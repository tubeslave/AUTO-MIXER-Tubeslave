"""Tests for the MuQ-Eval director offline experiment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.experiments.muq_eval_director import (
    CandidateResult,
    ChannelState,
    DirectorAction,
    DirectorState,
    MuqDirectorConfig,
    MuqEvalDirectorOfflineTest,
)


@dataclass
class FakeMuQResult:
    quality_score: float
    model_status: str = "available"
    confidence: float = 0.9


class FakeMuQService:
    def __init__(self, status: str = "available", score: float = 0.5):
        self.model_status = status
        self.score = score

    def evaluate(self, audio, sample_rate, timestamp=None):
        return FakeMuQResult(self.score, self.model_status)


def _write_track(path: Path, freq: float = 220.0, amp: float = 0.1, sr: int = 48000, duration: float = 0.2) -> None:
    t = np.arange(int(sr * duration), dtype=np.float32) / float(sr)
    audio = (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), audio, sr)


def _session(tmp_path: Path) -> Path:
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    _write_track(tracks / "KICK.wav", 60.0)
    _write_track(tracks / "Vocal.wav", 440.0)
    return tracks


def _config(tmp_path: Path, **overrides) -> MuqDirectorConfig:
    data = {
        "enabled": True,
        "offline_only": True,
        "send_osc": False,
        "shadow_mode": False,
        "window_sec": 0.05,
        "hop_sec": 0.05,
        "max_steps": 1,
        "candidates_per_step": 3,
        "beam_width": 2,
        "min_delta_accept": 0.005,
        "stop_if_no_improvement_steps": 1,
        "target_lufs": -14.0,
        "true_peak_ceiling_dbfs": -1.0,
        "max_eval_windows": 1,
        "output_root": str(tmp_path / "outputs"),
        "require_available_muq": True,
    }
    data.update(overrides)
    return MuqDirectorConfig.from_mapping(data)


def _state(score: float = 0.5) -> DirectorState:
    return DirectorState(
        state_id="parent",
        step=0,
        muq_score=score,
        final_score=score,
        channels={
            1: ChannelState(
                channel_id=1,
                name="Vocal",
                path="Vocal.wav",
                instrument="lead_vocal",
            )
        },
    )


def _candidate(candidate_id: str, score: float, accepted: bool = True, peak: float = -3.0) -> CandidateResult:
    state = _state(score)
    state.state_id = candidate_id
    return CandidateResult(
        step=1,
        parent_state_id="parent",
        candidate_id=candidate_id,
        action=DirectorAction("gain", 1, "lead_vocal", {"delta_db": 0.5}),
        state=state,
        render_path=f"{candidate_id}.wav",
        muq_score_before=0.5,
        muq_score_after=score,
        delta_muq=score - 0.5,
        loudness_before=-14.0,
        loudness_after=-14.0,
        peak_before=-3.0,
        peak_after=peak,
        phase_score=1.0,
        loudness_score=1.0,
        peak_headroom_score=1.0,
        anti_overprocessing_score=1.0,
        final_score=score,
        accepted=accepted,
        rejection_reason="" if accepted else "rejected",
        audio_window_scores=[score],
    )


def test_director_rejects_send_osc_true(tmp_path):
    with pytest.raises(ValueError, match="send_osc"):
        MuqEvalDirectorOfflineTest(_config(tmp_path, send_osc=True), muq_service=FakeMuQService())


def test_default_director_paths_split_audio_and_logs():
    config = MuqDirectorConfig()

    assert "Ai LOGS" in Path(config.output_root).parts
    assert "Ai MIXING" in Path(config.audio_output_root).parts


def test_muq_unavailable_graceful_failure(tmp_path):
    director = MuqEvalDirectorOfflineTest(_config(tmp_path), muq_service=FakeMuQService(status="unavailable"))

    report = director.run(_session(tmp_path))

    assert report["status"] == "failed"
    assert report["model_status"] == "unavailable"
    assert report["stopping_reason"] == "muq_eval_unavailable"
    assert (director.output_dir / "report.json").exists()


def test_selects_best_muq_candidate(tmp_path):
    director = MuqEvalDirectorOfflineTest(_config(tmp_path, enabled=False), muq_service=FakeMuQService())

    best = director.select_best_candidate([
        _candidate("low", 0.52),
        _candidate("high", 0.61),
        _candidate("rejected", 0.9, accepted=False),
    ])

    assert best is not None
    assert best.candidate_id == "high"


def test_clipping_candidate_is_rejected(tmp_path):
    director = MuqEvalDirectorOfflineTest(_config(tmp_path, enabled=False), muq_service=FakeMuQService())

    reason = director._candidate_rejection_reason(
        {
            "clipping_ratio": 0.01,
            "peak_dbfs": -0.5,
            "crest_factor_db": 10.0,
        },
        delta_muq=0.2,
        action=DirectorAction("gain", 1, "lead_vocal", {"delta_db": 1.0}),
        parent=_state(),
    )

    assert reason == "clipping"


def test_rollback_returns_parent_state(tmp_path):
    director = MuqEvalDirectorOfflineTest(_config(tmp_path, enabled=False), muq_service=FakeMuQService())
    parent = _state()
    director._state_archive[parent.state_id] = parent

    child = director.apply_best_action(parent, DirectorAction("gain", 1, "lead_vocal", {"delta_db": 1.0}))
    rolled_back = director.rollback(child)

    assert rolled_back is not None
    assert rolled_back.state_id == parent.state_id
    assert rolled_back.channels[1].gain_db == 0.0


def test_beam_search_keeps_top_n_and_writes_jsonl(tmp_path, monkeypatch):
    director = MuqEvalDirectorOfflineTest(_config(tmp_path), muq_service=FakeMuQService())
    actions = [
        DirectorAction("gain", 1, "kick", {"delta_db": 1.0}),
        DirectorAction("gain", 2, "lead_vocal", {"delta_db": 1.0}),
        DirectorAction("gain", 1, "kick", {"delta_db": -1.0}),
    ]
    monkeypatch.setattr(director, "propose_actions", lambda state: actions)

    def fake_evaluate(path, *, action=None, state=None):
        if action is None:
            score = 0.5
        elif action.channel_id == 2:
            score = 0.62
        elif action.parameters.get("delta_db", 0.0) > 0:
            score = 0.58
        else:
            score = 0.49
        return {
            "muq_score": score,
            "audio_window_scores": [score],
            "final_score": score,
            "lufs": -14.0,
            "peak_dbfs": -3.0,
            "phase_score": 1.0,
            "crest_factor_db": 10.0,
            "clipping_ratio": 0.0,
            "loudness_score": 1.0,
            "peak_headroom_score": 1.0,
            "anti_overprocessing_score": 1.0,
        }

    monkeypatch.setattr(director, "evaluate_candidate", fake_evaluate)

    report = director.run(_session(tmp_path))

    assert report["status"] == "completed"
    assert report["model_status"] == "available"
    assert report["score_curve"][0]["model_status"] == "available"
    assert len(director.beams) == 2
    assert director.beams[0].muq_score >= director.beams[1].muq_score
    rows = [
        json.loads(line)
        for line in (director.output_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 3
    assert all(row["model_status"] == "available" for row in rows)
    assert any(row["accepted"] for row in rows)


def test_stopping_when_all_candidates_are_worse(tmp_path, monkeypatch):
    director = MuqEvalDirectorOfflineTest(_config(tmp_path, candidates_per_step=1), muq_service=FakeMuQService())
    monkeypatch.setattr(director, "propose_actions", lambda state: [DirectorAction("gain", 1, "kick", {"delta_db": -1.0})])

    def fake_evaluate(path, *, action=None, state=None):
        score = 0.5 if action is None else 0.49
        return {
            "muq_score": score,
            "audio_window_scores": [score],
            "final_score": score,
            "lufs": -14.0,
            "peak_dbfs": -3.0,
            "phase_score": 1.0,
            "crest_factor_db": 10.0,
            "clipping_ratio": 0.0,
            "loudness_score": 1.0,
            "peak_headroom_score": 1.0,
            "anti_overprocessing_score": 1.0,
        }

    monkeypatch.setattr(director, "evaluate_candidate", fake_evaluate)

    report = director.run(_session(tmp_path))

    assert report["stopping_reason"] == "all_candidates_rejected_or_worse"
