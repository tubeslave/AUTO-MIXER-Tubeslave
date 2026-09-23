import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_workbench import causal
from audio_workbench import studio_iteration


def _write_tone(path: Path, *, gain: float = 0.08, sr: int = 48000, frames: int = 24000) -> None:
    t = np.arange(frames, dtype=np.float64) / sr
    mono = (gain * np.sin(2 * np.pi * 997 * t)).astype("float32")
    sf.write(path, np.column_stack([mono, mono * 0.7]), sr, subtype="PCM_24")


def _plan():
    return causal.make_plan(
        "sustained 2.5-6.5 kHz prominence exceeds the bounded harshness trigger",
        "a broad subtractive presence move may reduce the harshness proxy without protected regressions",
        "harshness",
        [{"type": "eq_bell", "params": {"frequency_hz": 4200, "gain_db": -0.8}, "label": "presence_cut"}],
        "harshness proxy decreases while density, width, foreground, punch and climax stay protected",
        ["density", "width_db", "foreground_db", "punch_db", "climax_lift_db"],
        confidence={"cause": 0.9, "intervention": 0.9},
    )


def _critic(decision: str, *, failures=None):
    failures = list(failures or [])
    return {
        "target": "harshness",
        "machine_decision": decision,
        "requires_human_listening": True,
        "failures": failures,
        "protected_regressions": [f for f in failures if f.endswith("_regression")],
        "evidence": [{"metric": "harshness", "role": "target", "passed": not failures}],
        "uncertainty": {"score": 0.2, "reasons": []},
    }


def _fake_master(monkeypatch, calls, *, status="pending_human_review"):
    def fake(source_path, output_dir, **kwargs):
        source_path = Path(source_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        calls.append(source_path.resolve())
        return {
            "status": status,
            "source_file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "artifacts": {},
            "post_master": {"integrated_lufs": kwargs["target_lufs"], "true_peak_dbtp": kwargs["ceiling_dbtp"]},
        }
    monkeypatch.setattr(studio_iteration, "deliver_master", fake)


def test_pending_perceptual_candidate_is_mastered_for_listening_without_baseline_promotion(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.wav"
    candidate = tmp_path / "candidate.wav"
    _write_tone(baseline, gain=0.08)
    _write_tone(candidate, gain=0.075)
    baseline_hash = hashlib.sha256(baseline.read_bytes()).hexdigest()
    candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    calls = []
    _fake_master(monkeypatch, calls)

    report = studio_iteration.run_studio_iteration(
        baseline, candidate, tmp_path / "run",
        baseline_id="baseline-1", candidate_id="candidate-2",
        plan=_plan(), critic_result=_critic("machine_safe"), evaluation_confidence=0.9,
        target_lufs=-16.0, ceiling_dbtp=-1.2,
    )

    assert report["status"] == "pending_human_review"
    assert report["delivery"]["role"] == "candidate_audition"
    assert report["delivery"]["rolled_back_to_baseline"] is False
    assert report["delivery"]["candidate_audition_exported"] is True
    assert report["baseline_after"] == "baseline-1"
    assert report["baseline_promoted"] is False
    assert report["transition"]["promote_baseline"] is False
    assert report["transition"]["next_action"] == "human_listening"
    assert calls == [candidate.resolve()]
    assert hashlib.sha256(baseline.read_bytes()).hexdigest() == baseline_hash
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == candidate_hash
    persisted = json.loads((tmp_path / "run" / "iteration_report.json").read_text())
    assert persisted["candidate"]["file_sha256"] == candidate_hash
    assert persisted["baseline"]["file_sha256"] == baseline_hash


def test_rejected_perceptual_candidate_rolls_back_before_real_mastering_boundary(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.wav"
    candidate = tmp_path / "candidate.wav"
    _write_tone(baseline, gain=0.08)
    _write_tone(candidate, gain=0.07)
    calls = []
    _fake_master(monkeypatch, calls)

    report = studio_iteration.run_studio_iteration(
        baseline, candidate, tmp_path / "run",
        baseline_id="baseline-1", candidate_id="candidate-2",
        plan=_plan(), critic_result=_critic("rejected", failures=["width_regression"]),
        evaluation_confidence=0.9,
    )

    assert report["status"] == "rejected"
    assert report["delivery"]["role"] == "baseline_rollback"
    assert report["delivery"]["rolled_back_to_baseline"] is True
    assert report["delivery"]["candidate_audition_exported"] is False
    assert report["transition"]["rollback_candidate"] is True
    assert report["baseline_after"] == "baseline-1"
    assert calls == [baseline.resolve()]


def test_mastering_rejection_cannot_promote_perceptual_candidate(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.wav"
    candidate = tmp_path / "candidate.wav"
    _write_tone(baseline, gain=0.08)
    _write_tone(candidate, gain=0.075)
    calls = []
    _fake_master(monkeypatch, calls, status="rejected")

    report = studio_iteration.run_studio_iteration(
        baseline, candidate, tmp_path / "run",
        baseline_id="baseline-1", candidate_id="candidate-2",
        plan=_plan(), critic_result=_critic("machine_safe"), evaluation_confidence=0.9,
    )

    assert report["status"] == "rejected_mastering"
    assert report["baseline_promoted"] is False
    assert report["baseline_after"] == "baseline-1"
    assert report["delivery"]["candidate_audition_exported"] is False
    assert calls == [candidate.resolve()]


def test_iteration_rejects_misaligned_candidate_before_mastering(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.wav"
    candidate = tmp_path / "candidate.wav"
    _write_tone(baseline, frames=24000)
    _write_tone(candidate, frames=23000)
    calls = []
    _fake_master(monkeypatch, calls)

    with pytest.raises(ValueError, match="shapes differ"):
        studio_iteration.run_studio_iteration(
            baseline, candidate, tmp_path / "run",
            baseline_id="baseline-1", candidate_id="candidate-2",
            plan=_plan(), critic_result=_critic("machine_safe"), evaluation_confidence=0.9,
        )
    assert calls == []
    assert not (tmp_path / "run").exists()


def test_iteration_requires_no_change_counterfactual(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.wav"
    candidate = tmp_path / "candidate.wav"
    _write_tone(baseline)
    _write_tone(candidate, gain=0.075)
    plan = _plan()
    plan["candidates"] = [c for c in plan["candidates"] if c["type"] != "bypass"]
    calls = []
    _fake_master(monkeypatch, calls)

    with pytest.raises(ValueError, match="bypass/no-change"):
        studio_iteration.run_studio_iteration(
            baseline, candidate, tmp_path / "run",
            baseline_id="baseline-1", candidate_id="candidate-2",
            plan=plan, critic_result=_critic("machine_safe"), evaluation_confidence=0.9,
        )
    assert calls == []
