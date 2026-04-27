"""Tests for the optional MuQ-Eval quality layer."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from evaluation import MuQEvalService


def _sine(freq_hz=440.0, amplitude=0.3, sample_rate=48000, duration_sec=1.0):
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / float(sample_rate)
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _clipped(sample_rate=48000):
    return np.clip(_sine(amplitude=2.5, sample_rate=sample_rate), -1.0, 1.0)


def _service(tmp_path=None, **overrides):
    config = {
        "enabled": True,
        "fallback_enabled": True,
        "log_scores": False,
        "window_sec": 1,
        "sample_rate": 24000,
        "local_files_only": True,
        "shadow_mode": False,
        "min_seconds_between_quality_decisions": 0,
        "min_improvement_threshold": 0.03,
    }
    if tmp_path is not None:
        config.update(
            {
                "log_scores": True,
                "log_path": str(tmp_path / "muq_eval_decisions.jsonl"),
                "training_log_path": str(tmp_path / "muq_eval_rewards.jsonl"),
            }
        )
    config.update(overrides)
    return MuQEvalService(config)


def test_works_without_installed_muq_eval():
    service = _service()

    result = service.evaluate(_sine(), 48000)

    assert 0.0 <= result.quality_score <= 1.0
    assert result.model_status == "unavailable"
    assert result.technical_artifacts["fallback"] is True
    assert "clipping_penalty" in result.technical_artifacts


def test_normalizes_muq_score():
    assert MuQEvalService.normalize_score(1.0) == 0.0
    assert MuQEvalService.normalize_score(3.0) == 0.5
    assert MuQEvalService.normalize_score(5.0) == 1.0
    assert MuQEvalService.normalize_score(8.0) == 1.0


def test_rejects_correction_when_quality_drops():
    service = _service()

    decision = service.validate_correction(
        before_audio=_sine(),
        after_audio=_clipped(),
        sample_rate=48000,
        proposed_action={"action_type": "ChannelFaderMove", "delta_db": 0.2},
    )

    assert decision.accepted is False
    assert decision.rejection_reason == "quality_drop"
    assert decision.delta < 0


def test_accepts_correction_when_quality_improves():
    service = _service()

    decision = service.validate_correction(
        before_audio=_clipped(),
        after_audio=_sine(),
        sample_rate=48000,
        proposed_action={"action_type": "ChannelFaderMove", "delta_db": 0.2},
        osc_commands=[{"address": "/ch/1/fdr", "value": -3.0}],
    )

    assert decision.accepted is True
    assert decision.rejection_reason == ""
    assert decision.delta > 0.03
    assert decision.to_dict()["osc_commands"]


def test_quality_decision_rate_limit():
    service = _service(min_seconds_between_quality_decisions=10)

    first = service.validate_correction(
        before_audio=_clipped(),
        after_audio=_sine(),
        sample_rate=48000,
        proposed_action={"action_type": "ChannelFaderMove", "delta_db": 0.2},
        timestamp=100.0,
    )
    second = service.validate_correction(
        before_audio=_clipped(),
        after_audio=_sine(),
        sample_rate=48000,
        proposed_action={"action_type": "ChannelFaderMove", "delta_db": 0.2},
        timestamp=101.0,
    )

    assert first.accepted is True
    assert second.accepted is False
    assert second.rejection_reason == "quality_decision_rate_limited"


def test_jsonl_logs_decision_and_reward(tmp_path):
    service = _service(tmp_path)

    decision = service.validate_correction(
        before_audio=_clipped(),
        after_audio=_sine(),
        sample_rate=48000,
        proposed_action={"action_type": "ChannelEQMove", "delta_db": 0.2},
        session_id="song-1",
        current_scene="verse",
        osc_commands=[{"address": "/ch/2/eq/1g", "value": -1.0}],
        timestamp=123.0,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "muq_eval_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    reward_rows = [
        json.loads(line)
        for line in (tmp_path / "muq_eval_rewards.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert decision.accepted is True
    assert rows[0]["session_id"] == "song-1"
    assert rows[0]["current_scene"] == "verse"
    assert rows[0]["accepted"] is True
    assert rows[0]["score_before"]["model_status"] == "unavailable"
    assert reward_rows[0]["reward"] == rows[0]["reward"]
