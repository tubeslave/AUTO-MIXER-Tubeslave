"""Tests for the proof-gated AutoFOH correction executor."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from autofoh_models import RuntimeState
from autofoh_safety import (
    AutoFOHSafetyConfig,
    AutoFOHSafetyController,
    ChannelEQMove,
    ChannelFaderMove,
    ChannelGainMove,
    ChannelMuteMove,
)
from correction_action_executor import (
    CorrectionActionExecutorConfig,
    GatedCorrectionActionExecutor,
)


class ProofMixer:
    def __init__(self):
        self.fader = {1: -10.0}
        self.gain = {1: 0.0}
        self.eq_gain = {(1, "2g"): 0.0}
        self.calls = []

    def get_fader(self, channel):
        return self.fader.get(channel)

    def set_fader(self, channel, value_db):
        self.fader[channel] = value_db
        self.calls.append(("set_fader", channel, value_db))
        return True

    def get_channel_gain(self, channel):
        return self.gain.get(channel)

    def set_gain(self, channel, value_db):
        self.gain[channel] = value_db
        self.calls.append(("set_gain", channel, value_db))
        return True

    def get_eq_band_gain(self, channel, band):
        return self.eq_gain.get((channel, band), 0.0)

    def set_eq_band(self, channel, band, freq, gain, q):
        self.eq_gain[(channel, f"{band}g")] = gain
        self.calls.append(("set_eq_band", channel, band, freq, gain, q))
        return True

    def get_channel_mute(self, channel):
        return self.eq_gain.get(("mute", channel), False)

    def set_channel_mute(self, channel, muted):
        self.calls.append(("set_channel_mute", channel, bool(muted)))
        self.eq_gain[("mute", channel)] = bool(muted)
        return True


class NoGainReadMixer(ProofMixer):
    def get_channel_gain(self, channel):
        return None


class StaleReadbackMixer(ProofMixer):
    def set_fader(self, channel, value_db):
        self.calls.append(("set_fader", channel, value_db))
        return True


def _executor(mixer, **config):
    config = dict(config)
    allow_direct_mute_writes = config.pop("allow_direct_mute_writes", False)
    safety = AutoFOHSafetyController(
        mixer,
        config=AutoFOHSafetyConfig(
            channel_fader_max_step_db=2.0,
            channel_fader_min_interval_sec=0.0,
            gain_min_interval_sec=0.0,
            allow_direct_mute_writes=allow_direct_mute_writes,
        ),
    )
    return GatedCorrectionActionExecutor(
        safety,
        config=CorrectionActionExecutorConfig.from_config(config),
    )


def test_promoted_fader_requires_pre_read_and_readback_before_verified_send():
    mixer = ProofMixer()
    executor = _executor(mixer)

    result = executor.execute(
        ChannelFaderMove(channel_id=1, target_db=-8.5, reason="line check balance"),
        RuntimeState.SOURCE_LEARNING,
    )

    assert result.sent is True
    assert result.write_sent is True
    assert result.pre_read_ok is True
    assert result.readback_ok is True
    assert result.rollback_available is True
    assert result.rollback_action.target_db == -10.0
    assert mixer.calls == [("set_fader", 1, -8.5)]


def test_dry_run_never_writes_but_returns_operator_recommendation_data():
    mixer = ProofMixer()
    executor = _executor(mixer)

    result = executor.execute(
        ChannelGainMove(channel_id=1, target_db=1.5, reason="trim up proof"),
        RuntimeState.SOURCE_LEARNING,
        dry_run=True,
    )
    decision = result.to_safety_decision()

    assert result.allowed is True
    assert result.sent is False
    assert result.message == "dry_run_only"
    assert mixer.calls == []
    assert decision.payload["gated_executor"]["pre_read"]["value"] == 0.0


def test_relative_artifact_path_is_cwd_independent_and_test_isolated(
    tmp_path,
    monkeypatch,
):
    runtime_root = tmp_path / "correction-runtime"
    working_directory = tmp_path / "checkout" / "backend"
    working_directory.mkdir(parents=True)
    monkeypatch.setenv(
        "AUTOMIXER_TEST_CORRECTION_RUNTIME_DIR",
        str(runtime_root),
    )
    monkeypatch.chdir(working_directory)

    executor = _executor(
        ProofMixer(),
        artifact_path="show_assist_auto_v1/gated_correction_events.jsonl",
    )
    executor.execute(
        ChannelGainMove(channel_id=1, target_db=1.5, reason="artifact isolation"),
        RuntimeState.SOURCE_LEARNING,
        dry_run=True,
    )

    expected = runtime_root / "show_assist_auto_v1" / "gated_correction_events.jsonl"
    assert expected.is_file()
    assert "gated_action_dry_run" in expected.read_text(encoding="utf-8")
    assert not (working_directory / "show_assist_auto_v1").exists()
    assert not (working_directory / "artifacts").exists()


def test_artifact_path_rejects_absolute_and_parent_traversal(tmp_path):
    for invalid in (str(tmp_path / "events.jsonl"), "../source-tree/events.jsonl"):
        with pytest.raises(ValueError, match="relative runtime path"):
            _executor(ProofMixer(), artifact_path=invalid)


def test_artifact_test_runtime_override_is_rejected_outside_pytest(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AUTOMIXER_TEST_CORRECTION_RUNTIME_DIR",
        str(tmp_path / "runtime"),
    )
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    with pytest.raises(RuntimeError, match="allowed only under pytest"):
        _executor(
            ProofMixer(),
            artifact_path="show_assist_auto_v1/events.jsonl",
        )


def test_disabled_gated_executor_fails_closed_without_calling_legacy_writer():
    mixer = ProofMixer()
    executor = _executor(mixer, enabled=False)

    result = executor.execute(
        ChannelFaderMove(channel_id=1, target_db=-8.5, reason="disabled gate"),
        RuntimeState.SOURCE_LEARNING,
    )

    assert result.allowed is False
    assert result.sent is False
    assert result.message == "gated_executor_disabled"
    assert mixer.calls == []


def test_unpromoted_eq_is_blocked_before_any_mixer_write():
    mixer = ProofMixer()
    executor = _executor(mixer)

    result = executor.execute(
        ChannelEQMove(
            channel_id=1,
            band=2,
            freq_hz=350.0,
            gain_db=-1.0,
            q=1.2,
            reason="mud cleanup",
        ),
        RuntimeState.SOURCE_LEARNING,
    )

    assert result.sent is False
    assert result.eligible is False
    assert result.message == "action_not_promoted_for_live_write"
    assert mixer.calls == []


def test_promoted_gain_fails_closed_when_pre_read_is_missing():
    mixer = NoGainReadMixer()
    executor = _executor(mixer)

    result = executor.execute(
        ChannelGainMove(channel_id=1, target_db=2.0, reason="trim up"),
        RuntimeState.SOURCE_LEARNING,
    )

    assert result.sent is False
    assert result.message == "pre_read_unavailable"
    assert result.demoted is True
    assert mixer.calls == []


def test_failed_readback_rolls_back_and_demotes_action_type():
    mixer = StaleReadbackMixer()
    executor = _executor(mixer)

    result = executor.execute(
        ChannelFaderMove(channel_id=1, target_db=-8.5, reason="line check balance"),
        RuntimeState.SOURCE_LEARNING,
    )

    assert result.sent is False
    assert result.write_sent is True
    assert result.readback_ok is False
    assert result.rollback_sent is True
    assert result.demoted is True
    assert mixer.calls == [
        ("set_fader", 1, -8.5),
        ("set_fader", 1, -10.0),
    ]

    blocked = executor.execute(
        ChannelFaderMove(channel_id=1, target_db=-8.0, reason="second move"),
        RuntimeState.SOURCE_LEARNING,
    )
    assert blocked.message == "action_not_promoted_for_live_write"


def test_promoted_mute_move_remains_quarantined_with_rollback_evidence():
    mixer = ProofMixer()
    executor = _executor(
        mixer,
        promoted_action_types={"ChannelMuteMove": True},
        allow_direct_mute_writes=True,
    )

    result = executor.execute(
        ChannelMuteMove(channel_id=1, muted=True, reason="safety isolation"),
        RuntimeState.SOURCE_LEARNING,
    )

    assert result.sent is False
    assert result.write_sent is False
    assert result.allowed is False
    assert result.message == "blocked by AutoFOH proposal-only direct-write policy"
    assert result.safety_decision is not None
    assert result.safety_decision.blocked is True
    assert result.safety_decision.payload["direct_mute_override_ignored"] is True
    assert result.safety_decision.payload["blocked_calls"][0]["cmd"] == "set_channel_mute"
    assert result.pre_read_ok is True
    assert result.readback_ok is False
    assert result.rollback_available is True
    assert result.rollback_action is not None
    assert isinstance(result.rollback_action, ChannelMuteMove)
    assert result.rollback_action.muted is False
    assert mixer.calls == []
