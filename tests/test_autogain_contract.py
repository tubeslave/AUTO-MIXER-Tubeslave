"""Shared AutoGain recommendation contract tests."""

from __future__ import annotations

import numpy as np

from arrangement_automation.live_input_trim_controller import LiveInputTrimController
from auto_soundcheck_engine import AutoSoundcheckEngine, ChannelInfo, ChannelSnapshot
from lufs_gain_staging import AnalysisState, LUFSGainStagingController, SafeGainCalibrator
from signal_metrics import ChannelMetrics


class _TrimMixer:
    is_connected = True

    def __init__(self, trim: float = 0.0):
        self.trim = float(trim)
        self.sent = []

    def get_channel_gain(self, channel):
        return self.trim

    def set_channel_gain(self, channel, value):
        self.sent.append((channel, float(value)))
        self.trim = float(value)
        return True

    def confirm_manual_write(self, operation, channel, desired_value, timeout_ms=300):
        return {
            "readback_status": "confirmed",
            "confirmed_value": float(self.trim),
            "verification_id": f"verify-{channel}",
            "failure_reason": None,
            "timeout_ms": timeout_ms,
        }


def _tone(freq=700.0, dbfs=-24.0, sample_rate=48_000, seconds=0.5):
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    amp = 10.0 ** (dbfs / 20.0)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _live_trim_config():
    return {
        "automation": {
            "live_input_trim": {
                "run_background_loop": False,
                "analysis_only_mode": False,
                "live_apply_enabled": True,
                "activity_attack_hold_sec": 0.0,
                "min_main_signal_confidence": 0.1,
                "analysis_window_sec": 0.5,
                "apply_cooldown_sec": 0.0,
                "max_step_db_per_tick": 0.25,
                "max_boost_step_db": 0.15,
                "deadband_db": 0.1,
                "role_targets": {
                    "lead_vocal": {"peak_dbfs": -10.0, "main_sec": 0.5, "windows": 2},
                },
            }
        }
    }


def test_safe_gain_calibrator_exposes_shared_recommendation_contract():
    mixer = _TrimMixer(trim=2.0)
    calibrator = SafeGainCalibrator(mixer_client=mixer)
    calibrator.add_channel(1, 1)
    calibrator.state = AnalysisState.READY
    stats = calibrator.channels[1]
    stats.integrated_lufs = -22.0
    stats.max_true_peak_db = -8.0
    stats.crest_factor_db = 14.0
    stats.signal_presence_ratio = 0.8
    stats.suggested_gain_db = 4.0
    stats.gain_limited_by = "peak"
    calibrator.suggestions[1] = stats.get_report()

    recommendations = calibrator.get_recommendations()

    rec = recommendations[1]
    assert rec["source"] == "safe_gain_calibrator"
    assert rec["current_trim_db"] == 2.0
    assert rec["recommended_target_trim_db"] == 6.0
    assert rec["delta_db"] == 4.0
    assert rec["reason"] == "safe_gain_peak_limited"
    assert rec["dry_run_only"] is True
    assert rec["live_apply_safe"] is False


def test_live_input_trim_exposes_shared_recommendation_contract():
    mixer = _TrimMixer()
    controller = LiveInputTrimController(
        mixer_client=mixer,
        config=_live_trim_config(),
    )
    controller.start([1], channel_names={1: "Lead Vox"})
    audio = {1: _tone()}
    controller.process_once(audio_by_channel=audio, timestamp=0.0)
    controller.process_once(audio_by_channel=audio, timestamp=0.6)

    recommendations = controller.get_recommendations()
    controller.stop()

    rec = recommendations[1]
    assert rec["source"] == "live_input_trim_controller"
    assert rec["analysis_state"] in {"ready_to_adjust", "hold"}
    assert rec["confidence"] >= 0.1
    assert rec["recommended_target_trim_db"] >= rec["current_trim_db"]
    assert "trim_bounds_db" in rec["metadata"]


def test_auto_soundcheck_engine_exposes_shared_gain_recommendation_contract():
    engine = AutoSoundcheckEngine(auto_apply=False)
    metrics = ChannelMetrics(channel=1)
    metrics.level.true_peak_dbtp = -18.0
    metrics.level.rms_db = -30.0
    metrics.level.crest_factor_db = 12.0
    info = ChannelInfo(
        channel=1,
        name="Kick",
        preset="kick",
        recognized=True,
        has_signal=True,
        peak_db=-18.0,
        rms_db=-30.0,
        metrics=metrics,
        original_snapshot=ChannelSnapshot(channel=1, gain_db=1.0),
    )
    engine.channels[1] = info

    recommendations = engine.get_gain_recommendations()

    rec = recommendations[1]
    assert rec["source"] == "auto_soundcheck_engine"
    assert rec["current_trim_db"] == 1.0
    assert rec["recommended_target_trim_db"] > 1.0
    assert rec["delta_db"] > 0.0
    assert rec["dry_run_only"] is True
    assert rec["metadata"]["auto_apply_enabled"] is False


def test_legacy_lufs_gain_controller_blocks_live_trim_without_confirmation():
    mixer = _TrimMixer(trim=0.0)
    controller = LUFSGainStagingController(
        mixer_client=mixer,
        config={"automation": {"lufs_gain_staging": {"max_apply_step_db": 1.0}}},
        dry_run=False,
        confirm_live_apply=False,
    )

    result = controller._apply_trim_change(
        1,
        6.0,
        source="test_legacy_lufs_realtime",
        current_trim=0.0,
        min_value=-18.0,
        max_value=18.0,
        max_step=1.0,
    )

    assert result.accepted is False
    assert result.blocked_reason == "confirm_live_apply_required"
    assert mixer.sent == []


def test_legacy_lufs_gain_controller_routes_trim_through_guarded_gate():
    mixer = _TrimMixer(trim=0.0)
    controller = LUFSGainStagingController(
        mixer_client=mixer,
        config={"automation": {"lufs_gain_staging": {"max_apply_step_db": 1.0}}},
        dry_run=False,
        confirm_live_apply=True,
    )

    result = controller._apply_trim_change(
        1,
        6.0,
        source="test_legacy_lufs_realtime",
        current_trim=0.0,
        min_value=-18.0,
        max_value=18.0,
        max_step=1.0,
    )

    assert result.accepted is True
    assert result.desired_value == 1.0
    assert result.confirmed_value == 1.0
    assert result.guard_reasons == ("max_step_clamped",)
    assert mixer.sent == [(1, 1.0)]
