import numpy as np

from lufs_gain_staging import ChannelPhase, SafeGainCalibrator


def _block(dbfs: float, size: int = 2048) -> np.ndarray:
    amp = 10 ** (dbfs / 20.0)
    return np.full(size, amp, dtype=np.float32)


def _calibrator_config(
    *,
    required_own_events: int = 2,
    capture_window_db: float = 3.0,
    own_trigger_delta_db: float = 6.0,
    retrigger_delta_db: float = 4.0,
    ceiling_dbtp: float = -1.0,
) -> dict:
    return {
        "automation": {
            "safe_gain_calibration": {
                "bleed_learn_blocks": 3,
                "own_trigger_delta_db": own_trigger_delta_db,
                "capture_window_db": capture_window_db,
                "required_own_events": required_own_events,
                "capture_timeout_sec": 5.0,
                "retrigger_delta_db": retrigger_delta_db,
                "bypass_trigger_peak_dbfs": -20.0,
                "max_gain_adjustment_db": 18.0,
                "gain_increase_true_peak_ceiling_dbtp": ceiling_dbtp,
                "auto_stop_when_ready": True,
            }
        }
    }


def _new_calibrator(config: dict, preset: str = "tom_mid") -> SafeGainCalibrator:
    cal = SafeGainCalibrator(mixer_client=None, sample_rate=48000, config=config, bleed_service=None)
    cal.add_channel(1)
    cal.channel_settings = {1: {"preset": preset}}
    assert cal.start_analysis() is True
    return cal


def test_bleed_first_then_short_own_capture_marks_ready():
    cal = _new_calibrator(_calibrator_config())

    for _ in range(3):
        cal.process_audio(1, _block(-50.0), bleed_ratio_override=0.95)

    stats = cal.channels[1]
    assert stats.phase == ChannelPhase.WAIT_FOR_OWN_SIGNAL

    cal.process_audio(1, _block(-45.0), bleed_ratio_override=0.95)  # below trigger
    assert stats.phase == ChannelPhase.WAIT_FOR_OWN_SIGNAL

    cal.process_audio(1, _block(-30.0), bleed_ratio_override=0.95)
    cal.process_audio(1, _block(-31.0), bleed_ratio_override=0.95)

    assert stats.phase == ChannelPhase.READY
    assert stats.accepted_count == 2


def test_wait_phase_does_not_penalize_long_duration_before_own_signal():
    cal = _new_calibrator(_calibrator_config())

    for _ in range(3):
        cal.process_audio(1, _block(-50.0), bleed_ratio_override=0.95)

    stats = cal.channels[1]
    assert stats.phase == ChannelPhase.WAIT_FOR_OWN_SIGNAL

    for _ in range(30):
        cal.process_audio(1, _block(-49.0), bleed_ratio_override=0.95)

    cal.process_audio(1, _block(-30.0), bleed_ratio_override=0.95)
    cal.process_audio(1, _block(-30.8), bleed_ratio_override=0.95)

    assert stats.phase == ChannelPhase.READY
    assert stats.accepted_count == 2
    assert stats.total_samples > 30


def test_capture_restarts_on_new_stronger_level():
    cal = _new_calibrator(_calibrator_config(required_own_events=2, retrigger_delta_db=4.0))

    for _ in range(3):
        cal.process_audio(1, _block(-50.0), bleed_ratio_override=0.95)

    stats = cal.channels[1]
    cal.process_audio(1, _block(-30.0), bleed_ratio_override=0.95)  # capture starts
    assert stats.phase == ChannelPhase.CAPTURE_OWN_LEVEL
    first_reference = stats.reference_peak_dbfs

    # A much stronger hit should re-center capture around the new reference.
    cal.process_audio(1, _block(-18.0), bleed_ratio_override=0.95)
    assert stats.reference_peak_dbfs is not None
    assert stats.reference_peak_dbfs > first_reference + 6.0
    assert stats.accepted_count >= 1

    cal.process_audio(1, _block(-18.5), bleed_ratio_override=0.95)
    assert stats.phase == ChannelPhase.READY
    assert stats.accepted_count == 2
    assert stats.max_true_peak_db > -21.0


def test_true_peak_ceiling_still_limits_boost_after_phase_capture():
    cal = _new_calibrator(
        _calibrator_config(required_own_events=2, ceiling_dbtp=-32.0),
        preset="tom_mid",
    )

    for _ in range(3):
        cal.process_audio(1, _block(-52.0), bleed_ratio_override=0.95)

    # Two stable own-signal hits complete capture and trigger finalize (auto_stop_when_ready=True).
    cal.process_audio(1, _block(-30.0), bleed_ratio_override=0.95)
    cal.process_audio(1, _block(-30.5), bleed_ratio_override=0.95)

    assert cal.state == cal.state.READY
    suggestion = cal.suggestions[1]
    assert suggestion["limited_by"] == "true_peak_ceiling"
    assert suggestion["suggested_gain_db"] == 0.0


def test_bypass_preset_uses_simple_peak_threshold_in_wait_phase():
    cal = _new_calibrator(_calibrator_config(), preset="bass")

    for _ in range(3):
        cal.process_audio(1, _block(-10.0), bleed_ratio_override=0.95)

    stats = cal.channels[1]
    assert stats.phase == ChannelPhase.WAIT_FOR_OWN_SIGNAL

    # For bypass presets, own trigger should use simple threshold (-20 dBFS),
    # not (baseline + delta) logic.
    cal.process_audio(1, _block(-19.0), bleed_ratio_override=0.95)
    assert stats.phase == ChannelPhase.CAPTURE_OWN_LEVEL


def test_bypass_preset_can_trigger_from_dominant_signal_below_threshold():
    cal = _new_calibrator(_calibrator_config(), preset="guitar")

    for _ in range(3):
        cal.process_audio(1, _block(-24.0), bleed_ratio_override=0.95)

    stats = cal.channels[1]
    assert stats.phase == ChannelPhase.WAIT_FOR_OWN_SIGNAL

    cal.process_audio(
        1,
        _block(-30.0),
        bleed_ratio_override=0.95,
        all_channel_levels_db={1: -30.0, 2: -45.0, 3: -44.0},
    )
    assert stats.phase == ChannelPhase.CAPTURE_OWN_LEVEL


def test_non_bypass_capture_accepts_loud_own_signal_with_unstable_dynamics():
    cal = _new_calibrator(
        _calibrator_config(required_own_events=3, capture_window_db=1.0),
        preset="snare",
    )

    # Learn baseline around -14 dBFS, so regular trigger for non-bypass channel is:
    # min(-14 + 6, -6) = -8 dBFS.
    for _ in range(3):
        cal.process_audio(1, _block(-14.0), bleed_ratio_override=0.95)

    stats = cal.channels[1]
    assert stats.phase == ChannelPhase.WAIT_FOR_OWN_SIGNAL

    # Enter capture on first own hit.
    cal.process_audio(1, _block(-6.0), bleed_ratio_override=0.95)
    assert stats.phase == ChannelPhase.CAPTURE_OWN_LEVEL

    # Strongly varying own hits (outside tiny +/-1 dB window) should still be accepted
    # via loud-own fallback and let capture complete.
    cal.process_audio(1, _block(-2.0), bleed_ratio_override=0.95)
    cal.process_audio(1, _block(-7.0), bleed_ratio_override=0.95)

    assert stats.phase == ChannelPhase.READY
    assert stats.accepted_count >= 3


def test_wait_phase_blocks_confident_bleed_from_starting_capture():
    cal = _new_calibrator(_calibrator_config(required_own_events=2), preset="snare")

    for _ in range(3):
        cal.process_audio(1, _block(-24.0), bleed_ratio_override=0.10, bleed_confidence_override=0.10)

    stats = cal.channels[1]
    assert stats.phase == ChannelPhase.WAIT_FOR_OWN_SIGNAL

    # Peak is high enough, but confident bleed must block transition to CAPTURE.
    cal.process_audio(
        1,
        _block(-7.0),
        bleed_ratio_override=0.90,
        bleed_confidence_override=0.95,
    )
    assert stats.phase == ChannelPhase.WAIT_FOR_OWN_SIGNAL
    assert stats.rejected_bleed_samples >= 1


def test_capture_phase_ignores_confident_bleed_blocks():
    cal = _new_calibrator(_calibrator_config(required_own_events=3), preset="tom_mid")

    for _ in range(3):
        cal.process_audio(1, _block(-24.0), bleed_ratio_override=0.10, bleed_confidence_override=0.10)

    stats = cal.channels[1]
    assert stats.phase == ChannelPhase.WAIT_FOR_OWN_SIGNAL

    # Start capture with clean own event.
    cal.process_audio(1, _block(-8.0), bleed_ratio_override=0.10, bleed_confidence_override=0.10)
    assert stats.phase == ChannelPhase.CAPTURE_OWN_LEVEL
    accepted_after_start = stats.accepted_count

    # Confident bleed should not be accepted into capture.
    cal.process_audio(1, _block(-8.2), bleed_ratio_override=0.90, bleed_confidence_override=0.90)
    assert stats.accepted_count == accepted_after_start
    assert stats.rejected_bleed_samples >= 1
