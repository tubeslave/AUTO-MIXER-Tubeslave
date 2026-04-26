"""Tests for the AutoFOH runtime policy and safety controller."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from autofoh_models import RuntimeState
from autofoh_safety import (
    AutoFOHSafetyConfig,
    AutoFOHSafetyController,
    ChannelEQMove,
    ChannelFaderMove,
    CompressorMakeupAdjust,
    EmergencyFeedbackNotch,
    HighPassAdjust,
)


class FakeMixer:
    def __init__(self):
        self.fader = {}
        self.eq_gain = {}
        self.calls = []

    def get_fader(self, channel):
        return self.fader.get(channel, -10.0)

    def set_fader(self, channel, value_db):
        self.fader[channel] = value_db
        self.calls.append(("set_fader", channel, value_db))
        return True

    def get_eq_band_gain(self, channel, band):
        return self.eq_gain.get((channel, band), 0.0)

    def set_eq_band(self, channel, band, freq, gain, q):
        self.eq_gain[(channel, f"{band}g")] = gain
        self.calls.append(("set_eq_band", channel, band, freq, gain, q))
        return True

    def set_compressor_gain(self, channel, gain):
        self.calls.append(("set_compressor_gain", channel, gain))
        return True


class NoHPFMixer(FakeMixer):
    pass


def test_state_machine_forbids_tonal_correction_during_song_start_stabilize():
    controller = AutoFOHSafetyController(FakeMixer())

    decision = controller.execute(
        ChannelEQMove(
            channel_id=1,
            band=2,
            freq_hz=3200.0,
            gain_db=-2.0,
            q=1.2,
            reason="masking cleanup",
        ),
        runtime_state=RuntimeState.SONG_START_STABILIZE,
    )

    assert decision.allowed is False
    assert decision.sent is False
    assert "not allowed" in decision.message


def test_emergency_feedback_action_is_allowed_in_emergency_feedback_state():
    mixer = FakeMixer()
    controller = AutoFOHSafetyController(mixer)

    decision = controller.execute(
        EmergencyFeedbackNotch(
            channel_id=3,
            band=4,
            freq_hz=3150.0,
            q=4.0,
            gain_db=-12.0,
            ttl_seconds=300.0,
            reason="feedback ring",
        ),
        runtime_state=RuntimeState.EMERGENCY_FEEDBACK,
    )

    assert decision.allowed is True
    assert decision.sent is True
    assert decision.bounded is True
    assert mixer.calls[-1] == ("set_eq_band", 3, 4, 3150.0, -6.0, 8.0)


def test_rate_limiter_blocks_excessive_rapid_fader_corrections():
    current_time = [100.0]

    def now():
        return current_time[0]

    mixer = FakeMixer()
    controller = AutoFOHSafetyController(mixer, time_provider=now)

    first = controller.execute(
        ChannelFaderMove(channel_id=1, target_db=-6.0, reason="first move"),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )
    second = controller.execute(
        ChannelFaderMove(channel_id=1, target_db=-5.0, reason="too soon"),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )

    assert first.sent is True
    assert second.rate_limited is True
    assert second.sent is False


def test_fader_bounds_do_not_lift_parked_channels_to_minus_30():
    mixer = FakeMixer()
    mixer.fader[1] = -100.0
    controller = AutoFOHSafetyController(
        mixer,
        config=AutoFOHSafetyConfig(channel_fader_max_step_db=1.0),
    )

    decision = controller.execute(
        ChannelFaderMove(channel_id=1, target_db=0.0, reason="large upward move"),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )

    assert decision.sent is True
    assert decision.bounded is True
    assert mixer.calls[-1] == ("set_fader", 1, -99.0)


def test_rollback_state_bypasses_rate_limit_for_reversible_actions():
    current_time = [100.0]

    def now():
        return current_time[0]

    mixer = FakeMixer()
    controller = AutoFOHSafetyController(mixer, time_provider=now)

    first = controller.execute(
        ChannelEQMove(
            channel_id=1,
            band=2,
            freq_hz=350.0,
            gain_db=-1.0,
            q=1.2,
            reason="initial cut",
        ),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )
    rollback = controller.execute(
        ChannelEQMove(
            channel_id=1,
            band=2,
            freq_hz=350.0,
            gain_db=0.0,
            q=1.2,
            reason="rollback cut",
        ),
        runtime_state=RuntimeState.ROLLBACK,
    )

    assert first.sent is True
    assert rollback.sent is True
    assert rollback.rate_limited is False


def test_safety_layer_blocks_unsupported_actions():
    controller = AutoFOHSafetyController(NoHPFMixer())

    decision = controller.execute(
        HighPassAdjust(channel_id=2, freq_hz=120.0, reason="cleanup"),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )

    assert decision.supported is False
    assert decision.sent is False
    assert decision.allowed is False


def test_safety_layer_bounds_outgoing_correction_values():
    mixer = FakeMixer()
    controller = AutoFOHSafetyController(
        mixer,
        config=AutoFOHSafetyConfig(
            broad_eq_max_step_db=1.0,
            broad_eq_max_total_db_from_snapshot=3.0,
        ),
    )

    decision = controller.execute(
        ChannelEQMove(
            channel_id=5,
            band=1,
            freq_hz=90.0,
            gain_db=6.0,
            q=1.0,
            reason="too much low boost",
        ),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )

    assert decision.sent is True
    assert decision.bounded is True
    assert mixer.calls[-1] == ("set_eq_band", 5, 1, 90.0, 1.0, 1.0)


def test_compressor_makeup_adjust_uses_safety_bounds():
    mixer = FakeMixer()
    controller = AutoFOHSafetyController(mixer)

    decision = controller.execute(
        CompressorMakeupAdjust(
            channel_id=2,
            makeup_db=18.0,
            reason="compensate compressor GR",
        ),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )

    assert decision.sent is True
    assert decision.bounded is True
    assert mixer.calls[-1] == ("set_compressor_gain", 2, 12.0)
