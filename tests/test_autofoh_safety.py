"""Tests for the AutoFOH runtime policy and safety controller."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from autofoh_models import RuntimeState
from autofoh_safety import (
    AutoFOHSafetyConfig,
    AutoFOHSafetyController,
    BusCompressorAdjust,
    BusEQMove,
    BusFaderMove,
    ChannelEQMove,
    ChannelFaderMove,
    CompressorMakeupAdjust,
    DCAFaderMove,
    DelayAdjust,
    EmergencyFeedbackNotch,
    HighPassAdjust,
    MasterFaderMove,
    PolarityAdjust,
)


class FakeMixer:
    def __init__(self):
        self.fader = {}
        self.bus_fader = {1: -8.0}
        self.dca_fader = {2: -9.0}
        self.main_fader = {1: -2.0}
        self.delay = {1: 1.0}
        self.polarity = {}
        self.eq_gain = {}
        self.bus_eq_gain = {}
        self.calls = []

    def get_fader(self, channel):
        return self.fader.get(channel, -10.0)

    def set_fader(self, channel, value_db):
        self.fader[channel] = value_db
        self.calls.append(("set_fader", channel, value_db))
        return True

    def get_main_fader(self, main):
        return self.main_fader.get(main, 0.0)

    def set_main_fader(self, main, value_db):
        self.main_fader[main] = value_db
        self.calls.append(("set_main_fader", main, value_db))
        return True

    def get_delay(self, channel):
        return self.delay.get(channel, 0.0)

    def set_delay(self, channel, delay_ms, enabled=True):
        self.delay[channel] = delay_ms
        self.calls.append(("set_delay", channel, delay_ms, enabled))
        return True

    def get_polarity(self, channel):
        return self.polarity.get(channel, False)

    def set_polarity(self, channel, inverted):
        self.polarity[channel] = inverted
        self.calls.append(("set_polarity", channel, inverted))
        return True

    def get_bus_fader(self, bus):
        return self.bus_fader.get(bus, -10.0)

    def set_bus_fader(self, bus, value_db):
        self.bus_fader[bus] = value_db
        self.calls.append(("set_bus_fader", bus, value_db))
        return True

    def get_dca_fader(self, dca):
        return self.dca_fader.get(dca, -10.0)

    def set_dca_fader(self, dca, value_db):
        self.dca_fader[dca] = value_db
        self.calls.append(("set_dca_fader", dca, value_db))
        return True

    def get_eq_band_gain(self, channel, band):
        return self.eq_gain.get((channel, band), 0.0)

    def set_eq_band(self, channel, band, freq, gain, q):
        self.eq_gain[(channel, f"{band}g")] = gain
        self.calls.append(("set_eq_band", channel, band, freq, gain, q))
        return True

    def get_bus_eq_band_gain(self, bus, band):
        return self.bus_eq_gain.get((bus, band), 0.0)

    def set_bus_eq_band(self, bus, band, freq, gain, q):
        self.bus_eq_gain[(bus, band)] = gain
        self.calls.append(("set_bus_eq_band", bus, band, freq, gain, q))
        return True

    def set_compressor_gain(self, channel, gain):
        self.calls.append(("set_compressor_gain", channel, gain))
        return True

    def set_bus_compressor(self, bus, threshold_db, ratio, attack_ms, release_ms, makeup_db=0.0, enabled=True):
        self.calls.append(("set_bus_compressor", bus, threshold_db, ratio, attack_ms, release_ms, makeup_db, enabled))
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


def test_master_fader_move_can_only_cut_and_is_step_limited():
    mixer = FakeMixer()
    mixer.main_fader[1] = -2.0
    controller = AutoFOHSafetyController(
        mixer,
        config=AutoFOHSafetyConfig(master_fader_max_cut_db=1.0),
    )

    cut = controller.execute(
        MasterFaderMove(
            main_id=1,
            target_db=-8.0,
            reason="master peak ceiling protection",
        ),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )

    assert cut.sent is True
    assert cut.bounded is True
    assert mixer.calls[-1] == ("set_main_fader", 1, -3.0)


def test_master_fader_move_never_raises_main_fader():
    mixer = FakeMixer()
    mixer.main_fader[1] = -6.0
    controller = AutoFOHSafetyController(mixer)

    decision = controller.execute(
        MasterFaderMove(
            main_id=1,
            target_db=-2.0,
            reason="invalid upward master move",
        ),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )

    assert decision.sent is True
    assert decision.bounded is True
    assert mixer.calls[-1] == ("set_main_fader", 1, -6.0)


def test_bus_and_dca_fader_moves_are_step_limited_and_never_above_zero():
    mixer = FakeMixer()
    controller = AutoFOHSafetyController(
        mixer,
        config=AutoFOHSafetyConfig(
            bus_fader_max_step_db=1.0,
            dca_fader_max_step_db=0.5,
        ),
    )

    bus = controller.execute(
        BusFaderMove(bus_id=1, target_db=2.0, reason="bus balance"),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )
    dca = controller.execute(
        DCAFaderMove(dca_id=2, target_db=0.0, reason="DCA balance"),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )

    assert bus.sent is True
    assert bus.bounded is True
    assert dca.sent is True
    assert dca.bounded is True
    assert ("set_bus_fader", 1, -7.0) in mixer.calls
    assert ("set_dca_fader", 2, -8.5) in mixer.calls


def test_bus_processing_actions_use_bus_translators_and_bounds():
    mixer = FakeMixer()
    controller = AutoFOHSafetyController(mixer)

    eq_decision = controller.execute(
        BusEQMove(
            bus_id=3,
            band=2,
            freq_hz=350.0,
            gain_db=-8.0,
            q=0.2,
            reason="bus mud cleanup",
        ),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )
    compressor_decision = controller.execute(
        BusCompressorAdjust(
            bus_id=3,
            threshold_db=-80.0,
            ratio=40.0,
            attack_ms=0.1,
            release_ms=2000.0,
            makeup_db=20.0,
            reason="bus glue",
        ),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )

    assert eq_decision.sent is True
    assert eq_decision.bounded is True
    assert ("set_bus_eq_band", 3, 2, 350.0, -1.0, 0.44) in mixer.calls
    assert compressor_decision.sent is True
    assert compressor_decision.bounded is True
    assert mixer.calls[-1] == ("set_bus_compressor", 3, -50.0, 20.0, 1.0, 600.0, 12.0, True)


def test_phase_actions_are_bounded_and_use_safety_translators():
    mixer = FakeMixer()
    controller = AutoFOHSafetyController(
        mixer,
        config=AutoFOHSafetyConfig(
            delay_max_step_ms=0.25,
            delay_max_ms=10.0,
        ),
    )

    delay = controller.execute(
        DelayAdjust(channel_id=1, delay_ms=2.0, reason="drum phase"),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )
    polarity = controller.execute(
        PolarityAdjust(channel_id=2, inverted=True, reason="polarity"),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )

    assert delay.sent is True
    assert delay.bounded is True
    assert polarity.sent is True
    assert ("set_delay", 1, 1.25, True) in mixer.calls
    assert ("set_polarity", 2, True) in mixer.calls
