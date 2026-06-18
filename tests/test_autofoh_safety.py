"""Tests for the AutoFOH runtime policy and safety controller."""

import os
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from autofoh_models import RuntimeState
from autofoh_safety import (
    AutoFOHSafetyConfig,
    AutoFOHSafetyController,
    TypedCorrectionAction,
    BusCompressorAdjust,
    BusEQMove,
    BusFaderMove,
    ChannelEQMove,
    ChannelFaderMove,
    ChannelMuteMove,
    CompressorMakeupAdjust,
    DCAFaderMove,
    DelayAdjust,
    EmergencyFeedbackNotch,
    HighPassAdjust,
    MasterFaderMove,
    PolarityAdjust,
)
from autofoh_runtime import ACTION_FAMILY_FX


def assert_blocked_evidence(decision, cmd):
    assert decision.allowed is False
    assert decision.sent is False
    assert decision.blocked is True
    assert decision.write_policy == "proposal_only_quarantined"
    assert decision.payload["blocked"] is True
    assert decision.payload["write_policy"] == "proposal_only_quarantined"
    assert decision.payload["live_write_allowed"] is False
    assert decision.payload["can_apply_live"] is False
    assert decision.payload["mutates_mixer"] is False
    assert any(call["cmd"] == cmd for call in decision.payload["blocked_calls"])


class FakeMixer:
    def __init__(self):
        self.fader = {}
        self.bus_fader = {1: -8.0}
        self.dca_fader = {2: -9.0}
        self.main_fader = {1: -2.0}
        self.channel_mute = {}
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

    def get_channel_mute(self, channel):
        return self.channel_mute.get(channel, False)

    def set_channel_mute(self, channel, muted):
        self.channel_mute[channel] = bool(muted)
        self.calls.append(("set_channel_mute", channel, self.channel_mute[channel]))
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


@dataclass
class InvalidCorrectionAction(TypedCorrectionAction):
    """Action that omits required contract behavior."""

    @property
    def family(self) -> str:
        raise NotImplementedError("invalid action family")

    @property
    def target_key(self) -> tuple[Any, ...]:
        return ("invalid",)


@dataclass
class InvalidTargetKeyAction(TypedCorrectionAction):
    """Action with valid family but invalid target key implementation."""

    @property
    def family(self) -> str:
        return ACTION_FAMILY_FX

    @property
    def target_key(self) -> tuple[Any, ...]:
        raise NotImplementedError("invalid target key")


def test_base_typed_action_contract_is_abstract():
    with pytest.raises(TypeError):
        TypedCorrectionAction(reason="invalid")


def test_invalid_action_contract_is_rejected_without_crash():
    controller = AutoFOHSafetyController(FakeMixer())

    decision = controller.execute(
        InvalidCorrectionAction(reason="bad contract"),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )

    assert decision.supported is False
    assert decision.allowed is False
    assert decision.sent is False
    assert "unsupported action family" in decision.message


def test_invalid_action_target_key_is_rejected_without_crash():
    controller = AutoFOHSafetyController(FakeMixer())

    decision = controller.execute(
        InvalidTargetKeyAction(reason="bad target key"),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )

    assert decision.supported is False
    assert decision.allowed is False
    assert decision.sent is False
    assert "invalid action target key" in decision.message


def test_mute_action_is_rejected_in_disallowed_state():
    controller = AutoFOHSafetyController(FakeMixer())

    decision = controller.execute(
        ChannelMuteMove(
            channel_id=1,
            muted=True,
            reason="prevent feedback path",
        ),
        runtime_state=RuntimeState.SONG_START_STABILIZE,
    )

    assert decision.allowed is False
    assert decision.sent is False
    assert decision.blocked is False
    assert "not allowed" in decision.message


def test_mute_action_is_quarantined_as_proposal_and_keeps_payload_shape():
    mixer = FakeMixer()
    controller = AutoFOHSafetyController(mixer)

    decision = controller.execute(
        ChannelMuteMove(
            channel_id=2,
            muted=True,
            reason="remove hot mic",
        ),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )

    assert_blocked_evidence(decision, "set_channel_mute")
    assert decision.bounded is False
    assert decision.payload["blocked_calls"][0] == {
        "cmd": "set_channel_mute",
        "channel": 2,
        "muted": True,
        "semantic": "channel_mute_move",
    }
    assert mixer.calls == []


def test_direct_mute_override_is_ignored_by_current_supervised_rail():
    mixer = FakeMixer()
    controller = AutoFOHSafetyController(
        mixer,
        config=AutoFOHSafetyConfig(
            allow_direct_mute_writes=True,
            channel_fader_min_interval_sec=0.0,
        ),
    )

    decision = controller.execute(
        ChannelMuteMove(
            channel_id=2,
            muted=True,
            reason="manual emergency mute",
        ),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )

    assert_blocked_evidence(decision, "set_channel_mute")
    assert decision.payload["direct_mute_override_ignored"] is True
    assert decision.payload["blocked_calls"][0]["muted"] is True
    assert mixer.calls == []


def test_mute_action_is_bounded_to_bool():
    mixer = FakeMixer()
    controller = AutoFOHSafetyController(mixer)

    decision = controller.execute(
        ChannelMuteMove(
            channel_id=2,
            muted=1,
            reason="coerce int mute command",
        ),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )

    assert_blocked_evidence(decision, "set_channel_mute")
    assert decision.bounded is True
    assert decision.payload["blocked_calls"][0]["muted"] is True


def test_mute_action_is_rate_limited_on_repeated_channel_commands():
    current_time = [100.0]

    def now():
        return current_time[0]

    mixer = FakeMixer()
    controller = AutoFOHSafetyController(mixer, time_provider=now)
    action = ChannelMuteMove(channel_id=5, muted=True, reason="hot mic guard")

    assert controller._is_rate_limited(action) is False

    controller._last_sent_at[action.target_key] = now()
    current_time[0] += 1.0
    assert controller._is_rate_limited(action) is True

    current_time[0] += 3.0
    assert controller._is_rate_limited(action) is False


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


def test_emergency_feedback_action_is_bounded_but_quarantined_in_emergency_state():
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

    assert_blocked_evidence(decision, "set_eq_band")
    assert decision.bounded is True
    assert decision.payload["blocked_calls"][0] == {
        "cmd": "set_eq_band",
        "channel": 3,
        "band": 4,
        "freq": 3150.0,
        "gain": -6.0,
        "q": 8.0,
        "ttl_seconds": 120.0,
        "semantic": "emergency_feedback_notch",
    }
    assert mixer.calls == []


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

    assert_blocked_evidence(first, "set_eq_band")
    assert_blocked_evidence(rollback, "set_eq_band")
    assert rollback.rate_limited is False
    assert mixer.calls == []


def test_safety_layer_quarantines_hpf_before_transport_support_matters():
    controller = AutoFOHSafetyController(NoHPFMixer())

    decision = controller.execute(
        HighPassAdjust(channel_id=2, freq_hz=120.0, reason="cleanup"),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )

    assert decision.supported is True
    assert_blocked_evidence(decision, "set_hpf")


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

    assert_blocked_evidence(decision, "set_eq_band")
    assert decision.bounded is True
    assert decision.payload["blocked_calls"][0] == {
        "cmd": "set_eq_band",
        "channel": 5,
        "band": 1,
        "freq": 90.0,
        "gain": 1.0,
        "q": 1.0,
        "semantic": "channel_eq_move",
    }
    assert mixer.calls == []


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

    assert_blocked_evidence(decision, "set_compressor_gain")
    assert decision.bounded is True
    assert decision.payload["blocked_calls"][0]["makeup_db"] == 12.0
    assert mixer.calls == []


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

    assert_blocked_evidence(cut, "set_main_fader")
    assert cut.bounded is True
    assert cut.payload["blocked_calls"][0]["value_db"] == -3.0
    assert mixer.calls == []


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

    assert_blocked_evidence(decision, "set_main_fader")
    assert decision.bounded is True
    assert decision.payload["blocked_calls"][0]["value_db"] == -6.0
    assert mixer.calls == []


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

    assert_blocked_evidence(bus, "set_bus_fader")
    assert bus.bounded is True
    assert_blocked_evidence(dca, "set_dca_fader")
    assert dca.bounded is True
    assert bus.payload["blocked_calls"][0]["value_db"] == -7.0
    assert dca.payload["blocked_calls"][0]["value_db"] == -8.5
    assert mixer.calls == []


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

    assert_blocked_evidence(eq_decision, "set_bus_eq_band")
    assert eq_decision.bounded is True
    assert eq_decision.payload["blocked_calls"][0]["gain"] == -1.0
    assert eq_decision.payload["blocked_calls"][0]["q"] == 0.44
    assert_blocked_evidence(compressor_decision, "set_bus_compressor")
    assert compressor_decision.bounded is True
    assert compressor_decision.payload["blocked_calls"][0]["threshold_db"] == -50.0
    assert compressor_decision.payload["blocked_calls"][0]["ratio"] == 20.0
    assert compressor_decision.payload["blocked_calls"][0]["makeup_db"] == 12.0
    assert mixer.calls == []


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

    assert_blocked_evidence(delay, "set_delay")
    assert delay.bounded is True
    assert_blocked_evidence(polarity, "set_polarity")
    assert delay.payload["blocked_calls"][0]["delay_ms"] == 1.25
    assert polarity.payload["blocked_calls"][0]["inverted"] is True
    assert mixer.calls == []


def test_autofoh_safety_source_has_no_direct_non_fader_gain_writes():
    source = Path("backend/autofoh_safety.py").read_text()

    assert "self.mixer_client.set_gain(" in source
    assert "self.mixer_client.set_fader(" in source
    for forbidden in (
        "self.mixer_client.set_channel_mute(",
        "self.mixer_client.set_bus_fader(",
        "self.mixer_client.set_dca_fader(",
        "self.mixer_client.set_main_fader(",
        "self.mixer_client.set_polarity(",
        "self.mixer_client.set_delay(",
        "self.mixer_client.set_eq_band(",
        "self.mixer_client.set_bus_eq_band(",
        "self.mixer_client.set_hpf(",
        "self.mixer_client.set_compressor(",
        "self.mixer_client.set_bus_compressor(",
        "self.mixer_client.set_compressor_gain(",
        "self.mixer_client.set_bus_compressor_gain(",
        "self.mixer_client.set_gate(",
        "self.mixer_client.set_pan(",
        "self.mixer_client.set_send_level(",
    ):
        assert forbidden not in source
    assert '"cmd": "set_eq_band"' in source
    assert '"cmd": "set_gate"' in source
