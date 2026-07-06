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
    # запаркованный канал (fader-mute — операторский выбор) не поднимается
    # вовсе, даже в саундчек-деривации
    assert mixer.calls[-1] == ("set_fader", 1, -100.0)


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

    # live-состояние (VERSE): пошаговый лимит 1 дБ от текущего значения
    decision = controller.execute(
        ChannelEQMove(
            channel_id=5,
            band=1,
            freq_hz=90.0,
            gain_db=6.0,
            q=1.0,
            reason="too much low boost",
        ),
        runtime_state=RuntimeState.VERSE,
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

    # саундчек-деривация: абсолютный кламп ±soundcheck_eq_max_abs_db,
    # без пошагового бюджета (one-shot цели)
    derivation = controller.execute(
        ChannelEQMove(
            channel_id=5,
            band=1,
            freq_hz=90.0,
            gain_db=-12.0,
            q=1.0,
            reason="resonance cut",
        ),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )
    assert derivation.bounded is True
    assert derivation.payload["blocked_calls"][0]["gain"] == -9.0
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
    # деривация: абсолютная цель clip(0.0, [-12, 0]), но буст всей группы
    # ограничен шагом +6 дБ от текущих -9.0 -> -3.0
    assert dca.payload["blocked_calls"][0]["value_db"] == -3.0
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

    # live-состояние: шаг задержки ограничен 0.25 мс от текущей (1.0)
    delay = controller.execute(
        DelayAdjust(channel_id=1, delay_ms=2.0, reason="drum phase"),
        runtime_state=RuntimeState.VERSE,
    )
    polarity = controller.execute(
        PolarityAdjust(channel_id=2, inverted=True, reason="polarity"),
        runtime_state=RuntimeState.VERSE,
    )

    assert_blocked_evidence(delay, "set_delay")
    assert delay.bounded is True
    assert_blocked_evidence(polarity, "set_polarity")
    assert delay.payload["blocked_calls"][0]["delay_ms"] == 1.25
    assert polarity.payload["blocked_calls"][0]["inverted"] is True
    assert mixer.calls == []

    # саундчек-деривация: абсолютная цель в пределах delay_max_ms
    derivation = controller.execute(
        DelayAdjust(channel_id=1, delay_ms=4.7, reason="drum phase align"),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )
    assert derivation.payload["blocked_calls"][0]["delay_ms"] == 4.7
    capped = controller.execute(
        DelayAdjust(channel_id=1, delay_ms=25.0, reason="cap check"),
        runtime_state=RuntimeState.SOURCE_LEARNING,
    )
    assert capped.payload["blocked_calls"][0]["delay_ms"] == 10.0
    assert mixer.calls == []


def test_autofoh_safety_default_config_keeps_non_fader_gain_writes_quarantined():
    """Трансляторы для EQ/comp/pan/delay/... существуют (операторский промоут
    2026-07-05), но БЕЗ промоута в promoted_direct_write_action_types любая
    такая запись блокируется proposal-only политикой — mixer не трогается."""
    mixer = FakeMixer()
    controller = AutoFOHSafetyController(mixer)

    quarantined_actions = (
        ChannelEQMove(channel_id=1, band=1, freq_hz=500.0, gain_db=-3.0, q=2.0,
                      reason="eq"),
        PolarityAdjust(channel_id=1, inverted=True, reason="pol"),
        DelayAdjust(channel_id=1, delay_ms=1.0, reason="dly"),
        HighPassAdjust(channel_id=1, freq_hz=100.0, reason="hpf"),
    )
    for action in quarantined_actions:
        decision = controller.execute(action, RuntimeState.SOURCE_LEARNING)
        assert decision.blocked is True, action.action_type
        assert decision.sent is False, action.action_type
        assert decision.write_policy == "proposal_only_quarantined"
    assert mixer.calls == []

    # мьюты/мастер/шинные записи транслятора не имеют вовсе —
    # промоут в конфиге их не оживит
    source = Path("backend/autofoh_safety.py").read_text()
    translator_src = source.split("def _translator_for")[1].split("def _require_method")[0]
    for forbidden in (
        "set_channel_mute(",
        "set_bus_fader(",
        "set_main_fader(",
        "set_bus_eq_band(",
        "set_bus_compressor(",
    ):
        assert forbidden not in translator_src
    # DCA-фейдер промоутнут оператором 2026-07-06 (финальный стем-баланс) —
    # транслятор существует, но без промоута в конфиге остаётся proposal-only
    assert "set_dca_fader(" in translator_src
    assert '"cmd": "set_eq_band"' in source
    assert '"cmd": "set_gate"' in source


def test_promoted_direct_write_types_allow_gated_console_writes():
    """Операторский промоут: типы из promoted_direct_write_action_types
    проходят до транслятора и реально пишутся."""
    mixer = FakeMixer()
    controller = AutoFOHSafetyController(
        mixer,
        config=AutoFOHSafetyConfig(
            promoted_direct_write_action_types=("PolarityAdjust",),
        ),
    )
    decision = controller.execute(
        PolarityAdjust(channel_id=2, inverted=True, reason="pol"),
        RuntimeState.SOURCE_LEARNING,
    )
    assert decision.blocked is False
    assert decision.sent is True
    assert ("set_polarity", 2, True) in mixer.calls

    # непромоутнутый тип по-прежнему в карантине
    blocked = controller.execute(
        DelayAdjust(channel_id=1, delay_ms=1.0, reason="dly"),
        RuntimeState.SOURCE_LEARNING,
    )
    assert blocked.blocked is True
    assert len(mixer.calls) == 1


def test_osc_throttle_drop_is_retried_once():
    """WING OSC-троттл молча дропает запись (send False): контроллер
    ретраит с паузой больше интервала троттлера (live 2026-07-06:
    mirror EQ терял записи как «send returned false»)."""
    class FlakySendMixer(FakeMixer):
        def __init__(self):
            super().__init__()
            self.fail_next = 1

        def set_fader(self, channel, value_db):
            if self.fail_next > 0:
                self.fail_next -= 1
                self.calls.append(("set_fader_dropped", channel, value_db))
                return False
            return super().set_fader(channel, value_db)

    mixer = FlakySendMixer()
    controller = AutoFOHSafetyController(mixer)
    decision = controller.execute(
        ChannelFaderMove(channel_id=1, target_db=-9.5, reason="throttle test"),
        RuntimeState.SNAPSHOT_LOCK,
    )
    assert decision.sent is True
    assert decision.payload.get("osc_throttle_retry") is True
    assert ("set_fader", 1, -9.5) in mixer.calls


def test_fader_absolute_target_in_derivation_stepped_in_live():
    """Оператор 2026-07-06: «очень слабая балансировка фейдерами для
    саундчека» — в деривации фейдер идёт к АБСОЛЮТНОЙ цели решателя
    (потолок 0 dBFS), микрошаги — только в live."""
    mixer = FakeMixer()
    controller = AutoFOHSafetyController(
        mixer,
        config=AutoFOHSafetyConfig(
            channel_fader_max_step_db=1.0,
            promoted_direct_write_action_types=("ChannelFaderMove",),
        ),
    )
    mixer.fader[1] = -2.0

    # деривация: сразу к цели -10.5
    derivation = controller.execute(
        ChannelFaderMove(channel_id=1, target_db=-10.5, reason="balance v3"),
        RuntimeState.SNAPSHOT_LOCK,
    )
    assert derivation.sent is True
    assert mixer.fader[1] == pytest.approx(-10.5)

    # подъём в деривации — шагом <= 6 дБ (открытый микрофон не взлетает)
    boost = controller.execute(
        ChannelFaderMove(channel_id=1, target_db=3.0, reason="boost check"),
        RuntimeState.SOURCE_LEARNING,
    )
    assert boost.action.target_db == pytest.approx(-4.5)  # -10.5 + 6

    # потолок 0 dBFS абсолютен и в деривации
    mixer.fader[3] = -2.0
    ceiling = controller.execute(
        ChannelFaderMove(channel_id=3, target_db=3.0, reason="ceiling check"),
        RuntimeState.SOURCE_LEARNING,
    )
    assert ceiling.action.target_db == pytest.approx(0.0)

    # live (VERSE): прежний микрошаг 1 дБ от текущего
    mixer.fader[2] = -2.0
    live = controller.execute(
        ChannelFaderMove(channel_id=2, target_db=-9.0, reason="live step"),
        RuntimeState.VERSE,
    )
    assert live.action.target_db == pytest.approx(-3.0)
