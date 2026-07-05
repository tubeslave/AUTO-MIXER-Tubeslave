"""Balance v3: конфигурируемый опорный канал и вырожденные случаи.

Пять сценарных кейсов спецификации + юниты: rebase, fallback, anchor_active
gate, guard на шине, segment_type в Dugan, drift-precheck, инструментальный
солвер.
"""

import numpy as np
import pytest

from heuristics.balance_v3 import (
    AnchorActivityGate,
    BalanceConfig,
    GENRE_PRESETS,
    INSTRUMENTAL_GUARD_MARGIN_LU,
    ROLE_ANCHOR_LUFS,
    VocalGuardState,
    build_target_vector,
    drift_correction_step,
    drift_precheck,
    make_guard_state,
    musical_dynamics_detected,
    rebase_targets,
    resolve_anchor,
    select_drift_reference,
    solve_faders,
    vocal_guard_step,
)
from dugan_gain_sharing import DuganGainSharing


class TestAnchorConfig:
    def test_rebase_changes_coordinates_not_hierarchy(self):
        targets = {"lead_vocal": 0.0, "kick": -4.0, "bass": -3.0}
        rebased = rebase_targets(targets, "kick")
        assert rebased["kick"] == 0.0
        assert rebased["lead_vocal"] == pytest.approx(4.0)
        # относительные разницы сохранены (система координат, не баланс)
        assert (rebased["lead_vocal"] - rebased["bass"]) == pytest.approx(
            targets["lead_vocal"] - targets["bass"])

    def test_rebase_missing_anchor_raises(self):
        with pytest.raises(KeyError):
            rebase_targets({"kick": -4.0}, "theremin")

    def test_missing_anchor_falls_back_with_warning(self):
        config = BalanceConfig()  # anchor lead_vocal
        anchor, warning = resolve_anchor(config, {"kick", "snare", "bass"})
        assert anchor == "kick"       # первый доступный из fallback
        assert warning is not None

    def test_configured_anchor_used_when_active(self):
        anchor, warning = resolve_anchor(
            BalanceConfig(), {"lead_vocal", "kick"})
        assert anchor == "lead_vocal" and warning is None

    def test_custom_targets_override_genre_preset(self):
        config = BalanceConfig(targets_LU={"lead_vocal": 0.0, "kick": -1.0})
        targets, anchor, _ = build_target_vector(
            config, {16: "lead_vocal", 1: "kick"})
        assert anchor == "lead_vocal"
        assert targets[1] - targets[16] == pytest.approx(-1.0)

    def test_anchor_absolute_plane_follows_role(self):
        """Инструментальный якорь ложится на СВОЮ Ayaic-плоскость."""
        config = BalanceConfig(anchor_channel="kick")
        targets, anchor, _ = build_target_vector(config, {1: "kick", 8: "bass"})
        assert anchor == "kick"
        assert targets[1] == pytest.approx(ROLE_ANCHOR_LUFS["kick"])


class TestScenario1VocalSilentWholeTrack:
    """Кейс 1: вокал молчит весь трек → фейдеры не двигаются от снапшота."""

    def test_guard_never_wakes_and_drift_holds(self):
        gate = AnchorActivityGate(bleed_floor_snapshot_db=-48.0)
        guard = VocalGuardState()
        for _ in range(120):
            active = gate.step(short_term_lufs=-47.0,        # только bleed
                               classifier_confirmed=False, dt_s=1.0)
            vocal_guard_step(guard, -47.0, -20.0, dt_s=1.0, vocal_active=active)
        assert not gate.active
        assert guard.boost_db == 0.0
        # дрейф-слой: канал на снапшот-цели — нулевые шаги
        assert drift_correction_step(-25.0, -25.0, 0.0, dt_s=1.0) == 0.0


class TestScenario2VocalDropsFor32Bars:
    """Кейс 2: вокал пропадает на проигрыше → guard спит, фундамент
    неподвижен, при возврате просыпается без скачка."""

    def test_sleep_and_wake_without_jump(self):
        gate = AnchorActivityGate(bleed_floor_snapshot_db=-48.0)
        guard = VocalGuardState(boost_db=1.5)  # накоплен до проигрыша

        # проигрыш ~60 с: вокал на bleed-уровне
        for _ in range(60):
            active = gate.step(-47.5, classifier_confirmed=False, dt_s=1.0)
            vocal_guard_step(guard, -47.5, -18.0, dt_s=1.0, vocal_active=active)
        assert guard.boost_db == pytest.approx(1.5)   # boost заморожен, скачка нет
        # фундамент (kick на цели) не трогается
        assert drift_correction_step(-25.2, -25.0, 0.0, dt_s=1.0) == 0.0

        # вокал вернулся: активация только после 2 с подтверждения
        assert not gate.step(-24.0, classifier_confirmed=True, dt_s=1.0)
        assert gate.step(-24.0, classifier_confirmed=True, dt_s=1.5)
        before = guard.boost_db
        vocal_guard_step(guard, -24.0, -20.0, dt_s=1.0, vocal_active=gate.active)
        # без скачка: изменение ограничено rate-limit'ом
        assert abs(guard.boost_db - before) <= 0.5 + 1e-9


class TestScenario3Duet:
    """Кейс 3: дуэт → оба лида target 0, guard по шине, Dugan молчит на пении."""

    def test_both_leads_get_zero_target(self):
        config = BalanceConfig()
        targets, anchor, _ = build_target_vector(
            config, {16: "lead_vocal", 17: "lead_vocal", 8: "bass"})
        assert targets[16] == targets[17] == pytest.approx(
            ROLE_ANCHOR_LUFS["lead_vocal"])

    def test_dugan_disabled_on_vocal_segments(self):
        sharing = DuganGainSharing(time_provider=lambda: 1.0)
        levels = {"lead_a": -20.0, "lead_b": -30.0}
        assert sharing.update(levels, segment_type="vocal") == {}
        assert sharing.update(levels, segment_type="music") == {}
        # на речевом сегменте слой работает
        assert sharing.update(levels, segment_type="speech") != {}

    def test_guard_bus_configured(self):
        config = BalanceConfig(guard_bus="vocal_dca")
        assert config.guard_bus == "vocal_dca"
        state = make_guard_state(config)
        assert state is not None and state.margin_lu == pytest.approx(2.5)


class TestScenario4Instrumental:
    """Кейс 4: инструментальный сет → солвер сходится с любым якорем,
    guard с инструментальным margin."""

    def test_solver_converges_with_instrument_anchor(self):
        config = BalanceConfig(anchor_channel="lead_instrument")
        roles = {5: "lead_instrument", 1: "kick", 8: "bass", 9: "guitar"}
        targets, anchor, warning = build_target_vector(config, roles)
        assert anchor == "lead_instrument" and warning is None
        measured = {5: -28.0, 1: -24.0, 8: -26.0, 9: -22.0}
        sol = solve_faders(measured, targets)
        assert sol.converged
        assert sol.passes <= 3
        result = {ch: measured[ch] + sol.faders_db[ch] for ch in measured}
        assert (result[5] - result[1]) == pytest.approx(
            targets[5] - targets[1], abs=0.5)

    def test_guard_reassigned_with_instrumental_margin(self):
        config = BalanceConfig(anchor_channel="lead_instrument")
        state = make_guard_state(config, instrumental=True)
        assert state is not None
        assert state.margin_lu == pytest.approx(INSTRUMENTAL_GUARD_MARGIN_LU)

    def test_guard_disable_option(self):
        config = BalanceConfig(guard_reassign_on_instrumental=False)
        assert make_guard_state(config, instrumental=True) is None
        config2 = BalanceConfig(guard_enabled=False)
        assert make_guard_state(config2) is None

    def test_dugan_announce_only(self):
        sharing = DuganGainSharing(time_provider=lambda: 1.0)
        levels = {"announce": -25.0, "kick": -20.0}
        out = sharing.update(levels, segment_type="speech",
                             enabled_channels=frozenset({"announce"}))
        assert set(out) == {"announce"}
        # нет announce-микрофонов -> слой выключен
        sharing2 = DuganGainSharing(time_provider=lambda: 1.0)
        assert sharing2.update(levels, segment_type="speech",
                               enabled_channels=frozenset()) == {}


class TestScenario5QuietSection:
    """Кейс 5: общая тихая секция (все −6 дБ коррелированно) → freeze."""

    def test_correlated_drop_freezes_all_corrections(self):
        deltas = {ch: -6.0 for ch in range(8)}
        assert drift_precheck(-6.0, deltas) == "freeze_musical"
        assert musical_dynamics_detected(deltas)  # подтверждающий этап
        step = drift_correction_step(
            -31.0, -25.0, 0.0, dt_s=1.0, musical_dynamics=True)
        assert step == 0.0  # никаких коррекций «вверх»


class TestDriftReference:
    def test_argmin_sigma_selects_stable_channel(self):
        series = {
            "kick": [-25.0, -25.2, -24.9, -25.1, -25.0],
            "vocal": [-22.0, -30.0, -24.0, -35.0, -23.0],
        }
        assert select_drift_reference(series) == "kick"

    def test_manual_override(self):
        assert select_drift_reference({}, manual_override="bass") == "bass"

    def test_short_series_skipped(self):
        assert select_drift_reference({"x": [-25.0, -25.0]}) is None

    def test_precheck_individual_drift(self):
        deltas = {"gtr": -2.5, "kick": 0.1, "bass": -0.2}
        assert drift_precheck(0.2, deltas) == "correct_individual_drift"

    def test_precheck_ambiguous_falls_to_full_check(self):
        deltas = {"gtr": -2.5, "kick": 2.4, "bass": 0.1}
        assert drift_precheck(-2.0, deltas) == "full_check"


class TestDriftFormulaRegression:
    """Пункт 2 спецификации: ошибка строго от снапшота, не от якоря."""

    def test_error_is_snapshot_absolute_not_anchor_relative(self):
        # якорь «дышит» (вокал стал громче) — фундамент на снапшот-цели
        # обязан остаться неподвижным: формула не содержит уровня якоря.
        step = drift_correction_step(
            measured_lu=-25.0, snapshot_target_lu=-25.0,
            accumulated_correction_db=0.0, dt_s=1.0)
        assert step == 0.0  # kick не «дышит» вслед за динамикой вокала
