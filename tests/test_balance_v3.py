"""Balance v3: цели от вокала, решатель, страж вокала, дрейф, bleed-робастность."""

import numpy as np
import pytest

from heuristics.balance_v3 import (
    FaderSolution,
    GENRE_PRESETS,
    VocalGuardState,
    drift_correction_step,
    musical_dynamics_detected,
    solve_faders,
    target_vector_lufs,
    vocal_guard_step,
)
from heuristics.bleed_leakage import (
    LeakageMatrix,
    bleed_floor_db,
    gated_channel_loudness_db,
)


class TestTargetVector:
    def test_vocal_is_reference_on_ayaic_anchor(self):
        targets = target_vector_lufs({16: "lead_vocal", 8: "bass"}, "pop_rock")
        assert targets[16] == pytest.approx(-22.0)   # Ayaic lead plane
        assert targets[8] == pytest.approx(-25.0)    # −3 LU от вокала

    def test_genre_presets_change_hierarchy(self):
        rock = target_vector_lufs({1: "kick"}, "pop_rock")[1]
        edm = target_vector_lufs({1: "kick"}, "bass_forward")[1]
        jazz = target_vector_lufs({1: "kick"}, "vocal_forward")[1]
        assert edm > rock > jazz                     # kick-доминантность EDM

    def test_vocal_always_top_priority_in_all_presets(self):
        for genre, offsets in GENRE_PRESETS.items():
            assert offsets["lead_vocal"] == 0.0
            assert all(v <= 0.0 for v in offsets.values()), genre

    def test_unknown_role_falls_back_to_line(self):
        targets = target_vector_lufs({5: "theremin"}, "pop_rock")
        assert targets[5] == pytest.approx(-22.0 + GENRE_PRESETS["pop_rock"]["line"])


class TestFaderSolver:
    def test_converges_to_targets(self):
        measured = {16: -30.0, 8: -24.0, 1: -26.0}
        targets = target_vector_lufs(
            {16: "lead_vocal", 8: "bass", 1: "kick"}, "pop_rock")
        sol = solve_faders(measured, targets, anchor_channel=16)
        assert sol.converged
        # линейная модель: measured + fader == target (после нормализации
        # вниз разница между каналами сохраняется)
        result = {ch: measured[ch] + sol.faders_db[ch] for ch in measured}
        diff_vocal_bass = result[16] - result[8]
        assert diff_vocal_bass == pytest.approx(targets[16] - targets[8], abs=0.5)

    def test_fader_ceiling_normalizes_down(self):
        """Тихий вокал требует +8 дБ — весь вектор уезжает вниз, не выше 0."""
        measured = {16: -34.0, 9: -22.0}
        targets = target_vector_lufs({16: "lead_vocal", 9: "guitar"}, "pop_rock")
        sol = solve_faders(measured, targets, anchor_channel=16)
        assert max(sol.faders_db.values()) <= 0.0 + 1e-9
        assert any("нормализация вниз" in n for n in sol.notes)

    def test_bounded_move(self):
        measured = {16: -60.0}
        targets = {16: -22.0}
        sol = solve_faders(measured, targets, max_move_db=12.0)
        assert sol.faders_db[16] <= 12.0 + 1e-9
        assert not sol.converged  # честно: цель недостижима bounded-шагом

    def test_serializable(self):
        import json
        sol = solve_faders({1: -25.0}, {1: -25.0})
        json.dumps(sol.to_dict())


class TestVocalGuard:
    def test_boost_after_hold(self):
        state = VocalGuardState()
        # вокал тонет: дефицит 3 LU; hold 3 c должен пройти до подъёма
        for _ in range(3):
            vocal_guard_step(state, vocal_lu=-25.0, band_bus_lu=-24.0, dt_s=1.0)
        assert state.boost_db == 0.0 or state.boost_db <= 0.5
        for _ in range(4):
            vocal_guard_step(state, vocal_lu=-25.0, band_bus_lu=-24.0, dt_s=1.0)
        assert state.boost_db > 0.5

    def test_cap_and_rate(self):
        state = VocalGuardState()
        for _ in range(60):
            vocal_guard_step(state, vocal_lu=-30.0, band_bus_lu=-20.0, dt_s=1.0)
        assert state.boost_db <= 3.0 + 1e-9          # кламп +3 дБ

    def test_short_overlap_ignored(self):
        """Короткое музыкальное перекрытие (< hold) не триггерит гвард."""
        state = VocalGuardState()
        vocal_guard_step(state, -25.0, -24.0, dt_s=2.0)
        vocal_guard_step(state, -20.0, -26.0, dt_s=1.0)  # дефицит ушёл
        assert state.boost_db == 0.0
        assert state.deficit_seconds == 0.0

    def test_slow_release_with_hysteresis(self):
        state = VocalGuardState(boost_db=2.0)
        vocal_guard_step(state, vocal_lu=-20.0, band_bus_lu=-26.0, dt_s=1.0)
        assert 0.0 < state.boost_db < 2.0            # медленный возврат

    def test_inactive_vocal_frozen(self):
        state = VocalGuardState(boost_db=1.0)
        vocal_guard_step(state, -60.0, -20.0, dt_s=5.0, vocal_active=False)
        assert state.boost_db == 1.0


class TestDriftAndMusicalDynamics:
    def test_deadband_blocks_small_errors(self):
        assert drift_correction_step(-26.5, -25.0, 0.0, dt_s=1.0) == 0.0

    def test_rate_and_cap(self):
        step = drift_correction_step(-30.0, -25.0, 0.0, dt_s=1.0)
        assert step == pytest.approx(0.3)            # 0.3 дБ/с
        step_at_cap = drift_correction_step(-30.0, -25.0, 2.9, dt_s=1.0)
        assert step_at_cap == pytest.approx(0.1)     # кламп ±3 суммарно

    def test_musical_dynamics_freeze(self):
        assert drift_correction_step(
            -30.0, -25.0, 0.0, dt_s=1.0, musical_dynamics=True) == 0.0

    def test_correlated_drop_is_music(self):
        drops = {ch: -3.0 for ch in range(8)}
        assert musical_dynamics_detected(drops)
        mixed = {0: -3.0, 1: 0.2, 2: 0.1, 3: -0.2, 4: 0.0}
        assert not musical_dynamics_detected(mixed)   # один канал упал — дрейф


class TestBleedRobustness:
    def test_bleed_floor_percentile(self):
        assert bleed_floor_db([-50.0, -48.0, -52.0, -20.0]) < -40.0  # выброс не рулит

    def test_gated_loudness_ignores_bleed_blocks(self):
        floor = -45.0
        blocks = [-44.0, -43.0, -20.0, -19.0]        # первые два — bleed
        gated = gated_channel_loudness_db(blocks, floor, margin_db=6.0)
        assert gated == pytest.approx(10 * np.log10(
            np.mean([10 ** (-2.0), 10 ** (-1.9)])), abs=0.2)

    def test_silent_instrument_returns_none(self):
        assert gated_channel_loudness_db([-44.0, -43.5], -45.0) is None

    def test_leakage_matrix_recovers_true_levels(self):
        """Вокальный мик ловит барабаны −10 дБ; NNLS вытаскивает правду."""
        solos = {
            "voc": {"voc": 0.0, "drum": -30.0},   # соло вокала: в драм-мик −30
            "drum": {"drum": 0.0, "voc": -10.0},  # соло барабанов: в вокал-мик −10
        }
        lm = LeakageMatrix.from_solo_measurements(solos)
        # истина: вокал −30 дБ, барабаны −20 дБ; вокальный мик слышит смесь
        true_voc, true_drum = 10 ** (-3.0), 10 ** (-2.0)
        measured = {
            "voc": 10 * np.log10(true_voc + true_drum * 10 ** (-1.0)),
            "drum": 10 * np.log10(true_drum + true_voc * 10 ** (-3.0)),
        }
        est = lm.estimate_true_levels_db(measured)
        assert est["voc"] == pytest.approx(-30.0, abs=1.0)
        assert est["drum"] == pytest.approx(-20.0, abs=1.0)

    def test_bleed_contaminated_vocal_not_underestimated(self):
        """Без коррекции утечки вокал казался бы громче (drum bleed) — после
        NNLS его истинный уровень ниже загрязнённого измерения."""
        solos = {
            "voc": {"voc": 0.0, "drum": -40.0},
            "drum": {"drum": 0.0, "voc": -8.0},
        }
        lm = LeakageMatrix.from_solo_measurements(solos)
        contaminated_voc = 10 * np.log10(10 ** (-3.0) + 10 ** (-2.0) * 10 ** (-0.8))
        measured = {"voc": contaminated_voc, "drum": -20.0}
        est = lm.estimate_true_levels_db(measured)
        assert est["voc"] < contaminated_voc - 1.0


def test_single_headroom_limited_channel_does_not_drag_mix_down():
    """Live 2026-07-06: kick с тримом −12 требовал фейдер +12 — старая
    нормализация стаскивала ВЕСЬ микс на −12 дБ каждый прогон
    (−34 -> −38.5 LUFS). Не-якорный дефицитный канал клампится на потолке,
    остальные остаются на плоскостях."""
    measured = {16: -22.0, 1: -34.0, 8: -25.0, 9: -25.0}   # kick глубоко в минусе
    targets = target_vector_lufs(
        {16: "lead_vocal", 1: "kick", 8: "bass", 9: "guitar"}, "pop_rock")
    sol = solve_faders(measured, targets, anchor_channel=16)
    assert sol.faders_db[1] <= 0.0 + 1e-9          # kick на потолке
    assert abs(sol.residual_lu[1]) > 1.0           # честный дефицит kick
    # вокал/бас/гитара достигли целей БЕЗ глобального сдвига
    for ch in (16, 8, 9):
        assert abs(sol.residual_lu[ch]) <= 1.0
    assert any("клампированы" in n for n in sol.notes)
