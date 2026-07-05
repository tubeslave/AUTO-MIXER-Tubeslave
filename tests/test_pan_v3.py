"""Pan v3: relevance-gate, якоря, слоты, пары, верификация, live-freeze."""

import numpy as np
import pytest

from heuristics.pan_v3 import (
    LivePanState,
    PanConfig,
    PanSolution,
    assign_pans,
    lcr_gains,
    lf_dominant,
    mono_fold_metrics,
    pan_gains,
    verify_pan_solution,
    width_budget,
)

SR = 48000
RNG = np.random.default_rng(41)


def _tone(freq, amp=0.3, n_sec=1.0):
    t = np.arange(int(n_sec * SR)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


class TestRelevanceGate:
    def test_mono_pa_bypasses_all_to_center(self):
        """Главный устранитель вреда: mono PA → всё в центр."""
        config = PanConfig(pa_type="mono")
        sol = assign_pans({1: {"role": "guitar"}, 2: {"role": "keys"}}, config)
        assert sol.bypass
        assert all(p == 0.0 for p in sol.pans.values())

    def test_default_config_is_conservative_mono(self):
        """Venue-профиль неизвестен → по умолчанию считаем PA моно."""
        assert PanConfig().pa_type == "mono"

    def test_width_budget_by_venue(self):
        assert width_budget(PanConfig(pa_type="stereo", venue_class="club")) == 0.50
        assert width_budget(PanConfig(pa_type="stereo", venue_class="festival")) == 0.25
        assert width_budget(PanConfig(pa_type="stereo", venue_class="theater")) == 0.70
        assert width_budget(PanConfig(pa_type="mono")) == 0.0

    def test_override_clamped(self):
        cfg = PanConfig(pa_type="stereo", width_cap_override=1.5)
        assert width_budget(cfg) == 0.85


class TestAnchors:
    CFG = PanConfig(pa_type="stereo", venue_class="club")

    def test_criticals_centered(self):
        sol = assign_pans({
            16: {"role": "lead_vocal"}, 1: {"role": "kick"},
            8: {"role": "bass"}, 2: {"role": "snare"},
        }, self.CFG)
        assert all(sol.pans[ch] == 0.0 for ch in (16, 1, 8, 2))
        assert all("critical" in sol.anchored[ch] for ch in (16, 1, 8, 2))

    def test_lf_dominant_centered(self):
        sub_synth = _tone(60.0, amp=0.5)
        sol = assign_pans({5: {"role": "keys", "audio": sub_synth}}, self.CFG)
        assert sol.pans[5] == 0.0
        assert sol.anchored[5] == "lf_dominant_center"

    def test_lf_dominant_function(self):
        assert lf_dominant(_tone(60.0), SR)
        assert not lf_dominant(_tone(2000.0), SR)

    def test_snare_offset_capped(self):
        cfg = PanConfig(pa_type="stereo", snare_offset=0.5)
        sol = assign_pans({2: {"role": "snare"}}, cfg)
        assert sol.pans[2] <= 0.10


class TestSlotsAndPairs:
    CFG = PanConfig(pa_type="stereo", venue_class="club")   # cap 0.5

    def test_movable_within_cap_alternating_sides(self):
        channels = {i: {"role": "guitar", "audio": _tone(300.0 * (i + 1))}
                    for i in range(4)}
        sol = assign_pans(channels, self.CFG, SR)
        pans = [sol.pans[i] for i in range(4)]
        assert all(abs(p) <= 0.50 + 1e-9 for p in pans)
        assert any(p > 0 for p in pans) and any(p < 0 for p in pans)
        # соседи по центроиду — на противоположных сторонах
        assert np.sign(pans[0]) != np.sign(pans[1])

    def test_stereo_pair_symmetric_width_only(self):
        channels = {
            "ohl": {"role": "overheads", "stereo_pair": "oh"},
            "ohr": {"role": "overheads", "stereo_pair": "oh"},
        }
        sol = assign_pans(channels, self.CFG, SR)
        assert sol.pans["ohl"] == pytest.approx(-sol.pans["ohr"])
        assert abs(sol.pans["ohl"]) == pytest.approx(0.5 * 0.7)  # cap·0.7
        assert "oh" in sol.pair_widths
        assert {"ohl", "ohr"} <= sol.pair_channels

    def test_audience_perspective_flip(self):
        cfg = PanConfig(pa_type="stereo", venue_class="club",
                        audience_perspective=False)
        channels = {
            "ohl": {"role": "overheads", "stereo_pair": "oh"},
            "ohr": {"role": "overheads", "stereo_pair": "oh"},
        }
        sol = assign_pans(channels, cfg, SR)
        assert sol.pans["ohl"] > 0  # перспектива исполнителя — зеркально


class TestPanLaw:
    def test_constant_power(self):
        for pan in (-1.0, -0.5, 0.0, 0.5, 1.0):
            g_l, g_r = pan_gains(pan)
            assert g_l ** 2 + g_r ** 2 == pytest.approx(1.0)
        g_l, g_r = pan_gains(0.0)
        assert g_l == pytest.approx(g_r)          # −3 дБ в центре

    def test_lcr_csr(self):
        center = lcr_gains(0.0, csr=1.0)
        assert center["C"] > 0 and center["L"] == pytest.approx(0.0)
        no_center = lcr_gains(0.0, csr=0.0)
        assert no_center["C"] == pytest.approx(0.0)


class TestVerification:
    def test_balanced_solution_passes(self):
        # широкополосные источники (программный материал), симметричные паны
        channels_audio = {
            "a": 0.3 * RNG.normal(0, 1, SR),
            "b": 0.3 * RNG.normal(0, 1, SR),
            "voc": 0.3 * RNG.normal(0, 1, SR),
        }
        pans = {"a": -0.3, "b": +0.3, "voc": 0.0}
        result = verify_pan_solution(channels_audio, pans, SR)
        assert result["passed"], result

    def test_lopsided_solution_fails(self):
        channels_audio = {
            "a": _tone(500.0, amp=0.6),
            "b": _tone(800.0, amp=0.6),
        }
        pans = {"a": -0.9, "b": -0.9}      # всё в одну сторону
        result = verify_pan_solution(channels_audio, pans, SR)
        assert not result["passed"]
        assert result["headroom_diff_db"] > 2.0

    def test_mono_fold_metrics(self):
        x = _tone(440.0)
        metrics = mono_fold_metrics(x, x)
        assert metrics["correlation"] == pytest.approx(1.0)
        assert metrics["headroom_diff_db"] == pytest.approx(0.0)


class TestLiveFreeze:
    def _state(self):
        return LivePanState(snapshot_pans={"gtr": 0.4, "keys": -0.3})

    def test_spotlight_requires_armed_and_confidence(self):
        state = self._state()
        trim = state.solo_spotlight_step("gtr", 1.0, solo_active=True,
                                         confidence=0.95, operator_armed=False)
        assert trim == 0.0                 # без arm — заморожено
        trim = state.solo_spotlight_step("gtr", 1.0, solo_active=True,
                                         confidence=0.5, operator_armed=True)
        assert trim == 0.0                 # низкая уверенность

    def test_spotlight_rate_and_cap(self):
        state = self._state()
        for _ in range(10):
            trim = state.solo_spotlight_step(
                "gtr", 1.0, solo_active=True,
                confidence=0.95, operator_armed=True)
        assert trim <= 0.0                 # к центру от +0.4
        assert abs(trim) <= 0.15 + 1e-9    # суммарный дрейф ≤15%

    def test_symmetry_guard_alert_after_hold(self):
        state = self._state()
        for _ in range(59):
            assert state.symmetry_guard_step(3.0, 1.0) is None
        alert = state.symmetry_guard_step(3.0, 1.0)
        assert alert is not None and alert["name"] == "LR_SYMMETRY_DRIFT"
        assert not alert["auto_applied"]   # без arm — только оповещение

    def test_symmetry_guard_armed_trim_bounded(self):
        state = self._state()
        for _ in range(61):
            alert = state.symmetry_guard_step(3.0, 1.0, operator_armed=True)
        assert alert is not None
        assert abs(alert["trim"]) <= 0.10

    def test_rollback(self):
        state = self._state()
        for _ in range(5):
            state.solo_spotlight_step("gtr", 1.0, solo_active=True,
                                      confidence=0.95, operator_armed=True)
        pans = state.rollback()
        assert pans == state.snapshot_pans
        assert state.trims == {}
