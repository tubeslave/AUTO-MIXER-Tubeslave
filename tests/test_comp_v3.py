"""Comp v3: ядро Giannoulis, статистики, гейт «не сжимать», верификация, live-супервизор."""

import numpy as np
import pytest

from heuristics.drc_giannoulis import (
    lf_release_guard_ms,
    pumping_metric_db,
    simulate_compressor,
    soft_knee_gain_db,
    smooth_decoupled_detector,
)
from heuristics.comp_v3 import (
    LiveGRSupervisorState,
    channel_dynamics_stats,
    derive_params,
    live_gr_supervise,
    safe_generic_params,
    should_compress,
    soundcheck_comp_v3,
)
from comp_params_ma2015 import (
    ballistics_giannoulis2013,
    knee_width_ma2015,
    low_frequency_weighting_multitrack,
    percussivity_weighting_multitrack,
)

SR = 48000
RNG = np.random.default_rng(11)


def _tone(freq, n_sec=2.0, amp=0.3):
    t = np.arange(int(n_sec * SR)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def _dynamic_vocal(n_sec=4.0):
    """«Нестабильный вокал»: чередование тихих и громких фраз."""
    t = np.arange(int(n_sec * SR)) / SR
    envelope = np.where((t % 2.0) < 1.0, 0.08, 0.5)
    return envelope * np.sin(2 * np.pi * 220.0 * t)


class TestGiannoulisCore:
    def test_soft_knee_regions(self):
        # ниже колена — без ослабления
        assert soft_knee_gain_db(-40.0, -20.0, 4.0, 6.0) == pytest.approx(0.0)
        # выше колена — GR = (x-T)*(1-1/R)
        assert soft_knee_gain_db(-10.0, -20.0, 4.0, 6.0) == pytest.approx(
            10.0 * (1 - 1 / 4.0))
        # непрерывность на краях колена (T±W/2) и монотонный рост внутри
        assert soft_knee_gain_db(-23.0, -20.0, 4.0, 6.0) == pytest.approx(0.0)
        assert soft_knee_gain_db(-17.0, -20.0, 4.0, 6.0) == pytest.approx(
            3.0 * (1 - 1 / 4.0))
        mid = soft_knee_gain_db(-20.0, -20.0, 4.0, 6.0)  # колено начинается ДО порога
        assert 0.0 < mid < 3.0 * (1 - 1 / 4.0)

    def test_hard_knee_degenerate(self):
        assert soft_knee_gain_db(-15.0, -20.0, 2.0, 0.0) == pytest.approx(2.5)

    def test_detector_attack_release_asymmetry(self):
        env_rate = 3000.0
        step = np.concatenate([np.zeros(50), np.full(100, 6.0), np.zeros(400)])
        gr = smooth_decoupled_detector(step, attack_ms=5.0, release_ms=100.0, env_rate_hz=env_rate)
        rise_t = np.argmax(gr > 3.0)          # до половины на атаке
        fall_region = gr[150:]
        fall_t = np.argmax(fall_region < 3.0)  # до половины на релизе
        assert rise_t < fall_t                 # релиз медленнее атаки
        assert gr.max() <= 6.0 + 1e-6

    def test_simulation_reduces_level_variance(self):
        """Компрессор сжимает ДИНАМИКУ: разброс блочных уровней падает."""
        x = _dynamic_vocal()
        sim = simulate_compressor(x, SR, threshold_db=-20.0, ratio=4.0,
                                  knee_db=6.0, attack_ms=5.0, release_ms=100.0)

        def sigma_level(v):
            block = SR // 4
            lv = [20 * np.log10(np.sqrt(np.mean(v[i:i + block] ** 2)) + 1e-12)
                  for i in range(0, len(v) - block, block)]
            return float(np.std(lv))

        assert sigma_level(sim.output) < sigma_level(x) - 1.0
        assert sim.mean_gr_db > 0.0

    def test_pumping_metric_detects_slow_modulation(self):
        env_rate = 3000.0
        t = np.arange(int(env_rate * 4)) / env_rate
        pumping_gain = -3.0 * (1 + np.sin(2 * np.pi * 4.0 * t))  # 4 Гц — худший случай
        steady_gain = np.full_like(t, -3.0)
        assert pumping_metric_db(pumping_gain, env_rate) > pumping_metric_db(steady_gain, env_rate) + 0.1

    def test_lf_release_guard(self):
        assert lf_release_guard_ms(50.0, 40.0) == pytest.approx(50.0)   # 25 мс < 50
        assert lf_release_guard_ms(10.0, 40.0) == pytest.approx(25.0)   # поднят до периода


class TestMaMultitrackFeatures:
    def test_percussivity_relative_range(self):
        w = percussivity_weighting_multitrack({"drum": 22.0, "pad": 4.0, "voc": 12.0})
        assert w["drum"] > 1.0 > w["pad"]          # перкуссивнее среднего > 1
        assert all(0.0 < v < 2.0 for v in w.values())

    def test_lf_weighting_normalized_by_mean(self):
        w = low_frequency_weighting_multitrack({"bass": 0.9, "hat": 0.05, "gtr": 0.4})
        assert w["bass"] > 1.0 > w["hat"]

    def test_ballistics_2013_formula(self):
        # crest 12 дБ -> x_crest≈3.98 -> tau_A≈10 мс, tau_R≈126 мс
        atk, rel = ballistics_giannoulis2013(12.0)
        assert atk == pytest.approx(2 * 80 / (10 ** (12 / 20)) ** 2, rel=0.05)
        assert rel == pytest.approx(2 * 1000 / (10 ** (12 / 20)) ** 2, rel=0.05)
        atk_pad, rel_pad = ballistics_giannoulis2013(3.0)
        assert atk_pad > atk and rel_pad > rel

    def test_knee_width(self):
        assert knee_width_ma2015(-24.0) == pytest.approx(12.0)


class TestSoundcheckPipeline:
    def test_stable_channel_bypassed(self):
        """Принцип «по умолчанию не сжимать»: стабильный пад не трогаем."""
        stats = channel_dynamics_stats(_tone(440.0), SR)
        decision = should_compress(stats, "keys")
        assert not decision["compress"]
        assert decision["reason"] == "stable_channel_bypass"

    def test_dynamic_vocal_compressed(self):
        stats = channel_dynamics_stats(_dynamic_vocal(), SR)
        assert stats.sigma_level_db > 3.5
        assert should_compress(stats, "lead_vocal")["compress"]

    def test_threshold_hits_target_gr_on_loud_sections(self):
        stats = channel_dynamics_stats(_dynamic_vocal(), SR)
        params = derive_params(stats, "lead_vocal", w_p=1.0, w_f=1.0)
        # T = L_loud − GR·R/(R−1) => GR на громком ≈ цель (hard-knee оценка)
        implied_gr = (stats.l10_db - params.threshold_db) * (1 - 1 / params.ratio)
        assert implied_gr == pytest.approx(params.gr_target_db, abs=0.5)

    def test_ratio_clamped_for_live(self):
        stats = channel_dynamics_stats(_dynamic_vocal(), SR)
        params = derive_params(stats, "guitar", w_p=1.9, w_f=1.9)
        assert params.ratio <= 4.0

    def test_release_guarded_against_lf(self):
        stats = channel_dynamics_stats(_tone(45.0, amp=0.4) * np.where(
            (np.arange(int(2 * SR)) / SR % 1.0) < 0.5, 1.0, 0.1), SR)
        params = derive_params(stats, "bass", w_p=1.0, w_f=1.5)
        assert params.release_ms >= 1000.0 / stats.f_low_hz

    def test_end_to_end_multichannel(self):
        result = soundcheck_comp_v3({
            16: {"audio": _dynamic_vocal(), "role": "lead_vocal"},
            13: {"audio": _tone(440.0), "role": "keys"},
        }, SR, tempo_bpm=120.0)
        assert result[16].compress and result[16].params is not None
        assert result[16].verification.get("iterations", 0) >= 1
        assert not result[13].compress
        import json
        json.dumps(result[16].to_dict())

    def test_makeup_bounded(self):
        result = soundcheck_comp_v3(
            {16: {"audio": _dynamic_vocal(), "role": "lead_vocal",
                  "true_peak_dbtp": -1.5}}, SR)
        params = result[16].params
        assert params is not None
        assert params.makeup_db <= 0.5 + 1e-9  # headroom-aware


class TestLiveSupervisor:
    def _state(self, **kw):
        return LiveGRSupervisorState(
            threshold_db=-24.0, threshold_snapshot_db=-24.0,
            gr_target_db=4.0, makeup_db=0.0, makeup_target_db=2.0, **kw)

    def test_threshold_drift_clamped(self):
        state = self._state()
        for _ in range(600):
            live_gr_supervise(state, gr_measured_db=12.0, dt_s=1.0)
        assert state.threshold_db <= -24.0 + 4.0 + 1e-6

    def test_over_gr_raises_threshold_not_ratio(self):
        state = self._state()
        before = state.threshold_db
        for _ in range(8):
            live_gr_supervise(state, gr_measured_db=10.0, dt_s=1.0)
        assert state.threshold_db > before

    def test_makeup_rate_limited_and_frozen_on_low_margin(self):
        state = self._state()
        live_gr_supervise(state, 4.0, dt_s=1.0, feedback_margin_db=12.0)
        assert state.makeup_db == pytest.approx(0.5)  # 0.5 дБ/с
        frozen = state.makeup_db
        live_gr_supervise(state, 4.0, dt_s=1.0, feedback_margin_db=3.0)
        assert state.makeup_db == pytest.approx(frozen)  # рост заморожен

    def test_freeze_in_silence(self):
        state = self._state()
        before = (state.threshold_db, state.makeup_db)
        live_gr_supervise(state, 10.0, dt_s=1.0, signal_active=False)
        assert (state.threshold_db, state.makeup_db) == before

    def test_safe_generic(self):
        params = safe_generic_params(l10_db=-18.0)
        assert params.ratio == 2.0
        assert params.makeup_db == 0.0
