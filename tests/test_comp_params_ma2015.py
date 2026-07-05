"""Регрессии Ma et al. 2015 и баллистика Giannoulis — синтетические сигналы."""

import numpy as np
import pytest

from comp_params_ma2015 import (
    ballistics_giannoulis,
    bounded_makeup_db,
    low_frequency_weighting,
    ma2015_params_from_audio,
    percussivity_weighting,
    ratio_ma2015,
    threshold_ma2015,
)
from auto_compressor_cf import AutoCFCompressorController, CFCompressorCalculator

SR = 48000


def _drum_hits(n_sec: float = 2.0, hit_hz: float = 2.0) -> np.ndarray:
    """Перкуссивный сигнал: экспоненциально затухающие удары."""
    n = int(n_sec * SR)
    x = np.zeros(n)
    period = int(SR / hit_hz)
    t = np.arange(period) / SR
    hit = np.exp(-t * 40.0) * np.sin(2 * np.pi * 180.0 * t)
    for start in range(0, n - period, period):
        x[start:start + period] += hit
    return x.astype(np.float64)


def _pad(n_sec: float = 2.0, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(n_sec * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def _bass(n_sec: float = 2.0) -> np.ndarray:
    t = np.arange(int(n_sec * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * 60.0 * t)).astype(np.float64)


class TestFeatures:
    def test_percussivity_high_for_drums_low_for_pad(self):
        assert percussivity_weighting(_drum_hits(), SR) > 0.3
        assert percussivity_weighting(_pad(), SR) < 0.1

    def test_lf_weighting_high_for_bass_low_for_hihat_range(self):
        assert low_frequency_weighting(_bass(), SR) > 0.9
        t = np.arange(SR) / SR
        hihat = np.sin(2 * np.pi * 6000.0 * t)
        assert low_frequency_weighting(hihat, SR) < 0.05

    def test_weights_bounded(self):
        rng = np.random.default_rng(3)
        noise = rng.normal(0, 0.3, SR)
        for fn in (percussivity_weighting, low_frequency_weighting):
            assert 0.0 <= fn(noise, SR) <= 1.0


class TestRegressions:
    def test_ratio_formula_exact(self):
        # R = 0.54*w_p + 0.764*w_f + 1
        assert ratio_ma2015(0.0, 0.0) == pytest.approx(1.0)
        assert ratio_ma2015(1.0, 1.0) == pytest.approx(2.304)
        assert ratio_ma2015(0.5, 0.25) == pytest.approx(1.461)

    def test_threshold_formula_exact(self):
        # T = -11.03 + 0.441*x_rms - 4.987*w_p (точный коэффициент статьи)
        assert threshold_ma2015(-20.0, 0.0) == pytest.approx(-19.85)
        assert threshold_ma2015(-20.0, 1.0) == pytest.approx(-24.837)

    def test_threshold_clamped(self):
        assert threshold_ma2015(-90.0, 1.0) == -40.0
        assert threshold_ma2015(20.0, 0.0) == -8.0

    def test_ballistics_shorter_for_transient(self):
        atk_perc, rel_perc = ballistics_giannoulis(crest_db=20.0, w_p=0.9)
        atk_pad, rel_pad = ballistics_giannoulis(crest_db=4.0, w_p=0.05)
        assert atk_perc < atk_pad
        assert rel_perc < rel_pad

    def test_makeup_bounded_and_headroom_aware(self):
        # GR = (x_rms - T)*(R-1)/R; makeup = 70% GR, cap 2 dB
        assert bounded_makeup_db(-18.0, -30.0, 3.0) == pytest.approx(2.0)
        # true-peak headroom режет makeup (правило CLAUDE.md)
        assert bounded_makeup_db(-18.0, -30.0, 3.0,
                                 true_peak_dbtp=-1.5) == pytest.approx(0.5)
        assert bounded_makeup_db(-40.0, -30.0, 3.0) == 0.0


class TestEndToEnd:
    def test_drums_get_higher_ratio_shorter_attack_than_pad(self):
        drums = ma2015_params_from_audio(_drum_hits(), SR)
        pad = ma2015_params_from_audio(_pad(), SR)
        assert drums.ratio > pad.ratio
        assert drums.attack_ms < pad.attack_ms

    def test_bass_lf_weighting_raises_ratio(self):
        bass = ma2015_params_from_audio(_bass(), SR)
        pad = ma2015_params_from_audio(_pad(), SR)  # 440 Гц — вне LF-полосы
        assert bass.w_f > pad.w_f
        assert bass.ratio > pad.ratio

    def test_calculator_blend_uses_ma_params(self):
        calc = CFCompressorCalculator()
        ma = ma2015_params_from_audio(_drum_hits(), SR)
        with_ma = calc.calculate_params(cf_db=15.0, lufs_momentary=-20.0,
                                        peak_db=-5.0, ma2015=ma)
        without = calc.calculate_params(cf_db=15.0, lufs_momentary=-20.0,
                                        peak_db=-5.0)
        assert with_ma.to_dict() != without.to_dict()
        assert with_ma.makeup_gain_db <= ma.makeup_db + 1e-9

    def test_controller_records_ma_explainability(self):
        ctl = AutoCFCompressorController(sample_rate=SR)
        ctl.is_active = True
        params = ctl.process_audio(_drum_hits(), lufs_momentary=-20.0, peak_db=-5.0)
        assert params is not None
        assert ctl.last_ma2015 is not None
        assert ctl.last_ma2015["source"] == "ma2015_regression"
        assert 0.0 <= ctl.last_ma2015["w_p"] <= 1.0
