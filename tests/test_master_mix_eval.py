"""Master mix eval: замкнутый контур оценки сведения (Main LR -> USB 47/48)."""

import numpy as np
import pytest

from master_mix_eval import evaluate_master_mix

SR = 48000
RNG = np.random.default_rng(23)


def _pink_ish(n, seed=0):
    """Розовеющий шум: белый через 1/sqrt(f) наклон."""
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.normal(0, 1, n))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spec[1:] /= np.sqrt(freqs[1:] / freqs[1])
    x = np.fft.irfft(spec, n)
    return x / (np.abs(x).max() + 1e-12)


def test_silence_is_no_signal_not_perfect_mix():
    z = np.zeros(3 * SR)
    v = evaluate_master_mix(z, z, SR)
    assert v.status == "NO_SIGNAL"
    assert any("тишина" in i for i in v.issues)


def test_reasonable_stereo_mix_ok():
    base = _pink_ish(6 * SR, seed=1) * 0.15
    l = base + _pink_ish(6 * SR, seed=2) * 0.02
    r = base + _pink_ish(6 * SR, seed=3) * 0.02
    v = evaluate_master_mix(l, r, SR)
    assert v.status in ("OK", "ISSUES")
    assert v.lufs_integrated is not None and v.lufs_integrated > -60
    assert abs(v.lr_balance_db) < 1.5
    assert v.correlation > 0.5
    assert v.mono_fold_ratio > 0.5
    # у сбалансированного розового наклон близок к live_standard
    assert not any("дисбаланс" in i for i in v.issues)
    assert not any("фазовые" in i for i in v.issues)


def test_lr_imbalance_flagged():
    base = _pink_ish(4 * SR, seed=4) * 0.2
    v = evaluate_master_mix(base, base * 0.4, SR)   # R на -8 дБ
    assert any("дисбаланс" in i for i in v.issues)


def test_phase_problem_flagged():
    base = _pink_ish(4 * SR, seed=5) * 0.2
    v = evaluate_master_mix(base, -base, SR)        # антифаза
    assert v.correlation < -0.9
    assert any("фазовые" in i for i in v.issues)
    assert any("mono-fold" in i for i in v.issues)


def test_true_peak_ceiling_flagged():
    base = _pink_ish(4 * SR, seed=6)
    hot = np.clip(base * 1.4, -1.0, 1.0)            # почти full-scale
    v = evaluate_master_mix(hot, hot, SR)
    assert v.true_peak_dbtp is not None
    assert any("true peak" in i for i in v.issues)


def test_masked_vocal_zone_flagged():
    # микс с продавленной зоной 1-4 кГц
    n = 5 * SR
    x = _pink_ish(n, seed=7) * 0.2
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    dip = (freqs >= 1000) & (freqs <= 4000)
    spec[dip] *= 10 ** (-30 / 20)
    x = np.fft.irfft(spec, n)
    v = evaluate_master_mix(x, x, SR)
    assert v.vocal_presence_db is not None
    assert any("вокала" in i for i in v.issues)
