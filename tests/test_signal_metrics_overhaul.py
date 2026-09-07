"""Focused tests for the finite streaming signal_metrics implementation."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from signal_metrics import SignalAnalyzer, compare_channels

SR = 48000


def _sine(freq: float, amp: float = 0.25, seconds: float = 1.0):
    t = np.arange(int(seconds * SR), dtype=np.float64) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_streaming_analyzer_is_finite_and_centroid_tracks_tone():
    analyzer = SignalAnalyzer(channel=1, sample_rate=SR, block_size=1024)
    tone = _sine(1000.0, seconds=1.0)
    tone[10] = np.nan
    tone[20] = np.inf
    for start in range(0, tone.size, 1024):
        analyzer.process(tone[start:start + 1024])
    metrics = analyzer.get_metrics()
    values = [
        metrics.level.peak_db,
        metrics.level.true_peak_dbtp,
        metrics.level.rms_db,
        metrics.level.lufs_integrated,
        metrics.spectral.centroid_hz,
        metrics.spectral.flatness,
        metrics.spectral.brightness,
    ]
    assert all(np.isfinite(float(v)) for v in values)
    assert 900.0 <= metrics.spectral.centroid_hz <= 1100.0


def test_streaming_reset_clears_accumulated_peak():
    analyzer = SignalAnalyzer(channel=1, sample_rate=SR, block_size=1024)
    analyzer.process(_sine(1000.0, amp=0.8, seconds=0.2))
    assert analyzer.get_metrics().level.peak_db > -3.0
    analyzer.reset()
    metrics = analyzer.get_metrics()
    assert metrics.level.peak_db <= -99.0
    assert metrics.level.true_peak_dbtp <= -99.0


def test_compare_channels_finds_known_delay_and_high_similarity():
    rng = np.random.default_rng(42)
    ref = rng.standard_normal(SR).astype(np.float32) * 0.1
    delay = 96
    target = np.zeros_like(ref)
    target[delay:] = ref[:-delay]
    result = compare_channels(ref, target, sample_rate=SR, ch_a=1, ch_b=2)
    assert abs(abs(result.delay_samples) - delay) <= 1
    assert abs(result.delay_ms - 2.0) < 0.05
    assert abs(result.cross_correlation) > 0.98
    assert result.spectral_similarity > 0.98
    assert 0.0 <= result.coherence <= 1.0


def test_coherence_separates_related_and_unrelated_noise():
    rng = np.random.default_rng(7)
    a = rng.standard_normal(SR * 2).astype(np.float32)
    b_related = a + 0.05 * rng.standard_normal(a.size).astype(np.float32)
    b_unrelated = rng.standard_normal(a.size).astype(np.float32)
    related = compare_channels(a, b_related, sample_rate=SR)
    unrelated = compare_channels(a, b_unrelated, sample_rate=SR)
    assert related.coherence > unrelated.coherence + 0.4
    assert related.cross_correlation > unrelated.cross_correlation + 0.4


def test_compare_channels_sanitizes_nonfinite_input():
    a = _sine(440.0, seconds=1.0)
    b = a.copy()
    a[5] = np.nan
    b[8] = np.inf
    result = compare_channels(a, b, sample_rate=SR)
    for value in (
        result.cross_correlation,
        result.delay_ms,
        result.coherence,
        result.spectral_similarity,
        result.level_difference_db,
    ):
        assert np.isfinite(float(value))
