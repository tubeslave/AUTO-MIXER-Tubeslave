"""Cross-module DSP invariants for the Automixer signal path.

These tests use synthetic audio only. They never instantiate mixer clients or
send OSC/MIDI commands. The goal is to catch DSP failures that ordinary unit
tests can miss: chunk-boundary dependence, time-window blindness, wrong source
contribution weighting, invalid numerics, and obviously incorrect spectral
measurements.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from lufs_gain_staging import KWeightingFilter, TruePeakMeter
from signal_metrics import SignalAnalyzer
from live_shared_mix import (
    LiveSharedMixChannel,
    _band_shares,
    _compensated_band_levels,
    _ltas_spectrum,
)

SR = 48000


def _sine(freq_hz: float, amp: float = 0.5, seconds: float = 1.0, phase: float = 0.0):
    t = np.arange(int(SR * seconds), dtype=np.float64) / SR
    return (amp * np.sin(2.0 * np.pi * freq_hz * t + phase)).astype(np.float32)


def _channel(channel_id: int, audio: np.ndarray, fader_db: float):
    return LiveSharedMixChannel(
        channel_id=channel_id,
        name=f"CH{channel_id}",
        role="guitars",
        stems=("GUITARS",),
        priority=0.5,
        audio=np.asarray(audio, dtype=np.float32),
        sample_rate=SR,
        fader_db=fader_db,
        muted=False,
        auto_corrections_enabled=True,
    )


def test_true_peak_is_independent_of_reasonable_chunk_boundaries():
    """Streaming TP must not materially depend on where the host cuts blocks."""
    # Broadband deterministic signal exposes FIR state loss better than a
    # periodic sine whose phase may accidentally hide the boundary error.
    rng = np.random.default_rng(20260907)
    audio = rng.standard_normal(SR).astype(np.float32)
    audio *= 0.5 / np.max(np.abs(audio))

    contiguous = TruePeakMeter(SR)
    contiguous.process(audio)
    contiguous_max = contiguous.get_max_peak_dbtp()

    chunked = TruePeakMeter(SR)
    for start in range(0, len(audio), 1024):
        chunked.process(audio[start:start + 1024])
    chunked_max = chunked.get_max_peak_dbtp()

    assert np.isfinite(contiguous_max)
    assert np.isfinite(chunked_max)
    assert abs(contiguous_max - chunked_max) < 0.15


def test_true_peak_and_k_weighting_stay_finite_for_silence():
    silence = np.zeros(4096, dtype=np.float32)
    tp = TruePeakMeter(SR)
    assert np.isfinite(tp.process(silence))
    weighted = KWeightingFilter(SR).process(silence)
    assert weighted.shape == silence.shape
    assert np.all(np.isfinite(weighted))
    assert np.max(np.abs(weighted)) == 0.0


def test_signal_analyzer_centroid_tracks_single_tone():
    analyzer = SignalAnalyzer(channel=1, sample_rate=SR, block_size=4096)
    tone = _sine(1000.0, amp=0.25, seconds=4096 / SR)
    analyzer.process(tone)
    metrics = analyzer.get_metrics()
    assert np.isfinite(metrics.spectral.centroid_hz)
    assert 930.0 <= metrics.spectral.centroid_hz <= 1070.0


def test_ltas_uses_material_beyond_first_fft_block():
    """LTAS must react to energy that appears later in the analysis window."""
    n = SR * 2
    first = np.zeros(n, dtype=np.float32)
    second = np.zeros(n, dtype=np.float32)
    prefix = _sine(500.0, amp=0.2, seconds=16384 / SR)
    first[: len(prefix)] = prefix
    second[: len(prefix)] = prefix

    # Keep the prefix identical, but make the remaining ~1.66 s radically
    # different. A genuine LTAS over the supplied audio must notice this.
    first[len(prefix):] = _sine(120.0, amp=0.5, seconds=(n - len(prefix)) / SR)
    second[len(prefix):] = _sine(8000.0, amp=0.5, seconds=(n - len(prefix)) / SR)

    a = _compensated_band_levels(first, SR)
    b = _compensated_band_levels(second, SR)
    delta = max(abs(float(a[k]) - float(b[k])) for k in a.keys() & b.keys())
    assert delta > 3.0


def test_ltas_spectrum_is_finite_for_short_and_silent_inputs():
    for audio in (np.zeros(1, dtype=np.float32), np.zeros(200, dtype=np.float32), _sine(1000.0, seconds=0.01)):
        freqs, spec = _ltas_spectrum(audio, SR)
        assert len(freqs) == len(spec)
        assert np.all(np.isfinite(freqs))
        assert np.all(np.isfinite(spec))
        assert np.all(spec >= 0.0)


def test_band_contribution_respects_channel_fader_gain():
    """A channel 40 dB down cannot still own ~50% of an audible band."""
    tone = _sine(1000.0, amp=0.3, seconds=1.0)
    channels = [_channel(1, tone, 0.0), _channel(2, tone, -40.0)]
    shares = _band_shares(channels, 0, len(tone), 700.0, 1400.0)

    assert shares[1] > 0.98
    assert shares[2] < 0.02
    assert abs(sum(shares.values()) - 1.0) < 1e-6


def test_signal_analyzer_outputs_remain_finite_with_nan_inf_input():
    analyzer = SignalAnalyzer(channel=1, sample_rate=SR, block_size=2048)
    audio = _sine(440.0, seconds=2048 / SR)
    audio[10] = np.nan
    audio[20] = np.inf
    audio[30] = -np.inf
    analyzer.process(audio)
    metrics = analyzer.get_metrics()

    numeric = [
        metrics.level.peak_db,
        metrics.level.true_peak_dbtp,
        metrics.level.rms_db,
        metrics.spectral.centroid_hz,
        metrics.spectral.flatness,
        metrics.spectral.brightness,
    ]
    assert all(np.isfinite(float(value)) for value in numeric)
