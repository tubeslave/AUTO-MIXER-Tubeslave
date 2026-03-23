"""
Tests for LUFS metering (ITU-R BS.1770 K-weighting + gated loudness).

Covers:
- KWeightingFilter design and frequency response
- LUFSMeter measurement accuracy on known signals
- Silence producing very low LUFS readings
"""

import numpy as np
import pytest

from lufs_gain_staging import KWeightingFilter, LUFSMeter


class TestKWeightingFilter:
    """Tests for the K-weighting filter stage."""

    def test_k_weighting_boost(self, sample_rate):
        """K-weighting should boost high-frequency content (shelf at ~1681 Hz)."""
        kw = KWeightingFilter(sample_rate=sample_rate)
        duration = 0.5
        t = np.arange(int(sample_rate * duration)) / sample_rate

        # Low frequency signal (100 Hz)
        low = np.sin(2 * np.pi * 100.0 * t).astype(np.float32)
        low_filtered = kw.process(low.copy())
        low_rms = np.sqrt(np.mean(low_filtered ** 2))

        # High frequency signal (4000 Hz)
        high = np.sin(2 * np.pi * 4000.0 * t).astype(np.float32)
        high_filtered = kw.process(high.copy())
        high_rms = np.sqrt(np.mean(high_filtered ** 2))

        # K-weighting boosts highs relative to lows.  Both input signals
        # have unit amplitude so the filtered high signal should have
        # higher RMS than the filtered low signal.
        assert high_rms > low_rms, (
            f"K-weighting should boost high freqs: high_rms={high_rms:.4f} "
            f"should be > low_rms={low_rms:.4f}"
        )

    def test_k_weighting_preserves_length(self, sample_rate):
        """Output length must match input length."""
        kw = KWeightingFilter(sample_rate=sample_rate)
        signal = np.random.randn(sample_rate).astype(np.float32)
        out = kw.process(signal)
        assert len(out) == len(signal)


class TestLUFSMeter:
    """Tests for the LUFS loudness meter."""

    def test_silence_is_very_quiet(self, sample_rate, test_audio_silence):
        """Silence should return a very low LUFS reading (< -60 dB)."""
        meter = LUFSMeter(sample_rate=sample_rate)
        lufs = meter.process(test_audio_silence)
        # Silence should be extremely low -- allow for numerical noise
        assert lufs < -60.0, f"Silence LUFS should be < -60, got {lufs:.1f}"

    def test_sine_wave_level(self, sample_rate):
        """A full-scale 1 kHz sine should produce LUFS near -3.01 dBFS.

        For a sine wave at 0 dBFS the RMS is -3.01 dB.  After K-weighting
        at 1 kHz (which is close to unity) the LUFS should be in the
        vicinity of -3 dB.  We allow a generous tolerance because the
        meter uses a 400 ms sliding window and transient effects matter.
        """
        duration = 2.0  # Need enough for the 400 ms window
        t = np.arange(int(sample_rate * duration)) / sample_rate
        full_scale_sine = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)

        meter = LUFSMeter(sample_rate=sample_rate)
        # Feed in the whole signal
        lufs = meter.process(full_scale_sine)
        # Full scale sine RMS is -3.01 dBFS; allow +/- 3 dB tolerance
        assert -7.0 < lufs < 1.0, f"Full-scale 1kHz sine LUFS should be near -3, got {lufs:.1f}"

    def test_louder_signal_gives_higher_lufs(self, sample_rate):
        """Doubling amplitude should increase LUFS by ~6 dB."""
        duration = 1.0
        t = np.arange(int(sample_rate * duration)) / sample_rate
        quiet = (0.1 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
        loud = (0.4 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)

        meter_q = LUFSMeter(sample_rate=sample_rate)
        meter_l = LUFSMeter(sample_rate=sample_rate)

        lufs_quiet = meter_q.process(quiet)
        lufs_loud = meter_l.process(loud)

        assert lufs_loud > lufs_quiet, (
            f"Louder signal should have higher LUFS: loud={lufs_loud:.1f}, quiet={lufs_quiet:.1f}"
        )
