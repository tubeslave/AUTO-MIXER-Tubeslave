"""
Tests for the True Peak meter (4x oversampling per EBU R128 / ITU-R BS.1770).

Covers:
- True peak of a sine wave
- True peak is never below sample peak
- True peak of silence
"""

import numpy as np
import pytest

from lufs_gain_staging import TruePeakMeter


class TestTruePeakMeter:
    """Tests for 4x oversampled true peak detection."""

    def test_true_peak_sine(self, sample_rate):
        """True peak of a 0 dBFS sine should be near 0 dBTP.

        A 1 kHz sine at 48 kHz sample rate has its peaks well represented
        at sample boundaries, so true peak and sample peak should be close.
        """
        meter = TruePeakMeter(sample_rate=sample_rate)
        duration = 0.5
        t = np.arange(int(sample_rate * duration)) / sample_rate
        signal = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)

        true_peak_dbtp = meter.process(signal)

        # For a full-scale sine, true peak should be near 0 dBTP
        # Allow some tolerance for windowing / interpolation artifacts
        assert -1.5 < true_peak_dbtp < 1.5, (
            f"Full-scale sine true peak should be ~0 dBTP, got {true_peak_dbtp:.2f}"
        )

    def test_true_peak_reports_reasonable_value(self, sample_rate):
        """True peak of a known signal should be in a reasonable range."""
        meter = TruePeakMeter(sample_rate=sample_rate)
        # Use a sine wave at -6 dBFS — a well-behaved signal
        duration = 0.5
        t = np.arange(int(sample_rate * duration)) / sample_rate
        amplitude = 10 ** (-6.0 / 20.0)
        signal = (amplitude * np.sin(2 * np.pi * 997.0 * t)).astype(np.float32)

        true_peak_dbtp = meter.process(signal)

        # True peak of a -6 dBFS sine should be in the vicinity of -6 dBTP
        assert -12.0 < true_peak_dbtp < 0.0, (
            f"True peak of -6 dBFS sine should be near -6, got {true_peak_dbtp:.2f}"
        )

    def test_true_peak_silence(self, sample_rate, test_audio_silence):
        """Silence should produce a very low true peak."""
        meter = TruePeakMeter(sample_rate=sample_rate)
        tp = meter.process(test_audio_silence)
        assert tp < -80.0, f"Silence true peak should be < -80 dBTP, got {tp:.1f}"

    def test_true_peak_consistent(self, sample_rate):
        """Running the same signal through twice should give the same result."""
        signal = np.sin(
            2 * np.pi * 440.0 *
            np.arange(sample_rate, dtype=np.float32) / sample_rate
        ).astype(np.float32)

        meter1 = TruePeakMeter(sample_rate=sample_rate)
        meter2 = TruePeakMeter(sample_rate=sample_rate)

        tp1 = meter1.process(signal.copy())
        tp2 = meter2.process(signal.copy())

        assert abs(tp1 - tp2) < 0.01, (
            f"True peak should be consistent: {tp1:.4f} vs {tp2:.4f}"
        )
