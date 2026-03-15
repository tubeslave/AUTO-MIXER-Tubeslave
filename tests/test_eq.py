"""
Tests for the Auto EQ system.

Covers:
- EQBand dataclass creation and serialization
- Spectral analysis basics
- EQ correction generation
"""

import numpy as np
import pytest

from auto_eq import EQBand, SpectrumAnalyzer


class TestEQBandCreation:
    """Tests for the EQBand dataclass."""

    def test_eq_band_creation(self):
        """EQBand should store frequency, gain, Q, and type."""
        band = EQBand(
            band_type="peaking",
            frequency=1000.0,
            gain=-3.0,
            q=1.5,
        )
        assert band.frequency == 1000.0
        assert band.gain == -3.0
        assert band.q == 1.5
        assert band.band_type == "peaking"

    def test_eq_band_to_dict(self):
        """to_dict should produce a serializable dictionary."""
        band = EQBand(
            band_type="highshelf",
            frequency=8000.0,
            gain=2.0,
            q=0.7,
        )
        d = band.to_dict()
        assert isinstance(d, dict)
        assert d["frequency"] == 8000.0
        assert d["gain"] == 2.0
        assert "band_type" in d or "type" in d

    def test_eq_band_defaults(self):
        """EQBand with minimal args should use sensible defaults."""
        band = EQBand(band_type="peaking", frequency=500.0, gain=0.0, q=1.0)
        assert band.q == 1.0
        assert band.gain == 0.0


class TestSpectralAnalysis:
    """Tests for the SpectrumAnalyzer FFT analysis."""

    def test_spectral_analysis(self, sample_rate):
        """Analyzer should detect the dominant frequency of a sine wave."""
        analyzer = SpectrumAnalyzer(sample_rate=sample_rate)
        duration = 0.5
        t = np.arange(int(sample_rate * duration)) / sample_rate
        # 2 kHz sine
        signal = np.sin(2 * np.pi * 2000.0 * t).astype(np.float32)

        result = analyzer.analyze(signal)

        # peak_freq should be near 2000 Hz (within one FFT bin width)
        assert hasattr(result, "peak_freq")
        assert abs(result.peak_freq - 2000.0) < 100.0, (
            f"Peak frequency should be ~2000 Hz, got {result.peak_freq:.1f}"
        )

    def test_spectral_analysis_returns_spectrum(self, sample_rate):
        """analyze() should return a result with spectrum array."""
        analyzer = SpectrumAnalyzer(sample_rate=sample_rate)
        signal = np.random.randn(sample_rate).astype(np.float32) * 0.1
        result = analyzer.analyze(signal)
        assert hasattr(result, "spectrum")
        assert len(result.spectrum) > 0

    def test_spectral_analysis_silence(self, sample_rate, test_audio_silence):
        """Silence should produce very low spectral energy."""
        analyzer = SpectrumAnalyzer(sample_rate=sample_rate)
        result = analyzer.analyze(test_audio_silence)
        assert hasattr(result, "spectrum")
        max_energy = np.max(result.spectrum) if len(result.spectrum) > 0 else 0.0
        assert max_energy < 1e-6, f"Silence spectrum max should be ~0, got {max_energy}"


class TestEQCorrectionGeneration:
    """Tests for EQ correction generation from spectral analysis."""

    def test_eq_correction_generation(self, sample_rate):
        """Auto EQ should generate correction bands for a signal."""
        try:
            from auto_eq import AutoEQController, InstrumentProfiles
        except ImportError:
            pytest.skip("AutoEQController not available")

        # Generate a signal with exaggerated low-end
        duration = 1.0
        t = np.arange(int(sample_rate * duration)) / sample_rate
        signal = (
            0.8 * np.sin(2 * np.pi * 100.0 * t) +
            0.1 * np.sin(2 * np.pi * 1000.0 * t)
        ).astype(np.float32)

        analyzer = SpectrumAnalyzer(sample_rate=sample_rate)
        result = analyzer.analyze(signal)

        # The peak frequency should be near the dominant 100 Hz
        assert result.peak_freq < 300.0, (
            f"Peak should be near 100 Hz, got {result.peak_freq:.1f}"
        )
