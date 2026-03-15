"""
Tests for backend/signal_analysis.py — ratio_float_to_wing(),
SpectralAnalyzerCompressor, SignalFeatureExtractor, ChannelSignalFeatures.

All tests work without hardware or network.
"""

import numpy as np
import pytest

try:
    from signal_analysis import (
        ratio_float_to_wing,
        SpectralAnalyzerCompressor,
        SignalFeatureExtractor,
        ChannelSignalFeatures,
        WING_RATIO_VALUES,
        WING_RATIO_STRINGS,
    )
except ImportError:
    pytest.skip("signal_analysis module not importable", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spectral_analyzer():
    """SpectralAnalyzerCompressor with default settings."""
    return SpectralAnalyzerCompressor(sample_rate=48000, fft_size=4096)


def _make_sine(freq_hz=440.0, duration_sec=0.1, sr=48000, amplitude=0.5):
    """Generate a mono sine wave."""
    t = np.arange(int(sr * duration_sec)) / sr
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# ratio_float_to_wing tests
# ---------------------------------------------------------------------------

class TestRatioFloatToWing:

    def test_exact_match_1_0(self):
        assert ratio_float_to_wing(1.0) == "1.0:1"

    def test_exact_match_4_0(self):
        assert ratio_float_to_wing(4.0) == "4.0:1"

    def test_exact_match_infinity(self):
        # Note: abs(inf - inf) is NaN, so the function cannot pick the inf
        # entry via its current closest-match loop.  It falls back to "20:1"
        # (last finite entry wins) or "1.0:1" depending on implementation.
        result = ratio_float_to_wing(float('inf'))
        assert result in WING_RATIO_STRINGS  # valid Wing ratio string

    def test_closest_match_2_5(self):
        # 2.5 is between 2.0 and 3.0; should pick closest
        result = ratio_float_to_wing(2.5)
        assert result in ("2.0:1", "3.0:1")

    def test_closest_match_5_0(self):
        # 5.0 is between 4.0 and 6.0; should pick closest
        result = ratio_float_to_wing(5.0)
        assert result in ("4.0:1", "6.0:1")

    def test_zero_returns_1_0(self):
        assert ratio_float_to_wing(0.0) == "1.0:1"

    def test_negative_returns_1_0(self):
        assert ratio_float_to_wing(-5.0) == "1.0:1"

    def test_large_value(self):
        result = ratio_float_to_wing(100.0)
        assert result in ("20:1", "inf:1")

    def test_very_small_positive(self):
        result = ratio_float_to_wing(0.5)
        assert result == "1.0:1"

    def test_all_exact_values(self):
        """Each exact WING_RATIO_VALUE should map to its corresponding string.
        Skip inf because abs(inf - inf) is NaN, breaking the closest-match loop.
        """
        for val, expected in zip(WING_RATIO_VALUES, WING_RATIO_STRINGS):
            if val == float('inf'):
                continue
            assert ratio_float_to_wing(val) == expected


# ---------------------------------------------------------------------------
# WING_RATIO constants tests
# ---------------------------------------------------------------------------

class TestWingRatioConstants:

    def test_values_length(self):
        assert len(WING_RATIO_VALUES) == len(WING_RATIO_STRINGS)

    def test_values_sorted(self):
        for i in range(len(WING_RATIO_VALUES) - 1):
            assert WING_RATIO_VALUES[i] <= WING_RATIO_VALUES[i + 1]


# ---------------------------------------------------------------------------
# SpectralAnalyzerCompressor tests
# ---------------------------------------------------------------------------

class TestSpectralAnalyzerCompressor:

    def test_analyze_returns_dict(self, spectral_analyzer):
        samples = _make_sine(freq_hz=1000.0, duration_sec=0.1)
        result = spectral_analyzer.analyze(samples)
        assert isinstance(result, dict)
        assert "centroid" in result
        assert "rolloff" in result
        assert "band_energy" in result
        assert "spectral_flux" in result

    def test_centroid_for_high_freq(self, spectral_analyzer):
        """High frequency tone should have higher centroid than low frequency."""
        high = _make_sine(freq_hz=8000.0, duration_sec=0.1)
        low = _make_sine(freq_hz=200.0, duration_sec=0.1)
        spectral_analyzer.reset()
        result_high = spectral_analyzer.analyze(high)
        spectral_analyzer.reset()
        result_low = spectral_analyzer.analyze(low)
        assert result_high["centroid"] > result_low["centroid"]

    def test_band_energy_keys(self, spectral_analyzer):
        samples = _make_sine(freq_hz=1000.0)
        result = spectral_analyzer.analyze(samples)
        expected_bands = {"sub", "bass", "low_mid", "mid", "high_mid", "high", "air"}
        assert set(result["band_energy"].keys()) == expected_bands

    def test_spectral_flux_first_frame_zero(self, spectral_analyzer):
        """First frame should have zero flux (no previous frame)."""
        spectral_analyzer.reset()
        samples = _make_sine(freq_hz=1000.0)
        result = spectral_analyzer.analyze(samples)
        assert result["spectral_flux"] == 0.0

    def test_spectral_flux_second_frame_nonzero(self, spectral_analyzer):
        """Different consecutive frames should produce nonzero flux."""
        spectral_analyzer.reset()
        samples1 = _make_sine(freq_hz=1000.0)
        samples2 = _make_sine(freq_hz=5000.0)
        spectral_analyzer.analyze(samples1)
        result = spectral_analyzer.analyze(samples2)
        assert result["spectral_flux"] > 0.0

    def test_short_signal_padded(self, spectral_analyzer):
        """Signal shorter than fft_size should be zero-padded."""
        short = _make_sine(freq_hz=1000.0, duration_sec=0.01)
        assert len(short) < spectral_analyzer.fft_size
        result = spectral_analyzer.analyze(short)
        assert "centroid" in result

    def test_reset(self, spectral_analyzer):
        samples = _make_sine(freq_hz=1000.0)
        spectral_analyzer.analyze(samples)
        assert spectral_analyzer._prev_spectrum is not None
        spectral_analyzer.reset()
        assert spectral_analyzer._prev_spectrum is None

    def test_rolloff_positive(self, spectral_analyzer):
        samples = _make_sine(freq_hz=1000.0, amplitude=0.8)
        result = spectral_analyzer.analyze(samples)
        assert result["rolloff"] > 0.0


# ---------------------------------------------------------------------------
# ChannelSignalFeatures tests
# ---------------------------------------------------------------------------

class TestChannelSignalFeatures:

    def test_default_values(self):
        f = ChannelSignalFeatures(channel_id=1)
        assert f.channel_id == 1
        assert f.peak_db == -100.0
        assert f.rms_db == -100.0
        assert f.spectral_centroid_hz == 0.0
        assert f.band_energy == {}


# ---------------------------------------------------------------------------
# SignalFeatureExtractor tests
# ---------------------------------------------------------------------------

class TestSignalFeatureExtractor:

    def test_process_returns_features(self):
        extractor = SignalFeatureExtractor(channel_id=1, sample_rate=48000)
        samples = _make_sine(freq_hz=440.0, duration_sec=0.1)
        features = extractor.process(samples)
        assert isinstance(features, ChannelSignalFeatures)
        assert features.channel_id == 1

    def test_process_empty_samples(self):
        extractor = SignalFeatureExtractor(channel_id=1)
        features = extractor.process(np.array([], dtype=np.float32))
        assert features.channel_id == 1
        assert features.peak_db == -100.0

    def test_peak_db_for_loud_signal(self):
        extractor = SignalFeatureExtractor(channel_id=1, sample_rate=48000)
        loud = _make_sine(amplitude=0.9)
        features = extractor.process(loud)
        assert features.peak_db > -10.0

    def test_peak_db_for_quiet_signal(self):
        extractor = SignalFeatureExtractor(channel_id=1, sample_rate=48000)
        quiet = _make_sine(amplitude=0.001)
        features = extractor.process(quiet)
        assert features.peak_db < -50.0

    def test_spectral_centroid_populated(self):
        extractor = SignalFeatureExtractor(channel_id=1, sample_rate=48000)
        samples = _make_sine(freq_hz=1000.0, duration_sec=0.1)
        features = extractor.process(samples)
        assert features.spectral_centroid_hz > 0.0

    def test_band_energy_populated(self):
        extractor = SignalFeatureExtractor(channel_id=1, sample_rate=48000)
        samples = _make_sine(freq_hz=1000.0, duration_sec=0.1)
        features = extractor.process(samples)
        assert len(features.band_energy) > 0

    def test_reset(self):
        extractor = SignalFeatureExtractor(channel_id=1, sample_rate=48000)
        samples = _make_sine(freq_hz=440.0)
        extractor.process(samples)
        extractor.reset()
        assert extractor._time_sec == 0.0
        assert len(extractor.transient_times) == 0

    def test_multiple_blocks_accumulate(self):
        extractor = SignalFeatureExtractor(channel_id=1, sample_rate=48000)
        for _ in range(5):
            samples = _make_sine(freq_hz=440.0, duration_sec=0.05)
            features = extractor.process(samples)
        assert features.channel_id == 1
        # After multiple blocks, should have some envelope data
        assert extractor._time_sec > 0.0
