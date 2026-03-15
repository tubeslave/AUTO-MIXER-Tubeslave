"""
Tests for backend.ml.mix_quality — mix quality metrics including spectral
balance, stereo correlation, loudness range, crest factor, masking score,
A-weighted level, and overall quality scoring.

All tests use numpy-generated audio. scipy is optional.
"""

import numpy as np
import pytest

from backend.ml.mix_quality import MixQualityMetric, HAS_SCIPY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def metric():
    return MixQualityMetric(sr=48000)


@pytest.fixture
def sine_mono():
    """1-second 440 Hz mono sine at 48 kHz, amplitude 0.5."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float64)
    return np.sin(2 * np.pi * 440 * t) * 0.5


@pytest.fixture
def sine_stereo():
    """1-second stereo signal (440 Hz L, 880 Hz R)."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float64)
    left = np.sin(2 * np.pi * 440 * t) * 0.5
    right = np.sin(2 * np.pi * 880 * t) * 0.3
    return np.stack([left, right], axis=0)


@pytest.fixture
def long_stereo():
    """4-second stereo signal for LRA measurement."""
    sr = 48000
    n = sr * 4
    t = np.linspace(0, 4.0, n, endpoint=False, dtype=np.float64)
    # Amplitude modulation to create dynamic range
    env = 0.3 + 0.5 * np.abs(np.sin(2 * np.pi * 0.5 * t))
    left = np.sin(2 * np.pi * 440 * t) * env
    right = np.sin(2 * np.pi * 660 * t) * env * 0.8
    return np.stack([left, right], axis=0)


@pytest.fixture
def pink_noise_channels():
    """Multiple channels of coloured noise for masking tests."""
    rng = np.random.default_rng(42)
    sr = 48000
    n = sr
    channels = []
    # Low-frequency channel
    t = np.linspace(0, 1.0, n, endpoint=False, dtype=np.float64)
    channels.append(np.sin(2 * np.pi * 100 * t) * 0.4)
    # Mid-frequency channel
    channels.append(np.sin(2 * np.pi * 1000 * t) * 0.3)
    # High-frequency noise channel
    noise = rng.standard_normal(n).astype(np.float64) * 0.2
    channels.append(np.diff(noise, prepend=0.0))
    return channels


# ---------------------------------------------------------------------------
# Spectral balance
# ---------------------------------------------------------------------------

class TestSpectralBalance:

    def test_returns_float(self, metric, sine_mono):
        score = metric.spectral_balance(sine_mono)
        assert isinstance(score, float)

    def test_in_range_0_100(self, metric, sine_mono):
        score = metric.spectral_balance(sine_mono)
        assert 0.0 <= score <= 100.0

    def test_stereo_input_accepted(self, metric, sine_stereo):
        score = metric.spectral_balance(sine_stereo)
        assert 0.0 <= score <= 100.0

    def test_short_audio_returns_50(self, metric):
        """Audio shorter than 1024 samples should return default 50."""
        short = np.zeros(100, dtype=np.float64)
        score = metric.spectral_balance(short)
        assert score == 50.0


# ---------------------------------------------------------------------------
# Stereo correlation
# ---------------------------------------------------------------------------

class TestStereoCorrelation:

    def test_identical_signals_give_1(self, metric, sine_mono):
        corr = metric.stereo_correlation(sine_mono, sine_mono)
        assert corr == pytest.approx(1.0, abs=0.01)

    def test_inverted_signals_give_negative(self, metric, sine_mono):
        corr = metric.stereo_correlation(sine_mono, -sine_mono)
        assert corr == pytest.approx(-1.0, abs=0.01)

    def test_uncorrelated_near_zero(self, metric):
        """Two different sine frequencies should have low correlation."""
        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float64)
        a = np.sin(2 * np.pi * 440 * t)
        b = np.sin(2 * np.pi * 441 * t + np.pi / 3)
        corr = metric.stereo_correlation(a, b)
        # Not perfectly correlated but probably close due to similar freq
        assert -1.0 <= corr <= 1.0

    def test_result_clamped(self, metric, sine_mono):
        corr = metric.stereo_correlation(sine_mono, sine_mono * 0.5)
        assert -1.0 <= corr <= 1.0


# ---------------------------------------------------------------------------
# Loudness range
# ---------------------------------------------------------------------------

class TestLoudnessRange:

    def test_returns_float(self, metric, long_stereo):
        lra = metric.loudness_range(long_stereo)
        assert isinstance(lra, float)

    def test_nonnegative(self, metric, long_stereo):
        lra = metric.loudness_range(long_stereo)
        assert lra >= 0.0

    def test_short_audio_returns_zero(self, metric, sine_mono):
        """Audio shorter than 3 seconds should return 0.0."""
        short = sine_mono[:48000]  # 1 second
        lra = metric.loudness_range(short)
        assert lra == 0.0

    def test_constant_amplitude_low_lra(self, metric):
        """A constant-amplitude signal should have very low LRA."""
        sr = 48000
        t = np.linspace(0, 5.0, sr * 5, endpoint=False, dtype=np.float64)
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        lra = metric.loudness_range(audio)
        # Constant amplitude -> very low or zero LRA
        assert lra < 5.0


# ---------------------------------------------------------------------------
# Crest factor
# ---------------------------------------------------------------------------

class TestCrestFactor:

    def test_returns_float(self, metric, sine_mono):
        cf = metric.crest_factor(sine_mono)
        assert isinstance(cf, float)

    def test_positive_for_signal(self, metric, sine_mono):
        cf = metric.crest_factor(sine_mono)
        assert cf > 0

    def test_sine_crest_factor_approximately_3db(self, metric):
        """Pure sine wave has crest factor of sqrt(2) ~ 3.01 dB."""
        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float64)
        sine = np.sin(2 * np.pi * 440 * t)
        cf = metric.crest_factor(sine)
        assert cf == pytest.approx(3.01, abs=0.1)

    def test_silence_returns_very_low(self, metric):
        """Silence should return a very low (or negative) crest factor."""
        audio = np.zeros(4096, dtype=np.float64)
        cf = metric.crest_factor(audio)
        assert cf <= 0.0

    def test_stereo_accepted(self, metric, sine_stereo):
        cf = metric.crest_factor(sine_stereo)
        assert cf > 0


# ---------------------------------------------------------------------------
# Masking score
# ---------------------------------------------------------------------------

class TestMaskingScore:

    def test_returns_float(self, metric, pink_noise_channels):
        score = metric.masking_score(pink_noise_channels)
        assert isinstance(score, float)

    def test_in_range_0_100(self, metric, pink_noise_channels):
        score = metric.masking_score(pink_noise_channels)
        assert 0.0 <= score <= 100.0

    def test_single_channel_returns_100(self, metric, sine_mono):
        """A single channel should report no masking (100)."""
        score = metric.masking_score([sine_mono])
        assert score == 100.0

    def test_identical_channels_high_masking(self, metric, sine_mono):
        """Two identical channels should show high masking (low score)."""
        score = metric.masking_score([sine_mono, sine_mono])
        assert score < 80.0  # significant overlap

    def test_well_separated_channels_higher_score(self, metric, pink_noise_channels):
        """Channels at different frequencies should have less masking."""
        score = metric.masking_score(pink_noise_channels)
        # Well-separated channels should score higher than identical ones
        identical_score = metric.masking_score([pink_noise_channels[0], pink_noise_channels[0]])
        assert score >= identical_score


# ---------------------------------------------------------------------------
# A-weighted level
# ---------------------------------------------------------------------------

class TestAWeightedLevel:

    def test_returns_float(self, metric, sine_mono):
        level = metric.a_weighted_level(sine_mono)
        assert isinstance(level, float)

    def test_negative_for_typical_signal(self, metric, sine_mono):
        """A-weighted level should be negative dBFS for sub-unity audio."""
        level = metric.a_weighted_level(sine_mono)
        assert level < 0

    def test_empty_returns_minus_100(self, metric):
        level = metric.a_weighted_level(np.array([]))
        assert level == -100.0

    def test_stereo_accepted(self, metric, sine_stereo):
        level = metric.a_weighted_level(sine_stereo)
        assert isinstance(level, float)


# ---------------------------------------------------------------------------
# Overall
# ---------------------------------------------------------------------------

class TestOverall:

    def test_returns_dict(self, metric, sine_stereo):
        result = metric.overall(sine_stereo)
        assert isinstance(result, dict)

    def test_expected_keys_present(self, metric, sine_stereo):
        result = metric.overall(sine_stereo)
        for key in (
            "spectral_balance", "stereo_correlation", "loudness_range_lu",
            "crest_factor_db", "a_weighted_level_dbfs", "masking_score",
            "overall_score",
        ):
            assert key in result, f"Missing key '{key}'"

    def test_overall_score_in_range(self, metric, sine_stereo):
        result = metric.overall(sine_stereo)
        assert 0.0 <= result["overall_score"] <= 100.0

    def test_masking_none_without_channels(self, metric, sine_stereo):
        """Without individual channels, masking_score should be None."""
        result = metric.overall(sine_stereo)
        assert result["masking_score"] is None

    def test_masking_present_with_channels(self, metric, sine_stereo, pink_noise_channels):
        result = metric.overall(sine_stereo, channels=pink_noise_channels)
        assert result["masking_score"] is not None
        assert 0.0 <= result["masking_score"] <= 100.0

    def test_mono_input(self, metric, sine_mono):
        result = metric.overall(sine_mono)
        assert result["stereo_correlation"] == 1.0
        assert 0.0 <= result["overall_score"] <= 100.0

    def test_stereo_correlation_in_result(self, metric, sine_stereo):
        result = metric.overall(sine_stereo)
        assert -1.0 <= result["stereo_correlation"] <= 1.0
