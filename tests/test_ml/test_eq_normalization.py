"""
Tests for backend.ml.eq_normalization — EQ band dataclass, reference profiles,
spectral profile computation, EQ correction fitting, and convenience API.

All tests use numpy-generated audio. scipy.optimize is optional.
"""

import numpy as np
import pytest

from backend.ml.eq_normalization import (
    EQBand,
    REFERENCE_PROFILES,
    compute_spectral_profile,
    compute_correction,
    _fallback_correction,
    get_reference_profile,
    compute_channel_eq,
    HAS_SCIPY_OPT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sine_1k():
    """1-second 1 kHz sine at 48 kHz."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float64)
    return np.sin(2 * np.pi * 1000 * t) * 0.5


@pytest.fixture
def low_freq_signal():
    """1-second 80 Hz sine."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float64)
    return np.sin(2 * np.pi * 80 * t) * 0.6


@pytest.fixture
def broadband_noise():
    """1-second white noise."""
    rng = np.random.default_rng(55)
    return rng.standard_normal(48000).astype(np.float64) * 0.3


# ---------------------------------------------------------------------------
# EQBand dataclass
# ---------------------------------------------------------------------------

class TestEQBand:

    def test_creation(self):
        band = EQBand(freq=1000.0, gain_db=-3.0, Q=1.4)
        assert band.freq == 1000.0
        assert band.gain_db == -3.0
        assert band.Q == 1.4

    def test_default_values_not_required(self):
        """EQBand should require all three fields."""
        with pytest.raises(TypeError):
            EQBand(freq=500.0)


# ---------------------------------------------------------------------------
# REFERENCE_PROFILES dictionary
# ---------------------------------------------------------------------------

class TestReferenceProfiles:

    def test_has_11_instruments(self):
        assert len(REFERENCE_PROFILES) == 11

    def test_known_instruments_present(self):
        expected = [
            "kick", "snare", "vocals", "bass_guitar", "electric_guitar",
            "acoustic_guitar", "keys", "hihat", "overheads", "brass", "strings",
        ]
        for inst in expected:
            assert inst in REFERENCE_PROFILES, f"{inst} missing"

    def test_each_profile_has_freqs_and_mags(self):
        for inst, profile in REFERENCE_PROFILES.items():
            assert "freqs" in profile, f"{inst} missing 'freqs'"
            assert "mags" in profile, f"{inst} missing 'mags'"
            assert len(profile["freqs"]) == len(profile["mags"]), (
                f"{inst} freqs/mags length mismatch"
            )

    def test_profiles_have_10_bands(self):
        """All profiles should have exactly 10 frequency bands."""
        for inst, profile in REFERENCE_PROFILES.items():
            assert len(profile["freqs"]) == 10, f"{inst} has {len(profile['freqs'])} bands"

    def test_freqs_are_monotonically_increasing(self):
        for inst, profile in REFERENCE_PROFILES.items():
            freqs = profile["freqs"]
            assert np.all(np.diff(freqs) > 0), f"{inst} freqs not monotonic"


# ---------------------------------------------------------------------------
# compute_spectral_profile
# ---------------------------------------------------------------------------

class TestComputeSpectralProfile:

    def test_returns_two_arrays(self, sine_1k):
        freqs, mags = compute_spectral_profile(sine_1k, sr=48000)
        assert isinstance(freqs, np.ndarray)
        assert isinstance(mags, np.ndarray)
        assert len(freqs) == len(mags)

    def test_freqs_nonnegative(self, sine_1k):
        freqs, _ = compute_spectral_profile(sine_1k, sr=48000)
        assert np.all(freqs >= 0)

    def test_peak_near_1khz(self, sine_1k):
        """A 1 kHz sine should have its peak near 1000 Hz."""
        freqs, mags = compute_spectral_profile(sine_1k, sr=48000)
        peak_idx = np.argmax(mags)
        peak_freq = freqs[peak_idx]
        assert 900 < peak_freq < 1100, f"Peak at {peak_freq} Hz, expected ~1000 Hz"

    def test_stereo_reduced_to_mono(self):
        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float64)
        stereo = np.stack([
            np.sin(2 * np.pi * 440 * t),
            np.sin(2 * np.pi * 880 * t),
        ])
        freqs, mags = compute_spectral_profile(stereo, sr=sr)
        assert len(freqs) > 0

    def test_short_signal(self):
        """Signal shorter than FFT size should still work."""
        audio = np.sin(np.linspace(0, 1, 256)) * 0.5
        freqs, mags = compute_spectral_profile(audio, sr=48000, n_fft=4096)
        assert len(freqs) == 4096 // 2 + 1


# ---------------------------------------------------------------------------
# get_reference_profile
# ---------------------------------------------------------------------------

class TestGetReferenceProfile:

    def test_known_instrument(self):
        result = get_reference_profile("vocals")
        assert result is not None
        freqs, mags = result
        assert len(freqs) == 10

    def test_unknown_returns_none(self):
        result = get_reference_profile("theremin")
        assert result is None

    def test_returns_copies(self):
        """Returned arrays should be copies, not references."""
        f1, m1 = get_reference_profile("kick")
        f2, m2 = get_reference_profile("kick")
        f1[0] = 99999.0
        assert f2[0] != 99999.0


# ---------------------------------------------------------------------------
# _fallback_correction
# ---------------------------------------------------------------------------

class TestFallbackCorrection:

    def test_returns_list_of_eqbands(self):
        freqs = np.linspace(20, 20000, 200)
        deviation = np.sin(np.linspace(0, 4 * np.pi, 200)) * 5.0
        bands = _fallback_correction(freqs, deviation, n_bands=4)
        assert len(bands) == 4
        for b in bands:
            assert isinstance(b, EQBand)

    def test_bands_sorted_by_frequency(self):
        freqs = np.linspace(20, 20000, 200)
        deviation = np.random.default_rng(11).standard_normal(200) * 3.0
        bands = _fallback_correction(freqs, deviation, n_bands=4)
        band_freqs = [b.freq for b in bands]
        assert band_freqs == sorted(band_freqs)

    def test_gains_within_max(self):
        freqs = np.linspace(20, 20000, 200)
        deviation = np.ones(200) * 20.0  # large deviation
        bands = _fallback_correction(freqs, deviation, n_bands=4, max_gain_db=12.0)
        for b in bands:
            assert abs(b.gain_db) <= 12.0


# ---------------------------------------------------------------------------
# compute_correction
# ---------------------------------------------------------------------------

class TestComputeCorrection:

    def test_returns_list_of_eqbands(self, sine_1k):
        actual = compute_spectral_profile(sine_1k, sr=48000)
        target = get_reference_profile("vocals")
        bands = compute_correction(actual, target, n_bands=4)
        assert len(bands) == 4
        for b in bands:
            assert isinstance(b, EQBand)

    def test_bands_sorted_by_frequency(self, broadband_noise):
        actual = compute_spectral_profile(broadband_noise, sr=48000)
        target = get_reference_profile("kick")
        bands = compute_correction(actual, target, n_bands=4)
        band_freqs = [b.freq for b in bands]
        assert band_freqs == sorted(band_freqs)

    def test_identical_profiles_give_small_gains(self):
        """If actual == target, corrections should have very small gains."""
        freqs = np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        mags = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        actual = (freqs, mags)
        target = (freqs.copy(), mags.copy())
        bands = compute_correction(actual, target, n_bands=4)
        for b in bands:
            assert abs(b.gain_db) < 3.0, (
                f"Expected small gain for identical profiles, got {b.gain_db}"
            )

    def test_empty_freq_range(self):
        """If frequency range is empty after filtering, return default bands."""
        freqs = np.array([0.0, 1.0, 2.0])  # below audible range
        mags = np.array([0.0, 0.0, 0.0])
        bands = compute_correction((freqs, mags), (freqs, mags), n_bands=4)
        assert len(bands) == 4


# ---------------------------------------------------------------------------
# compute_channel_eq
# ---------------------------------------------------------------------------

class TestComputeChannelEQ:

    def test_returns_list(self, sine_1k):
        bands = compute_channel_eq(sine_1k, sr=48000, instrument_type="vocals")
        assert isinstance(bands, list)

    def test_returns_n_bands(self, sine_1k):
        bands = compute_channel_eq(sine_1k, sr=48000, instrument_type="kick", n_bands=4)
        assert len(bands) == 4

    def test_unknown_instrument_uses_flat_target(self, broadband_noise):
        """Unknown instrument should still return EQ bands (flat target)."""
        bands = compute_channel_eq(broadband_noise, sr=48000, instrument_type="didgeridoo")
        assert len(bands) > 0

    def test_none_instrument(self, broadband_noise):
        bands = compute_channel_eq(broadband_noise, sr=48000, instrument_type=None)
        assert len(bands) > 0

    def test_all_bands_are_eqband(self, sine_1k):
        bands = compute_channel_eq(sine_1k, sr=48000, instrument_type="snare")
        for b in bands:
            assert isinstance(b, EQBand)
            assert isinstance(b.freq, float)
            assert isinstance(b.gain_db, float)
            assert isinstance(b.Q, float)
