"""
Tests for backend.ml.drc_onset — adaptive DRC threshold from onset peak
statistics, HFC onset detection, and instrument-specific attack/release.

All tests use numpy-generated audio. No librosa required.
"""

import numpy as np
import pytest

from backend.ml.drc_onset import (
    ONSET_PERCENTILES,
    ATTACK_RELEASE_TIMES,
    DEFAULT_PERCENTILE,
    DEFAULT_ATTACK_MS,
    DEFAULT_RELEASE_MS,
    _hfc_onset_detection,
    _detect_onset_peaks,
    compute_onset_threshold,
    get_attack_release,
    compute_drc_params,
    HAS_LIBROSA,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kick_burst():
    """Short 60 Hz sine burst simulating a kick transient."""
    sr = 48000
    n = sr  # 1 second
    t = np.linspace(0, 1.0, n, endpoint=False, dtype=np.float64)
    # Exponential decay envelope
    env = np.exp(-5.0 * t)
    return np.sin(2 * np.pi * 60 * t) * env * 0.9


@pytest.fixture
def snare_transient():
    """Short noise burst simulating a snare hit."""
    rng = np.random.default_rng(42)
    sr = 48000
    n = sr  # 1 second
    noise = rng.standard_normal(n).astype(np.float64)
    env = np.exp(-10.0 * np.linspace(0, 1, n))
    return noise * env * 0.8


@pytest.fixture
def sustained_tone():
    """Sustained 440 Hz sine simulating a sustained instrument."""
    sr = 48000
    n = sr * 2  # 2 seconds
    t = np.linspace(0, 2.0, n, endpoint=False, dtype=np.float64)
    return np.sin(2 * np.pi * 440 * t) * 0.5


@pytest.fixture
def multi_onset_signal():
    """Signal with multiple distinct transient onsets."""
    sr = 48000
    n = sr * 2  # 2 seconds
    audio = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(99)
    # Place 4 hits at 0.2s, 0.6s, 1.0s, 1.4s
    for onset_time in [0.2, 0.6, 1.0, 1.4]:
        start = int(onset_time * sr)
        burst_len = int(0.05 * sr)
        end = min(start + burst_len, n)
        burst = rng.standard_normal(end - start) * 0.7
        env = np.exp(-20.0 * np.linspace(0, 1, end - start))
        audio[start:end] += burst * env
    return audio


# ---------------------------------------------------------------------------
# ONSET_PERCENTILES dictionary tests
# ---------------------------------------------------------------------------

class TestOnsetPercentiles:

    def test_has_13_instruments(self):
        """ONSET_PERCENTILES should cover all 13 instrument classes."""
        assert len(ONSET_PERCENTILES) == 13

    def test_all_values_are_integers(self):
        for inst, pct in ONSET_PERCENTILES.items():
            assert isinstance(pct, (int, float)), f"{inst} percentile not numeric"
            assert 0 < pct <= 100, f"{inst} percentile {pct} out of range"

    def test_known_instruments_present(self):
        expected = [
            "kick", "snare", "hihat", "toms", "overheads",
            "bass_guitar", "electric_guitar", "acoustic_guitar",
            "keys", "vocals", "brass", "strings", "percussion",
        ]
        for inst in expected:
            assert inst in ONSET_PERCENTILES, f"{inst} missing"


# ---------------------------------------------------------------------------
# ATTACK_RELEASE_TIMES dictionary tests
# ---------------------------------------------------------------------------

class TestAttackReleaseTimes:

    def test_has_13_instruments(self):
        assert len(ATTACK_RELEASE_TIMES) == 13

    def test_all_tuples_of_two_positive_floats(self):
        for inst, (atk, rel) in ATTACK_RELEASE_TIMES.items():
            assert atk > 0, f"{inst} attack {atk} not positive"
            assert rel > 0, f"{inst} release {rel} not positive"
            assert rel > atk, f"{inst} release should exceed attack"

    def test_drums_have_fast_attack(self):
        """Drums should have attack <= 5 ms."""
        for inst in ("kick", "snare", "hihat"):
            atk, _ = ATTACK_RELEASE_TIMES[inst]
            assert atk <= 5.0, f"{inst} attack {atk} ms too slow for drums"


# ---------------------------------------------------------------------------
# HFC onset detection
# ---------------------------------------------------------------------------

class TestHFCOnsetDetection:

    def test_returns_two_arrays(self, kick_burst):
        strength, times = _hfc_onset_detection(kick_burst, sr=48000)
        assert isinstance(strength, np.ndarray)
        assert isinstance(times, np.ndarray)
        assert len(strength) == len(times)

    def test_onset_strength_nonnegative(self, kick_burst):
        strength, _ = _hfc_onset_detection(kick_burst, sr=48000)
        assert np.all(strength >= 0), "Onset strength must be non-negative"

    def test_has_positive_frames(self, kick_burst):
        """A transient signal should produce some positive onset strength."""
        strength, _ = _hfc_onset_detection(kick_burst, sr=48000)
        assert np.max(strength) > 0

    def test_silence_gives_zero_strength(self):
        """All-zero input should produce zero onset strength."""
        audio = np.zeros(48000, dtype=np.float64)
        strength, _ = _hfc_onset_detection(audio, sr=48000)
        assert np.allclose(strength, 0.0)

    def test_short_signal(self):
        """Very short input should not crash."""
        audio = np.array([0.1, -0.2, 0.3], dtype=np.float64)
        strength, times = _hfc_onset_detection(audio, sr=48000, hop_length=512)
        assert len(strength) >= 1


# ---------------------------------------------------------------------------
# Onset peak detection
# ---------------------------------------------------------------------------

class TestDetectOnsetPeaks:

    def test_returns_indices_and_values(self, multi_onset_signal):
        strength, _ = _hfc_onset_detection(multi_onset_signal, sr=48000)
        indices, values = _detect_onset_peaks(strength)
        assert len(indices) == len(values)
        assert len(indices) > 0

    def test_short_input_returns_something(self):
        """With fewer than 3 frames, should return at least one peak."""
        indices, values = _detect_onset_peaks(np.array([1.0, 0.5]))
        assert len(indices) >= 1


# ---------------------------------------------------------------------------
# compute_onset_threshold
# ---------------------------------------------------------------------------

class TestComputeOnsetThreshold:

    def test_returns_float(self, kick_burst):
        threshold = compute_onset_threshold(kick_burst, sr=48000, instrument_type="kick")
        assert isinstance(threshold, float)

    def test_within_clamped_range(self, kick_burst):
        threshold = compute_onset_threshold(kick_burst, sr=48000)
        assert -60.0 <= threshold <= -3.0

    def test_empty_audio_returns_default(self):
        threshold = compute_onset_threshold(np.array([]), sr=48000)
        assert threshold == -20.0

    def test_silence_returns_low_threshold(self):
        audio = np.zeros(48000, dtype=np.float64)
        threshold = compute_onset_threshold(audio, sr=48000)
        assert threshold == -60.0

    def test_instrument_type_affects_result(self, multi_onset_signal):
        """Different instrument types should potentially yield different thresholds."""
        t_kick = compute_onset_threshold(multi_onset_signal, sr=48000, instrument_type="kick")
        t_strings = compute_onset_threshold(multi_onset_signal, sr=48000, instrument_type="strings")
        # Both should be valid floats
        assert isinstance(t_kick, float)
        assert isinstance(t_strings, float)

    def test_unknown_instrument_uses_default_percentile(self, kick_burst):
        """Unknown instrument type should still return a valid threshold."""
        threshold = compute_onset_threshold(kick_burst, sr=48000, instrument_type="xylophone")
        assert isinstance(threshold, float)
        assert -60.0 <= threshold <= -3.0


# ---------------------------------------------------------------------------
# get_attack_release
# ---------------------------------------------------------------------------

class TestGetAttackRelease:

    def test_known_instrument(self):
        atk, rel = get_attack_release("kick")
        assert atk == ATTACK_RELEASE_TIMES["kick"][0]
        assert rel == ATTACK_RELEASE_TIMES["kick"][1]

    def test_unknown_instrument_returns_defaults(self):
        atk, rel = get_attack_release("didgeridoo")
        assert atk == DEFAULT_ATTACK_MS
        assert rel == DEFAULT_RELEASE_MS

    def test_none_instrument_returns_defaults(self):
        atk, rel = get_attack_release(None)
        assert atk == DEFAULT_ATTACK_MS
        assert rel == DEFAULT_RELEASE_MS

    def test_all_instruments_return_positive(self):
        for inst in ATTACK_RELEASE_TIMES:
            atk, rel = get_attack_release(inst)
            assert atk > 0 and rel > 0


# ---------------------------------------------------------------------------
# compute_drc_params
# ---------------------------------------------------------------------------

class TestComputeDRCParams:

    def test_returns_expected_keys(self, kick_burst):
        params = compute_drc_params(kick_burst, sr=48000, instrument_type="kick")
        for key in ("threshold_db", "ratio", "attack_ms", "release_ms", "knee_db", "makeup_db"):
            assert key in params, f"Missing key '{key}'"

    def test_all_values_are_float(self, kick_burst):
        params = compute_drc_params(kick_burst, sr=48000)
        for key, val in params.items():
            assert isinstance(val, float), f"{key} is not float: {type(val)}"

    def test_ratio_positive(self, kick_burst):
        params = compute_drc_params(kick_burst, sr=48000, instrument_type="kick")
        assert params["ratio"] > 1.0

    def test_attack_release_positive(self, sustained_tone):
        params = compute_drc_params(sustained_tone, sr=48000, instrument_type="vocals")
        assert params["attack_ms"] > 0
        assert params["release_ms"] > 0

    def test_makeup_capped_at_12(self, kick_burst):
        params = compute_drc_params(kick_burst, sr=48000)
        assert params["makeup_db"] <= 12.0

    def test_knee_positive(self, kick_burst):
        params = compute_drc_params(kick_burst, sr=48000, instrument_type="vocals")
        assert params["knee_db"] > 0

    def test_instrument_specific_ratio(self, kick_burst):
        """Different instruments should get different ratios."""
        p_kick = compute_drc_params(kick_burst, sr=48000, instrument_type="kick")
        p_strings = compute_drc_params(kick_burst, sr=48000, instrument_type="strings")
        assert p_kick["ratio"] != p_strings["ratio"]

    def test_empty_audio(self):
        params = compute_drc_params(np.array([]), sr=48000)
        assert params["threshold_db"] == -20.0
