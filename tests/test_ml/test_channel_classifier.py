"""
Tests for backend.ml.channel_classifier — MFCC + spectral feature extraction,
name-based classification, and ChannelClassifier ML pipeline.

Uses numpy-generated audio and skips sklearn-dependent tests gracefully.
"""

import numpy as np
import pytest

from backend.ml.channel_classifier import (
    extract_features,
    classify_from_name,
    ChannelClassifier,
    INSTRUMENT_CLASSES,
    librosa_mel_frequencies,
    _create_mel_filterbank,
    HAS_SKLEARN,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sine_440():
    """1-second 440 Hz sine at 48 kHz."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def low_freq_burst():
    """Short 60 Hz sine simulating a kick/bass."""
    sr = 48000
    n = 4096
    t = np.linspace(0, n / sr, n, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 60 * t) * 0.9


@pytest.fixture
def high_freq_transient():
    """Short noise burst with high-frequency content."""
    rng = np.random.default_rng(123)
    noise = rng.standard_normal(4096).astype(np.float32)
    # Simple high-pass via differencing
    return np.diff(noise, prepend=0.0)


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------

class TestExtractFeatures:

    def test_output_length(self, sine_440):
        features = extract_features(sine_440, sr=48000)
        assert features.shape == (36,), f"Expected 36 features, got {features.shape}"

    def test_output_dtype(self, sine_440):
        features = extract_features(sine_440, sr=48000)
        assert features.dtype == np.float32

    def test_no_nan_or_inf(self, sine_440):
        features = extract_features(sine_440, sr=48000)
        assert not np.any(np.isnan(features)), "Features contain NaN"
        assert not np.any(np.isinf(features)), "Features contain Inf"

    def test_stereo_reduced_to_mono(self):
        """Stereo input should be averaged to mono internally."""
        sr = 48000
        t = np.linspace(0, 0.5, sr // 2, endpoint=False, dtype=np.float32)
        stereo = np.stack([np.sin(2 * np.pi * 440 * t),
                           np.sin(2 * np.pi * 880 * t)])
        features = extract_features(stereo, sr=sr)
        assert features.shape == (36,)

    def test_silent_input(self):
        """All-zero input should still produce valid features."""
        audio = np.zeros(4096, dtype=np.float32)
        features = extract_features(audio, sr=48000)
        assert features.shape == (36,)
        assert not np.any(np.isnan(features))


# ---------------------------------------------------------------------------
# Mel-frequency helpers (numpy fallback path)
# ---------------------------------------------------------------------------

class TestMelHelpers:

    def test_mel_frequencies_count(self):
        freqs = librosa_mel_frequencies(42, fmin=0.0, fmax=24000.0)
        assert len(freqs) == 42

    def test_mel_frequencies_order(self):
        freqs = librosa_mel_frequencies(42, fmin=0.0, fmax=24000.0)
        assert np.all(np.diff(freqs) > 0), "Frequencies must be monotonically increasing"

    def test_mel_filterbank_shape(self):
        n_mels = 40
        n_fft = 2048
        sr = 48000
        mel_freqs = librosa_mel_frequencies(n_mels + 2, fmin=0, fmax=sr / 2)
        fb = _create_mel_filterbank(sr, n_fft, n_mels, mel_freqs)
        assert fb.shape == (n_mels, n_fft // 2 + 1)

    def test_mel_filterbank_nonnegative(self):
        n_mels = 40
        n_fft = 2048
        sr = 48000
        mel_freqs = librosa_mel_frequencies(n_mels + 2, fmin=0, fmax=sr / 2)
        fb = _create_mel_filterbank(sr, n_fft, n_mels, mel_freqs)
        assert np.all(fb >= 0), "Filter bank values must be non-negative"


# ---------------------------------------------------------------------------
# Name-based classification
# ---------------------------------------------------------------------------

class TestClassifyFromName:

    @pytest.mark.parametrize("name,expected", [
        ("Kick Drum", "kick"),
        ("BD", "kick"),
        ("Snare Top", "snare"),
        ("SNR", "snare"),
        ("Hi-Hat", "hihat"),
        ("HH", "hihat"),
        ("Tom 1", "toms"),
        ("Floor", "toms"),
        ("OH L", "overheads"),
        ("Overhead", "overheads"),
        ("Bass Guitar", "bass_guitar"),
        ("DI Bass", "bass_guitar"),
        ("E-Gtr", "electric_guitar"),
        ("Lead Gtr", "electric_guitar"),
        ("Acoustic Guitar", "acoustic_guitar"),
        ("A Gtr", "acoustic_guitar"),
        ("Keys", "keys"),
        ("Piano", "keys"),
        ("Synth", "keys"),
        ("Lead Voc", "vocals"),
        ("BGV", "vocals"),
        ("Trumpet", "brass"),
        ("Sax", "brass"),
        ("Violin", "strings"),
        ("Conga", "percussion"),
        ("Shaker", "percussion"),
    ])
    def test_known_patterns(self, name, expected):
        cls, conf = classify_from_name(name)
        assert cls == expected, f"'{name}' should classify as '{expected}', got '{cls}'"
        assert conf > 0

    def test_unknown_name_returns_none(self):
        cls, conf = classify_from_name("xyzzy")
        assert cls is None
        assert conf == 0.0

    def test_empty_name(self):
        cls, conf = classify_from_name("")
        assert cls is None
        assert conf == 0.0

    def test_none_name(self):
        cls, conf = classify_from_name(None)
        assert cls is None


# ---------------------------------------------------------------------------
# ChannelClassifier
# ---------------------------------------------------------------------------

class TestChannelClassifier:

    def test_heuristic_classify_returns_valid_class(self, low_freq_burst):
        cc = ChannelClassifier()
        cls, conf = cc._heuristic_classify(low_freq_burst, sr=48000)
        assert cls in INSTRUMENT_CLASSES
        assert 0 <= conf <= 1.0

    def test_heuristic_high_freq_not_kick(self, high_freq_transient):
        cc = ChannelClassifier()
        cls, _ = cc._heuristic_classify(high_freq_transient, sr=48000)
        assert cls != "kick", "High-freq transient should not be classified as kick"

    def test_classify_with_fallback_uses_name(self, sine_440):
        """When ML confidence is low, name fallback should take over."""
        cc = ChannelClassifier()
        cls, conf = cc.classify_with_fallback(sine_440, sr=48000, channel_name="Kick Drum")
        # Heuristic is low-confidence, so name match should win
        assert cls == "kick"
        assert conf >= 0.5

    def test_classify_short_audio(self):
        """Very short audio (<256 samples) returns default guess."""
        cc = ChannelClassifier()
        audio = np.zeros(100, dtype=np.float32)
        cls, conf = cc._heuristic_classify(audio, sr=48000)
        assert cls is not None

    def test_train_and_classify_sklearn(self):
        """Train a mini classifier and verify predictions (sklearn only)."""
        if not HAS_SKLEARN:
            pytest.skip("sklearn not installed")
        cc = ChannelClassifier()
        rng = np.random.default_rng(42)
        n_samples = 50
        n_features = 36
        X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
        y = rng.integers(0, len(INSTRUMENT_CLASSES), size=n_samples)
        acc = cc.train(X, y)
        assert 0 <= acc <= 1.0
        # Classify one of the training samples
        cls, conf = cc.classify(
            np.zeros(4096, dtype=np.float32), sr=48000
        )
        assert cls in INSTRUMENT_CLASSES
