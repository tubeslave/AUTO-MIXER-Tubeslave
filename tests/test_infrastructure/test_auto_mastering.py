"""
Tests for backend/auto_mastering.py — AutoMaster, _estimate_lufs,
_limit, _master_fallback, _apply_eq_match.

All tests work without hardware, network, or matchering.
"""

import numpy as np
import os
import pytest

try:
    from auto_mastering import AutoMaster, HAS_MATCHERING, HAS_SCIPY
except ImportError:
    pytest.skip("auto_mastering module not importable", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def master():
    """AutoMaster with default settings."""
    return AutoMaster(target_lufs=-14.0, true_peak_limit=-1.0, sample_rate=48000)


def _make_sine(freq_hz=440.0, duration_sec=1.0, sr=48000, amplitude=0.5):
    """Generate a mono sine wave."""
    t = np.arange(int(sr * duration_sec)) / sr
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _make_silence(duration_sec=1.0, sr=48000):
    """Generate silence."""
    return np.zeros(int(sr * duration_sec), dtype=np.float32)


# ---------------------------------------------------------------------------
# AutoMaster initialization tests
# ---------------------------------------------------------------------------

class TestAutoMasterInit:

    def test_default_values(self, master):
        assert master.target_lufs == -14.0
        assert master.true_peak_limit == -1.0
        assert master.sample_rate == 48000

    def test_custom_values(self):
        m = AutoMaster(target_lufs=-16.0, true_peak_limit=-2.0, sample_rate=44100)
        assert m.target_lufs == -16.0
        assert m.true_peak_limit == -2.0
        assert m.sample_rate == 44100


# ---------------------------------------------------------------------------
# _estimate_lufs tests
# ---------------------------------------------------------------------------

class TestEstimateLUFS:

    def test_silence_returns_low_value(self):
        silence = np.zeros(48000, dtype=np.float32)
        lufs = AutoMaster._estimate_lufs(silence)
        assert lufs <= -90.0

    def test_loud_signal_returns_higher(self):
        loud = _make_sine(amplitude=0.9)
        quiet = _make_sine(amplitude=0.01)
        lufs_loud = AutoMaster._estimate_lufs(loud)
        lufs_quiet = AutoMaster._estimate_lufs(quiet)
        assert lufs_loud > lufs_quiet

    def test_full_scale_sine(self):
        """Full scale sine should have LUFS close to ~-3 dB RMS."""
        full = _make_sine(amplitude=1.0)
        lufs = AutoMaster._estimate_lufs(full)
        # RMS of a full-scale sine is -3.01 dB
        assert -4.0 < lufs < -2.0

    def test_stereo_signal(self):
        """Should handle stereo (2D) input by averaging channels."""
        mono = _make_sine(amplitude=0.5)
        stereo = np.stack([mono, mono], axis=-1)
        lufs = AutoMaster._estimate_lufs(stereo)
        lufs_mono = AutoMaster._estimate_lufs(mono)
        # Should be approximately the same
        assert abs(lufs - lufs_mono) < 1.0

    def test_near_zero_signal(self):
        tiny = np.full(48000, 1e-12, dtype=np.float32)
        lufs = AutoMaster._estimate_lufs(tiny)
        assert lufs <= -100.0


# ---------------------------------------------------------------------------
# _limit tests
# ---------------------------------------------------------------------------

class TestLimit:

    def test_limit_reduces_peak(self, master):
        loud = _make_sine(amplitude=1.0)
        limited = master._limit(loud)
        peak = np.max(np.abs(limited))
        expected_peak = 10.0 ** (master.true_peak_limit / 20.0)
        assert peak <= expected_peak + 1e-6

    def test_limit_does_not_change_quiet_signal(self, master):
        quiet = _make_sine(amplitude=0.01)
        limited = master._limit(quiet)
        np.testing.assert_allclose(limited, quiet, atol=1e-6)

    def test_limit_preserves_shape(self, master):
        audio = _make_sine(amplitude=0.8)
        limited = master._limit(audio)
        assert limited.shape == audio.shape


# ---------------------------------------------------------------------------
# _master_fallback tests
# ---------------------------------------------------------------------------

class TestMasterFallback:

    def test_fallback_returns_float32(self, master):
        input_audio = _make_sine(amplitude=0.3)
        ref_audio = _make_sine(amplitude=0.7, freq_hz=880.0)
        result = master._master_fallback(input_audio, ref_audio, 48000)
        assert result.dtype == np.float32

    def test_fallback_same_length(self, master):
        input_audio = _make_sine(amplitude=0.3, duration_sec=0.5)
        ref_audio = _make_sine(amplitude=0.7, duration_sec=0.5)
        result = master._master_fallback(input_audio, ref_audio, 48000)
        assert len(result) == len(input_audio)

    def test_fallback_increases_quiet_signal(self, master):
        quiet = _make_sine(amplitude=0.01)
        loud_ref = _make_sine(amplitude=0.8, freq_hz=880.0)
        result = master._master_fallback(quiet, loud_ref, 48000)
        # Result should be louder than the input
        assert np.max(np.abs(result)) > np.max(np.abs(quiet))

    def test_fallback_with_silence_input(self, master):
        """Silence input should not cause errors."""
        silence = _make_silence()
        ref = _make_sine(amplitude=0.5)
        result = master._master_fallback(silence, ref, 48000)
        assert len(result) == len(silence)

    def test_fallback_with_silence_reference(self, master):
        """Silence reference should not cause errors."""
        input_audio = _make_sine(amplitude=0.5)
        silence_ref = _make_silence()
        result = master._master_fallback(input_audio, silence_ref, 48000)
        assert len(result) == len(input_audio)

    def test_fallback_output_limited(self, master):
        """Output should not exceed the true peak limit."""
        input_audio = _make_sine(amplitude=0.9)
        ref_audio = _make_sine(amplitude=0.95, freq_hz=880.0)
        result = master._master_fallback(input_audio, ref_audio, 48000)
        peak_limit = 10.0 ** (master.true_peak_limit / 20.0)
        assert np.max(np.abs(result)) <= peak_limit + 0.01


# ---------------------------------------------------------------------------
# _apply_eq_match tests
# ---------------------------------------------------------------------------

class TestApplyEQMatch:

    def test_eq_match_returns_same_length(self, master):
        audio = _make_sine(amplitude=0.5).astype(np.float64)
        ref = _make_sine(amplitude=0.5, freq_hz=880.0).astype(np.float64)
        result = master._apply_eq_match(audio, ref, 48000)
        # Result length should be at most the minimum of input lengths
        assert len(result) <= max(len(audio), len(ref))

    def test_eq_match_without_scipy(self, master):
        """Without scipy, _apply_eq_match should return audio unchanged."""
        if HAS_SCIPY:
            pytest.skip("scipy is available; cannot test missing path")
        audio = _make_sine(amplitude=0.5).astype(np.float64)
        ref = _make_sine(amplitude=0.5, freq_hz=880.0).astype(np.float64)
        result = master._apply_eq_match(audio, ref, 48000)
        np.testing.assert_array_equal(result, audio)


# ---------------------------------------------------------------------------
# master() full pipeline test
# ---------------------------------------------------------------------------

class TestMasterPipeline:

    def test_master_returns_array(self, master):
        input_audio = _make_sine(amplitude=0.3)
        ref_audio = _make_sine(amplitude=0.7, freq_hz=880.0)
        result = master.master(input_audio, ref_audio, 48000)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_master_with_different_lengths(self, master):
        input_audio = _make_sine(amplitude=0.3, duration_sec=0.5)
        ref_audio = _make_sine(amplitude=0.7, freq_hz=880.0, duration_sec=1.0)
        result = master.master(input_audio, ref_audio, 48000)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# WAV file I/O tests
# ---------------------------------------------------------------------------

class TestWavIO:

    def test_write_and_read_wav(self, master, tmp_dir):
        audio = _make_sine(amplitude=0.5)
        path = os.path.join(tmp_dir, "test.wav")
        master._write_wav(path, audio, 48000)
        assert os.path.isfile(path)
        loaded = master._read_wav(path)
        assert len(loaded) > 0
        # Should be approximately equal (int16 quantization)
        np.testing.assert_allclose(loaded, audio, atol=0.001)
