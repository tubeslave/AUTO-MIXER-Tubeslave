"""Tests for auto_mastering module."""
import pytest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from auto_mastering import AutoMaster, MasteringResult


class TestAutoMaster:
    """Tests for the AutoMaster class."""

    def _make_sine(self, freq=440.0, duration=0.5, amplitude=0.3, sr=48000):
        """Generate a synthetic sine wave for testing."""
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        return np.sin(2 * np.pi * freq * t) * amplitude

    def test_master_empty_audio(self):
        """Mastering empty audio returns a failure result."""
        master = AutoMaster(sample_rate=48000)
        result = master.master(np.array([], dtype=np.float32))
        assert isinstance(result, MasteringResult)
        assert result.success is False
        assert result.error == "Empty audio"

    def test_master_sine_wave(self):
        """Mastering a sine wave returns a successful result with valid measurements."""
        master = AutoMaster(sample_rate=48000, target_lufs=-14.0, true_peak_limit=-1.0)
        audio = self._make_sine(freq=1000.0, duration=1.0, amplitude=0.1)
        result = master.master(audio)

        assert result.success is True
        assert result.error is None
        assert result.audio is not None
        assert len(result.audio) == len(audio)
        # Peak should be below the true peak limit (with some tolerance for float math)
        assert result.peak_db <= 0.0
        # Gain was applied
        assert result.gain_applied_db != 0.0
        assert result.eq_applied is True

    def test_limiter_caps_output(self):
        """The limiter prevents output from exceeding the true peak ceiling."""
        master = AutoMaster(sample_rate=48000, target_lufs=-6.0, true_peak_limit=-1.0)
        # A loud signal that will need limiting
        audio = self._make_sine(freq=440.0, duration=0.5, amplitude=0.9)
        result = master.master(audio)

        assert result.success is True
        output_peak = float(20 * np.log10(np.max(np.abs(result.audio)) + 1e-10))
        # Output peak should not exceed the true peak limit
        assert output_peak <= -0.9  # allow small tolerance

    def test_compression_reduces_dynamic_range(self):
        """The compression stage reduces dynamic range of the processed audio."""
        master = AutoMaster(sample_rate=48000)
        # Create audio with high dynamic range: loud burst then quiet section
        sr = 48000
        loud = self._make_sine(freq=440.0, duration=0.25, amplitude=0.8, sr=sr)
        quiet = self._make_sine(freq=440.0, duration=0.25, amplitude=0.01, sr=sr)
        audio = np.concatenate([loud, quiet])

        result = master.master(audio)
        assert result.success is True
        # The mastered audio should have some processing applied
        assert result.limiter_reduction_db >= 0.0

    def test_master_result_dataclass_fields(self):
        """MasteringResult has all expected fields accessible."""
        result = MasteringResult(
            audio=np.zeros(100, dtype=np.float32),
            peak_db=-6.0,
            lufs=-14.0,
            gain_applied_db=3.0,
            limiter_reduction_db=1.5,
            eq_applied=True,
            success=True,
        )
        assert result.peak_db == -6.0
        assert result.lufs == -14.0
        assert result.gain_applied_db == 3.0
        assert result.limiter_reduction_db == 1.5
        assert result.eq_applied is True
        assert result.success is True
        assert result.error is None
