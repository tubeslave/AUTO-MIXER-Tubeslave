"""
Tests for backend.ml.losses — multi-resolution STFT, stereo, and combined
mixing losses.

All tests use numpy arrays directly. Torch-based classes are tested only
when torch is available (otherwise pytest.skip).
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import helpers with fallback guards
# ---------------------------------------------------------------------------
from backend.ml.losses import (
    _numpy_stft,
    _numpy_spectral_convergence,
    _numpy_log_magnitude_loss,
    MultiResolutionSTFTLoss,
    SumAndDifferenceLoss,
    MixingLoss,
    HAS_TORCH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sine_mono():
    """Generate a 1-second 440 Hz mono sine wave at 48 kHz."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def sine_stereo():
    """Generate a 1-second stereo signal (440 Hz L, 880 Hz R)."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    left = np.sin(2 * np.pi * 440 * t)
    right = np.sin(2 * np.pi * 880 * t)
    return np.stack([left, right], axis=0)


@pytest.fixture
def noise_mono():
    """Short white-noise burst (mono)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(4096).astype(np.float32)


# ---------------------------------------------------------------------------
# Numpy helper tests
# ---------------------------------------------------------------------------

class TestNumpySTFT:
    """Tests for the _numpy_stft helper."""

    def test_output_shape(self, sine_mono):
        fft_size, hop_size, win_size = 1024, 256, 1024
        mag = _numpy_stft(sine_mono, fft_size, hop_size, win_size)
        # Should have freq_bins = fft_size//2 + 1 columns
        assert mag.shape[1] == fft_size // 2 + 1
        # Should have a positive number of frames
        assert mag.shape[0] > 0

    def test_nonnegative_magnitude(self, noise_mono):
        mag = _numpy_stft(noise_mono, 512, 128, 512)
        assert np.all(mag >= 0), "STFT magnitudes must be non-negative"

    def test_identical_signals_low_sc(self, sine_mono):
        """Spectral convergence of identical signals should be ~0."""
        mag = _numpy_stft(sine_mono, 1024, 256, 1024)
        sc = _numpy_spectral_convergence(mag, mag)
        assert sc == pytest.approx(0.0, abs=1e-6)


class TestNumpySpectralHelpers:
    """Tests for numpy spectral convergence and log-magnitude helpers."""

    def test_spectral_convergence_identical(self):
        a = np.ones((10, 5), dtype=np.float32)
        assert _numpy_spectral_convergence(a, a) == pytest.approx(0.0, abs=1e-8)

    def test_spectral_convergence_different(self):
        a = np.ones((10, 5), dtype=np.float32)
        b = np.zeros((10, 5), dtype=np.float32)
        sc = _numpy_spectral_convergence(b, a)
        # pred=zeros, target=ones → sc = ||ones||/||ones|| = 1.0
        assert sc == pytest.approx(1.0, abs=1e-6)

    def test_log_magnitude_identical(self):
        a = np.ones((10, 5), dtype=np.float32) * 0.5
        loss = _numpy_log_magnitude_loss(a, a)
        assert loss == pytest.approx(0.0, abs=1e-5)

    def test_log_magnitude_positive(self):
        rng = np.random.default_rng(99)
        a = np.abs(rng.standard_normal((10, 5)).astype(np.float32)) + 0.01
        b = np.abs(rng.standard_normal((10, 5)).astype(np.float32)) + 0.01
        loss = _numpy_log_magnitude_loss(a, b)
        assert loss > 0


# ---------------------------------------------------------------------------
# MultiResolutionSTFTLoss (works in both torch and numpy paths)
# ---------------------------------------------------------------------------

class TestMultiResolutionSTFTLoss:

    def test_zero_loss_identical_mono(self, sine_mono):
        loss_fn = MultiResolutionSTFTLoss(fft_sizes=(512, 1024))
        if HAS_TORCH:
            import torch
            pred = torch.from_numpy(sine_mono).unsqueeze(0)
            target = pred.clone()
            val = loss_fn(pred, target)
            assert val.item() == pytest.approx(0.0, abs=1e-4)
        else:
            val = loss_fn(sine_mono, sine_mono)
            assert val == pytest.approx(0.0, abs=1e-4)

    def test_positive_loss_different_signals(self, sine_mono, noise_mono):
        loss_fn = MultiResolutionSTFTLoss(fft_sizes=(512,))
        # Ensure both signals have the same length
        min_len = min(len(sine_mono), len(noise_mono))
        sine_trimmed = sine_mono[:min_len]
        noise_trimmed = noise_mono[:min_len]
        if HAS_TORCH:
            import torch
            pred = torch.from_numpy(sine_trimmed).unsqueeze(0)
            target = torch.from_numpy(noise_trimmed).unsqueeze(0)
            val = loss_fn(pred, target)
            assert val.item() > 0
        else:
            val = loss_fn(sine_trimmed, noise_trimmed)
            assert val > 0

    def test_accepts_multichannel(self, sine_stereo):
        loss_fn = MultiResolutionSTFTLoss(fft_sizes=(512,))
        if HAS_TORCH:
            import torch
            pred = torch.from_numpy(sine_stereo).unsqueeze(0)  # (1,2,samples)
            val = loss_fn(pred, pred)
            assert val.item() == pytest.approx(0.0, abs=1e-4)
        else:
            val = loss_fn(sine_stereo, sine_stereo)
            assert val == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# SumAndDifferenceLoss
# ---------------------------------------------------------------------------

class TestSumAndDifferenceLoss:

    def test_zero_loss_identical_stereo(self, sine_stereo):
        loss_fn = SumAndDifferenceLoss()
        if HAS_TORCH:
            import torch
            pred = torch.from_numpy(sine_stereo).unsqueeze(0)
            val = loss_fn(pred, pred)
            assert val.item() == pytest.approx(0.0, abs=1e-6)
        else:
            val = loss_fn(sine_stereo, sine_stereo)
            assert val == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss_different_stereo(self, sine_stereo):
        rng = np.random.default_rng(7)
        noise = rng.standard_normal(sine_stereo.shape).astype(np.float32)
        loss_fn = SumAndDifferenceLoss()
        if HAS_TORCH:
            import torch
            pred = torch.from_numpy(sine_stereo).unsqueeze(0)
            tgt = torch.from_numpy(noise).unsqueeze(0)
            val = loss_fn(pred, tgt)
            assert val.item() > 0
        else:
            val = loss_fn(sine_stereo, noise)
            assert val > 0

    def test_weight_scaling(self, sine_stereo):
        rng = np.random.default_rng(8)
        noise = rng.standard_normal(sine_stereo.shape).astype(np.float32)
        loss_1 = SumAndDifferenceLoss(sum_weight=1.0, diff_weight=1.0)
        loss_2 = SumAndDifferenceLoss(sum_weight=2.0, diff_weight=2.0)
        if HAS_TORCH:
            import torch
            pred = torch.from_numpy(sine_stereo).unsqueeze(0)
            tgt = torch.from_numpy(noise).unsqueeze(0)
            v1 = loss_1(pred, tgt).item()
            v2 = loss_2(pred, tgt).item()
        else:
            v1 = loss_1(sine_stereo, noise)
            v2 = loss_2(sine_stereo, noise)
        # Double weights → double loss
        assert v2 == pytest.approx(2.0 * v1, rel=1e-5)


# ---------------------------------------------------------------------------
# MixingLoss (combined)
# ---------------------------------------------------------------------------

class TestMixingLoss:

    def test_returns_tuple(self, sine_stereo):
        loss_fn = MixingLoss(fft_sizes=(512,))
        if HAS_TORCH:
            import torch
            pred = torch.from_numpy(sine_stereo).unsqueeze(0)
            total, d = loss_fn(pred, pred, is_stereo=True)
            assert isinstance(d, dict)
            assert "spectral" in d and "stereo" in d and "total" in d
        else:
            total, d = loss_fn(sine_stereo, sine_stereo, is_stereo=True)
            assert isinstance(d, dict)
            assert "spectral" in d and "stereo" in d and "total" in d

    def test_zero_loss_identical(self, sine_stereo):
        loss_fn = MixingLoss(fft_sizes=(512,))
        if HAS_TORCH:
            import torch
            pred = torch.from_numpy(sine_stereo).unsqueeze(0)
            total, d = loss_fn(pred, pred, is_stereo=True)
            assert d["total"] == pytest.approx(0.0, abs=1e-3)
        else:
            total, d = loss_fn(sine_stereo, sine_stereo, is_stereo=True)
            assert d["total"] == pytest.approx(0.0, abs=1e-3)

    def test_stereo_component_absent_when_mono(self, sine_mono):
        loss_fn = MixingLoss(fft_sizes=(512,))
        if HAS_TORCH:
            import torch
            pred = torch.from_numpy(sine_mono).unsqueeze(0)
            _, d = loss_fn(pred, pred, is_stereo=False)
            assert d["stereo"] == 0.0
        else:
            _, d = loss_fn(sine_mono, sine_mono, is_stereo=False)
            assert d["stereo"] == 0.0

    def test_positive_loss_different_signals(self, sine_stereo):
        rng = np.random.default_rng(12)
        noise = rng.standard_normal(sine_stereo.shape).astype(np.float32)
        loss_fn = MixingLoss(fft_sizes=(512,))
        if HAS_TORCH:
            import torch
            pred = torch.from_numpy(sine_stereo).unsqueeze(0)
            tgt = torch.from_numpy(noise).unsqueeze(0)
            total, d = loss_fn(pred, tgt, is_stereo=True)
            assert d["total"] > 0
        else:
            total, d = loss_fn(sine_stereo, noise, is_stereo=True)
            assert d["total"] > 0
