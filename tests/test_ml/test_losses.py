"""Tests for ml.losses -- STFT loss, multi-resolution STFT, loudness, and mix consistency."""
import pytest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestSTFTLoss:
    """Tests for the single-resolution STFTLoss module."""

    def test_instantiation_default_params(self):
        from ml.losses import STFTLoss
        loss_fn = STFTLoss()
        assert loss_fn.fft_size == 1024
        assert loss_fn.hop_size == 256
        assert loss_fn.win_size == 1024

    def test_instantiation_custom_params(self):
        from ml.losses import STFTLoss
        loss_fn = STFTLoss(fft_size=512, hop_size=128, win_size=512)
        assert loss_fn.fft_size == 512
        assert loss_fn.hop_size == 128
        assert loss_fn.win_size == 512

    def test_forward_returns_scalar(self):
        from ml.losses import STFTLoss
        loss_fn = STFTLoss(fft_size=256, hop_size=64, win_size=256)
        predicted = torch.randn(1, 2048)
        target = torch.randn(1, 2048)
        result = loss_fn(predicted, target)
        assert result.dim() == 0, "Loss should be a scalar"

    def test_identical_signals_low_loss(self):
        from ml.losses import STFTLoss
        loss_fn = STFTLoss(fft_size=256, hop_size=64, win_size=256)
        signal = torch.randn(1, 2048)
        result = loss_fn(signal, signal)
        assert result.item() < 0.1, "Identical signals should yield near-zero loss"

    def test_different_signals_higher_loss(self):
        from ml.losses import STFTLoss
        loss_fn = STFTLoss(fft_size=256, hop_size=64, win_size=256)
        signal = torch.randn(1, 2048)
        noise = torch.randn(1, 2048)
        loss_same = loss_fn(signal, signal)
        loss_diff = loss_fn(signal, noise)
        assert loss_diff.item() > loss_same.item()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestMultiResolutionSTFTLoss:
    """Tests for MultiResolutionSTFTLoss combining multiple FFT sizes."""

    def test_instantiation_defaults(self):
        from ml.losses import MultiResolutionSTFTLoss
        loss_fn = MultiResolutionSTFTLoss()
        assert len(loss_fn.losses) == 3

    def test_instantiation_custom(self):
        from ml.losses import MultiResolutionSTFTLoss
        loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=(256, 512),
            hop_sizes=(64, 128),
            win_sizes=(256, 512),
        )
        assert len(loss_fn.losses) == 2

    def test_forward_returns_scalar(self):
        from ml.losses import MultiResolutionSTFTLoss
        loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=(256, 512),
            hop_sizes=(64, 128),
            win_sizes=(256, 512),
        )
        predicted = torch.randn(1, 4096)
        target = torch.randn(1, 4096)
        result = loss_fn(predicted, target)
        assert result.dim() == 0

    def test_identical_signals_low_loss(self):
        from ml.losses import MultiResolutionSTFTLoss
        loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=(256, 512),
            hop_sizes=(64, 128),
            win_sizes=(256, 512),
        )
        signal = torch.randn(1, 4096)
        result = loss_fn(signal, signal)
        assert result.item() < 0.1


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestLoudnessLoss:
    """Tests for perceptual loudness loss with A-weighting."""

    def test_instantiation(self):
        from ml.losses import LoudnessLoss
        loss_fn = LoudnessLoss(sample_rate=48000, block_size=1024)
        assert loss_fn.sample_rate == 48000
        assert loss_fn.block_size == 1024

    def test_forward_returns_scalar(self):
        from ml.losses import LoudnessLoss
        loss_fn = LoudnessLoss(sample_rate=48000, block_size=512)
        # Audio length must be divisible by block_size
        predicted = torch.randn(2048)
        target = torch.randn(2048)
        result = loss_fn(predicted, target)
        assert result.dim() == 0

    def test_identical_signals_low_loss(self):
        from ml.losses import LoudnessLoss
        loss_fn = LoudnessLoss(sample_rate=48000, block_size=512)
        signal = torch.randn(2048)
        result = loss_fn(signal, signal)
        assert result.item() < 1e-5

    def test_a_weight_shape(self):
        from ml.losses import LoudnessLoss
        loss_fn = LoudnessLoss()
        freqs = torch.linspace(20, 20000, 100)
        weights = loss_fn.a_weight(freqs)
        assert weights.shape == freqs.shape


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestMixConsistencyLoss:
    """Tests for mix consistency loss ensuring sum correctness."""

    def test_perfect_consistency_zero_loss(self):
        from ml.losses import MixConsistencyLoss
        loss_fn = MixConsistencyLoss()
        ch1 = torch.randn(1, 1024)
        ch2 = torch.randn(1, 1024)
        mix = ch1 + ch2
        result = loss_fn([ch1, ch2], mix)
        assert result.item() < 1e-6

    def test_inconsistent_mix_nonzero_loss(self):
        from ml.losses import MixConsistencyLoss
        loss_fn = MixConsistencyLoss()
        ch1 = torch.randn(1, 1024)
        ch2 = torch.randn(1, 1024)
        bad_mix = torch.randn(1, 1024)
        result = loss_fn([ch1, ch2], bad_mix)
        assert result.item() > 0

    def test_single_channel(self):
        from ml.losses import MixConsistencyLoss
        loss_fn = MixConsistencyLoss()
        ch = torch.randn(1, 1024)
        result = loss_fn([ch], ch)
        assert result.item() < 1e-6
