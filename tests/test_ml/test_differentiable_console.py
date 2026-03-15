"""Tests for ml.differentiable_console -- differentiable mixing console modules."""
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
class TestDifferentiableGain:
    """Tests for DifferentiableGain -- dB-parameterized gain stage."""

    def test_instantiation(self):
        from ml.differentiable_console import DifferentiableGain
        gain = DifferentiableGain(n_channels=4, init_db=-6.0)
        assert gain.gain_db.shape == (4,)
        assert torch.allclose(gain.gain_db, torch.full((4,), -6.0))

    def test_forward_shape_preserved(self):
        from ml.differentiable_console import DifferentiableGain
        gain = DifferentiableGain(n_channels=3)
        x = torch.randn(3, 1024)
        out = gain(x)
        assert out.shape == (3, 1024)

    def test_zero_db_unity_gain(self):
        from ml.differentiable_console import DifferentiableGain
        gain = DifferentiableGain(n_channels=2, init_db=0.0)
        x = torch.randn(2, 512)
        out = gain(x)
        assert torch.allclose(out, x, atol=1e-5)

    def test_negative_db_reduces_amplitude(self):
        from ml.differentiable_console import DifferentiableGain
        gain = DifferentiableGain(n_channels=1, init_db=-20.0)
        x = torch.ones(1, 256)
        out = gain(x)
        assert out.abs().max().item() < 1.0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestDifferentiablePan:
    """Tests for DifferentiablePan -- constant-power stereo panner."""

    def test_instantiation(self):
        from ml.differentiable_console import DifferentiablePan
        pan = DifferentiablePan(n_channels=4)
        assert pan.pan.shape == (4,)
        assert torch.allclose(pan.pan, torch.zeros(4))

    def test_center_pan_equal_left_right(self):
        from ml.differentiable_console import DifferentiablePan
        pan = DifferentiablePan(n_channels=1)
        x = torch.ones(1, 512)
        out = pan(x)
        # At center pan (0), left and right should be equal
        assert out.shape[1] == 2
        left_energy = out[0, 0, :].pow(2).sum().item()
        right_energy = out[0, 1, :].pow(2).sum().item()
        assert abs(left_energy - right_energy) < 1e-3

    def test_output_shape_2d_input(self):
        from ml.differentiable_console import DifferentiablePan
        pan = DifferentiablePan(n_channels=2)
        x = torch.randn(2, 256)
        out = pan(x)
        assert out.shape == (2, 2, 256)

    def test_pan_parameter_is_learnable(self):
        from ml.differentiable_console import DifferentiablePan
        pan = DifferentiablePan(n_channels=2)
        assert pan.pan.requires_grad is True


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestDifferentiableEQ:
    """Tests for DifferentiableEQ -- frequency-domain parametric EQ."""

    def test_instantiation(self):
        from ml.differentiable_console import DifferentiableEQ
        eq = DifferentiableEQ(n_channels=2, n_bands=4)
        assert eq.n_channels == 2
        assert eq.n_bands == 4
        assert eq.freq.shape == (2, 4)
        assert eq.gain_db.shape == (2, 4)
        assert eq.q.shape == (2, 4)

    def test_forward_shape_preserved(self):
        from ml.differentiable_console import DifferentiableEQ
        eq = DifferentiableEQ(n_channels=2, n_bands=3)
        x = torch.randn(2, 2048)
        out = eq(x)
        assert out.shape == x.shape

    def test_zero_gain_passthrough(self):
        from ml.differentiable_console import DifferentiableEQ
        eq = DifferentiableEQ(n_channels=1, n_bands=4)
        # Zero gain means no EQ applied; response should be ~1 everywhere
        with torch.no_grad():
            eq.gain_db.fill_(0.0)
        x = torch.randn(1, 2048)
        out = eq(x)
        # With zero gain the EQ transfer function should be unity
        assert torch.allclose(out, x, atol=1e-4)

    def test_compute_biquad_output_shape(self):
        from ml.differentiable_console import DifferentiableEQ
        eq = DifferentiableEQ(n_channels=2, n_bands=3)
        freq = torch.tensor([1000.0])
        gain_db = torch.tensor([6.0])
        q = torch.tensor([1.0])
        coeffs = eq.compute_biquad(freq, gain_db, q)
        assert coeffs.shape[-1] == 5


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestDifferentiableCompressor:
    """Tests for DifferentiableCompressor -- differentiable dynamics compressor."""

    def test_instantiation(self):
        from ml.differentiable_console import DifferentiableCompressor
        comp = DifferentiableCompressor(n_channels=4)
        assert comp.threshold_db.shape == (4,)
        assert comp.ratio.shape == (4,)

    def test_forward_shape_preserved(self):
        from ml.differentiable_console import DifferentiableCompressor
        comp = DifferentiableCompressor(n_channels=2)
        x = torch.randn(2, 1024)
        out = comp(x)
        assert out.shape == x.shape

    def test_quiet_signal_passthrough(self):
        from ml.differentiable_console import DifferentiableCompressor
        comp = DifferentiableCompressor(n_channels=1)
        # Very quiet signal should pass through unaffected (below threshold)
        x = torch.ones(1, 512) * 1e-6
        out = comp(x)
        assert torch.allclose(out, x, atol=1e-8)

    def test_compression_reduces_loud_signal(self):
        from ml.differentiable_console import DifferentiableCompressor
        comp = DifferentiableCompressor(n_channels=1)
        with torch.no_grad():
            comp.threshold_db.fill_(-20.0)
            comp.ratio.fill_(10.0)
        x = torch.ones(1, 512) * 0.9  # loud signal
        out = comp(x)
        # Compressed output should have lower peak
        assert out.abs().max().item() <= x.abs().max().item()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestDifferentiableMixingConsole:
    """Tests for the full DifferentiableMixingConsole chain."""

    def test_instantiation(self):
        from ml.differentiable_console import DifferentiableMixingConsole
        console = DifferentiableMixingConsole(n_channels=4, sample_rate=48000)
        assert console.n_channels == 4

    def test_forward_returns_mix_and_processed(self):
        from ml.differentiable_console import DifferentiableMixingConsole
        console = DifferentiableMixingConsole(n_channels=2, sample_rate=48000)
        channels = [torch.randn(1, 2048), torch.randn(1, 2048)]
        mix, processed = console(channels)
        assert mix.shape == (1, 2048)
        assert len(processed) == 2

    def test_get_parameters_dict_keys(self):
        from ml.differentiable_console import DifferentiableMixingConsole
        console = DifferentiableMixingConsole(n_channels=3)
        params = console.get_parameters_dict()
        expected_keys = {'gain_db', 'pan', 'eq_freq', 'eq_gain', 'eq_q',
                         'threshold', 'ratio'}
        assert set(params.keys()) == expected_keys

    def test_get_parameters_dict_lengths(self):
        from ml.differentiable_console import DifferentiableMixingConsole
        console = DifferentiableMixingConsole(n_channels=3)
        params = console.get_parameters_dict()
        assert len(params['gain_db']) == 3
        assert len(params['pan']) == 3
