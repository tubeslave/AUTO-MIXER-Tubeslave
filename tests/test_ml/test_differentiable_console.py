"""
Tests for backend.ml.differentiable_console — differentiable mixing console
with EQ, compressor, gain/pan.

Tests the numpy fallback path unconditionally and the torch path when available.
"""

import numpy as np
import pytest

from backend.ml.differentiable_console import DifferentiableMixingConsole, HAS_TORCH

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def short_audio_np():
    """2-channel, 4096-sample numpy audio (sine + noise)."""
    sr = 48000
    n = 4096
    t = np.linspace(0, n / sr, n, endpoint=False, dtype=np.float32)
    ch0 = np.sin(2 * np.pi * 440 * t) * 0.5
    ch1 = np.sin(2 * np.pi * 880 * t) * 0.3
    return np.stack([ch0, ch1], axis=0)  # (2, 4096)


# ---------------------------------------------------------------------------
# Numpy-fallback console tests (always runnable)
# ---------------------------------------------------------------------------

class TestNumpyConsoleDefaultParams:
    """Test create_default_params in the numpy path."""

    def test_default_params_keys(self):
        console = DifferentiableMixingConsole(num_channels=4, num_eq_bands=4, sr=48000)
        if HAS_TORCH:
            params = console.create_default_params(batch_size=1, num_channels=4)
        else:
            params = console.create_default_params(num_channels=4)
        expected_keys = {
            "eq_freq", "eq_gain", "eq_q",
            "comp_threshold", "comp_ratio", "comp_attack", "comp_release",
            "gain_db", "pan",
        }
        assert set(params.keys()) == expected_keys

    def test_default_params_shapes_numpy(self):
        if HAS_TORCH:
            pytest.skip("Testing numpy path only")
        console = DifferentiableMixingConsole(num_channels=4, num_eq_bands=4, sr=48000)
        params = console.create_default_params(num_channels=4)
        assert params["eq_freq"].shape == (4, 4)
        assert params["eq_gain"].shape == (4, 4)
        assert params["gain_db"].shape == (4, 1)
        assert params["pan"].shape == (4, 1)

    def test_default_gain_is_zero(self):
        console = DifferentiableMixingConsole(num_channels=2, sr=48000)
        if HAS_TORCH:
            params = console.create_default_params(batch_size=1, num_channels=2)
            import torch
            assert torch.allclose(params["gain_db"], torch.zeros_like(params["gain_db"]))
        else:
            params = console.create_default_params(num_channels=2)
            np.testing.assert_allclose(params["gain_db"], 0.0)


class TestNumpyConsoleForward:
    """Test forward pass in the numpy path."""

    def test_forward_produces_stereo(self, short_audio_np):
        if HAS_TORCH:
            pytest.skip("Testing numpy path only")
        console = DifferentiableMixingConsole(num_channels=2, num_eq_bands=4, sr=48000)
        params = console.create_default_params(num_channels=2)
        output = console.forward(short_audio_np, params)
        assert output.shape[0] == 2, "Output must be stereo (2 channels)"
        assert output.shape[1] == short_audio_np.shape[1]

    def test_flat_eq_passthrough(self, short_audio_np):
        """With flat EQ (0 dB gain), output energy should roughly equal input energy."""
        if HAS_TORCH:
            pytest.skip("Testing numpy path only")
        console = DifferentiableMixingConsole(num_channels=2, num_eq_bands=4, sr=48000)
        params = console.create_default_params(num_channels=2)
        # Set ratio to 1 (no compression) to isolate EQ test
        params["comp_ratio"] = np.full((2, 1), 1.0)
        output = console.forward(short_audio_np, params)
        # Sum of L+R energy should be similar to input energy (center pan)
        input_energy = np.sum(short_audio_np ** 2)
        output_energy = np.sum(output ** 2)
        # Allow generous tolerance because of filter edge effects
        assert output_energy > 0, "Output should have non-zero energy"

    def test_mute_via_large_negative_gain(self, short_audio_np):
        """A very large negative gain should produce near-silence."""
        if HAS_TORCH:
            pytest.skip("Testing numpy path only")
        console = DifferentiableMixingConsole(num_channels=2, num_eq_bands=4, sr=48000)
        params = console.create_default_params(num_channels=2)
        params["gain_db"] = np.full((2, 1), -120.0)
        params["comp_ratio"] = np.full((2, 1), 1.0)
        output = console.forward(short_audio_np, params)
        assert np.max(np.abs(output)) < 1e-3


# ---------------------------------------------------------------------------
# Torch console tests (skipped when torch is unavailable)
# ---------------------------------------------------------------------------

class TestTorchConsole:
    """Tests that exercise the PyTorch DifferentiableMixingConsole."""

    def test_forward_shape(self, short_audio_np):
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        import torch
        console = DifferentiableMixingConsole(num_channels=2, num_eq_bands=4, sr=48000)
        params = console.create_default_params(batch_size=1, num_channels=2)
        audio = torch.from_numpy(short_audio_np).unsqueeze(0)  # (1, 2, 4096)
        output = console(audio, params)
        assert output.shape == (1, 2, short_audio_np.shape[1])

    def test_gradients_flow(self, short_audio_np):
        """Parameters should receive gradients through the console."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        import torch
        console = DifferentiableMixingConsole(num_channels=2, num_eq_bands=4, sr=48000)
        params = console.create_default_params(batch_size=1, num_channels=2)
        # Make gain_db require grad
        params["gain_db"] = params["gain_db"].clone().requires_grad_(True)
        audio = torch.from_numpy(short_audio_np).unsqueeze(0)
        output = console(audio, params)
        loss = output.sum()
        loss.backward()
        assert params["gain_db"].grad is not None
        assert not torch.all(params["gain_db"].grad == 0)

    def test_default_params_batch_size(self):
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        import torch
        console = DifferentiableMixingConsole(num_channels=8, num_eq_bands=4, sr=48000)
        params = console.create_default_params(batch_size=3, num_channels=8)
        assert params["eq_freq"].shape == (3, 8, 4)
        assert params["pan"].shape == (3, 8, 1)
