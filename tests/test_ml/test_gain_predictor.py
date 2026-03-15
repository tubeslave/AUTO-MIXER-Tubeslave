"""
Tests for backend.ml.gain_pan_predictor — Squeeze-and-Excitation neural
network for gain/pan prediction, plus numpy fallback.

Tests both the torch-based and numpy-fallback codepaths.
"""

import os
import tempfile

import numpy as np
import pytest

from backend.ml.gain_pan_predictor import (
    SqueezeExcitation,
    GainPanPredictor,
    HAS_TORCH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sine_audio():
    """1-second 440 Hz mono sine at 48 kHz."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t) * 0.5


@pytest.fixture
def quiet_audio():
    """Very quiet 1-second sine."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t) * 0.001


@pytest.fixture
def loud_audio():
    """Full-scale 1-second sine."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t) * 0.95


@pytest.fixture
def predictor():
    return GainPanPredictor(input_dim=36, gain_range=(-30.0, 12.0))


# ---------------------------------------------------------------------------
# SqueezeExcitation
# ---------------------------------------------------------------------------

class TestSqueezeExcitation:

    def test_instantiation(self):
        se = SqueezeExcitation(channels=64, reduction=4)
        assert se is not None

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_forward_2d(self):
        import torch
        se = SqueezeExcitation(channels=64, reduction=4)
        x = torch.randn(4, 64)
        out = se(x)
        assert out.shape == (4, 64)

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_forward_3d(self):
        import torch
        se = SqueezeExcitation(channels=32, reduction=4)
        x = torch.randn(2, 32, 100)
        out = se(x)
        assert out.shape == (2, 32, 100)

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_output_same_sign_pattern(self):
        """SE should preserve the sign pattern of the input (scaling only)."""
        import torch
        se = SqueezeExcitation(channels=16, reduction=4)
        se.eval()
        x = torch.randn(1, 16)
        out = se(x)
        # Signs should be preserved since SE multiplies by positive sigmoid
        signs_in = torch.sign(x)
        signs_out = torch.sign(out)
        assert torch.all(signs_in == signs_out)


# ---------------------------------------------------------------------------
# GainPanPredictor (torch path)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestGainPanPredictorTorch:

    def test_forward_output_shapes(self, predictor):
        import torch
        x = torch.randn(8, 36)
        predictor.eval()
        gain, pan = predictor(x)
        assert gain.shape == (8, 1)
        assert pan.shape == (8, 1)

    def test_gain_in_range(self, predictor):
        """Output gain should be within the configured range."""
        import torch
        x = torch.randn(4, 36)
        predictor.eval()
        gain, _ = predictor(x)
        assert torch.all(gain >= -30.0)
        assert torch.all(gain <= 12.0)

    def test_pan_in_range(self, predictor):
        """Output pan should be within -1 to 1."""
        import torch
        x = torch.randn(4, 36)
        predictor.eval()
        _, pan = predictor(x)
        assert torch.all(pan >= -1.0)
        assert torch.all(pan <= 1.0)

    def test_predict_from_audio(self, predictor, sine_audio):
        """predict() should return (gain_db, pan) tuple of floats."""
        gain, pan = predictor.predict(sine_audio, sr=48000)
        assert isinstance(gain, float)
        assert isinstance(pan, float)
        assert -30.0 <= gain <= 12.0
        assert -1.0 <= pan <= 1.0

    def test_train_step(self, predictor):
        import torch
        optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)
        batch = {
            "features": torch.randn(4, 36),
            "gain_db": torch.randn(4, 1) * 5,
            "pan": torch.randn(4, 1) * 0.5,
        }
        loss = predictor.train_step(batch, optimizer)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_step_with_custom_loss(self, predictor):
        import torch
        optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)

        def custom_loss(pred_gain, target_gain, pred_pan, target_pan):
            return torch.nn.functional.l1_loss(pred_gain, target_gain) + \
                   torch.nn.functional.l1_loss(pred_pan, target_pan)

        batch = {
            "features": torch.randn(4, 36),
            "gain_db": torch.randn(4, 1) * 5,
            "pan": torch.randn(4, 1) * 0.5,
        }
        loss = predictor.train_step(batch, optimizer, loss_fn=custom_loss)
        assert isinstance(loss, float)

    def test_save_and_load_roundtrip(self, predictor):
        import torch
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            predictor.save_model(path)
            assert os.path.isfile(path)

            loaded = GainPanPredictor(input_dim=36, gain_range=(-30.0, 12.0))
            loaded.load_model(path)

            # Both should produce same output
            x = torch.randn(1, 36)
            predictor.eval()
            loaded.eval()
            g1, p1 = predictor(x)
            g2, p2 = loaded(x)
            assert g1.item() == pytest.approx(g2.item(), abs=1e-5)
            assert p1.item() == pytest.approx(p2.item(), abs=1e-5)

    def test_load_missing_file_raises(self, predictor):
        with pytest.raises(FileNotFoundError):
            predictor.load_model("/nonexistent/model.pt")

    def test_custom_gain_range(self):
        import torch
        model = GainPanPredictor(input_dim=36, gain_range=(-20.0, 6.0))
        model.eval()
        x = torch.randn(2, 36)
        gain, _ = model(x)
        assert torch.all(gain >= -20.0)
        assert torch.all(gain <= 6.0)


# ---------------------------------------------------------------------------
# GainPanPredictor (numpy fallback)
# ---------------------------------------------------------------------------

class TestGainPanPredictorNumpy:
    """Tests that work for both torch and numpy fallback."""

    def test_predict_returns_tuple(self, predictor, sine_audio):
        result = predictor.predict(sine_audio, sr=48000)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predict_gain_type(self, predictor, sine_audio):
        gain, _ = predictor.predict(sine_audio, sr=48000)
        assert isinstance(gain, float)

    def test_predict_pan_type(self, predictor, sine_audio):
        _, pan = predictor.predict(sine_audio, sr=48000)
        assert isinstance(pan, float)

    def test_predict_empty_audio(self):
        """Empty audio should return sensible defaults for the numpy fallback."""
        if HAS_TORCH:
            pytest.skip("Testing numpy fallback only")
        p = GainPanPredictor()
        gain, pan = p.predict(np.array([], dtype=np.float32), sr=48000)
        assert gain == -12.0
        assert pan == 0.0

    def test_forward_numpy(self):
        """Numpy fallback forward pass returns zero-centered arrays."""
        if HAS_TORCH:
            pytest.skip("Testing numpy fallback only")
        p = GainPanPredictor()
        x = np.random.randn(4, 36).astype(np.float32)
        gain, pan = p.forward(x)
        assert gain.shape == (4, 1)
        assert pan.shape == (4, 1)

    def test_train_step_noop(self):
        """Numpy fallback train_step should return 0."""
        if HAS_TORCH:
            pytest.skip("Testing numpy fallback only")
        p = GainPanPredictor()
        loss = p.train_step({})
        assert loss == 0.0

    def test_save_load_numpy(self):
        """Numpy fallback save/load roundtrip."""
        if HAS_TORCH:
            pytest.skip("Testing numpy fallback only")
        p = GainPanPredictor(input_dim=36, gain_range=(-25.0, 8.0))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.npz")
            p.save_model(path)
            assert os.path.isfile(path)

            p2 = GainPanPredictor()
            p2.load_model(path)
            assert p2.gain_min == -25.0
            assert p2.gain_max == 8.0

    def test_load_missing_numpy(self):
        """Numpy fallback should raise FileNotFoundError for missing file."""
        if HAS_TORCH:
            pytest.skip("Testing numpy fallback only")
        p = GainPanPredictor()
        with pytest.raises(FileNotFoundError):
            p.load_model("/nonexistent/model.npz")
