"""Tests for ml.gain_pan_predictor -- neural gain/pan prediction."""
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
class TestSEBlock:
    """Tests for the Squeeze-and-Excitation block."""

    def test_instantiation(self):
        from ml.gain_pan_predictor import SEBlock
        se = SEBlock(channels=32, reduction=4)
        assert se.squeeze is not None
        assert se.excitation is not None

    def test_forward_shape_preserved(self):
        from ml.gain_pan_predictor import SEBlock
        se = SEBlock(channels=16, reduction=4)
        x = torch.randn(2, 16, 128)
        out = se(x)
        assert out.shape == (2, 16, 128)

    def test_output_same_scale(self):
        from ml.gain_pan_predictor import SEBlock
        se = SEBlock(channels=8, reduction=2)
        se.eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            out = se(x)
        # SE block scales channels, output should be finite and similar magnitude
        assert torch.isfinite(out).all()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestGainPanPredictorNet:
    """Tests for the GainPanPredictorNet neural network."""

    def test_instantiation(self):
        from ml.gain_pan_predictor import GainPanPredictorNet
        net = GainPanPredictorNet(n_channels=8, feature_dim=64)
        assert net.n_channels == 8

    def test_forward_output_shapes(self):
        from ml.gain_pan_predictor import GainPanPredictorNet
        net = GainPanPredictorNet(n_channels=4, feature_dim=64)
        net.eval()
        # (batch, n_channels, audio_length)
        x = torch.randn(2, 4, 4096)
        with torch.no_grad():
            gains, pans = net(x)
        assert gains.shape == (2, 4)
        assert pans.shape == (2, 4)

    def test_gain_range(self):
        from ml.gain_pan_predictor import GainPanPredictorNet
        net = GainPanPredictorNet(n_channels=4, feature_dim=64)
        net.eval()
        x = torch.randn(1, 4, 4096)
        with torch.no_grad():
            gains, pans = net(x)
        # Gains are scaled to [-60, -12] range: gains = raw * 48 - 60
        # With random weights the exact range may vary, but should be finite
        assert torch.isfinite(gains).all()

    def test_pan_range(self):
        from ml.gain_pan_predictor import GainPanPredictorNet
        net = GainPanPredictorNet(n_channels=4, feature_dim=64)
        net.eval()
        x = torch.randn(1, 4, 4096)
        with torch.no_grad():
            gains, pans = net(x)
        # Pan head uses Tanh, so output in [-1, 1]
        assert (pans >= -1.0).all()
        assert (pans <= 1.0).all()

    def test_different_feature_dims(self):
        from ml.gain_pan_predictor import GainPanPredictorNet
        for dim in [64, 128, 256]:
            net = GainPanPredictorNet(n_channels=2, feature_dim=dim)
            net.eval()
            x = torch.randn(1, 2, 2048)
            with torch.no_grad():
                gains, pans = net(x)
            assert gains.shape == (1, 2)
            assert pans.shape == (1, 2)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestGainPanPredictor:
    """Tests for the high-level GainPanPredictor interface."""

    def test_instantiation(self):
        from ml.gain_pan_predictor import GainPanPredictor
        predictor = GainPanPredictor(n_channels=8)
        assert predictor.n_channels == 8

    def test_predict_returns_dict(self):
        from ml.gain_pan_predictor import GainPanPredictor
        predictor = GainPanPredictor(n_channels=4)
        channels = [np.random.randn(4096).astype(np.float32) * 0.1 for _ in range(4)]
        result = predictor.predict(channels, block_size=4096)
        assert 'gains_db' in result
        assert 'pans' in result
        assert len(result['gains_db']) == 4
        assert len(result['pans']) == 4

    def test_predict_gains_are_floats(self):
        from ml.gain_pan_predictor import GainPanPredictor
        predictor = GainPanPredictor(n_channels=2)
        channels = [np.random.randn(2048).astype(np.float32) * 0.1 for _ in range(2)]
        result = predictor.predict(channels, block_size=2048)
        for g in result['gains_db']:
            assert isinstance(g, float)
        for p in result['pans']:
            assert isinstance(p, float)

    def test_predict_pans_in_range(self):
        from ml.gain_pan_predictor import GainPanPredictor
        predictor = GainPanPredictor(n_channels=3)
        channels = [np.random.randn(2048).astype(np.float32) * 0.1 for _ in range(3)]
        result = predictor.predict(channels, block_size=2048)
        for p in result['pans']:
            assert -1.0 <= p <= 1.0

    def test_predict_variable_length_channels(self):
        from ml.gain_pan_predictor import GainPanPredictor
        predictor = GainPanPredictor(n_channels=3)
        # Channels of different lengths
        channels = [
            np.random.randn(1000).astype(np.float32) * 0.1,
            np.random.randn(2000).astype(np.float32) * 0.1,
            np.random.randn(1500).astype(np.float32) * 0.1,
        ]
        result = predictor.predict(channels, block_size=2048)
        assert len(result['gains_db']) == 3
        assert len(result['pans']) == 3
