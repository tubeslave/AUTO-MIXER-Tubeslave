"""Tests for ml.channel_classifier -- instrument classification from audio."""
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


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed (channel_classifier imports torch at module level)")
class TestInstrumentClasses:
    """Tests for the INSTRUMENT_CLASSES constant list."""

    def test_instrument_classes_not_empty(self):
        from ml.channel_classifier import INSTRUMENT_CLASSES
        assert len(INSTRUMENT_CLASSES) > 0

    def test_instrument_classes_contains_key_instruments(self):
        from ml.channel_classifier import INSTRUMENT_CLASSES
        expected = ['kick', 'snare', 'lead_vocal', 'bass_guitar', 'unknown']
        for inst in expected:
            assert inst in INSTRUMENT_CLASSES

    def test_num_classes_matches(self):
        from ml.channel_classifier import INSTRUMENT_CLASSES, NUM_CLASSES
        assert NUM_CLASSES == len(INSTRUMENT_CLASSES)

    def test_instrument_classes_are_unique(self):
        from ml.channel_classifier import INSTRUMENT_CLASSES
        assert len(INSTRUMENT_CLASSES) == len(set(INSTRUMENT_CLASSES))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestChannelClassifierNet:
    """Tests for the ChannelClassifierNet CNN."""

    def test_instantiation_default(self):
        from ml.channel_classifier import ChannelClassifierNet, NUM_CLASSES
        net = ChannelClassifierNet()
        assert net.fc2.out_features == NUM_CLASSES

    def test_instantiation_custom_classes(self):
        from ml.channel_classifier import ChannelClassifierNet
        net = ChannelClassifierNet(n_mels=32, n_classes=10)
        assert net.fc2.out_features == 10

    def test_forward_output_shape(self):
        from ml.channel_classifier import ChannelClassifierNet, NUM_CLASSES
        net = ChannelClassifierNet(n_mels=64)
        net.eval()
        # Input: (batch, n_mels, time_frames)
        x = torch.randn(2, 64, 32)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_forward_4d_input(self):
        from ml.channel_classifier import ChannelClassifierNet, NUM_CLASSES
        net = ChannelClassifierNet(n_mels=64)
        net.eval()
        # 4D input: (batch, 1, n_mels, time_frames)
        x = torch.randn(1, 1, 64, 32)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (1, NUM_CLASSES)

    def test_output_is_logits(self):
        from ml.channel_classifier import ChannelClassifierNet
        net = ChannelClassifierNet()
        net.eval()
        x = torch.randn(1, 64, 32)
        with torch.no_grad():
            out = net(x)
        # Logits can be positive or negative (not bounded 0-1)
        assert out.min().item() != out.max().item(), "Output should not be constant"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestChannelClassifier:
    """Tests for the high-level ChannelClassifier interface."""

    def test_instantiation(self):
        from ml.channel_classifier import ChannelClassifier
        clf = ChannelClassifier(sample_rate=48000)
        assert clf.sample_rate == 48000
        assert clf.n_mels == 64

    def test_classify_returns_dict(self):
        from ml.channel_classifier import ChannelClassifier, INSTRUMENT_CLASSES
        clf = ChannelClassifier()
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = clf.classify(audio)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(INSTRUMENT_CLASSES)

    def test_classify_probabilities_sum_to_one(self):
        from ml.channel_classifier import ChannelClassifier
        clf = ChannelClassifier()
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = clf.classify(audio)
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-3

    def test_classify_top_k(self):
        from ml.channel_classifier import ChannelClassifier
        clf = ChannelClassifier()
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        top_3 = clf.classify_top_k(audio, k=3)
        assert len(top_3) == 3
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in top_3)
        # Top-k should be in descending order
        assert top_3[0][1] >= top_3[1][1] >= top_3[2][1]

    def test_audio_to_mel_returns_tensor(self):
        from ml.channel_classifier import ChannelClassifier
        clf = ChannelClassifier()
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        mel = clf.audio_to_mel(audio)
        assert isinstance(mel, torch.Tensor)
        assert mel.dim() >= 2
