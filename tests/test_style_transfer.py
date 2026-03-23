"""Tests for neural_mix_extractor, style_transfer, and processing_graph modules."""
import pytest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from neural_mix_extractor import NeuralMixExtractor, MixStyle
from style_transfer import MixStyleTransfer, StyleTransferResult
from processing_graph import ProcessingGraph, ProcessingNode, NodeType, Connection


class TestNeuralMixExtractor:
    """Tests for the NeuralMixExtractor class."""

    def _make_test_audio(self, freq=440.0, duration=0.5, sr=48000):
        """Generate a mono sine wave for testing."""
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        return np.sin(2 * np.pi * freq * t) * 0.3

    def test_extract_mono_audio(self):
        """extract() returns a MixStyle with populated features from mono audio."""
        extractor = NeuralMixExtractor(sample_rate=48000, fft_size=2048)
        audio = self._make_test_audio(freq=1000.0, duration=1.0)
        style = extractor.extract(audio, name='test_mono')

        assert isinstance(style, MixStyle)
        assert style.name == 'test_mono'
        assert len(style.spectral_envelope) > 0
        assert len(style.frequencies) > 0
        assert style.spectral_centroid > 0
        assert style.spectral_rolloff > 0
        assert 0.0 <= style.spectral_flatness <= 1.0
        assert len(style.band_levels) == 7  # sub, bass, low_mid, mid, high_mid, high, air

    def test_extract_stereo_audio(self):
        """extract() handles 2-channel stereo audio correctly."""
        extractor = NeuralMixExtractor(sample_rate=48000, fft_size=2048)
        sr = 48000
        t = np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)
        left = np.sin(2 * np.pi * 440 * t) * 0.3
        right = np.sin(2 * np.pi * 440 * t + 0.5) * 0.3  # slight phase offset
        stereo = np.array([left, right])  # shape (2, N)

        style = extractor.extract(stereo, name='test_stereo')
        assert isinstance(style, MixStyle)
        assert style.stereo_width >= 0.0
        assert -1.0 <= style.stereo_correlation <= 1.0

    def test_compute_distance_same_style_is_zero(self):
        """Distance between a style and itself should be zero or near-zero."""
        extractor = NeuralMixExtractor(sample_rate=48000, fft_size=2048)
        audio = self._make_test_audio(freq=1000.0, duration=1.0)
        style = extractor.extract(audio, name='self_compare')
        distance = extractor.compute_distance(style, style)
        assert distance == pytest.approx(0.0, abs=0.01)

    def test_compute_distance_different_styles(self):
        """Distance between two different audio signals should be positive."""
        extractor = NeuralMixExtractor(sample_rate=48000, fft_size=2048)
        audio_low = self._make_test_audio(freq=200.0, duration=1.0)
        audio_high = self._make_test_audio(freq=8000.0, duration=1.0)
        style_low = extractor.extract(audio_low, name='low')
        style_high = extractor.extract(audio_high, name='high')
        distance = extractor.compute_distance(style_low, style_high)
        assert distance > 0.0


class TestMixStyleTransfer:
    """Tests for the MixStyleTransfer class."""

    def _make_style(self, loudness=-18.0, dynamic_range=12.0, band_boost=0.0):
        """Create a MixStyle with specified parameters for testing."""
        band_names = ['sub', 'bass', 'low_mid', 'mid', 'high_mid', 'high', 'air']
        return MixStyle(
            name='test',
            spectral_envelope=np.zeros(100),
            frequencies=np.linspace(0, 24000, 100),
            loudness_lufs=loudness,
            loudness_range_lu=8.0,
            dynamic_range_db=dynamic_range,
            crest_factor_db=8.0,
            spectral_centroid=2000.0,
            spectral_rolloff=8000.0,
            spectral_flatness=0.3,
            band_levels={b: -20.0 + band_boost for b in band_names},
            band_dynamics={b: 3.0 for b in band_names},
            stereo_width=0.5,
            stereo_correlation=0.8,
            avg_attack_ms=10.0,
            avg_release_ms=100.0,
        )

    def test_compute_transfer_gain_adjustment(self):
        """compute_transfer computes a gain adjustment based on loudness difference."""
        transfer = MixStyleTransfer(sample_rate=48000)
        current = self._make_style(loudness=-20.0)
        target = self._make_style(loudness=-14.0)
        result = transfer.compute_transfer(current, target)

        assert isinstance(result, StyleTransferResult)
        assert result.gain_adjustment_db == pytest.approx(6.0, abs=0.1)

    def test_compute_transfer_eq_corrections(self):
        """compute_transfer generates EQ corrections when band levels differ."""
        transfer = MixStyleTransfer(sample_rate=48000)
        current = self._make_style(loudness=-18.0, band_boost=0.0)
        target = self._make_style(loudness=-18.0, band_boost=5.0)
        result = transfer.compute_transfer(current, target)

        assert len(result.eq_corrections) > 0
        for eq in result.eq_corrections:
            assert 'frequency' in eq
            assert 'gain_db' in eq
            assert 'q' in eq
            assert 'type' in eq
            assert eq['gain_db'] > 0  # target is louder in all bands

    def test_apply_to_channels_generates_per_channel_adjustments(self):
        """apply_to_channels produces adjustments for each channel-instrument pair."""
        transfer = MixStyleTransfer(sample_rate=48000)
        current = self._make_style(loudness=-20.0)
        target = self._make_style(loudness=-14.0)
        result = transfer.compute_transfer(current, target)

        channels = {1: 'lead_vocal', 2: 'kick', 3: 'electric_guitar'}
        adjustments = transfer.apply_to_channels(result, channels)

        assert set(adjustments.keys()) == {1, 2, 3}
        for ch_id, adj in adjustments.items():
            assert 'gain_db' in adj
            assert 'eq_bands' in adj
            assert 'comp_threshold' in adj
            assert 'comp_ratio' in adj


class TestProcessingGraph:
    """Tests for the ProcessingGraph class."""

    def test_add_nodes_and_connect(self):
        """Nodes can be added and connected in the graph."""
        graph = ProcessingGraph()
        input_node = ProcessingNode('in', NodeType.INPUT)
        gain_node = ProcessingNode('gain', NodeType.GAIN, {'gain_db': -6.0})
        output_node = ProcessingNode('out', NodeType.OUTPUT)

        graph.add_node(input_node)
        graph.add_node(gain_node)
        graph.add_node(output_node)
        graph.connect('in', 'gain')
        graph.connect('gain', 'out')

        assert len(graph.nodes) == 3
        assert len(graph.connections) == 2

    def test_process_gain_node(self):
        """Processing through a gain node applies the expected gain."""
        graph = ProcessingGraph()
        graph.add_node(ProcessingNode('in', NodeType.INPUT))
        graph.add_node(ProcessingNode('gain', NodeType.GAIN, {'gain_db': -6.0}))
        graph.add_node(ProcessingNode('out', NodeType.OUTPUT))
        graph.connect('in', 'gain')
        graph.connect('gain', 'out')

        audio = np.ones(100, dtype=np.float32)
        outputs = graph.process({'in': audio})

        expected_gain = 10 ** (-6.0 / 20.0)
        assert 'out' in outputs
        np.testing.assert_allclose(outputs['out'], expected_gain, atol=1e-6)

    def test_build_channel_strip(self):
        """build_channel_strip creates a full chain with expected node types."""
        graph = ProcessingGraph()
        node_ids = graph.build_channel_strip(1)

        assert len(node_ids) == 8  # input, gate, eq, comp, gain, pan, meter, output
        assert node_ids[0] == 'ch1_input'
        assert node_ids[-1] == 'ch1_output'

        # All nodes should be in the graph
        for nid in node_ids:
            assert graph.get_node(nid) is not None

        # Connections should form a chain
        assert len(graph.connections) == 7

    def test_remove_node_and_connections(self):
        """Removing a node also removes its connections."""
        graph = ProcessingGraph()
        graph.add_node(ProcessingNode('a', NodeType.INPUT))
        graph.add_node(ProcessingNode('b', NodeType.GAIN, {'gain_db': 0.0}))
        graph.add_node(ProcessingNode('c', NodeType.OUTPUT))
        graph.connect('a', 'b')
        graph.connect('b', 'c')

        graph.remove_node('b')
        assert 'b' not in graph.nodes
        assert len(graph.connections) == 0  # both connections involved 'b'

    def test_set_parameter_and_get_graph_info(self):
        """set_parameter updates node parameters; get_graph_info returns topology."""
        graph = ProcessingGraph()
        graph.add_node(ProcessingNode('g', NodeType.GAIN, {'gain_db': 0.0}))
        graph.set_parameter('g', 'gain_db', -12.0)
        assert graph.nodes['g'].parameters['gain_db'] == -12.0

        info = graph.get_graph_info()
        assert 'nodes' in info
        assert 'connections' in info
        assert 'g' in info['nodes']
        assert info['nodes']['g']['type'] == 'gain'
