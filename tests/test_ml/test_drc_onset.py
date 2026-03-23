"""Tests for ml.drc_onset -- DRC onset detection for adaptive compressor timing."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))


class TestOnsetEvent:
    """Tests for the OnsetEvent dataclass."""

    def test_onset_event_creation(self):
        from ml.drc_onset import OnsetEvent
        event = OnsetEvent(
            time_sec=1.5, strength=0.8, type='transient',
            recommended_attack_ms=2.0, recommended_release_ms=100.0,
        )
        assert event.time_sec == 1.5
        assert event.strength == 0.8
        assert event.type == 'transient'
        assert event.recommended_attack_ms == 2.0
        assert event.recommended_release_ms == 100.0

    def test_onset_event_types(self):
        from ml.drc_onset import OnsetEvent
        for onset_type in ['transient', 'gradual', 'sustained']:
            event = OnsetEvent(0.0, 0.5, onset_type, 5.0, 100.0)
            assert event.type == onset_type

    def test_onset_event_fields_accessible(self):
        from ml.drc_onset import OnsetEvent
        event = OnsetEvent(2.0, 0.3, 'gradual', 15.0, 300.0)
        assert isinstance(event.time_sec, float)
        assert isinstance(event.strength, float)
        assert isinstance(event.type, str)


class TestDRCOnsetDetector:
    """Tests for the DRCOnsetDetector."""

    def test_instantiation(self):
        from ml.drc_onset import DRCOnsetDetector
        detector = DRCOnsetDetector(sample_rate=48000, block_size=1024)
        assert detector.sample_rate == 48000
        assert detector.block_size == 1024

    def test_process_silent_returns_none(self):
        from ml.drc_onset import DRCOnsetDetector
        detector = DRCOnsetDetector(sample_rate=48000, block_size=512)
        silence = np.zeros(512)
        result = detector.process(silence)
        assert result is None

    def test_process_detects_transient(self):
        from ml.drc_onset import DRCOnsetDetector, OnsetEvent
        detector = DRCOnsetDetector(sample_rate=48000, block_size=256)
        # First process a quiet block to establish baseline
        quiet = np.random.randn(256) * 0.0001
        detector.process(quiet)
        # Then process a loud block to trigger onset
        loud = np.random.randn(256) * 1.0
        result = detector.process(loud)
        assert result is not None
        assert isinstance(result, OnsetEvent)
        assert result.strength > 0.0

    def test_get_adaptive_params_default(self):
        from ml.drc_onset import DRCOnsetDetector
        detector = DRCOnsetDetector()
        params = detector.get_adaptive_params()
        assert 'attack_ms' in params
        assert 'release_ms' in params
        assert 'type' in params
        assert params['type'] == 'default'

    def test_get_adaptive_params_after_onsets(self):
        from ml.drc_onset import DRCOnsetDetector
        detector = DRCOnsetDetector(sample_rate=48000, block_size=256)
        # Generate onset by going from quiet to loud
        quiet = np.random.randn(256) * 0.0001
        loud = np.random.randn(256) * 1.0
        detector.process(quiet)
        result = detector.process(loud)
        if result is not None:
            params = detector.get_adaptive_params()
            assert params['type'] in ('transient', 'gradual', 'sustained')

    def test_get_density_initial(self):
        from ml.drc_onset import DRCOnsetDetector
        detector = DRCOnsetDetector()
        assert detector.get_density() == 0.0

    def test_reset_clears_state(self):
        from ml.drc_onset import DRCOnsetDetector
        detector = DRCOnsetDetector(sample_rate=48000, block_size=256)
        quiet = np.random.randn(256) * 0.0001
        loud = np.random.randn(256) * 1.0
        detector.process(quiet)
        detector.process(loud)
        detector.reset()
        assert detector.get_density() == 0.0
        params = detector.get_adaptive_params()
        assert params['type'] == 'default'

    def test_time_advances(self):
        from ml.drc_onset import DRCOnsetDetector
        detector = DRCOnsetDetector(sample_rate=48000, block_size=1024)
        block = np.random.randn(1024) * 0.01
        detector.process(block)
        expected_time = 1024.0 / 48000.0
        assert abs(detector._time - expected_time) < 1e-6
