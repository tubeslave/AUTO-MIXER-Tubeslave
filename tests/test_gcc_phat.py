"""Tests for auto_phase_gcc_phat module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pytest
from auto_phase_gcc_phat import GCCPHATAnalyzer, DelayMeasurement, AutoPhaseAligner


class TestGCCPHATAnalyzer:
    def test_zero_delay(self):
        sr = 48000
        analyzer = GCCPHATAnalyzer(sample_rate=sr, fft_size=4096)
        signal = np.random.randn(sr).astype(np.float32)
        signal = np.convolve(signal, [0.5, 0.5], mode='same')
        result = analyzer.compute_delay(signal, signal)
        assert abs(result.delay_ms) < 1.0

    def test_known_delay(self):
        sr = 48000
        analyzer = GCCPHATAnalyzer(sample_rate=sr, fft_size=4096, max_delay_ms=50)
        ref = np.random.randn(sr).astype(np.float32)
        ref = np.convolve(ref, [0.5, 0.5], mode='same')
        delay_samples = 48  # 1ms at 48kHz
        tgt = np.zeros_like(ref)
        tgt[delay_samples:] = ref[:-delay_samples]
        result = analyzer.compute_delay(ref, tgt)
        assert abs(result.delay_ms - 1.0) < 0.5

    def test_delay_measurement_valid(self):
        m = DelayMeasurement(1.0, 48.0, 0.8, 10.0, 30.0, 0.9, 0.8)
        assert m.is_valid()

    def test_delay_measurement_invalid(self):
        m = DelayMeasurement(1.0, 48.0, 0.1, 2.0, 5.0, 0.3, 0.2)
        assert not m.is_valid()

    def test_short_signal(self):
        analyzer = GCCPHATAnalyzer(fft_size=4096)
        short = np.random.randn(100).astype(np.float32)
        result = analyzer.compute_delay(short, short)
        assert result.delay_ms == 0
        assert result.confidence == 0

    def test_reset(self):
        analyzer = GCCPHATAnalyzer()
        analyzer.delay_history.append(1.0)
        analyzer.reset()
        assert len(analyzer.delay_history) == 0


class TestAutoPhaseAligner:
    def test_init(self):
        aligner = AutoPhaseAligner()
        assert aligner.reference_channel is None
        assert len(aligner.channels_to_align) == 0

    def test_add_remove_channel(self):
        aligner = AutoPhaseAligner()
        aligner.add_channel(1)
        aligner.add_channel(2)
        assert 1 in aligner.channels_to_align
        assert 2 in aligner.channels_to_align
        aligner.remove_channel(1)
        assert 1 not in aligner.channels_to_align

    def test_set_reference(self):
        aligner = AutoPhaseAligner()
        aligner.set_reference_channel(1)
        assert aligner.reference_channel == 1

    def test_get_status(self):
        aligner = AutoPhaseAligner()
        aligner.set_reference_channel(1)
        aligner.add_channel(2)
        status = aligner.get_status()
        assert status['reference_channel'] == 1
        assert 2 in status['channels']
