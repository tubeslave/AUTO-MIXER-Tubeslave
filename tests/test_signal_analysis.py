"""Tests for signal_analysis module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pytest


class TestRatioConversion:
    def test_ratio_float_to_wing_basic(self):
        from signal_analysis import ratio_float_to_wing
        assert ratio_float_to_wing(1.0) == "1:1"
        assert ratio_float_to_wing(2.0) == "2:1"
        assert ratio_float_to_wing(4.0) == "4:1"

    def test_ratio_negative_returns_1(self):
        from signal_analysis import ratio_float_to_wing
        assert ratio_float_to_wing(-1.0) == "1:1"
        assert ratio_float_to_wing(0) == "1:1"

    def test_ratio_infinity_clamps_to_max_supported_wing_ratio(self):
        from signal_analysis import ratio_float_to_wing
        assert ratio_float_to_wing(float('inf')) == "20:1"

    def test_ratio_nearest(self):
        from signal_analysis import ratio_float_to_wing
        assert ratio_float_to_wing(2.6) == "2.5:1"

    def test_wing_ratio_table_matches_documented_protocol_order(self):
        from signal_analysis import WING_RATIO_VALUES, WING_RATIO_STRINGS
        assert WING_RATIO_VALUES == [
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            4.0,
            5.0,
            7.0,
            10.0,
            20.0,
        ]
        assert WING_RATIO_STRINGS == [
            "1:1",
            "1.5:1",
            "2:1",
            "2.5:1",
            "3:1",
            "4:1",
            "5:1",
            "7:1",
            "10:1",
            "20:1",
        ]


class TestSpectralAnalyzer:
    def test_analyze_sine(self):
        from signal_analysis import SpectralAnalyzerCompressor
        sr = 48000
        analyzer = SpectralAnalyzerCompressor(sample_rate=sr, fft_size=4096)
        t = np.linspace(0, 0.1, int(sr * 0.1), dtype=np.float32)
        sine = np.sin(2 * np.pi * 1000 * t)
        result = analyzer.analyze(sine)
        assert 'centroid' in result
        assert 'rolloff' in result
        assert 'band_energy' in result
        assert 'spectral_flux' in result
        assert result['centroid'] > 0

    def test_analyze_silence(self):
        from signal_analysis import SpectralAnalyzerCompressor
        analyzer = SpectralAnalyzerCompressor()
        silence = np.zeros(4096, dtype=np.float32)
        result = analyzer.analyze(silence)
        assert result['spectral_flux'] == 0.0

    def test_spectral_flux_changes(self):
        from signal_analysis import SpectralAnalyzerCompressor
        sr = 48000
        analyzer = SpectralAnalyzerCompressor(sample_rate=sr, fft_size=2048)
        t = np.linspace(0, 0.1, 2048, dtype=np.float32)
        # First frame
        analyzer.analyze(np.sin(2 * np.pi * 500 * t))
        # Second frame with different frequency should produce non-zero flux
        result = analyzer.analyze(np.sin(2 * np.pi * 4000 * t))
        assert result['spectral_flux'] > 0

    def test_reset(self):
        from signal_analysis import SpectralAnalyzerCompressor
        analyzer = SpectralAnalyzerCompressor()
        analyzer.analyze(np.random.randn(4096).astype(np.float32))
        analyzer.reset()
        assert analyzer._prev_spectrum is None
