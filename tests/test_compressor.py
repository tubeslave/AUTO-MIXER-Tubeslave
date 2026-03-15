"""
Tests for auto compressor preset loading and signal feature extraction.

Covers:
- Loading compressor presets per instrument
- Signal feature extraction (RMS, peak, crest factor)
- Wing ratio float-to-string conversion
"""

import numpy as np
import pytest

from compressor_presets import DEFAULT_PRESETS, get_preset, get_available_tasks
from compressor_adaptation import ratio_float_to_wing, WING_RATIO_VALUES, WING_RATIO_STRINGS


class TestCompressorPresetLoading:
    """Tests for compressor preset retrieval."""

    def test_compressor_preset_loading(self):
        """Known instrument presets should load without errors."""
        instruments = ["kick", "snare", "bass", "leadVocal"]
        for inst in instruments:
            tasks = get_available_tasks(inst)
            assert len(tasks) > 0, f"No tasks found for instrument: {inst}"
            preset = get_preset(inst, tasks[0])
            assert preset is not None, f"Preset is None for {inst}/{tasks[0]}"
            assert "threshold" in preset or "ratio" in preset, (
                f"Preset for {inst} missing key parameters"
            )

    def test_default_presets_dict_structure(self):
        """DEFAULT_PRESETS should be a nested dict of instrument -> task -> params."""
        assert isinstance(DEFAULT_PRESETS, dict)
        assert len(DEFAULT_PRESETS) > 0
        for inst, tasks in DEFAULT_PRESETS.items():
            assert isinstance(tasks, dict), f"Tasks for {inst} should be a dict"
            for task_name, params in tasks.items():
                assert isinstance(params, dict), (
                    f"Params for {inst}/{task_name} should be a dict"
                )

    def test_unknown_instrument_returns_fallback(self):
        """get_preset for an unknown instrument should return a fallback preset."""
        result = get_preset("nonexistent_instrument_xyz", "base")
        assert isinstance(result, dict) and len(result) > 0, (
            "Unknown instrument should return a fallback preset dict"
        )


class TestSignalFeatureExtraction:
    """Tests for signal feature analysis used by compressor adaptation."""

    def test_signal_features_extraction(self, sample_rate):
        """SignalFeatureExtractor should compute basic features from audio."""
        try:
            from signal_analysis import SignalFeatureExtractor
        except ImportError:
            pytest.skip("SignalFeatureExtractor not available")

        extractor = SignalFeatureExtractor(channel_id=1, sample_rate=sample_rate)

        duration = 1.0
        t = np.arange(int(sample_rate * duration)) / sample_rate
        signal = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

        features = extractor.process(signal)

        assert hasattr(features, "rms_db")
        assert hasattr(features, "peak_db")
        assert hasattr(features, "crest_factor_db")
        assert features.peak_db >= features.rms_db, (
            "Peak should be >= RMS in dB"
        )

    def test_silence_features(self, sample_rate):
        """Silence should produce very low RMS and peak values."""
        try:
            from signal_analysis import SignalFeatureExtractor
        except ImportError:
            pytest.skip("SignalFeatureExtractor not available")

        extractor = SignalFeatureExtractor(channel_id=1, sample_rate=sample_rate)
        silence = np.zeros(sample_rate, dtype=np.float32)
        features = extractor.process(silence)

        assert features.rms_db < -80.0, f"Silence RMS should be < -80 dB, got {features.rms_db}"


class TestRatioConversion:
    """Tests for Wing OSC ratio float -> string conversion."""

    def test_ratio_conversion_exact(self):
        """Exact ratio values should map to the correct string."""
        assert ratio_float_to_wing(4.0) == "4.0"
        assert ratio_float_to_wing(2.0) == "2.0"
        assert ratio_float_to_wing(1.1) == "1.1"

    def test_ratio_conversion_nearest(self):
        """Non-exact ratios should map to the nearest Wing value."""
        # 3.7 is between 3.5 and 4.0 -- should map to nearest
        result = ratio_float_to_wing(3.7)
        assert result in ("3.5", "4.0")

    def test_ratio_conversion_edge_cases(self):
        """Edge cases: zero, negative, very large values."""
        result = ratio_float_to_wing(0)
        assert result == "1.1"  # Minimum Wing ratio

        result = ratio_float_to_wing(100.0)
        assert result == "100"

    def test_wing_ratio_lists_same_length(self):
        """WING_RATIO_STRINGS and WING_RATIO_VALUES must be the same length."""
        assert len(WING_RATIO_STRINGS) == len(WING_RATIO_VALUES)
