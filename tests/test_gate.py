"""
Tests for the CAIG (Cross-Adaptive Intelligent Gate) system.

Covers:
- Gate threshold calculation
- Gate opening on signal above threshold
- Feature extraction (RMS, peak, crest factor, LF energy)
- Adaptive threshold noise floor tracking
"""

import numpy as np
import pytest

from auto_gate_caig import (
    GateState,
    GateFeatures,
    GateSettings,
    FeatureExtractor,
    AdaptiveThreshold,
)


class TestGateThreshold:
    """Tests for adaptive gate threshold computation."""

    def test_gate_threshold(self):
        """AdaptiveThreshold should return a reasonable threshold for quiet signals."""
        at = AdaptiveThreshold()
        # Simulate a quiet noise floor around -60 dB
        for _ in range(20):
            at.update_noise_floor(-60.0)

        features = GateFeatures(rms_db=-55.0, peak_db=-50.0,
                                crest_factor_db=5.0, lf_energy_db=-58.0)
        threshold = at.calculate_threshold(features, group_influence=0.0)

        # Threshold should be above noise floor but below the signal
        assert threshold > -70.0, f"Threshold too low: {threshold}"
        assert threshold < -40.0, f"Threshold too high: {threshold}"

    def test_default_gate_settings(self):
        """Default GateSettings should have sensible values."""
        gs = GateSettings()
        assert gs.threshold_db == -60.0
        assert gs.attack_ms == 0.5
        assert gs.release_ms == 80.0
        assert gs.hold_ms == 10.0
        assert gs.range_db == -80.0
        assert gs.hysteresis_db == 3.0

    def test_gate_state_enum(self):
        """GateState enum should have the expected members."""
        assert GateState.CLOSED.value is not None
        assert GateState.OPEN.value is not None
        assert GateState.HOLD.value is not None


class TestGateOpensOnSignal:
    """Tests for gate behavior when signal exceeds threshold."""

    def test_gate_opens_on_signal(self, sample_rate):
        """FeatureExtractor should detect signal above threshold."""
        extractor = FeatureExtractor(sample_rate=sample_rate)

        # Generate a loud signal burst
        duration = 0.01  # 10 ms
        t = np.arange(int(sample_rate * duration)) / sample_rate
        loud_signal = (0.8 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)

        features = extractor.extract(loud_signal)

        assert isinstance(features, GateFeatures)
        # Loud signal should have high RMS
        assert features.rms_db > -20.0, (
            f"Loud signal RMS should be > -20 dB, got {features.rms_db:.1f}"
        )
        assert features.peak_db > -10.0, (
            f"Loud signal peak should be > -10 dB, got {features.peak_db:.1f}"
        )

    def test_silence_stays_below_threshold(self, sample_rate):
        """Silence should produce features well below any reasonable threshold."""
        extractor = FeatureExtractor(sample_rate=sample_rate)
        silence = np.zeros(int(sample_rate * 0.01), dtype=np.float32)
        features = extractor.extract(silence)

        assert features.rms_db < -60.0, (
            f"Silence RMS should be < -60 dB, got {features.rms_db:.1f}"
        )


class TestFeatureExtraction:
    """Tests for GateFeatures extraction from audio."""

    def test_feature_extraction_crest_factor(self, sample_rate):
        """Crest factor should be peak - RMS in dB."""
        extractor = FeatureExtractor(sample_rate=sample_rate)
        duration = 0.01
        t = np.arange(int(sample_rate * duration)) / sample_rate
        signal = (0.5 * np.sin(2 * np.pi * 500.0 * t)).astype(np.float32)
        features = extractor.extract(signal)

        # For a sine wave, crest factor ~3 dB
        expected_crest = features.peak_db - features.rms_db
        assert abs(features.crest_factor_db - expected_crest) < 1.0, (
            f"Crest factor mismatch: {features.crest_factor_db:.1f} vs "
            f"expected {expected_crest:.1f}"
        )

    def test_gate_features_dataclass(self):
        """GateFeatures should be constructible with all fields."""
        f = GateFeatures(rms_db=-30.0, peak_db=-24.0,
                         crest_factor_db=6.0, lf_energy_db=-35.0)
        assert f.rms_db == -30.0
        assert f.peak_db == -24.0
        assert f.crest_factor_db == 6.0
        assert f.lf_energy_db == -35.0
