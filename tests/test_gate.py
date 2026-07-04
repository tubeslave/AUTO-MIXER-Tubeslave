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
    AutoGateController,
    GateState,
    GateDecision,
    GateFeatures,
    GateSettings,
    GATE_READ_REQUIRED_SOURCE,
    FeatureExtractor,
    AdaptiveThreshold,
    InstrumentGroup,
    live_gate_settings_for_instrument,
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
        assert gs.release_ms == 120.0
        assert gs.hold_ms == 100.0
        assert gs.range_db == -80.0
        assert gs.hysteresis_db == 3.0

    def test_live_gate_settings_for_drums(self):
        """Drum presets should preserve transients and avoid chatter."""
        gs = live_gate_settings_for_instrument(
            "snare",
            nearest_source_db=-42.0,
            noise_floor_db=-70.0,
        )
        assert 0.01 <= gs.attack_ms <= 0.1
        assert 100.0 <= gs.hold_ms <= 300.0
        assert 100.0 <= gs.release_ms <= 300.0
        assert gs.threshold_db == -36.0

    def test_live_gate_settings_for_sustained_sources(self):
        """Non-drum presets should sit below the target peak."""
        gs = live_gate_settings_for_instrument("vocal", target_peak_db=-18.0)
        assert 0.1 <= gs.attack_ms <= 1.0
        assert 50.0 <= gs.hold_ms <= 200.0
        assert 50.0 <= gs.release_ms <= 200.0
        assert gs.threshold_db == -24.0

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


class FakeGateMixerClient:
    def __init__(self):
        self.calls = []

    def set_channel_gate(self, channel_id, gain_db):
        self.calls.append((channel_id, gain_db))


def _assert_top_level_denial(result):
    assert result["can_apply_live"] is False
    assert result["can_import_to_queue"] is False
    assert result["queue_import_allowed"] is False
    assert result["live_apply_allowed"] is False
    assert result["read_only_or_dry_run_only"] is True
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False
    assert result["queue_action"] == "not_queueable_until_promoted"
    assert result["live_write"] is False
    assert result["queue_mutation"] is False
    assert result["mutates_mixer"] is False


class TestAutoGateControllerWritePolicy:
    """Tests that legacy CAIG mixer-client paths stay fail-closed."""

    def test_process_with_mixer_client_does_not_write_gate(self, sample_rate):
        mixer = FakeGateMixerClient()
        controller = AutoGateController(
            num_channels=2,
            sample_rate=sample_rate,
            mixer_client=mixer,
        )
        controller.configure_channel(
            0,
            "snare",
            InstrumentGroup.DRUMS,
            is_drum=True,
        )

        duration = 0.02
        t = np.arange(int(sample_rate * duration)) / sample_rate
        loud_signal = (0.8 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)

        decisions = controller.process({0: loud_signal})

        assert decisions
        assert mixer.calls == []
        status = controller.get_status()
        assert status["live_write_allowed"] is False
        assert status["write_policy"] == "proposal_only_quarantined"
        assert status["mixer_client_present"] is True
        assert status["last_write_attempted"] is True
        assert status["last_write_blocked"] is True
        assert status["last_write_count"] == 0
        result = status["last_apply_result"]
        assert result["applied"] is False
        assert result["blocked"] is True
        assert result["reason"] == "auto_gate_caig_direct_apply_quarantined"
        assert result["confidence"] == 0.0
        assert result["confidence_basis"] == (
            "legacy_caig_write_path_quarantined_without_fresh_readonly_gate_context"
        )
        _assert_top_level_denial(result)
        assert result["blocked_call_count"] == 1
        assert result["blocked_calls"][0]["cmd"] == "set_channel_gate"
        correction = result["operator_correction"]
        assert correction["objective_family"] == "gate_expander"
        assert correction["future_wing_readback_contract"]["target_kind"] == "channel_gate"
        assert correction["bounded_delta"]["action_type"] == "auto_gate_caig_write_blocked"
        # Барабанный range −20 дБ (не полное закрытие): сохраняем
        # естественный хвост барабана — live-правило FOH-документа.
        assert correction["bounded_delta"]["delta_db"] == -20.0
        assert correction["bounded_delta"]["parameter_deltas"]["range_db"] == -20.0
        assert correction["bounded_delta"]["parameter_deltas"]["hysteresis_db"] == 3.0
        assert correction["bounded_delta"]["parameter_deltas"]["attack_ms"] == 0.05
        assert correction["live_apply_allowed"] is False
        assert correction["queue_import_allowed"] is False
        assert "chatter_guard" in correction["headroom_feedback_true_peak_checks"][
            "required_checks"
        ]
        assert "tails_and_attack" in correction["headroom_feedback_true_peak_checks"][
            "required_checks"
        ]
        assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
            "current_gate_bleed_context_cache_no_live_authority"
        ] is True
        source_evidence = correction["source_evidence"]["evidence"]
        _assert_top_level_denial(source_evidence)
        assert source_evidence["confidence"] == 0.0
        assert source_evidence["confidence_basis"] == result["confidence_basis"]
        assert source_evidence["old_value_source"] == GATE_READ_REQUIRED_SOURCE
        assert source_evidence["rollback_value_source"] == GATE_READ_REQUIRED_SOURCE
        assert source_evidence["rollback_gate_readback"] == GATE_READ_REQUIRED_SOURCE
        assert source_evidence["noise_floor"] == (
            "not_measured_in_legacy_caig_write_path"
        )
        assert source_evidence["bleed_risk"] == (
            "not_measured_in_legacy_caig_write_path"
        )
        assert source_evidence["quiet_words_tail_attack_preservation"][
            "legacy_gate_write_quarantined"
        ] is True
        evidence_checks = correction["headroom_feedback_true_peak_checks"][
            "evidence_checks"
        ]
        for key in (
            "noise_floor",
            "source_role",
            "bleed_risk",
            "current_gate_bleed_context_cache_freshness",
            "current_gate_bleed_context_cache_no_live_authority",
            "source_identity_downstream_trust",
            "old_value_source",
            "rollback_gate_readback",
            "chatter_guard",
            "hysteresis",
            "tail_risk",
            "quiet_words_tail_attack_preservation",
        ):
            assert evidence_checks[key] == source_evidence[key]
        completeness = correction["source_evidence"]["evidence_completeness"]
        assert completeness["status"] == "complete"
        assert completeness["missing"] == []
        assert correction["operator_proposal_review_gate"][
            "evidence_gate_status"
        ] == "required_evidence_complete_operator_review_still_required"
        assert source_evidence["queue_import_allowed"] is False
        assert source_evidence["live_apply_allowed"] is False
        assert source_evidence["read_only_or_dry_run_only"] is True
        assert "hardware_readback_and_rollback_not_proven" in correction[
            "safety_blockers"
        ]

    def test_direct_send_osc_call_is_quarantined(self):
        mixer = FakeGateMixerClient()
        controller = AutoGateController(num_channels=1, mixer_client=mixer)

        sent = controller._send_osc(
            {
                0: GateDecision(
                    channel_id=0,
                    state=GateState.OPEN,
                    gain_db=0.0,
                )
            }
        )

        assert sent == 0
        assert mixer.calls == []
        status = controller.get_status()
        assert status["last_write_attempted"] is True
        assert status["last_write_blocked"] is True
        assert status["last_write_count"] == 0
        result = status["last_apply_result"]
        assert result["blocked"] is True
        assert result["blocked_call_count"] == 1
        _assert_top_level_denial(result)
        assert result["operator_correction"]["objective_family"] == "gate_expander"
        assert result["operator_correction"]["read_only_or_dry_run_only"] is True
        assert result["operator_correction"]["live_apply_allowed"] is False
        assert result["operator_correction"]["source_evidence"]["evidence"][
            "rollback_gate_readback"
        ] == GATE_READ_REQUIRED_SOURCE


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
