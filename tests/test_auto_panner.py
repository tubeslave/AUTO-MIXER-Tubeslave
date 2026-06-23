"""Tests for spectral auto-panning heuristics."""

from pathlib import Path

from auto_panner import AutoPanner


class FakePanMixer:
    def __init__(self):
        self.calls = []

    def set_channel_pan(self, channel, pan_percent):
        self.calls.append(("set_channel_pan", channel, pan_percent))


def _assert_top_level_denial(result):
    assert result["live_write"] is False
    assert result["queue_mutation"] is False
    assert result["can_apply_live"] is False
    assert result["can_import_to_queue"] is False
    assert result["queue_import_allowed"] is False
    assert result["live_apply_allowed"] is False
    assert result["read_only_or_dry_run_only"] is True
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False
    assert result["queue_action"] == "not_queueable_until_promoted"
    assert result["mutates_mixer"] is False


def test_low_frequency_dominant_band_is_centered():
    panner = AutoPanner()

    decisions = panner.calculate_panning(
        channels=[1],
        instrument_types={1: "unknown"},
        spectral_centroids={1: 1000.0},
        channel_band_energy={1: {"sub": -10.0, "mid": -30.0}},
    )

    assert decisions[1].pan_percent == 0.0
    assert decisions[1].dominant_band == "sub"


def test_same_dominant_band_is_distributed_with_input_priority():
    panner = AutoPanner(max_pan_percent=80.0)

    channel_band_energy = {
        5: {"high_mid": -10.0, "bass": -40.0},
        2: {"high_mid": -11.0, "bass": -40.0},
        8: {"high_mid": -12.0, "bass": -40.0},
        4: {"high_mid": -13.0, "bass": -40.0},
    }
    decisions = panner.calculate_panning(
        channels=[5, 2, 8, 4],
        instrument_types={ch: "guitar" for ch in channel_band_energy},
        spectral_centroids={ch: 3000.0 for ch in channel_band_energy},
        channel_band_energy=channel_band_energy,
    )

    pans = {ch: decision.pan_percent for ch, decision in decisions.items()}
    assert any(value < 0 for value in pans.values())
    assert any(value > 0 for value in pans.values())
    assert abs(pans[2]) < abs(pans[5])
    assert abs(pans[4]) < abs(pans[8])


def test_apply_to_mixer_is_fail_closed_and_returns_blocked_pan_commands():
    panner = AutoPanner(max_pan_percent=80.0)
    decisions = panner.calculate_panning(
        channels=[1, 2, 3],
        instrument_types={1: "lead_vocal", 2: "electric_guitar", 3: "accordion"},
        spectral_centroids={1: 800.0, 2: 3000.0, 3: 3000.0},
    )
    mixer = FakePanMixer()

    result = panner.apply_to_mixer(mixer, decisions)

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["calls"] == []
    assert result["reason"] == "pan_direct_apply_quarantined"
    assert result["write_policy"] == "proposal_only_quarantined"
    assert result["live_write_allowed"] is False
    _assert_top_level_denial(result)
    assert result["mixer_client_present"] is True
    assert mixer.calls == []
    assert {
        "cmd": "set_channel_pan",
        "channel": 1,
        "pan_percent": decisions[1].pan_percent,
        "instrument": "lead_vocal",
        "reason": decisions[1].reason,
    } in result["blocked_calls"]
    expected_max_abs_pan = max(
        abs(call["pan_percent"]) for call in result["blocked_calls"]
    )
    correction = result["operator_correction"]
    assert correction["objective_family"] == "pan_stereo"
    assert correction["future_wing_readback_contract"]["target_kind"] == "channel_pan"
    assert correction["bounded_delta"]["action_type"] == "auto_panner_apply_blocked"
    assert correction["bounded_delta"]["parameter_deltas"]["delta_pan_percent"] == (
        expected_max_abs_pan
    )
    assert correction["bounded_delta"]["parameter_deltas"]["pan_percent"] == (
        expected_max_abs_pan
    )
    assert correction["confidence"] == 0.4
    assert correction["confidence_review"]["status"] == (
        "low_confidence_operator_review_required"
    )
    assert correction["source_evidence"]["evidence_completeness"]["status"] == (
        "complete"
    )
    assert correction["source_evidence"]["evidence_completeness"]["missing"] == []
    assert correction["evidence_gate"]["status"] == (
        "required_evidence_complete_operator_review_still_required"
    )
    assert correction["operator_proposal_review_gate"]["evidence_gate_status"] == (
        "required_evidence_complete_operator_review_still_required"
    )
    assert correction["read_only_or_dry_run_only"] is True
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    assert "mono_compatibility" in correction["headroom_feedback_true_peak_checks"][
        "required_checks"
    ]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "current_pan_cache_no_live_authority"
    ] is True
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "mono_phase_check_no_live_authority"
    ] is True
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_top_level_denial(source_evidence)
    assert source_evidence["stereo_pair"] == {
        "status": "not_verified_in_legacy_auto_panner_apply_path",
        "source": "legacy_auto_panner_plan_blocked_calls",
        "channel_count": 3,
        "stereo_pair_link_readback_required_before_live_promotion": True,
        "proposes_stereo_pair_link_write": False,
    }
    assert source_evidence["operator_context"] == {
        "status": "operator_review_required_before_live_promotion",
        "source": "legacy_auto_panner_plan_blocked_calls",
        "read_only_or_dry_run_only": True,
        "live_write": False,
        "queue_mutation": False,
        "requires_real_wing_proof": True,
        "requires_operator_approval": True,
    }
    assert source_evidence["mono_compatibility"] == {
        "status": "readback_required_before_live_promotion",
        "source": "legacy_auto_panner_plan_blocked_calls",
        "max_abs_pan_percent": expected_max_abs_pan,
        "mono_phase_check_required_before_live_promotion": True,
    }
    assert source_evidence["bounded_pan_delta"] == {
        "status": "passed",
        "max_abs_pan_percent": expected_max_abs_pan,
        "max_allowed_abs_pan_percent": 80.0,
        "transition_ms": 22.0,
        "source": "legacy_auto_panner_plan_blocked_calls",
    }
    assert source_evidence["rollback_pan_readback"] == {
        "status": "required_before_live_promotion",
        "source": "cached_or_physical_pan_read_required_before_promotion",
        "required": True,
        "channel_pan_readback_required": True,
        "stereo_pair_link_readback_required": True,
        "manual_operator_rollback_required_if_promoted": True,
    }
    assert source_evidence["mono_phase_safety"]["status"] == (
        "readback_required_before_live_promotion"
    )
    assert source_evidence["mono_compatibility_risk"] == (
        "not_measured_in_legacy_auto_panner_apply_path"
    )
    evidence_checks = correction["headroom_feedback_true_peak_checks"][
        "evidence_checks"
    ]
    for key in (
        "stereo_pair",
        "operator_context",
        "mono_compatibility",
        "bounded_pan_delta",
        "rollback_pan_readback",
        "mono_phase_safety",
        "mono_compatibility_risk",
        "current_pan_cache_freshness",
        "current_pan_cache_no_live_authority",
        "mono_phase_check_no_live_authority",
        "source_identity_downstream_trust",
    ):
        assert evidence_checks[key] == source_evidence[key]
    assert "pan_readback_not_proven" in correction["safety_blockers"]
    assert panner.last_apply_result is result


def test_auto_fader_v2_pan_separation_has_no_direct_set_channel_pan_call():
    source = Path("backend/auto_fader_v2/controller.py").read_text()

    assert ".set_channel_pan(" not in source
    assert '"cmd": "set_channel_pan"' in source
