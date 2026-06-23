from pathlib import Path

from backend.deesser_vocal_sibilance import (
    DEESSER_READ_REQUIRED_SOURCE,
    DeesserAnalysisInput,
    analyze_deesser_need,
    build_deesser_proposals,
    estimate_sibilance_ratio,
)
from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability


def _fresh(fields=6):
    return {
        "status": "fresh",
        "field_count": fields,
        "fresh_field_count": fields,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def _assert_top_level_denial(payload):
    assert payload["can_apply_live"] is False
    assert payload["can_import_to_queue"] is False
    assert payload["queue_import_allowed"] is False
    assert payload["live_apply_allowed"] is False
    assert payload["read_only_or_dry_run_only"] is True
    assert payload["live_authority_granted"] is False
    assert payload["readiness_closure_allowed"] is False
    assert payload["queue_action"] == "not_queueable_until_promoted"
    assert payload["live_write"] is False
    assert payload["queue_mutation"] is False
    assert payload["mutates_mixer"] is False


def _assert_deesser_operator_contract(payload, *, expected_blocker=None):
    assert payload["required_safety_checks"] == {
        "source_identity": "required_before_live_promotion",
        "arrangement_role": "required_before_live_promotion",
        "speech_intelligibility": "required_before_live_promotion",
        "bounded_attenuation": "required_before_live_promotion",
        "sibilance_persistence": "required_before_live_promotion",
        "true_peak_headroom": "required_before_live_promotion",
        "dynamic_eq_or_deesser_bypass_state": "required_before_live_promotion",
    }
    assert payload["rollback_expectation"] == {
        "required_before_live_promotion": True,
        "restore_previous_dynamics_or_dynamic_eq_values_after_readback": True,
        "restore_previous_bypass_state_after_physical_readback": True,
        "manual_operator_rollback_required_if_promoted": True,
    }
    readback = payload["future_wing_readback_contract"]
    assert readback["required"] is True
    assert readback["required_before_live_promotion"] is True
    assert readback["target_kind"] == "channel_deesser_or_dynamic_eq"
    assert readback["old_value_source"] == DEESSER_READ_REQUIRED_SOURCE
    assert readback["readback_targets"] == [
        "deesser_band",
        "dynamic_eq_band",
        "threshold",
        "ratio",
        "range",
        "max_gain_reduction",
        "bypass_state",
    ]
    assert readback["blocked_live_actions"] == [
        "deesser_parameter_write",
        "dynamic_eq_band_write",
        "threshold_range_write",
    ]
    assert readback["requires_real_wing_proof"] is True
    assert readback["requires_operator_approval"] is True
    assert readback["requires_separate_non_fader_adr"] is True
    assert "De-esser evidence is offline-only" in payload["operator_explanation"]
    assert payload["safety_blockers"][:7] == [
        "deesser_live_apply_quarantined",
        "unsupported_value_type:deesser",
        "not_in_supervised_fader_gain_executor",
        "real_wing_proof_required_before_live_apply",
        "non_fader_live_expansion_requires_separate_adr",
        "operator_approval_required_before_supervised_deesser_apply",
        "hardware_readback_and_rollback_not_proven",
    ]
    if expected_blocker is not None:
        assert expected_blocker in payload["safety_blockers"]


def test_deesser_module_is_transport_free():
    source = Path("backend/deesser_vocal_sibilance.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "set_channel_deesser",
        "set_deesser",
        "set_eq(",
        "set_compressor(",
    ]
    for token in forbidden:
        assert token not in source


def test_vocal_sibilance_creates_proposal_only_recommendation():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=7,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            arrangement_role="lead",
            band_energy_db={
                "low_mid": -38.0,
                "mid": -30.0,
                "presence": -24.0,
                "sibilance": -16.0,
                "air": -18.0,
            },
            sibilance_excess_db=4.5,
            peak_dbfs=-7.0,
            contribution_share=0.72,
            persistence_cycles=3,
            true_peak_headroom_db=5.0,
            current_state_cache_freshness={
                "status": "fresh",
                "field_count": 8,
                "fresh_field_count": 8,
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
            arrangement_context_cache_freshness={
                "status": "fresh",
                "field_count": 4,
                "fresh_field_count": 4,
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "deesser_proposal"
    assert 5500.0 <= recommendation.frequency_hz <= 10500.0
    assert -42.0 <= recommendation.threshold_db <= -12.0
    assert 1.5 <= recommendation.ratio <= 6.0
    assert 1.0 <= recommendation.max_gain_reduction_db <= 8.0
    assert recommendation.confidence >= 0.45
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["bounded_attenuation_only"] is True
    assert recommendation.evidence["makeup_gain_db"] == 0.0
    assert recommendation.evidence["boost_allowed"] is False
    assert recommendation.evidence["max_gain_reduction_db"] == (
        recommendation.max_gain_reduction_db
    )
    assert recommendation.evidence["arrangement_role"] == "lead"
    assert recommendation.evidence["arrangement_deesser_profile"] == "lead_vocal"
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_state_cache_freshness"]["fresh_field_count"] == 8
    assert recommendation.evidence["arrangement_context_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["arrangement_context_cache_freshness"]["fresh_field_count"] == 4
    assert recommendation.evidence["source_identity_downstream_trust"]["status"] == "legacy_unverified"
    summary = recommendation.deesser_safety_summary()
    assert summary["status"] == "passed"
    assert summary["source_role"] == "lead_vocal"
    assert summary["arrangement_deesser_profile"] == "lead_vocal"
    assert summary["bounded_attenuation_only"] is True
    assert summary["makeup_gain_db"] == 0.0
    assert summary["boost_allowed"] is False
    assert summary["max_gain_reduction_db"] == recommendation.max_gain_reduction_db
    assert summary["current_dynamics_or_dynamic_eq_cache_freshness_status"] == "fresh"
    assert summary["current_dynamics_or_dynamic_eq_cache_no_live_authority"] is True
    assert summary["arrangement_context_cache_freshness_status"] == "fresh"
    assert summary["current_arrangement_context_cache_no_live_authority"] is True
    assert summary["sibilance_persistence"] == {
        "persistence_cycles": 3,
        "minimum_cycles_before_proposal": 2,
        "status": "persistent",
    }
    assert summary["speech_intelligibility"] == {
        "status": "operator_review_required_before_live_promotion",
        "bounded_attenuation_only": True,
        "boost_allowed": False,
        "max_gain_reduction_db": recommendation.max_gain_reduction_db,
        "arrangement_deesser_profile": "lead_vocal",
    }
    _assert_deesser_operator_contract(summary)
    assert summary["readback_required_before_promotion"] is True
    assert summary["rollback_required_before_promotion"] is True
    assert summary["live_write"] is False
    assert summary["queue_mutation"] is False
    _assert_top_level_denial(summary)

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "deesser_vocal_sibilance"
    _assert_top_level_denial(proposal)
    _assert_deesser_operator_contract(proposal)
    assert proposal["safety_blockers"] == summary["safety_blockers"]
    assert "requested_change" not in proposal
    assert "unsupported_value_type:deesser" in proposal["blockers"]
    assert proposal["old_value"]["source"] == DEESSER_READ_REQUIRED_SOURCE
    assert proposal["rollback_value"]["source"] == DEESSER_READ_REQUIRED_SOURCE
    assert proposal["proposed_action"]["max_gain_reduction_db"] == (
        recommendation.max_gain_reduction_db
    )
    assert proposal["proposed_action"]["makeup_gain_db"] == 0.0
    assert proposal["proposed_action"]["boost_allowed"] is False
    assert proposal["proposed_action"]["required_safety_checks"] == summary[
        "required_safety_checks"
    ]
    assert proposal["proposed_action"]["rollback_expectation"] == summary[
        "rollback_expectation"
    ]
    assert proposal["proposed_action"]["future_wing_readback_contract"] == summary[
        "future_wing_readback_contract"
    ]
    assert proposal["deesser_safety_summary"] == summary
    assert proposal["proposed_action"]["deesser_safety_summary"] == summary
    assert proposal["evidence"]["deesser_safety_summary"] == summary
    assert proposal["evidence"]["operator_explanation"] == summary[
        "operator_explanation"
    ]
    assert proposal["evidence"]["required_safety_checks"] == summary[
        "required_safety_checks"
    ]
    assert proposal["evidence"]["rollback_expectation"] == summary[
        "rollback_expectation"
    ]
    assert proposal["evidence"]["future_wing_readback_contract"] == summary[
        "future_wing_readback_contract"
    ]
    assert proposal["evidence"]["old_value_source"] == DEESSER_READ_REQUIRED_SOURCE
    assert proposal["evidence"]["rollback_value_source"] == DEESSER_READ_REQUIRED_SOURCE
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["bounded_attenuation_only"] is True
    assert proposal["evidence"]["max_gain_reduction_db"] == (
        recommendation.max_gain_reduction_db
    )
    assert proposal["evidence"]["current_state_cache_freshness_status"] == "fresh"
    assert proposal["evidence"]["arrangement_context_cache_freshness_status"] == "fresh"
    assert proposal["evidence"]["vocal_source_role"] == "lead_vocal"
    assert proposal["evidence"]["true_peak_headroom"] == 5.0
    assert proposal["evidence"]["current_dynamics_or_dynamic_eq_cache_freshness"][
        "fresh_field_count"
    ] == 8
    assert proposal["evidence"]["current_arrangement_context_cache_freshness"][
        "fresh_field_count"
    ] == 4
    assert proposal["evidence"]["sibilance_persistence"] == summary[
        "sibilance_persistence"
    ]
    assert proposal["evidence"]["speech_intelligibility"] == summary[
        "speech_intelligibility"
    ]
    assert proposal["evidence"]["quiet_words_tail_attack_preservation"] == summary[
        "quiet_words_tail_attack_preservation"
    ]
    assert proposal["evidence"]["quiet_words_tail_attack_preservation"][
        "broadband_gate_or_expander"
    ] is False
    assert proposal["evidence"]["dynamic_eq_or_deesser_readback"] == (
        DEESSER_READ_REQUIRED_SOURCE
    )
    assert proposal["evidence"]["rollback_dynamics_or_dynamic_eq_readback"] == (
        DEESSER_READ_REQUIRED_SOURCE
    )
    assert proposal["evidence"]["bypass_state"] == (
        "readback_required_before_live_promotion"
    )
    correction = proposal["operator_correction"]
    assert correction["objective_family"] == "gate_expander_deesser"
    assert correction["source_evidence"]["evidence_completeness"]["status"] == (
        "complete"
    )
    assert {
        "bounded_attenuation_only",
        "max_gain_reduction_db",
        "speech_intelligibility",
        "quiet_words_tail_attack_preservation",
        "dynamic_eq_or_deesser_readback",
        "old_value_source",
        "rollback_dynamics_or_dynamic_eq_readback",
        "bypass_state",
    } <= set(correction["source_evidence"]["required_evidence_fields"])
    assert correction["evidence_gate"]["status"] == (
        "required_evidence_complete_operator_review_still_required"
    )
    assert correction["bounded_delta"]["delta_db"] == recommendation.max_gain_reduction_db
    assert correction["bounded_delta"]["parameter_deltas"]["threshold_db"] == (
        recommendation.threshold_db
    )
    assert correction["bounded_delta"]["parameter_deltas"]["max_gain_reduction_db"] == (
        recommendation.max_gain_reduction_db
    )
    assert correction["bounded_delta"]["parameter_deltas"]["boost_allowed"] is False
    assert correction["future_wing_readback_contract"]["target_kind"] == (
        "channel_deesser_or_dynamic_eq"
    )
    assert correction["future_wing_readback_contract"][
        "required_before_live_promotion"
    ] is True
    assert correction["read_only_or_dry_run_only"] is True
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    assert "speech_intelligibility" in correction[
        "headroom_feedback_true_peak_checks"
    ]["required_checks"]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "current_dynamics_or_dynamic_eq_cache_no_live_authority"
    ] is True
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_top_level_denial(source_evidence)
    evidence_checks = correction["headroom_feedback_true_peak_checks"][
        "evidence_checks"
    ]
    for key in (
        "vocal_source_role",
        "arrangement_role",
        "sibilance_ratio",
        "sibilance_persistence",
        "current_arrangement_context_cache_freshness",
        "current_arrangement_context_cache_no_live_authority",
        "current_dynamics_or_dynamic_eq_cache_freshness",
        "current_dynamics_or_dynamic_eq_cache_no_live_authority",
        "source_identity_downstream_trust",
        "true_peak_headroom",
        "bounded_attenuation_only",
        "max_gain_reduction_db",
        "speech_intelligibility",
        "quiet_words_tail_attack_preservation",
        "dynamic_eq_or_deesser_readback",
        "old_value_source",
        "rollback_dynamics_or_dynamic_eq_readback",
        "bypass_state",
    ):
        assert evidence_checks[key] == source_evidence[key]
    assert source_evidence["deesser_safety_summary"] == summary
    assert source_evidence["future_wing_readback_contract"] == summary[
        "future_wing_readback_contract"
    ]
    assert (
        source_evidence["current_dynamics_or_dynamic_eq_cache_no_live_authority"]
        is True
    )
    assert source_evidence["current_arrangement_context_cache_no_live_authority"] is True


def test_backing_vocal_deesser_profile_is_less_aggressive_than_lead():
    shared = {
        "channel_id": 7,
        "channel_label": "Vox",
        "source_role": "lead_vocal",
        "band_energy_db": {
            "low_mid": -38.0,
            "mid": -30.0,
            "presence": -24.0,
            "sibilance": -16.0,
            "air": -18.0,
        },
        "sibilance_excess_db": 4.5,
        "peak_dbfs": -7.0,
        "contribution_share": 0.72,
        "persistence_cycles": 3,
        "true_peak_headroom_db": 5.0,
        "current_state_cache_freshness": _fresh(8),
        "arrangement_context_cache_freshness": _fresh(4),
    }

    lead = analyze_deesser_need(
        DeesserAnalysisInput(**shared, arrangement_role="lead")
    )
    tucked = analyze_deesser_need(
        DeesserAnalysisInput(**shared, arrangement_role="tucked backing vocal")
    )

    assert lead.status == "proposal_only"
    assert tucked.status == "proposal_only"
    assert tucked.evidence["arrangement_deesser_profile"] == "tucked_or_backing_vocal"
    assert tucked.ratio < lead.ratio
    assert tucked.threshold_db > lead.threshold_db
    assert tucked.max_gain_reduction_db < lead.max_gain_reduction_db
    assert tucked.confidence < lead.confidence


def test_stale_arrangement_context_blocks_deesser_proposal():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=7,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            arrangement_role="lead",
            band_energy_db={
                "low_mid": -38.0,
                "mid": -30.0,
                "presence": -24.0,
                "sibilance": -16.0,
                "air": -18.0,
            },
            sibilance_excess_db=4.5,
            peak_dbfs=-7.0,
            contribution_share=0.72,
            persistence_cycles=3,
            true_peak_headroom_db=5.0,
            arrangement_context_cache_freshness={
                "status": "stale",
                "field_count": 4,
                "fresh_field_count": 2,
                "stale_field_count": 2,
                "stale_fields": ["arrangement_role", "section_role"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "deesser_arrangement_context_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["arrangement_context_cache_freshness_status"] == "stale"
    assert recommendation.evidence["arrangement_context_cache_freshness"]["stale_fields"] == [
        "arrangement_role",
        "section_role",
    ]
    summary = recommendation.deesser_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["arrangement_context_cache_freshness_status"] == "stale"
    assert "deesser_arrangement_context_cache_not_fresh" in summary["blockers"]
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert proposal["deesser_safety_summary"] == summary


def test_non_vocal_sibilance_is_not_deesser_proposal():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=3,
            channel_label="OH L",
            source_role="overhead",
            band_energy_db={"sibilance": -12.0, "presence": -24.0},
            sibilance_excess_db=7.0,
            contribution_share=0.8,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "source_not_vocal" in recommendation.blockers


def test_low_sibilance_vocal_is_analysis_only():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=8,
            source_role="backing_vocal",
            band_energy_db={"sibilance": -42.0, "presence": -25.0, "mid": -24.0},
            sibilance_excess_db=0.5,
            persistence_cycles=1,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "sibilance_below_threshold" in recommendation.blockers


def test_stale_current_deesser_state_blocks_proposal_only_recommendation():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=7,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            band_energy_db={
                "low_mid": -38.0,
                "mid": -30.0,
                "presence": -24.0,
                "sibilance": -16.0,
                "air": -18.0,
            },
            sibilance_excess_db=4.5,
            peak_dbfs=-7.0,
            contribution_share=0.72,
            persistence_cycles=3,
            true_peak_headroom_db=5.0,
            current_state_cache_freshness={
                "status": "stale",
                "field_count": 8,
                "fresh_field_count": 5,
                "stale_field_count": 3,
                "stale_fields": ["frequency_hz", "threshold_db", "ratio"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "deesser_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "stale"
    assert recommendation.evidence["current_state_cache_freshness"]["stale_fields"] == [
        "frequency_hz",
        "threshold_db",
        "ratio",
    ]
    summary = recommendation.deesser_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["current_dynamics_or_dynamic_eq_cache_freshness_status"] == "stale"
    assert "deesser_current_state_cache_not_fresh" in summary["blockers"]
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert proposal["deesser_safety_summary"] == summary


def test_missing_current_deesser_state_blocks_proposal_only_recommendation():
    analysis = DeesserAnalysisInput(
        channel_id=7,
        channel_label="Lead Vox",
        source_role="lead_vocal",
        arrangement_role="lead",
        band_energy_db={
            "low_mid": -38.0,
            "mid": -30.0,
            "presence": -24.0,
            "sibilance": -16.0,
            "air": -18.0,
        },
        sibilance_excess_db=4.5,
        peak_dbfs=-7.0,
        contribution_share=0.72,
        persistence_cycles=3,
        true_peak_headroom_db=5.0,
        arrangement_context_cache_freshness=_fresh(4),
    )
    recommendation = analyze_deesser_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "deesser_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "not_provided"
    assert build_deesser_proposals([analysis]) == ()


def test_missing_arrangement_context_blocks_deesser_proposal_only_recommendation():
    analysis = DeesserAnalysisInput(
        channel_id=7,
        channel_label="Lead Vox",
        source_role="lead_vocal",
        arrangement_role="lead",
        band_energy_db={
            "low_mid": -38.0,
            "mid": -30.0,
            "presence": -24.0,
            "sibilance": -16.0,
            "air": -18.0,
        },
        sibilance_excess_db=4.5,
        peak_dbfs=-7.0,
        contribution_share=0.72,
        persistence_cycles=3,
        true_peak_headroom_db=5.0,
        current_state_cache_freshness=_fresh(8),
    )
    recommendation = analyze_deesser_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "deesser_arrangement_context_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["arrangement_context_cache_freshness_status"] == "not_provided"
    assert build_deesser_proposals([analysis]) == ()


def test_live_authority_current_deesser_state_blocks_proposal_only_recommendation():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=7,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            arrangement_role="lead",
            band_energy_db={
                "low_mid": -38.0,
                "mid": -30.0,
                "presence": -24.0,
                "sibilance": -16.0,
                "air": -18.0,
            },
            sibilance_excess_db=4.5,
            peak_dbfs=-7.0,
            contribution_share=0.72,
            persistence_cycles=3,
            true_peak_headroom_db=5.0,
            current_state_cache_freshness={
                "status": "fresh",
                "field_count": 8,
                "fresh_field_count": 8,
                "read_only": False,
                "live_write": True,
                "queries_console": True,
            },
            arrangement_context_cache_freshness=_fresh(4),
        )
    )

    assert recommendation.status == "analysis_only"
    assert "deesser_current_state_cache_live_authority_present" in recommendation.blockers
    assert (
        recommendation.evidence[
            "current_dynamics_or_dynamic_eq_cache_no_live_authority"
        ]
        is False
    )
    summary = recommendation.deesser_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["current_dynamics_or_dynamic_eq_cache_no_live_authority"] is False
    assert "deesser_current_state_cache_live_authority_present" in summary["blockers"]
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert proposal["deesser_safety_summary"] == summary
    assert "deesser_current_state_cache_live_authority_present" in proposal["blockers"]


def test_live_authority_arrangement_context_blocks_deesser_proposal_only_recommendation():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=7,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            arrangement_role="lead",
            band_energy_db={
                "low_mid": -38.0,
                "mid": -30.0,
                "presence": -24.0,
                "sibilance": -16.0,
                "air": -18.0,
            },
            sibilance_excess_db=4.5,
            peak_dbfs=-7.0,
            contribution_share=0.72,
            persistence_cycles=3,
            true_peak_headroom_db=5.0,
            current_state_cache_freshness=_fresh(8),
            arrangement_context_cache_freshness={
                "status": "fresh",
                "field_count": 4,
                "fresh_field_count": 4,
                "read_only": False,
                "live_write": True,
                "queries_console": True,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert (
        "deesser_arrangement_context_cache_live_authority_present"
        in recommendation.blockers
    )
    assert (
        recommendation.evidence[
            "current_arrangement_context_cache_no_live_authority"
        ]
        is False
    )
    summary = recommendation.deesser_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["current_arrangement_context_cache_no_live_authority"] is False
    assert (
        "deesser_arrangement_context_cache_live_authority_present"
        in summary["blockers"]
    )
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert proposal["deesser_safety_summary"] == summary
    assert (
        "deesser_arrangement_context_cache_live_authority_present"
        in proposal["blockers"]
    )


def test_source_identity_hold_blocks_downstream_deesser_proposal():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=7,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            band_energy_db={
                "low_mid": -38.0,
                "mid": -30.0,
                "presence": -24.0,
                "sibilance": -16.0,
                "air": -18.0,
            },
            sibilance_excess_db=4.5,
            peak_dbfs=-7.0,
            contribution_share=0.72,
            persistence_cycles=3,
            true_peak_headroom_db=5.0,
            source_identity_trust_state="hold_for_operator_review",
            source_identity_trust_confidence=0.42,
            source_identity_replay_outcome_count=4,
            source_identity_calibration_freshness={
                "status": "fresh",
                "field_count": 4,
                "fresh_field_count": 4,
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "source_identity_not_trusted_for_downstream" in recommendation.blockers
    assert recommendation.severity == "medium"
    trust = recommendation.evidence["source_identity_downstream_trust"]
    assert trust["status"] == "held_for_operator_review"
    assert trust["usable_for_downstream"] is False
    assert trust["evidence"]["model_mutation"] is False
    summary = recommendation.deesser_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["source_identity_usable_for_downstream"] is False
    assert "source_identity_not_trusted_for_downstream" in summary["blockers"]
    _assert_deesser_operator_contract(
        summary,
        expected_blocker="source_identity_not_trusted_for_downstream",
    )
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    _assert_deesser_operator_contract(
        proposal,
        expected_blocker="source_identity_not_trusted_for_downstream",
    )
    assert proposal["deesser_safety_summary"] == summary


def test_stale_source_identity_replay_context_blocks_downstream_deesser_proposal():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=7,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            band_energy_db={
                "low_mid": -38.0,
                "mid": -30.0,
                "presence": -24.0,
                "sibilance": -16.0,
                "air": -18.0,
            },
            sibilance_excess_db=4.5,
            peak_dbfs=-7.0,
            contribution_share=0.72,
            persistence_cycles=3,
            true_peak_headroom_db=5.0,
            source_identity_trust_state="trusted_for_downstream_priority",
            source_identity_trust_confidence=0.9,
            source_identity_replay_outcome_count=5,
            source_identity_calibration_freshness={
                "status": "stale",
                "field_count": 4,
                "fresh_field_count": 2,
                "stale_field_count": 2,
                "stale_fields": ["accepted_labels", "replay_outcomes"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "source_identity_replay_context_not_fresh" in recommendation.blockers
    assert recommendation.severity == "high"
    trust = recommendation.evidence["source_identity_downstream_trust"]
    assert trust["status"] == "trusted"
    assert trust["usable_for_downstream"] is False
    assert trust["evidence"]["current_state_cache_freshness"]["stale_fields"] == [
        "accepted_labels",
        "replay_outcomes",
    ]


def test_deesser_ratio_estimator_uses_sibilance_band_share():
    ratio = estimate_sibilance_ratio(
        {"mid": -30.0, "presence": -28.0, "sibilance": -16.0, "air": -28.0}
    )

    assert ratio > 0.5


def test_build_deesser_proposals_filters_to_proposal_only_items():
    proposals = build_deesser_proposals(
        [
            DeesserAnalysisInput(
                channel_id=7,
                source_role="lead_vocal",
                band_energy_db={"sibilance": -15.0, "presence": -25.0},
                sibilance_excess_db=4.0,
                contribution_share=0.7,
                persistence_cycles=3,
                current_state_cache_freshness=_fresh(8),
                arrangement_context_cache_freshness=_fresh(4),
            ),
            DeesserAnalysisInput(
                channel_id=3,
                source_role="overhead",
                band_energy_db={"sibilance": -15.0, "presence": -25.0},
                sibilance_excess_db=4.0,
                contribution_share=0.7,
                persistence_cycles=3,
            ),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["channel"] == 7


def test_build_deesser_proposals_filters_stale_state_items():
    proposals = build_deesser_proposals(
        [
            DeesserAnalysisInput(
                channel_id=7,
                source_role="lead_vocal",
                band_energy_db={"sibilance": -15.0, "presence": -25.0},
                sibilance_excess_db=4.0,
                contribution_share=0.7,
                persistence_cycles=3,
                current_state_cache_freshness={"status": "stale"},
            ),
        ]
    )

    assert proposals == ()


def test_deesser_is_registered_as_proposal_only_goal_capability():
    method = get_processing_method("deesser_vocal_sibilance")
    capability = get_project_goal_capability("deessing")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_write_live is False
    assert capability is not None
    assert capability.status == "proposal_only"
    assert capability.method_ids == ("deesser_vocal_sibilance",)
    assert contracts["deesser_vocal_sibilance"]["proposal_contract"]["value_type"] == "deesser"
    evidence = contracts["deesser_vocal_sibilance"]["proposal_contract"]["safety_evidence_required"]
    assert "source_identity_downstream_trust" in evidence
    assert "current_arrangement_context_cache_freshness" in evidence
    assert "current_arrangement_context_cache_no_live_authority" in evidence
    assert "current_dynamics_or_dynamic_eq_cache_no_live_authority" in evidence
    assert "bounded_attenuation_only" in evidence
    assert "max_gain_reduction_db" in evidence
    assert "speech_intelligibility" in evidence
    assert "quiet_words_tail_attack_preservation" in evidence
    assert "dynamic_eq_or_deesser_readback" in evidence
    assert "old_value_source" in evidence
    assert "rollback_dynamics_or_dynamic_eq_readback" in evidence
    assert "bypass_state" in evidence
