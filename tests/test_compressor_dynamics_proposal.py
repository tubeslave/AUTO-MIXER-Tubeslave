from pathlib import Path

from backend.compressor_dynamics_proposal import (
    CompressorAnalysisInput,
    DYNAMICS_READ_REQUIRED_SOURCE,
    analyze_compressor_need,
    build_compressor_proposals,
    classify_source_family,
    estimate_crest_factor_db,
)
from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_pipeline_profile, get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability


def _fresh(fields=10):
    return {
        "status": "fresh",
        "field_count": fields,
        "fresh_field_count": fields,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def _assert_top_level_denial(payload):
    assert payload["live_write"] is False
    assert payload["queue_mutation"] is False
    assert payload["can_apply_live"] is False
    assert payload["can_import_to_queue"] is False
    assert payload["queue_import_allowed"] is False
    assert payload["live_apply_allowed"] is False
    assert payload["read_only_or_dry_run_only"] is True
    assert payload["live_authority_granted"] is False
    assert payload["readiness_closure_allowed"] is False
    assert payload["queue_action"] == "not_queueable_until_promoted"
    assert payload["mutates_mixer"] is False


def _assert_dynamics_operator_contract(payload, *, expected_blocker=None):
    assert payload["required_safety_checks"] == {
        "source_identity": "required_before_live_promotion",
        "gain_reduction_review": "required_before_live_promotion",
        "attack_release_bounds": "required_before_live_promotion",
        "makeup_gain_headroom": "required_before_live_promotion",
        "true_peak_headroom": "required_before_live_promotion",
        "feedback_risk": "required_before_live_promotion",
        "bypass_state_readback": "required_before_live_promotion",
    }
    assert payload["rollback_expectation"] == {
        "required_before_live_promotion": True,
        "restore_previous_dynamics_values_after_physical_readback": True,
        "restore_previous_bypass_state_after_physical_readback": True,
        "manual_operator_rollback_required_if_promoted": True,
    }
    readback = payload["future_wing_readback_contract"]
    assert readback["required"] is True
    assert readback["required_before_live_promotion"] is True
    assert readback["target_kind"] == "channel_dynamics"
    assert readback["old_value_source"] == DYNAMICS_READ_REQUIRED_SOURCE
    assert readback["readback_targets"] == [
        "dyn_on",
        "threshold",
        "ratio",
        "attack",
        "release",
        "makeup_gain",
        "gain_reduction_meter",
        "bypass_state",
    ]
    assert readback["blocked_live_actions"] == [
        "compressor_enable_write",
        "compressor_parameter_write",
        "compressor_write",
        "makeup_gain_write",
    ]
    assert readback["requires_real_wing_proof"] is True
    assert readback["requires_operator_approval"] is True
    assert readback["requires_separate_non_fader_adr"] is True
    assert "Compressor/Dynamics evidence is offline-only" in payload[
        "operator_explanation"
    ]
    assert payload["safety_blockers"][:7] == [
        "compressor_dynamics_live_apply_quarantined",
        "unsupported_value_type:compressor",
        "not_in_supervised_fader_gain_executor",
        "real_wing_proof_required_before_live_apply",
        "non_fader_live_expansion_requires_separate_adr",
        "operator_approval_required_before_supervised_dynamics_apply",
        "hardware_readback_and_rollback_not_proven",
    ]
    if expected_blocker is not None:
        assert expected_blocker in payload["safety_blockers"]


def test_compressor_proposal_module_is_transport_free():
    source = Path("backend/compressor_dynamics_proposal.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "set_compressor(",
        "set_compressor_gain",
        "mixer_client",
    ]
    for token in forbidden:
        assert token not in source


def test_dynamic_vocal_creates_proposal_only_compressor_recommendation():
    recommendation = analyze_compressor_need(
        CompressorAnalysisInput(
            channel_id=5,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            crest_factor_db=15.0,
            lra_db=8.5,
            rms_dbfs=-23.0,
            peak_dbfs=-8.0,
            lufs_short=-21.5,
            true_peak_headroom_db=5.0,
            existing_gain_reduction_db=1.0,
            contribution_share=0.78,
            persistence_cycles=3,
            transient_density=0.35,
            loudness_stability=0.34,
            current_state_cache_freshness={
                "status": "fresh",
                "field_count": 10,
                "fresh_field_count": 10,
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "compressor_proposal"
    assert recommendation.source_family == "vocal"
    assert -50.0 <= recommendation.threshold_db <= -6.0
    assert 1.2 <= recommendation.ratio <= 8.0
    assert 2.0 <= recommendation.attack_ms <= 120.0
    assert 40.0 <= recommendation.release_ms <= 1200.0
    assert 0.0 <= recommendation.makeup_gain_db <= 3.0
    assert recommendation.makeup_gain_db <= recommendation.evidence["true_peak_headroom_db"] - 1.0
    assert recommendation.confidence >= 0.42
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_state_cache_freshness"]["fresh_field_count"] == 10
    assert recommendation.evidence["feedback_safety"]["status"] == "passed"
    assert recommendation.evidence["feedback_risk"] == 0.0
    assert recommendation.evidence["makeup_gain_headroom_reserve_db"] == 1.0
    assert recommendation.evidence["requested_makeup_gain_db"] == (
        recommendation.makeup_gain_db
    )
    assert recommendation.evidence["makeup_gain_feedback_limited"] is False
    assert recommendation.evidence["source_identity_downstream_trust"]["status"] == "legacy_unverified"
    summary = recommendation.dynamics_safety_summary()
    assert summary["status"] == "passed"
    assert summary["source_family"] == "vocal"
    assert summary["makeup_gain_db"] == recommendation.makeup_gain_db
    assert summary["requested_makeup_gain_db"] == recommendation.makeup_gain_db
    assert summary["makeup_gain_is_bounded_by_headroom"] is True
    assert summary["feedback_safety_status"] == "passed"
    assert summary["feedback_blocks_makeup_gain"] is False
    assert summary["current_dynamics_cache_freshness_status"] == "fresh"
    assert summary["current_dynamics_cache_no_live_authority"] is True
    _assert_dynamics_operator_contract(summary)
    assert summary["readback_required_before_promotion"] is True
    assert summary["rollback_required_before_promotion"] is True
    assert summary["live_write"] is False
    assert summary["queue_mutation"] is False
    _assert_top_level_denial(summary)

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "compressor_dynamics_proposal"
    _assert_top_level_denial(proposal)
    _assert_dynamics_operator_contract(proposal)
    assert proposal["safety_blockers"] == summary["safety_blockers"]
    assert "requested_change" not in proposal
    assert "unsupported_value_type:compressor" in proposal["blockers"]
    assert proposal["old_value"]["source"] == DYNAMICS_READ_REQUIRED_SOURCE
    assert proposal["rollback_value"]["source"] == DYNAMICS_READ_REQUIRED_SOURCE
    assert proposal["proposed_action"]["makeup_gain_db"] == recommendation.makeup_gain_db
    assert proposal["dynamics_safety_summary"] == summary
    assert proposal["proposed_action"]["dynamics_safety_summary"] == summary
    assert proposal["proposed_action"]["required_safety_checks"] == summary[
        "required_safety_checks"
    ]
    assert proposal["proposed_action"]["rollback_expectation"] == summary[
        "rollback_expectation"
    ]
    assert proposal["proposed_action"]["future_wing_readback_contract"] == summary[
        "future_wing_readback_contract"
    ]
    assert proposal["evidence"]["dynamics_safety_summary"] == summary
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
    assert proposal["evidence"]["old_value_source"] == DYNAMICS_READ_REQUIRED_SOURCE
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["current_state_cache_freshness_status"] == "fresh"
    assert proposal["evidence"]["crest_factor"] == recommendation.evidence[
        "crest_factor_db"
    ]
    assert proposal["evidence"]["loudness_range"] == recommendation.evidence[
        "lra_db"
    ]
    assert proposal["evidence"]["gain_reduction_estimate"] == {
        "existing_gain_reduction_db": 1.0,
        "target_gain_reduction_db": recommendation.target_gain_reduction_db,
        "status": "operator_review_required_before_live_promotion",
    }
    assert proposal["evidence"]["bounded_makeup_headroom"]["bounded"] is True
    assert proposal["evidence"]["true_peak_headroom"] == 5.0
    assert proposal["evidence"]["rollback_dynamics_readback"] == (
        DYNAMICS_READ_REQUIRED_SOURCE
    )
    assert proposal["evidence"]["bypass_state"] == (
        "readback_required_before_live_promotion"
    )
    correction = proposal["operator_correction"]
    assert correction["objective_family"] == "compressor_dynamics"
    assert correction["source_evidence"]["evidence_completeness"]["status"] == (
        "complete"
    )
    assert correction["evidence_gate"]["status"] == (
        "required_evidence_complete_operator_review_still_required"
    )
    assert correction["bounded_delta"]["delta_db"] == recommendation.makeup_gain_db
    assert correction["bounded_delta"]["parameter_deltas"]["threshold_db"] == (
        recommendation.threshold_db
    )
    assert correction["bounded_delta"]["parameter_deltas"]["ratio"] == recommendation.ratio
    assert correction["bounded_delta"]["parameter_deltas"]["makeup_gain_db"] == (
        recommendation.makeup_gain_db
    )
    assert correction["future_wing_readback_contract"]["target_kind"] == "channel_dynamics"
    assert correction["future_wing_readback_contract"][
        "required_before_live_promotion"
    ] is True
    assert correction["read_only_or_dry_run_only"] is True
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    assert "makeup_gain" in correction["headroom_feedback_true_peak_checks"][
        "required_checks"
    ]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "current_dynamics_cache_no_live_authority"
    ] is True
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_top_level_denial(source_evidence)
    assert source_evidence["dynamics_safety_summary"] == summary
    assert source_evidence["future_wing_readback_contract"] == summary[
        "future_wing_readback_contract"
    ]
    assert source_evidence["current_dynamics_cache_no_live_authority"] is True
    assert source_evidence["makeup_gain_feedback_limited"] is False


def test_flat_or_already_controlled_source_is_analysis_only():
    recommendation = analyze_compressor_need(
        CompressorAnalysisInput(
            channel_id=9,
            channel_label="Pad",
            source_role="keys",
            crest_factor_db=4.0,
            lra_db=2.0,
            rms_dbfs=-24.0,
            peak_dbfs=-20.0,
            true_peak_headroom_db=8.0,
            existing_gain_reduction_db=0.0,
            loudness_stability=0.9,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "compression_not_needed" in recommendation.blockers


def test_insufficient_headroom_blocks_compressor_makeup_proposal():
    recommendation = analyze_compressor_need(
        CompressorAnalysisInput(
            channel_id=2,
            source_role="bass",
            crest_factor_db=9.0,
            lra_db=7.0,
            rms_dbfs=-18.0,
            peak_dbfs=-2.0,
            true_peak_headroom_db=0.4,
            contribution_share=0.8,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "insufficient_true_peak_headroom_for_makeup" in recommendation.blockers
    assert recommendation.makeup_gain_db == 0.0


def test_high_feedback_risk_blocks_compressor_makeup_proposal():
    recommendation = analyze_compressor_need(
        CompressorAnalysisInput(
            channel_id=5,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            crest_factor_db=15.0,
            lra_db=8.5,
            rms_dbfs=-23.0,
            peak_dbfs=-8.0,
            true_peak_headroom_db=5.0,
            feedback_risk=0.72,
            contribution_share=0.78,
            persistence_cycles=3,
            current_state_cache_freshness={
                "status": "fresh",
                "field_count": 10,
                "fresh_field_count": 10,
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "feedback_risk_blocks_compressor_makeup" in recommendation.blockers
    assert recommendation.makeup_gain_db == 0.0
    assert recommendation.evidence["requested_makeup_gain_db"] > 0.0
    assert recommendation.evidence["makeup_gain_feedback_limited"] is True
    assert recommendation.evidence["feedback_safety"]["status"] == "blocked"
    assert recommendation.evidence["feedback_safety"]["blocks_makeup_gain"] is True
    summary = recommendation.dynamics_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["feedback_safety_status"] == "blocked"
    assert summary["feedback_blocks_makeup_gain"] is True
    assert "feedback_risk_blocks_compressor_makeup" in summary["blockers"]
    _assert_dynamics_operator_contract(
        summary,
        expected_blocker="feedback_risk_blocks_compressor_makeup",
    )
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    _assert_dynamics_operator_contract(
        proposal,
        expected_blocker="feedback_risk_blocks_compressor_makeup",
    )
    assert proposal["proposed_action"]["makeup_gain_db"] == 0.0
    assert proposal["dynamics_safety_summary"] == summary
    assert proposal["evidence"]["requested_makeup_gain_db"] > 0.0
    assert proposal["evidence"]["makeup_gain_feedback_limited"] is True


def test_stale_current_compressor_state_blocks_proposal_only_recommendation():
    recommendation = analyze_compressor_need(
        CompressorAnalysisInput(
            channel_id=5,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            crest_factor_db=15.0,
            lra_db=8.5,
            rms_dbfs=-23.0,
            peak_dbfs=-8.0,
            lufs_short=-21.5,
            true_peak_headroom_db=5.0,
            existing_gain_reduction_db=1.0,
            contribution_share=0.78,
            persistence_cycles=3,
            transient_density=0.35,
            loudness_stability=0.34,
            current_state_cache_freshness={
                "status": "stale",
                "field_count": 10,
                "fresh_field_count": 7,
                "stale_field_count": 3,
                "stale_fields": ["threshold_db", "ratio", "makeup_db"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "compressor_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "stale"
    assert recommendation.evidence["current_state_cache_freshness"]["stale_fields"] == [
        "threshold_db",
        "ratio",
        "makeup_db",
    ]
    _assert_top_level_denial(recommendation.to_operator_proposal())


def test_missing_current_compressor_state_blocks_proposal_only_recommendation():
    analysis = CompressorAnalysisInput(
        channel_id=5,
        channel_label="Lead Vox",
        source_role="lead_vocal",
        crest_factor_db=15.0,
        lra_db=8.5,
        rms_dbfs=-23.0,
        peak_dbfs=-8.0,
        lufs_short=-21.5,
        true_peak_headroom_db=5.0,
        existing_gain_reduction_db=1.0,
        contribution_share=0.78,
        persistence_cycles=3,
        transient_density=0.35,
        loudness_stability=0.34,
    )
    recommendation = analyze_compressor_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "compressor_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "not_provided"
    assert build_compressor_proposals([analysis]) == ()


def test_live_authority_current_compressor_state_blocks_proposal_only_recommendation():
    recommendation = analyze_compressor_need(
        CompressorAnalysisInput(
            channel_id=5,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            crest_factor_db=15.0,
            lra_db=8.5,
            rms_dbfs=-23.0,
            peak_dbfs=-8.0,
            lufs_short=-21.5,
            true_peak_headroom_db=5.0,
            existing_gain_reduction_db=1.0,
            contribution_share=0.78,
            persistence_cycles=3,
            transient_density=0.35,
            loudness_stability=0.34,
            current_state_cache_freshness={
                "status": "fresh",
                "field_count": 10,
                "fresh_field_count": 10,
                "read_only": False,
                "live_write": True,
                "queries_console": True,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert (
        "compressor_current_state_cache_live_authority_present"
        in recommendation.blockers
    )
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert (
        recommendation.evidence["current_dynamics_cache_no_live_authority"] is False
    )
    summary = recommendation.dynamics_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["current_dynamics_cache_no_live_authority"] is False
    assert (
        "compressor_current_state_cache_live_authority_present"
        in summary["blockers"]
    )
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert proposal["dynamics_safety_summary"] == summary
    assert (
        "compressor_current_state_cache_live_authority_present"
        in proposal["blockers"]
    )


def test_source_identity_hold_blocks_downstream_compressor_proposal():
    recommendation = analyze_compressor_need(
        CompressorAnalysisInput(
            channel_id=5,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            crest_factor_db=15.0,
            lra_db=8.5,
            rms_dbfs=-23.0,
            peak_dbfs=-8.0,
            lufs_short=-21.5,
            true_peak_headroom_db=5.0,
            existing_gain_reduction_db=1.0,
            contribution_share=0.78,
            persistence_cycles=3,
            transient_density=0.35,
            loudness_stability=0.34,
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
    assert trust["evidence"]["queue_mutation"] is False
    assert trust["evidence"]["model_mutation"] is False
    _assert_top_level_denial(recommendation.to_operator_proposal())


def test_stale_source_identity_replay_context_blocks_downstream_compressor_proposal():
    recommendation = analyze_compressor_need(
        CompressorAnalysisInput(
            channel_id=5,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            crest_factor_db=15.0,
            lra_db=8.5,
            rms_dbfs=-23.0,
            peak_dbfs=-8.0,
            lufs_short=-21.5,
            true_peak_headroom_db=5.0,
            existing_gain_reduction_db=1.0,
            contribution_share=0.78,
            persistence_cycles=3,
            transient_density=0.35,
            loudness_stability=0.34,
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
    _assert_top_level_denial(recommendation.to_operator_proposal())


def test_compressor_helpers_classify_roles_and_derive_crest_factor():
    assert classify_source_family("Lead Vocal") == "vocal"
    assert classify_source_family("Bass DI") == "bass"
    assert classify_source_family("Snare Top") == "drum"
    assert estimate_crest_factor_db(-6.0, -21.0) == 15.0


def test_build_compressor_proposals_filters_to_proposal_only_items():
    proposals = build_compressor_proposals(
        [
            CompressorAnalysisInput(
                channel_id=5,
                source_role="lead_vocal",
                crest_factor_db=15.0,
                lra_db=8.0,
                rms_dbfs=-22.0,
                peak_dbfs=-7.0,
                true_peak_headroom_db=5.0,
                contribution_share=0.7,
                persistence_cycles=3,
                current_state_cache_freshness=_fresh(),
            ),
            CompressorAnalysisInput(
                channel_id=9,
                source_role="keys",
                crest_factor_db=3.5,
                lra_db=2.0,
                rms_dbfs=-24.0,
                peak_dbfs=-20.5,
            ),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["channel"] == 5
    _assert_top_level_denial(proposals[0])


def test_build_compressor_proposals_filters_stale_state_items():
    proposals = build_compressor_proposals(
        [
            CompressorAnalysisInput(
                channel_id=5,
                source_role="lead_vocal",
                crest_factor_db=15.0,
                lra_db=8.0,
                rms_dbfs=-22.0,
                peak_dbfs=-7.0,
                true_peak_headroom_db=5.0,
                contribution_share=0.7,
                persistence_cycles=3,
                current_state_cache_freshness={"status": "stale"},
            ),
        ]
    )

    assert proposals == ()


def test_compressor_proposal_is_registered_as_goal_capability():
    method = get_processing_method("compressor_dynamics_proposal")
    main = get_pipeline_profile("main_live_assist")
    capability = get_project_goal_capability("compression")
    console = get_project_goal_capability("console_compressor_control")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_write_live is False
    assert main is not None
    assert "compressor_dynamics_proposal" in main.proposal_method_ids
    assert capability is not None
    assert capability.status == "proposal_only"
    assert "compressor_dynamics_proposal" in capability.method_ids
    assert "backend/source_identity_downstream_gate.py" in capability.support_paths
    assert console is not None
    assert console.status == "proposal_only"
    assert "backend/source_identity_downstream_gate.py" in console.support_paths
    assert contracts["compressor_dynamics_proposal"]["proposal_contract"]["value_type"] == "compressor"
    assert (
        "source_identity_downstream_trust"
        in contracts["compressor_dynamics_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "feedback_risk"
        in contracts["compressor_dynamics_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "current_dynamics_cache_no_live_authority"
        in contracts["compressor_dynamics_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "old_value_source"
        in contracts["compressor_dynamics_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
