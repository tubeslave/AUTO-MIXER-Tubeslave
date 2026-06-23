from pathlib import Path

from backend.pan_spatial_proposal import (
    HIGH_MONO_RISK,
    PAN_READ_REQUIRED_SOURCE,
    PanAnalysisInput,
    analyze_pan_need,
    build_pan_proposals,
    classify_source_family,
)
from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_pipeline_profile, get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability


def _fresh(fields=4):
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


def test_pan_proposal_module_is_transport_free():
    source = Path("backend/pan_spatial_proposal.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "set_pan",
        "set_channel_pan",
        "mixer_client",
    ]
    for token in forbidden:
        assert token not in source


def test_center_anchor_can_create_proposal_only_pan_recommendation():
    recommendation = analyze_pan_need(
        PanAnalysisInput(
            channel_id=4,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_pan_percent=-24.0,
            mono_compatibility_risk=0.15,
            contribution_share=0.9,
            role_confidence=0.95,
            persistence_cycles=3,
            current_state_cache_freshness={
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
    assert recommendation.action_type == "pan_proposal"
    assert recommendation.target_pan_percent == 0.0
    assert recommendation.delta_pan_percent > 0
    assert recommendation.confidence >= 0.45
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_state_cache_freshness"]["fresh_field_count"] == 4
    assert recommendation.evidence["current_pan_cache_no_live_authority"] is True
    assert recommendation.evidence["source_identity_downstream_trust"]["status"] == "legacy_unverified"
    summary = recommendation.pan_spatial_safety_summary()
    assert summary["status"] == "passed"
    assert summary["source_family"] == "lead_vocal"
    assert summary["decision_source"] == "center_anchor_or_low_frequency"
    assert summary["bounded_pan_delta"]["status"] == "passed"
    assert summary["mono_phase_safety_status"] == "not_required_for_center_or_noop"
    assert summary["phase_required_for_non_center_pan"] is False
    assert summary["current_pan_cache_freshness_status"] == "fresh"
    assert summary["current_pan_cache_no_live_authority"] is True
    assert summary["rollback_pan_readback"]["status"] == (
        "required_before_live_promotion"
    )
    assert summary["required_safety_checks"] == {
        "source_identity": "required_before_live_promotion",
        "current_pan_readback": "required_before_live_promotion",
        "bounded_pan_delta": "required_before_live_promotion",
        "stereo_pair_or_stage_context": "required_before_live_promotion",
        "mono_compatibility": "required_before_live_promotion",
        "phase_correlation": "required_before_live_promotion",
        "mono_phase_safety": "required_before_live_promotion",
    }
    assert summary["rollback_expectation"] == {
        "required_before_live_promotion": True,
        "restore_previous_pan_value_after_physical_readback": True,
        "restore_previous_stereo_pair_link_after_physical_readback": True,
        "manual_operator_rollback_required_if_promoted": True,
    }
    readback = summary["future_wing_readback_contract"]
    assert readback["target_kind"] == "channel_pan"
    assert readback["old_value_source"] == PAN_READ_REQUIRED_SOURCE
    assert readback["required_before_live_promotion"] is True
    assert readback["requires_real_wing_proof"] is True
    assert readback["requires_operator_approval"] is True
    assert readback["requires_separate_non_fader_adr"] is True
    assert "pan_parameter_write" in readback["blocked_live_actions"]
    assert "stereo_pair_link_write" in readback["blocked_live_actions"]
    assert "non_fader_live_expansion_requires_separate_adr" in summary[
        "safety_blockers"
    ]
    assert summary["readback_required_before_promotion"] is True
    assert summary["rollback_required_before_promotion"] is True
    assert summary["live_write"] is False
    assert summary["queue_mutation"] is False
    _assert_top_level_denial(summary)

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "pan_spatial_proposal"
    _assert_top_level_denial(proposal)
    assert "requested_change" not in proposal
    assert "unsupported_value_type:pan" in proposal["blockers"]
    assert "pan_spatial_live_apply_quarantined" in proposal["blockers"]
    assert "non_fader_live_expansion_requires_separate_adr" in proposal["blockers"]
    assert proposal["old_value"]["source"] == PAN_READ_REQUIRED_SOURCE
    assert proposal["rollback_value"]["source"] == PAN_READ_REQUIRED_SOURCE
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["old_value_source"] == PAN_READ_REQUIRED_SOURCE
    assert proposal["evidence"]["mono_phase_safety"]["status"] == "not_required_for_center_or_noop"
    assert proposal["evidence"]["current_state_cache_freshness_status"] == "fresh"
    assert proposal["evidence"]["current_pan_cache_no_live_authority"] is True
    assert proposal["pan_spatial_safety_summary"] == summary
    assert proposal["proposed_action"]["pan_spatial_safety_summary"] == summary
    assert proposal["evidence"]["pan_spatial_safety_summary"] == summary
    assert proposal["safety_blockers"] == summary["safety_blockers"]
    assert proposal["required_safety_checks"] == summary["required_safety_checks"]
    assert proposal["rollback_expectation"] == summary["rollback_expectation"]
    assert proposal["future_wing_readback_contract"] == summary[
        "future_wing_readback_contract"
    ]
    assert proposal["proposed_action"]["bounded_pan_delta"] == summary[
        "bounded_pan_delta"
    ]
    assert proposal["proposed_action"]["rollback_pan_readback"] == summary[
        "rollback_pan_readback"
    ]
    assert proposal["evidence"]["rollback_pan_readback"] == summary[
        "rollback_pan_readback"
    ]
    correction = proposal["operator_correction"]
    assert correction["objective_family"] == "pan_stereo"
    assert correction["bounded_delta"]["delta_db"] == recommendation.delta_pan_percent
    assert correction["bounded_delta"]["parameter_deltas"]["delta_pan_percent"] == (
        recommendation.delta_pan_percent
    )
    assert correction["future_wing_readback_contract"]["target_kind"] == "channel_pan"
    assert correction["future_wing_readback_contract"][
        "required_before_live_promotion"
    ] is True
    assert correction["source_evidence"]["evidence_completeness"]["status"] == (
        "complete"
    )
    assert correction["source_evidence"]["evidence_completeness"]["missing"] == []
    assert correction["evidence_gate"]["status"] == (
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
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_top_level_denial(source_evidence)
    evidence_checks = correction["headroom_feedback_true_peak_checks"][
        "evidence_checks"
    ]
    for key in (
        "source_role",
        "stage_position",
        "phase_correlation",
        "mono_phase_safety",
        "bounded_pan_delta",
        "current_pan",
        "current_pan_cache_freshness",
        "current_pan_cache_no_live_authority",
        "mono_phase_check_no_live_authority",
        "source_identity_downstream_trust",
        "old_value_source",
        "rollback_pan_readback",
    ):
        assert evidence_checks[key] == source_evidence[key]
    assert source_evidence["mono_phase_safety"]["status"] == (
        "not_required_for_center_or_noop"
    )
    assert source_evidence["bounded_pan_delta"]["status"] == "passed"
    assert source_evidence["rollback_pan_readback"]["status"] == (
        "required_before_live_promotion"
    )
    assert source_evidence["future_wing_readback_contract"] == summary[
        "future_wing_readback_contract"
    ]
    assert source_evidence["current_pan_cache_no_live_authority"] is True


def test_stereo_pair_side_can_create_spatial_pan_proposal():
    recommendation = analyze_pan_need(
        PanAnalysisInput(
            channel_id=9,
            channel_label="GTR L",
            source_role="guitar",
            current_pan_percent=0.0,
            stereo_pair_id="gtr_pair",
            stereo_pair_side="left",
            mono_compatibility_risk=0.2,
            phase_correlation=0.84,
            mono_sum_delta_db=-0.4,
            mono_phase_check_freshness={
                "status": "fresh",
                "field_count": 3,
                "fresh_field_count": 3,
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
            contribution_share=0.65,
            role_confidence=0.8,
            persistence_cycles=3,
            current_state_cache_freshness=_fresh(),
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.source_family == "guitar"
    assert recommendation.target_pan_percent < 0.0
    assert recommendation.evidence["decision_source"] == "stereo_pair_side"
    assert recommendation.evidence["stereo_pair_side"] == "left"
    assert recommendation.evidence["mono_phase_safety"]["status"] == "passed"
    assert recommendation.evidence["mono_phase_check_freshness_status"] == "fresh"
    assert recommendation.evidence["mono_phase_check_no_live_authority"] is True
    assert recommendation.evidence["phase_correlation"] == 0.84


def test_high_mono_risk_blocks_non_center_pan_proposal():
    analysis = PanAnalysisInput(
        channel_id=10,
        source_role="keys",
        current_pan_percent=0.0,
        stage_position=0.8,
        mono_compatibility_risk=0.9,
        phase_correlation=0.82,
        mono_sum_delta_db=-0.3,
        mono_phase_check_freshness=_fresh(3),
        current_state_cache_freshness=_fresh(),
        role_confidence=0.9,
        persistence_cycles=3,
    )
    recommendation = analyze_pan_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "mono_compatibility_risk_high" in recommendation.blockers
    assert recommendation.evidence["mono_phase_safety"]["status"] == "blocked"
    assert (
        recommendation.evidence["mono_phase_safety"]["max_mono_compatibility_risk"]
        == HIGH_MONO_RISK
    )
    summary = recommendation.pan_spatial_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["mono_phase_safety_status"] == "blocked"
    assert summary["mono_compatibility_risk"] == 0.9
    assert "mono_compatibility_risk_high" in summary["blockers"]
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert proposal["pan_spatial_safety_summary"] == summary
    assert build_pan_proposals([analysis]) == ()


def test_non_center_pan_requires_fresh_mono_phase_evidence():
    missing_phase = analyze_pan_need(
        PanAnalysisInput(
            channel_id=10,
            source_role="keys",
            current_pan_percent=0.0,
            stage_position=0.8,
            mono_compatibility_risk=0.1,
            role_confidence=0.9,
            persistence_cycles=3,
        )
    )
    unsafe_phase = analyze_pan_need(
        PanAnalysisInput(
            channel_id=10,
            source_role="keys",
            current_pan_percent=0.0,
            stage_position=0.8,
            mono_compatibility_risk=0.1,
            phase_correlation=-0.2,
            mono_sum_delta_db=-4.5,
            mono_phase_check_freshness={"status": "fresh"},
            role_confidence=0.9,
            persistence_cycles=3,
        )
    )

    assert missing_phase.status == "analysis_only"
    assert "mono_phase_check_missing" in missing_phase.blockers
    assert "pan_mono_phase_check_not_fresh" in missing_phase.blockers
    assert missing_phase.evidence["mono_phase_safety"]["status"] == "blocked"
    assert unsafe_phase.status == "analysis_only"
    assert "pan_phase_correlation_unsafe" in unsafe_phase.blockers
    assert "pan_mono_sum_loss_high" in unsafe_phase.blockers


def test_stale_current_pan_state_blocks_proposal_only_recommendation():
    recommendation = analyze_pan_need(
        PanAnalysisInput(
            channel_id=9,
            channel_label="GTR L",
            source_role="guitar",
            current_pan_percent=0.0,
            stereo_pair_id="gtr_pair",
            stereo_pair_side="left",
            mono_compatibility_risk=0.2,
            contribution_share=0.65,
            role_confidence=0.8,
            persistence_cycles=3,
            current_state_cache_freshness={
                "status": "stale",
                "field_count": 4,
                "fresh_field_count": 2,
                "stale_field_count": 2,
                "stale_fields": ["pan_percent", "stereo_link"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "pan_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "stale"
    assert recommendation.evidence["current_state_cache_freshness"]["stale_fields"] == [
        "pan_percent",
        "stereo_link",
    ]
    summary = recommendation.pan_spatial_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["current_pan_cache_freshness_status"] == "stale"
    assert "pan_current_state_cache_not_fresh" in summary["blockers"]
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert proposal["pan_spatial_safety_summary"] == summary


def test_live_authority_current_pan_state_blocks_proposal_only_recommendation():
    analysis = PanAnalysisInput(
        channel_id=4,
        channel_label="Lead Vox",
        source_role="lead_vocal",
        current_pan_percent=-24.0,
        mono_compatibility_risk=0.15,
        contribution_share=0.9,
        role_confidence=0.95,
        persistence_cycles=3,
        current_state_cache_freshness={
            "status": "fresh",
            "field_count": 4,
            "fresh_field_count": 4,
            "read_only": False,
            "live_write": True,
            "queries_console": True,
        },
    )
    recommendation = analyze_pan_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "pan_current_state_cache_live_authority_present" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_pan_cache_no_live_authority"] is False
    assert recommendation.evidence["current_pan_cache_freshness"]["live_write"] is True
    summary = recommendation.pan_spatial_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["current_pan_cache_no_live_authority"] is False
    assert "pan_current_state_cache_live_authority_present" in summary["blockers"]
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert build_pan_proposals([analysis]) == ()


def test_live_authority_mono_phase_context_blocks_non_center_pan_proposal():
    analysis = PanAnalysisInput(
        channel_id=9,
        channel_label="GTR L",
        source_role="guitar",
        current_pan_percent=0.0,
        stereo_pair_id="gtr_pair",
        stereo_pair_side="left",
        mono_compatibility_risk=0.2,
        phase_correlation=0.84,
        mono_sum_delta_db=-0.4,
        mono_phase_check_freshness={
            "status": "fresh",
            "field_count": 3,
            "fresh_field_count": 3,
            "read_only": False,
            "live_write": True,
            "queries_console": True,
        },
        contribution_share=0.65,
        role_confidence=0.8,
        persistence_cycles=3,
        current_state_cache_freshness=_fresh(),
    )
    recommendation = analyze_pan_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "pan_mono_phase_check_live_authority_present" in recommendation.blockers
    assert recommendation.evidence["mono_phase_check_freshness_status"] == "fresh"
    assert recommendation.evidence["mono_phase_check_no_live_authority"] is False
    assert recommendation.evidence["mono_phase_check_freshness"]["queries_console"] is True
    summary = recommendation.pan_spatial_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["mono_phase_check_no_live_authority"] is False
    assert "pan_mono_phase_check_live_authority_present" in summary["blockers"]
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert build_pan_proposals([analysis]) == ()


def test_missing_current_pan_state_blocks_proposal_only_recommendation():
    analysis = PanAnalysisInput(
        channel_id=4,
        channel_label="Lead Vox",
        source_role="lead_vocal",
        current_pan_percent=-24.0,
        mono_compatibility_risk=0.15,
        contribution_share=0.9,
        role_confidence=0.95,
        persistence_cycles=3,
    )
    recommendation = analyze_pan_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "pan_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "not_provided"
    assert build_pan_proposals([analysis]) == ()


def test_source_identity_hold_blocks_downstream_pan_proposal():
    recommendation = analyze_pan_need(
        PanAnalysisInput(
            channel_id=9,
            channel_label="GTR L",
            source_role="guitar",
            current_pan_percent=0.0,
            stereo_pair_id="gtr_pair",
            stereo_pair_side="left",
            mono_compatibility_risk=0.2,
            contribution_share=0.65,
            role_confidence=0.8,
            persistence_cycles=3,
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
    summary = recommendation.pan_spatial_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["source_identity_usable_for_downstream"] is False
    assert "source_identity_not_trusted_for_downstream" in summary["blockers"]
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert proposal["pan_spatial_safety_summary"] == summary


def test_stale_source_identity_replay_context_blocks_downstream_pan_proposal():
    recommendation = analyze_pan_need(
        PanAnalysisInput(
            channel_id=9,
            channel_label="GTR L",
            source_role="guitar",
            current_pan_percent=0.0,
            stereo_pair_id="gtr_pair",
            stereo_pair_side="left",
            mono_compatibility_risk=0.2,
            contribution_share=0.65,
            role_confidence=0.8,
            persistence_cycles=3,
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
    summary = recommendation.pan_spatial_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["source_identity_usable_for_downstream"] is False
    assert "source_identity_replay_context_not_fresh" in summary["blockers"]
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert proposal["pan_spatial_safety_summary"] == summary


def test_non_anchor_source_without_spatial_context_is_analysis_only():
    recommendation = analyze_pan_need(
        PanAnalysisInput(
            channel_id=11,
            source_role="guitar",
            current_pan_percent=0.0,
            mono_compatibility_risk=0.1,
            role_confidence=0.8,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "stage_or_stereo_pair_context_missing" in recommendation.blockers


def test_pan_helpers_classify_roles():
    assert classify_source_family("Lead Vocal") == "lead_vocal"
    assert classify_source_family("Bass DI") == "bass"
    assert classify_source_family("GTR 57") == "guitar"


def test_build_pan_proposals_filters_to_proposal_only_items():
    proposals = build_pan_proposals(
        [
            PanAnalysisInput(
                channel_id=9,
                source_role="guitar",
                stereo_pair_id="gtr_pair",
                stereo_pair_side="right",
                current_pan_percent=0.0,
                mono_compatibility_risk=0.1,
                phase_correlation=0.8,
                mono_sum_delta_db=-0.2,
                mono_phase_check_freshness=_fresh(3),
                role_confidence=0.85,
                persistence_cycles=3,
                current_state_cache_freshness=_fresh(),
            ),
            PanAnalysisInput(
                channel_id=11,
                source_role="guitar",
                current_pan_percent=0.0,
            ),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["channel"] == 9


def test_build_pan_proposals_filters_stale_state_items():
    proposals = build_pan_proposals(
        [
            PanAnalysisInput(
                channel_id=9,
                source_role="guitar",
                stereo_pair_id="gtr_pair",
                stereo_pair_side="right",
                current_pan_percent=0.0,
                mono_compatibility_risk=0.1,
                phase_correlation=0.8,
                mono_sum_delta_db=-0.2,
                mono_phase_check_freshness={"status": "fresh"},
                role_confidence=0.85,
                persistence_cycles=3,
                current_state_cache_freshness={"status": "stale"},
            ),
        ]
    )

    assert proposals == ()


def test_pan_proposal_is_registered_as_goal_capability():
    method = get_processing_method("pan_spatial_proposal")
    main = get_pipeline_profile("main_live_assist")
    capability = get_project_goal_capability("panning")
    console = get_project_goal_capability("console_pan_control")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_write_live is False
    assert main is not None
    assert "pan_spatial_proposal" in main.proposal_method_ids
    assert capability is not None
    assert capability.status == "proposal_only"
    assert "pan_spatial_proposal" in capability.method_ids
    assert console is not None
    assert console.status == "proposal_only"
    assert "backend/source_identity_downstream_gate.py" in capability.support_paths
    assert "backend/source_identity_downstream_gate.py" in console.support_paths
    assert contracts["pan_spatial_proposal"]["proposal_contract"]["value_type"] == "pan"
    assert (
        "current_pan_cache_freshness"
        in contracts["pan_spatial_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "current_pan_cache_no_live_authority"
        in contracts["pan_spatial_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "mono_phase_check_no_live_authority"
        in contracts["pan_spatial_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "source_identity_downstream_trust"
        in contracts["pan_spatial_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "mono_phase_safety"
        in contracts["pan_spatial_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "old_value_source"
        in contracts["pan_spatial_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
