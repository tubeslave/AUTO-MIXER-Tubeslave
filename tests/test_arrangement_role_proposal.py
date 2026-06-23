from pathlib import Path

from backend.arrangement_role_proposal import (
    ArrangementRoleAnalysisInput,
    analyze_arrangement_role_need,
    build_arrangement_role_proposals,
    infer_candidate_role_from_scores,
    normalize_arrangement_role,
    normalize_source_type,
)
from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_pipeline_profile, get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability


def _fresh_state(field_count=7):
    return {
        "status": "fresh",
        "field_count": field_count,
        "fresh_field_count": field_count,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def test_arrangement_role_module_is_transport_free():
    source = Path("backend/arrangement_role_proposal.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "apply_operator_proposal",
    ]
    for token in forbidden:
        assert token not in source


def test_lead_vocal_creates_proposal_only_arrangement_role_recommendation():
    recommendation = analyze_arrangement_role_need(
        ArrangementRoleAnalysisInput(
            source_id="ch:1",
            source_type="lead vocal",
            channel_label="Lead Vox",
            current_arrangement_role="unknown",
            section_label="verse",
            source_type_confidence=0.92,
            role_confidence=0.9,
            context_confidence=0.86,
            arrangement_context_available=True,
            inter_channel_context_available=True,
            reference_role_available=True,
            accepted_history_available=True,
            activity_persistence_cycles=4,
            competing_role_count=0,
            recent_role_switches=0,
            lead_salience_score=0.9,
            masking_impact_score=0.55,
            current_state_cache_freshness=_fresh_state(),
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "arrangement_role_proposal"
    assert recommendation.source_type == "lead_vocal"
    assert recommendation.proposed_arrangement_role == "lead"
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["decision_evidence_packet"] is True
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_arrangement_role_context_cache_no_live_authority"] is True
    assert recommendation.evidence["current_state_cache_freshness"]["fresh_field_count"] == 7

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "arrangement_role_proposal"
    assert proposal["kind"] == "arrangement_role"
    assert proposal["can_import_to_queue"] is False
    assert proposal["can_apply_live"] is False
    assert proposal["queue_import_allowed"] is False
    assert proposal["live_apply_allowed"] is False
    assert "requested_change" not in proposal
    assert "requested_changes" not in proposal
    assert "unsupported_value_type:arrangement_role" in proposal["blockers"]
    assert proposal["proposed_action"]["proposed_arrangement_role"] == "lead"
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["current_state_cache_freshness_status"] == "fresh"
    assert proposal["evidence"]["current_arrangement_role_context_cache_no_live_authority"] is True
    correction = proposal["operator_correction"]
    assert correction["objective_family"] == "arrangement_role"
    assert correction["read_only_or_dry_run_only"] is True
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    assert correction["source_evidence"]["evidence_completeness"]["status"] == "complete"
    assert correction["source_evidence"]["evidence_completeness"]["missing"] == []
    assert correction["evidence_gate"]["status"] == (
        "required_evidence_complete_operator_review_still_required"
    )
    assert correction["future_wing_readback_contract"][
        "required_before_live_promotion"
    ] is True
    source_evidence = correction["source_evidence"]["evidence"]
    for required_field in correction["source_evidence"]["required_evidence_fields"]:
        assert required_field in source_evidence
    assert source_evidence["arrangement_context"]["section_label"] == "verse"
    assert source_evidence["role_persistence"]["persistent"] is True
    assert source_evidence["role_ambiguity"] == {
        "competing_role_count": 0,
        "max_competing_role_count": 1,
        "recent_role_switches": 0,
        "max_recent_role_switches": 1,
    }


def test_ambiguous_or_switching_role_is_analysis_only():
    recommendation = analyze_arrangement_role_need(
        ArrangementRoleAnalysisInput(
            source_id="ch:9",
            source_type="guitar",
            channel_label="GTR",
            current_arrangement_role="support",
            proposed_arrangement_role="featured",
            arrangement_context_available=True,
            inter_channel_context_available=True,
            source_type_confidence=0.9,
            role_confidence=0.9,
            context_confidence=0.9,
            activity_persistence_cycles=4,
            competing_role_count=3,
            recent_role_switches=3,
            harmonic_support_score=0.7,
            masking_impact_score=0.7,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "role_ambiguity_high" in recommendation.blockers
    assert "recent_role_switches_high" in recommendation.blockers
    assert recommendation.severity == "high"
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_missing_context_or_low_confidence_blocks_role_proposal():
    recommendation = analyze_arrangement_role_need(
        ArrangementRoleAnalysisInput(
            source_id="ch:12",
            source_type="keys",
            channel_label="Keys",
            current_arrangement_role="unknown",
            proposed_arrangement_role="harmonic_support",
            arrangement_context_available=False,
            inter_channel_context_available=False,
            operator_review_enabled=False,
            source_type_confidence=0.3,
            role_confidence=0.4,
            context_confidence=0.4,
            activity_persistence_cycles=1,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "arrangement_context_missing" in recommendation.blockers
    assert "inter_channel_context_missing" in recommendation.blockers
    assert "operator_review_disabled" in recommendation.blockers
    assert "source_type_confidence_low" in recommendation.blockers
    assert "role_confidence_low" in recommendation.blockers
    assert "context_confidence_low" in recommendation.blockers
    assert "role_persistence_low" in recommendation.blockers


def test_stale_arrangement_role_context_blocks_role_proposal():
    recommendation = analyze_arrangement_role_need(
        ArrangementRoleAnalysisInput(
            source_id="ch:1",
            source_type="lead vocal",
            channel_label="Lead Vox",
            current_arrangement_role="unknown",
            section_label="verse",
            source_type_confidence=0.92,
            role_confidence=0.9,
            context_confidence=0.86,
            arrangement_context_available=True,
            inter_channel_context_available=True,
            reference_role_available=True,
            accepted_history_available=True,
            activity_persistence_cycles=4,
            competing_role_count=0,
            recent_role_switches=0,
            lead_salience_score=0.9,
            masking_impact_score=0.55,
            current_state_cache_freshness={
                "status": "stale",
                "field_count": 7,
                "fresh_field_count": 5,
                "stale_field_count": 2,
                "stale_fields": ["arrangement_context", "inter_channel_context"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "arrangement_role_context_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "stale"
    assert recommendation.evidence["current_state_cache_freshness"]["stale_fields"] == [
        "arrangement_context",
        "inter_channel_context",
    ]
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_live_authority_arrangement_role_context_blocks_role_proposal():
    analysis = ArrangementRoleAnalysisInput(
        source_id="ch:1",
        source_type="lead vocal",
        channel_label="Lead Vox",
        current_arrangement_role="unknown",
        section_label="verse",
        source_type_confidence=0.92,
        role_confidence=0.9,
        context_confidence=0.86,
        arrangement_context_available=True,
        inter_channel_context_available=True,
        reference_role_available=True,
        accepted_history_available=True,
        activity_persistence_cycles=4,
        competing_role_count=0,
        recent_role_switches=0,
        lead_salience_score=0.9,
        masking_impact_score=0.55,
        current_state_cache_freshness={
            "status": "fresh",
            "field_count": 7,
            "fresh_field_count": 7,
            "read_only": False,
            "live_write": True,
            "queries_console": True,
        },
    )
    recommendation = analyze_arrangement_role_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "arrangement_role_context_cache_live_authority_present" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_arrangement_role_context_cache_no_live_authority"] is False
    assert recommendation.evidence["current_state_cache_freshness"]["live_write"] is True
    assert recommendation.to_operator_proposal()["can_apply_live"] is False
    assert build_arrangement_role_proposals([analysis]) == ()


def test_missing_arrangement_role_context_blocks_role_proposal():
    analysis = ArrangementRoleAnalysisInput(
        source_id="ch:1",
        source_type="lead vocal",
        channel_label="Lead Vox",
        current_arrangement_role="unknown",
        section_label="verse",
        source_type_confidence=0.92,
        role_confidence=0.9,
        context_confidence=0.86,
        arrangement_context_available=True,
        inter_channel_context_available=True,
        reference_role_available=True,
        accepted_history_available=True,
        activity_persistence_cycles=4,
        competing_role_count=0,
        recent_role_switches=0,
        lead_salience_score=0.9,
        masking_impact_score=0.55,
    )
    recommendation = analyze_arrangement_role_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "arrangement_role_context_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "not_provided"
    assert recommendation.evidence["current_state_cache_freshness"]["provided"] is False
    assert build_arrangement_role_proposals([analysis]) == ()


def test_source_identity_hold_blocks_downstream_arrangement_role_proposal():
    recommendation = analyze_arrangement_role_need(
        ArrangementRoleAnalysisInput(
            source_id="ch:1",
            source_type="lead vocal",
            channel_label="Lead Vox",
            current_arrangement_role="unknown",
            section_label="verse",
            source_type_confidence=0.92,
            role_confidence=0.9,
            context_confidence=0.86,
            arrangement_context_available=True,
            inter_channel_context_available=True,
            reference_role_available=True,
            accepted_history_available=True,
            activity_persistence_cycles=4,
            competing_role_count=0,
            recent_role_switches=0,
            lead_salience_score=0.9,
            source_identity_trust_state="hold_for_operator_review",
            source_identity_trust_confidence=0.46,
            source_identity_replay_outcome_count=5,
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
    trust = recommendation.evidence["source_identity_downstream_trust"]
    assert trust["status"] == "held_for_operator_review"
    assert trust["usable_for_downstream"] is False
    assert trust["evidence"]["model_mutation"] is False
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_unchanged_role_is_analysis_only():
    recommendation = analyze_arrangement_role_need(
        ArrangementRoleAnalysisInput(
            source_id="ch:2",
            source_type="bass",
            current_arrangement_role="low_anchor",
            arrangement_context_available=True,
            inter_channel_context_available=True,
            source_type_confidence=0.9,
            role_confidence=0.9,
            context_confidence=0.9,
            activity_persistence_cycles=4,
            low_anchor_score=0.95,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "arrangement_role_change_not_needed" in recommendation.blockers


def test_arrangement_role_helpers_normalize_and_infer_roles():
    assert normalize_source_type("lead vocal") == "lead_vocal"
    assert normalize_source_type("gtr") == "guitar"
    assert normalize_arrangement_role("main") == "lead"
    assert normalize_arrangement_role("support") == "harmonic_support"
    assert infer_candidate_role_from_scores("bass", low_anchor_score=0.9) == "low_anchor"
    assert infer_candidate_role_from_scores("fx", texture_score=0.9) == "effect"


def test_build_arrangement_role_proposals_filters_to_proposal_only_items():
    proposals = build_arrangement_role_proposals(
        [
            ArrangementRoleAnalysisInput(
                source_id="ch:1",
                source_type="lead vocal",
                current_arrangement_role="unknown",
                arrangement_context_available=True,
                inter_channel_context_available=True,
                reference_role_available=True,
                accepted_history_available=True,
                source_type_confidence=0.92,
                role_confidence=0.9,
                context_confidence=0.86,
                activity_persistence_cycles=4,
                lead_salience_score=0.9,
                current_state_cache_freshness=_fresh_state(),
            ),
            ArrangementRoleAnalysisInput(
                source_id="ch:2",
                source_type="bass",
                current_arrangement_role="low_anchor",
                arrangement_context_available=True,
                inter_channel_context_available=True,
                source_type_confidence=0.9,
                role_confidence=0.9,
                context_confidence=0.9,
                activity_persistence_cycles=4,
                low_anchor_score=0.95,
            ),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["target"] == "ch:1"
    assert proposals[0]["can_import_to_queue"] is False


def test_build_arrangement_role_proposals_filters_stale_context_items():
    proposals = build_arrangement_role_proposals(
        [
            ArrangementRoleAnalysisInput(
                source_id="ch:1",
                source_type="lead vocal",
                current_arrangement_role="unknown",
                arrangement_context_available=True,
                inter_channel_context_available=True,
                reference_role_available=True,
                accepted_history_available=True,
                source_type_confidence=0.92,
                role_confidence=0.9,
                context_confidence=0.86,
                activity_persistence_cycles=4,
                lead_salience_score=0.9,
                current_state_cache_freshness={"status": "stale"},
            ),
        ]
    )

    assert proposals == ()


def test_arrangement_role_proposal_is_registered_as_goal_capability():
    method = get_processing_method("arrangement_role_proposal")
    main = get_pipeline_profile("main_live_assist")
    capability = get_project_goal_capability("identify_arrangement_roles")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.domain == "arrangement"
    assert method.can_write_live is False
    assert main is not None
    assert "arrangement_role_proposal" in main.proposal_method_ids
    assert capability is not None
    assert capability.status == "proposal_only"
    assert capability.autonomy_mode_ceiling == "assisted"
    assert "arrangement_role_proposal" in capability.method_ids
    assert contracts["arrangement_role_proposal"]["proposal_contract"]["value_type"] == "arrangement_role"
    assert (
        "current_arrangement_role_context_cache_freshness"
        in contracts["arrangement_role_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "current_arrangement_role_context_cache_no_live_authority"
        in contracts["arrangement_role_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
