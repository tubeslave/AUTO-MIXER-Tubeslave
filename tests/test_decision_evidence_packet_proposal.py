from pathlib import Path

from backend.decision_evidence_packet_proposal import (
    DecisionEvidenceAnalysisInput,
    analyze_decision_evidence_packet_need,
    build_decision_evidence_packet_proposals,
    normalize_action_family,
    normalize_decision_kind,
)
from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_pipeline_profile, get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability


def _complete_input(**overrides):
    data = {
        "decision_id": "proposal:lead_vocal:fader_hold",
        "target_id": "ch:1",
        "decision_kind": "proposal",
        "proposed_action_family": "gain_fader",
        "operator_intent": "soundcheck lead vocal balance",
        "operator_review_enabled": True,
        "audio_analysis_available": True,
        "audio_analysis_confidence": 0.92,
        "psychoacoustic_model_available": True,
        "psychoacoustic_model_confidence": 0.84,
        "statistical_model_available": True,
        "statistical_model_confidence": 0.82,
        "project_history_available": True,
        "project_history_confidence": 0.78,
        "reference_comparison_available": True,
        "reference_comparison_confidence": 0.81,
        "previous_decision_outcomes_available": True,
        "previous_decision_outcomes_confidence": 0.76,
        "inter_channel_context_available": True,
        "inter_channel_context_confidence": 0.88,
        "arrangement_context_available": True,
        "arrangement_context_confidence": 0.86,
        "genre_context_available": True,
        "genre_context_confidence": 0.74,
        "current_state_readback_available": True,
        "rollback_plan_available": True,
        "true_peak_headroom_db": 5.5,
        "feedback_risk": 0.12,
        "latency_budget_ms": 120.0,
        "latency_budget_met": True,
        "decision_persistence_cycles": 4,
        "conflicting_evidence_count": 0,
        "manual_override_active": False,
        "auto_apply_requested": False,
        "active_performance": False,
        "current_state_cache_freshness": {
            "status": "fresh",
            "field_count": 9,
            "fresh_field_count": 9,
            "read_only": True,
            "live_write": False,
            "queries_console": False,
        },
    }
    data.update(overrides)
    return DecisionEvidenceAnalysisInput(**data)


def test_decision_evidence_packet_module_is_transport_free():
    source = Path("backend/decision_evidence_packet_proposal.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "apply_operator_proposal",
        "requested_change",
        "requested_changes",
    ]
    for token in forbidden:
        assert token not in source


def test_complete_evidence_packet_creates_proposal_only_recommendation():
    recommendation = analyze_decision_evidence_packet_need(_complete_input())

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "decision_evidence_packet_proposal"
    assert recommendation.proposed_action_family == "gain_fader"
    assert recommendation.evidence_family_count == 9
    assert recommendation.missing_evidence_families == ()
    assert recommendation.low_confidence_families == ()
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["decision_evidence_packet"] is True
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_state_cache_freshness"]["fresh_field_count"] == 9
    assert recommendation.evidence["source_identity_downstream_trust"]["status"] == (
        "legacy_unverified"
    )

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "decision_evidence_packet_proposal"
    assert proposal["kind"] == "decision_evidence_packet"
    assert proposal["can_import_to_queue"] is False
    assert proposal["can_apply_live"] is False
    assert "requested_change" not in proposal
    assert "requested_changes" not in proposal
    assert "unsupported_value_type:decision_evidence_packet" in proposal["blockers"]
    assert proposal["proposed_action"]["evidence_family_count"] == 9
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["queue_mutation"] is False
    assert proposal["evidence"]["rollback_required_before_promotion"] is True
    assert proposal["evidence"]["current_state_cache_freshness_status"] == "fresh"


def test_missing_critical_evidence_blocks_packet():
    recommendation = analyze_decision_evidence_packet_need(
        _complete_input(
            audio_analysis_available=False,
            inter_channel_context_available=False,
            arrangement_context_available=False,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "audio_analysis_missing" in recommendation.blockers
    assert "inter_channel_context_missing" in recommendation.blockers
    assert "arrangement_context_missing" in recommendation.blockers
    assert "insufficient_evidence_family_count" in recommendation.blockers
    assert recommendation.evidence_family_count == 6


def test_low_confidence_or_missing_review_blocks_packet():
    recommendation = analyze_decision_evidence_packet_need(
        _complete_input(
            operator_review_enabled=False,
            operator_intent="",
            audio_analysis_confidence=0.4,
            statistical_model_confidence=0.2,
            decision_persistence_cycles=1,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "operator_review_disabled" in recommendation.blockers
    assert "operator_intent_required" in recommendation.blockers
    assert "audio_analysis_confidence_low" in recommendation.blockers
    assert "statistical_model_confidence_low" in recommendation.blockers
    assert "decision_persistence_low" in recommendation.blockers


def test_low_headroom_or_feedback_blocks_console_affecting_packet():
    recommendation = analyze_decision_evidence_packet_need(
        _complete_input(
            true_peak_headroom_db=0.4,
            feedback_risk=0.8,
            proposed_action_family="master_bus",
            active_performance=True,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "true_peak_headroom_low" in recommendation.blockers
    assert "feedback_risk_high" in recommendation.blockers
    assert recommendation.severity == "high"


def test_conflicting_or_unreadbacked_packet_blocks_promotion():
    recommendation = analyze_decision_evidence_packet_need(
        _complete_input(
            current_state_readback_available=False,
            rollback_plan_available=False,
            latency_budget_met=False,
            conflicting_evidence_count=3,
            auto_apply_requested=True,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "current_state_readback_missing" in recommendation.blockers
    assert "rollback_plan_missing" in recommendation.blockers
    assert "latency_budget_missing" in recommendation.blockers
    assert "conflicting_evidence_high" in recommendation.blockers
    assert "auto_apply_not_supported" in recommendation.blockers


def test_stale_decision_evidence_context_blocks_packet():
    recommendation = analyze_decision_evidence_packet_need(
        _complete_input(
            current_state_cache_freshness={
                "status": "stale",
                "field_count": 9,
                "fresh_field_count": 7,
                "stale_field_count": 2,
                "stale_fields": ["audio_analysis", "inter_channel_context"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            }
        )
    )

    assert recommendation.status == "analysis_only"
    assert "decision_evidence_context_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "stale"
    assert recommendation.evidence["current_state_cache_freshness"]["stale_fields"] == [
        "audio_analysis",
        "inter_channel_context",
    ]
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_source_identity_hold_blocks_decision_evidence_packet():
    recommendation = analyze_decision_evidence_packet_need(
        _complete_input(
            source_identity_trust_state="hold_for_operator_review",
            source_identity_trust_confidence=0.56,
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
    assert trust["source_id"] == "ch:1"
    assert trust["source_type"] == "gain_fader"
    assert trust["status"] == "held_for_operator_review"
    assert trust["usable_for_downstream"] is False
    proposal = recommendation.to_operator_proposal()
    assert proposal["can_apply_live"] is False
    assert proposal["can_import_to_queue"] is False
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["queue_mutation"] is False
    assert "requested_change" not in proposal


def test_stale_source_identity_replay_context_blocks_decision_evidence_packet():
    analysis = _complete_input(
        source_identity_trust_state="trusted_for_downstream_priority",
        source_identity_trust_confidence=0.91,
        source_identity_replay_outcome_count=8,
        source_identity_calibration_freshness={
            "status": "stale",
            "field_count": 4,
            "fresh_field_count": 2,
            "stale_field_count": 2,
            "stale_fields": ["accepted_outcomes", "rejected_outcomes"],
            "read_only": True,
            "live_write": False,
            "queries_console": False,
        },
    )
    recommendation = analyze_decision_evidence_packet_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "source_identity_replay_context_not_fresh" in recommendation.blockers
    assert recommendation.severity == "high"
    trust = recommendation.evidence["source_identity_downstream_trust"]
    assert trust["status"] == "trusted"
    assert trust["usable_for_downstream"] is False
    assert trust["evidence"]["current_source_identity_replay_context_cache_freshness"] == "stale"
    assert trust["evidence"]["current_state_cache_freshness"]["stale_fields"] == [
        "accepted_outcomes",
        "rejected_outcomes",
    ]
    assert build_decision_evidence_packet_proposals([analysis]) == ()


def test_analysis_only_packet_does_not_require_console_rollback():
    recommendation = analyze_decision_evidence_packet_need(
        _complete_input(
            decision_kind="analysis",
            proposed_action_family="analysis",
            operator_intent="",
            current_state_readback_available=False,
            rollback_plan_available=False,
            true_peak_headroom_db=0.2,
            feedback_risk=0.9,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.proposed_action_family == "analysis_only"
    proposal = recommendation.to_operator_proposal()
    assert proposal["evidence"]["rollback_required_before_promotion"] is False


def test_helpers_normalize_decision_kind_and_action_family():
    assert normalize_decision_kind("review") == "operator_review"
    assert normalize_decision_kind("reject") == "shadow_rejection"
    assert normalize_action_family("fader") == "gain_fader"
    assert normalize_action_family("scene") == "scene_preset"
    assert normalize_action_family("analysis") == "analysis_only"


def test_build_decision_evidence_packet_proposals_filters_to_proposal_only_items():
    proposals = build_decision_evidence_packet_proposals(
        [
            _complete_input(decision_id="proposal:good", target_id="ch:1"),
            _complete_input(
                decision_id="proposal:blocked",
                target_id="ch:2",
                audio_analysis_available=False,
                inter_channel_context_available=False,
                arrangement_context_available=False,
            ),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["target"] == "ch:1"
    assert proposals[0]["can_import_to_queue"] is False


def test_build_decision_evidence_packet_proposals_filters_stale_context_items():
    proposals = build_decision_evidence_packet_proposals(
        [
            _complete_input(
                decision_id="proposal:stale",
                target_id="ch:1",
                current_state_cache_freshness={"status": "stale"},
            ),
        ]
    )

    assert proposals == ()


def test_decision_evidence_packet_proposal_is_registered_as_goal_capability():
    method = get_processing_method("decision_evidence_packet_proposal")
    main = get_pipeline_profile("main_live_assist")
    shadow = get_pipeline_profile("shadow_quality")
    capability = get_project_goal_capability("decision_basis")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.domain == "decision_evidence"
    assert method.can_write_live is False
    assert main is not None
    assert "decision_evidence_packet_proposal" in main.proposal_method_ids
    assert shadow is not None
    assert "decision_evidence_packet_proposal" in shadow.method_ids
    assert capability is not None
    assert capability.status == "proposal_only"
    assert capability.autonomy_mode_ceiling == "assisted"
    assert capability.current_live_write_kinds == ()
    assert "decision_evidence_packet_proposal" in capability.method_ids
    assert contracts["decision_evidence_packet_proposal"]["proposal_contract"]["value_type"] == (
        "decision_evidence_packet"
    )
    assert (
        "current_decision_evidence_context_cache_freshness"
        in contracts["decision_evidence_packet_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "source_identity_downstream_trust"
        in contracts["decision_evidence_packet_proposal"]["proposal_contract"][
            "safety_evidence_required"
        ]
    )
