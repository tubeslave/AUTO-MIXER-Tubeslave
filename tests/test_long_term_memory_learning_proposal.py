from pathlib import Path

from backend.long_term_memory_learning_proposal import (
    LongTermMemoryLearningInput,
    analyze_long_term_memory_learning_need,
    build_long_term_memory_learning_proposals,
    normalize_learning_action_family,
    normalize_scope_kind,
)
from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_pipeline_profile, get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability


def _complete_input(**overrides):
    data = {
        "learning_id": "learning:lead_vocal:gains",
        "scope_id": "project:lead_vocal",
        "scope_kind": "project",
        "proposed_action_family": "gain_fader",
        "operator_intent": "learn accepted lead vocal rides",
        "operator_review_enabled": True,
        "source_session_count": 2,
        "telemetry_session_count": 2,
        "verified_decision_count": 3,
        "offline_replay_count": 2,
        "accepted_decision_count": 4,
        "rejected_decision_count": 1,
        "evidence_packet_count": 3,
        "engineer_action_count": 4,
        "reference_comparison_count": 2,
        "genre_context_count": 1,
        "arrangement_context_count": 1,
        "replay_correlation_score": 0.88,
        "outcome_consistency_score": 0.9,
        "safety_regression_count": 0,
        "conflicting_outcome_count": 0,
        "recent_manual_override_count": 1,
        "latency_budget_met": True,
        "manual_review_required": True,
        "auto_apply_requested": False,
        "active_performance": False,
        "current_state_cache_freshness": {
            "status": "fresh",
            "field_count": 8,
            "fresh_field_count": 8,
            "read_only": True,
            "live_write": False,
            "queries_console": False,
        },
    }
    data.update(overrides)
    return LongTermMemoryLearningInput(**data)


def test_long_term_memory_learning_module_is_transport_free():
    source = Path("backend/long_term_memory_learning_proposal.py").read_text()

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


def test_complete_learning_evidence_creates_proposal_only_recommendation():
    recommendation = analyze_long_term_memory_learning_need(_complete_input())

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "long_term_memory_learning_proposal"
    assert recommendation.proposed_action_family == "gain_fader"
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["long_term_memory_learning"] is True
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_memory_learning_context_cache_no_live_authority"] is True
    assert recommendation.evidence["current_state_cache_freshness"]["fresh_field_count"] == 8
    assert recommendation.evidence["source_identity_downstream_trust"]["status"] == (
        "legacy_unverified"
    )

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "long_term_memory_learning_proposal"
    assert proposal["kind"] == "long_term_memory_learning"
    assert proposal["can_import_to_queue"] is False
    assert proposal["can_apply_live"] is False
    assert proposal["queue_import_allowed"] is False
    assert proposal["live_apply_allowed"] is False
    assert "requested_change" not in proposal
    assert "requested_changes" not in proposal
    assert "unsupported_value_type:long_term_memory_learning" in proposal["blockers"]
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["queue_mutation"] is False
    assert proposal["evidence"]["memory_mutation"] is False
    assert proposal["evidence"]["operator_acceptance_required_before_memory_write"] is True
    assert proposal["evidence"]["current_state_cache_freshness_status"] == "fresh"
    assert proposal["evidence"]["current_memory_learning_context_cache_no_live_authority"] is True
    correction = proposal["operator_correction"]
    assert correction["objective_family"] == "long_term_memory_learning"
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
    assert source_evidence["verified_decisions"]["count"] == 3
    assert source_evidence["offline_replay_runs"]["count"] == 2
    assert source_evidence["accepted_rejected_outcomes"] == {
        "accepted_count": 4,
        "rejected_count": 1,
        "outcome_label_count": 5,
        "minimum_outcome_label_count": 2,
    }
    assert source_evidence["safety_regressions"] == {
        "count": 0,
        "maximum_allowed": 0,
    }


def test_insufficient_evidence_blocks_learning_proposal():
    recommendation = analyze_long_term_memory_learning_need(
        _complete_input(
            verified_decision_count=0,
            offline_replay_count=0,
            accepted_decision_count=1,
            rejected_decision_count=0,
            evidence_packet_count=0,
            telemetry_session_count=0,
            replay_correlation_score=0.3,
            outcome_consistency_score=0.4,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "insufficient_verified_or_offline_evidence" in recommendation.blockers
    assert "decision_outcome_labels_missing" in recommendation.blockers
    assert "evidence_packets_missing" in recommendation.blockers
    assert "telemetry_history_missing" in recommendation.blockers
    assert "replay_correlation_low" in recommendation.blockers
    assert "outcome_consistency_low" in recommendation.blockers


def test_safety_regressions_conflicts_or_auto_apply_block_learning():
    recommendation = analyze_long_term_memory_learning_need(
        _complete_input(
            safety_regression_count=1,
            conflicting_outcome_count=3,
            recent_manual_override_count=5,
            auto_apply_requested=True,
            active_performance=True,
            latency_budget_met=False,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "safety_regression_observed" in recommendation.blockers
    assert "conflicting_outcomes_high" in recommendation.blockers
    assert "manual_override_burst_high" in recommendation.blockers
    assert "auto_apply_not_supported" in recommendation.blockers
    assert "active_performance_learning_write_blocked" in recommendation.blockers
    assert "latency_budget_missing" in recommendation.blockers
    assert recommendation.severity == "high"


def test_non_analysis_learning_requires_operator_intent_and_review():
    recommendation = analyze_long_term_memory_learning_need(
        _complete_input(
            operator_intent="",
            operator_review_enabled=False,
            manual_review_required=False,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "operator_intent_required" in recommendation.blockers
    assert "operator_review_disabled" in recommendation.blockers
    assert "manual_review_required_flag_missing" in recommendation.blockers


def test_stale_memory_learning_context_blocks_learning_proposal():
    recommendation = analyze_long_term_memory_learning_need(
        _complete_input(
            current_state_cache_freshness={
                "status": "stale",
                "field_count": 8,
                "fresh_field_count": 6,
                "stale_field_count": 2,
                "stale_fields": ["offline_replay_runs", "accepted_rejected_outcomes"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            }
        )
    )

    assert recommendation.status == "analysis_only"
    assert "long_term_memory_context_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "stale"
    assert recommendation.evidence["current_state_cache_freshness"]["stale_fields"] == [
        "offline_replay_runs",
        "accepted_rejected_outcomes",
    ]
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_live_authority_memory_learning_context_blocks_learning_proposal():
    analysis = _complete_input(
        current_state_cache_freshness={
            "status": "fresh",
            "field_count": 8,
            "fresh_field_count": 8,
            "read_only": False,
            "live_write": True,
            "queries_console": True,
        }
    )
    recommendation = analyze_long_term_memory_learning_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "memory_learning_context_cache_live_authority_present" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_memory_learning_context_cache_no_live_authority"] is False
    assert recommendation.evidence["current_state_cache_freshness"]["live_write"] is True
    assert recommendation.to_operator_proposal()["can_apply_live"] is False
    assert build_long_term_memory_learning_proposals([analysis]) == ()


def test_missing_memory_learning_context_blocks_learning_proposal():
    recommendation = analyze_long_term_memory_learning_need(
        _complete_input(current_state_cache_freshness=None)
    )

    assert recommendation.status == "analysis_only"
    assert "long_term_memory_context_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "not_provided"
    assert recommendation.evidence["current_state_cache_freshness"]["provided"] is False
    assert build_long_term_memory_learning_proposals(
        [_complete_input(current_state_cache_freshness=None)]
    ) == ()


def test_source_identity_hold_blocks_memory_learning_proposal():
    recommendation = analyze_long_term_memory_learning_need(
        _complete_input(
            source_identity_trust_state="hold_for_operator_review",
            source_identity_trust_confidence=0.55,
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
    assert trust["source_id"] == "project:lead_vocal"
    assert trust["source_type"] == "gain_fader"
    assert trust["status"] == "held_for_operator_review"
    assert trust["usable_for_downstream"] is False
    proposal = recommendation.to_operator_proposal()
    assert proposal["can_apply_live"] is False
    assert proposal["evidence"]["memory_mutation"] is False
    assert "requested_change" not in proposal


def test_stale_source_identity_replay_context_blocks_memory_learning_proposal():
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
    recommendation = analyze_long_term_memory_learning_need(analysis)

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
    assert build_long_term_memory_learning_proposals([analysis]) == ()


def test_analysis_only_learning_can_be_reviewed_without_console_intent():
    recommendation = analyze_long_term_memory_learning_need(
        _complete_input(
            proposed_action_family="analysis",
            operator_intent="",
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.proposed_action_family == "analysis_only"
    proposal = recommendation.to_operator_proposal()
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["memory_mutation"] is False


def test_helpers_normalize_scope_and_action_family():
    assert normalize_scope_kind("concert") == "show"
    assert normalize_scope_kind("input") == "channel"
    assert normalize_learning_action_family("fader") == "gain_fader"
    assert normalize_learning_action_family("scene") == "scene_preset"
    assert normalize_learning_action_family("evidence") == "decision_evidence"


def test_build_long_term_memory_learning_proposals_filters_to_proposal_only_items():
    proposals = build_long_term_memory_learning_proposals(
        [
            _complete_input(learning_id="learning:good", scope_id="project:good"),
            _complete_input(
                learning_id="learning:blocked",
                scope_id="project:blocked",
                verified_decision_count=0,
                offline_replay_count=0,
                evidence_packet_count=0,
            ),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["target"] == "project:good"
    assert proposals[0]["can_import_to_queue"] is False


def test_build_long_term_memory_learning_proposals_filters_stale_context_items():
    proposals = build_long_term_memory_learning_proposals(
        [
            _complete_input(
                learning_id="learning:stale",
                scope_id="project:stale",
                current_state_cache_freshness={"status": "stale"},
            ),
        ]
    )

    assert proposals == ()


def test_long_term_memory_learning_proposal_is_registered_as_goal_capability():
    method = get_processing_method("long_term_memory_learning_proposal")
    main = get_pipeline_profile("main_live_assist")
    shadow = get_pipeline_profile("shadow_quality")
    capability = get_project_goal_capability("long_term_memory_learning")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.domain == "memory_learning"
    assert method.can_write_live is False
    assert main is not None
    assert "long_term_memory_learning_proposal" in main.proposal_method_ids
    assert shadow is not None
    assert "long_term_memory_learning_proposal" in shadow.method_ids
    assert capability is not None
    assert capability.status == "proposal_only"
    assert capability.autonomy_mode_ceiling == "assisted"
    assert capability.current_live_write_kinds == ()
    assert "long_term_memory_learning_proposal" in capability.method_ids
    assert "backend/source_identity_downstream_gate.py" in capability.support_paths
    assert contracts["long_term_memory_learning_proposal"]["proposal_contract"]["value_type"] == (
        "long_term_memory_learning"
    )
    assert (
        "current_memory_learning_context_cache_freshness"
        in contracts["long_term_memory_learning_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "current_memory_learning_context_cache_no_live_authority"
        in contracts["long_term_memory_learning_proposal"]["proposal_contract"][
            "safety_evidence_required"
        ]
    )
    assert (
        "source_identity_downstream_trust"
        in contracts["long_term_memory_learning_proposal"]["proposal_contract"][
            "safety_evidence_required"
        ]
    )
