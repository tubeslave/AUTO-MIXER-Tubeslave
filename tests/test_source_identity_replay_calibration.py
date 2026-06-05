from pathlib import Path

from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_pipeline_profile, get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.source_identity_replay_calibration import (
    SourceIdentityReplayEvent,
    analyze_source_identity_replay_calibration,
    build_source_identity_replay_calibration_proposals,
)


def _freshness(status="fresh"):
    return {
        "status": status,
        "field_count": 6,
        "fresh_field_count": 6 if status == "fresh" else 4,
        "stale_field_count": 0 if status == "fresh" else 2,
        "stale_fields": [] if status == "fresh" else ["replay_outcomes", "accepted_labels"],
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def _event(
    event_id,
    source_type="lead_vocal",
    confidence=0.86,
    decision="accepted",
    accepted_source_type="lead_vocal",
    replay_matched=True,
    **overrides,
):
    data = {
        "event_id": event_id,
        "source_id": f"ch{event_id}",
        "predicted_source_type": source_type,
        "predicted_confidence": confidence,
        "operator_decision": decision,
        "accepted_source_type": accepted_source_type,
        "replay_matched": replay_matched,
        "downstream_family": "arrangement_role",
        "evidence_packet_id": f"packet:{event_id}",
        "session_id": "show:1",
    }
    data.update(overrides)
    return SourceIdentityReplayEvent(**data)


def test_accepted_replay_outcomes_create_downstream_trust_calibration():
    result = analyze_source_identity_replay_calibration(
        (
            _event("1"),
            _event("2", confidence=0.82),
            _event("3", confidence=0.9),
            _event("4", confidence=0.88),
        ),
        current_state_cache_freshness=_freshness(),
    )

    assert result.method_id == "source_identity_replay_calibration"
    assert result.status == "proposal_only"
    assert result.metadata["proposal_only"] is True
    assert result.metadata["live_write_allowed"] is False
    assert result.metadata["model_mutation_allowed"] is False
    assert result.metadata["memory_mutation_allowed"] is False
    assert result.metadata["current_state_cache_freshness_status"] == "fresh"

    recommendation = result.recommendations[0]
    assert recommendation.action_type == "source_identity_replay_calibration_candidate"
    assert recommendation.source_type == "lead_vocal"
    assert recommendation.recommended_trust_state == "trusted_for_downstream_priority"
    assert recommendation.blockers == ()
    assert recommendation.evidence["acceptance_rate"] == 1.0
    assert recommendation.evidence["replay_match_rate"] == 1.0
    assert recommendation.evidence["downstream_trust_requires_operator_acceptance"] is True
    assert recommendation.evidence["current_source_identity_replay_context_cache_freshness"] == "fresh"


def test_rejected_or_corrected_replay_outcomes_hold_downstream_trust():
    result = analyze_source_identity_replay_calibration(
        (
            _event("1", decision="accepted", replay_matched=True),
            _event(
                "2",
                source_type="electric_guitar",
                confidence=0.7,
                decision="corrected",
                accepted_source_type="keys",
                replay_matched=False,
            ),
            _event(
                "3",
                source_type="electric_guitar",
                confidence=0.67,
                decision="rejected",
                accepted_source_type="keys",
                replay_matched=False,
            ),
            _event(
                "4",
                source_type="electric_guitar",
                confidence=0.65,
                decision="accepted",
                accepted_source_type="electric_guitar",
                replay_matched=True,
            ),
            _event(
                "5",
                source_type="electric_guitar",
                confidence=0.62,
                decision="corrected",
                accepted_source_type="keys",
                replay_matched=False,
            ),
        ),
        current_state_cache_freshness=_freshness(),
    )

    by_source = {item.source_type: item for item in result.recommendations}
    guitar = by_source["electric_guitar"]

    assert guitar.recommended_trust_state == "hold_for_operator_review"
    assert guitar.severity == "medium"
    assert "source_identity_replay_match_low" in guitar.blockers
    assert guitar.evidence["corrected_away_count"] == 2
    assert guitar.evidence["rejected_count"] == 1
    assert guitar.evidence["acceptance_rate"] == 0.25


def test_safety_regression_blocks_downstream_source_identity_trust():
    result = analyze_source_identity_replay_calibration(
        (
            _event("1", source_type="bass", accepted_source_type="bass"),
            _event("2", source_type="bass", accepted_source_type="bass"),
            _event(
                "3",
                source_type="bass",
                accepted_source_type="bass",
                safety_regression=True,
            ),
        ),
        current_state_cache_freshness=_freshness(),
    )

    recommendation = result.recommendations[0]

    assert recommendation.recommended_trust_state == "blocked_from_downstream_trust"
    assert recommendation.severity == "high"
    assert "source_identity_safety_regression_observed" in recommendation.blockers
    assert recommendation.evidence["safety_regression_count"] == 1


def test_stale_replay_context_blocks_calibration_output():
    result = analyze_source_identity_replay_calibration(
        (_event("1"), _event("2"), _event("3")),
        current_state_cache_freshness=_freshness("stale"),
    )

    assert result.status == "analysis_only"
    assert result.recommendations == ()
    assert "source identity replay context cache is not fresh" in " ".join(result.warnings)
    assert result.metadata["current_state_cache_freshness_status"] == "stale"
    assert "source_identity_replay_context_cache_not_fresh" in result.metadata["blockers"]


def test_source_identity_replay_calibration_proposals_are_not_queueable_live_actions():
    proposals = build_source_identity_replay_calibration_proposals(
        (_event("1"), _event("2"), _event("3")),
        current_state_cache_freshness=_freshness(),
    )

    assert proposals
    proposal = proposals[0]

    assert proposal["method_id"] == "source_identity_replay_calibration"
    assert proposal["kind"] == "source_identity_calibration"
    assert proposal["can_import_to_queue"] is False
    assert proposal["can_apply_live"] is False
    assert "requested_change" not in proposal
    assert "requested_changes" not in proposal
    assert "unsupported_value_type:source_identity_calibration" in proposal["blockers"]
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["queue_mutation"] is False
    assert proposal["evidence"]["model_mutation"] is False
    assert proposal["evidence"]["memory_mutation"] is False


def test_source_identity_replay_calibration_registry_contract_and_goal_row():
    method = get_processing_method("source_identity_replay_calibration")
    main = get_pipeline_profile("main_live_assist")
    shadow = get_pipeline_profile("shadow_quality")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}
    row = get_project_goal_capability("identify_source_types")

    assert method is not None
    assert method.status == "proposal_only"
    assert method.domain == "source_identity"
    assert method.can_analyze is True
    assert method.can_propose is True
    assert method.can_write_live is False
    assert main is not None
    assert "source_identity_replay_calibration" in main.proposal_method_ids
    assert shadow is not None
    assert "source_identity_replay_calibration" in shadow.method_ids

    contract = contracts["source_identity_replay_calibration"]["proposal_contract"]
    evidence = contract["safety_evidence_required"]
    assert contract["value_type"] == "source_identity_calibration"
    assert contract["target_kind"] == "source_identity_replay_trust_state"
    assert "accepted_rejected_outcomes" in evidence
    assert "current_source_identity_replay_context_cache_freshness" in evidence
    assert "downstream_trust_state" in evidence
    assert contract["live_apply_allowed"] is False

    assert row is not None
    assert row.status == "canonical"
    assert "source_identity_replay_calibration" in row.method_ids
    assert row.current_live_write_kinds == ()
    assert "replay-calibrated" in row.next_step


def test_source_identity_replay_calibration_module_is_transport_free():
    source = Path("backend/source_identity_replay_calibration.py").read_text()

    forbidden_fragments = [
        "WingClient",
        "DLiveClient",
        "MixingStation",
        "mixer_client",
        "pythonosc",
        "python-osc",
        "set_gain",
        "set_eq",
        "apply_operator_proposal",
        ".send(",
    ]

    for fragment in forbidden_fragments:
        assert fragment not in source
