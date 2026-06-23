from pathlib import Path

from backend.processing_proposal_adapters import (
    adapt_source_identity_replay_calibration_proposals,
    build_proposal_only_method_contracts,
)
from backend.processing_registry import get_pipeline_profile, get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.source_identity_replay_calibration import (
    SourceIdentityReplayEvent,
    analyze_source_identity_replay_calibration,
    build_source_identity_replay_calibration_proposals,
)


def _freshness(status="fresh", **overrides):
    payload = {
        "status": status,
        "field_count": 6,
        "fresh_field_count": 6 if status == "fresh" else 4,
        "stale_field_count": 0 if status == "fresh" else 2,
        "stale_fields": [] if status == "fresh" else ["replay_outcomes", "accepted_labels"],
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    payload.update(overrides)
    return payload


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
    assert recommendation.evidence["current_source_identity_replay_context_cache_no_live_authority"] is True


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


def test_live_authority_replay_context_blocks_calibration_output():
    events = (_event("1"), _event("2"), _event("3"))
    result = analyze_source_identity_replay_calibration(
        events,
        current_state_cache_freshness=_freshness(
            read_only=False,
            live_write=True,
            queries_console=True,
        ),
    )

    assert result.status == "analysis_only"
    assert result.recommendations == ()
    assert result.metadata["current_source_identity_replay_context_cache_no_live_authority"] is False
    assert "source_identity_replay_context_cache_live_authority_present" in result.metadata["blockers"]
    assert build_source_identity_replay_calibration_proposals(
        events,
        current_state_cache_freshness=_freshness(
            read_only=False,
            live_write=True,
            queries_console=True,
        ),
    ) == ()


def test_missing_replay_context_blocks_calibration_output():
    events = (_event("1"), _event("2"), _event("3"))
    result = analyze_source_identity_replay_calibration(
        events,
        current_state_cache_freshness=None,
    )

    assert result.status == "analysis_only"
    assert result.recommendations == ()
    assert result.metadata["current_state_cache_freshness_status"] == "not_provided"
    assert result.metadata["current_state_cache_freshness"]["provided"] is False
    assert "source_identity_replay_context_cache_not_fresh" in result.metadata["blockers"]
    assert build_source_identity_replay_calibration_proposals(
        events,
        current_state_cache_freshness=None,
    ) == ()


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
    correction = proposal["operator_correction"]
    assert correction["objective_family"] == "source_identity_calibration"
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
    assert source_evidence["accepted_rejected_outcomes"] == {
        "accepted_count": 3,
        "corrected_away_count": 0,
        "rejected_count": 0,
        "dismissed_count": 0,
        "outcome_count": 3,
        "minimum_outcome_count": 3,
    }
    assert source_evidence["operator_acceptance"] == {
        "accepted_count": 3,
        "acceptance_rate": 1.0,
        "operator_review_required": True,
        "required_before_downstream_trust_write": True,
    }
    assert source_evidence["downstream_trust_state"] == {
        "recommended_trust_state": "trusted_for_downstream_priority",
        "write_allowed": False,
        "requires_operator_acceptance": True,
    }

    adapted = adapt_source_identity_replay_calibration_proposals(proposals)
    assert adapted[0]["adapter_id"] == "source_identity_replay_calibration"
    assert adapted[0]["queue_action"] == "preview_only_not_queued"
    assert adapted[0]["can_import_to_queue"] is False
    assert adapted[0]["can_apply_live"] is False
    assert "requested_change" not in adapted[0]
    assert "unsupported_value_type:source_identity_calibration" in adapted[0]["blockers"]
    assert adapted[0]["evidence"]["requested_change_allowed"] is False
    assert adapted[0]["evidence"]["queue_mutation"] is False
    assert adapted[0]["evidence"]["model_mutation"] is False
    assert adapted[0]["evidence"]["memory_mutation"] is False
    assert adapted[0]["evidence"]["downstream_priority_mutation"] is False
    forged = {
        **proposal,
        "evidence": {
            **proposal["evidence"],
            "live_authority_granted": True,
            "can_apply_live": True,
            "queue_action": "queue_now",
            "readback_required": False,
            "nested_authority_claim": {
                "apply_now": True,
                "dispatch_supervised_apply": True,
            },
        },
        "proposed_action": {
            **proposal["proposed_action"],
            "queue_action": "queue_now",
            "apply_now": True,
            "dispatch_supervised_apply": True,
            "live_authority_granted": True,
        },
    }
    forged_adapted = adapt_source_identity_replay_calibration_proposals([forged])[0]
    assert forged_adapted["evidence"]["live_authority_granted"] is False
    assert forged_adapted["evidence"]["can_apply_live"] is False
    assert forged_adapted["evidence"]["queue_action"] == "not_queueable_until_promoted"
    assert forged_adapted["evidence"]["readback_required"] is True
    assert forged_adapted["evidence"]["nested_authority_claim"]["apply_now"] is False
    assert forged_adapted["evidence"]["nested_authority_claim"][
        "dispatch_supervised_apply"
    ] is False
    assert forged_adapted["proposed_action"]["queue_action"] == (
        "not_queueable_until_promoted"
    )
    assert forged_adapted["proposed_action"]["apply_now"] is False
    assert forged_adapted["proposed_action"]["dispatch_supervised_apply"] is False
    assert forged_adapted["proposed_action"]["live_authority_granted"] is False
    claims_by_path = {
        claim["path"]: claim for claim in forged_adapted["raw_input_authority_claims"]
    }
    assert claims_by_path["evidence.live_authority_granted"]["requested_value"] is True
    assert claims_by_path["evidence.queue_action"]["requested_value"] == "queue_now"
    assert claims_by_path["evidence.readback_required"]["requested_value"] is False
    assert claims_by_path["evidence.nested_authority_claim.apply_now"][
        "requested_value"
    ] is True
    assert claims_by_path["proposed_action.apply_now"]["requested_value"] is True
    assert forged_adapted["evidence"]["raw_input_authority_claims"] == forged_adapted[
        "raw_input_authority_claims"
    ]


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
    assert "current_source_identity_replay_context_cache_no_live_authority" in evidence
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
