from pathlib import Path

from backend.processing_proposal_adapters import (
    adapt_source_identity_replay_calibration_proposals,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.source_identity_calibration_acceptance_ledger import (
    build_source_identity_calibration_acceptance_ledger,
)
from backend.source_identity_calibration_persistence_plan import (
    SOURCE_IDENTITY_CALIBRATION_PERSISTENCE_PLAN_METHOD_ID,
    build_source_identity_calibration_persistence_plan,
)
from backend.source_identity_replay_calibration import (
    SourceIdentityReplayEvent,
    build_source_identity_replay_calibration_proposals,
)


def _calibration_candidate():
    raw = build_source_identity_replay_calibration_proposals(
        [
            SourceIdentityReplayEvent(
                event_id=f"lead:{index}",
                source_id=f"ch{index}",
                predicted_source_type="lead_vocal",
                predicted_confidence=0.86 + index * 0.01,
                operator_decision="accepted",
                accepted_source_type="lead_vocal",
                replay_matched=True,
                downstream_family="arrangement_role",
                evidence_packet_id=f"packet:{index}",
                session_id="show:source-id",
            )
            for index in range(1, 4)
        ],
        current_state_cache_freshness={
            "status": "fresh",
            "read_only": True,
            "live_write": False,
            "queries_console": False,
        },
    )
    return adapt_source_identity_replay_calibration_proposals(raw)[0]


def _accepted_record(candidate):
    return {
        "review_id": "source-id-review-1",
        "candidate_id": candidate["id"],
        "calibration_method_id": "source_identity_replay_calibration",
        "source_type": candidate["proposed_action"]["source_type"],
        "recommended_trust_state": candidate["proposed_action"]["recommended_trust_state"],
        "decision": "accepted",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-06T08:30:00Z",
    }


def _accepted_ledger():
    candidate = _calibration_candidate()
    ledger = build_source_identity_calibration_acceptance_ledger(
        calibration_candidates=[candidate],
        acceptance_records=[_accepted_record(candidate)],
    )
    return candidate, ledger


def test_persistence_plan_requires_accepted_calibration_before_persistence():
    plan = build_source_identity_calibration_persistence_plan()

    assert plan["type"] == "source_identity_calibration_persistence_plan"
    assert plan["method_id"] == SOURCE_IDENTITY_CALIBRATION_PERSISTENCE_PLAN_METHOD_ID
    assert plan["status"] == "calibration_persistence_not_ready"
    assert plan["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "operator_required": True,
        "execution_allowed_by_plan": False,
        "calibration_persistence_write_allowed_by_plan": False,
        "queue_mutation_allowed_by_plan": False,
        "model_mutation_allowed_by_plan": False,
        "memory_mutation_allowed_by_plan": False,
        "downstream_priority_change_allowed_by_plan": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }
    assert plan["summary"]["persistence_candidate_count"] == 0
    assert plan["summary"]["calibration_persistence_write_allowed_by_plan"] is False
    assert plan["summary"]["memory_mutation_allowed_by_plan"] is False
    assert plan["summary"]["downstream_priority_change_allowed_by_plan"] is False
    assert "accepted_source_identity_calibration_required_before_persistence" in plan[
        "summary"
    ]["blocked_by"]


def test_persistence_plan_tracks_unreviewed_candidates_as_not_ready():
    candidate = _calibration_candidate()
    ledger = build_source_identity_calibration_acceptance_ledger(
        calibration_candidates=[candidate],
    )
    plan = build_source_identity_calibration_persistence_plan(acceptance_ledger=ledger)

    assert plan["status"] == "calibration_persistence_not_ready"
    assert plan["summary"]["calibration_candidate_count"] == 1
    assert plan["summary"]["unreviewed_candidate_count"] == 1
    assert plan["summary"]["unreviewed_candidate_ids"] == [candidate["id"]]
    assert plan["summary"]["persistence_candidate_count"] == 0
    assert plan["persistence_candidates"] == []


def test_persistence_plan_surfaces_accepted_records_as_manual_candidates_only():
    candidate, ledger = _accepted_ledger()
    plan = build_source_identity_calibration_persistence_plan(acceptance_ledger=ledger)

    assert plan["status"] == "manual_calibration_persistence_candidate_available"
    assert plan["summary"]["ledger_status"] == "operator_acceptance_records_available"
    assert plan["summary"]["accepted_record_count"] == 1
    assert plan["summary"]["persistence_candidate_count"] == 1
    assert plan["summary"]["persistence_candidate_ids"] == [candidate["id"]]
    assert plan["summary"]["accepted_source_types"] == ["lead_vocal"]
    assert plan["summary"]["calibration_persistence_write_allowed_by_plan"] is False
    assert plan["summary"]["queue_mutation_allowed_by_plan"] is False
    assert plan["summary"]["model_mutation_allowed_by_plan"] is False
    assert plan["summary"]["memory_mutation_allowed_by_plan"] is False
    assert plan["summary"]["downstream_priority_change_allowed_by_plan"] is False
    persistence_candidate = plan["persistence_candidates"][0]
    assert persistence_candidate["candidate_id"] == candidate["id"]
    assert persistence_candidate["manual_calibration_persistence_required"] is True
    assert persistence_candidate["calibration_persistence_write_allowed_by_plan"] is False
    assert persistence_candidate["memory_mutation_allowed_by_plan"] is False
    assert persistence_candidate["downstream_priority_change_allowed_by_plan"] is False
    assert persistence_candidate["live_authority_granted"] is False
    assert persistence_candidate["full_autonomous_ready"] is False


def test_persistence_plan_ignores_rejected_records_without_authority():
    candidate = _calibration_candidate()
    record = _accepted_record(candidate)
    record["review_id"] = "source-id-review-reject"
    record["decision"] = "rejected"
    ledger = build_source_identity_calibration_acceptance_ledger(
        calibration_candidates=[candidate],
        acceptance_records=[record],
    )
    plan = build_source_identity_calibration_persistence_plan(acceptance_ledger=ledger)

    assert plan["status"] == "calibration_persistence_not_ready"
    assert plan["summary"]["accepted_record_count"] == 0
    assert plan["summary"]["rejected_record_count"] == 1
    assert plan["summary"]["rejected_candidate_ids"] == [candidate["id"]]
    assert plan["summary"]["persistence_candidate_count"] == 0
    assert plan["persistence_candidates"] == []


def test_persistence_plan_rejects_unsafe_ledger_summary_authority_claims():
    _, ledger = _accepted_ledger()
    forged = {
        **ledger,
        "summary": {
            **ledger["summary"],
            "operator_review_required": False,
            "calibration_persistence_allowed_by_acceptance": True,
            "downstream_priority_change_allowed_by_acceptance": True,
            "queue_mutation_allowed_by_acceptance": True,
            "model_mutation_allowed_by_acceptance": True,
            "memory_mutation_allowed_by_acceptance": True,
            "live_authority_granted": True,
            "full_autonomous_ready": True,
        },
    }

    plan = build_source_identity_calibration_persistence_plan(acceptance_ledger=forged)

    assert plan["status"] == "calibration_persistence_replacement_required"
    assert plan["summary"]["invalid_record_ids"] == [
        "ledger_summary_operator_review_required_not_true",
        "ledger_summary_calibration_persistence_allowed_by_acceptance_not_false",
        "ledger_summary_downstream_priority_change_allowed_by_acceptance_not_false",
        "ledger_summary_queue_mutation_allowed_by_acceptance_not_false",
        "ledger_summary_model_mutation_allowed_by_acceptance_not_false",
        "ledger_summary_memory_mutation_allowed_by_acceptance_not_false",
        "ledger_summary_live_authority_granted_not_false",
        "ledger_summary_full_autonomous_ready_not_false",
    ]
    assert plan["summary"]["persistence_candidate_count"] == 0
    assert plan["persistence_candidates"] == []
    assert plan["summary"]["memory_mutation_allowed_by_plan"] is False
    assert plan["summary"]["downstream_priority_change_allowed_by_plan"] is False


def test_persistence_plan_rejects_summary_count_or_candidate_id_drift():
    candidate, ledger = _accepted_ledger()
    forged = {
        **ledger,
        "summary": {
            **ledger["summary"],
            "calibration_candidate_count": 7,
            "accepted_candidate_ids": [f"{candidate['id']}:drift"],
        },
    }

    plan = build_source_identity_calibration_persistence_plan(acceptance_ledger=forged)

    assert plan["status"] == "calibration_persistence_replacement_required"
    assert plan["summary"]["invalid_record_ids"] == [
        "ledger_summary_calibration_candidate_count_mismatch",
        "ledger_summary_accepted_candidate_ids_mismatch",
    ]
    assert plan["summary"]["persistence_candidate_count"] == 0


def test_persistence_plan_rejects_ambiguous_candidate_reviews():
    candidate = _calibration_candidate()
    first = _accepted_record(candidate)
    second = _accepted_record(candidate)
    second["review_id"] = "source-id-review-2"
    second["decision"] = "rejected"
    ledger = build_source_identity_calibration_acceptance_ledger(
        calibration_candidates=[candidate],
        acceptance_records=[first, second],
    )
    forged = {
        **ledger,
        "summary": {
            **ledger["summary"],
            "calibration_candidate_count": 1,
            "accepted_record_count": 1,
            "rejected_record_count": 1,
            "accepted_candidate_ids": [candidate["id"]],
            "rejected_candidate_ids": [candidate["id"]],
            "accepted_source_types": ["lead_vocal"],
            "rejected_source_types": ["lead_vocal"],
        },
    }

    plan = build_source_identity_calibration_persistence_plan(acceptance_ledger=forged)

    assert plan["status"] == "calibration_persistence_replacement_required"
    assert plan["summary"]["invalid_record_ids"] == [
        "source-id-review-1",
        "source-id-review-2",
    ]
    assert plan["summary"]["persistence_candidate_count"] == 0


def test_persistence_plan_is_transport_free():
    source = Path("backend/source_identity_calibration_persistence_plan.py").read_text(
        encoding="utf-8"
    )

    forbidden = [
        "WingClient",
        "DLiveClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "set_eq",
        "set_compressor",
        "apply_operator_proposal",
        "requested_change",
    ]
    for token in forbidden:
        assert token not in source


def test_persistence_plan_registry_and_goal_are_shadow_only():
    method = get_processing_method(SOURCE_IDENTITY_CALIBRATION_PERSISTENCE_PLAN_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "source_identity"
    assert method.can_analyze is True
    assert method.can_propose is False
    assert method.can_write_live is False
    assert method.live_write_kinds == ()

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert SOURCE_IDENTITY_CALIBRATION_PERSISTENCE_PLAN_METHOD_ID in main.method_ids
    assert SOURCE_IDENTITY_CALIBRATION_PERSISTENCE_PLAN_METHOD_ID not in (
        main.proposal_method_ids
    )
    assert SOURCE_IDENTITY_CALIBRATION_PERSISTENCE_PLAN_METHOD_ID not in (
        main.executor_method_ids
    )

    snapshot = build_processing_registry_snapshot(include_methods=False)
    plan = snapshot["source_identity_calibration_persistence_plan"]
    assert plan["safety"]["memory_mutation_allowed_by_plan"] is False
    assert plan["safety"]["downstream_priority_change_allowed_by_plan"] is False
    assert plan["summary"]["calibration_persistence_write_allowed_by_plan"] is False

    row = get_project_goal_capability("identify_source_types")
    assert row is not None
    assert SOURCE_IDENTITY_CALIBRATION_PERSISTENCE_PLAN_METHOD_ID in row.method_ids
    assert "backend/source_identity_calibration_persistence_plan.py" in row.support_paths
    assert row.current_live_write_kinds == ()
