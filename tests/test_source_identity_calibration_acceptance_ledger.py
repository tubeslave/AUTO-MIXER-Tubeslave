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
    SOURCE_IDENTITY_CALIBRATION_ACCEPTANCE_LEDGER_METHOD_ID,
    build_source_identity_calibration_acceptance_ledger,
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
        "reviewed_at": "2026-06-06T07:30:00Z",
    }


def test_acceptance_ledger_requires_candidates_before_acceptance():
    ledger = build_source_identity_calibration_acceptance_ledger()

    assert ledger["type"] == "source_identity_calibration_acceptance_ledger"
    assert ledger["method_id"] == SOURCE_IDENTITY_CALIBRATION_ACCEPTANCE_LEDGER_METHOD_ID
    assert ledger["status"] == "operator_acceptance_required"
    assert ledger["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "queue_mutation": False,
        "model_mutation": False,
        "memory_mutation": False,
        "downstream_priority_mutation": False,
        "operator_required": True,
        "calibration_persistence_allowed_by_acceptance": False,
        "downstream_priority_change_allowed_by_acceptance": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }
    assert ledger["summary"]["calibration_candidate_count"] == 0
    assert ledger["summary"]["acceptance_record_count"] == 0
    assert ledger["summary"]["calibration_persistence_allowed_by_acceptance"] is False
    assert ledger["summary"]["downstream_priority_change_allowed_by_acceptance"] is False
    assert ledger["summary"]["model_mutation_allowed_by_acceptance"] is False
    assert ledger["summary"]["memory_mutation_allowed_by_acceptance"] is False
    assert "operator_must_surface_valid_source_identity_calibration_candidates_first" in ledger[
        "summary"
    ]["blocked_by"]


def test_acceptance_ledger_tracks_unreviewed_calibration_candidates():
    candidate = _calibration_candidate()
    ledger = build_source_identity_calibration_acceptance_ledger(
        calibration_candidates=[candidate],
    )

    assert ledger["status"] == "operator_acceptance_required"
    assert ledger["summary"]["calibration_candidate_count"] == 1
    assert ledger["summary"]["unreviewed_candidate_count"] == 1
    assert ledger["summary"]["unreviewed_candidate_ids"] == [candidate["id"]]
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["downstream_priority_change_allowed_by_acceptance"] is False


def test_acceptance_ledger_records_accepted_candidate_without_mutation_authority():
    candidate = _calibration_candidate()
    ledger = build_source_identity_calibration_acceptance_ledger(
        calibration_candidates=[candidate],
        acceptance_records=[_accepted_record(candidate)],
    )

    assert ledger["status"] == "operator_acceptance_records_available"
    assert ledger["summary"]["accepted_record_count"] == 1
    assert ledger["summary"]["rejected_record_count"] == 0
    assert ledger["summary"]["invalid_record_count"] == 0
    assert ledger["summary"]["accepted_candidate_ids"] == [candidate["id"]]
    assert ledger["summary"]["accepted_source_types"] == ["lead_vocal"]
    assert ledger["summary"]["calibration_persistence_allowed_by_acceptance"] is False
    assert ledger["summary"]["downstream_priority_change_allowed_by_acceptance"] is False
    assert ledger["summary"]["queue_mutation_allowed_by_acceptance"] is False
    assert ledger["summary"]["model_mutation_allowed_by_acceptance"] is False
    assert ledger["summary"]["memory_mutation_allowed_by_acceptance"] is False
    record = ledger["records"][0]
    assert record["operator_accepted_calibration_candidate"] is True
    assert record["calibration_persistence_allowed_by_acceptance"] is False
    assert record["downstream_priority_change_allowed_by_acceptance"] is False
    assert record["live_authority_granted"] is False
    assert record["full_autonomous_ready"] is False


def test_acceptance_ledger_records_rejected_candidate_without_authority():
    candidate = _calibration_candidate()
    record = _accepted_record(candidate)
    record["review_id"] = "source-id-review-reject"
    record["decision"] = "rejected"
    ledger = build_source_identity_calibration_acceptance_ledger(
        calibration_candidates=[candidate],
        acceptance_records=[record],
    )

    assert ledger["status"] == "operator_acceptance_records_available"
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["rejected_record_count"] == 1
    assert ledger["summary"]["rejected_candidate_ids"] == [candidate["id"]]
    assert ledger["summary"]["rejected_source_types"] == ["lead_vocal"]
    assert ledger["records"][0]["operator_rejected_calibration_candidate"] is True
    assert ledger["records"][0]["memory_mutation_allowed_by_acceptance"] is False


def test_acceptance_ledger_rejects_unsafe_or_mismatched_records():
    candidate = _calibration_candidate()
    record = _accepted_record(candidate)
    record["review_id"] = "unsafe-source-id-review"
    record["recommended_trust_state"] = "blocked_from_downstream_trust"
    record["memory_mutation"] = True
    record["downstream_priority_change_allowed_by_acceptance"] = True
    ledger = build_source_identity_calibration_acceptance_ledger(
        calibration_candidates=[candidate],
        acceptance_records=[record],
    )

    assert ledger["status"] == "operator_acceptance_replacement_required"
    assert ledger["summary"]["invalid_record_count"] == 1
    assert ledger["summary"]["invalid_record_ids"] == ["unsafe-source-id-review"]
    assert ledger["records"][0]["valid"] is False
    assert "acceptance_record_recommended_trust_state_mismatch" in ledger["records"][0][
        "violations"
    ]
    assert "acceptance_record_unsafe_authority_claim" in ledger["records"][0]["violations"]
    assert ledger["records"][0]["unsafe_authority_flags"] == [
        "memory_mutation",
        "downstream_priority_change_allowed_by_acceptance",
    ]


def test_acceptance_ledger_rejects_non_utc_review_timestamp():
    candidate = _calibration_candidate()
    record = _accepted_record(candidate)
    record["review_id"] = "local-time-source-id-review"
    record["reviewed_at"] = "2026-06-06 10:30:00"
    ledger = build_source_identity_calibration_acceptance_ledger(
        calibration_candidates=[candidate],
        acceptance_records=[record],
    )

    assert ledger["status"] == "operator_acceptance_replacement_required"
    assert ledger["summary"]["invalid_record_ids"] == ["local-time-source-id-review"]
    assert ledger["records"][0]["valid"] is False
    assert "acceptance_record_reviewed_at_invalid" in ledger["records"][0]["violations"]


def test_acceptance_ledger_is_transport_free():
    source = Path("backend/source_identity_calibration_acceptance_ledger.py").read_text(
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
    ]
    for token in forbidden:
        assert token not in source


def test_acceptance_ledger_registry_and_goal_are_shadow_only():
    method = get_processing_method(SOURCE_IDENTITY_CALIBRATION_ACCEPTANCE_LEDGER_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "source_identity"
    assert method.can_analyze is True
    assert method.can_propose is False
    assert method.can_write_live is False
    assert method.live_write_kinds == ()

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert SOURCE_IDENTITY_CALIBRATION_ACCEPTANCE_LEDGER_METHOD_ID in main.method_ids
    assert SOURCE_IDENTITY_CALIBRATION_ACCEPTANCE_LEDGER_METHOD_ID not in (
        main.proposal_method_ids
    )
    assert SOURCE_IDENTITY_CALIBRATION_ACCEPTANCE_LEDGER_METHOD_ID not in (
        main.executor_method_ids
    )

    snapshot = build_processing_registry_snapshot(include_methods=False)
    ledger = snapshot["source_identity_calibration_acceptance_ledger"]
    assert ledger["safety"]["model_mutation"] is False
    assert ledger["safety"]["memory_mutation"] is False
    assert ledger["safety"]["downstream_priority_mutation"] is False
    assert ledger["summary"]["calibration_persistence_allowed_by_acceptance"] is False

    row = get_project_goal_capability("identify_source_types")
    assert row is not None
    assert SOURCE_IDENTITY_CALIBRATION_ACCEPTANCE_LEDGER_METHOD_ID in row.method_ids
    assert "backend/source_identity_calibration_acceptance_ledger.py" in row.support_paths
    assert row.current_live_write_kinds == ()
