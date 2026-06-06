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
from backend.source_identity_calibration_memory_draft import (
    SOURCE_IDENTITY_CALIBRATION_MEMORY_DRAFT_METHOD_ID,
    build_source_identity_calibration_memory_draft,
)
from backend.source_identity_calibration_persistence_plan import (
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
        "reviewed_at": "2026-06-06T09:30:00Z",
    }


def _persistence_plan():
    candidate = _calibration_candidate()
    ledger = build_source_identity_calibration_acceptance_ledger(
        calibration_candidates=[candidate],
        acceptance_records=[_accepted_record(candidate)],
    )
    return candidate, build_source_identity_calibration_persistence_plan(
        acceptance_ledger=ledger,
    )


def test_memory_draft_requires_manual_persistence_candidates():
    draft = build_source_identity_calibration_memory_draft()

    assert draft["type"] == "source_identity_calibration_memory_draft"
    assert draft["method_id"] == SOURCE_IDENTITY_CALIBRATION_MEMORY_DRAFT_METHOD_ID
    assert draft["status"] == "memory_draft_not_ready"
    assert draft["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "operator_required": True,
        "append_only_memory_write_allowed_by_draft": False,
        "queue_mutation_allowed_by_draft": False,
        "model_mutation_allowed_by_draft": False,
        "memory_mutation_allowed_by_draft": False,
        "downstream_priority_change_allowed_by_draft": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }
    assert draft["summary"]["draft_record_count"] == 0
    assert draft["summary"]["append_only_memory_write_allowed_by_draft"] is False
    assert draft["summary"]["memory_mutation_allowed_by_draft"] is False
    assert draft["summary"]["downstream_priority_change_allowed_by_draft"] is False
    assert "manual_persistence_candidate_required_before_memory_draft" in draft["summary"][
        "blocked_by"
    ]


def test_memory_draft_tracks_unreviewed_candidates_as_not_ready():
    candidate = _calibration_candidate()
    ledger = build_source_identity_calibration_acceptance_ledger(
        calibration_candidates=[candidate],
    )
    plan = build_source_identity_calibration_persistence_plan(acceptance_ledger=ledger)
    draft = build_source_identity_calibration_memory_draft(persistence_plan=plan)

    assert draft["status"] == "memory_draft_not_ready"
    assert draft["summary"]["unreviewed_candidate_count"] == 1
    assert draft["summary"]["unreviewed_candidate_ids"] == [candidate["id"]]
    assert draft["summary"]["draft_record_count"] == 0
    assert draft["draft_records"] == []


def test_memory_draft_surfaces_append_only_records_without_writing_memory():
    candidate, plan = _persistence_plan()
    draft = build_source_identity_calibration_memory_draft(persistence_plan=plan)

    assert draft["status"] == "append_only_memory_draft_available"
    assert draft["summary"]["persistence_plan_status"] == (
        "manual_calibration_persistence_candidate_available"
    )
    assert draft["summary"]["persistence_candidate_count"] == 1
    assert draft["summary"]["draft_record_count"] == 1
    assert draft["summary"]["draft_candidate_ids"] == [candidate["id"]]
    assert draft["summary"]["draft_source_types"] == ["lead_vocal"]
    assert draft["summary"]["append_only_memory_write_allowed_by_draft"] is False
    assert draft["summary"]["memory_mutation_allowed_by_draft"] is False
    assert draft["summary"]["downstream_priority_change_allowed_by_draft"] is False
    record = draft["draft_records"][0]
    assert record["target_memory_path"] == ".ai/memory/project.md"
    assert record["append_only"] is True
    assert record["memory_write_allowed_by_draft"] is False
    assert record["model_mutation_allowed_by_draft"] is False
    assert record["downstream_priority_change_allowed_by_draft"] is False
    assert record["source_candidate_id"] == candidate["id"]
    assert record["source_type"] == "lead_vocal"
    assert record["recommended_trust_state"] == "trusted_for_downstream_priority"
    assert "separate audited append-only write" in record["draft_text"]


def test_memory_draft_rejects_unsafe_persistence_plan_summary_claims():
    _, plan = _persistence_plan()
    forged = {
        **plan,
        "summary": {
            **plan["summary"],
            "operator_review_required": False,
            "calibration_persistence_write_allowed_by_plan": True,
            "queue_mutation_allowed_by_plan": True,
            "model_mutation_allowed_by_plan": True,
            "memory_mutation_allowed_by_plan": True,
            "downstream_priority_change_allowed_by_plan": True,
            "live_authority_granted": True,
            "full_autonomous_ready": True,
        },
    }

    draft = build_source_identity_calibration_memory_draft(persistence_plan=forged)

    assert draft["status"] == "memory_draft_replacement_required"
    assert draft["summary"]["invalid_record_ids"] == [
        "persistence_plan_summary_operator_review_required_not_true",
        "persistence_plan_summary_calibration_persistence_write_allowed_by_plan_not_false",
        "persistence_plan_summary_queue_mutation_allowed_by_plan_not_false",
        "persistence_plan_summary_model_mutation_allowed_by_plan_not_false",
        "persistence_plan_summary_memory_mutation_allowed_by_plan_not_false",
        "persistence_plan_summary_downstream_priority_change_allowed_by_plan_not_false",
        "persistence_plan_summary_live_authority_granted_not_false",
        "persistence_plan_summary_full_autonomous_ready_not_false",
    ]
    assert draft["summary"]["draft_record_count"] == 0
    assert draft["draft_records"] == []


def test_memory_draft_rejects_candidate_id_or_count_drift():
    candidate, plan = _persistence_plan()
    forged = {
        **plan,
        "summary": {
            **plan["summary"],
            "persistence_candidate_count": 7,
            "persistence_candidate_ids": [f"{candidate['id']}:drift"],
        },
    }

    draft = build_source_identity_calibration_memory_draft(persistence_plan=forged)

    assert draft["status"] == "memory_draft_replacement_required"
    assert draft["summary"]["invalid_record_ids"] == [
        "persistence_plan_summary_persistence_candidate_count_mismatch",
        "persistence_plan_summary_persistence_candidate_ids_mismatch",
    ]
    assert draft["summary"]["draft_record_count"] == 0


def test_memory_draft_rejects_unsafe_candidate_authority_claims():
    _, plan = _persistence_plan()
    forged_candidates = [
        {
            **plan["persistence_candidates"][0],
            "memory_mutation_allowed_by_plan": True,
            "downstream_priority_change_allowed_by_plan": True,
        }
    ]
    forged = {
        **plan,
        "persistence_candidates": forged_candidates,
    }

    draft = build_source_identity_calibration_memory_draft(persistence_plan=forged)

    assert draft["status"] == "memory_draft_replacement_required"
    assert draft["summary"]["invalid_record_ids"] == [
        plan["persistence_candidates"][0]["candidate_id"]
    ]
    assert draft["summary"]["draft_record_count"] == 0


def test_memory_draft_is_transport_free_and_read_only():
    source = Path("backend/source_identity_calibration_memory_draft.py").read_text(
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
        "write_text",
        "open(",
    ]
    for token in forbidden:
        assert token not in source


def test_memory_draft_registry_and_goal_are_shadow_only():
    method = get_processing_method(SOURCE_IDENTITY_CALIBRATION_MEMORY_DRAFT_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "memory_learning"
    assert method.can_analyze is True
    assert method.can_propose is False
    assert method.can_write_live is False
    assert method.live_write_kinds == ()

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_DRAFT_METHOD_ID in main.method_ids
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_DRAFT_METHOD_ID not in main.proposal_method_ids
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_DRAFT_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    draft = snapshot["source_identity_calibration_memory_draft"]
    assert draft["safety"]["memory_mutation_allowed_by_draft"] is False
    assert draft["safety"]["downstream_priority_change_allowed_by_draft"] is False
    assert draft["summary"]["append_only_memory_write_allowed_by_draft"] is False

    source_row = get_project_goal_capability("identify_source_types")
    memory_row = get_project_goal_capability("long_term_memory_learning")
    assert source_row is not None
    assert memory_row is not None
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_DRAFT_METHOD_ID in source_row.method_ids
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_DRAFT_METHOD_ID in memory_row.method_ids
    assert "backend/source_identity_calibration_memory_draft.py" in source_row.support_paths
    assert "backend/source_identity_calibration_memory_draft.py" in memory_row.support_paths
    assert source_row.current_live_write_kinds == ()
    assert memory_row.current_live_write_kinds == ()
