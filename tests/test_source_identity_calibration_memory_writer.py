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
    build_source_identity_calibration_memory_draft,
)
from backend.source_identity_calibration_memory_writer import (
    SOURCE_IDENTITY_CALIBRATION_MEMORY_WRITER_METHOD_ID,
    build_source_identity_calibration_memory_writer,
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
        "reviewed_at": "2026-06-06T10:30:00Z",
    }


def _memory_draft():
    candidate = _calibration_candidate()
    ledger = build_source_identity_calibration_acceptance_ledger(
        calibration_candidates=[candidate],
        acceptance_records=[_accepted_record(candidate)],
    )
    plan = build_source_identity_calibration_persistence_plan(
        acceptance_ledger=ledger,
    )
    return candidate, build_source_identity_calibration_memory_draft(
        persistence_plan=plan,
    )


def test_memory_writer_is_not_ready_without_draft_records():
    writer = build_source_identity_calibration_memory_writer()

    assert writer["type"] == "source_identity_calibration_memory_writer"
    assert writer["method_id"] == SOURCE_IDENTITY_CALIBRATION_MEMORY_WRITER_METHOD_ID
    assert writer["status"] == "memory_write_not_ready"
    assert writer["safety"]["read_only"] is True
    assert writer["safety"]["append_only_memory_file_write_applied"] is False
    assert writer["safety"]["runtime_memory_mutation"] is False
    assert writer["safety"]["downstream_priority_mutation"] is False
    assert writer["summary"]["write_result_count"] == 0
    assert "append_only_memory_draft_required_before_write" in writer["summary"][
        "blocked_by"
    ]


def test_memory_writer_dry_run_ready_does_not_write_memory(tmp_path):
    candidate, draft = _memory_draft()
    writer = build_source_identity_calibration_memory_writer(
        memory_draft=draft,
        root=tmp_path,
    )

    assert writer["status"] == "append_only_memory_write_dry_run_ready"
    assert writer["summary"]["draft_record_count"] == 1
    assert writer["summary"]["write_result_count"] == 0
    assert writer["summary"]["append_only_memory_write_applied"] is False
    assert writer["summary"]["runtime_memory_mutation"] is False
    assert writer["summary"]["downstream_priority_mutation"] is False
    assert "operator_ack_and_apply_required_before_append_only_memory_write" in writer[
        "summary"
    ]["blocked_by"]
    assert not (tmp_path / ".ai" / "memory" / "project.md").exists()
    assert draft["draft_records"][0]["source_candidate_id"] == candidate["id"]


def test_memory_writer_requires_operator_ack_for_apply(tmp_path):
    _, draft = _memory_draft()
    writer = build_source_identity_calibration_memory_writer(
        memory_draft=draft,
        root=tmp_path,
        apply=True,
        operator_ack=False,
    )

    assert writer["status"] == "memory_write_operator_ack_required"
    assert writer["summary"]["apply_requested"] is True
    assert writer["summary"]["operator_ack"] is False
    assert writer["summary"]["write_result_count"] == 0
    assert "operator_ack_required_before_append_only_memory_write" in writer["summary"][
        "blocked_by"
    ]
    assert not (tmp_path / ".ai" / "memory" / "project.md").exists()


def test_memory_writer_applies_append_only_record_with_operator_ack(tmp_path):
    candidate, draft = _memory_draft()
    writer = build_source_identity_calibration_memory_writer(
        memory_draft=draft,
        root=tmp_path,
        apply=True,
        operator_ack=True,
    )
    target = tmp_path / ".ai" / "memory" / "project.md"
    text = target.read_text(encoding="utf-8")

    assert writer["status"] == "append_only_memory_write_applied"
    assert writer["safety"]["read_only"] is False
    assert writer["safety"]["append_only_memory_file_write_applied"] is True
    assert writer["safety"]["runtime_memory_mutation"] is False
    assert writer["safety"]["model_mutation"] is False
    assert writer["safety"]["downstream_priority_mutation"] is False
    assert writer["summary"]["write_result_count"] == 1
    assert writer["write_results"][0]["target_memory_path"] == ".ai/memory/project.md"
    assert writer["write_results"][0]["applied"] is True
    assert candidate["id"] in text
    assert "Source identity calibration memory: lead_vocal" in text
    assert "Runtime authority: no model, memory-priority, queue, or live-console mutation" in text


def test_memory_writer_rejects_unsafe_target_or_summary_drift():
    _, draft = _memory_draft()
    forged = {
        **draft,
        "summary": {
            **draft["summary"],
            "draft_record_count": 7,
            "draft_candidate_ids": ["wrong-candidate"],
        },
        "draft_records": [
            {
                **draft["draft_records"][0],
                "target_memory_path": ".ai/memory/other.md",
            }
        ],
    }

    writer = build_source_identity_calibration_memory_writer(
        memory_draft=forged,
        apply=True,
        operator_ack=True,
    )

    assert writer["status"] == "memory_write_replacement_required"
    assert writer["summary"]["invalid_record_ids"] == [
        draft["draft_records"][0]["draft_id"],
        "memory_draft_summary_draft_record_count_mismatch",
        "memory_draft_summary_draft_candidate_ids_mismatch",
    ]
    assert writer["summary"]["write_result_count"] == 0
    assert writer["write_results"] == []


def test_memory_writer_rejects_unsafe_draft_envelope_claims():
    _, draft = _memory_draft()
    forged = {
        **draft,
        "safety": {
            **draft["safety"],
            "append_only_memory_write_allowed_by_draft": True,
            "memory_mutation_allowed_by_draft": True,
            "downstream_priority_change_allowed_by_draft": True,
            "live_authority_granted": True,
        },
    }

    writer = build_source_identity_calibration_memory_writer(memory_draft=forged)

    assert writer["status"] == "memory_write_replacement_required"
    assert writer["summary"]["invalid_record_ids"] == [
        "memory_draft_envelope_append_only_memory_write_allowed_by_draft_not_false",
        "memory_draft_envelope_memory_mutation_allowed_by_draft_not_false",
        "memory_draft_envelope_downstream_priority_change_allowed_by_draft_not_false",
        "memory_draft_envelope_live_authority_granted_not_false",
    ]
    assert writer["summary"]["write_result_count"] == 0


def test_memory_writer_registry_and_goal_are_shadow_only():
    method = get_processing_method(SOURCE_IDENTITY_CALIBRATION_MEMORY_WRITER_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "memory_learning"
    assert method.can_analyze is True
    assert method.can_propose is False
    assert method.can_write_live is False
    assert method.live_write_kinds == ()

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_WRITER_METHOD_ID in main.method_ids
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_WRITER_METHOD_ID not in main.proposal_method_ids
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_WRITER_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    writer = snapshot["source_identity_calibration_memory_writer"]
    assert writer["safety"]["runtime_memory_mutation"] is False
    assert writer["safety"]["downstream_priority_mutation"] is False
    assert writer["summary"]["append_only_memory_write_applied"] is False

    source_row = get_project_goal_capability("identify_source_types")
    memory_row = get_project_goal_capability("long_term_memory_learning")
    assert source_row is not None
    assert memory_row is not None
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_WRITER_METHOD_ID in source_row.method_ids
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_WRITER_METHOD_ID in memory_row.method_ids
    assert "backend/source_identity_calibration_memory_writer.py" in source_row.support_paths
    assert "backend/source_identity_calibration_memory_writer.py" in memory_row.support_paths
    assert source_row.current_live_write_kinds == ()
    assert memory_row.current_live_write_kinds == ()
