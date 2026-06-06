from pathlib import Path

from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.source_identity_calibration_memory_index import (
    SOURCE_IDENTITY_CALIBRATION_MEMORY_INDEX_METHOD_ID,
    build_source_identity_calibration_memory_index,
)


def _memory_text(*, reviewed_at: str = "2026-06-06T11:00:00Z") -> str:
    return "\n".join(
        [
            "# Project Memory",
            "",
            "---",
            "",
            "### Source identity calibration memory: lead_vocal",
            "",
            "- Draft ID: `source_identity_memory:candidate-1`",
            "- Source candidate: `candidate-1`",
            "- Source review: `review-1`",
            "- Recommended trust state: `trusted_for_downstream_priority`",
            "- Reviewed by: `operator`",
            f"- Reviewed at: `{reviewed_at}`",
            "- Runtime authority: no model, memory-priority, queue, or live-console mutation",
            "",
            "Source identity calibration for `lead_vocal` was operator-reviewed.",
            "",
        ]
    )


def test_memory_index_module_is_transport_free():
    source = Path("backend/source_identity_calibration_memory_index.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "apply_operator_proposal",
        "requested_change",
        "write_text",
    ]
    for token in forbidden:
        assert token not in source


def test_memory_index_is_not_ready_without_memory_source():
    index = build_source_identity_calibration_memory_index()

    assert index["type"] == "source_identity_calibration_memory_index"
    assert index["method_id"] == SOURCE_IDENTITY_CALIBRATION_MEMORY_INDEX_METHOD_ID
    assert index["status"] == "memory_index_not_ready"
    assert index["safety"]["read_only"] is True
    assert index["safety"]["memory_file_read"] is False
    assert index["safety"]["runtime_memory_mutation"] is False
    assert index["safety"]["downstream_priority_mutation"] is False
    assert index["summary"]["memory_record_count"] == 0
    assert "append_only_memory_file_required_before_index" in index["summary"]["blocked_by"]


def test_memory_index_parses_valid_append_only_memory_text():
    index = build_source_identity_calibration_memory_index(memory_text=_memory_text())

    assert index["status"] == "append_only_memory_index_available"
    assert index["summary"]["memory_record_count"] == 1
    assert index["summary"]["invalid_record_count"] == 0
    assert index["summary"]["indexed_candidate_ids"] == ["candidate-1"]
    assert index["summary"]["indexed_source_types"] == ["lead_vocal"]
    assert index["summary"]["runtime_memory_mutation"] is False
    assert index["summary"]["downstream_priority_mutation"] is False
    record = index["memory_records"][0]
    assert record["draft_id"] == "source_identity_memory:candidate-1"
    assert record["source_candidate_id"] == "candidate-1"
    assert record["source_review_id"] == "review-1"
    assert record["recommended_trust_state"] == "trusted_for_downstream_priority"
    assert record["usable_for_downstream_priority"] is False
    assert record["runtime_memory_mutation_allowed_by_index"] is False
    assert record["downstream_priority_change_allowed_by_index"] is False


def test_memory_index_can_read_supplied_root_without_writing(tmp_path):
    target = tmp_path / ".ai" / "memory" / "project.md"
    target.parent.mkdir(parents=True)
    target.write_text(_memory_text(), encoding="utf-8")

    index = build_source_identity_calibration_memory_index(root=tmp_path)

    assert index["status"] == "append_only_memory_index_available"
    assert index["safety"]["memory_file_read"] is True
    assert index["summary"]["target_memory_path"] == ".ai/memory/project.md"
    assert index["summary"]["memory_record_count"] == 1
    assert target.read_text(encoding="utf-8") == _memory_text()


def test_memory_index_rejects_malformed_or_unsafe_memory_blocks():
    text = _memory_text(reviewed_at="2026-06-06T11:00:00").replace(
        "- Runtime authority: no model, memory-priority, queue, or live-console mutation",
        "- Runtime authority: model and downstream priority can change",
    )

    index = build_source_identity_calibration_memory_index(memory_text=text)

    assert index["status"] == "memory_index_replacement_required"
    assert index["summary"]["memory_record_count"] == 0
    assert "source_identity_memory:candidate-1" in index["summary"]["invalid_record_ids"]
    assert "memory_index_record_reviewed_at_invalid" in index["summary"]["invalid_record_ids"]
    assert "memory_index_record_runtime_authority_missing" in index["summary"]["invalid_record_ids"]
    assert "memory_index_record_runtime_authority_unsafe" in index["summary"]["invalid_record_ids"]


def test_memory_index_rejects_unapproved_target_path(tmp_path):
    index = build_source_identity_calibration_memory_index(
        root=tmp_path,
        target_memory_path="../project.md",
    )

    assert index["status"] == "memory_index_replacement_required"
    assert index["summary"]["memory_record_count"] == 0
    assert index["summary"]["invalid_record_ids"] == ["memory_index_target_path_not_allowed"]


def test_memory_index_registry_and_goal_are_shadow_only():
    method = get_processing_method(SOURCE_IDENTITY_CALIBRATION_MEMORY_INDEX_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "memory_learning"
    assert method.can_analyze is True
    assert method.can_propose is False
    assert method.can_write_live is False
    assert method.live_write_kinds == ()

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_INDEX_METHOD_ID in main.method_ids
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_INDEX_METHOD_ID not in main.proposal_method_ids
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_INDEX_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    index = snapshot["source_identity_calibration_memory_index"]
    assert index["status"] == "memory_index_not_ready"
    assert index["safety"]["runtime_memory_mutation"] is False
    assert index["safety"]["downstream_priority_mutation"] is False

    source_row = get_project_goal_capability("identify_source_types")
    memory_row = get_project_goal_capability("long_term_memory_learning")
    assert source_row is not None
    assert memory_row is not None
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_INDEX_METHOD_ID in source_row.method_ids
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_INDEX_METHOD_ID in memory_row.method_ids
    assert "backend/source_identity_calibration_memory_index.py" in source_row.support_paths
    assert "backend/source_identity_calibration_memory_index.py" in memory_row.support_paths
    assert source_row.current_live_write_kinds == ()
    assert memory_row.current_live_write_kinds == ()
