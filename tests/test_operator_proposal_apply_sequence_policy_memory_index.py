from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_index import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_INDEX_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_index,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


def _memory_text(*, effective_value: str = "-8.5", value_type: str = "fader") -> str:
    return "\n".join(
        [
            "# Project Memory",
            "",
            "---",
            "",
            "### Supervised sequence policy memory",
            "",
            (
                "- Draft ID: "
                "`operator-sequence-policy-memory:memory-write-applied:"
                "memory-write:memory-write-applied`"
            ),
            "- Proposal ID: `memory-write-applied`",
            "- Learning scope: `supervised_fader_gain_sequence_execution`",
            f"- Value type: `{value_type}`",
            "- Channel: `2`",
            f"- Effective value: `{effective_value}`",
            "- Idempotency key: `memory-write:memory-write-applied`",
            "- Readback status: `verified_from_cache`",
            "- Proof status: `verified`",
            "- Runtime authority: no live, queue, model, or policy mutation",
            "",
            "Supervised fader/gain sequence execution was verified.",
            "",
        ]
    )


def test_sequence_policy_memory_index_module_is_transport_free():
    source = Path("backend/operator_proposal_apply_sequence_policy_memory_index.py").read_text()

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


def test_sequence_policy_memory_index_is_not_ready_without_memory_source():
    index = build_operator_proposal_apply_sequence_policy_memory_index()

    assert index["type"] == "operator_proposal_apply_sequence_policy_memory_index"
    assert index["method_id"] == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_INDEX_METHOD_ID
    assert index["status"] == "sequence_memory_index_not_ready"
    assert index["safety"]["read_only"] is True
    assert index["safety"]["memory_file_read"] is False
    assert index["safety"]["live_write"] is False
    assert index["safety"]["runtime_memory_mutation"] is False
    assert index["summary"]["memory_record_count"] == 0
    assert (
        "append_only_memory_file_required_before_sequence_index"
        in index["summary"]["blocked_by"]
    )


def test_sequence_policy_memory_index_parses_valid_append_only_memory_text():
    index = build_operator_proposal_apply_sequence_policy_memory_index(
        memory_text=_memory_text(),
    )

    assert index["status"] == "append_only_sequence_memory_index_available"
    assert index["summary"]["memory_record_count"] == 1
    assert index["summary"]["invalid_record_count"] == 0
    assert index["summary"]["indexed_proposal_ids"] == ["memory-write-applied"]
    assert index["summary"]["indexed_value_types"] == ["fader"]
    assert index["summary"]["execution_allowed_by_index"] is False
    assert index["summary"]["policy_mutation"] is False
    record = index["memory_records"][0]
    assert record["proposal_id"] == "memory-write-applied"
    assert record["learning_scope"] == "supervised_fader_gain_sequence_execution"
    assert record["channel"] == 2
    assert record["effective_value"] == -8.5
    assert record["idempotency_key"] == "memory-write:memory-write-applied"
    assert record["readback_status"] == "verified_from_cache"
    assert record["proof_status"] == "verified"
    assert record["usable_for_sequence_policy_review"] is True
    assert record["execution_allowed_by_index"] is False
    assert record["runtime_memory_mutation_allowed_by_index"] is False


def test_sequence_policy_memory_index_can_read_supplied_root_without_writing(tmp_path):
    target = tmp_path / ".ai" / "memory" / "project.md"
    target.parent.mkdir(parents=True)
    target.write_text(_memory_text(), encoding="utf-8")

    index = build_operator_proposal_apply_sequence_policy_memory_index(root=tmp_path)

    assert index["status"] == "append_only_sequence_memory_index_available"
    assert index["safety"]["memory_file_read"] is True
    assert index["summary"]["target_memory_path"] == ".ai/memory/project.md"
    assert index["summary"]["memory_record_count"] == 1
    assert target.read_text(encoding="utf-8") == _memory_text()


def test_sequence_policy_memory_index_rejects_malformed_or_unsafe_memory_blocks():
    text = _memory_text(effective_value="1.5").replace(
        "- Runtime authority: no live, queue, model, or policy mutation",
        "- Runtime authority: live execution and policy mutation allowed",
    )

    index = build_operator_proposal_apply_sequence_policy_memory_index(memory_text=text)

    assert index["status"] == "sequence_memory_index_replacement_required"
    assert index["summary"]["memory_record_count"] == 0
    invalid = index["summary"]["invalid_record_ids"]
    assert (
        "operator-sequence-policy-memory:memory-write-applied:"
        "memory-write:memory-write-applied"
    ) in invalid
    assert "sequence_memory_index_record_fader_ceiling_exceeded" in invalid
    assert "sequence_memory_index_record_runtime_authority_missing" in invalid
    assert "sequence_memory_index_record_runtime_authority_unsafe" in invalid


def test_sequence_policy_memory_index_rejects_pan_outside_wing_sequence_boundary():
    index = build_operator_proposal_apply_sequence_policy_memory_index(
        memory_text=_memory_text(value_type="pan", effective_value="0.0"),
    )

    assert index["status"] == "sequence_memory_index_replacement_required"
    assert index["summary"]["memory_record_count"] == 0
    assert "sequence_memory_index_record_value_type_not_allowed" in index["summary"][
        "invalid_record_ids"
    ]


def test_sequence_policy_memory_index_rejects_unapproved_target_path(tmp_path):
    index = build_operator_proposal_apply_sequence_policy_memory_index(
        root=tmp_path,
        target_memory_path="../project.md",
    )

    assert index["status"] == "sequence_memory_index_replacement_required"
    assert index["summary"]["memory_record_count"] == 0
    assert index["summary"]["invalid_record_ids"] == [
        "sequence_memory_index_target_path_not_allowed"
    ]


def test_sequence_policy_memory_index_registry_and_goal_are_shadow_only():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_INDEX_METHOD_ID
    )

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "memory_learning"
    assert method.can_analyze is True
    assert method.can_propose is False
    assert method.can_write_live is False
    assert method.live_write_kinds == ()

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_INDEX_METHOD_ID in main.method_ids
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_INDEX_METHOD_ID
        not in main.proposal_method_ids
    )
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_INDEX_METHOD_ID
        not in main.executor_method_ids
    )

    snapshot = build_processing_registry_snapshot(include_methods=False)
    index = snapshot["operator_proposal_apply_sequence_policy_memory_index"]
    assert index["status"] == "sequence_memory_index_not_ready"
    assert index["safety"]["runtime_memory_mutation"] is False
    assert index["safety"]["policy_mutation"] is False

    memory_row = get_project_goal_capability("long_term_memory_learning")
    assert memory_row is not None
    assert OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_INDEX_METHOD_ID in memory_row.method_ids
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_index.py"
        in memory_row.support_paths
    )
    assert memory_row.current_live_write_kinds == ()
