from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_index import (
    build_operator_proposal_apply_sequence_policy_memory_index,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer import (
    _memory_writer_dry_run_result_consumer,
    _writer_apply_payload,
)


def _freshness(status="fresh"):
    return {
        "status": status,
        "field_count": 4,
        "fresh_field_count": 4 if status == "fresh" else 2,
        "stale_field_count": 0 if status == "fresh" else 2,
        "read_only": True,
        "live_write": False,
        "apply_attempted": False,
        "queries_console": False,
        "queue_mutation": False,
        "policy_mutation": False,
    }


def _memory_writer_apply_result_consumer():
    dry_run_consumer = _memory_writer_dry_run_result_consumer()
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer(
            memory_writer_dry_run_result_consumer=dry_run_consumer,
            memory_writer_apply_result=_writer_apply_payload(dry_run_consumer),
        )
    )


def _memory_text_from_apply_consumer(apply_consumer=None) -> str:
    apply_consumer = apply_consumer or _memory_writer_apply_result_consumer()
    lines = ["# Project Memory", "", "---", ""]
    for spec in apply_consumer["memory_index_request_specs"]:
        proposal_id = spec["proposal_id"]
        draft_id = spec["draft_id"]
        lines.extend(
            [
                "### Supervised sequence policy memory",
                "",
                f"- Draft ID: `{draft_id}`",
                f"- Proposal ID: `{proposal_id}`",
                "- Learning scope: `supervised_fader_gain_sequence_execution`",
                "- Value type: `fader`",
                "- Channel: `2`",
                "- Effective value: `-8.5`",
                f"- Idempotency key: `seq-policy:{proposal_id}`",
                "- Readback status: `verified_from_cache`",
                "- Proof status: `verified`",
                "- Runtime authority: no live, queue, model, or policy mutation",
                "",
                "Supervised fader/gain sequence execution was verified.",
                "",
            ]
        )
    return "\n".join(lines)


def _memory_index(apply_consumer=None, root=None):
    apply_consumer = apply_consumer or _memory_writer_apply_result_consumer()
    memory_text = _memory_text_from_apply_consumer(apply_consumer)
    if root is not None:
        target = Path(root) / ".ai" / "memory" / "project.md"
        target.parent.mkdir(parents=True)
        target.write_text(memory_text, encoding="utf-8")
        return build_operator_proposal_apply_sequence_policy_memory_index(root=root)
    return build_operator_proposal_apply_sequence_policy_memory_index(
        memory_text=memory_text,
    )


def test_memory_index_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer.py"
    ).read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "set_trim",
        "set_input_trim",
        "set_channel_trim",
        "set_preamp_trim",
        "set_preamp_gain",
        "set_eq",
        "set_compressor",
        "route_output",
        "load_snap",
        "save_snap",
        "reset",
        "set_insert",
        "set_fx_model",
        "set_fx_on",
        "set_fx_mix",
        "set_fx_parameter",
        "write_text",
    ]
    for token in forbidden:
        assert token not in source


def test_memory_index_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer()
    )

    assert (
        consumer["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID
    )
    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_index_result_consumer_not_ready"
    )
    assert consumer["summary"]["memory_trust_review_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["memory_trust_review_handler_invoked"] is False
    assert consumer["summary"]["memory_trust_review_created"] is False
    assert consumer["summary"]["memory_file_mutation"] is False
    assert consumer["summary"]["live_write"] is False
    assert consumer["memory_trust_review_request"]["handler"] == (
        "review_operator_proposal_apply_sequence_policy_memory_trust"
    )
    assert consumer["memory_trust_review_request_specs"] == []


def test_memory_index_result_consumer_requires_fresh_context():
    apply_consumer = _memory_writer_apply_result_consumer()

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer(
            memory_writer_apply_result_consumer=apply_consumer,
            memory_index_result=_memory_index(apply_consumer),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_index_result_consumer_not_ready"
    )
    assert consumer["summary"]["memory_trust_review_request_available"] is False
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "missing"
    assert consumer["memory_trust_review_request_specs"] == []


def test_memory_index_result_consumer_emits_trust_review_request_for_available_index(tmp_path):
    apply_consumer = _memory_writer_apply_result_consumer()
    memory_index = _memory_index(apply_consumer, root=tmp_path)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer(
            memory_writer_apply_result_consumer=apply_consumer,
            memory_index_result=memory_index,
            current_sequence_review_freshness=_freshness(),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_trust_review_request_available"
    )
    assert consumer["summary"]["expected_memory_index_record_count"] == 2
    assert consumer["summary"]["memory_index_record_count"] == 2
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["memory_trust_review_request_available"] is True
    assert consumer["summary"]["memory_trust_review_handler_invoked"] is False
    assert consumer["summary"]["memory_trust_review_created"] is False
    assert consumer["summary"]["source_memory_index_available"] is True
    assert consumer["summary"]["fresh_sequence_review_context_status"] == "fresh"
    assert consumer["summary"]["memory_file_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    request = consumer["memory_trust_review_request"]
    assert request["handler"] == "review_operator_proposal_apply_sequence_policy_memory_trust"
    assert request["read_only_template"] is True
    assert request["memory_trust_review_handler_invoked"] is False
    assert request["memory_trust_review_created"] is False
    assert request["source_memory_index_available"] is True
    assert request["memory_index"]["status"] == "append_only_sequence_memory_index_available"
    assert request["current_sequence_review_freshness"]["status"] == "fresh"
    assert request["memory_file_mutation"] is False
    assert request["runtime_memory_mutation"] is False
    assert request["policy_mutation"] is False
    first = consumer["memory_trust_review_request_specs"][0]
    assert first["proposal_id"] == "proposal-first"
    assert first["draft_id"] == (
        "operator-sequence-policy-memory:proposal-first:seq-policy:proposal-first"
    )
    assert first["planned_trust_review_index"] == 0
    assert first["source_memory_index_available"] is True
    assert first["memory_trust_review_handler_invoked"] is False
    assert first["memory_file_mutation"] is False
    assert first["live_authority_granted"] is False


def test_memory_index_result_consumer_rejects_forged_index_result(tmp_path):
    apply_consumer = _memory_writer_apply_result_consumer()
    forged = deepcopy(_memory_index(apply_consumer, root=tmp_path))
    forged["summary"]["policy_mutation"] = True
    forged["summary"]["indexed_proposal_ids"] = ["proposal-first", "forged"]
    forged["safety"]["live_authority_granted"] = True
    forged["memory_records"][0]["draft_id"] = "forged"
    forged["memory_records"][1]["usable_for_sequence_policy_review"] = False

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer(
            memory_writer_apply_result_consumer=apply_consumer,
            memory_index_result=forged,
            current_sequence_review_freshness=_freshness(),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_index_result_consumer_replacement_required"
    )
    assert consumer["summary"]["memory_trust_review_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_memory_index_result_consumer_index_safety_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "sequence_policy_memory_index_result_consumer_index_summary_policy_mutation_not_false"
        in invalid
    )
    assert (
        "sequence_policy_memory_index_result_consumer_index_proposal_ids_mismatch"
        in invalid
    )
    assert any(item.endswith(":draft_id_mismatch") for item in invalid)
    assert any(
        item.endswith(":usable_for_sequence_policy_review_not_true")
        for item in invalid
    )
    assert consumer["memory_trust_review_request_specs"] == []


def test_memory_index_result_consumer_rejects_unsafe_freshness_context(tmp_path):
    apply_consumer = _memory_writer_apply_result_consumer()
    freshness = _freshness()
    freshness["read_only"] = False
    freshness["live_write"] = True
    freshness["policy_mutation"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer(
            memory_writer_apply_result_consumer=apply_consumer,
            memory_index_result=_memory_index(apply_consumer, root=tmp_path),
            current_sequence_review_freshness=freshness,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_index_result_consumer_replacement_required"
    )
    assert consumer["summary"]["memory_trust_review_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_memory_index_result_consumer_freshness_read_only_not_true"
        in invalid
    )
    assert (
        "sequence_policy_memory_index_result_consumer_freshness_live_write_not_false"
        in invalid
    )
    assert (
        "sequence_policy_memory_index_result_consumer_freshness_policy_mutation_not_false"
        in invalid
    )


def test_memory_index_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID
    )
    main = get_pipeline_profile("main_live_assist")
    snapshot = build_processing_registry_snapshot(include_methods=False)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "decision_evidence"
    assert method.can_analyze is True
    assert method.can_propose is False
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {
        "main_live_assist",
        "shadow_quality",
        "lab_dsp",
    }
    assert main is not None
    assert method.id in main.method_ids
    assert method.id not in main.proposal_method_ids
    assert method.id not in main.executor_method_ids
    consumer = snapshot[
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["summary"]["memory_trust_review_request_available"] is False


def test_memory_index_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
