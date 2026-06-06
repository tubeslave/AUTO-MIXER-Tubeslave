from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_APPLY_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer import (
    _memory_draft_result_consumer,
    _writer_dry_run_payload,
)


def _memory_writer_dry_run_result_consumer(prefix="seq-policy"):
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer(
            memory_draft_result_consumer=_memory_draft_result_consumer(prefix=prefix),
            memory_writer_dry_run_result=_writer_dry_run_payload(record_count=2),
        )
    )


def _writer_apply_payload(dry_run_consumer=None):
    dry_run_consumer = dry_run_consumer or _memory_writer_dry_run_result_consumer()
    records = dry_run_consumer["memory_writer_apply_request"]["memory_draft_payload"][
        "memory_draft_records"
    ]
    write_results = [
        {
            "draft_id": record["draft_id"],
            "target_memory_path": ".ai/memory/project.md",
            "applied": True,
            "bytes_appended": 128 + index,
        }
        for index, record in enumerate(records)
    ]
    return {
        "type": "operator_proposal_apply_sequence_policy_memory_writer",
        "status": "append_only_memory_write_applied",
        "success": True,
        "reason": None,
        "generated_at": 123.0,
        "source_memory_draft_status": "ready",
        "summary": {
            "memory_draft_record_count": len(records),
            "invalid_record_count": 0,
            "write_result_count": len(write_results),
            "append_only_memory_write_applied": True,
            "operator_ack_required": True,
            "operator_ack": True,
            "apply_requested": True,
            "target_memory_path": ".ai/memory/project.md",
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "mutates_console_state": False,
            "queries_console": False,
            "policy_mutation": False,
            "memory_file_mutation": True,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "blocked_by": [],
        },
        "safety": {
            "operator_required": True,
            "operator_ack_required": True,
            "apply_requested": True,
            "operator_ack": True,
            "append_only_memory_file_write_applied": True,
            "target_memory_path": ".ai/memory/project.md",
            "does_not_call_apply_operator_proposal": True,
            "does_not_call_manual_console_write": True,
            "allowed_live_write_kinds": ["fader", "gain"],
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "mutates_console_state": False,
            "queries_console": False,
            "policy_mutation": False,
            "memory_file_mutation": True,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "write_results": write_results,
        "blocked_results": [],
    }


def test_memory_writer_apply_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer.py"
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


def test_memory_writer_apply_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer()
    )

    assert (
        consumer["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_APPLY_RESULT_CONSUMER_METHOD_ID
    )
    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer_not_ready"
    )
    assert consumer["summary"]["memory_index_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["memory_index_handler_invoked"] is False
    assert consumer["summary"]["memory_index_created"] is False
    assert consumer["summary"]["source_memory_file_write_applied"] is False
    assert consumer["summary"]["memory_file_mutation"] is False
    assert consumer["summary"]["live_write"] is False
    assert consumer["memory_index_request"]["handler"] == (
        "index_operator_proposal_apply_sequence_policy_memory"
    )
    assert consumer["memory_index_request_specs"] == []


def test_memory_writer_apply_result_consumer_emits_index_request_for_applied_writer():
    dry_run_consumer = _memory_writer_dry_run_result_consumer()
    writer = _writer_apply_payload(dry_run_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer(
            memory_writer_dry_run_result_consumer=dry_run_consumer,
            memory_writer_apply_result=writer,
            root_path="/tmp/automixer-memory-proof",
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_index_request_available"
    )
    assert consumer["summary"]["expected_memory_write_result_count"] == 2
    assert consumer["summary"]["memory_write_result_count"] == 2
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["memory_index_request_available"] is True
    assert consumer["summary"]["memory_index_handler_invoked"] is False
    assert consumer["summary"]["memory_index_created"] is False
    assert consumer["summary"]["source_memory_file_write_applied"] is True
    assert consumer["summary"]["memory_file_write_applied_by_consumer"] is False
    assert consumer["summary"]["memory_file_mutation"] is False
    assert consumer["summary"]["runtime_memory_mutation"] is False
    assert consumer["summary"]["policy_mutation"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    request = consumer["memory_index_request"]
    assert request["handler"] == "index_operator_proposal_apply_sequence_policy_memory"
    assert request["root_path"] == "/tmp/automixer-memory-proof"
    assert request["target_memory_path"] == ".ai/memory/project.md"
    assert request["read_only_template"] is True
    assert request["memory_index_handler_invoked"] is False
    assert request["memory_index_created"] is False
    assert request["source_memory_file_write_applied"] is True
    assert request["memory_file_mutation"] is False
    assert request["runtime_memory_mutation"] is False
    assert request["policy_mutation"] is False
    assert request["live_authority_granted"] is False
    first = consumer["memory_index_request_specs"][0]
    assert first["proposal_id"] == "proposal-first"
    assert first["draft_id"] == (
        "operator-sequence-policy-memory:proposal-first:seq-policy:proposal-first"
    )
    assert first["planned_memory_index"] == 0
    assert first["target_memory_path"] == ".ai/memory/project.md"
    assert first["source_memory_file_write_applied"] is True
    assert first["memory_file_write_applied_by_consumer"] is False
    assert first["memory_file_mutation"] is False
    assert first["live_authority_granted"] is False


def test_memory_writer_apply_result_consumer_rejects_dry_run_writer_result():
    dry_run_consumer = _memory_writer_dry_run_result_consumer()
    writer = _writer_dry_run_payload(record_count=2)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer(
            memory_writer_dry_run_result_consumer=dry_run_consumer,
            memory_writer_apply_result=writer,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer_replacement_required"
    )
    assert consumer["summary"]["memory_index_request_available"] is False
    assert consumer["summary"]["source_memory_file_write_applied"] is False
    assert consumer["summary"]["memory_file_mutation"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_memory_writer_apply_consumer_writer_status_not_applied"
        in invalid
    )
    assert (
        "sequence_policy_memory_writer_apply_consumer_writer_success_not_true"
        in invalid
    )
    assert (
        "sequence_policy_memory_writer_apply_consumer_writer_summary_write_applied_not_true"
        in invalid
    )
    assert consumer["memory_index_request_specs"] == []


def test_memory_writer_apply_result_consumer_rejects_forged_applied_result():
    dry_run_consumer = _memory_writer_dry_run_result_consumer()
    writer = _writer_apply_payload(dry_run_consumer)
    forged = deepcopy(writer)
    forged["summary"]["live_authority_granted"] = True
    forged["safety"]["memory_file_mutation"] = False
    forged["write_results"][0]["draft_id"] = "forged"
    forged["write_results"][1]["target_memory_path"] = "../project.md"

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer(
            memory_writer_dry_run_result_consumer=dry_run_consumer,
            memory_writer_apply_result=forged,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer_replacement_required"
    )
    assert consumer["summary"]["memory_index_request_available"] is False
    assert consumer["summary"]["source_memory_file_write_applied"] is False
    assert consumer["summary"]["memory_file_mutation"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert (
        "sequence_policy_memory_writer_apply_consumer_writer_summary_live_authority_granted_not_false"
        in invalid
    )
    assert (
        "sequence_policy_memory_writer_apply_consumer_writer_safety_memory_file_mutation_not_true"
        in invalid
    )
    assert any(item.endswith(":write_result_draft_id_mismatch") for item in invalid)
    assert any(item.endswith(":write_result_target_path_mismatch") for item in invalid)
    assert consumer["memory_index_request_specs"] == []


def test_memory_writer_apply_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_APPLY_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_APPLY_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_mutation"] is False
    assert consumer["summary"]["memory_index_request_available"] is False


def test_memory_writer_apply_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_APPLY_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
