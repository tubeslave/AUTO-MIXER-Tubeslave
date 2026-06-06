from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_DRY_RUN_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer import (
    _audit_policy_gate_result_consumer,
    _memory_draft_payload,
)


def _memory_draft_result_consumer(prefix="seq-policy"):
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer(
            audit_policy_gate_result_consumer=_audit_policy_gate_result_consumer(
                prefix=prefix
            ),
            memory_draft_result=_memory_draft_payload(prefix=prefix),
        )
    )


def _writer_dry_run_payload(*, record_count=2):
    return {
        "type": "operator_proposal_apply_sequence_policy_memory_writer",
        "status": "append_only_memory_write_dry_run_ready",
        "success": False,
        "reason": "sequence_policy_memory_write_not_applied",
        "generated_at": 123.0,
        "source_memory_draft_status": "ready",
        "summary": {
            "memory_draft_record_count": record_count,
            "invalid_record_count": 1,
            "write_result_count": 0,
            "append_only_memory_write_applied": False,
            "operator_ack_required": True,
            "operator_ack": False,
            "apply_requested": False,
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
            "memory_file_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "blocked_by": [
                "operator_ack_and_apply_required_before_append_only_memory_write"
            ],
        },
        "safety": {
            "operator_required": True,
            "operator_ack_required": True,
            "apply_requested": False,
            "operator_ack": False,
            "append_only_memory_file_write_applied": False,
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
            "memory_file_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "write_results": [],
        "blocked_results": [
            {
                "draft_id": None,
                "memory_draft_index": None,
                "status": "blocked",
                "reason": "operator_ack_and_apply_required_before_append_only_memory_write",
            }
        ],
    }


def test_memory_writer_dry_run_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer.py"
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


def test_memory_writer_dry_run_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer()
    )

    assert (
        consumer["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_DRY_RUN_RESULT_CONSUMER_METHOD_ID
    )
    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer_not_ready"
    )
    assert consumer["summary"]["memory_writer_apply_request_available"] is False
    assert consumer["summary"]["memory_writer_handler_invoked"] is False
    assert consumer["summary"]["memory_file_write_applied"] is False
    assert consumer["summary"]["live_write"] is False
    assert consumer["memory_writer_apply_request"]["memory_draft_payload"] == {}
    assert consumer["memory_writer_apply_request_specs"] == []


def test_memory_writer_dry_run_result_consumer_emits_apply_request_for_dry_run_ready():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer(
            memory_draft_result_consumer=_memory_draft_result_consumer(),
            memory_writer_dry_run_result=_writer_dry_run_payload(),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_writer_apply_request_available"
    )
    assert consumer["summary"]["expected_memory_writer_dry_run_count"] == 2
    assert consumer["summary"]["memory_writer_dry_run_record_count"] == 2
    assert consumer["summary"]["dry_run_blocker_count"] == 1
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["memory_writer_apply_request_available"] is True
    assert consumer["summary"]["memory_writer_handler_invoked"] is False
    assert consumer["summary"]["memory_file_write_applied"] is False
    assert consumer["summary"]["memory_write_allowed_by_consumer"] is False
    assert consumer["summary"]["operator_ack_required_before_write"] is True
    assert consumer["summary"]["apply_required_before_write"] is True
    assert consumer["summary"]["live_write"] is False
    request = consumer["memory_writer_apply_request"]
    assert request["handler"] == "write_operator_proposal_apply_sequence_policy_memory"
    assert request["apply"] is False
    assert request["operator_ack"] is False
    assert request["operator_must_set_apply_true"] is True
    assert request["operator_must_set_operator_ack_true"] is True
    assert len(request["memory_draft_payload"]["memory_draft_records"]) == 2
    first = consumer["memory_writer_apply_request_specs"][0]
    assert first["proposal_id"] == "proposal-first"
    assert first["planned_memory_apply_index"] == 0
    assert first["target_memory_path"] == ".ai/memory/project.md"
    assert first["memory_writer_handler_invoked"] is False
    assert first["memory_file_write_applied"] is False
    assert first["memory_write_allowed_by_consumer"] is False
    assert first["live_authority_granted"] is False


def test_memory_writer_dry_run_result_consumer_rejects_applied_writer_result():
    writer = _writer_dry_run_payload()
    writer["status"] = "append_only_memory_write_applied"
    writer["success"] = True
    writer["summary"]["append_only_memory_write_applied"] = True
    writer["summary"]["memory_file_mutation"] = True
    writer["summary"]["invalid_record_count"] = 0
    writer["summary"]["write_result_count"] = 1
    writer["summary"]["blocked_by"] = []
    writer["safety"]["append_only_memory_file_write_applied"] = True
    writer["safety"]["memory_file_mutation"] = True
    writer["write_results"] = [
        {
            "draft_id": "operator-sequence-policy-memory:proposal-first:seq-policy:proposal-first",
            "target_memory_path": ".ai/memory/project.md",
            "applied": True,
            "bytes_appended": 128,
        }
    ]
    writer["blocked_results"] = []

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer(
            memory_draft_result_consumer=_memory_draft_result_consumer(),
            memory_writer_dry_run_result=writer,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer_replacement_required"
    )
    assert consumer["summary"]["memory_writer_apply_request_available"] is False
    assert (
        "sequence_policy_memory_writer_dry_run_consumer_writer_status_not_dry_run_ready"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "sequence_policy_memory_writer_dry_run_consumer_writer_has_write_results"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert consumer["memory_writer_apply_request"]["memory_draft_payload"] == {}
    assert consumer["summary"]["memory_file_write_applied"] is False


def test_memory_writer_dry_run_result_consumer_rejects_forged_dry_run_result():
    writer = _writer_dry_run_payload()
    writer["summary"]["live_authority_granted"] = True
    writer["summary"]["apply_requested"] = True
    writer["safety"]["allowed_live_write_kinds"] = ["fader", "gain", "eq"]
    writer["blocked_results"][0]["reason"] = "forged_blocker"

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer(
            memory_draft_result_consumer=_memory_draft_result_consumer(),
            memory_writer_dry_run_result=writer,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer_replacement_required"
    )
    assert consumer["summary"]["memory_writer_apply_request_available"] is False
    assert (
        "sequence_policy_memory_writer_dry_run_consumer_writer_summary_live_authority_granted_not_false"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "sequence_policy_memory_writer_dry_run_consumer_writer_summary_apply_requested_not_false"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "sequence_policy_memory_writer_dry_run_consumer_writer_safety_allowed_kinds_mismatch"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "sequence_policy_memory_writer_dry_run_consumer_writer_dry_run_blocker_mismatch"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert consumer["memory_writer_apply_request"]["memory_draft_payload"] == {}


def test_memory_writer_dry_run_result_consumer_rejects_non_ready_source_draft():
    writer = _writer_dry_run_payload()
    writer["source_memory_draft_status"] = "blocked"

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer(
            memory_draft_result_consumer=_memory_draft_result_consumer(),
            memory_writer_dry_run_result=writer,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer_replacement_required"
    )
    assert consumer["summary"]["memory_writer_apply_request_available"] is False
    assert (
        "sequence_policy_memory_writer_dry_run_consumer_writer_source_draft_not_ready"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert consumer["memory_writer_apply_request"]["memory_draft_payload"] == {}


def test_memory_writer_dry_run_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_DRY_RUN_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_DRY_RUN_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_write_applied"] is False
    assert consumer["summary"]["memory_writer_apply_request_available"] is False


def test_memory_writer_dry_run_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_DRY_RUN_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
