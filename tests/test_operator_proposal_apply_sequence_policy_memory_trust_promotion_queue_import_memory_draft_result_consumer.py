from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_DRAFT_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer import (
    _execution_audit_result_consumer,
    _policy_gate_payload,
)


def _audit_policy_gate_result_consumer(prefix="seq-policy"):
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer(
            execution_audit_result_consumer=_execution_audit_result_consumer(prefix=prefix),
            audit_policy_gate_result=_policy_gate_payload(prefix=prefix),
        )
    )


def _memory_draft_record(proposal_id, *, index, prefix="seq-policy", status="memory_draft_ready"):
    idempotency_key = f"{prefix}:{proposal_id}"
    return {
        "draft_id": f"operator-sequence-policy-memory:{proposal_id}:{idempotency_key}",
        "proposal_id": proposal_id,
        "policy_evidence_index": index,
        "status": status,
        "learning_scope": "supervised_fader_gain_sequence_execution",
        "target_memory_path": ".ai/memory/project.md",
        "target_section": "Project Memory Lessons",
        "append_only": True,
        "memory_write_allowed_by_draft": False,
        "append_only_memory_write_allowed_by_draft": False,
        "requires_separate_memory_writer": True,
        "operator_ack_required_for_write": True,
        "memory_mutation": False,
        "runtime_memory_mutation": False,
        "policy_mutation": False,
        "model_mutation": False,
        "queue_mutation": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "evidence": {
            "value_type": "fader",
            "channel": 2,
            "effective_value": -8.5,
            "idempotency_key": idempotency_key,
            "readback_status": "verified_from_cache",
            "proof_status": "verified",
            "rollback_available": True,
            "live_apply_entrypoint": "apply_operator_proposal",
        },
        "draft_text": (
            "Supervised fader/gain sequence execution was verified from "
            f"`{proposal_id}` with `fader` evidence, idempotency "
            f"`{idempotency_key}`, readback `verified_from_cache`, and rollback "
            "available. Future policy or autonomous use still requires separate "
            "audited gates and safety limits."
        ),
    }


def _memory_draft_payload(prefix="seq-policy", *, records=None, blocked=None, status="ready"):
    memory_draft_records = list(
        records
        if records is not None
        else [
            _memory_draft_record("proposal-first", index=0, prefix=prefix),
            _memory_draft_record("proposal-second", index=1, prefix=prefix),
        ]
    )
    blocked_results = list(blocked if blocked is not None else [])
    return {
        "type": "operator_proposal_apply_sequence_policy_memory_draft",
        "status": status,
        "success": status == "ready",
        "reason": None if status == "ready" else "sequence_policy_memory_draft_blocked",
        "generated_at": 123.0,
        "source_policy_gate_status": "ready",
        "summary": {
            "input_policy_evidence_count": 2,
            "memory_draft_count": len(memory_draft_records),
            "blocked_count": len(blocked_results),
            "memory_draft_proposal_ids": [
                item["proposal_id"] for item in memory_draft_records
            ],
            "blocked_proposal_ids": [
                item["proposal_id"] for item in blocked_results if item.get("proposal_id")
            ],
            "memory_draft_only": True,
            "policy_gate_payload_required": True,
            "append_only_memory_write_allowed_by_draft": False,
            "requires_separate_memory_writer": True,
            "execution_allowed_by_draft": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "mutates_console_state": False,
            "queries_console": False,
            "policy_mutation": False,
            "memory_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "safety": {
            "memory_draft_only": True,
            "policy_gate_payload_required": True,
            "does_not_call_apply_operator_proposal": True,
            "does_not_call_manual_console_write": True,
            "allowed_live_write_kinds": ["fader", "gain", "pan"],
            "append_only_memory_write_allowed_by_draft": False,
            "requires_separate_memory_writer": True,
            "execution_allowed_by_draft": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "mutates_console_state": False,
            "queries_console": False,
            "policy_mutation": False,
            "memory_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "memory_draft_records": memory_draft_records,
        "blocked_results": blocked_results,
    }


def test_memory_draft_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer.py"
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


def test_memory_draft_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer()
    )

    assert (
        consumer["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_DRAFT_RESULT_CONSUMER_METHOD_ID
    )
    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_draft_result_consumer_not_ready"
    )
    assert consumer["summary"]["memory_writer_request_available"] is False
    assert consumer["summary"]["memory_writer_handler_invoked"] is False
    assert consumer["summary"]["memory_file_write_applied"] is False
    assert consumer["summary"]["live_write"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["memory_writer_request"]["memory_draft_payload"] == {}
    assert consumer["memory_writer_request_specs"] == []


def test_memory_draft_result_consumer_emits_writer_request_for_ready_drafts():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer(
            audit_policy_gate_result_consumer=_audit_policy_gate_result_consumer(),
            memory_draft_result=_memory_draft_payload(),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_writer_request_available"
    )
    assert consumer["summary"]["expected_memory_draft_count"] == 2
    assert consumer["summary"]["memory_draft_record_count"] == 2
    assert consumer["summary"]["ready_memory_draft_record_count"] == 2
    assert consumer["summary"]["blocked_memory_draft_record_count"] == 0
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["memory_writer_request_available"] is True
    assert consumer["summary"]["memory_writer_handler_invoked"] is False
    assert consumer["summary"]["memory_file_write_applied"] is False
    assert consumer["summary"]["memory_write_allowed_by_consumer"] is False
    assert consumer["summary"]["execution_allowed_by_consumer"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["summary"]["memory_draft_proposal_ids"] == [
        "proposal-first",
        "proposal-second",
    ]
    request = consumer["memory_writer_request"]
    assert request["handler"] == "write_operator_proposal_apply_sequence_policy_memory"
    assert request["apply"] is False
    assert request["operator_ack"] is False
    assert request["operator_must_set_apply_true"] is True
    assert request["operator_must_set_operator_ack_true"] is True
    assert len(request["memory_draft_payload"]["memory_draft_records"]) == 2
    first = consumer["memory_writer_request_specs"][0]
    assert first["proposal_id"] == "proposal-first"
    assert first["planned_memory_write_index"] == 0
    assert first["idempotency_key"] == "seq-policy:proposal-first"
    assert first["memory_writer_handler"] == (
        "write_operator_proposal_apply_sequence_policy_memory"
    )
    assert first["memory_writer_handler_invoked"] is False
    assert first["memory_file_write_applied"] is False
    assert first["memory_write_allowed_by_consumer"] is False
    assert first["live_authority_granted"] is False


def test_memory_draft_result_consumer_rejects_forged_memory_draft_record():
    memory_draft = _memory_draft_payload()
    memory_draft["summary"]["live_authority_granted"] = True
    memory_draft["safety"]["allowed_live_write_kinds"] = ["fader", "gain", "eq"]
    memory_draft["memory_draft_records"][0]["draft_id"] = "forged"
    memory_draft["memory_draft_records"][0]["evidence"]["idempotency_key"] = "forged"
    memory_draft["memory_draft_records"][0]["evidence"]["effective_value"] = 1.0
    memory_draft["memory_draft_records"][1]["memory_mutation"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer(
            audit_policy_gate_result_consumer=_audit_policy_gate_result_consumer(),
            memory_draft_result=memory_draft,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_draft_result_consumer_replacement_required"
    )
    assert consumer["summary"]["memory_writer_request_available"] is False
    assert (
        "sequence_policy_memory_draft_result_consumer_draft_summary_live_authority_granted_not_false"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "sequence_policy_memory_draft_result_consumer_draft_safety_allowed_kinds_mismatch"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert "proposal-first:draft_id_mismatch" in consumer["summary"][
        "invalid_record_ids"
    ]
    assert "proposal-first:evidence_idempotency_key_mismatch" in consumer["summary"][
        "invalid_record_ids"
    ]
    assert "proposal-first:evidence_fader_ceiling_exceeded" in consumer["summary"][
        "invalid_record_ids"
    ]
    assert "proposal-second:memory_mutation_not_false" in consumer["summary"][
        "invalid_record_ids"
    ]
    assert consumer["memory_writer_request"]["memory_draft_payload"] == {}
    assert consumer["summary"]["live_write"] is False


def test_memory_draft_result_consumer_rejects_memory_draft_with_blocked_results():
    blocked = [
        {
            "proposal_id": "proposal-second",
            "policy_evidence_index": 1,
            "status": "blocked",
            "reason": "policy_evidence_readback_not_verified",
        }
    ]
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer(
            audit_policy_gate_result_consumer=_audit_policy_gate_result_consumer(),
            memory_draft_result=_memory_draft_payload(
                records=[_memory_draft_record("proposal-first", index=0)],
                blocked=blocked,
                status="blocked",
            ),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_draft_result_consumer_replacement_required"
    )
    assert consumer["summary"]["memory_writer_request_available"] is False
    assert consumer["memory_writer_request"]["memory_draft_payload"] == {}
    assert (
        "sequence_policy_memory_draft_result_consumer_draft_has_blocked_results"
        in consumer["summary"]["invalid_record_ids"]
    )


def test_memory_draft_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_DRAFT_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_DRAFT_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_file_write_applied"] is False
    assert consumer["summary"]["memory_writer_request_available"] is False


def test_memory_draft_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_DRAFT_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
