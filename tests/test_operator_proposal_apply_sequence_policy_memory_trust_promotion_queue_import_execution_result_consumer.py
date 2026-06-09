from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer import (
    _dry_run_result_consumer,
    _gate_payload,
)


def _execution_gate_result_consumer(prefix="seq-audit"):
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer(
            dry_run_result_consumer=_dry_run_result_consumer(prefix=prefix),
            execution_gate_result=_gate_payload(prefix=prefix),
        )
    )


def _execution_row(proposal_id, *, index, prefix="seq-audit", status="applied"):
    apply_status = "applied" if status == "applied" else "blocked"
    return {
        "planned_apply_index": index,
        "proposal_id": proposal_id,
        "candidate_index": index,
        "status": status,
        "reason": None if status == "applied" else "apply_not_applied",
        "apply_attempted": True,
        "apply_request": {
            "type": "apply_operator_proposal",
            "proposal_id": proposal_id,
            "approved": True,
            "idempotency_key": f"{prefix}:{proposal_id}",
        },
        "proposal": {
            "id": proposal_id,
            "status": apply_status,
            "can_apply": status != "applied",
        },
        "apply_result": {
            "status": apply_status,
            "channel": 2,
            "value_type": "fader",
            "effective_value": -8.5,
            "old_value": -8.0,
            "pre_read_status": "verified_from_cache",
            "pre_read_source": "mixer_state_cache",
            "readback_value": -8.5,
            "readback_status": "verified_from_cache",
            "readback_source": "mixer_state_cache",
            "proof_status": "verified",
            "rollback_available": True,
            "rollback_value": -8.0,
        },
    }


def _execution_payload(prefix="seq-audit", *, rows=None, status="applied"):
    execution_results = list(
        rows
        if rows is not None
        else [
            _execution_row("proposal-first", index=0, prefix=prefix),
            _execution_row("proposal-second", index=1, prefix=prefix),
        ]
    )
    applied = [item for item in execution_results if item.get("status") == "applied"]
    blocked = [item for item in execution_results if item.get("status") == "blocked"]
    return {
        "type": "operator_proposal_apply_sequence_execution_result",
        "status": status,
        "success": status == "applied",
        "reason": None if status == "applied" else "sequence_executor_stopped_on_block",
        "generated_at": 123.0,
        "operator_mode": {"mode": "supervised"},
        "summary": {
            "input_candidate_count": 2,
            "attempted_count": len(execution_results),
            "applied_count": len(applied),
            "blocked_count": len(blocked),
            "applied_proposal_ids": [item["proposal_id"] for item in applied],
            "blocked_proposal_ids": [item["proposal_id"] for item in blocked],
            "operator_mode_allows_apply_now": True,
            "operator_ack_received": True,
            "execute_requested": True,
            "approved": True,
            "requires_gate_ready": True,
            "uses_existing_apply_operator_proposal": True,
            "sequence_executor_called_apply_operator_proposal": True,
            "live_apply_entrypoint": "apply_operator_proposal",
            "live_write": True,
            "direct_live_write": False,
            "apply_attempted": True,
            "queue_mutation": True,
            "proposal_status_mutation": True,
            "idempotency_reserved": True,
            "idempotency_reserved_by_executor_directly": False,
            "idempotency_reservation_delegated_to_apply_operator_proposal": True,
            "manual_console_write_called_by_apply_operator_proposal": True,
            "manual_console_write_called_by_sequence_executor": False,
            "direct_manual_console_write_called": False,
            "mutates_console_state": True,
            "queries_console": False,
            "live_widening_allowed": False,
            "stopped_on_block": False,
            "queue_freshness_checked": True,
        },
        "safety": {
            "sequence_executor": True,
            "execution_gate_payload_required": True,
            "ready_for_executor_required": True,
            "requires_execute_true": True,
            "requires_operator_ack": True,
            "requires_approved_true": True,
            "requires_gate_ready": True,
            "uses_existing_apply_operator_proposal": True,
            "live_apply_entrypoint": "apply_operator_proposal",
            "allowed_live_write_kinds": ["fader", "gain", "pan"],
            "live_write": True,
            "direct_live_write": False,
            "apply_attempted": True,
            "queue_mutation_by_executor_directly": False,
            "proposal_status_mutation_by_executor_directly": False,
            "idempotency_reserved_by_executor_directly": False,
            "idempotency_reservation_delegated_to_apply_operator_proposal": True,
            "does_not_call_manual_console_write_directly": True,
            "manual_console_write_called_by_sequence_executor": False,
            "direct_manual_console_write_called": False,
            "manual_console_write_called_by_apply_operator_proposal": True,
            "mutates_console_state": True,
            "queries_console": False,
            "live_widening_allowed": False,
        },
        "execution_results": execution_results,
        "blocked_results": [],
    }


def test_execution_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer.py"
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
        "mute",
        "unmute",
        "set_insert",
        "set_fx_model",
        "set_fx_on",
        "set_fx_mix",
        "set_fx_parameter",
        "write_text",
    ]
    for token in forbidden:
        assert token not in source


def test_execution_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer()
    )

    assert (
        consumer["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_RESULT_CONSUMER_METHOD_ID
    )
    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_execution_result_consumer_not_ready"
    )
    assert consumer["summary"]["audit_request_available"] is False
    assert consumer["summary"]["audit_handler_invoked"] is False
    assert consumer["summary"]["audit_created"] is False
    assert consumer["summary"]["live_write"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["audit_request"]["execution_result_payload"] == {}
    assert consumer["audit_request_specs"] == []


def test_execution_result_consumer_emits_audit_request_for_applied_rows():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer(
            execution_gate_result_consumer=_execution_gate_result_consumer(),
            execution_result=_execution_payload(),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_audit_request_available"
    )
    assert consumer["summary"]["expected_execution_result_count"] == 2
    assert consumer["summary"]["execution_result_count"] == 2
    assert consumer["summary"]["applied_execution_result_count"] == 2
    assert consumer["summary"]["blocked_execution_result_count"] == 0
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["audit_request_available"] is True
    assert consumer["summary"]["audit_handler_invoked"] is False
    assert consumer["summary"]["audit_created"] is False
    assert consumer["summary"]["execution_allowed_by_consumer"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["summary"]["execution_result_proposal_ids"] == [
        "proposal-first",
        "proposal-second",
    ]
    assert consumer["audit_request"]["handler"] == (
        "audit_operator_proposal_apply_sequence_execution"
    )
    assert len(consumer["audit_request"]["execution_result_payload"]["execution_results"]) == 2
    first = consumer["audit_request_specs"][0]
    assert first["proposal_id"] == "proposal-first"
    assert first["planned_audit_index"] == 0
    assert first["idempotency_key"] == "seq-audit:proposal-first"
    assert first["audit_handler"] == "audit_operator_proposal_apply_sequence_execution"
    assert first["audit_handler_invoked"] is False
    assert first["live_authority_granted"] is False


def test_execution_result_consumer_blocks_when_executor_has_blocked_rows():
    rows = [
        _execution_row("proposal-first", index=0),
        _execution_row("proposal-second", index=1, status="blocked"),
    ]
    execution = _execution_payload(rows=rows, status="partial")
    execution["summary"]["stopped_on_block"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer(
            execution_gate_result_consumer=_execution_gate_result_consumer(),
            execution_result=execution,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_execution_result_consumer_blocked"
    )
    assert consumer["summary"]["applied_execution_result_count"] == 1
    assert consumer["summary"]["blocked_execution_result_count"] == 1
    assert consumer["summary"]["blocked_execution_proposal_ids"] == ["proposal-second"]
    assert consumer["summary"]["audit_request_available"] is False
    assert consumer["audit_request"]["execution_result_payload"] == {}
    assert consumer["summary"]["live_authority_granted"] is False


def test_execution_result_consumer_rejects_forged_execution_result():
    execution = _execution_payload()
    execution["summary"]["direct_manual_console_write_called"] = True
    execution["safety"]["allowed_live_write_kinds"] = ["fader", "gain", "eq"]
    execution["execution_results"][0]["apply_request"]["idempotency_key"] = "forged"
    execution["execution_results"][0]["apply_result"]["effective_value"] = 1.0

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer(
            execution_gate_result_consumer=_execution_gate_result_consumer(),
            execution_result=execution,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_execution_result_consumer_replacement_required"
    )
    assert consumer["summary"]["audit_request_available"] is False
    assert (
        "sequence_policy_execution_result_consumer_execution_summary_direct_manual_console_write_called_not_false"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "sequence_policy_execution_result_consumer_execution_safety_allowed_kinds_mismatch"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "proposal-first:apply_request_idempotency_key_mismatch"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "proposal-first:apply_result_fader_ceiling_exceeded"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert consumer["audit_request"]["execution_result_payload"] == {}
    assert consumer["summary"]["live_write"] is False


def test_execution_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["audit_created"] is False
    assert consumer["summary"]["audit_request_available"] is False


def test_execution_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
