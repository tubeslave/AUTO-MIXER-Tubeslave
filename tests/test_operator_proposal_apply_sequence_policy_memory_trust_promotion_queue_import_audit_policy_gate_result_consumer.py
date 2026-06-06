from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_AUDIT_POLICY_GATE_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer import (
    _audit_payload,
    _execution_result_consumer,
)


def _execution_audit_result_consumer(prefix="seq-policy"):
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer(
            execution_result_consumer=_execution_result_consumer(prefix=prefix),
            execution_audit_result=_audit_payload(prefix=prefix),
        )
    )


def _policy_row(proposal_id, *, index, prefix="seq-policy", status="policy_evidence_ready"):
    return {
        "proposal_id": proposal_id,
        "audit_row_index": index,
        "status": status,
        "eligible_for_supervised_sequence_policy": True,
        "execution_allowed_by_policy_gate": False,
        "policy_mutation_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "value_type": "fader",
        "channel": 2,
        "effective_value": -8.5,
        "idempotency_key": f"{prefix}:{proposal_id}",
        "readback_status": "verified_from_cache",
        "proof_status": "verified",
        "rollback_available": True,
        "live_apply_entrypoint": "apply_operator_proposal",
    }


def _policy_gate_payload(prefix="seq-policy", *, rows=None, blocked=None, status="ready"):
    policy_evidence = list(
        rows
        if rows is not None
        else [
            _policy_row("proposal-first", index=0, prefix=prefix),
            _policy_row("proposal-second", index=1, prefix=prefix),
        ]
    )
    blocked_results = list(blocked if blocked is not None else [])
    ready_rows = [
        row for row in policy_evidence if row.get("status") == "policy_evidence_ready"
    ]
    blocked_rows = [row for row in policy_evidence if row.get("status") == "blocked"]
    return {
        "type": "operator_proposal_apply_sequence_audit_policy_gate",
        "status": status,
        "success": status == "ready",
        "reason": None if status == "ready" else "sequence_audit_policy_gate_blocked",
        "generated_at": 123.0,
        "source_audit_status": "passed",
        "summary": {
            "input_audit_row_count": 2,
            "policy_evidence_ready_count": len(ready_rows),
            "blocked_count": len(blocked_rows) + len(blocked_results),
            "policy_evidence_ready_proposal_ids": [
                row["proposal_id"] for row in ready_rows
            ],
            "blocked_proposal_ids": [
                row["proposal_id"] for row in [*blocked_rows, *blocked_results]
            ],
            "policy_gate_only": True,
            "audit_payload_required": True,
            "execution_allowed_by_policy_gate": False,
            "execution_allowed_by_audit": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "mutates_console_state": False,
            "queries_console": False,
            "policy_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "safety": {
            "policy_gate_only": True,
            "audit_payload_required": True,
            "does_not_call_apply_operator_proposal": True,
            "does_not_call_manual_console_write": True,
            "allowed_live_write_kinds": ["fader", "gain"],
            "execution_allowed_by_policy_gate": False,
            "execution_allowed_by_audit": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "mutates_console_state": False,
            "queries_console": False,
            "policy_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "policy_evidence": policy_evidence,
        "blocked_results": blocked_results,
    }


def test_audit_policy_gate_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer.py"
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


def test_audit_policy_gate_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer()
    )

    assert (
        consumer["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_AUDIT_POLICY_GATE_RESULT_CONSUMER_METHOD_ID
    )
    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer_not_ready"
    )
    assert consumer["summary"]["memory_draft_request_available"] is False
    assert consumer["summary"]["memory_draft_handler_invoked"] is False
    assert consumer["summary"]["memory_draft_created"] is False
    assert consumer["summary"]["live_write"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["memory_draft_request"]["policy_gate_payload"] == {}
    assert consumer["memory_draft_request_specs"] == []


def test_audit_policy_gate_result_consumer_emits_memory_draft_request_for_ready_policy_evidence():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer(
            execution_audit_result_consumer=_execution_audit_result_consumer(),
            audit_policy_gate_result=_policy_gate_payload(),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_memory_draft_request_available"
    )
    assert consumer["summary"]["expected_policy_evidence_count"] == 2
    assert consumer["summary"]["policy_evidence_count"] == 2
    assert consumer["summary"]["ready_policy_evidence_count"] == 2
    assert consumer["summary"]["blocked_policy_evidence_count"] == 0
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["memory_draft_request_available"] is True
    assert consumer["summary"]["memory_draft_handler_invoked"] is False
    assert consumer["summary"]["memory_draft_created"] is False
    assert consumer["summary"]["memory_write_allowed_by_consumer"] is False
    assert consumer["summary"]["execution_allowed_by_consumer"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["summary"]["policy_evidence_proposal_ids"] == [
        "proposal-first",
        "proposal-second",
    ]
    assert consumer["memory_draft_request"]["handler"] == (
        "draft_operator_proposal_apply_sequence_policy_memory"
    )
    assert len(
        consumer["memory_draft_request"]["policy_gate_payload"]["policy_evidence"]
    ) == 2
    first = consumer["memory_draft_request_specs"][0]
    assert first["proposal_id"] == "proposal-first"
    assert first["planned_memory_draft_index"] == 0
    assert first["idempotency_key"] == "seq-policy:proposal-first"
    assert first["memory_draft_handler"] == (
        "draft_operator_proposal_apply_sequence_policy_memory"
    )
    assert first["memory_draft_handler_invoked"] is False
    assert first["memory_write_allowed_by_consumer"] is False
    assert first["live_authority_granted"] is False


def test_audit_policy_gate_result_consumer_rejects_forged_policy_evidence():
    policy_gate = _policy_gate_payload()
    policy_gate["summary"]["live_authority_granted"] = True
    policy_gate["safety"]["allowed_live_write_kinds"] = ["fader", "gain", "eq"]
    policy_gate["policy_evidence"][0]["idempotency_key"] = "forged"
    policy_gate["policy_evidence"][0]["effective_value"] = 1.0
    policy_gate["policy_evidence"][1]["live_authority_granted"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer(
            execution_audit_result_consumer=_execution_audit_result_consumer(),
            audit_policy_gate_result=policy_gate,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer_replacement_required"
    )
    assert consumer["summary"]["memory_draft_request_available"] is False
    assert (
        "sequence_policy_audit_policy_gate_result_consumer_policy_gate_summary_live_authority_granted_not_false"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "sequence_policy_audit_policy_gate_result_consumer_policy_gate_safety_allowed_kinds_mismatch"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert "proposal-first:idempotency_key_mismatch" in consumer["summary"][
        "invalid_record_ids"
    ]
    assert "proposal-first:fader_ceiling_exceeded" in consumer["summary"][
        "invalid_record_ids"
    ]
    assert "proposal-second:live_authority_granted_not_false" in consumer["summary"][
        "invalid_record_ids"
    ]
    assert consumer["memory_draft_request"]["policy_gate_payload"] == {}
    assert consumer["summary"]["live_write"] is False


def test_audit_policy_gate_result_consumer_rejects_non_finite_policy_value():
    policy_gate = _policy_gate_payload()
    policy_gate["policy_evidence"][0]["value_type"] = "gain"
    policy_gate["policy_evidence"][0]["effective_value"] = float("nan")

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer(
            execution_audit_result_consumer=_execution_audit_result_consumer(),
            audit_policy_gate_result=policy_gate,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer_replacement_required"
    )
    assert consumer["summary"]["memory_draft_request_available"] is False
    assert "proposal-first:effective_value_invalid" in consumer["summary"][
        "invalid_record_ids"
    ]
    assert consumer["memory_draft_request"]["policy_gate_payload"] == {}
    assert consumer["summary"]["live_authority_granted"] is False


def test_audit_policy_gate_result_consumer_rejects_policy_gate_with_blocked_results():
    blocked = [
        {
            "proposal_id": "proposal-second",
            "audit_row_index": 1,
            "status": "blocked",
            "reason": "audit_row_readback_not_verified",
        }
    ]
    rows = [
        _policy_row("proposal-first", index=0),
        _policy_row("proposal-second", index=1, status="blocked"),
    ]
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer(
            execution_audit_result_consumer=_execution_audit_result_consumer(),
            audit_policy_gate_result=_policy_gate_payload(
                rows=rows,
                blocked=blocked,
                status="blocked",
            ),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer_replacement_required"
    )
    assert consumer["summary"]["memory_draft_request_available"] is False
    assert consumer["memory_draft_request"]["policy_gate_payload"] == {}
    assert (
        "sequence_policy_audit_policy_gate_result_consumer_policy_gate_has_blocked_results"
        in consumer["summary"]["invalid_record_ids"]
    )


def test_audit_policy_gate_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_AUDIT_POLICY_GATE_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_AUDIT_POLICY_GATE_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["memory_draft_created"] is False
    assert consumer["summary"]["memory_draft_request_available"] is False


def test_audit_policy_gate_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_AUDIT_POLICY_GATE_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
