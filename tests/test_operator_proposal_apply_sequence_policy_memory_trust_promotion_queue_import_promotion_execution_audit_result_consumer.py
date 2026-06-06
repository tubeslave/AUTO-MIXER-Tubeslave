from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_AUDIT_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer import (
    _execution_payload,
    _promotion_execution_gate_consumer_ready,
)


def _promotion_execution_result_consumer_ready(tmp_path):
    prior_consumer = _promotion_execution_gate_consumer_ready(tmp_path)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer(
            promotion_execution_gate_result_consumer=prior_consumer,
            promotion_execution_result=_execution_payload(prior_consumer),
        )
    )


def _audit_row(proposal_id, *, index, prefix, status="passed"):
    return {
        "proposal_id": proposal_id,
        "execution_result_index": index,
        "status": status,
        "value_type": "fader",
        "channel": 2,
        "effective_value": -8.5,
        "idempotency_key": f"{prefix}:{proposal_id}",
        "readback_status": "verified_from_cache",
        "proof_status": "verified",
        "rollback_available": True,
        "live_apply_entrypoint": "apply_operator_proposal",
    }


def _audit_payload(prior_consumer, *, rows=None, blocked=None, status="passed"):
    prefix = prior_consumer["summary"]["idempotency_key_prefix"]
    ids = list(prior_consumer["summary"]["execution_result_proposal_ids"])
    audit_rows = list(
        rows
        if rows is not None
        else [
            _audit_row(
                proposal_id,
                index=index,
                prefix=prefix,
            )
            for index, proposal_id in enumerate(ids)
        ]
    )
    blocked_results = list(blocked if blocked is not None else [])
    passed_rows = [row for row in audit_rows if row.get("status") == "passed"]
    blocked_rows = [row for row in audit_rows if row.get("status") == "blocked"]
    return {
        "type": "operator_proposal_apply_sequence_execution_audit",
        "status": status,
        "success": status == "passed",
        "reason": None if status == "passed" else "sequence_execution_audit_blocked",
        "generated_at": 123.0,
        "source_execution_status": "applied",
        "summary": {
            "input_execution_result_count": len(ids),
            "audited_count": len(audit_rows),
            "passed_count": len(passed_rows),
            "blocked_count": len(blocked_rows) + len(blocked_results),
            "passed_proposal_ids": [row["proposal_id"] for row in passed_rows],
            "blocked_proposal_ids": [
                row["proposal_id"] for row in [*blocked_rows, *blocked_results]
            ],
            "audit_only": True,
            "execution_allowed_by_audit": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "mutates_console_state": False,
            "queries_console": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
        },
        "safety": {
            "audit_only": True,
            "execution_result_payload_required": True,
            "does_not_call_apply_operator_proposal": True,
            "does_not_call_manual_console_write": True,
            "live_apply_entrypoint_required": "apply_operator_proposal",
            "allowed_live_write_kinds": ["fader", "gain"],
            "execution_allowed_by_audit": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "mutates_console_state": False,
            "queries_console": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
        },
        "audit_rows": audit_rows,
        "blocked_results": blocked_results,
    }


def test_promotion_execution_audit_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer.py"
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


def test_promotion_execution_audit_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_AUDIT_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer_not_ready"
    )
    assert consumer["safety"]["read_only"] is True
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["policy_mutation"] is False
    assert consumer["safety"]["idempotency_reserved"] is False
    assert consumer["summary"]["promotion_audit_policy_gate_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert (
        consumer["promotion_audit_policy_gate_request"]["execution_audit_payload"]
        == {}
    )


def test_promotion_execution_audit_result_consumer_emits_policy_gate_request(tmp_path):
    prior_consumer = _promotion_execution_result_consumer_ready(tmp_path)
    audit = _audit_payload(prior_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer(
            promotion_execution_result_consumer=prior_consumer,
            promotion_execution_audit_result=audit,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_audit_policy_gate_request_available"
    )
    assert consumer["summary"]["expected_execution_audit_row_count"] == 1
    assert consumer["summary"]["execution_audit_row_count"] == 1
    assert consumer["summary"]["passed_execution_audit_row_count"] == 1
    assert consumer["summary"]["blocked_execution_audit_row_count"] == 0
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_audit_policy_gate_request_available"] is True
    assert consumer["summary"]["audit_policy_gate_handler_invoked"] is False
    assert consumer["summary"]["policy_gate_created"] is False
    assert consumer["summary"]["execution_allowed_by_consumer"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    assert consumer["generic_execution_audit_result_consumer"]["summary"][
        "audit_policy_gate_request_available"
    ] is True

    request = consumer["promotion_audit_policy_gate_request"]
    assert request["handler"] == "gate_operator_proposal_apply_sequence_audit_policy"
    assert request["read_only_template"] is True
    assert request["promotion_execution_audit_result_consumer_only"] is True
    assert request["promotion_audit_policy_gate_request_template_only"] is True
    assert len(request["execution_audit_payload"]["audit_rows"]) == 1
    assert request["policy_gate_created"] is False
    assert request["idempotency_reserved"] is False
    assert request["live_authority_granted"] is False
    spec = consumer["promotion_audit_policy_gate_request_specs"][0]
    assert spec["proposal_id"] == "proposal-first"
    assert spec["audit_policy_gate_handler_invoked"] is False
    assert spec["policy_gate_created"] is False
    assert spec["live_authority_granted"] is False


def test_promotion_execution_audit_result_consumer_blocks_blocked_rows(tmp_path):
    prior_consumer = _promotion_execution_result_consumer_ready(tmp_path)
    prefix = prior_consumer["summary"]["idempotency_key_prefix"]
    rows = [
        _audit_row(
            "proposal-first",
            index=0,
            prefix=prefix,
            status="blocked",
        )
    ]
    blocked = [
        {
            "proposal_id": "proposal-first",
            "execution_result_index": 0,
            "status": "blocked",
            "reason": "execution_row_readback_not_verified",
        }
    ]

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer(
            promotion_execution_result_consumer=prior_consumer,
            promotion_execution_audit_result=_audit_payload(
                prior_consumer,
                rows=rows,
                blocked=blocked,
                status="blocked",
            ),
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_audit_policy_gate_request_available"] is False
    assert (
        consumer["promotion_audit_policy_gate_request"]["execution_audit_payload"]
        == {}
    )
    assert consumer["summary"]["live_authority_granted"] is False
    assert any(
        item.endswith(
            "sequence_policy_execution_audit_result_consumer_audit_has_blocked_results"
        )
        for item in consumer["summary"]["invalid_record_ids"]
    )


def test_promotion_execution_audit_result_consumer_rejects_forged_audit(tmp_path):
    prior_consumer = _promotion_execution_result_consumer_ready(tmp_path)
    audit = _audit_payload(prior_consumer)
    audit["summary"]["live_authority_granted"] = True
    audit["safety"]["allowed_live_write_kinds"] = ["fader", "gain", "eq"]
    audit["audit_rows"][0]["idempotency_key"] = "forged"
    audit["audit_rows"][0]["effective_value"] = 1.0

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer(
            promotion_execution_result_consumer=prior_consumer,
            promotion_execution_audit_result=audit,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_audit_policy_gate_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert any(
        item.endswith(
            "sequence_policy_execution_audit_result_consumer_audit_summary_live_authority_granted_not_false"
        )
        for item in invalid
    )
    assert any(
        item.endswith(
            "sequence_policy_execution_audit_result_consumer_audit_safety_allowed_kinds_mismatch"
        )
        for item in invalid
    )
    assert any(item.endswith("proposal-first:idempotency_key_mismatch") for item in invalid)
    assert any(item.endswith("proposal-first:fader_ceiling_exceeded") for item in invalid)
    assert (
        consumer["promotion_audit_policy_gate_request"]["execution_audit_payload"]
        == {}
    )
    assert consumer["summary"]["live_write"] is False


def test_promotion_execution_audit_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_AUDIT_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_AUDIT_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["policy_gate_created"] is False
    assert consumer["summary"]["promotion_audit_policy_gate_request_available"] is False


def test_promotion_execution_audit_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_AUDIT_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
