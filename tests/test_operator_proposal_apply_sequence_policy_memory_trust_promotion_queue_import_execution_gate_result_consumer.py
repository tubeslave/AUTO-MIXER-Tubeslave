from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_GATE_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer import (
    _dry_run_payload,
    _sequence_plan_result_consumer,
)


def _dry_run_result_consumer(prefix="seq-exec"):
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer(
            sequence_plan_result_consumer=_sequence_plan_result_consumer(prefix=prefix),
            dry_run_result=_dry_run_payload(prefix=prefix),
        )
    )


def _gate_candidate(proposal_id, *, index, prefix="seq-exec", status="ready_for_executor"):
    return {
        "planned_apply_index": index,
        "proposal_id": proposal_id,
        "dry_run_result_index": index,
        "status": status,
        "execution_allowed_by_gate": False,
        "requires_separate_executor": True,
        "apply_request": {
            "type": "apply_operator_proposal",
            "proposal_id": proposal_id,
            "approved": True,
            "idempotency_key": f"{prefix}:{proposal_id}",
        },
        "current_proposal": {
            "id": proposal_id,
            "status": "pending",
            "can_apply": True,
        },
        "normalized_changes": [
            {
                "change_index": 0,
                "channel": 2,
                "value_type": "fader",
                "address": "/ch/2/fdr",
                "effective_value": -8.5,
                "delta_db": -0.5,
                "pre_read_value": -8.0,
                "pre_read_status": "verified_from_cache",
                "rollback_value": -8.0,
                "rollback_expected": True,
            }
        ],
    }


def _gate_payload(prefix="seq-exec", *, candidates=None, blocked=None, status="ready"):
    ready_for_executor = list(
        candidates
        if candidates is not None
        else [
            _gate_candidate("proposal-first", index=0, prefix=prefix),
            _gate_candidate("proposal-second", index=1, prefix=prefix),
        ]
    )
    blocked_results = list(blocked if blocked is not None else [])
    return {
        "type": "operator_proposal_apply_sequence_execution_gate",
        "status": status,
        "success": status == "ready",
        "reason": None if status == "ready" else "execution_gate_blocked",
        "generated_at": 123.0,
        "operator_mode": {"mode": "supervised"},
        "summary": {
            "input_dry_run_result_count": 2,
            "ready_for_executor_count": len(ready_for_executor),
            "blocked_count": len(blocked_results),
            "ready_for_executor_proposal_ids": [
                item["proposal_id"] for item in ready_for_executor
            ],
            "blocked_proposal_ids": [
                item["proposal_id"] for item in blocked_results if item.get("proposal_id")
            ],
            "operator_mode_allows_apply_now": True,
            "operator_ack_received": True,
            "execution_allowed_by_gate": False,
            "requires_separate_executor": True,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "mutates_console_state": False,
            "queries_console": False,
            "live_widening_allowed": False,
            "queue_freshness_checked": True,
        },
        "safety": {
            "execution_gate_only": True,
            "dry_run_results_required": True,
            "operator_ack_required": True,
            "current_queue_proposal_required": True,
            "does_not_call_apply_operator_proposal": True,
            "execution_allowed_by_gate": False,
            "requires_separate_executor": True,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "mutates_console_state": False,
            "queries_console": False,
            "live_widening_allowed": False,
            "live_apply_entrypoint": "apply_operator_proposal",
            "allowed_live_write_kinds": ["fader", "gain", "pan"],
        },
        "ready_for_executor": ready_for_executor,
        "blocked_results": blocked_results,
    }


def test_execution_gate_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer.py"
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


def test_execution_gate_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer()
    )

    assert (
        consumer["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_GATE_RESULT_CONSUMER_METHOD_ID
    )
    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_execution_gate_result_consumer_not_ready"
    )
    assert consumer["summary"]["executor_request_available"] is False
    assert consumer["summary"]["sequence_executor_handler_invoked"] is False
    assert consumer["summary"]["sequence_executor_created"] is False
    assert consumer["summary"]["live_write"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["executor_request"]["execution_gate_payload"] == {}
    assert consumer["executor_request_specs"] == []


def test_execution_gate_result_consumer_emits_executor_request_for_ready_rows():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer(
            dry_run_result_consumer=_dry_run_result_consumer(),
            execution_gate_result=_gate_payload(),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_executor_request_available"
    )
    assert consumer["summary"]["expected_gate_candidate_count"] == 2
    assert consumer["summary"]["ready_for_executor_count"] == 2
    assert consumer["summary"]["blocked_gate_result_count"] == 0
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["executor_request_available"] is True
    assert consumer["summary"]["sequence_executor_handler_invoked"] is False
    assert consumer["summary"]["sequence_executor_created"] is False
    assert consumer["summary"]["execution_allowed_by_consumer"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["summary"]["ready_for_executor_proposal_ids"] == [
        "proposal-first",
        "proposal-second",
    ]
    assert consumer["executor_request"]["handler"] == (
        "execute_operator_proposal_apply_sequence"
    )
    assert consumer["executor_request"]["execute"] is False
    assert consumer["executor_request"]["operator_ack"] is False
    assert consumer["executor_request"]["approved"] is False
    assert consumer["executor_request"]["operator_must_set_execute_true"] is True
    assert consumer["executor_request"]["operator_must_set_operator_ack_true"] is True
    assert consumer["executor_request"]["operator_must_set_approved_true"] is True
    assert len(consumer["executor_request"]["execution_gate_payload"]["ready_for_executor"]) == 2
    first = consumer["executor_request_specs"][0]
    assert first["proposal_id"] == "proposal-first"
    assert first["planned_executor_index"] == 0
    assert first["idempotency_key"] == "seq-exec:proposal-first"
    assert first["sequence_executor_handler"] == (
        "execute_operator_proposal_apply_sequence"
    )
    assert first["sequence_executor_invoked"] is False
    assert first["live_authority_granted"] is False


def test_execution_gate_result_consumer_blocks_when_gate_has_blocked_rows():
    blocked = [
        {
            "planned_apply_index": 0,
            "proposal_id": "proposal-first",
            "dry_run_result_index": 0,
            "status": "blocked",
            "reason": "operator_mode_blocks_live_write",
        },
        {
            "planned_apply_index": 1,
            "proposal_id": "proposal-second",
            "dry_run_result_index": 1,
            "status": "blocked",
            "reason": "operator_mode_blocks_live_write",
        },
    ]
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer(
            dry_run_result_consumer=_dry_run_result_consumer(),
            execution_gate_result=_gate_payload(
                candidates=[],
                blocked=blocked,
                status="blocked",
            ),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_execution_gate_result_consumer_blocked"
    )
    assert consumer["summary"]["ready_for_executor_count"] == 0
    assert consumer["summary"]["blocked_gate_result_count"] == 2
    assert consumer["summary"]["blocked_gate_proposal_ids"] == [
        "proposal-first",
        "proposal-second",
    ]
    assert consumer["summary"]["executor_request_available"] is False
    assert consumer["executor_request"]["execution_gate_payload"] == {}
    assert consumer["summary"]["live_authority_granted"] is False


def test_execution_gate_result_consumer_rejects_forged_gate_row():
    gate = _gate_payload()
    gate["ready_for_executor"][0]["apply_request"]["idempotency_key"] = "forged"
    gate["ready_for_executor"][0]["normalized_changes"][0]["effective_value"] = 1.0

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer(
            dry_run_result_consumer=_dry_run_result_consumer(),
            execution_gate_result=gate,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_execution_gate_result_consumer_replacement_required"
    )
    assert consumer["summary"]["executor_request_available"] is False
    assert (
        "proposal-first:apply_request_idempotency_key_mismatch"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "proposal-first:change_fader_ceiling_exceeded"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert consumer["executor_request"]["execution_gate_payload"] == {}
    assert consumer["summary"]["live_write"] is False


def test_execution_gate_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_GATE_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_GATE_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["sequence_executor_created"] is False
    assert consumer["summary"]["executor_request_available"] is False


def test_execution_gate_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_GATE_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
