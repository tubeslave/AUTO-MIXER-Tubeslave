from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_DRY_RUN_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer,
)
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer import (
    _preflight_result_consumer,
    _sequence_plan,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


def _sequence_plan_result_consumer(prefix="seq-gate"):
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer(
            preflight_result_consumer=_preflight_result_consumer(prefix=prefix),
            sequence_plan_result=_sequence_plan(prefix=prefix),
        )
    )


def _dry_run_result(proposal_id, *, index, prefix="seq-gate", status="would_apply"):
    return {
        "planned_apply_index": index,
        "proposal_id": proposal_id,
        "status": status,
        "reason": None if status == "would_apply" else "proposal_changed_since_plan",
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


def _dry_run_payload(prefix="seq-gate", *, results=None):
    dry_run_results = list(
        results
        if results is not None
        else [
            _dry_run_result("proposal-first", index=0, prefix=prefix),
            _dry_run_result("proposal-second", index=1, prefix=prefix),
        ]
    )
    would_apply_ids = [
        item["proposal_id"]
        for item in dry_run_results
        if item.get("status") == "would_apply"
    ]
    blocked_ids = [
        item["proposal_id"]
        for item in dry_run_results
        if item.get("status") == "blocked"
    ]
    return {
        "type": "operator_proposal_apply_sequence_dry_run",
        "status": "ok",
        "success": True,
        "reason": None,
        "generated_at": 123.0,
        "summary": {
            "input_step_count": 2,
            "result_count": len(dry_run_results),
            "would_apply_count": len(would_apply_ids),
            "blocked_count": len(blocked_ids),
            "would_apply_proposal_ids": would_apply_ids,
            "blocked_proposal_ids": blocked_ids,
            "execution_allowed_by_dry_run": False,
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
            "dry_run_only": True,
            "sequence_plan_required": True,
            "current_queue_proposal_required": True,
            "does_not_call_apply_operator_proposal": True,
            "execution_allowed_by_dry_run": False,
            "requires_operator_apply": True,
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
        "dry_run_results": dry_run_results,
    }


def test_dry_run_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer.py"
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


def test_dry_run_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer()
    )

    assert (
        consumer["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_DRY_RUN_RESULT_CONSUMER_METHOD_ID
    )
    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_dry_run_result_consumer_not_ready"
    )
    assert consumer["summary"]["execution_gate_request_available"] is False
    assert consumer["summary"]["execution_gate_handler_invoked"] is False
    assert consumer["summary"]["execution_gate_created"] is False
    assert consumer["summary"]["live_write"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["execution_gate_request"]["dry_run_payload"] == {}
    assert consumer["execution_gate_request_specs"] == []


def test_dry_run_result_consumer_emits_gate_request_for_would_apply_rows():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer(
            sequence_plan_result_consumer=_sequence_plan_result_consumer(),
            dry_run_result=_dry_run_payload(),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_execution_gate_request_available"
    )
    assert consumer["summary"]["expected_dry_run_result_count"] == 2
    assert consumer["summary"]["dry_run_result_count"] == 2
    assert consumer["summary"]["would_apply_result_count"] == 2
    assert consumer["summary"]["blocked_dry_run_result_count"] == 0
    assert consumer["summary"]["execution_gate_request_available"] is True
    assert consumer["summary"]["execution_gate_created"] is False
    assert consumer["summary"]["execution_allowed_by_consumer"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["summary"]["dry_run_result_proposal_ids"] == [
        "proposal-first",
        "proposal-second",
    ]
    assert consumer["execution_gate_request"]["handler"] == (
        "gate_operator_proposal_apply_sequence_execution"
    )
    assert consumer["execution_gate_request"]["operator_ack"] is False
    assert (
        consumer["execution_gate_request"]["operator_must_set_operator_ack_true"]
        is True
    )
    assert len(consumer["execution_gate_request"]["dry_run_payload"]["dry_run_results"]) == 2
    first = consumer["execution_gate_request_specs"][0]
    assert first["proposal_id"] == "proposal-first"
    assert first["planned_gate_index"] == 0
    assert first["idempotency_key"] == "seq-gate:proposal-first"
    assert first["execution_gate_handler"] == (
        "gate_operator_proposal_apply_sequence_execution"
    )
    assert first["execution_gate_invoked"] is False
    assert first["live_authority_granted"] is False


def test_dry_run_result_consumer_blocks_when_dry_run_has_blocked_rows():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer(
            sequence_plan_result_consumer=_sequence_plan_result_consumer(),
            dry_run_result=_dry_run_payload(
                results=[
                    _dry_run_result("proposal-first", index=0),
                    _dry_run_result("proposal-second", index=1, status="blocked"),
                ]
            ),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_dry_run_result_consumer_blocked"
    )
    assert consumer["summary"]["would_apply_result_count"] == 1
    assert consumer["summary"]["blocked_dry_run_result_count"] == 1
    assert consumer["summary"]["blocked_dry_run_proposal_ids"] == ["proposal-second"]
    assert consumer["summary"]["execution_gate_request_available"] is False
    assert consumer["execution_gate_request"]["dry_run_payload"] == {}
    assert consumer["summary"]["live_authority_granted"] is False


def test_dry_run_result_consumer_rejects_forged_would_apply_row():
    dry_run = _dry_run_payload()
    dry_run["dry_run_results"][0]["apply_request"]["idempotency_key"] = "forged"
    dry_run["dry_run_results"][0]["normalized_changes"][0]["effective_value"] = 1.0

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer(
            sequence_plan_result_consumer=_sequence_plan_result_consumer(),
            dry_run_result=dry_run,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_dry_run_result_consumer_replacement_required"
    )
    assert consumer["summary"]["execution_gate_request_available"] is False
    assert (
        "proposal-first:apply_request_idempotency_key_mismatch"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "proposal-first:change_fader_ceiling_exceeded"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert consumer["execution_gate_request"]["dry_run_payload"] == {}
    assert consumer["summary"]["live_write"] is False


def test_dry_run_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_DRY_RUN_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_DRY_RUN_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["execution_gate_created"] is False
    assert consumer["summary"]["execution_gate_request_available"] is False


def test_dry_run_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_DRY_RUN_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
