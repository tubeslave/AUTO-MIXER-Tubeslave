from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_DRY_RUN_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer import (
    _promotion_batch_preflight_consumer_ready,
    _sequence_plan,
)


def _promotion_sequence_plan_consumer_ready(tmp_path):
    prior_consumer = _promotion_batch_preflight_consumer_ready(tmp_path)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer(
            promotion_batch_preflight_result_consumer=prior_consumer,
            promotion_sequence_plan_result=_sequence_plan(prior_consumer),
        )
    )


def _dry_run_result(proposal_id, *, index, prefix, status="would_apply"):
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


def _dry_run_payload(prior_consumer, *, status="would_apply"):
    prefix = prior_consumer["summary"]["idempotency_key_prefix"]
    ids = list(prior_consumer["summary"]["sequence_plan_step_proposal_ids"])
    dry_run_results = [
        _dry_run_result(
            proposal_id,
            index=index,
            prefix=prefix,
            status=status,
        )
        for index, proposal_id in enumerate(ids)
    ]
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
            "input_step_count": len(ids),
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


def test_promotion_dry_run_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer.py"
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


def test_promotion_dry_run_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_DRY_RUN_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer_not_ready"
    )
    assert consumer["safety"]["read_only"] is True
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["queue_mutation"] is False
    assert consumer["safety"]["idempotency_reserved"] is False
    assert consumer["summary"]["promotion_execution_gate_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["promotion_execution_gate_request"]["dry_run_payload"] == {}


def test_promotion_dry_run_result_consumer_emits_execution_gate_request(tmp_path):
    prior_consumer = _promotion_sequence_plan_consumer_ready(tmp_path)
    dry_run = _dry_run_payload(prior_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer(
            promotion_sequence_plan_result_consumer=prior_consumer,
            promotion_dry_run_result=dry_run,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_execution_gate_request_available"
    )
    assert consumer["summary"]["expected_dry_run_result_count"] == 1
    assert consumer["summary"]["dry_run_result_count"] == 1
    assert consumer["summary"]["would_apply_result_count"] == 1
    assert consumer["summary"]["blocked_dry_run_result_count"] == 0
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_execution_gate_request_available"] is True
    assert consumer["summary"]["execution_gate_handler_invoked"] is False
    assert consumer["summary"]["execution_gate_created"] is False
    assert consumer["summary"]["execution_allowed_by_consumer"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    assert consumer["generic_dry_run_result_consumer"]["summary"][
        "execution_gate_request_available"
    ] is True

    request = consumer["promotion_execution_gate_request"]
    assert request["handler"] == "gate_operator_proposal_apply_sequence_execution"
    assert request["operator_ack"] is False
    assert request["operator_must_set_operator_ack_true"] is True
    assert request["read_only_template"] is True
    assert request["promotion_dry_run_result_consumer_only"] is True
    assert request["promotion_execution_gate_request_template_only"] is True
    assert len(request["dry_run_payload"]["dry_run_results"]) == 1
    assert request["execution_gate_created"] is False
    assert request["idempotency_reserved"] is False
    assert request["live_authority_granted"] is False
    spec = consumer["promotion_execution_gate_request_specs"][0]
    assert spec["proposal_id"] == "proposal-first"
    assert spec["execution_gate_invoked"] is False
    assert spec["live_authority_granted"] is False


def test_promotion_dry_run_result_consumer_blocks_blocked_rows(tmp_path):
    prior_consumer = _promotion_sequence_plan_consumer_ready(tmp_path)
    dry_run = _dry_run_payload(prior_consumer, status="blocked")

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer(
            promotion_sequence_plan_result_consumer=prior_consumer,
            promotion_dry_run_result=dry_run,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer_blocked"
    )
    assert consumer["summary"]["would_apply_result_count"] == 0
    assert consumer["summary"]["blocked_dry_run_result_count"] == 1
    assert consumer["summary"]["promotion_execution_gate_request_available"] is False
    assert consumer["promotion_execution_gate_request"]["dry_run_payload"] == {}


def test_promotion_dry_run_result_consumer_rejects_forged_dry_run(tmp_path):
    prior_consumer = _promotion_sequence_plan_consumer_ready(tmp_path)
    dry_run = _dry_run_payload(prior_consumer)
    dry_run["dry_run_results"][0]["apply_request"]["idempotency_key"] = "forged"
    dry_run["dry_run_results"][0]["normalized_changes"][0]["effective_value"] = 1.0

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer(
            promotion_sequence_plan_result_consumer=prior_consumer,
            promotion_dry_run_result=dry_run,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_execution_gate_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert any(
        item.endswith("proposal-first:apply_request_idempotency_key_mismatch")
        for item in invalid
    )
    assert any(
        item.endswith("proposal-first:change_fader_ceiling_exceeded")
        for item in invalid
    )


def test_promotion_dry_run_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_DRY_RUN_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_DRY_RUN_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["execution_gate_created"] is False
    assert consumer["summary"]["promotion_execution_gate_request_available"] is False


def test_promotion_dry_run_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_DRY_RUN_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
