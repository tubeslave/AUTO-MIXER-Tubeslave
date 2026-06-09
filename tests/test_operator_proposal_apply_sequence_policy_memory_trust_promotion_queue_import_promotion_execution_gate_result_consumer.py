from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_GATE_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer import (
    _dry_run_payload,
    _promotion_sequence_plan_consumer_ready,
)


def _promotion_dry_run_consumer_ready(tmp_path):
    prior_consumer = _promotion_sequence_plan_consumer_ready(tmp_path)
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer(
            promotion_sequence_plan_result_consumer=prior_consumer,
            promotion_dry_run_result=_dry_run_payload(prior_consumer),
        )
    )


def _gate_candidate(proposal_id, *, index, prefix, status="ready_for_executor"):
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


def _gate_payload(prior_consumer, *, candidates=None, blocked=None, status="ready"):
    prefix = prior_consumer["summary"]["idempotency_key_prefix"]
    ids = list(prior_consumer["summary"]["would_apply_proposal_ids"])
    ready_for_executor = list(
        candidates
        if candidates is not None
        else [
            _gate_candidate(
                proposal_id,
                index=index,
                prefix=prefix,
            )
            for index, proposal_id in enumerate(ids)
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
            "input_dry_run_result_count": len(ids),
            "ready_for_executor_count": len(ready_for_executor),
            "blocked_count": len(blocked_results),
            "ready_for_executor_proposal_ids": [
                item["proposal_id"] for item in ready_for_executor
            ],
            "blocked_proposal_ids": [
                item["proposal_id"]
                for item in blocked_results
                if item.get("proposal_id")
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


def test_promotion_execution_gate_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer.py"
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


def test_promotion_execution_gate_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer()
    )

    assert consumer["type"] == (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_GATE_RESULT_CONSUMER_METHOD_ID
    )
    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer_not_ready"
    )
    assert consumer["safety"]["read_only"] is True
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["queue_mutation"] is False
    assert consumer["safety"]["idempotency_reserved"] is False
    assert consumer["summary"]["promotion_executor_request_available"] is False
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["promotion_executor_request"]["execution_gate_payload"] == {}


def test_promotion_execution_gate_result_consumer_emits_executor_request(tmp_path):
    prior_consumer = _promotion_dry_run_consumer_ready(tmp_path)
    gate = _gate_payload(prior_consumer)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer(
            promotion_dry_run_result_consumer=prior_consumer,
            promotion_execution_gate_result=gate,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_executor_request_available"
    )
    assert consumer["summary"]["expected_gate_candidate_count"] == 1
    assert consumer["summary"]["ready_for_executor_count"] == 1
    assert consumer["summary"]["blocked_gate_result_count"] == 0
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["promotion_executor_request_available"] is True
    assert consumer["summary"]["sequence_executor_handler_invoked"] is False
    assert consumer["summary"]["sequence_executor_created"] is False
    assert consumer["summary"]["execution_allowed_by_consumer"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["summary"]["live_authority_granted"] is False
    assert consumer["generic_execution_gate_result_consumer"]["summary"][
        "executor_request_available"
    ] is True

    request = consumer["promotion_executor_request"]
    assert request["handler"] == "execute_operator_proposal_apply_sequence"
    assert request["execute"] is False
    assert request["operator_ack"] is False
    assert request["approved"] is False
    assert request["operator_must_set_execute_true"] is True
    assert request["operator_must_set_operator_ack_true"] is True
    assert request["operator_must_set_approved_true"] is True
    assert request["read_only_template"] is True
    assert request["promotion_execution_gate_result_consumer_only"] is True
    assert request["promotion_executor_request_template_only"] is True
    assert len(request["execution_gate_payload"]["ready_for_executor"]) == 1
    assert request["sequence_executor_created"] is False
    assert request["idempotency_reserved"] is False
    assert request["live_authority_granted"] is False
    spec = consumer["promotion_executor_request_specs"][0]
    assert spec["proposal_id"] == "proposal-first"
    assert spec["sequence_executor_handler_invoked"] is False
    assert spec["sequence_executor_created"] is False
    assert spec["live_authority_granted"] is False


def test_promotion_execution_gate_result_consumer_blocks_blocked_rows(tmp_path):
    prior_consumer = _promotion_dry_run_consumer_ready(tmp_path)
    blocked = [
        {
            "planned_apply_index": 0,
            "proposal_id": "proposal-first",
            "dry_run_result_index": 0,
            "status": "blocked",
            "reason": "operator_mode_blocks_live_write",
        }
    ]

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer(
            promotion_dry_run_result_consumer=prior_consumer,
            promotion_execution_gate_result=_gate_payload(
                prior_consumer,
                candidates=[],
                blocked=blocked,
                status="blocked",
            ),
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer_blocked"
    )
    assert consumer["summary"]["ready_for_executor_count"] == 0
    assert consumer["summary"]["blocked_gate_result_count"] == 1
    assert consumer["summary"]["promotion_executor_request_available"] is False
    assert consumer["promotion_executor_request"]["execution_gate_payload"] == {}
    assert consumer["summary"]["live_authority_granted"] is False


def test_promotion_execution_gate_result_consumer_rejects_forged_gate(tmp_path):
    prior_consumer = _promotion_dry_run_consumer_ready(tmp_path)
    gate = _gate_payload(prior_consumer)
    gate["ready_for_executor"][0]["apply_request"]["idempotency_key"] = "forged"
    gate["ready_for_executor"][0]["normalized_changes"][0]["effective_value"] = 1.0

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer(
            promotion_dry_run_result_consumer=prior_consumer,
            promotion_execution_gate_result=gate,
        )
    )

    assert consumer["status"] == (
        "sequence_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer_replacement_required"
    )
    assert consumer["summary"]["promotion_executor_request_available"] is False
    invalid = consumer["summary"]["invalid_record_ids"]
    assert any(
        item.endswith("proposal-first:apply_request_idempotency_key_mismatch")
        for item in invalid
    )
    assert any(
        item.endswith("proposal-first:change_fader_ceiling_exceeded")
        for item in invalid
    )
    assert consumer["promotion_executor_request"]["execution_gate_payload"] == {}
    assert consumer["summary"]["live_write"] is False


def test_promotion_execution_gate_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_GATE_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_GATE_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["sequence_executor_created"] is False
    assert consumer["summary"]["promotion_executor_request_available"] is False


def test_promotion_execution_gate_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_GATE_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
