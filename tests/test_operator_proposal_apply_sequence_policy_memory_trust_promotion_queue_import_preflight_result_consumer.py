from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


def _imported_action(proposal_id, *, index, can_apply=True, status="pending"):
    return {
        "id": proposal_id,
        "title": f"Imported {proposal_id}",
        "source": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID
        ),
        "status": status,
        "can_apply": can_apply,
        "raw": {
            "sequence_policy_queue_import": {
                "queue_ordering_plan_record_id": f"plan:{proposal_id}",
                "ordering_preview_record_id": f"preview:{proposal_id}",
                "consumer_record_id": f"consumer:{proposal_id}",
                "sequence_policy_gate_record_id": f"gate:{proposal_id}",
                "policy_key": f"{proposal_id}/memory-write:{proposal_id}",
                "planned_queue_index": index,
                "rank_source": "sequence_policy_consumer",
                "recommended_sequence_policy_bucket": "trusted_sequence_policy",
                "operator_ack_required_before_queue_import": True,
                "operator_ack_required_before_sequence_execution": True,
                "sequence_consumer_required_before_execution": True,
                "live_write": False,
                "can_apply_live": False,
            }
        },
    }


def _preflight_plan(prefix="seq-import"):
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan(
            pending_actions=[
                _imported_action("proposal-first", index=0),
                _imported_action("proposal-second", index=1),
            ],
            idempotency_key_prefix=prefix,
        )
    )


def _preflight_result(proposal_id, *, prefix="seq-import", ready=True):
    return {
        "type": "operator_proposal_apply_preflight",
        "status": "ready" if ready else "blocked",
        "success": ready,
        "ready_to_apply": ready,
        "reason": None if ready else "delta_limit_exceeded",
        "proposal": {
            "id": proposal_id,
            "title": f"Ready {proposal_id}",
            "status": "pending",
        },
        "idempotency": {
            "required": True,
            "provided": True,
            "idempotency_key": f"{prefix}:{proposal_id}",
            "reserved": False,
            "would_reserve_on_apply": ready,
            "duplicate": False,
        },
        "normalized_change": {
            "value_type": "fader",
            "channel": 2,
            "address": "/ch/2/fdr",
            "effective_value": -8.5,
            "old_value": -8.0,
            "pre_read_value": -8.0,
            "delta_db": -0.5,
        },
        "normalized_changes": [
            {
                "value_type": "fader",
                "channel": 2,
                "address": "/ch/2/fdr",
                "effective_value": -8.5,
                "old_value": -8.0,
                "pre_read_value": -8.0,
                "delta_db": -0.5,
            }
        ],
        "safety": {
            "preflight_only": True,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "allowed_live_write_kinds": ["fader", "gain", "pan"],
            "live_apply_entrypoint": "apply_operator_proposal",
        },
    }


def _batch_result(prefix="seq-import", *, results=None):
    preflight_results = list(
        results
        if results is not None
        else [
            _preflight_result("proposal-first", prefix=prefix),
            _preflight_result("proposal-second", prefix=prefix),
        ]
    )
    ready_ids = [
        item["proposal"]["id"]
        for item in preflight_results
        if item.get("ready_to_apply") is True
    ]
    blocked_ids = [
        item["proposal"]["id"]
        for item in preflight_results
        if item.get("ready_to_apply") is not True
    ]
    return {
        "type": "operator_proposal_apply_preflight_batch",
        "status": "ok",
        "success": True,
        "generated_at": 123.0,
        "summary": {
            "selected_count": len(preflight_results),
            "preflight_result_count": len(preflight_results),
            "ready_count": len(ready_ids),
            "blocked_count": len(blocked_ids),
            "skipped_analysis_only_count": 0,
            "not_found_count": 0,
            "ready_proposal_ids": ready_ids,
            "blocked_proposal_ids": blocked_ids,
            "skipped_analysis_only_ids": [],
            "not_found_ids": [],
            "approved": True,
            "idempotency_key_prefix": prefix,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "idempotency_reserved": False,
        },
        "safety": {
            "batch_preflight_only": True,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "live_apply_entrypoint": "apply_operator_proposal",
        },
        "preflight_results": preflight_results,
    }


def test_sequence_policy_preflight_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer.py"
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


def test_sequence_policy_preflight_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer()
    )

    assert (
        consumer["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_RESULT_CONSUMER_METHOD_ID
    )
    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_preflight_result_consumer_not_ready"
    )
    assert consumer["summary"]["sequence_plan_request_available"] is False
    assert consumer["summary"]["sequence_plan_handler_invoked"] is False
    assert consumer["summary"]["sequence_plan_created"] is False
    assert consumer["summary"]["live_write"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["sequence_plan_request"]["preflight_results"] == []
    assert consumer["sequence_plan_request_specs"] == []


def test_sequence_policy_preflight_result_consumer_emits_sequence_plan_request_for_ready_batch():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer(
            queue_import_preflight_plan=_preflight_plan(),
            batch_preflight_result=_batch_result(),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_sequence_plan_request_available"
    )
    assert consumer["summary"]["planned_preflight_candidate_count"] == 2
    assert consumer["summary"]["batch_preflight_result_count"] == 2
    assert consumer["summary"]["ready_preflight_result_count"] == 2
    assert consumer["summary"]["blocked_preflight_result_count"] == 0
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["sequence_plan_request_available"] is True
    assert consumer["summary"]["sequence_plan_created"] is False
    assert consumer["summary"]["sequence_execution_allowed_by_consumer"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["summary"]["planned_preflight_candidate_proposal_ids"] == [
        "proposal-first",
        "proposal-second",
    ]
    assert consumer["summary"]["batch_preflight_result_proposal_ids"] == [
        "proposal-first",
        "proposal-second",
    ]
    assert consumer["sequence_plan_request"]["handler"] == (
        "plan_operator_proposal_apply_sequence"
    )
    assert len(consumer["sequence_plan_request"]["preflight_results"]) == 2
    assert consumer["sequence_plan_request"]["sequence_plan_created"] is False
    assert (
        consumer["sequence_plan_request"]["sequence_execution_allowed_by_consumer"]
        is False
    )
    first = consumer["sequence_plan_request_specs"][0]
    assert first["proposal_id"] == "proposal-first"
    assert first["planned_sequence_index"] == 0
    assert first["idempotency_key"] == "seq-import:proposal-first"
    assert first["sequence_plan_handler"] == "plan_operator_proposal_apply_sequence"
    assert first["sequence_plan_created"] is False
    assert first["live_authority_granted"] is False


def test_sequence_policy_preflight_result_consumer_blocks_when_batch_has_blocked_rows():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer(
            queue_import_preflight_plan=_preflight_plan(),
            batch_preflight_result=_batch_result(
                results=[
                    _preflight_result("proposal-first"),
                    _preflight_result("proposal-second", ready=False),
                ]
            ),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_preflight_result_consumer_blocked"
    )
    assert consumer["summary"]["ready_preflight_result_count"] == 1
    assert consumer["summary"]["blocked_preflight_result_count"] == 1
    assert consumer["summary"]["blocked_preflight_result_proposal_ids"] == [
        "proposal-second"
    ]
    assert consumer["summary"]["sequence_plan_request_available"] is False
    assert consumer["sequence_plan_request"]["preflight_results"] == []
    assert consumer["summary"]["live_authority_granted"] is False


def test_sequence_policy_preflight_result_consumer_rejects_forged_order_and_safety():
    forged = _batch_result(
        results=[
            _preflight_result("proposal-second"),
            _preflight_result("proposal-first"),
        ]
    )
    forged["safety"]["live_apply_entrypoint"] = "_apply_manual_console_write"
    forged["preflight_results"][0]["safety"]["live_write"] = True
    forged["preflight_results"][0]["safety"][
        "live_apply_entrypoint"
    ] = "_apply_manual_console_write"

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer(
            queue_import_preflight_plan=_preflight_plan(),
            batch_preflight_result=forged,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_preflight_result_consumer_replacement_required"
    )
    assert consumer["summary"]["sequence_plan_request_available"] is False
    assert (
        "sequence_policy_preflight_result_consumer_result_order_mismatch"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "sequence_policy_preflight_result_consumer_batch_entrypoint_mismatch"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "proposal-second:preflight_result_live_write_not_false"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "proposal-second:preflight_result_live_apply_entrypoint_mismatch"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert consumer["sequence_plan_request"]["preflight_results"] == []
    assert consumer["summary"]["live_write"] is False


def test_sequence_policy_preflight_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["sequence_plan_created"] is False
    assert consumer["summary"]["sequence_plan_request_available"] is False


def test_sequence_policy_preflight_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
