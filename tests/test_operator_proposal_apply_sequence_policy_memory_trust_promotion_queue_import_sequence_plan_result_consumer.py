from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_SEQUENCE_PLAN_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer,
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


def _imported_action(proposal_id, *, index):
    return {
        "id": proposal_id,
        "title": f"Imported {proposal_id}",
        "source": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID
        ),
        "status": "pending",
        "can_apply": True,
        "raw": {
            "sequence_policy_queue_import": {
                "queue_ordering_plan_record_id": f"plan:{proposal_id}",
                "ordering_preview_record_id": f"preview:{proposal_id}",
                "consumer_record_id": f"consumer:{proposal_id}",
                "sequence_policy_gate_record_id": f"gate:{proposal_id}",
                "policy_key": f"{proposal_id}/memory-write:{proposal_id}",
                "source_idempotency_key": f"memory-write:{proposal_id}",
                "current_sequence_review_freshness": {
                    "status": "fresh",
                    "field_count": 4,
                    "fresh_field_count": 4,
                    "read_only": True,
                    "live_write": False,
                    "apply_attempted": False,
                    "queries_console": False,
                    "queue_mutation": False,
                    "policy_mutation": False,
                },
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


def _preflight_result(proposal_id, *, prefix="seq-plan"):
    return {
        "type": "operator_proposal_apply_preflight",
        "status": "ready",
        "success": True,
        "ready_to_apply": True,
        "reason": None,
        "proposal": {
            "id": proposal_id,
            "title": f"Ready {proposal_id}",
            "status": "pending",
            "can_apply": True,
        },
        "idempotency": {
            "required": True,
            "provided": True,
            "idempotency_key": f"{prefix}:{proposal_id}",
            "reserved": False,
            "would_reserve_on_apply": True,
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


def _batch_result(prefix="seq-plan"):
    preflight_results = [
        _preflight_result("proposal-first", prefix=prefix),
        _preflight_result("proposal-second", prefix=prefix),
    ]
    return {
        "type": "operator_proposal_apply_preflight_batch",
        "status": "ok",
        "success": True,
        "generated_at": 123.0,
        "summary": {
            "selected_count": 2,
            "preflight_result_count": 2,
            "ready_count": 2,
            "blocked_count": 0,
            "skipped_analysis_only_count": 0,
            "not_found_count": 0,
            "ready_proposal_ids": ["proposal-first", "proposal-second"],
            "blocked_proposal_ids": [],
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


def _preflight_result_consumer(prefix="seq-plan"):
    plan = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan(
        pending_actions=[
            _imported_action("proposal-first", index=0),
            _imported_action("proposal-second", index=1),
        ],
        idempotency_key_prefix=prefix,
    )
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer(
            queue_import_preflight_plan=plan,
            batch_preflight_result=_batch_result(prefix=prefix),
        )
    )


def _planned_step(proposal_id, *, index, prefix="seq-plan"):
    return {
        "planned_apply_index": index,
        "proposal_id": proposal_id,
        "proposal_title": f"Ready {proposal_id}",
        "idempotency_key": f"{prefix}:{proposal_id}",
        "approval_required": True,
        "requires_operator_apply": True,
        "live_apply_entrypoint": "apply_operator_proposal",
        "execution_allowed_by_plan": False,
        "current_proposal": {
            "id": proposal_id,
            "status": "pending",
            "can_apply": True,
        },
        "change_count": 1,
        "normalized_changes": [
            {
                "change_index": 0,
                "channel": 2,
                "value_type": "fader",
                "address": "/ch/2/fdr",
                "old_value": -8.0,
                "pre_read_value": -8.0,
                "effective_value": -8.5,
                "delta_db": -0.5,
                "rollback_value": -8.0,
                "rollback_expected": True,
            }
        ],
        "apply_request": {
            "type": "apply_operator_proposal",
            "proposal_id": proposal_id,
            "approved": True,
            "idempotency_key": f"{prefix}:{proposal_id}",
        },
    }


def _sequence_plan(prefix="seq-plan", *, steps=None, blocked_results=None):
    planned_steps = list(
        steps
        if steps is not None
        else [
            _planned_step("proposal-first", index=0, prefix=prefix),
            _planned_step("proposal-second", index=1, prefix=prefix),
        ]
    )
    blocked = list(blocked_results or [])
    return {
        "type": "operator_proposal_apply_sequence_plan",
        "status": "ok",
        "success": True,
        "reason": None,
        "generated_at": 123.0,
        "summary": {
            "input_preflight_result_count": 2,
            "ready_result_count": 2,
            "blocked_result_count": 0,
            "planned_apply_count": len(planned_steps),
            "plan_blocked_count": len(blocked),
            "planned_apply_proposal_ids": [
                item["proposal_id"] for item in planned_steps
            ],
            "blocked_proposal_ids": [
                item["proposal_id"] for item in blocked if item.get("proposal_id")
            ],
            "operator_mode_allows_apply_now": True,
            "execution_allowed_by_plan": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "mutates_console_state": False,
            "queries_console": False,
            "live_widening_allowed": False,
            "queue_freshness_checked": True,
        },
        "safety": {
            "sequence_plan_only": True,
            "preflight_results_required": True,
            "current_queue_proposal_required": True,
            "preflight_result_is_not_apply_authority": True,
            "requires_operator_apply": True,
            "execution_allowed_by_plan": False,
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
        "planned_apply_sequence": planned_steps,
        "blocked_results": blocked,
    }


def test_sequence_plan_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer.py"
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


def test_sequence_plan_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer()
    )

    assert (
        consumer["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_SEQUENCE_PLAN_RESULT_CONSUMER_METHOD_ID
    )
    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_sequence_plan_result_consumer_not_ready"
    )
    assert consumer["summary"]["dry_run_request_available"] is False
    assert consumer["summary"]["dry_run_handler_invoked"] is False
    assert consumer["summary"]["dry_run_created"] is False
    assert consumer["summary"]["live_write"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["dry_run_request"]["sequence_plan"] == {}
    assert consumer["dry_run_request_specs"] == []


def test_sequence_plan_result_consumer_emits_dry_run_request_for_ready_plan():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer(
            preflight_result_consumer=_preflight_result_consumer(),
            sequence_plan_result=_sequence_plan(),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_dry_run_request_available"
    )
    assert consumer["summary"]["expected_sequence_plan_step_count"] == 2
    assert consumer["summary"]["sequence_plan_step_count"] == 2
    assert consumer["summary"]["blocked_sequence_plan_step_count"] == 0
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["dry_run_request_available"] is True
    assert consumer["summary"]["dry_run_created"] is False
    assert consumer["summary"]["execution_allowed_by_consumer"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["summary"]["expected_sequence_plan_proposal_ids"] == [
        "proposal-first",
        "proposal-second",
    ]
    assert consumer["summary"]["sequence_plan_step_proposal_ids"] == [
        "proposal-first",
        "proposal-second",
    ]
    assert consumer["dry_run_request"]["handler"] == (
        "dry_run_operator_proposal_apply_sequence"
    )
    assert consumer["dry_run_request"]["sequence_plan"]["type"] == (
        "operator_proposal_apply_sequence_plan"
    )
    assert consumer["dry_run_request"]["dry_run_created"] is False
    assert consumer["dry_run_request"]["execution_allowed_by_consumer"] is False
    first = consumer["dry_run_request_specs"][0]
    assert first["proposal_id"] == "proposal-first"
    assert first["planned_dry_run_index"] == 0
    assert first["source_sequence_plan_request_spec_id"].endswith(":proposal-first")
    assert first["source_preflight_request_spec_id"].endswith(":proposal-first")
    assert first["queue_ordering_plan_record_id"] == "plan:proposal-first"
    assert first["consumer_record_id"] == "consumer:proposal-first"
    assert first["sequence_policy_gate_record_id"] == "gate:proposal-first"
    assert first["policy_key"] == "proposal-first/memory-write:proposal-first"
    assert first["source_idempotency_key"] == "memory-write:proposal-first"
    assert first["current_sequence_review_freshness"] == _preflight_result_consumer()[
        "sequence_plan_request_specs"
    ][0]["current_sequence_review_freshness"]
    assert first["rank_source"] == "sequence_policy_consumer"
    assert first["idempotency_key"] == "seq-plan:proposal-first"
    assert first["dry_run_handler"] == "dry_run_operator_proposal_apply_sequence"
    assert first["dry_run_invoked"] is False
    assert first["live_authority_granted"] is False
    assert consumer["dry_run_request"]["sequence_plan_request_specs"][0][
        "source_idempotency_key"
    ] == "memory-write:proposal-first"
    assert consumer["dry_run_request"]["sequence_plan_request_specs"][0][
        "current_sequence_review_freshness"
    ]["status"] == "fresh"


def test_sequence_plan_result_consumer_blocks_when_sequence_plan_has_blocked_rows():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer(
            preflight_result_consumer=_preflight_result_consumer(),
            sequence_plan_result=_sequence_plan(
                steps=[_planned_step("proposal-first", index=0)],
                blocked_results=[
                    {
                        "proposal_id": "proposal-second",
                        "preflight_result_index": 1,
                        "reason": "proposal_changed_since_preflight",
                    }
                ],
            ),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_sequence_plan_result_consumer_blocked"
    )
    assert consumer["summary"]["sequence_plan_step_count"] == 1
    assert consumer["summary"]["blocked_sequence_plan_step_count"] == 1
    assert consumer["summary"]["blocked_sequence_plan_proposal_ids"] == [
        "proposal-second"
    ]
    assert consumer["summary"]["dry_run_request_available"] is False
    assert consumer["dry_run_request"]["sequence_plan"] == {}
    assert consumer["summary"]["live_authority_granted"] is False


def test_sequence_plan_result_consumer_rejects_forged_step_authority():
    plan = _sequence_plan()
    plan["planned_apply_sequence"][0]["execution_allowed_by_plan"] = True
    plan["planned_apply_sequence"][0]["normalized_changes"][0][
        "effective_value"
    ] = 1.0

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer(
            preflight_result_consumer=_preflight_result_consumer(),
            sequence_plan_result=plan,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_sequence_plan_result_consumer_replacement_required"
    )
    assert consumer["summary"]["dry_run_request_available"] is False
    assert (
        "proposal-first:execution_allowed_by_plan_not_false"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "proposal-first:change_fader_ceiling_exceeded"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert consumer["dry_run_request"]["sequence_plan"] == {}
    assert consumer["summary"]["live_write"] is False


def test_sequence_plan_result_consumer_rejects_forged_sequence_request_identity():
    preflight_consumer = _preflight_result_consumer()
    preflight_consumer["sequence_plan_request_specs"][0][
        "source_idempotency_key"
    ] = "forged-memory-write:proposal-first"
    preflight_consumer["sequence_plan_request_specs"][1][
        "sequence_policy_gate_record_id"
    ] = ""

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer(
            preflight_result_consumer=preflight_consumer,
            sequence_plan_result=_sequence_plan(),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_sequence_plan_result_consumer_replacement_required"
    )
    assert consumer["summary"]["dry_run_request_available"] is False
    assert (
        "proposal-first:policy_key_identity_mismatch"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "proposal-second:sequence_policy_gate_record_id_missing"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert consumer["dry_run_request"]["sequence_plan"] == {}
    assert consumer["dry_run_request"]["sequence_plan_request_specs"] == []
    assert consumer["dry_run_request_specs"] == []


def test_sequence_plan_result_consumer_rejects_missing_sequence_context():
    preflight_consumer = _preflight_result_consumer()
    for spec in preflight_consumer["sequence_plan_request_specs"]:
        spec.pop("current_sequence_review_freshness", None)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer(
            preflight_result_consumer=preflight_consumer,
            sequence_plan_result=_sequence_plan(),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_sequence_plan_result_consumer_replacement_required"
    )
    assert consumer["summary"]["dry_run_request_available"] is False
    assert (
        "proposal-first:current_sequence_review_freshness_missing"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "proposal-second:current_sequence_review_freshness_missing"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert consumer["dry_run_request"]["sequence_plan"] == {}
    assert consumer["dry_run_request"]["sequence_plan_request_specs"] == []
    assert consumer["dry_run_request_specs"] == []


def test_sequence_plan_result_consumer_rejects_live_sequence_context():
    preflight_consumer = _preflight_result_consumer()
    preflight_consumer["sequence_plan_request_specs"][0][
        "current_sequence_review_freshness"
    ] = {
        "status": "fresh",
        "field_count": 4,
        "fresh_field_count": 4,
        "read_only": False,
        "live_write": True,
        "apply_attempted": True,
        "queries_console": True,
        "queue_mutation": True,
        "policy_mutation": True,
    }

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer(
            preflight_result_consumer=preflight_consumer,
            sequence_plan_result=_sequence_plan(),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_sequence_plan_result_consumer_replacement_required"
    )
    assert consumer["summary"]["dry_run_request_available"] is False
    assert (
        "proposal-first:current_sequence_review_freshness_read_only_not_true"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "proposal-first:current_sequence_review_freshness_live_write_not_false"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "proposal-first:current_sequence_review_freshness_apply_attempted_not_false"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert consumer["dry_run_request"]["sequence_plan"] == {}
    assert consumer["dry_run_request"]["sequence_plan_request_specs"] == []
    assert consumer["dry_run_request_specs"] == []


def test_sequence_plan_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_SEQUENCE_PLAN_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_SEQUENCE_PLAN_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["dry_run_created"] is False
    assert consumer["summary"]["dry_run_request_available"] is False


def test_sequence_plan_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_SEQUENCE_PLAN_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
