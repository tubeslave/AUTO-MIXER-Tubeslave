from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_AUDIT_RESULT_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from tests.test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer import (
    _execution_gate_result_consumer,
    _execution_payload,
)


def _execution_result_consumer(prefix="seq-policy"):
    return (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer(
            execution_gate_result_consumer=_execution_gate_result_consumer(prefix=prefix),
            execution_result=_execution_payload(prefix=prefix),
        )
    )


def _audit_row(proposal_id, *, index, prefix="seq-policy", status="passed"):
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


def _audit_payload(prefix="seq-policy", *, rows=None, blocked=None, status="passed"):
    audit_rows = list(
        rows
        if rows is not None
        else [
            _audit_row("proposal-first", index=0, prefix=prefix),
            _audit_row("proposal-second", index=1, prefix=prefix),
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
            "input_execution_result_count": 2,
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
            "allowed_live_write_kinds": ["fader", "gain", "pan"],
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


def test_execution_audit_result_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer.py"
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


def test_execution_audit_result_consumer_not_ready_without_inputs():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer()
    )

    assert (
        consumer["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_AUDIT_RESULT_CONSUMER_METHOD_ID
    )
    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_execution_audit_result_consumer_not_ready"
    )
    assert consumer["summary"]["audit_policy_gate_request_available"] is False
    assert consumer["summary"]["audit_policy_gate_handler_invoked"] is False
    assert consumer["summary"]["policy_gate_created"] is False
    assert consumer["summary"]["live_write"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["audit_policy_gate_request"]["execution_audit_payload"] == {}
    assert consumer["audit_policy_gate_request_specs"] == []


def test_execution_audit_result_consumer_emits_policy_gate_request_for_passed_rows():
    prior_consumer = _execution_result_consumer()
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer(
            execution_result_consumer=prior_consumer,
            execution_audit_result=_audit_payload(),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_audit_policy_gate_request_available"
    )
    assert consumer["summary"]["expected_execution_audit_row_count"] == 2
    assert consumer["summary"]["execution_audit_row_count"] == 2
    assert consumer["summary"]["passed_execution_audit_row_count"] == 2
    assert consumer["summary"]["blocked_execution_audit_row_count"] == 0
    assert consumer["summary"]["invalid_record_count"] == 0
    assert consumer["summary"]["audit_policy_gate_request_available"] is True
    assert consumer["summary"]["audit_policy_gate_handler_invoked"] is False
    assert consumer["summary"]["policy_gate_created"] is False
    assert consumer["summary"]["execution_allowed_by_consumer"] is False
    assert consumer["summary"]["idempotency_reserved"] is False
    assert consumer["summary"]["execution_audit_row_proposal_ids"] == [
        "proposal-first",
        "proposal-second",
    ]
    assert consumer["audit_policy_gate_request"]["handler"] == (
        "gate_operator_proposal_apply_sequence_audit_policy"
    )
    assert len(
        consumer["audit_policy_gate_request"]["execution_audit_payload"]["audit_rows"]
    ) == 2
    first = consumer["audit_policy_gate_request_specs"][0]
    assert first["proposal_id"] == "proposal-first"
    assert first["planned_policy_gate_index"] == 0
    assert first["source_audit_request_spec_id"].endswith(":proposal-first")
    assert first["source_executor_request_spec_id"].endswith(":proposal-first")
    assert first["source_execution_gate_request_spec_id"].endswith(":proposal-first")
    assert first["source_dry_run_request_spec_id"].endswith(":proposal-first")
    assert first["source_sequence_plan_request_spec_id"].endswith(":proposal-first")
    assert first["source_preflight_request_spec_id"].endswith(":proposal-first")
    assert first["queue_ordering_plan_record_id"] == "plan:proposal-first"
    assert first["consumer_record_id"] == "consumer:proposal-first"
    assert first["sequence_policy_gate_record_id"] == "gate:proposal-first"
    assert first["policy_key"] == "proposal-first/memory-write:proposal-first"
    assert first["source_idempotency_key"] == "memory-write:proposal-first"
    assert first["current_sequence_review_freshness"] == prior_consumer[
        "audit_request_specs"
    ][0]["current_sequence_review_freshness"]
    assert first["rank_source"] == "sequence_policy_consumer"
    assert first["idempotency_key"] == "seq-policy:proposal-first"
    assert first["audit_policy_gate_handler"] == (
        "gate_operator_proposal_apply_sequence_audit_policy"
    )
    assert first["audit_policy_gate_invoked"] is False
    assert first["live_authority_granted"] is False
    assert consumer["audit_policy_gate_request"]["execution_audit_payload"][
        "audit_request_specs"
    ][0]["source_idempotency_key"] == "memory-write:proposal-first"
    assert consumer["audit_policy_gate_request"]["execution_audit_payload"][
        "audit_request_specs"
    ][0]["current_sequence_review_freshness"]["status"] == "fresh"


def test_execution_audit_result_consumer_blocks_when_audit_has_blocked_rows():
    blocked = [
        {
            "proposal_id": "proposal-second",
            "execution_result_index": 1,
            "status": "blocked",
            "reason": "execution_row_readback_not_verified",
        }
    ]
    rows = [
        _audit_row("proposal-first", index=0),
        _audit_row("proposal-second", index=1, status="blocked"),
    ]
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer(
            execution_result_consumer=_execution_result_consumer(),
            execution_audit_result=_audit_payload(
                rows=rows,
                blocked=blocked,
                status="blocked",
            ),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_execution_audit_result_consumer_replacement_required"
    )
    assert consumer["summary"]["audit_policy_gate_request_available"] is False
    assert consumer["audit_policy_gate_request"]["execution_audit_payload"] == {}
    assert consumer["summary"]["live_authority_granted"] is False
    assert (
        "sequence_policy_execution_audit_result_consumer_audit_has_blocked_results"
        in consumer["summary"]["invalid_record_ids"]
    )


def test_execution_audit_result_consumer_rejects_forged_audit_row():
    audit = _audit_payload()
    audit["summary"]["live_authority_granted"] = True
    audit["safety"]["allowed_live_write_kinds"] = ["fader", "gain", "eq"]
    audit["audit_rows"][0]["idempotency_key"] = "forged"
    audit["audit_rows"][0]["effective_value"] = 1.0

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer(
            execution_result_consumer=_execution_result_consumer(),
            execution_audit_result=audit,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_execution_audit_result_consumer_replacement_required"
    )
    assert consumer["summary"]["audit_policy_gate_request_available"] is False
    assert (
        "sequence_policy_execution_audit_result_consumer_audit_summary_live_authority_granted_not_false"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "sequence_policy_execution_audit_result_consumer_audit_safety_allowed_kinds_mismatch"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert "proposal-first:idempotency_key_mismatch" in consumer["summary"][
        "invalid_record_ids"
    ]
    assert "proposal-first:fader_ceiling_exceeded" in consumer["summary"][
        "invalid_record_ids"
    ]
    assert consumer["audit_policy_gate_request"]["execution_audit_payload"] == {}
    assert consumer["summary"]["live_write"] is False


def test_execution_audit_result_consumer_rejects_forged_audit_request_identity():
    prior_consumer = _execution_result_consumer()
    prior_consumer["audit_request_specs"][0][
        "source_idempotency_key"
    ] = "forged-memory-write:proposal-first"
    prior_consumer["audit_request_specs"][1]["consumer_record_id"] = ""

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer(
            execution_result_consumer=prior_consumer,
            execution_audit_result=_audit_payload(),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_execution_audit_result_consumer_replacement_required"
    )
    assert consumer["summary"]["audit_policy_gate_request_available"] is False
    assert (
        "proposal-first:policy_key_identity_mismatch"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "proposal-second:consumer_record_id_missing"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert consumer["audit_policy_gate_request"]["execution_audit_payload"] == {}
    assert consumer["audit_policy_gate_request_specs"] == []


def test_execution_audit_result_consumer_rejects_missing_sequence_context():
    prior_consumer = _execution_result_consumer()
    for spec in prior_consumer["audit_request_specs"]:
        spec.pop("current_sequence_review_freshness", None)

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer(
            execution_result_consumer=prior_consumer,
            execution_audit_result=_audit_payload(),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_execution_audit_result_consumer_replacement_required"
    )
    assert consumer["summary"]["audit_policy_gate_request_available"] is False
    assert (
        "proposal-first:current_sequence_review_freshness_missing"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "proposal-second:current_sequence_review_freshness_missing"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert consumer["audit_policy_gate_request"]["execution_audit_payload"] == {}
    assert consumer["audit_policy_gate_request_specs"] == []


def test_execution_audit_result_consumer_rejects_live_sequence_context():
    prior_consumer = _execution_result_consumer()
    prior_consumer["audit_request_specs"][0][
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
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer(
            execution_result_consumer=prior_consumer,
            execution_audit_result=_audit_payload(),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_queue_import_execution_audit_result_consumer_replacement_required"
    )
    assert consumer["summary"]["audit_policy_gate_request_available"] is False
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
    assert consumer["audit_policy_gate_request"]["execution_audit_payload"] == {}
    assert consumer["audit_policy_gate_request_specs"] == []


def test_execution_audit_result_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_AUDIT_RESULT_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_AUDIT_RESULT_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["policy_gate_created"] is False
    assert consumer["summary"]["audit_policy_gate_request_available"] is False


def test_execution_audit_result_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_AUDIT_RESULT_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
