from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_SEQUENCE_POLICY_GATE_METHOD_ID,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


def _sequence_context():
    return {
        "status": "fresh",
        "field_count": 4,
        "fresh_field_count": 4,
        "read_only": True,
        "live_write": False,
        "apply_attempted": False,
        "queries_console": False,
        "queue_mutation": False,
        "policy_mutation": False,
    }


def _unsafe_sequence_context():
    return {
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


def _eligible_gate_record(proposal_id="proposal-sequence-fader"):
    return {
        "sequence_policy_gate_record_id": (
            "operator_sequence_memory_trust_promotion_sequence_policy_gate:apply-audit-1"
        ),
        "proposal_id": proposal_id,
        "learning_scope": "supervised_fader_gain_sequence_execution",
        "value_type": "fader",
        "source_idempotency_key": f"memory-write:{proposal_id}",
        "policy_key": f"{proposal_id}/memory-write:{proposal_id}",
        "target_policy_state": "trusted_for_sequence_policy",
        "sequence_policy_gate_status": "trusted",
        "current_sequence_review_freshness": _sequence_context(),
        "rollback_verified": True,
        "sequence_policy_eligible": True,
        "eligible_for_sequence_policy_consumer": True,
        "policy_gate_only": True,
        "sequence_policy_consumer_required": True,
        "read_only": True,
        "live_write": False,
        "apply_attempted": False,
        "mutates_console_state": False,
        "queries_console": False,
        "queue_mutation": False,
        "model_mutation": False,
        "runtime_memory_mutation": False,
        "sequence_policy_trust_store_mutation": False,
        "proposal_state_mutation": False,
        "sequence_execution_allowed_by_gate": False,
        "sequence_policy_change_allowed_by_gate": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "can_import_to_queue": False,
        "can_apply_live": False,
    }


def _sequence_policy_gate(records=None):
    policy_records = list(records or [_eligible_gate_record()])
    eligible_keys = [
        record["policy_key"]
        for record in policy_records
        if record.get("sequence_policy_eligible") is True
    ]
    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate"
        ),
        "generated_at": "2026-06-06T15:30:00Z",
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_SEQUENCE_POLICY_GATE_METHOD_ID
        ),
        "status": "verified_sequence_memory_trust_promotion_policy_records_available",
        "safety": {
            "read_only": True,
            "policy_gate_only": True,
            "live_write": False,
            "apply_attempted": False,
            "mutates_console_state": False,
            "queries_console": False,
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "sequence_policy_trust_store_mutation": False,
            "proposal_state_mutation": False,
            "sequence_execution_allowed_by_gate": False,
            "sequence_policy_change_allowed_by_gate": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "sequence_policy_gate_record_count": len(policy_records),
            "eligible_record_count": len(eligible_keys),
            "invalid_record_count": 0,
            "sequence_policy_eligible": bool(eligible_keys),
            "policy_gate_only": True,
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "sequence_policy_trust_store_mutation": False,
            "proposal_state_mutation": False,
            "sequence_execution_allowed_by_gate": False,
            "sequence_policy_change_allowed_by_gate": False,
            "live_write": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
            "eligible_policy_keys": eligible_keys,
            "invalid_record_ids": [],
        },
        "policy_records": policy_records,
    }


def test_sequence_policy_consumer_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer.py"
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
        "apply_operator_proposal",
        "requested_change",
        "requested_changes",
        "write_text",
    ]
    for token in forbidden:
        assert token not in source


def test_sequence_policy_consumer_not_ready_without_gate_records():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer(
            proposals=[],
        )
    )

    assert (
        consumer["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer"
    )
    assert (
        consumer["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_METHOD_ID
    )
    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_policy_consumer_not_ready"
    )
    assert consumer["safety"]["read_only"] is True
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["proposal_order_mutation"] is False
    assert consumer["safety"]["queue_mutation"] is False
    assert consumer["safety"]["sequence_policy_change_allowed_by_consumer"] is False
    assert consumer["safety"]["sequence_execution_allowed_by_consumer"] is False
    assert consumer["summary"]["consumer_record_count"] == 0
    assert consumer["summary"]["can_import_to_queue"] is False
    assert consumer["summary"]["can_apply_live"] is False
    assert consumer["consumer_records"] == []


def test_sequence_policy_consumer_projects_records_read_only():
    proposals = [
        {
            "proposal_id": "proposal-sequence-fader",
            "method_id": "safe_gain",
            "kind": "gain_trim_down",
            "target": "ch/02",
            "severity": "high",
            "confidence": 0.82,
            "evidence": {
                "idempotency_key": "memory-write:proposal-sequence-fader",
            },
        },
        {
            "proposal_id": "proposal-other",
            "method_id": "safe_gain",
            "kind": "gain_trim_down",
            "target": "ch/03",
            "severity": "critical",
            "confidence": 1.0,
        },
    ]

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer(
            proposals=proposals,
            sequence_policy_gate=_sequence_policy_gate(),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_policy_consumer_records_available"
    )
    assert consumer["summary"]["input_proposal_count"] == 2
    assert consumer["summary"]["sequence_policy_gate_record_count"] == 1
    assert consumer["summary"]["consumer_record_count"] == 1
    assert consumer["summary"]["policy_consumer_proposal_ids"] == [
        "proposal-sequence-fader"
    ]
    assert consumer["summary"]["unmatched_eligible_policy_keys"] == []
    assert consumer["summary"]["proposal_order_mutation"] is False
    assert consumer["summary"]["queue_mutation"] is False
    assert consumer["summary"]["sequence_policy_change_allowed_by_consumer"] is False
    assert consumer["summary"]["sequence_execution_allowed_by_consumer"] is False
    assert consumer["summary"]["live_widening_allowed"] is False

    record = consumer["consumer_records"][0]
    assert record["proposal_id"] == "proposal-sequence-fader"
    assert (
        record["policy_key"]
        == "proposal-sequence-fader/memory-write:proposal-sequence-fader"
    )
    assert record["recommended_sequence_policy_bucket"] == "trusted_sequence_memory"
    assert record["current_sequence_review_freshness"]["status"] == "fresh"
    assert record["current_sequence_review_freshness"]["read_only"] is True
    assert record["current_sequence_review_freshness"]["live_write"] is False
    assert record["adjusted_policy_score"] > record["base_policy_score"]
    assert record["policy_consumer_required_before_ordering"] is True
    assert record["read_only"] is True
    assert record["policy_evidence_only"] is True
    assert record["proposal_order_mutation"] is False
    assert record["mutates_console_state"] is False
    assert record["queries_console"] is False
    assert record["queue_mutation"] is False
    assert record["model_mutation"] is False
    assert record["runtime_memory_mutation"] is False
    assert record["sequence_policy_trust_store_mutation"] is False
    assert record["proposal_state_mutation"] is False
    assert record["sequence_execution_allowed_by_consumer"] is False
    assert record["sequence_policy_change_allowed_by_consumer"] is False
    assert record["live_widening_allowed"] is False
    assert record["live_write"] is False
    assert record["live_authority_granted"] is False
    assert record["full_autonomous_ready"] is False
    assert record["can_import_to_queue"] is False
    assert record["can_apply_live"] is False


def test_sequence_policy_consumer_requires_proposal_identity_match():
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer(
            proposals=[
                {
                    "proposal_id": "proposal-unrelated-fader",
                    "method_id": "safe_gain",
                    "kind": "gain_trim_down",
                    "target": "ch/03",
                    "severity": "critical",
                    "confidence": 1.0,
                    "learning_scope": "supervised_fader_gain_sequence_execution",
                    "value_type": "fader",
                }
            ],
            sequence_policy_gate=_sequence_policy_gate(),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_policy_consumer_not_ready"
    )
    assert consumer["summary"]["input_proposal_count"] == 1
    assert consumer["summary"]["eligible_sequence_policy_gate_record_count"] == 1
    assert consumer["summary"]["consumer_record_count"] == 0
    assert consumer["summary"]["unmatched_eligible_policy_count"] == 1
    assert consumer["summary"]["unmatched_eligible_policy_keys"] == [
        "proposal-sequence-fader/memory-write:proposal-sequence-fader"
    ]
    assert "matching_preview_proposals_required_for_sequence_policy_consumer" in (
        consumer["summary"]["blocked_by"]
    )
    assert consumer["summary"]["proposal_order_mutation"] is False
    assert consumer["summary"]["queue_mutation"] is False
    assert (
        consumer["summary"]["sequence_policy_change_allowed_by_consumer"]
        is False
    )
    assert (
        consumer["summary"]["sequence_execution_allowed_by_consumer"]
        is False
    )
    assert consumer["summary"]["live_authority_granted"] is False
    assert consumer["consumer_records"] == []


def test_sequence_policy_consumer_requires_fresh_context():
    gate_record = _eligible_gate_record()
    gate_record.pop("current_sequence_review_freshness")

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer(
            proposals=[
                {
                    "proposal_id": "proposal-sequence-fader",
                    "method_id": "safe_gain",
                    "kind": "gain_trim_down",
                    "target": "ch/02",
                    "severity": "high",
                    "confidence": 0.82,
                    "evidence": {
                        "idempotency_key": "memory-write:proposal-sequence-fader",
                    },
                }
            ],
            sequence_policy_gate=_sequence_policy_gate([gate_record]),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_policy_consumer_replacement_required"
    )
    assert consumer["summary"]["consumer_record_count"] == 0
    assert consumer["consumer_records"] == []
    assert (
        "operator_sequence_memory_trust_promotion_sequence_policy_gate:apply-audit-1"
        in consumer["summary"]["invalid_record_ids"]
    )


def test_sequence_policy_consumer_rejects_live_authority_context():
    gate_record = {
        **_eligible_gate_record(),
        "current_sequence_review_freshness": _unsafe_sequence_context(),
    }

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer(
            proposals=[
                {
                    "proposal_id": "proposal-sequence-fader",
                    "method_id": "safe_gain",
                    "kind": "gain_trim_down",
                    "target": "ch/02",
                    "severity": "high",
                    "confidence": 0.82,
                    "evidence": {
                        "idempotency_key": "memory-write:proposal-sequence-fader",
                    },
                }
            ],
            sequence_policy_gate=_sequence_policy_gate([gate_record]),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_policy_consumer_replacement_required"
    )
    assert consumer["summary"]["consumer_record_count"] == 0
    assert consumer["consumer_records"] == []


def test_sequence_policy_consumer_rejects_forged_gate_authority():
    gate = _sequence_policy_gate()
    gate["safety"]["queries_console"] = True
    gate["summary"]["sequence_execution_allowed_by_gate"] = True
    gate["policy_records"][0]["can_apply_live"] = True

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer(
            proposals=[
                {
                    "proposal_id": "proposal-sequence-fader",
                    "evidence": {
                        "idempotency_key": "memory-write:proposal-sequence-fader",
                    },
                }
            ],
            sequence_policy_gate=gate,
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_policy_consumer_replacement_required"
    )
    assert consumer["summary"]["consumer_record_count"] == 0
    assert consumer["consumer_records"] == []
    assert (
        "sequence_policy_consumer_gate_queries_console_not_false"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "sequence_policy_consumer_gate_summary_sequence_execution_allowed_by_gate_not_false"
        in consumer["summary"]["invalid_record_ids"]
    )
    assert (
        "operator_sequence_memory_trust_promotion_sequence_policy_gate:apply-audit-1"
        in consumer["summary"]["invalid_record_ids"]
    )


def test_sequence_policy_consumer_rejects_duplicate_policy_keys():
    first = _eligible_gate_record("proposal-sequence-fader")
    second = {
        **_eligible_gate_record("proposal-sequence-fader-2"),
        "sequence_policy_gate_record_id": (
            "operator_sequence_memory_trust_promotion_sequence_policy_gate:apply-audit-2"
        ),
        "policy_key": first["policy_key"],
    }

    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer(
            proposals=[
                {
                    "proposal_id": "proposal-sequence-fader",
                    "evidence": {
                        "idempotency_key": "memory-write:proposal-sequence-fader",
                    },
                }
            ],
            sequence_policy_gate=_sequence_policy_gate([first, second]),
        )
    )

    assert (
        consumer["status"]
        == "sequence_memory_trust_promotion_policy_consumer_replacement_required"
    )
    assert consumer["summary"]["consumer_record_count"] == 0
    assert consumer["consumer_records"] == []
    assert set(consumer["summary"]["invalid_record_ids"]) >= {
        "operator_sequence_memory_trust_promotion_sequence_policy_gate:apply-audit-1",
        "operator_sequence_memory_trust_promotion_sequence_policy_gate:apply-audit-2",
    }


def test_sequence_policy_consumer_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_METHOD_ID
    ]
    assert consumer["safety"]["live_write"] is False
    assert consumer["safety"]["proposal_order_mutation"] is False
    assert consumer["summary"]["consumer_record_count"] == 0


def test_sequence_policy_consumer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
