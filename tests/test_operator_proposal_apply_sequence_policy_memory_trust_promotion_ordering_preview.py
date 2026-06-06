from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer import (
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
        "generated_at": "2026-06-06T15:45:00Z",
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


def _preview_proposals():
    return [
        {
            "proposal_id": "proposal-other",
            "method_id": "safe_gain",
            "kind": "gain_trim_down",
            "target": "ch/03",
            "severity": "critical",
            "confidence": 1.0,
        },
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
    ]


def test_sequence_policy_ordering_preview_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview.py"
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


def test_sequence_policy_ordering_preview_not_ready_without_consumer_records():
    preview = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview(
            proposals=[],
        )
    )

    assert (
        preview["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview"
    )
    assert (
        preview["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_METHOD_ID
    )
    assert (
        preview["status"]
        == "sequence_memory_trust_promotion_ordering_preview_not_ready"
    )
    assert preview["safety"]["read_only"] is True
    assert preview["safety"]["ordering_preview_only"] is True
    assert preview["safety"]["proposal_order_mutation"] is False
    assert preview["safety"]["queue_mutation"] is False
    assert preview["safety"]["sequence_execution_allowed_by_ordering_preview"] is False
    assert preview["safety"]["sequence_policy_change_allowed_by_ordering_preview"] is False
    assert preview["summary"]["ordered_preview_count"] == 0
    assert preview["summary"]["can_import_to_queue"] is False
    assert preview["summary"]["can_apply_live"] is False
    assert preview["ordered_proposals"] == []


def test_sequence_policy_ordering_preview_recommends_order_without_mutation():
    proposals = _preview_proposals()
    original = deepcopy(proposals)
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer(
            proposals=proposals,
            sequence_policy_gate=_sequence_policy_gate(),
        )
    )

    preview = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview(
            proposals=proposals,
            policy_consumer=consumer,
        )
    )

    assert proposals == original
    assert (
        preview["status"]
        == "sequence_memory_trust_promotion_ordering_preview_available"
    )
    assert preview["summary"]["input_proposal_count"] == 2
    assert preview["summary"]["consumer_record_count"] == 1
    assert preview["summary"]["ordered_preview_count"] == 2
    assert preview["summary"]["policy_ranked_proposal_count"] == 1
    assert preview["summary"]["unranked_proposal_count"] == 1
    assert preview["summary"]["recommended_ordered_proposal_ids"] == [
        "proposal-sequence-fader",
        "proposal-other",
    ]
    assert preview["summary"]["policy_ranked_proposal_ids"] == [
        "proposal-sequence-fader",
    ]
    assert preview["summary"]["unranked_proposal_ids"] == ["proposal-other"]
    assert preview["summary"]["proposal_order_mutation"] is False
    assert preview["summary"]["queue_mutation"] is False
    assert preview["summary"]["sequence_execution_allowed_by_ordering_preview"] is False
    assert preview["summary"]["sequence_policy_change_allowed_by_ordering_preview"] is False
    assert preview["summary"]["live_widening_allowed"] is False

    ranked = preview["ordered_proposals"][0]
    unranked = preview["ordered_proposals"][1]
    assert ranked["proposal_id"] == "proposal-sequence-fader"
    assert ranked["original_index"] == 1
    assert ranked["recommended_index"] == 0
    assert ranked["rank_source"] == "sequence_policy_consumer"
    assert ranked["read_only"] is True
    assert ranked["ordering_preview_only"] is True
    assert ranked["policy_consumer_required_before_queue_mutation"] is True
    assert ranked["sequence_consumer_required_before_execution"] is True
    assert ranked["proposal_order_mutation"] is False
    assert ranked["queue_mutation"] is False
    assert ranked["model_mutation"] is False
    assert ranked["runtime_memory_mutation"] is False
    assert ranked["sequence_policy_trust_store_mutation"] is False
    assert ranked["proposal_state_mutation"] is False
    assert ranked["sequence_execution_allowed_by_ordering_preview"] is False
    assert ranked["sequence_policy_change_allowed_by_ordering_preview"] is False
    assert ranked["live_widening_allowed"] is False
    assert ranked["live_write"] is False
    assert ranked["live_authority_granted"] is False
    assert ranked["full_autonomous_ready"] is False
    assert ranked["can_import_to_queue"] is False
    assert ranked["can_apply_live"] is False
    assert unranked["proposal_id"] == "proposal-other"
    assert unranked["original_index"] == 0
    assert unranked["recommended_index"] == 1
    assert unranked["rank_source"] == "original_order"


def test_sequence_policy_ordering_preview_rejects_forged_consumer_authority():
    proposals = _preview_proposals()
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer(
            proposals=proposals,
            sequence_policy_gate=_sequence_policy_gate(),
        )
    )
    consumer["safety"]["proposal_order_mutation"] = True
    consumer["summary"]["sequence_execution_allowed_by_consumer"] = True
    consumer["consumer_records"][0]["can_apply_live"] = True

    preview = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview(
            proposals=proposals,
            policy_consumer=consumer,
        )
    )

    assert (
        preview["status"]
        == "sequence_memory_trust_promotion_ordering_preview_replacement_required"
    )
    assert preview["summary"]["consumer_record_count"] == 0
    assert preview["summary"]["ordered_preview_count"] == 0
    assert preview["ordered_proposals"] == []
    assert (
        "sequence_policy_ordering_preview_consumer_proposal_order_mutation_not_false"
        in preview["summary"]["invalid_record_ids"]
    )
    assert (
        "sequence_policy_ordering_preview_consumer_summary_sequence_execution_allowed_by_consumer_not_false"
        in preview["summary"]["invalid_record_ids"]
    )
    assert (
        "operator_sequence_policy_memory_trust_promotion_policy_consumer:proposal-sequence-fader"
        in preview["summary"]["invalid_record_ids"]
    )


def test_sequence_policy_ordering_preview_rejects_duplicate_consumer_records():
    proposals = _preview_proposals()
    consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer(
            proposals=proposals,
            sequence_policy_gate=_sequence_policy_gate(),
        )
    )
    duplicate = dict(consumer["consumer_records"][0])
    consumer["consumer_records"].append(duplicate)
    consumer["summary"]["consumer_record_count"] = 2
    consumer["summary"]["policy_consumer_proposal_ids"] = [
        duplicate["proposal_id"],
        duplicate["proposal_id"],
    ]

    preview = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview(
            proposals=proposals,
            policy_consumer=consumer,
        )
    )

    assert (
        preview["status"]
        == "sequence_memory_trust_promotion_ordering_preview_replacement_required"
    )
    assert preview["summary"]["ordered_preview_count"] == 0
    assert duplicate["consumer_record_id"] in preview["summary"]["invalid_record_ids"]


def test_sequence_policy_ordering_preview_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_METHOD_ID
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
    preview = snapshot[
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_METHOD_ID
    ]
    assert preview["safety"]["live_write"] is False
    assert preview["safety"]["proposal_order_mutation"] is False
    assert preview["summary"]["ordered_preview_count"] == 0


def test_sequence_policy_ordering_preview_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
