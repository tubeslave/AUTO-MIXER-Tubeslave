from copy import deepcopy
from pathlib import Path

from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.source_identity_calibration_memory_trust_promotion_priority_gate import (
    SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PRIORITY_GATE_METHOD_ID,
)
from backend.source_identity_proposal_ordering_preview import (
    build_source_identity_proposal_ordering_preview,
)
from backend.source_identity_proposal_priority_policy import (
    build_source_identity_proposal_priority_policy,
)
from backend.source_identity_proposal_queue_ordering_plan import (
    SOURCE_IDENTITY_PROPOSAL_QUEUE_ORDERING_PLAN_METHOD_ID,
    build_source_identity_proposal_queue_ordering_plan,
)


def _eligible_priority_record(source_candidate_id="candidate-1"):
    return {
        "priority_gate_record_id": (
            "source_identity_memory_trust_promotion_priority_gate:apply-audit-1"
        ),
        "source_candidate_id": source_candidate_id,
        "source_type": "lead_vocal",
        "target_trust_state": "trusted_for_downstream_priority",
        "source_identity_downstream_gate_status": "trusted",
        "rollback_verified": True,
        "downstream_priority_eligible": True,
        "eligible_for_downstream_priority_policy": True,
        "policy_gate_only": True,
        "proposal_priority_consumer_required": True,
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "queue_mutation": False,
        "model_mutation": False,
        "runtime_memory_mutation": False,
        "source_identity_trust_store_mutation": False,
        "direct_downstream_priority_mutation": False,
        "downstream_priority_change_allowed_by_gate": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "can_import_to_queue": False,
        "can_apply_live": False,
    }


def _priority_gate(records=None):
    priority_records = list(records or [_eligible_priority_record()])
    eligible_ids = [
        record["source_candidate_id"]
        for record in priority_records
        if record.get("downstream_priority_eligible") is True
    ]
    return {
        "type": "source_identity_calibration_memory_trust_promotion_priority_gate",
        "generated_at": "2026-06-06T15:00:00Z",
        "method_id": (
            SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PRIORITY_GATE_METHOD_ID
        ),
        "status": "verified_memory_trust_promotion_priority_records_available",
        "safety": {
            "read_only": True,
            "priority_gate_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "source_identity_trust_store_mutation": False,
            "direct_downstream_priority_mutation": False,
            "downstream_priority_change_allowed_by_gate": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "priority_gate_record_count": len(priority_records),
            "eligible_record_count": len(eligible_ids),
            "invalid_record_count": 0,
            "downstream_priority_eligible": bool(eligible_ids),
            "policy_gate_only": True,
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "source_identity_trust_store_mutation": False,
            "direct_downstream_priority_mutation": False,
            "downstream_priority_change_allowed_by_gate": False,
            "live_write": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
            "eligible_source_candidate_ids": eligible_ids,
            "invalid_record_ids": [],
        },
        "priority_records": priority_records,
    }


def _preview_proposals():
    return [
        {
            "proposal_id": "proposal-other",
            "method_id": "safe_gain",
            "kind": "gain_trim_down",
            "target": "ch/02",
            "severity": "critical",
            "confidence": 1.0,
            "evidence": {"source_candidate_id": "candidate-2"},
        },
        {
            "proposal_id": "proposal-vocal-gain",
            "method_id": "safe_gain",
            "kind": "gain_trim_down",
            "target": "ch/01",
            "severity": "high",
            "confidence": 0.8,
            "evidence": {"source_candidate_id": "candidate-1"},
        },
    ]


def _ordering_preview(proposals=None):
    proposal_rows = _preview_proposals() if proposals is None else proposals
    policy = build_source_identity_proposal_priority_policy(
        proposals=proposal_rows,
        priority_gate=_priority_gate(),
    )
    return build_source_identity_proposal_ordering_preview(
        proposals=proposal_rows,
        priority_policy=policy,
    )


def test_source_identity_proposal_queue_ordering_plan_module_is_transport_free():
    source = Path("backend/source_identity_proposal_queue_ordering_plan.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "set_trim",
        "set_eq",
        "set_compressor",
        "route_output",
        "load_snap",
        "save_snap",
        "mute",
        "unmute",
        "requested_change",
        "requested_changes",
        "write_text",
    ]
    for token in forbidden:
        assert token not in source


def test_source_identity_proposal_queue_ordering_plan_not_ready_without_preview_rows():
    plan = build_source_identity_proposal_queue_ordering_plan()

    assert plan["type"] == "source_identity_proposal_queue_ordering_plan"
    assert plan["method_id"] == SOURCE_IDENTITY_PROPOSAL_QUEUE_ORDERING_PLAN_METHOD_ID
    assert plan["status"] == "source_identity_proposal_queue_ordering_plan_not_ready"
    assert plan["safety"]["read_only"] is True
    assert plan["safety"]["queue_ordering_plan_only"] is True
    assert plan["safety"]["proposal_order_mutation"] is False
    assert plan["safety"]["queue_mutation"] is False
    assert plan["safety"]["queue_import_allowed_by_plan"] is False
    assert plan["safety"]["queue_reorder_allowed_by_plan"] is False
    assert plan["safety"]["direct_downstream_priority_mutation"] is False
    assert plan["safety"]["downstream_priority_change_allowed_by_policy"] is False
    assert plan["safety"]["live_widening_allowed"] is False
    assert plan["summary"]["queue_ordering_plan_record_count"] == 0
    assert plan["summary"]["can_import_to_queue"] is False
    assert plan["summary"]["can_apply_live"] is False
    assert plan["queue_ordering_plan_records"] == []


def test_source_identity_proposal_queue_ordering_plan_builds_records_without_mutation():
    proposals = _preview_proposals()
    preview = _ordering_preview(proposals)
    original_preview = deepcopy(preview)

    plan = build_source_identity_proposal_queue_ordering_plan(
        ordering_preview=preview,
    )

    assert preview == original_preview
    assert plan["status"] == "source_identity_proposal_queue_ordering_plan_available"
    assert plan["summary"]["input_ordering_preview_count"] == 2
    assert plan["summary"]["queue_ordering_plan_record_count"] == 2
    assert plan["summary"]["ranked_queue_candidate_count"] == 1
    assert plan["summary"]["unranked_queue_candidate_count"] == 1
    assert plan["summary"]["planned_queue_proposal_ids"] == [
        "proposal-vocal-gain",
        "proposal-other",
    ]
    assert plan["summary"]["ranked_queue_candidate_proposal_ids"] == [
        "proposal-vocal-gain",
    ]
    assert plan["summary"]["unranked_queue_candidate_proposal_ids"] == [
        "proposal-other",
    ]
    assert plan["summary"]["queue_import_allowed_by_plan"] is False
    assert plan["summary"]["queue_reorder_allowed_by_plan"] is False
    assert plan["summary"]["queue_mutation"] is False
    assert plan["summary"]["can_import_to_queue"] is False
    assert plan["summary"]["can_apply_live"] is False

    ranked = plan["queue_ordering_plan_records"][0]
    unranked = plan["queue_ordering_plan_records"][1]
    assert ranked["proposal_id"] == "proposal-vocal-gain"
    assert ranked["planned_queue_index"] == 0
    assert ranked["rank_source"] == "source_identity_policy"
    assert ranked["operator_review_required"] is True
    assert ranked["operator_ack_required_before_queue_import"] is True
    assert ranked["idempotency_key_required"] is True
    assert ranked["queue_ordering_plan_only"] is True
    assert ranked["read_only"] is True
    assert ranked["proposal_order_mutation"] is False
    assert ranked["queue_mutation"] is False
    assert ranked["queue_import_allowed_by_plan"] is False
    assert ranked["queue_reorder_allowed_by_plan"] is False
    assert ranked["model_mutation"] is False
    assert ranked["runtime_memory_mutation"] is False
    assert ranked["source_identity_trust_store_mutation"] is False
    assert ranked["direct_downstream_priority_mutation"] is False
    assert ranked["downstream_priority_change_allowed_by_policy"] is False
    assert ranked["live_widening_allowed"] is False
    assert ranked["live_write"] is False
    assert ranked["live_authority_granted"] is False
    assert ranked["full_autonomous_ready"] is False
    assert ranked["can_import_to_queue"] is False
    assert ranked["can_apply_live"] is False
    assert unranked["proposal_id"] == "proposal-other"
    assert unranked["planned_queue_index"] == 1
    assert unranked["rank_source"] == "original_order"


def test_source_identity_proposal_queue_ordering_plan_rejects_forged_preview_authority():
    preview = _ordering_preview()
    preview["safety"]["can_import_to_queue"] = True
    preview["summary"]["queue_mutation"] = True
    preview["ordered_proposals"][0]["live_write"] = True

    plan = build_source_identity_proposal_queue_ordering_plan(
        ordering_preview=preview,
    )

    assert (
        plan["status"]
        == "source_identity_proposal_queue_ordering_plan_replacement_required"
    )
    assert plan["summary"]["queue_ordering_plan_record_count"] == 0
    assert plan["queue_ordering_plan_records"] == []
    assert (
        "source_identity_proposal_queue_ordering_plan_preview_can_import_to_queue_not_false"
        in plan["summary"]["invalid_record_ids"]
    )
    assert (
        "source_identity_proposal_queue_ordering_plan_preview_summary_queue_mutation_not_false"
        in plan["summary"]["invalid_record_ids"]
    )
    assert (
        "source_identity_proposal_ordering_preview:proposal-vocal-gain"
        in plan["summary"]["invalid_record_ids"]
    )


def test_source_identity_proposal_queue_ordering_plan_is_registered_non_live():
    method = get_processing_method(SOURCE_IDENTITY_PROPOSAL_QUEUE_ORDERING_PLAN_METHOD_ID)
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
    plan = snapshot[SOURCE_IDENTITY_PROPOSAL_QUEUE_ORDERING_PLAN_METHOD_ID]
    assert plan["safety"]["live_write"] is False
    assert plan["safety"]["queue_mutation"] is False
    assert plan["safety"]["queue_import_allowed_by_plan"] is False
    assert plan["summary"]["queue_ordering_plan_record_count"] == 0


def test_source_identity_proposal_queue_ordering_plan_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert SOURCE_IDENTITY_PROPOSAL_QUEUE_ORDERING_PLAN_METHOD_ID in row.method_ids
        assert (
            "backend/source_identity_proposal_queue_ordering_plan.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
