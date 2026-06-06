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
    SOURCE_IDENTITY_PROPOSAL_ORDERING_PREVIEW_METHOD_ID,
    build_source_identity_proposal_ordering_preview,
)
from backend.source_identity_proposal_priority_policy import (
    build_source_identity_proposal_priority_policy,
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


def test_source_identity_proposal_ordering_preview_module_is_transport_free():
    source = Path("backend/source_identity_proposal_ordering_preview.py").read_text()

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


def test_source_identity_proposal_ordering_preview_not_ready_without_policy_records():
    preview = build_source_identity_proposal_ordering_preview(proposals=[])

    assert preview["type"] == "source_identity_proposal_ordering_preview"
    assert preview["method_id"] == SOURCE_IDENTITY_PROPOSAL_ORDERING_PREVIEW_METHOD_ID
    assert preview["status"] == "source_identity_proposal_ordering_preview_not_ready"
    assert preview["safety"]["read_only"] is True
    assert preview["safety"]["ordering_preview_only"] is True
    assert preview["safety"]["proposal_order_mutation"] is False
    assert preview["safety"]["queue_mutation"] is False
    assert preview["safety"]["direct_downstream_priority_mutation"] is False
    assert preview["safety"]["downstream_priority_change_allowed_by_policy"] is False
    assert preview["safety"]["live_widening_allowed"] is False
    assert preview["summary"]["ordered_preview_count"] == 0
    assert preview["summary"]["can_import_to_queue"] is False
    assert preview["summary"]["can_apply_live"] is False
    assert preview["ordered_proposals"] == []


def test_source_identity_proposal_ordering_preview_recommends_order_without_mutation():
    proposals = _preview_proposals()
    original = deepcopy(proposals)
    policy = build_source_identity_proposal_priority_policy(
        proposals=proposals,
        priority_gate=_priority_gate(),
    )

    preview = build_source_identity_proposal_ordering_preview(
        proposals=proposals,
        priority_policy=policy,
    )

    assert proposals == original
    assert preview["status"] == "source_identity_proposal_ordering_preview_available"
    assert preview["summary"]["input_proposal_count"] == 2
    assert preview["summary"]["policy_record_count"] == 1
    assert preview["summary"]["ordered_preview_count"] == 2
    assert preview["summary"]["policy_ranked_proposal_count"] == 1
    assert preview["summary"]["unranked_proposal_count"] == 1
    assert preview["summary"]["recommended_ordered_proposal_ids"] == [
        "proposal-vocal-gain",
        "proposal-other",
    ]
    assert preview["summary"]["policy_ranked_proposal_ids"] == [
        "proposal-vocal-gain",
    ]
    assert preview["summary"]["unranked_proposal_ids"] == ["proposal-other"]
    assert preview["summary"]["proposal_order_mutation"] is False
    assert preview["summary"]["queue_mutation"] is False
    assert (
        preview["summary"]["downstream_priority_change_allowed_by_policy"]
        is False
    )
    assert preview["summary"]["live_widening_allowed"] is False

    ranked = preview["ordered_proposals"][0]
    unranked = preview["ordered_proposals"][1]
    assert ranked["proposal_id"] == "proposal-vocal-gain"
    assert ranked["original_index"] == 1
    assert ranked["recommended_index"] == 0
    assert ranked["rank_source"] == "source_identity_policy"
    assert ranked["read_only"] is True
    assert ranked["ordering_preview_only"] is True
    assert ranked["policy_consumer_required_before_queue_mutation"] is True
    assert ranked["proposal_order_mutation"] is False
    assert ranked["queue_mutation"] is False
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
    assert unranked["original_index"] == 0
    assert unranked["recommended_index"] == 1
    assert unranked["rank_source"] == "original_order"


def test_source_identity_proposal_ordering_preview_rejects_forged_policy_authority():
    proposals = _preview_proposals()
    policy = build_source_identity_proposal_priority_policy(
        proposals=proposals,
        priority_gate=_priority_gate(),
    )
    policy["safety"]["proposal_order_mutation"] = True
    policy["summary"]["downstream_priority_change_allowed_by_policy"] = True
    policy["policy_records"][0]["can_apply_live"] = True

    preview = build_source_identity_proposal_ordering_preview(
        proposals=proposals,
        priority_policy=policy,
    )

    assert (
        preview["status"]
        == "source_identity_proposal_ordering_preview_replacement_required"
    )
    assert preview["summary"]["policy_record_count"] == 0
    assert preview["summary"]["ordered_preview_count"] == 0
    assert preview["ordered_proposals"] == []
    assert (
        "source_identity_proposal_ordering_preview_policy_proposal_order_mutation_not_false"
        in preview["summary"]["invalid_record_ids"]
    )
    assert (
        "source_identity_proposal_ordering_preview_policy_summary_downstream_priority_change_allowed_by_policy_not_false"
        in preview["summary"]["invalid_record_ids"]
    )
    assert (
        "source_identity_proposal_priority_policy:proposal-vocal-gain"
        in preview["summary"]["invalid_record_ids"]
    )


def test_source_identity_proposal_ordering_preview_is_registered_non_live():
    method = get_processing_method(SOURCE_IDENTITY_PROPOSAL_ORDERING_PREVIEW_METHOD_ID)
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
    preview = snapshot[SOURCE_IDENTITY_PROPOSAL_ORDERING_PREVIEW_METHOD_ID]
    assert preview["safety"]["live_write"] is False
    assert preview["safety"]["proposal_order_mutation"] is False
    assert preview["summary"]["ordered_preview_count"] == 0


def test_source_identity_proposal_ordering_preview_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert SOURCE_IDENTITY_PROPOSAL_ORDERING_PREVIEW_METHOD_ID in row.method_ids
        assert "backend/source_identity_proposal_ordering_preview.py" in row.support_paths
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
