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
from backend.source_identity_proposal_priority_policy import (
    SOURCE_IDENTITY_PROPOSAL_PRIORITY_POLICY_METHOD_ID,
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


def test_source_identity_proposal_priority_policy_module_is_transport_free():
    source = Path("backend/source_identity_proposal_priority_policy.py").read_text()

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


def test_source_identity_proposal_priority_policy_not_ready_without_gate_records():
    policy = build_source_identity_proposal_priority_policy(proposals=[])

    assert policy["type"] == "source_identity_proposal_priority_policy"
    assert policy["method_id"] == SOURCE_IDENTITY_PROPOSAL_PRIORITY_POLICY_METHOD_ID
    assert policy["status"] == "source_identity_proposal_priority_policy_not_ready"
    assert policy["safety"]["read_only"] is True
    assert policy["safety"]["live_write"] is False
    assert policy["safety"]["proposal_order_mutation"] is False
    assert policy["safety"]["queue_mutation"] is False
    assert policy["safety"]["direct_downstream_priority_mutation"] is False
    assert policy["safety"]["downstream_priority_change_allowed_by_policy"] is False
    assert policy["safety"]["live_widening_allowed"] is False
    assert policy["summary"]["policy_record_count"] == 0
    assert policy["summary"]["can_import_to_queue"] is False
    assert policy["summary"]["can_apply_live"] is False
    assert policy["policy_records"] == []


def test_source_identity_proposal_priority_policy_projects_ranked_records_read_only():
    proposals = [
        {
            "proposal_id": "proposal-vocal-gain",
            "method_id": "safe_gain",
            "kind": "gain_trim_down",
            "target": "ch/01",
            "severity": "high",
            "confidence": 0.8,
            "evidence": {"source_candidate_id": "candidate-1"},
        },
        {
            "proposal_id": "proposal-other",
            "method_id": "safe_gain",
            "kind": "gain_trim_down",
            "target": "ch/02",
            "severity": "critical",
            "confidence": 1.0,
            "evidence": {"source_candidate_id": "candidate-2"},
        },
    ]

    policy = build_source_identity_proposal_priority_policy(
        proposals=proposals,
        priority_gate=_priority_gate(),
    )

    assert policy["status"] == "source_identity_proposal_priority_policy_records_available"
    assert policy["summary"]["input_proposal_count"] == 2
    assert policy["summary"]["priority_gate_record_count"] == 1
    assert policy["summary"]["policy_record_count"] == 1
    assert policy["summary"]["ranked_proposal_ids"] == ["proposal-vocal-gain"]
    assert policy["summary"]["unmatched_eligible_source_candidate_ids"] == []
    assert policy["summary"]["proposal_order_mutation"] is False
    assert policy["summary"]["queue_mutation"] is False
    assert policy["summary"]["downstream_priority_change_allowed_by_policy"] is False
    assert policy["summary"]["live_widening_allowed"] is False

    record = policy["policy_records"][0]
    assert record["proposal_id"] == "proposal-vocal-gain"
    assert record["source_candidate_id"] == "candidate-1"
    assert record["recommended_rank_bucket"] == "trusted_source_identity"
    assert record["adjusted_priority_score"] > record["base_priority_score"]
    assert record["policy_consumer_required_before_ordering"] is True
    assert record["read_only"] is True
    assert record["policy_evidence_only"] is True
    assert record["proposal_order_mutation"] is False
    assert record["mutates_console_state"] is False
    assert record["queries_console"] is False
    assert record["queue_mutation"] is False
    assert record["model_mutation"] is False
    assert record["runtime_memory_mutation"] is False
    assert record["source_identity_trust_store_mutation"] is False
    assert record["direct_downstream_priority_mutation"] is False
    assert record["downstream_priority_change_allowed_by_policy"] is False
    assert record["live_widening_allowed"] is False
    assert record["live_write"] is False
    assert record["live_authority_granted"] is False
    assert record["full_autonomous_ready"] is False
    assert record["can_import_to_queue"] is False
    assert record["can_apply_live"] is False


def test_source_identity_proposal_priority_policy_rejects_forged_gate_authority():
    gate = _priority_gate()
    gate["safety"]["queries_console"] = True
    gate["summary"]["direct_downstream_priority_mutation"] = True
    gate["priority_records"][0]["can_apply_live"] = True

    policy = build_source_identity_proposal_priority_policy(
        proposals=[
            {
                "proposal_id": "proposal-vocal-gain",
                "evidence": {"source_candidate_id": "candidate-1"},
            }
        ],
        priority_gate=gate,
    )

    assert policy["status"] == "source_identity_proposal_priority_policy_replacement_required"
    assert policy["summary"]["policy_record_count"] == 0
    assert policy["policy_records"] == []
    assert (
        "source_identity_proposal_priority_policy_gate_queries_console_not_false"
        in policy["summary"]["invalid_record_ids"]
    )
    assert (
        "source_identity_proposal_priority_policy_gate_summary_direct_downstream_priority_mutation_not_false"
        in policy["summary"]["invalid_record_ids"]
    )
    assert (
        "source_identity_memory_trust_promotion_priority_gate:apply-audit-1"
        in policy["summary"]["invalid_record_ids"]
    )


def test_source_identity_proposal_priority_policy_rejects_duplicate_eligible_sources():
    first = _eligible_priority_record("candidate-1")
    second = {
        **_eligible_priority_record("candidate-1"),
        "priority_gate_record_id": (
            "source_identity_memory_trust_promotion_priority_gate:apply-audit-2"
        ),
    }

    policy = build_source_identity_proposal_priority_policy(
        proposals=[
            {
                "proposal_id": "proposal-vocal-gain",
                "evidence": {"source_candidate_id": "candidate-1"},
            }
        ],
        priority_gate=_priority_gate([first, second]),
    )

    assert policy["status"] == "source_identity_proposal_priority_policy_replacement_required"
    assert policy["summary"]["policy_record_count"] == 0
    assert policy["policy_records"] == []
    assert set(policy["summary"]["invalid_record_ids"]) >= {
        "source_identity_memory_trust_promotion_priority_gate:apply-audit-1",
        "source_identity_memory_trust_promotion_priority_gate:apply-audit-2",
    }


def test_source_identity_proposal_priority_policy_is_registered_non_live():
    method = get_processing_method(SOURCE_IDENTITY_PROPOSAL_PRIORITY_POLICY_METHOD_ID)
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
    policy = snapshot[SOURCE_IDENTITY_PROPOSAL_PRIORITY_POLICY_METHOD_ID]
    assert policy["safety"]["live_write"] is False
    assert policy["safety"]["proposal_order_mutation"] is False
    assert policy["summary"]["policy_record_count"] == 0


def test_source_identity_proposal_priority_policy_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert SOURCE_IDENTITY_PROPOSAL_PRIORITY_POLICY_METHOD_ID in row.method_ids
        assert "backend/source_identity_proposal_priority_policy.py" in row.support_paths
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
