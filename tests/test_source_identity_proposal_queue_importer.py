from copy import deepcopy
from pathlib import Path

from backend.operator_proposal_queue import OperatorProposalQueue
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
from backend.source_identity_proposal_queue_importer import (
    SOURCE_IDENTITY_PROPOSAL_QUEUE_IMPORTER_METHOD_ID,
    build_source_identity_proposal_queue_importer,
)
from backend.source_identity_proposal_queue_ordering_plan import (
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
            "title": "Other channel review",
            "method_id": "safe_gain",
            "kind": "analysis",
            "target": "ch/02",
            "severity": "medium",
            "confidence": 0.7,
            "evidence": {"source_candidate_id": "candidate-2"},
        },
        {
            "proposal_id": "proposal-vocal-gain",
            "title": "Lead Vocal: lower gain",
            "method_id": "safe_gain",
            "kind": "gain",
            "target": "ch/01",
            "severity": "high",
            "confidence": 0.8,
            "requested_change": {
                "value_type": "gain",
                "channel": 1,
                "value": -1.0,
                "current_value": 0.0,
                "rollback_value": 0.0,
            },
            "evidence": {"source_candidate_id": "candidate-1"},
        },
    ]


def _queue_ordering_plan(proposals=None):
    proposal_rows = _preview_proposals() if proposals is None else proposals
    policy = build_source_identity_proposal_priority_policy(
        proposals=proposal_rows,
        priority_gate=_priority_gate(),
    )
    preview = build_source_identity_proposal_ordering_preview(
        proposals=proposal_rows,
        priority_policy=policy,
    )
    return build_source_identity_proposal_queue_ordering_plan(
        ordering_preview=preview,
    )


def test_source_identity_proposal_queue_importer_module_is_transport_free():
    source = Path("backend/source_identity_proposal_queue_importer.py").read_text()

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
        "write_text",
    ]
    for token in forbidden:
        assert token not in source


def test_source_identity_proposal_queue_importer_not_ready_without_plan_records():
    importer = build_source_identity_proposal_queue_importer()

    assert importer["type"] == "source_identity_proposal_queue_importer"
    assert importer["method_id"] == SOURCE_IDENTITY_PROPOSAL_QUEUE_IMPORTER_METHOD_ID
    assert importer["status"] == "source_identity_proposal_queue_importer_not_ready"
    assert importer["safety"]["read_only"] is True
    assert importer["safety"]["queue_import_dry_run"] is True
    assert importer["safety"]["queue_import_applied"] is False
    assert importer["safety"]["queue_mutation"] is False
    assert importer["safety"]["live_write"] is False
    assert importer["summary"]["queue_import_spec_count"] == 0
    assert importer["summary"]["queue_import_result_count"] == 0
    assert importer["queue_import_specs"] == []
    assert importer["queue_import_results"] == []


def test_source_identity_proposal_queue_importer_requires_source_payloads():
    plan = _queue_ordering_plan()

    importer = build_source_identity_proposal_queue_importer(
        queue_ordering_plan=plan,
        proposal_payloads=[],
    )

    assert (
        importer["status"]
        == "source_identity_proposal_queue_importer_source_payload_required"
    )
    assert importer["summary"]["missing_source_payload_ids"] == [
        "proposal-vocal-gain",
        "proposal-other",
    ]
    assert importer["summary"]["queue_import_spec_count"] == 0
    assert importer["summary"]["queue_mutation"] is False


def test_source_identity_proposal_queue_importer_dry_run_preserves_inputs():
    proposals = _preview_proposals()
    plan = _queue_ordering_plan(proposals)
    original_plan = deepcopy(plan)
    original_proposals = deepcopy(proposals)

    importer = build_source_identity_proposal_queue_importer(
        queue_ordering_plan=plan,
        proposal_payloads=proposals,
    )

    assert plan == original_plan
    assert proposals == original_proposals
    assert importer["status"] == "source_identity_proposal_queue_importer_dry_run_ready"
    assert importer["summary"]["queue_ordering_plan_record_count"] == 2
    assert importer["summary"]["source_payload_count"] == 2
    assert importer["summary"]["queue_import_spec_count"] == 2
    assert importer["summary"]["queue_import_result_count"] == 0
    assert importer["summary"]["planned_import_proposal_ids"] == [
        "proposal-vocal-gain",
        "proposal-other",
    ]
    assert importer["summary"]["queue_import_dry_run"] is True
    assert importer["summary"]["queue_import_applied"] is False
    assert importer["summary"]["queue_mutation"] is False
    assert importer["summary"]["can_import_to_queue"] is False
    assert importer["summary"]["can_apply_live"] is False

    spec = importer["queue_import_specs"][0]
    assert spec["proposal_id"] == "proposal-vocal-gain"
    assert spec["planned_queue_index"] == 0
    assert spec["queue_import_dry_run"] is True
    assert spec["queue_import_applied"] is False
    assert spec["queue_mutation"] is False
    assert spec["live_write"] is False
    payload = spec["operator_proposal_payload"]
    assert payload["id"] == "proposal-vocal-gain"
    assert payload["title"] == "Lead Vocal: lower gain"
    assert payload["source"] == "source_identity_proposal_queue_importer"
    assert payload["requested_change"]["value_type"] == "gain"
    assert payload["source_identity_queue_import"]["planned_queue_index"] == 0
    assert payload["source_identity_queue_import"]["live_write"] is False


def test_source_identity_proposal_queue_importer_requires_operator_ack_and_target_queue():
    proposals = _preview_proposals()
    plan = _queue_ordering_plan(proposals)

    missing_ack = build_source_identity_proposal_queue_importer(
        queue_ordering_plan=plan,
        proposal_payloads=proposals,
        target_queue=OperatorProposalQueue(),
        apply=True,
        operator_ack=False,
    )
    missing_queue = build_source_identity_proposal_queue_importer(
        queue_ordering_plan=plan,
        proposal_payloads=proposals,
        apply=True,
        operator_ack=True,
    )

    assert (
        missing_ack["status"]
        == "source_identity_proposal_queue_importer_operator_ack_required"
    )
    assert missing_ack["summary"]["queue_import_result_count"] == 0
    assert missing_ack["summary"]["queue_mutation"] is False
    assert "operator_ack_required_before_source_identity_queue_import" in missing_ack[
        "summary"
    ]["blocked_by"]
    assert (
        missing_queue["status"]
        == "source_identity_proposal_queue_importer_target_queue_required"
    )
    assert missing_queue["summary"]["target_queue_provided"] is False
    assert missing_queue["summary"]["queue_import_result_count"] == 0


def test_source_identity_proposal_queue_importer_applies_to_explicit_queue_only():
    proposals = _preview_proposals()
    plan = _queue_ordering_plan(proposals)
    queue = OperatorProposalQueue()

    importer = build_source_identity_proposal_queue_importer(
        queue_ordering_plan=plan,
        proposal_payloads=proposals,
        target_queue=queue,
        apply=True,
        operator_ack=True,
    )

    assert importer["status"] == "source_identity_proposal_queue_importer_applied"
    assert importer["safety"]["read_only"] is False
    assert importer["safety"]["queue_import_applied"] is True
    assert importer["safety"]["queue_mutation"] is True
    assert importer["safety"]["live_write"] is False
    assert importer["safety"]["model_mutation"] is False
    assert importer["safety"]["runtime_memory_mutation"] is False
    assert importer["safety"]["source_identity_trust_store_mutation"] is False
    assert importer["safety"]["direct_downstream_priority_mutation"] is False
    assert importer["summary"]["queue_import_result_count"] == 2
    assert importer["summary"]["imported_proposal_ids"] == [
        "proposal-vocal-gain",
        "proposal-other",
    ]
    assert queue.summary()["pending_count"] == 2
    pending = queue.pending_actions()
    assert pending[0]["id"] == "proposal-vocal-gain"
    assert pending[0]["requested_change"]["value_type"] == "gain"
    assert pending[0]["can_apply"] is True
    assert pending[1]["id"] == "proposal-other"
    assert pending[1]["can_apply"] is False
    result = importer["queue_import_results"][0]
    assert result["queue_mutation"] is True
    assert result["queue_import_applied"] is True
    assert result["live_write"] is False
    assert result["can_apply_live"] is False


def test_source_identity_proposal_queue_importer_rejects_forged_plan_authority():
    proposals = _preview_proposals()
    plan = _queue_ordering_plan(proposals)
    plan["safety"]["queue_mutation"] = True
    plan["summary"]["can_import_to_queue"] = True
    plan["queue_ordering_plan_records"][0]["live_write"] = True

    importer = build_source_identity_proposal_queue_importer(
        queue_ordering_plan=plan,
        proposal_payloads=proposals,
        target_queue=OperatorProposalQueue(),
        apply=True,
        operator_ack=True,
    )

    assert (
        importer["status"]
        == "source_identity_proposal_queue_importer_replacement_required"
    )
    assert importer["summary"]["queue_import_spec_count"] == 0
    assert importer["summary"]["queue_import_result_count"] == 0
    assert (
        "source_identity_proposal_queue_importer_plan_queue_mutation_not_false"
        in importer["summary"]["invalid_record_ids"]
    )
    assert (
        "source_identity_proposal_queue_importer_plan_summary_can_import_to_queue_not_false"
        in importer["summary"]["invalid_record_ids"]
    )
    assert (
        "source_identity_proposal_queue_ordering_plan:proposal-vocal-gain"
        in importer["summary"]["invalid_record_ids"]
    )


def test_source_identity_proposal_queue_importer_is_registered_non_live():
    method = get_processing_method(SOURCE_IDENTITY_PROPOSAL_QUEUE_IMPORTER_METHOD_ID)
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
    importer = snapshot[SOURCE_IDENTITY_PROPOSAL_QUEUE_IMPORTER_METHOD_ID]
    assert importer["safety"]["live_write"] is False
    assert importer["safety"]["queue_mutation"] is False
    assert importer["summary"]["queue_import_spec_count"] == 0


def test_source_identity_proposal_queue_importer_tracks_goal_matrix_without_live_widening():
    decision_basis = get_project_goal_capability("decision_basis")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (decision_basis, memory):
        assert row is not None
        assert SOURCE_IDENTITY_PROPOSAL_QUEUE_IMPORTER_METHOD_ID in row.method_ids
        assert "backend/source_identity_proposal_queue_importer.py" in row.support_paths
        assert row.current_live_write_kinds == ()
    assert decision_basis.autonomy_mode_ceiling == "assisted"
    assert memory.autonomy_mode_ceiling == "assisted"
