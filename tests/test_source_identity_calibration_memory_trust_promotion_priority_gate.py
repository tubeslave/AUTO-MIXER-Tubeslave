from pathlib import Path

from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.source_identity_calibration_memory_index import (
    build_source_identity_calibration_memory_index,
)
from backend.source_identity_calibration_memory_trust_acceptance_ledger import (
    build_source_identity_calibration_memory_trust_acceptance_ledger,
)
from backend.source_identity_calibration_memory_trust_promotion_acceptance_ledger import (
    build_source_identity_calibration_memory_trust_promotion_acceptance_ledger,
)
from backend.source_identity_calibration_memory_trust_promotion_apply_audit import (
    build_source_identity_calibration_memory_trust_promotion_apply_audit,
)
from backend.source_identity_calibration_memory_trust_promotion_contract import (
    build_source_identity_calibration_memory_trust_promotion_contract,
)
from backend.source_identity_calibration_memory_trust_promotion_execution_plan import (
    build_source_identity_calibration_memory_trust_promotion_execution_plan,
)
from backend.source_identity_calibration_memory_trust_promotion_executor import (
    build_source_identity_calibration_memory_trust_promotion_executor,
)
from backend.source_identity_calibration_memory_trust_promotion_executor_dry_run import (
    build_source_identity_calibration_memory_trust_promotion_executor_dry_run,
)
from backend.source_identity_calibration_memory_trust_promotion_plan import (
    build_source_identity_calibration_memory_trust_promotion_plan,
)
from backend.source_identity_calibration_memory_trust_promotion_priority_gate import (
    SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PRIORITY_GATE_METHOD_ID,
    build_source_identity_calibration_memory_trust_promotion_priority_gate,
)
from backend.source_identity_calibration_memory_trust_promotion_rollback_verification import (
    build_source_identity_calibration_memory_trust_promotion_rollback_verification,
)
from backend.source_identity_calibration_memory_trust_review import (
    build_source_identity_calibration_memory_trust_review,
)


def _memory_text() -> str:
    return "\n".join(
        [
            "# Project Memory",
            "",
            "---",
            "",
            "### Source identity calibration memory: lead_vocal",
            "",
            "- Draft ID: `source_identity_memory:candidate-1`",
            "- Source candidate: `candidate-1`",
            "- Source review: `review-1`",
            "- Recommended trust state: `trusted_for_downstream_priority`",
            "- Reviewed by: `operator`",
            "- Reviewed at: `2026-06-06T11:00:00Z`",
            "- Runtime authority: no model, memory-priority, queue, or live-console mutation",
            "",
            "Source identity calibration for `lead_vocal` was operator-reviewed.",
            "",
        ]
    )


def _freshness():
    return {
        "status": "fresh",
        "field_count": 4,
        "fresh_field_count": 4,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def _rollback_evidence():
    return {
        "status": "ready",
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "queue_mutation": False,
        "model_mutation": False,
        "runtime_memory_mutation": False,
        "downstream_priority_mutation": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "rollback_strategy": "manual_revert_to_previous_trust_state",
        "previous_trust_state": "legacy_unverified",
        "target_store": "source_identity_downstream_gate",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-06T12:30:00Z",
    }


def _memory_trust_acceptance_record():
    return {
        "acceptance_id": "memory-trust-acceptance-1",
        "review_candidate_id": (
            "source_identity_memory_trust_review:"
            "source_identity_memory_index:source_identity_memory:candidate-1"
        ),
        "memory_record_id": "source_identity_memory_index:source_identity_memory:candidate-1",
        "source_candidate_id": "candidate-1",
        "source_type": "lead_vocal",
        "recommended_trust_state": "trusted_for_downstream_priority",
        "decision": "accepted",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-06T12:00:00Z",
    }


def _promotion_acceptance_record():
    return {
        "promotion_acceptance_id": "memory-trust-promotion-acceptance-1",
        "promotion_contract_record_id": (
            "source_identity_memory_trust_promotion_contract:"
            "source_identity_memory_trust_promotion:memory-trust-acceptance-1"
        ),
        "promotion_candidate_id": (
            "source_identity_memory_trust_promotion:memory-trust-acceptance-1"
        ),
        "source_candidate_id": "candidate-1",
        "source_type": "lead_vocal",
        "recommended_trust_state": "trusted_for_downstream_priority",
        "previous_trust_state": "legacy_unverified",
        "rollback_strategy": "manual_revert_to_previous_trust_state",
        "decision": "accepted",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-06T13:00:00Z",
    }


def _executor_dry_run():
    memory_index = build_source_identity_calibration_memory_index(memory_text=_memory_text())
    review = build_source_identity_calibration_memory_trust_review(
        memory_index=memory_index,
        current_state_cache_freshness=_freshness(),
    )
    trust_ledger = build_source_identity_calibration_memory_trust_acceptance_ledger(
        memory_trust_review=review,
        acceptance_records=[_memory_trust_acceptance_record()],
    )
    promotion_plan = build_source_identity_calibration_memory_trust_promotion_plan(
        acceptance_ledger=trust_ledger,
    )
    contract = build_source_identity_calibration_memory_trust_promotion_contract(
        promotion_plan=promotion_plan,
        current_state_cache_freshness=_freshness(),
        rollback_evidence=_rollback_evidence(),
    )
    acceptance_ledger = (
        build_source_identity_calibration_memory_trust_promotion_acceptance_ledger(
            promotion_contract=contract,
            acceptance_records=[_promotion_acceptance_record()],
        )
    )
    execution_plan = build_source_identity_calibration_memory_trust_promotion_execution_plan(
        promotion_acceptance_ledger=acceptance_ledger,
    )
    return build_source_identity_calibration_memory_trust_promotion_executor_dry_run(
        promotion_execution_plan=execution_plan,
    )


def _applied_executor():
    store = {"candidate-1": "legacy_unverified"}
    return build_source_identity_calibration_memory_trust_promotion_executor(
        executor_dry_run=_executor_dry_run(),
        target_store=store,
        apply=True,
        operator_ack=True,
    )


def _apply_audit():
    return build_source_identity_calibration_memory_trust_promotion_apply_audit(
        promotion_executor=_applied_executor(),
    )


def _verification_record(audit_record):
    return {
        "rollback_verification_id": "rollback-verification-1",
        "promotion_apply_audit_id": audit_record["promotion_apply_audit_id"],
        "promotion_executor_apply_id": audit_record["promotion_executor_apply_id"],
        "source_candidate_id": audit_record["source_candidate_id"],
        "source_type": audit_record["source_type"],
        "target_store": audit_record["target_store"],
        "target_path": audit_record["target_path"],
        "target_trust_state": audit_record["target_trust_state"],
        "rollback_target_trust_state": audit_record["rollback_target_trust_state"],
        "rollback_strategy": audit_record["rollback_strategy"],
        "verification_evidence_status": "reviewed",
        "decision": "verified",
        "verified_by": "operator",
        "verified_at": "2026-06-06T14:00:00Z",
        "rollback_verified": True,
        "read_only": True,
        "operator_review_required": True,
        "separate_downstream_priority_gate_required": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "queue_mutation": False,
        "model_mutation": False,
        "runtime_memory_mutation": False,
        "source_identity_trust_store_mutation": False,
        "direct_downstream_priority_mutation": False,
        "downstream_priority_change_allowed_by_verification": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "can_import_to_queue": False,
        "can_apply_live": False,
    }


def _rollback_verification():
    audit = _apply_audit()
    return build_source_identity_calibration_memory_trust_promotion_rollback_verification(
        apply_audit=audit,
        verification_records=[_verification_record(audit["audit_records"][0])],
    )


def test_memory_trust_promotion_priority_gate_module_is_transport_free():
    source = Path(
        "backend/source_identity_calibration_memory_trust_promotion_priority_gate.py"
    ).read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "apply_operator_proposal",
        "requested_change",
        "requested_changes",
        "write_text",
    ]
    for token in forbidden:
        assert token not in source


def test_memory_trust_promotion_priority_gate_not_ready_without_rollback_verification():
    gate = build_source_identity_calibration_memory_trust_promotion_priority_gate()

    assert gate["type"] == "source_identity_calibration_memory_trust_promotion_priority_gate"
    assert (
        gate["method_id"]
        == SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PRIORITY_GATE_METHOD_ID
    )
    assert gate["status"] == "memory_trust_promotion_priority_gate_not_ready"
    assert gate["safety"]["read_only"] is True
    assert gate["safety"]["live_write"] is False
    assert gate["safety"]["queue_mutation"] is False
    assert gate["safety"]["direct_downstream_priority_mutation"] is False
    assert gate["summary"]["priority_gate_record_count"] == 0
    assert gate["summary"]["eligible_record_count"] == 0
    assert gate["summary"]["can_import_to_queue"] is False
    assert gate["summary"]["can_apply_live"] is False
    assert gate["priority_records"] == []


def test_memory_trust_promotion_priority_gate_records_eligible_policy_evidence():
    gate = build_source_identity_calibration_memory_trust_promotion_priority_gate(
        rollback_verification=_rollback_verification(),
    )

    assert gate["status"] == "verified_memory_trust_promotion_priority_records_available"
    assert gate["safety"]["read_only"] is True
    assert gate["safety"]["downstream_priority_eligible"] is True
    assert gate["safety"]["downstream_priority_change_allowed_by_gate"] is False
    assert gate["safety"]["live_authority_granted"] is False
    assert gate["summary"]["rollback_verified_record_count"] == 1
    assert gate["summary"]["priority_gate_record_count"] == 1
    assert gate["summary"]["eligible_record_count"] == 1
    assert gate["summary"]["direct_downstream_priority_mutation"] is False
    assert gate["summary"]["can_import_to_queue"] is False
    assert gate["summary"]["can_apply_live"] is False

    record = gate["priority_records"][0]
    assert record["source_candidate_id"] == "candidate-1"
    assert record["target_trust_state"] == "trusted_for_downstream_priority"
    assert record["rollback_verified"] is True
    assert record["source_identity_downstream_gate_status"] == "trusted"
    assert record["source_identity_downstream_gate_usable"] is True
    assert record["downstream_priority_eligible"] is True
    assert record["eligible_for_downstream_priority_policy"] is True
    assert record["policy_gate_only"] is True
    assert record["proposal_priority_consumer_required"] is True
    assert record["queue_mutation"] is False
    assert record["model_mutation"] is False
    assert record["runtime_memory_mutation"] is False
    assert record["source_identity_trust_store_mutation"] is False
    assert record["direct_downstream_priority_mutation"] is False
    assert record["downstream_priority_change_allowed_by_gate"] is False
    assert record["can_import_to_queue"] is False
    assert record["can_apply_live"] is False
    assert "requested_change" not in gate


def test_memory_trust_promotion_priority_gate_rejects_forged_authority():
    rollback = _rollback_verification()
    forged = {
        **rollback,
        "summary": {
            **rollback["summary"],
            "direct_downstream_priority_mutation": True,
        },
        "verified_records": [
            {
                **rollback["verified_records"][0],
                "can_apply_live": True,
                "direct_downstream_priority_mutation": True,
            }
        ],
    }

    gate = build_source_identity_calibration_memory_trust_promotion_priority_gate(
        rollback_verification=forged,
    )

    assert gate["status"] == "memory_trust_promotion_priority_gate_replacement_required"
    assert gate["summary"]["priority_gate_record_count"] == 0
    assert gate["summary"]["eligible_record_count"] == 0
    assert (
        "memory_trust_promotion_priority_gate_rollback_summary_direct_downstream_priority_mutation_not_false"
        in gate["summary"]["invalid_record_ids"]
    )
    assert (
        "source_identity_memory_trust_promotion_rollback_verification:"
        f"{rollback['verified_records'][0]['promotion_apply_audit_id']}"
        in gate["summary"]["invalid_record_ids"]
    )


def test_memory_trust_promotion_priority_gate_blocks_untrusted_downstream_state():
    rollback = _rollback_verification()
    blocked = {
        **rollback,
        "verified_records": [
            {
                **rollback["verified_records"][0],
                "target_trust_state": "hold_for_operator_review",
            }
        ],
    }

    gate = build_source_identity_calibration_memory_trust_promotion_priority_gate(
        rollback_verification=blocked,
    )

    assert gate["status"] == "memory_trust_promotion_priority_gate_not_ready"
    assert gate["summary"]["priority_gate_record_count"] == 1
    assert gate["summary"]["eligible_record_count"] == 0
    assert gate["summary"]["ineligible_record_count"] == 1
    assert gate["summary"]["direct_downstream_priority_mutation"] is False
    assert gate["priority_records"][0]["source_identity_downstream_gate_status"] == (
        "held_for_operator_review"
    )
    assert gate["priority_records"][0]["downstream_priority_eligible"] is False
    assert "source_identity_not_trusted_for_downstream" in (
        gate["priority_records"][0]["source_identity_downstream_gate_blockers"]
    )


def test_memory_trust_promotion_priority_gate_is_registered_non_live():
    method = get_processing_method(
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PRIORITY_GATE_METHOD_ID
    )
    main = get_pipeline_profile("main_live_assist")
    snapshot = build_processing_registry_snapshot(include_methods=False)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "memory_learning"
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
    gate = snapshot[SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PRIORITY_GATE_METHOD_ID]
    assert gate["safety"]["live_write"] is False
    assert gate["safety"]["runtime_memory_mutation"] is False
    assert gate["summary"]["priority_gate_record_count"] == 0


def test_memory_trust_promotion_priority_gate_tracks_goal_matrix_without_live_widening():
    source_types = get_project_goal_capability("identify_source_types")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (source_types, memory):
        assert row is not None
        assert (
            SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PRIORITY_GATE_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/source_identity_calibration_memory_trust_promotion_priority_gate.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert source_types.autonomy_mode_ceiling == "analysis"
    assert memory.autonomy_mode_ceiling == "assisted"
