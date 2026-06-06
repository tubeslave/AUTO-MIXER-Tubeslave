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
    SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID,
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


def test_memory_trust_promotion_apply_audit_module_is_transport_free():
    source = Path(
        "backend/source_identity_calibration_memory_trust_promotion_apply_audit.py"
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


def test_memory_trust_promotion_apply_audit_not_ready_without_applied_executor():
    audit = build_source_identity_calibration_memory_trust_promotion_apply_audit()

    assert audit["type"] == "source_identity_calibration_memory_trust_promotion_apply_audit"
    assert audit["method_id"] == SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID
    assert audit["status"] == "memory_trust_promotion_apply_audit_not_ready"
    assert audit["safety"]["read_only"] is True
    assert audit["safety"]["live_write"] is False
    assert audit["safety"]["queue_mutation"] is False
    assert audit["safety"]["model_mutation"] is False
    assert audit["safety"]["runtime_memory_mutation"] is False
    assert audit["safety"]["direct_downstream_priority_mutation"] is False
    assert audit["summary"]["apply_audit_record_count"] == 0
    assert audit["audit_records"] == []
    assert "applied_memory_trust_promotion_result_required_before_apply_audit" in audit[
        "summary"
    ]["blocked_by"]


def test_memory_trust_promotion_apply_audit_records_rollback_target():
    executor = _applied_executor()

    audit = build_source_identity_calibration_memory_trust_promotion_apply_audit(
        promotion_executor=executor,
    )

    assert audit["status"] == "memory_trust_promotion_apply_audit_records_available"
    assert audit["safety"]["read_only"] is True
    assert audit["safety"]["runtime_memory_mutation"] is False
    assert audit["safety"]["source_identity_trust_store_mutation"] is False
    assert audit["safety"]["audits_applied_source_identity_trust_store_mutation"] is True
    assert audit["safety"]["direct_downstream_priority_mutation"] is False
    assert audit["safety"]["downstream_priority_change_allowed_by_audit"] is False
    assert audit["safety"]["live_authority_granted"] is False
    assert audit["summary"]["executor_apply_result_count"] == 1
    assert audit["summary"]["apply_audit_record_count"] == 1
    assert audit["summary"]["rollback_target_trust_states"] == ["legacy_unverified"]
    record = audit["audit_records"][0]
    assert record["promotion_executor_apply_id"] == executor["apply_results"][0][
        "promotion_executor_apply_id"
    ]
    assert record["source_candidate_id"] == "candidate-1"
    assert record["target_trust_state"] == "trusted_for_downstream_priority"
    assert record["rollback_target_trust_state"] == "legacy_unverified"
    assert record["rollback_evidence_required"] is True
    assert record["rollback_verified"] is False
    assert record["source_identity_trust_store_mutation_audited"] is True
    assert record["runtime_memory_mutation"] is False
    assert record["queue_mutation"] is False
    assert record["model_mutation"] is False
    assert record["direct_downstream_priority_mutation"] is False
    assert record["live_write"] is False
    assert record["live_authority_granted"] is False
    assert "requested_change" not in audit


def test_memory_trust_promotion_apply_audit_rejects_forged_executor_authority():
    executor = _applied_executor()
    apply_result = executor["apply_results"][0]
    forged = {
        **executor,
        "summary": {
            **executor["summary"],
            "apply_result_count": 2,
            "direct_downstream_priority_mutation": True,
        },
        "apply_results": [
            {
                **apply_result,
                "live_write": True,
                "direct_downstream_priority_mutation": True,
            }
        ],
    }

    audit = build_source_identity_calibration_memory_trust_promotion_apply_audit(
        promotion_executor=forged,
    )

    assert audit["status"] == "memory_trust_promotion_apply_audit_replacement_required"
    assert audit["summary"]["apply_audit_record_count"] == 0
    assert apply_result["promotion_executor_apply_id"] in audit["summary"][
        "invalid_record_ids"
    ]
    assert "memory_trust_promotion_apply_audit_summary_apply_result_count_mismatch" in audit[
        "summary"
    ]["invalid_record_ids"]
    assert "memory_trust_promotion_apply_audit_summary_direct_downstream_priority_mutation_not_false" in audit[
        "summary"
    ]["invalid_record_ids"]


def test_memory_trust_promotion_apply_audit_rejects_forged_apply_result_safety_envelope():
    executor = _applied_executor()
    apply_result = executor["apply_results"][0]
    forged = {
        **executor,
        "apply_results": [
            {
                **apply_result,
                "queries_console": True,
                "downstream_priority_change_allowed_by_executor": True,
                "live_widening_allowed": True,
            }
        ],
    }

    audit = build_source_identity_calibration_memory_trust_promotion_apply_audit(
        promotion_executor=forged,
    )

    assert audit["status"] == "memory_trust_promotion_apply_audit_replacement_required"
    assert audit["summary"]["apply_audit_record_count"] == 0
    assert apply_result["promotion_executor_apply_id"] in audit["summary"][
        "invalid_record_ids"
    ]
    assert not any(
        item.startswith("memory_trust_promotion_apply_audit_summary_")
        for item in audit["summary"]["invalid_record_ids"]
    )


def test_memory_trust_promotion_apply_audit_is_registered_non_live():
    method = get_processing_method(
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID
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
    audit = snapshot[
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID
    ]
    assert audit["safety"]["live_write"] is False
    assert audit["safety"]["runtime_memory_mutation"] is False
    assert audit["summary"]["apply_audit_record_count"] == 0


def test_memory_trust_promotion_apply_audit_tracks_goal_matrix_without_live_widening():
    source_types = get_project_goal_capability("identify_source_types")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (source_types, memory):
        assert row is not None
        assert (
            SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/source_identity_calibration_memory_trust_promotion_apply_audit.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert source_types.autonomy_mode_ceiling == "analysis"
    assert memory.autonomy_mode_ceiling == "assisted"
