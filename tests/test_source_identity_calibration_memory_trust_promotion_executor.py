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
from backend.source_identity_calibration_memory_trust_promotion_contract import (
    build_source_identity_calibration_memory_trust_promotion_contract,
)
from backend.source_identity_calibration_memory_trust_promotion_execution_plan import (
    build_source_identity_calibration_memory_trust_promotion_execution_plan,
)
from backend.source_identity_calibration_memory_trust_promotion_executor import (
    SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID,
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


def test_memory_trust_promotion_executor_module_is_transport_free():
    source = Path(
        "backend/source_identity_calibration_memory_trust_promotion_executor.py"
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


def test_memory_trust_promotion_executor_not_ready_without_dry_run():
    executor = build_source_identity_calibration_memory_trust_promotion_executor()

    assert executor["type"] == "source_identity_calibration_memory_trust_promotion_executor"
    assert executor["method_id"] == SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID
    assert executor["status"] == "memory_trust_promotion_executor_not_ready"
    assert executor["safety"]["read_only"] is True
    assert executor["safety"]["memory_trust_promotion_applied"] is False
    assert executor["safety"]["live_write"] is False
    assert executor["safety"]["queue_mutation"] is False
    assert executor["summary"]["apply_result_count"] == 0
    assert "dry_run_record_required_before_memory_trust_promotion_apply" in executor[
        "summary"
    ]["blocked_by"]


def test_memory_trust_promotion_executor_preview_ready_without_mutation():
    store = {"candidate-1": "legacy_unverified"}

    executor = build_source_identity_calibration_memory_trust_promotion_executor(
        executor_dry_run=_executor_dry_run(),
        target_store=store,
    )

    assert executor["status"] == "memory_trust_promotion_executor_ready"
    assert executor["summary"]["dry_run_record_count"] == 1
    assert executor["summary"]["apply_result_count"] == 0
    assert executor["summary"]["apply_requested"] is False
    assert executor["summary"]["memory_trust_promotion_applied"] is False
    assert executor["summary"]["runtime_memory_mutation"] is False
    assert store == {"candidate-1": "legacy_unverified"}
    assert executor["updated_trust_store"] == store


def test_memory_trust_promotion_executor_requires_operator_ack_and_target_store():
    dry_run = _executor_dry_run()

    missing_ack = build_source_identity_calibration_memory_trust_promotion_executor(
        executor_dry_run=dry_run,
        target_store={"candidate-1": "legacy_unverified"},
        apply=True,
        operator_ack=False,
    )
    missing_store = build_source_identity_calibration_memory_trust_promotion_executor(
        executor_dry_run=dry_run,
        apply=True,
        operator_ack=True,
    )

    assert missing_ack["status"] == "memory_trust_promotion_executor_operator_ack_required"
    assert missing_ack["summary"]["apply_result_count"] == 0
    assert missing_ack["summary"]["runtime_memory_mutation"] is False
    assert "operator_ack_required_before_memory_trust_promotion" in missing_ack[
        "summary"
    ]["blocked_by"]
    assert missing_store["status"] == "memory_trust_promotion_executor_target_store_required"
    assert missing_store["summary"]["target_store_provided"] is False
    assert "target_trust_store_required_before_memory_trust_promotion" in missing_store[
        "summary"
    ]["blocked_by"]


def test_memory_trust_promotion_executor_applies_to_explicit_store_only():
    store = {"candidate-1": "legacy_unverified"}

    executor = build_source_identity_calibration_memory_trust_promotion_executor(
        executor_dry_run=_executor_dry_run(),
        target_store=store,
        apply=True,
        operator_ack=True,
    )

    assert executor["status"] == "memory_trust_promotion_executor_applied"
    assert executor["safety"]["read_only"] is False
    assert executor["safety"]["memory_trust_promotion_applied"] is True
    assert executor["safety"]["runtime_memory_mutation"] is True
    assert executor["safety"]["source_identity_trust_store_mutation"] is True
    assert executor["safety"]["live_write"] is False
    assert executor["safety"]["queue_mutation"] is False
    assert executor["safety"]["model_mutation"] is False
    assert executor["safety"]["direct_downstream_priority_mutation"] is False
    assert executor["summary"]["apply_result_count"] == 1
    assert executor["summary"]["updated_source_candidate_ids"] == ["candidate-1"]
    assert store == {"candidate-1": "trusted_for_downstream_priority"}
    assert executor["updated_trust_store"] == store
    result = executor["apply_results"][0]
    assert result["source_candidate_id"] == "candidate-1"
    assert result["previous_trust_state"] == "legacy_unverified"
    assert result["target_trust_state"] == "trusted_for_downstream_priority"
    assert result["applied"] is True
    assert result["live_write"] is False
    assert result["queries_console"] is False
    assert result["queue_mutation"] is False
    assert result["direct_downstream_priority_mutation"] is False
    assert result["downstream_priority_change_allowed_by_executor"] is False
    assert result["live_widening_allowed"] is False
    assert "requested_change" not in executor


def test_memory_trust_promotion_executor_rejects_forged_dry_run():
    dry_run = _executor_dry_run()
    forged = {
        **dry_run,
        "summary": {
            **dry_run["summary"],
            "dry_run_record_count": 2,
        },
        "dry_run_records": [
            {
                **dry_run["dry_run_records"][0],
                "runtime_memory_mutation_allowed_by_dry_run": True,
            }
        ],
    }

    executor = build_source_identity_calibration_memory_trust_promotion_executor(
        executor_dry_run=forged,
        target_store={"candidate-1": "legacy_unverified"},
        apply=True,
        operator_ack=True,
    )

    assert executor["status"] == "memory_trust_promotion_executor_replacement_required"
    assert executor["summary"]["apply_result_count"] == 0
    assert dry_run["dry_run_records"][0]["promotion_executor_dry_run_id"] in executor[
        "summary"
    ]["invalid_record_ids"]
    assert "memory_trust_promotion_executor_dry_run_summary_dry_run_record_count_mismatch" in executor[
        "summary"
    ]["invalid_record_ids"]


def test_memory_trust_promotion_executor_is_registered_non_live():
    method = get_processing_method(
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID
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
    executor = snapshot[SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID]
    assert executor["safety"]["live_write"] is False
    assert executor["safety"]["memory_trust_promotion_applied"] is False
    assert executor["summary"]["apply_result_count"] == 0


def test_memory_trust_promotion_executor_tracks_goal_matrix_without_live_widening():
    source_types = get_project_goal_capability("identify_source_types")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (source_types, memory):
        assert row is not None
        assert SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID in row.method_ids
        assert (
            "backend/source_identity_calibration_memory_trust_promotion_executor.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert source_types.autonomy_mode_ceiling == "analysis"
    assert memory.autonomy_mode_ceiling == "assisted"
