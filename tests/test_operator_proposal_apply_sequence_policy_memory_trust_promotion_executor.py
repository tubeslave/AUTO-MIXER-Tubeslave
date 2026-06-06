from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_index import (
    build_operator_proposal_apply_sequence_policy_memory_index,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger import (
    build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_contract import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_executor import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_plan import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_review import (
    build_operator_proposal_apply_sequence_policy_memory_trust_review,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


def _memory_text() -> str:
    return "\n".join(
        [
            "# Project Memory",
            "",
            "---",
            "",
            "### Supervised sequence policy memory",
            "",
            (
                "- Draft ID: "
                "`operator-sequence-policy-memory:memory-write-applied:"
                "memory-write:memory-write-applied`"
            ),
            "- Proposal ID: `memory-write-applied`",
            "- Learning scope: `supervised_fader_gain_sequence_execution`",
            "- Value type: `fader`",
            "- Channel: `2`",
            "- Effective value: `-8.5`",
            "- Idempotency key: `memory-write:memory-write-applied`",
            "- Readback status: `verified_from_cache`",
            "- Proof status: `verified`",
            "- Runtime authority: no live, queue, model, or policy mutation",
            "",
            "Supervised fader/gain sequence execution was verified.",
            "",
        ]
    )


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


def _rollback_evidence():
    return {
        "status": "ready",
        "read_only": True,
        "live_write": False,
        "apply_attempted": False,
        "mutates_console_state": False,
        "queries_console": False,
        "queue_mutation": False,
        "policy_mutation": False,
        "model_mutation": False,
        "runtime_memory_mutation": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "rollback_strategy": "manual_revert_to_previous_sequence_policy_state",
        "previous_policy_state": "not_trusted_for_sequence_policy",
        "target_policy_store": "operator_sequence_policy_trust_gate",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-06T13:00:00Z",
    }


def _memory_trust_acceptance_record():
    return {
        "acceptance_id": "sequence-memory-trust-acceptance-1",
        "review_candidate_id": (
            "operator_sequence_memory_trust_review:"
            "operator_sequence_policy_memory_index:"
            "operator-sequence-policy-memory:memory-write-applied:"
            "memory-write:memory-write-applied"
        ),
        "memory_record_id": (
            "operator_sequence_policy_memory_index:"
            "operator-sequence-policy-memory:memory-write-applied:"
            "memory-write:memory-write-applied"
        ),
        "proposal_id": "memory-write-applied",
        "learning_scope": "supervised_fader_gain_sequence_execution",
        "value_type": "fader",
        "idempotency_key": "memory-write:memory-write-applied",
        "decision": "accepted",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-06T12:30:00Z",
    }


def _promotion_acceptance_record():
    return {
        "promotion_acceptance_id": "sequence-memory-promotion-acceptance-1",
        "promotion_contract_record_id": (
            "operator_sequence_memory_trust_promotion_contract:"
            "operator_sequence_memory_trust_promotion:"
            "sequence-memory-trust-acceptance-1"
        ),
        "promotion_candidate_id": (
            "operator_sequence_memory_trust_promotion:"
            "sequence-memory-trust-acceptance-1"
        ),
        "proposal_id": "memory-write-applied",
        "learning_scope": "supervised_fader_gain_sequence_execution",
        "value_type": "fader",
        "idempotency_key": "memory-write:memory-write-applied",
        "previous_policy_state": "not_trusted_for_sequence_policy",
        "rollback_strategy": "manual_revert_to_previous_sequence_policy_state",
        "decision": "accepted",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-06T13:30:00Z",
    }


def _executor_dry_run():
    index = build_operator_proposal_apply_sequence_policy_memory_index(
        memory_text=_memory_text(),
    )
    review = build_operator_proposal_apply_sequence_policy_memory_trust_review(
        memory_index=index,
        current_sequence_review_freshness=_sequence_context(),
    )
    trust_ledger = build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger(
        memory_trust_review=review,
        acceptance_records=[_memory_trust_acceptance_record()],
    )
    promotion_plan = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan(
        acceptance_ledger=trust_ledger,
    )
    contract = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract(
        promotion_plan=promotion_plan,
        current_sequence_policy_context=_sequence_context(),
        rollback_evidence=_rollback_evidence(),
    )
    acceptance_ledger = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger(
            promotion_contract=contract,
            acceptance_records=[_promotion_acceptance_record()],
        )
    )
    execution_plan = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan(
        promotion_acceptance_ledger=acceptance_ledger,
    )
    return build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run(
        promotion_execution_plan=execution_plan,
    )


def test_sequence_memory_trust_promotion_executor_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_executor.py"
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


def test_sequence_memory_trust_promotion_executor_not_ready_without_dry_run():
    executor = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor()

    assert (
        executor["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor"
    )
    assert (
        executor["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID
    )
    assert executor["status"] == "sequence_memory_trust_promotion_executor_not_ready"
    assert executor["safety"]["read_only"] is True
    assert executor["safety"]["sequence_policy_promotion_applied"] is False
    assert executor["safety"]["live_write"] is False
    assert executor["safety"]["queue_mutation"] is False
    assert executor["summary"]["apply_result_count"] == 0
    assert "dry_run_record_required_before_sequence_policy_promotion_apply" in executor[
        "summary"
    ]["blocked_by"]


def test_sequence_memory_trust_promotion_executor_preview_ready_without_mutation():
    store = {
        "memory-write-applied/memory-write:memory-write-applied": (
            "not_trusted_for_sequence_policy"
        )
    }

    executor = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor(
        executor_dry_run=_executor_dry_run(),
        target_store=store,
    )

    assert executor["status"] == "sequence_memory_trust_promotion_executor_ready"
    assert executor["summary"]["dry_run_record_count"] == 1
    assert executor["summary"]["apply_result_count"] == 0
    assert executor["summary"]["apply_requested"] is False
    assert executor["summary"]["sequence_policy_promotion_applied"] is False
    assert executor["summary"]["runtime_memory_mutation"] is False
    assert store == {
        "memory-write-applied/memory-write:memory-write-applied": (
            "not_trusted_for_sequence_policy"
        )
    }
    assert executor["updated_policy_store"] == store


def test_sequence_memory_trust_promotion_executor_requires_operator_ack_and_target_store():
    dry_run = _executor_dry_run()

    missing_ack = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor(
        executor_dry_run=dry_run,
        target_store={
            "memory-write-applied/memory-write:memory-write-applied": (
                "not_trusted_for_sequence_policy"
            )
        },
        apply=True,
        operator_ack=False,
    )
    missing_store = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor(
        executor_dry_run=dry_run,
        apply=True,
        operator_ack=True,
    )

    assert (
        missing_ack["status"]
        == "sequence_memory_trust_promotion_executor_operator_ack_required"
    )
    assert missing_ack["summary"]["apply_result_count"] == 0
    assert missing_ack["summary"]["runtime_memory_mutation"] is False
    assert "operator_ack_required_before_sequence_policy_promotion" in missing_ack[
        "summary"
    ]["blocked_by"]
    assert (
        missing_store["status"]
        == "sequence_memory_trust_promotion_executor_target_store_required"
    )
    assert missing_store["summary"]["target_store_provided"] is False
    assert (
        "target_policy_store_required_before_sequence_policy_promotion"
        in missing_store["summary"]["blocked_by"]
    )


def test_sequence_memory_trust_promotion_executor_applies_to_explicit_store_only():
    store = {
        "memory-write-applied/memory-write:memory-write-applied": (
            "not_trusted_for_sequence_policy"
        )
    }

    executor = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor(
        executor_dry_run=_executor_dry_run(),
        target_store=store,
        apply=True,
        operator_ack=True,
    )

    assert executor["status"] == "sequence_memory_trust_promotion_executor_applied"
    assert executor["safety"]["read_only"] is False
    assert executor["safety"]["sequence_policy_promotion_applied"] is True
    assert executor["safety"]["runtime_memory_mutation"] is True
    assert executor["safety"]["sequence_policy_trust_store_mutation"] is True
    assert executor["safety"]["live_write"] is False
    assert executor["safety"]["queue_mutation"] is False
    assert executor["safety"]["model_mutation"] is False
    assert executor["safety"]["proposal_state_mutation"] is False
    assert executor["safety"]["sequence_execution_allowed_by_executor"] is False
    assert executor["summary"]["apply_result_count"] == 1
    assert executor["summary"]["updated_policy_keys"] == [
        "memory-write-applied/memory-write:memory-write-applied"
    ]
    assert store == {
        "memory-write-applied/memory-write:memory-write-applied": (
            "trusted_for_sequence_policy"
        )
    }
    assert executor["updated_policy_store"] == store
    result = executor["apply_results"][0]
    assert result["policy_key"] == "memory-write-applied/memory-write:memory-write-applied"
    assert result["previous_policy_state"] == "not_trusted_for_sequence_policy"
    assert result["target_policy_state"] == "trusted_for_sequence_policy"
    assert result["applied"] is True
    assert result["live_write"] is False
    assert result["queries_console"] is False
    assert result["queue_mutation"] is False
    assert result["proposal_state_mutation"] is False
    assert result["sequence_execution_allowed_by_executor"] is False
    assert result["live_widening_allowed"] is False
    assert "requested_change" not in executor


def test_sequence_memory_trust_promotion_executor_rejects_forged_dry_run():
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
                "policy_mutation_allowed_by_dry_run": True,
            }
        ],
    }

    executor = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor(
        executor_dry_run=forged,
        target_store={
            "memory-write-applied/memory-write:memory-write-applied": (
                "not_trusted_for_sequence_policy"
            )
        },
        apply=True,
        operator_ack=True,
    )

    assert (
        executor["status"]
        == "sequence_memory_trust_promotion_executor_replacement_required"
    )
    assert executor["summary"]["apply_result_count"] == 0
    assert dry_run["dry_run_records"][0]["promotion_executor_dry_run_id"] in executor[
        "summary"
    ]["invalid_record_ids"]
    assert (
        "sequence_memory_trust_promotion_executor_dry_run_summary_dry_run_record_count_mismatch"
        in executor["summary"]["invalid_record_ids"]
    )


def test_sequence_memory_trust_promotion_executor_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID
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
    executor = snapshot[
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID
    ]
    assert executor["safety"]["live_write"] is False
    assert executor["safety"]["sequence_policy_promotion_applied"] is False
    assert executor["summary"]["apply_result_count"] == 0


def test_sequence_memory_trust_promotion_executor_tracks_goal_matrix_without_live_widening():
    memory = get_project_goal_capability("long_term_memory_learning")

    assert memory is not None
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID
        in memory.method_ids
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_executor.py"
        in memory.support_paths
    )
    assert memory.current_live_write_kinds == ()
    assert memory.autonomy_mode_ceiling == "assisted"
