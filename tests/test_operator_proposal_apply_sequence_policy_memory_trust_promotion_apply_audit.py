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
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_contract import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_executor import (
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


def _applied_executor():
    store = {
        "memory-write-applied/memory-write:memory-write-applied": (
            "not_trusted_for_sequence_policy"
        )
    }
    return build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor(
        executor_dry_run=_executor_dry_run(),
        target_store=store,
        apply=True,
        operator_ack=True,
    )


def test_sequence_memory_trust_promotion_apply_audit_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit.py"
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


def test_sequence_memory_trust_promotion_apply_audit_not_ready_without_applied_executor():
    audit = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit()

    assert (
        audit["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit"
    )
    assert (
        audit["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID
    )
    assert audit["status"] == "sequence_memory_trust_promotion_apply_audit_not_ready"
    assert audit["safety"]["read_only"] is True
    assert audit["safety"]["live_write"] is False
    assert audit["safety"]["queue_mutation"] is False
    assert audit["safety"]["model_mutation"] is False
    assert audit["safety"]["runtime_memory_mutation"] is False
    assert audit["safety"]["sequence_execution_allowed_by_audit"] is False
    assert audit["summary"]["apply_audit_record_count"] == 0
    assert audit["audit_records"] == []
    assert (
        "applied_sequence_policy_promotion_result_required_before_apply_audit"
        in audit["summary"]["blocked_by"]
    )


def test_sequence_memory_trust_promotion_apply_audit_records_rollback_target():
    executor = _applied_executor()

    audit = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit(
        promotion_executor=executor,
    )

    assert audit["status"] == "sequence_memory_trust_promotion_apply_audit_records_available"
    assert audit["safety"]["read_only"] is True
    assert audit["safety"]["runtime_memory_mutation"] is False
    assert audit["safety"]["sequence_policy_trust_store_mutation"] is False
    assert audit["safety"][
        "audits_applied_sequence_policy_trust_store_mutation"
    ] is True
    assert audit["safety"]["proposal_state_mutation"] is False
    assert audit["safety"]["sequence_execution_allowed_by_audit"] is False
    assert audit["safety"]["live_authority_granted"] is False
    assert audit["summary"]["executor_apply_result_count"] == 1
    assert audit["summary"]["apply_audit_record_count"] == 1
    assert audit["summary"]["rollback_target_policy_states"] == [
        "not_trusted_for_sequence_policy"
    ]
    record = audit["audit_records"][0]
    assert record["promotion_executor_apply_id"] == executor["apply_results"][0][
        "promotion_executor_apply_id"
    ]
    assert record["policy_key"] == "memory-write-applied/memory-write:memory-write-applied"
    assert record["proposal_id"] == "memory-write-applied"
    assert record["target_policy_state"] == "trusted_for_sequence_policy"
    assert record["rollback_target_policy_state"] == "not_trusted_for_sequence_policy"
    assert record["rollback_evidence_required"] is True
    assert record["rollback_verified"] is False
    assert record["audits_applied_sequence_policy_trust_store_mutation"] is True
    assert record["runtime_memory_mutation"] is False
    assert record["queue_mutation"] is False
    assert record["model_mutation"] is False
    assert record["proposal_state_mutation"] is False
    assert record["sequence_execution_allowed_by_audit"] is False
    assert record["live_write"] is False
    assert record["live_authority_granted"] is False
    assert "requested_change" not in audit


def test_sequence_memory_trust_promotion_apply_audit_rejects_forged_executor_authority():
    executor = _applied_executor()
    apply_result = executor["apply_results"][0]
    forged = {
        **executor,
        "summary": {
            **executor["summary"],
            "apply_result_count": 2,
            "sequence_execution_allowed_by_executor": True,
        },
        "apply_results": [
            {
                **apply_result,
                "live_write": True,
                "sequence_execution_allowed_by_executor": True,
            }
        ],
    }

    audit = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit(
        promotion_executor=forged,
    )

    assert (
        audit["status"]
        == "sequence_memory_trust_promotion_apply_audit_replacement_required"
    )
    assert audit["summary"]["apply_audit_record_count"] == 0
    assert apply_result["promotion_executor_apply_id"] in audit["summary"][
        "invalid_record_ids"
    ]
    assert (
        "sequence_memory_trust_promotion_apply_audit_summary_apply_result_count_mismatch"
        in audit["summary"]["invalid_record_ids"]
    )
    assert (
        "sequence_memory_trust_promotion_apply_audit_summary_sequence_execution_allowed_by_executor_not_false"
        in audit["summary"]["invalid_record_ids"]
    )


def test_sequence_memory_trust_promotion_apply_audit_rejects_forged_apply_result_safety_envelope():
    executor = _applied_executor()
    apply_result = executor["apply_results"][0]
    forged = {
        **executor,
        "apply_results": [
            {
                **apply_result,
                "queries_console": True,
                "proposal_state_mutation": True,
                "live_widening_allowed": True,
            }
        ],
    }

    audit = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit(
        promotion_executor=forged,
    )

    assert (
        audit["status"]
        == "sequence_memory_trust_promotion_apply_audit_replacement_required"
    )
    assert audit["summary"]["apply_audit_record_count"] == 0
    assert apply_result["promotion_executor_apply_id"] in audit["summary"][
        "invalid_record_ids"
    ]
    assert not any(
        item.startswith("sequence_memory_trust_promotion_apply_audit_summary_")
        for item in audit["summary"]["invalid_record_ids"]
    )


def test_sequence_memory_trust_promotion_apply_audit_is_registered_non_live():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID
    ]
    assert audit["safety"]["live_write"] is False
    assert audit["safety"]["runtime_memory_mutation"] is False
    assert audit["summary"]["apply_audit_record_count"] == 0


def test_sequence_memory_trust_promotion_apply_audit_tracks_goal_matrix_without_live_widening():
    memory = get_project_goal_capability("long_term_memory_learning")

    assert memory is not None
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID
        in memory.method_ids
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit.py"
        in memory.support_paths
    )
    assert memory.current_live_write_kinds == ()
    assert memory.autonomy_mode_ceiling == "assisted"
