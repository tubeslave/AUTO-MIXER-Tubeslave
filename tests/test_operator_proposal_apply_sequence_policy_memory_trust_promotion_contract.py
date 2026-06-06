from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_index import (
    build_operator_proposal_apply_sequence_policy_memory_index,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger import (
    build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_contract import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract,
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


def _sequence_context(**overrides):
    payload = {
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
    payload.update(overrides)
    return payload


def _rollback_evidence(**overrides):
    payload = {
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
    payload.update(overrides)
    return payload


def _acceptance_record(**overrides):
    record = {
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
    record.update(overrides)
    return record


def _promotion_plan(records=None):
    index = build_operator_proposal_apply_sequence_policy_memory_index(
        memory_text=_memory_text(),
    )
    review = build_operator_proposal_apply_sequence_policy_memory_trust_review(
        memory_index=index,
        current_sequence_review_freshness=_sequence_context(),
    )
    ledger = build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger(
        memory_trust_review=review,
        acceptance_records=records if records is not None else [_acceptance_record()],
    )
    return build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan(
        acceptance_ledger=ledger,
    )


def test_sequence_memory_trust_promotion_contract_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_contract.py"
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


def test_sequence_memory_trust_promotion_contract_not_ready_without_candidates():
    contract = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract()
    )

    assert (
        contract["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_contract"
    )
    assert (
        contract["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID
    )
    assert contract["status"] == "sequence_memory_trust_promotion_contract_not_ready"
    assert contract["safety"]["read_only"] is True
    assert contract["safety"]["execution_allowed_by_contract"] is False
    assert contract["safety"]["policy_mutation_allowed_by_contract"] is False
    assert contract["safety"]["runtime_memory_mutation_allowed_by_contract"] is False
    assert contract["summary"]["contract_record_count"] == 0
    assert (
        "sequence_promotion_plan_candidate_required_before_contract"
        in contract["summary"]["blocked_by"]
    )


def test_sequence_memory_trust_promotion_contract_ready_without_authority():
    contract = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract(
        promotion_plan=_promotion_plan(),
        current_sequence_policy_context=_sequence_context(),
        rollback_evidence=_rollback_evidence(),
    )

    assert contract["status"] == "manual_sequence_memory_trust_promotion_contract_ready"
    assert contract["summary"]["promotion_candidate_count"] == 1
    assert contract["summary"]["contract_record_count"] == 1
    assert contract["summary"]["rollback_strategy"] == (
        "manual_revert_to_previous_sequence_policy_state"
    )
    assert contract["summary"]["sequence_policy_promotion_allowed_by_contract"] is False
    assert contract["summary"]["execution_allowed_by_contract"] is False
    assert contract["summary"]["runtime_memory_mutation_allowed_by_contract"] is False
    record = contract["contract_records"][0]
    assert record["proposal_id"] == "memory-write-applied"
    assert record["previous_policy_state"] == "not_trusted_for_sequence_policy"
    assert record["read_only"] is True
    assert record["live_write"] is False
    assert record["apply_attempted"] is False
    assert record["mutates_console_state"] is False
    assert record["queries_console"] is False
    assert record["rollback_evidence_required"] is True
    assert record["sequence_policy_promotion_ready_for_operator_review"] is True
    assert record["promotion_executor_required"] is True
    assert record["sequence_policy_promotion_allowed_by_contract"] is False
    assert record["execution_allowed_by_contract"] is False
    assert record["policy_mutation_allowed_by_contract"] is False
    assert record["runtime_memory_mutation_allowed_by_contract"] is False
    assert record["live_widening_allowed"] is False
    assert record["live_authority_granted"] is False
    assert "requested_change" not in contract


def test_sequence_memory_trust_promotion_contract_requires_fresh_context():
    contract = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract(
        promotion_plan=_promotion_plan(),
        current_sequence_policy_context=_sequence_context(status="stale"),
        rollback_evidence=_rollback_evidence(),
    )

    assert contract["status"] == "sequence_memory_trust_promotion_contract_not_ready"
    assert contract["summary"]["contract_record_count"] == 0
    assert (
        "fresh_sequence_policy_context_required_before_promotion_contract"
        in contract["summary"]["blocked_by"]
    )


def test_sequence_memory_trust_promotion_contract_requires_rollback_evidence():
    contract = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract(
        promotion_plan=_promotion_plan(),
        current_sequence_policy_context=_sequence_context(),
    )

    assert contract["status"] == "sequence_memory_trust_promotion_contract_not_ready"
    assert contract["summary"]["contract_record_count"] == 0
    assert (
        "rollback_evidence_required_before_sequence_promotion_contract"
        in contract["summary"]["blocked_by"]
    )


def test_sequence_memory_trust_promotion_contract_rejects_unsafe_plan_or_rollback():
    plan = _promotion_plan()
    forged_plan = {
        **plan,
        "summary": {
            **plan["summary"],
            "promotion_candidate_count": 2,
        },
        "promotion_candidates": [
            {
                **plan["promotion_candidates"][0],
                "sequence_policy_promotion_allowed_by_plan": True,
            }
        ],
    }

    contract = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract(
        promotion_plan=forged_plan,
        current_sequence_policy_context=_sequence_context(live_write=True),
        rollback_evidence=_rollback_evidence(
            policy_mutation=True,
            rollback_strategy="apply_immediately",
        ),
    )

    assert contract["status"] == "sequence_memory_trust_promotion_contract_replacement_required"
    assert contract["summary"]["contract_record_count"] == 0
    invalid = contract["summary"]["invalid_record_ids"]
    assert (
        "sequence_memory_trust_promotion_plan_summary_promotion_candidate_count_mismatch"
        in invalid
    )
    assert plan["promotion_candidates"][0]["promotion_candidate_id"] in invalid
    assert (
        "sequence_memory_trust_promotion_contract_context_live_write_not_false"
        in invalid
    )
    assert (
        "sequence_memory_trust_promotion_contract_rollback_policy_mutation_not_false"
        in invalid
    )
    assert (
        "sequence_memory_trust_promotion_contract_rollback_strategy_invalid"
        in invalid
    )


def test_sequence_memory_trust_promotion_contract_registry_and_goal_are_shadow_only():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID
    )

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "memory_learning"
    assert method.can_analyze is True
    assert method.can_propose is False
    assert method.can_write_live is False
    assert method.live_write_kinds == ()

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID
        in main.method_ids
    )
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID
        not in main.proposal_method_ids
    )
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID
        not in main.executor_method_ids
    )

    snapshot = build_processing_registry_snapshot(include_methods=False)
    contract = snapshot[
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_contract"
    ]
    assert contract["status"] == "sequence_memory_trust_promotion_contract_not_ready"
    assert contract["safety"]["runtime_memory_mutation_allowed_by_contract"] is False
    assert contract["safety"]["policy_mutation_allowed_by_contract"] is False

    memory_row = get_project_goal_capability("long_term_memory_learning")
    assert memory_row is not None
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID
        in memory_row.method_ids
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_contract.py"
        in memory_row.support_paths
    )
    assert memory_row.current_live_write_kinds == ()
