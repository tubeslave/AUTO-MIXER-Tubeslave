from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_index import (
    build_operator_proposal_apply_sequence_policy_memory_index,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger import (
    build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_plan import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID,
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


def _freshness():
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


def _acceptance_ledger(records=None):
    index = build_operator_proposal_apply_sequence_policy_memory_index(
        memory_text=_memory_text(),
    )
    review = build_operator_proposal_apply_sequence_policy_memory_trust_review(
        memory_index=index,
        current_sequence_review_freshness=_freshness(),
    )
    return build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger(
        memory_trust_review=review,
        acceptance_records=records if records is not None else [_acceptance_record()],
    )


def test_sequence_memory_trust_promotion_plan_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_plan.py"
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


def test_sequence_memory_trust_promotion_plan_not_ready_without_accepted_records():
    plan = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan()

    assert (
        plan["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_plan"
    )
    assert (
        plan["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID
    )
    assert plan["status"] == "sequence_memory_trust_promotion_plan_not_ready"
    assert plan["safety"]["read_only"] is True
    assert plan["safety"]["execution_allowed_by_plan"] is False
    assert plan["safety"]["runtime_memory_mutation_allowed_by_plan"] is False
    assert plan["safety"]["policy_mutation_allowed_by_plan"] is False
    assert plan["summary"]["promotion_candidate_count"] == 0
    assert (
        "accepted_sequence_memory_trust_record_required_before_promotion"
        in plan["summary"]["blocked_by"]
    )


def test_sequence_memory_trust_promotion_plan_surfaces_manual_candidate_without_authority():
    plan = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan(
        acceptance_ledger=_acceptance_ledger(),
    )

    assert plan["status"] == "sequence_memory_trust_promotion_plan_candidate_available"
    assert plan["summary"]["accepted_record_count"] == 1
    assert plan["summary"]["promotion_candidate_count"] == 1
    assert plan["summary"]["accepted_proposal_ids"] == ["memory-write-applied"]
    assert plan["summary"]["sequence_policy_promotion_allowed_by_plan"] is False
    assert plan["summary"]["execution_allowed_by_plan"] is False
    assert plan["summary"]["runtime_memory_mutation_allowed_by_plan"] is False
    candidate = plan["promotion_candidates"][0]
    assert candidate["acceptance_id"] == "sequence-memory-trust-acceptance-1"
    assert candidate["proposal_id"] == "memory-write-applied"
    assert candidate["value_type"] == "fader"
    assert candidate["manual_sequence_policy_promotion_required"] is True
    assert candidate["sequence_policy_promotion_allowed_by_plan"] is False
    assert candidate["execution_allowed_by_plan"] is False
    assert candidate["policy_mutation_allowed_by_plan"] is False
    assert candidate["runtime_memory_mutation_allowed_by_plan"] is False
    assert candidate["live_authority_granted"] is False
    assert "requested_change" not in plan


def test_sequence_memory_trust_promotion_plan_waits_for_unreviewed_candidates():
    ledger = _acceptance_ledger(records=[])
    plan = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan(
        acceptance_ledger=ledger,
    )

    assert plan["status"] == "sequence_memory_trust_promotion_plan_not_ready"
    assert plan["summary"]["promotion_candidate_count"] == 0
    assert (
        "operator_must_review_all_sequence_memory_trust_candidates_before_promotion"
        in plan["summary"]["blocked_by"]
    )


def test_sequence_memory_trust_promotion_plan_rejects_unsafe_ledger_or_records():
    ledger = _acceptance_ledger()
    forged = {
        **ledger,
        "safety": {
            **ledger["safety"],
            "policy_mutation": True,
            "runtime_memory_mutation": True,
        },
        "summary": {
            **ledger["summary"],
            "accepted_record_count": 2,
            "sequence_policy_promotion_allowed_by_acceptance": True,
        },
        "records": [
            {
                **ledger["records"][0],
                "sequence_policy_promotion_allowed_by_acceptance": True,
                "execution_allowed_by_acceptance": True,
            }
        ],
    }

    plan = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan(
        acceptance_ledger=forged,
    )

    assert plan["status"] == "sequence_memory_trust_promotion_plan_replacement_required"
    assert plan["summary"]["promotion_candidate_count"] == 0
    invalid = plan["summary"]["invalid_record_ids"]
    assert "sequence_memory_trust_acceptance_ledger_policy_mutation_not_false" in invalid
    assert (
        "sequence_memory_trust_acceptance_summary_accepted_record_count_mismatch"
        in invalid
    )
    assert (
        "sequence_memory_trust_acceptance_summary_sequence_policy_promotion_allowed_by_acceptance_not_false"
        in invalid
    )
    assert "sequence-memory-trust-acceptance-1" in invalid


def test_sequence_memory_trust_promotion_plan_rejects_forged_duplicate_reviews():
    ledger = _acceptance_ledger()
    forged = {
        **ledger,
        "summary": {
            **ledger["summary"],
            "acceptance_record_count": 2,
            "accepted_record_count": 1,
            "rejected_record_count": 1,
            "rejected_proposal_ids": ["memory-write-applied"],
        },
        "records": [
            ledger["records"][0],
            {
                **ledger["records"][0],
                "acceptance_id": "sequence-memory-trust-acceptance-2",
                "decision": "rejected",
                "operator_accepted_sequence_memory_trust_candidate": False,
                "operator_rejected_sequence_memory_trust_candidate": True,
            },
        ],
    }

    plan = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan(
        acceptance_ledger=forged,
    )

    assert plan["status"] == "sequence_memory_trust_promotion_plan_replacement_required"
    assert plan["summary"]["promotion_candidate_count"] == 0
    assert plan["summary"]["invalid_record_ids"] == [
        "sequence-memory-trust-acceptance-1",
        "sequence-memory-trust-acceptance-2",
    ]


def test_sequence_memory_trust_promotion_plan_registry_and_goal_are_shadow_only():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID
        in main.method_ids
    )
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID
        not in main.proposal_method_ids
    )
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID
        not in main.executor_method_ids
    )

    snapshot = build_processing_registry_snapshot(include_methods=False)
    plan = snapshot[
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_plan"
    ]
    assert plan["status"] == "sequence_memory_trust_promotion_plan_not_ready"
    assert plan["safety"]["runtime_memory_mutation_allowed_by_plan"] is False
    assert plan["safety"]["policy_mutation_allowed_by_plan"] is False

    memory_row = get_project_goal_capability("long_term_memory_learning")
    assert memory_row is not None
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID
        in memory_row.method_ids
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_plan.py"
        in memory_row.support_paths
    )
    assert memory_row.current_live_write_kinds == ()
