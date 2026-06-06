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
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan,
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


def _memory_trust_acceptance_record(**overrides):
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


def _promotion_contract():
    index = build_operator_proposal_apply_sequence_policy_memory_index(
        memory_text=_memory_text(),
    )
    review = build_operator_proposal_apply_sequence_policy_memory_trust_review(
        memory_index=index,
        current_sequence_review_freshness=_sequence_context(),
    )
    ledger = build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger(
        memory_trust_review=review,
        acceptance_records=[_memory_trust_acceptance_record()],
    )
    plan = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan(
        acceptance_ledger=ledger,
    )
    return build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract(
        promotion_plan=plan,
        current_sequence_policy_context=_sequence_context(),
        rollback_evidence=_rollback_evidence(),
    )


def _promotion_acceptance_record(**overrides):
    record = {
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
    record.update(overrides)
    return record


def _promotion_acceptance_ledger(records=None):
    return build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger(
        promotion_contract=_promotion_contract(),
        acceptance_records=records
        if records is not None
        else [_promotion_acceptance_record()],
    )


def test_sequence_memory_trust_promotion_execution_plan_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan.py"
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


def test_sequence_memory_trust_promotion_execution_plan_not_ready_without_acceptance():
    plan = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan()

    assert (
        plan["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan"
    )
    assert (
        plan["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_METHOD_ID
    )
    assert plan["status"] == "sequence_memory_trust_promotion_execution_plan_not_ready"
    assert plan["safety"]["read_only"] is True
    assert plan["safety"]["execution_allowed_by_plan"] is False
    assert plan["safety"]["sequence_policy_promotion_allowed_by_plan"] is False
    assert plan["safety"]["policy_mutation_allowed_by_plan"] is False
    assert plan["safety"]["runtime_memory_mutation_allowed_by_plan"] is False
    assert plan["summary"]["execution_candidate_count"] == 0
    assert (
        "accepted_sequence_memory_trust_promotion_record_required_before_execution_plan"
        in plan["summary"]["blocked_by"]
    )


def test_sequence_memory_trust_promotion_execution_plan_waits_for_all_contract_reviews():
    ledger = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger(
        promotion_contract=_promotion_contract(),
    )

    plan = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan(
        promotion_acceptance_ledger=ledger,
    )

    assert plan["status"] == "sequence_memory_trust_promotion_execution_plan_not_ready"
    assert plan["summary"]["execution_candidate_count"] == 0
    assert plan["summary"]["unreviewed_contract_record_count"] == 1
    assert (
        "operator_must_review_all_sequence_promotion_contract_records_before_execution_plan"
        in plan["summary"]["blocked_by"]
    )


def test_sequence_memory_trust_promotion_execution_plan_surfaces_candidates_without_authority():
    plan = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan(
        promotion_acceptance_ledger=_promotion_acceptance_ledger(),
    )

    assert (
        plan["status"]
        == "manual_sequence_memory_trust_promotion_execution_candidate_available"
    )
    assert plan["summary"]["acceptance_record_count"] == 1
    assert plan["summary"]["accepted_record_count"] == 1
    assert plan["summary"]["execution_candidate_count"] == 1
    assert plan["summary"]["accepted_proposal_ids"] == ["memory-write-applied"]
    assert plan["summary"]["execution_allowed_by_plan"] is False
    assert plan["summary"]["sequence_policy_promotion_allowed_by_plan"] is False
    assert plan["summary"]["policy_mutation_allowed_by_plan"] is False
    assert plan["summary"]["runtime_memory_mutation_allowed_by_plan"] is False
    candidate = plan["execution_candidates"][0]
    assert candidate["proposal_id"] == "memory-write-applied"
    assert candidate["read_only"] is True
    assert candidate["live_write"] is False
    assert candidate["apply_attempted"] is False
    assert candidate["mutates_console_state"] is False
    assert candidate["queries_console"] is False
    assert candidate["separate_executor_required"] is True
    assert candidate["execution_allowed_by_plan"] is False
    assert candidate["sequence_policy_promotion_allowed_by_plan"] is False
    assert candidate["queue_mutation_allowed_by_plan"] is False
    assert candidate["policy_mutation_allowed_by_plan"] is False
    assert candidate["model_mutation_allowed_by_plan"] is False
    assert candidate["runtime_memory_mutation_allowed_by_plan"] is False
    assert candidate["live_widening_allowed"] is False
    assert candidate["live_authority_granted"] is False
    assert "requested_change" not in plan


def test_sequence_memory_trust_promotion_execution_plan_rejects_unsafe_or_forged_ledger():
    ledger = _promotion_acceptance_ledger()
    forged_ledger = {
        **ledger,
        "summary": {
            **ledger["summary"],
            "accepted_record_count": 2,
        },
        "records": [
            {
                **ledger["records"][0],
                "policy_mutation_allowed_by_acceptance": True,
            }
        ],
    }

    plan = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan(
        promotion_acceptance_ledger=forged_ledger,
    )

    assert plan["status"] == "sequence_memory_trust_promotion_execution_plan_replacement_required"
    assert plan["summary"]["execution_candidate_count"] == 0
    assert "sequence-memory-promotion-acceptance-1" in plan["summary"][
        "invalid_record_ids"
    ]
    assert (
        "sequence_memory_trust_promotion_acceptance_summary_accepted_record_count_mismatch"
        in plan["summary"]["invalid_record_ids"]
    )


def test_sequence_memory_trust_promotion_execution_plan_rejects_record_level_live_authority():
    ledger = _promotion_acceptance_ledger()
    forged_ledger = {
        **ledger,
        "records": [
            {
                **ledger["records"][0],
                "live_write": True,
                "apply_attempted": True,
                "policy_mutation": True,
                "unsafe_authority_flags": [],
            }
        ],
    }

    plan = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan(
        promotion_acceptance_ledger=forged_ledger,
    )

    assert plan["status"] == "sequence_memory_trust_promotion_execution_plan_replacement_required"
    assert plan["summary"]["execution_candidate_count"] == 0
    assert plan["summary"]["invalid_record_ids"] == [
        "sequence-memory-promotion-acceptance-1"
    ]


def test_sequence_memory_trust_promotion_execution_plan_is_registered_shadow_only():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_METHOD_ID
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
    assert (
        snapshot[
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan"
        ]["safety"]["live_write"]
        is False
    )


def test_sequence_memory_trust_promotion_execution_plan_tracks_goal_matrix_without_live_widening():
    memory = get_project_goal_capability("long_term_memory_learning")

    assert memory is not None
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_METHOD_ID
        in memory.method_ids
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan.py"
        in memory.support_paths
    )
    assert memory.current_live_write_kinds == ()
    assert memory.autonomy_mode_ceiling == "assisted"
