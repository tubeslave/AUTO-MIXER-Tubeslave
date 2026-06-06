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
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID,
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


def _execution_plan(records=None):
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
            acceptance_records=records
            if records is not None
            else [_promotion_acceptance_record()],
        )
    )
    return build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan(
        promotion_acceptance_ledger=acceptance_ledger,
    )


def test_sequence_memory_trust_promotion_executor_dry_run_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run.py"
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


def test_sequence_memory_trust_promotion_executor_dry_run_not_ready_without_execution_plan():
    dry_run = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run()
    )

    assert (
        dry_run["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run"
    )
    assert (
        dry_run["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID
    )
    assert dry_run["status"] == "sequence_memory_trust_promotion_executor_dry_run_not_ready"
    assert dry_run["safety"]["read_only"] is True
    assert dry_run["safety"]["dry_run"] is True
    assert dry_run["safety"]["execution_allowed_by_dry_run"] is False
    assert dry_run["safety"]["sequence_policy_promotion_applied"] is False
    assert dry_run["safety"]["policy_mutation_allowed_by_dry_run"] is False
    assert dry_run["safety"]["runtime_memory_mutation_allowed_by_dry_run"] is False
    assert dry_run["summary"]["dry_run_record_count"] == 0
    assert (
        "execution_candidate_required_before_sequence_executor_dry_run"
        in dry_run["summary"]["blocked_by"]
    )


def test_sequence_memory_trust_promotion_executor_dry_run_builds_apply_specs_without_apply():
    dry_run = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run(
        promotion_execution_plan=_execution_plan(),
    )

    assert dry_run["status"] == "sequence_memory_trust_promotion_executor_dry_run_ready"
    assert dry_run["summary"]["execution_candidate_count"] == 1
    assert dry_run["summary"]["dry_run_record_count"] == 1
    assert dry_run["summary"]["would_change_policy_state_count"] == 1
    assert dry_run["summary"]["target_policy_store"] == "operator_sequence_policy_trust_gate"
    assert dry_run["summary"]["accepted_proposal_ids"] == ["memory-write-applied"]
    assert dry_run["summary"]["execution_allowed_by_dry_run"] is False
    assert dry_run["summary"]["sequence_policy_promotion_applied"] is False
    assert dry_run["summary"]["sequence_policy_promotion_allowed_by_dry_run"] is False
    assert dry_run["summary"]["policy_mutation_allowed_by_dry_run"] is False
    record = dry_run["dry_run_records"][0]
    assert record["target_policy_store"] == "operator_sequence_policy_trust_gate"
    assert record["target_policy_path"] == (
        "operator_sequence_policy_trust_gate/"
        "memory-write-applied/memory-write:memory-write-applied/trust_state"
    )
    assert record["operation"] == "set_operator_sequence_policy_trust_state"
    assert record["previous_policy_state"] == "not_trusted_for_sequence_policy"
    assert record["target_policy_state"] == "trusted_for_sequence_policy"
    assert record["idempotency_key"] == (
        "operator_sequence_policy_memory_trust_promotion:"
        "memory-write-applied:memory-write:memory-write-applied:"
        "sequence-memory-promotion-acceptance-1"
    )
    assert record["dry_run"] is True
    assert record["requires_operator_ack_before_apply"] is True
    assert record["requires_executor_contract_before_apply"] is True
    assert record["execution_allowed_by_dry_run"] is False
    assert record["sequence_policy_promotion_applied"] is False
    assert record["sequence_policy_promotion_allowed_by_dry_run"] is False
    assert record["policy_mutation_allowed_by_dry_run"] is False
    assert record["runtime_memory_mutation_allowed_by_dry_run"] is False
    assert record["live_authority_granted"] is False
    assert "requested_change" not in dry_run


def test_sequence_memory_trust_promotion_executor_dry_run_blocks_apply_request():
    dry_run = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run(
        promotion_execution_plan=_execution_plan(),
        apply=True,
        operator_ack=True,
    )

    assert dry_run["status"] == "sequence_memory_trust_promotion_executor_apply_blocked"
    assert dry_run["safety"]["apply_requested"] is True
    assert dry_run["safety"]["operator_ack"] is True
    assert dry_run["summary"]["dry_run_record_count"] == 1
    assert dry_run["summary"]["sequence_policy_promotion_applied"] is False
    assert "dry_run_does_not_apply_sequence_policy_promotion" in dry_run["summary"][
        "blocked_by"
    ]


def test_sequence_memory_trust_promotion_executor_dry_run_rejects_unsafe_or_forged_plan():
    plan = _execution_plan()
    forged = {
        **plan,
        "summary": {
            **plan["summary"],
            "execution_candidate_count": 2,
        },
        "execution_candidates": [
            {
                **plan["execution_candidates"][0],
                "policy_mutation_allowed_by_plan": True,
            }
        ],
    }

    dry_run = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run(
        promotion_execution_plan=forged,
    )

    assert (
        dry_run["status"]
        == "sequence_memory_trust_promotion_executor_dry_run_replacement_required"
    )
    assert dry_run["summary"]["dry_run_record_count"] == 0
    assert plan["execution_candidates"][0]["promotion_execution_candidate_id"] in dry_run[
        "summary"
    ]["invalid_record_ids"]
    assert (
        "sequence_memory_trust_promotion_execution_plan_summary_execution_candidate_count_mismatch"
        in dry_run["summary"]["invalid_record_ids"]
    )


def test_sequence_memory_trust_promotion_executor_dry_run_rejects_duplicate_policy_specs():
    plan = _execution_plan()
    candidate = plan["execution_candidates"][0]
    forged = {
        **plan,
        "summary": {
            **plan["summary"],
            "execution_candidate_count": 2,
            "execution_candidate_ids": [
                candidate["promotion_execution_candidate_id"],
                "operator_sequence_memory_trust_promotion_execution:duplicate",
            ],
            "accepted_proposal_ids": ["memory-write-applied", "memory-write-applied"],
        },
        "execution_candidates": [
            candidate,
            {
                **candidate,
                "promotion_execution_candidate_id": (
                    "operator_sequence_memory_trust_promotion_execution:duplicate"
                ),
            },
        ],
    }

    dry_run = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run(
        promotion_execution_plan=forged,
    )

    assert (
        dry_run["status"]
        == "sequence_memory_trust_promotion_executor_dry_run_replacement_required"
    )
    assert dry_run["summary"]["dry_run_record_count"] == 0
    assert candidate["promotion_execution_candidate_id"] in dry_run["summary"][
        "invalid_record_ids"
    ]
    assert (
        "operator_sequence_memory_trust_promotion_execution:duplicate"
        in dry_run["summary"]["invalid_record_ids"]
    )


def test_sequence_memory_trust_promotion_executor_dry_run_is_registered_shadow_only():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID
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
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID
        ]["safety"]["live_write"]
        is False
    )
    assert (
        snapshot[
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID
        ]["summary"]["dry_run_record_count"]
        == 0
    )


def test_sequence_memory_trust_promotion_executor_dry_run_tracks_goal_matrix_without_live_widening():
    memory = get_project_goal_capability("long_term_memory_learning")

    assert memory is not None
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID
        in memory.method_ids
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run.py"
        in memory.support_paths
    )
    assert memory.current_live_write_kinds == ()
    assert memory.autonomy_mode_ceiling == "assisted"
