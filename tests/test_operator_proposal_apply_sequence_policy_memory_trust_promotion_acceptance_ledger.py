from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_index import (
    build_operator_proposal_apply_sequence_policy_memory_index,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger import (
    build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_contract import (
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


def test_sequence_memory_trust_promotion_acceptance_ledger_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger.py"
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


def test_sequence_memory_trust_promotion_acceptance_requires_contract_records():
    ledger = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger()
    )

    assert (
        ledger["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger"
    )
    assert (
        ledger["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID
    )
    assert ledger["status"] == (
        "operator_sequence_memory_trust_promotion_acceptance_required"
    )
    assert ledger["safety"]["read_only"] is True
    assert ledger["safety"]["runtime_memory_mutation_allowed_by_acceptance"] is False
    assert ledger["safety"]["policy_mutation_allowed_by_acceptance"] is False
    assert ledger["summary"]["promotion_contract_record_count"] == 0
    assert ledger["summary"]["acceptance_record_count"] == 0
    assert (
        "valid_sequence_memory_trust_promotion_contract_records_required_before_acceptance"
        in ledger["summary"]["blocked_by"]
    )


def test_sequence_memory_trust_promotion_acceptance_requires_operator_records_for_contract():
    ledger = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger(
        promotion_contract=_promotion_contract(),
    )

    assert ledger["status"] == (
        "operator_sequence_memory_trust_promotion_acceptance_required"
    )
    assert ledger["summary"]["promotion_contract_record_count"] == 1
    assert ledger["summary"]["unreviewed_contract_record_count"] == 1
    assert (
        "operator_must_review_sequence_memory_trust_promotion_contract_records"
        in ledger["summary"]["blocked_by"]
    )


def test_sequence_memory_trust_promotion_acceptance_records_operator_decision_without_authority():
    ledger = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger(
        promotion_contract=_promotion_contract(),
        acceptance_records=[_promotion_acceptance_record()],
    )

    assert ledger["status"] == (
        "operator_sequence_memory_trust_promotion_acceptance_records_available"
    )
    assert ledger["summary"]["acceptance_record_count"] == 1
    assert ledger["summary"]["accepted_record_count"] == 1
    assert ledger["summary"]["rejected_record_count"] == 0
    assert ledger["summary"]["accepted_proposal_ids"] == ["memory-write-applied"]
    assert ledger["summary"]["sequence_policy_promotion_allowed_by_acceptance"] is False
    assert ledger["summary"]["execution_allowed_by_acceptance"] is False
    assert ledger["summary"]["runtime_memory_mutation_allowed_by_acceptance"] is False
    record = ledger["records"][0]
    assert record["operator_accepted_sequence_memory_trust_promotion_contract"] is True
    assert record["operator_rejected_sequence_memory_trust_promotion_contract"] is False
    assert record["read_only"] is True
    assert record["live_write"] is False
    assert record["apply_attempted"] is False
    assert record["mutates_console_state"] is False
    assert record["queries_console"] is False
    assert record["operator_review_required"] is True
    assert record["queue_mutation"] is False
    assert record["policy_mutation"] is False
    assert record["model_mutation"] is False
    assert record["runtime_memory_mutation"] is False
    assert record["sequence_policy_promotion_allowed_by_acceptance"] is False
    assert record["execution_allowed_by_acceptance"] is False
    assert record["policy_mutation_allowed_by_acceptance"] is False
    assert record["runtime_memory_mutation_allowed_by_acceptance"] is False
    assert record["live_widening_allowed"] is False
    assert record["live_authority_granted"] is False
    assert record["can_import_to_queue"] is False
    assert record["can_apply_live"] is False
    assert "requested_change" not in ledger


def test_sequence_memory_trust_promotion_acceptance_rejects_unsafe_or_mismatched_records():
    ledger = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger(
        promotion_contract=_promotion_contract(),
        acceptance_records=[
            _promotion_acceptance_record(
                value_type="gain",
                sequence_policy_promotion_allowed_by_acceptance=True,
                live_authority_granted=True,
            )
        ],
    )

    assert ledger["status"] == (
        "operator_sequence_memory_trust_promotion_acceptance_replacement_required"
    )
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["invalid_record_ids"] == [
        "sequence-memory-promotion-acceptance-1"
    ]
    record = ledger["records"][0]
    assert record["valid"] is False
    assert (
        "sequence_memory_trust_promotion_acceptance_record_value_type_mismatch"
        in record["violations"]
    )
    assert (
        "sequence_memory_trust_promotion_acceptance_record_unsafe_authority_claim"
        in record["violations"]
    )
    assert "sequence_policy_promotion_allowed_by_acceptance" in record[
        "unsafe_authority_flags"
    ]
    assert "live_authority_granted" in record["unsafe_authority_flags"]


def test_sequence_memory_trust_promotion_acceptance_rejects_unsafe_contract_payload():
    contract = _promotion_contract()
    forged = {
        **contract,
        "safety": {
            **contract["safety"],
            "policy_mutation_allowed_by_contract": True,
        },
        "summary": {
            **contract["summary"],
            "contract_record_count": 2,
            "policy_mutation_allowed_by_contract": True,
        },
        "contract_records": [
            {
                **contract["contract_records"][0],
                "policy_mutation_allowed_by_contract": True,
            }
        ],
    }

    ledger = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger(
        promotion_contract=forged,
        acceptance_records=[_promotion_acceptance_record()],
    )

    assert ledger["status"] == (
        "operator_sequence_memory_trust_promotion_acceptance_replacement_required"
    )
    invalid = ledger["summary"]["invalid_record_ids"]
    assert (
        "sequence_memory_trust_promotion_contract_policy_mutation_allowed_by_contract_not_false"
        in invalid
    )
    assert (
        "sequence_memory_trust_promotion_contract_summary_contract_record_count_mismatch"
        in invalid
    )
    assert (
        "sequence_memory_trust_promotion_contract_summary_policy_mutation_allowed_by_contract_not_false"
        in invalid
    )
    assert contract["contract_records"][0]["promotion_contract_record_id"] in invalid


def test_sequence_memory_trust_promotion_acceptance_registry_and_goal_are_shadow_only():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID
        in main.method_ids
    )
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID
        not in main.proposal_method_ids
    )
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID
        not in main.executor_method_ids
    )

    snapshot = build_processing_registry_snapshot(include_methods=False)
    ledger = snapshot[
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger"
    ]
    assert ledger["status"] == (
        "operator_sequence_memory_trust_promotion_acceptance_required"
    )
    assert ledger["safety"]["runtime_memory_mutation_allowed_by_acceptance"] is False
    assert ledger["safety"]["policy_mutation_allowed_by_acceptance"] is False

    memory_row = get_project_goal_capability("long_term_memory_learning")
    assert memory_row is not None
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID
        in memory_row.method_ids
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger.py"
        in memory_row.support_paths
    )
    assert memory_row.current_live_write_kinds == ()
