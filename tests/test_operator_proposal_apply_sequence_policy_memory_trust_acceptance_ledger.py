from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_index import (
    build_operator_proposal_apply_sequence_policy_memory_index,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_ACCEPTANCE_LEDGER_METHOD_ID,
    build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger,
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


def _memory_trust_review():
    index = build_operator_proposal_apply_sequence_policy_memory_index(
        memory_text=_memory_text(),
    )
    return build_operator_proposal_apply_sequence_policy_memory_trust_review(
        memory_index=index,
        current_sequence_review_freshness=_freshness(),
    )


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


def test_sequence_memory_trust_acceptance_ledger_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger.py"
    ).read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "apply_operator_proposal",
        "write_text",
    ]
    for token in forbidden:
        assert token not in source


def test_sequence_memory_trust_acceptance_requires_review_candidates():
    ledger = build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger()

    assert (
        ledger["type"]
        == "operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger"
    )
    assert (
        ledger["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_ACCEPTANCE_LEDGER_METHOD_ID
    )
    assert ledger["status"] == "operator_sequence_memory_trust_acceptance_required"
    assert ledger["safety"]["read_only"] is True
    assert ledger["safety"]["runtime_memory_mutation"] is False
    assert ledger["safety"]["policy_mutation"] is False
    assert ledger["summary"]["trust_review_candidate_count"] == 0
    assert ledger["summary"]["acceptance_record_count"] == 0
    assert (
        "valid_sequence_memory_trust_review_candidates_required_before_acceptance"
        in ledger["summary"]["blocked_by"]
    )


def test_sequence_memory_trust_acceptance_requires_operator_records_for_candidates():
    ledger = build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger(
        memory_trust_review=_memory_trust_review(),
    )

    assert ledger["status"] == "operator_sequence_memory_trust_acceptance_required"
    assert ledger["summary"]["trust_review_candidate_count"] == 1
    assert ledger["summary"]["unreviewed_candidate_count"] == 1
    assert (
        "operator_must_review_sequence_memory_trust_candidates_before_promotion"
        in ledger["summary"]["blocked_by"]
    )


def test_sequence_memory_trust_acceptance_records_operator_acceptance_without_promotion():
    ledger = build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger(
        memory_trust_review=_memory_trust_review(),
        acceptance_records=[_acceptance_record()],
    )

    assert ledger["status"] == "operator_sequence_memory_trust_acceptance_records_available"
    assert ledger["summary"]["acceptance_record_count"] == 1
    assert ledger["summary"]["accepted_record_count"] == 1
    assert ledger["summary"]["rejected_record_count"] == 0
    assert ledger["summary"]["accepted_proposal_ids"] == ["memory-write-applied"]
    assert ledger["summary"]["sequence_policy_promotion_allowed_by_acceptance"] is False
    assert ledger["summary"]["execution_allowed_by_acceptance"] is False
    assert ledger["summary"]["runtime_memory_mutation_allowed_by_acceptance"] is False
    record = ledger["records"][0]
    assert record["operator_accepted_sequence_memory_trust_candidate"] is True
    assert record["operator_rejected_sequence_memory_trust_candidate"] is False
    assert record["sequence_policy_promotion_allowed_by_acceptance"] is False
    assert record["execution_allowed_by_acceptance"] is False
    assert record["runtime_memory_mutation_allowed_by_acceptance"] is False
    assert record["live_authority_granted"] is False
    assert "requested_change" not in ledger


def test_sequence_memory_trust_acceptance_rejects_unsafe_or_mismatched_records():
    ledger = build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger(
        memory_trust_review=_memory_trust_review(),
        acceptance_records=[
            _acceptance_record(
                value_type="gain",
                execution_allowed_by_acceptance=True,
                policy_mutation_allowed_by_acceptance=True,
                live_authority_granted=True,
            )
        ],
    )

    assert ledger["status"] == "operator_sequence_memory_trust_acceptance_replacement_required"
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["invalid_record_ids"] == [
        "sequence-memory-trust-acceptance-1"
    ]
    record = ledger["records"][0]
    assert record["valid"] is False
    assert "sequence_memory_acceptance_record_value_type_mismatch" in record["violations"]
    assert "sequence_memory_acceptance_record_unsafe_authority_claim" in record[
        "violations"
    ]
    assert "execution_allowed_by_acceptance" in record["unsafe_authority_flags"]
    assert "policy_mutation_allowed_by_acceptance" in record["unsafe_authority_flags"]
    assert "live_authority_granted" in record["unsafe_authority_flags"]


def test_sequence_memory_trust_acceptance_rejects_duplicate_candidate_decisions():
    ledger = build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger(
        memory_trust_review=_memory_trust_review(),
        acceptance_records=[
            _acceptance_record(acceptance_id="sequence-memory-trust-acceptance-1"),
            _acceptance_record(
                acceptance_id="sequence-memory-trust-acceptance-2",
                decision="rejected",
            ),
        ],
    )

    assert ledger["status"] == "operator_sequence_memory_trust_acceptance_replacement_required"
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["rejected_record_count"] == 0
    assert ledger["summary"]["invalid_record_count"] == 2
    assert ledger["summary"]["invalid_record_ids"] == [
        "sequence-memory-trust-acceptance-1",
        "sequence-memory-trust-acceptance-2",
    ]
    assert ledger["summary"]["unreviewed_candidate_count"] == 1
    assert ledger["summary"]["execution_allowed_by_acceptance"] is False
    assert ledger["summary"]["policy_mutation_allowed_by_acceptance"] is False
    assert ledger["summary"]["runtime_memory_mutation_allowed_by_acceptance"] is False
    assert ledger["summary"]["live_authority_granted"] is False
    for record in ledger["records"]:
        assert record["valid"] is False
        assert (
            "sequence_memory_acceptance_record_review_candidate_duplicate"
            in record["violations"]
        )
        assert record["operator_accepted_sequence_memory_trust_candidate"] is False
        assert record["operator_rejected_sequence_memory_trust_candidate"] is False
        assert record["execution_allowed_by_acceptance"] is False
        assert record["policy_mutation_allowed_by_acceptance"] is False
        assert record["runtime_memory_mutation_allowed_by_acceptance"] is False
        assert record["live_authority_granted"] is False


def test_sequence_memory_trust_acceptance_rejects_unsafe_review_payload():
    review = _memory_trust_review()
    forged = {
        **review,
        "safety": {
            **review["safety"],
            "policy_mutation": True,
            "runtime_memory_mutation": True,
        },
        "summary": {
            **review["summary"],
            "trust_review_candidate_count": 2,
            "policy_mutation": True,
        },
    }

    ledger = build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger(
        memory_trust_review=forged,
        acceptance_records=[_acceptance_record()],
    )

    assert ledger["status"] == "operator_sequence_memory_trust_acceptance_replacement_required"
    invalid = ledger["summary"]["invalid_record_ids"]
    assert "sequence_memory_trust_review_envelope_policy_mutation_not_false" in invalid
    assert "sequence_memory_trust_review_summary_candidate_count_mismatch" in invalid
    assert "sequence_memory_trust_review_summary_policy_mutation_not_false" in invalid


def test_sequence_memory_trust_acceptance_registry_and_goal_are_shadow_only():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_ACCEPTANCE_LEDGER_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_ACCEPTANCE_LEDGER_METHOD_ID
        in main.method_ids
    )
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_ACCEPTANCE_LEDGER_METHOD_ID
        not in main.proposal_method_ids
    )
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_ACCEPTANCE_LEDGER_METHOD_ID
        not in main.executor_method_ids
    )

    snapshot = build_processing_registry_snapshot(include_methods=False)
    ledger = snapshot[
        "operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger"
    ]
    assert ledger["status"] == "operator_sequence_memory_trust_acceptance_required"
    assert ledger["safety"]["runtime_memory_mutation"] is False
    assert ledger["safety"]["policy_mutation"] is False

    memory_row = get_project_goal_capability("long_term_memory_learning")
    assert memory_row is not None
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_ACCEPTANCE_LEDGER_METHOD_ID
        in memory_row.method_ids
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger.py"
        in memory_row.support_paths
    )
    assert memory_row.current_live_write_kinds == ()
