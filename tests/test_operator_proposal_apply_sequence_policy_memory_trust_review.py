from pathlib import Path

from backend.operator_proposal_apply_sequence_policy_memory_index import (
    build_operator_proposal_apply_sequence_policy_memory_index,
)
from backend.operator_proposal_apply_sequence_policy_memory_trust_review import (
    OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_METHOD_ID,
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


def _freshness(status: str = "fresh"):
    return {
        "status": status,
        "field_count": 4,
        "fresh_field_count": 4 if status == "fresh" else 2,
        "stale_field_count": 0 if status == "fresh" else 2,
        "read_only": True,
        "live_write": False,
        "apply_attempted": False,
        "queries_console": False,
        "queue_mutation": False,
        "policy_mutation": False,
    }


def _memory_index():
    return build_operator_proposal_apply_sequence_policy_memory_index(
        memory_text=_memory_text(),
    )


def test_sequence_memory_trust_review_module_is_transport_free():
    source = Path(
        "backend/operator_proposal_apply_sequence_policy_memory_trust_review.py"
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


def test_sequence_memory_trust_review_is_not_ready_without_index_records():
    review = build_operator_proposal_apply_sequence_policy_memory_trust_review(
        current_sequence_review_freshness=_freshness(),
    )

    assert review["type"] == "operator_proposal_apply_sequence_policy_memory_trust_review"
    assert (
        review["method_id"]
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_METHOD_ID
    )
    assert review["status"] == "sequence_memory_trust_review_not_ready"
    assert review["safety"]["read_only"] is True
    assert review["safety"]["runtime_memory_mutation"] is False
    assert review["safety"]["policy_mutation"] is False
    assert review["summary"]["trust_review_candidate_count"] == 0
    assert "indexed_sequence_policy_memory_required_before_trust_review" in review[
        "summary"
    ]["blocked_by"]


def test_sequence_memory_trust_review_creates_operator_review_candidate_with_fresh_context():
    review = build_operator_proposal_apply_sequence_policy_memory_trust_review(
        memory_index=_memory_index(),
        current_sequence_review_freshness=_freshness(),
    )

    assert review["status"] == "sequence_memory_trust_review_candidate_available"
    assert review["summary"]["memory_record_count"] == 1
    assert review["summary"]["trust_review_candidate_count"] == 1
    assert review["summary"]["current_sequence_review_freshness_status"] == "fresh"
    assert review["summary"]["execution_allowed_by_review"] is False
    assert review["summary"]["policy_mutation"] is False
    assert review["summary"]["runtime_memory_mutation"] is False
    candidate = review["trust_review_candidates"][0]
    assert candidate["proposal_id"] == "memory-write-applied"
    assert candidate["learning_scope"] == "supervised_fader_gain_sequence_execution"
    assert candidate["value_type"] == "fader"
    assert candidate["channel"] == 2
    assert candidate["effective_value"] == -8.5
    assert candidate["idempotency_key"] == "memory-write:memory-write-applied"
    assert candidate["operator_review_required"] is True
    assert candidate["execution_allowed_by_review"] is False
    assert candidate["queue_mutation_allowed_by_review"] is False
    assert candidate["policy_mutation_allowed_by_review"] is False
    assert candidate["model_mutation_allowed_by_review"] is False
    assert candidate["runtime_memory_mutation_allowed_by_review"] is False
    assert candidate["live_authority_granted"] is False
    assert candidate["full_autonomous_ready"] is False


def test_sequence_memory_trust_review_requires_fresh_review_context():
    review = build_operator_proposal_apply_sequence_policy_memory_trust_review(
        memory_index=_memory_index(),
        current_sequence_review_freshness=_freshness("stale"),
    )

    assert review["status"] == "sequence_memory_trust_review_not_ready"
    assert review["summary"]["trust_review_candidate_count"] == 0
    assert review["summary"]["current_sequence_review_freshness_status"] == "stale"
    assert "fresh_sequence_review_context_required_before_trust_review" in review[
        "summary"
    ]["blocked_by"]


def test_sequence_memory_trust_review_requires_provided_review_context():
    review = build_operator_proposal_apply_sequence_policy_memory_trust_review(
        memory_index=_memory_index(),
    )

    assert review["status"] == "sequence_memory_trust_review_not_ready"
    assert review["summary"]["trust_review_candidate_count"] == 0
    assert review["summary"]["current_sequence_review_freshness_status"] == "not_provided"
    assert "fresh_sequence_review_context_required_before_trust_review" in review[
        "summary"
    ]["blocked_by"]
    assert review["summary"]["execution_allowed_by_review"] is False
    assert review["summary"]["policy_mutation"] is False
    assert review["summary"]["runtime_memory_mutation"] is False


def test_sequence_memory_trust_review_rejects_unsafe_index_envelope_and_records():
    index = _memory_index()
    forged = {
        **index,
        "safety": {
            **index["safety"],
            "policy_mutation": True,
            "runtime_memory_mutation": True,
            "live_authority_granted": True,
        },
        "summary": {
            **index["summary"],
            "memory_record_count": 2,
            "indexed_value_types": ["gain"],
            "policy_mutation": True,
        },
        "memory_records": [
            {
                **index["memory_records"][0],
                "usable_for_sequence_policy_review": False,
                "execution_allowed_by_index": True,
                "policy_mutation_allowed_by_index": True,
            }
        ],
    }

    review = build_operator_proposal_apply_sequence_policy_memory_trust_review(
        memory_index=forged,
        current_sequence_review_freshness=_freshness(),
    )

    assert review["status"] == "sequence_memory_trust_review_replacement_required"
    assert review["summary"]["trust_review_candidate_count"] == 0
    invalid = review["summary"]["invalid_record_ids"]
    assert "sequence_memory_index_envelope_policy_mutation_not_false" in invalid
    assert "sequence_memory_index_summary_memory_record_count_mismatch" in invalid
    assert "sequence_memory_index_summary_indexed_value_types_mismatch" in invalid
    assert "sequence_memory_trust_review_index_record_policy_usable_not_true" in invalid
    assert (
        "sequence_memory_trust_review_index_record_execution_allowed_by_index_not_false"
        in invalid
    )
    assert (
        "sequence_memory_trust_review_index_record_policy_mutation_allowed_by_index_not_false"
        in invalid
    )


def test_sequence_memory_trust_review_rejects_inconsistent_index_status_with_records():
    index = _memory_index()
    forged = {
        **index,
        "status": "sequence_memory_index_not_ready",
    }

    review = build_operator_proposal_apply_sequence_policy_memory_trust_review(
        memory_index=forged,
        current_sequence_review_freshness=_freshness(),
    )

    assert review["status"] == "sequence_memory_trust_review_replacement_required"
    assert review["summary"]["trust_review_candidate_count"] == 0
    assert "sequence_memory_index_status_not_ready_with_records" in review["summary"][
        "invalid_record_ids"
    ]


def test_sequence_memory_trust_review_rejects_unsafe_freshness_context():
    review = build_operator_proposal_apply_sequence_policy_memory_trust_review(
        memory_index=_memory_index(),
        current_sequence_review_freshness={
            "status": "fresh",
            "read_only": False,
            "live_write": True,
            "apply_attempted": True,
            "queries_console": True,
            "queue_mutation": True,
            "policy_mutation": True,
        },
    )

    assert review["status"] == "sequence_memory_trust_review_replacement_required"
    assert review["summary"]["trust_review_candidate_count"] == 0
    invalid = review["summary"]["invalid_record_ids"]
    assert "sequence_memory_trust_review_freshness_read_only_not_true" in invalid
    assert "sequence_memory_trust_review_freshness_live_write_not_false" in invalid
    assert "sequence_memory_trust_review_freshness_apply_attempted_not_false" in invalid
    assert "sequence_memory_trust_review_freshness_queries_console_not_false" in invalid
    assert "sequence_memory_trust_review_freshness_queue_mutation_not_false" in invalid
    assert "sequence_memory_trust_review_freshness_policy_mutation_not_false" in invalid


def test_sequence_memory_trust_review_registry_and_goal_are_shadow_only():
    method = get_processing_method(
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_METHOD_ID
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
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_METHOD_ID
        in main.method_ids
    )
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_METHOD_ID
        not in main.proposal_method_ids
    )
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_METHOD_ID
        not in main.executor_method_ids
    )

    snapshot = build_processing_registry_snapshot(include_methods=False)
    review = snapshot["operator_proposal_apply_sequence_policy_memory_trust_review"]
    assert review["status"] == "sequence_memory_trust_review_not_ready"
    assert review["safety"]["runtime_memory_mutation"] is False
    assert review["safety"]["policy_mutation"] is False

    memory_row = get_project_goal_capability("long_term_memory_learning")
    assert memory_row is not None
    assert (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_METHOD_ID
        in memory_row.method_ids
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_review.py"
        in memory_row.support_paths
    )
    assert memory_row.current_live_write_kinds == ()
