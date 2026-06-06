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
from backend.source_identity_calibration_memory_trust_review import (
    SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_REVIEW_METHOD_ID,
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


def _freshness(status: str = "fresh"):
    return {
        "status": status,
        "field_count": 4,
        "fresh_field_count": 4 if status == "fresh" else 2,
        "stale_field_count": 0 if status == "fresh" else 2,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def _memory_index():
    return build_source_identity_calibration_memory_index(memory_text=_memory_text())


def test_memory_trust_review_module_is_transport_free():
    source = Path("backend/source_identity_calibration_memory_trust_review.py").read_text()

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


def test_memory_trust_review_is_not_ready_without_index_records():
    review = build_source_identity_calibration_memory_trust_review(
        current_state_cache_freshness=_freshness(),
    )

    assert review["type"] == "source_identity_calibration_memory_trust_review"
    assert review["method_id"] == SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_REVIEW_METHOD_ID
    assert review["status"] == "memory_trust_review_not_ready"
    assert review["safety"]["read_only"] is True
    assert review["safety"]["runtime_memory_mutation"] is False
    assert review["safety"]["downstream_priority_mutation"] is False
    assert review["summary"]["trust_review_candidate_count"] == 0
    assert "indexed_source_identity_memory_required_before_trust_review" in review[
        "summary"
    ]["blocked_by"]


def test_memory_trust_review_creates_operator_review_candidate_with_fresh_context():
    review = build_source_identity_calibration_memory_trust_review(
        memory_index=_memory_index(),
        current_state_cache_freshness=_freshness(),
    )

    assert review["status"] == "memory_trust_review_candidate_available"
    assert review["summary"]["memory_record_count"] == 1
    assert review["summary"]["trust_review_candidate_count"] == 1
    assert review["summary"]["current_state_cache_freshness_status"] == "fresh"
    assert review["summary"]["runtime_memory_mutation"] is False
    assert review["summary"]["downstream_priority_mutation"] is False
    candidate = review["trust_review_candidates"][0]
    assert candidate["source_candidate_id"] == "candidate-1"
    assert candidate["source_review_id"] == "review-1"
    assert candidate["source_type"] == "lead_vocal"
    assert candidate["recommended_trust_state"] == "trusted_for_downstream_priority"
    assert candidate["operator_review_required"] is True
    assert candidate["queue_mutation_allowed_by_review"] is False
    assert candidate["model_mutation_allowed_by_review"] is False
    assert candidate["runtime_memory_mutation_allowed_by_review"] is False
    assert candidate["downstream_priority_change_allowed_by_review"] is False
    assert candidate["live_authority_granted"] is False
    assert candidate["full_autonomous_ready"] is False


def test_memory_trust_review_requires_fresh_replay_context():
    review = build_source_identity_calibration_memory_trust_review(
        memory_index=_memory_index(),
        current_state_cache_freshness=_freshness("stale"),
    )

    assert review["status"] == "memory_trust_review_not_ready"
    assert review["summary"]["trust_review_candidate_count"] == 0
    assert review["summary"]["current_state_cache_freshness_status"] == "stale"
    assert "fresh_source_identity_replay_context_required_before_trust_review" in review[
        "summary"
    ]["blocked_by"]


def test_memory_trust_review_rejects_unsafe_index_envelope_and_records():
    index = _memory_index()
    forged = {
        **index,
        "safety": {
            **index["safety"],
            "downstream_priority_mutation": True,
            "runtime_memory_mutation": True,
            "live_authority_granted": True,
        },
        "summary": {
            **index["summary"],
            "memory_record_count": 2,
            "indexed_source_types": ["wrong"],
            "downstream_priority_mutation": True,
        },
        "memory_records": [
            {
                **index["memory_records"][0],
                "usable_for_downstream_priority": True,
                "downstream_priority_change_allowed_by_index": True,
            }
        ],
    }

    review = build_source_identity_calibration_memory_trust_review(
        memory_index=forged,
        current_state_cache_freshness=_freshness(),
    )

    assert review["status"] == "memory_trust_review_replacement_required"
    assert review["summary"]["trust_review_candidate_count"] == 0
    assert "memory_index_envelope_downstream_priority_mutation_not_false" in review[
        "summary"
    ]["invalid_record_ids"]
    assert "memory_index_summary_memory_record_count_mismatch" in review["summary"][
        "invalid_record_ids"
    ]
    assert "memory_index_summary_indexed_source_types_mismatch" in review["summary"][
        "invalid_record_ids"
    ]
    assert (
        "memory_trust_review_index_record_downstream_usable_not_false"
        in review["summary"]["invalid_record_ids"]
    )
    assert (
        "memory_trust_review_index_record_downstream_priority_change_allowed_by_index_not_false"
        in review["summary"]["invalid_record_ids"]
    )


def test_memory_trust_review_rejects_inconsistent_index_status_with_records():
    index = _memory_index()
    forged = {
        **index,
        "status": "memory_index_not_ready",
    }

    review = build_source_identity_calibration_memory_trust_review(
        memory_index=forged,
        current_state_cache_freshness=_freshness(),
    )

    assert review["status"] == "memory_trust_review_replacement_required"
    assert review["summary"]["trust_review_candidate_count"] == 0
    assert "memory_index_status_not_ready_with_records" in review["summary"][
        "invalid_record_ids"
    ]


def test_memory_trust_review_rejects_unsafe_freshness_context():
    review = build_source_identity_calibration_memory_trust_review(
        memory_index=_memory_index(),
        current_state_cache_freshness={
            "status": "fresh",
            "read_only": False,
            "live_write": True,
            "queries_console": True,
        },
    )

    assert review["status"] == "memory_trust_review_replacement_required"
    assert review["summary"]["trust_review_candidate_count"] == 0
    assert "memory_trust_review_freshness_read_only_not_true" in review["summary"][
        "invalid_record_ids"
    ]
    assert "memory_trust_review_freshness_live_write_not_false" in review["summary"][
        "invalid_record_ids"
    ]
    assert "memory_trust_review_freshness_queries_console_not_false" in review[
        "summary"
    ]["invalid_record_ids"]


def test_memory_trust_review_registry_and_goal_are_shadow_only():
    method = get_processing_method(SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_REVIEW_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "memory_learning"
    assert method.can_analyze is True
    assert method.can_propose is False
    assert method.can_write_live is False
    assert method.live_write_kinds == ()

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_REVIEW_METHOD_ID in main.method_ids
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_REVIEW_METHOD_ID not in main.proposal_method_ids
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_REVIEW_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    review = snapshot["source_identity_calibration_memory_trust_review"]
    assert review["status"] == "memory_trust_review_not_ready"
    assert review["safety"]["runtime_memory_mutation"] is False
    assert review["safety"]["downstream_priority_mutation"] is False

    source_row = get_project_goal_capability("identify_source_types")
    memory_row = get_project_goal_capability("long_term_memory_learning")
    assert source_row is not None
    assert memory_row is not None
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_REVIEW_METHOD_ID in source_row.method_ids
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_REVIEW_METHOD_ID in memory_row.method_ids
    assert (
        "backend/source_identity_calibration_memory_trust_review.py"
        in source_row.support_paths
    )
    assert (
        "backend/source_identity_calibration_memory_trust_review.py"
        in memory_row.support_paths
    )
    assert source_row.current_live_write_kinds == ()
    assert memory_row.current_live_write_kinds == ()
