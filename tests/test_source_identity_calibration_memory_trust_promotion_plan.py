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
from backend.source_identity_calibration_memory_trust_acceptance_ledger import (
    build_source_identity_calibration_memory_trust_acceptance_ledger,
)
from backend.source_identity_calibration_memory_trust_promotion_plan import (
    SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID,
    build_source_identity_calibration_memory_trust_promotion_plan,
)
from backend.source_identity_calibration_memory_trust_review import (
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


def _freshness():
    return {
        "status": "fresh",
        "field_count": 4,
        "fresh_field_count": 4,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def _acceptance_record(**overrides):
    record = {
        "acceptance_id": "memory-trust-acceptance-1",
        "review_candidate_id": (
            "source_identity_memory_trust_review:"
            "source_identity_memory_index:source_identity_memory:candidate-1"
        ),
        "memory_record_id": "source_identity_memory_index:source_identity_memory:candidate-1",
        "source_candidate_id": "candidate-1",
        "source_type": "lead_vocal",
        "recommended_trust_state": "trusted_for_downstream_priority",
        "decision": "accepted",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-06T12:00:00Z",
    }
    record.update(overrides)
    return record


def _acceptance_ledger(records=None):
    memory_index = build_source_identity_calibration_memory_index(memory_text=_memory_text())
    review = build_source_identity_calibration_memory_trust_review(
        memory_index=memory_index,
        current_state_cache_freshness=_freshness(),
    )
    return build_source_identity_calibration_memory_trust_acceptance_ledger(
        memory_trust_review=review,
        acceptance_records=records if records is not None else [_acceptance_record()],
    )


def test_memory_trust_promotion_plan_module_is_transport_free():
    source = Path(
        "backend/source_identity_calibration_memory_trust_promotion_plan.py"
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


def test_memory_trust_promotion_not_ready_without_accepted_records():
    plan = build_source_identity_calibration_memory_trust_promotion_plan()

    assert plan["type"] == "source_identity_calibration_memory_trust_promotion_plan"
    assert plan["method_id"] == SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID
    assert plan["status"] == "memory_trust_promotion_not_ready"
    assert plan["safety"]["read_only"] is True
    assert plan["safety"]["runtime_memory_mutation_allowed_by_plan"] is False
    assert plan["safety"]["downstream_priority_change_allowed_by_plan"] is False
    assert plan["summary"]["promotion_candidate_count"] == 0
    assert "accepted_memory_trust_record_required_before_promotion" in plan["summary"][
        "blocked_by"
    ]


def test_memory_trust_promotion_surfaces_manual_candidate_without_authority():
    plan = build_source_identity_calibration_memory_trust_promotion_plan(
        acceptance_ledger=_acceptance_ledger(),
    )

    assert plan["status"] == "manual_memory_trust_promotion_candidate_available"
    assert plan["summary"]["accepted_record_count"] == 1
    assert plan["summary"]["promotion_candidate_count"] == 1
    assert plan["summary"]["accepted_source_types"] == ["lead_vocal"]
    assert plan["summary"]["memory_trust_promotion_allowed_by_plan"] is False
    assert plan["summary"]["downstream_priority_change_allowed_by_plan"] is False
    assert plan["summary"]["runtime_memory_mutation_allowed_by_plan"] is False
    candidate = plan["promotion_candidates"][0]
    assert candidate["acceptance_id"] == "memory-trust-acceptance-1"
    assert candidate["source_type"] == "lead_vocal"
    assert candidate["recommended_trust_state"] == "trusted_for_downstream_priority"
    assert candidate["manual_memory_trust_promotion_required"] is True
    assert candidate["memory_trust_promotion_allowed_by_plan"] is False
    assert candidate["downstream_priority_change_allowed_by_plan"] is False
    assert candidate["runtime_memory_mutation_allowed_by_plan"] is False
    assert candidate["live_authority_granted"] is False
    assert "requested_change" not in plan


def test_memory_trust_promotion_waits_for_unreviewed_candidates():
    ledger = _acceptance_ledger(records=[])
    plan = build_source_identity_calibration_memory_trust_promotion_plan(
        acceptance_ledger=ledger,
    )

    assert plan["status"] == "memory_trust_promotion_not_ready"
    assert plan["summary"]["promotion_candidate_count"] == 0
    assert "operator_must_review_all_memory_trust_candidates_before_promotion" in plan[
        "summary"
    ]["blocked_by"]


def test_memory_trust_promotion_rejects_unsafe_ledger_or_records():
    ledger = _acceptance_ledger()
    forged = {
        **ledger,
        "safety": {
            **ledger["safety"],
            "downstream_priority_mutation": True,
            "runtime_memory_mutation": True,
        },
        "summary": {
            **ledger["summary"],
            "accepted_record_count": 2,
            "downstream_priority_change_allowed_by_acceptance": True,
        },
        "records": [
            {
                **ledger["records"][0],
                "downstream_priority_change_allowed_by_acceptance": True,
                "memory_trust_promotion_allowed_by_acceptance": True,
            }
        ],
    }

    plan = build_source_identity_calibration_memory_trust_promotion_plan(
        acceptance_ledger=forged,
    )

    assert plan["status"] == "memory_trust_promotion_replacement_required"
    assert plan["summary"]["promotion_candidate_count"] == 0
    assert "memory_trust_acceptance_ledger_downstream_priority_mutation_not_false" in plan[
        "summary"
    ]["invalid_record_ids"]
    assert "memory_trust_acceptance_summary_accepted_record_count_mismatch" in plan[
        "summary"
    ]["invalid_record_ids"]
    assert "memory_trust_acceptance_summary_downstream_priority_change_allowed_by_acceptance_not_false" in plan[
        "summary"
    ]["invalid_record_ids"]
    assert "memory-trust-acceptance-1" in plan["summary"]["invalid_record_ids"]


def test_memory_trust_promotion_registry_and_goal_are_shadow_only():
    method = get_processing_method(
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID
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
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID in main.method_ids
    assert (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID
        not in main.proposal_method_ids
    )
    assert (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID
        not in main.executor_method_ids
    )

    snapshot = build_processing_registry_snapshot(include_methods=False)
    plan = snapshot["source_identity_calibration_memory_trust_promotion_plan"]
    assert plan["status"] == "memory_trust_promotion_not_ready"
    assert plan["safety"]["runtime_memory_mutation_allowed_by_plan"] is False
    assert plan["safety"]["downstream_priority_change_allowed_by_plan"] is False

    source_row = get_project_goal_capability("identify_source_types")
    memory_row = get_project_goal_capability("long_term_memory_learning")
    assert source_row is not None
    assert memory_row is not None
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID in source_row.method_ids
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID in memory_row.method_ids
    assert (
        "backend/source_identity_calibration_memory_trust_promotion_plan.py"
        in source_row.support_paths
    )
    assert (
        "backend/source_identity_calibration_memory_trust_promotion_plan.py"
        in memory_row.support_paths
    )
    assert source_row.current_live_write_kinds == ()
    assert memory_row.current_live_write_kinds == ()
