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
from backend.source_identity_calibration_memory_trust_promotion_acceptance_ledger import (
    build_source_identity_calibration_memory_trust_promotion_acceptance_ledger,
)
from backend.source_identity_calibration_memory_trust_promotion_contract import (
    build_source_identity_calibration_memory_trust_promotion_contract,
)
from backend.source_identity_calibration_memory_trust_promotion_execution_plan import (
    SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_METHOD_ID,
    build_source_identity_calibration_memory_trust_promotion_execution_plan,
)
from backend.source_identity_calibration_memory_trust_promotion_plan import (
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


def _freshness(**overrides):
    payload = {
        "status": "fresh",
        "field_count": 4,
        "fresh_field_count": 4,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    payload.update(overrides)
    return payload


def _rollback_evidence(**overrides):
    payload = {
        "status": "ready",
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "queue_mutation": False,
        "model_mutation": False,
        "runtime_memory_mutation": False,
        "downstream_priority_mutation": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "rollback_strategy": "manual_revert_to_previous_trust_state",
        "previous_trust_state": "legacy_unverified",
        "target_store": "source_identity_downstream_gate",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-06T12:30:00Z",
    }
    payload.update(overrides)
    return payload


def _memory_trust_acceptance_record(**overrides):
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


def _promotion_contract():
    memory_index = build_source_identity_calibration_memory_index(memory_text=_memory_text())
    review = build_source_identity_calibration_memory_trust_review(
        memory_index=memory_index,
        current_state_cache_freshness=_freshness(),
    )
    ledger = build_source_identity_calibration_memory_trust_acceptance_ledger(
        memory_trust_review=review,
        acceptance_records=[_memory_trust_acceptance_record()],
    )
    plan = build_source_identity_calibration_memory_trust_promotion_plan(
        acceptance_ledger=ledger,
    )
    return build_source_identity_calibration_memory_trust_promotion_contract(
        promotion_plan=plan,
        current_state_cache_freshness=_freshness(),
        rollback_evidence=_rollback_evidence(),
    )


def _promotion_acceptance_record(**overrides):
    record = {
        "promotion_acceptance_id": "memory-trust-promotion-acceptance-1",
        "promotion_contract_record_id": (
            "source_identity_memory_trust_promotion_contract:"
            "source_identity_memory_trust_promotion:memory-trust-acceptance-1"
        ),
        "promotion_candidate_id": (
            "source_identity_memory_trust_promotion:memory-trust-acceptance-1"
        ),
        "source_candidate_id": "candidate-1",
        "source_type": "lead_vocal",
        "recommended_trust_state": "trusted_for_downstream_priority",
        "previous_trust_state": "legacy_unverified",
        "rollback_strategy": "manual_revert_to_previous_trust_state",
        "decision": "accepted",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-06T13:00:00Z",
    }
    record.update(overrides)
    return record


def _promotion_acceptance_ledger(records=None):
    return build_source_identity_calibration_memory_trust_promotion_acceptance_ledger(
        promotion_contract=_promotion_contract(),
        acceptance_records=records
        if records is not None
        else [_promotion_acceptance_record()],
    )


def test_memory_trust_promotion_execution_plan_module_is_transport_free():
    source = Path(
        "backend/source_identity_calibration_memory_trust_promotion_execution_plan.py"
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


def test_memory_trust_promotion_execution_plan_not_ready_without_acceptance():
    plan = build_source_identity_calibration_memory_trust_promotion_execution_plan()

    assert plan["type"] == "source_identity_calibration_memory_trust_promotion_execution_plan"
    assert (
        plan["method_id"]
        == SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_METHOD_ID
    )
    assert plan["status"] == "memory_trust_promotion_execution_plan_not_ready"
    assert plan["safety"]["read_only"] is True
    assert plan["safety"]["execution_allowed_by_plan"] is False
    assert plan["safety"]["memory_trust_promotion_allowed_by_plan"] is False
    assert plan["safety"]["downstream_priority_change_allowed_by_plan"] is False
    assert plan["safety"]["runtime_memory_mutation_allowed_by_plan"] is False
    assert plan["summary"]["execution_candidate_count"] == 0
    assert (
        "accepted_memory_trust_promotion_record_required_before_execution_plan"
        in plan["summary"]["blocked_by"]
    )


def test_memory_trust_promotion_execution_plan_waits_for_all_contract_reviews():
    ledger = build_source_identity_calibration_memory_trust_promotion_acceptance_ledger(
        promotion_contract=_promotion_contract(),
    )

    plan = build_source_identity_calibration_memory_trust_promotion_execution_plan(
        promotion_acceptance_ledger=ledger,
    )

    assert plan["status"] == "memory_trust_promotion_execution_plan_not_ready"
    assert plan["summary"]["execution_candidate_count"] == 0
    assert plan["summary"]["unreviewed_contract_record_count"] == 1
    assert (
        "operator_must_review_all_promotion_contract_records_before_execution_plan"
        in plan["summary"]["blocked_by"]
    )


def test_memory_trust_promotion_execution_plan_surfaces_candidates_without_authority():
    plan = build_source_identity_calibration_memory_trust_promotion_execution_plan(
        promotion_acceptance_ledger=_promotion_acceptance_ledger(),
    )

    assert plan["status"] == "manual_memory_trust_promotion_execution_candidate_available"
    assert plan["summary"]["acceptance_record_count"] == 1
    assert plan["summary"]["accepted_record_count"] == 1
    assert plan["summary"]["execution_candidate_count"] == 1
    assert plan["summary"]["accepted_source_types"] == ["lead_vocal"]
    assert plan["summary"]["execution_allowed_by_plan"] is False
    assert plan["summary"]["memory_trust_promotion_allowed_by_plan"] is False
    assert plan["summary"]["downstream_priority_change_allowed_by_plan"] is False
    assert plan["summary"]["runtime_memory_mutation_allowed_by_plan"] is False
    candidate = plan["execution_candidates"][0]
    assert candidate["source_type"] == "lead_vocal"
    assert candidate["read_only"] is True
    assert candidate["live_write"] is False
    assert candidate["mutates_console_state"] is False
    assert candidate["queries_console"] is False
    assert candidate["separate_executor_required"] is True
    assert candidate["execution_allowed_by_plan"] is False
    assert candidate["memory_trust_promotion_allowed_by_plan"] is False
    assert candidate["downstream_priority_change_allowed_by_plan"] is False
    assert candidate["queue_mutation_allowed_by_plan"] is False
    assert candidate["model_mutation_allowed_by_plan"] is False
    assert candidate["runtime_memory_mutation_allowed_by_plan"] is False
    assert candidate["live_widening_allowed"] is False
    assert candidate["live_authority_granted"] is False
    assert "requested_change" not in plan


def test_memory_trust_promotion_execution_plan_rejects_unsafe_or_forged_ledger():
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
                "downstream_priority_change_allowed_by_acceptance": True,
            }
        ],
    }

    plan = build_source_identity_calibration_memory_trust_promotion_execution_plan(
        promotion_acceptance_ledger=forged_ledger,
    )

    assert plan["status"] == "memory_trust_promotion_execution_plan_replacement_required"
    assert plan["summary"]["execution_candidate_count"] == 0
    assert "memory-trust-promotion-acceptance-1" in plan["summary"]["invalid_record_ids"]
    assert (
        "memory_trust_promotion_acceptance_summary_accepted_record_count_mismatch"
        in plan["summary"]["invalid_record_ids"]
    )


def test_memory_trust_promotion_execution_plan_rejects_record_level_live_authority():
    ledger = _promotion_acceptance_ledger()
    forged_ledger = {
        **ledger,
        "records": [
            {
                **ledger["records"][0],
                "live_write": True,
                "queries_console": True,
                "downstream_priority_mutation": True,
                "unsafe_authority_flags": [],
            }
        ],
    }

    plan = build_source_identity_calibration_memory_trust_promotion_execution_plan(
        promotion_acceptance_ledger=forged_ledger,
    )

    assert plan["status"] == "memory_trust_promotion_execution_plan_replacement_required"
    assert plan["summary"]["execution_candidate_count"] == 0
    assert plan["summary"]["invalid_record_ids"] == [
        "memory-trust-promotion-acceptance-1"
    ]


def test_memory_trust_promotion_execution_plan_is_registered_shadow_only():
    method = get_processing_method(
        "source_identity_calibration_memory_trust_promotion_execution_plan"
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
        snapshot["source_identity_calibration_memory_trust_promotion_execution_plan"][
            "safety"
        ]["live_write"]
        is False
    )


def test_memory_trust_promotion_execution_plan_tracks_goal_matrix_without_live_widening():
    source_types = get_project_goal_capability("identify_source_types")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (source_types, memory):
        assert row is not None
        assert (
            "source_identity_calibration_memory_trust_promotion_execution_plan"
            in row.method_ids
        )
        assert (
            "backend/source_identity_calibration_memory_trust_promotion_execution_plan.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert source_types.autonomy_mode_ceiling == "analysis"
    assert memory.autonomy_mode_ceiling == "assisted"
