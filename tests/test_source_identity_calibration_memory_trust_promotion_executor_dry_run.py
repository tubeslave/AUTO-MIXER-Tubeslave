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
    build_source_identity_calibration_memory_trust_promotion_execution_plan,
)
from backend.source_identity_calibration_memory_trust_promotion_executor_dry_run import (
    SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID,
    build_source_identity_calibration_memory_trust_promotion_executor_dry_run,
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


def _freshness():
    return {
        "status": "fresh",
        "field_count": 4,
        "fresh_field_count": 4,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def _rollback_evidence():
    return {
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


def _memory_trust_acceptance_record():
    return {
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


def _execution_plan(records=None):
    memory_index = build_source_identity_calibration_memory_index(memory_text=_memory_text())
    review = build_source_identity_calibration_memory_trust_review(
        memory_index=memory_index,
        current_state_cache_freshness=_freshness(),
    )
    trust_ledger = build_source_identity_calibration_memory_trust_acceptance_ledger(
        memory_trust_review=review,
        acceptance_records=[_memory_trust_acceptance_record()],
    )
    promotion_plan = build_source_identity_calibration_memory_trust_promotion_plan(
        acceptance_ledger=trust_ledger,
    )
    contract = build_source_identity_calibration_memory_trust_promotion_contract(
        promotion_plan=promotion_plan,
        current_state_cache_freshness=_freshness(),
        rollback_evidence=_rollback_evidence(),
    )
    acceptance_ledger = (
        build_source_identity_calibration_memory_trust_promotion_acceptance_ledger(
            promotion_contract=contract,
            acceptance_records=records
            if records is not None
            else [_promotion_acceptance_record()],
        )
    )
    return build_source_identity_calibration_memory_trust_promotion_execution_plan(
        promotion_acceptance_ledger=acceptance_ledger,
    )


def test_memory_trust_promotion_executor_dry_run_module_is_transport_free():
    source = Path(
        "backend/source_identity_calibration_memory_trust_promotion_executor_dry_run.py"
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


def test_memory_trust_promotion_executor_dry_run_not_ready_without_execution_plan():
    dry_run = build_source_identity_calibration_memory_trust_promotion_executor_dry_run()

    assert dry_run["type"] == "source_identity_calibration_memory_trust_promotion_executor_dry_run"
    assert (
        dry_run["method_id"]
        == SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID
    )
    assert dry_run["status"] == "memory_trust_promotion_executor_dry_run_not_ready"
    assert dry_run["safety"]["read_only"] is True
    assert dry_run["safety"]["dry_run"] is True
    assert dry_run["safety"]["execution_allowed_by_dry_run"] is False
    assert dry_run["safety"]["memory_trust_promotion_applied"] is False
    assert dry_run["safety"]["runtime_memory_mutation_allowed_by_dry_run"] is False
    assert dry_run["summary"]["dry_run_record_count"] == 0
    assert (
        "execution_candidate_required_before_executor_dry_run"
        in dry_run["summary"]["blocked_by"]
    )


def test_memory_trust_promotion_executor_dry_run_builds_apply_specs_without_apply():
    dry_run = build_source_identity_calibration_memory_trust_promotion_executor_dry_run(
        promotion_execution_plan=_execution_plan(),
    )

    assert dry_run["status"] == "memory_trust_promotion_executor_dry_run_ready"
    assert dry_run["summary"]["execution_candidate_count"] == 1
    assert dry_run["summary"]["dry_run_record_count"] == 1
    assert dry_run["summary"]["would_change_trust_state_count"] == 1
    assert dry_run["summary"]["target_store"] == "source_identity_downstream_gate"
    assert dry_run["summary"]["execution_allowed_by_dry_run"] is False
    assert dry_run["summary"]["memory_trust_promotion_applied"] is False
    assert dry_run["summary"]["downstream_priority_change_allowed_by_dry_run"] is False
    record = dry_run["dry_run_records"][0]
    assert record["target_store"] == "source_identity_downstream_gate"
    assert record["target_path"] == "source_identity_downstream_gate/candidate-1/trust_state"
    assert record["operation"] == "set_source_identity_trust_state"
    assert record["previous_trust_state"] == "legacy_unverified"
    assert record["target_trust_state"] == "trusted_for_downstream_priority"
    assert record["idempotency_key"] == (
        "source_identity_memory_trust_promotion:"
        "candidate-1:trusted_for_downstream_priority:memory-trust-promotion-acceptance-1"
    )
    assert record["dry_run"] is True
    assert record["requires_operator_ack_before_apply"] is True
    assert record["requires_executor_contract_before_apply"] is True
    assert record["execution_allowed_by_dry_run"] is False
    assert record["memory_trust_promotion_applied"] is False
    assert record["memory_trust_promotion_allowed_by_dry_run"] is False
    assert record["runtime_memory_mutation_allowed_by_dry_run"] is False
    assert record["live_authority_granted"] is False
    assert "requested_change" not in dry_run


def test_memory_trust_promotion_executor_dry_run_blocks_apply_request():
    dry_run = build_source_identity_calibration_memory_trust_promotion_executor_dry_run(
        promotion_execution_plan=_execution_plan(),
        apply=True,
        operator_ack=True,
    )

    assert dry_run["status"] == "memory_trust_promotion_executor_apply_blocked"
    assert dry_run["safety"]["apply_requested"] is True
    assert dry_run["safety"]["operator_ack"] is True
    assert dry_run["summary"]["dry_run_record_count"] == 1
    assert dry_run["summary"]["memory_trust_promotion_applied"] is False
    assert "dry_run_does_not_apply_memory_trust_promotion" in dry_run["summary"][
        "blocked_by"
    ]


def test_memory_trust_promotion_executor_dry_run_rejects_unsafe_or_forged_plan():
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
                "runtime_memory_mutation_allowed_by_plan": True,
            }
        ],
    }

    dry_run = build_source_identity_calibration_memory_trust_promotion_executor_dry_run(
        promotion_execution_plan=forged,
    )

    assert dry_run["status"] == "memory_trust_promotion_executor_dry_run_replacement_required"
    assert dry_run["summary"]["dry_run_record_count"] == 0
    assert plan["execution_candidates"][0]["promotion_execution_candidate_id"] in dry_run[
        "summary"
    ]["invalid_record_ids"]
    assert (
        "memory_trust_promotion_execution_plan_summary_execution_candidate_count_mismatch"
        in dry_run["summary"]["invalid_record_ids"]
    )


def test_memory_trust_promotion_executor_dry_run_rejects_duplicate_source_specs():
    plan = _execution_plan()
    candidate = plan["execution_candidates"][0]
    forged = {
        **plan,
        "summary": {
            **plan["summary"],
            "execution_candidate_count": 2,
            "execution_candidate_ids": [
                candidate["promotion_execution_candidate_id"],
                "source_identity_memory_trust_promotion_execution:duplicate",
            ],
            "accepted_source_types": ["lead_vocal", "lead_vocal"],
        },
        "execution_candidates": [
            candidate,
            {
                **candidate,
                "promotion_execution_candidate_id": (
                    "source_identity_memory_trust_promotion_execution:duplicate"
                ),
            },
        ],
    }

    dry_run = build_source_identity_calibration_memory_trust_promotion_executor_dry_run(
        promotion_execution_plan=forged,
    )

    assert dry_run["status"] == "memory_trust_promotion_executor_dry_run_replacement_required"
    assert dry_run["summary"]["dry_run_record_count"] == 0
    assert candidate["promotion_execution_candidate_id"] in dry_run["summary"][
        "invalid_record_ids"
    ]
    assert (
        "source_identity_memory_trust_promotion_execution:duplicate"
        in dry_run["summary"]["invalid_record_ids"]
    )


def test_memory_trust_promotion_executor_dry_run_is_registered_shadow_only():
    method = get_processing_method(
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID
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
            SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID
        ]["safety"]["live_write"]
        is False
    )
    assert (
        snapshot[
            SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID
        ]["summary"]["dry_run_record_count"]
        == 0
    )


def test_memory_trust_promotion_executor_dry_run_tracks_goal_matrix_without_live_widening():
    source_types = get_project_goal_capability("identify_source_types")
    memory = get_project_goal_capability("long_term_memory_learning")

    for row in (source_types, memory):
        assert row is not None
        assert (
            SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID
            in row.method_ids
        )
        assert (
            "backend/source_identity_calibration_memory_trust_promotion_executor_dry_run.py"
            in row.support_paths
        )
        assert row.current_live_write_kinds == ()
    assert source_types.autonomy_mode_ceiling == "analysis"
    assert memory.autonomy_mode_ceiling == "assisted"
