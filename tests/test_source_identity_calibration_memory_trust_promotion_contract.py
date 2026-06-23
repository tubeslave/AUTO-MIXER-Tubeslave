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
from backend.source_identity_calibration_memory_trust_promotion_contract import (
    SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID,
    build_source_identity_calibration_memory_trust_promotion_contract,
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


def _promotion_plan(records=None):
    memory_index = build_source_identity_calibration_memory_index(memory_text=_memory_text())
    review = build_source_identity_calibration_memory_trust_review(
        memory_index=memory_index,
        current_state_cache_freshness=_freshness(),
    )
    ledger = build_source_identity_calibration_memory_trust_acceptance_ledger(
        memory_trust_review=review,
        acceptance_records=records if records is not None else [_acceptance_record()],
    )
    return build_source_identity_calibration_memory_trust_promotion_plan(
        acceptance_ledger=ledger,
    )


def test_memory_trust_promotion_contract_module_is_transport_free():
    source = Path(
        "backend/source_identity_calibration_memory_trust_promotion_contract.py"
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


def test_memory_trust_promotion_contract_not_ready_without_candidates():
    contract = build_source_identity_calibration_memory_trust_promotion_contract()

    assert contract["type"] == "source_identity_calibration_memory_trust_promotion_contract"
    assert (
        contract["method_id"]
        == SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID
    )
    assert contract["status"] == "memory_trust_promotion_contract_not_ready"
    assert contract["safety"]["read_only"] is True
    assert contract["safety"]["runtime_memory_mutation_allowed_by_contract"] is False
    assert contract["safety"]["downstream_priority_change_allowed_by_contract"] is False
    assert contract["summary"]["contract_record_count"] == 0
    assert "promotion_plan_candidate_required_before_contract" in contract["summary"][
        "blocked_by"
    ]


def test_memory_trust_promotion_contract_ready_without_authority():
    contract = build_source_identity_calibration_memory_trust_promotion_contract(
        promotion_plan=_promotion_plan(),
        current_state_cache_freshness=_freshness(),
        rollback_evidence=_rollback_evidence(),
    )

    assert contract["status"] == "manual_memory_trust_promotion_contract_ready"
    assert contract["summary"]["promotion_candidate_count"] == 1
    assert contract["summary"]["contract_record_count"] == 1
    assert contract["summary"]["rollback_strategy"] == "manual_revert_to_previous_trust_state"
    assert contract["summary"]["memory_trust_promotion_allowed_by_contract"] is False
    assert contract["summary"]["downstream_priority_change_allowed_by_contract"] is False
    assert contract["summary"]["runtime_memory_mutation_allowed_by_contract"] is False
    record = contract["contract_records"][0]
    assert record["source_type"] == "lead_vocal"
    assert record["previous_trust_state"] == "legacy_unverified"
    assert record["read_only"] is True
    assert record["live_write"] is False
    assert record["mutates_console_state"] is False
    assert record["queries_console"] is False
    assert record["rollback_evidence_required"] is True
    assert record["memory_trust_promotion_ready_for_operator_review"] is True
    assert record["promotion_executor_required"] is True
    assert record["memory_trust_promotion_allowed_by_contract"] is False
    assert record["downstream_priority_change_allowed_by_contract"] is False
    assert record["runtime_memory_mutation_allowed_by_contract"] is False
    assert record["live_widening_allowed"] is False
    assert record["live_authority_granted"] is False
    assert "requested_change" not in contract


def test_memory_trust_promotion_contract_requires_fresh_replay_context():
    contract = build_source_identity_calibration_memory_trust_promotion_contract(
        promotion_plan=_promotion_plan(),
        current_state_cache_freshness=_freshness(status="stale"),
        rollback_evidence=_rollback_evidence(),
    )

    assert contract["status"] == "memory_trust_promotion_contract_not_ready"
    assert contract["summary"]["contract_record_count"] == 0
    assert "fresh_replay_context_required_before_promotion_contract" in contract[
        "summary"
    ]["blocked_by"]


def test_memory_trust_promotion_contract_requires_provided_replay_context():
    contract = build_source_identity_calibration_memory_trust_promotion_contract(
        promotion_plan=_promotion_plan(),
        rollback_evidence=_rollback_evidence(),
    )

    assert contract["status"] == "memory_trust_promotion_contract_not_ready"
    assert contract["summary"]["current_state_cache_freshness_status"] == "missing"
    assert contract["summary"]["contract_record_count"] == 0
    assert "fresh_replay_context_required_before_promotion_contract" in contract[
        "summary"
    ]["blocked_by"]
    assert contract["safety"]["runtime_memory_mutation_allowed_by_contract"] is False
    assert contract["safety"]["downstream_priority_change_allowed_by_contract"] is False


def test_memory_trust_promotion_contract_requires_rollback_evidence():
    contract = build_source_identity_calibration_memory_trust_promotion_contract(
        promotion_plan=_promotion_plan(),
        current_state_cache_freshness=_freshness(),
    )

    assert contract["status"] == "memory_trust_promotion_contract_not_ready"
    assert contract["summary"]["contract_record_count"] == 0
    assert "rollback_evidence_required_before_promotion_contract" in contract["summary"][
        "blocked_by"
    ]


def test_memory_trust_promotion_contract_rejects_unsafe_plan_or_rollback():
    plan = _promotion_plan()
    forged_plan = {
        **plan,
        "summary": {
            **plan["summary"],
            "promotion_candidate_count": 2,
        },
        "promotion_candidates": [
            {
                **plan["promotion_candidates"][0],
                "downstream_priority_change_allowed_by_plan": True,
            }
        ],
    }

    contract = build_source_identity_calibration_memory_trust_promotion_contract(
        promotion_plan=forged_plan,
        current_state_cache_freshness=_freshness(live_write=True),
        rollback_evidence=_rollback_evidence(
            runtime_memory_mutation=True,
            rollback_strategy="apply_immediately",
        ),
    )

    assert contract["status"] == "memory_trust_promotion_contract_replacement_required"
    assert contract["summary"]["contract_record_count"] == 0
    assert "memory_trust_promotion_plan_summary_promotion_candidate_count_mismatch" in contract[
        "summary"
    ]["invalid_record_ids"]
    assert plan["promotion_candidates"][0]["promotion_candidate_id"] in contract[
        "summary"
    ]["invalid_record_ids"]
    assert "memory_trust_promotion_contract_freshness_live_write_not_false" in contract[
        "summary"
    ]["invalid_record_ids"]
    assert "memory_trust_promotion_contract_rollback_runtime_memory_mutation_not_false" in contract[
        "summary"
    ]["invalid_record_ids"]
    assert "memory_trust_promotion_contract_rollback_strategy_invalid" in contract[
        "summary"
    ]["invalid_record_ids"]


def test_memory_trust_promotion_contract_registry_and_goal_are_shadow_only():
    method = get_processing_method(
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID
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
    assert SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID in main.method_ids
    assert (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID
        not in main.proposal_method_ids
    )
    assert (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID
        not in main.executor_method_ids
    )

    snapshot = build_processing_registry_snapshot(include_methods=False)
    contract = snapshot["source_identity_calibration_memory_trust_promotion_contract"]
    assert contract["status"] == "memory_trust_promotion_contract_not_ready"
    assert contract["safety"]["runtime_memory_mutation_allowed_by_contract"] is False
    assert contract["safety"]["downstream_priority_change_allowed_by_contract"] is False

    source_row = get_project_goal_capability("identify_source_types")
    memory_row = get_project_goal_capability("long_term_memory_learning")
    assert source_row is not None
    assert memory_row is not None
    assert (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID
        in source_row.method_ids
    )
    assert (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID
        in memory_row.method_ids
    )
    assert (
        "backend/source_identity_calibration_memory_trust_promotion_contract.py"
        in source_row.support_paths
    )
    assert (
        "backend/source_identity_calibration_memory_trust_promotion_contract.py"
        in memory_row.support_paths
    )
    assert source_row.current_live_write_kinds == ()
    assert memory_row.current_live_write_kinds == ()
