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
    SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID,
    build_source_identity_calibration_memory_trust_promotion_acceptance_ledger,
)
from backend.source_identity_calibration_memory_trust_promotion_contract import (
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


def test_memory_trust_promotion_acceptance_ledger_module_is_transport_free():
    source = Path(
        "backend/source_identity_calibration_memory_trust_promotion_acceptance_ledger.py"
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


def test_memory_trust_promotion_acceptance_requires_contract_records():
    ledger = build_source_identity_calibration_memory_trust_promotion_acceptance_ledger()

    assert (
        ledger["type"]
        == "source_identity_calibration_memory_trust_promotion_acceptance_ledger"
    )
    assert (
        ledger["method_id"]
        == SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID
    )
    assert ledger["status"] == "operator_memory_trust_promotion_acceptance_required"
    assert ledger["safety"]["read_only"] is True
    assert ledger["safety"]["runtime_memory_mutation_allowed_by_acceptance"] is False
    assert ledger["safety"]["downstream_priority_change_allowed_by_acceptance"] is False
    assert ledger["summary"]["promotion_contract_record_count"] == 0
    assert ledger["summary"]["acceptance_record_count"] == 0
    assert "valid_memory_trust_promotion_contract_records_required_before_acceptance" in ledger[
        "summary"
    ]["blocked_by"]


def test_memory_trust_promotion_acceptance_requires_operator_records_for_contract():
    ledger = build_source_identity_calibration_memory_trust_promotion_acceptance_ledger(
        promotion_contract=_promotion_contract(),
    )

    assert ledger["status"] == "operator_memory_trust_promotion_acceptance_required"
    assert ledger["summary"]["promotion_contract_record_count"] == 1
    assert ledger["summary"]["unreviewed_contract_record_count"] == 1
    assert "operator_must_review_memory_trust_promotion_contract_records" in ledger[
        "summary"
    ]["blocked_by"]


def test_memory_trust_promotion_acceptance_records_operator_decision_without_authority():
    ledger = build_source_identity_calibration_memory_trust_promotion_acceptance_ledger(
        promotion_contract=_promotion_contract(),
        acceptance_records=[_promotion_acceptance_record()],
    )

    assert ledger["status"] == "operator_memory_trust_promotion_acceptance_records_available"
    assert ledger["summary"]["acceptance_record_count"] == 1
    assert ledger["summary"]["accepted_record_count"] == 1
    assert ledger["summary"]["rejected_record_count"] == 0
    assert ledger["summary"]["accepted_source_types"] == ["lead_vocal"]
    assert ledger["summary"]["memory_trust_promotion_allowed_by_acceptance"] is False
    assert ledger["summary"]["downstream_priority_change_allowed_by_acceptance"] is False
    assert ledger["summary"]["runtime_memory_mutation_allowed_by_acceptance"] is False
    record = ledger["records"][0]
    assert record["operator_accepted_memory_trust_promotion_contract"] is True
    assert record["operator_rejected_memory_trust_promotion_contract"] is False
    assert record["memory_trust_promotion_allowed_by_acceptance"] is False
    assert record["downstream_priority_change_allowed_by_acceptance"] is False
    assert record["runtime_memory_mutation_allowed_by_acceptance"] is False
    assert record["live_authority_granted"] is False
    assert "requested_change" not in ledger


def test_memory_trust_promotion_acceptance_rejects_unsafe_or_mismatched_records():
    ledger = build_source_identity_calibration_memory_trust_promotion_acceptance_ledger(
        promotion_contract=_promotion_contract(),
        acceptance_records=[
            _promotion_acceptance_record(
                source_type="wrong_source",
                downstream_priority_change_allowed_by_acceptance=True,
                live_authority_granted=True,
            )
        ],
    )

    assert (
        ledger["status"]
        == "operator_memory_trust_promotion_acceptance_replacement_required"
    )
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["invalid_record_ids"] == [
        "memory-trust-promotion-acceptance-1"
    ]
    record = ledger["records"][0]
    assert record["valid"] is False
    assert (
        "memory_trust_promotion_acceptance_record_source_type_mismatch"
        in record["violations"]
    )
    assert (
        "memory_trust_promotion_acceptance_record_unsafe_authority_claim"
        in record["violations"]
    )
    assert "downstream_priority_change_allowed_by_acceptance" in record[
        "unsafe_authority_flags"
    ]
    assert "live_authority_granted" in record["unsafe_authority_flags"]


def test_memory_trust_promotion_acceptance_rejects_unsafe_contract_payload():
    contract = _promotion_contract()
    forged = {
        **contract,
        "safety": {
            **contract["safety"],
            "downstream_priority_change_allowed_by_contract": True,
        },
        "summary": {
            **contract["summary"],
            "contract_record_count": 2,
            "downstream_priority_change_allowed_by_contract": True,
        },
        "contract_records": [
            {
                **contract["contract_records"][0],
                "downstream_priority_change_allowed_by_contract": True,
            }
        ],
    }

    ledger = build_source_identity_calibration_memory_trust_promotion_acceptance_ledger(
        promotion_contract=forged,
        acceptance_records=[_promotion_acceptance_record()],
    )

    assert (
        ledger["status"]
        == "operator_memory_trust_promotion_acceptance_replacement_required"
    )
    assert "memory_trust_promotion_contract_downstream_priority_change_allowed_by_contract_not_false" in ledger[
        "summary"
    ]["invalid_record_ids"]
    assert "memory_trust_promotion_contract_summary_contract_record_count_mismatch" in ledger[
        "summary"
    ]["invalid_record_ids"]
    assert "memory_trust_promotion_contract_summary_downstream_priority_change_allowed_by_contract_not_false" in ledger[
        "summary"
    ]["invalid_record_ids"]
    assert contract["contract_records"][0]["promotion_contract_record_id"] in ledger[
        "summary"
    ]["invalid_record_ids"]


def test_memory_trust_promotion_acceptance_registry_and_goal_are_shadow_only():
    method = get_processing_method(
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID
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
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID
        in main.method_ids
    )
    assert (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID
        not in main.proposal_method_ids
    )
    assert (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID
        not in main.executor_method_ids
    )

    snapshot = build_processing_registry_snapshot(include_methods=False)
    ledger = snapshot[
        "source_identity_calibration_memory_trust_promotion_acceptance_ledger"
    ]
    assert ledger["status"] == "operator_memory_trust_promotion_acceptance_required"
    assert ledger["safety"]["runtime_memory_mutation_allowed_by_acceptance"] is False
    assert ledger["safety"]["downstream_priority_change_allowed_by_acceptance"] is False

    source_row = get_project_goal_capability("identify_source_types")
    memory_row = get_project_goal_capability("long_term_memory_learning")
    assert source_row is not None
    assert memory_row is not None
    assert (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID
        in source_row.method_ids
    )
    assert (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID
        in memory_row.method_ids
    )
    assert (
        "backend/source_identity_calibration_memory_trust_promotion_acceptance_ledger.py"
        in source_row.support_paths
    )
    assert (
        "backend/source_identity_calibration_memory_trust_promotion_acceptance_ledger.py"
        in memory_row.support_paths
    )
    assert source_row.current_live_write_kinds == ()
    assert memory_row.current_live_write_kinds == ()
