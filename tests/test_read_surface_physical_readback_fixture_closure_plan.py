from pathlib import Path

from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.read_surface_physical_readback_fixture_acceptance_ledger import (
    build_read_surface_physical_readback_fixture_acceptance_ledger,
)
from backend.read_surface_physical_readback_fixture_closure_plan import (
    READ_SURFACE_PHYSICAL_READBACK_FIXTURE_CLOSURE_PLAN_METHOD_ID,
    build_read_surface_physical_readback_fixture_closure_plan,
)
from tests.test_read_surface_physical_readback_fixture_acceptance_ledger import (
    _queue_with_valid_candidate,
)


def _accepted_ledger():
    spec, queue = _queue_with_valid_candidate()
    ledger = build_read_surface_physical_readback_fixture_acceptance_ledger(
        review_queue=queue,
        acceptance_records=[
            {
                "review_id": "review-raw-1",
                "surface_id": spec["surface_id"],
                "artifact_source_id": "raw-mixer-physical.json",
                "required_contract_method_id": spec["required_contract_method_id"],
                "decision": "accepted",
                "reviewed_by": "operator",
                "reviewed_at": "2026-06-05T21:10:00Z",
            }
        ],
    )
    return spec, ledger


def test_fixture_closure_plan_requires_accepted_candidates():
    plan = build_read_surface_physical_readback_fixture_closure_plan()

    assert plan["type"] == "read_surface_physical_readback_fixture_closure_plan"
    assert plan["method_id"] == READ_SURFACE_PHYSICAL_READBACK_FIXTURE_CLOSURE_PLAN_METHOD_ID
    assert plan["status"] == "fixture_closure_not_ready"
    assert plan["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "operator_required": True,
        "execution_allowed_by_plan": False,
        "fixture_update_allowed_by_plan": False,
        "fixture_closure_allowed_by_plan": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }
    assert plan["summary"]["closure_candidate_count"] == 0
    assert plan["summary"]["manual_fixture_closure_candidate_available"] is False
    assert plan["summary"]["manual_fixture_closure_required"] is True
    assert plan["summary"]["physical_fixture_gap_remains"] is True
    assert plan["summary"]["fixture_update_allowed_by_plan"] is False
    assert plan["summary"]["fixture_closure_allowed_by_plan"] is False
    assert plan["summary"]["live_authority_granted"] is False
    assert plan["summary"]["full_autonomous_ready"] is False
    assert "accepted_physical_readback_candidate_required_before_fixture_closure" in plan[
        "summary"
    ]["blocked_by"]


def test_fixture_closure_plan_surfaces_manual_candidate_without_closing_gap():
    spec, ledger = _accepted_ledger()

    plan = build_read_surface_physical_readback_fixture_closure_plan(
        acceptance_ledger=ledger
    )

    assert plan["status"] == "manual_fixture_closure_candidate_available"
    assert plan["summary"]["closure_candidate_count"] == 1
    assert plan["summary"]["closure_candidate_surface_ids"] == [spec["surface_id"]]
    assert plan["summary"]["manual_fixture_closure_candidate_available"] is True
    assert plan["summary"]["physical_fixture_gap_remains"] is True
    assert plan["summary"]["fixture_update_allowed_by_plan"] is False
    assert plan["summary"]["fixture_closure_allowed_by_plan"] is False
    assert plan["summary"]["live_authority_granted"] is False
    assert plan["summary"]["full_autonomous_ready"] is False
    candidate = plan["closure_candidates"][0]
    assert candidate["surface_id"] == spec["surface_id"]
    assert candidate["artifact_source_id"] == "raw-mixer-physical.json"
    assert candidate["manual_fixture_closure_required"] is True
    assert candidate["fixture_update_allowed_by_plan"] is False
    assert candidate["fixture_closure_allowed_by_plan"] is False
    assert candidate["live_authority_granted"] is False
    assert candidate["full_autonomous_ready"] is False


def test_fixture_closure_plan_rejects_incomplete_accepted_ledger_records():
    spec, ledger = _accepted_ledger()
    forged_ledger = {
        **ledger,
        "records": [
            {
                **ledger["records"][0],
                "review_id": "accepted-without-source",
                "artifact_source_id": "",
            },
            {
                **ledger["records"][0],
                "review_id": "accepted-with-local-time",
                "reviewed_at": "2026-06-05 21:10:00",
            },
        ],
        "summary": {
            **ledger["summary"],
            "acceptance_record_count": 2,
            "invalid_record_ids": [],
            "accepted_record_count": 2,
            "accepted_surface_ids": [spec["surface_id"], spec["surface_id"]],
            "unreviewed_candidate_surface_ids": [],
        },
    }

    plan = build_read_surface_physical_readback_fixture_closure_plan(
        acceptance_ledger=forged_ledger
    )

    assert plan["status"] == "fixture_closure_replacement_required"
    assert plan["summary"]["closure_candidate_count"] == 0
    assert plan["summary"]["invalid_record_ids"] == [
        "accepted-without-source",
        "accepted-with-local-time",
    ]
    assert "operator_must_replace_invalid_acceptance_records_before_fixture_closure" in plan[
        "summary"
    ]["blocked_by"]
    assert plan["closure_candidates"] == []
    assert plan["summary"]["physical_fixture_gap_remains"] is True
    assert plan["summary"]["fixture_update_allowed_by_plan"] is False
    assert plan["summary"]["fixture_closure_allowed_by_plan"] is False


def test_fixture_closure_plan_rejects_accepted_record_decision_mismatch():
    spec, ledger = _accepted_ledger()
    forged_ledger = {
        **ledger,
        "records": [
            {
                **ledger["records"][0],
                "review_id": "accepted-flag-rejected-decision",
                "decision": "rejected",
            },
        ],
        "summary": {
            **ledger["summary"],
            "invalid_record_ids": [],
            "accepted_surface_ids": [spec["surface_id"]],
            "unreviewed_candidate_surface_ids": [],
        },
    }

    plan = build_read_surface_physical_readback_fixture_closure_plan(
        acceptance_ledger=forged_ledger
    )

    assert plan["status"] == "fixture_closure_replacement_required"
    assert plan["summary"]["invalid_record_ids"] == [
        "accepted-flag-rejected-decision"
    ]
    assert plan["summary"]["closure_candidate_count"] == 0
    assert plan["summary"]["closure_candidate_surface_ids"] == []
    assert plan["closure_candidates"] == []
    assert "operator_must_replace_invalid_acceptance_records_before_fixture_closure" in plan[
        "summary"
    ]["blocked_by"]
    assert plan["summary"]["fixture_update_allowed_by_plan"] is False
    assert plan["summary"]["fixture_closure_allowed_by_plan"] is False
    assert plan["summary"]["live_authority_granted"] is False


def test_fixture_closure_plan_rejects_accepted_record_without_candidate_link():
    spec, ledger = _accepted_ledger()
    forged_ledger = {
        **ledger,
        "records": [
            {
                **ledger["records"][0],
                "review_id": "accepted-without-candidate-link",
                "candidate_present": False,
            },
        ],
        "summary": {
            **ledger["summary"],
            "invalid_record_ids": [],
            "accepted_surface_ids": [spec["surface_id"]],
            "unreviewed_candidate_surface_ids": [],
        },
    }

    plan = build_read_surface_physical_readback_fixture_closure_plan(
        acceptance_ledger=forged_ledger
    )

    assert plan["status"] == "fixture_closure_replacement_required"
    assert plan["summary"]["invalid_record_ids"] == [
        "accepted-without-candidate-link"
    ]
    assert plan["summary"]["closure_candidate_count"] == 0
    assert plan["summary"]["closure_candidate_surface_ids"] == []
    assert plan["closure_candidates"] == []
    assert "operator_must_replace_invalid_acceptance_records_before_fixture_closure" in plan[
        "summary"
    ]["blocked_by"]
    assert plan["summary"]["fixture_update_allowed_by_plan"] is False
    assert plan["summary"]["fixture_closure_allowed_by_plan"] is False
    assert plan["summary"]["live_authority_granted"] is False


def test_fixture_closure_plan_rejects_ambiguous_surface_review_records():
    spec, ledger = _accepted_ledger()
    accepted_record = ledger["records"][0]
    forged_ledger = {
        **ledger,
        "records": [
            accepted_record,
            {
                **accepted_record,
                "review_id": "review-raw-reject",
                "decision": "rejected",
                "operator_accepted_review_candidate": False,
                "operator_rejected_review_candidate": True,
            },
            {
                **accepted_record,
                "review_id": "review-raw-duplicate",
            },
        ],
        "summary": {
            **ledger["summary"],
            "acceptance_record_count": 3,
            "invalid_record_ids": [],
            "accepted_record_count": 2,
            "rejected_record_count": 1,
            "accepted_surface_ids": [spec["surface_id"], spec["surface_id"]],
            "rejected_surface_ids": [spec["surface_id"]],
            "unreviewed_candidate_surface_ids": [],
        },
    }

    plan = build_read_surface_physical_readback_fixture_closure_plan(
        acceptance_ledger=forged_ledger
    )

    assert plan["status"] == "fixture_closure_replacement_required"
    assert plan["summary"]["accepted_record_count"] == 2
    assert plan["summary"]["invalid_record_ids"] == [
        "review-raw-1",
        "review-raw-reject",
        "review-raw-duplicate",
    ]
    assert plan["summary"]["closure_candidate_count"] == 0
    assert plan["closure_candidates"] == []
    assert "operator_must_replace_invalid_acceptance_records_before_fixture_closure" in plan[
        "summary"
    ]["blocked_by"]
    assert plan["summary"]["physical_fixture_gap_remains"] is True
    assert plan["summary"]["fixture_update_allowed_by_plan"] is False
    assert plan["summary"]["fixture_closure_allowed_by_plan"] is False


def test_fixture_closure_plan_rejects_malformed_ledger_summary_counts():
    _, ledger = _accepted_ledger()
    forged_ledger = {
        **ledger,
        "summary": {
            **ledger["summary"],
            "unreviewed_candidate_count": 1,
            "unreviewed_candidate_surface_ids": [],
        },
    }

    plan = build_read_surface_physical_readback_fixture_closure_plan(
        acceptance_ledger=forged_ledger
    )

    assert plan["status"] == "fixture_closure_replacement_required"
    assert plan["summary"]["invalid_record_ids"] == [
        "ledger_summary_unreviewed_candidate_count_mismatch"
    ]
    assert plan["summary"]["closure_candidate_count"] == 0
    assert plan["summary"]["closure_candidate_surface_ids"] == []
    assert plan["closure_candidates"] == []
    assert "operator_must_replace_invalid_acceptance_records_before_fixture_closure" in plan[
        "summary"
    ]["blocked_by"]
    assert plan["summary"]["physical_fixture_gap_remains"] is True
    assert plan["summary"]["fixture_update_allowed_by_plan"] is False
    assert plan["summary"]["fixture_closure_allowed_by_plan"] is False


def test_fixture_closure_plan_rejects_accepted_surface_id_summary_drift():
    _, ledger = _accepted_ledger()
    forged_ledger = {
        **ledger,
        "summary": {
            **ledger["summary"],
            "accepted_surface_ids": ["wrong_surface"],
        },
    }

    plan = build_read_surface_physical_readback_fixture_closure_plan(
        acceptance_ledger=forged_ledger
    )

    assert plan["status"] == "fixture_closure_replacement_required"
    assert plan["summary"]["invalid_record_ids"] == [
        "ledger_summary_accepted_surface_ids_mismatch"
    ]
    assert plan["summary"]["closure_candidate_count"] == 0
    assert plan["summary"]["closure_candidate_surface_ids"] == []
    assert plan["closure_candidates"] == []
    assert plan["summary"]["physical_fixture_gap_remains"] is True
    assert plan["summary"]["fixture_update_allowed_by_plan"] is False
    assert plan["summary"]["fixture_closure_allowed_by_plan"] is False


def test_fixture_closure_plan_rejects_review_candidate_count_drift():
    _, ledger = _accepted_ledger()
    forged_ledger = {
        **ledger,
        "summary": {
            **ledger["summary"],
            "review_candidate_count": 7,
        },
    }

    plan = build_read_surface_physical_readback_fixture_closure_plan(
        acceptance_ledger=forged_ledger
    )

    assert plan["status"] == "fixture_closure_replacement_required"
    assert plan["summary"]["invalid_record_ids"] == [
        "ledger_summary_review_candidate_count_mismatch"
    ]
    assert plan["summary"]["closure_candidate_count"] == 0
    assert plan["summary"]["closure_candidate_surface_ids"] == []
    assert plan["closure_candidates"] == []
    assert plan["summary"]["physical_fixture_gap_remains"] is True
    assert plan["summary"]["fixture_update_allowed_by_plan"] is False
    assert plan["summary"]["fixture_closure_allowed_by_plan"] is False


def test_fixture_closure_plan_rejects_unsafe_ledger_summary_authority_claims():
    _, ledger = _accepted_ledger()
    forged_ledger = {
        **ledger,
        "summary": {
            **ledger["summary"],
            "operator_review_required": False,
            "physical_fixture_gap_remains": False,
            "fixture_closure_allowed_by_acceptance": True,
            "live_authority_granted": True,
            "full_autonomous_ready": True,
        },
    }

    plan = build_read_surface_physical_readback_fixture_closure_plan(
        acceptance_ledger=forged_ledger
    )

    assert plan["status"] == "fixture_closure_replacement_required"
    assert plan["summary"]["invalid_record_ids"] == [
        "ledger_summary_operator_review_required_not_true",
        "ledger_summary_physical_fixture_gap_remains_not_true",
        "ledger_summary_fixture_closure_allowed_by_acceptance_not_false",
        "ledger_summary_live_authority_granted_not_false",
        "ledger_summary_full_autonomous_ready_not_false",
    ]
    assert plan["summary"]["closure_candidate_count"] == 0
    assert plan["summary"]["closure_candidate_surface_ids"] == []
    assert plan["closure_candidates"] == []
    assert "operator_must_replace_invalid_acceptance_records_before_fixture_closure" in plan[
        "summary"
    ]["blocked_by"]
    assert plan["summary"]["physical_fixture_gap_remains"] is True
    assert plan["summary"]["fixture_update_allowed_by_plan"] is False
    assert plan["summary"]["fixture_closure_allowed_by_plan"] is False
    assert plan["summary"]["live_authority_granted"] is False
    assert plan["summary"]["full_autonomous_ready"] is False


def test_fixture_closure_plan_rejects_unsafe_ledger_envelope():
    _, ledger = _accepted_ledger()
    forged_ledger = {
        **ledger,
        "method_id": "forged_acceptance_ledger",
        "status": "fixture_closure_already_allowed",
        "safety": {
            **ledger["safety"],
            "read_only": False,
            "live_write": True,
            "fixture_closure_allowed_by_acceptance": True,
            "live_authority_granted": True,
        },
    }

    plan = build_read_surface_physical_readback_fixture_closure_plan(
        acceptance_ledger=forged_ledger
    )

    assert plan["status"] == "fixture_closure_replacement_required"
    assert plan["summary"]["invalid_record_ids"] == [
        "ledger_envelope_method_id_mismatch",
        "ledger_envelope_status_unknown",
        "ledger_envelope_read_only_not_true",
        "ledger_envelope_live_write_not_false",
        "ledger_envelope_fixture_closure_allowed_by_acceptance_not_false",
        "ledger_envelope_live_authority_granted_not_false",
    ]
    assert plan["summary"]["closure_candidate_count"] == 0
    assert plan["summary"]["closure_candidate_surface_ids"] == []
    assert plan["closure_candidates"] == []
    assert "operator_must_replace_invalid_acceptance_records_before_fixture_closure" in plan[
        "summary"
    ]["blocked_by"]
    assert plan["summary"]["physical_fixture_gap_remains"] is True
    assert plan["summary"]["fixture_update_allowed_by_plan"] is False
    assert plan["summary"]["fixture_closure_allowed_by_plan"] is False
    assert plan["summary"]["live_authority_granted"] is False
    assert plan["summary"]["full_autonomous_ready"] is False


def test_fixture_closure_plan_rejects_malformed_ledger_identity_envelope():
    _, ledger = _accepted_ledger()
    forged_ledger = {
        **ledger,
        "type": "forged_acceptance_ledger",
        "generated_at": "2026-06-05 21:10:00",
    }

    plan = build_read_surface_physical_readback_fixture_closure_plan(
        acceptance_ledger=forged_ledger
    )

    assert plan["status"] == "fixture_closure_replacement_required"
    assert plan["summary"]["invalid_record_ids"] == [
        "ledger_envelope_type_mismatch",
        "ledger_envelope_generated_at_invalid",
    ]
    assert plan["summary"]["closure_candidate_count"] == 0
    assert plan["summary"]["closure_candidate_surface_ids"] == []
    assert plan["closure_candidates"] == []
    assert "operator_must_replace_invalid_acceptance_records_before_fixture_closure" in plan[
        "summary"
    ]["blocked_by"]
    assert plan["summary"]["physical_fixture_gap_remains"] is True
    assert plan["summary"]["fixture_update_allowed_by_plan"] is False
    assert plan["summary"]["fixture_closure_allowed_by_plan"] is False
    assert plan["summary"]["live_authority_granted"] is False
    assert plan["summary"]["full_autonomous_ready"] is False


def test_fixture_closure_plan_rejects_rejected_surface_id_summary_drift():
    spec, queue = _queue_with_valid_candidate()
    ledger = build_read_surface_physical_readback_fixture_acceptance_ledger(
        review_queue=queue,
        acceptance_records=[
            {
                "review_id": "review-raw-reject",
                "surface_id": spec["surface_id"],
                "artifact_source_id": "raw-mixer-physical.json",
                "required_contract_method_id": spec["required_contract_method_id"],
                "decision": "rejected",
                "reviewed_by": "operator",
                "reviewed_at": "2026-06-05T21:10:00Z",
            }
        ],
    )
    forged_ledger = {
        **ledger,
        "summary": {
            **ledger["summary"],
            "rejected_surface_ids": ["wrong_surface"],
        },
    }

    plan = build_read_surface_physical_readback_fixture_closure_plan(
        acceptance_ledger=forged_ledger
    )

    assert plan["status"] == "fixture_closure_replacement_required"
    assert plan["summary"]["invalid_record_ids"] == [
        "ledger_summary_rejected_surface_ids_mismatch"
    ]
    assert plan["summary"]["closure_candidate_count"] == 0
    assert plan["summary"]["closure_candidate_surface_ids"] == []
    assert plan["closure_candidates"] == []
    assert plan["summary"]["physical_fixture_gap_remains"] is True
    assert plan["summary"]["fixture_update_allowed_by_plan"] is False
    assert plan["summary"]["fixture_closure_allowed_by_plan"] is False


def test_fixture_closure_plan_does_not_create_candidate_for_rejected_records():
    spec, queue = _queue_with_valid_candidate()
    ledger = build_read_surface_physical_readback_fixture_acceptance_ledger(
        review_queue=queue,
        acceptance_records=[
            {
                "review_id": "review-raw-reject",
                "surface_id": spec["surface_id"],
                "artifact_source_id": "raw-mixer-physical.json",
                "required_contract_method_id": spec["required_contract_method_id"],
                "decision": "rejected",
                "reviewed_by": "operator",
                "reviewed_at": "2026-06-05T21:10:00Z",
            }
        ],
    )

    plan = build_read_surface_physical_readback_fixture_closure_plan(
        acceptance_ledger=ledger
    )

    assert plan["status"] == "fixture_closure_not_ready"
    assert plan["summary"]["closure_candidate_count"] == 0
    assert plan["summary"]["rejected_surface_ids"] == [spec["surface_id"]]
    assert "accepted_physical_readback_candidate_required_before_fixture_closure" in plan[
        "summary"
    ]["blocked_by"]


def test_fixture_closure_plan_blocks_invalid_or_unreviewed_candidates():
    spec, queue = _queue_with_valid_candidate()
    invalid_ledger = build_read_surface_physical_readback_fixture_acceptance_ledger(
        review_queue=queue,
        acceptance_records=[
            {
                "review_id": "unsafe-review",
                "surface_id": spec["surface_id"],
                "artifact_source_id": "wrong.json",
                "required_contract_method_id": spec["required_contract_method_id"],
                "decision": "accepted",
                "reviewed_by": "operator",
                "reviewed_at": "2026-06-05T21:10:00Z",
                "live_authority_granted": True,
            }
        ],
    )
    unreviewed_ledger = build_read_surface_physical_readback_fixture_acceptance_ledger(
        review_queue=queue
    )

    invalid_plan = build_read_surface_physical_readback_fixture_closure_plan(
        acceptance_ledger=invalid_ledger
    )
    unreviewed_plan = build_read_surface_physical_readback_fixture_closure_plan(
        acceptance_ledger=unreviewed_ledger
    )

    assert invalid_plan["status"] == "fixture_closure_replacement_required"
    assert invalid_plan["summary"]["invalid_record_ids"] == ["unsafe-review"]
    assert invalid_plan["summary"]["closure_candidate_count"] == 0
    assert "operator_must_replace_invalid_acceptance_records_before_fixture_closure" in invalid_plan[
        "summary"
    ]["blocked_by"]

    assert unreviewed_plan["status"] == "fixture_closure_not_ready"
    assert unreviewed_plan["summary"]["unreviewed_candidate_surface_ids"] == [
        spec["surface_id"]
    ]
    assert "operator_must_review_all_candidates_before_fixture_closure" in unreviewed_plan[
        "summary"
    ]["blocked_by"]


def test_fixture_closure_plan_is_transport_free():
    source = Path("backend/read_surface_physical_readback_fixture_closure_plan.py").read_text(
        encoding="utf-8"
    )

    forbidden = [
        "WingClient",
        "DLiveClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "set_eq",
        "set_compressor",
        "apply_operator_proposal",
    ]
    for token in forbidden:
        assert token not in source


def test_fixture_closure_plan_registry_and_goal_are_shadow_only():
    method = get_processing_method(
        READ_SURFACE_PHYSICAL_READBACK_FIXTURE_CLOSURE_PLAN_METHOD_ID
    )

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert READ_SURFACE_PHYSICAL_READBACK_FIXTURE_CLOSURE_PLAN_METHOD_ID not in (
        main.executor_method_ids
    )

    snapshot = build_processing_registry_snapshot(include_methods=False)
    plan = snapshot["read_surface_physical_readback_fixture_closure_plan"]
    assert plan["safety"]["fixture_update_allowed_by_plan"] is False
    assert plan["safety"]["fixture_closure_allowed_by_plan"] is False
    assert plan["summary"]["physical_fixture_gap_remains"] is True

    row = get_project_goal_capability("console_read_parameters")
    assert row is not None
    assert READ_SURFACE_PHYSICAL_READBACK_FIXTURE_CLOSURE_PLAN_METHOD_ID in row.method_ids
    assert "backend/read_surface_physical_readback_fixture_closure_plan.py" in row.support_paths
    assert row.current_live_write_kinds == ()
