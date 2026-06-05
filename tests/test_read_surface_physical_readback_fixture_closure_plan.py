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
