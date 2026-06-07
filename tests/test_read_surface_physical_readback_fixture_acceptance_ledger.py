from pathlib import Path

from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.read_surface_physical_readback_artifact_review_queue import INTAKE_SPECS
from backend.read_surface_physical_readback_fixture_acceptance_ledger import (
    READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_METHOD_ID,
    build_read_surface_physical_readback_fixture_acceptance_ledger,
)
from tests.test_read_surface_physical_readback_artifact_review_queue import (
    _snapshot_for_spec,
    _snapshot_map,
)


def _queue_with_valid_candidate():
    spec = INTAKE_SPECS[0]
    snapshot = _snapshot_for_spec(spec, valid=1)
    snapshot["summary"]["artifact_source_ids"] = ["raw-mixer-physical.json"]
    queue = build_processing_registry_snapshot(include_methods=False)[
        "read_surface_physical_readback_artifact_review_queue"
    ]
    from backend.read_surface_physical_readback_artifact_review_queue import (
        build_read_surface_physical_readback_artifact_review_queue,
    )

    queue = build_read_surface_physical_readback_artifact_review_queue(
        intake_snapshots=_snapshot_map(**{spec["method_id"]: snapshot})
    )
    return spec, queue


def _candidate_for_spec(queue: dict, spec: dict) -> dict:
    return next(
        item
        for item in queue["queue_items"]
        if item["method_id"] == spec["method_id"]
    )


def _candidate_trace(queue: dict, spec: dict) -> dict:
    candidate = _candidate_for_spec(queue, spec)
    return {
        "source_queue_method_id": queue["method_id"],
        "source_queue_status": queue["status"],
        "candidate_status": candidate["status"],
        "intake_status": candidate["intake_status"],
        "artifact_source_ids": list(candidate["artifact_source_ids"]),
        "valid_candidate_count": candidate["valid_candidate_count"],
        "invalid_candidate_count": candidate["invalid_candidate_count"],
        "target_consistency": candidate["target_consistency"],
        "candidate_source_id_missing": candidate["candidate_source_id_missing"],
        "accepted_as_physical_fixture_candidate": candidate[
            "accepted_as_physical_fixture_candidate"
        ],
        "fixture_closure_allowed_by_queue": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }


def _acceptance_record(
    spec: dict,
    queue: dict,
    *,
    review_id: str,
    decision: str,
    **overrides,
) -> dict:
    record = {
        "review_id": review_id,
        "surface_id": spec["surface_id"],
        "artifact_source_id": "raw-mixer-physical.json",
        "required_contract_method_id": spec["required_contract_method_id"],
        "candidate_trace": _candidate_trace(queue, spec),
        "decision": decision,
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-05T19:45:00Z",
    }
    for key, value in overrides.items():
        if value is None:
            record.pop(key, None)
        else:
            record[key] = value
    return record


def test_acceptance_ledger_requires_candidates_before_acceptance():
    ledger = build_read_surface_physical_readback_fixture_acceptance_ledger()

    assert ledger["type"] == "read_surface_physical_readback_fixture_acceptance_ledger"
    assert ledger["method_id"] == READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_METHOD_ID
    assert ledger["status"] == "operator_acceptance_required"
    assert ledger["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "operator_required": True,
        "execution_allowed_by_acceptance": False,
        "fixture_closure_allowed_by_acceptance": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }
    assert ledger["summary"]["review_candidate_count"] == 0
    assert ledger["summary"]["acceptance_record_count"] == 0
    assert ledger["summary"]["candidate_trace_missing_count"] == 0
    assert ledger["summary"]["candidate_trace_mismatch_count"] == 0
    assert ledger["summary"]["physical_fixture_gap_remains"] is True
    assert ledger["summary"]["fixture_closure_allowed_by_acceptance"] is False
    assert ledger["summary"]["live_authority_granted"] is False
    assert ledger["summary"]["full_autonomous_ready"] is False
    assert "operator_must_capture_or_surface_valid_candidates_before_acceptance" in ledger[
        "summary"
    ]["blocked_by"]


def test_acceptance_ledger_tracks_unreviewed_queue_candidates():
    spec, queue = _queue_with_valid_candidate()
    ledger = build_read_surface_physical_readback_fixture_acceptance_ledger(review_queue=queue)

    assert ledger["status"] == "operator_acceptance_required"
    assert ledger["summary"]["review_candidate_count"] == 1
    assert ledger["summary"]["unreviewed_candidate_count"] == 1
    assert ledger["summary"]["unreviewed_candidate_surface_ids"] == [spec["surface_id"]]
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["candidate_trace_missing_count"] == 0
    assert ledger["summary"]["fixture_closure_allowed_by_acceptance"] is False


def test_acceptance_ledger_records_accepted_candidate_without_closing_fixture_gap():
    spec, queue = _queue_with_valid_candidate()
    ledger = build_read_surface_physical_readback_fixture_acceptance_ledger(
        review_queue=queue,
        acceptance_records=[
            _acceptance_record(
                spec,
                queue,
                review_id="review-raw-1",
                decision="accepted",
            )
        ],
    )

    assert ledger["status"] == "operator_acceptance_records_available"
    assert ledger["summary"]["accepted_record_count"] == 1
    assert ledger["summary"]["rejected_record_count"] == 0
    assert ledger["summary"]["invalid_record_count"] == 0
    assert ledger["summary"]["candidate_trace_missing_count"] == 0
    assert ledger["summary"]["candidate_trace_mismatch_count"] == 0
    assert ledger["summary"]["accepted_surface_ids"] == [spec["surface_id"]]
    assert ledger["summary"]["physical_fixture_gap_remains"] is True
    assert ledger["summary"]["fixture_closure_allowed_by_acceptance"] is False
    assert ledger["summary"]["live_authority_granted"] is False
    record = ledger["records"][0]
    assert record["operator_accepted_review_candidate"] is True
    assert record["candidate_trace"]["artifact_source_ids"] == ["raw-mixer-physical.json"]
    assert record["fixture_closure_allowed_by_acceptance"] is False
    assert record["full_autonomous_ready"] is False


def test_acceptance_ledger_records_rejected_candidate_without_authority():
    spec, queue = _queue_with_valid_candidate()
    ledger = build_read_surface_physical_readback_fixture_acceptance_ledger(
        review_queue=queue,
        acceptance_records=[
            _acceptance_record(
                spec,
                queue,
                review_id="review-raw-reject",
                decision="rejected",
            )
        ],
    )

    assert ledger["status"] == "operator_acceptance_records_available"
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["rejected_record_count"] == 1
    assert ledger["summary"]["rejected_surface_ids"] == [spec["surface_id"]]
    assert ledger["records"][0]["operator_rejected_review_candidate"] is True
    assert ledger["summary"]["fixture_closure_allowed_by_acceptance"] is False


def test_acceptance_ledger_rejects_unsafe_or_unmatched_records():
    spec, queue = _queue_with_valid_candidate()
    ledger = build_read_surface_physical_readback_fixture_acceptance_ledger(
        review_queue=queue,
        acceptance_records=[
            _acceptance_record(
                spec,
                queue,
                review_id="unsafe-review",
                decision="accepted",
                artifact_source_id="wrong.json",
                fixture_closure_allowed_by_acceptance=True,
            )
        ],
    )

    assert ledger["status"] == "operator_acceptance_replacement_required"
    assert ledger["summary"]["invalid_record_count"] == 1
    assert ledger["summary"]["invalid_record_ids"] == ["unsafe-review"]
    assert ledger["records"][0]["valid"] is False
    assert "acceptance_record_artifact_source_mismatch" in ledger["records"][0]["violations"]
    assert "acceptance_record_unsafe_authority_claim" in ledger["records"][0]["violations"]
    assert ledger["records"][0]["unsafe_authority_flags"] == [
        "fixture_closure_allowed_by_acceptance"
    ]


def test_acceptance_ledger_rejects_missing_artifact_source_id():
    spec, queue = _queue_with_valid_candidate()
    ledger = build_read_surface_physical_readback_fixture_acceptance_ledger(
        review_queue=queue,
        acceptance_records=[
            _acceptance_record(
                spec,
                queue,
                review_id="missing-artifact-source",
                decision="accepted",
                artifact_source_id=None,
            )
        ],
    )

    assert ledger["status"] == "operator_acceptance_replacement_required"
    assert ledger["summary"]["invalid_record_ids"] == ["missing-artifact-source"]
    assert ledger["records"][0]["valid"] is False
    assert "acceptance_record_artifact_source_missing" in ledger["records"][0][
        "violations"
    ]


def test_acceptance_ledger_rejects_non_utc_review_timestamp():
    spec, queue = _queue_with_valid_candidate()
    ledger = build_read_surface_physical_readback_fixture_acceptance_ledger(
        review_queue=queue,
        acceptance_records=[
            _acceptance_record(
                spec,
                queue,
                review_id="local-time-review",
                decision="accepted",
                reviewed_at="2026-06-05 22:45:00",
            )
        ],
    )

    assert ledger["status"] == "operator_acceptance_replacement_required"
    assert ledger["summary"]["invalid_record_ids"] == ["local-time-review"]
    assert ledger["records"][0]["valid"] is False
    assert "acceptance_record_reviewed_at_invalid" in ledger["records"][0][
        "violations"
    ]


def test_acceptance_ledger_rejects_missing_candidate_trace():
    spec, queue = _queue_with_valid_candidate()
    ledger = build_read_surface_physical_readback_fixture_acceptance_ledger(
        review_queue=queue,
        acceptance_records=[
            _acceptance_record(
                spec,
                queue,
                review_id="missing-trace",
                decision="accepted",
                candidate_trace=None,
            )
        ],
    )

    assert ledger["status"] == "operator_acceptance_replacement_required"
    assert ledger["summary"]["candidate_trace_missing_count"] == 1
    assert ledger["summary"]["candidate_trace_mismatch_count"] == 0
    assert "missing-trace" in ledger["summary"]["invalid_record_ids"]
    assert "acceptance_record_candidate_trace_missing" in ledger["records"][0][
        "violations"
    ]


def test_acceptance_ledger_rejects_candidate_trace_mismatch():
    spec, queue = _queue_with_valid_candidate()
    trace = _candidate_trace(queue, spec)
    trace["candidate_status"] = "wrong-status"
    ledger = build_read_surface_physical_readback_fixture_acceptance_ledger(
        review_queue=queue,
        acceptance_records=[
            _acceptance_record(
                spec,
                queue,
                review_id="bad-trace",
                decision="accepted",
                candidate_trace=trace,
            )
        ],
    )

    assert ledger["status"] == "operator_acceptance_replacement_required"
    assert ledger["summary"]["candidate_trace_missing_count"] == 0
    assert ledger["summary"]["candidate_trace_mismatch_count"] == 1
    assert "bad-trace" in ledger["summary"]["invalid_record_ids"]
    assert (
        "acceptance_record_candidate_trace_candidate_status_mismatch"
        in ledger["records"][0]["violations"]
    )


def test_acceptance_ledger_is_transport_free():
    source = Path(
        "backend/read_surface_physical_readback_fixture_acceptance_ledger.py"
    ).read_text(encoding="utf-8")

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


def test_acceptance_ledger_registry_and_goal_are_shadow_only():
    method = get_processing_method(
        READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_METHOD_ID
    )

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_METHOD_ID not in (
        main.executor_method_ids
    )

    snapshot = build_processing_registry_snapshot(include_methods=False)
    ledger = snapshot["read_surface_physical_readback_fixture_acceptance_ledger"]
    assert ledger["safety"]["fixture_closure_allowed_by_acceptance"] is False
    assert ledger["summary"]["physical_fixture_gap_remains"] is True

    row = get_project_goal_capability("console_read_parameters")
    assert row is not None
    assert READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_METHOD_ID in row.method_ids
    assert (
        "backend/read_surface_physical_readback_fixture_acceptance_ledger.py"
        in row.support_paths
    )
    assert row.current_live_write_kinds == ()
