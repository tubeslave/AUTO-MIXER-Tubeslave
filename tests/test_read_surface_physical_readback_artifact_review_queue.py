from pathlib import Path

from backend.read_surface_physical_readback_artifact_review_queue import (
    INTAKE_SPECS,
    READ_SURFACE_PHYSICAL_READBACK_ARTIFACT_REVIEW_QUEUE_METHOD_ID,
    build_read_surface_physical_readback_artifact_review_queue,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


def _snapshot_for_spec(spec: dict, *, valid: int = 0, invalid: int = 0, live_write: bool = False) -> dict:
    artifact_count = valid + invalid
    return {
        "type": spec["method_id"],
        "method_id": spec["method_id"],
        "status": (
            "valid_operator_artifact_available"
            if valid
            else "invalid_operator_artifact_present"
            if invalid
            else "operator_artifact_missing"
        ),
        "schema_id": "physical_readback.v1",
        "surface_id": spec["surface_id"],
        "remediation_action_id": spec["remediation_action_id"],
        "required_contract_method_id": spec["required_contract_method_id"],
        "safety": {
            "read_only": True,
            "live_write": live_write,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_intake": False,
            "fixture_closure_allowed_by_intake": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "artifact_count": artifact_count,
            "valid_candidate_count": valid,
            "invalid_candidate_count": invalid,
            "accepted_as_physical_fixture_candidate": valid > 0 and invalid == 0,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_intake": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "promotion_families_requiring_surface": ["fader", "gain"],
            "artifact_source_ids": [f"{spec['surface_id']}.json"] if artifact_count else [],
            "blocked_by": [
                "operator_must_review_valid_candidate_before_fixture_closure"
                if valid
                else "operator_must_capture_physical_readback_artifact"
            ],
            "violations": ["bad_artifact"] if invalid else [],
        },
        "artifacts": [],
    }


def _snapshot_map(**overrides: dict) -> dict:
    snapshots = {
        spec["method_id"]: _snapshot_for_spec(spec)
        for spec in INTAKE_SPECS
    }
    snapshots.update(overrides)
    return snapshots


def test_artifact_review_queue_reports_missing_artifacts_without_authority():
    queue = build_read_surface_physical_readback_artifact_review_queue()

    assert queue["type"] == "read_surface_physical_readback_artifact_review_queue"
    assert queue["method_id"] == READ_SURFACE_PHYSICAL_READBACK_ARTIFACT_REVIEW_QUEUE_METHOD_ID
    assert queue["status"] == "operator_artifact_capture_required"
    assert queue["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "operator_required": True,
        "execution_allowed_by_queue": False,
        "fixture_closure_allowed_by_queue": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }
    assert queue["summary"]["intake_surface_count"] == len(INTAKE_SPECS)
    assert queue["summary"]["available_intake_count"] == len(INTAKE_SPECS)
    assert queue["summary"]["artifact_count"] == 0
    assert queue["summary"]["valid_candidate_count"] == 0
    assert queue["summary"]["invalid_candidate_count"] == 0
    assert queue["summary"]["review_candidate_count"] == 0
    assert queue["summary"]["missing_artifact_surface_count"] == len(INTAKE_SPECS)
    assert queue["summary"]["operator_review_required"] is True
    assert queue["summary"]["physical_fixture_gap_remains"] is True
    assert queue["summary"]["fixture_closure_allowed_by_queue"] is False
    assert queue["summary"]["live_authority_granted"] is False
    assert queue["summary"]["full_autonomous_ready"] is False
    assert "operator_must_capture_missing_physical_readback_artifacts" in queue["summary"][
        "blocked_by"
    ]
    assert len(queue["queue_items"]) == len(INTAKE_SPECS)
    assert all(item["fixture_closure_allowed_by_queue"] is False for item in queue["queue_items"])
    assert all(item["live_authority_granted"] is False for item in queue["queue_items"])


def test_artifact_review_queue_surfaces_valid_and_invalid_operator_work():
    valid_spec = INTAKE_SPECS[0]
    invalid_spec = INTAKE_SPECS[1]
    queue = build_read_surface_physical_readback_artifact_review_queue(
        intake_snapshots=_snapshot_map(
            **{
                valid_spec["method_id"]: _snapshot_for_spec(valid_spec, valid=1),
                invalid_spec["method_id"]: _snapshot_for_spec(invalid_spec, invalid=1),
            }
        )
    )

    assert queue["status"] == "operator_artifact_replacement_required"
    assert queue["summary"]["artifact_count"] == 2
    assert queue["summary"]["valid_candidate_count"] == 1
    assert queue["summary"]["invalid_candidate_count"] == 1
    assert queue["summary"]["review_candidate_count"] == 1
    assert queue["summary"]["review_candidate_surface_ids"] == [valid_spec["surface_id"]]
    assert invalid_spec["surface_id"] in queue["summary"]["invalid_surface_ids"]
    assert queue["summary"]["fixture_closure_allowed_by_queue"] is False
    assert queue["summary"]["live_authority_granted"] is False
    valid_item = next(item for item in queue["queue_items"] if item["method_id"] == valid_spec["method_id"])
    invalid_item = next(
        item for item in queue["queue_items"] if item["method_id"] == invalid_spec["method_id"]
    )
    assert valid_item["accepted_as_physical_fixture_candidate"] is True
    assert invalid_item["accepted_as_physical_fixture_candidate"] is False


def test_artifact_review_queue_rejects_unsafe_authority_claims():
    spec = INTAKE_SPECS[0]
    queue = build_read_surface_physical_readback_artifact_review_queue(
        intake_snapshots=_snapshot_map(
            **{
                spec["method_id"]: _snapshot_for_spec(spec, valid=1, live_write=True),
            }
        )
    )

    assert queue["status"] == "operator_artifact_replacement_required"
    assert queue["summary"]["review_candidate_count"] == 0
    assert queue["summary"]["unsafe_authority_claim_count"] == 1
    assert queue["summary"]["unsafe_authority_surface_ids"] == [spec["surface_id"]]
    item = next(item for item in queue["queue_items"] if item["method_id"] == spec["method_id"])
    assert item["accepted_as_physical_fixture_candidate"] is False
    assert item["unsafe_authority_flags"] == ["live_write"]
    assert item["fixture_closure_allowed_by_queue"] is False
    assert item["full_autonomous_ready"] is False


def test_artifact_review_queue_is_transport_free():
    source = Path("backend/read_surface_physical_readback_artifact_review_queue.py").read_text(
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


def test_artifact_review_queue_is_registry_shadow_only_not_executor():
    method = get_processing_method(READ_SURFACE_PHYSICAL_READBACK_ARTIFACT_REVIEW_QUEUE_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert READ_SURFACE_PHYSICAL_READBACK_ARTIFACT_REVIEW_QUEUE_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    queue = snapshot["read_surface_physical_readback_artifact_review_queue"]
    assert queue["type"] == "read_surface_physical_readback_artifact_review_queue"
    assert queue["safety"]["live_write"] is False
    assert queue["safety"]["fixture_closure_allowed_by_queue"] is False
    assert queue["summary"]["physical_fixture_gap_remains"] is True


def test_artifact_review_queue_is_console_read_goal_support_path():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert READ_SURFACE_PHYSICAL_READBACK_ARTIFACT_REVIEW_QUEUE_METHOD_ID in row.method_ids
    assert "backend/read_surface_physical_readback_artifact_review_queue.py" in row.support_paths
    assert row.current_live_write_kinds == ()
