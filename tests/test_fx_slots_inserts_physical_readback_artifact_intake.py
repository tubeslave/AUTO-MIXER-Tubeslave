import copy
import json
from pathlib import Path

from backend.fx_slots_inserts_physical_readback_artifact_intake import (
    DEFAULT_FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_ROOTS,
    FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_fx_slots_inserts_physical_readback_artifact_intake,
)
from backend.fx_slots_inserts_physical_readback_contract import (
    FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_TYPE,
    FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID,
    FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID,
    FX_SLOTS_INSERTS_SURFACE_ID,
    MINIMUM_FX_PARAMETER_COUNT,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


def _parameters() -> list[dict]:
    return [
        {
            "parameter_index": index,
            "value": float(index) / 10.0,
            "schema_status": "mapped",
            "schema_family": "reverb",
            "schema_source": "physical_model_schema",
            "semantic_name": f"parameter_{index}",
            "unit": "raw",
            "safe_range": {"min": 0.0, "max": 1.0},
        }
        for index in range(1, MINIMUM_FX_PARAMETER_COUNT + 1)
    ]


def _valid_artifact() -> dict:
    parameters = _parameters()
    return {
        "type": FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_TYPE,
        "schema_id": FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID,
        "surface_id": FX_SLOTS_INSERTS_SURFACE_ID,
        "remediation_action_id": FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID,
        "captured_at": "2026-06-05T00:00:00Z",
        "source": "physical_readback",
        "console_model": "wing",
        "fx_slot_count": 1,
        "insert_count": 1,
        "fx_slots": [
            {
                "slot": "FX1",
                "slot_number": 1,
                "readback_status": "verified_from_physical_readback",
                "readback_source": "physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "model": "Hall Reverb",
                "on": True,
                "mix": 25.0,
                "source": "Bus 1",
                "mode": "insert",
                "assigned_channel": 1,
                "assigned_position": "post",
                "active": True,
                "parameters": parameters,
                "parameter_schema": {
                    "status": "mapped",
                    "schema_family": "reverb",
                    "schema_source": "physical_model_schema",
                    "mapped_parameter_count": len(parameters),
                    "generic_parameter_count": 0,
                    "cached_parameter_count": len(parameters),
                    "physical_model_schema_verified": True,
                    "live_write_authority": False,
                },
                "coverage": {
                    "identity": True,
                    "assignment": True,
                    "parameters": True,
                    "parameter_schema": True,
                    "cache_matches_physical_readback": True,
                },
            }
        ],
        "inserts": [
            {
                "target_type": "channel",
                "target": 1,
                "position": "post",
                "readback_status": "verified_from_physical_readback",
                "readback_source": "physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "on": True,
                "slot": "FX1",
                "status": "active",
                "mode": "post_fader",
                "weight": 100.0,
                "linked_fx_slot": "FX1",
                "active": True,
                "insert_assignment_policy": "blocked_manual_only",
                "destructive_change_marker": "insert_assignment_would_repatch_processing_chain",
                "coverage": {
                    "assignment": True,
                    "status": True,
                    "scoped_diff_marker": True,
                    "cache_matches_physical_readback": True,
                },
            }
        ],
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
        },
    }


def test_fx_slots_inserts_artifact_intake_reports_missing_without_authority():
    intake = build_fx_slots_inserts_physical_readback_artifact_intake(
        artifact_roots=("does-not-exist/fx-slots-inserts",)
    )

    assert intake["type"] == "fx_slots_inserts_physical_readback_artifact_intake"
    assert intake["method_id"] == FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID
    assert intake["status"] == "operator_artifact_missing"
    assert intake["surface_id"] == FX_SLOTS_INSERTS_SURFACE_ID
    assert intake["remediation_action_id"] == FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID
    assert intake["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "operator_required": True,
        "execution_allowed_by_intake": False,
        "fixture_closure_allowed_by_intake": False,
        "live_widening_allowed": False,
        "full_autonomous_ready": False,
    }
    assert intake["summary"]["artifact_search_roots"] == [
        "does-not-exist/fx-slots-inserts"
    ]
    assert intake["summary"]["artifact_count"] == 0
    assert intake["summary"]["valid_candidate_count"] == 0
    assert intake["summary"]["invalid_candidate_count"] == 0
    assert intake["summary"]["valid_fx_slot_count"] == 0
    assert intake["summary"]["valid_insert_count"] == 0
    assert intake["summary"]["accepted_as_physical_fixture_candidate"] is False
    assert intake["summary"]["operator_review_required"] is True
    assert intake["summary"]["physical_fixture_gap_remains"] is True
    assert intake["summary"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["live_authority_granted"] is False
    assert intake["summary"]["full_autonomous_ready"] is False
    assert intake["summary"]["promotion_families_requiring_surface"] == [
        "fx_send_insert",
        "routing",
        "mix_balance",
        "performance_adaptation",
    ]
    assert intake["summary"]["blocked_by"] == [
        "operator_must_capture_fx_slots_inserts_physical_readback_artifact"
    ]
    assert intake["artifacts"] == []


def test_fx_slots_inserts_artifact_intake_accepts_valid_candidate_but_keeps_review_gate():
    intake = build_fx_slots_inserts_physical_readback_artifact_intake(
        artifact_payloads=(_valid_artifact(),),
        artifact_roots=(),
    )

    assert intake["status"] == "valid_operator_artifact_available"
    assert intake["summary"]["artifact_count"] == 1
    assert intake["summary"]["valid_candidate_count"] == 1
    assert intake["summary"]["invalid_candidate_count"] == 0
    assert intake["summary"]["valid_fx_slot_count"] == 1
    assert intake["summary"]["valid_insert_count"] == 1
    assert intake["summary"]["accepted_as_physical_fixture_candidate"] is True
    assert intake["summary"]["operator_review_required"] is True
    assert intake["summary"]["physical_fixture_gap_remains"] is True
    assert intake["summary"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["live_authority_granted"] is False
    assert intake["summary"]["latest_valid_candidate_source_id"] == "inline_payload_0"
    assert intake["summary"]["blocked_by"] == [
        "operator_must_review_valid_fx_slots_inserts_candidate_before_fixture_closure"
    ]
    assert intake["artifacts"][0]["status"] == "valid_physical_readback_artifact"
    assert intake["artifacts"][0]["accepted_as_physical_fixture_candidate"] is True
    assert intake["artifacts"][0]["fx_slot_count"] == 1
    assert intake["artifacts"][0]["insert_count"] == 1
    assert intake["artifacts"][0]["violations"] == []


def test_fx_slots_inserts_artifact_intake_rejects_invalid_or_live_claiming_candidate():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["source"] = "mixer_state_cache"
    artifact["safety"]["live_write"] = True
    artifact["fx_write_events"] = []
    artifact["fx_slots"][0]["readback_status"] = "verified_from_cache"
    artifact["fx_slots"][0]["readback_source"] = "mixer_state_cache"
    artifact["fx_slots"][0]["coverage"]["cache_matches_physical_readback"] = False
    artifact["inserts"][0]["readback_status"] = "verified_from_cache"
    artifact["inserts"][0]["readback_source"] = "mixer_state_cache"
    artifact["inserts"][0]["on"] = "true"

    intake = build_fx_slots_inserts_physical_readback_artifact_intake(
        artifact_payloads=(artifact,),
        artifact_roots=(),
    )

    assert intake["status"] == "invalid_operator_artifact_present"
    assert intake["summary"]["artifact_count"] == 1
    assert intake["summary"]["valid_candidate_count"] == 0
    assert intake["summary"]["invalid_candidate_count"] == 1
    assert intake["summary"]["accepted_as_physical_fixture_candidate"] is False
    assert intake["summary"]["physical_fixture_gap_remains"] is True
    assert intake["summary"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["live_authority_granted"] is False
    assert "source_not_physical_readback" in intake["summary"]["violations"]
    assert "safety_live_write_not_false" in intake["summary"]["violations"]
    assert "mutation_evidence_field_present:fx_write_events" in intake["summary"]["violations"]
    assert "fx_slot_0_readback_status_not_physical_verified" in intake["summary"][
        "violations"
    ]
    assert "fx_slot_0_readback_source_not_physical_readback" in intake["summary"][
        "violations"
    ]
    assert (
        "fx_slot_0_coverage_cache_matches_physical_readback_not_true"
        in intake["summary"]["violations"]
    )
    assert "insert_0_readback_status_not_physical_verified" in intake["summary"][
        "violations"
    ]
    assert "insert_0_readback_source_not_physical_readback" in intake["summary"][
        "violations"
    ]
    assert "insert_0_on_not_bool_or_0_1" in intake["summary"]["violations"]
    assert intake["artifacts"][0]["accepted_as_physical_fixture_candidate"] is False


def test_fx_slots_inserts_artifact_intake_discovers_json_files(tmp_path):
    artifact_dir = tmp_path / "fx-slots-inserts"
    artifact_dir.mkdir()
    artifact_path = artifact_dir / "candidate.json"
    artifact_path.write_text(json.dumps(_valid_artifact()), encoding="utf-8")
    (artifact_dir / "broken.json").write_text("{", encoding="utf-8")

    intake = build_fx_slots_inserts_physical_readback_artifact_intake(
        artifact_roots=(artifact_dir,)
    )

    assert intake["status"] == "invalid_operator_artifact_present"
    assert intake["summary"]["artifact_search_roots"] == [str(artifact_dir)]
    assert intake["summary"]["artifact_count"] == 2
    assert intake["summary"]["valid_candidate_count"] == 1
    assert intake["summary"]["invalid_candidate_count"] == 1
    assert intake["summary"]["accepted_as_physical_fixture_candidate"] is False
    assert intake["summary"]["blocked_by"] == [
        "operator_must_replace_invalid_fx_slots_inserts_physical_readback_artifact"
    ]
    assert any(row["source_id"] == str(artifact_path) for row in intake["artifacts"])
    assert any(
        "artifact_load_error:JSONDecodeError" in row["violations"]
        for row in intake["artifacts"]
    )


def test_fx_slots_inserts_artifact_intake_uses_repo_default_roots():
    intake = build_fx_slots_inserts_physical_readback_artifact_intake()

    assert intake["summary"]["artifact_search_roots"] == list(
        DEFAULT_FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_ROOTS
    )
    assert intake["safety"]["queries_console"] is False
    assert intake["safety"]["executes_tests"] is False


def test_fx_slots_inserts_artifact_intake_is_transport_free():
    source = Path("backend/fx_slots_inserts_physical_readback_artifact_intake.py").read_text(
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
        "set_fx_slot",
        "set_fx_parameter",
        "set_insert",
        "set_bus_send",
        "set_matrix_send",
        "apply_operator_proposal",
    ]
    for token in forbidden:
        assert token not in source


def test_fx_slots_inserts_artifact_intake_is_registry_shadow_only_not_executor():
    method = get_processing_method(FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    intake = snapshot["fx_slots_inserts_physical_readback_artifact_intake"]
    assert intake["type"] == "fx_slots_inserts_physical_readback_artifact_intake"
    assert intake["safety"]["live_write"] is False
    assert intake["safety"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["physical_fixture_gap_remains"] is True


def test_fx_slots_inserts_artifact_intake_is_console_read_goal_support_path():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID in row.method_ids
    assert "backend/fx_slots_inserts_physical_readback_artifact_intake.py" in row.support_paths
    assert row.current_live_write_kinds == ()
