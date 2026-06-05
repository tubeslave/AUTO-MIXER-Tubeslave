import copy
import json
from pathlib import Path

from backend.bus_group_main_dca_physical_readback_artifact_intake import (
    BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    DEFAULT_BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_ROOTS,
    build_bus_group_main_dca_physical_readback_artifact_intake,
)
from backend.bus_group_main_dca_physical_readback_contract import (
    BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_TYPE,
    BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SCHEMA_ID,
    BUS_GROUP_MAIN_DCA_REMEDIATION_ACTION_ID,
    BUS_GROUP_MAIN_DCA_SURFACE_ID,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


def _valid_artifact() -> dict:
    return {
        "type": BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_TYPE,
        "schema_id": BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SCHEMA_ID,
        "surface_id": BUS_GROUP_MAIN_DCA_SURFACE_ID,
        "remediation_action_id": BUS_GROUP_MAIN_DCA_REMEDIATION_ACTION_ID,
        "captured_at": "2026-06-05T00:00:00Z",
        "source": "physical_readback",
        "console_model": "wing",
        "bus_count": 1,
        "main_count": 1,
        "dca_count": 1,
        "buses": [
            {
                "bus_id": 1,
                "bus_label": "Vocal Bus",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "controls": {
                    "fader_db": -5.0,
                    "mute": False,
                    "pan": 0.0,
                    "width": 100.0,
                    "mono": 0,
                },
                "assignments": {
                    "dca_assignments": [1],
                    "main_send_on": True,
                    "main_send_level_db": 0.0,
                    "main_send_pre": False,
                },
                "metering": {
                    "lufs": -25.0,
                    "true_peak_dbtp": -6.0,
                    "headroom_db": 5.0,
                },
                "coverage": {
                    "controls": True,
                    "assignments": True,
                    "metering": True,
                    "cache_matches_physical_readback": True,
                },
            }
        ],
        "mains": [
            {
                "main_id": 1,
                "main_label": "Main LR",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "controls": {
                    "fader_db": -3.0,
                    "mute": False,
                    "pan": 0.0,
                },
                "metering": {
                    "lufs": -22.0,
                    "true_peak_dbtp": -4.0,
                    "headroom_db": 3.0,
                },
                "coverage": {
                    "controls": True,
                    "metering": True,
                    "cache_matches_physical_readback": True,
                },
            }
        ],
        "dcas": [
            {
                "dca_id": 1,
                "dca_label": "Vocals",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "controls": {
                    "fader_db": -2.0,
                    "mute": False,
                    "monitor": 0,
                },
                "assignments": {
                    "member_channels": [1, 2],
                    "member_buses": [1],
                },
                "coverage": {
                    "identity": True,
                    "controls": True,
                    "assignments": True,
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


def test_bus_group_main_dca_artifact_intake_reports_missing_without_authority():
    intake = build_bus_group_main_dca_physical_readback_artifact_intake(
        artifact_roots=("does-not-exist/bus-group-main-dca",)
    )

    assert intake["type"] == "bus_group_main_dca_physical_readback_artifact_intake"
    assert intake["method_id"] == BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID
    assert intake["status"] == "operator_artifact_missing"
    assert intake["surface_id"] == BUS_GROUP_MAIN_DCA_SURFACE_ID
    assert intake["remediation_action_id"] == BUS_GROUP_MAIN_DCA_REMEDIATION_ACTION_ID
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
        "does-not-exist/bus-group-main-dca"
    ]
    assert intake["summary"]["artifact_count"] == 0
    assert intake["summary"]["valid_candidate_count"] == 0
    assert intake["summary"]["invalid_candidate_count"] == 0
    assert intake["summary"]["valid_bus_count"] == 0
    assert intake["summary"]["valid_main_count"] == 0
    assert intake["summary"]["valid_dca_count"] == 0
    assert intake["summary"]["accepted_as_physical_fixture_candidate"] is False
    assert intake["summary"]["physical_fixture_gap_remains"] is True
    assert intake["summary"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["live_authority_granted"] is False
    assert intake["summary"]["full_autonomous_ready"] is False
    assert intake["summary"]["promotion_families_requiring_surface"] == [
        "bus_group_matrix",
        "dca",
        "master_bus",
        "mix_balance",
        "performance_adaptation",
    ]
    assert intake["summary"]["blocked_by"] == [
        "operator_must_capture_bus_group_main_dca_physical_readback_artifact"
    ]
    assert intake["artifacts"] == []


def test_bus_group_main_dca_artifact_intake_accepts_valid_candidate_but_keeps_review_gate():
    intake = build_bus_group_main_dca_physical_readback_artifact_intake(
        artifact_payloads=(_valid_artifact(),),
        artifact_roots=(),
    )

    assert intake["status"] == "valid_operator_artifact_available"
    assert intake["summary"]["artifact_count"] == 1
    assert intake["summary"]["valid_candidate_count"] == 1
    assert intake["summary"]["invalid_candidate_count"] == 0
    assert intake["summary"]["valid_bus_count"] == 1
    assert intake["summary"]["valid_main_count"] == 1
    assert intake["summary"]["valid_dca_count"] == 1
    assert intake["summary"]["accepted_as_physical_fixture_candidate"] is True
    assert intake["summary"]["operator_review_required"] is True
    assert intake["summary"]["physical_fixture_gap_remains"] is True
    assert intake["summary"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["live_authority_granted"] is False
    assert intake["summary"]["latest_valid_candidate_source_id"] == "inline_payload_0"
    assert intake["summary"]["blocked_by"] == [
        "operator_must_review_valid_bus_group_main_dca_candidate_before_fixture_closure"
    ]
    assert intake["artifacts"][0]["status"] == "valid_physical_readback_artifact"
    assert intake["artifacts"][0]["accepted_as_physical_fixture_candidate"] is True
    assert intake["artifacts"][0]["bus_count"] == 1
    assert intake["artifacts"][0]["main_count"] == 1
    assert intake["artifacts"][0]["dca_count"] == 1
    assert intake["artifacts"][0]["violations"] == []


def test_bus_group_main_dca_artifact_intake_rejects_invalid_or_live_claiming_candidate():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["source"] = "mixer_state_cache"
    artifact["safety"]["live_write"] = True
    artifact["dca_write_events"] = []
    artifact["buses"][0]["status"] = "verified_from_cache"
    artifact["buses"][0]["source"] = "mixer_state_cache"
    artifact["buses"][0]["controls"]["mute"] = "false"
    artifact["buses"][0]["coverage"]["cache_matches_physical_readback"] = False

    intake = build_bus_group_main_dca_physical_readback_artifact_intake(
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
    assert "mutation_evidence_field_present:dca_write_events" in intake["summary"]["violations"]
    assert "bus_0_status_not_physical_verified" in intake["summary"]["violations"]
    assert "bus_0_source_not_physical_readback" in intake["summary"]["violations"]
    assert "bus_0_controls_mute_not_bool_or_0_1" in intake["summary"]["violations"]
    assert (
        "bus_0_coverage_cache_matches_physical_readback_not_true"
        in intake["summary"]["violations"]
    )
    assert intake["artifacts"][0]["accepted_as_physical_fixture_candidate"] is False


def test_bus_group_main_dca_artifact_intake_discovers_json_files(tmp_path):
    artifact_dir = tmp_path / "bus-group-main-dca"
    artifact_dir.mkdir()
    artifact_path = artifact_dir / "candidate.json"
    artifact_path.write_text(json.dumps(_valid_artifact()), encoding="utf-8")
    (artifact_dir / "broken.json").write_text("{", encoding="utf-8")

    intake = build_bus_group_main_dca_physical_readback_artifact_intake(
        artifact_roots=(artifact_dir,)
    )

    assert intake["status"] == "valid_operator_artifact_available"
    assert intake["summary"]["artifact_search_roots"] == [str(artifact_dir)]
    assert intake["summary"]["artifact_count"] == 2
    assert intake["summary"]["valid_candidate_count"] == 1
    assert intake["summary"]["invalid_candidate_count"] == 1
    assert any(row["source_id"] == str(artifact_path) for row in intake["artifacts"])
    assert any(
        "artifact_load_error:JSONDecodeError" in row["violations"]
        for row in intake["artifacts"]
    )


def test_bus_group_main_dca_artifact_intake_uses_repo_default_roots():
    intake = build_bus_group_main_dca_physical_readback_artifact_intake()

    assert intake["summary"]["artifact_search_roots"] == list(
        DEFAULT_BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_ROOTS
    )
    assert intake["safety"]["queries_console"] is False
    assert intake["safety"]["executes_tests"] is False


def test_bus_group_main_dca_artifact_intake_is_transport_free():
    source = Path("backend/bus_group_main_dca_physical_readback_artifact_intake.py").read_text(
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
        "set_bus_fader",
        "set_group_fader",
        "set_dca_fader",
        "set_main_fader",
        "set_mute",
        "set_bus_mute",
        "set_dca_mute",
        "set_group_mute",
        "set_main_mute",
        "apply_operator_proposal",
    ]
    for token in forbidden:
        assert token not in source


def test_bus_group_main_dca_artifact_intake_is_registry_shadow_only_not_executor():
    method = get_processing_method(BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert (
        BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID
        not in main.executor_method_ids
    )

    snapshot = build_processing_registry_snapshot(include_methods=False)
    intake = snapshot["bus_group_main_dca_physical_readback_artifact_intake"]
    assert intake["type"] == "bus_group_main_dca_physical_readback_artifact_intake"
    assert intake["safety"]["live_write"] is False
    assert intake["safety"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["physical_fixture_gap_remains"] is True


def test_bus_group_main_dca_artifact_intake_is_console_read_goal_support_path():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID in row.method_ids
    assert "backend/bus_group_main_dca_physical_readback_artifact_intake.py" in row.support_paths
    assert row.current_live_write_kinds == ()
