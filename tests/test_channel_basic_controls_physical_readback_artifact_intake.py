import copy
import json
from pathlib import Path

from backend.channel_basic_controls_physical_readback_artifact_intake import (
    CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    DEFAULT_CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_ROOTS,
    build_channel_basic_controls_physical_readback_artifact_intake,
)
from backend.channel_basic_controls_physical_readback_contract import (
    CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_TYPE,
    CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SCHEMA_ID,
    CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID,
    CHANNEL_BASIC_CONTROLS_SURFACE_ID,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


def _valid_artifact() -> dict:
    return {
        "type": CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_TYPE,
        "schema_id": CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SCHEMA_ID,
        "surface_id": CHANNEL_BASIC_CONTROLS_SURFACE_ID,
        "remediation_action_id": CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID,
        "captured_at": "2026-06-05T00:00:00Z",
        "source": "physical_readback",
        "console_model": "wing",
        "channel_count": 1,
        "channels": [
            {
                "channel_id": 1,
                "channel_label": "Lead Vox",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "identity": {
                    "source_name": "Lead Vox",
                    "source_role": "lead_vocal",
                    "input_source": "local_in_1",
                },
                "controls": {
                    "fader_db": -8.5,
                    "trim_db": 1.0,
                    "mute": False,
                    "pan": 0.0,
                },
                "coverage": {
                    "identity": True,
                    "basic_controls": True,
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


def test_channel_basic_controls_artifact_intake_reports_missing_without_authority():
    intake = build_channel_basic_controls_physical_readback_artifact_intake(
        artifact_roots=("does-not-exist/channel-basic",)
    )

    assert intake["type"] == "channel_basic_controls_physical_readback_artifact_intake"
    assert intake["method_id"] == CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID
    assert intake["status"] == "operator_artifact_missing"
    assert intake["surface_id"] == CHANNEL_BASIC_CONTROLS_SURFACE_ID
    assert intake["remediation_action_id"] == CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID
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
    assert intake["summary"]["artifact_search_roots"] == ["does-not-exist/channel-basic"]
    assert intake["summary"]["artifact_count"] == 0
    assert intake["summary"]["valid_candidate_count"] == 0
    assert intake["summary"]["invalid_candidate_count"] == 0
    assert intake["summary"]["valid_channel_count"] == 0
    assert intake["summary"]["accepted_as_physical_fixture_candidate"] is False
    assert intake["summary"]["physical_fixture_gap_remains"] is True
    assert intake["summary"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["live_authority_granted"] is False
    assert intake["summary"]["full_autonomous_ready"] is False
    assert intake["summary"]["promotion_families_requiring_surface"] == [
        "fader",
        "gain",
        "mute",
        "pan_phase_delay",
    ]
    assert intake["summary"]["blocked_by"] == [
        "operator_must_capture_channel_basic_controls_physical_readback_artifact"
    ]
    assert intake["artifacts"] == []


def test_channel_basic_controls_artifact_intake_accepts_valid_candidate_but_keeps_review_gate():
    intake = build_channel_basic_controls_physical_readback_artifact_intake(
        artifact_payloads=(_valid_artifact(),),
        artifact_roots=(),
    )

    assert intake["status"] == "valid_operator_artifact_available"
    assert intake["summary"]["artifact_count"] == 1
    assert intake["summary"]["valid_candidate_count"] == 1
    assert intake["summary"]["invalid_candidate_count"] == 0
    assert intake["summary"]["valid_channel_count"] == 1
    assert intake["summary"]["accepted_as_physical_fixture_candidate"] is True
    assert intake["summary"]["operator_review_required"] is True
    assert intake["summary"]["physical_fixture_gap_remains"] is True
    assert intake["summary"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["live_authority_granted"] is False
    assert intake["summary"]["latest_valid_candidate_source_id"] == "inline_payload_0"
    assert intake["summary"]["blocked_by"] == [
        "operator_must_review_valid_channel_basic_controls_candidate_before_fixture_closure"
    ]
    assert intake["artifacts"][0]["status"] == "valid_physical_readback_artifact"
    assert intake["artifacts"][0]["accepted_as_physical_fixture_candidate"] is True
    assert intake["artifacts"][0]["channel_count"] == 1
    assert intake["artifacts"][0]["violations"] == []


def test_channel_basic_controls_artifact_intake_rejects_invalid_or_live_claiming_candidate():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["source"] = "mixer_state_cache"
    artifact["safety"]["live_write"] = True
    artifact["mute_write_events"] = []
    artifact["channels"][0]["status"] = "verified_from_cache"
    artifact["channels"][0]["source"] = "mixer_state_cache"
    artifact["channels"][0]["controls"]["mute"] = "false"

    intake = build_channel_basic_controls_physical_readback_artifact_intake(
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
    assert "channel_0_status_not_physical_verified" in intake["summary"]["violations"]
    assert "channel_0_source_not_physical_readback" in intake["summary"]["violations"]
    assert "channel_0_controls_mute_not_bool" in intake["summary"]["violations"]
    assert "mutation_evidence_field_present:mute_write_events" in intake["summary"]["violations"]
    assert intake["artifacts"][0]["accepted_as_physical_fixture_candidate"] is False


def test_channel_basic_controls_artifact_intake_discovers_json_files(tmp_path):
    artifact_dir = tmp_path / "channel-basic"
    artifact_dir.mkdir()
    artifact_path = artifact_dir / "candidate.json"
    artifact_path.write_text(json.dumps(_valid_artifact()), encoding="utf-8")
    (artifact_dir / "broken.json").write_text("{", encoding="utf-8")

    intake = build_channel_basic_controls_physical_readback_artifact_intake(
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


def test_channel_basic_controls_artifact_intake_uses_repo_default_roots():
    intake = build_channel_basic_controls_physical_readback_artifact_intake()

    assert intake["summary"]["artifact_search_roots"] == list(
        DEFAULT_CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_ROOTS
    )
    assert intake["safety"]["queries_console"] is False
    assert intake["safety"]["executes_tests"] is False


def test_channel_basic_controls_artifact_intake_is_transport_free():
    source = Path("backend/channel_basic_controls_physical_readback_artifact_intake.py").read_text(
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
        "set_mute",
        "set_pan",
        "apply_operator_proposal",
    ]
    for token in forbidden:
        assert token not in source


def test_channel_basic_controls_artifact_intake_is_registry_shadow_only_not_executor():
    method = get_processing_method(
        CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID
    )

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    intake = snapshot["channel_basic_controls_physical_readback_artifact_intake"]
    assert intake["type"] == "channel_basic_controls_physical_readback_artifact_intake"
    assert intake["safety"]["live_write"] is False
    assert intake["safety"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["physical_fixture_gap_remains"] is True


def test_channel_basic_controls_artifact_intake_is_console_read_goal_support_path():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID in row.method_ids
    assert "backend/channel_basic_controls_physical_readback_artifact_intake.py" in row.support_paths
    assert row.current_live_write_kinds == ()
