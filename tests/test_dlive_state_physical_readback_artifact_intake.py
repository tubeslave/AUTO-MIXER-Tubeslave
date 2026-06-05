import copy
import json
from pathlib import Path

from backend.dlive_state_physical_readback_artifact_intake import (
    DEFAULT_DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_ROOTS,
    DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_dlive_state_physical_readback_artifact_intake,
)
from backend.dlive_state_physical_readback_contract import (
    DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE,
    DLIVE_STATE_PHYSICAL_READBACK_SCHEMA_ID,
    DLIVE_STATE_REMEDIATION_ACTION_ID,
    DLIVE_STATE_SURFACE_ID,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


def _coverage() -> dict:
    return {
        "channel_identity": True,
        "fader_scale": True,
        "mute_state": True,
        "raw_midi_state": True,
        "midi_base_channel": True,
        "wing_parity_mapping": True,
        "cache_matches_physical_readback": True,
    }


def _valid_artifact() -> dict:
    coverage = _coverage()
    return {
        "type": DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE,
        "schema_id": DLIVE_STATE_PHYSICAL_READBACK_SCHEMA_ID,
        "surface_id": DLIVE_STATE_SURFACE_ID,
        "remediation_action_id": DLIVE_STATE_REMEDIATION_ACTION_ID,
        "captured_at": "2026-06-05T00:00:00Z",
        "source": "dlive_physical_readback",
        "console_model": "Allen & Heath dLive",
        "channel_count": 1,
        "channels": [
            {
                "channel": 1,
                "cached_channel_name": "Lead Vox",
                "fader_raw_14bit": 10922,
                "fader_db": 0.0,
                "mute": False,
                "raw_note_velocity": 0,
                "midi_base_channel": 1,
                "readback_status": "verified_from_physical_readback",
                "readback_source": "dlive_physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "coverage": dict(coverage),
            }
        ],
        "midi_state": {
            "midi_base_channel": 1,
            "raw_state_key_count": 4,
            "raw_note_event_count": 1,
            "raw_cc_event_count": 1,
            "raw_pitchbend_event_count": 1,
            "tcp_session_status": "verified_passive_state",
            "midi_transport_status": "verified_passive_state",
            "readback_status": "verified_from_physical_readback",
            "readback_source": "dlive_physical_readback",
            "captured_at": "2026-06-05T00:00:00Z",
            "coverage": dict(coverage),
        },
        "parity": {
            "wing_surface_equivalent": "channel_identity_and_basic_controls",
            "mapping_status": "partial_mapping_verified",
            "channel_identity_parity": "verified",
            "fader_scale_parity": "verified",
            "mute_state_parity": "verified",
            "midi_velocity_parity": "verified",
            "unsupported_field_policy": "block_autonomous_promotion",
            "coverage": dict(coverage),
        },
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "midi_send_allowed": False,
            "tcp_send_allowed": False,
            "dlive_write_allowed": False,
            "dlive_apply_allowed": False,
        },
    }


def test_dlive_state_artifact_intake_reports_missing_without_authority():
    intake = build_dlive_state_physical_readback_artifact_intake(
        artifact_roots=("does-not-exist/dlive-state",)
    )

    assert intake["type"] == "dlive_state_physical_readback_artifact_intake"
    assert intake["method_id"] == DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID
    assert intake["status"] == "operator_artifact_missing"
    assert intake["surface_id"] == DLIVE_STATE_SURFACE_ID
    assert intake["remediation_action_id"] == DLIVE_STATE_REMEDIATION_ACTION_ID
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
        "midi_send_allowed": False,
        "tcp_send_allowed": False,
        "dlive_write_allowed": False,
        "dlive_apply_allowed": False,
    }
    assert intake["summary"]["artifact_search_roots"] == [
        "does-not-exist/dlive-state"
    ]
    assert intake["summary"]["artifact_count"] == 0
    assert intake["summary"]["valid_candidate_count"] == 0
    assert intake["summary"]["invalid_candidate_count"] == 0
    assert intake["summary"]["valid_channel_count"] == 0
    assert intake["summary"]["accepted_as_physical_fixture_candidate"] is False
    assert intake["summary"]["operator_review_required"] is True
    assert intake["summary"]["physical_fixture_gap_remains"] is True
    assert intake["summary"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["midi_send_allowed"] is False
    assert intake["summary"]["tcp_send_allowed"] is False
    assert intake["summary"]["dlive_write_allowed"] is False
    assert intake["summary"]["dlive_apply_allowed"] is False
    assert intake["summary"]["live_authority_granted"] is False
    assert intake["summary"]["full_autonomous_ready"] is False
    assert intake["summary"]["promotion_families_requiring_surface"] == [
        "fader",
        "gain",
        "mute",
        "bus_group_matrix",
        "routing",
        "scene_preset",
        "mix_balance",
        "performance_adaptation",
    ]
    assert intake["summary"]["blocked_by"] == [
        "operator_must_capture_dlive_state_physical_readback_artifact"
    ]
    assert intake["artifacts"] == []


def test_dlive_state_artifact_intake_accepts_valid_candidate_but_keeps_gate():
    intake = build_dlive_state_physical_readback_artifact_intake(
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
    assert intake["summary"]["midi_send_allowed"] is False
    assert intake["summary"]["tcp_send_allowed"] is False
    assert intake["summary"]["dlive_write_allowed"] is False
    assert intake["summary"]["dlive_apply_allowed"] is False
    assert intake["summary"]["live_authority_granted"] is False
    assert intake["summary"]["latest_valid_candidate_source_id"] == "inline_payload_0"
    assert intake["summary"]["blocked_by"] == [
        "operator_must_review_valid_dlive_state_candidate_before_fixture_closure"
    ]
    assert intake["artifacts"][0]["status"] == "valid_physical_readback_artifact"
    assert intake["artifacts"][0]["accepted_as_physical_fixture_candidate"] is True
    assert intake["artifacts"][0]["channel_count"] == 1
    assert intake["artifacts"][0]["violations"] == []


def test_dlive_state_artifact_intake_rejects_invalid_or_live_claiming_candidate():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["source"] = "mixer_state_cache"
    artifact["live_write"] = True
    artifact["safety"]["midi_send_allowed"] = True
    artifact["safety"]["tcp_send_allowed"] = True
    artifact["safety"]["dlive_write_allowed"] = True
    artifact["safety"]["dlive_apply_allowed"] = True
    artifact["channels"][0]["readback_status"] = "verified_from_cache"
    artifact["channels"][0]["readback_source"] = "mixer_state_cache"
    artifact["channels"][0]["coverage"]["cache_matches_physical_readback"] = False
    artifact["channels"][0]["midi_send_events"] = [{"channel": 1}]

    intake = build_dlive_state_physical_readback_artifact_intake(
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
    assert intake["summary"]["midi_send_allowed"] is False
    assert intake["summary"]["tcp_send_allowed"] is False
    assert intake["summary"]["dlive_write_allowed"] is False
    assert intake["summary"]["dlive_apply_allowed"] is False
    assert intake["summary"]["live_authority_granted"] is False
    assert "source_not_physical_readback" in intake["summary"]["violations"]
    assert "safety_midi_send_allowed_not_false" in intake["summary"]["violations"]
    assert "safety_tcp_send_allowed_not_false" in intake["summary"]["violations"]
    assert "safety_dlive_write_allowed_not_false" in intake["summary"]["violations"]
    assert "safety_dlive_apply_allowed_not_false" in intake["summary"]["violations"]
    assert "top_level_live_write_not_false" in intake["summary"]["violations"]
    assert "channel_0_readback_status_not_physical_verified" in intake["summary"][
        "violations"
    ]
    assert "channel_0_readback_source_not_physical_readback" in intake["summary"][
        "violations"
    ]
    assert (
        "channel_0_coverage_cache_matches_physical_readback_not_true"
        in intake["summary"]["violations"]
    )
    assert (
        "mutation_evidence_field_present:channels[0].midi_send_events"
        in intake["summary"]["violations"]
    )
    assert intake["artifacts"][0]["accepted_as_physical_fixture_candidate"] is False


def test_dlive_state_artifact_intake_discovers_json_files(tmp_path):
    artifact_dir = tmp_path / "dlive-state"
    artifact_dir.mkdir()
    artifact_path = artifact_dir / "candidate.json"
    artifact_path.write_text(json.dumps(_valid_artifact()), encoding="utf-8")
    (artifact_dir / "broken.json").write_text("{", encoding="utf-8")

    intake = build_dlive_state_physical_readback_artifact_intake(
        artifact_roots=(artifact_dir,)
    )

    assert intake["status"] == "invalid_operator_artifact_present"
    assert intake["summary"]["artifact_search_roots"] == [str(artifact_dir)]
    assert intake["summary"]["artifact_count"] == 2
    assert intake["summary"]["valid_candidate_count"] == 1
    assert intake["summary"]["invalid_candidate_count"] == 1
    assert intake["summary"]["accepted_as_physical_fixture_candidate"] is False
    assert intake["summary"]["blocked_by"] == [
        "operator_must_replace_invalid_dlive_state_physical_readback_artifact"
    ]
    assert any(row["source_id"] == str(artifact_path) for row in intake["artifacts"])
    assert any(
        "artifact_load_error:JSONDecodeError" in row["violations"]
        for row in intake["artifacts"]
    )


def test_dlive_state_artifact_intake_uses_repo_default_roots():
    intake = build_dlive_state_physical_readback_artifact_intake()

    assert intake["summary"]["artifact_search_roots"] == list(
        DEFAULT_DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_ROOTS
    )
    assert intake["safety"]["queries_console"] is False
    assert intake["safety"]["executes_tests"] is False
    assert intake["safety"]["midi_send_allowed"] is False
    assert intake["safety"]["tcp_send_allowed"] is False
    assert intake["safety"]["dlive_write_allowed"] is False
    assert intake["safety"]["dlive_apply_allowed"] is False


def test_dlive_state_artifact_intake_is_transport_free():
    source = Path("backend/dlive_state_physical_readback_artifact_intake.py").read_text(
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
        "send_midi",
        "_send_raw",
        "apply_operator_proposal",
    ]
    for token in forbidden:
        assert token not in source


def test_dlive_state_artifact_intake_is_registry_shadow_only_not_executor():
    method = get_processing_method(DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    intake = snapshot["dlive_state_physical_readback_artifact_intake"]
    assert intake["type"] == "dlive_state_physical_readback_artifact_intake"
    assert intake["safety"]["live_write"] is False
    assert intake["safety"]["fixture_closure_allowed_by_intake"] is False
    assert intake["safety"]["midi_send_allowed"] is False
    assert intake["safety"]["tcp_send_allowed"] is False
    assert intake["safety"]["dlive_write_allowed"] is False
    assert intake["safety"]["dlive_apply_allowed"] is False
    assert intake["summary"]["physical_fixture_gap_remains"] is True


def test_dlive_state_artifact_intake_is_console_read_goal_support_path():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID in row.method_ids
    assert "backend/dlive_state_physical_readback_artifact_intake.py" in row.support_paths
    assert row.current_live_write_kinds == ()
