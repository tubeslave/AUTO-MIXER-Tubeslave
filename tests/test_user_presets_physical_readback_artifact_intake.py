import copy
import json
from pathlib import Path

from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.user_presets_physical_readback_artifact_intake import (
    DEFAULT_USER_PRESETS_PHYSICAL_READBACK_ARTIFACT_ROOTS,
    USER_PRESETS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_user_presets_physical_readback_artifact_intake,
)
from backend.user_presets_physical_readback_contract import (
    USER_PRESETS_PHYSICAL_READBACK_ARTIFACT_TYPE,
    USER_PRESETS_PHYSICAL_READBACK_SCHEMA_ID,
    USER_PRESETS_REMEDIATION_ACTION_ID,
    USER_PRESETS_SURFACE_ID,
)


def _valid_artifact() -> dict:
    return {
        "type": USER_PRESETS_PHYSICAL_READBACK_ARTIFACT_TYPE,
        "schema_id": USER_PRESETS_PHYSICAL_READBACK_SCHEMA_ID,
        "surface_id": USER_PRESETS_SURFACE_ID,
        "remediation_action_id": USER_PRESETS_REMEDIATION_ACTION_ID,
        "captured_at": "2026-06-05T00:00:00Z",
        "source": "wing_physical_readback",
        "console_model": "WING Rack",
        "preset_count": 1,
        "presets": [
            {
                "preset_index": 1,
                "preset_name": "Lead Vox Chain",
                "preset_scope": "channel_processing",
                "target_type": "channel",
                "target_label": "Lead Vox",
                "readback_status": "verified_from_physical_readback",
                "readback_source": "wing_physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "preset_recall_policy": "blocked_manual_only",
                "preset_store_policy": "blocked_manual_only",
                "destructive_recall_marker": "preset_recall_would_change_parameters",
                "destructive_store_marker": "preset_store_would_persist_parameters",
                "preset_content_fingerprint": "sha256:preset-content",
                "scoped_diff_marker": "diff:preset-1-before-after",
                "rollback_reference": "rollback:preset-1",
                "coverage": {
                    "catalog_identity": True,
                    "scope": True,
                    "target": True,
                    "preset_content_fingerprint": True,
                    "scoped_diff_marker": True,
                    "rollback_reference": True,
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
            "preset_recall_allowed": False,
            "preset_store_allowed": False,
        },
    }


def test_user_presets_artifact_intake_reports_missing_without_authority():
    intake = build_user_presets_physical_readback_artifact_intake(
        artifact_roots=("does-not-exist/user-presets",)
    )

    assert intake["type"] == "user_presets_physical_readback_artifact_intake"
    assert intake["method_id"] == USER_PRESETS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID
    assert intake["status"] == "operator_artifact_missing"
    assert intake["surface_id"] == USER_PRESETS_SURFACE_ID
    assert intake["remediation_action_id"] == USER_PRESETS_REMEDIATION_ACTION_ID
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
        "preset_recall_allowed": False,
        "preset_store_allowed": False,
    }
    assert intake["summary"]["artifact_search_roots"] == [
        "does-not-exist/user-presets"
    ]
    assert intake["summary"]["artifact_count"] == 0
    assert intake["summary"]["valid_candidate_count"] == 0
    assert intake["summary"]["invalid_candidate_count"] == 0
    assert intake["summary"]["valid_preset_count"] == 0
    assert intake["summary"]["accepted_as_physical_fixture_candidate"] is False
    assert intake["summary"]["operator_review_required"] is True
    assert intake["summary"]["physical_fixture_gap_remains"] is True
    assert intake["summary"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["preset_recall_allowed"] is False
    assert intake["summary"]["preset_store_allowed"] is False
    assert intake["summary"]["live_authority_granted"] is False
    assert intake["summary"]["full_autonomous_ready"] is False
    assert intake["summary"]["promotion_families_requiring_surface"] == [
        "scene_preset",
        "routing",
        "mix_balance",
        "performance_adaptation",
    ]
    assert intake["summary"]["blocked_by"] == [
        "operator_must_capture_user_presets_physical_readback_artifact"
    ]
    assert intake["artifacts"] == []


def test_user_presets_artifact_intake_accepts_valid_candidate_but_keeps_gate():
    intake = build_user_presets_physical_readback_artifact_intake(
        artifact_payloads=(_valid_artifact(),),
        artifact_roots=(),
    )

    assert intake["status"] == "valid_operator_artifact_available"
    assert intake["summary"]["artifact_count"] == 1
    assert intake["summary"]["valid_candidate_count"] == 1
    assert intake["summary"]["invalid_candidate_count"] == 0
    assert intake["summary"]["valid_preset_count"] == 1
    assert intake["summary"]["accepted_as_physical_fixture_candidate"] is True
    assert intake["summary"]["operator_review_required"] is True
    assert intake["summary"]["physical_fixture_gap_remains"] is True
    assert intake["summary"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["preset_recall_allowed"] is False
    assert intake["summary"]["preset_store_allowed"] is False
    assert intake["summary"]["live_authority_granted"] is False
    assert intake["summary"]["latest_valid_candidate_source_id"] == "inline_payload_0"
    assert intake["summary"]["blocked_by"] == [
        "operator_must_review_valid_user_presets_candidate_before_fixture_closure"
    ]
    assert intake["artifacts"][0]["status"] == "valid_physical_readback_artifact"
    assert intake["artifacts"][0]["accepted_as_physical_fixture_candidate"] is True
    assert intake["artifacts"][0]["preset_count"] == 1
    assert intake["artifacts"][0]["violations"] == []


def test_user_presets_artifact_intake_rejects_invalid_or_live_claiming_candidate():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["source"] = "mixer_state_cache"
    artifact["live_write"] = True
    artifact["safety"]["preset_store_allowed"] = True
    artifact["preset_recall_events"] = [{"preset": 1}]
    artifact["presets"][0]["readback_status"] = "verified_from_cache"
    artifact["presets"][0]["readback_source"] = "mixer_state_cache"
    artifact["presets"][0]["coverage"]["cache_matches_physical_readback"] = False

    intake = build_user_presets_physical_readback_artifact_intake(
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
    assert intake["summary"]["preset_recall_allowed"] is False
    assert intake["summary"]["preset_store_allowed"] is False
    assert intake["summary"]["live_authority_granted"] is False
    assert "source_not_physical_readback" in intake["summary"]["violations"]
    assert "safety_preset_store_allowed_not_false" in intake["summary"]["violations"]
    assert "top_level_live_write_not_false" in intake["summary"]["violations"]
    assert (
        "mutation_evidence_field_present:preset_recall_events"
        in intake["summary"]["violations"]
    )
    assert "preset_0_readback_status_not_physical_verified" in intake["summary"][
        "violations"
    ]
    assert "preset_0_readback_source_not_physical_readback" in intake["summary"][
        "violations"
    ]
    assert (
        "preset_0_coverage_cache_matches_physical_readback_not_true"
        in intake["summary"]["violations"]
    )
    assert intake["artifacts"][0]["accepted_as_physical_fixture_candidate"] is False


def test_user_presets_artifact_intake_discovers_json_files(tmp_path):
    artifact_dir = tmp_path / "user-presets"
    artifact_dir.mkdir()
    artifact_path = artifact_dir / "candidate.json"
    artifact_path.write_text(json.dumps(_valid_artifact()), encoding="utf-8")
    (artifact_dir / "broken.json").write_text("{", encoding="utf-8")

    intake = build_user_presets_physical_readback_artifact_intake(
        artifact_roots=(artifact_dir,)
    )

    assert intake["status"] == "invalid_operator_artifact_present"
    assert intake["summary"]["artifact_search_roots"] == [str(artifact_dir)]
    assert intake["summary"]["artifact_count"] == 2
    assert intake["summary"]["valid_candidate_count"] == 1
    assert intake["summary"]["invalid_candidate_count"] == 1
    assert intake["summary"]["accepted_as_physical_fixture_candidate"] is False
    assert intake["summary"]["blocked_by"] == [
        "operator_must_replace_invalid_user_presets_physical_readback_artifact"
    ]
    assert any(row["source_id"] == str(artifact_path) for row in intake["artifacts"])
    assert any(
        "artifact_load_error:JSONDecodeError" in row["violations"]
        for row in intake["artifacts"]
    )


def test_user_presets_artifact_intake_uses_repo_default_roots():
    intake = build_user_presets_physical_readback_artifact_intake()

    assert intake["summary"]["artifact_search_roots"] == list(
        DEFAULT_USER_PRESETS_PHYSICAL_READBACK_ARTIFACT_ROOTS
    )
    assert intake["safety"]["queries_console"] is False
    assert intake["safety"]["executes_tests"] is False
    assert intake["safety"]["preset_recall_allowed"] is False
    assert intake["safety"]["preset_store_allowed"] is False


def test_user_presets_artifact_intake_is_transport_free():
    source = Path("backend/user_presets_physical_readback_artifact_intake.py").read_text(
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
        "recall_preset",
        "store_preset",
        "load_preset",
        "save_preset",
        "apply_operator_proposal",
    ]
    for token in forbidden:
        assert token not in source


def test_user_presets_artifact_intake_is_registry_shadow_only_not_executor():
    method = get_processing_method(USER_PRESETS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert USER_PRESETS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    intake = snapshot["user_presets_physical_readback_artifact_intake"]
    assert intake["type"] == "user_presets_physical_readback_artifact_intake"
    assert intake["safety"]["live_write"] is False
    assert intake["safety"]["fixture_closure_allowed_by_intake"] is False
    assert intake["safety"]["preset_recall_allowed"] is False
    assert intake["safety"]["preset_store_allowed"] is False
    assert intake["summary"]["physical_fixture_gap_remains"] is True


def test_user_presets_artifact_intake_is_console_read_goal_support_path():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert USER_PRESETS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID in row.method_ids
    assert "backend/user_presets_physical_readback_artifact_intake.py" in row.support_paths
    assert row.current_live_write_kinds == ()
