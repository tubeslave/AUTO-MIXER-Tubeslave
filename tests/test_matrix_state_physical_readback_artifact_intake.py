import copy
import json
from pathlib import Path

from backend.matrix_state_physical_readback_artifact_intake import (
    DEFAULT_MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_ROOTS,
    MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_matrix_state_physical_readback_artifact_intake,
)
from backend.matrix_state_physical_readback_contract import (
    MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE,
    MATRIX_STATE_PHYSICAL_READBACK_SCHEMA_ID,
    MATRIX_STATE_REMEDIATION_ACTION_ID,
    MATRIX_STATE_SURFACE_ID,
    MINIMUM_MATRIX_EQ_BAND_COUNT,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


def _eq_bands() -> list[dict]:
    return [
        {
            "band": band,
            "frequency_hz": float(100 * band),
            "gain_db": 0.0,
            "q": 1.0,
        }
        for band in range(1, MINIMUM_MATRIX_EQ_BAND_COUNT + 1)
    ]


def _valid_artifact() -> dict:
    return {
        "type": MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE,
        "schema_id": MATRIX_STATE_PHYSICAL_READBACK_SCHEMA_ID,
        "surface_id": MATRIX_STATE_SURFACE_ID,
        "remediation_action_id": MATRIX_STATE_REMEDIATION_ACTION_ID,
        "captured_at": "2026-06-05T00:00:00Z",
        "source": "physical_readback",
        "console_model": "wing",
        "matrix_count": 1,
        "matrices": [
            {
                "matrix_id": 1,
                "matrix_label": "Matrix 1",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "controls": {
                    "fader_db": -4.0,
                    "mute": False,
                    "pan": 0.0,
                },
                "eq": {
                    "enabled": True,
                    "bands": _eq_bands(),
                },
                "dynamics": {
                    "enabled": True,
                    "model": "matrix_compressor",
                    "threshold_db": -18.0,
                    "ratio": 2.0,
                    "knee": 3.0,
                    "detector": "peak",
                    "attack_ms": 12.0,
                    "release_ms": 120.0,
                    "makeup_db": 1.0,
                    "mix_pct": 100.0,
                    "gain_reduction_db": 2.5,
                },
                "inserts": {
                    "pre_insert_on": False,
                    "pre_insert_slot": "none",
                    "pre_insert_status": "empty",
                    "post_insert_on": True,
                    "post_insert_slot": "limiter",
                    "post_insert_status": "active",
                    "post_insert_mode": "post_fader",
                    "post_insert_weight": 100.0,
                },
                "metering": {
                    "lufs": -25.0,
                    "true_peak_dbtp": -5.0,
                    "headroom_db": 4.0,
                },
                "coverage": {
                    "controls": True,
                    "eq": True,
                    "dynamics": True,
                    "inserts": True,
                    "metering": True,
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


def test_matrix_state_artifact_intake_reports_missing_without_authority():
    intake = build_matrix_state_physical_readback_artifact_intake(
        artifact_roots=("does-not-exist/matrix-state",)
    )

    assert intake["type"] == "matrix_state_physical_readback_artifact_intake"
    assert intake["method_id"] == MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID
    assert intake["status"] == "operator_artifact_missing"
    assert intake["surface_id"] == MATRIX_STATE_SURFACE_ID
    assert intake["remediation_action_id"] == MATRIX_STATE_REMEDIATION_ACTION_ID
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
    assert intake["summary"]["artifact_search_roots"] == ["does-not-exist/matrix-state"]
    assert intake["summary"]["artifact_count"] == 0
    assert intake["summary"]["valid_candidate_count"] == 0
    assert intake["summary"]["invalid_candidate_count"] == 0
    assert intake["summary"]["valid_matrix_count"] == 0
    assert intake["summary"]["accepted_as_physical_fixture_candidate"] is False
    assert intake["summary"]["operator_review_required"] is True
    assert intake["summary"]["physical_fixture_gap_remains"] is True
    assert intake["summary"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["live_authority_granted"] is False
    assert intake["summary"]["full_autonomous_ready"] is False
    assert intake["summary"]["promotion_families_requiring_surface"] == [
        "bus_group_matrix",
        "fx_send_insert",
        "master_bus",
        "mix_balance",
        "routing",
        "performance_adaptation",
    ]
    assert intake["summary"]["blocked_by"] == [
        "operator_must_capture_matrix_state_physical_readback_artifact"
    ]
    assert intake["artifacts"] == []


def test_matrix_state_artifact_intake_accepts_valid_candidate_but_keeps_review_gate():
    intake = build_matrix_state_physical_readback_artifact_intake(
        artifact_payloads=(_valid_artifact(),),
        artifact_roots=(),
    )

    assert intake["status"] == "valid_operator_artifact_available"
    assert intake["summary"]["artifact_count"] == 1
    assert intake["summary"]["valid_candidate_count"] == 1
    assert intake["summary"]["invalid_candidate_count"] == 0
    assert intake["summary"]["valid_matrix_count"] == 1
    assert intake["summary"]["accepted_as_physical_fixture_candidate"] is True
    assert intake["summary"]["operator_review_required"] is True
    assert intake["summary"]["physical_fixture_gap_remains"] is True
    assert intake["summary"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["live_authority_granted"] is False
    assert intake["summary"]["latest_valid_candidate_source_id"] == "inline_payload_0"
    assert intake["summary"]["blocked_by"] == [
        "operator_must_review_valid_matrix_state_candidate_before_fixture_closure"
    ]
    assert intake["artifacts"][0]["status"] == "valid_physical_readback_artifact"
    assert intake["artifacts"][0]["accepted_as_physical_fixture_candidate"] is True
    assert intake["artifacts"][0]["matrix_count"] == 1
    assert intake["artifacts"][0]["violations"] == []


def test_matrix_state_artifact_intake_rejects_invalid_or_live_claiming_candidate():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["source"] = "mixer_state_cache"
    artifact["safety"]["live_write"] = True
    artifact["matrix_write_events"] = []
    artifact["matrices"][0]["status"] = "verified_from_cache"
    artifact["matrices"][0]["source"] = "mixer_state_cache"
    artifact["matrices"][0]["controls"]["mute"] = "false"
    artifact["matrices"][0]["coverage"]["cache_matches_physical_readback"] = False

    intake = build_matrix_state_physical_readback_artifact_intake(
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
    assert "mutation_evidence_field_present:matrix_write_events" in intake["summary"][
        "violations"
    ]
    assert "matrix_0_status_not_physical_verified" in intake["summary"]["violations"]
    assert "matrix_0_source_not_physical_readback" in intake["summary"]["violations"]
    assert "matrix_0_controls_mute_not_bool_or_0_1" in intake["summary"]["violations"]
    assert (
        "matrix_0_coverage_cache_matches_physical_readback_not_true"
        in intake["summary"]["violations"]
    )
    assert intake["artifacts"][0]["accepted_as_physical_fixture_candidate"] is False


def test_matrix_state_artifact_intake_discovers_json_files(tmp_path):
    artifact_dir = tmp_path / "matrix-state"
    artifact_dir.mkdir()
    artifact_path = artifact_dir / "candidate.json"
    artifact_path.write_text(json.dumps(_valid_artifact()), encoding="utf-8")
    (artifact_dir / "broken.json").write_text("{", encoding="utf-8")

    intake = build_matrix_state_physical_readback_artifact_intake(
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


def test_matrix_state_artifact_intake_uses_repo_default_roots():
    intake = build_matrix_state_physical_readback_artifact_intake()

    assert intake["summary"]["artifact_search_roots"] == list(
        DEFAULT_MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_ROOTS
    )
    assert intake["safety"]["queries_console"] is False
    assert intake["safety"]["executes_tests"] is False


def test_matrix_state_artifact_intake_is_transport_free():
    source = Path("backend/matrix_state_physical_readback_artifact_intake.py").read_text(
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
        "set_matrix_fader",
        "set_matrix_mute",
        "set_matrix_send",
        "set_matrix_eq",
        "set_matrix_dynamics",
        "set_insert",
        "apply_operator_proposal",
    ]
    for token in forbidden:
        assert token not in source


def test_matrix_state_artifact_intake_is_registry_shadow_only_not_executor():
    method = get_processing_method(MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    intake = snapshot["matrix_state_physical_readback_artifact_intake"]
    assert intake["type"] == "matrix_state_physical_readback_artifact_intake"
    assert intake["safety"]["live_write"] is False
    assert intake["safety"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["physical_fixture_gap_remains"] is True


def test_matrix_state_artifact_intake_is_console_read_goal_support_path():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID in row.method_ids
    assert "backend/matrix_state_physical_readback_artifact_intake.py" in row.support_paths
    assert row.current_live_write_kinds == ()
