import copy
import json
from pathlib import Path

from backend.channel_processing_physical_readback_artifact_intake import (
    CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    DEFAULT_CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_ROOTS,
    build_channel_processing_physical_readback_artifact_intake,
)
from backend.channel_processing_physical_readback_contract import (
    CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_TYPE,
    CHANNEL_PROCESSING_PHYSICAL_READBACK_SCHEMA_ID,
    CHANNEL_PROCESSING_REMEDIATION_ACTION_ID,
    CHANNEL_PROCESSING_SURFACE_ID,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


def _processing_channel(channel_id: int, label: str) -> dict:
    return {
        "channel_id": channel_id,
        "channel_label": label,
        "status": "verified_from_physical_readback",
        "source": "physical_readback",
        "captured_at": "2026-06-05T00:00:00Z",
        "hpf": {
            "enabled": 1,
            "frequency_hz": 120.0,
        },
        "eq": {
            "enabled": 1,
            "bands": [
                {"band": 1, "frequency_hz": 100.0, "gain_db": -1.0, "q": 0.8},
                {"band": 2, "frequency_hz": 250.0, "gain_db": -2.0, "q": 0.9},
                {"band": 3, "frequency_hz": 1200.0, "gain_db": 1.0, "q": 1.2},
                {"band": 4, "frequency_hz": 4500.0, "gain_db": -1.5, "q": 1.0},
            ],
        },
        "compressor": {
            "enabled": 1,
            "model": "COMP",
            "threshold_db": -18.0,
            "ratio": "3.0",
            "attack_ms": 12.0,
            "release_ms": 180.0,
            "makeup_db": 2.0,
            "mix_pct": 80.0,
            "detector": "PEAK",
            "gain_reduction_db": 2.4,
        },
        "gate": {
            "enabled": 1,
            "model": "GATE",
            "threshold_db": -47.5,
            "range_db": 18.0,
            "attack_ms": 4.0,
            "hold_ms": 80.0,
            "release_ms": 240.0,
            "accent": 3.0,
            "ratio": "4.0",
        },
        "coverage": {
            "hpf": True,
            "eq": True,
            "compressor": True,
            "gate": True,
            "cache_matches_physical_readback": True,
        },
    }


def _valid_artifact() -> dict:
    return {
        "type": CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_TYPE,
        "schema_id": CHANNEL_PROCESSING_PHYSICAL_READBACK_SCHEMA_ID,
        "surface_id": CHANNEL_PROCESSING_SURFACE_ID,
        "remediation_action_id": CHANNEL_PROCESSING_REMEDIATION_ACTION_ID,
        "captured_at": "2026-06-05T00:00:00Z",
        "source": "physical_readback",
        "console_model": "wing",
        "channel_count": 2,
        "channels": [
            _processing_channel(1, "Lead Vox"),
            _processing_channel(2, "Snare Top"),
        ],
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
        },
    }


def test_channel_processing_artifact_intake_reports_missing_without_authority():
    intake = build_channel_processing_physical_readback_artifact_intake(
        artifact_roots=("does-not-exist/channel-processing",)
    )

    assert intake["type"] == "channel_processing_physical_readback_artifact_intake"
    assert intake["method_id"] == CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID
    assert intake["status"] == "operator_artifact_missing"
    assert intake["surface_id"] == CHANNEL_PROCESSING_SURFACE_ID
    assert intake["remediation_action_id"] == CHANNEL_PROCESSING_REMEDIATION_ACTION_ID
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
        "does-not-exist/channel-processing"
    ]
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
        "eq",
        "compressor",
        "gate",
        "deessing",
        "frequency_separation",
        "cross_channel_unmasking",
    ]
    assert intake["summary"]["blocked_by"] == [
        "operator_must_capture_channel_processing_physical_readback_artifact"
    ]
    assert intake["artifacts"] == []


def test_channel_processing_artifact_intake_accepts_valid_candidate_but_keeps_review_gate():
    intake = build_channel_processing_physical_readback_artifact_intake(
        artifact_payloads=(_valid_artifact(),),
        artifact_roots=(),
    )

    assert intake["status"] == "valid_operator_artifact_available"
    assert intake["summary"]["artifact_count"] == 1
    assert intake["summary"]["valid_candidate_count"] == 1
    assert intake["summary"]["invalid_candidate_count"] == 0
    assert intake["summary"]["valid_channel_count"] == 2
    assert intake["summary"]["accepted_as_physical_fixture_candidate"] is True
    assert intake["summary"]["operator_review_required"] is True
    assert intake["summary"]["physical_fixture_gap_remains"] is True
    assert intake["summary"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["live_authority_granted"] is False
    assert intake["summary"]["latest_valid_candidate_source_id"] == "inline_payload_0"
    assert intake["summary"]["blocked_by"] == [
        "operator_must_review_valid_channel_processing_candidate_before_fixture_closure"
    ]
    assert intake["artifacts"][0]["status"] == "valid_physical_readback_artifact"
    assert intake["artifacts"][0]["accepted_as_physical_fixture_candidate"] is True
    assert intake["artifacts"][0]["channel_count"] == 2
    assert intake["artifacts"][0]["violations"] == []


def test_channel_processing_artifact_intake_rejects_invalid_or_live_claiming_candidate():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["source"] = "mixer_state_cache"
    artifact["safety"]["live_write"] = True
    artifact["eq_write_events"] = []
    artifact["channels"][0]["status"] = "verified_from_cache"
    artifact["channels"][0]["source"] = "mixer_state_cache"
    artifact["channels"][0]["hpf"]["enabled"] = "on"
    artifact["channels"][0]["coverage"]["cache_matches_physical_readback"] = False

    intake = build_channel_processing_physical_readback_artifact_intake(
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
    assert "mutation_evidence_field_present:eq_write_events" in intake["summary"]["violations"]
    assert "channel_0_status_not_physical_verified" in intake["summary"]["violations"]
    assert "channel_0_source_not_physical_readback" in intake["summary"]["violations"]
    assert "channel_0_hpf_enabled_not_bool_or_0_1" in intake["summary"]["violations"]
    assert (
        "channel_0_coverage_cache_matches_physical_readback_not_true"
        in intake["summary"]["violations"]
    )
    assert intake["artifacts"][0]["accepted_as_physical_fixture_candidate"] is False


def test_channel_processing_artifact_intake_discovers_json_files(tmp_path):
    artifact_dir = tmp_path / "channel-processing"
    artifact_dir.mkdir()
    artifact_path = artifact_dir / "candidate.json"
    artifact_path.write_text(json.dumps(_valid_artifact()), encoding="utf-8")
    (artifact_dir / "broken.json").write_text("{", encoding="utf-8")

    intake = build_channel_processing_physical_readback_artifact_intake(
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


def test_channel_processing_artifact_intake_uses_repo_default_roots():
    intake = build_channel_processing_physical_readback_artifact_intake()

    assert intake["summary"]["artifact_search_roots"] == list(
        DEFAULT_CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_ROOTS
    )
    assert intake["safety"]["queries_console"] is False
    assert intake["safety"]["executes_tests"] is False


def test_channel_processing_artifact_intake_is_transport_free():
    source = Path("backend/channel_processing_physical_readback_artifact_intake.py").read_text(
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
        "set_gate",
        "reset_channel_processing",
        "apply_operator_proposal",
    ]
    for token in forbidden:
        assert token not in source


def test_channel_processing_artifact_intake_is_registry_shadow_only_not_executor():
    method = get_processing_method(
        CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID
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
    assert CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    intake = snapshot["channel_processing_physical_readback_artifact_intake"]
    assert intake["type"] == "channel_processing_physical_readback_artifact_intake"
    assert intake["safety"]["live_write"] is False
    assert intake["safety"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["physical_fixture_gap_remains"] is True


def test_channel_processing_artifact_intake_is_console_read_goal_support_path():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID in row.method_ids
    assert "backend/channel_processing_physical_readback_artifact_intake.py" in row.support_paths
    assert row.current_live_write_kinds == ()
