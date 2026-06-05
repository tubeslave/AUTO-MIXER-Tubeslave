import copy
import json
from pathlib import Path

from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.routing_sends_physical_readback_artifact_intake import (
    DEFAULT_ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_ROOTS,
    ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_routing_sends_physical_readback_artifact_intake,
)
from backend.routing_sends_physical_readback_contract import (
    ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_TYPE,
    ROUTING_SENDS_PHYSICAL_READBACK_SCHEMA_ID,
    ROUTING_SENDS_REMEDIATION_ACTION_ID,
    ROUTING_SENDS_SURFACE_ID,
)


def _valid_artifact() -> dict:
    return {
        "type": ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_TYPE,
        "schema_id": ROUTING_SENDS_PHYSICAL_READBACK_SCHEMA_ID,
        "surface_id": ROUTING_SENDS_SURFACE_ID,
        "remediation_action_id": ROUTING_SENDS_REMEDIATION_ACTION_ID,
        "captured_at": "2026-06-05T00:00:00Z",
        "source": "physical_readback",
        "console_model": "wing",
        "static_dante_route_count": 1,
        "channel_bus_send_count": 1,
        "channel_main_send_count": 1,
        "channel_matrix_send_count": 1,
        "output_route_count": 1,
        "static_dante_routes": [
            {
                "start": 1,
                "end": 8,
                "role": "drums",
                "tap_point": "direct_out",
                "label_short": "DR",
                "label_full": "Drums",
                "required": True,
                "wing_routing_hint": "ch:1-8",
                "used_by": ["channel_inventory"],
                "readback_status": "verified_from_physical_readback",
                "readback_source": "physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "coverage": {
                    "identity": True,
                    "static_config_match": True,
                    "physical_route_mapping": True,
                    "scoped_diff_marker": True,
                },
            }
        ],
        "channel_bus_sends": [
            {
                "channel": 2,
                "send": 1,
                "target_bus": 1,
                "readback_status": "verified_from_physical_readback",
                "readback_source": "physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "on": True,
                "level_db": -12.0,
                "pre_on": True,
                "mode": "PRE",
                "plink": False,
                "pan": 0.0,
                "active": True,
                "destructive_change_marker": "bus_send_write_would_change_console_routing",
                "coverage": {
                    "assignment": True,
                    "level": True,
                    "mode": True,
                    "scoped_diff_marker": True,
                    "cache_matches_physical_readback": True,
                },
            }
        ],
        "channel_main_sends": [
            {
                "channel": 2,
                "main": 1,
                "readback_status": "verified_from_physical_readback",
                "readback_source": "physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "on": True,
                "level_db": -3.0,
                "pre": False,
                "active": True,
                "destructive_change_marker": "main_send_write_would_change_console_routing",
                "coverage": {
                    "assignment": True,
                    "level": True,
                    "mode": True,
                    "scoped_diff_marker": True,
                    "cache_matches_physical_readback": True,
                },
            }
        ],
        "channel_matrix_sends": [
            {
                "channel": 2,
                "send": "MX1",
                "target_matrix": 1,
                "readback_status": "verified_from_physical_readback",
                "readback_source": "physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "on": True,
                "level_db": -18.0,
                "active": True,
                "destructive_change_marker": "matrix_send_write_would_change_console_routing",
                "coverage": {
                    "assignment": True,
                    "level": True,
                    "mode": True,
                    "scoped_diff_marker": True,
                    "cache_matches_physical_readback": True,
                },
            }
        ],
        "output_routes": [
            {
                "output_group": "MOD",
                "output_number": 1,
                "readback_status": "verified_from_physical_readback",
                "readback_source": "physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "source_group": "CH",
                "source_channel": 2,
                "destructive_change_marker": "output_routing_write_would_repatch_console",
                "coverage": {
                    "assignment": True,
                    "source": True,
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


def test_routing_sends_artifact_intake_reports_missing_without_authority():
    intake = build_routing_sends_physical_readback_artifact_intake(
        artifact_roots=("does-not-exist/routing-sends",)
    )

    assert intake["type"] == "routing_sends_physical_readback_artifact_intake"
    assert intake["method_id"] == ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID
    assert intake["status"] == "operator_artifact_missing"
    assert intake["surface_id"] == ROUTING_SENDS_SURFACE_ID
    assert intake["remediation_action_id"] == ROUTING_SENDS_REMEDIATION_ACTION_ID
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
        "does-not-exist/routing-sends"
    ]
    assert intake["summary"]["artifact_count"] == 0
    assert intake["summary"]["valid_candidate_count"] == 0
    assert intake["summary"]["invalid_candidate_count"] == 0
    assert intake["summary"]["valid_static_dante_route_count"] == 0
    assert intake["summary"]["valid_channel_bus_send_count"] == 0
    assert intake["summary"]["valid_channel_main_send_count"] == 0
    assert intake["summary"]["valid_channel_matrix_send_count"] == 0
    assert intake["summary"]["valid_output_route_count"] == 0
    assert intake["summary"]["accepted_as_physical_fixture_candidate"] is False
    assert intake["summary"]["operator_review_required"] is True
    assert intake["summary"]["physical_fixture_gap_remains"] is True
    assert intake["summary"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["live_authority_granted"] is False
    assert intake["summary"]["full_autonomous_ready"] is False
    assert intake["summary"]["promotion_families_requiring_surface"] == [
        "routing",
        "fx_send_insert",
        "bus_group_matrix",
        "scene_preset",
        "mix_balance",
        "performance_adaptation",
    ]
    assert intake["summary"]["blocked_by"] == [
        "operator_must_capture_routing_sends_physical_readback_artifact"
    ]
    assert intake["artifacts"] == []


def test_routing_sends_artifact_intake_accepts_valid_candidate_but_keeps_review_gate():
    intake = build_routing_sends_physical_readback_artifact_intake(
        artifact_payloads=(_valid_artifact(),),
        artifact_roots=(),
    )

    assert intake["status"] == "valid_operator_artifact_available"
    assert intake["summary"]["artifact_count"] == 1
    assert intake["summary"]["valid_candidate_count"] == 1
    assert intake["summary"]["invalid_candidate_count"] == 0
    assert intake["summary"]["valid_static_dante_route_count"] == 1
    assert intake["summary"]["valid_channel_bus_send_count"] == 1
    assert intake["summary"]["valid_channel_main_send_count"] == 1
    assert intake["summary"]["valid_channel_matrix_send_count"] == 1
    assert intake["summary"]["valid_output_route_count"] == 1
    assert intake["summary"]["accepted_as_physical_fixture_candidate"] is True
    assert intake["summary"]["operator_review_required"] is True
    assert intake["summary"]["physical_fixture_gap_remains"] is True
    assert intake["summary"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["live_authority_granted"] is False
    assert intake["summary"]["latest_valid_candidate_source_id"] == "inline_payload_0"
    assert intake["summary"]["blocked_by"] == [
        "operator_must_review_valid_routing_sends_candidate_before_fixture_closure"
    ]
    assert intake["artifacts"][0]["status"] == "valid_physical_readback_artifact"
    assert intake["artifacts"][0]["accepted_as_physical_fixture_candidate"] is True
    assert intake["artifacts"][0]["static_dante_route_count"] == 1
    assert intake["artifacts"][0]["channel_bus_send_count"] == 1
    assert intake["artifacts"][0]["channel_main_send_count"] == 1
    assert intake["artifacts"][0]["channel_matrix_send_count"] == 1
    assert intake["artifacts"][0]["output_route_count"] == 1
    assert intake["artifacts"][0]["violations"] == []


def test_routing_sends_artifact_intake_rejects_invalid_or_live_claiming_candidate():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["source"] = "mixer_state_cache"
    artifact["safety"]["live_write"] = True
    artifact["routing_write_events"] = []
    artifact["channel_bus_sends"][0]["readback_status"] = "verified_from_cache"
    artifact["channel_bus_sends"][0]["readback_source"] = "mixer_state_cache"
    artifact["channel_bus_sends"][0]["on"] = "true"
    artifact["channel_bus_sends"][0]["coverage"]["cache_matches_physical_readback"] = False
    artifact["output_routes"][0]["readback_status"] = "verified_from_cache"
    artifact["output_routes"][0]["readback_source"] = "mixer_state_cache"

    intake = build_routing_sends_physical_readback_artifact_intake(
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
    assert (
        "mutation_evidence_field_present:routing_write_events"
        in intake["summary"]["violations"]
    )
    assert "bus_send_0_readback_status_not_physical_verified" in intake["summary"][
        "violations"
    ]
    assert "bus_send_0_readback_source_not_physical_readback" in intake["summary"][
        "violations"
    ]
    assert "bus_send_0_on_not_bool_or_0_1" in intake["summary"]["violations"]
    assert (
        "bus_send_0_coverage_cache_matches_physical_readback_not_true"
        in intake["summary"]["violations"]
    )
    assert "output_route_0_readback_status_not_physical_verified" in intake["summary"][
        "violations"
    ]
    assert "output_route_0_readback_source_not_physical_readback" in intake["summary"][
        "violations"
    ]
    assert intake["artifacts"][0]["accepted_as_physical_fixture_candidate"] is False


def test_routing_sends_artifact_intake_discovers_json_files(tmp_path):
    artifact_dir = tmp_path / "routing-sends"
    artifact_dir.mkdir()
    artifact_path = artifact_dir / "candidate.json"
    artifact_path.write_text(json.dumps(_valid_artifact()), encoding="utf-8")
    (artifact_dir / "broken.json").write_text("{", encoding="utf-8")

    intake = build_routing_sends_physical_readback_artifact_intake(
        artifact_roots=(artifact_dir,)
    )

    assert intake["status"] == "invalid_operator_artifact_present"
    assert intake["summary"]["artifact_search_roots"] == [str(artifact_dir)]
    assert intake["summary"]["artifact_count"] == 2
    assert intake["summary"]["valid_candidate_count"] == 1
    assert intake["summary"]["invalid_candidate_count"] == 1
    assert intake["summary"]["accepted_as_physical_fixture_candidate"] is False
    assert intake["summary"]["blocked_by"] == [
        "operator_must_replace_invalid_routing_sends_physical_readback_artifact"
    ]
    assert any(row["source_id"] == str(artifact_path) for row in intake["artifacts"])
    assert any(
        "artifact_load_error:JSONDecodeError" in row["violations"]
        for row in intake["artifacts"]
    )


def test_routing_sends_artifact_intake_uses_repo_default_roots():
    intake = build_routing_sends_physical_readback_artifact_intake()

    assert intake["summary"]["artifact_search_roots"] == list(
        DEFAULT_ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_ROOTS
    )
    assert intake["safety"]["queries_console"] is False
    assert intake["safety"]["executes_tests"] is False


def test_routing_sends_artifact_intake_is_transport_free():
    source = Path("backend/routing_sends_physical_readback_artifact_intake.py").read_text(
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
        "set_bus_send",
        "set_main_send",
        "set_matrix_send",
        "set_output_route",
        "set_routing",
        "route_output",
        "load_snapshot",
        "recall_scene",
        "apply_operator_proposal",
    ]
    for token in forbidden:
        assert token not in source


def test_routing_sends_artifact_intake_is_registry_shadow_only_not_executor():
    method = get_processing_method(ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    intake = snapshot["routing_sends_physical_readback_artifact_intake"]
    assert intake["type"] == "routing_sends_physical_readback_artifact_intake"
    assert intake["safety"]["live_write"] is False
    assert intake["safety"]["fixture_closure_allowed_by_intake"] is False
    assert intake["summary"]["physical_fixture_gap_remains"] is True


def test_routing_sends_artifact_intake_is_console_read_goal_support_path():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID in row.method_ids
    assert "backend/routing_sends_physical_readback_artifact_intake.py" in row.support_paths
    assert row.current_live_write_kinds == ()
