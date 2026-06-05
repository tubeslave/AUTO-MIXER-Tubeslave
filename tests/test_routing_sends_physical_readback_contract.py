import copy
from pathlib import Path

from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.routing_sends_physical_readback_contract import (
    ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_TYPE,
    ROUTING_SENDS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    ROUTING_SENDS_PHYSICAL_READBACK_SCHEMA_ID,
    ROUTING_SENDS_REMEDIATION_ACTION_ID,
    ROUTING_SENDS_SURFACE_ID,
    build_routing_sends_physical_readback_contract,
    validate_routing_sends_physical_readback_artifact,
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


def test_routing_sends_contract_describes_future_physical_artifact():
    contract = build_routing_sends_physical_readback_contract()

    assert contract["type"] == "routing_sends_physical_readback_contract"
    assert contract["method_id"] == ROUTING_SENDS_PHYSICAL_READBACK_CONTRACT_METHOD_ID
    assert contract["schema_id"] == ROUTING_SENDS_PHYSICAL_READBACK_SCHEMA_ID
    assert contract["status"] == "contract_ready_physical_fixture_missing"
    assert contract["surface_id"] == ROUTING_SENDS_SURFACE_ID
    assert contract["remediation_action_id"] == ROUTING_SENDS_REMEDIATION_ACTION_ID
    assert contract["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "live_widening_allowed": False,
        "full_autonomous_ready": False,
    }
    artifact_contract = contract["artifact_contract"]
    assert artifact_contract["type"] == ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_TYPE
    assert "channel_bus_sends" in artifact_contract["required_top_level_fields"]
    assert "wing_routing_hint" in artifact_contract["required_static_route_fields"]
    assert "pre_on" in artifact_contract["required_bus_send_fields"]
    assert "target_matrix" in artifact_contract["required_matrix_send_fields"]
    assert "source_channel" in artifact_contract["required_output_route_fields"]
    assert "cache_matches_physical_readback" in artifact_contract[
        "required_send_coverage_flags"
    ]
    assert "physical_readback" in artifact_contract["allowed_sources"]
    assert "routing_write_events" in artifact_contract[
        "forbidden_mutation_evidence_fields"
    ]
    assert contract["summary"]["physical_readback_surface_id"] == ROUTING_SENDS_SURFACE_ID
    assert contract["summary"]["physical_fixture_gap_remains"] is True
    assert contract["summary"]["live_authority_granted"] is False
    assert "routing" in contract["summary"]["promotion_families_requiring_surface"]
    assert "scene_preset" in contract["summary"]["promotion_families_requiring_surface"]


def test_routing_sends_contract_validates_minimal_physical_artifact():
    result = validate_routing_sends_physical_readback_artifact(_valid_artifact())

    assert result["status"] == "valid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is True
    assert result["live_authority_granted"] is False
    assert result["full_autonomous_ready"] is False
    assert result["static_dante_route_count"] == 1
    assert result["channel_bus_send_count"] == 1
    assert result["channel_main_send_count"] == 1
    assert result["channel_matrix_send_count"] == 1
    assert result["output_route_count"] == 1
    assert result["violations"] == []
    assert result["evidence"]["source"] == "physical_readback"


def test_routing_sends_contract_rejects_live_or_cached_claims():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["source"] = "mixer_state_cache"
    artifact["full_autonomous_ready"] = True
    artifact["routing_write_events"] = []
    artifact["safety"]["live_write"] = True
    artifact["channel_bus_sends"][0]["readback_status"] = "verified_from_cache"
    artifact["channel_bus_sends"][0]["readback_source"] = "mixer_state_cache"
    artifact["channel_bus_sends"][0]["on"] = "true"
    artifact["channel_bus_sends"][0]["coverage"]["cache_matches_physical_readback"] = False
    artifact["output_routes"][0]["readback_status"] = "verified_from_cache"
    artifact["output_routes"][0]["readback_source"] = "mixer_state_cache"

    result = validate_routing_sends_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is False
    assert result["live_authority_granted"] is False
    assert result["full_autonomous_ready"] is False
    assert "source_not_physical_readback" in result["violations"]
    assert "top_level_full_autonomous_ready_not_false" in result["violations"]
    assert "safety_live_write_not_false" in result["violations"]
    assert "mutation_evidence_field_present:routing_write_events" in result["violations"]
    assert "bus_send_0_readback_status_not_physical_verified" in result["violations"]
    assert "bus_send_0_readback_source_not_physical_readback" in result["violations"]
    assert "bus_send_0_on_not_bool_or_0_1" in result["violations"]
    assert "bus_send_0_coverage_cache_matches_physical_readback_not_true" in result[
        "violations"
    ]
    assert "output_route_0_readback_status_not_physical_verified" in result["violations"]
    assert "output_route_0_readback_source_not_physical_readback" in result[
        "violations"
    ]


def test_routing_sends_contract_rejects_incomplete_rows():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["static_dante_route_count"] = 2
    artifact["channel_bus_send_count"] = 2
    artifact["channel_matrix_send_count"] = 2
    artifact["output_route_count"] = 2
    artifact["static_dante_routes"][0]["start"] = 9
    artifact["static_dante_routes"][0]["end"] = 8
    artifact["channel_bus_sends"][0]["send"] = 2
    artifact["channel_bus_sends"][0]["target_bus"] = 1
    artifact["channel_bus_sends"][0]["level_db"] = "-12"
    artifact["channel_matrix_sends"][0]["send"] = "BUS1"
    artifact["output_routes"][0]["output_number"] = 0
    del artifact["output_routes"][0]["destructive_change_marker"]

    result = validate_routing_sends_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert "static_dante_route_count_mismatch" in result["violations"]
    assert "channel_bus_send_count_mismatch" in result["violations"]
    assert "channel_matrix_send_count_mismatch" in result["violations"]
    assert "output_route_count_mismatch" in result["violations"]
    assert "static_route_0_range_invalid" in result["violations"]
    assert "bus_send_0_send_target_bus_mismatch" in result["violations"]
    assert "bus_send_0_level_db_not_number" in result["violations"]
    assert "matrix_send_0_send_target_matrix_mismatch" in result["violations"]
    assert "output_route_0_output_number_invalid" in result["violations"]
    assert "output_route_0_missing_field:destructive_change_marker" in result[
        "violations"
    ]


def test_routing_sends_contract_rejects_non_finite_numbers():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["channel_bus_sends"][0]["level_db"] = float("nan")
    artifact["channel_bus_sends"][0]["pan"] = float("inf")
    artifact["channel_main_sends"][0]["level_db"] = float("-inf")
    artifact["channel_matrix_sends"][0]["level_db"] = float("nan")

    result = validate_routing_sends_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is False
    assert result["live_authority_granted"] is False
    assert "bus_send_0_level_db_not_number" in result["violations"]
    assert "bus_send_0_pan_not_number" in result["violations"]
    assert "main_send_0_level_db_not_number" in result["violations"]
    assert "matrix_send_0_level_db_not_number" in result["violations"]


def test_routing_sends_contract_is_transport_free():
    source = Path("backend/routing_sends_physical_readback_contract.py").read_text(
        encoding="utf-8"
    )

    forbidden_transport_tokens = [
        "WingClient",
        "DLiveClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
        ".send(",
        "mixer_client",
    ]
    forbidden_write_call_tokens = [
        "set_bus_send(",
        "set_matrix_send(",
        "set_routing(",
        "load_snapshot(",
        "recall_scene(",
    ]
    for token in forbidden_transport_tokens + forbidden_write_call_tokens:
        assert token not in source


def test_routing_sends_contract_is_registry_shadow_only_not_executor():
    method = get_processing_method(ROUTING_SENDS_PHYSICAL_READBACK_CONTRACT_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert ROUTING_SENDS_PHYSICAL_READBACK_CONTRACT_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    contract = snapshot["routing_sends_physical_readback_contract"]
    assert contract["type"] == "routing_sends_physical_readback_contract"
    assert contract["safety"]["live_write"] is False
    assert contract["summary"]["physical_fixture_gap_remains"] is True


def test_routing_sends_contract_is_console_read_goal_support_path():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert ROUTING_SENDS_PHYSICAL_READBACK_CONTRACT_METHOD_ID in row.method_ids
    assert "backend/routing_sends_physical_readback_contract.py" in row.support_paths
    assert row.current_live_write_kinds == ()
