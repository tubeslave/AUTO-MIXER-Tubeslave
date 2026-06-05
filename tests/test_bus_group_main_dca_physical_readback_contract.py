import copy
from pathlib import Path

from backend.bus_group_main_dca_physical_readback_contract import (
    BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_TYPE,
    BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SCHEMA_ID,
    BUS_GROUP_MAIN_DCA_REMEDIATION_ACTION_ID,
    BUS_GROUP_MAIN_DCA_SURFACE_ID,
    build_bus_group_main_dca_physical_readback_contract,
    validate_bus_group_main_dca_physical_readback_artifact,
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


def test_bus_group_main_dca_contract_describes_future_physical_artifact():
    contract = build_bus_group_main_dca_physical_readback_contract()

    assert contract["type"] == "bus_group_main_dca_physical_readback_contract"
    assert contract["method_id"] == BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_CONTRACT_METHOD_ID
    assert contract["schema_id"] == BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SCHEMA_ID
    assert contract["status"] == "contract_ready_physical_fixture_missing"
    assert contract["surface_id"] == BUS_GROUP_MAIN_DCA_SURFACE_ID
    assert contract["remediation_action_id"] == BUS_GROUP_MAIN_DCA_REMEDIATION_ACTION_ID
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
    assert artifact_contract["type"] == BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_TYPE
    assert "buses" in artifact_contract["required_top_level_fields"]
    assert "mains" in artifact_contract["required_top_level_fields"]
    assert "dcas" in artifact_contract["required_top_level_fields"]
    assert artifact_contract["required_bus_control_fields"] == [
        "fader_db",
        "mute",
        "pan",
        "width",
        "mono",
    ]
    assert artifact_contract["required_dca_assignment_fields"] == [
        "member_channels",
        "member_buses",
    ]
    assert "true_peak_dbtp" in artifact_contract["required_metering_fields"]
    assert "cache_matches_physical_readback" in artifact_contract[
        "required_bus_coverage_flags"
    ]
    assert "physical_readback" in artifact_contract["allowed_sources"]
    assert "dca_write_events" in artifact_contract["forbidden_mutation_evidence_fields"]
    assert contract["summary"]["physical_readback_surface_id"] == BUS_GROUP_MAIN_DCA_SURFACE_ID
    assert contract["summary"]["physical_fixture_gap_remains"] is True
    assert contract["summary"]["live_authority_granted"] is False
    assert "bus_group_matrix" in contract["summary"]["promotion_families_requiring_surface"]
    assert "master_bus" in contract["summary"]["promotion_families_requiring_surface"]


def test_bus_group_main_dca_contract_validates_minimal_physical_artifact():
    result = validate_bus_group_main_dca_physical_readback_artifact(_valid_artifact())

    assert result["status"] == "valid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is True
    assert result["live_authority_granted"] is False
    assert result["full_autonomous_ready"] is False
    assert result["bus_count"] == 1
    assert result["main_count"] == 1
    assert result["dca_count"] == 1
    assert result["violations"] == []
    assert result["evidence"]["source"] == "physical_readback"


def test_bus_group_main_dca_contract_rejects_live_or_cached_claims():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["source"] = "mixer_state_cache"
    artifact["full_autonomous_ready"] = True
    artifact["dca_write_events"] = []
    artifact["safety"]["live_write"] = True
    artifact["buses"][0]["status"] = "verified_from_cache"
    artifact["buses"][0]["source"] = "mixer_state_cache"
    artifact["buses"][0]["controls"]["mute"] = "false"
    artifact["buses"][0]["coverage"]["cache_matches_physical_readback"] = False

    result = validate_bus_group_main_dca_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is False
    assert result["live_authority_granted"] is False
    assert result["full_autonomous_ready"] is False
    assert "source_not_physical_readback" in result["violations"]
    assert "top_level_full_autonomous_ready_not_false" in result["violations"]
    assert "safety_live_write_not_false" in result["violations"]
    assert "mutation_evidence_field_present:dca_write_events" in result["violations"]
    assert "bus_0_status_not_physical_verified" in result["violations"]
    assert "bus_0_source_not_physical_readback" in result["violations"]
    assert "bus_0_controls_mute_not_bool_or_0_1" in result["violations"]
    assert "bus_0_coverage_cache_matches_physical_readback_not_true" in result["violations"]


def test_bus_group_main_dca_contract_rejects_incomplete_rows():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["bus_count"] = 2
    artifact["dcas"][0]["dca_id"] = 0
    artifact["buses"][0]["assignments"]["dca_assignments"] = ["1"]
    del artifact["mains"][0]["metering"]["true_peak_dbtp"]
    del artifact["dcas"][0]["assignments"]["member_buses"]

    result = validate_bus_group_main_dca_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert "bus_count_mismatch" in result["violations"]
    assert "bus_0_assignments_dca_assignments_not_int_list" in result["violations"]
    assert "main_0_metering_missing_field:true_peak_dbtp" in result["violations"]
    assert "dca_0_dca_id_invalid" in result["violations"]
    assert "dca_0_assignments_missing_field:member_buses" in result["violations"]


def test_bus_group_main_dca_contract_is_transport_free():
    source = Path("backend/bus_group_main_dca_physical_readback_contract.py").read_text(
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
        "set_fader(",
        "set_bus_fader(",
        "set_group_fader(",
        "set_dca_fader(",
        "set_dca_level(",
        "set_main_fader(",
        "set_mute(",
        "set_bus_mute(",
        "set_dca_mute(",
        "set_group_mute(",
        "set_main_mute(",
    ]
    for token in forbidden_transport_tokens + forbidden_write_call_tokens:
        assert token not in source


def test_bus_group_main_dca_contract_is_registry_shadow_only_not_executor():
    method = get_processing_method(BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_CONTRACT_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_CONTRACT_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    contract = snapshot["bus_group_main_dca_physical_readback_contract"]
    assert contract["type"] == "bus_group_main_dca_physical_readback_contract"
    assert contract["safety"]["live_write"] is False
    assert contract["summary"]["physical_fixture_gap_remains"] is True


def test_bus_group_main_dca_contract_is_console_read_goal_support_path():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_CONTRACT_METHOD_ID in row.method_ids
    assert "backend/bus_group_main_dca_physical_readback_contract.py" in row.support_paths
    assert row.current_live_write_kinds == ()
