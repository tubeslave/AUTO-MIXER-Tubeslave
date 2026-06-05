from pathlib import Path

from backend.read_surface_inventory import ReadSurface
from backend.read_surface_remediation_plan import build_read_surface_remediation_plan


def test_read_surface_remediation_plan_module_stays_transport_free():
    source = Path("backend/read_surface_remediation_plan.py").read_text(encoding="utf-8")

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
    ]
    for token in forbidden:
        assert token not in source


def test_plan_creates_actions_for_current_partial_and_missing_read_surfaces():
    plan = build_read_surface_remediation_plan()

    assert plan["type"] == "read_surface_remediation_plan"
    assert plan["status"] == "read_remediation_required"
    assert plan["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "live_widening_allowed": False,
        "full_autonomous_ready": False,
    }
    assert plan["summary"]["incomplete_surface_count"] == len(plan["actions"])
    assert "channel_eq_dynamics_gate" in plan["summary"]["incomplete_surface_ids"]
    assert "user_presets" in plan["summary"]["incomplete_surface_ids"]
    assert "raw_mixer_state_cache" in plan["summary"]["incomplete_surface_ids"]
    assert plan["summary"]["next_physical_readback_surface_id"] == "raw_mixer_state_cache"
    assert plan["summary"]["next_physical_readback_action_id"] == "remediate_raw_mixer_state_cache"
    assert plan["summary"]["physical_fixture_missing_surface_ids"][0] == "raw_mixer_state_cache"
    assert plan["summary"]["status_counts"]["partial"] >= 5
    assert plan["summary"]["status_counts"]["missing"] == 0
    assert "full_console_read_coverage_incomplete" in plan["summary"]["global_blockers"]
    assert "physical_readback_fixtures_incomplete" in plan["summary"]["global_blockers"]

    actions = {action["surface_id"]: action for action in plan["actions"]}
    raw_cache = actions["raw_mixer_state_cache"]
    assert raw_cache["action_type"] == "add_physical_readback_fixture"
    assert raw_cache["status"] == "cached"
    assert raw_cache["physical_readback_gap"] is True
    assert raw_cache["next_physical_readback_target"] is True
    assert "physical_readback_fixture" in raw_cache["missing_contracts"]
    assert "physical_readback_fixture_missing" in raw_cache["fixture_remaining_gaps"]
    assert any(
        "raw_mixer_state_physical_readback_contract" in artifact
        for artifact in raw_cache["expected_artifacts"]
    )
    assert raw_cache["mutates_console_state"] is False

    channel_processing = actions["channel_eq_dynamics_gate"]
    assert channel_processing["action_type"] == "complete_read_dto"
    assert "physical_query_readback_fixture" in channel_processing["missing_contracts"]
    assert channel_processing["physical_readback_gap"] is True
    assert channel_processing["next_physical_readback_target"] is False
    assert any(
        "channel_processing_physical_readback_contract" in artifact
        for artifact in channel_processing["expected_artifacts"]
    )
    assert channel_processing["mutates_console_state"] is False

    channel_basic = actions["channel_identity_and_basic_controls"]
    assert channel_basic["action_type"] == "complete_read_dto"
    assert channel_basic["physical_readback_gap"] is True
    assert channel_basic["next_physical_readback_target"] is False
    assert "physical_readback_fixture" in channel_basic["missing_contracts"]
    assert any(
        "channel_basic_controls_physical_readback_contract" in artifact
        for artifact in channel_basic["expected_artifacts"]
    )
    assert channel_basic["mutates_console_state"] is False

    bus_group_main_dca = actions["bus_group_main_dca_state"]
    assert bus_group_main_dca["action_type"] == "complete_read_dto"
    assert bus_group_main_dca["physical_readback_gap"] is True
    assert bus_group_main_dca["next_physical_readback_target"] is False
    assert "physical_readback_fixture" in bus_group_main_dca["missing_contracts"]
    assert any(
        "bus_group_main_dca_physical_readback_contract" in artifact
        for artifact in bus_group_main_dca["expected_artifacts"]
    )
    assert bus_group_main_dca["mutates_console_state"] is False

    matrix_state = actions["matrix_state"]
    assert matrix_state["action_type"] == "complete_read_dto"
    assert matrix_state["physical_readback_gap"] is True
    assert matrix_state["next_physical_readback_target"] is False
    assert "physical_readback_fixture" in matrix_state["missing_contracts"]
    assert any(
        "matrix_state_physical_readback_contract" in artifact
        for artifact in matrix_state["expected_artifacts"]
    )
    assert matrix_state["mutates_console_state"] is False

    presets = actions["user_presets"]
    assert presets["action_type"] == "complete_read_dto"
    assert "enumeration_contract" in presets["missing_contracts"]
    assert "destructive_change_diff_marker" in presets["missing_contracts"]
    assert presets["requires_connection"] is False


def test_complete_surfaces_return_complete_status_without_actions():
    surface = ReadSurface(
        id="complete_fx",
        label="Complete FX",
        target_console_surface="fx",
        status="queryable",
        read_mode="osc_read_query",
        paths=("backend/wing_client.py",),
        entrypoints=("get_fx_overview",),
        read_fields=("fx_model",),
        current_coverage="Complete fixture row.",
    )

    plan = build_read_surface_remediation_plan(
        [surface],
        fixture_evidence_snapshot={
            "summary": {
                "physical_fixture_missing_surface_ids": [],
                "next_physical_readback_surface_id": None,
            },
            "surfaces": [
                {
                    "surface_id": "complete_fx",
                    "status": "fixture_evidence_present",
                    "remaining_gaps": [],
                }
            ],
        },
    )

    assert plan["status"] == "read_coverage_complete"
    assert plan["summary"]["incomplete_surface_count"] == 0
    assert plan["summary"]["physical_fixture_missing_surface_ids"] == []
    assert plan["summary"]["next_physical_readback_surface_id"] is None
    assert plan["summary"]["next_physical_readback_action_id"] is None
    assert plan["summary"]["global_blockers"] == []
    assert plan["actions"] == []
