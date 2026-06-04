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
    assert plan["summary"]["status_counts"]["partial"] >= 5
    assert plan["summary"]["status_counts"]["missing"] == 0
    assert "full_console_read_coverage_incomplete" in plan["summary"]["global_blockers"]

    actions = {action["surface_id"]: action for action in plan["actions"]}
    channel_processing = actions["channel_eq_dynamics_gate"]
    assert channel_processing["action_type"] == "complete_read_dto"
    assert "physical_query_readback_fixture" in channel_processing["missing_contracts"]
    assert channel_processing["mutates_console_state"] is False

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

    plan = build_read_surface_remediation_plan([surface])

    assert plan["status"] == "read_coverage_complete"
    assert plan["summary"]["incomplete_surface_count"] == 0
    assert plan["summary"]["global_blockers"] == []
    assert plan["actions"] == []
