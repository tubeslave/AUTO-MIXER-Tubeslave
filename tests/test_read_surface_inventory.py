from pathlib import Path

from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.read_surface_inventory import (
    READ_MODES,
    READ_SURFACE_STATUSES,
    build_read_surface_inventory_snapshot,
    get_read_surface,
    iter_read_surfaces,
)
from backend.read_surface_remediation_plan import build_read_surface_remediation_plan


def test_read_surface_inventory_is_read_only_and_unique():
    rows = list(iter_read_surfaces())
    ids = [row.id for row in rows]

    assert rows
    assert len(ids) == len(set(ids))
    for row in rows:
        assert row.status in READ_SURFACE_STATUSES
        assert row.read_mode in READ_MODES
        assert row.paths
        assert row.entrypoints
        assert row.mutates_console_state is False


def test_inventory_covers_goal_console_read_surfaces():
    surface_ids = {row.id for row in iter_read_surfaces()}

    assert {
        "raw_mixer_state_cache",
        "channel_identity_and_basic_controls",
        "channel_eq_dynamics_gate",
        "bus_group_main_dca_state",
        "matrix_state",
        "fx_slots_and_inserts",
        "routing_and_sends",
        "snapshots_scenes",
        "user_presets",
        "dlive_state",
    } <= surface_ids


def test_partial_and_missing_surfaces_block_full_readiness_claim():
    snapshot = build_read_surface_inventory_snapshot()
    remediation = build_read_surface_remediation_plan()

    assert snapshot["type"] == "read_surface_inventory"
    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
    }
    assert snapshot["summary"]["claim"] == "read coverage is inventoried but not complete"
    assert "user_presets" in snapshot["summary"]["partial_or_missing_surface_ids"]
    assert "matrix_state" in snapshot["summary"]["partial_or_missing_surface_ids"]
    assert snapshot["summary"]["status_counts"]["missing"] == 0
    assert snapshot["summary"]["status_counts"]["partial"] >= 6
    assert remediation["status"] == "read_remediation_required"
    assert remediation["safety"]["live_write"] is False
    assert set(snapshot["summary"]["partial_or_missing_surface_ids"]) == set(
        remediation["summary"]["incomplete_surface_ids"]
    )


def test_fx_and_snapshot_read_surfaces_are_not_write_authority():
    fx = get_read_surface("fx_slots_and_inserts")
    snapshots = get_read_surface("snapshots_scenes")
    presets = get_read_surface("user_presets")

    assert fx is not None
    assert fx.status == "queryable"
    assert "get_fx_slot_info" in fx.entrypoints
    assert fx.mutates_console_state is False
    assert snapshots is not None
    assert snapshots.status == "partial"
    assert snapshots.read_mode == "product_snapshot"
    assert "build_snapshots_scenes_snapshot" in snapshots.entrypoints
    assert "find_snap_by_name" in snapshots.entrypoints
    assert "destructive_change_marker" in snapshots.read_fields
    assert "find_snap_by_name_must_not_load_snapshot" in snapshots.notes
    assert presets is not None
    assert presets.status == "partial"
    assert presets.read_mode == "product_snapshot"
    assert "build_user_presets_snapshot" in presets.entrypoints
    assert "destructive_recall_marker" in presets.read_fields


def test_project_goal_console_read_capability_references_inventory():
    capability = get_project_goal_capability("console_read_parameters")

    assert capability is not None
    assert "backend/read_surface_inventory.py" in capability.support_paths
    assert capability.status == "proposal_only"
    assert "partial" in capability.blocker


def test_read_surface_inventory_module_stays_transport_free():
    source = Path("backend/read_surface_inventory.py").read_text()

    forbidden = [
        "WingClient",
        "DLiveClient",
        "EnhancedOSCClient",
        ".send(",
        "socket",
        "websockets",
        "set_fader",
        "set_gain",
        "set_eq",
        "set_compressor",
    ]
    for token in forbidden:
        assert token not in source
