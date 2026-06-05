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
    assert set(snapshot["summary"]["partial_or_missing_surface_ids"]) <= set(
        remediation["summary"]["incomplete_surface_ids"]
    )
    assert "raw_mixer_state_cache" in remediation["summary"]["physical_fixture_missing_surface_ids"]
    assert remediation["summary"]["next_physical_readback_surface_id"] == "raw_mixer_state_cache"


def test_fx_and_snapshot_read_surfaces_are_not_write_authority():
    fx = get_read_surface("fx_slots_and_inserts")
    snapshots = get_read_surface("snapshots_scenes")
    presets = get_read_surface("user_presets")
    dlive = get_read_surface("dlive_state")

    assert fx is not None
    assert fx.status == "partial"
    assert fx.read_mode == "product_snapshot"
    assert "build_fx_inserts_snapshot" in fx.entrypoints
    assert "get_fx_slot_info" in fx.entrypoints
    assert "destructive_change_marker" in fx.read_fields
    assert fx.mutates_console_state is False
    assert snapshots is not None
    assert snapshots.status == "partial"
    assert snapshots.read_mode == "product_snapshot"
    assert "build_snapshots_scenes_snapshot" in snapshots.entrypoints
    assert "find_snap_by_name" in snapshots.entrypoints
    assert "destructive_change_marker" in snapshots.read_fields
    assert "cache_freshness" in snapshots.read_fields
    assert "find_snap_by_name_must_not_load_snapshot" in snapshots.notes
    assert presets is not None
    assert presets.status == "partial"
    assert presets.read_mode == "product_snapshot"
    assert "build_user_presets_snapshot" in presets.entrypoints
    assert "destructive_recall_marker" in presets.read_fields
    assert "cache_freshness" in presets.read_fields
    assert dlive is not None
    assert dlive.status == "partial"
    assert dlive.read_mode == "product_snapshot"
    assert "build_dlive_state_snapshot" in dlive.entrypoints
    assert "get_dlive_state_snapshot" in dlive.entrypoints
    assert "wing_parity_status" in dlive.read_fields
    assert "parity" in dlive.blocker.lower()


def test_fx_insert_read_surface_tracks_cache_freshness_without_write_authority():
    fx = get_read_surface("fx_slots_and_inserts")

    assert fx is not None
    assert fx.status == "partial"
    assert "cache_freshness" in fx.read_fields
    assert "fx_parameter_schema" in fx.read_fields
    assert "cache freshness" in fx.current_coverage
    assert "semantic schema hints" in fx.current_coverage
    assert "stale-cache timestamps" not in fx.blocker
    assert "complete model-specific FX parameter schemas" in fx.blocker
    assert "FX safety review" in fx.blocker
    assert fx.mutates_console_state is False


def test_bus_main_dca_read_surface_tracks_cache_freshness_without_write_authority():
    surface = get_read_surface("bus_group_main_dca_state")

    assert surface is not None
    assert surface.status == "partial"
    assert "cache_freshness" in surface.read_fields
    assert "cache freshness" in surface.current_coverage
    assert "stale-cache timestamps" not in surface.blocker
    assert "full group membership" in surface.blocker
    assert "bus/main/DCA safety review" in surface.blocker
    assert surface.mutates_console_state is False


def test_matrix_read_surface_tracks_cached_dynamics_without_write_authority():
    matrix = get_read_surface("matrix_state")

    assert matrix is not None
    assert matrix.status == "partial"
    assert "matrix_dynamics" in matrix.read_fields
    assert "dynamics" in matrix.current_coverage
    assert "dynamics" not in matrix.blocker
    assert matrix.mutates_console_state is False


def test_matrix_read_surface_tracks_cached_matrix_sends_without_write_authority():
    matrix = get_read_surface("matrix_state")
    routing = get_read_surface("routing_and_sends")

    assert matrix is not None
    assert "matrix_send_state" in matrix.read_fields
    assert "channel-to-matrix send rows" in matrix.current_coverage
    assert "matrix sends" not in matrix.blocker
    assert "route diff proof" in matrix.blocker
    assert matrix.mutates_console_state is False
    assert routing is not None
    assert "channel_matrix_send_on" in routing.read_fields
    assert "channel_matrix_send_level" in routing.read_fields
    assert "channel matrix sends" in routing.current_coverage
    assert routing.mutates_console_state is False


def test_matrix_read_surface_tracks_cache_freshness_without_write_authority():
    matrix = get_read_surface("matrix_state")

    assert matrix is not None
    assert "cache_freshness" in matrix.read_fields
    assert "cache freshness" in matrix.current_coverage
    assert "stale-cache timestamps" not in matrix.blocker
    assert "matrix safety review" in matrix.blocker
    assert matrix.mutates_console_state is False


def test_routing_send_read_surface_tracks_cache_freshness_without_write_authority():
    routing = get_read_surface("routing_and_sends")

    assert routing is not None
    assert routing.status == "partial"
    assert "cache_freshness" in routing.read_fields
    assert "cache freshness" in routing.current_coverage
    assert "stale-cache timestamps" not in routing.blocker
    assert "route/send diffing" in routing.blocker
    assert routing.mutates_console_state is False


def test_scene_preset_read_surfaces_track_cache_freshness_without_write_authority():
    snapshots = get_read_surface("snapshots_scenes")
    presets = get_read_surface("user_presets")

    assert snapshots is not None
    assert snapshots.status == "partial"
    assert "cache_freshness" in snapshots.read_fields
    assert "cache freshness" in snapshots.current_coverage
    assert "stale-cache timestamps" not in snapshots.blocker
    assert "rollback proof" in snapshots.blocker
    assert snapshots.mutates_console_state is False

    assert presets is not None
    assert presets.status == "partial"
    assert "cache_freshness" in presets.read_fields
    assert "cache freshness" in presets.current_coverage
    assert "stale-cache timestamps" not in presets.blocker
    assert "scoped diff" in presets.blocker
    assert presets.mutates_console_state is False


def test_project_goal_console_read_capability_references_inventory():
    capability = get_project_goal_capability("console_read_parameters")

    assert capability is not None
    assert "backend/read_surface_inventory.py" in capability.support_paths
    assert capability.status == "proposal_only"
    assert "partial" in capability.blocker


def test_channel_identity_read_surface_tracks_source_patch_without_write_authority():
    surface = get_read_surface("channel_identity_and_basic_controls")

    assert surface is not None
    assert "source_patch" in surface.read_fields
    assert "fallback_source" in surface.read_fields
    assert "source patch" in surface.current_coverage
    assert "physical source-patch readback" in surface.blocker
    assert surface.mutates_console_state is False


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
