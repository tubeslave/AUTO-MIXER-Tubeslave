from pathlib import Path

from backend.processing_registry import build_processing_registry_snapshot, get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.scene_preset_read_inventory import build_scene_preset_read_inventory


def _fresh(fields=6):
    return {
        "status": "fresh",
        "field_count": fields,
        "fresh_field_count": fields,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def test_scene_preset_read_inventory_module_is_transport_free():
    source = Path("backend/scene_preset_read_inventory.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "mixer_client",
        "load_snap",
        "save_snap",
        "load_snap_by_index",
        "recall_scene",
        "store_preset",
        "/snap/load",
        "/snap/store",
        "/$ctl/lib/$action",
    ]
    for token in forbidden:
        assert token not in source


def test_fresh_rows_build_read_only_scene_preset_inventory():
    inventory = build_scene_preset_read_inventory(
        surface_freshness={
            "snapshots_scenes": _fresh(),
            "user_presets": _fresh(),
        },
        scene_rows=[
            {
                "snapshot_id": "scene:12",
                "snapshot_index": 12,
                "snapshot_name": "Band A vocal strip",
                "active_snapshot_name": "Pre-show baseline",
                "current_show": "summer_show",
                "scene_load_policy": "manual_operator_only",
                "scene_save_policy": "manual_operator_only",
                "destructive_change_marker": "scene_load_would_change_console_state",
                "cache_freshness": _fresh(),
            }
        ],
        preset_rows=[
            {
                "preset_id": "preset:vocal_a",
                "preset_index": 4,
                "preset_name": "Vocal A",
                "preset_scope": "channel_strip",
                "target_type": "channel",
                "target_label": "Lead Vox",
                "preset_recall_policy": "manual_operator_only",
                "preset_store_policy": "manual_operator_only",
                "destructive_recall_marker": "preset_recall_would_change_channel_strip",
                "cache_freshness": _fresh(),
            }
        ],
    )

    assert inventory["status"] == "read_inventory_ready"
    assert inventory["safety"]["read_only"] is True
    assert inventory["safety"]["live_write"] is False
    assert inventory["safety"]["mutates_console_state"] is False
    assert inventory["safety"]["queries_console"] is False
    assert inventory["safety"]["scene_load_allowed"] is False
    assert inventory["safety"]["preset_recall_allowed"] is False
    assert inventory["summary"]["blockers"] == []
    assert inventory["summary"]["scene_count"] == 1
    assert inventory["summary"]["preset_count"] == 1
    assert inventory["inventory"]["scenes"][0]["destructive_change_marker_present"] is True
    assert inventory["inventory"]["presets"][0]["preset_scope"] == "channel_strip"
    assert inventory["inventory"]["presets"][0]["destructive_recall_marker_present"] is True


def test_missing_or_stale_surfaces_and_policy_fields_block_inventory_claim():
    inventory = build_scene_preset_read_inventory(
        surface_freshness={
            "snapshots_scenes": {"status": "stale", "read_only": True},
        },
        scene_rows=[
            {
                "snapshot_id": "scene:12",
                "snapshot_name": "Band A vocal strip",
            }
        ],
        preset_rows=[
            {
                "preset_id": "preset:vocal_a",
                "preset_name": "Vocal A",
            }
        ],
    )

    blockers = inventory["summary"]["blockers"]
    assert inventory["status"] == "read_inventory_incomplete"
    assert "surface_not_fresh:snapshots_scenes:stale" in blockers
    assert "surface_snapshot_missing:user_presets" in blockers
    assert "scene_row_0_destructive_change_marker_missing" in blockers
    assert "scene_row_0_load_policy_missing" in blockers
    assert "scene_row_0_save_policy_missing" in blockers
    assert "preset_row_0_scope_missing" in blockers
    assert "preset_row_0_destructive_recall_marker_missing" in blockers
    assert "preset_row_0_recall_policy_missing" in blockers
    assert "preset_row_0_store_policy_missing" in blockers
    assert inventory["safety"]["queue_mutation"] is False


def test_stale_row_cache_blocks_scene_preset_ready_claim():
    inventory = build_scene_preset_read_inventory(
        surface_freshness={
            "snapshots_scenes": _fresh(),
            "user_presets": _fresh(),
        },
        scene_rows=[
            {
                "snapshot_id": "scene:12",
                "snapshot_name": "Band A vocal strip",
                "scene_load_policy": "manual_operator_only",
                "scene_save_policy": "manual_operator_only",
                "destructive_change_marker": "scene_load_would_change_console_state",
                "cache_freshness": {"status": "stale", "read_only": True},
            }
        ],
        preset_rows=[
            {
                "preset_id": "preset:vocal_a",
                "preset_name": "Vocal A",
                "preset_scope": "channel_strip",
                "preset_recall_policy": "manual_operator_only",
                "preset_store_policy": "manual_operator_only",
                "destructive_recall_marker": "preset_recall_would_change_channel_strip",
                "cache_freshness": _fresh(),
            }
        ],
    )

    assert inventory["status"] == "read_inventory_incomplete"
    assert "scene_row_0_cache_not_fresh:stale" in inventory["summary"]["blockers"]
    assert inventory["safety"]["preset_recall_allowed"] is False


def test_default_registry_scene_preset_inventory_is_incomplete_and_non_authoritative():
    snapshot = build_processing_registry_snapshot(include_methods=False)
    inventory = snapshot["scene_preset_read_inventory"]

    assert inventory["type"] == "scene_preset_read_inventory"
    assert inventory["status"] == "read_inventory_incomplete"
    assert inventory["safety"]["live_write"] is False
    assert inventory["safety"]["scene_load_allowed"] is False
    assert "scene_preset_read_inventory_empty" in inventory["summary"]["blockers"]


def test_scene_preset_read_inventory_is_registered_without_live_widening():
    method = get_processing_method("scene_preset_read_inventory")
    console_read = get_project_goal_capability("console_read_parameters")
    scene_preset = get_project_goal_capability("console_scene_preset_control")

    assert method is not None
    assert method.status == "shadow_only"
    assert method.can_write_live is False
    assert console_read is not None and "scene_preset_read_inventory" in console_read.method_ids
    assert scene_preset is not None and "scene_preset_read_inventory" in scene_preset.method_ids
    assert "backend/scene_preset_read_inventory.py" in scene_preset.support_paths
    assert scene_preset.current_live_write_kinds == ()
