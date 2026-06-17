from pathlib import Path

from backend.processing_registry import build_processing_registry_snapshot, get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.routing_group_read_map import build_routing_group_read_map


def _fresh(fields=4):
    return {
        "status": "fresh",
        "field_count": fields,
        "fresh_field_count": fields,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def test_routing_group_read_map_module_is_transport_free():
    source = Path("backend/routing_group_read_map.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "route_output",
        "route_multiple_outputs",
        "set_channel_input",
        "set_matrix",
        "set_dca",
        "mixer_client",
    ]
    for token in forbidden:
        assert token not in source


def test_fresh_rows_build_read_only_routing_group_topology_map():
    read_map = build_routing_group_read_map(
        surface_freshness={
            "routing_and_sends": _fresh(),
            "bus_group_main_dca_state": _fresh(),
            "matrix_state": _fresh(),
        },
        routing_rows=[
            {
                "route_id": "out:MOD:1",
                "route_kind": "output_patch",
                "source_group": "CH",
                "source_channel": 1,
                "output_group": "MOD",
                "output_number": 1,
                "destructive_change_marker": "output_routing_write_would_repatch_console",
                "cache_freshness": _fresh(),
            }
        ],
        bus_group_rows=[
            {
                "bus_id": "dca:1",
                "bus_kind": "dca",
                "label": "Vocals",
                "members": ["CH:1", "CH:2"],
                "fader_db": -3.0,
                "cache_freshness": _fresh(),
            }
        ],
        matrix_rows=[
            {
                "matrix_id": "matrix:1",
                "sources": [{"source_group": "BUS", "source_channel": 1, "level_db": -6.0}],
                "cache_freshness": _fresh(),
            }
        ],
    )

    assert read_map["status"] == "read_map_ready"
    assert read_map["safety"]["read_only"] is True
    assert read_map["safety"]["live_write"] is False
    assert read_map["safety"]["mutates_console_state"] is False
    assert read_map["summary"]["blockers"] == []
    assert read_map["summary"]["route_count"] == 1
    assert read_map["summary"]["bus_group_count"] == 1
    assert read_map["summary"]["matrix_count"] == 1
    assert read_map["summary"]["edge_count"] == 4
    assert read_map["topology"]["routing_rows"][0]["destructive_change_marker_present"] is True
    assert {"kind": "route", "from": "CH:1", "to": "MOD:1", "surface_id": "routing_and_sends"} in read_map["topology"]["edges"]


def test_missing_or_stale_surfaces_block_read_map_claim():
    read_map = build_routing_group_read_map(
        surface_freshness={
            "routing_and_sends": {"status": "stale", "read_only": True},
            "bus_group_main_dca_state": _fresh(),
        },
        routing_rows=[
            {
                "route_id": "out:MOD:1",
                "source_group": "CH",
                "source_channel": 1,
                "output_group": "MOD",
                "output_number": 1,
            }
        ],
        bus_group_rows=[
            {
                "bus_id": "dca:1",
                "bus_kind": "dca",
                "members": [],
            }
        ],
    )

    blockers = read_map["summary"]["blockers"]
    assert read_map["status"] == "read_map_incomplete"
    assert "surface_not_fresh:routing_and_sends:stale" in blockers
    assert "surface_snapshot_missing:matrix_state" in blockers
    assert "routing_row_0_destructive_change_marker_missing" in blockers
    assert "bus_group_row_0_membership_missing" in blockers
    assert read_map["safety"]["queries_console"] is False


def test_stale_row_cache_blocks_routing_group_ready_claim():
    read_map = build_routing_group_read_map(
        surface_freshness={
            "routing_and_sends": _fresh(),
            "bus_group_main_dca_state": _fresh(),
            "matrix_state": _fresh(),
        },
        routing_rows=[
            {
                "route_id": "out:MOD:1",
                "source_group": "CH",
                "source_channel": 1,
                "output_group": "MOD",
                "output_number": 1,
                "destructive_change_marker": "output_routing_write_would_repatch_console",
                "cache_freshness": {"status": "stale", "read_only": True},
            }
        ],
        bus_group_rows=[
            {
                "bus_id": "dca:1",
                "bus_kind": "dca",
                "members": ["CH:1"],
                "cache_freshness": _fresh(),
            }
        ],
        matrix_rows=[
            {
                "matrix_id": "matrix:1",
                "sources": ["BUS:1"],
                "cache_freshness": _fresh(),
            }
        ],
    )

    assert read_map["status"] == "read_map_incomplete"
    assert "routing_row_0_cache_not_fresh:stale" in read_map["summary"]["blockers"]
    assert read_map["safety"]["live_write"] is False


def test_default_registry_read_map_is_incomplete_and_non_authoritative():
    snapshot = build_processing_registry_snapshot(include_methods=False)
    read_map = snapshot["routing_group_read_map"]

    assert read_map["type"] == "routing_group_read_map"
    assert read_map["status"] == "read_map_incomplete"
    assert read_map["safety"]["live_write"] is False
    assert "routing_group_read_map_empty" in read_map["summary"]["blockers"]


def test_routing_group_read_map_is_registered_without_live_widening():
    method = get_processing_method("routing_group_read_map")
    console_read = get_project_goal_capability("console_read_parameters")
    routing = get_project_goal_capability("console_routing_control")
    groups = get_project_goal_capability("console_dca_group_matrix_control")

    assert method is not None
    assert method.status == "shadow_only"
    assert method.can_write_live is False
    assert console_read is not None and "routing_group_read_map" in console_read.method_ids
    assert routing is not None and "routing_group_read_map" in routing.method_ids
    assert groups is not None and "routing_group_read_map" in groups.method_ids
