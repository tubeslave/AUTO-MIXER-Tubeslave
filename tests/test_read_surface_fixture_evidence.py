from pathlib import Path

from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.read_surface_fixture_evidence import (
    build_read_surface_fixture_evidence_snapshot,
)


def test_read_surface_fixture_evidence_snapshot_tracks_cached_dto_fixtures_only():
    snapshot = build_read_surface_fixture_evidence_snapshot()
    surfaces = {row["surface_id"]: row for row in snapshot["surfaces"]}

    assert snapshot["type"] == "read_surface_fixture_evidence"
    assert snapshot["status"] == "fixture_evidence_incomplete"
    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "full_autonomous_ready": False,
    }
    assert snapshot["summary"]["surface_count"] >= 10
    assert snapshot["summary"]["surface_with_fixture_count"] >= 9
    assert "channel_identity_and_basic_controls" in surfaces
    assert (
        "test_channel_inventory_publishes_cached_source_patch_readback_only"
        in surfaces["channel_identity_and_basic_controls"]["test_names"]
    )
    assert "cached_source_patch" in surfaces["channel_identity_and_basic_controls"]["covered_contracts"]
    assert "channel_eq_dynamics_gate" in surfaces
    assert surfaces["channel_eq_dynamics_gate"]["status"] == "partial_fixture_evidence"
    assert (
        "test_channel_processing_snapshot_publishes_cached_processing_readback_only"
        in surfaces["channel_eq_dynamics_gate"]["test_names"]
    )
    assert "stale_cache_detection" in surfaces["channel_eq_dynamics_gate"]["covered_contracts"]
    assert "bus_main_dca_cache_freshness" in surfaces["bus_group_main_dca_state"]["covered_contracts"]
    assert "stale_cache_detection" in surfaces["bus_group_main_dca_state"]["covered_contracts"]
    assert (
        "per_field_stale_cache_fixture_incomplete"
        not in surfaces["bus_group_main_dca_state"]["remaining_gaps"]
    )
    assert "fx_insert_cache_freshness" in surfaces["fx_slots_and_inserts"]["covered_contracts"]
    assert "fx_parameter_schema_hints" in surfaces["fx_slots_and_inserts"]["covered_contracts"]
    assert (
        "test_fx_inserts_snapshot_uses_generic_schema_for_unknown_fx_models"
        in surfaces["fx_slots_and_inserts"]["test_names"]
    )
    assert "stale_cache_detection" in surfaces["fx_slots_and_inserts"]["covered_contracts"]
    assert (
        "per_field_stale_cache_fixture_incomplete"
        not in surfaces["fx_slots_and_inserts"]["remaining_gaps"]
    )
    assert "cached_matrix_dynamics" in surfaces["matrix_state"]["covered_contracts"]
    assert "cached_matrix_sends" in surfaces["matrix_state"]["covered_contracts"]
    assert "matrix_cache_freshness" in surfaces["matrix_state"]["covered_contracts"]
    assert "stale_cache_detection" in surfaces["matrix_state"]["covered_contracts"]
    assert (
        "per_field_stale_cache_fixture_incomplete"
        not in surfaces["matrix_state"]["remaining_gaps"]
    )
    assert "cached_matrix_send_state" in surfaces["routing_and_sends"]["covered_contracts"]
    assert "send_cache_freshness" in surfaces["routing_and_sends"]["covered_contracts"]
    assert "stale_cache_detection" in surfaces["routing_and_sends"]["covered_contracts"]
    assert (
        "per_field_stale_cache_fixture_incomplete"
        not in surfaces["routing_and_sends"]["remaining_gaps"]
    )
    assert "physical_readback_fixture_missing" in surfaces["matrix_state"]["remaining_gaps"]
    assert "route_send_diff_fixture_missing" in surfaces["routing_and_sends"]["remaining_gaps"]
    assert "snapshot_scene_cache_freshness" in surfaces["snapshots_scenes"]["covered_contracts"]
    assert "stale_cache_detection" in surfaces["snapshots_scenes"]["covered_contracts"]
    assert (
        "test_snapshots_scenes_snapshot_marks_stale_cache_timestamps_without_queries"
        in surfaces["snapshots_scenes"]["test_names"]
    )
    assert (
        "per_field_stale_cache_fixture_incomplete"
        not in surfaces["snapshots_scenes"]["remaining_gaps"]
    )
    assert "scoped_diff_fixture_missing" in surfaces["snapshots_scenes"]["remaining_gaps"]
    assert "rollback_diff_fixture_missing" in surfaces["snapshots_scenes"]["remaining_gaps"]
    assert "user_preset_cache_freshness" in surfaces["user_presets"]["covered_contracts"]
    assert "stale_cache_detection" in surfaces["user_presets"]["covered_contracts"]
    assert (
        "test_user_presets_snapshot_marks_stale_cache_timestamps_without_queries"
        in surfaces["user_presets"]["test_names"]
    )
    assert (
        "per_field_stale_cache_fixture_incomplete"
        not in surfaces["user_presets"]["remaining_gaps"]
    )
    assert "scoped_diff_fixture_missing" in surfaces["user_presets"]["remaining_gaps"]
    assert "rollback_diff_fixture_missing" in surfaces["user_presets"]["remaining_gaps"]
    assert "dlive_wing_parity_mapping_missing" in surfaces["dlive_state"]["remaining_gaps"]
    assert surfaces["dlive_state"]["full_autonomous_ready"] is False


def test_read_surface_fixture_evidence_method_is_shadow_only_not_executor():
    method = get_processing_method("read_surface_fixture_evidence")

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert "read_surface_fixture_evidence" not in main.executor_method_ids


def test_processing_registry_snapshot_includes_read_fixture_evidence_without_live_claim():
    snapshot = build_processing_registry_snapshot(include_methods=False)
    evidence = snapshot["read_surface_fixture_evidence"]

    assert evidence["type"] == "read_surface_fixture_evidence"
    assert evidence["safety"]["live_write"] is False
    assert evidence["safety"]["executes_tests"] is False
    assert evidence["summary"]["claim"] == (
        "read DTO fixtures are tracked but physical readback proof is incomplete"
    )


def test_console_read_goal_references_fixture_evidence_without_widening_live_scope():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert row.status == "proposal_only"
    assert "read_surface_fixture_evidence" in row.method_ids
    assert "backend/read_surface_fixture_evidence.py" in row.support_paths
    assert row.current_live_write_kinds == ()
    assert "physical fixtures" in row.next_step


def test_read_surface_fixture_evidence_module_stays_transport_free():
    source = Path("backend/read_surface_fixture_evidence.py").read_text()

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
