import ast
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


ROOT = Path(__file__).resolve().parents[1]


def _declared_test_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


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
    assert snapshot["summary"]["next_physical_readback_surface_id"] == (
        snapshot["summary"]["physical_fixture_missing_surface_ids"][0]
    )
    assert "channel_identity_and_basic_controls" in surfaces
    assert (
        "physical_readback_artifact_contract"
        in surfaces["raw_mixer_state_cache"]["covered_contracts"]
    )
    assert (
        "test_raw_mixer_state_contract_validates_minimal_physical_artifact"
        in surfaces["raw_mixer_state_cache"]["test_names"]
    )
    assert "physical_readback_fixture" not in surfaces["raw_mixer_state_cache"]["covered_contracts"]
    assert "physical_readback_fixture_missing" in surfaces["raw_mixer_state_cache"]["remaining_gaps"]
    assert (
        "test_channel_inventory_publishes_cached_source_patch_readback_only"
        in surfaces["channel_identity_and_basic_controls"]["test_names"]
    )
    assert "cached_source_patch" in surfaces["channel_identity_and_basic_controls"]["covered_contracts"]
    assert (
        "test_channel_basic_controls_contract_validates_minimal_physical_artifact"
        in surfaces["channel_identity_and_basic_controls"]["test_names"]
    )
    assert (
        "physical_readback_artifact_contract"
        in surfaces["channel_identity_and_basic_controls"]["covered_contracts"]
    )
    assert (
        "physical_readback_fixture"
        not in surfaces["channel_identity_and_basic_controls"]["covered_contracts"]
    )
    assert "channel_eq_dynamics_gate" in surfaces
    assert surfaces["channel_eq_dynamics_gate"]["status"] == "partial_fixture_evidence"
    assert (
        "test_channel_processing_snapshot_publishes_cached_processing_readback_only"
        in surfaces["channel_eq_dynamics_gate"]["test_names"]
    )
    assert (
        "test_channel_processing_contract_validates_minimal_physical_artifact"
        in surfaces["channel_eq_dynamics_gate"]["test_names"]
    )
    assert "stale_cache_detection" in surfaces["channel_eq_dynamics_gate"]["covered_contracts"]
    assert (
        "physical_readback_artifact_contract"
        in surfaces["channel_eq_dynamics_gate"]["covered_contracts"]
    )
    assert "physical_readback_fixture" not in surfaces["channel_eq_dynamics_gate"]["covered_contracts"]
    assert "bus_main_dca_cache_freshness" in surfaces["bus_group_main_dca_state"]["covered_contracts"]
    assert "stale_cache_detection" in surfaces["bus_group_main_dca_state"]["covered_contracts"]
    assert (
        "physical_readback_artifact_contract"
        in surfaces["bus_group_main_dca_state"]["covered_contracts"]
    )
    assert (
        "test_bus_group_main_dca_contract_validates_minimal_physical_artifact"
        in surfaces["bus_group_main_dca_state"]["test_names"]
    )
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
    assert (
        "test_fx_slots_inserts_contract_validates_minimal_physical_artifact"
        in surfaces["fx_slots_and_inserts"]["test_names"]
    )
    assert (
        "physical_readback_artifact_contract"
        in surfaces["fx_slots_and_inserts"]["covered_contracts"]
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
        "test_routing_sends_contract_validates_minimal_physical_artifact"
        in surfaces["routing_and_sends"]["test_names"]
    )
    assert (
        "physical_readback_artifact_contract"
        in surfaces["routing_and_sends"]["covered_contracts"]
    )
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
    assert "raw_mixer_state_physical_readback_contract" in row.method_ids
    assert "channel_basic_controls_physical_readback_contract" in row.method_ids
    assert "channel_processing_physical_readback_contract" in row.method_ids
    assert "bus_group_main_dca_physical_readback_contract" in row.method_ids
    assert "matrix_state_physical_readback_contract" in row.method_ids
    assert "fx_slots_inserts_physical_readback_contract" in row.method_ids
    assert "routing_sends_physical_readback_contract" in row.method_ids
    assert "backend/read_surface_fixture_evidence.py" in row.support_paths
    assert "backend/raw_mixer_state_physical_readback_contract.py" in row.support_paths
    assert "backend/channel_basic_controls_physical_readback_contract.py" in row.support_paths
    assert "backend/channel_processing_physical_readback_contract.py" in row.support_paths
    assert "backend/bus_group_main_dca_physical_readback_contract.py" in row.support_paths
    assert "backend/matrix_state_physical_readback_contract.py" in row.support_paths
    assert "backend/fx_slots_inserts_physical_readback_contract.py" in row.support_paths
    assert "backend/routing_sends_physical_readback_contract.py" in row.support_paths
    assert row.current_live_write_kinds == ()
    assert "physical fixtures" in row.next_step


def test_read_surface_fixture_evidence_declares_existing_fixture_tests():
    snapshot = build_read_surface_fixture_evidence_snapshot()

    for surface in snapshot["surfaces"]:
        surface_id = surface["surface_id"]
        fixture_paths = surface["fixture_paths"]
        test_names = surface["test_names"]
        if surface["status"] == "missing_fixture_evidence":
            assert fixture_paths == []
            assert test_names == []
            continue

        assert fixture_paths, surface_id
        assert test_names, surface_id
        available_tests_by_path = {}
        for relative_path in fixture_paths:
            assert relative_path.startswith("tests/"), (surface_id, relative_path)
            fixture_path = ROOT / relative_path
            assert fixture_path.exists(), (surface_id, relative_path)
            available_tests_by_path[relative_path] = _declared_test_functions(fixture_path)

        missing_tests = [
            name
            for name in test_names
            if not any(name in names for names in available_tests_by_path.values())
        ]
        assert missing_tests == [], (surface_id, missing_tests)


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
