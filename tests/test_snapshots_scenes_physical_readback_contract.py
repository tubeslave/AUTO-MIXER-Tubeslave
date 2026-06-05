from pathlib import Path

from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.snapshots_scenes_physical_readback_contract import (
    SNAPSHOTS_SCENES_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    build_snapshots_scenes_physical_readback_contract,
    validate_snapshots_scenes_physical_readback_artifact,
)


def valid_snapshots_scenes_artifact():
    return {
        "type": "snapshots_scenes_physical_readback",
        "schema_id": "snapshots_scenes_physical_readback.v1",
        "surface_id": "snapshots_scenes",
        "remediation_action_id": "remediate_snapshots_scenes",
        "captured_at": "2026-06-05T00:00:00Z",
        "source": "wing_physical_readback",
        "console_model": "WING Rack",
        "snapshot_count": 1,
        "active_scene": {
            "active_snapshot_index": 1,
            "active_snapshot_name": "Show Start",
            "current_show": "Festival A",
            "readback_status": "verified_from_physical_readback",
            "readback_source": "wing_physical_readback",
            "captured_at": "2026-06-05T00:00:00Z",
            "coverage": {
                "catalog_identity": True,
                "active_scene": True,
                "scene_content_fingerprint": True,
                "scoped_diff_marker": True,
                "rollback_reference": True,
                "cache_matches_physical_readback": True,
            },
        },
        "snapshots": [
            {
                "snapshot_index": 1,
                "snapshot_name": "Show Start",
                "readback_status": "verified_from_physical_readback",
                "readback_source": "wing_physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "active": True,
                "scene_load_policy": "blocked_manual_only",
                "scene_save_policy": "blocked_manual_only",
                "destructive_change_marker": "scene_recall_would_overwrite_console_state",
                "destructive_persistence_marker": "scene_save_would_overwrite_or_create_console_state",
                "scene_content_fingerprint": "sha256:scene-content",
                "scoped_diff_marker": "diff:scene-1-before-after",
                "rollback_reference": "rollback:scene-1",
                "coverage": {
                    "catalog_identity": True,
                    "active_scene": True,
                    "scene_content_fingerprint": True,
                    "scoped_diff_marker": True,
                    "rollback_reference": True,
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
            "scene_load_allowed": False,
            "scene_save_allowed": False,
        },
    }


def test_snapshots_scenes_contract_describes_future_physical_artifact():
    contract = build_snapshots_scenes_physical_readback_contract()

    assert contract["type"] == "snapshots_scenes_physical_readback_contract"
    assert contract["status"] == "contract_ready_physical_fixture_missing"
    assert contract["surface_id"] == "snapshots_scenes"
    assert contract["remediation_action_id"] == "remediate_snapshots_scenes"
    assert contract["safety"]["live_write"] is False
    assert contract["safety"]["scene_load_allowed"] is False
    assert contract["safety"]["scene_save_allowed"] is False
    assert contract["artifact_contract"]["minimum_snapshot_count"] == 1
    assert "scene_content_fingerprint" in contract["artifact_contract"]["required_snapshot_fields"]
    assert "rollback_reference" in contract["artifact_contract"]["required_coverage_flags"]
    assert "scene_recall_events" in contract["artifact_contract"][
        "forbidden_mutation_evidence_fields"
    ]
    assert contract["summary"]["physical_fixture_gap_remains"] is True
    assert contract["summary"]["accepted_as_physical_fixture_candidate"] is False
    assert contract["summary"]["live_authority_granted"] is False
    assert "scene_preset" in contract["summary"]["promotion_families_requiring_surface"]


def test_snapshots_scenes_contract_validates_minimal_physical_artifact():
    result = validate_snapshots_scenes_physical_readback_artifact(
        valid_snapshots_scenes_artifact()
    )

    assert result["method_id"] == SNAPSHOTS_SCENES_PHYSICAL_READBACK_CONTRACT_METHOD_ID
    assert result["status"] == "valid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is True
    assert result["live_authority_granted"] is False
    assert result["full_autonomous_ready"] is False
    assert result["snapshot_count"] == 1
    assert result["violations"] == []


def test_snapshots_scenes_contract_rejects_live_or_cached_claims():
    artifact = valid_snapshots_scenes_artifact()
    artifact["source"] = "mixer_state_cache"
    artifact["live_write"] = True
    artifact["safety"]["scene_load_allowed"] = True
    artifact["snapshots"][0]["readback_source"] = "mixer_state_cache"
    artifact["snapshots"][0]["scene_load_events"] = [{"snapshot": 1}]

    result = validate_snapshots_scenes_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is False
    assert result["live_authority_granted"] is False
    assert "source_not_physical_readback" in result["violations"]
    assert "safety_scene_load_allowed_not_false" in result["violations"]
    assert "top_level_live_write_not_false" in result["violations"]
    assert "snapshot_0_readback_source_not_physical_readback" in result["violations"]
    assert (
        "mutation_evidence_field_present:snapshots[0].scene_load_events"
        in result["violations"]
    )


def test_snapshots_scenes_contract_rejects_incomplete_rows():
    artifact = valid_snapshots_scenes_artifact()
    artifact["snapshot_count"] = 2
    artifact["snapshots"][0].pop("rollback_reference")
    artifact["snapshots"][0]["coverage"]["scoped_diff_marker"] = False

    result = validate_snapshots_scenes_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert "snapshot_count_mismatch" in result["violations"]
    assert "snapshot_0_field_missing_field:rollback_reference" in result["violations"]
    assert "snapshot_0_rollback_reference_missing" in result["violations"]
    assert "snapshot_0_coverage_scoped_diff_marker_not_true" in result["violations"]


def test_snapshots_scenes_contract_module_stays_transport_free():
    source = Path("backend/snapshots_scenes_physical_readback_contract.py").read_text(
        encoding="utf-8"
    )

    forbidden = (
        "socket",
        "requests",
        "python-osc",
        "udp_client",
        "wing_client",
        "dlive_client",
        "load_scene",
        "save_scene",
        "recall_scene",
    )
    for token in forbidden:
        assert token not in source


def test_snapshots_scenes_contract_is_shadow_only_not_executor():
    method = get_processing_method("snapshots_scenes_physical_readback_contract")

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert "snapshots_scenes_physical_readback_contract" not in main.executor_method_ids


def test_snapshots_scenes_contract_registry_snapshot_is_read_only():
    snapshot = build_processing_registry_snapshot(include_methods=False)
    contract = snapshot["snapshots_scenes_physical_readback_contract"]

    assert contract["type"] == "snapshots_scenes_physical_readback_contract"
    assert contract["safety"]["live_write"] is False
    assert contract["safety"]["scene_load_allowed"] is False
    assert contract["summary"]["physical_fixture_gap_remains"] is True


def test_snapshots_scenes_contract_supports_console_read_goal_without_live_scope():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert "snapshots_scenes_physical_readback_contract" in row.method_ids
    assert "backend/snapshots_scenes_physical_readback_contract.py" in row.support_paths
    assert row.current_live_write_kinds == ()
