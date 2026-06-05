from pathlib import Path

from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.user_presets_physical_readback_contract import (
    USER_PRESETS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    build_user_presets_physical_readback_contract,
    validate_user_presets_physical_readback_artifact,
)


def valid_user_presets_artifact():
    return {
        "type": "user_presets_physical_readback",
        "schema_id": "user_presets_physical_readback.v1",
        "surface_id": "user_presets",
        "remediation_action_id": "remediate_user_presets",
        "captured_at": "2026-06-05T00:00:00Z",
        "source": "wing_physical_readback",
        "console_model": "WING Rack",
        "preset_count": 1,
        "presets": [
            {
                "preset_index": 1,
                "preset_name": "Lead Vox Chain",
                "preset_scope": "channel_processing",
                "target_type": "channel",
                "target_label": "Lead Vox",
                "readback_status": "verified_from_physical_readback",
                "readback_source": "wing_physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "preset_recall_policy": "blocked_manual_only",
                "preset_store_policy": "blocked_manual_only",
                "destructive_recall_marker": "preset_recall_would_change_console_parameters",
                "destructive_store_marker": "preset_store_would_persist_console_parameters",
                "preset_content_fingerprint": "sha256:preset-content",
                "scoped_diff_marker": "diff:preset-1-before-after",
                "rollback_reference": "rollback:preset-1",
                "coverage": {
                    "catalog_identity": True,
                    "scope": True,
                    "target": True,
                    "preset_content_fingerprint": True,
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
            "preset_recall_allowed": False,
            "preset_store_allowed": False,
        },
    }


def test_user_presets_contract_describes_future_physical_artifact():
    contract = build_user_presets_physical_readback_contract()

    assert contract["type"] == "user_presets_physical_readback_contract"
    assert contract["status"] == "contract_ready_physical_fixture_missing"
    assert contract["surface_id"] == "user_presets"
    assert contract["remediation_action_id"] == "remediate_user_presets"
    assert contract["safety"]["live_write"] is False
    assert contract["safety"]["preset_recall_allowed"] is False
    assert contract["safety"]["preset_store_allowed"] is False
    assert contract["artifact_contract"]["minimum_preset_count"] == 1
    assert "preset_content_fingerprint" in contract["artifact_contract"]["required_preset_fields"]
    assert "rollback_reference" in contract["artifact_contract"]["required_coverage_flags"]
    assert "preset_store_events" in contract["artifact_contract"][
        "forbidden_mutation_evidence_fields"
    ]
    assert contract["summary"]["physical_fixture_gap_remains"] is True
    assert contract["summary"]["accepted_as_physical_fixture_candidate"] is False
    assert contract["summary"]["live_authority_granted"] is False
    assert "scene_preset" in contract["summary"]["promotion_families_requiring_surface"]


def test_user_presets_contract_validates_minimal_physical_artifact():
    result = validate_user_presets_physical_readback_artifact(
        valid_user_presets_artifact()
    )

    assert result["method_id"] == USER_PRESETS_PHYSICAL_READBACK_CONTRACT_METHOD_ID
    assert result["status"] == "valid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is True
    assert result["live_authority_granted"] is False
    assert result["full_autonomous_ready"] is False
    assert result["preset_count"] == 1
    assert result["violations"] == []


def test_user_presets_contract_rejects_live_or_cached_claims():
    artifact = valid_user_presets_artifact()
    artifact["source"] = "mixer_state_cache"
    artifact["live_write"] = True
    artifact["safety"]["preset_store_allowed"] = True
    artifact["presets"][0]["readback_source"] = "mixer_state_cache"
    artifact["presets"][0]["preset_store_events"] = [{"preset": 1}]

    result = validate_user_presets_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is False
    assert result["live_authority_granted"] is False
    assert "source_not_physical_readback" in result["violations"]
    assert "safety_preset_store_allowed_not_false" in result["violations"]
    assert "top_level_live_write_not_false" in result["violations"]
    assert "preset_0_readback_source_not_physical_readback" in result["violations"]
    assert (
        "mutation_evidence_field_present:presets[0].preset_store_events"
        in result["violations"]
    )


def test_user_presets_contract_rejects_incomplete_rows():
    artifact = valid_user_presets_artifact()
    artifact["preset_count"] = 2
    artifact["presets"][0].pop("rollback_reference")
    artifact["presets"][0]["coverage"]["scoped_diff_marker"] = False

    result = validate_user_presets_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert "preset_count_mismatch" in result["violations"]
    assert "preset_0_field_missing_field:rollback_reference" in result["violations"]
    assert "preset_0_rollback_reference_missing" in result["violations"]
    assert "preset_0_coverage_scoped_diff_marker_not_true" in result["violations"]


def test_user_presets_contract_module_stays_transport_free():
    source = Path("backend/user_presets_physical_readback_contract.py").read_text(
        encoding="utf-8"
    )

    forbidden = (
        "socket",
        "requests",
        "python-osc",
        "udp_client",
        "wing_client",
        "dlive_client",
        "recall_preset",
        "store_preset",
        "load_preset",
        "save_preset",
    )
    for token in forbidden:
        assert token not in source


def test_user_presets_contract_is_shadow_only_not_executor():
    method = get_processing_method("user_presets_physical_readback_contract")

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert "user_presets_physical_readback_contract" not in main.executor_method_ids


def test_user_presets_contract_registry_snapshot_is_read_only():
    snapshot = build_processing_registry_snapshot(include_methods=False)
    contract = snapshot["user_presets_physical_readback_contract"]

    assert contract["type"] == "user_presets_physical_readback_contract"
    assert contract["safety"]["live_write"] is False
    assert contract["safety"]["preset_recall_allowed"] is False
    assert contract["safety"]["preset_store_allowed"] is False
    assert contract["summary"]["physical_fixture_gap_remains"] is True


def test_user_presets_contract_supports_console_read_goal_without_live_scope():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert "user_presets_physical_readback_contract" in row.method_ids
    assert "backend/user_presets_physical_readback_contract.py" in row.support_paths
    assert row.current_live_write_kinds == ()
