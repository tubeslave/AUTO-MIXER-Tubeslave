import copy
from pathlib import Path

from backend.fx_slots_inserts_physical_readback_contract import (
    FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_TYPE,
    FX_SLOTS_INSERTS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID,
    FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID,
    FX_SLOTS_INSERTS_SURFACE_ID,
    MINIMUM_FX_PARAMETER_COUNT,
    build_fx_slots_inserts_physical_readback_contract,
    validate_fx_slots_inserts_physical_readback_artifact,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


def _parameters() -> list[dict]:
    return [
        {
            "parameter_index": index,
            "value": float(index) / 10.0,
            "schema_status": "mapped",
            "schema_family": "reverb",
            "schema_source": "physical_model_schema",
            "semantic_name": f"parameter_{index}",
            "unit": "raw",
            "safe_range": {"min": 0.0, "max": 1.0},
        }
        for index in range(1, MINIMUM_FX_PARAMETER_COUNT + 1)
    ]


def _valid_artifact() -> dict:
    parameters = _parameters()
    return {
        "type": FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_TYPE,
        "schema_id": FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID,
        "surface_id": FX_SLOTS_INSERTS_SURFACE_ID,
        "remediation_action_id": FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID,
        "captured_at": "2026-06-05T00:00:00Z",
        "source": "physical_readback",
        "console_model": "wing",
        "fx_slot_count": 1,
        "insert_count": 1,
        "fx_slots": [
            {
                "slot": "FX1",
                "slot_number": 1,
                "readback_status": "verified_from_physical_readback",
                "readback_source": "physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "model": "Hall Reverb",
                "on": True,
                "mix": 25.0,
                "source": "Bus 1",
                "mode": "insert",
                "assigned_channel": 1,
                "assigned_position": "post",
                "active": True,
                "parameters": parameters,
                "parameter_schema": {
                    "status": "mapped",
                    "schema_family": "reverb",
                    "schema_source": "physical_model_schema",
                    "mapped_parameter_count": len(parameters),
                    "generic_parameter_count": 0,
                    "cached_parameter_count": len(parameters),
                    "physical_model_schema_verified": True,
                    "live_write_authority": False,
                },
                "coverage": {
                    "identity": True,
                    "assignment": True,
                    "parameters": True,
                    "parameter_schema": True,
                    "cache_matches_physical_readback": True,
                },
            }
        ],
        "inserts": [
            {
                "target_type": "channel",
                "target": 1,
                "position": "post",
                "readback_status": "verified_from_physical_readback",
                "readback_source": "physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "on": True,
                "slot": "FX1",
                "status": "active",
                "mode": "post_fader",
                "weight": 100.0,
                "linked_fx_slot": "FX1",
                "active": True,
                "insert_assignment_policy": "blocked_manual_only",
                "destructive_change_marker": "insert_assignment_would_repatch_processing_chain",
                "coverage": {
                    "assignment": True,
                    "status": True,
                    "scoped_diff_marker": True,
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
        },
    }


def test_fx_slots_inserts_contract_describes_future_physical_artifact():
    contract = build_fx_slots_inserts_physical_readback_contract()

    assert contract["type"] == "fx_slots_inserts_physical_readback_contract"
    assert contract["method_id"] == FX_SLOTS_INSERTS_PHYSICAL_READBACK_CONTRACT_METHOD_ID
    assert contract["schema_id"] == FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID
    assert contract["status"] == "contract_ready_physical_fixture_missing"
    assert contract["surface_id"] == FX_SLOTS_INSERTS_SURFACE_ID
    assert contract["remediation_action_id"] == FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID
    assert contract["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "live_widening_allowed": False,
        "full_autonomous_ready": False,
    }
    artifact_contract = contract["artifact_contract"]
    assert artifact_contract["type"] == FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_TYPE
    assert "fx_slots" in artifact_contract["required_top_level_fields"]
    assert "parameter_schema" in artifact_contract["required_fx_slot_fields"]
    assert "safe_range" in artifact_contract["required_parameter_fields"]
    assert "physical_model_schema_verified" in artifact_contract[
        "required_parameter_schema_fields"
    ]
    assert "scoped_diff_marker" in artifact_contract["required_insert_coverage_flags"]
    assert artifact_contract["minimum_fx_parameter_count"] == MINIMUM_FX_PARAMETER_COUNT
    assert "physical_readback" in artifact_contract["allowed_sources"]
    assert "fx_parameter_write_events" in artifact_contract[
        "forbidden_mutation_evidence_fields"
    ]
    assert contract["summary"]["physical_readback_surface_id"] == FX_SLOTS_INSERTS_SURFACE_ID
    assert contract["summary"]["physical_fixture_gap_remains"] is True
    assert contract["summary"]["live_authority_granted"] is False
    assert "fx_send_insert" in contract["summary"]["promotion_families_requiring_surface"]
    assert "routing" in contract["summary"]["promotion_families_requiring_surface"]


def test_fx_slots_inserts_contract_validates_minimal_physical_artifact():
    result = validate_fx_slots_inserts_physical_readback_artifact(_valid_artifact())

    assert result["status"] == "valid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is True
    assert result["live_authority_granted"] is False
    assert result["full_autonomous_ready"] is False
    assert result["fx_slot_count"] == 1
    assert result["insert_count"] == 1
    assert result["violations"] == []
    assert result["evidence"]["source"] == "physical_readback"


def test_fx_slots_inserts_contract_rejects_live_or_cached_claims():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["source"] = "mixer_state_cache"
    artifact["full_autonomous_ready"] = True
    artifact["fx_write_events"] = []
    artifact["safety"]["live_write"] = True
    artifact["fx_slots"][0]["readback_status"] = "verified_from_cache"
    artifact["fx_slots"][0]["readback_source"] = "mixer_state_cache"
    artifact["fx_slots"][0]["parameter_schema"]["physical_model_schema_verified"] = False
    artifact["fx_slots"][0]["coverage"]["cache_matches_physical_readback"] = False
    artifact["inserts"][0]["readback_status"] = "verified_from_cache"
    artifact["inserts"][0]["readback_source"] = "mixer_state_cache"
    artifact["inserts"][0]["on"] = "true"
    artifact["inserts"][0]["linked_fx_slot"] = None

    result = validate_fx_slots_inserts_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is False
    assert result["live_authority_granted"] is False
    assert result["full_autonomous_ready"] is False
    assert "source_not_physical_readback" in result["violations"]
    assert "top_level_full_autonomous_ready_not_false" in result["violations"]
    assert "safety_live_write_not_false" in result["violations"]
    assert "mutation_evidence_field_present:fx_write_events" in result["violations"]
    assert "fx_slot_0_readback_status_not_physical_verified" in result["violations"]
    assert "fx_slot_0_readback_source_not_physical_readback" in result["violations"]
    assert "fx_slot_0_parameter_schema_physical_model_schema_not_verified" in result[
        "violations"
    ]
    assert "fx_slot_0_coverage_cache_matches_physical_readback_not_true" in result[
        "violations"
    ]
    assert "insert_0_readback_status_not_physical_verified" in result["violations"]
    assert "insert_0_readback_source_not_physical_readback" in result["violations"]
    assert "insert_0_on_not_bool_or_0_1" in result["violations"]
    assert "insert_0_active_insert_missing_linked_fx_slot" in result["violations"]


def test_fx_slots_inserts_contract_rejects_incomplete_rows():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["fx_slot_count"] = 2
    artifact["insert_count"] = 2
    artifact["fx_slots"][0]["slot_number"] = 0
    artifact["fx_slots"][0]["parameters"] = artifact["fx_slots"][0]["parameters"][:2]
    artifact["fx_slots"][0]["parameter_schema"]["cached_parameter_count"] = 99
    artifact["fx_slots"][0]["parameter_schema"]["generic_parameter_count"] = 1
    artifact["inserts"][0]["target"] = 0
    artifact["inserts"][0]["position"] = "side"
    artifact["inserts"][0]["weight"] = "full"
    del artifact["inserts"][0]["destructive_change_marker"]

    result = validate_fx_slots_inserts_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert "fx_slot_count_mismatch" in result["violations"]
    assert "insert_count_mismatch" in result["violations"]
    assert "fx_slot_0_slot_number_invalid" in result["violations"]
    assert "fx_slot_0_parameters_missing" in result["violations"]
    assert "fx_slot_0_parameter_schema_cached_parameter_count_mismatch" in result[
        "violations"
    ]
    assert "fx_slot_0_parameter_schema_generic_parameter_count_not_zero" in result[
        "violations"
    ]
    assert "insert_0_target_invalid" in result["violations"]
    assert "insert_0_position_invalid" in result["violations"]
    assert "insert_0_weight_not_number" in result["violations"]
    assert "insert_0_missing_field:destructive_change_marker" in result["violations"]


def test_fx_slots_inserts_contract_is_transport_free():
    source = Path("backend/fx_slots_inserts_physical_readback_contract.py").read_text(
        encoding="utf-8"
    )

    forbidden_transport_tokens = [
        "WingClient",
        "DLiveClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
        ".send(",
        "mixer_client",
    ]
    forbidden_write_call_tokens = [
        "set_fx_slot(",
        "set_fx_parameter(",
        "set_insert(",
        "set_bus_send(",
        "set_matrix_send(",
    ]
    for token in forbidden_transport_tokens + forbidden_write_call_tokens:
        assert token not in source


def test_fx_slots_inserts_contract_is_registry_shadow_only_not_executor():
    method = get_processing_method(FX_SLOTS_INSERTS_PHYSICAL_READBACK_CONTRACT_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert FX_SLOTS_INSERTS_PHYSICAL_READBACK_CONTRACT_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    contract = snapshot["fx_slots_inserts_physical_readback_contract"]
    assert contract["type"] == "fx_slots_inserts_physical_readback_contract"
    assert contract["safety"]["live_write"] is False
    assert contract["summary"]["physical_fixture_gap_remains"] is True


def test_fx_slots_inserts_contract_is_console_read_goal_support_path():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert FX_SLOTS_INSERTS_PHYSICAL_READBACK_CONTRACT_METHOD_ID in row.method_ids
    assert "backend/fx_slots_inserts_physical_readback_contract.py" in row.support_paths
    assert row.current_live_write_kinds == ()
