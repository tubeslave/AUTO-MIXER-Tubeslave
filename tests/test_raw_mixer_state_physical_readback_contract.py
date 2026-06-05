import copy
from pathlib import Path

from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.raw_mixer_state_physical_readback_contract import (
    RAW_MIXER_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE,
    RAW_MIXER_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    RAW_MIXER_STATE_PHYSICAL_READBACK_SCHEMA_ID,
    RAW_MIXER_STATE_REMEDIATION_ACTION_ID,
    RAW_MIXER_STATE_SURFACE_ID,
    build_raw_mixer_state_physical_readback_contract,
    validate_raw_mixer_state_physical_readback_artifact,
)


def _valid_artifact() -> dict:
    return {
        "type": RAW_MIXER_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE,
        "schema_id": RAW_MIXER_STATE_PHYSICAL_READBACK_SCHEMA_ID,
        "surface_id": RAW_MIXER_STATE_SURFACE_ID,
        "remediation_action_id": RAW_MIXER_STATE_REMEDIATION_ACTION_ID,
        "captured_at": "2026-06-05T00:00:00Z",
        "source": "physical_readback",
        "console_model": "wing",
        "address_count": 2,
        "readback_rows": [
            {
                "address": "/ch/1/mix/fader",
                "value": -12.0,
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
            },
            {
                "address": "/ch/1/in/set/trim",
                "value": 1.5,
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
            },
        ],
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
        },
    }


def test_raw_mixer_state_contract_describes_future_physical_artifact():
    contract = build_raw_mixer_state_physical_readback_contract()

    assert contract["type"] == "raw_mixer_state_physical_readback_contract"
    assert contract["method_id"] == RAW_MIXER_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID
    assert contract["schema_id"] == RAW_MIXER_STATE_PHYSICAL_READBACK_SCHEMA_ID
    assert contract["status"] == "contract_ready_physical_fixture_missing"
    assert contract["surface_id"] == RAW_MIXER_STATE_SURFACE_ID
    assert contract["remediation_action_id"] == RAW_MIXER_STATE_REMEDIATION_ACTION_ID
    assert contract["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "live_widening_allowed": False,
        "full_autonomous_ready": False,
    }
    assert contract["artifact_contract"]["type"] == RAW_MIXER_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE
    assert "readback_rows" in contract["artifact_contract"]["required_top_level_fields"]
    assert "physical_readback" in contract["artifact_contract"]["allowed_sources"]
    assert "manual_write_events" in contract["artifact_contract"]["forbidden_mutation_evidence_fields"]
    assert contract["summary"]["next_physical_readback_surface_id"] == RAW_MIXER_STATE_SURFACE_ID
    assert contract["summary"]["next_physical_readback_action_id"] == RAW_MIXER_STATE_REMEDIATION_ACTION_ID
    assert contract["summary"]["physical_fixture_gap_remains"] is True
    assert contract["summary"]["live_authority_granted"] is False


def test_raw_mixer_state_contract_validates_minimal_physical_artifact():
    result = validate_raw_mixer_state_physical_readback_artifact(_valid_artifact())

    assert result["status"] == "valid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is True
    assert result["live_authority_granted"] is False
    assert result["full_autonomous_ready"] is False
    assert result["row_count"] == 2
    assert result["violations"] == []
    assert result["evidence"]["source"] == "physical_readback"


def test_raw_mixer_state_contract_rejects_live_or_cached_claims():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["surface_id"] = "channel_identity_and_basic_controls"
    artifact["source"] = "mixer_state_cache"
    artifact["full_autonomous_ready"] = True
    artifact["manual_write_events"] = []
    artifact["safety"]["live_write"] = True
    artifact["readback_rows"][0]["status"] = "verified_from_cache"
    artifact["readback_rows"][0]["source"] = "mixer_state_cache"

    result = validate_raw_mixer_state_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is False
    assert result["live_authority_granted"] is False
    assert result["full_autonomous_ready"] is False
    assert "surface_id_mismatch" in result["violations"]
    assert "source_not_physical_readback" in result["violations"]
    assert "top_level_full_autonomous_ready_not_false" in result["violations"]
    assert "safety_live_write_not_false" in result["violations"]
    assert "mutation_evidence_field_present:manual_write_events" in result["violations"]
    assert "row_0_status_not_physical_verified" in result["violations"]
    assert "row_0_source_not_physical_readback" in result["violations"]


def test_raw_mixer_state_contract_is_transport_free():
    source = Path("backend/raw_mixer_state_physical_readback_contract.py").read_text(
        encoding="utf-8"
    )

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


def test_raw_mixer_state_contract_is_registry_shadow_only_not_executor():
    method = get_processing_method(RAW_MIXER_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert RAW_MIXER_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    contract = snapshot["raw_mixer_state_physical_readback_contract"]
    assert contract["type"] == "raw_mixer_state_physical_readback_contract"
    assert contract["safety"]["live_write"] is False
    assert contract["summary"]["physical_fixture_gap_remains"] is True


def test_raw_mixer_state_contract_is_console_read_goal_support_path():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert RAW_MIXER_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID in row.method_ids
    assert "backend/raw_mixer_state_physical_readback_contract.py" in row.support_paths
    assert row.current_live_write_kinds == ()
