import copy
from pathlib import Path

from backend.channel_processing_physical_readback_contract import (
    CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_TYPE,
    CHANNEL_PROCESSING_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    CHANNEL_PROCESSING_PHYSICAL_READBACK_SCHEMA_ID,
    CHANNEL_PROCESSING_REMEDIATION_ACTION_ID,
    CHANNEL_PROCESSING_SURFACE_ID,
    build_channel_processing_physical_readback_contract,
    validate_channel_processing_physical_readback_artifact,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


def _processing_channel(channel_id: int, label: str) -> dict:
    return {
        "channel_id": channel_id,
        "channel_label": label,
        "status": "verified_from_physical_readback",
        "source": "physical_readback",
        "captured_at": "2026-06-05T00:00:00Z",
        "hpf": {
            "enabled": 1,
            "frequency_hz": 120.0,
        },
        "eq": {
            "enabled": 1,
            "bands": [
                {"band": 1, "frequency_hz": 100.0, "gain_db": -1.0, "q": 0.8},
                {"band": 2, "frequency_hz": 250.0, "gain_db": -2.0, "q": 0.9},
                {"band": 3, "frequency_hz": 1200.0, "gain_db": 1.0, "q": 1.2},
                {"band": 4, "frequency_hz": 4500.0, "gain_db": -1.5, "q": 1.0},
            ],
        },
        "compressor": {
            "enabled": 1,
            "model": "COMP",
            "threshold_db": -18.0,
            "ratio": "3.0",
            "attack_ms": 12.0,
            "release_ms": 180.0,
            "makeup_db": 2.0,
            "mix_pct": 80.0,
            "detector": "PEAK",
            "gain_reduction_db": 2.4,
        },
        "gate": {
            "enabled": 1,
            "model": "GATE",
            "threshold_db": -47.5,
            "range_db": 18.0,
            "attack_ms": 4.0,
            "hold_ms": 80.0,
            "release_ms": 240.0,
            "accent": 3.0,
            "ratio": "4.0",
        },
        "coverage": {
            "hpf": True,
            "eq": True,
            "compressor": True,
            "gate": True,
            "cache_matches_physical_readback": True,
        },
    }


def _valid_artifact() -> dict:
    return {
        "type": CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_TYPE,
        "schema_id": CHANNEL_PROCESSING_PHYSICAL_READBACK_SCHEMA_ID,
        "surface_id": CHANNEL_PROCESSING_SURFACE_ID,
        "remediation_action_id": CHANNEL_PROCESSING_REMEDIATION_ACTION_ID,
        "captured_at": "2026-06-05T00:00:00Z",
        "source": "physical_readback",
        "console_model": "wing",
        "channel_count": 2,
        "channels": [
            _processing_channel(1, "Lead Vox"),
            _processing_channel(2, "Snare Top"),
        ],
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
        },
    }


def test_channel_processing_contract_describes_future_physical_artifact():
    contract = build_channel_processing_physical_readback_contract()

    assert contract["type"] == "channel_processing_physical_readback_contract"
    assert contract["method_id"] == CHANNEL_PROCESSING_PHYSICAL_READBACK_CONTRACT_METHOD_ID
    assert contract["schema_id"] == CHANNEL_PROCESSING_PHYSICAL_READBACK_SCHEMA_ID
    assert contract["status"] == "contract_ready_physical_fixture_missing"
    assert contract["surface_id"] == CHANNEL_PROCESSING_SURFACE_ID
    assert contract["remediation_action_id"] == CHANNEL_PROCESSING_REMEDIATION_ACTION_ID
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
    assert artifact_contract["type"] == CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_TYPE
    assert "channels" in artifact_contract["required_top_level_fields"]
    assert "eq" in artifact_contract["required_channel_fields"]
    assert artifact_contract["required_hpf_fields"] == ["enabled", "frequency_hz"]
    assert artifact_contract["required_eq_band_fields"] == ["band", "frequency_hz", "gain_db", "q"]
    assert "gain_reduction_db" in artifact_contract["required_compressor_fields"]
    assert "range_db" in artifact_contract["required_gate_fields"]
    assert "cache_matches_physical_readback" in artifact_contract["required_coverage_flags"]
    assert "physical_readback" in artifact_contract["allowed_sources"]
    assert "eq_write_events" in artifact_contract["forbidden_mutation_evidence_fields"]
    assert "reset_channel_processing_events" in artifact_contract[
        "forbidden_mutation_evidence_fields"
    ]
    assert artifact_contract["minimum_eq_band_count"] == 4
    assert contract["summary"]["physical_readback_surface_id"] == CHANNEL_PROCESSING_SURFACE_ID
    assert contract["summary"]["physical_fixture_gap_remains"] is True
    assert contract["summary"]["live_authority_granted"] is False
    assert "deessing" in contract["summary"]["promotion_families_requiring_surface"]
    assert "cross_channel_unmasking" in contract["summary"]["promotion_families_requiring_surface"]


def test_channel_processing_contract_validates_minimal_physical_artifact():
    result = validate_channel_processing_physical_readback_artifact(_valid_artifact())

    assert result["status"] == "valid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is True
    assert result["live_authority_granted"] is False
    assert result["full_autonomous_ready"] is False
    assert result["channel_count"] == 2
    assert result["violations"] == []
    assert result["evidence"]["source"] == "physical_readback"


def test_channel_processing_contract_rejects_live_or_cached_claims():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["source"] = "mixer_state_cache"
    artifact["full_autonomous_ready"] = True
    artifact["eq_write_events"] = []
    artifact["safety"]["live_write"] = True
    artifact["channels"][0]["status"] = "verified_from_cache"
    artifact["channels"][0]["source"] = "mixer_state_cache"
    artifact["channels"][0]["hpf"]["enabled"] = "on"
    artifact["channels"][0]["coverage"]["cache_matches_physical_readback"] = False

    result = validate_channel_processing_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is False
    assert result["live_authority_granted"] is False
    assert result["full_autonomous_ready"] is False
    assert "source_not_physical_readback" in result["violations"]
    assert "top_level_full_autonomous_ready_not_false" in result["violations"]
    assert "safety_live_write_not_false" in result["violations"]
    assert "mutation_evidence_field_present:eq_write_events" in result["violations"]
    assert "channel_0_status_not_physical_verified" in result["violations"]
    assert "channel_0_source_not_physical_readback" in result["violations"]
    assert "channel_0_hpf_enabled_not_bool_or_0_1" in result["violations"]
    assert "channel_0_coverage_cache_matches_physical_readback_not_true" in result["violations"]


def test_channel_processing_contract_rejects_incomplete_processing_rows():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["channel_count"] = 3
    artifact["channels"][1]["channel_id"] = 1
    artifact["channels"][1]["eq"]["bands"] = [
        {"band": 1, "frequency_hz": 100.0, "gain_db": -1.0, "q": 0.8},
    ]
    del artifact["channels"][1]["compressor"]["gain_reduction_db"]
    del artifact["channels"][1]["gate"]["range_db"]

    result = validate_channel_processing_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert "channel_count_mismatch" in result["violations"]
    assert "channel_1_channel_id_duplicate" in result["violations"]
    assert "channel_1_eq_bands_missing" in result["violations"]
    assert "channel_1_compressor_missing_field:gain_reduction_db" in result["violations"]
    assert "channel_1_gate_missing_field:range_db" in result["violations"]


def test_channel_processing_contract_is_transport_free():
    source = Path("backend/channel_processing_physical_readback_contract.py").read_text(
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
        "set_fader(",
        "set_gain(",
        "set_eq(",
        "set_compressor(",
        "set_gate(",
        "reset_channel_processing(",
    ]
    for token in forbidden_transport_tokens + forbidden_write_call_tokens:
        assert token not in source


def test_channel_processing_contract_is_registry_shadow_only_not_executor():
    method = get_processing_method(CHANNEL_PROCESSING_PHYSICAL_READBACK_CONTRACT_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert CHANNEL_PROCESSING_PHYSICAL_READBACK_CONTRACT_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    contract = snapshot["channel_processing_physical_readback_contract"]
    assert contract["type"] == "channel_processing_physical_readback_contract"
    assert contract["safety"]["live_write"] is False
    assert contract["summary"]["physical_fixture_gap_remains"] is True


def test_channel_processing_contract_is_console_read_goal_support_path():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert CHANNEL_PROCESSING_PHYSICAL_READBACK_CONTRACT_METHOD_ID in row.method_ids
    assert "backend/channel_processing_physical_readback_contract.py" in row.support_paths
    assert row.current_live_write_kinds == ()
