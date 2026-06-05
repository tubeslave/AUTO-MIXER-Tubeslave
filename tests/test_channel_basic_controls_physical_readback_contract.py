import copy
from pathlib import Path

from backend.channel_basic_controls_physical_readback_contract import (
    CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_TYPE,
    CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SCHEMA_ID,
    CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID,
    CHANNEL_BASIC_CONTROLS_SURFACE_ID,
    build_channel_basic_controls_physical_readback_contract,
    validate_channel_basic_controls_physical_readback_artifact,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


def _valid_artifact() -> dict:
    return {
        "type": CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_TYPE,
        "schema_id": CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SCHEMA_ID,
        "surface_id": CHANNEL_BASIC_CONTROLS_SURFACE_ID,
        "remediation_action_id": CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID,
        "captured_at": "2026-06-05T00:00:00Z",
        "source": "physical_readback",
        "console_model": "wing",
        "channel_count": 2,
        "channels": [
            {
                "channel_id": 1,
                "channel_label": "Lead Vox",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "identity": {
                    "source_name": "Lead Vox",
                    "source_role": "lead_vocal",
                    "input_source": "local_in_1",
                },
                "controls": {
                    "fader_db": -8.5,
                    "trim_db": 1.0,
                    "mute": False,
                    "pan": 0.0,
                },
                "coverage": {
                    "identity": True,
                    "basic_controls": True,
                    "cache_matches_physical_readback": True,
                },
            },
            {
                "channel_id": 2,
                "channel_label": "Snare Top",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "identity": {
                    "source_name": "Snare Top",
                    "source_role": "snare_top",
                    "input_source": "local_in_2",
                },
                "controls": {
                    "fader_db": -12.0,
                    "trim_db": 0.0,
                    "mute": False,
                    "pan": -5.0,
                },
                "coverage": {
                    "identity": True,
                    "basic_controls": True,
                    "cache_matches_physical_readback": True,
                },
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


def test_channel_basic_controls_contract_describes_future_physical_artifact():
    contract = build_channel_basic_controls_physical_readback_contract()

    assert contract["type"] == "channel_basic_controls_physical_readback_contract"
    assert contract["method_id"] == CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_CONTRACT_METHOD_ID
    assert contract["schema_id"] == CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SCHEMA_ID
    assert contract["status"] == "contract_ready_physical_fixture_missing"
    assert contract["surface_id"] == CHANNEL_BASIC_CONTROLS_SURFACE_ID
    assert contract["remediation_action_id"] == CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID
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
    assert artifact_contract["type"] == CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_TYPE
    assert "channels" in artifact_contract["required_top_level_fields"]
    assert "controls" in artifact_contract["required_channel_fields"]
    assert artifact_contract["required_control_fields"] == ["fader_db", "trim_db", "mute", "pan"]
    assert "cache_matches_physical_readback" in artifact_contract["required_coverage_flags"]
    assert "physical_readback" in artifact_contract["allowed_sources"]
    assert "mute_write_events" in artifact_contract["forbidden_mutation_evidence_fields"]
    assert contract["summary"]["physical_readback_surface_id"] == CHANNEL_BASIC_CONTROLS_SURFACE_ID
    assert contract["summary"]["physical_fixture_gap_remains"] is True
    assert contract["summary"]["live_authority_granted"] is False
    assert "mute" in contract["summary"]["promotion_families_requiring_surface"]


def test_channel_basic_controls_contract_validates_minimal_physical_artifact():
    result = validate_channel_basic_controls_physical_readback_artifact(_valid_artifact())

    assert result["status"] == "valid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is True
    assert result["live_authority_granted"] is False
    assert result["full_autonomous_ready"] is False
    assert result["channel_count"] == 2
    assert result["violations"] == []
    assert result["evidence"]["source"] == "physical_readback"


def test_channel_basic_controls_contract_rejects_live_or_cached_claims():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["source"] = "mixer_state_cache"
    artifact["full_autonomous_ready"] = True
    artifact["mute_write_events"] = []
    artifact["safety"]["live_write"] = True
    artifact["channels"][0]["status"] = "verified_from_cache"
    artifact["channels"][0]["source"] = "mixer_state_cache"
    artifact["channels"][0]["controls"]["mute"] = "false"
    artifact["channels"][0]["coverage"]["cache_matches_physical_readback"] = False

    result = validate_channel_basic_controls_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is False
    assert result["live_authority_granted"] is False
    assert result["full_autonomous_ready"] is False
    assert "source_not_physical_readback" in result["violations"]
    assert "top_level_full_autonomous_ready_not_false" in result["violations"]
    assert "safety_live_write_not_false" in result["violations"]
    assert "mutation_evidence_field_present:mute_write_events" in result["violations"]
    assert "channel_0_status_not_physical_verified" in result["violations"]
    assert "channel_0_source_not_physical_readback" in result["violations"]
    assert "channel_0_controls_mute_not_bool" in result["violations"]
    assert "channel_0_coverage_cache_matches_physical_readback_not_true" in result["violations"]


def test_channel_basic_controls_contract_rejects_incomplete_channel_rows():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["channel_count"] = 3
    artifact["channels"][1]["channel_id"] = 1
    del artifact["channels"][1]["identity"]["input_source"]
    del artifact["channels"][1]["controls"]["pan"]

    result = validate_channel_basic_controls_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert "channel_count_mismatch" in result["violations"]
    assert "channel_1_channel_id_duplicate" in result["violations"]
    assert "channel_1_identity_missing_field:input_source" in result["violations"]
    assert "channel_1_controls_missing_field:pan" in result["violations"]


def test_channel_basic_controls_contract_is_transport_free():
    source = Path("backend/channel_basic_controls_physical_readback_contract.py").read_text(
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
        "set_mute",
        "set_pan",
    ]
    for token in forbidden:
        assert token not in source


def test_channel_basic_controls_contract_is_registry_shadow_only_not_executor():
    method = get_processing_method(CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_CONTRACT_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_CONTRACT_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    contract = snapshot["channel_basic_controls_physical_readback_contract"]
    assert contract["type"] == "channel_basic_controls_physical_readback_contract"
    assert contract["safety"]["live_write"] is False
    assert contract["summary"]["physical_fixture_gap_remains"] is True


def test_channel_basic_controls_contract_is_console_read_goal_support_path():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_CONTRACT_METHOD_ID in row.method_ids
    assert "backend/channel_basic_controls_physical_readback_contract.py" in row.support_paths
    assert row.current_live_write_kinds == ()
