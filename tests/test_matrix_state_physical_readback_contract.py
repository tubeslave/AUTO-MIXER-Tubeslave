import copy
from pathlib import Path

from backend.matrix_state_physical_readback_contract import (
    MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE,
    MATRIX_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    MATRIX_STATE_PHYSICAL_READBACK_SCHEMA_ID,
    MATRIX_STATE_REMEDIATION_ACTION_ID,
    MATRIX_STATE_SURFACE_ID,
    MINIMUM_MATRIX_EQ_BAND_COUNT,
    build_matrix_state_physical_readback_contract,
    validate_matrix_state_physical_readback_artifact,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


def _eq_bands() -> list[dict]:
    return [
        {
            "band": band,
            "frequency_hz": float(100 * band),
            "gain_db": 0.0,
            "q": 1.0,
        }
        for band in range(1, MINIMUM_MATRIX_EQ_BAND_COUNT + 1)
    ]


def _valid_artifact() -> dict:
    return {
        "type": MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE,
        "schema_id": MATRIX_STATE_PHYSICAL_READBACK_SCHEMA_ID,
        "surface_id": MATRIX_STATE_SURFACE_ID,
        "remediation_action_id": MATRIX_STATE_REMEDIATION_ACTION_ID,
        "captured_at": "2026-06-05T00:00:00Z",
        "source": "physical_readback",
        "console_model": "wing",
        "matrix_count": 1,
        "matrices": [
            {
                "matrix_id": 1,
                "matrix_label": "Matrix 1",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "controls": {
                    "fader_db": -4.0,
                    "mute": False,
                    "pan": 0.0,
                },
                "eq": {
                    "enabled": True,
                    "bands": _eq_bands(),
                },
                "dynamics": {
                    "enabled": True,
                    "model": "matrix_compressor",
                    "threshold_db": -18.0,
                    "ratio": 2.0,
                    "knee": 3.0,
                    "detector": "peak",
                    "attack_ms": 12.0,
                    "release_ms": 120.0,
                    "makeup_db": 1.0,
                    "mix_pct": 100.0,
                    "gain_reduction_db": 2.5,
                },
                "inserts": {
                    "pre_insert_on": False,
                    "pre_insert_slot": "none",
                    "pre_insert_status": "empty",
                    "post_insert_on": True,
                    "post_insert_slot": "limiter",
                    "post_insert_status": "active",
                    "post_insert_mode": "post_fader",
                    "post_insert_weight": 100.0,
                },
                "metering": {
                    "lufs": -25.0,
                    "true_peak_dbtp": -5.0,
                    "headroom_db": 4.0,
                },
                "coverage": {
                    "controls": True,
                    "eq": True,
                    "dynamics": True,
                    "inserts": True,
                    "metering": True,
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


def test_matrix_state_contract_describes_future_physical_artifact():
    contract = build_matrix_state_physical_readback_contract()

    assert contract["type"] == "matrix_state_physical_readback_contract"
    assert contract["method_id"] == MATRIX_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID
    assert contract["schema_id"] == MATRIX_STATE_PHYSICAL_READBACK_SCHEMA_ID
    assert contract["status"] == "contract_ready_physical_fixture_missing"
    assert contract["surface_id"] == MATRIX_STATE_SURFACE_ID
    assert contract["remediation_action_id"] == MATRIX_STATE_REMEDIATION_ACTION_ID
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
    assert artifact_contract["type"] == MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE
    assert "matrices" in artifact_contract["required_top_level_fields"]
    assert artifact_contract["required_control_fields"] == ["fader_db", "mute", "pan"]
    assert artifact_contract["required_eq_band_fields"] == [
        "band",
        "frequency_hz",
        "gain_db",
        "q",
    ]
    assert "gain_reduction_db" in artifact_contract["required_dynamics_fields"]
    assert "post_insert_weight" in artifact_contract["required_insert_fields"]
    assert "true_peak_dbtp" in artifact_contract["required_metering_fields"]
    assert "cache_matches_physical_readback" in artifact_contract[
        "required_coverage_flags"
    ]
    assert artifact_contract["minimum_eq_band_count"] == 8
    assert "physical_readback" in artifact_contract["allowed_sources"]
    assert "matrix_write_events" in artifact_contract[
        "forbidden_mutation_evidence_fields"
    ]
    assert contract["summary"]["physical_readback_surface_id"] == MATRIX_STATE_SURFACE_ID
    assert contract["summary"]["physical_fixture_gap_remains"] is True
    assert contract["summary"]["live_authority_granted"] is False
    assert "bus_group_matrix" in contract["summary"]["promotion_families_requiring_surface"]
    assert "routing" in contract["summary"]["promotion_families_requiring_surface"]


def test_matrix_state_contract_validates_minimal_physical_artifact():
    result = validate_matrix_state_physical_readback_artifact(_valid_artifact())

    assert result["status"] == "valid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is True
    assert result["live_authority_granted"] is False
    assert result["full_autonomous_ready"] is False
    assert result["matrix_count"] == 1
    assert result["violations"] == []
    assert result["evidence"]["source"] == "physical_readback"


def test_matrix_state_contract_rejects_live_or_cached_claims():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["source"] = "mixer_state_cache"
    artifact["full_autonomous_ready"] = True
    artifact["matrix_write_events"] = []
    artifact["safety"]["live_write"] = True
    artifact["matrices"][0]["status"] = "verified_from_cache"
    artifact["matrices"][0]["source"] = "mixer_state_cache"
    artifact["matrices"][0]["controls"]["mute"] = "false"
    artifact["matrices"][0]["coverage"]["cache_matches_physical_readback"] = False

    result = validate_matrix_state_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is False
    assert result["live_authority_granted"] is False
    assert result["full_autonomous_ready"] is False
    assert "source_not_physical_readback" in result["violations"]
    assert "top_level_full_autonomous_ready_not_false" in result["violations"]
    assert "safety_live_write_not_false" in result["violations"]
    assert "mutation_evidence_field_present:matrix_write_events" in result["violations"]
    assert "matrix_0_status_not_physical_verified" in result["violations"]
    assert "matrix_0_source_not_physical_readback" in result["violations"]
    assert "matrix_0_controls_mute_not_bool_or_0_1" in result["violations"]
    assert "matrix_0_coverage_cache_matches_physical_readback_not_true" in result[
        "violations"
    ]


def test_matrix_state_contract_rejects_incomplete_rows():
    artifact = copy.deepcopy(_valid_artifact())
    artifact["matrix_count"] = 2
    artifact["matrices"][0]["matrix_id"] = 0
    artifact["matrices"][0]["eq"]["bands"] = artifact["matrices"][0]["eq"]["bands"][:4]
    artifact["matrices"][0]["dynamics"]["ratio"] = "2:1"
    del artifact["matrices"][0]["metering"]["true_peak_dbtp"]
    del artifact["matrices"][0]["inserts"]["post_insert_weight"]

    result = validate_matrix_state_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert "matrix_count_mismatch" in result["violations"]
    assert "matrix_0_matrix_id_invalid" in result["violations"]
    assert "matrix_0_eq_bands_missing" in result["violations"]
    assert "matrix_0_dynamics_ratio_not_number" in result["violations"]
    assert "matrix_0_metering_missing_field:true_peak_dbtp" in result["violations"]
    assert "matrix_0_inserts_missing_field:post_insert_weight" in result["violations"]


def test_matrix_state_contract_is_transport_free():
    source = Path("backend/matrix_state_physical_readback_contract.py").read_text(
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
        "set_matrix_fader(",
        "set_matrix_mute(",
        "set_matrix_send(",
        "set_matrix_eq(",
        "set_matrix_dynamics(",
        "set_insert(",
    ]
    for token in forbidden_transport_tokens + forbidden_write_call_tokens:
        assert token not in source


def test_matrix_state_contract_is_registry_shadow_only_not_executor():
    method = get_processing_method(MATRIX_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert MATRIX_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    contract = snapshot["matrix_state_physical_readback_contract"]
    assert contract["type"] == "matrix_state_physical_readback_contract"
    assert contract["safety"]["live_write"] is False
    assert contract["summary"]["physical_fixture_gap_remains"] is True


def test_matrix_state_contract_is_console_read_goal_support_path():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert MATRIX_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID in row.method_ids
    assert "backend/matrix_state_physical_readback_contract.py" in row.support_paths
    assert row.current_live_write_kinds == ()
