from pathlib import Path

from backend.dlive_state_physical_readback_contract import (
    DLIVE_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    build_dlive_state_physical_readback_contract,
    validate_dlive_state_physical_readback_artifact,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


def valid_dlive_state_artifact():
    coverage = {
        "channel_identity": True,
        "fader_scale": True,
        "mute_state": True,
        "raw_midi_state": True,
        "midi_base_channel": True,
        "wing_parity_mapping": True,
        "cache_matches_physical_readback": True,
    }
    return {
        "type": "dlive_state_physical_readback",
        "schema_id": "dlive_state_physical_readback.v1",
        "surface_id": "dlive_state",
        "remediation_action_id": "remediate_dlive_state",
        "captured_at": "2026-06-05T00:00:00Z",
        "source": "dlive_physical_readback",
        "console_model": "Allen & Heath dLive",
        "channel_count": 1,
        "channels": [
            {
                "channel": 1,
                "cached_channel_name": "Lead Vox",
                "fader_raw_14bit": 10922,
                "fader_db": 0.0,
                "mute": False,
                "raw_note_velocity": 0,
                "midi_base_channel": 1,
                "readback_status": "verified_from_physical_readback",
                "readback_source": "dlive_physical_readback",
                "captured_at": "2026-06-05T00:00:00Z",
                "coverage": dict(coverage),
            }
        ],
        "midi_state": {
            "midi_base_channel": 1,
            "raw_state_key_count": 4,
            "raw_note_event_count": 1,
            "raw_cc_event_count": 1,
            "raw_pitchbend_event_count": 1,
            "tcp_session_status": "verified_passive_state",
            "midi_transport_status": "verified_passive_state",
            "readback_status": "verified_from_physical_readback",
            "readback_source": "dlive_physical_readback",
            "captured_at": "2026-06-05T00:00:00Z",
            "coverage": dict(coverage),
        },
        "parity": {
            "wing_surface_equivalent": "channel_identity_and_basic_controls",
            "mapping_status": "partial_mapping_verified",
            "channel_identity_parity": "verified",
            "fader_scale_parity": "verified",
            "mute_state_parity": "verified",
            "midi_velocity_parity": "verified",
            "unsupported_field_policy": "block_autonomous_promotion",
            "coverage": dict(coverage),
        },
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "midi_send_allowed": False,
            "tcp_send_allowed": False,
            "dlive_write_allowed": False,
            "dlive_apply_allowed": False,
        },
    }


def test_dlive_state_contract_describes_future_physical_artifact():
    contract = build_dlive_state_physical_readback_contract()

    assert contract["type"] == "dlive_state_physical_readback_contract"
    assert contract["status"] == "contract_ready_physical_fixture_missing"
    assert contract["surface_id"] == "dlive_state"
    assert contract["remediation_action_id"] == "remediate_dlive_state"
    assert contract["safety"]["live_write"] is False
    assert contract["safety"]["midi_send_allowed"] is False
    assert contract["safety"]["tcp_send_allowed"] is False
    assert contract["safety"]["dlive_write_allowed"] is False
    assert contract["artifact_contract"]["minimum_channel_count"] == 1
    assert "fader_raw_14bit" in contract["artifact_contract"]["required_channel_fields"]
    assert "midi_base_channel" in contract["artifact_contract"]["required_midi_state_fields"]
    assert "wing_surface_equivalent" in contract["artifact_contract"]["required_parity_fields"]
    assert "wing_parity_mapping" in contract["artifact_contract"]["required_coverage_flags"]
    assert "midi_send_events" in contract["artifact_contract"][
        "forbidden_mutation_evidence_fields"
    ]
    assert contract["summary"]["physical_fixture_gap_remains"] is True
    assert contract["summary"]["accepted_as_physical_fixture_candidate"] is False
    assert contract["summary"]["live_authority_granted"] is False
    assert "performance_adaptation" in contract["summary"][
        "promotion_families_requiring_surface"
    ]


def test_dlive_state_contract_validates_minimal_physical_artifact():
    result = validate_dlive_state_physical_readback_artifact(
        valid_dlive_state_artifact()
    )

    assert result["method_id"] == DLIVE_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID
    assert result["status"] == "valid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is True
    assert result["live_authority_granted"] is False
    assert result["full_autonomous_ready"] is False
    assert result["channel_count"] == 1
    assert result["violations"] == []


def test_dlive_state_contract_rejects_live_or_cached_claims():
    artifact = valid_dlive_state_artifact()
    artifact["source"] = "mixer_state_cache"
    artifact["live_write"] = True
    artifact["safety"]["midi_send_allowed"] = True
    artifact["channels"][0]["readback_source"] = "mixer_state_cache"
    artifact["channels"][0]["midi_send_events"] = [{"channel": 1}]

    result = validate_dlive_state_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert result["accepted_as_physical_fixture_candidate"] is False
    assert result["live_authority_granted"] is False
    assert "source_not_physical_readback" in result["violations"]
    assert "safety_midi_send_allowed_not_false" in result["violations"]
    assert "top_level_live_write_not_false" in result["violations"]
    assert "channel_0_readback_source_not_physical_readback" in result["violations"]
    assert (
        "mutation_evidence_field_present:channels[0].midi_send_events"
        in result["violations"]
    )


def test_dlive_state_contract_rejects_incomplete_rows_and_parity():
    artifact = valid_dlive_state_artifact()
    artifact["channel_count"] = 2
    artifact["channels"][0].pop("fader_raw_14bit")
    artifact["channels"][0]["coverage"]["wing_parity_mapping"] = False
    artifact["parity"].pop("wing_surface_equivalent")

    result = validate_dlive_state_physical_readback_artifact(artifact)

    assert result["status"] == "invalid_physical_readback_artifact"
    assert "channel_count_mismatch" in result["violations"]
    assert "channel_0_field_missing_field:fader_raw_14bit" in result["violations"]
    assert "channel_0_fader_raw_14bit_invalid" in result["violations"]
    assert "channel_0_coverage_wing_parity_mapping_not_true" in result["violations"]
    assert "parity_field_missing_field:wing_surface_equivalent" in result["violations"]
    assert "parity_wing_surface_equivalent_missing" in result["violations"]


def test_dlive_state_contract_module_stays_transport_free():
    source = Path("backend/dlive_state_physical_readback_contract.py").read_text(
        encoding="utf-8"
    )

    forbidden = (
        "socket",
        "requests",
        "mido",
        "rtmidi",
        "dlive_client",
        "wing_client",
        "send_midi",
        "send(",
        "_send_raw",
    )
    for token in forbidden:
        assert token not in source


def test_dlive_state_contract_is_shadow_only_not_executor():
    method = get_processing_method("dlive_state_physical_readback_contract")

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert "dlive_state_physical_readback_contract" not in main.executor_method_ids


def test_dlive_state_contract_registry_snapshot_is_read_only():
    snapshot = build_processing_registry_snapshot(include_methods=False)
    contract = snapshot["dlive_state_physical_readback_contract"]

    assert contract["type"] == "dlive_state_physical_readback_contract"
    assert contract["safety"]["live_write"] is False
    assert contract["safety"]["midi_send_allowed"] is False
    assert contract["safety"]["tcp_send_allowed"] is False
    assert contract["safety"]["dlive_write_allowed"] is False
    assert contract["summary"]["physical_fixture_gap_remains"] is True


def test_dlive_state_contract_supports_console_read_goal_without_live_scope():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert "dlive_state_physical_readback_contract" in row.method_ids
    assert "backend/dlive_state_physical_readback_contract.py" in row.support_paths
    assert row.current_live_write_kinds == ()
