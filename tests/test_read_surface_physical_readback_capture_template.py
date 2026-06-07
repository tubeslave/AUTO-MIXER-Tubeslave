import json
from pathlib import Path

from tools import build_read_surface_physical_readback_capture_template as template_tool


def _packet_with_two_targets() -> dict:
    return {
        "type": "read_surface_physical_readback_capture_packet",
        "status": "operator_capture_required",
        "source_method_id": "read_surface_physical_readback_capture_plan",
        "blocked_by": ["operator_must_capture_and_review_physical_readback_artifact"],
        "next_physical_readback_surface_id": "raw_mixer_state_cache",
        "remaining_capture_target_count": 2,
        "remaining_capture_targets": [
            {
                "surface_id": "raw_mixer_state_cache",
                "remediation_action_id": "remediate_raw_mixer_state_cache",
                "target_console_surface": "all_cached_addresses",
                "read_mode": "cached_state",
                "requires_connection": True,
                "contract_method_id": "raw_mixer_state_physical_readback_contract",
                "required_artifact_contracts": [
                    "physical_readback_fixture",
                    "raw_mixer_state_physical_readback_contract",
                ],
                "expected_artifacts": [
                    "Raw mixer-state physical readback artifact.",
                ],
                "blockers_to_clear": [
                    "physical_readback_fixture_missing",
                    "hardware_parity_unproven",
                ],
            },
            {
                "surface_id": "channel_identity_and_basic_controls",
                "remediation_action_id": "remediate_channel_identity_and_basic_controls",
                "target_console_surface": "channels",
                "read_mode": "product_snapshot",
                "requires_connection": True,
                "contract_method_id": "channel_basic_controls_physical_readback_contract",
                "required_artifact_contracts": [
                    "physical_readback_fixture",
                    "channel_basic_controls_physical_readback_contract",
                ],
                "expected_artifacts": [
                    "Channel identity/basic-control physical readback artifact.",
                ],
                "blockers_to_clear": [
                    "physical_readback_fixture_missing",
                    "hardware_parity_unproven",
                ],
            },
        ],
    }


def test_capture_template_builds_next_target_without_authority():
    template = template_tool.build_capture_template(
        capture_packet=_packet_with_two_targets(),
        generated_at="2026-06-07T00:00:00Z",
    )

    assert template["type"] == "read_surface_physical_readback_artifact_capture_template"
    assert template["status"] == "operator_capture_template_ready"
    assert template["source_packet_status"] == "operator_capture_required"
    assert template["read_only"] is True
    assert template["live_write"] is False
    assert template["mutates_console_state"] is False
    assert template["queries_console"] is False
    assert template["executes_tests"] is False
    assert template["operator_required"] is True
    assert template["execution_allowed_by_template"] is False
    assert template["accepted_as_physical_fixture_candidate"] is False
    assert template["fixture_closure_allowed_by_template"] is False
    assert template["live_widening_allowed"] is False
    assert template["live_authority_granted"] is False
    assert template["full_autonomous_ready"] is False
    assert template["all_remaining_target_count"] == 2
    assert template["capture_template_count"] == 1
    assert template["selected_surface_id"] == "raw_mixer_state_cache"
    assert "operator_must_replace_capture_template_placeholders" in template["blocked_by"]

    record = template["capture_templates"][0]
    assert record["surface_id"] == "raw_mixer_state_cache"
    assert record["remediation_action_id"] == "remediate_raw_mixer_state_cache"
    assert record["required_contract_method_id"] == (
        "raw_mixer_state_physical_readback_contract"
    )
    assert record["artifact_source_id"] == "<path-to-operator-captured-json>"
    assert record["captured_at"].startswith("<UTC ISO timestamp")
    assert record["source"].startswith("<physical_readback")
    assert record["physical_readback_payload"].startswith("<replace with JSON payload")
    payload_template = record["physical_readback_payload_template"]
    assert payload_template["template_status"] == (
        "operator_must_fill_from_physical_readback"
    )
    assert payload_template["required_contract_method_id"] == (
        "raw_mixer_state_physical_readback_contract"
    )
    assert payload_template["artifact_type"] == "raw_mixer_state_physical_readback"
    assert payload_template["schema_id"] == "raw_mixer_state_physical_readback.v1"
    assert payload_template["required_field_sets"]["required_false_safety_flags"] == [
        "live_write",
        "mutates_console_state",
        "live_widening_allowed",
        "full_autonomous_ready",
    ]
    assert payload_template["required_top_level_fields"] == [
        "type",
        "schema_id",
        "surface_id",
        "remediation_action_id",
        "captured_at",
        "source",
        "console_model",
        "address_count",
        "readback_rows",
        "safety",
    ]
    payload = payload_template["payload"]
    assert payload["type"] == "raw_mixer_state_physical_readback"
    assert payload["schema_id"] == "raw_mixer_state_physical_readback.v1"
    assert payload["surface_id"] == "raw_mixer_state_cache"
    assert payload["remediation_action_id"] == "remediate_raw_mixer_state_cache"
    assert payload["captured_at"].startswith("<UTC ISO timestamp")
    assert payload["source"].startswith("<physical_readback")
    assert payload["address_count"] == "<integer address_count>"
    assert payload["readback_rows"][0]["address"] == "<operator-captured address>"
    assert payload["readback_rows"][0]["status"] == "<verified_from_physical_readback>"
    assert payload["safety"]["read_only"] is True
    assert payload["safety"]["live_write"] is False
    assert payload["safety"]["accepted_as_physical_fixture_candidate"] is False
    assert record["safety"]["read_only"] is True
    assert record["safety"]["live_write"] is False
    assert record["safety"]["accepted_as_physical_fixture_candidate"] is False
    assert record["safety"]["fixture_closure_allowed_by_template"] is False
    assert record["safety"]["live_authority_granted"] is False
    assert record["safety"]["full_autonomous_ready"] is False


def test_capture_template_can_select_surface_or_all_targets():
    packet = _packet_with_two_targets()
    selected = template_tool.build_capture_template(
        capture_packet=packet,
        surface_id="channel_identity_and_basic_controls",
        generated_at="2026-06-07T00:00:00Z",
    )
    all_targets = template_tool.build_capture_template(
        capture_packet=packet,
        include_all_targets=True,
        generated_at="2026-06-07T00:00:00Z",
    )
    missing = template_tool.build_capture_template(
        capture_packet=packet,
        surface_id="dlive_state",
        generated_at="2026-06-07T00:00:00Z",
    )

    assert selected["selected_surface_id"] == "channel_identity_and_basic_controls"
    assert selected["capture_template_count"] == 1
    assert selected["capture_templates"][0]["surface_id"] == (
        "channel_identity_and_basic_controls"
    )
    selected_payload_template = selected["capture_templates"][0][
        "physical_readback_payload_template"
    ]
    assert selected_payload_template["artifact_type"] == (
        "channel_basic_controls_physical_readback"
    )
    assert selected_payload_template["payload"]["channel_count"] == (
        "<integer channel_count>"
    )
    assert selected_payload_template["payload"]["channels"][0]["identity"] == {
        "source_name": "<operator-captured source_name>",
        "source_role": "<operator-captured source_role>",
        "input_source": "<operator-captured input_source>",
    }
    assert selected_payload_template["payload"]["channels"][0]["controls"] == {
        "fader_db": "<operator-captured fader_db>",
        "trim_db": "<operator-captured trim_db>",
        "mute": "<operator-captured mute>",
        "pan": "<operator-captured pan>",
    }
    assert all_targets["include_all_targets"] is True
    assert all_targets["capture_template_count"] == 2
    assert [record["surface_id"] for record in all_targets["capture_templates"]] == [
        "raw_mixer_state_cache",
        "channel_identity_and_basic_controls",
    ]
    assert [
        record["physical_readback_payload_template"]["required_contract_method_id"]
        for record in all_targets["capture_templates"]
    ] == [
        "raw_mixer_state_physical_readback_contract",
        "channel_basic_controls_physical_readback_contract",
    ]
    assert missing["status"] == "operator_capture_target_missing"
    assert missing["capture_templates"] == []
    assert missing["blocked_by"] == [
        "requested_surface_id_has_no_remaining_capture_target"
    ]


def test_capture_template_reports_complete_packet_without_records():
    template = template_tool.build_capture_template(
        capture_packet={
            "type": "read_surface_physical_readback_capture_packet",
            "status": "read_surface_physical_readback_complete",
            "source_method_id": "read_surface_physical_readback_capture_plan",
            "remaining_capture_target_count": 0,
            "remaining_capture_targets": [],
            "blocked_by": [],
        },
        generated_at="2026-06-07T00:00:00Z",
    )

    assert template["status"] == "read_surface_physical_readback_complete"
    assert template["capture_template_count"] == 0
    assert template["capture_templates"] == []
    assert template["live_authority_granted"] is False


def test_capture_template_tool_main_writes_json_and_markdown(tmp_path: Path):
    json_output = tmp_path / "capture-template.json"
    markdown_output = tmp_path / "capture-template.md"

    assert template_tool.main(["--output", str(json_output), "--quiet"]) == 0
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["type"] == "read_surface_physical_readback_artifact_capture_template"
    assert payload["live_authority_granted"] is False

    assert (
        template_tool.main(
            ["--format", "markdown", "--output", str(markdown_output), "--quiet"]
        )
        == 0
    )
    markdown = markdown_output.read_text(encoding="utf-8")
    assert "Read Surface Physical Readback Capture Template" in markdown
    assert "live_authority_granted: `false`" in markdown
    assert "physical_readback_payload" in markdown
    assert "physical_readback_payload_template" in markdown


def test_capture_template_tool_is_transport_free():
    source = Path(
        "tools/build_read_surface_physical_readback_capture_template.py"
    ).read_text(encoding="utf-8")

    forbidden = [
        "WingClient",
        "DLiveClient",
        "EnhancedOSCClient",
        "import socket",
        "websockets",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "apply_operator_proposal",
    ]
    for token in forbidden:
        assert token not in source
