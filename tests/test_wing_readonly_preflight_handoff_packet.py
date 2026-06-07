import json
from pathlib import Path

from tools import build_wing_readonly_preflight_handoff_packet as handoff
from tools import no_wing_release_check


def test_handoff_packet_builds_readonly_operator_sequence():
    packet = handoff.build_handoff_packet(generated_at="2026-06-07T00:00:00Z")

    assert packet["type"] == "wing_readonly_preflight_handoff_packet"
    assert packet["status"] == "operator_hardware_step_required"
    assert packet["read_only"] is True
    assert packet["live_write"] is False
    assert packet["mutates_console_state"] is False
    assert packet["queries_console"] is False
    assert packet["executes_tests"] is False
    assert packet["operator_required"] is True
    assert packet["expected_plan_required_for_real_run"] is True
    assert packet["physical_preflight_requires_existing_backend_connection"] is True
    assert packet["manual_review_required_before_readiness_update"] is True
    assert packet["execution_allowed_by_packet"] is False
    assert packet["readiness_matrix_update_allowed_by_packet"] is False
    assert packet["live_authority_granted"] is False
    assert packet["full_autonomous_ready"] is False
    assert packet["channels"] == [3]
    assert packet["value_types"] == ["fader", "gain"]
    assert packet["commands"]["preview_readonly_request_plan"] == [
        "python3",
        "tools/wing_readonly_preflight_runner.py",
        "--url",
        "ws://localhost:8765",
        "--channels",
        "3",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
        "--value-types",
        "fader,gain",
        "--require-physical-readback",
        "--artifact-root",
        "artifacts/wing_readonly_preflight",
        "--plan-only",
        "--plan-output",
        "artifacts/wing_readonly_preflight_plan/latest_plan.json",
    ]
    assert packet["commands"]["capture_readonly_preflight"] == [
        "python3",
        "tools/wing_readonly_preflight_runner.py",
        "--url",
        "ws://localhost:8765",
        "--channels",
        "3",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
        "--value-types",
        "fader,gain",
        "--require-physical-readback",
        "--artifact-root",
        "artifacts/wing_readonly_preflight",
        "--expected-plan",
        "artifacts/wing_readonly_preflight_plan/latest_plan.json",
    ]
    assert "--decision-template-output" in packet["commands"][
        "review_artifact_and_write_pending_decision_template"
    ]
    assert packet["commands"]["review_edited_decision_json"] == [
        "python3",
        "tools/review_wing_readonly_preflight_decisions.py",
        "--artifact-root",
        "artifacts/wing_readonly_preflight",
        "--decision-root",
        "artifacts/wing_readonly_preflight_review_decisions",
        "--output",
        "Docs/reports/WING_READONLY_PREFLIGHT_DECISION_REVIEW_LATEST.md",
    ]
    assert "confirm_no_live_authority" in packet["required_review_checklist_ids"]
    assert "confirm_operator_session_ready" in packet["required_review_checklist_ids"]
    assert packet["required_review_checklist_ids"] == list(
        no_wing_release_check.WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS
    )
    forbidden_request_types = set(packet["forbidden_live_request_types"])
    assert "apply_operator_proposal" in forbidden_request_types
    assert "create_operator_proposal" in forbidden_request_types
    assert "rollback_last_manual_write" in forbidden_request_types
    assert "emergency_stop" in forbidden_request_types
    assert "set_fader" in forbidden_request_types
    assert "set_gain" in forbidden_request_types
    assert "set_trim" in forbidden_request_types
    assert "set_input_trim" in forbidden_request_types
    assert "set_channel_trim" in forbidden_request_types
    assert "set_preamp_gain" in forbidden_request_types
    assert "set_eq" in forbidden_request_types
    assert "set_compressor" in forbidden_request_types
    assert "route_output" in forbidden_request_types
    assert "load_snap" in forbidden_request_types
    assert "save_snap" in forbidden_request_types
    assert "reset" in forbidden_request_types
    assert "mute" in forbidden_request_types
    assert "unmute" in forbidden_request_types
    assert "set_fx_parameter" in forbidden_request_types
    assert len(packet["forbidden_live_request_types"]) == len(forbidden_request_types)
    assert forbidden_request_types == set(
        no_wing_release_check.WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES
    )
    assert "wing_readonly_connection_proof" in packet["blockers_not_closed_by_packet"]


def test_handoff_packet_forbidden_live_request_types_matches_release_check_contract():
    packet = handoff.build_handoff_packet(generated_at="2026-06-07T00:00:00Z")
    expected_forbidden = set(
        no_wing_release_check.WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES
    )
    packet_forbidden = set(packet["forbidden_live_request_types"])

    assert packet_forbidden == expected_forbidden
    assert len(packet["forbidden_live_request_types"]) == len(packet_forbidden)


def test_handoff_packet_supports_custom_channel_confirmation():
    packet = handoff.build_handoff_packet(channels="2,4", generated_at="2026-06-07T00:00:00Z")

    assert packet["channels"] == [2, 4]
    preview = packet["commands"]["preview_readonly_request_plan"]
    assert preview[preview.index("--channels") + 1] == "2,4"
    assert preview[preview.index("--operator-confirm-channels") + 1] == "2,4"
    assert "--operator-confirm-session-ready" in preview
    command = packet["commands"]["capture_readonly_preflight"]
    assert command[command.index("--channels") + 1] == "2,4"
    assert command[command.index("--operator-confirm-channels") + 1] == "2,4"
    assert "--operator-confirm-session-ready" in command


def test_handoff_packet_rejects_invalid_channels():
    try:
        handoff.build_handoff_packet(channels="0")
    except ValueError as exc:
        assert str(exc) == "channels_must_be_positive"
    else:
        raise AssertionError("expected invalid channel to fail")


def test_handoff_packet_rejects_channels_outside_wing_input_range(tmp_path: Path):
    try:
        handoff.build_handoff_packet(channels="49")
    except ValueError as exc:
        assert str(exc) == "channels_must_be_in_1_48"
    else:
        raise AssertionError("expected out-of-range channel to fail")

    output = tmp_path / "handoff.md"
    assert handoff.main(["--channels", "49", "--output", str(output), "--quiet"]) == 2
    assert not output.exists()


def test_handoff_markdown_renders_commands_and_boundaries():
    packet = handoff.build_handoff_packet(generated_at="2026-06-07T00:00:00Z")
    markdown = handoff.render_markdown(packet)

    assert "WING Read-Only Preflight Handoff Packet" in markdown
    assert "live_write: `false`" in markdown
    assert "expected_plan_required_for_real_run: `true`" in markdown
    assert "physical_preflight_requires_existing_backend_connection: `true`" in markdown
    assert "manual_review_required_before_readiness_update: `true`" in markdown
    assert "preview_readonly_request_plan" in markdown
    assert "--plan-only" in markdown
    assert "--operator-confirm-session-ready" in markdown
    assert "capture_readonly_preflight" in markdown
    assert "review_edited_decision_json" in markdown
    assert "pending_review_decision_template.json" in markdown
    assert "confirm_operator_session_ready" in markdown
    assert "full_autonomous_ready: `false`" in markdown


def test_handoff_tool_main_writes_markdown_and_json(tmp_path: Path):
    markdown_output = tmp_path / "handoff.md"
    json_output = tmp_path / "handoff.json"

    assert handoff.main(["--output", str(markdown_output), "--quiet"]) == 0
    assert "WING Read-Only Preflight Handoff Packet" in markdown_output.read_text(
        encoding="utf-8"
    )

    assert handoff.main(["--format", "json", "--output", str(json_output), "--quiet"]) == 0
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["type"] == "wing_readonly_preflight_handoff_packet"
    assert payload["expected_plan_required_for_real_run"] is True
    assert payload["physical_preflight_requires_existing_backend_connection"] is True
    assert payload["manual_review_required_before_readiness_update"] is True
    assert payload["live_authority_granted"] is False


def test_handoff_tool_main_fails_closed_for_invalid_channel(tmp_path: Path):
    output = tmp_path / "handoff.md"

    assert handoff.main(["--channels", "bad", "--output", str(output), "--quiet"]) == 2
    assert not output.exists()


def test_handoff_tool_is_transport_free():
    source = Path("tools/build_wing_readonly_preflight_handoff_packet.py").read_text(
        encoding="utf-8"
    )

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
        ".send(",
        "mixer_client",
        "set_fader(",
        "set_gain(",
        "apply_operator_proposal(",
    ]
    for token in forbidden:
        assert token not in source
