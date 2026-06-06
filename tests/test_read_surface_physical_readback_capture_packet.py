import json
from pathlib import Path

from tools import build_read_surface_physical_readback_capture_packet as packet_tool


def test_capture_packet_wraps_read_surface_capture_plan_without_live_authority():
    packet = packet_tool.build_capture_packet(generated_at="2026-06-07T00:00:00Z")

    assert packet["type"] == "read_surface_physical_readback_capture_packet"
    assert packet["status"] == "operator_capture_required"
    assert packet["source_plan_status"] == "operator_capture_required"
    assert packet["read_only"] is True
    assert packet["live_write"] is False
    assert packet["mutates_console_state"] is False
    assert packet["queries_console"] is False
    assert packet["executes_tests"] is False
    assert packet["operator_required"] is True
    assert packet["execution_allowed_by_packet"] is False
    assert packet["live_widening_allowed"] is False
    assert packet["live_authority_granted"] is False
    assert packet["full_autonomous_ready"] is False
    assert packet["next_physical_readback_surface_id"] == "raw_mixer_state_cache"
    assert packet["next_physical_readback_action_id"] == "remediate_raw_mixer_state_cache"
    assert packet["target_action_consistency"] is True
    assert packet["capture_step_count"] == 1
    assert "physical_readback_fixture" in packet["required_artifact_contracts"]
    assert "operator_must_capture_and_review_physical_readback_artifact" in packet["blocked_by"]
    assert packet["step"]["contract_method_id"] == "raw_mixer_state_physical_readback_contract"
    assert any("do not transmit OSC" in item for item in packet["step"]["forbidden_actions"])


def test_capture_packet_reports_complete_plan_without_steps():
    packet = packet_tool.build_capture_packet(
        generated_at="2026-06-07T00:00:00Z",
        capture_plan={
            "type": "read_surface_physical_readback_capture_plan",
            "method_id": "read_surface_physical_readback_capture_plan",
            "status": "read_surface_physical_readback_complete",
            "safety": {
                "read_only": True,
                "live_write": False,
                "mutates_console_state": False,
                "queries_console": False,
                "executes_tests": False,
                "operator_required": True,
                "execution_allowed_by_plan": False,
                "live_widening_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
            },
            "summary": {
                "physical_fixture_missing_surface_ids": [],
                "next_physical_readback_surface_id": None,
                "next_physical_readback_action_id": None,
                "target_action_consistency": False,
                "required_artifact_contracts": [],
                "capture_step_count": 0,
                "blocked_by": [],
            },
            "steps": [],
        },
    )

    assert packet["status"] == "read_surface_physical_readback_complete"
    assert packet["capture_step_count"] == 0
    assert packet["step"] == {}
    assert packet["live_authority_granted"] is False


def test_capture_packet_markdown_renders_boundary_target_and_forbidden_actions():
    packet = packet_tool.build_capture_packet(generated_at="2026-06-07T00:00:00Z")
    markdown = packet_tool.render_markdown(packet)

    assert "Read Surface Physical Readback Capture Packet" in markdown
    assert "live_write: `false`" in markdown
    assert "execution_allowed_by_packet: `false`" in markdown
    assert "raw_mixer_state_cache" in markdown
    assert "raw_mixer_state_physical_readback_contract" in markdown
    assert "do not transmit OSC" in markdown
    assert "tests/test_read_surface_physical_readback_capture_plan.py" in markdown


def test_capture_packet_tool_main_writes_markdown_and_json(tmp_path: Path):
    markdown_output = tmp_path / "capture-packet.md"
    json_output = tmp_path / "capture-packet.json"

    assert packet_tool.main(["--output", str(markdown_output), "--quiet"]) == 0
    assert "Read Surface Physical Readback Capture Packet" in markdown_output.read_text(
        encoding="utf-8"
    )

    assert packet_tool.main(["--format", "json", "--output", str(json_output), "--quiet"]) == 0
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["type"] == "read_surface_physical_readback_capture_packet"
    assert payload["live_authority_granted"] is False


def test_capture_packet_tool_is_transport_free():
    source = Path("tools/build_read_surface_physical_readback_capture_packet.py").read_text(
        encoding="utf-8"
    )

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
