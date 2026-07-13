from unittest.mock import AsyncMock, MagicMock

import pytest

from handlers import arrangement_automation_handlers
from server import AutoMixerServer


def _bare_server() -> AutoMixerServer:
    server = AutoMixerServer.__new__(AutoMixerServer)
    server.safe_gain_calibrator = None
    server.auto_soundcheck_engine = None
    server.live_input_trim_controller = None
    server.arrangement_automation_controller = None
    return server


def test_real_server_gain_recommendation_endpoints_are_read_only_and_available():
    server = _bare_server()
    server.safe_gain_calibrator = MagicMock()
    server.safe_gain_calibrator.get_recommendations.return_value = {
        3: {"channel": 3, "recommended_target_trim_db": -2.0}
    }
    server.live_input_trim_controller = MagicMock()
    server.live_input_trim_controller.get_recommendations.return_value = {
        7: {"channel": 7, "recommended_target_trim_db": -4.0}
    }

    safe_gain = server.get_gain_staging_recommendations()
    live_trim = server.get_live_input_trim_recommendations()

    assert safe_gain["recommendation_count"] == 1
    assert safe_gain["recommendations"][3]["recommended_target_trim_db"] == -2.0
    assert safe_gain["dry_run"] is True
    assert safe_gain["live_apply_enabled"] is False
    assert safe_gain["live_authority_granted"] is False
    assert live_trim["recommendation_count"] == 1
    assert live_trim["recommendations"][7]["recommended_target_trim_db"] == -4.0
    assert live_trim["dry_run"] is True
    assert live_trim["live_authority_granted"] is False


def test_real_server_gain_recommendation_endpoints_fail_closed_when_inactive_or_broken():
    server = _bare_server()
    assert server.get_gain_staging_recommendations()["recommendations"] == {}
    assert server.get_live_input_trim_recommendations()["recommendations"] == {}

    server.safe_gain_calibrator = MagicMock()
    server.safe_gain_calibrator.get_recommendations.side_effect = RuntimeError("boom")
    payload = server.get_gain_staging_recommendations()
    assert payload["status"] == "unavailable"
    assert payload["reason"] == "recommendations_unavailable"
    assert "boom" not in str(payload)


@pytest.mark.asyncio
async def test_arrangement_automation_websocket_surface_is_explicitly_fail_closed():
    server = _bare_server()
    server.send_to_client = AsyncMock()
    handlers = arrangement_automation_handlers.register_handlers(server)

    await handlers["start_arrangement_automation"](
        "ws",
        {"channels": [1], "settings": {"analysis_only_mode": False}},
    )
    started = server.send_to_client.await_args.args[1]
    assert started["status"] == "blocked"
    assert started["reason"] == "arrangement_automation_not_promoted"
    assert started["live_apply_enabled"] is False
    assert started["live_authority_granted"] is False
    assert server.arrangement_automation_controller is None

    await handlers["set_arrangement_automation_live_apply"](
        "ws",
        {
            "live_apply_enabled": True,
            "analysis_only_mode": False,
            "confirm_live_apply": True,
        },
    )
    enabled = server.send_to_client.await_args.args[1]
    assert enabled["status"] == "blocked"
    assert enabled["live_apply_enabled"] is False
    assert enabled["operation"] == "set_arrangement_automation_live_apply"

    await handlers["get_arrangement_automation_status"]("ws", {})
    status = server.send_to_client.await_args.args[1]
    assert status["active"] is False
    assert status["arrangement_automation_state"]["status"] == "blocked"
