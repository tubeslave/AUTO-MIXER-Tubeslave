"""Tests for read-only gain staging recommendation websocket handlers."""

import pytest

import handlers.gain_staging_handlers as gain_staging_handlers


class FakeServer:
    def __init__(self):
        self.sent_messages = []

    async def send_to_client(self, websocket, message):
        self.sent_messages.append((websocket, message))

    def get_gain_staging_recommendations(self):
        return {
            "active": True,
            "source": "safe_gain_calibrator",
            "live_apply_enabled": False,
            "dry_run": True,
            "recommendations": {
                5: {
                    "channel": 5,
                    "recommended_target_trim_db": -3.0,
                    "delta_db": 1.5,
                }
            },
        }

    def get_live_input_trim_recommendations(self):
        return {
            "active": True,
            "source": "live_input_trim_controller",
            "live_apply_enabled": False,
            "dry_run": True,
            "recommendations": {
                7: {
                    "channel": 7,
                    "recommended_target_trim_db": -2.0,
                    "delta_db": -0.5,
                }
            },
        }


@pytest.mark.asyncio
async def test_get_gain_staging_recommendations_returns_read_only_payload():
    server = FakeServer()
    handlers = gain_staging_handlers.register_handlers(server)

    await handlers["get_gain_staging_recommendations"](
        "ws",
        {"request_id": "req-gain"},
    )

    message = server.sent_messages[-1][1]
    assert message["request_id"] == "req-gain"
    assert message["type"] == "gain_staging_recommendations"
    assert message["active"] is True
    assert message["source"] == "safe_gain_calibrator"
    assert message["dry_run"] is True
    assert message["recommendations"][5]["recommended_target_trim_db"] == -3.0


@pytest.mark.asyncio
async def test_get_live_input_trim_recommendations_returns_read_only_payload():
    server = FakeServer()
    handlers = gain_staging_handlers.register_handlers(server)

    await handlers["get_live_input_trim_recommendations"](
        "ws",
        {"request_id": "req-live-trim"},
    )

    message = server.sent_messages[-1][1]
    assert message["request_id"] == "req-live-trim"
    assert message["type"] == "live_input_trim_recommendations"
    assert message["active"] is True
    assert message["source"] == "live_input_trim_controller"
    assert message["dry_run"] is True
    assert message["recommendations"][7]["delta_db"] == -0.5
