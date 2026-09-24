"""Composition-root tests for WING live capture configuration in websocket handlers."""

import importlib.util
import os
import sys
from unittest.mock import AsyncMock

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, BACKEND)

_spec = importlib.util.spec_from_file_location(
    "soundcheck_handlers_composition_target",
    os.path.join(BACKEND, "handlers", "soundcheck_handlers.py"),
)
_module = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_module)
register_handlers = _module.register_handlers


class FakeEngine:
    def start_async(self):
        pass

    def stop(self):
        pass


class FakeLiveService:
    def __init__(self):
        self.requests = []
        self.engine = FakeEngine()
        self.active = False

    def is_active(self):
        return self.active

    def start(self, request, **callbacks):
        self.requests.append(request)
        self.active = True
        return self.engine

    def stop(self):
        self.active = False
        return True

    def get_status(self):
        return {"state": "discover" if self.active else "idle"}


class DummyServer:
    def __init__(self):
        self.config = {
            "mixer": {"type": "wing", "ip": "10.0.0.5", "port": 2223},
            "audio": {"device_name": "WING USB"},
        }
        self._live_soundcheck_service = FakeLiveService()
        self.auto_soundcheck_running = False
        self.auto_soundcheck_observe_only = False
        self.sent = []
        self.send_to_client = AsyncMock(side_effect=self._send)
        self.broadcast = AsyncMock()

    async def _send(self, websocket, payload):
        self.sent.append(payload)


def _capture_config():
    return {
        "enabled": True,
        "roles": {"1": "lead_vocal", "2": "guitar"},
        "channel_names": {"1": "Lead Vocal", "2": "Guitar"},
        "main_tap": {
            "left_channel": 47,
            "right_channel": 48,
            "routes": [
                {"usb_slot": 47, "source_group": "MAIN", "source_channel": 1},
                {"usb_slot": 48, "source_group": "MAIN", "source_channel": 1},
            ],
        },
    }


@pytest.mark.asyncio
async def test_handler_passes_explicit_roles_and_patch_contract_to_live_service():
    server = DummyServer()
    server.config["live_soundcheck"] = {"capture_bridge": _capture_config()}
    handlers = register_handlers(server)

    await handlers["start_auto_engine"]("ws", {
        "mode": "observe",
        "channels": [1, 2],
    })

    request = server._live_soundcheck_service.requests[-1]
    assert request.num_channels == 48
    assert request.selected_channels == [1, 2]
    assert request.capture_bridge is not None
    assert request.capture_bridge.roles == {1: "lead_vocal", 2: "guitar"}
    assert request.capture_bridge.patch_contract.tap.channels == (47, 48)
    assert server.sent[-1]["capture_bridge_configured"] is True


@pytest.mark.asyncio
async def test_handler_blocks_invalid_explicit_capture_config_before_service_start():
    server = DummyServer()
    config = _capture_config()
    del config["roles"]["2"]
    server.config["live_soundcheck"] = {"capture_bridge": config}
    handlers = register_handlers(server)

    await handlers["start_auto_engine"]("ws", {
        "mode": "observe",
        "channels": [1, 2],
    })

    assert server._live_soundcheck_service.requests == []
    assert server.sent[-1]["status"] == "blocked"
    assert "explicit roles" in server.sent[-1]["error"]


@pytest.mark.asyncio
async def test_handler_does_not_invent_main_tap_when_config_is_absent():
    server = DummyServer()
    handlers = register_handlers(server)

    await handlers["start_auto_engine"]("ws", {
        "mode": "observe",
        "channels": [1, 2],
    })

    request = server._live_soundcheck_service.requests[-1]
    assert request.capture_bridge is None
    assert request.num_channels == 2
    assert server.sent[-1]["capture_bridge_configured"] is False
