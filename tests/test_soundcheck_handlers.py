"""Tests for backend/handlers/soundcheck_handlers.py."""

import asyncio
import importlib.util
import os
import sys
from unittest.mock import AsyncMock

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, BACKEND)

# Load this handler directly so the focused live-runtime test does not import
# every legacy handler via handlers/__init__.py.  That package-wide import fanout
# is itself part of the composition-root modernization work.
_spec = importlib.util.spec_from_file_location(
    "soundcheck_handlers_test_target",
    os.path.join(BACKEND, "handlers", "soundcheck_handlers.py"),
)
_module = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_module)
register_handlers = _module.register_handlers


class DummyServer:
    def __init__(self):
        self.config = {
            "mixer": {"type": "wing", "ip": "10.0.0.5", "port": 2223},
            "audio": {"device_name": "Test Device"},
        }
        self.auto_soundcheck_running = False
        self.auto_soundcheck_observe_only = False
        self.auto_soundcheck_engine = None
        self.sent_messages = []
        self.broadcast_messages = []
        self.send_to_client = AsyncMock(side_effect=self._capture_send)
        self.broadcast = AsyncMock(side_effect=self._capture_broadcast)
        self.start_auto_soundcheck = AsyncMock()

    async def _capture_send(self, websocket, payload):
        self.sent_messages.append((websocket, payload))

    async def _capture_broadcast(self, payload):
        self.broadcast_messages.append(payload)


class FakeEngine:
    def __init__(self):
        self.state = type("State", (), {"value": "idle"})()
        self.started = False

    def start_async(self):
        self.started = True

    def stop(self):
        self.state = type("State", (), {"value": "stopped"})()


class FakeLiveService:
    def __init__(self):
        self.requests = []
        self.callbacks = []
        self.engine = FakeEngine()

    def create_engine(self, request, **callbacks):
        self.requests.append(request)
        self.callbacks.append(callbacks)
        return self.engine


@pytest.mark.asyncio
async def test_get_auto_soundcheck_status_includes_legacy_aliases():
    server = DummyServer()
    handlers = register_handlers(server)

    await handlers["get_auto_soundcheck_status"]("ws", {})

    _, payload = server.sent_messages[-1]
    assert payload["type"] == "auto_soundcheck_status"
    assert payload["is_running"] is False
    assert payload["running"] is False
    assert payload["observe_only"] is False
    assert payload["step_progress"] == 0
    assert payload["progress"] == 0


@pytest.mark.asyncio
async def test_get_auto_soundcheck_status_forwards_session_report_summary():
    server = DummyServer()

    class StatusEngine:
        def get_status(self):
            return {
                "state": "running",
                "autofoh_session_report_summary": "AutoFOH session report: events=4; sent=1; blocked=3; guard_blocks=2",
            }

    server.auto_soundcheck_engine = StatusEngine()
    server.auto_soundcheck_running = True
    handlers = register_handlers(server)

    await handlers["get_auto_soundcheck_status"]("ws", {})

    _, payload = server.sent_messages[-1]
    assert payload["type"] == "auto_soundcheck_status"
    assert payload["autofoh_session_report_summary"].startswith("AutoFOH session report:")


@pytest.mark.asyncio
async def test_start_auto_soundcheck_routes_construction_through_live_runtime():
    server = DummyServer()
    service = FakeLiveService()
    server._live_soundcheck_service = service
    handlers = register_handlers(server)

    await handlers["start_auto_soundcheck"]("ws", {
        "device_id": "dev1",
        "channels": [1, 2],
        "observe_only": True,
    })

    request = service.requests[-1]
    assert request.selected_channels == [1, 2]
    assert request.num_channels == 2
    assert request.audio_device_name == "dev1"
    assert request.mode.value == "observe"
    assert server.auto_soundcheck_observe_only is True
    assert server.auto_soundcheck_running is True
    assert service.engine.started is True


@pytest.mark.asyncio
async def test_missing_mode_defaults_to_observe_not_legacy_auto_write():
    server = DummyServer()
    service = FakeLiveService()
    server._live_soundcheck_service = service
    handlers = register_handlers(server)

    await handlers["start_auto_engine"]("ws", {})

    request = service.requests[-1]
    assert request.mode.value == "observe"
    _, payload = server.sent_messages[-1]
    assert payload["mode"] == "observe"
    assert payload["observe_only"] is True


@pytest.mark.asyncio
async def test_explicit_bench_test_is_preserved_for_visible_wing_testing():
    server = DummyServer()
    service = FakeLiveService()
    server._live_soundcheck_service = service
    handlers = register_handlers(server)

    await handlers["start_auto_engine"]("ws", {"mode": "bench_test"})

    request = service.requests[-1]
    assert request.mode.value == "bench_test"
    assert server.auto_soundcheck_observe_only is False
    _, payload = server.sent_messages[-1]
    assert payload["mode"] == "bench_test"


@pytest.mark.asyncio
async def test_legacy_explicit_observe_false_maps_to_supervised_not_bench_test():
    server = DummyServer()
    service = FakeLiveService()
    server._live_soundcheck_service = service
    handlers = register_handlers(server)

    await handlers["start_auto_engine"]("ws", {"observe_only": False})

    assert service.requests[-1].mode.value == "supervised"


@pytest.mark.asyncio
async def test_invalid_mode_is_blocked_before_engine_creation():
    server = DummyServer()
    service = FakeLiveService()
    server._live_soundcheck_service = service
    handlers = register_handlers(server)

    await handlers["start_auto_engine"]("ws", {"mode": "cowboy"})

    assert service.requests == []
    _, payload = server.sent_messages[-1]
    assert payload["status"] == "blocked"
    assert payload["running"] is False


@pytest.mark.asyncio
async def test_start_auto_engine_uses_config_and_wires_callbacks():
    server = DummyServer()
    service = FakeLiveService()
    server._live_soundcheck_service = service
    handlers = register_handlers(server)

    await handlers["start_auto_engine"]("ws", {"mode": "propose"})

    request = service.requests[-1]
    callbacks = service.callbacks[-1]
    assert request.mixer_type == "wing"
    assert request.mixer_ip == "10.0.0.5"
    assert request.mixer_port == 2223
    assert request.audio_device_name == "Test Device"
    assert callable(callbacks["on_state_change"])
    assert callable(callbacks["on_channel_update"])

    callbacks["on_state_change"]("running", "Engine started")
    callbacks["on_channel_update"](3, {"preset": "kick"})
    await asyncio.sleep(0)

    assert {
        "type": "auto_engine_state",
        "state": "running",
        "message": "Engine started",
        "mode": "propose",
    } in server.broadcast_messages
    assert {"type": "auto_engine_channel", "channel": 3, "data": {"preset": "kick"}} in server.broadcast_messages
