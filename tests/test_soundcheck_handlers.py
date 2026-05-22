"""Tests for backend/handlers/soundcheck_handlers.py."""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from handlers.soundcheck_handlers import register_handlers


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
        self.imported_soundcheck_bundles = []
        self.send_to_client = AsyncMock(side_effect=self._capture_send)
        self.broadcast = AsyncMock(side_effect=self._capture_broadcast)
        self.start_auto_soundcheck = AsyncMock()

    async def _capture_send(self, websocket, payload):
        self.sent_messages.append((websocket, payload))

    async def _capture_broadcast(self, payload):
        self.broadcast_messages.append(payload)

    def import_soundcheck_recommendations_to_operator_queue(self, bundle=None):
        self.imported_soundcheck_bundles.append(bundle)
        return {
            "type": "operator_proposals_imported",
            "status": "ok",
            "source": "soundcheck_recommendation_bundle",
            "created_count": len((bundle or {}).get("proposals", [])) if isinstance(bundle, dict) else 0,
            "existing_count": 0,
            "blocked_count": 0,
            "imported_count": len((bundle or {}).get("proposals", [])) if isinstance(bundle, dict) else 0,
            "results": [],
        }


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

    class FakeEngine:
        def get_status(self):
            return {
                "state": "running",
                "autofoh_session_report_summary": "AutoFOH session report: events=4; sent=1; blocked=3; guard_blocks=2",
            }

    server.auto_soundcheck_engine = FakeEngine()
    server.auto_soundcheck_running = True
    handlers = register_handlers(server)

    await handlers["get_auto_soundcheck_status"]("ws", {})

    _, payload = server.sent_messages[-1]
    assert payload["type"] == "auto_soundcheck_status"
    assert payload["autofoh_session_report_summary"].startswith("AutoFOH session report:")


@pytest.mark.asyncio
async def test_start_auto_soundcheck_handler_starts_new_engine_with_observe_only(monkeypatch):
    server = DummyServer()
    handlers = register_handlers(server)
    created = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.state = type("State", (), {"value": "idle"})()

        def start_async(self):
            self.started = True

    monkeypatch.setattr("handlers.soundcheck_handlers.AutoSoundcheckEngine", FakeEngine)

    await handlers["start_auto_soundcheck"]("ws", {
        "device_id": "dev1",
        "channels": [1, 2],
        "channel_settings": {},
        "channel_mapping": {},
        "timings": {"gain_staging": 10},
        "observe_only": True,
    })

    assert created["selected_channels"] == [1, 2]
    assert created["num_channels"] == 2
    assert created["audio_device_name"] == "dev1"
    assert created["observe_only"] is True
    assert server.auto_soundcheck_running is True


@pytest.mark.asyncio
async def test_start_auto_engine_uses_config_and_wires_callbacks(monkeypatch):
    server = DummyServer()
    handlers = register_handlers(server)
    created = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.state = type("State", (), {"value": "idle"})()
            self.started = False

        def start_async(self):
            self.started = True

    monkeypatch.setattr("handlers.soundcheck_handlers.AutoSoundcheckEngine", FakeEngine)

    await handlers["start_auto_engine"]("ws", {})

    assert created["mixer_type"] == "wing"
    assert created["mixer_ip"] == "10.0.0.5"
    assert created["mixer_port"] == 2223
    assert created["audio_device_name"] == "Test Device"
    assert callable(created["on_state_change"])
    assert callable(created["on_channel_update"])

    created["on_state_change"]("running", "Engine started")
    created["on_channel_update"](3, {"preset": "kick"})
    await asyncio.sleep(0)

    assert {"type": "auto_engine_state", "state": "running", "message": "Engine started"} in server.broadcast_messages
    assert {"type": "auto_engine_channel", "channel": 3, "data": {"preset": "kick"}} in server.broadcast_messages


@pytest.mark.asyncio
async def test_auto_engine_imports_soundcheck_recommendations_on_running_state(monkeypatch):
    server = DummyServer()
    handlers = register_handlers(server)
    bundle = {
        "schema_version": "soundcheck_recommendation_bundle/v1",
        "proposals": [
            {
                "proposal_id": "unit-fader",
                "action_type": "set_fader_db",
                "target": {"mixer_channel": 2},
                "current_state": {"fader_db": -10.0},
                "requested_state": {"fader_db": -8.5},
            }
        ],
    }

    class FakeEngine:
        def __init__(self, **kwargs):
            self.on_state_change = kwargs["on_state_change"]
            self.state = type("State", (), {"value": "idle"})()
            self.started = False
            self.get_soundcheck_recommendation_bundle = MagicMock(return_value=bundle)

        def start_async(self):
            self.started = True

    monkeypatch.setattr("handlers.soundcheck_handlers.AutoSoundcheckEngine", FakeEngine)

    await handlers["start_auto_engine"]("ws", {})
    server.auto_soundcheck_engine.on_state_change("running", "Auto-soundcheck complete, monitoring...")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert server.imported_soundcheck_bundles == [bundle]
    assert {
        "type": "operator_proposals_imported",
        "status": "ok",
        "source": "soundcheck_recommendation_bundle",
        "created_count": 1,
        "existing_count": 0,
        "blocked_count": 0,
        "imported_count": 1,
        "results": [],
        "trigger": "auto_soundcheck_state",
        "trigger_state": "running",
    } in server.broadcast_messages
