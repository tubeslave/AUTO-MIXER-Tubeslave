"""Tests for websocket safety gating on the headless auto-soundcheck engine."""

from types import SimpleNamespace

import pytest

import handlers.soundcheck_handlers as soundcheck_handlers


class FakeServer:
    def __init__(self):
        self.auto_soundcheck_engine = None
        self.sent_messages = []
        self.broadcast_messages = []
        self.auto_soundcheck_running = False
        self.start_auto_soundcheck_calls = []

    async def send_to_client(self, websocket, message):
        self.sent_messages.append((websocket, message))

    async def broadcast(self, message):
        self.broadcast_messages.append(message)

    async def start_auto_soundcheck(
        self,
        websocket,
        device_id,
        channels,
        channel_settings,
        channel_mapping,
        timings,
        live_apply_enabled=False,
    ):
        self.start_auto_soundcheck_calls.append({
            "websocket": websocket,
            "device_id": device_id,
            "channels": channels,
            "channel_settings": channel_settings,
            "channel_mapping": channel_mapping,
            "timings": timings,
            "live_apply_enabled": live_apply_enabled,
        })

    def get_auto_engine_gain_recommendations(self):
        return {
            "active": self.auto_soundcheck_engine is not None,
            "source": "auto_soundcheck_engine",
            "live_apply_enabled": False,
            "dry_run": True,
            "recommendations": {
                3: {
                    "channel": 3,
                    "recommended_target_trim_db": -4.0,
                    "delta_db": 2.0,
                }
            } if self.auto_soundcheck_engine is not None else {},
        }


class FakeEngine:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.state = SimpleNamespace(value="idle")
        FakeEngine.instances.append(self)

    def start_async(self):
        self.started = True

    def get_status(self):
        return {
            "state": self.state.value,
            "live_apply_enabled": self.kwargs["auto_apply"],
            "dry_run": not self.kwargs["auto_apply"],
            "mixer_connected": False,
            "audio_running": False,
        }


@pytest.fixture(autouse=True)
def clear_fake_engine_instances():
    FakeEngine.instances.clear()
    yield
    FakeEngine.instances.clear()


@pytest.mark.asyncio
async def test_start_auto_engine_defaults_to_dry_run(monkeypatch):
    monkeypatch.setattr(soundcheck_handlers, "AutoSoundcheckEngine", FakeEngine)
    server = FakeServer()
    handlers = soundcheck_handlers.register_handlers(server)

    await handlers["start_auto_engine"]("ws", {})

    engine = FakeEngine.instances[-1]
    assert engine.kwargs["auto_apply"] is False
    assert engine.started is True
    assert server.sent_messages[-1][1]["live_apply_enabled"] is False
    assert server.sent_messages[-1][1]["dry_run"] is True
    assert server.sent_messages[-1][1]["warning"] is None


@pytest.mark.asyncio
async def test_start_auto_engine_blocks_unconfirmed_live_apply(monkeypatch):
    monkeypatch.setattr(soundcheck_handlers, "AutoSoundcheckEngine", FakeEngine)
    server = FakeServer()
    handlers = soundcheck_handlers.register_handlers(server)

    await handlers["start_auto_engine"]("ws", {"live_apply_enabled": True})

    engine = FakeEngine.instances[-1]
    assert engine.kwargs["auto_apply"] is False
    assert server.sent_messages[-1][1]["warning"] == "live_apply_blocked_missing_confirmation"
    assert server.sent_messages[-1][1]["dry_run"] is True


@pytest.mark.asyncio
async def test_start_auto_engine_auto_apply_alias_still_requires_confirmation(monkeypatch):
    monkeypatch.setattr(soundcheck_handlers, "AutoSoundcheckEngine", FakeEngine)
    server = FakeServer()
    handlers = soundcheck_handlers.register_handlers(server)

    await handlers["start_auto_engine"]("ws", {"auto_apply": True})

    engine = FakeEngine.instances[-1]
    assert engine.kwargs["auto_apply"] is False
    assert server.sent_messages[-1][1]["live_apply_enabled"] is False
    assert server.sent_messages[-1][1]["dry_run"] is True
    assert server.sent_messages[-1][1]["warning"] == "live_apply_blocked_missing_confirmation"


@pytest.mark.asyncio
async def test_start_auto_engine_confirmation_alone_does_not_enable_live_apply(monkeypatch):
    monkeypatch.setattr(soundcheck_handlers, "AutoSoundcheckEngine", FakeEngine)
    server = FakeServer()
    handlers = soundcheck_handlers.register_handlers(server)

    await handlers["start_auto_engine"]("ws", {"confirm_live_apply": True})

    engine = FakeEngine.instances[-1]
    assert engine.kwargs["auto_apply"] is False
    assert server.sent_messages[-1][1]["live_apply_enabled"] is False
    assert server.sent_messages[-1][1]["dry_run"] is True
    assert server.sent_messages[-1][1]["warning"] is None


@pytest.mark.asyncio
async def test_start_auto_engine_allows_explicit_confirmed_live_apply(monkeypatch):
    monkeypatch.setattr(soundcheck_handlers, "AutoSoundcheckEngine", FakeEngine)
    server = FakeServer()
    handlers = soundcheck_handlers.register_handlers(server)

    await handlers["start_auto_engine"](
        "ws",
        {"live_apply_enabled": True, "confirm_live_apply": True},
    )

    engine = FakeEngine.instances[-1]
    assert engine.kwargs["auto_apply"] is True
    assert server.sent_messages[-1][1]["live_apply_enabled"] is True
    assert server.sent_messages[-1][1]["dry_run"] is False
    assert server.sent_messages[-1][1]["warning"] is None


@pytest.mark.asyncio
async def test_start_auto_soundcheck_defaults_to_dry_run():
    server = FakeServer()
    handlers = soundcheck_handlers.register_handlers(server)

    await handlers["start_auto_soundcheck"]("ws", {"device_id": "dev0", "channels": [1, 2]})

    call = server.start_auto_soundcheck_calls[-1]
    assert call["live_apply_enabled"] is False
    assert server.sent_messages[-1][1]["live_apply_enabled"] is False
    assert server.sent_messages[-1][1]["dry_run"] is True
    assert server.sent_messages[-1][1]["warning"] is None


@pytest.mark.asyncio
async def test_start_auto_soundcheck_blocks_unconfirmed_live_apply():
    server = FakeServer()
    handlers = soundcheck_handlers.register_handlers(server)

    await handlers["start_auto_soundcheck"](
        "ws",
        {"device_id": "dev0", "channels": [1], "live_apply_enabled": True},
    )

    call = server.start_auto_soundcheck_calls[-1]
    assert call["live_apply_enabled"] is False
    assert server.sent_messages[-1][1]["dry_run"] is True
    assert server.sent_messages[-1][1]["warning"] == "live_apply_blocked_missing_confirmation"


@pytest.mark.asyncio
async def test_start_auto_soundcheck_auto_apply_alias_still_requires_confirmation():
    server = FakeServer()
    handlers = soundcheck_handlers.register_handlers(server)

    await handlers["start_auto_soundcheck"](
        "ws",
        {"device_id": "dev0", "channels": [1], "auto_apply": True},
    )

    call = server.start_auto_soundcheck_calls[-1]
    assert call["live_apply_enabled"] is False
    assert server.sent_messages[-1][1]["live_apply_enabled"] is False
    assert server.sent_messages[-1][1]["dry_run"] is True
    assert server.sent_messages[-1][1]["warning"] == "live_apply_blocked_missing_confirmation"


@pytest.mark.asyncio
async def test_start_auto_soundcheck_confirmation_alone_does_not_enable_live_apply():
    server = FakeServer()
    handlers = soundcheck_handlers.register_handlers(server)

    await handlers["start_auto_soundcheck"](
        "ws",
        {"device_id": "dev0", "channels": [1], "confirm_live_apply": True},
    )

    call = server.start_auto_soundcheck_calls[-1]
    assert call["live_apply_enabled"] is False
    assert server.sent_messages[-1][1]["live_apply_enabled"] is False
    assert server.sent_messages[-1][1]["dry_run"] is True
    assert server.sent_messages[-1][1]["warning"] is None


@pytest.mark.asyncio
async def test_start_auto_soundcheck_allows_explicit_confirmed_live_apply():
    server = FakeServer()
    handlers = soundcheck_handlers.register_handlers(server)

    await handlers["start_auto_soundcheck"](
        "ws",
        {
            "device_id": "dev0",
            "channels": [1],
            "live_apply_enabled": True,
            "confirm_live_apply": True,
        },
    )

    call = server.start_auto_soundcheck_calls[-1]
    assert call["live_apply_enabled"] is True
    assert server.sent_messages[-1][1]["live_apply_enabled"] is True
    assert server.sent_messages[-1][1]["dry_run"] is False
    assert server.sent_messages[-1][1]["warning"] is None


@pytest.mark.asyncio
async def test_get_auto_engine_gain_recommendations_returns_read_only_payload():
    server = FakeServer()
    server.auto_soundcheck_engine = object()
    handlers = soundcheck_handlers.register_handlers(server)

    await handlers["get_auto_engine_gain_recommendations"](
        "ws",
        {"request_id": "req-auto-engine"},
    )

    message = server.sent_messages[-1][1]
    assert message["request_id"] == "req-auto-engine"
    assert message["type"] == "auto_engine_gain_recommendations"
    assert message["active"] is True
    assert message["source"] == "auto_soundcheck_engine"
    assert message["dry_run"] is True
    assert message["recommendations"][3]["recommended_target_trim_db"] == -4.0


@pytest.mark.asyncio
async def test_get_auto_engine_status_defaults_to_idle_dry_run_payload():
    server = FakeServer()
    handlers = soundcheck_handlers.register_handlers(server)

    await handlers["get_auto_engine_status"]("ws", {})

    message = server.sent_messages[-1][1]
    assert message["type"] == "auto_engine_status"
    assert message["state"] == "idle"
    assert message["live_apply_enabled"] is False
    assert message["dry_run"] is True


@pytest.mark.asyncio
async def test_get_auto_engine_status_relays_confirmed_live_apply_state():
    server = FakeServer()

    class RunningEngine:
        def get_status(self):
            return {
                "state": "running",
                "live_apply_enabled": True,
                "dry_run": False,
                "mixer_connected": True,
                "audio_running": True,
            }

    server.auto_soundcheck_engine = RunningEngine()
    handlers = soundcheck_handlers.register_handlers(server)

    await handlers["get_auto_engine_status"]("ws", {})

    message = server.sent_messages[-1][1]
    assert message["type"] == "auto_engine_status"
    assert message["state"] == "running"
    assert message["live_apply_enabled"] is True
    assert message["dry_run"] is False
