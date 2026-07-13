import asyncio

import pytest

from handlers.audio_handlers import register_handlers


class FakeCapture:
    def __init__(self):
        self.stopped = False
        self.running = True

    def get_status(self):
        return {
            "running": True,
            "source_type": "sounddevice",
            "num_channels": 48,
            "sample_rate": 48000,
            "buffer_seconds": 8.0,
            "subscribers": 0,
        }

    def stop(self):
        self.stopped = True


class FakeServer:
    def __init__(self):
        self.sent = []
        self.audio_capture = None
        self.capture_calls = []
        self._audio_capture_preview_owner = None
        self._audio_capture_lock = asyncio.Lock()
        self._system_measurement_task = None
        self.auto_soundcheck_running = False
        self.auto_soundcheck_engine = None

    async def send_to_client(self, websocket, payload):
        self.sent.append(payload)

    def _ensure_audio_capture(self, device_id, *, num_channels, sample_rate, buffer_seconds):
        self.capture_calls.append({
            "device_id": device_id,
            "num_channels": num_channels,
            "sample_rate": sample_rate,
            "buffer_seconds": buffer_seconds,
        })
        self.audio_capture = FakeCapture()
        return self.audio_capture

    async def _ensure_audio_capture_async(
        self,
        device_id,
        *,
        num_channels,
        sample_rate,
        buffer_seconds,
    ):
        return self._ensure_audio_capture(
            device_id,
            num_channels=num_channels,
            sample_rate=sample_rate,
            buffer_seconds=buffer_seconds,
        )


@pytest.mark.asyncio
async def test_start_audio_capture_preview_normalizes_numeric_device_id():
    server = FakeServer()
    handlers = register_handlers(server)

    await handlers["start_audio_capture_preview"](
        object(),
        {
            "device_id": "1",
            "num_channels": "48",
            "sample_rate": "48000",
        },
    )

    assert server.capture_calls == [{
        "device_id": 1,
        "num_channels": 48,
        "sample_rate": 48000,
        "buffer_seconds": 8.0,
    }]
    assert server.sent[-1]["type"] == "audio_capture_status"
    assert server.sent[-1]["preview"] is True
    assert server.sent[-1]["running"] is True
    assert server.sent[-1]["device_id"] == 1
    assert server.sent[-1]["owned_by_requester"] is True


@pytest.mark.asyncio
async def test_stop_audio_capture_preview_stops_and_clears_capture():
    server = FakeServer()
    server.audio_capture = FakeCapture()
    capture = server.audio_capture
    handlers = register_handlers(server)
    websocket = object()
    server._audio_capture_preview_owner = websocket

    await handlers["stop_audio_capture_preview"](websocket, {})

    assert capture.stopped is True
    assert server.audio_capture is None
    assert server.sent[-1] == {
        "type": "audio_capture_status",
        "running": False,
        "preview": True,
        "owned_by_requester": True,
    }


@pytest.mark.asyncio
async def test_audio_preview_cannot_stop_capture_owned_by_another_websocket():
    server = FakeServer()
    server.audio_capture = FakeCapture()
    capture = server.audio_capture
    server._audio_capture_preview_owner = object()
    handlers = register_handlers(server)

    await handlers["stop_audio_capture_preview"](object(), {})

    assert capture.stopped is False
    assert server.audio_capture is capture
    assert server.sent[-1]["running"] is True
    assert server.sent[-1]["owned_by_requester"] is False
    assert server.sent[-1]["error"] == "audio_capture_preview_not_owned_by_requester"


@pytest.mark.asyncio
async def test_start_audio_capture_preview_rejects_invalid_channel_count():
    server = FakeServer()
    handlers = register_handlers(server)

    await handlers["start_audio_capture_preview"](
        object(),
        {"device_id": "1", "num_channels": "not-a-number"},
    )

    assert server.capture_calls == []
    assert server.sent[-1]["type"] == "audio_capture_status"
    assert server.sent[-1]["running"] is False
    assert "Invalid audio capture preview request" in server.sent[-1]["error"]


@pytest.mark.asyncio
async def test_start_audio_capture_preview_rejects_oversized_allocation():
    server = FakeServer()
    handlers = register_handlers(server)

    await handlers["start_audio_capture_preview"](
        object(),
        {
            "device_id": "1",
            "num_channels": 64,
            "sample_rate": 192000,
            "buffer_seconds": 30,
        },
    )

    assert server.capture_calls == []
    assert "256 MiB" in server.sent[-1]["error"]


@pytest.mark.asyncio
async def test_start_audio_capture_preview_rejects_active_auto_soundcheck():
    server = FakeServer()
    server.auto_soundcheck_running = True
    server.auto_soundcheck_engine = type(
        "Engine",
        (),
        {"state": type("State", (), {"value": "running"})()},
    )()
    handlers = register_handlers(server)

    await handlers["start_audio_capture_preview"](
        object(),
        {"device_id": "1", "num_channels": 48},
    )

    assert server.capture_calls == []
    assert server.sent[-1]["error"] == "audio_capture_in_use_by_auto_soundcheck"
