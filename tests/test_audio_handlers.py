import pytest

from handlers.audio_handlers import register_handlers


class FakeCapture:
    def __init__(self):
        self.stopped = False

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


@pytest.mark.asyncio
async def test_stop_audio_capture_preview_stops_and_clears_capture():
    server = FakeServer()
    server.audio_capture = FakeCapture()
    capture = server.audio_capture
    handlers = register_handlers(server)

    await handlers["stop_audio_capture_preview"](object(), {})

    assert capture.stopped is True
    assert server.audio_capture is None
    assert server.sent[-1] == {
        "type": "audio_capture_status",
        "running": False,
        "preview": True,
    }


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
