"""Tests for the read-only realtime meter stream (dashboard telemetry)."""

import asyncio
import math

import numpy as np
import pytest

from handlers.meter_stream_handlers import (
    METER_STREAM_INTERVAL_SEC,
    register_handlers,
)
from operator_product_state import (
    SPECTRUM_BIN_COUNT,
    _analyzer_metrics,
    build_meter_frame,
)


class FakeCapture:
    def __init__(self, num_channels=4, sample_rate=48000):
        self.running = True
        self.num_channels = num_channels
        self.sample_rate = sample_rate

    def get_rms(self, channel, num_samples=1024):
        return -20.0 - channel

    def get_peak(self, channel, num_samples=1024):
        return -10.0 - channel

    def get_lufs(self, channel, window_seconds=0.4):
        return -22.0 - channel

    def get_spectrum(self, channel, fft_size=2048):
        freqs = np.fft.rfftfreq(fft_size, 1.0 / self.sample_rate)
        # flat -30 dB spectrum with a louder low-mid bump
        mags = np.full(freqs.shape, -30.0)
        mags[(freqs > 100) & (freqs < 300)] = -12.0
        return {"frequencies": freqs, "magnitude_db": mags}


class FakeServer:
    def __init__(self, capture):
        self.audio_capture = capture
        self.sent = []

    async def send_to_client(self, websocket, payload):
        self.sent.append((websocket, payload))


def test_meter_frame_reports_channels_master_and_analyzer():
    server = FakeServer(FakeCapture(num_channels=4))
    frame = build_meter_frame(server)

    assert frame["type"] == "meter_frame"
    assert frame["available"] is True
    assert frame["read_only"] is True
    assert frame["live_write"] is False
    assert [row["channel"] for row in frame["channels"]] == [1, 2, 3, 4]
    assert frame["channels"][0]["rms_db"] == -21.0
    assert frame["channels"][0]["peak_db"] == -11.0
    assert frame["master_bus"]["available"] is True
    assert frame["analyzer"]["available"] is True


def test_meter_frame_fails_closed_without_running_capture():
    capture = FakeCapture()
    capture.running = False
    frame = build_meter_frame(FakeServer(capture))

    assert frame["available"] is False
    assert frame["channels"] == []
    assert frame["master_bus"]["available"] is False
    assert frame["analyzer"]["available"] is False

    frame_no_capture = build_meter_frame(FakeServer(None))
    assert frame_no_capture["available"] is False


def test_meter_frame_respects_channel_cap():
    server = FakeServer(FakeCapture(num_channels=64))
    frame = build_meter_frame(server, max_channels=48)
    assert len(frame["channels"]) == 48


def test_analyzer_bins_are_display_scaled_and_log_spaced():
    server = FakeServer(FakeCapture())
    analyzer = _analyzer_metrics(server)

    bins = analyzer["spectrum_bins"]
    assert analyzer["available"] is True
    assert len(bins) == SPECTRUM_BIN_COUNT
    assert all(0.0 <= value <= 100.0 for value in bins)
    # -30 dB flat floor maps to 50 on the display scale; the 100-300 Hz bump
    # must stand clearly above the floor bins.
    floor = [value for value in bins if abs(value - 50.0) < 3.0]
    assert floor, "expected flat-floor bins near 50"
    assert max(bins) > 65.0
    peak_index = bins.index(max(bins))
    log_min, log_max = math.log10(20.0), math.log10(20000.0)
    bin_center_hz = 10 ** (log_min + (peak_index + 0.5) / SPECTRUM_BIN_COUNT * (log_max - log_min))
    assert 80.0 < bin_center_hz < 400.0


@pytest.mark.asyncio
async def test_subscribe_streams_frames_and_unsubscribe_stops():
    server = FakeServer(FakeCapture(num_channels=2))
    handlers = register_handlers(server)
    websocket = object()

    await handlers["subscribe_meters"](websocket, {})
    assert websocket in server._meter_stream_subscribers
    status = server.sent[0][1]
    assert status["type"] == "meter_stream_status"
    assert status["subscribed"] is True
    assert status["read_only"] is True

    await asyncio.sleep(METER_STREAM_INTERVAL_SEC * 3)
    frames = [payload for _, payload in server.sent if payload["type"] == "meter_frame"]
    assert len(frames) >= 2
    assert frames[0]["available"] is True
    assert frames[0]["channels"][0]["channel"] == 1

    await handlers["unsubscribe_meters"](websocket, {})
    assert websocket not in server._meter_stream_subscribers
    await asyncio.sleep(METER_STREAM_INTERVAL_SEC * 2)
    assert server._meter_stream_task is None or server._meter_stream_task.done()


@pytest.mark.asyncio
async def test_stream_drops_stale_subscribers_on_send_failure():
    server = FakeServer(FakeCapture(num_channels=2))

    failing = object()
    original_send = server.send_to_client

    async def send(websocket, payload):
        if websocket is failing and payload.get("type") == "meter_frame":
            raise ConnectionError("client gone")
        await original_send(websocket, payload)

    server.send_to_client = send
    handlers = register_handlers(server)
    await handlers["subscribe_meters"](failing, {})
    await asyncio.sleep(METER_STREAM_INTERVAL_SEC * 3)
    assert failing not in server._meter_stream_subscribers
