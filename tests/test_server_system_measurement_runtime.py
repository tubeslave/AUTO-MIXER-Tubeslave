"""Regression tests for exclusive, fresh system-measurement audio capture."""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

import server as server_module
from server import AutoMixerServer


class FreshCapture:
    def __init__(self, *, real_input: bool = True, available_samples: int = 100):
        self.sample_rate = 100
        self.running = True
        self.real_input = real_input
        self.available_samples = available_samples
        self.stopped = False
        self._count_reads = 0

    @property
    def captured_sample_count(self) -> int:
        self._count_reads += 1
        return 0 if self._count_reads == 1 else 100

    def get_status(self):
        return {
            "real_input": self.real_input,
            "callback_age_sec": 0.01,
            "running": True,
        }

    def get_available_samples(self, channel: int) -> int:
        del channel
        return self.available_samples

    def get_buffer(self, channel: int, num_samples: int):
        del channel
        return np.full(num_samples, 0.1, dtype=np.float32)

    def get_synchronized_buffers(self, channels, num_samples):
        return {
            "buffers": {
                int(channel): np.full(num_samples, 0.1, dtype=np.float32)
                for channel in channels
            },
            "available_samples": {
                int(channel): self.available_samples for channel in channels
            },
            "captured_sample_count": 100,
            "last_callback_monotonic": time.monotonic(),
        }

    def stop(self):
        self.stopped = True
        self.running = False


def make_server(capture: FreshCapture) -> AutoMixerServer:
    with (
        patch.object(AutoMixerServer, "_load_config", return_value={}),
        patch("server.BleedService"),
    ):
        instance = AutoMixerServer()
    instance.mixer_client = MagicMock()
    instance.mixer_client.is_connected = True
    instance.audio_capture = capture
    instance.send_to_client = AsyncMock()
    instance.broadcast = AsyncMock()
    instance._ensure_audio_capture_async = AsyncMock(return_value=capture)
    return instance


def test_ensure_audio_capture_restarts_when_existing_ring_is_too_short(
    monkeypatch: pytest.MonkeyPatch,
):
    capture = FreshCapture()
    instance = make_server(capture)
    existing = MagicMock(
        running=True,
        device_name=None,
        num_channels=48,
        sample_rate=48000,
        buffer_seconds=8.0,
    )
    replacement = MagicMock(
        running=True,
        source_type=server_module.AudioSourceType.SOUNDDEVICE,
    )
    instance.audio_capture = existing
    monkeypatch.setattr(server_module, "HAS_SOUNDDEVICE", True)
    constructor = MagicMock(return_value=replacement)
    monkeypatch.setattr(server_module, "AudioCapture", constructor)

    result = instance._ensure_audio_capture(
        None,
        num_channels=48,
        sample_rate=48000,
        buffer_seconds=22.0,
    )

    existing.stop.assert_called_once_with()
    constructor.assert_called_once_with(
        num_channels=48,
        sample_rate=48000,
        buffer_seconds=22.0,
        source_type=server_module.AudioSourceType.SOUNDDEVICE,
        device_name=None,
    )
    replacement.start.assert_called_once_with()
    assert result is replacement


def test_ensure_audio_capture_refuses_replacement_when_old_stop_fails(monkeypatch):
    instance = make_server(FreshCapture())
    existing = MagicMock(
        running=True,
        device_name=None,
        num_channels=2,
        sample_rate=48000,
        buffer_seconds=1.0,
    )
    existing.stop.side_effect = RuntimeError("stop failed")
    instance.audio_capture = existing
    monkeypatch.setattr(server_module, "HAS_SOUNDDEVICE", True)
    constructor = MagicMock()
    monkeypatch.setattr(server_module, "AudioCapture", constructor)

    with pytest.raises(RuntimeError, match="refusing replacement"):
        instance._ensure_audio_capture(
            None,
            num_channels=2,
            sample_rate=48000,
            buffer_seconds=8.0,
        )

    constructor.assert_not_called()
    assert instance.audio_capture is existing


@pytest.fixture(autouse=True)
def short_measurement_duration(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server_module, "SYSTEM_MEASUREMENT_MIN_DURATION_SEC", 0.01)
    monkeypatch.setattr(server_module, "SYSTEM_MEASUREMENT_MAX_DURATION_SEC", 1.0)
    monkeypatch.setattr(server_module, "SYSTEM_MEASUREMENT_CAPTURE_GRACE_SEC", 0.05)
    monkeypatch.setattr(server_module, "SYSTEM_MEASUREMENT_CAPTURE_POLL_SEC", 0.001)


@pytest.mark.asyncio
async def test_system_measurement_reuses_requester_preview_without_stopping_it():
    capture = FreshCapture()
    instance = make_server(capture)
    websocket = object()
    instance._audio_capture_preview_owner = websocket
    controller = MagicMock()
    controller.analyze_reference_measurement.return_value = {
        "quality": 1.0,
        "corrections": [],
    }
    controller.get_recommended_target.return_value = "flat"

    with patch("server.SystemMeasurementController", return_value=controller):
        await instance.start_system_measurement(
            websocket,
            device_id="1",
            reference_channel=1,
            measurement_channel=2,
            duration_sec=0.02,
        )

    assert capture.stopped is False
    assert instance.audio_capture is capture
    assert instance._audio_capture_preview_owner is websocket
    assert instance.system_measurement_controller is controller
    result = instance.broadcast.await_args_list[-1].args[0]
    assert result["type"] == "system_measurement_result"
    assert result["success"] is True
    assert result["capture_real_input"] is True
    assert result["captured_samples"] == 2


@pytest.mark.asyncio
async def test_system_measurement_releases_capture_it_started_on_success():
    capture = FreshCapture()
    instance = make_server(capture)
    controller = MagicMock()
    controller.analyze_reference_measurement.return_value = {
        "quality": 1.0,
        "corrections": [],
    }
    controller.get_recommended_target.return_value = "flat"

    with patch("server.SystemMeasurementController", return_value=controller):
        await instance.start_system_measurement(
            object(),
            device_id="1",
            reference_channel=1,
            measurement_channel=2,
            duration_sec=0.02,
        )

    assert capture.stopped is True
    assert instance.audio_capture is None
    assert instance._system_measurement_task is None


@pytest.mark.asyncio
async def test_system_measurement_rejects_capture_owned_by_other_preview():
    capture = FreshCapture()
    instance = make_server(capture)
    instance._audio_capture_preview_owner = object()

    await instance.start_system_measurement(
        object(),
        device_id="1",
        reference_channel=1,
        measurement_channel=2,
        duration_sec=0.02,
    )

    instance._ensure_audio_capture_async.assert_not_awaited()
    payload = instance.send_to_client.await_args.args[1]
    assert payload["success"] is False
    assert payload["error"] == "audio_capture_in_use_by_another_runtime_owner"


@pytest.mark.asyncio
async def test_system_measurement_rejects_internally_owned_auto_soundcheck_capture():
    capture = FreshCapture()
    instance = make_server(capture)
    instance.audio_capture = None
    instance.auto_soundcheck_running = True
    instance.auto_soundcheck_engine = MagicMock()
    instance.auto_soundcheck_engine.state.value = "running"

    await instance.start_system_measurement(
        object(),
        device_id="1",
        reference_channel=1,
        measurement_channel=2,
        duration_sec=0.02,
    )

    instance._ensure_audio_capture_async.assert_not_awaited()
    payload = instance.send_to_client.await_args.args[1]
    assert payload["success"] is False
    assert payload["error"] == "audio_capture_in_use_by_auto_soundcheck"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("real_input", "available_samples", "error_text"),
    [
        (False, 100, "stale or unavailable"),
        (True, 1, "full requested measurement duration"),
    ],
)
async def test_system_measurement_rejects_stale_or_truncated_audio(
    real_input,
    available_samples,
    error_text,
):
    capture = FreshCapture(
        real_input=real_input,
        available_samples=available_samples,
    )
    instance = make_server(capture)

    await instance.start_system_measurement(
        object(),
        device_id="1",
        reference_channel=1,
        measurement_channel=2,
        duration_sec=0.02,
    )

    payload = instance.send_to_client.await_args.args[1]
    assert payload["success"] is False
    assert error_text in payload["error"]
    assert capture.stopped is True
    assert instance.audio_capture is None
    assert instance.system_measurement_controller is None


@pytest.mark.asyncio
async def test_failed_new_measurement_invalidates_previous_controller():
    capture = FreshCapture(real_input=False)
    instance = make_server(capture)
    instance.system_measurement_controller = MagicMock()

    await instance.start_system_measurement(
        object(),
        device_id="1",
        reference_channel=1,
        measurement_channel=2,
        duration_sec=0.02,
    )

    assert instance.system_measurement_controller is None


@pytest.mark.asyncio
async def test_concurrent_system_measurement_is_rejected():
    instance = make_server(FreshCapture())
    active = asyncio.create_task(asyncio.sleep(10))
    instance._system_measurement_task = active
    try:
        await instance.start_system_measurement(
            object(),
            device_id="1",
            reference_channel=1,
            measurement_channel=2,
            duration_sec=0.02,
        )
    finally:
        active.cancel()
        await asyncio.gather(active, return_exceptions=True)

    payload = instance.send_to_client.await_args.args[1]
    assert payload["success"] is False
    assert payload["error"] == "system_measurement_already_running"


@pytest.mark.asyncio
async def test_audio_preview_stop_does_not_block_event_loop():
    instance = make_server(FreshCapture())
    websocket = object()
    instance._audio_capture_preview_owner = websocket
    stop_started = threading.Event()

    def slow_stop():
        stop_started.set()
        time.sleep(0.1)

    instance.audio_capture.stop = slow_stop
    task = asyncio.create_task(instance._stop_owned_audio_capture_preview(websocket))
    assert await asyncio.to_thread(stop_started.wait, 1.0)

    await asyncio.sleep(0.01)
    assert task.done() is False

    assert await task is True
    assert instance.audio_capture is None


@pytest.mark.asyncio
async def test_cancelled_audio_stop_drains_worker_before_releasing_owner():
    instance = make_server(FreshCapture())
    websocket = object()
    instance._audio_capture_preview_owner = websocket
    stop_started = threading.Event()
    stop_finished = threading.Event()

    def slow_stop():
        stop_started.set()
        time.sleep(0.05)
        stop_finished.set()

    instance.audio_capture.stop = slow_stop
    task = asyncio.create_task(instance._stop_owned_audio_capture_preview(websocket))
    assert await asyncio.to_thread(stop_started.wait, 1.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert stop_finished.is_set()
    assert instance._audio_capture_preview_owner is None
    assert instance.audio_capture is None


@pytest.mark.asyncio
async def test_failed_preview_stop_retains_owner_and_capture_for_retry():
    instance = make_server(FreshCapture())
    websocket = object()
    capture = instance.audio_capture
    instance._audio_capture_preview_owner = websocket
    capture.stop = MagicMock(side_effect=RuntimeError("stop failed"))

    with pytest.raises(RuntimeError, match="stop failed"):
        await instance._stop_owned_audio_capture_preview(websocket)

    assert instance.audio_capture is capture
    assert instance._audio_capture_preview_owner is websocket


@pytest.mark.asyncio
async def test_cancelled_audio_start_stops_unowned_replacement_before_unlock():
    instance = make_server(FreshCapture())
    instance.audio_capture = None
    instance._audio_capture_preview_owner = None
    instance._ensure_audio_capture_async = AutoMixerServer._ensure_audio_capture_async.__get__(
        instance,
        AutoMixerServer,
    )
    start_entered = threading.Event()
    allow_start_to_finish = threading.Event()
    replacement = FreshCapture()

    def slow_start(*_args, **_kwargs):
        start_entered.set()
        assert allow_start_to_finish.wait(1.0)
        instance.audio_capture = replacement
        return replacement

    instance._ensure_audio_capture = slow_start
    task = asyncio.create_task(instance._ensure_audio_capture_async(
        None,
        num_channels=2,
        sample_rate=48000,
        buffer_seconds=8.0,
    ))
    assert await asyncio.to_thread(start_entered.wait, 1.0)
    task.cancel()
    allow_start_to_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert replacement.stopped is True
    assert instance.audio_capture is None
    assert instance._audio_capture_preview_owner is None


@pytest.mark.asyncio
async def test_cancelled_audio_start_retains_orphan_when_cleanup_fails():
    instance = make_server(FreshCapture())
    instance.audio_capture = None
    instance._audio_capture_preview_owner = None
    instance._ensure_audio_capture_async = AutoMixerServer._ensure_audio_capture_async.__get__(
        instance,
        AutoMixerServer,
    )
    start_entered = threading.Event()
    allow_start_to_finish = threading.Event()
    replacement = FreshCapture()
    replacement.stop = MagicMock(side_effect=RuntimeError("stop failed"))

    def slow_start(*_args, **_kwargs):
        start_entered.set()
        assert allow_start_to_finish.wait(1.0)
        instance.audio_capture = replacement
        return replacement

    instance._ensure_audio_capture = slow_start
    task = asyncio.create_task(instance._ensure_audio_capture_async(
        None,
        num_channels=2,
        sample_rate=48000,
        buffer_seconds=8.0,
    ))
    assert await asyncio.to_thread(start_entered.wait, 1.0)
    task.cancel()
    allow_start_to_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert instance.audio_capture is replacement
    assert instance._audio_capture_preview_owner is None


def test_cleanup_all_controllers_is_terminally_idempotent():
    instance = make_server(FreshCapture())
    instance.mixing_agent = MagicMock()
    mixer = instance.mixer_client
    capture = instance.audio_capture
    capture.stop = MagicMock(side_effect=capture.stop)

    instance.cleanup_all_controllers()
    instance.cleanup_all_controllers()

    assert instance.mixing_agent is None
    capture.stop.assert_called_once_with()
    mixer.disconnect.assert_called_once_with()
