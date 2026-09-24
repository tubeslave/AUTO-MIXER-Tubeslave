"""Tests for canonical live AudioCapture selection and lifecycle ownership."""

import os
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND)

from audio_capture import AudioSourceType  # noqa: E402
from audio_device_scanner import AudioDevice, AudioProtocol  # noqa: E402
from live_runtime.audio_capture_session import (  # noqa: E402
    LiveAudioCaptureConfig,
    LiveAudioCaptureError,
    LiveAudioCaptureSession,
)


def _device(
    *,
    index=7,
    name="WING USB Audio",
    channels=48,
    rate=48000,
    protocol=AudioProtocol.USB,
):
    return AudioDevice(
        index=index,
        name=name,
        max_input_channels=channels,
        default_samplerate=rate,
        protocol=protocol,
        is_multichannel=channels >= 8,
        score=100,
    )


class FakeCapture:
    def __init__(self, *, fallback_to_silence=False, **kwargs):
        self.kwargs = dict(kwargs)
        self.num_channels = kwargs["num_channels"]
        self.sample_rate = kwargs["sample_rate"]
        self.source_type = kwargs["source_type"]
        self.fallback_to_silence = fallback_to_silence
        self.start_calls = 0
        self.stop_calls = 0

    def start(self):
        self.start_calls += 1
        if self.fallback_to_silence:
            self.source_type = AudioSourceType.SILENCE

    def stop(self):
        self.stop_calls += 1


def test_session_owns_exact_48k_48ch_usb_capture_and_teardown():
    created = []
    audits = []

    def factory(**kwargs):
        capture = FakeCapture(**kwargs)
        created.append(capture)
        return capture

    session = LiveAudioCaptureSession(
        LiveAudioCaptureConfig(
            audio_device_name="WING USB",
            num_channels=48,
            sample_rate=48000,
            required_channel_ids=(1, 47, 48),
        ),
        scan_devices=lambda: [_device()],
        capture_factory=factory,
        audit_sink=audits.append,
    )

    capture = session.start()
    assert capture is session.capture
    assert capture.start_calls == 1
    assert capture.kwargs["num_channels"] == 48
    assert capture.kwargs["sample_rate"] == 48000
    assert capture.kwargs["device_name"] == 7
    assert capture.kwargs["source_type"] is AudioSourceType.SOUNDDEVICE
    assert session.status().running is True
    assert session.status().effective_channels == 48

    # Start is idempotent and does not create/open a second physical stream.
    assert session.start() is capture
    assert len(created) == 1
    assert capture.start_calls == 1

    assert session.stop() is True
    assert capture.stop_calls == 1
    assert session.stop() is False
    assert capture.stop_calls == 1
    assert [event["event"] for event in audits] == [
        "live_audio_capture_started",
        "live_audio_capture_stopped",
    ]


def test_dante_name_prefers_dante_and_maps_source_type():
    devices = [
        _device(index=1, name="Generic 64", channels=64, protocol=AudioProtocol.USB),
        _device(
            index=2,
            name="Dante Virtual Soundcard",
            channels=64,
            protocol=AudioProtocol.DANTE,
        ),
    ]
    created = []

    def factory(**kwargs):
        capture = FakeCapture(**kwargs)
        created.append(capture)
        return capture

    session = LiveAudioCaptureSession(
        LiveAudioCaptureConfig(
            audio_device_name="Dante",
            num_channels=48,
        ),
        scan_devices=lambda: devices,
        capture_factory=factory,
    )

    session.start()
    assert session.selected_device is devices[1]
    assert created[0].source_type is AudioSourceType.DANTE
    session.stop()


def test_insufficient_hardware_channels_fail_closed_without_constructing_capture():
    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return FakeCapture(**kwargs)

    session = LiveAudioCaptureSession(
        LiveAudioCaptureConfig(
            num_channels=48,
            required_channel_ids=(47, 48),
        ),
        scan_devices=lambda: [_device(channels=32)],
        capture_factory=factory,
    )

    with pytest.raises(LiveAudioCaptureError, match="48 are required"):
        session.start()

    assert factory_calls == []
    assert session.capture is None
    assert session.running is False


def test_no_live_audio_device_fails_closed_instead_of_inventing_silence():
    session = LiveAudioCaptureSession(
        LiveAudioCaptureConfig(num_channels=48),
        scan_devices=lambda: [],
    )

    with pytest.raises(LiveAudioCaptureError, match="No live audio input device"):
        session.start()

    assert session.capture is None
    assert session.running is False


def test_real_capture_fallback_to_silence_is_rejected_and_closed():
    created = []

    def factory(**kwargs):
        capture = FakeCapture(fallback_to_silence=True, **kwargs)
        created.append(capture)
        return capture

    session = LiveAudioCaptureSession(
        LiveAudioCaptureConfig(num_channels=48),
        scan_devices=lambda: [_device()],
        capture_factory=factory,
    )

    with pytest.raises(LiveAudioCaptureError, match="fell back to SILENCE"):
        session.start()

    assert created[0].start_calls == 1
    assert created[0].stop_calls == 1
    assert session.capture is None
    assert session.running is False


def test_config_rejects_required_channel_outside_declared_stream():
    with pytest.raises(ValueError, match="outside 1..48"):
        LiveAudioCaptureConfig(
            num_channels=48,
            required_channel_ids=(49,),
        )
