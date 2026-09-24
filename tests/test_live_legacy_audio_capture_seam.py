"""Migration tests for handing canonical live AudioCapture to legacy engine."""

import os
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND)

from live_runtime.legacy_audio_capture_seam import (  # noqa: E402
    LegacyAudioCaptureSeamError,
    LegacyExternalAudioCaptureSeam,
)


class FakeCapture:
    def __init__(self):
        self.start_calls = 0
        self.stop_calls = 0
        self.sample_rate = 48000
        self.num_channels = 48
        self.buffers_read = 0

    def start(self):
        self.start_calls += 1

    def stop(self):
        self.stop_calls += 1

    def get_buffer(self, frames):
        self.buffers_read += 1
        return (frames, self.num_channels)


class FakeLegacyEngine:
    def __init__(self):
        self.audio_capture = None
        self.internal_start_calls = 0
        self.internal_capture = None

    def _start_audio(self):
        self.internal_start_calls += 1
        self.internal_capture = FakeCapture()
        self.audio_capture = self.internal_capture
        self.audio_capture.start()
        return True

    def stop(self):
        if self.audio_capture is not None:
            self.audio_capture.stop()


def test_external_capture_is_used_but_legacy_cannot_start_or_stop_physical_stream():
    capture = FakeCapture()
    capture.start()  # Canonical LiveAudioCaptureSession owns this call.
    engine = FakeLegacyEngine()
    audits = []
    seam = LegacyExternalAudioCaptureSeam(engine, capture, audit_sink=audits.append)

    assert seam.bind() is True
    assert seam.bind() is False
    assert seam.status().bound is True
    assert seam.status().engine_uses_proxy is True

    # Legacy startup is bypassed, so no second physical capture is constructed.
    assert engine._start_audio() is True
    assert engine.internal_start_calls == 0
    assert engine.internal_capture is None
    assert capture.start_calls == 1

    # Analysis/read APIs still delegate to the canonical physical capture.
    assert engine.audio_capture.sample_rate == 48000
    assert engine.audio_capture.num_channels == 48
    assert engine.audio_capture.get_buffer(256) == (256, 48)
    assert capture.buffers_read == 1

    # A direct lifecycle call through legacy ownership is suppressed as well.
    engine.audio_capture.start()
    assert capture.start_calls == 1

    # Legacy teardown may run normally, but it cannot close the external stream.
    engine.stop()
    assert capture.stop_calls == 0

    assert seam.detach() is True
    assert seam.detach() is False
    assert engine.audio_capture is None
    assert seam.status().bound is False

    # Only the canonical owner closes the physical stream.
    capture.stop()
    assert capture.stop_calls == 1

    events = [event["event"] for event in audits]
    assert events == [
        "legacy_external_audio_capture_bound",
        "legacy_external_audio_capture_start_bypassed",
        "legacy_external_audio_capture_start_suppressed",
        "legacy_external_audio_capture_stop_suppressed",
        "legacy_external_audio_capture_detached",
    ]


def test_bind_fails_closed_if_legacy_engine_already_owns_a_capture():
    external = FakeCapture()
    engine = FakeLegacyEngine()
    engine.audio_capture = FakeCapture()
    seam = LegacyExternalAudioCaptureSeam(engine, external)

    with pytest.raises(LegacyAudioCaptureSeamError, match="ambiguous ownership"):
        seam.bind()

    assert engine.audio_capture is not external
    assert seam.bound is False


def test_bind_fails_closed_without_legacy_start_audio_seam():
    class NoAudioStartEngine:
        audio_capture = None

    with pytest.raises(LegacyAudioCaptureSeamError, match="_start_audio"):
        LegacyExternalAudioCaptureSeam(NoAudioStartEngine(), FakeCapture())


def test_detach_refuses_to_clobber_capture_replaced_while_bound():
    capture = FakeCapture()
    engine = FakeLegacyEngine()
    seam = LegacyExternalAudioCaptureSeam(engine, capture)
    seam.bind()

    replacement = FakeCapture()
    engine.audio_capture = replacement

    with pytest.raises(LegacyAudioCaptureSeamError, match="changed while external ownership"):
        seam.detach()

    assert engine.audio_capture is replacement
    assert seam.bound is True


def test_start_bypass_detects_displaced_proxy_before_legacy_startup():
    capture = FakeCapture()
    engine = FakeLegacyEngine()
    seam = LegacyExternalAudioCaptureSeam(engine, capture)
    seam.bind()
    engine.audio_capture = None

    with pytest.raises(LegacyAudioCaptureSeamError, match="displaced"):
        engine._start_audio()
