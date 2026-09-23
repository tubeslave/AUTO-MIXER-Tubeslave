import threading
import time

import numpy as np
import pytest

from backend.live_runtime.capture_bridge import LiveAudioCaptureBridge
from backend.live_runtime.feature_stream import MainFeatureEvidence


class FakeCapture:
    def __init__(self, *, sample_rate=48_000, num_channels=48):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self._subscribers = {}
        self._buffers = {
            channel: np.zeros(2048, dtype=np.float32)
            for channel in range(1, num_channels + 1)
        }

    def subscribe(self, name, callback):
        self._subscribers[name] = callback

    def unsubscribe(self, name):
        self._subscribers.pop(name, None)

    def get_buffer(self, channel, num_samples=0):
        data = self._buffers[channel]
        return data[-num_samples:].copy() if num_samples > 0 else data.copy()

    def emit(self, block):
        for channel in range(1, self.num_channels + 1):
            self._buffers[channel] = np.asarray(block[:, channel - 1], dtype=np.float32).copy()
        for callback in list(self._subscribers.values()):
            callback()


class FakeService:
    def __init__(self):
        self.calls = []
        self.called = threading.Event()

    def process_feature_snapshot(self, features, roles, **kwargs):
        self.calls.append((features, roles, threading.get_ident()))
        self.called.set()
        return "processed"


def _main_at(timestamp_s):
    return MainFeatureEvidence(
        rms_dbfs=-18.0,
        peak_dbfs=-6.0,
        crest_db=12.0,
        timestamp_s=timestamp_s,
    )


def _wait_for(predicate, timeout=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_bridge_rejects_noncanonical_capture_shape():
    service = FakeService()
    with pytest.raises(ValueError, match="48000"):
        LiveAudioCaptureBridge(
            FakeCapture(sample_rate=44_100),
            service,
            roles={},
            main_evidence_provider=_main_at,
        )
    with pytest.raises(ValueError, match="48 AudioCapture channels"):
        LiveAudioCaptureBridge(
            FakeCapture(num_channels=32),
            service,
            roles={},
            main_evidence_provider=_main_at,
        )


def test_bridge_snapshots_48_channels_and_feeds_service_off_callback_thread():
    capture = FakeCapture()
    service = FakeService()
    roles = {1: "kick", 2: "snare"}
    bridge = LiveAudioCaptureBridge(
        capture,
        service,
        roles=roles,
        channel_names={1: "Kick In", 2: "Snare Top"},
        main_evidence_provider=_main_at,
        window_frames=2048,
        analysis_interval_s=0.0,
    )
    block = np.zeros((2048, 48), dtype=np.float32)
    block[:, 0] = 0.10
    block[:, 1] = 0.20

    callback_thread = threading.get_ident()
    bridge.start()
    try:
        capture.emit(block)
        assert service.called.wait(1.0)
        assert len(service.calls) == 1
        features, received_roles, worker_thread = service.calls[0]
        assert worker_thread != callback_thread
        assert received_roles == roles
        assert len(features.channels) == 48
        assert features.channels[0].name == "Kick In"
        assert features.channels[1].name == "Snare Top"
        assert features.channels[0].rms_dbfs == pytest.approx(-20.0, abs=0.1)
        assert features.channels[1].rms_dbfs == pytest.approx(-13.98, abs=0.1)
        assert features.main_peak_dbfs == -6.0
        assert bridge.status().snapshots_processed == 1
    finally:
        bridge.stop()


def test_missing_main_evidence_fails_closed_without_service_call():
    capture = FakeCapture()
    service = FakeService()
    bridge = LiveAudioCaptureBridge(
        capture,
        service,
        roles={},
        main_evidence_provider=lambda _timestamp: None,
        analysis_interval_s=0.0,
    )
    bridge.start()
    try:
        capture.emit(np.zeros((2048, 48), dtype=np.float32))
        assert _wait_for(lambda: bridge.status().missing_main_evidence == 1)
        assert not service.called.is_set()
        status = bridge.status()
        assert status.snapshots_processed == 0
        assert status.last_error == "missing_main_evidence"
    finally:
        bridge.stop()


def test_incomplete_capture_window_is_dropped_before_analysis():
    capture = FakeCapture()
    capture._buffers[17] = np.zeros(1024, dtype=np.float32)
    service = FakeService()
    bridge = LiveAudioCaptureBridge(
        capture,
        service,
        roles={},
        main_evidence_provider=_main_at,
        window_frames=2048,
        analysis_interval_s=0.0,
    )
    bridge.start()
    try:
        # Call the actual subscriber without replacing the deliberately short buffer.
        for callback in list(capture._subscribers.values()):
            callback()
        assert _wait_for(lambda: bridge.status().incomplete_windows == 1)
        assert not service.called.is_set()
        assert bridge.status().snapshots_captured == 0
    finally:
        bridge.stop()


def test_stale_main_evidence_cannot_reach_service():
    capture = FakeCapture()
    service = FakeService()

    def stale_main(timestamp_s):
        return MainFeatureEvidence(
            rms_dbfs=-18.0,
            peak_dbfs=-6.0,
            crest_db=12.0,
            timestamp_s=timestamp_s - 1.0,
        )

    bridge = LiveAudioCaptureBridge(
        capture,
        service,
        roles={},
        main_evidence_provider=stale_main,
        analysis_interval_s=0.0,
        max_main_age_s=0.250,
    )
    bridge.start()
    try:
        capture.emit(np.zeros((2048, 48), dtype=np.float32))
        assert _wait_for(lambda: bridge.status().processing_failures == 1)
        assert not service.called.is_set()
        assert "Main evidence is stale" in (bridge.status().last_error or "")
    finally:
        bridge.stop()


def test_stop_unsubscribes_without_stopping_capture_transport():
    capture = FakeCapture()
    service = FakeService()
    bridge = LiveAudioCaptureBridge(
        capture,
        service,
        roles={},
        main_evidence_provider=_main_at,
    )
    bridge.start()
    assert "live_runtime_feature_bridge" in capture._subscribers
    bridge.stop()
    assert "live_runtime_feature_bridge" not in capture._subscribers
    assert not bridge.running
