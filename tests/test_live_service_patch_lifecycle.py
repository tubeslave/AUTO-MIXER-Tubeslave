import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from live_runtime.capture_bridge import LiveAudioCaptureBridge
from live_runtime.contracts import ChannelFeatures, LiveMode, MixFeatures, SoundcheckState
from live_runtime.feature_stream import MainFeatureEvidence
from live_runtime.main_evidence import PostConsoleMainTap, PostConsoleMainTapEvidenceProvider
from live_runtime.patch_startup import MainTapPatchStartupCoordinator
from live_runtime.patch_verify import (
    MainTapPatchContract,
    MainTapPatchVerifier,
    MainTapRouteExpectation,
    PhysicalMainMeterEvidence,
)
from live_runtime.service import LiveSoundcheckService, LiveStartRequest


class FakeWingClient:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.callbacks = {}
        self.sent = []

    def subscribe(self, address, callback):
        self.callbacks.setdefault(address, []).append(callback)

    def send(self, address, *values):
        self.sent.append((address, values))
        if values:
            self.values[address] = values[0] if len(values) == 1 else values
            return True
        actual = self.values.get(address)
        for callback in self.callbacks.get(address, []):
            callback(address, str(actual), 0.5, actual)
        return True


class FakeMeterProvider:
    def __init__(self, evidence=None):
        self.evidence = evidence or PhysicalMainMeterEvidence(
            peak_dbfs=-6.4,
            rms_dbfs=-18.3,
            timestamp_s=10.05,
            source="test-main-meter",
        )
        self.calls = 0

    def read(self):
        self.calls += 1
        return self.evidence


class FakeCapture:
    sample_rate = 48_000
    num_channels = 48

    def __init__(self):
        self._subscribers = {}
        self._buffers = {
            channel: np.zeros(2048, dtype=np.float32)
            for channel in range(1, self.num_channels + 1)
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


class ConnectedWingEngine:
    transport = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.state = type("State", (), {"value": "idle"})()
        self._real_mixer_client = type(self).transport
        self.mixer_client = self._real_mixer_client

    def start_async(self):
        self.state = type("State", (), {"value": "running"})()

    def stop(self):
        self.state = type("State", (), {"value": "stopped"})()

    def get_status(self):
        return {"state": self.state.value, "mixer_connected": True, "audio_running": True}


def _request():
    return LiveStartRequest(
        mixer_type="wing",
        mixer_ip="10.0.0.5",
        mixer_port=2223,
        audio_device_name="USB",
        num_channels=48,
        selected_channels=[1, 2],
        mode=LiveMode.AUTO_SAFE,
    )


def _contract():
    return MainTapPatchContract(
        tap=PostConsoleMainTap(left_channel=47, right_channel=48),
        routes=(
            MainTapRouteExpectation(usb_slot=47, source_channel=1),
            MainTapRouteExpectation(usb_slot=48, source_channel=2),
        ),
    )


def _tap():
    return MainFeatureEvidence(
        rms_dbfs=-18.0,
        peak_dbfs=-6.0,
        crest_db=12.0,
        timestamp_s=10.0,
    )


def _features():
    return MixFeatures(
        channels=[
            ChannelFeatures(
                channel=1,
                name="Lead Vocal",
                rms_dbfs=-20.0,
                peak_dbfs=-9.0,
                crest_db=11.0,
                activity=0.8,
            )
        ],
        main_rms_dbfs=-14.0,
        main_peak_dbfs=-0.5,
        main_crest_db=13.5,
    )


def _routes():
    return {
        "/io/out/USB/46/grp": "MAIN",
        "/io/out/USB/46/in": 1,
        "/io/out/USB/47/grp": "MAIN",
        "/io/out/USB/47/in": 2,
    }


def _service(client, meter):
    ConnectedWingEngine.transport = client

    def factory(adapter, _host, audit_sink):
        return MainTapPatchStartupCoordinator(
            MainTapPatchVerifier(adapter),
            meter,
            audit_sink=audit_sink,
        )

    service = LiveSoundcheckService(
        engine_factory=ConnectedWingEngine,
        patch_startup_factory=factory,
    )
    service.start(_request())
    return service


def _wait_for(predicate, timeout=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_service_starts_in_discover_and_blocks_feature_loop_before_patch_verify():
    client = FakeWingClient(_routes())
    meter = FakeMeterProvider()
    service = _service(client, meter)

    assert service.get_status()["soundcheck_state"] == "discover"

    result = service.process_feature_snapshot(_features(), roles={})

    assert result.state is SoundcheckState.DISCOVER
    assert result.reason == "startup_state_blocked:discover"
    assert client.sent == []
    assert meter.calls == 0


def test_service_patch_verify_success_advances_to_listen_and_exposes_status():
    client = FakeWingClient(_routes())
    meter = FakeMeterProvider()
    service = _service(client, meter)

    result = service.verify_main_tap_patch(_contract(), _tap())
    status = service.get_status()

    assert result.verified is True
    assert result.state is SoundcheckState.LISTEN
    assert status["soundcheck_state"] == "listen"
    assert status["patch_verify_verified"] is True
    assert status["patch_verify_reason"] == "verified"
    assert status["patch_verify_physical_source"] == "test-main-meter"
    assert meter.calls == 1
    assert all(values == () for _, values in client.sent)
    assert [event["event"] for event in service.control_audit_events] == [
        "live_state_transition",
        "live_patch_verify_complete",
    ]


def test_service_patch_verify_route_failure_advances_to_hold_without_meter_read():
    routes = _routes()
    routes["/io/out/USB/47/grp"] = "BUS"
    client = FakeWingClient(routes)
    meter = FakeMeterProvider()
    service = _service(client, meter)

    result = service.verify_main_tap_patch(_contract(), _tap())
    status = service.get_status()

    assert result.verified is False
    assert result.state is SoundcheckState.HOLD
    assert result.reason.startswith("route_failed:")
    assert status["soundcheck_state"] == "hold"
    assert status["patch_verify_verified"] is False
    assert status["patch_verify_reason"].startswith("route_failed:")
    assert meter.calls == 0
    assert all(values == () for _, values in client.sent)


def test_capture_bridge_closes_discover_patch_verify_listen_seam():
    client = FakeWingClient(_routes())
    meter = FakeMeterProvider(
        PhysicalMainMeterEvidence(
            peak_dbfs=-6.02,
            rms_dbfs=-6.02,
            timestamp_s=10.02,
            source="test-main-meter",
        )
    )
    service = _service(client, meter)
    capture = FakeCapture()
    bridge = LiveAudioCaptureBridge(
        capture,
        service,
        roles={},
        snapshot_main_evidence_provider=PostConsoleMainTapEvidenceProvider(47, 48),
        patch_contract=_contract(),
        window_frames=2048,
        analysis_interval_s=0.0,
        wall_clock=lambda: 10.0,
    )
    block = np.zeros((2048, 48), dtype=np.float32)
    block[:, 46] = 0.5
    block[:, 47] = 0.5

    bridge.start()
    try:
        capture.emit(block)
        assert _wait_for(lambda: service.get_status()["soundcheck_state"] == "listen")
        assert meter.calls == 1
        assert bridge.status().snapshots_processed == 1
        assert service.get_status()["patch_verify_verified"] is True

        # The proof frame is consumed only by startup. A later coherent frame is
        # the first one eligible for the normal Director/Critic loop.
        capture.emit(block)
        assert _wait_for(lambda: bridge.status().snapshots_processed == 2)
        assert meter.calls == 1
        assert service.get_status()["soundcheck_state"] not in {"discover", "patch_verify"}
    finally:
        bridge.stop()


def test_stop_clears_startup_proof_state():
    client = FakeWingClient(_routes())
    meter = FakeMeterProvider()
    service = _service(client, meter)
    assert service.verify_main_tap_patch(_contract(), _tap()).verified is True

    assert service.stop() is True
    status = service.get_status()

    assert status["soundcheck_state"] == "idle"
    assert status["patch_verify_verified"] is None
    assert status["patch_verify_reason"] is None
