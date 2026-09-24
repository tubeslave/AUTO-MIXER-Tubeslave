from dataclasses import dataclass

import pytest

from backend.live_runtime.contracts import LiveMode
from backend.live_runtime.main_evidence import PostConsoleMainTap
from backend.live_runtime.patch_verify import MainTapPatchContract, MainTapRouteExpectation
from backend.live_runtime.service import (
    LiveCaptureBridgeConfig,
    LiveSoundcheckService,
    LiveStartRequest,
)
from backend.live_runtime.session_lifecycle import LiveSessionLifecycle


class FakeCapture:
    sample_rate = 48_000
    num_channels = 48

    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


class FakeCaptureSession:
    instances = []
    order = None

    def __init__(self, config, *, audit_sink=None):
        self.config = config
        self.audit_sink = audit_sink
        self.capture = FakeCapture()
        self.running = False
        type(self).instances.append(self)

    def start(self):
        self.running = True
        if type(self).order is not None:
            type(self).order.append("audio_start")
        return self.capture

    def stop(self):
        if not self.running:
            return False
        self.running = False
        self.capture.stop()
        if type(self).order is not None:
            type(self).order.append("audio_stop")
        return True


class FailingCaptureSession(FakeCaptureSession):
    instances = []
    order = None

    def start(self):
        if type(self).order is not None:
            type(self).order.append("audio_start_failed")
        raise RuntimeError("audio owner boom")


class FakeMixerClient:
    def __init__(self):
        self.is_connected = True
        self.disconnect_calls = 0
        self.sent = []
        self.subscriptions = []

    def send(self, *args, **kwargs):
        self.sent.append((args, kwargs))

    def subscribe(self, *args, **kwargs):
        self.subscriptions.append((args, kwargs))

    def disconnect(self):
        self.disconnect_calls += 1
        self.is_connected = False


@dataclass(frozen=True)
class FakeMixerTarget:
    mixer_type: str = "wing"
    ip: str = "10.0.0.5"
    port: int = 2223


class FakeMixerSession:
    instances = []
    order = None

    def __init__(self, config, *, audit_sink=None):
        self.config = config
        self.audit_sink = audit_sink
        self.client = None
        self.target = None
        self.connected = False
        type(self).instances.append(self)

    def start(self):
        self.client = FakeMixerClient()
        self.target = FakeMixerTarget(
            mixer_type=self.config.mixer_type or "wing",
            ip=self.config.mixer_ip or "10.0.0.5",
            port=self.config.mixer_port or 2223,
        )
        self.connected = True
        if type(self).order is not None:
            type(self).order.append("mixer_start")
        return self.client

    def stop(self):
        if not self.connected:
            return False
        self.connected = False
        self.client.disconnect()
        if type(self).order is not None:
            type(self).order.append("mixer_stop")
        return True


@dataclass(frozen=True)
class FakeBridgeStatus:
    running: bool
    snapshots_captured: int = 0
    snapshots_processed: int = 0
    snapshots_replaced: int = 0
    incomplete_windows: int = 0
    missing_main_evidence: int = 0
    processing_failures: int = 0
    last_error: str | None = None


class FakeBridge:
    instances = []
    fail_start = False
    order = None

    def __init__(self, capture, service, **kwargs):
        self.capture = capture
        self.service = service
        self.kwargs = kwargs
        self.running = False
        self.stopped = False
        type(self).instances.append(self)

    def start(self):
        if type(self).fail_start:
            raise RuntimeError("bridge boom")
        self.running = True
        if type(self).order is not None:
            type(self).order.append("bridge_start")

    def stop(self):
        self.running = False
        self.stopped = True
        if type(self).order is not None:
            type(self).order.append("bridge_stop")

    def status(self):
        return FakeBridgeStatus(running=self.running)


class FakeEngine:
    order = None
    legacy_discover_calls = 0
    legacy_connect_calls = 0
    constructed = 0

    def __init__(self, **kwargs):
        type(self).constructed += 1
        self.kwargs = kwargs
        self.state = type("State", (), {"value": "idle"})()
        self.audio_capture = None
        self.started = False
        self.stopped = False
        self.mixer_client = None
        self._real_mixer_client = None
        self.mixer_type = kwargs.get("mixer_type")
        self.mixer_ip = kwargs.get("mixer_ip")
        self.mixer_port = kwargs.get("mixer_port")

    def _discover_mixer(self):
        type(self).legacy_discover_calls += 1
        if type(self).order is not None:
            type(self).order.append("legacy_mixer_discover")
        return True

    def _connect_mixer(self):
        type(self).legacy_connect_calls += 1
        if type(self).order is not None:
            type(self).order.append("legacy_mixer_connect")
        client = FakeMixerClient()
        self.mixer_client = client
        self._real_mixer_client = client
        return True

    def _start_audio(self):
        if type(self).order is not None:
            type(self).order.append("legacy_audio_start")
        self.audio_capture = FakeCapture()
        return True

    def start_async(self):
        self.started = True
        self.state = type("State", (), {"value": "starting"})()
        if type(self).order is not None:
            type(self).order.append("engine_start")
        assert self._discover_mixer()
        assert self._connect_mixer()
        self._start_audio()
        callback = self.kwargs.get("on_state_change")
        if callback is not None:
            callback("capturing", "audio ready")

    def stop(self):
        if type(self).order is not None:
            type(self).order.append("engine_stop")
        if self.audio_capture is not None:
            self.audio_capture.stop()
        if self.mixer_client is not None:
            try:
                self.mixer_client.disconnect()
            except Exception:
                pass
        callback = self.kwargs.get("on_state_change")
        if callback is not None:
            callback("stopped", "stopped")
        self.stopped = True
        self.state = type("State", (), {"value": "stopped"})()

    def get_status(self):
        return {
            "state": self.state.value,
            "mixer_connected": True,
            "audio_running": self.started and not self.stopped,
        }


def _contract():
    return MainTapPatchContract(
        tap=PostConsoleMainTap(left_channel=47, right_channel=48),
        routes=(
            MainTapRouteExpectation(usb_slot=47, source_channel=1),
            MainTapRouteExpectation(usb_slot=48, source_channel=2),
        ),
    )


def _request(*, capture_bridge=True):
    bridge = None
    if capture_bridge:
        bridge = LiveCaptureBridgeConfig(
            patch_contract=_contract(),
            roles={1: "kick", 2: "snare"},
            channel_names={1: "Kick In", 2: "Snare Top"},
            window_frames=1024,
            analysis_interval_s=0.0,
        )
    return LiveStartRequest(
        mixer_type="wing",
        mixer_ip="10.0.0.5",
        mixer_port=2223,
        audio_device_name="USB",
        num_channels=48,
        selected_channels=[1, 2],
        mode=LiveMode.OBSERVE,
        capture_bridge=bridge,
    )


def _service(engine_factory=FakeEngine, *, capture_session_factory=FakeCaptureSession):
    return LiveSoundcheckService(
        engine_factory=engine_factory,
        capture_bridge_factory=FakeBridge,
        audio_capture_session_factory=capture_session_factory,
        mixer_session_factory=FakeMixerSession,
    )


def setup_function():
    FakeBridge.instances = []
    FakeBridge.fail_start = False
    FakeBridge.order = None
    FakeCaptureSession.instances = []
    FakeCaptureSession.order = None
    FailingCaptureSession.instances = []
    FailingCaptureSession.order = None
    FakeMixerSession.instances = []
    FakeMixerSession.order = None
    FakeEngine.order = None
    FakeEngine.legacy_discover_calls = 0
    FakeEngine.legacy_connect_calls = 0
    FakeEngine.constructed = 0


def test_configured_service_owns_hardware_without_constructing_legacy_engine():
    order = []
    FakeMixerSession.order = order
    FakeCaptureSession.order = order
    FakeEngine.order = order
    FakeBridge.order = order
    service = _service()

    handle = service.start(_request())

    assert isinstance(handle, LiveSessionLifecycle)
    assert handle.state.value == "running"
    assert order == ["mixer_start", "audio_start", "bridge_start"]
    assert FakeEngine.constructed == 0
    assert FakeEngine.legacy_discover_calls == 0
    assert FakeEngine.legacy_connect_calls == 0

    assert len(FakeMixerSession.instances) == 1
    mixer_session = FakeMixerSession.instances[0]
    assert mixer_session.config.mixer_type == "wing"
    assert mixer_session.config.mixer_ip == "10.0.0.5"
    assert mixer_session.config.mixer_port == 2223
    assert service._active_wing_transport() is mixer_session.client

    assert len(FakeCaptureSession.instances) == 1
    session = FakeCaptureSession.instances[0]
    assert session.config.audio_device_name == "USB"
    assert session.config.num_channels == 48
    assert session.config.sample_rate == 48_000
    assert session.config.required_channel_ids == (1, 2, 47, 48)

    assert len(FakeBridge.instances) == 1
    assert FakeBridge.instances[0].capture is session.capture
    status = service.get_status()
    assert status["state"] == "running"
    assert status["legacy_engine_attached"] is False
    assert status["mixer_transport_owned_by_live_runtime"] is True
    assert status["mixer_connected"] is True
    assert status["control_plane_ready"] is True
    assert status["audio_capture_owned_by_live_runtime"] is True
    assert status["capture_bridge_running"] is True


def test_stop_order_has_no_legacy_engine_and_closes_physical_owners_once():
    order = []
    FakeMixerSession.order = order
    FakeCaptureSession.order = order
    FakeEngine.order = order
    FakeBridge.order = order
    service = _service()
    handle = service.start(_request())
    audio_session = FakeCaptureSession.instances[0]
    mixer_session = FakeMixerSession.instances[0]
    physical_client = mixer_session.client
    order.clear()

    assert service.stop() is True

    assert order == ["bridge_stop", "audio_stop", "mixer_stop"]
    assert "engine_stop" not in order
    assert audio_session.capture.stop_calls == 1
    assert physical_client.disconnect_calls == 1
    assert handle.state.value == "stopped"
    assert FakeEngine.constructed == 0
    status = service.get_status()
    assert status["audio_capture_owned_by_live_runtime"] is False
    assert status["mixer_transport_owned_by_live_runtime"] is False
    assert status["capture_bridge_running"] is False


def test_bridge_start_failure_holds_without_falling_back_to_legacy_engine():
    FakeBridge.fail_start = True
    service = _service()

    handle = service.start(_request())

    assert isinstance(handle, LiveSessionLifecycle)
    assert handle.state.value == "running"
    assert FakeEngine.constructed == 0
    assert len(FakeMixerSession.instances) == 1
    assert len(FakeCaptureSession.instances) == 1
    status = service.get_status()
    assert status["soundcheck_state"] == "hold"
    assert status["capture_bridge_running"] is False
    assert status["mixer_transport_owned_by_live_runtime"] is True
    assert status["audio_capture_owned_by_live_runtime"] is True
    assert status["legacy_engine_attached"] is False
    assert "bridge boom" in status["capture_bridge_error"]

    service.stop()


def test_canonical_audio_owner_start_failure_releases_mixer_without_legacy_fallback():
    order = []
    FakeMixerSession.order = order
    FailingCaptureSession.order = order
    FakeEngine.order = order
    service = _service(capture_session_factory=FailingCaptureSession)

    with pytest.raises(RuntimeError, match="audio owner boom"):
        service.start(_request())

    assert order == ["mixer_start", "audio_start_failed", "mixer_stop"]
    assert FakeEngine.constructed == 0
    mixer_session = FakeMixerSession.instances[0]
    assert mixer_session.client.disconnect_calls == 1
    assert mixer_session.connected is False
    assert service.active_engine is None
    assert service.is_active() is False
    status = service.get_status()
    assert status["audio_capture_owned_by_live_runtime"] is False
    assert status["mixer_transport_owned_by_live_runtime"] is False


def test_returned_canonical_handle_stops_through_service_owner():
    service = _service()
    handle = service.start(_request())

    assert handle.stop() is True
    assert handle.state.value == "stopped"
    assert service.is_active() is False
    assert FakeCaptureSession.instances[0].capture.stop_calls == 1
    assert FakeMixerSession.instances[0].client.disconnect_calls == 1


def test_unconfigured_compatibility_session_retains_legacy_hardware_owners():
    order = []
    FakeMixerSession.order = order
    FakeCaptureSession.order = order
    FakeEngine.order = order
    service = _service()

    engine = service.start(_request(capture_bridge=False))

    assert FakeEngine.constructed == 1
    assert FakeMixerSession.instances == []
    assert FakeCaptureSession.instances == []
    assert order == [
        "engine_start",
        "legacy_mixer_discover",
        "legacy_mixer_connect",
        "legacy_audio_start",
    ]
    assert FakeEngine.legacy_discover_calls == 1
    assert FakeEngine.legacy_connect_calls == 1
    assert isinstance(engine.audio_capture, FakeCapture)
    status = service.get_status()
    assert status["capture_bridge_configured"] is False
    assert status["audio_capture_owned_by_live_runtime"] is False
    assert status["mixer_transport_owned_by_live_runtime"] is False
    assert status["legacy_engine_attached"] is True
    assert status["soundcheck_state"] == "discover"

    service.stop()
