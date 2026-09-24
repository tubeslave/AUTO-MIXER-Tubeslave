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

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.state = type("State", (), {"value": "idle"})()
        self.audio_capture = None
        self.started = False
        self.stopped = False
        self.mixer_client = None
        self._real_mixer_client = None

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
        self._start_audio()
        callback = self.kwargs.get("on_state_change")
        if callback is not None:
            callback("capturing", "audio ready")

    def stop(self):
        if type(self).order is not None:
            type(self).order.append("engine_stop")
        if self.audio_capture is not None:
            self.audio_capture.stop()
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


class FailingEngine(FakeEngine):
    def start_async(self):
        self.started = True
        if type(self).order is not None:
            type(self).order.append("engine_start")
        self._start_audio()
        raise RuntimeError("engine boom")


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


def _service(engine_factory=FakeEngine):
    return LiveSoundcheckService(
        engine_factory=engine_factory,
        capture_bridge_factory=FakeBridge,
        audio_capture_session_factory=FakeCaptureSession,
    )


def setup_function():
    FakeBridge.instances = []
    FakeBridge.fail_start = False
    FakeBridge.order = None
    FakeCaptureSession.instances = []
    FakeCaptureSession.order = None
    FakeEngine.order = None
    FailingEngine.order = None


def test_service_owns_physical_capture_before_legacy_engine_and_bridge_uses_it():
    order = []
    FakeCaptureSession.order = order
    FakeEngine.order = order
    FakeBridge.order = order
    service = _service()

    engine = service.start(_request())

    assert order == ["audio_start", "engine_start", "bridge_start"]
    assert len(FakeCaptureSession.instances) == 1
    session = FakeCaptureSession.instances[0]
    assert session.config.audio_device_name == "USB"
    assert session.config.num_channels == 48
    assert session.config.sample_rate == 48_000
    assert session.config.required_channel_ids == (1, 2, 47, 48)
    assert len(FakeBridge.instances) == 1
    bridge = FakeBridge.instances[0]
    assert bridge.capture is session.capture
    assert engine.audio_capture.physical_capture is session.capture
    assert "legacy_audio_start" not in order
    assert service.get_status()["audio_capture_owned_by_live_runtime"] is True
    assert service.get_status()["capture_bridge_running"] is True


def test_stop_order_keeps_seam_bound_until_legacy_stop_then_closes_physical_capture():
    order = []
    FakeCaptureSession.order = order
    FakeEngine.order = order
    FakeBridge.order = order
    service = _service()
    service.start(_request())
    session = FakeCaptureSession.instances[0]
    engine = service.active_engine
    order.clear()

    assert service.stop() is True

    assert order == ["bridge_stop", "engine_stop", "audio_stop"]
    assert session.capture.stop_calls == 1
    assert engine.audio_capture is None
    assert engine.stopped is True
    assert service.get_status()["audio_capture_owned_by_live_runtime"] is False
    assert service.get_status()["capture_bridge_running"] is False


def test_bridge_start_failure_holds_but_does_not_fall_back_to_legacy_audio_owner():
    FakeBridge.fail_start = True
    service = _service()

    engine = service.start(_request())

    assert engine.started is True
    assert len(FakeCaptureSession.instances) == 1
    assert engine.audio_capture.physical_capture is FakeCaptureSession.instances[0].capture
    status = service.get_status()
    assert status["soundcheck_state"] == "hold"
    assert status["capture_bridge_running"] is False
    assert status["audio_capture_owned_by_live_runtime"] is True
    assert "bridge boom" in status["capture_bridge_error"]

    service.stop()


def test_legacy_start_failure_releases_owned_capture_after_legacy_cleanup():
    order = []
    FakeCaptureSession.order = order
    FailingEngine.order = order
    service = _service(FailingEngine)

    with pytest.raises(RuntimeError, match="engine boom"):
        service.start(_request())

    assert order == ["audio_start", "engine_start", "engine_stop", "audio_stop"]
    session = FakeCaptureSession.instances[0]
    assert session.capture.stop_calls == 1
    assert session.running is False
    assert service.active_engine is None
    assert service.get_status()["audio_capture_owned_by_live_runtime"] is False


def test_unconfigured_compatibility_session_retains_legacy_audio_owner():
    order = []
    FakeCaptureSession.order = order
    FakeEngine.order = order
    service = _service()

    engine = service.start(_request(capture_bridge=False))

    assert FakeCaptureSession.instances == []
    assert order == ["engine_start", "legacy_audio_start"]
    assert isinstance(engine.audio_capture, FakeCapture)
    assert service.get_status()["capture_bridge_configured"] is False
    assert service.get_status()["audio_capture_owned_by_live_runtime"] is False
    assert service.get_status()["soundcheck_state"] == "discover"

    service.stop()
