from dataclasses import dataclass

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
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.state = type("State", (), {"value": "idle"})()
        self.audio_capture = None
        self.started = False
        self.stopped = False
        self.mixer_client = None
        self._real_mixer_client = None

    def start_async(self):
        self.started = True
        self.state = type("State", (), {"value": "starting"})()

    def make_capture_ready(self):
        self.audio_capture = FakeCapture()
        callback = self.kwargs.get("on_state_change")
        if callback is not None:
            callback("capturing", "audio ready")

    def stop(self):
        if FakeBridge.order is not None:
            FakeBridge.order.append("engine_stop")
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


class SynchronousCaptureEngine(FakeEngine):
    def start_async(self):
        super().start_async()
        self.make_capture_ready()


def _contract():
    return MainTapPatchContract(
        tap=PostConsoleMainTap(left_channel=47, right_channel=48),
        routes=(
            MainTapRouteExpectation(usb_slot=47, source_channel=1),
            MainTapRouteExpectation(usb_slot=48, source_channel=2),
        ),
    )


def _request():
    return LiveStartRequest(
        mixer_type="wing",
        mixer_ip="10.0.0.5",
        mixer_port=2223,
        audio_device_name="USB",
        num_channels=48,
        selected_channels=[1, 2],
        mode=LiveMode.OBSERVE,
        capture_bridge=LiveCaptureBridgeConfig(
            patch_contract=_contract(),
            roles={1: "kick", 2: "snare"},
            channel_names={1: "Kick In", 2: "Snare Top"},
            window_frames=1024,
            analysis_interval_s=0.0,
        ),
    )


def setup_function():
    FakeBridge.instances = []
    FakeBridge.fail_start = False
    FakeBridge.order = None


def test_service_starts_bridge_when_async_capture_becomes_ready():
    service = LiveSoundcheckService(
        engine_factory=FakeEngine,
        capture_bridge_factory=FakeBridge,
    )
    engine = service.start(_request())

    assert FakeBridge.instances == []
    assert service.get_status()["capture_bridge_configured"] is True
    assert service.get_status()["capture_bridge_running"] is False

    engine.make_capture_ready()

    assert len(FakeBridge.instances) == 1
    bridge = FakeBridge.instances[0]
    assert bridge.running is True
    assert bridge.capture is engine.audio_capture
    assert bridge.kwargs["roles"] == {1: "kick", 2: "snare"}
    assert bridge.kwargs["channel_names"] == {1: "Kick In", 2: "Snare Top"}
    assert bridge.kwargs["patch_contract"] == _contract()
    assert bridge.kwargs["snapshot_main_evidence_provider"].reserved_capture_channels == (47, 48)
    assert service.get_status()["capture_bridge_running"] is True
    assert service.control_audit_events[-1]["event"] == "live_capture_bridge_started"


def test_service_also_catches_capture_exposed_during_start_async():
    service = LiveSoundcheckService(
        engine_factory=SynchronousCaptureEngine,
        capture_bridge_factory=FakeBridge,
    )

    service.start(_request())

    assert len(FakeBridge.instances) == 1
    assert FakeBridge.instances[0].running is True


def test_stop_unsubscribes_bridge_before_legacy_engine_and_does_not_restart_it():
    FakeBridge.order = []
    service = LiveSoundcheckService(
        engine_factory=SynchronousCaptureEngine,
        capture_bridge_factory=FakeBridge,
    )
    service.start(_request())
    engine = service.active_engine

    assert service.stop() is True

    assert FakeBridge.order == ["bridge_start", "bridge_stop", "engine_stop"]
    assert len(FakeBridge.instances) == 1
    assert engine.stopped is True
    status = service.get_status()
    assert status["capture_bridge_running"] is False
    assert status["capture_bridge_configured"] is False


def test_bridge_start_failure_fails_closed_to_hold_without_raising_from_legacy_callback():
    FakeBridge.fail_start = True
    service = LiveSoundcheckService(
        engine_factory=SynchronousCaptureEngine,
        capture_bridge_factory=FakeBridge,
    )

    engine = service.start(_request())

    assert engine.started is True
    status = service.get_status()
    assert status["soundcheck_state"] == "hold"
    assert status["capture_bridge_running"] is False
    assert "bridge boom" in status["capture_bridge_error"]
    assert service.control_audit_events[-1]["event"] == "live_capture_bridge_start_failed"


def test_unconfigured_compatibility_session_does_not_attach_bridge():
    service = LiveSoundcheckService(
        engine_factory=SynchronousCaptureEngine,
        capture_bridge_factory=FakeBridge,
    )
    request = LiveStartRequest(
        mixer_type="wing",
        mixer_ip="10.0.0.5",
        mixer_port=2223,
        audio_device_name="USB",
        num_channels=48,
        selected_channels=[1, 2],
        mode=LiveMode.OBSERVE,
    )

    service.start(request)

    assert FakeBridge.instances == []
    assert service.get_status()["capture_bridge_configured"] is False
    assert service.get_status()["soundcheck_state"] == "discover"
