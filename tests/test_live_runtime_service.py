import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest

from live_runtime.contracts import LiveMode, ProposedAction
from live_runtime.service import LiveSoundcheckService, LiveStartRequest


def _request(mode, mixer_type="wing"):
    return LiveStartRequest(
        mixer_type=mixer_type,
        mixer_ip="10.0.0.5",
        mixer_port=2223,
        audio_device_name="USB",
        num_channels=48,
        selected_channels=[1, 2],
        mode=mode,
    )


class FakeEngine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.state = type("State", (), {"value": "idle"})()
        self.started = False
        self.stopped = False
        self.mixer_client = None
        self._real_mixer_client = None

    def start_async(self):
        self.started = True
        self.state = type("State", (), {"value": "running"})()

    def stop(self):
        self.stopped = True
        self.state = type("State", (), {"value": "stopped"})()

    def get_status(self):
        return {"state": self.state.value, "mixer_connected": True, "audio_running": self.started and not self.stopped}


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


class ConnectedWingEngine(FakeEngine):
    transport = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._real_mixer_client = type(self).transport


def _fader_action(value=-4.0):
    return ProposedAction(
        target="ch:1",
        parameter="fader_db",
        value=value,
        reason="service control-plane cutover test",
        confidence=.9,
        max_step=1.0,
        reversible=True,
        risk="low",
    )


def test_observe_and_propose_do_not_auto_apply_through_bridge():
    created = []

    class CapturingEngine(FakeEngine):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created.append(kwargs)

    service = LiveSoundcheckService(engine_factory=CapturingEngine)
    service.create_engine(_request(LiveMode.OBSERVE))
    service.create_engine(_request(LiveMode.PROPOSE))

    assert created[0]["observe_only"] is True
    assert created[0]["auto_apply"] is False
    assert created[1]["observe_only"] is True
    assert created[1]["auto_apply"] is False


def test_bench_test_enables_legacy_write_path_without_being_inferred():
    created = []

    class CapturingEngine(FakeEngine):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created.append(kwargs)

    service = LiveSoundcheckService(engine_factory=CapturingEngine)
    engine = service.create_engine(_request(LiveMode.BENCH_TEST))

    assert created[0]["observe_only"] is False
    assert created[0]["auto_apply"] is True
    assert engine.live_runtime_mode == "bench_test"


def test_production_write_modes_use_write_capable_bridge():
    for mode in (LiveMode.SUPERVISED, LiveMode.AUTO_SAFE, LiveMode.EMERGENCY):
        created = []

        class CapturingEngine(FakeEngine):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                created.append(kwargs)

        service = LiveSoundcheckService(engine_factory=CapturingEngine)
        service.create_engine(_request(mode))
        assert created[0]["observe_only"] is False
        assert created[0]["auto_apply"] is True


def test_service_owns_start_stop_and_status_lifecycle():
    service = LiveSoundcheckService(engine_factory=FakeEngine)

    engine = service.start(_request(LiveMode.PROPOSE))

    assert engine.started is True
    assert service.active_engine is engine
    assert service.active_mode is LiveMode.PROPOSE
    assert service.is_active() is True
    assert service.get_status()["mode"] == "propose"
    assert service.get_status()["state"] == "running"
    assert service.get_status()["control_plane_ready"] is False

    assert service.stop() is True
    assert engine.stopped is True
    assert service.active_engine is None
    assert service.active_mode is None
    assert service.is_active() is False
    assert service.get_status()["state"] == "idle"


def test_service_rejects_parallel_live_engine_start():
    service = LiveSoundcheckService(engine_factory=FakeEngine)
    service.start(_request(LiveMode.OBSERVE))

    with pytest.raises(RuntimeError, match="already running"):
        service.start(_request(LiveMode.BENCH_TEST))


def test_failed_engine_start_does_not_leave_phantom_active_runtime():
    class BrokenEngine(FakeEngine):
        def start_async(self):
            raise RuntimeError("capture failed")

    service = LiveSoundcheckService(engine_factory=BrokenEngine)

    with pytest.raises(RuntimeError, match="capture failed"):
        service.start(_request(LiveMode.OBSERVE))

    assert service.active_engine is None
    assert service.active_mode is None
    assert service.is_active() is False


def test_bench_test_action_uses_service_owned_wing_control_plane_and_fresh_readback():
    client = FakeWingClient({"/ch/1/fdr": -5.0})
    ConnectedWingEngine.transport = client
    service = LiveSoundcheckService(engine_factory=ConnectedWingEngine)
    service.start(_request(LiveMode.BENCH_TEST))

    result = service.execute_action(_fader_action(-4.0))

    assert result.wrote is True
    assert result.verified.before == -5.0
    assert result.verified.readback == -4.0
    assert result.verified.accepted is True
    assert result.verified.rollback_value == -5.0
    assert client.sent == [
        ("/ch/1/fdr", ()),
        ("/ch/1/fdr", (-4.0,)),
        ("/ch/1/fdr", ()),
    ]
    assert service.get_status()["control_plane_ready"] is True
    assert service.get_status()["control_audit_count"] == 1
    assert service.control_audit_events[0]["event"] == "live_write_verified"


def test_observe_action_is_audited_but_never_writes_wing():
    client = FakeWingClient({"/ch/1/fdr": -5.0})
    ConnectedWingEngine.transport = client
    service = LiveSoundcheckService(engine_factory=ConnectedWingEngine)
    service.start(_request(LiveMode.OBSERVE))

    result = service.execute_action(_fader_action(-4.0))

    assert result.wrote is False
    assert result.authorization_reason == "writes_disabled"
    assert client.values["/ch/1/fdr"] == -5.0
    assert client.sent == [("/ch/1/fdr", ())]
    assert service.control_audit_events[0]["event"] == "live_write_blocked"


def test_service_control_plane_requires_active_physical_wing_transport():
    service = LiveSoundcheckService(engine_factory=FakeEngine)

    with pytest.raises(RuntimeError, match="not running"):
        service.execute_action(_fader_action())

    service.start(_request(LiveMode.BENCH_TEST))
    with pytest.raises(RuntimeError, match="transport is not ready"):
        service.execute_action(_fader_action())


def test_non_wing_runtime_fails_closed_until_adapter_is_migrated():
    client = FakeWingClient({"/ch/1/fdr": -5.0})
    ConnectedWingEngine.transport = client
    service = LiveSoundcheckService(engine_factory=ConnectedWingEngine)
    service.start(_request(LiveMode.BENCH_TEST, mixer_type="dlive"))

    with pytest.raises(NotImplementedError, match="not migrated"):
        service.execute_action(_fader_action())
