import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest

from live_runtime.contracts import LiveMode
from live_runtime.service import LiveSoundcheckService, LiveStartRequest


def _request(mode):
    return LiveStartRequest(
        mixer_type="wing",
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

    def start_async(self):
        self.started = True
        self.state = type("State", (), {"value": "running"})()

    def stop(self):
        self.stopped = True
        self.state = type("State", (), {"value": "stopped"})()

    def get_status(self):
        return {"state": self.state.value, "mixer_connected": True, "audio_running": self.started and not self.stopped}


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
