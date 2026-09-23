import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

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


def test_observe_and_propose_do_not_auto_apply_through_bridge():
    created = []

    class FakeEngine:
        def __init__(self, **kwargs):
            created.append(kwargs)

    service = LiveSoundcheckService(engine_factory=FakeEngine)
    service.create_engine(_request(LiveMode.OBSERVE))
    service.create_engine(_request(LiveMode.PROPOSE))

    assert created[0]["observe_only"] is True
    assert created[0]["auto_apply"] is False
    assert created[1]["observe_only"] is True
    assert created[1]["auto_apply"] is False


def test_bench_test_enables_legacy_write_path_without_being_inferred():
    created = []

    class FakeEngine:
        def __init__(self, **kwargs):
            created.append(kwargs)

    service = LiveSoundcheckService(engine_factory=FakeEngine)
    engine = service.create_engine(_request(LiveMode.BENCH_TEST))

    assert created[0]["observe_only"] is False
    assert created[0]["auto_apply"] is True
    assert engine.live_runtime_mode == "bench_test"


def test_production_write_modes_use_write_capable_bridge():
    for mode in (LiveMode.SUPERVISED, LiveMode.AUTO_SAFE, LiveMode.EMERGENCY):
        created = []

        class FakeEngine:
            def __init__(self, **kwargs):
                created.append(kwargs)

        service = LiveSoundcheckService(engine_factory=FakeEngine)
        service.create_engine(_request(mode))
        assert created[0]["observe_only"] is False
        assert created[0]["auto_apply"] is True
