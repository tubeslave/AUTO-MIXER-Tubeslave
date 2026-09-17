"""Check production connection wiring, not just an isolated renderer."""
import sys
import types

import pytest

from auto_soundcheck_engine import AutoSoundcheckEngine
from autofoh_models import RuntimeState
from autofoh_safety import ChannelFaderMove, MasterFaderMove
from guarded_mixer import GuardedMixerClient
from observation_mixer import ObservationMixerClient


class FakeLiveClient:
    is_connected = True
    state = {}
    callbacks = {}
    def __init__(self, **kwargs):
        self.calls = []
    def connect(self, **kwargs):
        return True
    def get_fader(self, channel):
        return -6
    def get_main_fader(self, main):
        return -3
    def set_fader(self, channel, value):
        self.calls.append(("fader", channel, value))
    def set_main_fader(self, main, value):
        self.calls.append(("main", main, value))
    def reset_channel_processing(self, channel):
        self.calls.append(("reset", channel))


@pytest.mark.parametrize("mode,proxy", [("shadow", ObservationMixerClient), ("guarded", GuardedMixerClient)])
def test_real_engine_connection_installs_shadow_policy(monkeypatch, tmp_path, mode, proxy):
    module = types.ModuleType("dlive_client"); module.DLiveClient = FakeLiveClient
    monkeypatch.setitem(sys.modules, "dlive_client", module)
    config = tmp_path / "settings.yaml"
    config.write_text(f"autofoh:\n  shadow:\n    mode: {mode}\n    context: rock:chorus\n")
    engine = AutoSoundcheckEngine(mixer_type="dlive", mixer_ip="127.0.0.1", auto_discover=False,
                                  config_path=str(config))
    assert engine._connect_mixer()
    assert isinstance(engine.mixer_client, proxy)
    assert engine.safety_controller.reviewer is not None
    base = engine.mixer_client._base_client
    engine.mixer_client.reset_channel_processing(1)
    rejected = engine.safety_controller.execute(ChannelFaderMove("test", 1, -6.5),
                                                RuntimeState.FULL_BAND_LEARNING)
    assert not rejected.sent and not base.calls
    emergency = engine.safety_controller.execute(MasterFaderMove("peak", 1, -4),
                                                 RuntimeState.EMERGENCY_SPL)
    if mode == "shadow":
        assert emergency.simulated and not emergency.sent and not base.calls
    else:
        assert emergency.sent and base.calls == [("main", 1, -4)]
