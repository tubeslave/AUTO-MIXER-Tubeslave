from mix_agent.backend_bridge import MixAgentBackendBridge
from mix_agent.models import BackendChannelSnapshot, MixAction

from autofoh_models import RuntimeState
from autofoh_safety import AutoFOHSafetyConfig, AutoFOHSafetyController


class FakeMixer:
    def __init__(self):
        self.faders = {1: -10.0}
        self.eq_gain = {}
        self.calls = []

    def get_fader(self, channel):
        return self.faders.get(channel, -12.0)

    def set_fader(self, channel, value_db):
        self.faders[channel] = value_db
        self.calls.append(("set_fader", channel, value_db))
        return True

    def get_eq_band_gain(self, channel, band):
        return self.eq_gain.get((channel, band), 0.0)

    def set_eq_band(self, channel, band, freq, gain, q):
        self.eq_gain[(channel, f"{band}g")] = gain
        self.calls.append(("set_eq_band", channel, band, freq, gain, q))
        return True

    def set_hpf(self, channel, freq_hz, enabled=True):
        self.calls.append(("set_hpf", channel, freq_hz, enabled))
        return True


def test_backend_bridge_blocks_ambiguous_mix_bus_gain():
    bridge = MixAgentBackendBridge(channel_map={"lead_vocal": 1})
    action = MixAction(
        id="gain_stage.lower_mix",
        action_type="gain_adjustment",
        target="mix",
        parameters={"gain_db": -1.0},
    )

    result = bridge.validate_or_apply([action], apply=False)

    assert result.translated == []
    assert result.blocked[0]["reason"] == "mix_bus_gain_changes_require_explicit_operator_target_mapping"


def test_backend_bridge_applies_channel_gain_through_safety_controller():
    mixer = FakeMixer()
    controller = AutoFOHSafetyController(
        mixer,
        config=AutoFOHSafetyConfig(channel_fader_max_step_db=0.5, channel_fader_min_interval_sec=0.0),
    )
    bridge = MixAgentBackendBridge(
        safety_controller=controller,
        snapshots={1: BackendChannelSnapshot(channel_id=1, name="Lead", role="lead_vocal", current_fader_db=-10.0)},
    )
    action = MixAction(
        id="balance.lower_lead",
        action_type="gain_adjustment",
        target="lead_vocal",
        parameters={"gain_db": -2.0},
        reason="test conservative cut",
    )

    result = bridge.validate_or_apply([action], runtime_state=RuntimeState.SOURCE_LEARNING, apply=True)

    assert result.decisions[0]["sent"] is True
    assert result.decisions[0]["bounded"] is True
    assert mixer.calls[-1] == ("set_fader", 1, -10.5)


def test_backend_bridge_keeps_sidechain_as_advisory():
    bridge = MixAgentBackendBridge(channel_map={"guitars": 2})
    action = MixAction(
        id="masking.dynamic_eq",
        action_type="sidechain_suggestion",
        target="guitars",
        parameters={"trigger": "lead_vocal", "frequency_hz": 3000.0},
    )

    result = bridge.validate_or_apply([action], apply=False)

    assert result.translated == []
    assert result.blocked[0]["reason"] == "advisory_placeholder_requires_operator_or_plugin_chain"
