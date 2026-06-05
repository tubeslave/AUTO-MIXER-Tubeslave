import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from auto_reverb import AutoReverb


class FakeReverbMixer:
    def __init__(self):
        self.calls = []

    def set_reverb_send(self, channel, level):
        self.calls.append(("set_reverb_send", channel, level))

    def set_reverb_params(self, channel, **params):
        self.calls.append(("set_reverb_params", channel, params))


def test_auto_reverb_calculates_dry_side_decisions():
    reverb = AutoReverb()

    decisions = reverb.calculate_reverb(
        channels=[1, 2],
        instrument_types={1: "lead_vocal", 2: "snare"},
        spectral_centroids={1: 800.0, 2: 4500.0},
        spectral_fluxes={2: 5.0},
    )

    lead = decisions[1]
    snare = decisions[2]
    assert lead.reverb_type == "plate"
    assert lead.predelay_ms == 40.0
    assert lead.reverb_level_lu == -16.0
    assert snare.decay_time_s < 0.8
    assert snare.brightness == 0.3


def test_apply_to_mixer_is_fail_closed_and_returns_blocked_reverb_commands():
    reverb = AutoReverb()
    decisions = reverb.calculate_reverb(
        channels=[1],
        instrument_types={1: "lead_vocal"},
    )
    mixer = FakeReverbMixer()

    result = reverb.apply_to_mixer(mixer, decisions)

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["calls"] == []
    assert result["reason"] == "reverb_direct_apply_quarantined"
    assert result["write_policy"] == "proposal_only_quarantined"
    assert result["live_write_allowed"] is False
    assert result["can_apply_live"] is False
    assert result["mixer_client_present"] is True
    assert result["mutates_mixer"] is False
    assert mixer.calls == []
    assert {
        "cmd": "set_reverb_send",
        "channel": 1,
        "level_lu": decisions[1].reverb_level_lu,
    } in result["blocked_calls"]
    assert {
        "cmd": "set_reverb_params",
        "channel": 1,
        "reverb_type": "plate",
        "decay_time": decisions[1].decay_time_s,
        "predelay": decisions[1].predelay_ms,
        "brightness": decisions[1].brightness,
        "density": decisions[1].density,
        "hpf": decisions[1].hpf_hz,
        "lpf": decisions[1].lpf_hz,
    } in result["blocked_calls"]


def test_apply_to_mixer_without_decisions_reports_no_mutation():
    reverb = AutoReverb()
    mixer = FakeReverbMixer()

    result = reverb.apply_to_mixer(mixer, {})

    assert result["applied"] is False
    assert result["blocked"] is False
    assert result["blocked_calls"] == []
    assert result["reason"] == "no_decisions"
    assert result["mutates_mixer"] is False
    assert mixer.calls == []
