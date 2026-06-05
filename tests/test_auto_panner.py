"""Tests for spectral auto-panning heuristics."""

from pathlib import Path

from auto_panner import AutoPanner


class FakePanMixer:
    def __init__(self):
        self.calls = []

    def set_channel_pan(self, channel, pan_percent):
        self.calls.append(("set_channel_pan", channel, pan_percent))


def test_low_frequency_dominant_band_is_centered():
    panner = AutoPanner()

    decisions = panner.calculate_panning(
        channels=[1],
        instrument_types={1: "unknown"},
        spectral_centroids={1: 1000.0},
        channel_band_energy={1: {"sub": -10.0, "mid": -30.0}},
    )

    assert decisions[1].pan_percent == 0.0
    assert decisions[1].dominant_band == "sub"


def test_same_dominant_band_is_distributed_with_input_priority():
    panner = AutoPanner(max_pan_percent=80.0)

    channel_band_energy = {
        5: {"high_mid": -10.0, "bass": -40.0},
        2: {"high_mid": -11.0, "bass": -40.0},
        8: {"high_mid": -12.0, "bass": -40.0},
        4: {"high_mid": -13.0, "bass": -40.0},
    }
    decisions = panner.calculate_panning(
        channels=[5, 2, 8, 4],
        instrument_types={ch: "guitar" for ch in channel_band_energy},
        spectral_centroids={ch: 3000.0 for ch in channel_band_energy},
        channel_band_energy=channel_band_energy,
    )

    pans = {ch: decision.pan_percent for ch, decision in decisions.items()}
    assert any(value < 0 for value in pans.values())
    assert any(value > 0 for value in pans.values())
    assert abs(pans[2]) < abs(pans[5])
    assert abs(pans[4]) < abs(pans[8])


def test_apply_to_mixer_is_fail_closed_and_returns_blocked_pan_commands():
    panner = AutoPanner(max_pan_percent=80.0)
    decisions = panner.calculate_panning(
        channels=[1, 2],
        instrument_types={1: "lead_vocal", 2: "electric_guitar"},
        spectral_centroids={1: 800.0, 2: 3000.0},
    )
    mixer = FakePanMixer()

    result = panner.apply_to_mixer(mixer, decisions)

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["calls"] == []
    assert result["reason"] == "pan_direct_apply_quarantined"
    assert result["write_policy"] == "proposal_only_quarantined"
    assert result["live_write_allowed"] is False
    assert result["can_apply_live"] is False
    assert result["mixer_client_present"] is True
    assert result["mutates_mixer"] is False
    assert mixer.calls == []
    assert {
        "cmd": "set_channel_pan",
        "channel": 1,
        "pan_percent": decisions[1].pan_percent,
        "instrument": "lead_vocal",
        "reason": decisions[1].reason,
    } in result["blocked_calls"]


def test_auto_fader_v2_pan_separation_has_no_direct_set_channel_pan_call():
    source = Path("backend/auto_fader_v2/controller.py").read_text()

    assert ".set_channel_pan(" not in source
    assert '"cmd": "set_channel_pan"' in source
