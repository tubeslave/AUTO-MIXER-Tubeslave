import numpy as np
from pathlib import Path

from auto_panner_adaptive import AutoPannerController, Genre, PanPosition, PanningDecision


class FakeAdaptivePanMixer:
    def __init__(self):
        self.calls = []

    def set_channel_pan(self, channel, pan_value):
        self.calls.append(("set_channel_pan", channel, pan_value))


class StableCrossover:
    def process(self, audio):
        return audio, audio, audio


def test_adaptive_send_osc_is_fail_closed_and_returns_blocked_pan_commands():
    mixer = FakeAdaptivePanMixer()
    controller = AutoPannerController(num_channels=2, mixer_client=mixer)
    decision = PanningDecision(
        channel_id=0,
        position=0.0,
        position_enum=PanPosition.CENTER,
        confidence=0.9,
        genre=Genre.ROCK,
        instrument_type="vocal",
    )

    result = controller._send_osc({0: (0.707, 0.707)}, {0: decision})

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["calls"] == []
    assert result["reason"] == "adaptive_pan_direct_apply_quarantined"
    assert result["write_policy"] == "proposal_only_quarantined"
    assert result["live_write_allowed"] is False
    assert result["can_apply_live"] is False
    assert result["mixer_client_present"] is True
    assert result["mutates_mixer"] is False
    assert mixer.calls == []
    assert {
        "cmd": "set_channel_pan",
        "channel": 0,
        "pan_value": 0.0,
        "position_degrees": 0.0,
        "position_name": "CENTER",
        "instrument_type": "vocal",
        "confidence": 0.9,
    } in result["blocked_calls"]


def test_adaptive_process_with_mixer_client_does_not_write_pan():
    mixer = FakeAdaptivePanMixer()
    controller = AutoPannerController(num_channels=1, sample_rate=48000, mixer_client=mixer)
    controller.crossover = StableCrossover()
    t = np.linspace(0.0, 0.02, 960, endpoint=False)
    audio = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)

    decisions = controller.process({0: audio})
    status = controller.get_status()

    assert decisions
    assert mixer.calls == []
    assert status["live_write_allowed"] is False
    assert status["write_policy"] == "proposal_only_quarantined"
    assert status["mixer_client_present"] is True
    assert status["last_write_attempted"] is True
    assert status["last_write_blocked"] is True
    assert status["last_write_count"] == 0
    assert status["last_blocked_call_count"] > 0


def test_adaptive_panner_has_no_direct_set_channel_pan_call():
    source = Path("backend/auto_panner_adaptive.py").read_text()

    assert ".set_channel_pan(" not in source
    assert '"cmd": "set_channel_pan"' in source
