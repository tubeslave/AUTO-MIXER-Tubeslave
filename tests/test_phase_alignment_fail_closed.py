from pathlib import Path

from phase_alignment import PhaseAlignmentController


class FakePhaseMixer:
    def __init__(self):
        self.calls = []

    def set_channel_delay(self, *args, **kwargs):
        self.calls.append(("set_channel_delay", args, kwargs))

    def set_channel_phase_invert(self, *args, **kwargs):
        self.calls.append(("set_channel_phase_invert", args, kwargs))

    def send(self, *args, **kwargs):
        self.calls.append(("send", args, kwargs))


def test_phase_alignment_apply_is_fail_closed_and_returns_blocked_evidence():
    mixer = FakePhaseMixer()
    controller = PhaseAlignmentController(mixer_client=mixer)
    controller.reference_channel = 1
    controller.channels_to_align = [2]
    controller.all_channels = [1, 2]

    result = controller.apply_corrections(
        {
            (1, 2): {
                "optimal_delay_ms": 3.25,
                "phase_invert": 1,
                "coherence": 0.82,
            }
        }
    )

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["calls"] == []
    assert result["reason"] == "phase_delay_direct_apply_quarantined"
    assert result["write_policy"] == "proposal_only_quarantined"
    assert result["live_write_allowed"] is False
    assert result["can_apply_live"] is False
    assert result["mixer_client_present"] is True
    assert result["mutates_mixer"] is False
    assert mixer.calls == []
    assert {
        "cmd": "set_channel_delay",
        "channel": 2,
        "delay_ms": 3.25,
        "mode": "MS",
        "reference_channel": 1,
        "measured_delay_ms": 3.25,
        "coherence": 0.82,
    } in result["blocked_calls"]
    assert {
        "cmd": "set_channel_phase_invert",
        "channel": 2,
        "phase_invert": 1,
        "reference_channel": 1,
        "coherence": 0.82,
    } in result["blocked_calls"]
    assert controller.corrections[2] == {"delay_ms": 3.25, "phase_invert": 1}


def test_phase_alignment_reset_is_fail_closed_and_preserves_local_corrections():
    mixer = FakePhaseMixer()
    controller = PhaseAlignmentController(mixer_client=mixer)
    controller.corrections = {2: {"delay_ms": 3.25, "phase_invert": 1}}

    result = controller.reset_all_phase_delay([2])

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["calls"] == []
    assert result["reason"] == "phase_delay_reset_quarantined"
    assert result["blocked_call_count"] == 4
    assert mixer.calls == []
    assert controller.corrections == {2: {"delay_ms": 3.25, "phase_invert": 1}}
    assert {
        "cmd": "set_channel_phase_invert",
        "channel": 2,
        "phase_invert": 0,
        "semantic": "reset_phase_invert",
    } in result["blocked_calls"]
    assert {
        "cmd": "send",
        "path": "/ch/2/in/set/dly",
        "value": 0.5,
        "semantic": "reset_delay_value",
        "channel": 2,
    } in result["blocked_calls"]


def test_phase_alignment_has_no_direct_delay_or_polarity_mixer_writes():
    source = Path("backend/phase_alignment.py").read_text()

    assert "mixer_client.set_channel_delay(" not in source
    assert "mixer_client.set_channel_phase_invert(" not in source
    assert "mixer_client.send(" not in source
    assert '"cmd": "set_channel_delay"' in source
    assert '"cmd": "set_channel_phase_invert"' in source
