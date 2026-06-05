from pathlib import Path

from auto_eq import AutoEQController, EQBand, MultiChannelAutoEQController


class FakeEQMixer:
    is_connected = True

    def __init__(self):
        self.calls = []

    def set_eq_on(self, *args, **kwargs):
        self.calls.append(("set_eq_on", args, kwargs))

    def set_eq_low_shelf(self, *args, **kwargs):
        self.calls.append(("set_eq_low_shelf", args, kwargs))

    def set_eq_high_shelf(self, *args, **kwargs):
        self.calls.append(("set_eq_high_shelf", args, kwargs))

    def set_eq_band(self, *args, **kwargs):
        self.calls.append(("set_eq_band", args, kwargs))

    def send(self, *args, **kwargs):
        self.calls.append(("send", args, kwargs))


def _corrections():
    return [
        EQBand("peak", 120.0, -3.0, 1.2),
        EQBand("peak", 800.0, 2.0, 1.5),
        EQBand("peak", 5000.0, 1.5, 2.0),
    ]


def test_auto_eq_apply_to_mixer_is_fail_closed_evidence():
    mixer = FakeEQMixer()
    controller = AutoEQController(mixer_client=mixer)
    controller.current_channel = 2
    controller.current_corrections = _corrections()

    result = controller.apply_to_mixer()
    status = controller.get_status()

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["calls"] == []
    assert result["reason"] == "auto_eq_direct_apply_quarantined"
    assert result["write_policy"] == "proposal_only_quarantined"
    assert result["live_write_allowed"] is False
    assert result["can_apply_live"] is False
    assert result["mixer_client_present"] is True
    assert result["mutates_mixer"] is False
    assert mixer.calls == []
    assert controller.applied_corrections_history == []
    assert status["last_write_attempted"] is True
    assert status["last_write_blocked"] is True
    assert status["last_write_count"] == 0
    assert status["last_blocked_call_count"] == len(result["blocked_calls"])
    assert {"cmd": "set_eq_on", "channel": 2, "on": 1, "semantic": "enable_eq_before_apply"} in result["blocked_calls"]
    assert any(call["cmd"] == "set_eq_low_shelf" for call in result["blocked_calls"])
    assert any(call["cmd"] == "set_eq_band" for call in result["blocked_calls"])
    assert any(call["cmd"] == "set_eq_high_shelf" for call in result["blocked_calls"])


def test_auto_eq_reset_is_fail_closed_and_preserves_corrections():
    mixer = FakeEQMixer()
    controller = AutoEQController(mixer_client=mixer)
    controller.current_channel = 4
    controller.current_corrections = _corrections()

    result = controller.reset_eq()

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["calls"] == []
    assert result["reason"] == "auto_eq_reset_quarantined"
    assert result["mutates_mixer"] is False
    assert result["blocked_call_count"] == 7
    assert mixer.calls == []
    assert controller.current_corrections
    assert {
        "cmd": "send",
        "path": "/ch/4/eq/2g",
        "value": 0.0,
        "semantic": "reset_peak_band_gain",
        "channel": 4,
        "band": 2,
    } in result["blocked_calls"]


def test_multi_channel_auto_eq_apply_is_fail_closed_evidence():
    mixer = FakeEQMixer()
    controller = MultiChannelAutoEQController(mixer_client=mixer)
    controller.active_channels[7] = {
        "analyzer": None,
        "profile": None,
        "auto_apply": False,
        "corrections": _corrections(),
        "spectrum_buffer": [],
        "analysis_start_time": None,
        "device_id": None,
    }

    result = controller.apply_channel_correction(7)
    status = controller.get_status()

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["reason"] == "auto_eq_direct_apply_quarantined"
    assert result["mutates_mixer"] is False
    assert mixer.calls == []
    assert status["last_write_attempted"] is True
    assert status["last_blocked_call_count"] == len(result["blocked_calls"])


def test_auto_eq_module_has_no_direct_mixer_eq_writes():
    source = Path("backend/auto_eq.py").read_text()

    assert "mixer_client.set_eq_on(" not in source
    assert "mixer_client.set_eq_low_shelf(" not in source
    assert "mixer_client.set_eq_high_shelf(" not in source
    assert "mixer_client.set_eq_band(" not in source
    assert "mixer_client.send(" not in source
    assert '"cmd": "set_eq_on"' in source
    assert '"cmd": "set_eq_band"' in source
