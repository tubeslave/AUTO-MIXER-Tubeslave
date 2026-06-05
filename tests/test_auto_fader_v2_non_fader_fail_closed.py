from pathlib import Path
from types import SimpleNamespace

from auto_fader_v2.controller import AutoFaderControllerV2
from auto_fader_v2.core.spectral_masking import EQAdjustment


class FakeMixer:
    is_connected = True

    def __init__(self):
        self.calls = []

    def set_gate_on(self, *args, **kwargs):
        self.calls.append(("set_gate_on", args, kwargs))

    def set_channel_fader(self, *args, **kwargs):
        self.calls.append(("set_channel_fader", args, kwargs))

    def set_eq_band(self, *args, **kwargs):
        self.calls.append(("set_eq_band", args, kwargs))

    def get_channel_fader(self, channel_id):
        return 0.0


def _make_controller(settings=None):
    config = {
        "automation": {
            "auto_fader": {
                "adaptive_noise_gate_enabled": True,
                "noise_gate_hold_sec": 0.0,
                "noise_gate_threshold_lufs": -55.0,
                **(settings or {}),
            }
        }
    }
    return AutoFaderControllerV2(FakeMixer(), config)


def test_auto_fader_v2_adaptive_gate_is_fail_closed_evidence():
    controller = _make_controller()
    metric = SimpleNamespace(lufs_momentary=-80.0)

    result = controller._apply_adaptive_noise_gate(3, metric, now_ts=100.0)
    status = controller.get_status()

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["calls"] == []
    assert result["reason"] == "auto_fader_v2_adaptive_noise_gate_quarantined"
    assert result["write_policy"] == "proposal_only_quarantined"
    assert result["live_write_allowed"] is False
    assert result["can_apply_live"] is False
    assert result["mutates_mixer"] is False
    assert controller.mixer_client.calls == []
    assert controller._gate_closed_channels[3] is True
    assert status["last_non_fader_write_attempted"] is True
    assert status["last_non_fader_write_blocked"] is True
    assert status["last_non_fader_write_count"] == 0
    assert status["last_blocked_non_fader_call_count"] == result["blocked_call_count"]
    assert {
        "cmd": "set_gate_on",
        "channel": 3,
        "on": 1,
        "semantic": "adaptive_noise_gate_close",
        "level_lufs": -80.0,
        "threshold_lufs": -55.0,
        "inactive_for_sec": 0.0,
    } in result["blocked_calls"]
    assert any(call["cmd"] == "set_channel_fader" for call in result["blocked_calls"])


def test_auto_fader_v2_adaptive_gate_reopen_is_fail_closed_evidence():
    controller = _make_controller()
    controller._gate_closed_channels[3] = True
    metric = SimpleNamespace(lufs_momentary=-20.0)

    result = controller._apply_adaptive_noise_gate(3, metric, now_ts=101.0)

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["reason"] == "auto_fader_v2_adaptive_noise_gate_quarantined"
    assert controller.mixer_client.calls == []
    assert controller._gate_closed_channels[3] is False
    assert result["blocked_calls"] == [
        {
            "cmd": "set_gate_on",
            "channel": 3,
            "on": 0,
            "semantic": "adaptive_noise_gate_open",
            "level_lufs": -20.0,
            "threshold_lufs": -55.0,
        }
    ]


def test_auto_fader_v2_masking_eq_is_fail_closed_evidence():
    controller = _make_controller()

    result = controller._block_eq_adjustments(
        [EQAdjustment(channel_id=4, frequency_hz=2500.0, gain_db=-3.5, q=3.0)],
        reason="auto_fader_v2_spectral_masking_eq_quarantined",
        semantic="spectral_masking_eq_cut",
    )

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["calls"] == []
    assert result["reason"] == "auto_fader_v2_spectral_masking_eq_quarantined"
    assert result["write_policy"] == "proposal_only_quarantined"
    assert result["live_write_allowed"] is False
    assert result["can_apply_live"] is False
    assert result["mutates_mixer"] is False
    assert controller.mixer_client.calls == []
    assert result["blocked_calls"] == [
        {
            "cmd": "set_eq_band",
            "channel": 4,
            "band": 3,
            "freq": 2500.0,
            "gain": -3.5,
            "q": 3.0,
            "semantic": "spectral_masking_eq_cut",
        }
    ]


def test_auto_fader_v2_source_has_no_direct_gate_or_eq_mixer_writes():
    source = Path("backend/auto_fader_v2/controller.py").read_text()

    assert "mixer_client.set_gate_on(" not in source
    assert "mixer_client.set_eq_band(" not in source
    assert "mixer_client.set_channel_fader(channel_id, -120.0)" not in source
    assert '"cmd": "set_gate_on"' in source
    assert '"cmd": "set_eq_band"' in source
