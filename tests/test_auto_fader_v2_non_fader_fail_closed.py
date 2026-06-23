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

    def set_channel_pan(self, *args, **kwargs):
        self.calls.append(("set_channel_pan", args, kwargs))

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


def _assert_non_fader_denial(result):
    assert result["write_policy"] == "proposal_only_quarantined"
    assert result["live_write"] is False
    assert result["queue_mutation"] is False
    assert result["live_write_allowed"] is False
    assert result["can_apply_live"] is False
    assert result["can_import_to_queue"] is False
    assert result["queue_import_allowed"] is False
    assert result["live_apply_allowed"] is False
    assert result["read_only_or_dry_run_only"] is True
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False
    assert result["queue_action"] == "not_queueable_until_promoted"
    assert result["mutates_mixer"] is False
    assert result["safety_blockers"] == [
        "auto_fader_v2_non_fader_direct_apply_quarantined",
        "real_wing_proof_required_before_live_apply",
        "non_fader_live_expansion_requires_separate_adr",
        "operator_approval_required_before_supervised_non_fader_apply",
        "hardware_readback_and_rollback_not_proven",
    ]
    assert "Review it offline" in result["operator_explanation"]
    assert result["required_safety_checks"] == {
        "source_identity": "required_before_live_promotion",
        "true_peak_headroom": "required_before_live_promotion",
        "feedback_risk": "required_before_live_promotion",
        "mono_phase_safety": "required_for_pan_separation_before_promotion",
        "manual_band_protection": "required_for_eq_before_promotion",
        "chatter_guard": "required_for_gate_before_promotion",
    }
    assert result["rollback_expectation"] == {
        "required_before_live_promotion": True,
        "restore_previous_console_value_after_physical_readback": True,
        "manual_operator_rollback_required_if_promoted": True,
    }
    assert result["future_wing_readback_contract"] == {
        "required": True,
        "target_kind": "auto_fader_v2_non_fader_control",
        "readback_targets": [
            "gate_state",
            "eq_band_state",
            "pan_state",
            "fader_fallback_state",
        ],
        "blocked_live_actions": [
            "set_gate_on",
            "set_eq_band",
            "set_channel_pan",
            "set_channel_fader",
        ],
        "requires_real_wing_proof": True,
        "requires_operator_approval": True,
        "requires_separate_non_fader_adr": True,
    }


def test_auto_fader_v2_adaptive_gate_is_fail_closed_evidence():
    controller = _make_controller()
    metric = SimpleNamespace(lufs_momentary=-80.0)

    result = controller._apply_adaptive_noise_gate(3, metric, now_ts=100.0)
    status = controller.get_status()

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["calls"] == []
    assert result["reason"] == "auto_fader_v2_adaptive_noise_gate_quarantined"
    _assert_non_fader_denial(result)
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
    _assert_non_fader_denial(result)
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
    _assert_non_fader_denial(result)
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


def test_auto_fader_v2_non_fader_noop_still_denies_live_and_queue_authority():
    controller = _make_controller()
    metric = SimpleNamespace(lufs_momentary=-20.0)

    result = controller._apply_adaptive_noise_gate(3, metric, now_ts=100.0)

    assert result["applied"] is False
    assert result["blocked"] is False
    assert result["calls"] == []
    assert result["blocked_calls"] == []
    assert result["reason"] == "adaptive_noise_gate_no_state_change"
    _assert_non_fader_denial(result)
    assert controller.mixer_client.calls == []


def test_auto_fader_v2_pan_separation_noop_still_denies_live_and_queue_authority():
    controller = _make_controller({"pan_separation_enabled": True})

    result = controller._apply_pan_separation(
        [1],
        [2],
        {1: SimpleNamespace(lufs_momentary=-22.0)},
    )

    assert result["applied"] is False
    assert result["blocked"] is False
    assert result["calls"] == []
    assert result["blocked_calls"] == []
    assert result["reason"] == "stereo_metrics_unavailable"
    _assert_non_fader_denial(result)
    assert controller.mixer_client.calls == []


def test_auto_fader_v2_pan_separation_is_blocked_proposal_evidence():
    controller = _make_controller({"pan_separation_enabled": True})
    stereo_metric = SimpleNamespace(left_rms=0.2, right_rms=0.18)

    result = controller._apply_pan_separation(
        [1],
        [2, 3],
        {1: stereo_metric, 2: stereo_metric, 3: stereo_metric},
    )

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["calls"] == []
    assert result["reason"] == "pan_separation_direct_apply_quarantined"
    _assert_non_fader_denial(result)
    assert result["blocked_call_count"] == 3
    assert result["blocked_calls"] == [
        {
            "cmd": "set_channel_pan",
            "channel": 1,
            "pan_percent": 0.0,
            "reason": "center lead vocal for pan separation",
        },
        {
            "cmd": "set_channel_pan",
            "channel": 2,
            "pan_percent": -20.0,
            "reason": "background pan separation",
        },
        {
            "cmd": "set_channel_pan",
            "channel": 3,
            "pan_percent": 20.0,
            "reason": "background pan separation",
        },
    ]
    assert controller.mixer_client.calls == []
    assert controller.last_non_fader_write_attempted is True
    assert controller.last_non_fader_write_blocked is True
    assert controller.last_non_fader_write_count == 0
    assert controller.last_blocked_non_fader_calls == result["blocked_calls"]
