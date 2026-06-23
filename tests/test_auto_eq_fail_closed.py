from pathlib import Path

from auto_eq import (
    AutoEQController,
    EQ_READ_REQUIRED_SOURCE,
    EQBand,
    MultiChannelAutoEQController,
)


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


def _assert_top_level_denial(result):
    assert result["can_apply_live"] is False
    assert result["can_import_to_queue"] is False
    assert result["queue_import_allowed"] is False
    assert result["live_apply_allowed"] is False
    assert result["read_only_or_dry_run_only"] is True
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False
    assert result["queue_action"] == "not_queueable_until_promoted"
    assert result["live_write"] is False
    assert result["queue_mutation"] is False
    assert result["mutates_mixer"] is False


def _assert_complete_evidence_gate(correction):
    assert correction["source_evidence"]["evidence_completeness"]["status"] == (
        "complete"
    )
    assert correction["source_evidence"]["evidence_completeness"]["missing"] == []
    assert correction["evidence_gate"]["status"] == (
        "required_evidence_complete_operator_review_still_required"
    )


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
    _assert_top_level_denial(result)
    assert result["mixer_client_present"] is True
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
    correction = result["operator_correction"]
    assert correction["objective_family"] == "eq"
    assert correction["future_wing_readback_contract"]["target_kind"] == "channel_eq_band"
    assert correction["bounded_delta"]["action_type"] == "auto_eq_apply_blocked"
    assert correction["bounded_delta"]["delta_db"] == -3.0
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    _assert_complete_evidence_gate(correction)
    assert "hardware_readback_and_rollback_not_proven" in correction["safety_blockers"]
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_top_level_denial(source_evidence)
    assert source_evidence["old_value_source"] == EQ_READ_REQUIRED_SOURCE
    assert source_evidence["rollback_value_source"] == EQ_READ_REQUIRED_SOURCE
    assert source_evidence["frequency_match"]["status"] == (
        "readback_required_before_live_promotion"
    )
    assert source_evidence["cut_or_bounded_boost"]["bounded_preview_only"] is True
    assert source_evidence["feedback_notch_preserved"] == (
        "readback_required_before_live_promotion"
    )
    assert source_evidence["manual_band_preserved"] == (
        "readback_required_before_live_promotion"
    )
    assert source_evidence["true_peak_headroom"] == "required_before_live_promotion"
    assert source_evidence["eq_band_readback"] == {
        "status": "required_before_live_promotion",
        "readback_targets": [
            "eq_on",
            "band_id",
            "band_frequency",
            "band_gain",
            "band_q",
            "filter_type",
        ],
        "blocked_call_count": len(result["blocked_calls"]),
        "old_value_source": EQ_READ_REQUIRED_SOURCE,
        "read_only_or_dry_run_only": True,
        "live_write": False,
    }
    assert source_evidence["current_eq_context_cache_freshness"] == {
        "status": "readback_required_before_live_promotion",
        "source": EQ_READ_REQUIRED_SOURCE,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    assert source_evidence["current_eq_context_cache_no_live_authority"] is True
    assert source_evidence["rollback_eq_band_readback"] == {
        "status": "required_before_live_promotion",
        "rollback_targets": [
            "eq_on",
            "band_id",
            "band_frequency",
            "band_gain",
            "band_q",
            "filter_type",
        ],
        "old_value_source": EQ_READ_REQUIRED_SOURCE,
        "blocked_call_count": len(result["blocked_calls"]),
        "read_only_or_dry_run_only": True,
        "live_write": False,
    }
    evidence_checks = correction["headroom_feedback_true_peak_checks"][
        "evidence_checks"
    ]
    for key in (
        "frequency_match",
        "cut_or_bounded_boost",
        "feedback_notch_preserved",
        "manual_band_preserved",
        "true_peak_headroom",
        "eq_band_readback",
        "current_eq_context_cache_freshness",
        "current_eq_context_cache_no_live_authority",
        "rollback_eq_band_readback",
    ):
        assert evidence_checks[key] == source_evidence[key]


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
    _assert_top_level_denial(result)
    assert result["blocked_call_count"] == 7
    assert mixer.calls == []
    assert controller.current_corrections
    correction = result["operator_correction"]
    assert correction["objective_family"] == "eq"
    assert correction["bounded_delta"]["action_type"] == "auto_eq_reset_blocked"
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    _assert_complete_evidence_gate(correction)
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_top_level_denial(source_evidence)
    assert source_evidence["blocked_call_count"] == result["blocked_call_count"]
    assert source_evidence["feedback_notch_preserved"] == (
        "readback_required_before_live_promotion"
    )
    assert source_evidence["manual_band_preserved"] == (
        "readback_required_before_live_promotion"
    )
    assert source_evidence["current_eq_context_cache_no_live_authority"] is True
    assert source_evidence["rollback_eq_band_readback"]["status"] == (
        "required_before_live_promotion"
    )
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
    _assert_top_level_denial(result)
    assert mixer.calls == []
    assert status["last_write_attempted"] is True
    assert status["last_blocked_call_count"] == len(result["blocked_calls"])
    correction = result["operator_correction"]
    assert correction["objective_family"] == "eq"
    assert correction["bounded_delta"]["action_type"] == "auto_eq_apply_blocked"
    assert correction["bounded_delta"]["target"] == 7
    assert correction["live_apply_allowed"] is False
    _assert_complete_evidence_gate(correction)
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_top_level_denial(source_evidence)
    assert source_evidence["blocked_call_count"] == result["blocked_call_count"]
    assert source_evidence["frequency_match"]["eq_command_count"] == len(
        result["blocked_calls"]
    )
    assert source_evidence["current_eq_context_cache_no_live_authority"] is True
    assert source_evidence["rollback_eq_band_readback"]["blocked_call_count"] == len(
        result["blocked_calls"]
    )


def test_auto_eq_module_has_no_direct_mixer_eq_writes():
    source = Path("backend/auto_eq.py").read_text()

    assert "mixer_client.set_eq_on(" not in source
    assert "mixer_client.set_eq_low_shelf(" not in source
    assert "mixer_client.set_eq_high_shelf(" not in source
    assert "mixer_client.set_eq_band(" not in source
    assert "mixer_client.send(" not in source
    assert '"cmd": "set_eq_on"' in source
    assert '"cmd": "set_eq_band"' in source
