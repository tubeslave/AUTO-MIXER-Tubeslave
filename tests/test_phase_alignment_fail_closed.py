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


def _assert_top_level_denial(result):
    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["calls"] == []
    _assert_operator_denial(result)


def _assert_operator_denial(result):
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
    _assert_top_level_denial(result)
    assert result["mixer_client_present"] is True
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
    summary = result["phase_alignment_safety_summary"]
    assert summary["status"] == "readback_required_before_live_promotion"
    assert summary["correlation_delta"] == 0.82
    assert summary["gcc_phat_confidence"] == 0.82
    assert summary["polarity_readback"] == (
        "cached_or_physical_delay_polarity_read_required_before_promotion"
    )
    assert summary["delay_readback"] == (
        "cached_or_physical_delay_polarity_read_required_before_promotion"
    )
    assert summary["read_only_or_dry_run_only"] is True
    assert summary["live_apply_allowed"] is False
    assert summary["queue_import_allowed"] is False
    _assert_operator_denial(summary)
    correction = result["operator_correction"]
    assert correction["objective_family"] == "phase_alignment"
    assert correction["future_wing_readback_contract"]["target_kind"] == (
        "channel_delay_or_polarity"
    )
    assert correction["bounded_delta"]["action_type"] == (
        "phase_delay_direct_apply_quarantined"
    )
    assert correction["bounded_delta"]["target"] == 2
    assert correction["bounded_delta"]["parameter_deltas"]["delay_ms"] == 3.25
    assert correction["bounded_delta"]["parameter_deltas"]["measured_delay_ms"] == 3.25
    assert correction["bounded_delta"]["parameter_deltas"]["phase_invert"] == 1
    assert correction["bounded_delta"]["parameter_deltas"]["correlation_delta"] == 0.82
    assert correction["bounded_delta"]["parameter_deltas"]["gcc_phat_confidence"] == 0.82
    assert correction["read_only_or_dry_run_only"] is True
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    assert "phase_correlation" in correction["headroom_feedback_true_peak_checks"][
        "required_checks"
    ]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "phase_correlation"
    ] == 0.82
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_operator_denial(source_evidence)
    assert source_evidence["phase_safety"] == summary
    assert source_evidence["current_phase_audio_context_cache_no_live_authority"] is True
    assert source_evidence["readback_required_before_promotion"] is True
    assert source_evidence["rollback_required_before_promotion"] is True
    assert "delay_polarity_readback_not_proven" in correction["safety_blockers"]


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
    _assert_top_level_denial(result)
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
    correction = result["operator_correction"]
    assert correction["objective_family"] == "phase_alignment"
    assert correction["bounded_delta"]["action_type"] == "phase_delay_reset_quarantined"
    assert correction["bounded_delta"]["parameter_deltas"]["delay_ms"] == 0.5
    assert correction["bounded_delta"]["parameter_deltas"]["phase_invert"] == 0
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_operator_denial(source_evidence)
    assert source_evidence["phase_safety"] == result["phase_alignment_safety_summary"]


def test_phase_alignment_has_no_direct_delay_or_polarity_mixer_writes():
    source = Path("backend/phase_alignment.py").read_text()

    assert "mixer_client.set_channel_delay(" not in source
    assert "mixer_client.set_channel_phase_invert(" not in source
    assert "mixer_client.send(" not in source
    assert '"cmd": "set_channel_delay"' in source
    assert '"cmd": "set_channel_phase_invert"' in source
