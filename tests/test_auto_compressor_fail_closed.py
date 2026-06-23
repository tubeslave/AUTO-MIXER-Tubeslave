from pathlib import Path

from auto_compressor import AutoCompressorController, DYNAMICS_READ_REQUIRED_SOURCE
from auto_compressor_cf import AutoCFCompressorController, CFClass, CFCompressorParams


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


class FakeCompressorMixer:
    is_connected = True

    def __init__(self):
        self.calls = []

    def get_compressor_on(self, channel):
        self.calls.append(("get_compressor_on", channel))
        return 0

    def set_compressor_on(self, *args, **kwargs):
        self.calls.append(("set_compressor_on", args, kwargs))

    def set_compressor(self, *args, **kwargs):
        self.calls.append(("set_compressor", args, kwargs))


def test_auto_compressor_smooth_apply_is_fail_closed_evidence():
    mixer = FakeCompressorMixer()
    controller = AutoCompressorController(mixer_client=mixer)

    result = controller._apply_compressor_smooth(
        3,
        {
            "threshold": -18.5,
            "ratio_wing": "3.0",
            "attack_ms": 12.0,
            "release_ms": 180.0,
            "knee": 3.0,
            "gain": 1.5,
        },
    )

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["calls"] == []
    assert result["reason"] == "auto_compressor_direct_apply_quarantined"
    assert result["write_policy"] == "proposal_only_quarantined"
    assert result["live_write_allowed"] is False
    _assert_top_level_denial(result)
    assert result["mixer_client_present"] is True
    correction = result["operator_correction"]
    assert correction["objective_family"] == "compressor_dynamics"
    assert correction["future_wing_readback_contract"]["target_kind"] == "channel_dynamics"
    assert correction["bounded_delta"]["action_type"] == "auto_compressor_apply_blocked"
    assert correction["bounded_delta"]["delta_db"] == 1.5
    assert correction["bounded_delta"]["parameter_deltas"]["threshold_db"] == -18.5
    assert correction["bounded_delta"]["parameter_deltas"]["makeup_gain_db"] == 1.5
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    _assert_complete_evidence_gate(correction)
    assert "hardware_readback_and_rollback_not_proven" in correction["safety_blockers"]
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_top_level_denial(source_evidence)
    assert source_evidence["old_value_source"] == DYNAMICS_READ_REQUIRED_SOURCE
    assert source_evidence["rollback_value_source"] == DYNAMICS_READ_REQUIRED_SOURCE
    assert source_evidence["gain_reduction_estimate"]["status"] == (
        "required_before_live_promotion"
    )
    assert source_evidence["target_gain_reduction_db"] is None
    assert source_evidence["requested_makeup_gain_db"] == 1.5
    assert source_evidence["makeup_headroom"]["makeup_gain_db"] == 1.5
    assert source_evidence["true_peak_headroom"] == "required_before_live_promotion"
    assert source_evidence["current_dynamics_cache_freshness"] == {
        "status": "required_before_live_promotion",
        "source": DYNAMICS_READ_REQUIRED_SOURCE,
    }
    assert source_evidence["current_dynamics_cache_no_live_authority"] is True
    assert source_evidence["rollback_dynamics_readback"] == (
        DYNAMICS_READ_REQUIRED_SOURCE
    )
    assert source_evidence["bypass_state"] == "readback_required_before_live_promotion"
    assert mixer.calls == []
    assert controller.current_params[3]["threshold"] == -18.5
    assert {
        "cmd": "set_compressor_on",
        "channel": 3,
        "on": 1,
        "semantic": "enable_before_compressor_params",
    } in result["blocked_calls"]
    assert {
        "cmd": "set_compressor",
        "channel": 3,
        "threshold": -18.5,
        "ratio": "3.0",
        "attack": 12.0,
        "release": 180.0,
        "knee": 3,
        "gain": 1.5,
        "semantic": "apply_compressor_params",
    } in result["blocked_calls"]


def test_auto_compressor_on_toggle_is_fail_closed():
    mixer = FakeCompressorMixer()
    controller = AutoCompressorController(mixer_client=mixer)

    result = controller._block_compressor_on(4, 0, "soundcheck_clean_analysis_disable")

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["reason"] == "auto_compressor_direct_apply_quarantined"
    _assert_top_level_denial(result)
    assert mixer.calls == []
    assert result["blocked_calls"] == [
        {
            "cmd": "set_compressor_on",
            "channel": 4,
            "on": 0,
            "semantic": "soundcheck_clean_analysis_disable",
        }
    ]
    correction = result["operator_correction"]
    assert correction["objective_family"] == "compressor_dynamics"
    assert correction["bounded_delta"]["action_type"] == "auto_compressor_on_toggle_blocked"
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    _assert_complete_evidence_gate(correction)
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_top_level_denial(source_evidence)
    assert source_evidence["blocked_call_count"] == result["blocked_call_count"]
    assert source_evidence["current_dynamics_cache_no_live_authority"] is True
    assert source_evidence["rollback_dynamics_readback"] == (
        DYNAMICS_READ_REQUIRED_SOURCE
    )
    assert source_evidence["bypass_state"] == "readback_required_before_live_promotion"


def test_auto_cf_compressor_apply_params_is_fail_closed_evidence():
    mixer = FakeCompressorMixer()
    controller = AutoCFCompressorController(mixer_client=mixer)
    params = CFCompressorParams(
        attack_ms=8.0,
        release_ms=110.0,
        ratio=4.0,
        threshold_db=-21.0,
        knee_db=2.0,
        makeup_gain_db=1.0,
        cf_class=CFClass.DRUMS,
        cf_db=14.0,
    )

    result = controller.apply_params(5, params)

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["calls"] == []
    assert result["reason"] == "auto_cf_compressor_direct_apply_quarantined"
    assert result["write_policy"] == "proposal_only_quarantined"
    _assert_top_level_denial(result)
    correction = result["operator_correction"]
    assert correction["objective_family"] == "compressor_dynamics"
    assert correction["future_wing_readback_contract"]["target_kind"] == "channel_dynamics"
    assert correction["bounded_delta"]["action_type"] == "auto_cf_compressor_apply_blocked"
    assert correction["bounded_delta"]["delta_db"] == 1.0
    assert correction["bounded_delta"]["parameter_deltas"]["threshold_db"] == -21.0
    assert correction["bounded_delta"]["parameter_deltas"]["ratio"] == 4.0
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    _assert_complete_evidence_gate(correction)
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_top_level_denial(source_evidence)
    summary = result["cf_compressor_safety_summary"]
    assert summary == source_evidence["cf_compressor_safety_summary"]
    assert summary["summary_type"] == "auto_cf_compressor_fail_closed_review"
    _assert_top_level_denial(summary)
    assert summary["channel"] == 5
    assert summary["blocked_call_count"] == 2
    assert summary["threshold_db"] == -21.0
    assert summary["ratio"] == 4.0
    assert summary["makeup_gain_db"] == 1.0
    assert summary["headroom_check"] == "required_before_apply"
    assert summary["feedback_safety"] == "required_before_apply"
    assert summary["true_peak_headroom"] == "required_before_apply"
    assert summary["gain_reduction_review"] == "required_before_apply"
    assert summary["makeup_gain_safety"] == "required_before_apply"
    assert source_evidence["crest_factor_class"] == "drums"
    assert source_evidence["lufs_lra_context"]["target_lufs"] == -18.0
    assert source_evidence["gain_reduction_estimate"]["cf_db"] == 14.0
    assert source_evidence["bounded_makeup_headroom"]["makeup_gain_db"] == 1.0
    assert source_evidence["bypass_state"] == "readback_required_before_live_promotion"
    assert summary["read_only_or_dry_run_only"] is True
    assert summary["live_apply_allowed"] is False
    assert summary["queue_import_allowed"] is False
    assert mixer.calls == []
    assert {
        "cmd": "set_compressor_on",
        "channel": 5,
        "on": 1,
        "semantic": "enable_before_cf_params",
    } in result["blocked_calls"]
    assert {
        "cmd": "set_compressor",
        "channel": 5,
        "threshold": -21.0,
        "ratio": "4.0",
        "attack": 8.0,
        "release": 110.0,
        "knee": 2,
        "gain": 1.0,
        "cf_db": 14.0,
        "cf_class": "drums",
        "semantic": "apply_cf_compressor_params",
    } in result["blocked_calls"]


def test_auto_compressor_modules_have_no_direct_mixer_compressor_writes():
    for source_path in ("backend/auto_compressor.py", "backend/auto_compressor_cf.py"):
        source = Path(source_path).read_text()

        assert "mixer_client.set_compressor(" not in source
        assert "mixer_client.set_compressor_on(" not in source
        assert '"cmd": "set_compressor"' in source
        assert '"cmd": "set_compressor_on"' in source
