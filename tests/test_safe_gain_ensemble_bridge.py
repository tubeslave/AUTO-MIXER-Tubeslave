import time

from operator_recommendation_bridge import (
    _cache_freshness_for_change,
    build_safe_gain_operator_proposals,
    build_soundcheck_operator_proposals,
)
from soundcheck_recommendation_workflow import NoWriteSoundcheckRecommendationWorkflow


class _MixerWithCachedTrim:
    def get_channel_gain(self, channel):
        return 0.0


class _MixerWithStaleCachedTrim:
    state = {"/ch/1/in/set/trim": 0.0}
    state_timestamps = {"/ch/1/in/set/trim": time.time() - 120.0}


class _MixerWithCachedPan:
    state = {"/ch/3/pan": 0.0}
    state_timestamps = {"/ch/3/pan": time.time()}


def _assert_operator_denial(payload):
    assert payload["can_apply_live"] is False
    assert payload["can_import_to_queue"] is False
    assert payload["queue_import_allowed"] is False
    assert payload["live_apply_allowed"] is False
    assert payload["read_only_or_dry_run_only"] is True
    assert payload["live_authority_granted"] is False
    assert payload["readiness_closure_allowed"] is False
    assert payload["queue_action"] == "not_queueable_until_promoted"
    assert payload["live_write"] is False
    assert payload["queue_mutation"] is False
    assert payload["mutates_mixer"] is False


def test_recommendation_bridge_treats_pan_as_outside_current_wing_apply_scope():
    freshness = _cache_freshness_for_change(
        _MixerWithCachedPan(),
        {"value_type": "pan", "channel": 3, "value": 0.0},
    )

    assert freshness["status"] == "unsupported"
    assert freshness["value_type"] == "pan"
    assert freshness["read_only"] is True
    assert freshness["live_write"] is False
    assert freshness["queries_console"] is False


def test_safe_gain_proposal_omits_time_evidence_when_ensemble_time_coherence_disabled():
    suggestion = {
        "channel": 1,
        "method_id": "input_trim_preamp_character",
        "input_trim_role": "input_headroom_not_mix_balance",
        "peak_db": -21.0,
        "target_peak_db": -10.0,
        "suggested_gain_db": 6.0,
        "target_trim_db": 6.0,
        "confidence": 0.82,
        "input_gain_character": {
            "character": "clean",
            "current_zone": "clean",
            "target_zone": "clean",
            "distortion_risk": "low",
            "soundcheck_ensemble_context": {
                "provided": True,
                "active": True,
                "reason": "ensemble_context_vocal_understaged",
                "time_coherence_enabled": False,
                "time_coherence_evaluable": False,
                "time_coherent": True,
                "time_coherent_loud_sources": 3,
                "required_time_coherent_sources": 2,
                "reference_max_lag_samples": 0,
                "max_context_lag_samples": 2,
                "operator_review_only": True,
            },
        },
    }

    proposals = build_safe_gain_operator_proposals(
        {1: suggestion},
        channel_mapping={1: 1},
        mixer_client=_MixerWithCachedTrim(),
    )

    proposal = proposals[0]
    correction = proposal["operator_correction"]
    _assert_operator_denial(proposal)
    evidence = {
        item["label"]: item
        for item in proposal["requested_change"]["safety_evidence"]
    }

    assert correction["objective_family"] == "gain_trim"
    assert correction["future_wing_readback_contract"]["target_kind"] == (
        "channel_input_trim_or_preamp_gain"
    )
    assert correction["bounded_delta"]["delta_db"] == 6.0
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    assert "real_wing_proof_required_before_live_apply" in correction["safety_blockers"]
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_operator_denial(source_evidence)
    assert source_evidence["input_level_evidence"]["peak_db"] == -21.0
    assert source_evidence["input_level_evidence"]["target_peak_db"] == -10.0
    assert source_evidence["bounded_gain_delta"]["delta_db"] == 6.0
    assert source_evidence["gain_down_or_operator_review_gain_up"] == (
        "gain_up_requires_operator_review"
    )
    assert source_evidence["rollback_input_trim_readback"]["value_db"] == 0.0
    assert evidence["Ensemble context"]["value"] == "active"
    assert evidence["Ensemble apply policy"]["value"] == "operator_review_only"
    assert "Ensemble time coherence" not in evidence
    assert "Ensemble time sources" not in evidence
    assert "Ensemble lag samples" not in evidence
    assert "Ensemble context active (ensemble_context_vocal_understaged)" in proposal["reason"]
    assert "time missing" not in proposal["reason"]
    assert "sources 3/2" not in proposal["reason"]
    assert "lag 0/2 samples" not in proposal["reason"]


def test_safe_gain_operator_correction_reflects_stale_cache_blocker():
    suggestion = {
        "channel": 1,
        "suggested_gain_db": -2.0,
        "target_trim_db": -2.0,
        "confidence": 0.9,
        "peak_db": -0.8,
    }

    proposal = build_safe_gain_operator_proposals(
        {1: suggestion},
        channel_mapping={1: 1},
        mixer_client=_MixerWithStaleCachedTrim(),
    )[0]

    correction = proposal["operator_correction"]
    _assert_operator_denial(proposal)
    assert proposal["kind"] == "analysis"
    assert proposal["requested_change"] is None
    assert correction["objective_family"] == "gain_trim"
    assert correction["proposed_correction"]["status"] == "blocked_before_applyable_proposal"
    assert correction["proposed_correction"]["blocked_requested_change"]["delta_db"] == -2.0
    assert "cache_freshness_stale:/ch/1/in/set/trim" in correction["safety_blockers"]
    assert "cache_freshness_required_before_applyable_proposal" in correction[
        "safety_blockers"
    ]
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_operator_denial(source_evidence)
    assert source_evidence["blocked_requested_change"]["delta_db"] == -2.0
    assert source_evidence["apply_status"] == "blocked_cache_freshness_not_fresh"
    assert source_evidence["cache_freshness"][0]["status"] == "stale"
    assert source_evidence["bounded_gain_delta"]["delta_db"] == -2.0


def test_soundcheck_gain_and_fader_proposals_include_operator_correction_payloads():
    workflow = NoWriteSoundcheckRecommendationWorkflow()
    workflow.recommend_input_gain(channel=1, current_trim_db=0.0, target_trim_db=-3.0)
    workflow.recommend_fader(channel=2, current_fader_db=-8.0, target_fader_db=-9.5)

    proposals = build_soundcheck_operator_proposals(workflow.build_bundle())

    gain_correction = proposals[0]["operator_correction"]
    fader_correction = proposals[1]["operator_correction"]
    for proposal in proposals:
        _assert_operator_denial(proposal)
    assert gain_correction["objective_family"] == "gain_trim_and_fader_balance"
    assert gain_correction["future_wing_readback_contract"]["target_kind"] == (
        "bounded_soundcheck_gain_or_fader_preview"
    )
    assert gain_correction["bounded_delta"]["delta_db"] == -3.0
    assert proposals[0]["requested_change"]["rollback_value"] == 0.0
    assert fader_correction["objective_family"] == "gain_trim_and_fader_balance"
    assert fader_correction["future_wing_readback_contract"]["target_kind"] == (
        "bounded_soundcheck_gain_or_fader_preview"
    )
    assert fader_correction["bounded_delta"]["delta_db"] == -1.5
    assert proposals[1]["requested_change"]["rollback_value"] == -8.0
    assert gain_correction["live_apply_allowed"] is False
    assert fader_correction["live_apply_allowed"] is False
    gain_source_evidence = gain_correction["source_evidence"]["evidence"]
    fader_source_evidence = fader_correction["source_evidence"]["evidence"]
    _assert_operator_denial(gain_source_evidence)
    _assert_operator_denial(fader_source_evidence)
    assert gain_source_evidence["bounded_gain_delta"]["delta_db"] == -3.0
    assert gain_source_evidence["rollback_input_trim_readback"]["value_db"] == 0.0
    assert fader_source_evidence["bounded_fader_delta"]["delta_db"] == -1.5
    assert fader_source_evidence["rollback_fader_readback"]["value_db"] == -8.0
