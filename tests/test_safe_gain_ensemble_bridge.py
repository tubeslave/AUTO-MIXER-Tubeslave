from operator_recommendation_bridge import build_safe_gain_operator_proposals


class _MixerWithCachedTrim:
    def get_channel_gain(self, channel):
        return 0.0


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
    evidence = {
        item["label"]: item
        for item in proposal["requested_change"]["safety_evidence"]
    }

    assert evidence["Ensemble context"]["value"] == "active"
    assert evidence["Ensemble apply policy"]["value"] == "operator_review_only"
    assert "Ensemble time coherence" not in evidence
    assert "Ensemble time sources" not in evidence
    assert "Ensemble lag samples" not in evidence
    assert "Ensemble context active (ensemble_context_vocal_understaged)" in proposal["reason"]
    assert "time missing" not in proposal["reason"]
    assert "sources 3/2" not in proposal["reason"]
    assert "lag 0/2 samples" not in proposal["reason"]
