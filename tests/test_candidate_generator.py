from ai_mixing_pipeline.decision_layer.candidate_generator import CandidateGenerator


def test_candidate_generator_creates_basic_candidates_and_no_change():
    generator = CandidateGenerator({"safety": {"max_gain_change_db_per_step": 1.0}})
    candidates = generator.generate(
        {
            "01_kick": "kick",
            "02_bass": "bass",
            "03_vocal": "vocal",
            "04_guitar": "guitars",
        },
        {"muddiness_proxy": 0.25, "harshness_proxy": 0.2},
        max_candidates=10,
    )

    ids = [candidate.candidate_id for candidate in candidates]

    assert ids[0] == "candidate_000_no_change"
    assert "candidate_001_vocal_up_0_5db" in ids
    assert "candidate_003_low_mid_cleanup" in ids
    assert "candidate_005_bass_kick_balance" in ids
    assert "candidate_009_gain_balance_polish" in ids
