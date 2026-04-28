from ai_mixing_pipeline.decision_layer.nevergrad_optimizer import NevergradActionOptimizer


def test_nevergrad_optimizer_fallback_does_not_crash_when_missing():
    optimizer = NevergradActionOptimizer(
        {},
        {"channels": {"vocal": "vocal"}, "safety": {"max_gain_change_db_per_step": 1.0}},
        random_seed=1,
    )

    candidates = optimizer.ask_candidates(2)

    if optimizer.status.available:
        assert len(candidates) == 2
        assert candidates[0].source == "nevergrad"
    else:
        assert candidates == []
        assert optimizer.status.warnings
