from ai_mixing_pipeline.decision_layer.optuna_optimizer import OptunaActionOptimizer


def test_optuna_optimizer_fallback_does_not_crash_when_missing(tmp_path):
    optimizer = OptunaActionOptimizer({}, {"channels": {"vocal": "vocal"}}, random_seed=1)

    candidates = optimizer.ask_candidates(2)
    history_path = optimizer.save_history(tmp_path / "optimizer_history.json")

    assert history_path.exists()
    if optimizer.status.available:
        assert len(candidates) == 2
        assert candidates[0].source == "optuna"
    else:
        assert candidates == []
        assert optimizer.status.warnings
