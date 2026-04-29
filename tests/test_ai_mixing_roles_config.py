from ai_mixing_pipeline.config import enabled_critic_weights, load_roles_config


def test_ai_mixing_roles_config_loads_expected_roles():
    config = load_roles_config("configs/ai_mixing_roles.yaml")

    assert config["critics"]["muq_eval"]["role"] == "chief_music_critic"
    assert config["critics"]["audiobox_aesthetics"]["weight"] == 0.20
    assert config["critics"]["demucs_or_openunmix"]["realtime_allowed"] is False
    assert config["safety"]["max_true_peak_dbfs"] == -1.0
    assert config["offline_test"]["create_no_change_candidate"] is True
    assert config["offline_test"]["safe_render_peak_margin_db"] == 0.6


def test_enabled_critic_weights_include_safety_and_skip_separator():
    config = load_roles_config("configs/ai_mixing_roles.yaml")
    weights = enabled_critic_weights(config)

    assert "muq_eval" in weights
    assert "safety" in weights
    assert "demucs_or_openunmix" not in weights
    assert weights["muq_eval"] > weights["safety"]
