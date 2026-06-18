from backend.operator_mode_policy import (
    FULL_AUTONOMOUS_TARGET_SURFACES,
    build_operator_mode_status,
    get_operator_mode_options,
    normalize_manual_live_write_kinds,
    normalize_operator_mode,
)


def test_operator_mode_options_expose_project_four_mode_contract():
    options = get_operator_mode_options()

    assert [option["key"] for option in options] == [
        "analyze",
        "assist",
        "supervised",
        "auto",
    ]
    assert [option["project_mode_name"] for option in options] == [
        "Analysis Mode",
        "Assisted Mixing Mode",
        "Supervised AutoMix Mode",
        "Full Autonomous Mixing Mode",
    ]
    assert [option["autonomy_tier"] for option in options] == ["1", "2", "3", "4"]


def test_goal_language_aliases_normalize_to_stable_backend_modes():
    assert normalize_operator_mode("Analysis Mode") == "analyze"
    assert normalize_operator_mode("Assisted Mixing Mode") == "assist"
    assert normalize_operator_mode("Supervised AutoMix Mode") == "supervised"
    assert normalize_operator_mode("Full Autonomous Mixing Mode") == "auto"
    assert normalize_operator_mode("autonomous") == "auto"
    assert normalize_operator_mode("analysis   mode") == "analyze"
    assert normalize_operator_mode("analysis-mode-mode") == "analyze"
    assert normalize_operator_mode("analysis_mode_mode") == "analyze"
    assert normalize_operator_mode("supervised auto mix mode") == "supervised"
    assert normalize_operator_mode("supervised_auto_mode") == "supervised"
    assert normalize_operator_mode("full autonomous mode") == "auto"
    assert normalize_operator_mode("full autonomous mix") == "auto"
    assert normalize_operator_mode("full autonomous mix mode") == "auto"
    assert normalize_operator_mode("full_autonomous_mixing") == "auto"
    assert normalize_operator_mode("full_autonomous_mix") == "auto"
    assert normalize_operator_mode("full_autonomous_mix_mode") == "auto"
    assert normalize_operator_mode("full_autonomous_mode") == "auto"


def test_goal_language_aliases_handle_robust_separator_variants():
    for value in (
        "full-autonomous-mix",
        "full -- autonomous -- mix",
        "full   autonomous    mix",
        "full_autonomous_mode--",
        "analysis----mode",
        "supervised-auto_mix",
        "supervised   automix  mode",
    ):
        assert normalize_operator_mode(value) == "auto" if "full" in value else (
            "supervised" if "super" in value else "analyze"
        )


def test_goal_language_aliases_handle_russian_variants():
    assert normalize_operator_mode("анализ") == "analyze"
    assert normalize_operator_mode("режим_анализа") == "analyze"
    assert normalize_operator_mode("ассист") == "assist"
    assert normalize_operator_mode("assistant") == "assist"
    assert normalize_operator_mode("супервизор") == "supervised"
    assert normalize_operator_mode("режим_супервизии") == "supervised"
    assert normalize_operator_mode("полный_автономный_режим") == "auto"
    assert normalize_operator_mode("автономный_режим") == "auto"
    assert normalize_operator_mode("full_autonomous") == "auto"


def test_goal_language_aliases_fallback_to_assist_for_unknown_inputs():
    assert normalize_operator_mode(None) == "assist"
    assert normalize_operator_mode("") == "assist"
    assert normalize_operator_mode("   ") == "assist"
    assert normalize_operator_mode("random mode name") == "assist"
    assert normalize_operator_mode(123) == "assist"


def test_analysis_and_assist_modes_never_apply_to_console():
    for mode in ("analysis_mode", "assisted_mixing_mode"):
        status = build_operator_mode_status(mode=mode, connection_mode="wing")

        assert status["capabilities"]["can_analyze"] is True
        assert status["capabilities"]["can_apply_to_console"] is False
        assert status["capabilities"]["can_autonomous_apply"] is False
        assert status["capabilities"]["allowed_live_write_kinds"] == []
        assert status["capabilities"]["allowed_live_write_surfaces"] == []
        assert status["project_mode"]["current_live_write_kinds"] == []
        assert status["project_mode"]["effective_live_write_surfaces"] == []


def test_supervised_mode_maps_to_target_architecture_but_current_scope_stays_proven():
    status = build_operator_mode_status(mode="supervised_automix_mode")

    assert status["mode"] == "supervised"
    assert status["project_mode"]["name"] == "Supervised AutoMix Mode"
    assert status["project_mode"]["target_live_write_surfaces"] == list(
        FULL_AUTONOMOUS_TARGET_SURFACES
    )
    assert status["capabilities"]["requires_approval"] is True
    assert status["capabilities"]["can_autonomous_apply"] is False
    assert status["capabilities"]["allowed_live_write_kinds"] == ["fader", "gain", "pan"]
    assert status["capabilities"]["allowed_live_write_surfaces"] == [
        "preamp",
        "trim",
        "fader",
        "pan",
    ]
    assert status["project_mode"]["effective_live_write_surfaces"] == [
        "preamp",
        "trim",
        "fader",
        "pan",
    ]


def test_supervised_wing_deployment_scope_is_limited_to_fader_gain():
    status = build_operator_mode_status(
        mode="supervised_automix_mode",
        connection_mode="wing",
        wing_boundary_active=True,
    )

    assert status["mode"] == "supervised"
    assert status["project_mode"]["target_live_write_surfaces"] == list(
        FULL_AUTONOMOUS_TARGET_SURFACES
    )
    assert status["capabilities"]["can_apply_to_console"] is True
    assert status["capabilities"]["requires_approval"] is True
    assert status["capabilities"]["allowed_live_write_kinds"] == ["fader", "gain"]
    assert status["capabilities"]["allowed_live_write_surfaces"] == [
        "preamp",
        "trim",
        "fader",
    ]
    assert status["project_mode"]["current_live_write_kinds"] == ["fader", "gain"]
    assert status["project_mode"]["effective_live_write_surfaces"] == [
        "preamp",
        "trim",
        "fader",
    ]


def test_full_autonomous_mode_is_locked_on_wing_even_if_runtime_unlock_is_set():
    status = build_operator_mode_status(
        mode="full_autonomous_mixing_mode",
        connection_mode="wing",
        wing_boundary_active=True,
        auto_mode_unlocked=True,
    )

    assert status["mode"] == "auto"
    assert status["project_mode"]["name"] == "Full Autonomous Mixing Mode"
    assert status["live_write_policy"] == "auto_locked_on_wing_deployment"
    assert status["capabilities"]["can_apply_to_console"] is False
    assert status["capabilities"]["can_autonomous_apply"] is False
    assert status["capabilities"]["allowed_live_write_kinds"] == []
    assert "auto_mode_not_approved_for_wing_deployment" in status["blocked_reasons"]


def test_unlocked_non_wing_auto_still_exposes_only_current_proven_write_scope():
    status = build_operator_mode_status(
        mode="auto",
        connection_mode="offline_fixture",
        wing_boundary_active=False,
        auto_mode_unlocked=True,
    )

    assert status["capabilities"]["can_apply_to_console"] is True
    assert status["capabilities"]["can_autonomous_apply"] is True
    assert status["capabilities"]["requires_approval"] is False
    assert status["capabilities"]["allowed_live_write_kinds"] == ["fader", "gain", "pan"]
    assert status["capabilities"]["allowed_live_write_surfaces"] == [
        "preamp",
        "trim",
        "fader",
        "pan",
    ]
    assert status["project_mode"]["target_live_write_surfaces"] == list(
        FULL_AUTONOMOUS_TARGET_SURFACES
    )
    assert status["project_mode"]["current_live_write_kinds"] == ["fader", "gain", "pan"]
    assert status["project_mode"]["effective_live_write_surfaces"] == [
        "preamp",
        "trim",
        "fader",
        "pan",
    ]
    assert "autonomous_scope_limited_to_current_proven_write_kinds" in status["warnings"]


def test_normalize_manual_live_write_kinds_filters_unsupported_and_aliases_values():
    assert normalize_manual_live_write_kinds(
        [
            "fader",
            "gain",
            "eq",
            "trim",
            "input_gain",
            "preamp_gain",
            "fader",
            "mute",
            "pan_percent",
            "pan",
        ]
    ) == ["fader", "gain", "mute", "pan"]
