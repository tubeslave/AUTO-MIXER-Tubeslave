from backend.operator_mode_policy import (
    FULL_AUTONOMOUS_TARGET_SURFACES,
    build_operator_mode_status,
    get_operator_mode_options,
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


def test_analysis_and_assist_modes_never_apply_to_console():
    for mode in ("analysis_mode", "assisted_mixing_mode"):
        status = build_operator_mode_status(mode=mode, connection_mode="wing")

        assert status["capabilities"]["can_analyze"] is True
        assert status["capabilities"]["can_apply_to_console"] is False
        assert status["capabilities"]["can_autonomous_apply"] is False
        assert status["capabilities"]["allowed_live_write_kinds"] == []
        assert status["project_mode"]["current_live_write_kinds"] == []


def test_supervised_mode_maps_to_target_architecture_but_current_scope_stays_proven():
    status = build_operator_mode_status(mode="supervised_automix_mode")

    assert status["mode"] == "supervised"
    assert status["project_mode"]["name"] == "Supervised AutoMix Mode"
    assert status["project_mode"]["target_live_write_surfaces"] == list(
        FULL_AUTONOMOUS_TARGET_SURFACES
    )
    assert status["capabilities"]["requires_approval"] is True
    assert status["capabilities"]["can_autonomous_apply"] is False
    assert status["capabilities"]["allowed_live_write_kinds"] == ["fader", "gain"]


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
    assert status["capabilities"]["allowed_live_write_kinds"] == ["fader", "gain"]
    assert status["project_mode"]["target_live_write_surfaces"] == list(
        FULL_AUTONOMOUS_TARGET_SURFACES
    )
    assert status["project_mode"]["current_live_write_kinds"] == ["fader", "gain"]
    assert "autonomous_scope_limited_to_current_proven_write_kinds" in status["warnings"]
