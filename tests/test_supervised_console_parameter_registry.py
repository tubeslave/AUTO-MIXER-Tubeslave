from supervised_console_parameter_registry import (
    FAMILY_CONTRACTS,
    LIFECYCLE_HARDWARE_PROVEN,
    LIFECYCLE_PROPOSAL_ONLY,
    family_id_for_write_kind,
    get_supervised_console_family,
    supervised_console_family_registry,
    validate_supervised_console_family_registry,
)


def test_registry_covers_all_requested_soundcheck_families_fail_closed():
    expected = {
        "fader",
        "gain_trim",
        "mute",
        "pan_phase_delay",
        "eq_hpf",
        "gate_expander",
        "dynamics_compressor",
        "fx_sends_inserts",
        "bus_group_dca_matrix",
        "master_output",
        "routing",
        "scene_preset",
    }

    assert {row.family_id for row in FAMILY_CONTRACTS} == expected
    assert validate_supervised_console_family_registry() == ()
    assert get_supervised_console_family("eq_hpf").live_allowed_write_kinds == ()
    assert get_supervised_console_family("routing").live_allowed_write_kinds == ()


def test_hardware_proof_does_not_grant_normal_supervised_live_authority():
    assert get_supervised_console_family("fader").lifecycle == LIFECYCLE_HARDWARE_PROVEN
    assert get_supervised_console_family("gain_trim").lifecycle == LIFECYCLE_HARDWARE_PROVEN
    assert get_supervised_console_family("eq_hpf").lifecycle == LIFECYCLE_HARDWARE_PROVEN
    assert not get_supervised_console_family("gate_expander").permits_execution(write_kind="gate", proof_run=False)
    assert not get_supervised_console_family("gate_expander").permits_execution(write_kind="expander", proof_run=False)
    assert get_supervised_console_family("gate_expander").permits_execution(write_kind="gate", proof_run=True)
    assert get_supervised_console_family("scene_preset").lifecycle == LIFECYCLE_PROPOSAL_ONLY


def test_write_kind_mapping_is_unique_and_operator_visible():
    assert family_id_for_write_kind("gain") == "gain_trim"
    assert family_id_for_write_kind("eq") == "eq_hpf"
    assert family_id_for_write_kind("snapshot") == "scene_preset"
    registry = supervised_console_family_registry()
    assert registry["routing"]["normal_supervised_apply_allowed"] is False
    assert registry["fx_sends_inserts"]["live_allowed_write_kinds"] == []
    assert registry["fx_sends_inserts"]["fully_live_allowed"] is False
    assert registry["fader"]["proof_run_allowed"] is True
    assert registry["fader"]["blockers"] == ["owner_readiness_promotion_required"]
