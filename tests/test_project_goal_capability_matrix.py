from backend.processing_registry import iter_processing_methods
from backend.project_goal_capability_matrix import (
    AUTONOMY_MODE_CEILINGS,
    CURRENT_PROVEN_LIVE_WRITE_KINDS,
    GOAL_CAPABILITY_STATUSES,
    build_project_goal_capability_matrix_snapshot,
    get_project_goal_capability,
    iter_project_goal_capabilities,
)


def test_project_goal_matrix_covers_explicit_goal_requirements():
    ids = {row.id for row in iter_project_goal_capabilities()}

    expected = {
        "mode_analysis",
        "mode_assisted",
        "mode_supervised_automix",
        "mode_full_autonomous",
        "analyze_multichannel_audio",
        "identify_source_types",
        "identify_arrangement_roles",
        "gain_staging",
        "equalization",
        "compression",
        "gating",
        "deessing",
        "panning",
        "frequency_separation",
        "cross_channel_unmasking",
        "phase_correction",
        "effects_management",
        "effect_sends",
        "mix_balance",
        "musician_level_adaptation",
        "live_adaptation",
        "genre_variants",
        "master_bus_optimization",
        "realtime_corrections",
        "console_read_parameters",
        "console_analyze_state",
        "console_preamp_trim_control",
        "console_fader_control",
        "console_mute_control",
        "console_pan_control",
        "console_eq_control",
        "console_compressor_control",
        "console_gate_control",
        "console_routing_control",
        "console_dca_group_matrix_control",
        "console_fx_control",
        "console_scene_preset_control",
        "decision_basis",
        "long_term_memory_learning",
    }

    assert expected <= ids


def test_project_goal_rows_are_valid_and_reference_registered_methods():
    method_ids = {method.id for method in iter_processing_methods()}
    rows = list(iter_project_goal_capabilities())
    ids = [row.id for row in rows]

    assert len(ids) == len(set(ids))
    for row in rows:
        assert row.status in GOAL_CAPABILITY_STATUSES
        assert row.autonomy_mode_ceiling in AUTONOMY_MODE_CEILINGS
        assert set(row.method_ids) <= method_ids
        for kind in row.current_live_write_kinds:
            assert kind in CURRENT_PROVEN_LIVE_WRITE_KINDS


def test_full_autonomous_and_broad_console_writes_remain_blocked():
    full_auto = get_project_goal_capability("mode_full_autonomous")
    routing = get_project_goal_capability("console_routing_control")
    scene = get_project_goal_capability("console_scene_preset_control")

    assert full_auto is not None
    assert full_auto.status == "blocked_live"
    assert full_auto.current_live_write_kinds == ()
    assert routing is not None and routing.status == "proposal_only"
    assert routing.current_live_write_kinds == ()
    assert scene is not None and scene.status == "proposal_only"
    assert scene.current_live_write_kinds == ()


def test_deessing_is_tracked_as_proposal_only_method_not_ignored():
    row = get_project_goal_capability("deessing")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "assisted"
    assert row.method_ids == ("deesser_vocal_sibilance",)
    assert "de-esser" in row.next_step


def test_arrangement_roles_are_tracked_as_proposal_layer_without_live_widening():
    row = get_project_goal_capability("identify_arrangement_roles")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "assisted"
    assert "arrangement_role_proposal" in row.method_ids
    assert row.current_live_write_kinds == ()
    assert "operator review" in row.next_step


def test_compression_is_tracked_as_transport_free_proposal_layer():
    row = get_project_goal_capability("compression")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "assisted"
    assert "compressor_dynamics_proposal" in row.method_ids
    assert "operator review" in row.next_step


def test_panning_is_tracked_as_transport_free_proposal_layer():
    row = get_project_goal_capability("panning")
    console = get_project_goal_capability("console_pan_control")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "assisted"
    assert "pan_spatial_proposal" in row.method_ids
    assert "operator review" in row.next_step
    assert console is not None
    assert console.status == "proposal_only"
    assert "pan_spatial_proposal" in console.method_ids


def test_fx_is_tracked_as_transport_free_proposal_layer():
    effects = get_project_goal_capability("effects_management")
    sends = get_project_goal_capability("effect_sends")
    console = get_project_goal_capability("console_fx_control")

    assert effects is not None
    assert effects.status == "proposal_only"
    assert effects.autonomy_mode_ceiling == "assisted"
    assert "fx_send_proposal" in effects.method_ids
    assert sends is not None
    assert sends.status == "proposal_only"
    assert "fx_send_proposal" in sends.method_ids
    assert console is not None
    assert console.status == "proposal_only"
    assert "fx_send_proposal" in console.method_ids


def test_master_bus_is_tracked_as_transport_free_proposal_layer():
    row = get_project_goal_capability("master_bus_optimization")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "assisted"
    assert "master_bus_optimization_proposal" in row.method_ids
    assert "operator review" in row.next_step
    assert row.current_live_write_kinds == ()


def test_genre_variants_are_tracked_as_proposal_layer_without_live_widening():
    row = get_project_goal_capability("genre_variants")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "assisted"
    assert "mix_variant_profile_proposal" in row.method_ids
    assert row.current_live_write_kinds == ()
    assert "operator review" in row.next_step


def test_performance_adaptation_rows_are_tracked_as_proposal_layer_without_live_widening():
    for capability_id in (
        "musician_level_adaptation",
        "live_adaptation",
        "realtime_corrections",
    ):
        row = get_project_goal_capability(capability_id)

        assert row is not None
        assert row.status == "proposal_only"
        assert row.autonomy_mode_ceiling == "assisted"
        assert "performance_adaptation_proposal" in row.method_ids
        assert row.current_live_write_kinds == ()


def test_decision_basis_is_tracked_as_proposal_layer_without_live_widening():
    row = get_project_goal_capability("decision_basis")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "assisted"
    assert "decision_evidence_packet_proposal" in row.method_ids
    assert row.current_live_write_kinds == ()
    assert "replay" in row.next_step


def test_long_term_memory_learning_is_tracked_as_proposal_layer_without_live_widening():
    row = get_project_goal_capability("long_term_memory_learning")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "assisted"
    assert "long_term_memory_learning_proposal" in row.method_ids
    assert row.current_live_write_kinds == ()
    assert "replay" in row.next_step


def test_bus_group_balance_is_tracked_as_proposal_layer_without_live_widening():
    mix_balance = get_project_goal_capability("mix_balance")
    console = get_project_goal_capability("console_dca_group_matrix_control")

    assert mix_balance is not None
    assert mix_balance.status == "proposal_only"
    assert "bus_group_balance_proposal" in mix_balance.method_ids
    assert mix_balance.current_live_write_kinds == ("fader",)
    assert console is not None
    assert console.status == "proposal_only"
    assert console.autonomy_mode_ceiling == "assisted"
    assert "bus_group_balance_proposal" in console.method_ids
    assert console.current_live_write_kinds == ()


def test_mute_control_is_tracked_as_proposal_layer_without_live_widening():
    row = get_project_goal_capability("console_mute_control")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "assisted"
    assert "mute_management_proposal" in row.method_ids
    assert row.current_live_write_kinds == ()
    assert "operator review" in row.next_step


def test_routing_control_is_tracked_as_proposal_layer_without_live_widening():
    row = get_project_goal_capability("console_routing_control")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "assisted"
    assert "routing_management_proposal" in row.method_ids
    assert row.current_live_write_kinds == ()
    assert "operator review" in row.next_step


def test_scene_preset_control_is_tracked_as_proposal_layer_without_live_widening():
    row = get_project_goal_capability("console_scene_preset_control")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "assisted"
    assert "scene_preset_management_proposal" in row.method_ids
    assert row.current_live_write_kinds == ()
    assert "operator review" in row.next_step


def test_snapshot_is_read_only_and_fail_closed_for_live_widening():
    snapshot = build_project_goal_capability_matrix_snapshot()

    assert snapshot["type"] == "project_goal_capability_matrix"
    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "live_widening_allowed": False,
        "current_proven_live_write_kinds": ["fader", "gain"],
    }
    assert "deessing" not in snapshot["summary"]["missing_capability_ids"]
    assert "mode_full_autonomous" in snapshot["summary"]["blocked_live_capability_ids"]
    assert "console_mute_control" not in snapshot["summary"]["blocked_live_capability_ids"]
    assert "console_scene_preset_control" not in snapshot["summary"]["blocked_live_capability_ids"]
    assert "console_fader_control" in snapshot["summary"]["currently_live_writable_capability_ids"]
    assert snapshot["summary"]["status_counts"]["blocked_live"] >= 1
