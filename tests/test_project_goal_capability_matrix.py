from backend.processing_registry import iter_processing_methods
from backend.project_goal_capability_matrix import (
    AUTOMIXER_OBJECTIVE_CAPABILITY_IDS,
    AUTOMIXER_OBJECTIVE_CAPABILITY_IDS,
    AUTONOMY_MODE_CEILINGS,
    CURRENT_PROVEN_LIVE_WRITE_KINDS,
    GOAL_CAPABILITY_STATUSES,
    PROJECT_GOAL_LIVE_WRITE_KINDS,
    build_project_goal_objective_alignment_snapshot,
    build_project_goal_capability_matrix_snapshot,
    build_project_goal_capability_progress_snapshot,
    get_project_goal_capability,
    iter_project_goal_capabilities,
)
from backend.operator_mode_policy import build_operator_mode_status


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


def test_project_goal_objective_alignment_snapshot_is_read_only_and_complete():
    snapshot = build_project_goal_objective_alignment_snapshot()
    assert snapshot["type"] == "project_goal_capability_objective_alignment_snapshot"
    assert snapshot["safety"]["read_only"] is True
    assert snapshot["safety"]["live_write"] is False
    assert snapshot["safety"]["full_autonomous_live_allowed"] is False

    summary = snapshot["summary"]
    assert summary["missing_row_count"] == 0
    assert summary["required_row_count"] == len(AUTOMIXER_OBJECTIVE_CAPABILITY_IDS)
    assert summary["required_row_count"] == summary["present_row_count"]
    assert summary["full_autonomous_status"] == "blocked_live"
    assert summary["autonomy_readiness_score"] >= 0
    assert summary["autonomy_readiness_score"] <= 100
    assert (
        summary["autonomy_readiness_ready_row_count"]
        + summary["autonomy_readiness_gap_row_count"]
        == summary["required_row_count"]
    )
    required_rows_by_area = {}
    required_status_counts_by_area: Dict[str, Dict[str, int]] = {}
    for row_id in AUTOMIXER_OBJECTIVE_CAPABILITY_IDS:
        row = get_project_goal_capability(row_id)
        assert row is not None
        required_rows_by_area.setdefault(row.goal_area, set()).add(row.id)
        area_status_counts = required_status_counts_by_area.setdefault(row.goal_area, {})
        area_status_counts[row.status] = area_status_counts.get(row.status, 0) + 1
    expected_score = (
        int(
            (summary["autonomy_readiness_ready_row_count"] / summary["required_row_count"])
            * 100
        )
        if summary["required_row_count"]
        else 0
    )
    assert summary["autonomy_readiness_score"] == expected_score
    readiness_by_area = summary["autonomy_readiness_by_goal_area"]
    assert isinstance(readiness_by_area, dict)
    assert readiness_by_area
    assert (
        sum(item["required_row_count"] for item in readiness_by_area.values())
        == summary["required_row_count"]
    )
    assert (
        sum(item["ready_row_count"] for item in readiness_by_area.values())
        == summary["autonomy_readiness_ready_row_count"]
    )
    assert (
        sum(item["gap_row_count"] for item in readiness_by_area.values())
        == summary["autonomy_readiness_gap_row_count"]
    )
    priority_expected = [
        {
            "goal_area": area_id,
            "required_row_count": area_summary["required_row_count"],
            "ready_row_count": area_summary["ready_row_count"],
            "gap_row_count": area_summary["gap_row_count"],
            "readiness_score": area_summary["readiness_score"],
        }
        for area_id, area_summary in sorted(
            readiness_by_area.items(),
            key=lambda item: (
                -item[1]["gap_row_count"],
                item[1]["readiness_score"],
                item[0],
            ),
        )
    ]
    assert summary["autonomy_readiness_goal_area_priorities"] == priority_expected
    for area_id, area_summary in readiness_by_area.items():
        assert area_summary["required_row_count"] >= 0
        assert area_summary["ready_row_count"] >= 0
        assert area_summary["gap_row_count"] >= 0
        assert isinstance(area_summary["ready_row_ids"], list)
        assert isinstance(area_summary["gap_row_ids"], list)
        assert isinstance(area_summary["status_counts"], dict)
        assert area_summary["ready_row_count"] + area_summary["gap_row_count"] == area_summary["required_row_count"]
        assert (
            set(area_summary["ready_row_ids"]).isdisjoint(
                set(area_summary["gap_row_ids"]),
            )
        )
        assert (
            len(area_summary["ready_row_ids"])
            + len(area_summary["gap_row_ids"])
            == area_summary["required_row_count"]
        )
        assert set(area_summary["ready_row_ids"]).union(area_summary["gap_row_ids"]) == set(
            required_rows_by_area[area_id]
        )
        assert sum(area_summary["status_counts"].values()) == area_summary["required_row_count"]
        assert area_summary["status_counts"] == required_status_counts_by_area[area_id]
        expected_area_score = (
            int((area_summary["ready_row_count"] / area_summary["required_row_count"]) * 100)
            if area_summary["required_row_count"]
            else 0
        )
        assert area_summary["readiness_score"] == expected_area_score
        assert (
            (area_summary["readiness_score"] == 100)
            == (area_summary["gap_row_count"] == 0)
            or area_summary["required_row_count"] == 0
        )
    assert summary["blocked_live_row_ids"] == ["mode_full_autonomous"]
    assert "mode_supervised_automix" in summary["proposal_only_row_ids"]
    assert summary["supervised_candidate_row_count"] == len(summary["supervised_candidate_row_ids"])
    assert summary["blocked_live_row_actions"]
    assert summary["blocked_live_rows"] == ["mode_full_autonomous"]
    assert summary["currently_live_writable_rows"]
    next_focus = summary["autonomy_readiness_next_focus"]
    assert next_focus["reason"] == "highest_gap_goal_area_first_gap_row"
    assert next_focus["goal_area"] == priority_expected[0]["goal_area"]
    assert next_focus["capability_id"] in readiness_by_area[
        next_focus["goal_area"]
    ]["gap_row_ids"]
    focused_row = get_project_goal_capability(next_focus["capability_id"])
    assert focused_row is not None
    assert next_focus["label"] == focused_row.label
    assert next_focus["status"] == focused_row.status
    assert next_focus["autonomy_mode_ceiling"] == focused_row.autonomy_mode_ceiling
    assert next_focus["blocker"] == focused_row.blocker
    assert next_focus["next_step"] == focused_row.next_step
    assert next_focus["method_ids"] == list(focused_row.method_ids)
    assert next_focus["support_paths"] == list(focused_row.support_paths)
    assert next_focus["target_console_surfaces"] == list(
        focused_row.target_console_surfaces
    )


def test_runtime_mode_capability_rows_match_operator_mode_policy():
    runtime_mode_rows = {
        "mode_analysis": ("analyze", "analysis", False),
        "mode_assisted": ("assist", "assisted", False),
        "mode_supervised_automix": ("supervised", "supervised_narrow", True),
        "mode_full_autonomous": ("auto", "blocked", False),
    }

    for row_id, (operator_mode, expected_ceiling, expected_live_apply) in (
        runtime_mode_rows.items()
    ):
        row = get_project_goal_capability(row_id)
        assert row is not None, row_id

        policy = build_operator_mode_status(
            mode=operator_mode,
            connection_mode="wing",
            wing_boundary_active=True,
        )

        assert row.autonomy_mode_ceiling == expected_ceiling
        assert row.goal_area == "runtime_mode"
        assert row.project_requirement
        assert row.current_live_write_kinds == tuple(
            policy["capabilities"]["allowed_live_write_kinds"]
        )
        assert policy["capabilities"]["can_apply_to_console"] is expected_live_apply

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
            assert kind in PROJECT_GOAL_LIVE_WRITE_KINDS


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


def test_supervised_automix_goal_requires_wing_candidate_prestart_trace():
    row = get_project_goal_capability("mode_supervised_automix")

    assert row is not None
    assert row.autonomy_mode_ceiling == "supervised_narrow"
    assert row.current_live_write_kinds == PROJECT_GOAL_LIVE_WRITE_KINDS
    assert row.current_live_write_kinds == ("fader", "gain")
    assert "pan" not in row.current_live_write_kinds
    assert "accepted-candidate prestart trace" in row.next_step
    assert "read-only preflight review" in row.next_step
    assert "supervised_prestart_gate" in row.next_step
    assert "expected_supervised_prestart_gate" in row.next_step
    assert "before widening" in row.next_step


def test_console_read_parameters_tracks_physical_readback_capture_plan():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "analysis"
    assert "read_surface_remediation_plan" in row.method_ids
    assert "routing_group_read_map" in row.method_ids
    assert "scene_preset_read_inventory" in row.method_ids
    assert "read_surface_physical_readback_capture_plan" in row.method_ids
    assert "read_surface_fixture_evidence" in row.method_ids
    assert "snapshots_scenes_physical_readback_artifact_intake" in row.method_ids
    assert "read_surface_physical_readback_artifact_review_queue" in row.method_ids
    assert "read_surface_physical_readback_fixture_acceptance_ledger" in row.method_ids
    assert "read_surface_physical_readback_fixture_closure_plan" in row.method_ids
    assert "backend/read_surface_physical_readback_capture_plan.py" in row.support_paths
    assert "backend/routing_group_read_map.py" in row.support_paths
    assert "backend/scene_preset_read_inventory.py" in row.support_paths
    assert (
        "backend/snapshots_scenes_physical_readback_artifact_intake.py"
        in row.support_paths
    )
    assert (
        "backend/read_surface_physical_readback_artifact_review_queue.py"
        in row.support_paths
    )
    assert (
        "backend/read_surface_physical_readback_fixture_acceptance_ledger.py"
        in row.support_paths
    )
    assert (
        "backend/read_surface_physical_readback_fixture_closure_plan.py"
        in row.support_paths
    )
    assert "physical fixtures" in row.next_step
    assert row.current_live_write_kinds == ()


def test_deessing_is_tracked_as_proposal_only_method_not_ignored():
    row = get_project_goal_capability("deessing")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "assisted"
    assert row.method_ids == ("deesser_vocal_sibilance",)
    assert "de-esser" in row.next_step


def test_gating_is_freshness_and_identity_guarded_without_live_widening():
    gating = get_project_goal_capability("gating")
    console_gate = get_project_goal_capability("console_gate_control")

    for row in (gating, console_gate):
        assert row is not None
        assert row.status == "proposal_only"
        assert row.autonomy_mode_ceiling == "assisted"
        assert "context_aware_gate_agent" in row.method_ids
        assert "backend/auto_fader_v2/controller.py" in row.support_paths
        assert "backend/source_identity_downstream_gate.py" in row.support_paths
        assert row.current_live_write_kinds == ()
        assert "trusted source identity" in row.next_step
        assert "chatter guard" in row.next_step
        assert "AutoFaderV2" in row.blocker


def test_equalization_tracks_fail_closed_auto_eq_boundary():
    row = get_project_goal_capability("equalization")
    console = get_project_goal_capability("console_eq_control")

    for capability in (row, console):
        assert capability is not None
        assert capability.status == "proposal_only"
        assert capability.autonomy_mode_ceiling == "assisted"
        assert "auto_eq" in capability.method_ids
        assert "corrective_eq_proposal" in capability.method_ids
        assert "backend/auto_eq.py" in capability.support_paths
        assert "backend/corrective_eq_proposal.py" in capability.support_paths
        assert "backend/auto_fader_v2/controller.py" in capability.support_paths
        assert "backend/auto_soundcheck_engine.py" in capability.support_paths
        assert "blocked evidence" in capability.blocker
        assert capability.current_live_write_kinds == ()
    assert "AutoSoundcheck EQ-enable writes" in console.blocker
    assert "operator review" in row.next_step
    assert "protected-band" in row.next_step
    assert "readback and rollback proof" in console.next_step


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
    console = get_project_goal_capability("console_compressor_control")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "assisted"
    assert "compressor_dynamics_proposal" in row.method_ids
    assert "auto_compressor" in row.method_ids
    assert "auto_compressor_cf" in row.method_ids
    assert "backend/auto_compressor.py" in row.support_paths
    assert "backend/auto_compressor_cf.py" in row.support_paths
    assert "backend/server.py" in row.support_paths
    assert "backend/auto_soundcheck_engine.py" in console.support_paths
    assert "server reset helper" in row.blocker
    assert "blocked evidence" in row.blocker
    assert "AutoSoundcheck compressor-threshold writes" in console.blocker
    assert "operator review" in row.next_step
    assert "feedback safety" in row.next_step
    assert console is not None
    assert console.status == "proposal_only"
    assert "auto_compressor" in console.method_ids
    assert "auto_compressor_cf" in console.method_ids
    assert "backend/auto_compressor.py" in console.support_paths
    assert "backend/auto_compressor_cf.py" in console.support_paths
    assert "blocked evidence" in console.blocker
    assert "feedback safety" in console.next_step


def test_panning_is_tracked_as_transport_free_proposal_layer():
    row = get_project_goal_capability("panning")
    console = get_project_goal_capability("console_pan_control")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "assisted"
    assert "pan_spatial_proposal" in row.method_ids
    assert "auto_panner" in row.method_ids
    assert "backend/auto_panner.py" in row.support_paths
    assert "backend/auto_panner_adaptive.py" in row.support_paths
    assert "fail-closed" in row.blocker
    assert "mono/phase check" in row.blocker
    assert "operator review" in row.next_step
    assert "mono/phase-safe" in row.next_step
    assert console is not None
    assert console.status == "proposal_only"
    assert "pan_spatial_proposal" in console.method_ids
    assert "backend/auto_panner.py" in console.support_paths
    assert "backend/auto_panner_adaptive.py" in console.support_paths
    assert "adaptive panner" in console.blocker
    assert "blocked evidence" in console.blocker
    assert "mono/phase compatibility" in console.next_step


def test_phase_correction_tracks_fail_closed_delay_polarity_boundary():
    row = get_project_goal_capability("phase_correction")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "assisted"
    assert "phase_alignment" in row.method_ids
    assert "blocked evidence" in row.blocker
    assert "readback and rollback proof" in row.next_step


def test_fx_is_tracked_as_transport_free_proposal_layer():
    effects = get_project_goal_capability("effects_management")
    sends = get_project_goal_capability("effect_sends")
    console = get_project_goal_capability("console_fx_control")

    assert effects is not None
    assert effects.status == "proposal_only"
    assert effects.autonomy_mode_ceiling == "assisted"
    assert "fx_send_proposal" in effects.method_ids
    assert "auto_fx" in effects.method_ids
    assert "backend/operator_product_state.py" in effects.support_paths
    assert "backend/auto_fx.py" in effects.support_paths
    assert "AutoReverb" in effects.blocker
    assert "fail-closed" in effects.blocker
    assert "read-only FX schema hints" in effects.next_step
    assert sends is not None
    assert sends.status == "proposal_only"
    assert "fx_send_proposal" in sends.method_ids
    assert "backend/auto_fx.py" in sends.support_paths
    assert console is not None
    assert console.status == "proposal_only"
    assert "fx_send_proposal" in console.method_ids
    assert "backend/operator_product_state.py" in console.support_paths
    assert "backend/auto_fx.py" in console.support_paths
    assert "AutoReverb" in console.blocker
    assert "blocked evidence" in console.blocker
    assert "schema hints" in console.next_step


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
    assert "backend/source_identity_downstream_gate.py" in row.support_paths
    assert row.current_live_write_kinds == ()
    assert "trusted-source-identity" in row.next_step
    assert "replay" in row.next_step


def test_long_term_memory_learning_is_tracked_as_proposal_layer_without_live_widening():
    row = get_project_goal_capability("long_term_memory_learning")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "assisted"
    assert "long_term_memory_learning_proposal" in row.method_ids
    assert "operator_proposal_apply_sequence_policy_memory_index" in row.method_ids
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_review"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_plan"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_contract"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer"
        in row.method_ids
    )
    assert (
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer"
        in row.method_ids
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_index.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_review.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_plan.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_contract.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_executor.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer.py"
        in row.support_paths
    )
    assert (
        "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer.py"
        in row.support_paths
    )
    assert row.current_live_write_kinds == ()
    assert "replay" in row.next_step


def test_bus_group_balance_is_tracked_as_proposal_layer_without_live_widening():
    mix_balance = get_project_goal_capability("mix_balance")
    console = get_project_goal_capability("console_dca_group_matrix_control")

    assert mix_balance is not None
    assert mix_balance.status == "proposal_only"
    assert "bus_group_balance_proposal" in mix_balance.method_ids
    assert mix_balance.current_live_write_kinds == ("fader",)
    assert "AutoFOH bus/DCA/master typed actions are blocked evidence" in mix_balance.blocker
    assert console is not None
    assert console.status == "proposal_only"
    assert console.autonomy_mode_ceiling == "assisted"
    assert "routing_group_read_map" in console.method_ids
    assert "bus_group_balance_proposal" in console.method_ids
    assert "backend/routing_group_read_map.py" in console.support_paths
    assert "backend/autofoh_safety.py" in console.support_paths
    assert "blocked evidence only" in console.blocker
    assert console.current_live_write_kinds == ()
    assert "topology map" in console.next_step


def test_fader_control_stays_product_layer_only_with_autosoundcheck_restore_evidence():
    row = get_project_goal_capability("console_fader_control")

    assert row is not None
    assert row.status == "canonical"
    assert row.current_live_write_kinds == ("fader",)
    assert "backend/auto_soundcheck_engine.py" in row.support_paths
    assert "AutoSoundcheck fader restore is blocked evidence" in row.blocker
    assert "Product Layer approval" in row.next_step


def test_mute_control_is_tracked_as_proposal_layer_without_live_widening():
    row = get_project_goal_capability("console_mute_control")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "assisted"
    assert "mute_management_proposal" in row.method_ids
    assert "backend/auto_soundcheck_engine.py" in row.support_paths
    assert row.current_live_write_kinds == ()
    assert "AutoSoundcheck mute restore returns blocked evidence" in row.blocker
    assert "operator review" in row.next_step


def test_routing_control_is_tracked_as_proposal_layer_without_live_widening():
    row = get_project_goal_capability("console_routing_control")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "assisted"
    assert "routing_group_read_map" in row.method_ids
    assert "routing_management_proposal" in row.method_ids
    assert "backend/routing_group_read_map.py" in row.support_paths
    assert row.current_live_write_kinds == ()
    assert "topology map" in row.next_step
    assert "operator review" in row.next_step


def test_scene_preset_control_is_tracked_as_proposal_layer_without_live_widening():
    row = get_project_goal_capability("console_scene_preset_control")

    assert row is not None
    assert row.status == "proposal_only"
    assert row.autonomy_mode_ceiling == "assisted"
    assert "scene_preset_read_inventory" in row.method_ids
    assert "scene_preset_management_proposal" in row.method_ids
    assert "backend/scene_preset_read_inventory.py" in row.support_paths
    assert "backend/operator_product_state.py" in row.support_paths
    assert row.current_live_write_kinds == ()
    assert "operator review" in row.next_step
    assert "read-only scoped scene/preset diffs" in row.next_step
    assert "Product Layer snapshot freshness" in row.next_step
    assert "read-only rollback evidence" in row.next_step


def test_snapshot_is_read_only_and_fail_closed_for_live_widening():
    snapshot = build_project_goal_capability_matrix_snapshot()

    assert snapshot["type"] == "project_goal_capability_matrix"
    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "live_widening_allowed": False,
        "current_proven_live_write_kinds": list(CURRENT_PROVEN_LIVE_WRITE_KINDS),
        "wing_deployment_live_write_kinds": list(PROJECT_GOAL_LIVE_WRITE_KINDS),
        "project_goal_live_write_kinds": list(PROJECT_GOAL_LIVE_WRITE_KINDS),
    }
    assert "deessing" not in snapshot["summary"]["missing_capability_ids"]
    assert "mode_full_autonomous" in snapshot["summary"]["blocked_live_capability_ids"]
    assert "console_mute_control" not in snapshot["summary"]["blocked_live_capability_ids"]
    assert "console_scene_preset_control" not in snapshot["summary"]["blocked_live_capability_ids"]
    assert "console_fader_control" in snapshot["summary"]["currently_live_writable_capability_ids"]
    assert snapshot["summary"]["status_counts"]["blocked_live"] >= 1


def test_goal_capability_progress_snapshot_aligns_with_matrix_and_highlights_active_blockers():
    matrix = build_project_goal_capability_matrix_snapshot()
    progress = build_project_goal_capability_progress_snapshot()

    assert progress["type"] == "project_goal_capability_progress_snapshot"
    assert progress["status"] == "ok"
    assert progress["summary"]["row_count"] == len(iter_project_goal_capabilities())
    assert progress["summary"]["row_count"] == matrix["summary"]["row_count"]
    assert progress["summary"]["missing_capability_ids"] == matrix["summary"]["missing_capability_ids"]
    assert progress["summary"]["blocked_live_capability_ids"] == matrix["summary"]["blocked_live_capability_ids"]
    assert progress["summary"]["currently_live_writable_capability_ids"] == matrix["summary"][
        "currently_live_writable_capability_ids"
    ]
    goal_area_status_counts = {}
    for row in progress["rows"]:
        area = row["goal_area"]
        area_status = goal_area_status_counts.setdefault(
            area, {status: 0 for status in GOAL_CAPABILITY_STATUSES}
        )
        area_status[row["status"]] += 1
    assert progress["summary"]["goal_area_status_counts"] == goal_area_status_counts
    assert (
        isinstance(progress["summary"]["goal_areas_with_blockers"], list)
        and progress["summary"]["goal_areas_with_blockers"]
    )
    assert "runtime_mode" in progress["summary"]["goal_area_counts"]
    assert "mode_analysis" in progress["summary"]["runtime_mode_ceiling_by_id"]
    assert progress["summary"]["runtime_mode_ceiling_by_id"] == {
        row["id"]: row["autonomy_mode_ceiling"] for row in progress["rows"] if row["goal_area"] == "runtime_mode"
    }
    assert progress["summary"]["supervised_narrow_live_candidates"]
    assert progress["summary"]["goal_area_counts"].keys() == {
        row.goal_area for row in iter_project_goal_capabilities()
    }

    assert isinstance(progress["next_steps"], list)
    for step in progress["next_steps"]:
        assert step["capability_id"] and step["goal_area"] and step["next_step"]

    status_counts = progress["summary"]["status_counts"]
    for status in GOAL_CAPABILITY_STATUSES:
        assert status in status_counts

    objective_alignment = build_project_goal_objective_alignment_snapshot()
    summary = progress["summary"]
    assert summary["project_goal_objective_autonomy_readiness_gap_row_count"] == (
        objective_alignment["summary"]["autonomy_readiness_gap_row_count"]
    )
    assert summary["project_goal_objective_autonomy_readiness_ready_row_count"] == (
        objective_alignment["summary"]["autonomy_readiness_ready_row_count"]
    )
    assert summary["project_goal_objective_autonomy_readiness_by_goal_area"] == (
        objective_alignment["summary"]["autonomy_readiness_by_goal_area"]
    )
    assert summary[
        "project_goal_objective_autonomy_readiness_goal_area_priorities"
    ] == objective_alignment["summary"]["autonomy_readiness_goal_area_priorities"]
    assert summary["project_goal_objective_autonomy_readiness_next_focus"] == (
        objective_alignment["summary"]["autonomy_readiness_next_focus"]
    )
    hotspots = summary["project_goal_objective_autonomy_readiness_hotspots"]
    assert isinstance(hotspots, list)
    assert hotspots
    assert all(
        item["goal_area"]
        and isinstance(item["gap_row_count"], int)
        and isinstance(item["readiness_score"], int)
        and isinstance(item["blocker_row_count"], int)
        and isinstance(item["blocker_row_ids"], list)
        for item in hotspots
    )
    assert all(
        item["gap_row_count"] > 0
        for item in hotspots
    )
    assert sum(item["gap_row_count"] for item in hotspots) == objective_alignment["summary"]["autonomy_readiness_gap_row_count"]
    assert hotspots == sorted(
        [
            {
                "goal_area": area_id,
                "gap_row_count": area_data["gap_row_count"],
                "readiness_score": area_data["readiness_score"],
                "blocker_row_count": len(
                    [
                        row["id"]
                        for row in objective_alignment["rows"]
                        if row["goal_area"] == area_id
                        and row["status"] not in {"canonical", "supervised_narrow"}
                        and row["blocker"]
                    ]
                ),
                "blocker_row_ids": sorted(
                    [
                        row["id"]
                        for row in objective_alignment["rows"]
                        if row["goal_area"] == area_id
                        and row["status"] not in {"canonical", "supervised_narrow"}
                        and row["blocker"]
                    ]
                ),
            }
            for area_id, area_data in objective_alignment["summary"][
                "autonomy_readiness_by_goal_area"
            ].items()
            if area_data["gap_row_count"] > 0
        ],
        key=lambda item: (-item["gap_row_count"], item["readiness_score"], item["goal_area"]),
    )

    required_rows = [row_id for row_id in AUTOMIXER_OBJECTIVE_CAPABILITY_IDS if row_id in set(
        row["id"] for row in progress["rows"]
    )]
    assert len(required_rows) == objective_alignment["summary"]["required_row_count"]
