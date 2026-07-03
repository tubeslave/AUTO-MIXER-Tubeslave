from unittest.mock import patch

import backend.processing_registry as processing_registry

from backend.processing_registry import (
    LIVE_ALLOWED_WRITE_KINDS,
    PROCESSING_DOMAINS,
    PROCESSING_STATUSES,
    build_processing_registry_snapshot,
    build_pipeline_runtime_summary,
    get_pipeline_profile,
    get_processing_method,
    iter_pipeline_profiles,
    iter_processing_methods,
)
from backend.read_surface_inventory import (
    build_read_surface_inventory_snapshot,
    get_read_surface,
    iter_read_surfaces,
)
from backend.safety_coverage_matrix import (
    COVERAGE_STATUSES,
    PROOF_STATES,
    build_safety_coverage_matrix_snapshot,
    get_safety_coverage_row,
    iter_safety_coverage_rows,
)
from backend.write_surface_inventory import (
    LIVE_ALLOWED_WRITE_KINDS as WRITE_SURFACE_LIVE_ALLOWED_WRITE_KINDS,
    WRITE_SURFACE_STATUSES,
    build_write_surface_inventory_snapshot,
    get_write_surface,
    iter_write_surfaces,
)


def test_proposal_adapter_module_stays_free_of_live_mixer_imports():
    from pathlib import Path

    source = Path("backend/processing_proposal_adapters.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        "mixer_client.set_",
        ".send(",
        "set_eq(",
        "set_compressor(",
    ]
    for token in forbidden:
        assert token not in source


def test_processing_methods_have_valid_statuses_domains_and_unique_ids():
    methods = list(iter_processing_methods())
    ids = [method.id for method in methods]

    assert len(ids) == len(set(ids))
    assert methods
    for method in methods:
        assert method.status in PROCESSING_STATUSES
        assert method.domain in PROCESSING_DOMAINS
        assert method.module


def test_live_capable_methods_require_supervision_and_only_fader_gain():
    live_methods = [method for method in iter_processing_methods() if method.can_write_live]

    assert LIVE_ALLOWED_WRITE_KINDS == ("fader", "gain")
    assert [method.id for method in live_methods] == ["supervised_manual_fader_gain"]
    for method in live_methods:
        assert method.requires_supervision is True
        assert tuple(method.live_write_kinds) == LIVE_ALLOWED_WRITE_KINDS
        assert method.status == "canonical"


def test_auto_gate_caig_is_proposal_only_with_quarantined_direct_send():
    method = get_processing_method("auto_gate_caig")

    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_analyze is True
    assert method.can_propose is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert "direct_gate_send_quarantined" in method.notes
    assert "no_live_writes" in method.notes


def test_auto_eq_is_proposal_only_with_quarantined_direct_apply():
    method = get_processing_method("auto_eq")

    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_analyze is True
    assert method.can_propose is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert "direct_eq_apply_quarantined" in method.notes
    assert "direct_eq_reset_quarantined" in method.notes
    assert "blocked_apply_evidence" in method.notes
    assert "no_live_writes" in method.notes


def test_auto_fx_planner_is_proposal_only_with_quarantined_direct_apply():
    method = get_processing_method("auto_fx")

    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_analyze is True
    assert method.can_propose is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert "direct_fx_apply_quarantined" in method.notes
    assert "direct_reverb_apply_quarantined" in method.notes
    assert "blocked_apply_evidence" in method.notes
    assert "no_live_writes" in method.notes


def test_auto_panner_is_proposal_only_with_quarantined_direct_apply():
    method = get_processing_method("auto_panner")

    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_analyze is True
    assert method.can_propose is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert "direct_pan_apply_quarantined" in method.notes
    assert "adaptive_pan_send_quarantined" in method.notes
    assert "blocked_apply_evidence" in method.notes
    assert "no_live_writes" in method.notes


def test_phase_alignment_is_proposal_only_with_quarantined_direct_apply():
    method = get_processing_method("phase_alignment")

    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_analyze is True
    assert method.can_propose is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert "phase_delay_apply_quarantined" in method.notes
    assert "phase_delay_reset_quarantined" in method.notes
    assert "blocked_apply_evidence" in method.notes
    assert "no_live_writes" in method.notes


def test_auto_compressor_is_proposal_only_with_quarantined_direct_apply():
    method = get_processing_method("auto_compressor")
    cf_method = get_processing_method("auto_compressor_cf")

    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_analyze is True
    assert method.can_propose is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert "direct_compressor_apply_quarantined" in method.notes
    assert "blocked_apply_evidence" in method.notes
    assert "no_live_writes" in method.notes
    assert cf_method is not None
    assert cf_method.status == "proposal_only"
    assert "direct_cf_compressor_apply_quarantined" in cf_method.notes
    assert "blocked_apply_evidence" in cf_method.notes


def test_main_live_pipeline_executor_is_only_supervised_fader_gain():
    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert main.status == "canonical"
    assert main.can_write_live is True
    assert main.requires_supervision is True
    assert tuple(main.live_write_kinds) == LIVE_ALLOWED_WRITE_KINDS
    assert tuple(main.executor_method_ids) == ("supervised_manual_fader_gain",)

    blocked_statuses = {"placeholder", "quarantined", "deprecated"}
    for method_id in main.method_ids:
        method = get_processing_method(method_id)
        assert method is not None
        assert method.status not in blocked_statuses

    context_gate = get_processing_method("context_aware_gate_agent")
    assert context_gate is not None
    assert context_gate.can_propose is True
    assert context_gate.can_write_live is False
    assert context_gate.status == "proposal_only"
    assert "source_identity_downstream_trust" in context_gate.notes
    assert "context_aware_gate_agent" in main.proposal_method_ids


def test_pipeline_runtime_summary_filters_manual_scope_to_canonical_live_kinds():
    operator_mode = {
        "mode": "supervised",
        "capabilities": {
            "allowed_live_write_kinds": ["fader", "gain", "pan"],
            "can_apply_to_console": True,
        },
    }
    manual_write_supervision = {
        "approved_live_write_kinds": [
            "fader",
            "gain",
            "eq",
            "trim",
            "gain",
            "input_gain",
            "mute",
            "fader",
            "preamp_gain",
        ]
    }

    runtime = build_pipeline_runtime_summary(
        operator_mode,
        manual_write_supervision,
    )

    assert runtime["allowed_live_write_kinds"] == [
        "fader",
        "gain",
    ]


def test_pipeline_runtime_summary_filters_supervised_fallback_to_canonical_live_kinds():
    operator_mode = {
        "mode": "supervised",
        "capabilities": {
            "allowed_live_write_kinds": ["fader", "gain", "pan"],
            "can_apply_to_console": True,
        },
    }

    runtime = build_pipeline_runtime_summary(operator_mode, {})

    assert runtime["allowed_live_write_kinds"] == ["fader", "gain"]


def test_pipeline_runtime_summary_shows_no_live_write_kinds_in_assisted_mode_without_manual_override():
    operator_mode = {
        "mode": "assist",
        "capabilities": {"allowed_live_write_kinds": [], "can_apply_to_console": False},
    }

    runtime = build_pipeline_runtime_summary(operator_mode, {})

    assert runtime["allowed_live_write_kinds"] == []


def test_pipeline_runtime_summary_shows_no_live_write_kinds_in_locked_auto_mode():
    operator_mode = {
        "mode": "auto",
        "capabilities": {
            "allowed_live_write_kinds": [],
            "can_apply_to_console": False,
        },
    }

    runtime = build_pipeline_runtime_summary(operator_mode, {})

    assert runtime["allowed_live_write_kinds"] == []


def test_supervised_proof_artifact_evaluator_is_shadow_only_not_executor():
    method = get_processing_method("supervised_proof_artifact_evaluator")
    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert "supervised_proof_artifact_evaluator" not in main.executor_method_ids


def test_supervised_promotion_evidence_bridge_is_shadow_only_not_executor():
    method = get_processing_method("supervised_promotion_evidence_bridge")
    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert "supervised_promotion_evidence_bridge" not in main.executor_method_ids


def test_supervised_wing_proof_plan_is_shadow_only_not_executor():
    method = get_processing_method("supervised_wing_proof_plan")
    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}
    assert (
        method.blocked_reason
        == "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
    )
    assert "accepted read-only preflight decision" in method.summary
    assert "backend-port prestart contract" in method.summary
    assert "candidate_prestart_trace_required" in method.notes
    assert "explicit operator approval" in method.promotion_next

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert "supervised_wing_proof_plan" not in main.executor_method_ids


def test_supervised_proof_remediation_plan_is_shadow_only_not_executor():
    method = get_processing_method("supervised_proof_remediation_plan")
    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert "supervised_proof_remediation_plan" not in main.executor_method_ids


def test_autonomy_evidence_overlay_is_shadow_only_not_executor():
    method = get_processing_method("autonomy_evidence_overlay")
    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert "autonomy_evidence_overlay" not in main.executor_method_ids


def test_autonomy_mode_transition_gate_is_shadow_only_not_executor():
    method = get_processing_method("autonomy_mode_transition_gate")
    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert "autonomy_mode_transition_gate" not in main.executor_method_ids


def test_read_surface_remediation_plan_is_shadow_only_not_executor():
    method = get_processing_method("read_surface_remediation_plan")
    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert "read_surface_remediation_plan" not in main.executor_method_ids


def test_read_surface_physical_readback_capture_plan_is_shadow_only_not_executor():
    method = get_processing_method("read_surface_physical_readback_capture_plan")
    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}
    assert "execution_allowed_by_plan_false" in method.notes

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert "read_surface_physical_readback_capture_plan" not in main.executor_method_ids


def test_secondary_pipelines_do_not_declare_live_write_executor():
    for pipeline_id in ("offline_mix", "shadow_quality", "lab_dsp", "quarantine"):
        profile = get_pipeline_profile(pipeline_id)
        assert profile is not None
        assert profile.can_write_live is False
        assert profile.live_write_kinds == ()


def test_quarantined_and_placeholder_methods_are_not_pipeline_executors():
    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    executor_ids = set(main.executor_method_ids)

    for method in iter_processing_methods():
        if method.status in {"placeholder", "quarantined", "deprecated"}:
            assert method.can_write_live is False
            assert method.id not in executor_ids


def test_registry_snapshot_can_hide_full_method_inventory():
    snapshot = build_processing_registry_snapshot(include_methods=False)

    assert snapshot["type"] == "processing_pipeline_snapshot"
    assert snapshot["summary"]["main_pipeline_id"] == "main_live_assist"
    assert snapshot["summary"]["live_allowed_write_kinds"] == list(LIVE_ALLOWED_WRITE_KINDS)
    assert snapshot["summary"]["live_executor_method_ids"] == ["supervised_manual_fader_gain"]
    assert snapshot["write_surface_inventory"]["summary"]["allowed_live_surface_ids"] == [
        "product_layer_fader_gain_apply"
    ]
    assert snapshot["safety_coverage_matrix"]["summary"]["live_allowed_row_ids"] == []
    assert "fader_supervised" in snapshot["safety_coverage_matrix"]["summary"][
        "missing_or_partial_real_wing_proof_ids"
    ]
    assert snapshot["read_surface_inventory"]["summary"]["claim"] == (
        "read coverage is inventoried but not complete"
    )
    assert snapshot["read_surface_inventory"]["safety"]["live_write"] is False
    assert snapshot["read_surface_remediation_plan"]["type"] == "read_surface_remediation_plan"
    assert snapshot["read_surface_remediation_plan"]["safety"]["live_write"] is False
    assert snapshot["routing_group_read_map"]["type"] == "routing_group_read_map"
    assert snapshot["routing_group_read_map"]["safety"]["live_write"] is False
    assert snapshot["scene_preset_read_inventory"]["type"] == "scene_preset_read_inventory"
    assert snapshot["scene_preset_read_inventory"]["safety"]["live_write"] is False
    assert snapshot["read_surface_remediation_plan"]["summary"]["incomplete_surface_count"] >= 1
    assert (
        snapshot["read_surface_physical_readback_capture_plan"]["type"]
        == "read_surface_physical_readback_capture_plan"
    )
    assert snapshot["read_surface_physical_readback_capture_plan"]["safety"][
        "live_write"
    ] is False
    assert snapshot["read_surface_physical_readback_capture_plan"]["safety"][
        "execution_allowed_by_plan"
    ] is False
    assert snapshot["read_surface_physical_readback_capture_plan"]["summary"][
        "next_physical_readback_surface_id"
    ] == "raw_mixer_state_cache"
    assert snapshot["read_surface_physical_readback_capture_plan"]["summary"][
        "target_action_consistency"
    ] is True
    assert (
        snapshot["raw_mixer_state_physical_readback_contract"]["type"]
        == "raw_mixer_state_physical_readback_contract"
    )
    assert snapshot["raw_mixer_state_physical_readback_contract"]["safety"]["live_write"] is False
    assert (
        snapshot["raw_mixer_state_physical_readback_contract"]["summary"][
            "physical_fixture_gap_remains"
        ]
        is True
    )
    assert (
        snapshot["channel_basic_controls_physical_readback_contract"]["type"]
        == "channel_basic_controls_physical_readback_contract"
    )
    assert snapshot["channel_basic_controls_physical_readback_contract"]["safety"]["live_write"] is False
    assert (
        snapshot["channel_basic_controls_physical_readback_contract"]["summary"][
            "physical_fixture_gap_remains"
        ]
        is True
    )
    assert (
        snapshot["channel_processing_physical_readback_contract"]["type"]
        == "channel_processing_physical_readback_contract"
    )
    assert snapshot["channel_processing_physical_readback_contract"]["safety"]["live_write"] is False
    assert (
        snapshot["channel_processing_physical_readback_contract"]["summary"][
            "physical_fixture_gap_remains"
        ]
        is True
    )
    assert (
        snapshot["bus_group_main_dca_physical_readback_contract"]["type"]
        == "bus_group_main_dca_physical_readback_contract"
    )
    assert snapshot["bus_group_main_dca_physical_readback_contract"]["safety"]["live_write"] is False
    assert (
        snapshot["bus_group_main_dca_physical_readback_contract"]["summary"][
            "physical_fixture_gap_remains"
        ]
        is True
    )
    assert (
        snapshot["matrix_state_physical_readback_contract"]["type"]
        == "matrix_state_physical_readback_contract"
    )
    assert snapshot["matrix_state_physical_readback_contract"]["safety"]["live_write"] is False
    assert (
        snapshot["matrix_state_physical_readback_contract"]["summary"][
            "physical_fixture_gap_remains"
        ]
        is True
    )
    assert (
        snapshot["fx_slots_inserts_physical_readback_contract"]["type"]
        == "fx_slots_inserts_physical_readback_contract"
    )
    assert snapshot["fx_slots_inserts_physical_readback_contract"]["safety"]["live_write"] is False
    assert (
        snapshot["fx_slots_inserts_physical_readback_contract"]["summary"][
            "physical_fixture_gap_remains"
        ]
        is True
    )
    assert (
        snapshot["routing_sends_physical_readback_contract"]["type"]
        == "routing_sends_physical_readback_contract"
    )
    assert snapshot["routing_sends_physical_readback_contract"]["safety"]["live_write"] is False
    assert (
        snapshot["routing_sends_physical_readback_contract"]["summary"][
            "physical_fixture_gap_remains"
        ]
        is True
    )
    assert (
        snapshot["snapshots_scenes_physical_readback_contract"]["type"]
        == "snapshots_scenes_physical_readback_contract"
    )
    assert snapshot["snapshots_scenes_physical_readback_contract"]["safety"]["live_write"] is False
    assert snapshot["snapshots_scenes_physical_readback_contract"]["safety"][
        "scene_load_allowed"
    ] is False
    assert (
        snapshot["snapshots_scenes_physical_readback_contract"]["summary"][
            "physical_fixture_gap_remains"
        ]
        is True
    )
    assert (
        snapshot["snapshots_scenes_physical_readback_artifact_intake"]["type"]
        == "snapshots_scenes_physical_readback_artifact_intake"
    )
    assert (
        snapshot["snapshots_scenes_physical_readback_artifact_intake"]["safety"][
            "live_write"
        ]
        is False
    )
    assert (
        snapshot["snapshots_scenes_physical_readback_artifact_intake"]["safety"][
            "scene_load_allowed"
        ]
        is False
    )
    assert (
        snapshot["snapshots_scenes_physical_readback_artifact_intake"]["safety"][
            "scene_save_allowed"
        ]
        is False
    )
    assert (
        snapshot["snapshots_scenes_physical_readback_artifact_intake"]["summary"][
            "physical_fixture_gap_remains"
        ]
        is True
    )
    assert (
        snapshot["user_presets_physical_readback_contract"]["type"]
        == "user_presets_physical_readback_contract"
    )
    assert snapshot["user_presets_physical_readback_contract"]["safety"]["live_write"] is False
    assert snapshot["user_presets_physical_readback_contract"]["safety"][
        "preset_recall_allowed"
    ] is False
    assert snapshot["user_presets_physical_readback_contract"]["safety"][
        "preset_store_allowed"
    ] is False
    assert (
        snapshot["user_presets_physical_readback_contract"]["summary"][
            "physical_fixture_gap_remains"
        ]
        is True
    )
    assert (
        snapshot["dlive_state_physical_readback_contract"]["type"]
        == "dlive_state_physical_readback_contract"
    )
    assert snapshot["dlive_state_physical_readback_contract"]["safety"]["live_write"] is False
    assert snapshot["dlive_state_physical_readback_contract"]["safety"][
        "midi_send_allowed"
    ] is False
    assert snapshot["dlive_state_physical_readback_contract"]["safety"][
        "tcp_send_allowed"
    ] is False
    assert snapshot["dlive_state_physical_readback_contract"]["safety"][
        "dlive_write_allowed"
    ] is False
    assert (
        snapshot["dlive_state_physical_readback_contract"]["summary"][
            "physical_fixture_gap_remains"
        ]
        is True
    )
    assert snapshot["project_goal_capability_matrix"]["type"] == "project_goal_capability_matrix"
    assert snapshot["project_goal_capability_matrix"]["safety"][
        "live_widening_allowed"
    ] is False
    assert snapshot["autonomy_promotion_readiness"]["type"] == "autonomy_promotion_readiness"
    assert snapshot["autonomy_promotion_readiness"]["safety"]["live_widening_allowed"] is False
    assert snapshot["autonomy_promotion_readiness"]["summary"]["autonomous_ready_target_ids"] == []
    assert snapshot["autonomy_mode_transition_gate"]["type"] == "autonomy_mode_transition_gate"
    assert snapshot["autonomy_mode_transition_gate"]["safety"]["live_write"] is False
    assert snapshot["autonomy_mode_transition_gate"]["summary"]["autonomous_apply_allowed_modes"] == []
    assert snapshot["autonomy_mode_transition_gate"]["summary"][
        "supervised_current_live_write_kinds"
    ] == []
    assert snapshot["autonomy_mode_transition_gate"]["summary"][
        "supervised_current_live_write_surfaces"
    ] == []
    assert "auto" in snapshot["autonomy_mode_transition_gate"]["summary"]["blocked_modes"]
    assert "deessing" not in snapshot["project_goal_capability_matrix"]["summary"][
        "missing_capability_ids"
    ]
    assert "mode_full_autonomous" in snapshot["project_goal_capability_matrix"]["summary"][
        "blocked_live_capability_ids"
    ]
    assert (
        snapshot["project_goal_capability_progress_snapshot"]["type"]
        == "project_goal_capability_progress_snapshot"
    )
    assert (
        snapshot["project_goal_capability_progress_snapshot"]["safety"][
            "live_widening_allowed"
        ]
        is False
    )
    assert (
        snapshot["project_goal_capability_progress_snapshot"]["summary"][
            "row_count"
        ]
        == snapshot["project_goal_capability_matrix"]["summary"]["row_count"]
    )
    assert (
        snapshot["project_goal_capability_progress_snapshot"]["summary"][
            "missing_capability_ids"
        ]
        == snapshot["project_goal_capability_matrix"]["summary"][
            "missing_capability_ids"
        ]
    )
    assert (
        snapshot["project_goal_capability_progress_snapshot"]["summary"][
            "goal_area_counts"
        ]
    )
    assert (
        sum(
            snapshot["project_goal_capability_progress_snapshot"]["summary"][
                "goal_area_counts"
            ].values()
        )
        == snapshot["project_goal_capability_progress_snapshot"]["summary"][
            "row_count"
        ]
    )
    assert (
        snapshot["project_goal_capability_progress_snapshot"]["goal_area_progress"]
        and snapshot["project_goal_capability_progress_snapshot"]["next_steps"]
    )
    assert snapshot["methods"] == []
    assert {profile["id"] for profile in snapshot["pipeline_profiles"]} == {
        profile.id for profile in iter_pipeline_profiles()
    }


def test_read_surface_inventory_is_visible_but_not_live_authority():
    snapshot = build_read_surface_inventory_snapshot()
    rows = list(iter_read_surfaces())

    assert snapshot["type"] == "read_surface_inventory"
    assert rows
    assert snapshot["safety"]["read_only"] is True
    assert snapshot["safety"]["live_write"] is False
    assert snapshot["summary"]["surface_count"] == len(rows)
    assert "user_presets" in snapshot["summary"]["partial_or_missing_surface_ids"]
    assert get_read_surface("fx_slots_and_inserts").mutates_console_state is False


def test_registry_snapshot_includes_objective_alignment_projection():
    snapshot = build_processing_registry_snapshot(include_methods=False)
    objective_alignment = snapshot[
        "project_goal_capability_objective_alignment_snapshot"
    ]

    assert objective_alignment["type"] == "project_goal_capability_objective_alignment_snapshot"
    assert objective_alignment["safety"]["read_only"] is True
    assert objective_alignment["safety"]["live_write"] is False
    assert objective_alignment["summary"]["missing_row_count"] == 0
    assert objective_alignment["summary"]["required_row_count"] == objective_alignment[
        "summary"
    ]["present_row_count"]
    assert (
        objective_alignment["summary"]["required_row_count"]
        <= snapshot["project_goal_capability_matrix"]["summary"]["row_count"]
    )
    assert (
        snapshot["summary"]["project_goal_capability_objective_required_row_count"]
        == objective_alignment["summary"]["required_row_count"]
    )
    assert (
        snapshot["summary"]["project_goal_capability_objective_present_row_count"]
        == objective_alignment["summary"]["present_row_count"]
    )
    assert (
        snapshot["summary"]["project_goal_capability_objective_missing_row_count"]
        == objective_alignment["summary"]["missing_row_count"]
    )
    assert (
        snapshot["summary"]["project_goal_capability_objective_readiness_score"]
        == objective_alignment["summary"]["autonomy_readiness_score"]
    )
    assert (
        snapshot["summary"][
            "project_goal_capability_objective_readiness_ready_row_count"
        ]
        == objective_alignment["summary"]["autonomy_readiness_ready_row_count"]
    )
    assert (
        snapshot["summary"][
            "project_goal_capability_objective_readiness_gap_row_count"
        ]
        == objective_alignment["summary"]["autonomy_readiness_gap_row_count"]
    )
    assert (
        snapshot["summary"][
            "project_goal_capability_objective_readiness_goal_area_priorities"
        ]
        == objective_alignment["summary"]["autonomy_readiness_goal_area_priorities"]
    )
    assert (
        snapshot["summary"][
            "project_goal_capability_objective_readiness_by_goal_area"
        ]
        == processing_registry._normalize_goal_capability_readiness_rows(
            objective_alignment["summary"]["autonomy_readiness_by_goal_area"]
        )
    )
    assert (
        snapshot["summary"][
            "project_goal_capability_objective_readiness_hotspots"
        ]
        == processing_registry._normalize_goal_capability_readiness_hotspots(
            snapshot["project_goal_capability_progress_snapshot"]["summary"][
            "project_goal_objective_autonomy_readiness_hotspots"
        ]
        )
    )
    assert (
        snapshot["summary"][
            "project_goal_capability_objective_readiness_next_focus"
        ]
        == objective_alignment["summary"]["autonomy_readiness_next_focus"]
    )
    assert snapshot["summary"][
        "project_goal_capability_objective_readiness_next_focus"
    ]["capability_id"]
    assert (
        snapshot["summary"]["project_goal_capability_objective_blocked_live_row_count"]
        == len(objective_alignment["summary"]["blocked_live_rows"])
    )
    assert (
        snapshot["summary"]["project_goal_capability_objective_currently_live_writable_row_count"]
        == len(objective_alignment["summary"]["currently_live_writable_rows"])
    )
    assert (
        snapshot["summary"]["project_goal_capability_objective_full_autonomous_status"]
        == objective_alignment["summary"]["full_autonomous_status"]
    )
    assert objective_alignment["summary"]["blocked_live_rows"] == ["mode_full_autonomous"]
    assert objective_alignment["summary"]["currently_live_writable_rows"]

    matrix_rows = {
        row["id"]: row for row in snapshot["project_goal_capability_matrix"]["rows"]
    }
    for row in objective_alignment["rows"]:
        matrix_row = matrix_rows[row["id"]]
        assert matrix_row["id"] == row["id"]
        assert matrix_row["status"] == row["status"]
        assert matrix_row["goal_area"] == row["goal_area"]
        assert matrix_row["currently_live_writable"] == row["currently_live_writable"]


def test_processing_registry_snapshot_reflects_runtime_auto_unlock():
    snapshot = build_processing_registry_snapshot(
        include_methods=False,
        connection_mode="local",
        wing_boundary_active=False,
        auto_mode_unlocked=True,
    )
    auto_row = {row["mode"]: row for row in snapshot["autonomy_mode_transition_gate"]["rows"]}["auto"]

    assert auto_row["status"] == "blocked"
    assert auto_row["runtime_entry_allowed"] is False
    assert "auto_mode_requires_explicit_runtime_unlock" not in auto_row["blockers"]


def test_registry_snapshot_normalizes_goal_capability_readiness_payload():
    malformed_alignment_snapshot = {
        "type": "project_goal_capability_objective_alignment_snapshot",
        "summary": {
            "required_row_count": "4",
            "present_row_count": "3",
            "missing_row_count": "1",
            "autonomy_readiness_score": "88",
            "autonomy_readiness_ready_row_count": "2",
            "autonomy_readiness_gap_row_count": "1",
            "autonomy_readiness_goal_area_priorities": ("eq", "input_gain"),
            "autonomy_readiness_by_goal_area": {
                "eq": {
                    "goal_area": "eq",
                    "gap_row_count": "7",
                    "blocker_row_ids": ["cg_eq_1"],
                    "blocker_rows": ["cg_eq_2"],
                    "target_live_writable_row_count": "3",
                    "target_proof_gap_count": "2",
                    "ready_row_count": "1",
                    "required_row_count": "2",
                    "blocker_row_count": "4",
                    "readiness_score": "5",
                    "full_autonomous_status": "blocked_live",
                    "full_autonomous_status_label": "blocked",
                },
                "malformed": "value",
            },
            "blocked_live_rows": ["mode_full_autonomous"],
            "currently_live_writable_row_count": "9",
            "full_autonomous_status": "candidate",
        },
        "rows": [],
    }
    malformed_progress_snapshot = {
        "type": "project_goal_capability_progress_snapshot",
        "summary": {
            "row_count": 4,
            "project_goal_objective_autonomy_readiness_hotspots": [
                {"goal_area": "eq", "gap_row_count": "3", "blocker_row_ids": ["cg_eq_1"]},
                {"goal_area": "", "gap_row_count": "9"},
                "bad-entry",
            ],
        },
        "goal_area_progress": [],
        "next_steps": [],
        "rows": [],
    }

    with patch.object(processing_registry, "build_project_goal_objective_alignment_snapshot", return_value=malformed_alignment_snapshot), patch.object(processing_registry, "build_project_goal_capability_progress_snapshot", return_value=malformed_progress_snapshot):
        snapshot = build_processing_registry_snapshot(include_methods=False)

    assert (
        snapshot["summary"][
            "project_goal_capability_objective_readiness_goal_area_priorities"
        ]
        == ["eq", "input_gain"]
    )
    assert snapshot["summary"][
        "project_goal_capability_objective_readiness_by_goal_area"
    ] == {
        "eq": {
            "goal_area": "eq",
            "gap_row_count": 7,
            "blocker_row_ids": ["cg_eq_1"],
            "blocker_rows": ["cg_eq_2"],
            "target_live_writable_row_count": 3,
            "target_proof_gap_count": 2,
            "ready_row_count": 1,
            "required_row_count": 2,
            "blocker_row_count": 4,
            "readiness_score": 5,
            "full_autonomous_status": "blocked_live",
            "full_autonomous_status_label": "blocked",
        },
    }
    assert snapshot["summary"]["project_goal_capability_objective_readiness_hotspots"] == [
        {"goal_area": "eq", "gap_row_count": 3, "blocker_row_ids": ["cg_eq_1"]}
    ]


def test_write_surface_inventory_is_fail_closed_by_default():
    surfaces = list(iter_write_surfaces())
    ids = [surface.id for surface in surfaces]

    assert surfaces
    assert len(ids) == len(set(ids))

    live_surfaces = [surface for surface in surfaces if surface.allowed_live]
    assert [surface.id for surface in live_surfaces] == ["product_layer_fader_gain_apply"]

    for surface in surfaces:
        assert surface.status in WRITE_SURFACE_STATUSES
        assert surface.paths
        assert surface.entrypoints
        if surface.allowed_live:
            assert surface.status == "canonical_supervised"
            assert tuple(surface.write_kinds) == WRITE_SURFACE_LIVE_ALLOWED_WRITE_KINDS
            assert "apply_operator_proposal" in surface.gate
        else:
            assert surface.status in {"proposal_only", "lab_only", "quarantined", "blocked"}


def test_write_surface_inventory_covers_known_legacy_write_paths():
    by_path = {}
    for surface in iter_write_surfaces():
        for path in surface.paths:
            by_path.setdefault(path, set()).add(surface.id)

    expected_paths = {
        "backend/handlers/routing_handlers.py": "routing_snapshot_scene_writes",
        "backend/handlers/snapshot_handlers.py": "routing_snapshot_scene_writes",
        "backend/auto_test_routing.py": "routing_snapshot_scene_writes",
        "backend/handlers/eq_handlers.py": "eq_direct_apply_and_reset",
        "backend/system_measurement.py": "eq_direct_apply_and_reset",
        "backend/auto_soundcheck_engine.py": "eq_direct_apply_and_reset",
        "backend/autofoh_safety.py": "bus_dca_master_fader_direct_apply",
        "backend/auto_compressor.py": "compressor_direct_apply",
        "backend/auto_gate_caig.py": "gate_hpf_send_direct_apply",
        "backend/auto_panner.py": "pan_phase_delay_direct_apply",
        "backend/auto_panner_adaptive.py": "pan_phase_delay_direct_apply",
        "backend/auto_fader_v2/controller.py": "pan_phase_delay_direct_apply",
        "backend/auto_fx.py": "fx_insert_send_direct_apply",
        "backend/wing_readonly_preflight_contract.py": "proof_and_release_check_tools",
        "backend/voice_control.py": "voice_ai_direct_commands",
        "backend/ai/agent.py": "voice_ai_direct_commands",
        "backend/reset_all_channels.py": "manual_reset_restore_scripts",
        "frontend/src/services/websocket.js": "legacy_frontend_direct_ws_methods",
        "backend/routing.py": "legacy_ws_to_osc_router",
        "backend/ml/style_transfer.py": "ml_generated_osc_commands",
    }

    for path, surface_id in expected_paths.items():
        assert surface_id in by_path.get(path, set()), f"{path} missing from {surface_id}"


def test_write_surface_snapshot_status_counts_and_blocked_surfaces():
    snapshot = build_write_surface_inventory_snapshot()

    assert snapshot["type"] == "write_surface_inventory"
    assert snapshot["summary"]["live_apply_policy"] == "canonical_supervised_fader_gain_only"
    assert snapshot["summary"]["allowed_live_write_kinds"] == list(
        WRITE_SURFACE_LIVE_ALLOWED_WRITE_KINDS
    )
    assert snapshot["summary"]["allowed_live_surface_ids"] == ["product_layer_fader_gain_apply"]
    assert snapshot["summary"]["status_counts"]["blocked"] >= 3
    assert snapshot["summary"]["status_counts"]["quarantined"] >= 6

    blocked = [row for row in snapshot["surfaces"] if row["status"] == "blocked"]
    assert {row["id"] for row in blocked} >= {
        "manual_reset_restore_scripts",
        "legacy_frontend_direct_ws_methods",
        "legacy_ws_to_osc_router",
    }


def test_specific_write_surface_statuses_match_policy():
    assert get_write_surface("product_layer_fader_gain_apply").status == "canonical_supervised"
    assert get_write_surface("context_gate_bleed_recommendations").status == "proposal_only"
    gate_surface = get_write_surface("gate_hpf_send_direct_apply")
    assert gate_surface.status == "quarantined"
    assert gate_surface.allowed_live is False
    assert "CAIG _send_osc is fail-closed" in gate_surface.gate
    assert "AutoFaderV2/AutoFOH adaptive or typed gate actions return blocked evidence" in gate_surface.gate
    assert "AutoFOH gate/HPF/send actions are blocked" in gate_surface.current_policy
    pan_surface = get_write_surface("pan_phase_delay_direct_apply")
    assert pan_surface.status == "quarantined"
    assert pan_surface.allowed_live is False
    assert "set_channel_pan" in pan_surface.entrypoints
    assert "adaptive panner" in pan_surface.gate
    assert "pan separation are fail-closed" in pan_surface.gate
    assert "phase alignment" in pan_surface.gate
    assert "AutoFOH phase actions are fail-closed" in pan_surface.gate
    assert "phase/delay apply/reset paths return blocked evidence" in pan_surface.current_policy
    compressor_surface = get_write_surface("compressor_direct_apply")
    assert compressor_surface.status == "quarantined"
    assert compressor_surface.allowed_live is False
    assert "direct apply paths are fail-closed" in compressor_surface.gate
    assert "AutoMixerServer._reset_compressor_on_channel" in compressor_surface.entrypoints
    assert "AutoFOH compressor typed actions" in compressor_surface.current_policy
    assert "server compressor reset helper" in compressor_surface.current_policy
    assert "AutoSoundcheck legacy compressor-threshold writes" in compressor_surface.current_policy
    bus_surface = get_write_surface("bus_dca_master_fader_direct_apply")
    assert bus_surface.status == "quarantined"
    assert bus_surface.allowed_live is False
    assert "AutoFOH bus/DCA/master fader actions return blocked evidence" in bus_surface.gate
    assert "No bus, DCA, group, matrix, or master fader live writes" in bus_surface.current_policy
    eq_surface = get_write_surface("eq_direct_apply_and_reset")
    assert eq_surface.status == "quarantined"
    assert eq_surface.allowed_live is False
    assert "AutoEQ, AutoFaderV2 masking EQ, AutoFOH EQ actions, and AutoSoundcheck legacy EQ-enable writes are fail-closed" in eq_surface.gate
    assert "AutoSoundcheck EQ-enable writes return blocked evidence" in eq_surface.current_policy
    fx_surface = get_write_surface("fx_insert_send_direct_apply")
    assert fx_surface.status == "quarantined"
    assert fx_surface.allowed_live is False
    assert "AutoFX/AutoReverb/AutoFOH send apply paths are fail-closed" in fx_surface.gate
    assert "set_reverb_send" in fx_surface.entrypoints
    assert "set_reverb_params" in fx_surface.entrypoints
    assert get_write_surface("latency_benchmark_tools").status == "lab_only"
    assert get_write_surface("routing_snapshot_scene_writes").status == "quarantined"
    reset_surface = get_write_surface("manual_reset_restore_scripts")
    assert reset_surface.status == "blocked"
    assert "backend/auto_soundcheck_engine.py" in reset_surface.paths
    assert "AutoSoundcheck fader/mute restore emits blocked evidence only" in reset_surface.current_policy


def test_safety_coverage_matrix_records_live_proof_gaps():
    rows = list(iter_safety_coverage_rows())
    ids = [row.id for row in rows]

    assert rows
    assert len(ids) == len(set(ids))
    for row in rows:
        assert row.status in COVERAGE_STATUSES
        assert row.unit_tests in PROOF_STATES
        assert row.pre_read in PROOF_STATES
        if row.live_allowed:
            assert row.status == "hardware_proven_narrow"
            assert "apply_operator_proposal" in row.canonical_gate
        else:
            assert row.status in {
                "supervised_candidate",
                "dry_run_ready",
                "proposal_only",
                "quarantined",
                "blocked",
            }

    fader = get_safety_coverage_row("fader_supervised")
    assert fader.live_allowed is False
    assert fader.status == "supervised_candidate"
    assert fader.rollback_readback == "failed"
    assert fader.emergency_stop_proof == "missing"
    assert "rollback_not_applied" in fader.current_blocker
    assert fader.required_evidence_components == (
        "physical_pre_read",
        "bounded_fader_down_apply",
        "apply_readback",
        "rollback_readback",
        "emergency_stop_block",
        "operator_approval",
    )
    gain = get_safety_coverage_row("gain_supervised")
    assert gain.live_allowed is False
    assert gain.required_evidence_components == (
        "physical_pre_read",
        "bounded_gain_down_apply",
        "apply_readback",
        "rollback_readback",
        "emergency_stop_block",
        "unsafe_gain_up_followup_blocked",
        "gain_up_operator_authored_opt_in",
    )
    bus = get_safety_coverage_row("bus_group_matrix")
    mute = get_safety_coverage_row("mute_supervised")

    assert bus is not None
    assert bus.status == "quarantined"
    assert bus.live_allowed is False
    assert bus.action_type == "bus_group_matrix"
    assert bus.fake_mixer_proof == "missing"
    assert bus.real_wing_proof == "missing"
    assert "true-peak headroom" in bus.next_required_evidence
    assert "no_bus_dca_matrix_live_writes" in bus.notes

    assert mute is not None
    assert mute.status == "quarantined"
    assert mute.live_allowed is False
    assert mute.action_type == "mute"
    assert mute.pre_read == "missing"
    assert mute.rollback_readback == "missing"
    assert "manual_mute_authority_preserved" in mute.notes
    assert gain.real_wing_proof == "missing"
    assert get_safety_coverage_row("routing_snapshots_scenes").status == "blocked"


def test_safety_coverage_matrix_snapshot_is_fail_closed():
    snapshot = build_safety_coverage_matrix_snapshot()

    assert snapshot["type"] == "safety_coverage_matrix"
    assert (
        snapshot["summary"]["live_apply_policy"]
        == "supervised_fader_gain_policy_only_release_live_blocked"
    )
    assert snapshot["summary"]["live_allowed_row_ids"] == []
    assert "fader_supervised" in snapshot["summary"]["missing_or_partial_real_wing_proof_ids"]
    assert "gain_supervised" in snapshot["summary"]["missing_or_partial_real_wing_proof_ids"]
    assert snapshot["summary"]["status_counts"]["blocked"] >= 1
