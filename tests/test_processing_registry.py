from backend.processing_registry import (
    LIVE_ALLOWED_WRITE_KINDS,
    PROCESSING_DOMAINS,
    PROCESSING_STATUSES,
    build_processing_registry_snapshot,
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

    assert [method.id for method in live_methods] == ["supervised_manual_fader_gain"]
    for method in live_methods:
        assert method.requires_supervision is True
        assert tuple(method.live_write_kinds) == LIVE_ALLOWED_WRITE_KINDS
        assert method.status == "canonical"


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
    assert "context_aware_gate_agent" in main.proposal_method_ids


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
    assert snapshot["safety_coverage_matrix"]["summary"]["live_allowed_row_ids"] == [
        "fader_supervised"
    ]
    assert snapshot["read_surface_inventory"]["summary"]["claim"] == (
        "read coverage is inventoried but not complete"
    )
    assert snapshot["read_surface_inventory"]["safety"]["live_write"] is False
    assert snapshot["read_surface_remediation_plan"]["type"] == "read_surface_remediation_plan"
    assert snapshot["read_surface_remediation_plan"]["safety"]["live_write"] is False
    assert snapshot["read_surface_remediation_plan"]["summary"]["incomplete_surface_count"] >= 1
    assert snapshot["project_goal_capability_matrix"]["type"] == "project_goal_capability_matrix"
    assert snapshot["project_goal_capability_matrix"]["safety"][
        "live_widening_allowed"
    ] is False
    assert snapshot["autonomy_promotion_readiness"]["type"] == "autonomy_promotion_readiness"
    assert snapshot["autonomy_promotion_readiness"]["safety"]["live_widening_allowed"] is False
    assert snapshot["autonomy_promotion_readiness"]["summary"]["autonomous_ready_target_ids"] == []
    assert "deessing" not in snapshot["project_goal_capability_matrix"]["summary"][
        "missing_capability_ids"
    ]
    assert "mode_full_autonomous" in snapshot["project_goal_capability_matrix"]["summary"][
        "blocked_live_capability_ids"
    ]
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
            assert tuple(surface.write_kinds) == LIVE_ALLOWED_WRITE_KINDS
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
        "backend/auto_compressor.py": "compressor_direct_apply",
        "backend/auto_gate_caig.py": "gate_hpf_send_direct_apply",
        "backend/auto_panner_adaptive.py": "pan_phase_delay_direct_apply",
        "backend/auto_fx.py": "fx_insert_send_direct_apply",
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
    assert snapshot["summary"]["allowed_live_write_kinds"] == list(LIVE_ALLOWED_WRITE_KINDS)
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
    assert get_write_surface("latency_benchmark_tools").status == "lab_only"
    assert get_write_surface("routing_snapshot_scene_writes").status == "quarantined"
    assert get_write_surface("manual_reset_restore_scripts").status == "blocked"


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
            assert row.status in {"dry_run_ready", "proposal_only", "quarantined", "blocked"}

    assert get_safety_coverage_row("fader_supervised").live_allowed is True
    assert get_safety_coverage_row("gain_supervised").live_allowed is False
    assert get_safety_coverage_row("gain_supervised").real_wing_proof == "missing"
    assert get_safety_coverage_row("routing_snapshots_scenes").status == "blocked"


def test_safety_coverage_matrix_snapshot_is_fail_closed():
    snapshot = build_safety_coverage_matrix_snapshot()

    assert snapshot["type"] == "safety_coverage_matrix"
    assert snapshot["summary"]["live_apply_policy"] == "supervised_fader_gain_only"
    assert snapshot["summary"]["live_allowed_row_ids"] == ["fader_supervised"]
    assert "gain_supervised" in snapshot["summary"]["missing_or_partial_real_wing_proof_ids"]
    assert snapshot["summary"]["status_counts"]["blocked"] >= 1
