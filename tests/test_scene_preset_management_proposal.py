from pathlib import Path

from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_pipeline_profile, get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.scene_preset_management_proposal import (
    ScenePresetManagementAnalysisInput,
    analyze_scene_preset_management_need,
    build_scene_preset_management_proposals,
    normalize_artifact_type,
    normalize_scene_preset_action,
    normalize_target_scope,
)


def test_scene_preset_management_module_is_transport_free():
    source = Path("backend/scene_preset_management_proposal.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "mixer_client",
        "load_snap",
        "save_snap",
        "load_snap_by_index",
        "recall_scene",
        "find_snap_by_name",
        "/snap/store",
        "/$ctl/lib/$action",
    ]
    for token in forbidden:
        assert token not in source


def test_soundcheck_scoped_scene_recall_creates_proposal_only_recommendation():
    recommendation = analyze_scene_preset_management_need(
        ScenePresetManagementAnalysisInput(
            artifact_id="scene:12",
            artifact_name="Band A vocal strip",
            artifact_type="scene",
            action_type="recall",
            target_scope="channel_strip",
            target_label="Lead Vox scene recall",
            active_scene_id="scene:pre_show",
            active_scene_name="Pre-show baseline",
            artifact_exists=True,
            target_name_match=True,
            current_state_readback_available=True,
            target_preview_available=True,
            scoped_diff_available=True,
            rollback_snapshot_available=True,
            rollback_snapshot_id="rollback:pre_scene_recall",
            operator_intent="operator_scene_request",
            change_window="soundcheck",
            active_performance=False,
            affected_channel_count=1,
            affected_bus_count=0,
            affected_fx_slot_count=0,
            estimated_changed_parameter_count=12,
            destructive_scope_score=0.08,
            current_mix_signal_dbfs=-82.0,
            affected_sources_active=False,
            target_confidence=0.92,
            scope_confidence=0.9,
            rollback_confidence=0.88,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "scene_recall"
    assert recommendation.artifact_type == "scene"
    assert recommendation.target_scope == "channel_strip"
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["destructive_console_state"] is True

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "scene_preset_management_proposal"
    assert proposal["kind"] == "scene_preset"
    assert proposal["can_import_to_queue"] is False
    assert proposal["can_apply_live"] is False
    assert "requested_change" not in proposal
    assert "requested_changes" not in proposal
    assert "unsupported_value_type:scene_preset" in proposal["blockers"]
    assert proposal["proposed_action"]["action_type"] == "scene_recall"
    assert proposal["evidence"]["live_write"] is False


def test_active_performance_and_broad_scope_block_scene_preset_proposal():
    recommendation = analyze_scene_preset_management_need(
        ScenePresetManagementAnalysisInput(
            artifact_id="scene:99",
            artifact_name="Full show reset",
            artifact_type="snapshot",
            action_type="recall",
            target_scope="scoped_scene",
            artifact_exists=True,
            target_name_match=True,
            current_state_readback_available=True,
            target_preview_available=True,
            scoped_diff_available=True,
            rollback_snapshot_available=True,
            rollback_snapshot_id="rollback:show",
            operator_intent="operator_scene_request",
            change_window="show",
            active_performance=True,
            affected_channel_count=16,
            affected_bus_count=4,
            affected_fx_slot_count=3,
            estimated_changed_parameter_count=180,
            destructive_scope_score=0.82,
            current_mix_signal_dbfs=-18.0,
            affected_sources_active=True,
            target_confidence=0.9,
            scope_confidence=0.9,
            rollback_confidence=0.9,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "active_performance_scene_preset_blocked" in recommendation.blockers
    assert "scene_preset_scope_too_broad" in recommendation.blockers
    assert "active_signal_requires_manual_scene_preset_change" in recommendation.blockers
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_missing_preview_or_rollback_blocks_scene_preset_proposal():
    recommendation = analyze_scene_preset_management_need(
        ScenePresetManagementAnalysisInput(
            artifact_id="preset:vocal_a",
            artifact_name="Vocal A",
            artifact_type="user_preset",
            action_type="recall",
            target_scope="channel_strip",
            artifact_exists=True,
            target_name_match=True,
            current_state_readback_available=True,
            target_preview_available=False,
            scoped_diff_available=False,
            rollback_snapshot_available=False,
            operator_intent="operator_preset_request",
            affected_channel_count=1,
            estimated_changed_parameter_count=8,
            destructive_scope_score=0.08,
            current_mix_signal_dbfs=-90.0,
            target_confidence=0.9,
            scope_confidence=0.9,
            rollback_confidence=0.9,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "scene_preset_preview_missing" in recommendation.blockers
    assert "scene_preset_scoped_diff_missing" in recommendation.blockers
    assert "rollback_snapshot_missing" in recommendation.blockers


def test_preset_save_requires_explicit_overwrite_permission():
    recommendation = analyze_scene_preset_management_need(
        ScenePresetManagementAnalysisInput(
            artifact_id="preset:lead_vocal_slot",
            artifact_name="Lead Vocal Slot",
            artifact_type="user_preset",
            action_type="save",
            target_scope="preset_slot",
            current_state_readback_available=True,
            target_preview_available=True,
            scoped_diff_available=True,
            rollback_snapshot_available=True,
            rollback_snapshot_id="rollback:before_preset_save",
            operator_intent="operator_preset_save_request",
            change_window="soundcheck",
            target_slot_occupied=True,
            allow_overwrite=False,
            affected_channel_count=1,
            estimated_changed_parameter_count=10,
            destructive_scope_score=0.12,
            current_mix_signal_dbfs=-88.0,
            target_confidence=0.9,
            scope_confidence=0.9,
            rollback_confidence=0.9,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "overwrite_requires_explicit_operator_intent" in recommendation.blockers

    allowed = analyze_scene_preset_management_need(
        ScenePresetManagementAnalysisInput(
            artifact_id="preset:lead_vocal_slot",
            artifact_name="Lead Vocal Slot",
            artifact_type="user_preset",
            action_type="save",
            target_scope="preset_slot",
            current_state_readback_available=True,
            target_preview_available=True,
            scoped_diff_available=True,
            rollback_snapshot_available=True,
            rollback_snapshot_id="rollback:before_preset_save",
            operator_intent="operator_preset_save_request",
            change_window="soundcheck",
            target_slot_occupied=True,
            allow_overwrite=True,
            affected_channel_count=1,
            estimated_changed_parameter_count=10,
            destructive_scope_score=0.12,
            current_mix_signal_dbfs=-88.0,
            target_confidence=0.9,
            scope_confidence=0.9,
            rollback_confidence=0.9,
            persistence_cycles=3,
        )
    )

    assert allowed.status == "proposal_only"
    assert allowed.action_type == "preset_save"


def test_unchanged_scene_recall_is_analysis_only():
    recommendation = analyze_scene_preset_management_need(
        ScenePresetManagementAnalysisInput(
            artifact_id="scene:12",
            artifact_name="Current scene",
            artifact_type="scene",
            action_type="recall",
            active_scene_id="scene:12",
            artifact_exists=True,
            target_name_match=True,
            current_state_readback_available=True,
            target_preview_available=True,
            scoped_diff_available=True,
            rollback_snapshot_available=True,
            rollback_snapshot_id="rollback:current",
            operator_intent="operator_scene_request",
            estimated_changed_parameter_count=0,
            current_mix_signal_dbfs=-88.0,
            target_confidence=0.9,
            scope_confidence=0.9,
            rollback_confidence=0.9,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "scene_change_not_needed" in recommendation.blockers
    assert "scene_preset_change_not_needed" in recommendation.blockers


def test_scene_preset_helpers_normalize_action_and_scope():
    assert normalize_artifact_type("snap") == "snapshot"
    assert normalize_artifact_type("fx") == "fx_preset"
    assert normalize_scene_preset_action("recall", "scene") == "scene_recall"
    assert normalize_scene_preset_action("save", "user_preset") == "preset_save"
    assert normalize_target_scope("strip") == "channel_strip"
    assert normalize_target_scope("preset") == "preset_slot"


def test_build_scene_preset_management_proposals_filters_to_proposal_only_items():
    proposals = build_scene_preset_management_proposals(
        [
            ScenePresetManagementAnalysisInput(
                artifact_id="scene:12",
                artifact_name="Vocal recall",
                artifact_type="scene",
                action_type="recall",
                target_scope="channel_strip",
                artifact_exists=True,
                target_name_match=True,
                current_state_readback_available=True,
                target_preview_available=True,
                scoped_diff_available=True,
                rollback_snapshot_available=True,
                rollback_snapshot_id="rollback:scene_12",
                operator_intent="operator_scene_request",
                change_window="soundcheck",
                affected_channel_count=1,
                estimated_changed_parameter_count=6,
                current_mix_signal_dbfs=-90.0,
                target_confidence=0.9,
                scope_confidence=0.9,
                rollback_confidence=0.9,
                persistence_cycles=3,
            ),
            ScenePresetManagementAnalysisInput(
                artifact_id="scene:13",
                artifact_name="Missing preview",
                artifact_type="scene",
                action_type="recall",
                artifact_exists=True,
                target_name_match=True,
                current_state_readback_available=True,
                target_preview_available=False,
                scoped_diff_available=False,
                rollback_snapshot_available=True,
                rollback_snapshot_id="rollback:scene_13",
                operator_intent="operator_scene_request",
                estimated_changed_parameter_count=6,
                current_mix_signal_dbfs=-90.0,
                target_confidence=0.9,
                scope_confidence=0.9,
                rollback_confidence=0.9,
            ),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["target"] == "scene:12"


def test_scene_preset_management_proposal_is_registered_as_goal_capability():
    method = get_processing_method("scene_preset_management_proposal")
    direct_write = get_processing_method("routing_snapshot_reset_writes")
    main = get_pipeline_profile("main_live_assist")
    capability = get_project_goal_capability("console_scene_preset_control")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.domain == "scene_preset"
    assert method.can_write_live is False
    assert direct_write is not None
    assert direct_write.status == "quarantined"
    assert main is not None
    assert "scene_preset_management_proposal" in main.proposal_method_ids
    assert capability is not None
    assert capability.status == "proposal_only"
    assert capability.autonomy_mode_ceiling == "assisted"
    assert "scene_preset_management_proposal" in capability.method_ids
    assert contracts["scene_preset_management_proposal"]["proposal_contract"]["value_type"] == "scene_preset"
