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
    scene_preset_cache_freshness_from_snapshots,
    scene_preset_rollback_payload,
    scene_preset_scoped_diff_payload,
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
            current_state_cache_freshness={
                "status": "fresh",
                "field_count": 7,
                "fresh_field_count": 7,
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "scene_recall"
    assert recommendation.artifact_type == "scene"
    assert recommendation.target_scope == "channel_strip"
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["destructive_console_state"] is True
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_state_cache_freshness"]["fresh_field_count"] == 7

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
    assert proposal["evidence"]["current_state_cache_freshness_status"] == "fresh"


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


def test_read_only_scoped_diff_evidence_can_drive_scene_preset_proposal():
    diff = {
        "id": "diff:scene_12:vocal_strip",
        "source": "product_layer_cached_scene_diff",
        "available": True,
        "target_preview_available": True,
        "rollback_snapshot_available": True,
        "rollback_snapshot_id": "rollback:pre_scene_recall",
        "estimated_changed_parameter_count": 12,
        "affected_channel_count": 1,
        "affected_bus_count": 0,
        "affected_fx_slot_count": 0,
        "destructive_scope_score": 0.08,
        "scope_confidence": 0.91,
        "rollback_confidence": 0.89,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
        "mutates_console_state": False,
        "destructive_change_marker": "scene_recall_would_change_channel_strip",
    }

    normalized = scene_preset_scoped_diff_payload(diff)
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
            operator_intent="operator_scene_request",
            change_window="soundcheck",
            active_performance=False,
            current_mix_signal_dbfs=-82.0,
            affected_sources_active=False,
            target_confidence=0.92,
            persistence_cycles=3,
            current_state_cache_freshness={"status": "fresh"},
            scoped_diff_evidence=diff,
        )
    )

    assert normalized["available"] is True
    assert normalized["rollback_snapshot_id"] == "rollback:pre_scene_recall"
    assert recommendation.status == "proposal_only"
    assert recommendation.blockers == ()
    assert recommendation.evidence["target_preview_available"] is True
    assert recommendation.evidence["scoped_diff_available"] is True
    assert recommendation.evidence["rollback_snapshot_available"] is True
    assert recommendation.evidence["rollback_snapshot_id"] == "rollback:pre_scene_recall"
    assert recommendation.evidence["estimated_changed_parameter_count"] == 12
    assert recommendation.evidence["affected_channel_count"] == 1
    assert recommendation.evidence["scoped_diff_evidence"]["diff_id"] == "diff:scene_12:vocal_strip"
    assert recommendation.evidence["scoped_diff_evidence"]["read_only"] is True


def test_scoped_diff_evidence_with_live_or_query_claim_blocks_proposal():
    recommendation = analyze_scene_preset_management_need(
        ScenePresetManagementAnalysisInput(
            artifact_id="scene:12",
            artifact_name="Band A vocal strip",
            artifact_type="scene",
            action_type="recall",
            target_scope="channel_strip",
            artifact_exists=True,
            target_name_match=True,
            current_state_readback_available=True,
            operator_intent="operator_scene_request",
            change_window="soundcheck",
            active_performance=False,
            current_mix_signal_dbfs=-82.0,
            target_confidence=0.92,
            persistence_cycles=3,
            current_state_cache_freshness={"status": "fresh"},
            scoped_diff_evidence={
                "available": True,
                "target_preview_available": True,
                "rollback_snapshot_id": "rollback:pre_scene_recall",
                "estimated_changed_parameter_count": 12,
                "affected_channel_count": 1,
                "destructive_scope_score": 0.08,
                "scope_confidence": 0.91,
                "rollback_confidence": 0.89,
                "read_only": False,
                "live_write": True,
                "queries_console": True,
                "mutates_console_state": True,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "scene_preset_scoped_diff_not_read_only" in recommendation.blockers
    assert "scene_preset_scoped_diff_live_write_claim" in recommendation.blockers
    assert "scene_preset_scoped_diff_queries_console" in recommendation.blockers
    assert "scene_preset_scoped_diff_mutates_console_state" in recommendation.blockers
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_scoped_diff_evidence_broad_scope_blocks_scene_preset_proposal():
    recommendation = analyze_scene_preset_management_need(
        ScenePresetManagementAnalysisInput(
            artifact_id="scene:99",
            artifact_name="Broad scene",
            artifact_type="scene",
            action_type="recall",
            target_scope="scoped_scene",
            artifact_exists=True,
            target_name_match=True,
            current_state_readback_available=True,
            operator_intent="operator_scene_request",
            change_window="soundcheck",
            active_performance=False,
            current_mix_signal_dbfs=-82.0,
            target_confidence=0.92,
            persistence_cycles=3,
            current_state_cache_freshness={"status": "fresh"},
            scoped_diff_evidence={
                "available": True,
                "target_preview_available": True,
                "rollback_snapshot_id": "rollback:broad_scene",
                "estimated_changed_parameter_count": 80,
                "affected_channel_count": 5,
                "affected_bus_count": 2,
                "affected_fx_slot_count": 2,
                "destructive_scope_score": 0.62,
                "scope_confidence": 0.91,
                "rollback_confidence": 0.89,
                "read_only": True,
                "live_write": False,
                "queries_console": False,
                "mutates_console_state": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.severity == "high"
    assert "scene_preset_scope_too_broad" in recommendation.blockers
    assert recommendation.evidence["affected_channel_count"] == 5
    assert recommendation.evidence["affected_bus_count"] == 2
    assert recommendation.evidence["affected_fx_slot_count"] == 2
    assert recommendation.evidence["destructive_scope_score"] == 0.62


def test_read_only_rollback_evidence_can_drive_scene_preset_proposal():
    rollback = {
        "id": "rollback-evidence:scene_12",
        "source": "product_layer_cached_rollback_marker",
        "available": True,
        "rollback_snapshot_id": "rollback:pre_scene_recall",
        "rollback_confidence": 0.91,
        "target_scope_matches": True,
        "target_scope": "channel_strip",
        "read_only": True,
        "live_write": False,
        "queries_console": False,
        "mutates_console_state": False,
    }

    normalized = scene_preset_rollback_payload(rollback)
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
            operator_intent="operator_scene_request",
            change_window="soundcheck",
            active_performance=False,
            affected_channel_count=1,
            estimated_changed_parameter_count=12,
            destructive_scope_score=0.08,
            current_mix_signal_dbfs=-82.0,
            affected_sources_active=False,
            target_confidence=0.92,
            scope_confidence=0.9,
            persistence_cycles=3,
            current_state_cache_freshness={"status": "fresh"},
            rollback_evidence=rollback,
        )
    )

    assert normalized["available"] is True
    assert normalized["rollback_snapshot_id"] == "rollback:pre_scene_recall"
    assert recommendation.status == "proposal_only"
    assert recommendation.blockers == ()
    assert recommendation.evidence["rollback_snapshot_available"] is True
    assert recommendation.evidence["rollback_snapshot_id"] == "rollback:pre_scene_recall"
    assert recommendation.evidence["rollback_confidence"] == 0.91
    assert recommendation.evidence["rollback_evidence"]["rollback_evidence_id"] == "rollback-evidence:scene_12"
    assert recommendation.evidence["rollback_evidence"]["read_only"] is True


def test_rollback_evidence_with_live_or_query_claim_blocks_proposal():
    recommendation = analyze_scene_preset_management_need(
        ScenePresetManagementAnalysisInput(
            artifact_id="scene:12",
            artifact_name="Band A vocal strip",
            artifact_type="scene",
            action_type="recall",
            target_scope="channel_strip",
            artifact_exists=True,
            target_name_match=True,
            current_state_readback_available=True,
            target_preview_available=True,
            scoped_diff_available=True,
            operator_intent="operator_scene_request",
            change_window="soundcheck",
            active_performance=False,
            affected_channel_count=1,
            estimated_changed_parameter_count=12,
            destructive_scope_score=0.08,
            current_mix_signal_dbfs=-82.0,
            target_confidence=0.92,
            scope_confidence=0.9,
            persistence_cycles=3,
            current_state_cache_freshness={"status": "fresh"},
            rollback_evidence={
                "available": True,
                "rollback_snapshot_id": "rollback:pre_scene_recall",
                "rollback_confidence": 0.91,
                "target_scope_matches": True,
                "read_only": False,
                "live_write": True,
                "queries_console": True,
                "mutates_console_state": True,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "scene_preset_rollback_not_read_only" in recommendation.blockers
    assert "scene_preset_rollback_live_write_claim" in recommendation.blockers
    assert "scene_preset_rollback_queries_console" in recommendation.blockers
    assert "scene_preset_rollback_mutates_console_state" in recommendation.blockers
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_rollback_evidence_target_scope_mismatch_blocks_proposal():
    recommendation = analyze_scene_preset_management_need(
        ScenePresetManagementAnalysisInput(
            artifact_id="scene:12",
            artifact_name="Band A vocal strip",
            artifact_type="scene",
            action_type="recall",
            target_scope="channel_strip",
            artifact_exists=True,
            target_name_match=True,
            current_state_readback_available=True,
            target_preview_available=True,
            scoped_diff_available=True,
            operator_intent="operator_scene_request",
            change_window="soundcheck",
            active_performance=False,
            affected_channel_count=1,
            estimated_changed_parameter_count=12,
            destructive_scope_score=0.08,
            current_mix_signal_dbfs=-82.0,
            target_confidence=0.92,
            scope_confidence=0.9,
            persistence_cycles=3,
            current_state_cache_freshness={"status": "fresh"},
            rollback_evidence={
                "available": True,
                "rollback_snapshot_id": "rollback:wrong_scope",
                "rollback_confidence": 0.91,
                "target_scope_matches": False,
                "target_scope": "scoped_scene",
                "read_only": True,
                "live_write": False,
                "queries_console": False,
                "mutates_console_state": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "scene_preset_rollback_target_scope_mismatch" in recommendation.blockers
    assert recommendation.evidence["rollback_evidence"]["target_scope_matches"] is False


def test_stale_current_scene_preset_state_blocks_proposal_only_recommendation():
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
            estimated_changed_parameter_count=12,
            destructive_scope_score=0.08,
            current_mix_signal_dbfs=-82.0,
            affected_sources_active=False,
            target_confidence=0.92,
            scope_confidence=0.9,
            rollback_confidence=0.88,
            persistence_cycles=3,
            current_state_cache_freshness={
                "status": "stale",
                "field_count": 7,
                "fresh_field_count": 5,
                "stale_field_count": 2,
                "stale_fields": ["active_scene", "rollback_snapshot"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "scene_preset_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "stale"
    assert recommendation.evidence["current_state_cache_freshness"]["stale_fields"] == [
        "active_scene",
        "rollback_snapshot",
    ]
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_scene_preset_snapshot_freshness_bridge_uses_fresh_scene_snapshot():
    snapshot = {
        "type": "snapshots_scenes_snapshot",
        "source": "runtime_cached_state",
        "safety": {
            "read_only": True,
            "live_write": False,
            "queries_console": False,
        },
        "summary": {
            "read_coverage_status": "partial",
            "cache_freshness_status": "fresh",
            "live_apply_allowed": False,
        },
        "cache_freshness": {
            "status": "fresh",
            "field_count": 12,
            "fresh_field_count": 12,
            "read_only": True,
            "live_write": False,
            "queries_console": False,
        },
    }

    bridged = scene_preset_cache_freshness_from_snapshots(
        "scene",
        current_snapshots_scenes_snapshot=snapshot,
    )
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
            estimated_changed_parameter_count=12,
            destructive_scope_score=0.08,
            current_mix_signal_dbfs=-82.0,
            affected_sources_active=False,
            target_confidence=0.92,
            scope_confidence=0.9,
            rollback_confidence=0.88,
            persistence_cycles=3,
            current_snapshots_scenes_snapshot=snapshot,
        )
    )

    assert bridged is not None
    assert bridged["status"] == "fresh"
    assert bridged["snapshot_type"] == "snapshots_scenes_snapshot"
    assert recommendation.status == "proposal_only"
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert (
        recommendation.evidence["current_state_cache_freshness"]["snapshot_type"]
        == "snapshots_scenes_snapshot"
    )
    assert recommendation.evidence["current_state_cache_freshness"]["snapshot_queries_console"] is False


def test_scene_preset_snapshot_freshness_bridge_blocks_stale_shared_scene_snapshot():
    snapshot = {
        "type": "snapshots_scenes_snapshot",
        "source": "runtime_cached_state",
        "safety": {
            "read_only": True,
            "live_write": False,
            "queries_console": False,
        },
        "summary": {
            "read_coverage_status": "partial",
            "cache_freshness_status": "stale",
            "live_apply_allowed": False,
        },
        "cache_freshness": {
            "status": "stale",
            "field_count": 12,
            "fresh_field_count": 8,
            "stale_field_count": 4,
            "stale_fields": ["active_scene", "current_show"],
            "read_only": True,
            "live_write": False,
            "queries_console": False,
        },
    }
    analysis = ScenePresetManagementAnalysisInput(
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
        estimated_changed_parameter_count=12,
        destructive_scope_score=0.08,
        current_mix_signal_dbfs=-82.0,
        affected_sources_active=False,
        target_confidence=0.92,
        scope_confidence=0.9,
        rollback_confidence=0.88,
        persistence_cycles=3,
        current_snapshots_scenes_snapshot=snapshot,
    )

    recommendation = analyze_scene_preset_management_need(analysis)
    proposals = build_scene_preset_management_proposals(
        [
            ScenePresetManagementAnalysisInput(
                artifact_id="scene:12",
                artifact_name="Band A vocal strip",
                artifact_type="scene",
                action_type="recall",
                target_scope="channel_strip",
                artifact_exists=True,
                target_name_match=True,
                current_state_readback_available=True,
                target_preview_available=True,
                scoped_diff_available=True,
                rollback_snapshot_available=True,
                rollback_snapshot_id="rollback:pre_scene_recall",
                operator_intent="operator_scene_request",
                change_window="soundcheck",
                affected_channel_count=1,
                estimated_changed_parameter_count=12,
                current_mix_signal_dbfs=-82.0,
                target_confidence=0.92,
                scope_confidence=0.9,
                rollback_confidence=0.88,
                persistence_cycles=3,
            ),
        ],
        current_snapshots_scenes_snapshot=snapshot,
    )

    assert recommendation.status == "analysis_only"
    assert "scene_preset_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "stale"
    assert recommendation.evidence["current_state_cache_freshness"]["stale_fields"] == [
        "active_scene",
        "current_show",
    ]
    assert proposals == ()


def test_scene_preset_snapshot_freshness_bridge_blocks_stale_user_preset_snapshot():
    snapshot = {
        "type": "user_presets_snapshot",
        "source": "runtime_cached_state",
        "safety": {
            "read_only": True,
            "live_write": False,
            "queries_console": False,
        },
        "summary": {
            "read_coverage_status": "partial",
            "cache_freshness_status": "stale",
            "live_apply_allowed": False,
        },
        "cache_freshness": {
            "status": "stale",
            "field_count": 4,
            "fresh_field_count": 2,
            "stale_field_count": 2,
            "stale_fields": ["preset_name"],
            "read_only": True,
            "live_write": False,
            "queries_console": False,
        },
    }

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
            target_preview_available=True,
            scoped_diff_available=True,
            rollback_snapshot_available=True,
            rollback_snapshot_id="rollback:preset_vocal_a",
            operator_intent="operator_preset_request",
            change_window="soundcheck",
            affected_channel_count=1,
            estimated_changed_parameter_count=8,
            destructive_scope_score=0.08,
            current_mix_signal_dbfs=-90.0,
            target_confidence=0.9,
            scope_confidence=0.9,
            rollback_confidence=0.9,
            persistence_cycles=3,
            current_user_presets_snapshot=snapshot,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "scene_preset_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "stale"
    assert recommendation.evidence["current_state_cache_freshness"]["snapshot_type"] == "user_presets_snapshot"


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


def test_build_scene_preset_management_proposals_filters_stale_state_items():
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
                current_state_cache_freshness={"status": "stale"},
            ),
        ]
    )

    assert proposals == ()


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
    assert "scoped_diff_evidence_contract" in method.notes
    assert "rollback_evidence_contract" in method.notes
    assert "product_snapshot_freshness_bridge" in method.notes
    assert "snapshots_scenes_snapshot_bridge" in method.notes
    assert "user_presets_snapshot_bridge" in method.notes
    assert direct_write is not None
    assert direct_write.status == "quarantined"
    assert main is not None
    assert "scene_preset_management_proposal" in main.proposal_method_ids
    assert capability is not None
    assert capability.status == "proposal_only"
    assert capability.autonomy_mode_ceiling == "assisted"
    assert "scene_preset_management_proposal" in capability.method_ids
    assert contracts["scene_preset_management_proposal"]["proposal_contract"]["value_type"] == "scene_preset"
    assert (
        "current_scene_preset_cache_freshness"
        in contracts["scene_preset_management_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "snapshots_scenes_snapshot_cache_freshness"
        in contracts["scene_preset_management_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "user_presets_snapshot_cache_freshness"
        in contracts["scene_preset_management_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "read_only_scoped_diff_evidence"
        in contracts["scene_preset_management_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "read_only_rollback_evidence"
        in contracts["scene_preset_management_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
