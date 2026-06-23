from pathlib import Path

from backend.mute_management_proposal import (
    MuteManagementAnalysisInput,
    analyze_mute_management_need,
    build_mute_management_proposals,
    classify_source_family,
)
from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_pipeline_profile, get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability


def _fresh_state(field_count=5):
    return {
        "status": "fresh",
        "field_count": field_count,
        "fresh_field_count": field_count,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def _assert_top_level_denial(payload):
    assert payload["queue_action"] == "not_queueable_until_promoted"
    assert payload["live_write"] is False
    assert payload["queue_mutation"] is False
    assert payload["can_import_to_queue"] is False
    assert payload["can_apply_live"] is False
    assert payload["queue_import_allowed"] is False
    assert payload["live_apply_allowed"] is False
    assert payload["read_only_or_dry_run_only"] is True
    assert payload["live_authority_granted"] is False
    assert payload["readiness_closure_allowed"] is False


def test_mute_management_module_is_transport_free():
    source = Path("backend/mute_management_proposal.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "set_mute",
        "set_channel_mute",
        "mute_channel(",
        "unmute_channel(",
        "mixer_client",
    ]
    for token in forbidden:
        assert token not in source


def test_idle_close_mic_creates_proposal_only_mute_recommendation():
    recommendation = analyze_mute_management_need(
        MuteManagementAnalysisInput(
            channel_id=12,
            channel_label="Floor Tom",
            source_role="floor_tom",
            current_muted=False,
            allow_auto_mute_candidate=True,
            rms_dbfs=-67.0,
            peak_dbfs=-55.0,
            noise_floor_dbfs=-66.0,
            idle_duration_sec=48.0,
            source_confidence=0.9,
            bleed_risk=0.25,
            feedback_risk=0.05,
            dropout_risk=0.1,
            expected_next_cue_sec=40.0,
            persistence_cycles=3,
            current_state_cache_freshness=_fresh_state(),
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "idle_mute_proposal"
    assert recommendation.source_family == "tom"
    assert recommendation.current_muted is False
    assert recommendation.proposed_muted is True
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["manual_mute_authority_preserved"] is True
    assert recommendation.evidence["current_mute_state"] is False
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_mute_cache_freshness"]["status"] == "fresh"
    assert recommendation.evidence["current_mute_cache_no_live_authority"] is True
    assert recommendation.evidence["signal_activity"]["idle_signal"] is True
    assert recommendation.evidence["signal_activity"]["active_signal"] is False
    assert recommendation.evidence["rollback_value"] is False
    assert recommendation.evidence["rollback_mute_readback"] == {
        "required_before_live_promotion": True,
        "previous_mute_state": False,
        "read_only_or_dry_run_only": True,
        "live_write": False,
    }
    assert recommendation.evidence["current_state_cache_freshness"]["fresh_field_count"] == 5
    assert recommendation.evidence["source_identity_downstream_trust"]["status"] == "legacy_unverified"

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "mute_management_proposal"
    assert proposal["kind"] == "mute_state"
    _assert_top_level_denial(proposal)
    assert "requested_change" not in proposal
    assert "requested_changes" not in proposal
    assert "unsupported_value_type:mute_state" in proposal["blockers"]
    assert proposal["proposed_action"]["proposed_muted"] is True
    assert proposal["proposed_action"]["current_mute_state"] is False
    assert proposal["proposed_action"]["signal_activity"] == proposal["evidence"]["signal_activity"]
    assert proposal["proposed_action"]["rollback_value"] is False
    assert proposal["proposed_action"]["rollback_mute_readback"] == proposal["evidence"][
        "rollback_mute_readback"
    ]
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["current_state_cache_freshness_status"] == "fresh"
    assert proposal["evidence"]["current_mute_cache_freshness"]["status"] == "fresh"
    assert proposal["evidence"]["current_mute_cache_no_live_authority"] is True
    assert proposal["evidence"]["manual_mute_authority_preserved"] is True
    summary = proposal["evidence"]["mute_safety_summary"]
    assert summary == proposal["proposed_action"]["mute_safety_summary"]
    assert summary["summary_type"] == "mute_management_review"
    assert summary["channel_id"] == 12
    assert summary["current_muted"] is False
    assert summary["current_mute_state"] is False
    assert summary["proposed_muted"] is True
    assert summary["active_signal"] is False
    assert summary["signal_activity"]["idle_signal"] is True
    assert summary["feedback_risk"] == 0.05
    assert summary["dropout_risk"] == 0.1
    assert summary["manual_mute_authority_preserved"] is True
    assert summary["current_mute_cache_no_live_authority"] is True
    assert summary["rollback_value"] is False
    assert summary["rollback_mute_readback"] == proposal["evidence"]["rollback_mute_readback"]
    assert summary["active_signal_guard"] == "required_before_apply"
    assert summary["destructive_scope"] == "channel_mute_state_only"
    _assert_top_level_denial(summary)
    assert summary["can_authorize_live"] is False
    correction = proposal["operator_correction"]
    assert correction["objective_family"] == "routing_groups_scenes"
    assert correction["bounded_delta"]["unit"] == "mute_state_diff"
    assert correction["bounded_delta"]["old_value"] is False
    assert correction["bounded_delta"]["proposed_value"] is True
    assert correction["bounded_delta"]["parameter_deltas"]["proposed_muted"] is True
    assert correction["future_wing_readback_contract"]["target_kind"] == "channel_mute_state"
    assert correction["source_evidence"]["evidence_completeness"]["status"] == (
        "complete"
    )
    assert correction["source_evidence"]["evidence_completeness"]["missing"] == []
    assert {
        "rollback_mute_readback",
        "manual_mute_authority_preserved",
    } <= set(correction["source_evidence"]["required_evidence_fields"])
    assert correction["evidence_gate"]["status"] == (
        "required_evidence_complete_operator_review_still_required"
    )
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "current_mute_cache_no_live_authority"
    ] is True
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "rollback_mute_readback"
    ] == proposal["evidence"]["rollback_mute_readback"]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "manual_mute_authority_preserved"
    ] is True


def test_auto_muted_channel_with_returned_signal_creates_unmute_proposal():
    recommendation = analyze_mute_management_need(
        MuteManagementAnalysisInput(
            channel_id=12,
            channel_label="Floor Tom",
            source_role="floor_tom",
            current_muted=True,
            auto_muted_by_system=True,
            allow_auto_mute_candidate=True,
            rms_dbfs=-35.0,
            peak_dbfs=-22.0,
            recent_signal_cycles=3,
            source_confidence=0.9,
            dropout_risk=0.1,
            feedback_risk=0.05,
            persistence_cycles=3,
            current_state_cache_freshness=_fresh_state(),
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "unmute_proposal"
    assert recommendation.proposed_muted is False
    _assert_top_level_denial(recommendation.to_operator_proposal())


def test_emergency_feedback_context_creates_high_severity_mute_proposal_only():
    recommendation = analyze_mute_management_need(
        MuteManagementAnalysisInput(
            channel_id=7,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_muted=False,
            rms_dbfs=-18.0,
            peak_dbfs=-5.0,
            source_confidence=0.95,
            feedback_risk=0.9,
            operator_intent="emergency_feedback",
            persistence_cycles=3,
            current_state_cache_freshness=_fresh_state(),
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "emergency_mute_proposal"
    assert recommendation.severity == "high"
    assert recommendation.proposed_muted is True
    _assert_top_level_denial(recommendation.to_operator_proposal())


def test_critical_idle_source_remains_analysis_only():
    recommendation = analyze_mute_management_need(
        MuteManagementAnalysisInput(
            channel_id=7,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_muted=False,
            allow_auto_mute_candidate=True,
            rms_dbfs=-68.0,
            peak_dbfs=-56.0,
            idle_duration_sec=60.0,
            source_confidence=0.9,
            dropout_risk=0.1,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "critical_source_manual_only" in recommendation.blockers


def test_manual_mute_is_not_unmuted_without_authority():
    recommendation = analyze_mute_management_need(
        MuteManagementAnalysisInput(
            channel_id=8,
            channel_label="Guitar",
            source_role="guitar",
            current_muted=True,
            auto_muted_by_system=False,
            allow_auto_mute_candidate=True,
            rms_dbfs=-32.0,
            peak_dbfs=-18.0,
            recent_signal_cycles=3,
            source_confidence=0.9,
            dropout_risk=0.1,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "manual_mute_authority_required" in recommendation.blockers


def test_stale_current_mute_state_blocks_proposal_only_recommendation():
    recommendation = analyze_mute_management_need(
        MuteManagementAnalysisInput(
            channel_id=12,
            channel_label="Floor Tom",
            source_role="floor_tom",
            current_muted=False,
            allow_auto_mute_candidate=True,
            rms_dbfs=-67.0,
            peak_dbfs=-55.0,
            noise_floor_dbfs=-66.0,
            idle_duration_sec=48.0,
            source_confidence=0.9,
            bleed_risk=0.25,
            feedback_risk=0.05,
            dropout_risk=0.1,
            expected_next_cue_sec=40.0,
            persistence_cycles=3,
            current_state_cache_freshness={
                "status": "stale",
                "field_count": 5,
                "fresh_field_count": 3,
                "stale_field_count": 2,
                "stale_fields": ["current_muted", "mute_owner"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "mute_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "stale"
    assert recommendation.evidence["current_state_cache_freshness"]["stale_fields"] == [
        "current_muted",
        "mute_owner",
    ]
    _assert_top_level_denial(recommendation.to_operator_proposal())


def test_live_authority_current_mute_state_blocks_proposal_only_recommendation():
    analysis = MuteManagementAnalysisInput(
        channel_id=12,
        channel_label="Floor Tom",
        source_role="floor_tom",
        current_muted=False,
        allow_auto_mute_candidate=True,
        rms_dbfs=-67.0,
        peak_dbfs=-55.0,
        noise_floor_dbfs=-66.0,
        idle_duration_sec=48.0,
        source_confidence=0.9,
        bleed_risk=0.25,
        feedback_risk=0.05,
        dropout_risk=0.1,
        expected_next_cue_sec=40.0,
        persistence_cycles=3,
        current_state_cache_freshness={
            "status": "fresh",
            "field_count": 5,
            "fresh_field_count": 5,
            "read_only": False,
            "live_write": True,
            "queries_console": True,
        },
    )
    recommendation = analyze_mute_management_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "mute_current_state_cache_live_authority_present" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_mute_cache_no_live_authority"] is False
    assert recommendation.evidence["current_state_cache_freshness"]["queries_console"] is True
    _assert_top_level_denial(recommendation.to_operator_proposal())
    assert build_mute_management_proposals([analysis]) == ()


def test_missing_current_mute_state_blocks_proposal_only_recommendation():
    analysis = MuteManagementAnalysisInput(
        channel_id=12,
        channel_label="Floor Tom",
        source_role="floor_tom",
        current_muted=False,
        allow_auto_mute_candidate=True,
        rms_dbfs=-67.0,
        peak_dbfs=-55.0,
        noise_floor_dbfs=-66.0,
        idle_duration_sec=48.0,
        source_confidence=0.9,
        bleed_risk=0.25,
        feedback_risk=0.05,
        dropout_risk=0.1,
        expected_next_cue_sec=40.0,
        persistence_cycles=3,
    )
    recommendation = analyze_mute_management_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "mute_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "not_provided"
    assert recommendation.evidence["current_state_cache_freshness"]["provided"] is False
    _assert_top_level_denial(recommendation.to_operator_proposal())
    assert build_mute_management_proposals([analysis]) == ()


def test_source_identity_hold_blocks_downstream_mute_proposal():
    recommendation = analyze_mute_management_need(
        MuteManagementAnalysisInput(
            channel_id=12,
            channel_label="Floor Tom",
            source_role="floor_tom",
            current_muted=False,
            allow_auto_mute_candidate=True,
            rms_dbfs=-67.0,
            peak_dbfs=-55.0,
            noise_floor_dbfs=-66.0,
            idle_duration_sec=48.0,
            source_confidence=0.9,
            bleed_risk=0.25,
            feedback_risk=0.05,
            dropout_risk=0.1,
            expected_next_cue_sec=40.0,
            persistence_cycles=3,
            current_state_cache_freshness=_fresh_state(),
            source_identity_trust_state="hold_for_operator_review",
            source_identity_trust_confidence=0.42,
            source_identity_replay_outcome_count=4,
            source_identity_calibration_freshness={
                "status": "fresh",
                "field_count": 4,
                "fresh_field_count": 4,
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "source_identity_not_trusted_for_downstream" in recommendation.blockers
    assert recommendation.severity == "medium"
    trust = recommendation.evidence["source_identity_downstream_trust"]
    assert trust["status"] == "held_for_operator_review"
    assert trust["usable_for_downstream"] is False
    assert trust["evidence"]["queue_mutation"] is False
    assert trust["evidence"]["model_mutation"] is False
    _assert_top_level_denial(recommendation.to_operator_proposal())


def test_stale_source_identity_replay_context_blocks_downstream_mute_proposal():
    recommendation = analyze_mute_management_need(
        MuteManagementAnalysisInput(
            channel_id=12,
            channel_label="Floor Tom",
            source_role="floor_tom",
            current_muted=False,
            allow_auto_mute_candidate=True,
            rms_dbfs=-67.0,
            peak_dbfs=-55.0,
            noise_floor_dbfs=-66.0,
            idle_duration_sec=48.0,
            source_confidence=0.9,
            bleed_risk=0.25,
            feedback_risk=0.05,
            dropout_risk=0.1,
            expected_next_cue_sec=40.0,
            persistence_cycles=3,
            current_state_cache_freshness=_fresh_state(),
            source_identity_trust_state="trusted_for_downstream_priority",
            source_identity_trust_confidence=0.9,
            source_identity_replay_outcome_count=5,
            source_identity_calibration_freshness={
                "status": "stale",
                "field_count": 4,
                "fresh_field_count": 2,
                "stale_field_count": 2,
                "stale_fields": ["accepted_labels", "replay_outcomes"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "source_identity_replay_context_not_fresh" in recommendation.blockers
    assert recommendation.severity == "high"
    trust = recommendation.evidence["source_identity_downstream_trust"]
    assert trust["status"] == "trusted"
    assert trust["usable_for_downstream"] is False
    assert trust["evidence"]["current_state_cache_freshness"]["stale_fields"] == [
        "accepted_labels",
        "replay_outcomes",
    ]
    _assert_top_level_denial(recommendation.to_operator_proposal())


def test_mute_helpers_classify_source_family():
    assert classify_source_family("Lead Vocal") == "lead_vocal"
    assert classify_source_family("Floor Tom") == "tom"
    assert classify_source_family("Vocal Plate Return") == "fx_return"


def test_build_mute_management_proposals_filters_to_proposal_only_items():
    proposals = build_mute_management_proposals(
        [
            MuteManagementAnalysisInput(
                channel_id=12,
                source_role="floor_tom",
                current_muted=False,
                allow_auto_mute_candidate=True,
                rms_dbfs=-67.0,
                peak_dbfs=-55.0,
                idle_duration_sec=48.0,
                source_confidence=0.9,
                dropout_risk=0.1,
                expected_next_cue_sec=40.0,
                persistence_cycles=3,
                current_state_cache_freshness=_fresh_state(),
            ),
            MuteManagementAnalysisInput(
                channel_id=7,
                source_role="lead_vocal",
                current_muted=False,
                allow_auto_mute_candidate=True,
                rms_dbfs=-67.0,
                peak_dbfs=-55.0,
                idle_duration_sec=48.0,
                source_confidence=0.9,
            ),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["channel"] == 12
    _assert_top_level_denial(proposals[0])
    _assert_top_level_denial(proposals[0]["evidence"]["mute_safety_summary"])


def test_build_mute_management_proposals_filters_stale_state_items():
    proposals = build_mute_management_proposals(
        [
            MuteManagementAnalysisInput(
                channel_id=12,
                source_role="floor_tom",
                current_muted=False,
                allow_auto_mute_candidate=True,
                rms_dbfs=-67.0,
                peak_dbfs=-55.0,
                idle_duration_sec=48.0,
                source_confidence=0.9,
                dropout_risk=0.1,
                expected_next_cue_sec=40.0,
                persistence_cycles=3,
                current_state_cache_freshness={"status": "stale"},
            ),
        ]
    )

    assert proposals == ()


def test_mute_management_proposal_is_registered_as_goal_capability():
    method = get_processing_method("mute_management_proposal")
    main = get_pipeline_profile("main_live_assist")
    capability = get_project_goal_capability("console_mute_control")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.domain == "mute"
    assert method.can_write_live is False
    assert main is not None
    assert "mute_management_proposal" in main.proposal_method_ids
    assert capability is not None
    assert capability.status == "proposal_only"
    assert capability.autonomy_mode_ceiling == "assisted"
    assert "mute_management_proposal" in capability.method_ids
    assert "backend/source_identity_downstream_gate.py" in capability.support_paths
    assert contracts["mute_management_proposal"]["proposal_contract"]["value_type"] == "mute_state"
    assert (
        "current_mute_cache_freshness"
        in contracts["mute_management_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "current_mute_cache_no_live_authority"
        in contracts["mute_management_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "source_identity_downstream_trust"
        in contracts["mute_management_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "rollback_mute_readback"
        in contracts["mute_management_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "manual_mute_authority_preserved"
        in contracts["mute_management_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
