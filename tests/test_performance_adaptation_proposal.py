from pathlib import Path

from backend.performance_adaptation_proposal import (
    PerformanceAdaptationAnalysisInput,
    analyze_performance_adaptation_need,
    build_performance_adaptation_proposals,
    normalize_target_kind,
)
from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_pipeline_profile, get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability


def _fresh_state(field_count=8):
    return {
        "status": "fresh",
        "field_count": field_count,
        "fresh_field_count": field_count,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def test_performance_adaptation_module_is_transport_free():
    source = Path("backend/performance_adaptation_proposal.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "apply_operator_proposal",
    ]
    for token in forbidden:
        assert token not in source


def test_persistent_loud_musician_creates_proposal_only_downward_adaptation():
    recommendation = analyze_performance_adaptation_need(
        PerformanceAdaptationAnalysisInput(
            target_id="ch:7",
            target_kind="channel",
            target_label="Lead Guitar",
            source_role="electric_guitar",
            arrangement_role="support",
            baseline_level_lufs=-25.0,
            current_level_lufs=-21.6,
            short_term_lufs=-21.2,
            current_fader_db=-6.0,
            true_peak_dbfs=-7.0,
            main_bus_headroom_db=5.0,
            feedback_risk=0.12,
            bleed_risk=0.2,
            baseline_available=True,
            current_readback_available=True,
            rollback_available=True,
            operator_intent="operator_adaptation_review",
            adaptation_window="performance",
            active_performance=True,
            cooldown_elapsed_sec=12.0,
            persistence_cycles=4,
            latency_budget_ms=180.0,
            latency_budget_met=True,
            source_confidence=0.9,
            trend_confidence=0.88,
            safety_confidence=0.86,
            current_state_cache_freshness=_fresh_state(),
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "performance_adaptation_proposal"
    assert recommendation.target_kind == "channel"
    assert recommendation.level_drift_db > 3.0
    assert recommendation.suggested_delta_db == -1.5
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["rate_limited"] is True
    assert recommendation.evidence["active_performance"] is True
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_performance_context_cache_no_live_authority"] is True
    assert recommendation.evidence["current_state_cache_freshness"]["fresh_field_count"] == 8
    assert recommendation.evidence["source_identity_downstream_trust"]["status"] == "legacy_unverified"

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "performance_adaptation_proposal"
    assert proposal["kind"] == "performance_adaptation"
    assert proposal["can_import_to_queue"] is False
    assert proposal["can_apply_live"] is False
    assert "requested_change" not in proposal
    assert "requested_changes" not in proposal
    assert "unsupported_value_type:performance_adaptation" in proposal["blockers"]
    assert proposal["proposed_action"]["suggested_delta_db"] == -1.5
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["current_state_cache_freshness_status"] == "fresh"
    assert proposal["evidence"]["current_performance_context_cache_no_live_authority"] is True


def test_upward_adaptation_requires_headroom_and_low_feedback_bleed():
    recommendation = analyze_performance_adaptation_need(
        PerformanceAdaptationAnalysisInput(
            target_id="ch:3",
            target_kind="channel",
            target_label="Rack Tom",
            source_role="tom",
            baseline_level_lufs=-25.0,
            current_level_lufs=-28.4,
            short_term_lufs=-28.0,
            current_fader_db=-2.0,
            main_bus_headroom_db=0.6,
            feedback_risk=0.7,
            bleed_risk=0.75,
            baseline_available=True,
            current_readback_available=True,
            rollback_available=True,
            operator_intent="operator_adaptation_review",
            adaptation_window="performance",
            cooldown_elapsed_sec=12.0,
            persistence_cycles=4,
            latency_budget_met=True,
            source_confidence=0.9,
            trend_confidence=0.9,
            safety_confidence=0.9,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "true_peak_headroom_insufficient_for_upward_adaptation" in recommendation.blockers
    assert "feedback_risk_high_for_upward_adaptation" in recommendation.blockers
    assert "bleed_risk_high_for_upward_adaptation" in recommendation.blockers
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_missing_evidence_or_manual_override_blocks_adaptation():
    recommendation = analyze_performance_adaptation_need(
        PerformanceAdaptationAnalysisInput(
            target_id="group:vocals",
            target_kind="group",
            source_role="lead_vocal",
            baseline_level_lufs=-22.0,
            current_level_lufs=-19.0,
            main_bus_headroom_db=4.0,
            baseline_available=False,
            current_readback_available=False,
            rollback_available=False,
            operator_intent="",
            manual_override_active=True,
            cooldown_elapsed_sec=1.0,
            persistence_cycles=1,
            latency_budget_met=False,
            source_confidence=0.4,
            trend_confidence=0.4,
            safety_confidence=0.4,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "baseline_level_missing" in recommendation.blockers
    assert "current_level_readback_missing" in recommendation.blockers
    assert "rollback_state_missing" in recommendation.blockers
    assert "operator_intent_required" in recommendation.blockers
    assert "manual_override_active" in recommendation.blockers
    assert "cooldown_not_elapsed" in recommendation.blockers
    assert "trend_persistence_low" in recommendation.blockers
    assert "latency_budget_missing" in recommendation.blockers


def test_stale_current_performance_context_blocks_adaptation_proposal():
    recommendation = analyze_performance_adaptation_need(
        PerformanceAdaptationAnalysisInput(
            target_id="ch:7",
            target_kind="channel",
            target_label="Lead Guitar",
            source_role="electric_guitar",
            arrangement_role="support",
            baseline_level_lufs=-25.0,
            current_level_lufs=-21.6,
            short_term_lufs=-21.2,
            current_fader_db=-6.0,
            true_peak_dbfs=-7.0,
            main_bus_headroom_db=5.0,
            feedback_risk=0.12,
            bleed_risk=0.2,
            baseline_available=True,
            current_readback_available=True,
            rollback_available=True,
            operator_intent="operator_adaptation_review",
            adaptation_window="performance",
            active_performance=True,
            cooldown_elapsed_sec=12.0,
            persistence_cycles=4,
            latency_budget_ms=180.0,
            latency_budget_met=True,
            source_confidence=0.9,
            trend_confidence=0.88,
            safety_confidence=0.86,
            current_state_cache_freshness={
                "status": "stale",
                "field_count": 8,
                "fresh_field_count": 6,
                "stale_field_count": 2,
                "stale_fields": ["current_level_lufs", "trend_persistence"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "performance_adaptation_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "stale"
    assert recommendation.evidence["current_state_cache_freshness"]["stale_fields"] == [
        "current_level_lufs",
        "trend_persistence",
    ]
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_live_authority_current_performance_context_blocks_adaptation_proposal():
    analysis = PerformanceAdaptationAnalysisInput(
        target_id="ch:7",
        target_kind="channel",
        target_label="Lead Guitar",
        source_role="electric_guitar",
        arrangement_role="support",
        baseline_level_lufs=-25.0,
        current_level_lufs=-21.6,
        short_term_lufs=-21.2,
        current_fader_db=-6.0,
        true_peak_dbfs=-7.0,
        main_bus_headroom_db=5.0,
        feedback_risk=0.12,
        bleed_risk=0.2,
        baseline_available=True,
        current_readback_available=True,
        rollback_available=True,
        operator_intent="operator_adaptation_review",
        adaptation_window="performance",
        active_performance=True,
        cooldown_elapsed_sec=12.0,
        persistence_cycles=4,
        latency_budget_ms=180.0,
        latency_budget_met=True,
        source_confidence=0.9,
        trend_confidence=0.88,
        safety_confidence=0.86,
        current_state_cache_freshness={
            "status": "fresh",
            "field_count": 8,
            "fresh_field_count": 8,
            "read_only": False,
            "live_write": True,
            "queries_console": True,
        },
    )
    recommendation = analyze_performance_adaptation_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "performance_adaptation_current_state_cache_live_authority_present" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_performance_context_cache_no_live_authority"] is False
    assert recommendation.evidence["current_state_cache_freshness"]["live_write"] is True
    assert recommendation.to_operator_proposal()["can_apply_live"] is False
    assert build_performance_adaptation_proposals([analysis]) == ()


def test_missing_current_performance_context_blocks_adaptation_proposal():
    analysis = PerformanceAdaptationAnalysisInput(
        target_id="ch:7",
        target_kind="channel",
        target_label="Lead Guitar",
        source_role="electric_guitar",
        arrangement_role="support",
        baseline_level_lufs=-25.0,
        current_level_lufs=-21.6,
        short_term_lufs=-21.2,
        current_fader_db=-6.0,
        true_peak_dbfs=-7.0,
        main_bus_headroom_db=5.0,
        feedback_risk=0.12,
        bleed_risk=0.2,
        baseline_available=True,
        current_readback_available=True,
        rollback_available=True,
        operator_intent="operator_adaptation_review",
        adaptation_window="performance",
        active_performance=True,
        cooldown_elapsed_sec=12.0,
        persistence_cycles=4,
        latency_budget_ms=180.0,
        latency_budget_met=True,
        source_confidence=0.9,
        trend_confidence=0.88,
        safety_confidence=0.86,
    )
    recommendation = analyze_performance_adaptation_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "performance_adaptation_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "not_provided"
    assert recommendation.evidence["current_state_cache_freshness"]["provided"] is False
    assert build_performance_adaptation_proposals([analysis]) == ()


def test_source_identity_hold_blocks_downstream_performance_adaptation():
    recommendation = analyze_performance_adaptation_need(
        PerformanceAdaptationAnalysisInput(
            target_id="ch:7",
            target_kind="channel",
            target_label="Lead Guitar",
            source_role="electric_guitar",
            arrangement_role="support",
            baseline_level_lufs=-25.0,
            current_level_lufs=-21.6,
            short_term_lufs=-21.2,
            current_fader_db=-6.0,
            true_peak_dbfs=-7.0,
            main_bus_headroom_db=5.0,
            feedback_risk=0.12,
            bleed_risk=0.2,
            baseline_available=True,
            current_readback_available=True,
            rollback_available=True,
            operator_intent="operator_adaptation_review",
            adaptation_window="performance",
            active_performance=True,
            cooldown_elapsed_sec=12.0,
            persistence_cycles=4,
            latency_budget_ms=180.0,
            latency_budget_met=True,
            source_confidence=0.9,
            trend_confidence=0.88,
            safety_confidence=0.86,
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
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_stale_source_identity_replay_context_blocks_downstream_performance_adaptation():
    recommendation = analyze_performance_adaptation_need(
        PerformanceAdaptationAnalysisInput(
            target_id="ch:7",
            target_kind="channel",
            target_label="Lead Guitar",
            source_role="electric_guitar",
            arrangement_role="support",
            baseline_level_lufs=-25.0,
            current_level_lufs=-21.6,
            short_term_lufs=-21.2,
            current_fader_db=-6.0,
            true_peak_dbfs=-7.0,
            main_bus_headroom_db=5.0,
            feedback_risk=0.12,
            bleed_risk=0.2,
            baseline_available=True,
            current_readback_available=True,
            rollback_available=True,
            operator_intent="operator_adaptation_review",
            adaptation_window="performance",
            active_performance=True,
            cooldown_elapsed_sec=12.0,
            persistence_cycles=4,
            latency_budget_ms=180.0,
            latency_budget_met=True,
            source_confidence=0.9,
            trend_confidence=0.88,
            safety_confidence=0.86,
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
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_small_or_recent_drift_is_analysis_only():
    recommendation = analyze_performance_adaptation_need(
        PerformanceAdaptationAnalysisInput(
            target_id="ch:8",
            target_kind="ch",
            source_role="keys",
            baseline_level_lufs=-30.0,
            current_level_lufs=-29.8,
            current_fader_db=-8.0,
            main_bus_headroom_db=4.0,
            baseline_available=True,
            current_readback_available=True,
            rollback_available=True,
            operator_intent="operator_adaptation_review",
            cooldown_elapsed_sec=2.0,
            persistence_cycles=1,
            source_confidence=0.9,
            trend_confidence=0.9,
            safety_confidence=0.9,
        )
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.target_kind == "channel"
    assert "adaptation_change_not_needed" in recommendation.blockers
    assert "cooldown_not_elapsed" in recommendation.blockers
    assert "trend_persistence_low" in recommendation.blockers


def test_build_performance_adaptation_proposals_filters_to_proposal_only_items():
    proposals = build_performance_adaptation_proposals(
        [
            PerformanceAdaptationAnalysisInput(
                target_id="ch:7",
                target_kind="channel",
                source_role="electric_guitar",
                baseline_level_lufs=-25.0,
                current_level_lufs=-21.6,
                current_fader_db=-6.0,
                main_bus_headroom_db=5.0,
                baseline_available=True,
                current_readback_available=True,
                rollback_available=True,
                operator_intent="operator_adaptation_review",
                cooldown_elapsed_sec=12.0,
                persistence_cycles=4,
                source_confidence=0.9,
                trend_confidence=0.88,
                safety_confidence=0.86,
                current_state_cache_freshness=_fresh_state(),
            ),
            PerformanceAdaptationAnalysisInput(
                target_id="ch:8",
                target_kind="channel",
                source_role="keys",
                baseline_level_lufs=-30.0,
                current_level_lufs=-29.9,
                baseline_available=True,
                current_readback_available=True,
                rollback_available=True,
                operator_intent="operator_adaptation_review",
                cooldown_elapsed_sec=12.0,
                persistence_cycles=4,
                source_confidence=0.9,
                trend_confidence=0.88,
                safety_confidence=0.86,
            ),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["target"] == "ch:7"
    assert proposals[0]["can_import_to_queue"] is False


def test_build_performance_adaptation_proposals_filters_stale_context_items():
    proposals = build_performance_adaptation_proposals(
        [
            PerformanceAdaptationAnalysisInput(
                target_id="ch:7",
                target_kind="channel",
                source_role="electric_guitar",
                baseline_level_lufs=-25.0,
                current_level_lufs=-21.6,
                current_fader_db=-6.0,
                main_bus_headroom_db=5.0,
                baseline_available=True,
                current_readback_available=True,
                rollback_available=True,
                operator_intent="operator_adaptation_review",
                cooldown_elapsed_sec=12.0,
                persistence_cycles=4,
                source_confidence=0.9,
                trend_confidence=0.88,
                safety_confidence=0.86,
                current_state_cache_freshness={"status": "stale"},
            ),
        ]
    )

    assert proposals == ()


def test_performance_adaptation_helpers_normalize_target_kind():
    assert normalize_target_kind("ch") == "channel"
    assert normalize_target_kind("subgroup") == "group"
    assert normalize_target_kind("mtx") == "matrix"


def test_performance_adaptation_proposal_is_registered_as_goal_capability():
    method = get_processing_method("performance_adaptation_proposal")
    main = get_pipeline_profile("main_live_assist")
    capability = get_project_goal_capability("musician_level_adaptation")
    live = get_project_goal_capability("live_adaptation")
    realtime = get_project_goal_capability("realtime_corrections")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.domain == "adaptation"
    assert method.can_write_live is False
    assert main is not None
    assert "performance_adaptation_proposal" in main.proposal_method_ids
    assert capability is not None
    assert capability.status == "proposal_only"
    assert capability.autonomy_mode_ceiling == "assisted"
    assert "performance_adaptation_proposal" in capability.method_ids
    assert "backend/source_identity_downstream_gate.py" in capability.support_paths
    assert live is not None and live.status == "proposal_only"
    assert "performance_adaptation_proposal" in live.method_ids
    assert "backend/source_identity_downstream_gate.py" in live.support_paths
    assert realtime is not None and realtime.status == "proposal_only"
    assert "performance_adaptation_proposal" in realtime.method_ids
    assert "backend/source_identity_downstream_gate.py" in realtime.support_paths
    assert contracts["performance_adaptation_proposal"]["proposal_contract"]["value_type"] == "performance_adaptation"
    assert (
        "current_performance_context_cache_freshness"
        in contracts["performance_adaptation_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "current_performance_context_cache_no_live_authority"
        in contracts["performance_adaptation_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "source_identity_downstream_trust"
        in contracts["performance_adaptation_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
