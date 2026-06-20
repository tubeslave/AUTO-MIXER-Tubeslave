from pathlib import Path

from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_pipeline_profile, get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.routing_management_proposal import (
    RoutingManagementAnalysisInput,
    analyze_routing_management_need,
    build_routing_management_proposals,
    normalize_route_kind,
)


def _fresh_state(field_count=6):
    return {
        "status": "fresh",
        "field_count": field_count,
        "fresh_field_count": field_count,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def test_routing_management_module_is_transport_free():
    source = Path("backend/routing_management_proposal.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "route_output",
        "route_multiple_outputs",
        "set_channel_input",
        "load_snap",
        "save_snap",
        "recall_scene",
        "mixer_client",
    ]
    for token in forbidden:
        assert token not in source


def test_soundcheck_channel_route_diff_creates_proposal_only_recommendation():
    recommendation = analyze_routing_management_need(
        RoutingManagementAnalysisInput(
            route_id="ch:10:input",
            route_kind="channel_input",
            target_label="Lead Vox input",
            current_source_group="LOCAL",
            current_source_channel=10,
            current_source_label="Local 10",
            proposed_source_group="DANTE",
            proposed_source_channel=10,
            proposed_source_label="Dante Vox 10",
            expected_source_role="lead_vocal",
            route_readback_available=True,
            rollback_route_available=True,
            snapshot_baseline_id="soundcheck_pre_patch",
            operator_intent="operator_patch_request",
            route_change_window="soundcheck",
            active_performance=False,
            current_route_signal_dbfs=-82.0,
            proposed_source_signal_dbfs=-28.0,
            proposed_source_verified=True,
            destructive_scope_score=0.1,
            affected_target_count=1,
            source_confidence=0.9,
            route_confidence=0.92,
            persistence_cycles=3,
            current_state_cache_freshness=_fresh_state(),
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "routing_proposal"
    assert recommendation.route_kind == "channel_input"
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["destructive_console_state"] is True
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_route_cache_no_live_authority"] is True
    assert recommendation.evidence["current_state_cache_freshness"]["fresh_field_count"] == 6
    assert recommendation.evidence["source_identity_downstream_trust"]["status"] == (
        "legacy_unverified"
    )

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "routing_management_proposal"
    assert proposal["kind"] == "routing"
    assert proposal["can_import_to_queue"] is False
    assert proposal["can_apply_live"] is False
    assert "requested_change" not in proposal
    assert "requested_changes" not in proposal
    assert "unsupported_value_type:routing" in proposal["blockers"]
    assert proposal["proposed_action"]["proposed_source"]["group"] == "DANTE"
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["current_state_cache_freshness_status"] == "fresh"
    assert proposal["evidence"]["current_route_cache_no_live_authority"] is True


def test_active_performance_and_active_route_block_routing_proposal():
    recommendation = analyze_routing_management_need(
        RoutingManagementAnalysisInput(
            route_id="ch:10:input",
            route_kind="channel_input",
            current_source_group="LOCAL",
            current_source_channel=10,
            proposed_source_group="DANTE",
            proposed_source_channel=10,
            route_readback_available=True,
            rollback_route_available=True,
            snapshot_baseline_id="show_baseline",
            operator_intent="operator_patch_request",
            route_change_window="show",
            active_performance=True,
            current_route_signal_dbfs=-20.0,
            proposed_source_verified=True,
            source_confidence=0.9,
            route_confidence=0.9,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "active_performance_routing_blocked" in recommendation.blockers
    assert "current_route_active_requires_manual_patch" in recommendation.blockers
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_source_identity_hold_blocks_routing_proposal():
    recommendation = analyze_routing_management_need(
        RoutingManagementAnalysisInput(
            route_id="ch:10:input",
            route_kind="channel_input",
            target_label="Lead Vox input",
            current_source_group="LOCAL",
            current_source_channel=10,
            current_source_label="Local 10",
            proposed_source_group="DANTE",
            proposed_source_channel=10,
            proposed_source_label="Dante Vox 10",
            expected_source_role="lead_vocal",
            route_readback_available=True,
            rollback_route_available=True,
            snapshot_baseline_id="soundcheck_pre_patch",
            operator_intent="operator_patch_request",
            route_change_window="soundcheck",
            active_performance=False,
            current_route_signal_dbfs=-82.0,
            proposed_source_signal_dbfs=-28.0,
            proposed_source_verified=True,
            destructive_scope_score=0.1,
            affected_target_count=1,
            source_confidence=0.9,
            route_confidence=0.92,
            persistence_cycles=3,
            current_state_cache_freshness={"status": "fresh", "read_only": True},
            source_identity_trust_state="hold_for_operator_review",
            source_identity_trust_confidence=0.58,
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
    assert trust["source_id"] == "DANTE:10"
    assert trust["source_type"] == "lead_vocal"
    assert trust["status"] == "held_for_operator_review"
    assert trust["usable_for_downstream"] is False
    proposal = recommendation.to_operator_proposal()
    assert proposal["can_apply_live"] is False
    assert "requested_change" not in proposal


def test_stale_source_identity_replay_context_blocks_routing_proposal():
    analysis = RoutingManagementAnalysisInput(
        route_id="ch:10:input",
        route_kind="channel_input",
        current_source_group="LOCAL",
        current_source_channel=10,
        proposed_source_group="DANTE",
        proposed_source_channel=10,
        expected_source_role="lead_vocal",
        route_readback_available=True,
        rollback_route_available=True,
        snapshot_baseline_id="soundcheck_pre_patch",
        operator_intent="operator_patch_request",
        route_change_window="soundcheck",
        active_performance=False,
        current_route_signal_dbfs=-82.0,
        proposed_source_verified=True,
        destructive_scope_score=0.1,
        affected_target_count=1,
        source_confidence=0.9,
        route_confidence=0.92,
        persistence_cycles=3,
        current_state_cache_freshness={"status": "fresh", "read_only": True},
        source_identity_trust_state="trusted_for_downstream_priority",
        source_identity_trust_confidence=0.91,
        source_identity_replay_outcome_count=8,
        source_identity_calibration_freshness={
            "status": "stale",
            "field_count": 4,
            "fresh_field_count": 2,
            "stale_field_count": 2,
            "stale_fields": ["accepted_outcomes", "rejected_outcomes"],
            "read_only": True,
            "live_write": False,
            "queries_console": False,
        },
    )
    recommendation = analyze_routing_management_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "source_identity_replay_context_not_fresh" in recommendation.blockers
    assert recommendation.severity == "high"
    trust = recommendation.evidence["source_identity_downstream_trust"]
    assert trust["status"] == "trusted"
    assert trust["usable_for_downstream"] is False
    assert trust["evidence"]["current_source_identity_replay_context_cache_freshness"] == "stale"
    assert trust["evidence"]["current_state_cache_freshness"]["stale_fields"] == [
        "accepted_outcomes",
        "rejected_outcomes",
    ]
    assert build_routing_management_proposals([analysis]) == ()


def test_missing_readback_or_rollback_blocks_routing_proposal():
    recommendation = analyze_routing_management_need(
        RoutingManagementAnalysisInput(
            route_id="out:1",
            route_kind="output_patch",
            current_source_group="MAIN",
            current_source_channel=1,
            proposed_source_group="MTX",
            proposed_source_channel=1,
            route_readback_available=False,
            rollback_route_available=False,
            operator_intent="operator_patch_request",
            route_change_window="soundcheck",
            current_route_signal_dbfs=-90.0,
            proposed_source_verified=True,
            source_confidence=0.9,
            route_confidence=0.9,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "route_readback_missing" in recommendation.blockers
    assert "rollback_route_missing" in recommendation.blockers


def test_stale_current_routing_state_blocks_proposal_only_recommendation():
    recommendation = analyze_routing_management_need(
        RoutingManagementAnalysisInput(
            route_id="ch:10:input",
            route_kind="channel_input",
            target_label="Lead Vox input",
            current_source_group="LOCAL",
            current_source_channel=10,
            current_source_label="Local 10",
            proposed_source_group="DANTE",
            proposed_source_channel=10,
            proposed_source_label="Dante Vox 10",
            expected_source_role="lead_vocal",
            route_readback_available=True,
            rollback_route_available=True,
            snapshot_baseline_id="soundcheck_pre_patch",
            operator_intent="operator_patch_request",
            route_change_window="soundcheck",
            active_performance=False,
            current_route_signal_dbfs=-82.0,
            proposed_source_signal_dbfs=-28.0,
            proposed_source_verified=True,
            destructive_scope_score=0.1,
            affected_target_count=1,
            source_confidence=0.9,
            route_confidence=0.92,
            persistence_cycles=3,
            current_state_cache_freshness={
                "status": "stale",
                "field_count": 6,
                "fresh_field_count": 4,
                "stale_field_count": 2,
                "stale_fields": ["current_source", "rollback_route"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "routing_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "stale"
    assert recommendation.evidence["current_state_cache_freshness"]["stale_fields"] == [
        "current_source",
        "rollback_route",
    ]
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_live_authority_current_routing_state_blocks_proposal_only_recommendation():
    analysis = RoutingManagementAnalysisInput(
        route_id="ch:10:input",
        route_kind="channel_input",
        target_label="Lead Vox input",
        current_source_group="LOCAL",
        current_source_channel=10,
        current_source_label="Local 10",
        proposed_source_group="DANTE",
        proposed_source_channel=10,
        proposed_source_label="Dante Vox 10",
        expected_source_role="lead_vocal",
        route_readback_available=True,
        rollback_route_available=True,
        snapshot_baseline_id="soundcheck_pre_patch",
        operator_intent="operator_patch_request",
        route_change_window="soundcheck",
        active_performance=False,
        current_route_signal_dbfs=-82.0,
        proposed_source_signal_dbfs=-28.0,
        proposed_source_verified=True,
        destructive_scope_score=0.1,
        affected_target_count=1,
        source_confidence=0.9,
        route_confidence=0.92,
        persistence_cycles=3,
        current_state_cache_freshness={
            "status": "fresh",
            "field_count": 6,
            "fresh_field_count": 6,
            "read_only": False,
            "live_write": True,
            "queries_console": True,
        },
    )
    recommendation = analyze_routing_management_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "routing_current_state_cache_live_authority_present" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_route_cache_no_live_authority"] is False
    assert recommendation.evidence["current_state_cache_freshness"]["live_write"] is True
    assert recommendation.to_operator_proposal()["can_apply_live"] is False
    assert build_routing_management_proposals([analysis]) == ()


def test_missing_current_routing_state_blocks_proposal_only_recommendation():
    analysis = RoutingManagementAnalysisInput(
        route_id="ch:10:input",
        route_kind="channel_input",
        target_label="Lead Vox input",
        current_source_group="LOCAL",
        current_source_channel=10,
        current_source_label="Local 10",
        proposed_source_group="DANTE",
        proposed_source_channel=10,
        proposed_source_label="Dante Vox 10",
        expected_source_role="lead_vocal",
        route_readback_available=True,
        rollback_route_available=True,
        snapshot_baseline_id="soundcheck_pre_patch",
        operator_intent="operator_patch_request",
        route_change_window="soundcheck",
        active_performance=False,
        current_route_signal_dbfs=-82.0,
        proposed_source_signal_dbfs=-28.0,
        proposed_source_verified=True,
        destructive_scope_score=0.1,
        affected_target_count=1,
        source_confidence=0.9,
        route_confidence=0.92,
        persistence_cycles=3,
    )
    recommendation = analyze_routing_management_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "routing_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "not_provided"
    assert recommendation.evidence["current_state_cache_freshness"]["provided"] is False
    assert build_routing_management_proposals([analysis]) == ()


def test_unchanged_route_is_analysis_only():
    recommendation = analyze_routing_management_need(
        RoutingManagementAnalysisInput(
            route_id="ch:10:input",
            route_kind="input",
            current_source_group="DANTE",
            current_source_channel=10,
            proposed_source_group="DANTE",
            proposed_source_channel=10,
            route_readback_available=True,
            rollback_route_available=True,
            snapshot_baseline_id="baseline",
            operator_intent="operator_patch_request",
            current_route_signal_dbfs=-90.0,
            source_confidence=0.9,
            route_confidence=0.9,
        )
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.route_kind == "channel_input"
    assert "route_change_not_needed" in recommendation.blockers


def test_routing_helpers_normalize_route_kind():
    assert normalize_route_kind("input") == "channel_input"
    assert normalize_route_kind("output") == "output_patch"
    assert normalize_route_kind("matrix") == "matrix_source"


def test_build_routing_management_proposals_filters_to_proposal_only_items():
    proposals = build_routing_management_proposals(
        [
            RoutingManagementAnalysisInput(
                route_id="ch:10:input",
                route_kind="channel_input",
                current_source_group="LOCAL",
                current_source_channel=10,
                proposed_source_group="DANTE",
                proposed_source_channel=10,
                route_readback_available=True,
                rollback_route_available=True,
                snapshot_baseline_id="baseline",
                operator_intent="operator_patch_request",
                route_change_window="soundcheck",
                current_route_signal_dbfs=-82.0,
                proposed_source_verified=True,
                source_confidence=0.9,
                route_confidence=0.9,
                persistence_cycles=3,
                current_state_cache_freshness=_fresh_state(),
            ),
            RoutingManagementAnalysisInput(
                route_id="ch:11:input",
                route_kind="channel_input",
                current_source_group="DANTE",
                current_source_channel=11,
                proposed_source_group="DANTE",
                proposed_source_channel=11,
                route_readback_available=True,
                rollback_route_available=True,
                snapshot_baseline_id="baseline",
                operator_intent="operator_patch_request",
            ),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["target"] == "ch:10:input"


def test_build_routing_management_proposals_filters_stale_state_items():
    proposals = build_routing_management_proposals(
        [
            RoutingManagementAnalysisInput(
                route_id="ch:10:input",
                route_kind="channel_input",
                current_source_group="LOCAL",
                current_source_channel=10,
                proposed_source_group="DANTE",
                proposed_source_channel=10,
                route_readback_available=True,
                rollback_route_available=True,
                snapshot_baseline_id="baseline",
                operator_intent="operator_patch_request",
                route_change_window="soundcheck",
                current_route_signal_dbfs=-82.0,
                proposed_source_verified=True,
                source_confidence=0.9,
                route_confidence=0.9,
                persistence_cycles=3,
                current_state_cache_freshness={"status": "stale"},
            ),
        ]
    )

    assert proposals == ()


def test_routing_management_proposal_is_registered_as_goal_capability():
    method = get_processing_method("routing_management_proposal")
    direct_write = get_processing_method("routing_snapshot_reset_writes")
    main = get_pipeline_profile("main_live_assist")
    capability = get_project_goal_capability("console_routing_control")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.domain == "routing"
    assert method.can_write_live is False
    assert direct_write is not None
    assert direct_write.status == "quarantined"
    assert main is not None
    assert "routing_management_proposal" in main.proposal_method_ids
    assert capability is not None
    assert capability.status == "proposal_only"
    assert capability.autonomy_mode_ceiling == "assisted"
    assert "routing_management_proposal" in capability.method_ids
    assert "backend/source_identity_downstream_gate.py" in capability.support_paths
    assert contracts["routing_management_proposal"]["proposal_contract"]["value_type"] == "routing"
    assert (
        "current_route_cache_freshness"
        in contracts["routing_management_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "current_route_cache_no_live_authority"
        in contracts["routing_management_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "source_identity_downstream_trust"
        in contracts["routing_management_proposal"]["proposal_contract"][
            "safety_evidence_required"
        ]
    )
