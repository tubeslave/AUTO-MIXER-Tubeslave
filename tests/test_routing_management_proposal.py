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
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "routing_proposal"
    assert recommendation.route_kind == "channel_input"
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["destructive_console_state"] is True

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
    assert contracts["routing_management_proposal"]["proposal_contract"]["value_type"] == "routing"
