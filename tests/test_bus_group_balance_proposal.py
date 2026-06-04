from pathlib import Path

from backend.bus_group_balance_proposal import (
    BusGroupBalanceAnalysisInput,
    analyze_bus_group_balance_need,
    build_bus_group_balance_proposals,
    classify_bus_family,
    resolve_ayaic_bus_target_lufs,
)
from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_pipeline_profile, get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability


def test_bus_group_balance_module_is_transport_free():
    source = Path("backend/bus_group_balance_proposal.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "set_dca",
        "set_matrix",
        "set_bus",
        "DCAFaderMove",
        "mixer_client",
    ]
    for token in forbidden:
        assert token not in source


def test_safe_drum_group_creates_proposal_only_balance_recommendation():
    recommendation = analyze_bus_group_balance_need(
        BusGroupBalanceAnalysisInput(
            bus_id="group:drums",
            bus_kind="group",
            bus_label="DRUMS",
            source_role="drums",
            current_fader_db=-5.0,
            integrated_lufs=-26.1,
            short_term_lufs=-25.8,
            true_peak_dbfs=-8.0,
            summed_child_lufs=-25.6,
            child_count=8,
            dominant_child_share=0.35,
            main_bus_headroom_db=6.0,
            feedback_risk=0.1,
            assignment_confidence=0.9,
            confidence=0.9,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "bus_group_balance_proposal"
    assert recommendation.bus_kind == "group"
    assert recommendation.source_family == "drums"
    assert recommendation.target_plane_lufs == -25.0
    assert 0.4 <= recommendation.suggested_delta_db <= 1.0
    assert recommendation.target_fader_db <= 0.0
    assert recommendation.blockers == ()
    assert recommendation.evidence["ayaic_level_plane"] is True
    assert recommendation.evidence["transport_free"] is True

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "bus_group_balance_proposal"
    assert proposal["kind"] == "bus_group_balance"
    assert proposal["can_import_to_queue"] is False
    assert proposal["can_apply_live"] is False
    assert "requested_change" not in proposal
    assert "requested_changes" not in proposal
    assert "unsupported_value_type:bus_group_balance" in proposal["blockers"]
    assert proposal["proposed_action"]["suggested_delta_db"] == recommendation.suggested_delta_db
    assert proposal["evidence"]["live_write"] is False


def test_peak_limited_bus_up_is_analysis_only():
    recommendation = analyze_bus_group_balance_need(
        BusGroupBalanceAnalysisInput(
            bus_id="dca:band",
            bus_kind="dca",
            bus_label="Band DCA",
            source_role="band",
            current_fader_db=-4.0,
            integrated_lufs=-30.0,
            true_peak_dbfs=-3.5,
            child_count=12,
            main_bus_headroom_db=0.5,
            assignment_confidence=0.9,
            confidence=0.95,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.suggested_delta_db == 0.0
    assert "true_peak_headroom_insufficient_for_bus_up" in recommendation.blockers
    assert "main_bus_headroom_insufficient_for_bus_up" in recommendation.blockers
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_low_assignment_confidence_blocks_group_balance():
    recommendation = analyze_bus_group_balance_need(
        BusGroupBalanceAnalysisInput(
            bus_id="matrix:fill",
            bus_kind="matrix",
            bus_label="Front Fill",
            source_role="matrix",
            current_fader_db=-6.0,
            integrated_lufs=-27.0,
            true_peak_dbfs=-10.0,
            child_count=2,
            main_bus_headroom_db=8.0,
            assignment_confidence=0.2,
            confidence=0.8,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "bus_assignment_confidence_low" in recommendation.blockers


def test_bus_group_helpers_resolve_ayaic_planes():
    assert classify_bus_family("Lead Vocal", "Lead Vox", "group") == "lead_vocal"
    assert classify_bus_family("Backing Vocal", "BGV", "group") == "backing_vocal"
    assert resolve_ayaic_bus_target_lufs("group", "Lead Vocal", "lead")[0] == -22.0
    assert resolve_ayaic_bus_target_lufs("group", "Backing Vocal")[0] == -27.0
    assert resolve_ayaic_bus_target_lufs("fx_return", "Vocal Plate", "deep")[0] == -35.0


def test_build_bus_group_balance_proposals_filters_to_proposal_only_items():
    proposals = build_bus_group_balance_proposals(
        [
            BusGroupBalanceAnalysisInput(
                bus_id="group:drums",
                bus_kind="group",
                source_role="drums",
                current_fader_db=-5.0,
                integrated_lufs=-26.1,
                true_peak_dbfs=-8.0,
                child_count=8,
                main_bus_headroom_db=6.0,
                assignment_confidence=0.9,
                confidence=0.9,
                persistence_cycles=3,
            ),
            BusGroupBalanceAnalysisInput(
                bus_id="group:drums",
                bus_kind="group",
                source_role="drums",
                current_fader_db=-5.0,
                integrated_lufs=-25.0,
                true_peak_dbfs=-8.0,
                child_count=8,
            ),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["target"] == "group:drums"
    assert proposals[0]["can_import_to_queue"] is False


def test_bus_group_balance_proposal_is_registered_as_goal_capability():
    method = get_processing_method("bus_group_balance_proposal")
    main = get_pipeline_profile("main_live_assist")
    mix_balance = get_project_goal_capability("mix_balance")
    console = get_project_goal_capability("console_dca_group_matrix_control")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.domain == "bus_group"
    assert method.can_write_live is False
    assert main is not None
    assert "bus_group_balance_proposal" in main.proposal_method_ids
    assert mix_balance is not None
    assert "bus_group_balance_proposal" in mix_balance.method_ids
    assert console is not None
    assert console.status == "proposal_only"
    assert console.autonomy_mode_ceiling == "assisted"
    assert "bus_group_balance_proposal" in console.method_ids
    assert contracts["bus_group_balance_proposal"]["proposal_contract"]["value_type"] == "bus_group_balance"
