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


def _fresh(fields=6):
    return {
        "status": "fresh",
        "field_count": fields,
        "fresh_field_count": fields,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


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
    assert recommendation.action_type == "bus_group_balance_proposal"
    assert recommendation.bus_kind == "group"
    assert recommendation.source_family == "drums"
    assert recommendation.target_plane_lufs == -25.0
    assert 0.4 <= recommendation.suggested_delta_db <= 1.0
    assert recommendation.target_fader_db <= 0.0
    assert recommendation.blockers == ()
    assert recommendation.evidence["ayaic_level_plane"] is True
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_bus_group_dca_matrix_cache_no_live_authority"] is True
    assert recommendation.evidence["current_state_cache_freshness"]["fresh_field_count"] == 7
    assert recommendation.evidence["source_identity_downstream_trust"]["status"] == (
        "legacy_unverified"
    )

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
    assert proposal["evidence"]["current_state_cache_freshness_status"] == "fresh"
    assert proposal["evidence"]["current_bus_group_dca_matrix_cache_no_live_authority"] is True


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


def test_source_identity_hold_blocks_bus_group_balance_proposal():
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
            current_state_cache_freshness={"status": "fresh", "read_only": True},
            source_identity_trust_state="hold_for_operator_review",
            source_identity_trust_confidence=0.52,
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
    assert trust["source_id"] == "group:drums"
    assert trust["source_type"] == "drums"
    assert trust["status"] == "held_for_operator_review"
    assert trust["usable_for_downstream"] is False
    proposal = recommendation.to_operator_proposal()
    assert proposal["can_apply_live"] is False
    assert "requested_change" not in proposal


def test_stale_source_identity_replay_context_blocks_bus_group_balance_proposal():
    analysis = BusGroupBalanceAnalysisInput(
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
    recommendation = analyze_bus_group_balance_need(analysis)

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
    assert build_bus_group_balance_proposals([analysis]) == ()


def test_stale_current_bus_group_state_blocks_proposal_only_recommendation():
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
            current_state_cache_freshness={
                "status": "stale",
                "field_count": 7,
                "fresh_field_count": 4,
                "stale_field_count": 3,
                "stale_fields": ["bus_fader_db", "assignment_members", "matrix_send_db"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "bus_group_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "stale"
    assert recommendation.evidence["current_state_cache_freshness"]["stale_fields"] == [
        "bus_fader_db",
        "assignment_members",
        "matrix_send_db",
    ]
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_live_authority_current_bus_group_state_blocks_proposal_only_recommendation():
    analysis = BusGroupBalanceAnalysisInput(
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
        current_state_cache_freshness={
            "status": "fresh",
            "field_count": 7,
            "fresh_field_count": 7,
            "read_only": False,
            "live_write": True,
            "queries_console": True,
        },
    )
    recommendation = analyze_bus_group_balance_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "bus_group_current_state_cache_live_authority_present" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_bus_group_dca_matrix_cache_no_live_authority"] is False
    assert recommendation.evidence["current_state_cache_freshness"]["live_write"] is True
    assert recommendation.to_operator_proposal()["can_apply_live"] is False
    assert build_bus_group_balance_proposals([analysis]) == ()


def test_missing_current_bus_group_state_blocks_proposal_only_recommendation():
    analysis = BusGroupBalanceAnalysisInput(
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
    recommendation = analyze_bus_group_balance_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "bus_group_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "not_provided"
    assert build_bus_group_balance_proposals([analysis]) == ()


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
                current_state_cache_freshness=_fresh(7),
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


def test_build_bus_group_balance_proposals_filters_stale_state_items():
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
                current_state_cache_freshness={"status": "stale"},
            ),
        ]
    )

    assert proposals == ()


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
    assert "backend/source_identity_downstream_gate.py" in mix_balance.support_paths
    assert "backend/source_identity_downstream_gate.py" in console.support_paths
    assert contracts["bus_group_balance_proposal"]["proposal_contract"]["value_type"] == "bus_group_balance"
    assert (
        "current_bus_group_dca_matrix_cache_freshness"
        in contracts["bus_group_balance_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "current_bus_group_dca_matrix_cache_no_live_authority"
        in contracts["bus_group_balance_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "source_identity_downstream_trust"
        in contracts["bus_group_balance_proposal"]["proposal_contract"][
            "safety_evidence_required"
        ]
    )
