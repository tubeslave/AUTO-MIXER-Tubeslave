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
    assert payload["mutates_mixer"] is False


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
    _assert_top_level_denial(proposal)
    assert "requested_change" not in proposal
    assert "requested_changes" not in proposal
    assert "unsupported_value_type:bus_group_balance" in proposal["blockers"]
    assert proposal["proposed_action"]["suggested_delta_db"] == recommendation.suggested_delta_db
    level_plane_summary = proposal["ayaic_level_plane_summary"]
    assert level_plane_summary["summary_type"] == "ayaic_bus_group_level_plane_review"
    assert level_plane_summary["ayaic_level_plane"] is True
    assert level_plane_summary["target_plane_lufs"] == -25.0
    assert level_plane_summary["target_plane_reason"] == "strong_group"
    assert level_plane_summary["source_family"] == "drums"
    assert level_plane_summary["current_integrated_lufs"] == recommendation.current_integrated_lufs
    assert level_plane_summary["summed_child_lufs"] == recommendation.summed_child_lufs
    assert level_plane_summary["child_to_bus_delta_db"] == 0.5
    assert level_plane_summary["suggested_delta_db"] == recommendation.suggested_delta_db
    assert level_plane_summary["target_fader_db"] == recommendation.target_fader_db
    assert level_plane_summary["true_peak_headroom_db"] == recommendation.true_peak_headroom_db
    assert level_plane_summary["main_bus_headroom_db"] == 6.0
    assert level_plane_summary["feedback_risk"] == 0.1
    assert level_plane_summary["assignment_confidence"] == 0.9
    assert level_plane_summary["child_count"] == 8
    assert level_plane_summary["dominant_child_share"] == 0.35
    assert level_plane_summary["readback_required_before_promotion"] is True
    assert level_plane_summary["rollback_required_before_promotion"] is True
    assert level_plane_summary["operator_review_required"] is True
    assert level_plane_summary["read_only_or_dry_run_only"] is True
    assert level_plane_summary["live_apply_allowed"] is False
    assert level_plane_summary["queue_import_allowed"] is False
    _assert_top_level_denial(level_plane_summary)
    assert level_plane_summary["can_authorize_live"] is False
    assert proposal["proposed_action"]["ayaic_level_plane_summary"] == level_plane_summary
    assert proposal["evidence"]["ayaic_level_plane_summary"] == level_plane_summary
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["current_state_cache_freshness_status"] == "fresh"
    assert proposal["evidence"]["current_bus_group_dca_matrix_cache_no_live_authority"] is True
    assert proposal["evidence"]["ayaic_target_plane"]["target_plane_lufs"] == -25.0
    assert proposal["evidence"]["ayaic_target_plane"]["target_plane_reason"] == "strong_group"
    assert proposal["evidence"]["assignment_readback"]["child_count"] == 8
    assert proposal["evidence"]["assignment_readback"]["summed_child_lufs"] == -25.6
    assert proposal["evidence"]["assignment_readback"]["assignment_confidence"] == 0.9
    assert proposal["evidence"]["bounded_bus_group_delta"] == {
        "bus_id": "group:drums",
        "bus_kind": "group",
        "current_fader_db": -5.0,
        "target_fader_db": recommendation.target_fader_db,
        "suggested_delta_db": recommendation.suggested_delta_db,
        "max_up_delta_db": 1.0,
        "max_down_delta_db": 2.5,
        "fader_ceiling_db": 0.0,
        "true_peak_headroom_db": recommendation.true_peak_headroom_db,
        "main_bus_headroom_db": 6.0,
        "bounded": True,
        "read_only_or_dry_run_only": True,
        "live_write": False,
    }
    assert proposal["evidence"]["current_bus_fader"]["current_fader_db"] == -5.0
    assert proposal["evidence"]["current_bus_fader"]["suggested_delta_db"] == (
        recommendation.suggested_delta_db
    )
    assert proposal["evidence"]["rollback_bus_group_dca_matrix_readback"] == {
        "bus_id": "group:drums",
        "bus_kind": "group",
        "rollback_fader_db": -5.0,
        "assignment_map_readback_required": True,
        "cache_freshness_status": "fresh",
        "cache_no_live_authority": True,
        "source": "cached_or_operator_supplied_bus_group_dca_matrix_state",
        "required_before_live_promotion": True,
        "read_only_or_dry_run_only": True,
        "live_write": False,
    }
    assert proposal["evidence"]["current_bus_group_dca_matrix_cache_freshness"][
        "status"
    ] == "fresh"
    correction = proposal["operator_correction"]
    assert correction["objective_family"] == "fader_balance"
    assert correction["bounded_delta"]["delta_db"] == recommendation.suggested_delta_db
    assert correction["future_wing_readback_contract"]["target_kind"] == (
        "bus_group_dca_matrix_fader"
    )
    assert correction["future_wing_readback_contract"][
        "required_before_live_promotion"
    ] is True
    assert correction["read_only_or_dry_run_only"] is True
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    assert "main_bus_headroom" in correction["headroom_feedback_true_peak_checks"][
        "required_checks"
    ]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "ayaic_target_plane"
    ] == proposal["evidence"]["ayaic_target_plane"]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "integrated_lufs"
    ] == proposal["evidence"]["integrated_lufs"]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "summed_child_lufs"
    ] == proposal["evidence"]["summed_child_lufs"]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "assignment_readback"
    ] == proposal["evidence"]["assignment_readback"]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "current_bus_fader"
    ] == proposal["evidence"]["current_bus_fader"]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "current_bus_group_dca_matrix_cache_freshness"
    ] == proposal["evidence"]["current_bus_group_dca_matrix_cache_freshness"]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "bounded_bus_group_delta"
    ] == proposal["evidence"]["bounded_bus_group_delta"]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "rollback_bus_group_dca_matrix_readback"
    ] == proposal["evidence"]["rollback_bus_group_dca_matrix_readback"]
    assert correction["source_evidence"]["evidence"]["ayaic_level_plane_summary"] == level_plane_summary
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_top_level_denial(source_evidence)
    assert source_evidence["ayaic_level_plane_summary"] == level_plane_summary
    assert source_evidence["ayaic_target_plane"] == proposal["evidence"][
        "ayaic_target_plane"
    ]
    assert source_evidence["assignment_readback"] == proposal["evidence"][
        "assignment_readback"
    ]
    assert source_evidence["bounded_bus_group_delta"] == proposal["evidence"][
        "bounded_bus_group_delta"
    ]
    assert source_evidence["current_bus_fader"] == proposal["evidence"][
        "current_bus_fader"
    ]
    assert source_evidence["rollback_bus_group_dca_matrix_readback"] == (
        proposal["evidence"]["rollback_bus_group_dca_matrix_readback"]
    )
    assert source_evidence["current_bus_group_dca_matrix_cache_freshness"] == (
        proposal["evidence"]["current_bus_group_dca_matrix_cache_freshness"]
    )
    assert source_evidence["current_bus_group_dca_matrix_cache_no_live_authority"] is True
    completeness = correction["source_evidence"]["evidence_completeness"]
    assert completeness["status"] == "complete"
    assert completeness["missing"] == []
    assert correction["operator_proposal_review_gate"]["evidence_gate_status"] == (
        "required_evidence_complete_operator_review_still_required"
    )
    assert "ayaic_level_plane_summary" in correction["operator_explanation_packet"][
        "source_evidence_summary"
    ]["structured_evidence_fields"]


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
    _assert_top_level_denial(recommendation.ayaic_level_plane_summary())
    _assert_top_level_denial(recommendation.to_operator_proposal())


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
    _assert_top_level_denial(recommendation.ayaic_level_plane_summary())
    _assert_top_level_denial(proposal)
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
    _assert_top_level_denial(recommendation.ayaic_level_plane_summary())
    _assert_top_level_denial(recommendation.to_operator_proposal())


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
    _assert_top_level_denial(recommendation.ayaic_level_plane_summary())
    _assert_top_level_denial(recommendation.to_operator_proposal())
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
    _assert_top_level_denial(recommendation.ayaic_level_plane_summary())
    _assert_top_level_denial(recommendation.to_operator_proposal())
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
    _assert_top_level_denial(proposals[0])
    _assert_top_level_denial(proposals[0]["ayaic_level_plane_summary"])


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
        "bounded_bus_group_delta"
        in contracts["bus_group_balance_proposal"]["proposal_contract"][
            "safety_evidence_required"
        ]
    )
    assert (
        "rollback_bus_group_dca_matrix_readback"
        in contracts["bus_group_balance_proposal"]["proposal_contract"][
            "safety_evidence_required"
        ]
    )
    assert (
        "source_identity_downstream_trust"
        in contracts["bus_group_balance_proposal"]["proposal_contract"][
            "safety_evidence_required"
        ]
    )
