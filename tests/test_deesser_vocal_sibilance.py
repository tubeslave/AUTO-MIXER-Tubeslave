from pathlib import Path

from backend.deesser_vocal_sibilance import (
    DeesserAnalysisInput,
    analyze_deesser_need,
    build_deesser_proposals,
    estimate_sibilance_ratio,
)
from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_processing_method
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


def test_deesser_module_is_transport_free():
    source = Path("backend/deesser_vocal_sibilance.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "set_channel_deesser",
        "set_deesser",
        "set_eq(",
        "set_compressor(",
    ]
    for token in forbidden:
        assert token not in source


def test_vocal_sibilance_creates_proposal_only_recommendation():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=7,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            arrangement_role="lead",
            band_energy_db={
                "low_mid": -38.0,
                "mid": -30.0,
                "presence": -24.0,
                "sibilance": -16.0,
                "air": -18.0,
            },
            sibilance_excess_db=4.5,
            peak_dbfs=-7.0,
            contribution_share=0.72,
            persistence_cycles=3,
            true_peak_headroom_db=5.0,
            current_state_cache_freshness={
                "status": "fresh",
                "field_count": 8,
                "fresh_field_count": 8,
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
            arrangement_context_cache_freshness={
                "status": "fresh",
                "field_count": 4,
                "fresh_field_count": 4,
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "deesser_proposal"
    assert 5500.0 <= recommendation.frequency_hz <= 10500.0
    assert -42.0 <= recommendation.threshold_db <= -12.0
    assert 1.5 <= recommendation.ratio <= 6.0
    assert 1.0 <= recommendation.max_gain_reduction_db <= 8.0
    assert recommendation.confidence >= 0.45
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["bounded_attenuation_only"] is True
    assert recommendation.evidence["makeup_gain_db"] == 0.0
    assert recommendation.evidence["boost_allowed"] is False
    assert recommendation.evidence["max_gain_reduction_db"] == (
        recommendation.max_gain_reduction_db
    )
    assert recommendation.evidence["arrangement_role"] == "lead"
    assert recommendation.evidence["arrangement_deesser_profile"] == "lead_vocal"
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_state_cache_freshness"]["fresh_field_count"] == 8
    assert recommendation.evidence["arrangement_context_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["arrangement_context_cache_freshness"]["fresh_field_count"] == 4
    assert recommendation.evidence["source_identity_downstream_trust"]["status"] == "legacy_unverified"

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "deesser_vocal_sibilance"
    assert proposal["can_import_to_queue"] is False
    assert proposal["can_apply_live"] is False
    assert "requested_change" not in proposal
    assert "unsupported_value_type:deesser" in proposal["blockers"]
    assert proposal["proposed_action"]["max_gain_reduction_db"] == (
        recommendation.max_gain_reduction_db
    )
    assert proposal["proposed_action"]["makeup_gain_db"] == 0.0
    assert proposal["proposed_action"]["boost_allowed"] is False
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["bounded_attenuation_only"] is True
    assert proposal["evidence"]["current_state_cache_freshness_status"] == "fresh"
    assert proposal["evidence"]["arrangement_context_cache_freshness_status"] == "fresh"


def test_backing_vocal_deesser_profile_is_less_aggressive_than_lead():
    shared = {
        "channel_id": 7,
        "channel_label": "Vox",
        "source_role": "lead_vocal",
        "band_energy_db": {
            "low_mid": -38.0,
            "mid": -30.0,
            "presence": -24.0,
            "sibilance": -16.0,
            "air": -18.0,
        },
        "sibilance_excess_db": 4.5,
        "peak_dbfs": -7.0,
        "contribution_share": 0.72,
        "persistence_cycles": 3,
        "true_peak_headroom_db": 5.0,
        "current_state_cache_freshness": _fresh(8),
        "arrangement_context_cache_freshness": _fresh(4),
    }

    lead = analyze_deesser_need(
        DeesserAnalysisInput(**shared, arrangement_role="lead")
    )
    tucked = analyze_deesser_need(
        DeesserAnalysisInput(**shared, arrangement_role="tucked backing vocal")
    )

    assert lead.status == "proposal_only"
    assert tucked.status == "proposal_only"
    assert tucked.evidence["arrangement_deesser_profile"] == "tucked_or_backing_vocal"
    assert tucked.ratio < lead.ratio
    assert tucked.threshold_db > lead.threshold_db
    assert tucked.max_gain_reduction_db < lead.max_gain_reduction_db
    assert tucked.confidence < lead.confidence


def test_stale_arrangement_context_blocks_deesser_proposal():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=7,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            arrangement_role="lead",
            band_energy_db={
                "low_mid": -38.0,
                "mid": -30.0,
                "presence": -24.0,
                "sibilance": -16.0,
                "air": -18.0,
            },
            sibilance_excess_db=4.5,
            peak_dbfs=-7.0,
            contribution_share=0.72,
            persistence_cycles=3,
            true_peak_headroom_db=5.0,
            arrangement_context_cache_freshness={
                "status": "stale",
                "field_count": 4,
                "fresh_field_count": 2,
                "stale_field_count": 2,
                "stale_fields": ["arrangement_role", "section_role"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "deesser_arrangement_context_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["arrangement_context_cache_freshness_status"] == "stale"
    assert recommendation.evidence["arrangement_context_cache_freshness"]["stale_fields"] == [
        "arrangement_role",
        "section_role",
    ]
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_non_vocal_sibilance_is_not_deesser_proposal():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=3,
            channel_label="OH L",
            source_role="overhead",
            band_energy_db={"sibilance": -12.0, "presence": -24.0},
            sibilance_excess_db=7.0,
            contribution_share=0.8,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "source_not_vocal" in recommendation.blockers


def test_low_sibilance_vocal_is_analysis_only():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=8,
            source_role="backing_vocal",
            band_energy_db={"sibilance": -42.0, "presence": -25.0, "mid": -24.0},
            sibilance_excess_db=0.5,
            persistence_cycles=1,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "sibilance_below_threshold" in recommendation.blockers


def test_stale_current_deesser_state_blocks_proposal_only_recommendation():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=7,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            band_energy_db={
                "low_mid": -38.0,
                "mid": -30.0,
                "presence": -24.0,
                "sibilance": -16.0,
                "air": -18.0,
            },
            sibilance_excess_db=4.5,
            peak_dbfs=-7.0,
            contribution_share=0.72,
            persistence_cycles=3,
            true_peak_headroom_db=5.0,
            current_state_cache_freshness={
                "status": "stale",
                "field_count": 8,
                "fresh_field_count": 5,
                "stale_field_count": 3,
                "stale_fields": ["frequency_hz", "threshold_db", "ratio"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "deesser_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "stale"
    assert recommendation.evidence["current_state_cache_freshness"]["stale_fields"] == [
        "frequency_hz",
        "threshold_db",
        "ratio",
    ]
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_missing_current_deesser_state_blocks_proposal_only_recommendation():
    analysis = DeesserAnalysisInput(
        channel_id=7,
        channel_label="Lead Vox",
        source_role="lead_vocal",
        arrangement_role="lead",
        band_energy_db={
            "low_mid": -38.0,
            "mid": -30.0,
            "presence": -24.0,
            "sibilance": -16.0,
            "air": -18.0,
        },
        sibilance_excess_db=4.5,
        peak_dbfs=-7.0,
        contribution_share=0.72,
        persistence_cycles=3,
        true_peak_headroom_db=5.0,
        arrangement_context_cache_freshness=_fresh(4),
    )
    recommendation = analyze_deesser_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "deesser_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "not_provided"
    assert build_deesser_proposals([analysis]) == ()


def test_missing_arrangement_context_blocks_deesser_proposal_only_recommendation():
    analysis = DeesserAnalysisInput(
        channel_id=7,
        channel_label="Lead Vox",
        source_role="lead_vocal",
        arrangement_role="lead",
        band_energy_db={
            "low_mid": -38.0,
            "mid": -30.0,
            "presence": -24.0,
            "sibilance": -16.0,
            "air": -18.0,
        },
        sibilance_excess_db=4.5,
        peak_dbfs=-7.0,
        contribution_share=0.72,
        persistence_cycles=3,
        true_peak_headroom_db=5.0,
        current_state_cache_freshness=_fresh(8),
    )
    recommendation = analyze_deesser_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "deesser_arrangement_context_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["arrangement_context_cache_freshness_status"] == "not_provided"
    assert build_deesser_proposals([analysis]) == ()


def test_live_authority_current_deesser_state_blocks_proposal_only_recommendation():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=7,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            arrangement_role="lead",
            band_energy_db={
                "low_mid": -38.0,
                "mid": -30.0,
                "presence": -24.0,
                "sibilance": -16.0,
                "air": -18.0,
            },
            sibilance_excess_db=4.5,
            peak_dbfs=-7.0,
            contribution_share=0.72,
            persistence_cycles=3,
            true_peak_headroom_db=5.0,
            current_state_cache_freshness={
                "status": "fresh",
                "field_count": 8,
                "fresh_field_count": 8,
                "read_only": False,
                "live_write": True,
                "queries_console": True,
            },
            arrangement_context_cache_freshness=_fresh(4),
        )
    )

    assert recommendation.status == "analysis_only"
    assert "deesser_current_state_cache_live_authority_present" in recommendation.blockers
    assert (
        recommendation.evidence[
            "current_dynamics_or_dynamic_eq_cache_no_live_authority"
        ]
        is False
    )
    proposal = recommendation.to_operator_proposal()
    assert proposal["can_apply_live"] is False
    assert "deesser_current_state_cache_live_authority_present" in proposal["blockers"]


def test_live_authority_arrangement_context_blocks_deesser_proposal_only_recommendation():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=7,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            arrangement_role="lead",
            band_energy_db={
                "low_mid": -38.0,
                "mid": -30.0,
                "presence": -24.0,
                "sibilance": -16.0,
                "air": -18.0,
            },
            sibilance_excess_db=4.5,
            peak_dbfs=-7.0,
            contribution_share=0.72,
            persistence_cycles=3,
            true_peak_headroom_db=5.0,
            current_state_cache_freshness=_fresh(8),
            arrangement_context_cache_freshness={
                "status": "fresh",
                "field_count": 4,
                "fresh_field_count": 4,
                "read_only": False,
                "live_write": True,
                "queries_console": True,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert (
        "deesser_arrangement_context_cache_live_authority_present"
        in recommendation.blockers
    )
    assert (
        recommendation.evidence[
            "current_arrangement_context_cache_no_live_authority"
        ]
        is False
    )
    proposal = recommendation.to_operator_proposal()
    assert proposal["can_apply_live"] is False
    assert (
        "deesser_arrangement_context_cache_live_authority_present"
        in proposal["blockers"]
    )


def test_source_identity_hold_blocks_downstream_deesser_proposal():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=7,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            band_energy_db={
                "low_mid": -38.0,
                "mid": -30.0,
                "presence": -24.0,
                "sibilance": -16.0,
                "air": -18.0,
            },
            sibilance_excess_db=4.5,
            peak_dbfs=-7.0,
            contribution_share=0.72,
            persistence_cycles=3,
            true_peak_headroom_db=5.0,
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
    assert trust["evidence"]["model_mutation"] is False
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_stale_source_identity_replay_context_blocks_downstream_deesser_proposal():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=7,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            band_energy_db={
                "low_mid": -38.0,
                "mid": -30.0,
                "presence": -24.0,
                "sibilance": -16.0,
                "air": -18.0,
            },
            sibilance_excess_db=4.5,
            peak_dbfs=-7.0,
            contribution_share=0.72,
            persistence_cycles=3,
            true_peak_headroom_db=5.0,
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


def test_deesser_ratio_estimator_uses_sibilance_band_share():
    ratio = estimate_sibilance_ratio(
        {"mid": -30.0, "presence": -28.0, "sibilance": -16.0, "air": -28.0}
    )

    assert ratio > 0.5


def test_build_deesser_proposals_filters_to_proposal_only_items():
    proposals = build_deesser_proposals(
        [
            DeesserAnalysisInput(
                channel_id=7,
                source_role="lead_vocal",
                band_energy_db={"sibilance": -15.0, "presence": -25.0},
                sibilance_excess_db=4.0,
                contribution_share=0.7,
                persistence_cycles=3,
                current_state_cache_freshness=_fresh(8),
                arrangement_context_cache_freshness=_fresh(4),
            ),
            DeesserAnalysisInput(
                channel_id=3,
                source_role="overhead",
                band_energy_db={"sibilance": -15.0, "presence": -25.0},
                sibilance_excess_db=4.0,
                contribution_share=0.7,
                persistence_cycles=3,
            ),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["channel"] == 7


def test_build_deesser_proposals_filters_stale_state_items():
    proposals = build_deesser_proposals(
        [
            DeesserAnalysisInput(
                channel_id=7,
                source_role="lead_vocal",
                band_energy_db={"sibilance": -15.0, "presence": -25.0},
                sibilance_excess_db=4.0,
                contribution_share=0.7,
                persistence_cycles=3,
                current_state_cache_freshness={"status": "stale"},
            ),
        ]
    )

    assert proposals == ()


def test_deesser_is_registered_as_proposal_only_goal_capability():
    method = get_processing_method("deesser_vocal_sibilance")
    capability = get_project_goal_capability("deessing")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_write_live is False
    assert capability is not None
    assert capability.status == "proposal_only"
    assert capability.method_ids == ("deesser_vocal_sibilance",)
    assert contracts["deesser_vocal_sibilance"]["proposal_contract"]["value_type"] == "deesser"
    evidence = contracts["deesser_vocal_sibilance"]["proposal_contract"]["safety_evidence_required"]
    assert "source_identity_downstream_trust" in evidence
    assert "current_arrangement_context_cache_freshness" in evidence
    assert "current_arrangement_context_cache_no_live_authority" in evidence
    assert "current_dynamics_or_dynamic_eq_cache_no_live_authority" in evidence
