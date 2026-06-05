from pathlib import Path

from backend.compressor_dynamics_proposal import (
    CompressorAnalysisInput,
    analyze_compressor_need,
    build_compressor_proposals,
    classify_source_family,
    estimate_crest_factor_db,
)
from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_pipeline_profile, get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability


def test_compressor_proposal_module_is_transport_free():
    source = Path("backend/compressor_dynamics_proposal.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "set_compressor(",
        "set_compressor_gain",
        "mixer_client",
    ]
    for token in forbidden:
        assert token not in source


def test_dynamic_vocal_creates_proposal_only_compressor_recommendation():
    recommendation = analyze_compressor_need(
        CompressorAnalysisInput(
            channel_id=5,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            crest_factor_db=15.0,
            lra_db=8.5,
            rms_dbfs=-23.0,
            peak_dbfs=-8.0,
            lufs_short=-21.5,
            true_peak_headroom_db=5.0,
            existing_gain_reduction_db=1.0,
            contribution_share=0.78,
            persistence_cycles=3,
            transient_density=0.35,
            loudness_stability=0.34,
            current_state_cache_freshness={
                "status": "fresh",
                "field_count": 10,
                "fresh_field_count": 10,
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "compressor_proposal"
    assert recommendation.source_family == "vocal"
    assert -50.0 <= recommendation.threshold_db <= -6.0
    assert 1.2 <= recommendation.ratio <= 8.0
    assert 2.0 <= recommendation.attack_ms <= 120.0
    assert 40.0 <= recommendation.release_ms <= 1200.0
    assert 0.0 <= recommendation.makeup_gain_db <= 3.0
    assert recommendation.makeup_gain_db <= recommendation.evidence["true_peak_headroom_db"] - 1.0
    assert recommendation.confidence >= 0.42
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_state_cache_freshness"]["fresh_field_count"] == 10
    assert recommendation.evidence["source_identity_downstream_trust"]["status"] == "legacy_unverified"

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "compressor_dynamics_proposal"
    assert proposal["can_import_to_queue"] is False
    assert proposal["can_apply_live"] is False
    assert "requested_change" not in proposal
    assert "unsupported_value_type:compressor" in proposal["blockers"]
    assert proposal["proposed_action"]["makeup_gain_db"] == recommendation.makeup_gain_db
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["current_state_cache_freshness_status"] == "fresh"


def test_flat_or_already_controlled_source_is_analysis_only():
    recommendation = analyze_compressor_need(
        CompressorAnalysisInput(
            channel_id=9,
            channel_label="Pad",
            source_role="keys",
            crest_factor_db=4.0,
            lra_db=2.0,
            rms_dbfs=-24.0,
            peak_dbfs=-20.0,
            true_peak_headroom_db=8.0,
            existing_gain_reduction_db=0.0,
            loudness_stability=0.9,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "compression_not_needed" in recommendation.blockers


def test_insufficient_headroom_blocks_compressor_makeup_proposal():
    recommendation = analyze_compressor_need(
        CompressorAnalysisInput(
            channel_id=2,
            source_role="bass",
            crest_factor_db=9.0,
            lra_db=7.0,
            rms_dbfs=-18.0,
            peak_dbfs=-2.0,
            true_peak_headroom_db=0.4,
            contribution_share=0.8,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "insufficient_true_peak_headroom_for_makeup" in recommendation.blockers
    assert recommendation.makeup_gain_db == 0.0


def test_stale_current_compressor_state_blocks_proposal_only_recommendation():
    recommendation = analyze_compressor_need(
        CompressorAnalysisInput(
            channel_id=5,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            crest_factor_db=15.0,
            lra_db=8.5,
            rms_dbfs=-23.0,
            peak_dbfs=-8.0,
            lufs_short=-21.5,
            true_peak_headroom_db=5.0,
            existing_gain_reduction_db=1.0,
            contribution_share=0.78,
            persistence_cycles=3,
            transient_density=0.35,
            loudness_stability=0.34,
            current_state_cache_freshness={
                "status": "stale",
                "field_count": 10,
                "fresh_field_count": 7,
                "stale_field_count": 3,
                "stale_fields": ["threshold_db", "ratio", "makeup_db"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "compressor_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "stale"
    assert recommendation.evidence["current_state_cache_freshness"]["stale_fields"] == [
        "threshold_db",
        "ratio",
        "makeup_db",
    ]
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_source_identity_hold_blocks_downstream_compressor_proposal():
    recommendation = analyze_compressor_need(
        CompressorAnalysisInput(
            channel_id=5,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            crest_factor_db=15.0,
            lra_db=8.5,
            rms_dbfs=-23.0,
            peak_dbfs=-8.0,
            lufs_short=-21.5,
            true_peak_headroom_db=5.0,
            existing_gain_reduction_db=1.0,
            contribution_share=0.78,
            persistence_cycles=3,
            transient_density=0.35,
            loudness_stability=0.34,
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


def test_stale_source_identity_replay_context_blocks_downstream_compressor_proposal():
    recommendation = analyze_compressor_need(
        CompressorAnalysisInput(
            channel_id=5,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            crest_factor_db=15.0,
            lra_db=8.5,
            rms_dbfs=-23.0,
            peak_dbfs=-8.0,
            lufs_short=-21.5,
            true_peak_headroom_db=5.0,
            existing_gain_reduction_db=1.0,
            contribution_share=0.78,
            persistence_cycles=3,
            transient_density=0.35,
            loudness_stability=0.34,
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


def test_compressor_helpers_classify_roles_and_derive_crest_factor():
    assert classify_source_family("Lead Vocal") == "vocal"
    assert classify_source_family("Bass DI") == "bass"
    assert classify_source_family("Snare Top") == "drum"
    assert estimate_crest_factor_db(-6.0, -21.0) == 15.0


def test_build_compressor_proposals_filters_to_proposal_only_items():
    proposals = build_compressor_proposals(
        [
            CompressorAnalysisInput(
                channel_id=5,
                source_role="lead_vocal",
                crest_factor_db=15.0,
                lra_db=8.0,
                rms_dbfs=-22.0,
                peak_dbfs=-7.0,
                true_peak_headroom_db=5.0,
                contribution_share=0.7,
                persistence_cycles=3,
            ),
            CompressorAnalysisInput(
                channel_id=9,
                source_role="keys",
                crest_factor_db=3.5,
                lra_db=2.0,
                rms_dbfs=-24.0,
                peak_dbfs=-20.5,
            ),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["channel"] == 5


def test_build_compressor_proposals_filters_stale_state_items():
    proposals = build_compressor_proposals(
        [
            CompressorAnalysisInput(
                channel_id=5,
                source_role="lead_vocal",
                crest_factor_db=15.0,
                lra_db=8.0,
                rms_dbfs=-22.0,
                peak_dbfs=-7.0,
                true_peak_headroom_db=5.0,
                contribution_share=0.7,
                persistence_cycles=3,
                current_state_cache_freshness={"status": "stale"},
            ),
        ]
    )

    assert proposals == ()


def test_compressor_proposal_is_registered_as_goal_capability():
    method = get_processing_method("compressor_dynamics_proposal")
    main = get_pipeline_profile("main_live_assist")
    capability = get_project_goal_capability("compression")
    console = get_project_goal_capability("console_compressor_control")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_write_live is False
    assert main is not None
    assert "compressor_dynamics_proposal" in main.proposal_method_ids
    assert capability is not None
    assert capability.status == "proposal_only"
    assert "compressor_dynamics_proposal" in capability.method_ids
    assert "backend/source_identity_downstream_gate.py" in capability.support_paths
    assert console is not None
    assert console.status == "proposal_only"
    assert "backend/source_identity_downstream_gate.py" in console.support_paths
    assert contracts["compressor_dynamics_proposal"]["proposal_contract"]["value_type"] == "compressor"
    assert (
        "source_identity_downstream_trust"
        in contracts["compressor_dynamics_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
