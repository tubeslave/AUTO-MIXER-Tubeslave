from pathlib import Path

from backend.master_bus_optimization_proposal import (
    MasterBusAnalysisInput,
    analyze_master_bus_need,
    build_master_bus_proposals,
    resolve_target_profile,
)
from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_pipeline_profile, get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability


def test_master_bus_proposal_module_is_transport_free():
    source = Path("backend/master_bus_optimization_proposal.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "set_main",
        "set_bus",
        "set_limiter",
        "mixer_client",
    ]
    for token in forbidden:
        assert token not in source


def test_healthy_master_creates_proposal_only_recommendation():
    recommendation = analyze_master_bus_need(
        MasterBusAnalysisInput(
            bus_label="Main L/R",
            target_profile="live",
            integrated_lufs=-18.7,
            short_term_lufs=-18.3,
            true_peak_dbfs=-4.5,
            crest_factor_db=10.5,
            lra_db=5.0,
            spectral_tilt_db=1.8,
            low_end_excess_db=1.4,
            harshness_db=0.4,
            stereo_correlation=0.1,
            confidence=0.9,
            persistence_cycles=3,
            current_state_cache_freshness={
                "status": "fresh",
                "field_count": 8,
                "fresh_field_count": 8,
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "master_bus_proposal"
    assert recommendation.target_profile == "live"
    assert recommendation.target_integrated_lufs == -18.0
    assert recommendation.max_true_peak_dbfs == -2.0
    assert 0.0 < recommendation.suggested_trim_db <= 1.0
    assert recommendation.true_peak_headroom_db >= 2.0
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["lufs_only_loudness_up_is_not_listening_master"] is True
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_state_cache_freshness"]["fresh_field_count"] == 8

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "master_bus_optimization_proposal"
    assert proposal["kind"] == "master_bus"
    assert proposal["can_import_to_queue"] is False
    assert proposal["can_apply_live"] is False
    assert "requested_change" not in proposal
    assert "requested_changes" not in proposal
    assert "unsupported_value_type:master_bus" in proposal["blockers"]
    assert proposal["proposed_action"]["suggested_trim_db"] == recommendation.suggested_trim_db
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["current_state_cache_freshness_status"] == "fresh"


def test_peak_limited_loudness_up_is_analysis_only_without_dynamics_proof():
    recommendation = analyze_master_bus_need(
        MasterBusAnalysisInput(
            target_profile="streaming",
            integrated_lufs=-24.0,
            short_term_lufs=-23.5,
            true_peak_dbfs=-1.2,
            crest_factor_db=8.0,
            lra_db=5.0,
            confidence=0.95,
            persistence_cycles=3,
            limiter_or_dynamics_proof=False,
        )
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.suggested_trim_db == 0.0
    assert "true_peak_headroom_insufficient" in recommendation.blockers
    assert "master_loudness_requires_approved_limiter_or_dynamics_stage" in recommendation.blockers
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_low_end_and_harshness_create_broad_eq_fields():
    recommendation = analyze_master_bus_need(
        MasterBusAnalysisInput(
            target_profile="live",
            integrated_lufs=-18.5,
            short_term_lufs=-18.0,
            true_peak_dbfs=-5.5,
            crest_factor_db=10.0,
            lra_db=5.0,
            spectral_tilt_db=3.0,
            low_end_excess_db=4.0,
            harshness_db=3.0,
            stereo_correlation=0.3,
            confidence=0.85,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.broad_eq_tilt_db < 0.0
    assert recommendation.low_shelf_db < 0.0
    assert recommendation.high_shelf_db < 0.0
    assert recommendation.evidence["low_end_excess_db"] == 4.0
    assert recommendation.evidence["harshness_db"] == 3.0


def test_stale_current_master_bus_state_blocks_proposal_only_recommendation():
    recommendation = analyze_master_bus_need(
        MasterBusAnalysisInput(
            bus_label="Main L/R",
            target_profile="live",
            integrated_lufs=-18.7,
            short_term_lufs=-18.3,
            true_peak_dbfs=-4.5,
            crest_factor_db=10.5,
            lra_db=5.0,
            spectral_tilt_db=1.8,
            low_end_excess_db=1.4,
            harshness_db=0.4,
            stereo_correlation=0.1,
            confidence=0.9,
            persistence_cycles=3,
            current_state_cache_freshness={
                "status": "stale",
                "field_count": 8,
                "fresh_field_count": 5,
                "stale_field_count": 3,
                "stale_fields": ["limiter_ceiling_dbfs", "main_eq", "bus_compressor"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "master_bus_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "stale"
    assert recommendation.evidence["current_state_cache_freshness"]["stale_fields"] == [
        "limiter_ceiling_dbfs",
        "main_eq",
        "bus_compressor",
    ]
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_build_master_bus_proposals_filters_to_proposal_only_items():
    proposals = build_master_bus_proposals(
        [
            MasterBusAnalysisInput(
                bus_id="main_lr",
                target_profile="live",
                integrated_lufs=-18.7,
                true_peak_dbfs=-4.5,
                confidence=0.9,
                persistence_cycles=3,
            ),
            MasterBusAnalysisInput(
                bus_id="main_lr",
                target_profile="live",
                integrated_lufs=-18.0,
                true_peak_dbfs=-4.0,
                confidence=0.9,
                persistence_cycles=3,
            ),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["target"] == "main_lr"
    assert proposals[0]["can_import_to_queue"] is False


def test_build_master_bus_proposals_filters_stale_state_items():
    proposals = build_master_bus_proposals(
        [
            MasterBusAnalysisInput(
                bus_id="main_lr",
                target_profile="live",
                integrated_lufs=-18.7,
                true_peak_dbfs=-4.5,
                confidence=0.9,
                persistence_cycles=3,
                current_state_cache_freshness={"status": "stale"},
            ),
        ]
    )

    assert proposals == ()


def test_master_bus_helpers_resolve_reference_target_safely():
    profile = resolve_target_profile("streaming", reference_lufs=-9.0)

    assert profile.id == "streaming"
    assert profile.target_integrated_lufs == -12.0
    assert profile.max_true_peak_dbfs == -1.0


def test_master_bus_proposal_is_registered_as_goal_capability():
    method = get_processing_method("master_bus_optimization_proposal")
    main = get_pipeline_profile("main_live_assist")
    capability = get_project_goal_capability("master_bus_optimization")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.domain == "master_bus"
    assert method.can_write_live is False
    assert main is not None
    assert "master_bus_optimization_proposal" in main.proposal_method_ids
    assert capability is not None
    assert capability.status == "proposal_only"
    assert capability.autonomy_mode_ceiling == "assisted"
    assert "master_bus_optimization_proposal" in capability.method_ids
    assert contracts["master_bus_optimization_proposal"]["proposal_contract"]["value_type"] == "master_bus"
    assert (
        "current_main_bus_processing_cache_freshness"
        in contracts["master_bus_optimization_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
