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
    assert payload["live_write"] is False
    assert payload["queue_mutation"] is False
    assert payload["queue_action"] == "not_queueable_until_promoted"
    assert payload["can_import_to_queue"] is False
    assert payload["can_apply_live"] is False
    assert payload["queue_import_allowed"] is False
    assert payload["live_apply_allowed"] is False
    assert payload["read_only_or_dry_run_only"] is True
    assert payload["live_authority_granted"] is False
    assert payload["readiness_closure_allowed"] is False
    assert payload["mutates_mixer"] is False


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
            pa_feed_readback_available=True,
            pa_feed_safe_for_master_change=True,
            pa_feed_cache_freshness=_fresh(),
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
    assert recommendation.evidence["output_feed_role"] == "pa_main"
    assert recommendation.evidence["pa_feed_readback_available"] is True
    assert recommendation.evidence["pa_feed_readback"] == {
        "status": "available",
        "required_before_live_promotion": True,
        "output_feed_role": "pa_main",
        "output_feed_label": "PA main",
        "cache_freshness_status": "fresh",
        "cache_no_live_authority": True,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    assert recommendation.evidence["pa_feed_safe_for_master_change"] is True
    assert recommendation.evidence["pa_feed_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["pa_feed_cache_no_live_authority"] is True
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_main_bus_processing_cache_no_live_authority"] is True
    assert recommendation.evidence["current_state_cache_freshness"]["fresh_field_count"] == 8
    summary = recommendation.master_bus_safety_summary()
    assert summary["status"] == "passed"
    assert summary["target_profile"] == "live"
    assert summary["reference_profile"]["profile_id"] == "live"
    assert summary["suggested_trim_db"] == recommendation.suggested_trim_db
    assert summary["true_peak_headroom_db"] == recommendation.true_peak_headroom_db
    assert summary["loudness_range"]["status"] == "within_target"
    assert summary["loudness_range"]["lra_db"] == 5.0
    assert summary["output_feed_role"] == "pa_main"
    assert summary["pa_feed_readback_available"] is True
    assert summary["pa_feed_readback"] == recommendation.evidence["pa_feed_readback"]
    assert summary["pa_feed_safe_for_master_change"] is True
    assert summary["pa_feed_safety"]["status"] == "passed"
    assert summary["pa_feed_cache_freshness_status"] == "fresh"
    assert summary["pa_feed_cache_no_live_authority"] is True
    assert summary["is_reference_feed"] is False
    assert summary["reference_feed_exclusion"]["status"] == "passed"
    assert summary["current_main_bus_processing_cache_freshness_status"] == "fresh"
    assert summary["current_main_bus_processing_cache_freshness"]["status"] == "fresh"
    assert summary["current_main_bus_processing_cache_no_live_authority"] is True
    assert summary["rollback_master_processing_readback"] == {
        "status": "required_before_live_promotion",
        "target_kind": "main_bus_processing",
        "bus_id": "main_lr",
        "old_value_source": "cached_or_physical_main_bus_processing_read_required_before_promotion",
        "current_state_cache_freshness_status": "fresh",
        "current_state_cache_no_live_authority": True,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    assert summary["rollback_pa_feed_matrix_readback"] == {
        "status": "required_before_live_promotion",
        "target_kind": "matrix_or_pa_feed",
        "output_feed_role": "pa_main",
        "output_feed_label": "PA main",
        "old_value_source": "cached_or_physical_pa_feed_matrix_read_required_before_promotion",
        "pa_feed_readback_available": True,
        "pa_feed_cache_freshness_status": "fresh",
        "pa_feed_cache_no_live_authority": True,
        "reference_feed_excluded": True,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    assert summary["required_safety_checks"] == {
        "integrated_lufs": "required_before_live_promotion",
        "true_peak_headroom": "required_before_live_promotion",
        "crest_factor": "required_before_live_promotion",
        "loudness_range": "required_before_live_promotion",
        "spectral_tilt": "required_before_live_promotion",
        "reference_profile": "required_before_live_promotion",
        "limiter_or_dynamics_proof": "required_before_live_promotion",
        "pa_feed_readback": "required_before_live_promotion",
        "pa_feed_safety": "required_before_live_promotion",
        "pa_feed_cache_freshness": "required_before_live_promotion",
        "pa_feed_cache_no_live_authority": "required_before_live_promotion",
        "reference_feed_exclusion": "required_before_live_promotion",
        "current_main_bus_processing_cache_freshness": "required_before_live_promotion",
        "current_main_bus_processing_cache_no_live_authority": "required_before_live_promotion",
        "rollback_master_processing_readback": "required_before_live_promotion",
        "rollback_pa_feed_matrix_readback": "required_before_live_promotion",
    }
    assert summary["rollback_expectation"] == {
        "required_before_live_promotion": True,
        "restore_previous_master_processing_after_physical_readback": True,
        "restore_previous_limiter_or_dynamics_state_after_physical_readback": True,
        "restore_previous_pa_feed_matrix_state_after_physical_readback": True,
        "manual_operator_rollback_required_if_promoted": True,
    }
    readback = summary["future_wing_readback_contract"]
    assert readback["target_kind"] == "main_bus_processing"
    assert readback["old_value_source"] == (
        "cached_or_physical_main_bus_processing_read_required_before_promotion"
    )
    assert readback["required_before_live_promotion"] is True
    assert readback["requires_real_wing_proof"] is True
    assert readback["requires_operator_approval"] is True
    assert readback["requires_separate_non_fader_adr"] is True
    assert "master_bus_processing_write" in readback["blocked_live_actions"]
    assert "matrix_pa_feed_write" in readback["blocked_live_actions"]
    assert "master_bus_live_apply_quarantined" in summary["safety_blockers"]
    assert "main_bus_processing_read_only_until_promoted" in summary["safety_blockers"]
    assert summary["readback_required_before_promotion"] is True
    assert summary["rollback_required_before_promotion"] is True
    assert summary["live_write"] is False
    assert summary["queue_mutation"] is False
    _assert_top_level_denial(summary)

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "master_bus_optimization_proposal"
    assert proposal["kind"] == "master_bus"
    _assert_top_level_denial(proposal)
    assert "requested_change" not in proposal
    assert "requested_changes" not in proposal
    assert "unsupported_value_type:master_bus" in proposal["blockers"]
    assert "master_bus_live_apply_quarantined" in proposal["blockers"]
    assert "non_fader_live_expansion_requires_separate_adr" in proposal["blockers"]
    assert proposal["proposed_action"]["suggested_trim_db"] == recommendation.suggested_trim_db
    assert proposal["proposed_action"]["reference_profile"] == summary["reference_profile"]
    assert proposal["proposed_action"]["loudness_range"] == summary["loudness_range"]
    assert proposal["proposed_action"]["pa_feed_readback"] == summary["pa_feed_readback"]
    assert proposal["proposed_action"]["pa_feed_safety"] == summary["pa_feed_safety"]
    assert proposal["proposed_action"]["reference_feed_exclusion"] == summary[
        "reference_feed_exclusion"
    ]
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["pa_feed_readback"] == summary["pa_feed_readback"]
    assert proposal["evidence"]["pa_feed_safe_for_master_change"] is True
    assert proposal["evidence"]["pa_feed_cache_no_live_authority"] is True
    assert proposal["evidence"]["pa_feed_safety"] == summary["pa_feed_safety"]
    assert proposal["evidence"]["reference_feed_exclusion"] == summary[
        "reference_feed_exclusion"
    ]
    assert proposal["evidence"]["current_state_cache_freshness_status"] == "fresh"
    assert proposal["evidence"]["current_main_bus_processing_cache_no_live_authority"] is True
    assert proposal["evidence"]["rollback_master_processing_readback"] == summary[
        "rollback_master_processing_readback"
    ]
    assert proposal["evidence"]["rollback_pa_feed_matrix_readback"] == summary[
        "rollback_pa_feed_matrix_readback"
    ]
    assert proposal["master_bus_safety_summary"] == summary
    assert proposal["proposed_action"]["master_bus_safety_summary"] == summary
    assert proposal["evidence"]["master_bus_safety_summary"] == summary
    assert proposal["safety_blockers"] == summary["safety_blockers"]
    assert proposal["required_safety_checks"] == summary["required_safety_checks"]
    assert proposal["rollback_expectation"] == summary["rollback_expectation"]
    assert proposal["future_wing_readback_contract"] == summary[
        "future_wing_readback_contract"
    ]
    correction = proposal["operator_correction"]
    assert correction["objective_family"] == "master_output"
    assert correction["bounded_delta"]["delta_db"] == recommendation.suggested_trim_db
    assert correction["bounded_delta"]["parameter_deltas"]["suggested_trim_db"] == (
        recommendation.suggested_trim_db
    )
    assert correction["bounded_delta"]["parameter_deltas"]["broad_eq_tilt_db"] == (
        recommendation.broad_eq_tilt_db
    )
    assert correction["future_wing_readback_contract"]["target_kind"] == (
        "main_bus_processing"
    )
    assert correction["future_wing_readback_contract"][
        "required_before_live_promotion"
    ] is True
    assert correction["source_evidence"]["evidence_completeness"]["status"] == (
        "complete"
    )
    assert correction["source_evidence"]["evidence_completeness"]["missing"] == []
    assert correction["evidence_gate"]["status"] == (
        "required_evidence_complete_operator_review_still_required"
    )
    assert correction["read_only_or_dry_run_only"] is True
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    assert "pa_feed_safety" in correction["headroom_feedback_true_peak_checks"][
        "required_checks"
    ]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "limiter_or_dynamics_proof"
    ] is False
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "pa_feed_readback"
    ] == summary["pa_feed_readback"]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "pa_feed_safety"
    ] == summary["pa_feed_safety"]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "pa_feed_cache_freshness"
    ] == proposal["evidence"]["pa_feed_cache_freshness"]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "pa_feed_cache_no_live_authority"
    ] is True
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "reference_feed_exclusion"
    ] == summary["reference_feed_exclusion"]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "current_main_bus_processing_cache_freshness"
    ] == proposal["evidence"]["current_main_bus_processing_cache_freshness"]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "rollback_master_processing_readback"
    ] == summary["rollback_master_processing_readback"]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "rollback_pa_feed_matrix_readback"
    ] == summary["rollback_pa_feed_matrix_readback"]
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_top_level_denial(source_evidence)
    assert source_evidence["master_bus_safety_summary"] == summary
    assert source_evidence["pa_feed_readback"] == summary["pa_feed_readback"]
    assert source_evidence["pa_feed_cache_no_live_authority"] is True
    assert source_evidence["pa_feed_safety"]["status"] == "passed"
    assert source_evidence["reference_feed_exclusion"]["status"] == "passed"
    assert source_evidence["rollback_master_processing_readback"] == summary[
        "rollback_master_processing_readback"
    ]
    assert source_evidence["rollback_pa_feed_matrix_readback"] == summary[
        "rollback_pa_feed_matrix_readback"
    ]
    assert source_evidence["future_wing_readback_contract"] == summary[
        "future_wing_readback_contract"
    ]


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
    _assert_top_level_denial(recommendation.master_bus_safety_summary())
    _assert_top_level_denial(recommendation.to_operator_proposal())


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
            pa_feed_readback_available=True,
            pa_feed_safe_for_master_change=True,
            pa_feed_cache_freshness=_fresh(),
            current_state_cache_freshness=_fresh(8),
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


def test_missing_pa_feed_safety_blocks_master_bus_proposal():
    recommendation = analyze_master_bus_need(
        MasterBusAnalysisInput(
            target_profile="live",
            integrated_lufs=-18.7,
            short_term_lufs=-18.3,
            true_peak_dbfs=-4.5,
            crest_factor_db=10.5,
            lra_db=5.0,
            spectral_tilt_db=1.8,
            confidence=0.9,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "pa_feed_readback_missing" in recommendation.blockers
    assert "pa_feed_safety_not_confirmed" in recommendation.blockers
    assert "pa_feed_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["pa_feed_cache_freshness_status"] == "not_provided"
    _assert_top_level_denial(recommendation.master_bus_safety_summary())
    _assert_top_level_denial(recommendation.to_operator_proposal())


def test_reference_feed_is_excluded_from_master_output_proposals():
    recommendation = analyze_master_bus_need(
        MasterBusAnalysisInput(
            bus_id="dante_reference",
            bus_label="Master Reference Capture",
            target_profile="live",
            integrated_lufs=-18.7,
            short_term_lufs=-18.3,
            true_peak_dbfs=-4.5,
            crest_factor_db=10.5,
            lra_db=5.0,
            spectral_tilt_db=1.8,
            output_feed_role="master_reference_capture",
            is_reference_feed=True,
            pa_feed_readback_available=True,
            pa_feed_safe_for_master_change=True,
            pa_feed_cache_freshness=_fresh(),
            confidence=0.9,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "master_reference_feed_excluded" in recommendation.blockers
    assert recommendation.severity == "high"
    assert recommendation.evidence["is_reference_feed"] is True
    assert recommendation.evidence["output_feed_role"] == "master_reference_capture"
    summary = recommendation.master_bus_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["is_reference_feed"] is True
    assert "master_reference_feed_excluded" in summary["blockers"]
    _assert_top_level_denial(summary)
    _assert_top_level_denial(recommendation.to_operator_proposal())


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
            pa_feed_readback_available=True,
            pa_feed_safe_for_master_change=True,
            pa_feed_cache_freshness=_fresh(),
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
    _assert_top_level_denial(recommendation.master_bus_safety_summary())
    _assert_top_level_denial(recommendation.to_operator_proposal())


def test_live_authority_pa_feed_cache_blocks_master_bus_proposal_only_recommendation():
    analysis = MasterBusAnalysisInput(
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
        pa_feed_readback_available=True,
        pa_feed_safe_for_master_change=True,
        pa_feed_cache_freshness={
            "status": "fresh",
            "field_count": 6,
            "fresh_field_count": 6,
            "read_only": False,
            "live_write": True,
            "queries_console": True,
        },
        current_state_cache_freshness=_fresh(8),
        confidence=0.9,
        persistence_cycles=3,
    )
    recommendation = analyze_master_bus_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "pa_feed_cache_live_authority_present" in recommendation.blockers
    assert recommendation.evidence["pa_feed_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["pa_feed_cache_no_live_authority"] is False
    assert recommendation.evidence["pa_feed_cache_freshness"]["live_write"] is True
    summary = recommendation.master_bus_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["pa_feed_cache_no_live_authority"] is False
    assert "pa_feed_cache_live_authority_present" in summary["blockers"]
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert proposal["master_bus_safety_summary"] == summary
    assert build_master_bus_proposals([analysis]) == ()


def test_live_authority_current_master_bus_state_blocks_proposal_only_recommendation():
    analysis = MasterBusAnalysisInput(
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
        pa_feed_readback_available=True,
        pa_feed_safe_for_master_change=True,
        pa_feed_cache_freshness=_fresh(),
        current_state_cache_freshness={
            "status": "fresh",
            "field_count": 8,
            "fresh_field_count": 8,
            "read_only": False,
            "live_write": True,
            "queries_console": True,
        },
        confidence=0.9,
        persistence_cycles=3,
    )
    recommendation = analyze_master_bus_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "master_bus_current_state_cache_live_authority_present" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_main_bus_processing_cache_no_live_authority"] is False
    assert recommendation.evidence["current_state_cache_freshness"]["queries_console"] is True
    summary = recommendation.master_bus_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["current_main_bus_processing_cache_no_live_authority"] is False
    assert "master_bus_current_state_cache_live_authority_present" in summary["blockers"]
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert proposal["master_bus_safety_summary"] == summary
    assert build_master_bus_proposals([analysis]) == ()


def test_missing_current_master_bus_state_blocks_proposal_only_recommendation():
    analysis = MasterBusAnalysisInput(
        bus_label="Main L/R",
        target_profile="live",
        integrated_lufs=-18.7,
        short_term_lufs=-18.3,
        true_peak_dbfs=-4.5,
        crest_factor_db=10.5,
        lra_db=5.0,
        spectral_tilt_db=1.8,
        pa_feed_readback_available=True,
        pa_feed_safe_for_master_change=True,
        pa_feed_cache_freshness=_fresh(),
        confidence=0.9,
        persistence_cycles=3,
    )
    recommendation = analyze_master_bus_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "master_bus_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "not_provided"
    _assert_top_level_denial(recommendation.master_bus_safety_summary())
    _assert_top_level_denial(recommendation.to_operator_proposal())
    assert build_master_bus_proposals([analysis]) == ()


def test_build_master_bus_proposals_filters_to_proposal_only_items():
    proposals = build_master_bus_proposals(
        [
            MasterBusAnalysisInput(
                bus_id="main_lr",
                target_profile="live",
                integrated_lufs=-18.7,
                true_peak_dbfs=-4.5,
                pa_feed_readback_available=True,
                pa_feed_safe_for_master_change=True,
                pa_feed_cache_freshness=_fresh(),
                current_state_cache_freshness=_fresh(8),
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
    _assert_top_level_denial(proposals[0])


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
    assert (
        "pa_feed_safety"
        in contracts["master_bus_optimization_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "pa_feed_cache_no_live_authority"
        in contracts["master_bus_optimization_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "reference_feed_exclusion"
        in contracts["master_bus_optimization_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "current_main_bus_processing_cache_no_live_authority"
        in contracts["master_bus_optimization_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "rollback_master_processing_readback"
        in contracts["master_bus_optimization_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "rollback_pa_feed_matrix_readback"
        in contracts["master_bus_optimization_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
