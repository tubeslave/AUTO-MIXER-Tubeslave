from pathlib import Path

from backend.corrective_eq_proposal import (
    EQ_READ_REQUIRED_SOURCE,
    CorrectiveEQAnalysisInput,
    analyze_corrective_eq_need,
    build_corrective_eq_proposals,
    classify_source_family,
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


def _lead_vocal_eq(**overrides):
    payload = {
        "channel_id": 5,
        "channel_label": "Lead Vox",
        "source_role": "lead_vocal",
        "problem": "harsh_presence",
        "current_band_id": 2,
        "current_band_frequency_hz": 3100.0,
        "current_band_gain_db": 0.0,
        "current_band_q": 1.2,
        "target_frequency_hz": 3000.0,
        "proposed_gain_db": -2.8,
        "proposed_q": 2.0,
        "true_peak_headroom_db": 6.0,
        "feedback_risk": 0.12,
        "phase_risk": 0.1,
        "confidence_hint": 0.84,
        "current_state_cache_freshness": _fresh(),
        "phase_context_freshness": _fresh(4),
    }
    payload.update(overrides)
    return CorrectiveEQAnalysisInput(**payload)


def test_corrective_eq_proposal_module_is_transport_free():
    source = Path("backend/corrective_eq_proposal.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "set_eq",
        "set_eq_band",
        "mixer_client",
    ]
    for token in forbidden:
        assert token not in source


def test_safe_cut_creates_non_queueable_corrective_eq_proposal():
    recommendation = analyze_corrective_eq_need(_lead_vocal_eq())

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "eq_band_proposal"
    assert recommendation.source_family == "vocal"
    assert recommendation.current_band_id == 2
    assert recommendation.proposed_gain_db < 0.0
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["band_identity"]["status"] == "passed"
    assert recommendation.evidence["bounded_gain_delta"]["status"] == "passed"
    assert recommendation.evidence["protected_band_policy"]["status"] == "passed"
    assert recommendation.evidence["feedback_notch_preserved"] is True
    assert recommendation.evidence["current_eq_context_cache_freshness"]["status"] == "fresh"
    assert recommendation.evidence["current_eq_context_cache_no_live_authority"] is True
    assert (
        recommendation.evidence["current_phase_audio_context_cache_freshness"]["status"]
        == "fresh"
    )
    assert recommendation.evidence["current_phase_audio_context_cache_no_live_authority"] is True
    assert recommendation.evidence["source_identity_downstream_trust"]["status"] == (
        "legacy_unverified"
    )
    summary = recommendation.eq_band_protection_summary()
    assert summary["status"] == "passed"
    assert summary["band_identity_status"] == "passed"
    assert summary["protected_band_policy_status"] == "passed"
    assert summary["bounded_gain_status"] == "passed"
    assert summary["feedback_notch_band_protected"] is True
    assert summary["manual_eq_band_protected"] is True
    assert summary["current_eq_context_cache_no_live_authority"] is True
    assert summary["current_phase_audio_context_cache_no_live_authority"] is True
    assert summary["readback_required_before_promotion"] is True
    assert summary["rollback_required_before_promotion"] is True
    assert summary["live_write"] is False
    assert summary["queue_mutation"] is False
    _assert_top_level_denial(summary)

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "corrective_eq_proposal"
    assert proposal["kind"] == "eq"
    _assert_top_level_denial(proposal)
    assert "requested_change" not in proposal
    assert "unsupported_value_type:eq" in proposal["blockers"]
    assert proposal["old_value"]["source"] == EQ_READ_REQUIRED_SOURCE
    assert proposal["rollback_value"]["source"] == EQ_READ_REQUIRED_SOURCE
    assert proposal["proposed_action"]["band_id"] == 2
    assert proposal["eq_band_protection_summary"] == summary
    assert proposal["proposed_action"]["eq_band_protection_summary"] == summary
    assert proposal["evidence"]["eq_band_protection_summary"] == summary
    assert proposal["evidence"]["old_value_source"] == EQ_READ_REQUIRED_SOURCE
    assert proposal["evidence"]["rollback_required_before_promotion"] is True
    assert proposal["evidence"]["rollback_eq_band_readback"] == {
        "channel_id": 5,
        "band_id": 2,
        "filter_type": recommendation.filter_type,
        "frequency_hz": recommendation.evidence["band_identity"][
            "current_band_frequency_hz"
        ],
        "gain_db": recommendation.evidence["bounded_gain_delta"][
            "current_gain_db"
        ],
        "q": recommendation.evidence["band_identity"]["current_band_q"],
        "source": EQ_READ_REQUIRED_SOURCE,
        "required_before_live_promotion": True,
        "live_write": False,
    }
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["current_eq_context_cache_no_live_authority"] is True
    assert proposal["evidence"]["current_phase_audio_context_cache_no_live_authority"] is True
    correction = proposal["operator_correction"]
    assert correction["objective_family"] == "eq"
    completeness = correction["source_evidence"]["evidence_completeness"]
    assert completeness["status"] == "complete"
    assert completeness["missing"] == []
    assert correction["operator_proposal_review_gate"]["evidence_gate_status"] == (
        "required_evidence_complete_operator_review_still_required"
    )
    assert correction["bounded_delta"]["delta_db"] == recommendation.proposed_gain_db
    assert correction["bounded_delta"]["parameter_deltas"]["gain_db"] == (
        recommendation.proposed_gain_db
    )
    assert correction["bounded_delta"]["parameter_deltas"]["frequency_hz"] == (
        recommendation.target_frequency_hz
    )
    assert correction["future_wing_readback_contract"]["target_kind"] == "channel_eq_band"
    assert correction["future_wing_readback_contract"][
        "required_before_live_promotion"
    ] is True
    assert correction["read_only_or_dry_run_only"] is True
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    assert "manual_band" in correction["headroom_feedback_true_peak_checks"][
        "required_checks"
    ]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "current_eq_context_cache_no_live_authority"
    ] is True
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_top_level_denial(source_evidence)
    assert source_evidence["eq_band_protection_summary"] == summary
    assert source_evidence["rollback_eq_band_readback"] == proposal["evidence"][
        "rollback_eq_band_readback"
    ]
    assert source_evidence["current_eq_context_cache_no_live_authority"] is True
    assert source_evidence["current_phase_audio_context_cache_no_live_authority"] is True


def test_corrective_eq_blocks_feedback_notch_and_manual_band_overwrite():
    feedback = analyze_corrective_eq_need(
        _lead_vocal_eq(current_band_is_feedback_notch=True)
    )
    manual = analyze_corrective_eq_need(_lead_vocal_eq(current_band_is_manual=True))

    assert feedback.status == "analysis_only"
    assert "feedback_notch_band_preserved" in feedback.blockers
    assert feedback.severity == "high"
    assert feedback.evidence["protected_band_policy"]["status"] == "blocked"
    assert feedback.eq_band_protection_summary()["status"] == "blocked"
    assert (
        feedback.eq_band_protection_summary()["feedback_notch_band_protected"]
        is False
    )
    assert manual.status == "analysis_only"
    assert "manual_eq_band_preserved" in manual.blockers
    assert manual.evidence["protected_band_policy"]["status"] == "blocked"
    assert manual.eq_band_protection_summary()["status"] == "blocked"
    assert manual.eq_band_protection_summary()["manual_eq_band_protected"] is False


def test_corrective_eq_blocks_unsafe_boost_and_missing_headroom():
    boost = analyze_corrective_eq_need(
        _lead_vocal_eq(proposed_gain_db=3.5, true_peak_headroom_db=6.0)
    )
    no_headroom = analyze_corrective_eq_need(
        _lead_vocal_eq(proposed_gain_db=1.8, true_peak_headroom_db=2.2)
    )
    feedback_boost = analyze_corrective_eq_need(
        _lead_vocal_eq(proposed_gain_db=1.5, feedback_risk=0.7)
    )

    assert boost.status == "analysis_only"
    assert "eq_boost_exceeds_default_bound" in boost.blockers
    assert no_headroom.status == "analysis_only"
    assert "insufficient_true_peak_headroom_for_eq_boost" in no_headroom.blockers
    assert feedback_boost.status == "analysis_only"
    assert "feedback_risk_blocks_eq_boost" in feedback_boost.blockers


def test_corrective_eq_blocks_stale_context_and_band_mismatch():
    stale = analyze_corrective_eq_need(
        _lead_vocal_eq(
            current_state_cache_freshness={"status": "stale"},
            phase_context_freshness={"status": "stale"},
        )
    )
    mismatch = analyze_corrective_eq_need(
        _lead_vocal_eq(current_band_frequency_hz=600.0)
    )

    assert stale.status == "analysis_only"
    assert "eq_current_state_cache_not_fresh" in stale.blockers
    assert "eq_phase_context_not_fresh" in stale.blockers
    assert mismatch.status == "analysis_only"
    assert "eq_band_frequency_mismatch" in mismatch.blockers
    assert mismatch.evidence["band_identity"]["status"] == "blocked"


def test_corrective_eq_blocks_live_authority_current_eq_context():
    analysis = _lead_vocal_eq(
        current_state_cache_freshness={
            "status": "fresh",
            "field_count": 6,
            "fresh_field_count": 6,
            "read_only": False,
            "live_write": True,
            "queries_console": True,
        }
    )
    recommendation = analyze_corrective_eq_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "eq_current_state_cache_live_authority_present" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_eq_context_cache_no_live_authority"] is False
    assert recommendation.evidence["current_eq_context_cache_freshness"]["live_write"] is True
    summary = recommendation.eq_band_protection_summary()
    assert summary["status"] == "blocked"
    assert summary["current_eq_context_cache_no_live_authority"] is False
    assert "eq_current_state_cache_live_authority_present" in summary["blockers"]
    _assert_top_level_denial(summary)
    _assert_top_level_denial(recommendation.to_operator_proposal())
    assert build_corrective_eq_proposals([analysis]) == ()


def test_corrective_eq_blocks_live_authority_phase_context():
    analysis = _lead_vocal_eq(
        phase_context_freshness={
            "status": "fresh",
            "field_count": 4,
            "fresh_field_count": 4,
            "read_only": False,
            "live_write": True,
            "queries_console": True,
        }
    )
    recommendation = analyze_corrective_eq_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "eq_phase_context_cache_live_authority_present" in recommendation.blockers
    assert recommendation.evidence["current_phase_audio_context_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_phase_audio_context_cache_no_live_authority"] is False
    assert (
        recommendation.evidence["current_phase_audio_context_cache_freshness"]["queries_console"]
        is True
    )
    _assert_top_level_denial(recommendation.eq_band_protection_summary())
    _assert_top_level_denial(recommendation.to_operator_proposal())
    assert build_corrective_eq_proposals([analysis]) == ()


def test_corrective_eq_blocks_unsupported_filter_type():
    recommendation = analyze_corrective_eq_need(
        _lead_vocal_eq(filter_type="load_snap")
    )

    assert recommendation.status == "analysis_only"
    assert "unsupported_eq_filter_type" in recommendation.blockers
    assert recommendation.filter_type == "unsupported"
    assert recommendation.evidence["filter_type"] == "unsupported"
    assert recommendation.evidence["filter_type_policy"]["status"] == "blocked"
    assert (
        recommendation.evidence["filter_type_policy"]["requested_filter_type"]
        == "load_snap"
    )
    assert "peaking" in recommendation.evidence["filter_type_policy"][
        "allowed_filter_types"
    ]

    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(recommendation.eq_band_protection_summary())
    _assert_top_level_denial(proposal)
    assert "unsupported_eq_filter_type" in proposal["blockers"]
    assert proposal["proposed_action"]["filter_type"] == "unsupported"


def test_build_corrective_eq_proposals_filters_to_proposal_only_items():
    proposals = build_corrective_eq_proposals(
        [
            _lead_vocal_eq(),
            _lead_vocal_eq(channel_id=6, current_band_is_feedback_notch=True),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["channel"] == 5
    _assert_top_level_denial(proposals[0])
    _assert_top_level_denial(proposals[0]["eq_band_protection_summary"])


def test_corrective_eq_helpers_and_registration():
    method = get_processing_method("corrective_eq_proposal")
    main = get_pipeline_profile("main_live_assist")
    row = get_project_goal_capability("equalization")
    console = get_project_goal_capability("console_eq_control")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert classify_source_family("Lead Vocal") == "vocal"
    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_write_live is False
    assert main is not None
    assert "corrective_eq_proposal" in main.proposal_method_ids
    assert row is not None and "corrective_eq_proposal" in row.method_ids
    assert console is not None and "corrective_eq_proposal" in console.method_ids
    assert contracts["corrective_eq_proposal"]["proposal_contract"]["value_type"] == "eq"
    required = contracts["corrective_eq_proposal"]["proposal_contract"][
        "safety_evidence_required"
    ]
    assert "band_identity" in required
    assert "filter_type_policy" in required
    assert "protected_band_policy" in required
    assert "current_eq_context_cache_no_live_authority" in required
    assert "current_phase_audio_context_cache_no_live_authority" in required
    assert "old_value_source" in required
