from pathlib import Path

from backend.fx_send_proposal import (
    FX_READ_REQUIRED_SOURCE,
    FXSendAnalysisInput,
    analyze_fx_send_need,
    build_fx_send_proposals,
    classify_source_family,
    fx_parameter_schema_from_snapshot,
    tempo_delay_ms,
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


def _schema(family="reverb"):
    return {
        "status": "mapped",
        "schema_family": family,
        "schema_source": "automixer_generic_fx_parameter_hints",
        "mapped_parameter_count": 6,
        "generic_parameter_count": 0,
        "physical_model_schema_verified": False,
        "live_write_authority": False,
    }


def _assert_top_level_denial(payload):
    assert payload["live_write"] is False
    assert payload["queue_mutation"] is False
    assert payload["can_apply_live"] is False
    assert payload["can_import_to_queue"] is False
    assert payload["queue_import_allowed"] is False
    assert payload["live_apply_allowed"] is False
    assert payload["read_only_or_dry_run_only"] is True
    assert payload["live_authority_granted"] is False
    assert payload["readiness_closure_allowed"] is False
    assert payload["queue_action"] == "not_queueable_until_promoted"
    assert payload["mutates_mixer"] is False


def test_fx_send_proposal_module_is_transport_free():
    source = Path("backend/fx_send_proposal.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "set_fx",
        "set_channel_send",
        "set_send_level",
        "mixer_client",
    ]
    for token in forbidden:
        assert token not in source


def test_lead_vocal_creates_proposal_only_fx_send_recommendation():
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_send_db=-28.0,
            tempo_bpm=120.0,
            arrangement_role="lead_space",
            contribution_share=0.85,
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            source_confidence=0.92,
            persistence_cycles=3,
            current_state_cache_freshness=_fresh(),
            current_fx_parameter_schema=_schema("reverb"),
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "fx_send_proposal"
    assert recommendation.source_family == "lead_vocal"
    assert recommendation.fx_type == "reverb"
    assert -36.0 <= recommendation.target_send_db <= -12.0
    assert recommendation.delta_send_db > 0
    assert recommendation.return_level_db <= -3.0
    assert recommendation.ducking_db > 0.0
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_state_cache_freshness"]["fresh_field_count"] == 6
    assert recommendation.evidence["current_fx_parameter_schema_status"] == "mapped"
    assert recommendation.evidence["current_fx_parameter_schema"]["schema_family"] == "reverb"
    assert recommendation.evidence["expected_fx_schema_family"] == "reverb"
    assert recommendation.evidence["return_level_safety"]["status"] == "passed"
    assert recommendation.evidence["monitor_safety"]["status"] == "passed"
    assert recommendation.evidence["monitor_safety"]["post_fader_send_required"] is True
    assert recommendation.evidence["current_fx_send_return_cache_freshness"]["status"] == "fresh"
    assert recommendation.evidence["current_fx_send_return_cache_no_live_authority"] is True
    assert recommendation.evidence["source_identity_downstream_trust"]["status"] == "legacy_unverified"
    summary = recommendation.fx_send_safety_summary()
    assert summary["status"] == "passed"
    assert summary["fx_type"] == "reverb"
    assert summary["return_level_safety_status"] == "passed"
    assert summary["send_level_limit"]["status"] == "passed"
    assert summary["send_level_limit_status"] == "passed"
    assert summary["fx_return_headroom"]["status"] == "passed"
    assert summary["fx_return_headroom_status"] == "passed"
    assert summary["dry_wet_bounds"]["status"] == "passed"
    assert summary["dry_wet_bounds_status"] == "passed"
    assert summary["rollback_fx_readback"]["status"] == (
        "required_before_live_promotion"
    )
    assert summary["monitor_safety_status"] == "passed"
    assert summary["post_fader_send_required_before_promotion"] is True
    assert summary["current_fx_send_return_cache_freshness_status"] == "fresh"
    assert summary["current_fx_send_return_cache_no_live_authority"] is True
    assert summary["current_fx_parameter_schema_status"] == "mapped"
    assert summary["current_fx_parameter_schema_family"] == "reverb"
    assert summary["expected_fx_schema_family"] == "reverb"
    assert summary["fx_parameter_schema_no_live_authority"] is True
    assert summary["required_safety_checks"] == {
        "source_identity": "required_before_live_promotion",
        "send_level_limit": "required_before_live_promotion",
        "fx_return_headroom": "required_before_live_promotion",
        "return_level_safety": "required_before_live_promotion",
        "feedback_risk": "required_before_live_promotion",
        "intelligibility_risk": "required_before_live_promotion",
        "monitor_safety": "required_before_live_promotion",
        "dry_wet_bounds": "required_before_live_promotion",
        "fx_parameter_schema": "required_before_live_promotion",
    }
    assert summary["rollback_expectation"] == {
        "required_before_live_promotion": True,
        "restore_previous_send_level_after_physical_readback": True,
        "restore_previous_fx_return_level_after_physical_readback": True,
        "restore_previous_fx_slot_parameters_after_physical_readback": True,
        "manual_operator_rollback_required_if_promoted": True,
    }
    readback = summary["future_wing_readback_contract"]
    assert readback["target_kind"] == "send_or_fx_parameter"
    assert readback["required_before_live_promotion"] is True
    assert readback["old_value_source"] == FX_READ_REQUIRED_SOURCE
    assert readback["requires_real_wing_proof"] is True
    assert readback["requires_operator_approval"] is True
    assert readback["requires_separate_non_fader_adr"] is True
    assert "fx_send_level_write" in readback["blocked_live_actions"]
    assert "monitor_send_write" in readback["blocked_live_actions"]
    assert "non_fader_live_expansion_requires_separate_adr" in summary[
        "safety_blockers"
    ]
    assert summary["readback_required_before_promotion"] is True
    assert summary["rollback_required_before_promotion"] is True
    assert summary["live_write"] is False
    assert summary["queue_mutation"] is False
    _assert_top_level_denial(summary)

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "fx_send_proposal"
    _assert_top_level_denial(proposal)
    assert "requested_change" not in proposal
    assert "unsupported_value_type:fx_send" in proposal["blockers"]
    assert "fx_send_live_apply_quarantined" in proposal["blockers"]
    assert "non_fader_live_expansion_requires_separate_adr" in proposal["blockers"]
    assert proposal["old_value"]["source"] == FX_READ_REQUIRED_SOURCE
    assert proposal["rollback_value"]["source"] == FX_READ_REQUIRED_SOURCE
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["old_value_source"] == FX_READ_REQUIRED_SOURCE
    assert proposal["evidence"]["rollback_value_source"] == FX_READ_REQUIRED_SOURCE
    assert proposal["evidence"]["monitor_safety"]["status"] == "passed"
    assert proposal["evidence"]["return_level_safety"]["status"] == "passed"
    assert proposal["evidence"]["current_state_cache_freshness_status"] == "fresh"
    assert proposal["evidence"]["current_fx_send_return_cache_no_live_authority"] is True
    assert proposal["evidence"]["current_fx_parameter_schema_status"] == "mapped"
    assert proposal["fx_send_safety_summary"] == summary
    assert proposal["proposed_action"]["fx_send_safety_summary"] == summary
    assert proposal["evidence"]["fx_send_safety_summary"] == summary
    assert proposal["safety_blockers"] == summary["safety_blockers"]
    assert proposal["required_safety_checks"] == summary["required_safety_checks"]
    assert proposal["rollback_expectation"] == summary["rollback_expectation"]
    assert proposal["future_wing_readback_contract"] == summary[
        "future_wing_readback_contract"
    ]
    assert proposal["proposed_action"]["send_level_limit"] == summary[
        "send_level_limit"
    ]
    assert proposal["proposed_action"]["fx_return_headroom"] == summary[
        "fx_return_headroom"
    ]
    assert proposal["proposed_action"]["dry_wet_bounds"] == summary[
        "dry_wet_bounds"
    ]
    assert proposal["proposed_action"]["rollback_fx_readback"] == summary[
        "rollback_fx_readback"
    ]
    assert proposal["evidence"]["current_fx_parameter_schema_hints"] == (
        recommendation.evidence["current_fx_parameter_schema"]
    )
    assert proposal["evidence"]["rollback_fx_readback"] == summary[
        "rollback_fx_readback"
    ]
    correction = proposal["operator_correction"]
    assert correction["objective_family"] == "fx_sends"
    assert correction["bounded_delta"]["delta_db"] == recommendation.delta_send_db
    assert correction["bounded_delta"]["parameter_deltas"]["delta_send_db"] == (
        recommendation.delta_send_db
    )
    assert correction["bounded_delta"]["parameter_deltas"]["return_level_db"] == (
        recommendation.return_level_db
    )
    assert correction["future_wing_readback_contract"]["target_kind"] == (
        "send_or_fx_parameter"
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
    assert "monitor_safety" in correction["headroom_feedback_true_peak_checks"][
        "required_checks"
    ]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "current_fx_send_return_cache_no_live_authority"
    ] is True
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_top_level_denial(source_evidence)
    evidence_checks = correction["headroom_feedback_true_peak_checks"][
        "evidence_checks"
    ]
    for key in (
        "source_role",
        "send_level_limit",
        "fx_return_headroom",
        "return_level_safety",
        "feedback_risk",
        "intelligibility_risk",
        "monitor_safety",
        "dry_wet_bounds",
        "current_fx_send_return_cache_freshness",
        "current_fx_send_return_cache_no_live_authority",
        "current_fx_parameter_schema_hints",
        "fx_parameter_schema_no_live_authority",
        "source_identity_downstream_trust",
        "old_value_source",
        "rollback_fx_readback",
    ):
        assert evidence_checks[key] == source_evidence[key]
    assert source_evidence["monitor_safety"]["status"] == "passed"
    assert source_evidence["return_level_safety"]["status"] == "passed"
    assert source_evidence["send_level_limit"]["status"] == "passed"
    assert source_evidence["fx_return_headroom"]["status"] == "passed"
    assert source_evidence["dry_wet_bounds"]["status"] == "passed"
    assert source_evidence["rollback_fx_readback"]["status"] == (
        "required_before_live_promotion"
    )
    assert source_evidence["future_wing_readback_contract"] == summary[
        "future_wing_readback_contract"
    ]


def test_fx_send_can_derive_schema_evidence_from_fx_inserts_snapshot():
    snapshot = {
        "type": "fx_inserts_snapshot",
        "safety": {
            "queries_console": False,
            "live_write": False,
        },
        "summary": {
            "read_coverage_status": "partial",
            "cache_freshness_status": "fresh",
        },
        "cache_freshness": {"status": "fresh"},
        "parameter_schema": {
            "status": "mapped",
            "mapped_parameter_count": 6,
            "generic_parameter_count": 0,
            "physical_model_schema_verified": False,
            "live_write_authority": False,
        },
        "fx_slots": [
            {
                "slot": "FX1",
                "model": "PLATE",
                "parameter_schema": {
                    "status": "mapped",
                    "schema_family": "reverb",
                    "schema_source": "automixer_generic_fx_parameter_hints",
                    "mapped_parameter_count": 6,
                    "generic_parameter_count": 0,
                    "cached_parameter_count": 6,
                    "physical_model_schema_verified": False,
                    "live_write_authority": False,
                },
            },
        ],
    }

    schema = fx_parameter_schema_from_snapshot(snapshot, "reverb")
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_send_db=-28.0,
            arrangement_role="lead_space",
            contribution_share=0.85,
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            source_confidence=0.92,
            persistence_cycles=3,
            current_state_cache_freshness=_fresh(),
            current_fx_inserts_snapshot=snapshot,
        )
    )

    assert schema["status"] == "mapped"
    assert schema["schema_family"] == "reverb"
    assert schema["fx_slot"] == "FX1"
    assert recommendation.status == "proposal_only"
    assert recommendation.evidence["current_fx_parameter_schema"]["fx_slot"] == "FX1"
    assert recommendation.evidence["current_fx_parameter_schema"]["fx_model"] == "PLATE"
    assert recommendation.evidence["current_fx_parameter_schema"]["snapshot_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_fx_parameter_schema"]["snapshot_queries_console"] is False
    assert recommendation.evidence["current_fx_parameter_schema"]["snapshot_live_write"] is False


def test_requested_tempo_delay_uses_tempo_and_remains_non_applyable():
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            source_role="lead_vocal",
            current_send_db=-34.0,
            tempo_bpm=100.0,
            requested_fx_type="delay",
            return_headroom_db=7.0,
            feedback_risk=0.1,
            intelligibility_risk=0.15,
            source_confidence=0.9,
            persistence_cycles=3,
            current_state_cache_freshness=_fresh(),
            current_fx_parameter_schema=_schema("delay"),
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.fx_type == "delay"
    assert recommendation.delay_ms == 450.0
    summary = recommendation.fx_send_safety_summary()
    assert summary["status"] == "passed"
    assert summary["fx_type"] == "delay"
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert proposal["fx_send_safety_summary"] == summary


def test_dry_anchor_source_is_analysis_only():
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=1,
            source_role="kick",
            current_send_db=-36.0,
            return_headroom_db=8.0,
            feedback_risk=0.1,
            intelligibility_risk=0.1,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "dry_anchor_source" in recommendation.blockers


def test_feedback_or_headroom_blocks_fx_send_proposal():
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            source_role="lead_vocal",
            current_send_db=-30.0,
            return_headroom_db=1.0,
            feedback_risk=0.7,
            intelligibility_risk=0.2,
            source_confidence=0.9,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "insufficient_fx_return_headroom" in recommendation.blockers
    assert "feedback_risk_high" in recommendation.blockers


def test_monitor_safety_blocks_fx_send_proposal_without_fresh_safe_context():
    stale_monitor = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            source_role="lead_vocal",
            current_send_db=-30.0,
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            monitor_send_exposure=True,
            monitor_safety_risk=0.1,
            monitor_safety_freshness={"status": "stale"},
            source_confidence=0.9,
            persistence_cycles=3,
            current_state_cache_freshness=_fresh(),
            current_fx_parameter_schema=_schema("reverb"),
        )
    )
    risky_monitor = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            source_role="lead_vocal",
            current_send_db=-30.0,
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            monitor_send_exposure=True,
            monitor_safety_risk=0.72,
            monitor_safety_freshness={"status": "fresh"},
            source_confidence=0.9,
            persistence_cycles=3,
            current_state_cache_freshness=_fresh(),
            current_fx_parameter_schema=_schema("reverb"),
        )
    )

    assert stale_monitor.status == "analysis_only"
    assert "fx_monitor_safety_context_not_fresh" in stale_monitor.blockers
    assert stale_monitor.evidence["monitor_safety"]["status"] == "blocked"
    assert stale_monitor.severity == "high"
    assert risky_monitor.status == "analysis_only"
    assert "monitor_safety_risk_high" in risky_monitor.blockers
    assert risky_monitor.evidence["monitor_safety"]["monitor_safety_risk"] == 0.72
    assert risky_monitor.evidence["monitor_safety"]["status"] == "blocked"


def test_stale_current_fx_state_blocks_proposal_only_recommendation():
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_send_db=-28.0,
            tempo_bpm=120.0,
            arrangement_role="lead_space",
            contribution_share=0.85,
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            source_confidence=0.92,
            persistence_cycles=3,
            current_state_cache_freshness={
                "status": "stale",
                "field_count": 6,
                "fresh_field_count": 3,
                "stale_field_count": 3,
                "stale_fields": ["send_db", "return_level_db", "fx_slot_type"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "fx_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "stale"
    assert recommendation.evidence["current_state_cache_freshness"]["stale_fields"] == [
        "send_db",
        "return_level_db",
        "fx_slot_type",
    ]
    summary = recommendation.fx_send_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["current_fx_send_return_cache_freshness_status"] == "stale"
    assert "fx_current_state_cache_not_fresh" in summary["blockers"]
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert proposal["fx_send_safety_summary"] == summary


def test_live_authority_current_fx_state_blocks_proposal_only_recommendation():
    analysis = FXSendAnalysisInput(
        channel_id=6,
        channel_label="Lead Vox",
        source_role="lead_vocal",
        current_send_db=-28.0,
        arrangement_role="lead_space",
        contribution_share=0.85,
        return_headroom_db=6.0,
        feedback_risk=0.1,
        intelligibility_risk=0.2,
        source_confidence=0.92,
        persistence_cycles=3,
        current_state_cache_freshness={
            "status": "fresh",
            "field_count": 6,
            "fresh_field_count": 6,
            "read_only": False,
            "live_write": True,
            "queries_console": True,
        },
        current_fx_parameter_schema=_schema("reverb"),
    )
    recommendation = analyze_fx_send_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "fx_current_state_cache_live_authority_present" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_fx_send_return_cache_no_live_authority"] is False
    assert recommendation.evidence["current_fx_send_return_cache_freshness"]["live_write"] is True
    summary = recommendation.fx_send_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["current_fx_send_return_cache_no_live_authority"] is False
    assert "fx_current_state_cache_live_authority_present" in summary["blockers"]
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert build_fx_send_proposals([analysis]) == ()


def test_missing_current_fx_state_blocks_proposal_only_recommendation():
    analysis = FXSendAnalysisInput(
        channel_id=6,
        channel_label="Lead Vox",
        source_role="lead_vocal",
        current_send_db=-28.0,
        arrangement_role="lead_space",
        contribution_share=0.85,
        return_headroom_db=6.0,
        feedback_risk=0.1,
        intelligibility_risk=0.2,
        source_confidence=0.92,
        persistence_cycles=3,
        current_fx_parameter_schema=_schema("reverb"),
    )
    recommendation = analyze_fx_send_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "fx_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "not_provided"
    assert build_fx_send_proposals([analysis]) == ()


def test_missing_fx_parameter_schema_blocks_proposal_only_recommendation():
    analysis = FXSendAnalysisInput(
        channel_id=6,
        channel_label="Lead Vox",
        source_role="lead_vocal",
        current_send_db=-28.0,
        arrangement_role="lead_space",
        contribution_share=0.85,
        return_headroom_db=6.0,
        feedback_risk=0.1,
        intelligibility_risk=0.2,
        source_confidence=0.92,
        persistence_cycles=3,
        current_state_cache_freshness=_fresh(),
    )
    recommendation = analyze_fx_send_need(analysis)

    assert recommendation.status == "analysis_only"
    assert "fx_parameter_schema_not_mapped" in recommendation.blockers
    assert recommendation.evidence["current_fx_parameter_schema_status"] == "not_provided"
    assert build_fx_send_proposals([analysis]) == ()


def test_generic_fx_parameter_schema_blocks_fx_send_proposal():
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_send_db=-28.0,
            tempo_bpm=120.0,
            arrangement_role="lead_space",
            contribution_share=0.85,
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            source_confidence=0.92,
            persistence_cycles=3,
            current_state_cache_freshness={"status": "fresh"},
            current_fx_parameter_schema={
                "status": "partial",
                "schema_family": "unknown",
                "schema_source": "automixer_generic_fx_parameter_hints",
                "mapped_parameter_count": 0,
                "generic_parameter_count": 1,
                "physical_model_schema_verified": False,
                "live_write_authority": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "fx_parameter_schema_not_mapped" in recommendation.blockers
    assert recommendation.severity == "medium"
    assert recommendation.evidence["current_fx_parameter_schema_status"] == "partial"
    assert recommendation.evidence["current_fx_parameter_schema"]["generic_parameter_count"] == 1
    summary = recommendation.fx_send_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["current_fx_parameter_schema_status"] == "partial"
    assert summary["fx_parameter_schema_no_live_authority"] is True
    assert "fx_parameter_schema_not_mapped" in summary["blockers"]
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert proposal["fx_send_safety_summary"] == summary


def test_generic_fx_inserts_snapshot_schema_blocks_fx_send_proposal():
    snapshot = {
        "type": "fx_inserts_snapshot",
        "safety": {
            "queries_console": False,
            "live_write": False,
        },
        "summary": {
            "read_coverage_status": "partial",
            "cache_freshness_status": "fresh",
        },
        "cache_freshness": {"status": "fresh"},
        "parameter_schema": {
            "status": "partial",
            "mapped_parameter_count": 0,
            "generic_parameter_count": 1,
            "physical_model_schema_verified": False,
            "live_write_authority": False,
        },
        "fx_slots": [
            {
                "slot": "FX1",
                "model": "CUSTOM-FX",
                "parameter_schema": {
                    "status": "partial",
                    "schema_family": "unknown",
                    "mapped_parameter_count": 0,
                    "generic_parameter_count": 1,
                    "cached_parameter_count": 1,
                    "physical_model_schema_verified": False,
                    "live_write_authority": False,
                },
            },
        ],
    }

    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_send_db=-28.0,
            arrangement_role="lead_space",
            contribution_share=0.85,
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            source_confidence=0.92,
            persistence_cycles=3,
            current_state_cache_freshness={"status": "fresh"},
            current_fx_inserts_snapshot=snapshot,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "fx_parameter_schema_not_mapped" in recommendation.blockers
    assert recommendation.evidence["current_fx_parameter_schema_status"] == "partial"
    assert recommendation.evidence["current_fx_parameter_schema"]["generic_parameter_count"] == 1
    assert recommendation.evidence["current_fx_parameter_schema"]["snapshot_cache_freshness_status"] == "fresh"
    summary = recommendation.fx_send_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["current_fx_parameter_schema_status"] == "partial"
    assert summary["fx_parameter_schema_no_live_authority"] is True
    assert "fx_parameter_schema_not_mapped" in summary["blockers"]
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert proposal["fx_send_safety_summary"] == summary


def test_mismatched_fx_parameter_schema_family_blocks_fx_send_proposal():
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_send_db=-28.0,
            tempo_bpm=120.0,
            requested_fx_type="delay",
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            source_confidence=0.92,
            persistence_cycles=3,
            current_state_cache_freshness={"status": "fresh"},
            current_fx_parameter_schema={
                "status": "mapped",
                "schema_family": "reverb",
                "mapped_parameter_count": 6,
                "generic_parameter_count": 0,
                "physical_model_schema_verified": False,
                "live_write_authority": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.fx_type == "delay"
    assert "fx_parameter_schema_family_mismatch" in recommendation.blockers
    assert recommendation.evidence["expected_fx_schema_family"] == "delay"
    assert recommendation.evidence["current_fx_parameter_schema"]["schema_family"] == "reverb"


def test_fx_parameter_schema_with_live_authority_blocks_fx_send_proposal():
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_send_db=-28.0,
            tempo_bpm=120.0,
            arrangement_role="lead_space",
            contribution_share=0.85,
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            source_confidence=0.92,
            persistence_cycles=3,
            current_state_cache_freshness={"status": "fresh"},
            current_fx_parameter_schema={
                "status": "mapped",
                "schema_family": "reverb",
                "schema_source": "unit_test_live_authority_schema",
                "mapped_parameter_count": 6,
                "generic_parameter_count": 0,
                "physical_model_schema_verified": False,
                "live_write_authority": True,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "fx_parameter_schema_live_authority_present" in recommendation.blockers
    assert recommendation.evidence["current_fx_parameter_schema_status"] == "mapped"
    assert (
        recommendation.evidence["current_fx_parameter_schema"]["live_write_authority"]
        is True
    )
    assert recommendation.evidence["fx_parameter_schema_no_live_authority"] is False
    summary = recommendation.fx_send_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["fx_parameter_schema_no_live_authority"] is False
    assert "fx_parameter_schema_live_authority_present" in summary["blockers"]
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert proposal["fx_send_safety_summary"] == summary
    assert "fx_parameter_schema_live_authority_present" in proposal["blockers"]


def test_source_identity_hold_blocks_downstream_fx_send_proposal():
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_send_db=-28.0,
            tempo_bpm=120.0,
            arrangement_role="lead_space",
            contribution_share=0.85,
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            source_confidence=0.92,
            persistence_cycles=3,
            source_identity_trust_state="hold_for_operator_review",
            source_identity_trust_confidence=0.41,
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
    summary = recommendation.fx_send_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["source_identity_usable_for_downstream"] is False
    assert "source_identity_not_trusted_for_downstream" in summary["blockers"]
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert proposal["fx_send_safety_summary"] == summary


def test_stale_source_identity_replay_context_blocks_downstream_fx_send_proposal():
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_send_db=-28.0,
            tempo_bpm=120.0,
            arrangement_role="lead_space",
            contribution_share=0.85,
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            source_confidence=0.92,
            persistence_cycles=3,
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
    summary = recommendation.fx_send_safety_summary()
    assert summary["status"] == "blocked"
    assert summary["source_identity_usable_for_downstream"] is False
    assert "source_identity_replay_context_not_fresh" in summary["blockers"]
    _assert_top_level_denial(summary)
    proposal = recommendation.to_operator_proposal()
    _assert_top_level_denial(proposal)
    assert proposal["fx_send_safety_summary"] == summary


def test_fx_helpers_classify_roles_and_tempo_notes():
    assert classify_source_family("Lead Vocal") == "lead_vocal"
    assert classify_source_family("Snare Top") == "snare"
    assert classify_source_family("Bass DI") == "bass"
    assert tempo_delay_ms(100.0, "dotted_eighth") == 450.0
    assert tempo_delay_ms(120.0, "quarter") == 500.0


def test_build_fx_send_proposals_filters_to_proposal_only_items():
    proposals = build_fx_send_proposals(
        [
            FXSendAnalysisInput(
                channel_id=6,
                source_role="lead_vocal",
                current_send_db=-30.0,
                return_headroom_db=6.0,
                feedback_risk=0.1,
                intelligibility_risk=0.2,
                source_confidence=0.9,
                persistence_cycles=3,
                current_state_cache_freshness=_fresh(),
                current_fx_parameter_schema=_schema("reverb"),
            ),
            FXSendAnalysisInput(
                channel_id=1,
                source_role="kick",
                current_send_db=-36.0,
            ),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["channel"] == 6


def test_build_fx_send_proposals_filters_stale_state_items():
    proposals = build_fx_send_proposals(
        [
            FXSendAnalysisInput(
                channel_id=6,
                source_role="lead_vocal",
                current_send_db=-30.0,
                return_headroom_db=6.0,
                feedback_risk=0.1,
                intelligibility_risk=0.2,
                source_confidence=0.9,
                persistence_cycles=3,
                current_state_cache_freshness={"status": "stale"},
            ),
        ]
    )

    assert proposals == ()


def test_build_fx_send_proposals_uses_shared_fx_snapshot_schema():
    snapshot = {
        "type": "fx_inserts_snapshot",
        "safety": {
            "queries_console": False,
            "live_write": False,
        },
        "summary": {
            "read_coverage_status": "partial",
            "cache_freshness_status": "fresh",
        },
        "cache_freshness": {"status": "fresh"},
        "parameter_schema": {
            "status": "mapped",
            "mapped_parameter_count": 6,
            "generic_parameter_count": 0,
            "physical_model_schema_verified": False,
            "live_write_authority": False,
        },
        "fx_slots": [
            {
                "slot": "FX1",
                "model": "PLATE",
                "parameter_schema": {
                    "status": "mapped",
                    "schema_family": "reverb",
                    "schema_source": "automixer_generic_fx_parameter_hints",
                    "mapped_parameter_count": 6,
                    "generic_parameter_count": 0,
                    "cached_parameter_count": 6,
                    "physical_model_schema_verified": False,
                    "live_write_authority": False,
                },
            },
        ],
    }

    proposals = build_fx_send_proposals(
        [
            FXSendAnalysisInput(
                channel_id=6,
                source_role="lead_vocal",
                current_send_db=-30.0,
                arrangement_role="lead_space",
                return_headroom_db=6.0,
                feedback_risk=0.1,
                intelligibility_risk=0.2,
                source_confidence=0.9,
                persistence_cycles=3,
                current_state_cache_freshness=_fresh(),
            ),
            FXSendAnalysisInput(
                channel_id=1,
                source_role="kick",
                current_send_db=-36.0,
                current_state_cache_freshness=_fresh(),
            ),
        ],
        current_fx_inserts_snapshot=snapshot,
    )

    assert len(proposals) == 1
    assert proposals[0]["channel"] == 6
    assert proposals[0]["evidence"]["current_fx_parameter_schema_status"] == "mapped"
    assert proposals[0]["evidence"]["current_fx_parameter_schema"]["fx_slot"] == "FX1"
    assert proposals[0]["evidence"]["current_fx_parameter_schema"]["snapshot_live_write"] is False


def test_build_fx_send_proposals_blocks_stale_shared_fx_snapshot_schema():
    snapshot = {
        "type": "fx_inserts_snapshot",
        "safety": {
            "queries_console": False,
            "live_write": False,
        },
        "summary": {
            "read_coverage_status": "partial",
            "cache_freshness_status": "stale",
        },
        "cache_freshness": {"status": "stale"},
        "parameter_schema": {
            "status": "mapped",
            "mapped_parameter_count": 6,
            "generic_parameter_count": 0,
            "physical_model_schema_verified": False,
            "live_write_authority": False,
        },
        "fx_slots": [
            {
                "slot": "FX1",
                "model": "PLATE",
                "parameter_schema": {
                    "status": "mapped",
                    "schema_family": "reverb",
                    "schema_source": "automixer_generic_fx_parameter_hints",
                    "mapped_parameter_count": 6,
                    "generic_parameter_count": 0,
                    "cached_parameter_count": 6,
                    "physical_model_schema_verified": False,
                    "live_write_authority": False,
                },
            },
        ],
    }
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_send_db=-30.0,
            arrangement_role="lead_space",
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            source_confidence=0.9,
            persistence_cycles=3,
            current_fx_inserts_snapshot=snapshot,
        )
    )
    proposals = build_fx_send_proposals(
        [
            FXSendAnalysisInput(
                channel_id=6,
                source_role="lead_vocal",
                current_send_db=-30.0,
                arrangement_role="lead_space",
                return_headroom_db=6.0,
                feedback_risk=0.1,
                intelligibility_risk=0.2,
                source_confidence=0.9,
                persistence_cycles=3,
            ),
        ],
        current_fx_inserts_snapshot=snapshot,
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.severity == "medium"
    assert "fx_inserts_snapshot_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_fx_parameter_schema_status"] == "mapped"
    assert (
        recommendation.evidence["current_fx_parameter_schema"]["snapshot_cache_freshness_status"]
        == "stale"
    )
    assert proposals == ()


def test_build_fx_send_proposals_filters_generic_shared_fx_snapshot_schema():
    snapshot = {
        "type": "fx_inserts_snapshot",
        "safety": {
            "queries_console": False,
            "live_write": False,
        },
        "summary": {
            "read_coverage_status": "partial",
            "cache_freshness_status": "fresh",
        },
        "cache_freshness": {"status": "fresh"},
        "parameter_schema": {
            "status": "partial",
            "mapped_parameter_count": 0,
            "generic_parameter_count": 1,
            "physical_model_schema_verified": False,
            "live_write_authority": False,
        },
        "fx_slots": [
            {
                "slot": "FX1",
                "model": "CUSTOM-FX",
                "parameter_schema": {
                    "status": "partial",
                    "schema_family": "unknown",
                    "mapped_parameter_count": 0,
                    "generic_parameter_count": 1,
                    "cached_parameter_count": 1,
                    "physical_model_schema_verified": False,
                    "live_write_authority": False,
                },
            },
        ],
    }

    proposals = build_fx_send_proposals(
        [
            FXSendAnalysisInput(
                channel_id=6,
                source_role="lead_vocal",
                current_send_db=-30.0,
                arrangement_role="lead_space",
                return_headroom_db=6.0,
                feedback_risk=0.1,
                intelligibility_risk=0.2,
                source_confidence=0.9,
                persistence_cycles=3,
                current_state_cache_freshness={"status": "fresh"},
            ),
        ],
        current_fx_inserts_snapshot=snapshot,
    )

    assert proposals == ()


def test_fx_send_proposal_is_registered_as_goal_capability():
    method = get_processing_method("fx_send_proposal")
    main = get_pipeline_profile("main_live_assist")
    effects = get_project_goal_capability("effects_management")
    sends = get_project_goal_capability("effect_sends")
    console = get_project_goal_capability("console_fx_control")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_write_live is False
    assert "schema hints" in method.summary
    assert "fx_snapshot_freshness_gate" in method.notes
    assert "fx_parameter_schema_gate" in method.notes
    assert "fx_inserts_snapshot_schema_bridge" in method.notes
    assert "batch_fx_snapshot_schema_bridge" in method.notes
    assert main is not None
    assert "fx_send_proposal" in main.proposal_method_ids
    assert effects is not None and effects.status == "proposal_only"
    assert sends is not None and sends.status == "proposal_only"
    assert console is not None and console.status == "proposal_only"
    assert "backend/source_identity_downstream_gate.py" in effects.support_paths
    assert "backend/source_identity_downstream_gate.py" in sends.support_paths
    assert "backend/source_identity_downstream_gate.py" in console.support_paths
    assert contracts["fx_send_proposal"]["proposal_contract"]["value_type"] == "fx_send"
    assert (
        "current_fx_send_return_cache_freshness"
        in contracts["fx_send_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "current_fx_send_return_cache_no_live_authority"
        in contracts["fx_send_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "current_fx_parameter_schema_hints"
        in contracts["fx_send_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "source_identity_downstream_trust"
        in contracts["fx_send_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "monitor_safety"
        in contracts["fx_send_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "return_level_safety"
        in contracts["fx_send_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "old_value_source"
        in contracts["fx_send_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
