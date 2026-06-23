from pathlib import Path

from backend.frequency_separation_proposal import (
    READ_REQUIRED_SOURCE,
    FrequencySeparationSource,
    analyze_frequency_separation_need,
    build_frequency_separation_proposals,
)
from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability


def _freshness(status="fresh"):
    return {
        "status": status,
        "field_count": 6,
        "fresh_field_count": 6 if status == "fresh" else 4,
        "stale_field_count": 0 if status == "fresh" else 2,
        "stale_fields": [] if status == "fresh" else ["stem_spectrum", "eq_band_snapshot"],
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def _assert_top_level_denial(payload):
    assert payload["queue_action"] == "not_queueable_until_promoted"
    assert payload["can_import_to_queue"] is False
    assert payload["can_apply_live"] is False
    assert payload["queue_import_allowed"] is False
    assert payload["live_apply_allowed"] is False
    assert payload["read_only_or_dry_run_only"] is True
    assert payload["live_authority_granted"] is False
    assert payload["readiness_closure_allowed"] is False
    assert payload["mutates_mixer"] is False


def _sources():
    return (
        FrequencySeparationSource(
            source_id="lead_vox",
            channel_id=1,
            label="Lead Vox",
            source_type="lead_vocal",
            arrangement_role="lead",
            stem_id="band",
            priority=1,
            band_energy_db={"high_mid": -20.0},
            stem_contribution={"high_mid": 0.44},
            confidence=0.88,
        ),
        FrequencySeparationSource(
            source_id="gtr_l",
            channel_id=7,
            label="GTR L",
            source_type="guitar",
            arrangement_role="harmonic_support",
            stem_id="band",
            priority=3,
            band_energy_db={"high_mid": -21.0},
            stem_contribution={"high_mid": 0.37},
            confidence=0.81,
        ),
    )


def test_frequency_separation_emits_fresh_context_lane_proposal():
    result = analyze_frequency_separation_need(
        _sources(),
        current_state_cache_freshness=_freshness(),
    )

    assert result.method_id == "frequency_separation_proposal"
    assert result.status == "proposal_only"
    assert result.metadata["proposal_only"] is True
    assert result.metadata["live_write_allowed"] is False
    _assert_top_level_denial(result.metadata)
    assert result.metadata["current_state_cache_freshness_status"] == "fresh"
    assert result.metadata["current_frequency_context_cache_no_live_authority"] is True
    assert result.recommendations

    recommendation = result.recommendations[0]
    assert recommendation.action_type == "frequency_lane_separation_candidate"
    assert recommendation.target_source_id == "gtr_l"
    assert recommendation.protected_source_id == "lead_vox"
    assert recommendation.band == "high_mid"
    assert recommendation.suggested_eq_delta_db < 0.0
    assert recommendation.evidence["source_stem_contribution"]["target"] == 0.37
    assert recommendation.evidence["lane_conflict_pair"]["protected_source_id"] == "lead_vox"
    assert recommendation.evidence["bounded_subtractive_delta"] is True
    assert recommendation.evidence["current_frequency_context_cache_freshness"] == "fresh"
    assert recommendation.evidence["current_frequency_context_cache_no_live_authority"] is True
    assert recommendation.evidence["sum_bus_check"] == "required_before_apply"
    assert recommendation.evidence["eq_band_readback"] == "required_before_apply"
    assert recommendation.evidence["rollback_value_source"] == READ_REQUIRED_SOURCE


def test_stale_frequency_context_blocks_frequency_separation_proposals():
    result = analyze_frequency_separation_need(
        _sources(),
        current_state_cache_freshness=_freshness("stale"),
    )

    assert result.status == "analysis_only"
    assert result.recommendations == ()
    assert "frequency separation context cache is not fresh" in " ".join(result.warnings)
    assert result.metadata["current_state_cache_freshness_status"] == "stale"
    assert "frequency_separation_context_cache_not_fresh" in result.metadata["blockers"]
    assert result.metadata["live_write_allowed"] is False
    _assert_top_level_denial(result.metadata)


def test_live_authority_frequency_context_blocks_frequency_separation_proposals():
    live_authority_context = {
        "status": "fresh",
        "field_count": 6,
        "fresh_field_count": 6,
        "read_only": False,
        "live_write": True,
        "queries_console": True,
    }
    result = analyze_frequency_separation_need(
        _sources(),
        current_state_cache_freshness=live_authority_context,
    )

    assert result.status == "analysis_only"
    assert result.recommendations == ()
    assert "frequency separation context cache has live authority" in " ".join(
        result.warnings
    )
    assert result.metadata["current_state_cache_freshness_status"] == "fresh"
    assert result.metadata["current_frequency_context_cache_no_live_authority"] is False
    assert result.metadata["current_state_cache_freshness"]["live_write"] is True
    _assert_top_level_denial(result.metadata)
    assert "frequency_separation_context_cache_live_authority_present" in result.metadata[
        "blockers"
    ]
    assert (
        build_frequency_separation_proposals(
            _sources(),
            current_state_cache_freshness=live_authority_context,
        )
        == ()
    )


def test_missing_frequency_context_blocks_frequency_separation_proposals():
    result = analyze_frequency_separation_need(_sources())

    assert result.status == "analysis_only"
    assert result.recommendations == ()
    assert "frequency separation context cache is not fresh" in " ".join(result.warnings)
    assert result.metadata["current_state_cache_freshness_status"] == "not_provided"
    assert result.metadata["current_state_cache_freshness"]["provided"] is False
    assert "frequency_separation_context_cache_not_fresh" in result.metadata["blockers"]
    _assert_top_level_denial(result.metadata)
    assert build_frequency_separation_proposals(_sources()) == ()


def test_blocked_source_identity_blocks_frequency_separation_proposals():
    lead, guitar = _sources()
    result = analyze_frequency_separation_need(
        (
            lead,
            FrequencySeparationSource(
                source_id=guitar.source_id,
                channel_id=guitar.channel_id,
                label=guitar.label,
                source_type=guitar.source_type,
                arrangement_role=guitar.arrangement_role,
                stem_id=guitar.stem_id,
                priority=guitar.priority,
                band_energy_db=guitar.band_energy_db,
                stem_contribution=guitar.stem_contribution,
                confidence=guitar.confidence,
                source_identity_trust_state="blocked_from_downstream_trust",
                source_identity_trust_confidence=0.8,
                source_identity_replay_outcome_count=5,
                source_identity_calibration_freshness=_freshness(),
            ),
        ),
        current_state_cache_freshness=_freshness(),
    )

    assert result.status == "analysis_only"
    assert result.recommendations == ()
    assert "source identity replay calibration blocks frequency separation" in " ".join(result.warnings)
    assert "source_identity_blocked_from_downstream_trust" in result.metadata["blockers"]
    assert result.metadata["source_identity_blocked_source_count"] == 1
    blocked = result.metadata["source_identity_blocked_sources"][0]
    assert blocked["source_id"] == "gtr_l"
    assert blocked["usable_for_downstream"] is False
    assert blocked["evidence"]["memory_mutation"] is False
    assert result.metadata["live_write_allowed"] is False
    _assert_top_level_denial(result.metadata)


def test_stale_source_identity_replay_context_blocks_frequency_separation():
    lead, guitar = _sources()
    result = analyze_frequency_separation_need(
        (
            FrequencySeparationSource(
                source_id=lead.source_id,
                channel_id=lead.channel_id,
                label=lead.label,
                source_type=lead.source_type,
                arrangement_role=lead.arrangement_role,
                stem_id=lead.stem_id,
                priority=lead.priority,
                band_energy_db=lead.band_energy_db,
                stem_contribution=lead.stem_contribution,
                confidence=lead.confidence,
                source_identity_trust_state="trusted_for_downstream_priority",
                source_identity_trust_confidence=0.9,
                source_identity_replay_outcome_count=6,
                source_identity_calibration_freshness=_freshness("stale"),
            ),
            guitar,
        ),
        current_state_cache_freshness=_freshness(),
    )

    assert result.status == "analysis_only"
    assert result.recommendations == ()
    assert "source_identity_replay_context_not_fresh" in result.metadata["blockers"]
    assert result.metadata["source_identity_blocked_sources"][0]["source_id"] == "lead_vox"
    _assert_top_level_denial(result.metadata)


def test_frequency_separation_operator_proposals_are_not_queueable_live_actions():
    proposals = build_frequency_separation_proposals(
        _sources(),
        current_state_cache_freshness=_freshness(),
    )

    assert proposals
    proposal = proposals[0]

    assert proposal["method_id"] == "frequency_separation_proposal"
    assert proposal["kind"] == "frequency_split"
    _assert_top_level_denial(proposal)
    assert "unsupported_value_type:frequency_split" in proposal["blockers"]
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["current_frequency_context_cache_no_live_authority"] is True
    assert proposal["evidence"]["readback_required_before_promotion"] is True
    assert proposal["evidence"]["rollback_required_before_promotion"] is True
    summary = proposal["evidence"]["frequency_lane_safety_summary"]
    assert summary == proposal["proposed_action"]["frequency_lane_safety_summary"]
    assert summary["summary_type"] == "frequency_lane_separation_review"
    assert summary["target_source_id"] == "gtr_l"
    assert summary["protected_source_id"] == "lead_vox"
    assert summary["band"] == "high_mid"
    assert summary["bounded_subtractive_delta"] is True
    assert summary["current_frequency_context_cache_no_live_authority"] is True
    assert summary["feedback_safety"] == "required_before_apply"
    assert summary["true_peak_headroom"] == "required_before_apply"
    _assert_top_level_denial(summary)
    assert summary["can_authorize_live"] is False
    correction = proposal["operator_correction"]
    assert correction["objective_family"] == "eq"
    assert correction["bounded_delta"]["delta_db"] == round(
        proposal["proposed_action"]["suggested_eq_delta_db"], 1
    )
    assert correction["bounded_delta"]["parameter_deltas"]["band"] == "high_mid"
    assert correction["bounded_delta"]["parameter_deltas"]["center_hz"] == 3000.0
    assert correction["future_wing_readback_contract"]["target_kind"] == "source_stem_frequency_lane"
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "current_frequency_context_cache_no_live_authority"
    ] is True
    completeness = correction["source_evidence"]["evidence_completeness"]
    assert completeness["status"] == "complete"
    assert completeness["missing"] == []
    assert correction["operator_proposal_review_gate"]["evidence_gate_status"] == (
        "required_evidence_complete_operator_review_still_required"
    )
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_top_level_denial(source_evidence)
    assert source_evidence["frequency_lane_safety_summary"] == summary
    assert source_evidence["current_frequency_context_cache_no_live_authority"] is True
    assert source_evidence["source_identity_downstream_trust"]["target_source"][
        "source_id"
    ] == "gtr_l"
    assert source_evidence["source_identity_downstream_trust"]["protected_source"][
        "source_id"
    ] == "lead_vox"
    assert source_evidence["rollback_eq_band_readback"] == {
        "target_source_id": "gtr_l",
        "protected_source_id": "lead_vox",
        "stem_id": "band",
        "band": "high_mid",
        "center_hz": 3000.0,
        "suggested_eq_delta_db": proposal["proposed_action"][
            "suggested_eq_delta_db"
        ],
        "source": READ_REQUIRED_SOURCE,
        "required_before_live_promotion": True,
        "live_write": False,
    }
    assert source_evidence["readback_required_before_promotion"] is True
    assert source_evidence["rollback_required_before_promotion"] is True


def test_frequency_separation_registry_contract_and_goal_row():
    method = get_processing_method("frequency_separation_proposal")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}
    row = get_project_goal_capability("frequency_separation")

    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_analyze is True
    assert method.can_propose is True
    assert method.can_write_live is False
    assert "main_live_assist" in method.pipeline_profiles
    assert method.blocked_reason is not None
    assert "fresh_source_stem_spectrum" in method.blocked_reason

    contract = contracts["frequency_separation_proposal"]["proposal_contract"]
    evidence = contract["safety_evidence_required"]

    assert contract["value_type"] == "frequency_split"
    assert contract["target_kind"] == "source_stem_frequency_lane"
    assert "current_frequency_context_cache_freshness" in evidence
    assert "current_frequency_context_cache_no_live_authority" in evidence
    assert "source_stem_contribution" in evidence
    assert "rollback_eq_band_readback" in evidence
    assert contract["live_apply_allowed"] is False

    assert row is not None
    assert row.status == "proposal_only"
    assert "frequency_separation_proposal" in row.method_ids
    assert row.current_live_write_kinds == ()
    assert "operator review" in row.next_step


def test_frequency_separation_module_is_transport_free():
    source = Path("backend/frequency_separation_proposal.py").read_text()

    forbidden_fragments = [
        "WingClient",
        "DLiveClient",
        "MixingStation",
        "mixer_client",
        "pythonosc",
        "python-osc",
        "set_eq_band",
        ".send(",
    ]

    for fragment in forbidden_fragments:
        assert fragment not in source
