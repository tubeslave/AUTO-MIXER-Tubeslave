from backend.cross_adaptive_eq import CrossAdaptiveEQ, READ_REQUIRED_SOURCE
from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability


def _freshness(status="fresh"):
    return {
        "status": status,
        "field_count": 5,
        "fresh_field_count": 5 if status == "fresh" else 3,
        "stale_field_count": 0 if status == "fresh" else 2,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def _identity(
    source_id,
    source_type,
    trust_state="trusted_for_downstream_priority",
    freshness="fresh",
):
    return {
        "source_id": source_id,
        "source_type": source_type,
        "trust_state": trust_state,
        "trust_confidence": 0.86,
        "replay_outcome_count": 5,
        "current_state_cache_freshness": _freshness(freshness),
    }


def _arrangement(role, source_role, confidence=0.84):
    return {
        "arrangement_role": role,
        "source_role": source_role,
        "role_confidence": confidence,
        "read_only": True,
        "live_write": False,
    }


def _arrangement_context():
    return {
        1: _arrangement("lead", "lead_vocal"),
        2: _arrangement("harmonic_support", "guitar"),
    }


def _band_energy():
    return {
        1: {"mid": -20.0},
        2: {"mid": -21.0},
    }


def test_cross_adaptive_eq_emits_linked_proposal_actions_with_rollback_evidence():
    processor = CrossAdaptiveEQ(overlap_tolerance_db=6.0, max_cut_db=-6.0, max_boost_db=3.0)

    result = processor.analyze_unmasking_proposals(
        channel_band_energy=_band_energy(),
        channel_priorities={1: 1, 2: 3},
        current_state_cache_freshness=_freshness(),
        arrangement_context_cache_freshness=_freshness(),
        channel_arrangement_context=_arrangement_context(),
    )

    assert result.method_id == "cross_adaptive_eq"
    assert result.metadata["proposal_only"] is True
    assert result.metadata["live_write_allowed"] is False
    assert result.metadata["current_state_cache_freshness_status"] == "fresh"
    assert result.metadata["current_masking_context_cache_no_live_authority"] is True
    assert result.metadata["current_arrangement_context_cache_no_live_authority"] is True
    assert result.metadata["source_identity_downstream_trust"][0]["status"] == "legacy_unverified"
    assert len(result.adjustments) == 2
    assert len(result.suggested_actions) == 2

    cut = next(item for item in result.adjustments if item.role == "masker_cut")
    boost = next(item for item in result.adjustments if item.role == "masked_source_boost")

    assert cut.channel_id == 2
    assert cut.paired_channel_id == 1
    assert cut.masked_channel_id == 1
    assert cut.masker_channel_id == 2
    assert boost.channel_id == 1
    assert boost.paired_channel_id == 2
    assert boost.frequency_hz == cut.frequency_hz

    cut_action = next(item for item in result.suggested_actions if item["role"] == "masker_cut")
    evidence = cut_action["safety_evidence"]

    assert cut_action["action_type"] == "cross_channel_unmasking_eq_candidate"
    assert cut_action["target_kind"] == "masking_eq_band"
    assert cut_action["live_write_allowed"] is False
    assert cut_action["can_import_to_queue"] is False
    assert cut_action["old_value"]["source"] == READ_REQUIRED_SOURCE
    assert cut_action["rollback_value"]["source"] == READ_REQUIRED_SOURCE
    assert cut_action["rollback_required"] is True
    assert evidence["masking_pair"] is True
    assert evidence["linked_cut_boost_pair"] is True
    assert evidence["bounded_gain_delta"] is True
    assert evidence["current_masking_context_cache_freshness"] == "fresh"
    assert evidence["current_masking_context_cache_no_live_authority"] is True
    assert evidence["current_arrangement_context_cache_freshness"] == "fresh"
    assert evidence["current_arrangement_context_cache_no_live_authority"] is True
    assert evidence["source_identity_downstream_trust"]["masked_channel"]["status"] == (
        "legacy_unverified"
    )
    assert evidence["source_identity_downstream_trust"]["masker_channel"]["status"] == (
        "legacy_unverified"
    )
    assert evidence["sum_bus_check"] == "required_before_apply"
    assert evidence["eq_band_readback"] == "required_before_apply"


def test_fresh_arrangement_context_can_supply_unmasking_priorities():
    processor = CrossAdaptiveEQ(overlap_tolerance_db=6.0)

    result = processor.analyze_unmasking_proposals(
        channel_band_energy=_band_energy(),
        channel_priorities={},
        current_state_cache_freshness=_freshness(),
        arrangement_context_cache_freshness=_freshness(),
        channel_arrangement_context=_arrangement_context(),
    )

    assert len(result.adjustments) == 2
    assert result.metadata["arrangement_context_cache_freshness_status"] == "fresh"
    assert result.metadata["effective_channel_priorities"] == {"1": 1, "2": 3}
    assert result.metadata["channel_priority_sources"] == {
        "1": "arrangement_role_hint",
        "2": "arrangement_role_hint",
    }

    cut = next(item for item in result.adjustments if item.role == "masker_cut")
    boost = next(item for item in result.adjustments if item.role == "masked_source_boost")
    assert cut.channel_id == 2
    assert boost.channel_id == 1

    cut_action = next(item for item in result.suggested_actions if item["role"] == "masker_cut")
    arrangement_evidence = cut_action["safety_evidence"]["arrangement_priority_context"]
    assert arrangement_evidence["masked_channel"]["arrangement_role"] == "lead"
    assert arrangement_evidence["masked_channel"]["priority_source"] == "arrangement_role_hint"
    assert arrangement_evidence["masker_channel"]["arrangement_role"] == "harmonic_support"
    assert cut_action["safety_evidence"]["current_arrangement_context_cache_freshness"] == "fresh"
    assert cut_action["live_write_allowed"] is False


def test_stale_masking_context_blocks_unmasking_proposals():
    processor = CrossAdaptiveEQ()

    result = processor.analyze_unmasking_proposals(
        channel_band_energy=_band_energy(),
        channel_priorities={1: 1, 2: 3},
        current_state_cache_freshness=_freshness("stale"),
        arrangement_context_cache_freshness=_freshness(),
        channel_arrangement_context=_arrangement_context(),
    )

    assert result.adjustments == ()
    assert result.suggested_actions == ()
    assert "cross-channel masking context cache is not fresh" in " ".join(result.warnings)
    assert result.metadata["current_state_cache_freshness_status"] == "stale"
    assert "cross_adaptive_eq_context_cache_not_fresh" in result.metadata["blockers"]
    assert result.metadata["live_write_allowed"] is False


def test_missing_masking_context_blocks_unmasking_proposals():
    processor = CrossAdaptiveEQ()

    result = processor.analyze_unmasking_proposals(
        channel_band_energy=_band_energy(),
        channel_priorities={1: 1, 2: 3},
        arrangement_context_cache_freshness=_freshness(),
        channel_arrangement_context=_arrangement_context(),
    )

    assert result.adjustments == ()
    assert result.suggested_actions == ()
    assert "cross-channel masking context cache is not fresh" in " ".join(result.warnings)
    assert result.metadata["current_state_cache_freshness_status"] == "not_provided"
    assert result.metadata["current_state_cache_freshness"]["provided"] is False
    assert "cross_adaptive_eq_context_cache_not_fresh" in result.metadata["blockers"]
    assert result.metadata["live_write_allowed"] is False


def test_live_authority_masking_context_blocks_unmasking_proposals():
    processor = CrossAdaptiveEQ()
    live_authority_context = {
        "status": "fresh",
        "field_count": 5,
        "fresh_field_count": 5,
        "read_only": False,
        "live_write": True,
        "queries_console": True,
    }

    result = processor.analyze_unmasking_proposals(
        channel_band_energy=_band_energy(),
        channel_priorities={1: 1, 2: 3},
        current_state_cache_freshness=live_authority_context,
        arrangement_context_cache_freshness=_freshness(),
        channel_arrangement_context=_arrangement_context(),
    )

    assert result.adjustments == ()
    assert result.suggested_actions == ()
    assert "cross-channel masking context cache has live authority" in " ".join(
        result.warnings
    )
    assert result.metadata["current_state_cache_freshness_status"] == "fresh"
    assert result.metadata["current_masking_context_cache_no_live_authority"] is False
    assert result.metadata["current_state_cache_freshness"]["live_write"] is True
    assert "cross_adaptive_eq_context_cache_live_authority_present" in result.metadata[
        "blockers"
    ]
    assert result.metadata["live_write_allowed"] is False


def test_stale_arrangement_context_blocks_unmasking_proposals():
    processor = CrossAdaptiveEQ()

    result = processor.analyze_unmasking_proposals(
        channel_band_energy=_band_energy(),
        channel_priorities={1: 1, 2: 3},
        current_state_cache_freshness=_freshness(),
        arrangement_context_cache_freshness=_freshness("stale"),
        channel_arrangement_context={
            1: _arrangement("lead", "lead_vocal"),
            2: _arrangement("harmonic_support", "guitar"),
        },
    )

    assert result.adjustments == ()
    assert result.suggested_actions == ()
    assert "cross-channel arrangement context cache is not fresh" in " ".join(result.warnings)
    assert result.metadata["arrangement_context_cache_freshness_status"] == "stale"
    assert "cross_adaptive_eq_arrangement_context_cache_not_fresh" in result.metadata["blockers"]
    assert result.metadata["live_write_allowed"] is False


def test_live_authority_arrangement_context_blocks_unmasking_proposals():
    processor = CrossAdaptiveEQ()
    live_authority_context = {
        "status": "fresh",
        "field_count": 5,
        "fresh_field_count": 5,
        "read_only": False,
        "live_write": True,
        "queries_console": True,
    }

    result = processor.analyze_unmasking_proposals(
        channel_band_energy=_band_energy(),
        channel_priorities={1: 1, 2: 3},
        current_state_cache_freshness=_freshness(),
        arrangement_context_cache_freshness=live_authority_context,
        channel_arrangement_context=_arrangement_context(),
    )

    assert result.adjustments == ()
    assert result.suggested_actions == ()
    assert "cross-channel arrangement context cache has live authority" in " ".join(
        result.warnings
    )
    assert result.metadata["arrangement_context_cache_freshness_status"] == "fresh"
    assert result.metadata["current_arrangement_context_cache_no_live_authority"] is False
    assert result.metadata["arrangement_context_cache_freshness"]["queries_console"] is True
    assert (
        "cross_adaptive_eq_arrangement_context_cache_live_authority_present"
        in result.metadata["blockers"]
    )
    assert result.metadata["live_write_allowed"] is False


def test_missing_arrangement_freshness_blocks_unmasking_proposals():
    processor = CrossAdaptiveEQ()

    result = processor.analyze_unmasking_proposals(
        channel_band_energy=_band_energy(),
        channel_priorities={1: 1, 2: 3},
        current_state_cache_freshness=_freshness(),
        channel_arrangement_context=_arrangement_context(),
    )

    assert result.adjustments == ()
    assert result.suggested_actions == ()
    assert "cross-channel arrangement context cache is not fresh" in " ".join(result.warnings)
    assert result.metadata["arrangement_context_cache_freshness_status"] == "not_provided"
    assert result.metadata["arrangement_context_cache_freshness"]["provided"] is False
    assert "cross_adaptive_eq_arrangement_context_cache_not_fresh" in result.metadata["blockers"]
    assert result.metadata["live_write_allowed"] is False


def test_missing_arrangement_context_blocks_unmasking_proposals():
    processor = CrossAdaptiveEQ()

    result = processor.analyze_unmasking_proposals(
        channel_band_energy=_band_energy(),
        channel_priorities={1: 1, 2: 3},
        current_state_cache_freshness=_freshness(),
        arrangement_context_cache_freshness=_freshness(),
    )

    assert result.adjustments == ()
    assert result.suggested_actions == ()
    assert "arrangement-role evidence blocks cross-channel unmasking" in " ".join(result.warnings)
    assert "cross_adaptive_eq_arrangement_context_missing_for_channel" in result.metadata["blockers"]
    assert result.metadata["arrangement_context_blocked_channel_count"] == 2
    assert result.metadata["live_write_allowed"] is False


def test_low_confidence_arrangement_context_blocks_unmasking_proposals():
    processor = CrossAdaptiveEQ()

    result = processor.analyze_unmasking_proposals(
        channel_band_energy=_band_energy(),
        channel_priorities={},
        current_state_cache_freshness=_freshness(),
        arrangement_context_cache_freshness=_freshness(),
        channel_arrangement_context={
            1: _arrangement("lead", "lead_vocal", confidence=0.42),
            2: _arrangement("harmonic_support", "guitar"),
        },
    )

    assert result.adjustments == ()
    assert result.suggested_actions == ()
    assert "arrangement-role evidence blocks cross-channel unmasking" in " ".join(result.warnings)
    assert "cross_adaptive_eq_arrangement_role_confidence_low" in result.metadata["blockers"]
    blocked = result.metadata["arrangement_context_blocked_channels"][0]
    assert blocked["channel_id"] == 1
    assert blocked["usable_for_unmasking_priority"] is False
    assert result.metadata["live_write_allowed"] is False


def test_held_source_identity_blocks_unmasking_proposals():
    processor = CrossAdaptiveEQ()

    result = processor.analyze_unmasking_proposals(
        channel_band_energy=_band_energy(),
        channel_priorities={1: 1, 2: 3},
        current_state_cache_freshness=_freshness(),
        arrangement_context_cache_freshness=_freshness(),
        channel_arrangement_context=_arrangement_context(),
        channel_source_identity={
            1: _identity("lead_vox", "lead_vocal"),
            2: _identity("gtr_l", "guitar", trust_state="hold_for_operator_review"),
        },
    )

    assert result.adjustments == ()
    assert result.suggested_actions == ()
    assert "source identity replay calibration blocks cross-channel unmasking" in " ".join(result.warnings)
    assert "source_identity_not_trusted_for_downstream" in result.metadata["blockers"]
    assert result.metadata["source_identity_blocked_channel_count"] == 1
    blocked = result.metadata["source_identity_blocked_channels"][0]
    assert blocked["channel_id"] == 2
    assert blocked["source_id"] == "gtr_l"
    assert blocked["usable_for_downstream"] is False
    assert blocked["evidence"]["live_write"] is False
    assert blocked["evidence"]["queue_mutation"] is False
    assert blocked["evidence"]["model_mutation"] is False
    assert blocked["evidence"]["memory_mutation"] is False
    assert result.metadata["live_write_allowed"] is False
    assert result.metadata["can_import_to_queue"] is False


def test_stale_source_identity_replay_context_blocks_unmasking_proposals():
    processor = CrossAdaptiveEQ()

    result = processor.analyze_unmasking_proposals(
        channel_band_energy=_band_energy(),
        channel_priorities={1: 1, 2: 3},
        current_state_cache_freshness=_freshness(),
        arrangement_context_cache_freshness=_freshness(),
        channel_arrangement_context=_arrangement_context(),
        channel_source_identity={
            1: _identity("lead_vox", "lead_vocal", freshness="stale"),
            2: _identity("gtr_l", "guitar"),
        },
    )

    assert result.adjustments == ()
    assert result.suggested_actions == ()
    assert "source_identity_replay_context_not_fresh" in result.metadata["blockers"]
    blocked = result.metadata["source_identity_blocked_channels"][0]
    assert blocked["channel_id"] == 1
    assert blocked["source_id"] == "lead_vox"
    assert blocked["status"] == "trusted"
    assert blocked["usable_for_downstream"] is False
    assert blocked["evidence"]["current_source_identity_replay_context_cache_freshness"] == "stale"


def test_legacy_cross_adaptive_calculate_corrections_still_returns_adjustments():
    processor = CrossAdaptiveEQ()

    adjustments = processor.calculate_corrections(
        channel_band_energy=_band_energy(),
        channel_priorities={1: 1, 2: 3},
    )

    assert len(adjustments) == 2
    assert {item.role for item in adjustments} == {"masker_cut", "masked_source_boost"}
    assert {item.paired_channel_id for item in adjustments} == {1, 2}


def test_cross_adaptive_eq_contract_and_goal_row_require_fresh_context():
    method = get_processing_method("cross_adaptive_eq")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}
    row = get_project_goal_capability("cross_channel_unmasking")

    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_write_live is False
    assert method.blocked_reason is not None
    assert "fresh_masking_context" in method.blocked_reason
    assert "rollback EQ-band evidence" in (method.promotion_next or "")
    assert row is not None
    assert row.status == "proposal_only"
    assert "backend/arrangement_role_proposal.py" in row.support_paths
    assert "backend/source_identity_downstream_gate.py" in row.support_paths
    assert row.current_live_write_kinds == ()
    assert "trusted source identity" in row.next_step
    assert "fresh masking and arrangement context" in row.next_step

    contract = contracts["cross_adaptive_eq"]["proposal_contract"]
    evidence = contract["safety_evidence_required"]

    assert contract["value_type"] == "eq"
    assert contract["target_kind"] == "masking_eq_band"
    assert "current_masking_context_cache_freshness" in evidence
    assert "current_masking_context_cache_no_live_authority" in evidence
    assert "current_arrangement_context_cache_freshness" in evidence
    assert "current_arrangement_context_cache_no_live_authority" in evidence
    assert "arrangement_priority_context" in evidence
    assert "source_identity_downstream_trust" in evidence
    assert "rollback_eq_band_readback" in evidence
    assert contract["live_apply_allowed"] is False
