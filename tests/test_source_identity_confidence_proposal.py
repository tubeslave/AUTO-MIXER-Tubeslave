from pathlib import Path

from backend.channel_recognizer import classify_channel_name
from backend.processing_proposal_adapters import (
    adapt_source_identity_confidence_proposals,
    build_proposal_only_method_contracts,
)
from backend.processing_registry import get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability
from backend.source_identity_confidence_proposal import (
    SourceIdentityAlternative,
    SourceIdentityObservation,
    analyze_source_identity_confidence,
    build_source_identity_confidence_proposals,
)


def _freshness(status="fresh"):
    return {
        "status": status,
        "field_count": 5,
        "fresh_field_count": 5 if status == "fresh" else 3,
        "stale_field_count": 0 if status == "fresh" else 2,
        "stale_fields": [] if status == "fresh" else ["classifier_output", "operator_label"],
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def test_low_confidence_source_identity_emits_review_proposal_with_downstream_rejections():
    observation = SourceIdentityObservation(
        source_id="ch12",
        channel_id=12,
        channel_label="Mystery Pod",
        predicted_source_type="unknown",
        source_type_confidence=0.32,
        match_type="unknown",
        classifier_source="channel_recognizer",
        signal_present=True,
        activity_confidence=0.72,
    )

    result = analyze_source_identity_confidence(
        (observation,),
        current_state_cache_freshness=_freshness(),
    )

    assert result.method_id == "source_identity_confidence_proposal"
    assert result.status == "proposal_only"
    assert result.metadata["proposal_only"] is True
    assert result.metadata["live_write_allowed"] is False
    assert result.metadata["current_state_cache_freshness_status"] == "fresh"
    assert result.metadata["current_source_identity_context_cache_no_live_authority"] is True
    assert result.metadata["review_candidate_count"] == 1

    recommendation = result.recommendations[0]
    assert recommendation.action_type == "source_identity_review_candidate"
    assert recommendation.proposed_source_type == "unknown"
    assert recommendation.proposal_priority == 1
    assert recommendation.downstream_priority_multiplier == 0.0
    assert "source_identity_unknown" in recommendation.review_reasons
    assert "source_type_confidence_low" in recommendation.review_reasons
    assert (
        "source_type_confidence_below_downstream_threshold"
        in recommendation.downstream_rejection_reasons
    )
    assert recommendation.evidence["current_source_identity_context_cache_freshness"] == "fresh"
    assert (
        recommendation.evidence["current_source_identity_context_cache_no_live_authority"]
        is True
    )
    assert recommendation.evidence["transport_free"] is True


def test_ambiguous_source_identity_uses_alternative_margin_for_review_priority():
    observation = SourceIdentityObservation(
        source_id="ch05",
        channel_id=5,
        channel_label="GTR Maybe",
        predicted_source_type="electric_guitar",
        source_type_confidence=0.66,
        match_type="ml_top_k",
        classifier_source="ml.channel_classifier",
        alternatives=(
            SourceIdentityAlternative("electric_guitar", 0.66, "ml"),
            SourceIdentityAlternative("keys_piano", 0.58, "ml"),
        ),
        stem_roles=("GUITARS",),
        allowed_controls=("eq", "compressor", "fader"),
    )

    result = analyze_source_identity_confidence(
        (observation,),
        current_state_cache_freshness=_freshness(),
    )

    recommendation = result.recommendations[0]
    assert recommendation.proposal_priority == 2
    assert recommendation.downstream_priority_multiplier == 0.5
    assert "source_identity_ambiguous" in recommendation.review_reasons
    assert recommendation.evidence["competing_source_type"] == "keys"
    assert recommendation.evidence["alternative_margin"] == 0.08


def test_stale_source_identity_context_blocks_review_proposals():
    observation = SourceIdentityObservation(
        source_id="ch01",
        channel_id=1,
        channel_label="Lead Vox",
        predicted_source_type="lead_vocal",
        source_type_confidence=0.55,
        match_type="regex",
    )

    result = analyze_source_identity_confidence(
        (observation,),
        current_state_cache_freshness=_freshness("stale"),
    )

    assert result.status == "analysis_only"
    assert result.recommendations == ()
    assert "source identity context cache is not fresh" in " ".join(result.warnings)
    assert result.metadata["current_state_cache_freshness_status"] == "stale"
    assert "source_identity_context_cache_not_fresh" in result.metadata["blockers"]
    assert result.metadata["live_write_allowed"] is False


def test_missing_source_identity_context_blocks_review_proposals():
    observation = SourceIdentityObservation(
        source_id="ch12",
        channel_id=12,
        channel_label="Mystery Pod",
        predicted_source_type="unknown",
        source_type_confidence=0.32,
        match_type="unknown",
        classifier_source="channel_recognizer",
        signal_present=True,
        activity_confidence=0.72,
    )

    result = analyze_source_identity_confidence((observation,))

    assert result.status == "analysis_only"
    assert result.recommendations == ()
    assert "source identity context cache is not fresh" in " ".join(result.warnings)
    assert result.metadata["current_state_cache_freshness_status"] == "not_provided"
    assert result.metadata["current_state_cache_freshness"]["provided"] is False
    assert "source_identity_context_cache_not_fresh" in result.metadata["blockers"]
    assert result.metadata["live_write_allowed"] is False
    assert build_source_identity_confidence_proposals((observation,)) == ()


def test_live_authority_source_identity_context_blocks_review_proposals():
    observation = SourceIdentityObservation(
        source_id="ch12",
        channel_id=12,
        channel_label="Mystery Pod",
        predicted_source_type="unknown",
        source_type_confidence=0.32,
        match_type="unknown",
        classifier_source="channel_recognizer",
        signal_present=True,
        activity_confidence=0.72,
    )
    live_authority_context = {
        "status": "fresh",
        "field_count": 5,
        "fresh_field_count": 5,
        "read_only": False,
        "live_write": True,
        "queries_console": True,
    }

    result = analyze_source_identity_confidence(
        (observation,),
        current_state_cache_freshness=live_authority_context,
    )

    assert result.status == "analysis_only"
    assert result.recommendations == ()
    assert "source identity context cache has live authority" in " ".join(result.warnings)
    assert result.metadata["current_state_cache_freshness_status"] == "fresh"
    assert result.metadata["current_source_identity_context_cache_no_live_authority"] is False
    assert result.metadata["current_state_cache_freshness"]["live_write"] is True
    assert "source_identity_context_cache_live_authority_present" in result.metadata[
        "blockers"
    ]
    assert (
        build_source_identity_confidence_proposals(
            (observation,),
            current_state_cache_freshness=live_authority_context,
        )
        == ()
    )


def test_high_confidence_channel_recognizer_result_does_not_create_review_proposal():
    classification = classify_channel_name("LD VOX")
    observation = SourceIdentityObservation.from_channel_classification(
        "ch01",
        classification,
        channel_id=1,
    )

    result = analyze_source_identity_confidence(
        (observation,),
        current_state_cache_freshness=_freshness(),
    )

    assert result.status == "analysis_only"
    assert result.recommendations == ()
    assert result.metadata["review_candidate_count"] == 0


def test_source_identity_operator_proposals_and_adapter_are_not_queueable_live_actions():
    proposals = build_source_identity_confidence_proposals(
        (
            SourceIdentityObservation(
                source_id="ch09",
                channel_id=9,
                channel_label="Maybe Vocal",
                predicted_source_type="lead_vocal",
                source_type_confidence=0.48,
                match_type="heuristic_name",
            ),
        ),
        current_state_cache_freshness=_freshness(),
    )

    assert proposals
    proposal = proposals[0]
    assert proposal["method_id"] == "source_identity_confidence_proposal"
    assert proposal["kind"] == "source_identity"
    assert proposal["can_import_to_queue"] is False
    assert proposal["can_apply_live"] is False
    assert "requested_change" not in proposal
    assert "unsupported_value_type:source_identity" in proposal["blockers"]
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["current_source_identity_context_cache_no_live_authority"] is True
    assert proposal["evidence"]["operator_acceptance_required_before_promotion"] is True

    adapted = adapt_source_identity_confidence_proposals(proposals)
    assert adapted[0]["adapter_id"] == "source_identity_confidence"
    assert adapted[0]["can_import_to_queue"] is False
    assert adapted[0]["can_apply_live"] is False
    assert adapted[0]["queue_action"] == "preview_only_not_queued"
    assert "requested_change" not in adapted[0]
    assert adapted[0]["evidence"]["requested_change_allowed"] is False


def test_source_identity_registry_contract_and_goal_row():
    method = get_processing_method("source_identity_confidence_proposal")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}
    row = get_project_goal_capability("identify_source_types")

    assert method is not None
    assert method.status == "proposal_only"
    assert method.domain == "source_identity"
    assert method.can_analyze is True
    assert method.can_propose is True
    assert method.can_write_live is False
    assert "main_live_assist" in method.pipeline_profiles
    assert method.blocked_reason is not None
    assert "source_identity_review" in method.blocked_reason

    contract = contracts["source_identity_confidence_proposal"]["proposal_contract"]
    evidence = contract["safety_evidence_required"]

    assert contract["value_type"] == "source_identity"
    assert contract["target_kind"] == "source_identity_review"
    assert "source_type_confidence" in evidence
    assert "downstream_rejection_reasons" in evidence
    assert "current_source_identity_context_cache_no_live_authority" in evidence
    assert "operator_acceptance" in evidence
    assert contract["live_apply_allowed"] is False

    assert row is not None
    assert row.status == "canonical"
    assert "source_identity_confidence_proposal" in row.method_ids
    assert row.current_live_write_kinds == ()
    assert "operator review" in row.next_step


def test_source_identity_module_is_transport_free():
    source = Path("backend/source_identity_confidence_proposal.py").read_text()

    forbidden_fragments = [
        "WingClient",
        "DLiveClient",
        "MixingStation",
        "mixer_client",
        "pythonosc",
        "python-osc",
        "set_gain",
        "set_eq",
        ".send(",
    ]

    for fragment in forbidden_fragments:
        assert fragment not in source
