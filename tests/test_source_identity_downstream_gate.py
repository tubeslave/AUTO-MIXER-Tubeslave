from pathlib import Path

from backend.source_identity_downstream_gate import evaluate_source_identity_downstream_trust


def _freshness(status="fresh", **overrides):
    payload = {
        "status": status,
        "field_count": 4,
        "fresh_field_count": 4 if status == "fresh" else 2,
        "stale_field_count": 0 if status == "fresh" else 2,
        "stale_fields": [] if status == "fresh" else ["accepted_labels", "replay_outcomes"],
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    payload.update(overrides)
    return payload


def test_trusted_source_identity_is_usable_for_downstream_proposals():
    decision = evaluate_source_identity_downstream_trust(
        source_id="ch:1",
        source_type="lead_vocal",
        trust_state="trusted_for_downstream_priority",
        trust_confidence=0.91,
        replay_outcome_count=6,
        current_state_cache_freshness=_freshness(),
    )

    assert decision.status == "trusted"
    assert decision.usable_for_downstream is True
    assert decision.blockers == ()
    assert decision.evidence["trust_confidence"] == 0.91
    assert decision.evidence["current_source_identity_replay_context_cache_freshness"] == "fresh"
    assert decision.evidence["current_source_identity_replay_context_cache_no_live_authority"] is True
    assert decision.evidence["model_mutation"] is False
    assert decision.evidence["memory_mutation"] is False


def test_missing_calibration_is_legacy_allowed_but_marked_unverified():
    decision = evaluate_source_identity_downstream_trust(
        source_id="ch:2",
        source_type="guitar",
    )

    assert decision.status == "legacy_unverified"
    assert decision.usable_for_downstream is True
    assert decision.blockers == ()
    assert decision.evidence["legacy_missing_calibration_allowed"] is True


def test_trusted_source_identity_without_fresh_replay_context_is_not_usable_downstream():
    decision = evaluate_source_identity_downstream_trust(
        source_id="ch:2",
        source_type="guitar",
        trust_state="trusted_for_downstream_priority",
        trust_confidence=0.92,
        replay_outcome_count=6,
    )

    assert decision.status == "trusted"
    assert decision.usable_for_downstream is False
    assert "source_identity_replay_context_not_fresh" in decision.blockers
    assert decision.evidence["current_source_identity_replay_context_cache_freshness"] == "not_provided"
    assert decision.evidence["current_state_cache_freshness"]["provided"] is False
    assert decision.evidence["legacy_missing_calibration_allowed"] is False


def test_hold_or_blocked_source_identity_is_not_usable_downstream():
    held = evaluate_source_identity_downstream_trust(
        source_id="ch:3",
        source_type="keys",
        trust_state="hold_for_operator_review",
        trust_confidence=0.48,
        replay_outcome_count=4,
        current_state_cache_freshness=_freshness(),
    )
    blocked = evaluate_source_identity_downstream_trust(
        source_id="ch:4",
        source_type="bass",
        trust_state="blocked_from_downstream_trust",
        trust_confidence=0.8,
        replay_outcome_count=5,
        current_state_cache_freshness=_freshness(),
    )

    assert held.usable_for_downstream is False
    assert "source_identity_not_trusted_for_downstream" in held.blockers
    assert blocked.usable_for_downstream is False
    assert "source_identity_blocked_from_downstream_trust" in blocked.blockers


def test_stale_source_identity_replay_context_blocks_downstream_trust():
    decision = evaluate_source_identity_downstream_trust(
        source_id="ch:5",
        source_type="lead_vocal",
        trust_state="trusted_for_downstream_priority",
        trust_confidence=0.9,
        replay_outcome_count=5,
        current_state_cache_freshness=_freshness("stale"),
    )

    assert decision.usable_for_downstream is False
    assert "source_identity_replay_context_not_fresh" in decision.blockers
    assert decision.evidence["current_state_cache_freshness"]["stale_fields"] == [
        "accepted_labels",
        "replay_outcomes",
    ]


def test_live_authority_replay_context_blocks_downstream_trust():
    decision = evaluate_source_identity_downstream_trust(
        source_id="ch:6",
        source_type="lead_vocal",
        trust_state="trusted_for_downstream_priority",
        trust_confidence=0.91,
        replay_outcome_count=6,
        current_state_cache_freshness=_freshness(
            read_only=False,
            live_write=True,
            queries_console=True,
        ),
    )

    assert decision.status == "trusted"
    assert decision.usable_for_downstream is False
    assert "source_identity_replay_context_live_authority_present" in decision.blockers
    assert decision.evidence["current_source_identity_replay_context_cache_freshness"] == "fresh"
    assert decision.evidence["current_source_identity_replay_context_cache_no_live_authority"] is False
    assert decision.evidence["current_state_cache_freshness"]["live_write"] is True


def test_source_identity_downstream_gate_is_transport_free():
    source = Path("backend/source_identity_downstream_gate.py").read_text()

    forbidden_fragments = [
        "WingClient",
        "DLiveClient",
        "MixingStation",
        "mixer_client",
        "pythonosc",
        "python-osc",
        "set_gain",
        "set_eq",
        "apply_operator_proposal",
        ".send(",
    ]

    for fragment in forbidden_fragments:
        assert fragment not in source
