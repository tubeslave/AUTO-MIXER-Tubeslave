"""Replay-safe proposal building and ranking tests."""

from __future__ import annotations

import json
from pathlib import Path

from ai_mixing_pipeline.decision_layer.replay_proposal_ranking import (
    REPLAY_TRACE_SCHEMA_VERSION,
    ReplayDecisionProposal,
    ReplayGraphCheckpoint,
    build_replay_graph_checkpoint,
    build_replay_timeline_artifact,
    build_fader_proposals,
    build_feedback_mitigation_proposals,
    build_gain_proposals,
    build_operator_assistance_proposals,
    build_performer_stage_interaction_proposals,
    build_room_compensation_proposals,
    build_speech_priority_arbitration_proposals,
    compare_replay_graph_checkpoints,
    rank_replay_proposals,
    write_replay_timeline_artifact,
)
from ai_mixing_pipeline.decision_layer.replay_executor import simulate_replay_decisions


def _gain_rec(source: str, channel: int, current: float, target: float, confidence: float = 0.8):
    return {
        "source": source,
        "mixer_channel": channel,
        "current_trim_db": current,
        "recommended_target_trim_db": target,
        "confidence": confidence,
        "reason": "prototype_gain",
        "role": "lead_vocal",
        "replay_correlation_id": f"corr::{source}::{channel}",
    }


def _fader_rec(channel: int, current: float, target: float, confidence: float = 0.7):
    return {
        "source": "auto_fader_v2",
        "mixer_channel": channel,
        "current_fader_db": current,
        "target_fader_db": target,
        "confidence": confidence,
        "live_apply_safe": True,
        "reason": "prototype_balance",
    }


def _room_plan(target: str = "group", confidence: float = 0.75):
    return {
        "event_id": "r1",
        "room_classification": "reverberant",
        "recommended_target": target,
        "room_quality_indicator": "good",
        "room_quality_score": 0.73,
        "confidence_score": confidence,
        "dry_run_only": True,
        "blocked": False,
        "planned_filters": [
            {
                "frequency": 250.0,
                "gain_db": -2.0,
                "q": 1.8,
                "filter_type": "peak",
            }
        ],
    }


def _feedback_event(channel: int, action: str = "notch", confidence: float = 0.62):
    return {
        "event_id": "fb1",
        "channel": channel,
        "source": "feedback_detector",
        "action": action,
        "frequency_hz": 2500,
        "magnitude_db": -7.0,
        "confidence": confidence,
        "live_apply_allowed": False,
    }


def _speech_payload(leader: int = 3, follower: int = 4):
    return {
        "event_id": "sp1",
        "source": "speech_priority_engine",
        "leader_channel": leader,
        "suppressed_channel": follower,
        "leader_boost_db": 0.8,
        "suppressed_cut_db": 1.2,
        "dominant_speaker": "lead_vocal",
        "confidence": 0.9,
        "live_apply_safe": False,
    }


def _operator_payload():
    return {
        "rec_id": "op1",
        "source": "operator_coach",
        "type": "scene_reset",
        "target": "mix",
        "message": "Consider reloading the scene after loud feedback burst.",
        "confidence": 0.4,
        "urgency": "medium",
    }


def _performer_payload():
    return {
        "event_id": "pr1",
        "source": "stage_interaction_model",
        "performer_id": "performer_1",
        "mixer_channel": 2,
        "zone_from": "center",
        "zone_to": "stage_left",
        "orientation_deg": 45.0,
        "orientation_change_deg": 10.0,
        "distance_m": 2.2,
        "proximity_delta_m": 0.2,
        "movement_level_delta_db": 0.5,
        "movement_spill_delta_db": 0.2,
        "risk_profile": "medium",
        "confidence": 0.72,
        "replay_correlation_id": "corr::stage::performer::1",
    }


def test_replay_proposal_builders_emit_dry_run_and_traceable_payloads():
    gain = build_gain_proposals(recommendations=[_gain_rec("auto_soundcheck_engine", 1, 0.0, 1.2)])
    fader = build_fader_proposals(fader_updates=[_fader_rec(2, -4.0, -3.0)])
    speech = build_speech_priority_arbitration_proposals(arbitration_payloads=[_speech_payload()])
    room = build_room_compensation_proposals(room_plans=[_room_plan()])
    feedback = build_feedback_mitigation_proposals(feedback_events=[_feedback_event(5)])
    operator = build_operator_assistance_proposals(operator_recommendations=[_operator_payload()])
    performer = build_performer_stage_interaction_proposals(interaction_events=[_performer_payload()])

    all_proposals = [*gain, *fader, *speech, *room, *feedback, *operator, *performer]

    assert len(all_proposals) == 7
    assert all(isinstance(item, ReplayDecisionProposal) for item in all_proposals)
    assert all(item.dry_run_only for item in all_proposals)
    for item in all_proposals:
        assert item.replay_signature
        assert item.proposal_id
        assert item.family in {
            "gain_adjustment",
            "fader_adjustment",
            "speech_arbitration",
            "room_compensation",
            "feedback_mitigation",
            "operator_assistance",
            "performer_stage_interaction",
        }


def test_rank_replay_proposals_is_deterministic_and_confidence_aware():
    proposals = [
        *build_gain_proposals(
            recommendations=[
                _gain_rec("auto_soundcheck_engine", 1, 0.0, 2.0, confidence=0.15),
                _gain_rec("safe_gain_calibrator", 2, -5.0, -4.9, confidence=0.95),
            ]
        ),
        ReplayDecisionProposal(
            proposal_id="op::manual::1",
            family="operator_assistance",
            source_system="operator_advice",
            action_type="operator_prompt",
            target={"target": "mix"},
            current_state={},
            requested_state={"message": "check"},
            confidence=0.99,
            safety={"requires_operator": True, "requires_confirmation": True},
            auto_apply_blocked=True,
            critic_scores={
                "audiobox_aesthetics": {
                    "delta": {"overall": 0.2},
                    "scores": {"overall": 0.2},
                    "confidence": 0.9,
                }
            },
        ),
    ]

    proposals[0].critic_scores.update(
        {
            "audiobox_aesthetics": {"delta": {"overall": 0.1}, "scores": {"overall": 0.1}, "confidence": 0.2}
        }
    )
    proposals[1].critic_scores.update(
        {
            "audiobox_aesthetics": {"delta": {"overall": 0.15}, "scores": {"overall": 0.15}, "confidence": 0.95}
        }
    )

    first = rank_replay_proposals(
        proposals,
        config={"critics": {"audiobox_aesthetics": {"enabled": True, "weight": 1.0}}},
        family_priority={"operator_assistance": 0.0},
    )
    second = rank_replay_proposals(
        proposals,
        config={"critics": {"audiobox_aesthetics": {"enabled": True, "weight": 1.0}}},
        family_priority={"operator_assistance": 0.0},
    )

    assert first.selected_proposal_id is not None
    assert first.selected_proposal_id == second.selected_proposal_id
    assert first.run_signature == second.run_signature
    assert first.ranked_proposals[0]["proposal_id"].startswith("op::")
    assert first.ranked_proposals[0]["confidence_weighted_score"] > first.ranked_proposals[-1]["confidence_weighted_score"]


def test_rank_trace_contains_traceability_fields_for_replay():
    proposals = [
        *build_feedback_mitigation_proposals(feedback_events=[_feedback_event(1, action="notch", confidence=0.72)]),
        *build_room_compensation_proposals(room_plans=[_room_plan(target="master", confidence=0.55)]),
    ]
    proposals[1].critic_scores.update(
        {
            "mert": {"delta": {"overall": 0.5}, "scores": {"overall": 0.5}, "confidence": 0.7},
            "safety": {"safety_score": 0.9},
        }
    )

    result = rank_replay_proposals(proposals)
    proposal = result.ranked_proposals[0]

    assert isinstance(proposal, dict)
    assert proposal["dry_run_only"] is True
    assert proposal["replay_signature"]
    assert "score_breakdown" in proposal
    assert proposal["score_breakdown"]["critic_values"] in ({}, {"mert": 0.5, "safety": 0.9})
    assert result.ranking_trace[0]["proposal_id"]
    assert result.ranking_trace[0]["family"] in {"room_compensation", "feedback_mitigation"}
    assert result.ranking_trace[0]["replay_correlation_id"]


def test_replay_proposals_preserve_supplied_correlation_ids():
    proposals = build_gain_proposals(
        recommendations=[_gain_rec("auto_soundcheck_engine", 8, -1.0, 0.5)]
    )

    assert proposals[0].replay_correlation_id == "corr::auto_soundcheck_engine::8"
    ranked = rank_replay_proposals(proposals)
    assert ranked.ranked_proposals[0]["replay_correlation_id"] == "corr::auto_soundcheck_engine::8"
    assert ranked.ranking_trace[0]["replay_correlation_id"] == "corr::auto_soundcheck_engine::8"


def test_room_and_operator_payloads_are_blocking_transport_by_default():
    proposals = [
        *build_room_compensation_proposals(room_plans=[_room_plan(target="matrix")]),
        *build_operator_assistance_proposals(operator_recommendations=[_operator_payload()]),
    ]

    ranked = rank_replay_proposals(proposals)
    assert all(item["dry_run_only"] for item in ranked.ranked_proposals)
    assert all(item["auto_apply_blocked"] for item in ranked.ranked_proposals)
    assert all(item["family"] in {"room_compensation", "operator_assistance"} for item in ranked.ranked_proposals)


def test_replay_timeline_artifact_is_deterministic_and_replay_safe():
    fixture_path = Path(__file__).parent / "fixtures" / "replay_timeline_artifact_v1.json"
    proposals = [
        *build_gain_proposals(
            recommendations=[
                {
                    **_gain_rec("auto_soundcheck_engine", 12, -3.0, -1.5, confidence=0.82),
                    "metadata": {
                        "max_step_db": 1.0,
                        "path": "/tmp/unsafe.wav",
                        "host": "192.168.0.77",
                    },
                }
            ]
        ),
        *build_feedback_mitigation_proposals(
            feedback_events=[
                {
                    **_feedback_event(5, action="notch", confidence=0.62),
                    "operator_override_required": True,
                }
            ]
        ),
    ]
    proposals[0].critic_scores.update(
        {
            "clarity": {
                "scores": {"overall": 0.61},
                "delta": {"overall": 0.14},
                "confidence": 0.91,
                "metadata": {
                    "path": "/tmp/critic-before.wav",
                    "detail": "safe",
                },
            }
        }
    )
    proposals[1].critic_scores.update(
        {
            "safety": {
                "safety_score": 0.88,
                "warnings": ["operator_override_required"],
            }
        }
    )
    ranking = rank_replay_proposals(
        list(reversed(proposals)),
        config={"critics": {"clarity": {"enabled": True, "weight": 1.0}}},
    )

    first = build_replay_timeline_artifact(
        proposals,
        ranking,
        metadata={"scenario": "deterministic_trace_v1", "path": "/tmp/leak.json"},
    )
    second = build_replay_timeline_artifact(
        list(reversed(proposals)),
        ranking,
        metadata={"path": "/tmp/leak.json", "scenario": "deterministic_trace_v1"},
    )

    expected = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert first == second
    assert first == expected
    assert first["schema_version"] == REPLAY_TRACE_SCHEMA_VERSION
    assert first["dry_run_only"] is True
    assert first["live_mixer_writes"] is False
    assert first["metadata"] == {"scenario": "deterministic_trace_v1"}
    assert all("path" not in json.dumps(event, sort_keys=True) for event in first["timeline"])
    assert all("host" not in json.dumps(event, sort_keys=True) for event in first["timeline"])
    assert [event["seq"] for event in first["timeline"]] == list(range(1, len(first["timeline"]) + 1))
    assert [event["stage"] for event in first["timeline"][:4]] == ["analyzer", "critic", "analyzer", "critic"]
    assert first["timeline"][2]["replay_correlation_id"] == "corr::auto_soundcheck_engine::12"
    assert first["timeline"][5]["replay_correlation_id"] == "corr::auto_soundcheck_engine::12"


def test_write_replay_timeline_artifact_emits_sorted_json(tmp_path: Path):
    proposals = build_room_compensation_proposals(room_plans=[_room_plan(target="master", confidence=0.55)])
    ranking = rank_replay_proposals(proposals)

    target = write_replay_timeline_artifact(
        tmp_path / "replay_timeline.json",
        proposals,
        ranking,
        metadata={"scenario": "write_path"},
    )

    payload = json.loads(target.read_text(encoding="utf-8"))

    assert target.exists()
    assert payload["schema_version"] == REPLAY_TRACE_SCHEMA_VERSION
    assert payload["selected_proposal_id"] == ranking.selected_proposal_id
    assert payload["timeline"][-1]["stage"] == "decision"


def test_replay_graph_checkpoint_round_trip_is_deterministic():
    proposals = [
        *build_gain_proposals(
            recommendations=[_gain_rec("auto_soundcheck_engine", 1, 0.0, 1.2, confidence=0.8)]
        ),
        *build_feedback_mitigation_proposals(
            feedback_events=[_feedback_event(5, action="notch", confidence=0.62)]
        ),
    ]
    proposals[0].critic_scores.update(
        {"clarity": {"delta": {"overall": 0.25}, "scores": {"overall": 0.25}, "confidence": 0.9}}
    )
    ranking = rank_replay_proposals(proposals)

    first = build_replay_graph_checkpoint(
        proposals,
        ranking,
        metadata={"scenario": "checkpoint_round_trip"},
    )
    second = build_replay_graph_checkpoint(
        list(reversed(proposals)),
        ranking,
        metadata={"scenario": "checkpoint_round_trip"},
    )
    restored = ReplayGraphCheckpoint.from_dict(first.to_dict())

    assert first.checkpoint_id == second.checkpoint_id
    assert restored.checkpoint_id == first.checkpoint_id
    assert [proposal.proposal_id for proposal in restored.proposals] == sorted(
        proposal.proposal_id for proposal in proposals
    )
    replayed = rank_replay_proposals(restored.proposals)
    assert replayed.run_signature == ranking.run_signature
    assert replayed.selected_proposal_id == ranking.selected_proposal_id


def test_replay_graph_checkpoint_branch_compare_detects_decision_change():
    proposals = [
        *build_gain_proposals(
            recommendations=[
                _gain_rec("auto_soundcheck_engine", 1, 0.0, 2.0, confidence=0.85),
            ]
        ),
        ReplayDecisionProposal(
            proposal_id="operator::manual::branch",
            family="operator_assistance",
            source_system="operator_advice",
            action_type="operator_prompt",
            target={"target": "mix"},
            current_state={},
            requested_state={"message": "hold automation"},
            confidence=0.95,
            safety={"requires_operator": True, "requires_confirmation": True},
            auto_apply_blocked=True,
        ),
    ]
    proposals[0].critic_scores.update(
        {"clarity": {"delta": {"overall": 0.4}, "scores": {"overall": 0.4}, "confidence": 0.85}}
    )
    proposals[1].critic_scores.update(
        {"clarity": {"delta": {"overall": 0.3}, "scores": {"overall": 0.3}, "confidence": 0.95}}
    )

    baseline_ranking = rank_replay_proposals(
        proposals,
        config={"critics": {"clarity": {"enabled": True, "weight": 1.0}}},
        family_priority={"operator_assistance": 0.0},
    )
    branch_ranking = rank_replay_proposals(
        proposals,
        config={"critics": {"clarity": {"enabled": True, "weight": 1.0}}},
        family_priority={"operator_assistance": 0.5},
    )

    baseline_checkpoint = build_replay_graph_checkpoint(
        proposals,
        baseline_ranking,
        metadata={"branch": "baseline"},
    )
    branch_checkpoint = build_replay_graph_checkpoint(
        proposals,
        branch_ranking,
        metadata={"branch": "operator_priority"},
    )
    comparison = compare_replay_graph_checkpoints(
        baseline_checkpoint,
        branch_checkpoint,
    )

    assert comparison["same_selected_proposal"] is False
    assert comparison["selected_proposal_changed"] is True
    assert comparison["baseline_selected_proposal_id"] != comparison["candidate_selected_proposal_id"]
    assert comparison["proposal_ids_added"] == []
    assert comparison["proposal_ids_removed"] == []


def test_performer_stage_interaction_proposals_are_replay_simulatable():
    proposals = build_performer_stage_interaction_proposals(
        interaction_events=[_performer_payload()]
    )
    proposals[0].critic_scores.update(
        {
            "safety": {
                "safety_score": 0.84,
                "safety_flags": [],
            }
        }
    )
    ranking = rank_replay_proposals(
        proposals,
        config={"critics": {"safety": {"enabled": True, "weight": 1.0}}},
    )
    executor = simulate_replay_decisions(
        proposals,
        selected_proposal_id=ranking.selected_proposal_id,
    )

    assert executor.dry_run_only is True
    assert executor.selected_proposal_id == proposals[0].proposal_id
    assert len(executor.events) == 1
    assert executor.events[0].family == "performer_stage_interaction"
    assert executor.events[0].status == "applied"
    assert executor.events[0].dry_run is True
    assert executor.events[0].details["performer_id"] == "performer_1"
