from __future__ import annotations

from ai_mixing_pipeline.decision_layer.replay_proposal_ranking import (
    build_gain_proposals,
    build_operator_assistance_proposals,
    build_replay_graph_checkpoint,
    compare_replay_graph_checkpoints,
    rank_replay_proposals,
)


def test_rank_replay_proposals_is_deterministic_and_confidence_aware():
    proposals = []
    gain = build_gain_proposals(
        recommendations=[
            {
                "source": "auto_soundcheck_engine",
                "mixer_channel": 1,
                "current_trim_db": 0.0,
                "recommended_target_trim_db": 1.2,
                "confidence": 0.85,
                "replay_correlation_id": "corr::gain::rank",
                "critic_scores": {
                    "clarity": {
                        "scores": {"overall": 0.4},
                        "confidence": 0.85,
                    }
                },
            }
        ]
    )[0]
    operator = build_operator_assistance_proposals(
        operator_recommendations=[
            {
                "source": "operator_console",
                "type": "manual_prompt",
                "rec_id": "operator::rank",
                "message": "hold automation",
                "confidence": 0.95,
                "replay_correlation_id": "corr::operator::rank",
                "critic_scores": {
                    "clarity": {
                        "scores": {"overall": 0.3},
                        "confidence": 0.95,
                    }
                },
            }
        ]
    )[0]
    proposals.extend([gain, operator])

    baseline = rank_replay_proposals(
        proposals,
        config={"critics": {"clarity": {"enabled": True, "weight": 1.0}}},
        family_priority={"operator_assistance": 0.0},
    )
    branch = rank_replay_proposals(
        proposals,
        config={"critics": {"clarity": {"enabled": True, "weight": 1.0}}},
        family_priority={"operator_assistance": 0.5},
    )

    assert baseline.selected_proposal_id == gain.proposal_id
    assert branch.selected_proposal_id == operator.proposal_id


def test_replay_graph_checkpoint_branch_compare_detects_decision_change():
    gain = build_gain_proposals(
        recommendations=[
            {
                "source": "auto_soundcheck_engine",
                "mixer_channel": 1,
                "current_trim_db": 0.0,
                "recommended_target_trim_db": 1.2,
                "confidence": 0.85,
                "replay_correlation_id": "corr::gain::branch",
                "critic_scores": {"clarity": {"scores": {"overall": 0.4}, "confidence": 0.85}},
            }
        ]
    )[0]
    operator = build_operator_assistance_proposals(
        operator_recommendations=[
            {
                "source": "operator_console",
                "type": "manual_prompt",
                "rec_id": "operator::branch",
                "message": "hold automation",
                "confidence": 0.95,
                "replay_correlation_id": "corr::operator::branch",
                "critic_scores": {"clarity": {"scores": {"overall": 0.3}, "confidence": 0.95}},
            }
        ]
    )[0]
    proposals = [gain, operator]
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
    baseline_checkpoint = build_replay_graph_checkpoint(proposals, baseline_ranking, metadata={"branch": "baseline"})
    branch_checkpoint = build_replay_graph_checkpoint(proposals, branch_ranking, metadata={"branch": "operator_priority"})

    comparison = compare_replay_graph_checkpoints(baseline_checkpoint, branch_checkpoint)

    assert comparison["same_selected_proposal"] is False
    assert comparison["selected_proposal_changed"] is True
    assert comparison["baseline_selected_proposal_id"] != comparison["candidate_selected_proposal_id"]
    assert comparison["proposal_ids_added"] == []
    assert comparison["proposal_ids_removed"] == []
