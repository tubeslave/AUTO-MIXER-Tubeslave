from __future__ import annotations

from ai_mixing_pipeline.decision_layer.replay_proposal_ranking import (
    build_fader_proposals,
    build_gain_proposals,
    build_speech_priority_arbitration_proposals,
    build_replay_timeline_artifact,
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


def test_replay_timeline_artifact_preserves_source_payload_for_inventory_restore():
    gain = build_gain_proposals(
        recommendations=[
            {
                "source": "auto_soundcheck_engine",
                "mixer_channel": 7,
                "current_trim_db": -1.0,
                "recommended_target_trim_db": 1.2,
                "confidence": 0.85,
                "replay_correlation_id": "corr::gain::inventory",
                "critic_scores": {"clarity": {"scores": {"overall": 0.4}, "confidence": 0.85}},
                "arrangement_id": "intro_a",
                "live_apply": True,
            }
        ]
    )[0]
    ranking = rank_replay_proposals([gain])

    artifact = build_replay_timeline_artifact([gain], ranking, metadata={"scenario": "inventory_restore"})

    assert artifact["events"][0]["source_payload"]["arrangement_id"] == "intro_a"
    assert artifact["events"][0]["source_payload"]["recommended_target_trim_db"] == 1.2
    assert artifact["events"][0]["source_payload"]["live_apply"] is True


def test_replay_timeline_artifact_preserves_auto_fader_inventory_metadata():
    fader = build_fader_proposals(
        fader_updates=[
            {
                "source": "auto_fader_v2",
                "mixer_channel": 11,
                "current_fader_db": -6.0,
                "recommended_target_fader_db": -3.5,
                "confidence": 0.92,
                "replay_correlation_id": "corr::fader::inventory",
                "critic_scores": {"clarity": {"scores": {"overall": 0.6}, "confidence": 0.92}},
                "arrangement_id": "chorus_b",
                "live_apply": True,
                "readback_source": "console_state",
                "requires_confirmation": True,
            }
        ]
    )[0]
    ranking = rank_replay_proposals([fader])

    artifact = build_replay_timeline_artifact([fader], ranking, metadata={"scenario": "auto_fader_inventory_restore"})

    assert artifact["events"][0]["source_payload"]["arrangement_id"] == "chorus_b"
    assert artifact["events"][0]["source_payload"]["recommended_target_fader_db"] == -3.5
    assert artifact["events"][0]["source_payload"]["live_apply"] is True
    assert artifact["events"][0]["source_payload"]["readback_source"] == "console_state"
    assert artifact["events"][0]["source_payload"]["requires_confirmation"] is True


def test_speech_priority_proposals_preserve_environment_metadata_in_timeline():
    speech = build_speech_priority_arbitration_proposals(
        arbitration_payloads=[
            {
                "source": "speech_priority_engine",
                "target_bus": "main",
                "current_state": {
                    "duck_music_db": -2.0,
                    "speech_weight": 0.6,
                    "music_weight": 0.8,
                },
                "requested_state": {
                    "duck_music_db": -6.0,
                    "speech_weight": 1.0,
                    "music_weight": 0.35,
                    "environment": "speech",
                },
                "environment_label": "speech_focus",
                "priority_source": "podium_mic",
                "confidence": 0.97,
                "replay_correlation_id": "corr::speech::timeline",
            }
        ]
    )[0]
    ranking = rank_replay_proposals([speech])

    artifact = build_replay_timeline_artifact([speech], ranking, metadata={"scenario": "speech_environment"})

    assert artifact["events"][0]["source_payload"]["environment_label"] == "speech_focus"
    assert artifact["events"][0]["source_payload"]["priority_source"] == "podium_mic"
    assert artifact["events"][0]["requested_state"]["environment"] == "speech"
    assert artifact["events"][0]["target"]["bus"] == "main"
