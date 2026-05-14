"""Replay executor simulation tests."""

from __future__ import annotations

from ai_mixing_pipeline.decision_layer.replay_executor import build_replay_executor_from_rewind_context, simulate_replay_decisions
from ai_mixing_pipeline.decision_layer.replay_proposal_ranking import ReplayDecisionProposal
from ai_mixing_pipeline.decision_layer.replay_proposal_ranking import (
    build_fader_proposals,
    build_feedback_mitigation_proposals,
    build_gain_proposals,
    build_room_compensation_proposals,
    build_replay_graph_checkpoint,
    build_speech_priority_arbitration_proposals,
    rank_replay_proposals,
)
from ai_mixing_pipeline.decision_layer.replay_checkpoint_manifest import build_replay_checkpoint_manifest
from ai_mixing_pipeline.decision_layer.replay_restore_loader import load_replay_rewind_context
from arrangement_automation.live_input_trim_controller import LiveInputTrimChannelState, LiveInputTrimController


def _speech_payload(leader: int = 5, follower: int = 6):
    return {
        "event_id": "sp1",
        "source": "speech_priority_engine",
        "leader_channel": leader,
        "suppressed_channel": follower,
        "leader_boost_db": 0.8,
        "suppressed_cut_db": 1.2,
        "dominant_speaker": "lead_vocal",
        "confidence": 0.9,
    }


def _build_eq_and_compression_proposals():
    eq_proposal = ReplayDecisionProposal(
        proposal_id="eq::ch1::v1",
        family="eq_adjustment",
        source_system="manual_eq_tester",
        action_type="set_eq_band",
        target={"channel": 1, "band_id": "band_2"},
        current_state={"gain_db": 0.0},
        requested_state={"freq_hz": 1600.0, "gain_db": 1.0, "q": 1.0, "filter_type": "peaking"},
        confidence=0.7,
        safety={"max_gain_db": 4.0},
        auto_apply_blocked=True,
        metadata={"reason": "test_eq"},
        analyst_observation={"sweep": "mid"},
    )

    compression_proposal = ReplayDecisionProposal(
        proposal_id="comp::ch1::v1",
        family="compression",
        source_system="manual_compressor_tester",
        action_type="set_compressor",
        target={"channel": 1},
        current_state={"ratio": 1.5},
        requested_state={"threshold_db": -18.0, "ratio": 2.0, "attack_ms": 12.0, "release_ms": 120.0, "makeup_gain_db": 1.0},
        confidence=0.65,
        safety={"max_ratio": 4.0},
        auto_apply_blocked=True,
        metadata={"reason": "test_compression"},
        analyst_observation={"zone": "mid"},
    )
    return eq_proposal, compression_proposal


def _live_trim_config():
    return {
        "automation": {
            "live_input_trim": {
                "run_background_loop": False,
                "analysis_only_mode": False,
                "live_apply_enabled": True,
                "confirm_live_apply": True,
                "activity_attack_hold_sec": 0.0,
                "min_main_signal_confidence": 0.1,
                "analysis_window_sec": 0.5,
                "apply_cooldown_sec": 0.0,
                "max_step_db_per_tick": 0.25,
                "max_boost_step_db": 0.15,
                "deadband_db": 0.1,
            }
        }
    }


def test_replay_executor_simulates_all_supported_action_families_deterministically():
    gain = build_gain_proposals(
        recommendations=[{"source": "auto_soundcheck_engine", "mixer_channel": 1, "current_trim_db": 0.0, "recommended_target_trim_db": 1.0, "confidence": 0.9}]
    )[0]
    fader = build_fader_proposals(
        fader_updates=[{"mixer_channel": 2, "current_fader_db": -3.0, "target_fader_db": -2.0}]
    )[0]
    room = build_room_compensation_proposals(
        room_plans=[{"event_id": "r1", "room_classification": "flat", "recommended_target": "master", "room_quality_indicator": "fair", "room_quality_score": 0.6, "planned_filters": [{"frequency": 250.0, "gain_db": -1.2, "q": 1.8, "filter_type": "peak"}], "confidence_score": 0.55}]
    )[0]
    speech = build_speech_priority_arbitration_proposals(arbitration_payloads=[_speech_payload()])[0]
    feedback = build_feedback_mitigation_proposals(
        feedback_events=[{"event_id": "fb1", "source": "feedback_detector", "channel": 3, "action": "notch", "frequency_hz": 2000.0, "magnitude_db": -7.0, "confidence": 0.64}]
    )[0]
    eq, compressor = _build_eq_and_compression_proposals()
    proposals = [gain, fader, eq, compressor, room, speech, feedback]

    first = simulate_replay_decisions(proposals)
    second = simulate_replay_decisions(proposals)
    assert first.dry_run_only is True
    assert first.trace_signature == second.trace_signature
    assert [event.seq for event in first.events] == list(range(1, len(first.events) + 1))
    assert first.to_dict()["schema_version"] == "replay_executor/v1"
    assert len(first.events) == len(proposals)
    assert first.state_snapshots and first.events[0].applied
    assert first.events[0].status == "applied"
    assert any(event.family == "eq_adjustment" for event in first.events)
    assert any(event.family == "compression" for event in first.events)
    assert any(event.family == "speech_arbitration" for event in first.events)


def test_replay_executor_allows_rollback_to_intermediate_state():
    gain = build_gain_proposals(
        recommendations=[{"source": "auto_soundcheck_engine", "mixer_channel": 1, "current_trim_db": 0.0, "recommended_target_trim_db": 1.0, "confidence": 0.9}]
    )[0]
    fader = build_fader_proposals(
        fader_updates=[{"mixer_channel": 2, "current_fader_db": -3.0, "target_fader_db": -2.0}]
    )[0]
    result = simulate_replay_decisions([gain, fader], rollback_to=1)

    assert result.restored_state is not None
    assert result.restored_state["gains"] == {"1": 1.0}
    assert result.rollback_state == result.restored_state


def test_replay_executor_from_rewind_context_initializes_existing_state():
    trim_controller = LiveInputTrimController(config=_live_trim_config())
    trim_controller.channels = [1]
    trim_controller.channel_mapping = {1: 1}
    trim_controller.states = {
        1: LiveInputTrimChannelState(
            audio_channel=1,
            mixer_channel=1,
            channel_name="Lead Vox",
            role="lead_vocal",
            role_confidence=1.0,
            current_trim_db=2.5,
        ),
    }
    trim_snapshot = trim_controller.build_replay_snapshot(
        event_seq=3,
        replay_metadata={"scenario": "replay_executor"},
    ).to_dict()
    trim_snapshot["transport_policy"]["dry_run_only"] = True

    gain = build_gain_proposals(
        recommendations=[{"source": "auto_soundcheck_engine", "mixer_channel": 1, "current_trim_db": 2.5, "recommended_target_trim_db": 2.5, "confidence": 0.7}]
    )[0]
    ranking = rank_replay_proposals([gain])
    graph_checkpoint = build_replay_graph_checkpoint([gain], ranking, metadata={"scenario": "replay_executor"}).to_dict()
    manifest = build_replay_checkpoint_manifest(trim_snapshot=trim_snapshot, graph_checkpoint=graph_checkpoint)
    context = load_replay_rewind_context(manifest)

    initial_state = build_replay_executor_from_rewind_context(context.to_dict())
    result = simulate_replay_decisions([gain], initial_state=initial_state)

    assert result.events[0].status == "applied"
    assert result.events[0].post_state["gains"]["1"] == 2.5
