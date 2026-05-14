from __future__ import annotations

from ai_mixing_pipeline.decision_layer.replay_checkpoint_manifest import (
    build_replay_checkpoint_manifest,
)
from ai_mixing_pipeline.decision_layer.replay_executor import (
    simulate_replay_decisions,
)
from ai_mixing_pipeline.decision_layer.replay_policy_validator import (
    validate_replay_policy_consistency,
)
from ai_mixing_pipeline.decision_layer.replay_proposal_ranking import (
    ReplayDecisionProposal,
    build_gain_proposals,
    build_replay_graph_checkpoint,
    rank_replay_proposals,
)
from arrangement_automation.live_input_trim_controller import (
    LiveInputTrimController,
)


def _trim_config():
    return {
        "automation": {
            "live_input_trim": {
                "run_background_loop": False,
                "analysis_only_mode": True,
                "live_apply_enabled": False,
                "confirm_live_apply": False,
            }
        }
    }


def _build_manifest(*, graph_event_seq: int = 41):
    controller = LiveInputTrimController(config=_trim_config())
    controller.channels = [1]
    controller.channel_mapping = {1: 1}
    controller.last_state = {
        "applied": [
            {
                "channel": 1,
                "replay_correlation_id": "corr::containment::1",
                "target_trim_db": 0.5,
            }
        ]
    }
    trim_snapshot = controller.build_replay_snapshot(
        event_seq=41,
        replay_metadata={"stage": "containment_prereqs"},
    ).to_dict()

    proposal = build_gain_proposals(
        recommendations=[
            {
                "source": "auto_soundcheck_engine",
                "mixer_channel": 1,
                "current_trim_db": 0.0,
                "recommended_target_trim_db": 0.5,
                "confidence": 0.82,
                "reason": "containment_prereq",
                "role": "lead_vocal",
                "replay_correlation_id": "corr::containment::1",
            }
        ]
    )[0]
    ranking = rank_replay_proposals([proposal])
    graph_checkpoint = build_replay_graph_checkpoint(
        [proposal],
        ranking,
        metadata={"event_seq": graph_event_seq, "stage": "containment_prereqs"},
    ).to_dict()
    return (
        build_replay_checkpoint_manifest(
            trim_snapshot=trim_snapshot,
            graph_checkpoint=graph_checkpoint,
            critic_artifacts={"replay_correlation_ids": ["corr::containment::1"]},
            metadata={"scenario": "replay_failure_containment_prereqs"},
        ),
        proposal,
        ranking,
    )


def test_replay_policy_validator_flags_event_seq_corruption_and_missing_block_reason():
    manifest, proposal, ranking = _build_manifest(graph_event_seq=999)
    executor = simulate_replay_decisions(
        [proposal],
        selected_proposal_id=ranking.selected_proposal_id,
    ).to_dict()
    executor["events"][0]["status"] = "blocked"
    executor["events"][0]["blocked_reason"] = ""

    report = validate_replay_policy_consistency(
        manifest,
        executor_result=executor,
    )

    codes = {item.code for item in report.findings}
    assert report.valid is False
    assert "replay_event_seq_mismatch" in codes
    assert "executor_blocked_without_reason" in codes


def test_replay_executor_fail_closes_non_dry_run_auto_apply_blocked_proposals():
    proposal = ReplayDecisionProposal(
        proposal_id="gain::containment::1",
        family="gain_adjustment",
        source_system="containment_test",
        action_type="set_gain",
        target={"channel": 1},
        current_state={"gain_db": 0.0},
        requested_state={"gain_db": 1.0},
        confidence=0.9,
        safety={"max_step_db": 2.0},
        dry_run_only=False,
        auto_apply_blocked=True,
        replay_correlation_id="corr::containment::blocked",
        metadata={"reason": "fail_closed_check"},
    )

    result = simulate_replay_decisions([proposal])

    assert result.dry_run_only is True
    assert result.proposals_executed == []
    assert len(result.events) == 1
    assert result.events[0].status == "blocked"
    assert result.events[0].applied is False
    assert result.events[0].dry_run is False
    assert result.events[0].blocked_reason == "auto_apply_blocked_without_dry_run"
