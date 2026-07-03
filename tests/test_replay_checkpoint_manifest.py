from __future__ import annotations

from ai_mixing_pipeline.decision_layer.replay_checkpoint_manifest import (
    ReplayCheckpointManifest,
    build_replay_checkpoint_manifest,
)
from ai_mixing_pipeline.decision_layer.replay_proposal_ranking import (
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


def test_replay_checkpoint_manifest_links_trim_graph_and_critic_artifacts():
    controller = LiveInputTrimController(config=_trim_config())
    controller.channels = [1]
    controller.channel_mapping = {1: 1}
    controller.last_state = {
        "applied": [
            {
                "channel": 1,
                "replay_correlation_id": "corr::shared::1",
            }
        ]
    }
    trim_snapshot = controller.build_replay_snapshot(
        event_seq=14,
        replay_metadata={"stage": "pre_graph"},
    ).to_dict()

    proposals = build_gain_proposals(
        recommendations=[
            {
                "source": "auto_soundcheck_engine",
                "mixer_channel": 1,
                "current_trim_db": 0.0,
                "recommended_target_trim_db": 1.2,
                "confidence": 0.85,
                "reason": "manifest_link_test",
                "role": "lead_vocal",
                "replay_correlation_id": "corr::shared::1",
            }
        ]
    )
    ranking = rank_replay_proposals(proposals)
    graph_checkpoint = build_replay_graph_checkpoint(
        proposals,
        ranking,
        metadata={"event_seq": 14, "stage": "graph"},
    ).to_dict()

    manifest = build_replay_checkpoint_manifest(
        trim_snapshot=trim_snapshot,
        graph_checkpoint=graph_checkpoint,
        critic_artifacts={
            "artifact_paths": {"stable_snapshot": "/tmp/replay_stable_snapshot.json"},
            "replay_correlation_ids": ["corr::shared::1"],
        },
        metadata={"scenario": "manifest_root"},
    )

    assert manifest.version == "replay_checkpoint_manifest/v1"
    assert manifest.event_seq == 14
    assert manifest.dry_run_only is False
    assert manifest.live_mixer_writes is False
    assert manifest.replay_correlation_ids == ["corr::shared::1"]
    assert manifest.critic_artifacts["artifact_paths"]["stable_snapshot"].endswith(
        "replay_stable_snapshot.json"
    )


def test_replay_checkpoint_manifest_round_trip_is_deterministic():
    trim_snapshot = {
        "version": "replay_state_snapshot/v1",
        "transport_policy": {"dry_run_only": True},
        "replay_metadata": {"event_seq": 7},
        "states": {},
        "last_state": {},
    }
    graph_checkpoint = {
        "version": "replay_graph_checkpoint/v1",
        "replay_metadata": {"event_seq": 7},
        "proposals": [
            {"proposal_id": "gain::a", "replay_correlation_id": "corr::a"},
            {"proposal_id": "gain::b", "replay_correlation_id": "corr::b"},
        ],
        "timeline_artifact": {
            "dry_run_only": True,
            "live_mixer_writes": False,
            "timeline": [
                {"replay_correlation_id": "corr::b"},
                {"replay_correlation_id": "corr::a"},
            ],
        },
    }

    first = build_replay_checkpoint_manifest(
        trim_snapshot=trim_snapshot,
        graph_checkpoint=graph_checkpoint,
        critic_artifacts={"replay_correlation_ids": ["corr::b", "corr::a"]},
        metadata={"scenario": "stable_manifest", "branch": "baseline"},
    )
    second = build_replay_checkpoint_manifest(
        trim_snapshot=trim_snapshot,
        graph_checkpoint=graph_checkpoint,
        critic_artifacts={"replay_correlation_ids": ["corr::a", "corr::b"]},
        metadata={"branch": "baseline", "scenario": "stable_manifest"},
    )
    restored = ReplayCheckpointManifest.from_dict(first.to_dict())

    assert first.manifest_id == second.manifest_id
    assert restored.manifest_id == first.manifest_id
    assert restored.replay_correlation_ids == ["corr::a", "corr::b"]
