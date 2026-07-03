from __future__ import annotations

import pytest

from ai_mixing_pipeline.decision_layer.replay_checkpoint_manifest import (
    build_replay_checkpoint_manifest,
)
from ai_mixing_pipeline.decision_layer.replay_proposal_ranking import (
    build_gain_proposals,
    build_replay_graph_checkpoint,
    rank_replay_proposals,
)
from ai_mixing_pipeline.decision_layer.replay_restore_loader import (
    load_replay_rewind_context,
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


def _build_manifest(*, dry_run_only: bool = True, live_mixer_writes: bool = False):
    controller = LiveInputTrimController(config=_trim_config())
    controller.channels = [1]
    controller.channel_mapping = {1: 1}
    controller.last_state = {
        "applied": [
            {
                "channel": 1,
                "replay_correlation_id": "corr::restore::1",
            }
        ]
    }
    trim_snapshot = controller.build_replay_snapshot(
        event_seq=21,
        replay_metadata={"stage": "restore_root"},
    ).to_dict()
    trim_snapshot["transport_policy"]["dry_run_only"] = dry_run_only

    proposals = build_gain_proposals(
        recommendations=[
            {
                "source": "auto_soundcheck_engine",
                "mixer_channel": 1,
                "current_trim_db": 0.0,
                "recommended_target_trim_db": 0.5,
                "confidence": 0.8,
                "reason": "restore_loader_test",
                "role": "lead_vocal",
                "replay_correlation_id": "corr::restore::1",
            }
        ]
    )
    ranking = rank_replay_proposals(proposals)
    graph_checkpoint = build_replay_graph_checkpoint(
        proposals,
        ranking,
        metadata={"event_seq": 21},
    ).to_dict()
    graph_checkpoint["timeline_artifact"]["dry_run_only"] = dry_run_only
    graph_checkpoint["timeline_artifact"]["live_mixer_writes"] = live_mixer_writes

    return build_replay_checkpoint_manifest(
        trim_snapshot=trim_snapshot,
        graph_checkpoint=graph_checkpoint,
        critic_artifacts={"replay_correlation_ids": ["corr::restore::1"]},
        metadata={"scenario": "restore_loader"},
    )


def test_replay_restore_loader_returns_typed_dry_run_context():
    manifest = _build_manifest()

    context = load_replay_rewind_context(manifest)

    assert context.manifest_id == manifest.manifest_id
    assert context.event_seq == 21
    assert context.dry_run_only is True
    assert context.live_mixer_writes is False
    assert context.replay_correlation_ids == ["corr::restore::1"]
    assert context.trim_snapshot.snapshot_id.startswith("replay_checkpoint::live_input_trim::")
    assert context.graph_checkpoint.checkpoint_id.startswith("replay_graph::")


def test_replay_restore_loader_rejects_live_write_manifests():
    manifest = _build_manifest(dry_run_only=False, live_mixer_writes=True)

    with pytest.raises(ValueError, match="dry_run_only manifest"):
        load_replay_rewind_context(manifest)
