from __future__ import annotations

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


def _build_manifest(*, dry_run_only: bool = True, live_mixer_writes: bool = False):
    trim_snapshot = {
        "version": "replay_state_snapshot/v1",
        "snapshot_id": "replay_checkpoint::live_input_trim::21",
        "event_seq": 21,
        "transport_policy": {
            "dry_run_only": dry_run_only,
            "live_mixer_writes": live_mixer_writes,
        },
        "replay_metadata": {"stage": "restore_root"},
        "replay_correlation_ids": ["corr::restore::1"],
        "last_state": {
            "applied": [
                {
                    "channel": 1,
                    "replay_correlation_id": "corr::restore::1",
                }
            ]
        },
    }
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
    checkpoint = build_replay_graph_checkpoint(proposals, ranking, metadata={"event_seq": 21}).to_dict()
    checkpoint["timeline_artifact"]["dry_run_only"] = dry_run_only
    checkpoint["timeline_artifact"]["live_mixer_writes"] = live_mixer_writes
    return build_replay_checkpoint_manifest(
        trim_snapshot=trim_snapshot,
        graph_checkpoint=checkpoint,
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
    try:
        load_replay_rewind_context(manifest)
    except ValueError as exc:
        assert "dry_run_only manifest" in str(exc)
    else:
        raise AssertionError("Expected live-write replay manifest to be rejected")
