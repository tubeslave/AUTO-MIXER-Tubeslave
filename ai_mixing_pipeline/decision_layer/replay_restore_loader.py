"""Replay restore loader for deterministic dry-run rewind contexts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from arrangement_automation.live_input_trim_controller import LiveInputTrimReplaySnapshot
except ImportError:  # pragma: no cover - package import from repo root.
    from backend.arrangement_automation.live_input_trim_controller import LiveInputTrimReplaySnapshot

from .replay_checkpoint_manifest import ReplayCheckpointManifest
from .replay_proposal_ranking import ReplayGraphCheckpoint


@dataclass(frozen=True)
class ReplayRewindContext:
    manifest_id: str
    event_seq: int | None
    replay_correlation_ids: list[str]
    dry_run_only: bool
    live_mixer_writes: bool
    trim_snapshot: LiveInputTrimReplaySnapshot
    graph_checkpoint: ReplayGraphCheckpoint
    critic_artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "event_seq": self.event_seq,
            "replay_correlation_ids": list(self.replay_correlation_ids),
            "dry_run_only": bool(self.dry_run_only),
            "live_mixer_writes": bool(self.live_mixer_writes),
            "trim_snapshot": self.trim_snapshot.to_dict(),
            "graph_checkpoint": self.graph_checkpoint.to_dict(),
            "critic_artifacts": dict(self.critic_artifacts),
            "metadata": dict(self.metadata),
        }


def load_replay_rewind_context(
    manifest: ReplayCheckpointManifest | dict[str, Any],
) -> ReplayRewindContext:
    resolved_manifest = (
        manifest
        if isinstance(manifest, ReplayCheckpointManifest)
        else ReplayCheckpointManifest.from_dict(dict(manifest))
    )
    if not resolved_manifest.dry_run_only:
        raise ValueError("Replay rewind context requires dry_run_only manifest")
    if resolved_manifest.live_mixer_writes:
        raise ValueError("Replay rewind context rejects manifests with live mixer writes")

    trim_snapshot = LiveInputTrimReplaySnapshot.from_dict(
        dict(resolved_manifest.trim_snapshot)
    )
    graph_checkpoint = ReplayGraphCheckpoint.from_dict(
        dict(resolved_manifest.graph_checkpoint)
    )
    return ReplayRewindContext(
        manifest_id=resolved_manifest.manifest_id,
        event_seq=resolved_manifest.event_seq,
        replay_correlation_ids=list(resolved_manifest.replay_correlation_ids),
        dry_run_only=resolved_manifest.dry_run_only,
        live_mixer_writes=resolved_manifest.live_mixer_writes,
        trim_snapshot=trim_snapshot,
        graph_checkpoint=graph_checkpoint,
        critic_artifacts=dict(resolved_manifest.critic_artifacts),
        metadata=dict(resolved_manifest.metadata),
    )
