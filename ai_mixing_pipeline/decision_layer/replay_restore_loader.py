"""Typed replay restore loader for dry-run rewind artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .replay_checkpoint_manifest import ReplayCheckpointManifest
from .replay_proposal_ranking import ReplayGraphCheckpoint


@dataclass(frozen=True)
class ReplayTrimSnapshot:
    snapshot_id: str
    schema_version: str
    event_seq: int
    transport_policy: dict[str, Any]
    replay_metadata: dict[str, Any]
    replay_correlation_ids: list[str]
    payload: dict[str, Any]
    last_state: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReplayTrimSnapshot":
        snapshot_id = str(payload.get("snapshot_id") or f"replay_checkpoint::live_input_trim::{payload.get('event_seq', 0)}")
        return cls(
            snapshot_id=snapshot_id,
            schema_version=str(payload.get("version", payload.get("schema_version", "replay_state_snapshot/v1"))),
            event_seq=int(payload.get("event_seq", 0)),
            transport_policy=dict(payload.get("transport_policy") or {}),
            replay_metadata=dict(payload.get("replay_metadata") or {}),
            replay_correlation_ids=[str(item) for item in payload.get("replay_correlation_ids", [])],
            payload=dict(payload),
            last_state=dict(payload.get("last_state") or {}),
        )


@dataclass(frozen=True)
class ReplayRewindContext:
    manifest_id: str
    event_seq: int
    dry_run_only: bool
    live_mixer_writes: bool
    replay_correlation_ids: list[str]
    trim_snapshot: ReplayTrimSnapshot
    graph_checkpoint: ReplayGraphCheckpoint
    manifest: ReplayCheckpointManifest


def load_replay_rewind_context(
    manifest: ReplayCheckpointManifest | dict[str, Any],
) -> ReplayRewindContext:
    resolved = manifest if isinstance(manifest, ReplayCheckpointManifest) else ReplayCheckpointManifest.from_dict(manifest)
    if not resolved.dry_run_only or resolved.live_mixer_writes:
        raise ValueError("dry_run_only manifest required for replay restore loading")
    trim_snapshot = ReplayTrimSnapshot.from_dict(resolved.trim_snapshot)
    graph_checkpoint = ReplayGraphCheckpoint.from_dict(resolved.graph_checkpoint)
    event_seq = trim_snapshot.event_seq or int(graph_checkpoint.metadata.get("event_seq", 0))
    return ReplayRewindContext(
        manifest_id=resolved.manifest_id,
        event_seq=event_seq,
        dry_run_only=resolved.dry_run_only,
        live_mixer_writes=resolved.live_mixer_writes,
        replay_correlation_ids=list(resolved.replay_correlation_ids),
        trim_snapshot=trim_snapshot,
        graph_checkpoint=graph_checkpoint,
        manifest=resolved,
    )
