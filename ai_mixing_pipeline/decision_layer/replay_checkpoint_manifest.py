"""Replay checkpoint manifest helpers for end-to-end deterministic rewind."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _sanitize(payload: Any) -> Any:
    return json.loads(_stable_json(payload))


def _collect_graph_correlation_ids(graph_checkpoint: dict[str, Any]) -> list[str]:
    values: set[str] = set()
    for proposal in graph_checkpoint.get("proposals", []):
        correlation_id = proposal.get("replay_correlation_id")
        if correlation_id:
            values.add(str(correlation_id))
    for event in (graph_checkpoint.get("timeline_artifact") or {}).get("events", []):
        correlation_id = event.get("replay_correlation_id")
        if correlation_id:
            values.add(str(correlation_id))
    return sorted(values)


def _collect_trim_correlation_ids(trim_snapshot: dict[str, Any]) -> list[str]:
    values: set[str] = set()
    for item in trim_snapshot.get("replay_correlation_ids", []):
        if item:
            values.add(str(item))
    last_state = trim_snapshot.get("last_state") or {}
    for records in last_state.values():
        if isinstance(records, list):
            for record in records:
                correlation_id = (record or {}).get("replay_correlation_id")
                if correlation_id:
                    values.add(str(correlation_id))
    return sorted(values)


def _normalize_critic_artifacts(payload: dict[str, Any] | None) -> dict[str, Any]:
    artifacts = dict(payload or {})
    correlation_ids = sorted({str(item) for item in artifacts.get("replay_correlation_ids", []) if item})
    if correlation_ids:
        artifacts["replay_correlation_ids"] = correlation_ids
    return _sanitize(artifacts)


@dataclass(frozen=True)
class ReplayCheckpointManifest:
    manifest_id: str
    schema_version: str
    trim_snapshot: dict[str, Any]
    graph_checkpoint: dict[str, Any]
    critic_artifacts: dict[str, Any]
    metadata: dict[str, Any]
    replay_correlation_ids: list[str]
    dry_run_only: bool
    live_mixer_writes: bool
    manifest_signature: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReplayCheckpointManifest":
        return cls(
            manifest_id=str(payload["manifest_id"]),
            schema_version=str(payload.get("schema_version", "replay_checkpoint_manifest/v1")),
            trim_snapshot=dict(payload.get("trim_snapshot") or {}),
            graph_checkpoint=dict(payload.get("graph_checkpoint") or {}),
            critic_artifacts=dict(payload.get("critic_artifacts") or {}),
            metadata=dict(payload.get("metadata") or {}),
            replay_correlation_ids=[str(item) for item in payload.get("replay_correlation_ids", [])],
            dry_run_only=bool(payload.get("dry_run_only", True)),
            live_mixer_writes=bool(payload.get("live_mixer_writes", False)),
            manifest_signature=str(payload.get("manifest_signature") or ""),
        )


def build_replay_checkpoint_manifest(
    *,
    trim_snapshot: dict[str, Any],
    graph_checkpoint: dict[str, Any],
    critic_artifacts: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ReplayCheckpointManifest:
    safe_trim_snapshot = _sanitize(trim_snapshot)
    safe_graph_checkpoint = _sanitize(graph_checkpoint)
    safe_critic_artifacts = _normalize_critic_artifacts(critic_artifacts)
    safe_metadata = _sanitize(metadata or {})
    replay_correlation_ids = sorted(
        {
            *(_collect_graph_correlation_ids(safe_graph_checkpoint)),
            *(_collect_trim_correlation_ids(safe_trim_snapshot)),
            *(str(item) for item in safe_critic_artifacts.get("replay_correlation_ids", []) if item),
        }
    )
    trim_policy = safe_trim_snapshot.get("transport_policy") or {}
    graph_policy = (safe_graph_checkpoint.get("timeline_artifact") or {})
    live_mixer_writes = bool(trim_policy.get("live_mixer_writes")) or bool(graph_policy.get("live_mixer_writes"))
    dry_run_only = bool(trim_policy.get("dry_run_only", True)) and bool(graph_policy.get("dry_run_only", True))
    manifest_payload = {
        "schema_version": "replay_checkpoint_manifest/v1",
        "trim_snapshot": safe_trim_snapshot,
        "graph_checkpoint": safe_graph_checkpoint,
        "critic_artifacts": safe_critic_artifacts,
        "metadata": safe_metadata,
        "replay_correlation_ids": replay_correlation_ids,
        "dry_run_only": dry_run_only,
        "live_mixer_writes": live_mixer_writes,
    }
    manifest_signature = _stable_hash(manifest_payload)
    return ReplayCheckpointManifest(
        manifest_id=f"replay_manifest::{manifest_signature[:16]}",
        schema_version="replay_checkpoint_manifest/v1",
        trim_snapshot=safe_trim_snapshot,
        graph_checkpoint=safe_graph_checkpoint,
        critic_artifacts=safe_critic_artifacts,
        metadata=safe_metadata,
        replay_correlation_ids=replay_correlation_ids,
        dry_run_only=dry_run_only,
        live_mixer_writes=live_mixer_writes,
        manifest_signature=manifest_signature,
    )
