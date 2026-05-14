"""Replay checkpoint manifest helpers for end-to-end deterministic rewind."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item) for item in value]
    return value


def _collect_graph_correlation_ids(graph_checkpoint: dict[str, Any]) -> list[str]:
    correlation_ids: set[str] = set()
    for proposal in graph_checkpoint.get("proposals") or ():
        value = str((proposal or {}).get("replay_correlation_id", "")).strip()
        if value:
            correlation_ids.add(value)
    timeline = (graph_checkpoint.get("timeline_artifact") or {}).get("timeline") or ()
    for event in timeline:
        value = str((event or {}).get("replay_correlation_id", "")).strip()
        if value:
            correlation_ids.add(value)
    return sorted(correlation_ids)


def _collect_trim_correlation_ids(trim_snapshot: dict[str, Any]) -> list[str]:
    correlation_ids: set[str] = set()
    states = trim_snapshot.get("states") or {}
    for state in states.values():
        value = str((state or {}).get("replay_correlation_id", "")).strip()
        if value:
            correlation_ids.add(value)
    last_state = trim_snapshot.get("last_state") or {}
    for item in last_state.get("applied") or ():
        value = str((item or {}).get("replay_correlation_id", "")).strip()
        if value:
            correlation_ids.add(value)
    return sorted(correlation_ids)


def _normalize_critic_artifacts(critic_artifacts: dict[str, Any]) -> dict[str, Any]:
    normalized = _sanitize(critic_artifacts)
    correlation_ids = normalized.get("replay_correlation_ids")
    if isinstance(correlation_ids, list):
        normalized["replay_correlation_ids"] = sorted(
            {str(item).strip() for item in correlation_ids if str(item).strip()}
        )
    return normalized


@dataclass(frozen=True)
class ReplayCheckpointManifest:
    version: str
    manifest_id: str
    event_seq: int | None
    replay_correlation_ids: list[str]
    dry_run_only: bool
    live_mixer_writes: bool
    trim_snapshot: dict[str, Any]
    graph_checkpoint: dict[str, Any]
    critic_artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "manifest_id": self.manifest_id,
            "event_seq": self.event_seq,
            "replay_correlation_ids": list(self.replay_correlation_ids),
            "dry_run_only": bool(self.dry_run_only),
            "live_mixer_writes": bool(self.live_mixer_writes),
            "trim_snapshot": _sanitize(self.trim_snapshot),
            "graph_checkpoint": _sanitize(self.graph_checkpoint),
            "critic_artifacts": _sanitize(self.critic_artifacts),
            "metadata": _sanitize(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReplayCheckpointManifest":
        event_seq = payload.get("event_seq")
        return cls(
            version=str(payload.get("version", "replay_checkpoint_manifest/v1")),
            manifest_id=str(payload.get("manifest_id", "")),
            event_seq=int(event_seq) if event_seq is not None else None,
            replay_correlation_ids=[
                str(item) for item in payload.get("replay_correlation_ids") or ()
            ],
            dry_run_only=bool(payload.get("dry_run_only", True)),
            live_mixer_writes=bool(payload.get("live_mixer_writes", False)),
            trim_snapshot=dict(payload.get("trim_snapshot") or {}),
            graph_checkpoint=dict(payload.get("graph_checkpoint") or {}),
            critic_artifacts=dict(payload.get("critic_artifacts") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )


def build_replay_checkpoint_manifest(
    *,
    trim_snapshot: dict[str, Any],
    graph_checkpoint: dict[str, Any],
    critic_artifacts: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ReplayCheckpointManifest:
    critic_payload = _normalize_critic_artifacts(critic_artifacts or {})
    trim_payload = _sanitize(trim_snapshot)
    graph_payload = _sanitize(graph_checkpoint)
    event_seq = (
        trim_payload.get("replay_metadata", {}).get("event_seq")
        if trim_payload.get("replay_metadata", {}).get("event_seq") is not None
        else graph_payload.get("replay_metadata", {}).get("event_seq")
    )
    correlation_ids = sorted(
        set(_collect_trim_correlation_ids(trim_payload))
        | set(_collect_graph_correlation_ids(graph_payload))
        | set(str(item) for item in critic_payload.get("replay_correlation_ids") or () if str(item).strip())
    )
    dry_run_only = bool(
        trim_payload.get("transport_policy", {}).get("dry_run_only", True)
        and graph_payload.get("timeline_artifact", {}).get("dry_run_only", True)
    )
    live_mixer_writes = bool(
        graph_payload.get("timeline_artifact", {}).get("live_mixer_writes", False)
    )
    core_payload = {
        "version": "replay_checkpoint_manifest/v1",
        "event_seq": int(event_seq) if event_seq is not None else None,
        "replay_correlation_ids": correlation_ids,
        "dry_run_only": dry_run_only,
        "live_mixer_writes": live_mixer_writes,
        "trim_snapshot": trim_payload,
        "graph_checkpoint": graph_payload,
        "critic_artifacts": critic_payload,
        "metadata": _sanitize(metadata or {}),
    }
    digest = hashlib.sha256(
        json.dumps(
            core_payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return ReplayCheckpointManifest(
        version="replay_checkpoint_manifest/v1",
        manifest_id=f"replay_manifest::{digest[:16]}",
        event_seq=int(event_seq) if event_seq is not None else None,
        replay_correlation_ids=correlation_ids,
        dry_run_only=dry_run_only,
        live_mixer_writes=live_mixer_writes,
        trim_snapshot=trim_payload,
        graph_checkpoint=graph_payload,
        critic_artifacts=critic_payload,
        metadata=_sanitize(metadata or {}),
    )
