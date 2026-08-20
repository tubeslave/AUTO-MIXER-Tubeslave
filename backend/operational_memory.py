"""Centralized append-only operational memory for Automixer runtime evidence."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
import uuid
from typing import Any, Iterable, Mapping


REDACTED = "[REDACTED]"
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bot_token",
    "cookie",
    "credential",
    "passwd",
    "password",
    "private_key",
    "secret",
    "token",
)
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[^\s,;]+")
_URL_CREDENTIAL_RE = re.compile(r"(?i)(https?://)[^/@\s:]+:[^/@\s]+@")
_KNOWN_TOKEN_RE = re.compile(
    r"(?i)(?:sk-[a-z0-9_-]{12,}|pplx-[a-z0-9_-]{12,}|\b\d{6,12}:[a-z0-9_-]{20,})"
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime | None = None) -> str:
    moment = value or _utc_now()
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _redact_text(value: str) -> str:
    if "-----BEGIN " in value and "PRIVATE KEY-----" in value:
        return REDACTED
    value = _BEARER_RE.sub(r"\1" + REDACTED, value)
    value = _URL_CREDENTIAL_RE.sub(r"\1" + REDACTED + "@", value)
    return _KNOWN_TOKEN_RE.sub(REDACTED, value)


def _is_sensitive_key(key: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if _is_sensitive_key(key) else _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return _redact_text(value)
    return value


class OperationalMemoryStore:
    """Persist operational evidence into one GPT-readable append-only store."""

    def __init__(self, base_dir: str | Path = "artifacts/operational_memory", max_recent: int = 25):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.base_dir / "events.jsonl"
        self.snapshot_path = self.base_dir / "latest_snapshot.json"
        self.summary_path = self.base_dir / "latest_summary.md"
        self.knowledge_dir = self.base_dir / "knowledge_snapshots"
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.max_recent = max(1, int(max_recent))
        self._lock = threading.Lock()

    def record_event(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        source: str | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        event = {
            "event_id": uuid.uuid4().hex,
            "event_type": _redact_text(str(event_type)),
            "recorded_at": _isoformat(),
            "source": _redact_text(str(source or "unknown")),
            "summary": _redact_text(str(summary or "").strip()),
            "payload": _json_safe(dict(payload or {})),
        }
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")
            self._rewrite_views_locked()
        return event

    def record_session_event(
        self,
        *,
        action: str,
        session_id: str,
        session_name: str,
        config: Mapping[str, Any] | None = None,
        active: bool | None = None,
        event_count: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "session_id": session_id,
            "session_name": session_name,
            "config": _json_safe(dict(config or {})),
            "active": active,
            "event_count": event_count,
            "extra": _json_safe(dict(extra or {})),
        }
        return self.record_event(
            "session_event",
            payload,
            source="session_manager",
            summary=f"Session {session_name} {action}",
        )

    def record_write_intent(self, record: Mapping[str, Any]) -> dict[str, Any]:
        payload = _json_safe(dict(record))
        channel = payload.get("mixer_channel")
        write_kind = payload.get("write_kind") or "unknown"
        target = f"ch {channel}" if channel is not None else str(payload.get("target_bus") or "global")
        return self.record_event(
            "write_intent",
            payload,
            source=str(payload.get("source_system") or "replay_write_intent_adapter"),
            summary=f"Dry-run {write_kind} intent captured for {target}",
        )

    def record_telemetry_session(
        self,
        *,
        phase: str,
        status: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "phase": str(phase),
            "status": _json_safe(dict(status)),
            "metadata": _json_safe(dict(metadata or {})),
        }
        return self.record_event(
            "telemetry_session",
            payload,
            source="wing_telemetry_recorder",
            summary=f"WING telemetry session {phase}",
        )

    def record_task_outcome(
        self,
        *,
        issue_identifier: str,
        title: str,
        status: str,
        summary: str,
        evidence_paths: Iterable[str | Path] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "issue_identifier": issue_identifier,
            "title": title,
            "status": status,
            "summary": summary,
            "evidence_paths": [str(path) for path in (evidence_paths or [])],
            "context": _json_safe(dict(context or {})),
        }
        return self.record_event(
            "task_outcome",
            payload,
            source="paperclip",
            summary=f"{issue_identifier} {status}: {title}",
        )

    def record_context_sync(
        self,
        *,
        agent_id: str,
        issue_identifier: str,
        summary: str,
        related_event_ids: Iterable[str] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "agent_id": agent_id,
            "issue_identifier": issue_identifier,
            "summary": summary,
            "related_event_ids": list(related_event_ids or []),
            "context": _json_safe(dict(context or {})),
        }
        return self.record_event(
            "context_sync",
            payload,
            source="agent_handoff",
            summary=f"Context sync for {issue_identifier}",
        )

    def write_knowledge_snapshot(
        self,
        *,
        snapshot_name: str,
        summary: str,
        content: Mapping[str, Any],
        tags: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in snapshot_name).strip("_")
        filename = f"{_utc_now().strftime('%Y%m%dT%H%M%SZ')}_{slug or 'snapshot'}.json"
        path = self.knowledge_dir / filename
        payload = {
            "schema_version": 1,
            "snapshot_name": snapshot_name,
            "summary": summary,
            "tags": list(tags or []),
            "generated_at": _isoformat(),
            "content": _json_safe(dict(content)),
        }
        with self._lock:
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")
        event = self.record_event(
            "knowledge_snapshot",
            {"snapshot_path": str(path), **payload},
            source="knowledge_snapshot",
            summary=summary or f"Knowledge snapshot {snapshot_name}",
        )
        return {"snapshot_path": str(path), "event": event}

    def read_snapshot(self) -> dict[str, Any]:
        if not self.snapshot_path.exists():
            return {
                "schema_version": 1,
                "generated_at": None,
                "event_count": 0,
                "counts_by_type": {},
                "recent_events": [],
            }
        return json.loads(self.snapshot_path.read_text(encoding="utf-8"))

    def _rewrite_views_locked(self) -> None:
        events = self._load_events_locked()
        snapshot = self._build_snapshot(events)
        self.snapshot_path.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        self.summary_path.write_text(self._build_markdown_summary(snapshot), encoding="utf-8")

    def _load_events_locked(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self.events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                events.append(json.loads(text))
        return events

    def _build_snapshot(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        counts_by_type: dict[str, int] = {}
        latest_by_type: dict[str, dict[str, Any]] = {}
        for event in events:
            event_type = str(event.get("event_type") or "unknown")
            counts_by_type[event_type] = counts_by_type.get(event_type, 0) + 1
            latest_by_type[event_type] = event

        def _recent(event_type: str) -> list[dict[str, Any]]:
            return [event for event in events if event.get("event_type") == event_type][-self.max_recent:]

        return {
            "schema_version": 1,
            "generated_at": _isoformat(),
            "event_count": len(events),
            "counts_by_type": dict(sorted(counts_by_type.items())),
            "recent_events": events[-self.max_recent:],
            "latest_by_type": latest_by_type,
            "task_outcomes": _recent("task_outcome"),
            "write_intents": _recent("write_intent"),
            "session_events": _recent("session_event"),
            "telemetry_sessions": _recent("telemetry_session"),
            "knowledge_snapshots": _recent("knowledge_snapshot"),
            "context_sync_events": _recent("context_sync"),
        }

    def _build_markdown_summary(self, snapshot: Mapping[str, Any]) -> str:
        counts = snapshot.get("counts_by_type") or {}
        recent_events = snapshot.get("recent_events") or []
        lines = [
            "# Operational Memory Summary",
            "",
            f"- Generated at: {snapshot.get('generated_at')}",
            f"- Total events: {snapshot.get('event_count', 0)}",
            "",
            "## Counts by type",
        ]
        if counts:
            for event_type, count in counts.items():
                lines.append(f"- {event_type}: {count}")
        else:
            lines.append("- none")

        lines.extend(["", "## Recent events"])
        if recent_events:
            for event in recent_events[-10:]:
                lines.append(
                    f"- {event.get('recorded_at')}: {event.get('event_type')} - "
                    f"{event.get('summary') or 'no summary'}"
                )
        else:
            lines.append("- none")
        return "\n".join(lines) + "\n"
