"""Append-only JSON/JSONL writer for learning telemetry sessions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

try:
    from .schemas import EVENT_FILES, TelemetryManifest, utc_now
except ImportError:  # pragma: no cover - direct PYTHONPATH=backend compatibility
    from telemetry.schemas import EVENT_FILES, TelemetryManifest, utc_now


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def write_json_atomic(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


class TelemetryEventWriter:
    """Writes telemetry event rows and keeps per-file counters."""

    def __init__(self, session_dir: str | Path, *, session_id: str):
        self.session_dir = Path(session_dir)
        self.session_id = str(session_id)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.counters: Dict[str, int] = {event_type: 0 for event_type in EVENT_FILES}
        self.warnings: list[str] = []

    def event_path(self, event_type: str) -> Path:
        if event_type not in EVENT_FILES:
            raise ValueError(f"Unknown telemetry event type: {event_type}")
        return self.session_dir / EVENT_FILES[event_type]

    def event_path_map(self) -> Dict[str, str]:
        return dict(EVENT_FILES)

    def append_jsonl(self, event_type: str, payload: Any) -> Dict[str, Any]:
        row = _json_safe(payload)
        if not isinstance(row, dict):
            row = {"value": row}
        row.setdefault("session_id", self.session_id)
        path = self.event_path(event_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        self.counters[event_type] = self.counters.get(event_type, 0) + 1
        return row

    def write_manifest(self, manifest: TelemetryManifest) -> Dict[str, Any]:
        manifest.files = {
            "session": "session.json",
            "scene_metadata": "scene_metadata.json",
            "manifest": "manifest.json",
            **{event_type: filename for event_type, filename in EVENT_FILES.items()},
        }
        manifest.counters = dict(self.counters)
        manifest.warnings = list(dict.fromkeys([*manifest.warnings, *self.warnings]))
        payload = manifest.to_dict()
        payload["updated_at"] = utc_now()
        write_json_atomic(self.session_dir / "manifest.json", payload)
        return payload

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.flush()
