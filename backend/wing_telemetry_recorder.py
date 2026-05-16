"""
Persistent JSONL recorder for WING observation and replay telemetry sessions.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Convert values into JSON-serializable primitives."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class WingTelemetryRecorder:
    """Record a WING telemetry session as a durable JSONL dataset."""

    def __init__(
        self,
        base_dir: str = "artifacts/wing_telemetry",
        session_label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        clock: Optional[callable] = None,
    ):
        self._base_dir = Path(base_dir)
        self._session_label = session_label or "wing_session"
        self._metadata = dict(metadata or {})
        self._clock = clock or time.time
        self._lock = threading.Lock()

        session_stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(self._clock()))
        slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in self._session_label).strip("_")
        slug = slug or "wing_session"

        self.session_id = f"{session_stamp}_{slug}_{uuid.uuid4().hex[:8]}"
        self.session_dir = self._base_dir / self.session_id
        self.events_path = self.session_dir / "events.jsonl"
        self.metadata_path = self.session_dir / "metadata.json"

        self._active = False
        self._event_count = 0
        self._seq = 0
        self._started_at: Optional[float] = None
        self._stopped_at: Optional[float] = None

    def start(self) -> Dict[str, Any]:
        """Create the dataset directory and metadata file."""
        with self._lock:
            if self._active:
                return self.get_status()

            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._started_at = self._clock()
            self._stopped_at = None
            self._active = True
            self._write_metadata_locked()

        logger.info("Started WING telemetry capture: %s", self.session_dir)
        return self.get_status()

    def stop(self, reason: str = "") -> Dict[str, Any]:
        """Flush final metadata and close the dataset."""
        with self._lock:
            if not self._active:
                return self.get_status()

            self._stopped_at = self._clock()
            self._active = False
            if reason:
                self._metadata["stop_reason"] = reason
            self._write_metadata_locked()

        logger.info("Stopped WING telemetry capture: %s", self.session_dir)
        return self.get_status()

    def record_event(
        self,
        event_type: str,
        *,
        address: Optional[str] = None,
        values: Any = None,
        source: Optional[str] = None,
        **extra: Any,
    ) -> bool:
        """Append one event to the JSONL stream."""
        with self._lock:
            if not self._active:
                return False

            self._seq += 1
            event = {
                "seq": self._seq,
                "timestamp": self._clock(),
                "event_type": event_type,
            }
            if address is not None:
                event["address"] = address
            if values is not None:
                event["values"] = _json_safe(values)
            if source is not None:
                event["source"] = source
            if extra:
                event.update(_json_safe(extra))

            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=True) + "\n")

            self._event_count += 1
            return True

    @property
    def active(self) -> bool:
        return self._active

    def get_status(self) -> Dict[str, Any]:
        """Return current capture status and artifact paths."""
        return {
            "active": self._active,
            "session_id": self.session_id,
            "session_label": self._session_label,
            "session_dir": str(self.session_dir),
            "events_path": str(self.events_path),
            "metadata_path": str(self.metadata_path),
            "started_at": self._started_at,
            "stopped_at": self._stopped_at,
            "event_count": self._event_count,
        }

    def _write_metadata_locked(self):
        payload = {
            "schema_version": 1,
            "session_id": self.session_id,
            "session_label": self._session_label,
            "started_at": self._started_at,
            "stopped_at": self._stopped_at,
            "active": self._active,
            "event_count": self._event_count,
            "artifacts": {
                "events_path": str(self.events_path),
                "metadata_path": str(self.metadata_path),
            },
            "metadata": _json_safe(self._metadata),
        }
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
