"""JSONL logging for Automixer -> Mixing Station correction attempts."""

from __future__ import annotations

import json
import threading
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config import resolve_repo_path
from .models import AutomixCorrection, CorrectionResult, parse_timestamp


class MixingStationJSONLLogger:
    """Small synchronous JSONL logger for correction results."""

    def __init__(self, path: str | Path):
        self.path = resolve_repo_path(path)
        self._lock = threading.Lock()

    def log_result(self, result: CorrectionResult) -> None:
        """Append a correction result in the requested JSONL shape."""
        correction = result.correction
        timestamp = correction.timestamp.astimezone(timezone.utc).isoformat()
        row: Dict[str, Any] = {
            "timestamp": timestamp,
            "correction_id": correction.id,
            "console_profile": correction.console_profile,
            "mode": correction.mode,
            "channel_index": correction.channel_index,
            "channel_name": correction.channel_name,
            "strip_type": correction.strip_type,
            "parameter": correction.parameter,
            "requested_value": result.requested_value,
            "sent_value": result.sent_value,
            "unit": correction.value_unit,
            "transport": result.transport,
            "data_path": result.data_path,
            "safety_status": result.safety_status,
            "dry_run": result.dry_run,
            "success": result.success,
            "sent": result.sent,
            "blocked": result.blocked,
            "error": result.error,
            "message": result.message,
            "reason": correction.reason,
            "confidence": correction.confidence,
            "source_metrics": correction.source_metrics,
        }
        self.log_row(row)

    def log_row(self, row: Dict[str, Any]) -> None:
        """Append an arbitrary JSON row."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def correction_from_log_row(row: Dict[str, Any], *, dry_run: Optional[bool] = None) -> AutomixCorrection:
    """Recreate an AutomixCorrection from a JSONL log row."""
    return AutomixCorrection(
        id=str(row.get("correction_id") or row.get("id") or ""),
        timestamp=parse_timestamp(row.get("timestamp")),
        console_profile=str(row.get("console_profile", "wing_rack")),
        mode=str(row.get("mode", "offline_visualization")),
        channel_index=int(row.get("channel_index", 0)),
        channel_name=row.get("channel_name"),
        strip_type=str(row.get("strip_type", "input")),
        parameter=str(row.get("parameter", "fader")),
        value=row.get("requested_value", row.get("value")),
        value_unit=str(row.get("unit", row.get("value_unit", "db"))),
        previous_value=row.get("previous_value"),
        reason=str(row.get("reason", "replay")),
        confidence=float(row.get("confidence", 1.0)),
        source_metrics=dict(row.get("source_metrics") or {}),
        safety_status=str(row.get("safety_status", "allowed")),
        dry_run=bool(row.get("dry_run", True) if dry_run is None else dry_run),
    )
