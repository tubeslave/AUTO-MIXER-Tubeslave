"""Read-only streaming alert extraction for Automixer telemetry.

This module consumes already-recorded telemetry rows and emits local artifacts
for dashboards and training labels. It intentionally has no mixer transport and
does not import OSC/WING clients.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional

try:
    from .learning_export import export_training_timeline
    from .schemas import SCHEMA_VERSION, utc_now
    from .session_manager import DEFAULT_STORAGE_DIR
except ImportError:  # pragma: no cover - direct PYTHONPATH=backend compatibility
    from telemetry.learning_export import export_training_timeline
    from telemetry.schemas import SCHEMA_VERSION, utc_now
    from telemetry.session_manager import DEFAULT_STORAGE_DIR


ALERT_SCHEMA_VERSION = "0.1"
MANUAL_OVERRIDE_ACTIONS = {
    "fader_move",
    "trim_change",
    "mute",
    "unmute",
    "eq_change",
    "comp_change",
    "gate_change",
}


@dataclass(frozen=True)
class StreamingAlertConfig:
    lufs_drift_lu: float = 2.0
    lufs_window_ms: int = 5000
    fader_jump_db: float = 6.0
    fader_window_ms: int = 200
    manual_override_limit: int = 3
    manual_override_window_ms: int = 30000
    clipping_peak_dbfs: float = -0.1
    headroom_peak_dbfs: float = -3.0


@dataclass
class StreamingAlert:
    timestamp_ms: int
    session_id: str
    rule: str
    severity: str
    target_type: str
    target_id: str
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    window_ms: Optional[int] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _trim_window(rows: Deque[tuple[int, float]], now_ms: int, window_ms: int) -> None:
    cutoff = now_ms - window_ms
    while rows and rows[0][0] < cutoff:
        rows.popleft()


def _target_for_event(event_type: str, payload: Mapping[str, Any]) -> tuple[str, str]:
    if event_type == "bus_telemetry":
        return "bus", str(payload.get("bus_name") or "master")
    if event_type == "engineer_actions":
        return str(payload.get("target_type") or "unknown"), str(payload.get("target_id") or "unknown")
    return "channel", str(payload.get("channel_id") or payload.get("channel_name") or "unknown")


class StreamingAlertEngine:
    """Detect bounded telemetry anomalies from local read-only event streams."""

    def __init__(self, config: Optional[StreamingAlertConfig] = None):
        self.config = config or StreamingAlertConfig()
        self._lufs_history: Dict[str, Deque[tuple[int, float]]] = defaultdict(deque)
        self._fader_history: Dict[str, Deque[tuple[int, float]]] = defaultdict(deque)
        self._manual_override_history: Dict[str, Deque[int]] = defaultdict(deque)
        self.alerts: List[StreamingAlert] = []
        self.dashboard_rows: List[Dict[str, Any]] = []

    def process(self, events: Iterable[Mapping[str, Any]]) -> List[StreamingAlert]:
        for event in sorted(events, key=lambda row: (_safe_int(row.get("timestamp_ms")), str(row.get("event_type") or ""))):
            self.process_event(event)
        return self.alerts

    def process_event(self, event: Mapping[str, Any]) -> List[StreamingAlert]:
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else event
        if not isinstance(payload, Mapping):
            return []

        timestamp_ms = _safe_int(payload.get("timestamp_ms") or event.get("timestamp_ms"))
        session_id = str(payload.get("session_id") or event.get("session_id") or "")
        target_type, target_id = _target_for_event(event_type, payload)
        before_count = len(self.alerts)

        if event_type in {"bus_telemetry", "channel_telemetry"}:
            self._check_lufs_drift(timestamp_ms, session_id, target_type, target_id, payload)
            self._check_peak_headroom(timestamp_ms, session_id, target_type, target_id, payload)
        if event_type == "channel_telemetry":
            self._check_fader_jump(timestamp_ms, session_id, target_id, payload)
        if event_type == "engineer_actions":
            self._check_manual_override_burst(timestamp_ms, session_id, target_type, target_id, payload)

        new_alerts = self.alerts[before_count:]
        self.dashboard_rows.append(self._dashboard_row(event_type, payload, target_type, target_id, new_alerts))
        return new_alerts

    def _append_alert(self, alert: StreamingAlert) -> None:
        self.alerts.append(alert)

    def _check_lufs_drift(
        self,
        timestamp_ms: int,
        session_id: str,
        target_type: str,
        target_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        lufs = _safe_float(payload.get("lufs_short"))
        if lufs is None:
            return
        key = f"{target_type}:{target_id}"
        rows = self._lufs_history[key]
        _trim_window(rows, timestamp_ms, self.config.lufs_window_ms)
        if rows:
            baseline_ts, baseline = rows[0]
            drift = lufs - baseline
            if abs(drift) >= self.config.lufs_drift_lu:
                self._append_alert(
                    StreamingAlert(
                        timestamp_ms=timestamp_ms,
                        session_id=session_id,
                        rule="lufs_drift",
                        severity="warning",
                        target_type=target_type,
                        target_id=target_id,
                        message=f"Short loudness changed by {drift:.2f} LU within {timestamp_ms - baseline_ts} ms.",
                        value=round(drift, 3),
                        threshold=self.config.lufs_drift_lu,
                        window_ms=self.config.lufs_window_ms,
                        context={"current_lufs_short": lufs, "baseline_lufs_short": baseline},
                    )
                )
        rows.append((timestamp_ms, lufs))

    def _check_fader_jump(
        self,
        timestamp_ms: int,
        session_id: str,
        target_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        fader_db = _safe_float(payload.get("fader_db"))
        if fader_db is None:
            return
        key = f"channel:{target_id}"
        rows = self._fader_history[key]
        _trim_window(rows, timestamp_ms, self.config.fader_window_ms)
        if rows:
            baseline_ts, baseline = rows[-1]
            delta = fader_db - baseline
            elapsed = timestamp_ms - baseline_ts
            if elapsed <= self.config.fader_window_ms and abs(delta) >= self.config.fader_jump_db:
                self._append_alert(
                    StreamingAlert(
                        timestamp_ms=timestamp_ms,
                        session_id=session_id,
                        rule="fader_jump",
                        severity="critical" if delta > 0 else "warning",
                        target_type="channel",
                        target_id=target_id,
                        message=f"Fader moved {delta:.2f} dB in {elapsed} ms.",
                        value=round(delta, 3),
                        threshold=self.config.fader_jump_db,
                        window_ms=self.config.fader_window_ms,
                        context={"current_fader_db": fader_db, "previous_fader_db": baseline},
                    )
                )
        rows.append((timestamp_ms, fader_db))

    def _check_manual_override_burst(
        self,
        timestamp_ms: int,
        session_id: str,
        target_type: str,
        target_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        if str(payload.get("actor") or "") != "human":
            return
        if str(payload.get("action_type") or "") not in MANUAL_OVERRIDE_ACTIONS:
            return
        key = f"{target_type}:{target_id}"
        rows = self._manual_override_history[key]
        cutoff = timestamp_ms - self.config.manual_override_window_ms
        while rows and rows[0] < cutoff:
            rows.popleft()
        rows.append(timestamp_ms)
        if len(rows) > self.config.manual_override_limit:
            self._append_alert(
                StreamingAlert(
                    timestamp_ms=timestamp_ms,
                    session_id=session_id,
                    rule="manual_override_burst",
                    severity="warning",
                    target_type=target_type,
                    target_id=target_id,
                    message=f"{len(rows)} manual overrides in {self.config.manual_override_window_ms} ms.",
                    value=float(len(rows)),
                    threshold=float(self.config.manual_override_limit),
                    window_ms=self.config.manual_override_window_ms,
                    context={"action_type": payload.get("action_type"), "actor": payload.get("actor")},
                )
            )

    def _check_peak_headroom(
        self,
        timestamp_ms: int,
        session_id: str,
        target_type: str,
        target_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        peak = _safe_float(payload.get("peak_dbfs"))
        if peak is None:
            peak = _safe_float(payload.get("meter_peak_dbfs"))
        if peak is None:
            return
        if peak >= self.config.clipping_peak_dbfs:
            self._append_alert(
                StreamingAlert(
                    timestamp_ms=timestamp_ms,
                    session_id=session_id,
                    rule="clipping_risk",
                    severity="critical",
                    target_type=target_type,
                    target_id=target_id,
                    message=f"Peak {peak:.2f} dBFS is at or above clipping guard.",
                    value=round(peak, 3),
                    threshold=self.config.clipping_peak_dbfs,
                    context={"peak_dbfs": peak},
                )
            )
            return
        if peak >= self.config.headroom_peak_dbfs:
            self._append_alert(
                StreamingAlert(
                    timestamp_ms=timestamp_ms,
                    session_id=session_id,
                    rule="headroom_low",
                    severity="warning",
                    target_type=target_type,
                    target_id=target_id,
                    message=f"Peak {peak:.2f} dBFS leaves limited headroom.",
                    value=round(peak, 3),
                    threshold=self.config.headroom_peak_dbfs,
                    context={"peak_dbfs": peak},
                )
            )

    def _dashboard_row(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        target_type: str,
        target_id: str,
        alerts: List[StreamingAlert],
    ) -> Dict[str, Any]:
        key = f"{target_type}:{target_id}"
        return {
            "timestamp_ms": _safe_int(payload.get("timestamp_ms")),
            "session_id": str(payload.get("session_id") or ""),
            "event_type": event_type,
            "target_type": target_type,
            "target_id": target_id,
            "lufs_short": payload.get("lufs_short", ""),
            "rms_dbfs": payload.get("rms_dbfs", payload.get("meter_rms_dbfs", "")),
            "peak_dbfs": payload.get("peak_dbfs", payload.get("meter_peak_dbfs", "")),
            "fader_db": payload.get("fader_db", ""),
            "override_count_30s": len(self._manual_override_history.get(key, [])),
            "alert_count": len(alerts),
            "alerts": "|".join(alert.rule for alert in alerts),
        }


def write_alerts_jsonl(path: str | Path, alerts: Iterable[StreamingAlert]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for alert in alerts:
            handle.write(json.dumps(alert.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def write_dashboard_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp_ms",
        "session_id",
        "event_type",
        "target_type",
        "target_id",
        "lufs_short",
        "rms_dbfs",
        "peak_dbfs",
        "fader_db",
        "override_count_30s",
        "alert_count",
        "alerts",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_streaming_alerts_for_session(
    session_id: str,
    *,
    storage_dir: str | Path = DEFAULT_STORAGE_DIR,
    output_dir: str | Path | None = None,
    config: Optional[StreamingAlertConfig] = None,
) -> Dict[str, Any]:
    timeline = export_training_timeline(session_id, storage_dir=storage_dir)
    engine = StreamingAlertEngine(config=config)
    alerts = engine.process(timeline)

    output_root = Path(output_dir) if output_dir is not None else Path(storage_dir) / session_id / "alerts"
    alerts_path = output_root / "alerts.jsonl"
    dashboard_path = output_root / "dashboard.csv"
    summary_path = output_root / "summary.json"
    write_alerts_jsonl(alerts_path, alerts)
    write_dashboard_csv(dashboard_path, engine.dashboard_rows)

    summary = {
        "type": "telemetry_streaming_alerts_summary",
        "schema_version": ALERT_SCHEMA_VERSION,
        "telemetry_schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "session_id": session_id,
        "event_count": len(timeline),
        "alert_count": len(alerts),
        "dashboard_row_count": len(engine.dashboard_rows),
        "alerts_path": str(alerts_path),
        "dashboard_path": str(dashboard_path),
        "safety": {"read_only": True, "live_write": False},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract read-only streaming alerts from Automixer telemetry")
    parser.add_argument("--storage-dir", default=DEFAULT_STORAGE_DIR)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    summary = run_streaming_alerts_for_session(
        args.session_id,
        storage_dir=args.storage_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
