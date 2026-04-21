"""
Non-blocking JSONL logging for AutoFOH actions and evaluations.

Writes happen on a background thread so the main engine loop does not block on
disk IO.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any, Dict, Iterable, List, Optional


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize(item) for item in value]
    if is_dataclass(value):
        return {key: _normalize(item) for key, item in value.__dict__.items()}
    if hasattr(value, "__dict__"):
        return {key: _normalize(item) for key, item in value.__dict__.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return repr(value)
    return repr(value)


@dataclass
class AutoFOHLoggerStats:
    enqueued: int = 0
    written: int = 0
    dropped: int = 0
    errors: int = 0
    last_error: str = ""


@dataclass
class AutoFOHStructuredLogger:
    path: Path
    queue_maxsize: int = 1024
    stats: AutoFOHLoggerStats = field(default_factory=AutoFOHLoggerStats)

    def __post_init__(self):
        self.path = Path(self.path)
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, self.queue_maxsize))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.is_running:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0):
        if self._thread is None:
            return
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)
        self._thread = None

    def log_event(self, event_type: str, **payload) -> bool:
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            **_normalize(payload),
        }
        try:
            self._queue.put_nowait(event)
            self.stats.enqueued += 1
            return True
        except queue.Full:
            self.stats.dropped += 1
            return False

    def _worker(self):
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                while not self._stop_event.is_set() or not self._queue.empty():
                    try:
                        item = self._queue.get(timeout=0.25)
                    except queue.Empty:
                        continue
                    if item is None:
                        continue
                    try:
                        handle.write(json.dumps(item, ensure_ascii=True) + "\n")
                        handle.flush()
                        self.stats.written += 1
                    except Exception as exc:
                        self.stats.errors += 1
                        self.stats.last_error = str(exc)
        except Exception as exc:
            self.stats.errors += 1
            self.stats.last_error = str(exc)


@dataclass
class AutoFOHSessionReport:
    generated_at: float
    log_path: str
    total_events: int = 0
    event_counts: Dict[str, int] = field(default_factory=dict)
    action_sent_count: int = 0
    action_blocked_count: int = 0
    rollback_count: int = 0
    evaluation_count: int = 0
    guard_block_count: int = 0
    guard_blocks_by_reason: Dict[str, int] = field(default_factory=dict)
    guard_blocks_by_action_type: Dict[str, int] = field(default_factory=dict)
    guard_blocks_by_runtime_state: Dict[str, int] = field(default_factory=dict)
    guard_blocks_by_phase: Dict[str, int] = field(default_factory=dict)
    guard_blocks_by_channel: Dict[str, int] = field(default_factory=dict)
    channels_with_guard_blocks: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _normalize(self)


def _increment(counter: Dict[str, int], key: Any):
    normalized_key = str(key)
    counter[normalized_key] = int(counter.get(normalized_key, 0)) + 1


def iter_jsonl_events(path: Path | str) -> Iterable[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []

    def _iterator():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    return _iterator()


def build_session_report(events: Iterable[Dict[str, Any]], *, log_path: Path | str = "") -> AutoFOHSessionReport:
    report = AutoFOHSessionReport(
        generated_at=time.time(),
        log_path=str(log_path),
    )

    for event in events:
        event_type = str(event.get("event_type", "unknown"))
        report.total_events += 1
        _increment(report.event_counts, event_type)

        if event_type == "action_decision":
            if bool(event.get("sent", False)):
                report.action_sent_count += 1
            else:
                report.action_blocked_count += 1

        if event_type == "action_rollback":
            report.rollback_count += 1

        if event_type == "action_evaluation":
            report.evaluation_count += 1

        if event_type != "phase_target_guard_blocked":
            continue

        report.guard_block_count += 1
        _increment(
            report.guard_blocks_by_reason,
            event.get("message", "unknown"),
        )
        action_payload = event.get("action", {})
        _increment(
            report.guard_blocks_by_action_type,
            action_payload.get("action_type") or action_payload.get("type") or "unknown",
        )
        _increment(
            report.guard_blocks_by_runtime_state,
            event.get("runtime_state", "unknown"),
        )
        metadata = event.get("metadata", {}) or {}
        _increment(
            report.guard_blocks_by_phase,
            metadata.get("phase_name", "unknown"),
        )
        channel_id = event.get("channel_id", metadata.get("channel_id"))
        if channel_id is not None:
            _increment(report.guard_blocks_by_channel, channel_id)

    report.channels_with_guard_blocks = sorted(
        int(channel_id)
        for channel_id in report.guard_blocks_by_channel.keys()
        if str(channel_id).isdigit()
    )
    return report


def build_session_report_from_jsonl(path: Path | str) -> AutoFOHSessionReport:
    path = Path(path)
    return build_session_report(iter_jsonl_events(path), log_path=path)


def render_session_report_summary(
    report: Optional[AutoFOHSessionReport],
    *,
    max_channels: int = 4,
) -> str:
    if report is None:
        return ""

    parts = [
        f"events={report.total_events}",
        f"sent={report.action_sent_count}",
        f"blocked={report.action_blocked_count}",
    ]
    if report.guard_block_count:
        parts.append(f"guard_blocks={report.guard_block_count}")
        if report.channels_with_guard_blocks:
            channels = ", ".join(
                str(channel_id)
                for channel_id in report.channels_with_guard_blocks[:max_channels]
            )
            if len(report.channels_with_guard_blocks) > max_channels:
                channels += ", ..."
            parts.append(f"channels=[{channels}]")
        if report.guard_blocks_by_action_type:
            top_action_type = max(
                report.guard_blocks_by_action_type.items(),
                key=lambda item: item[1],
            )[0]
            parts.append(f"top_guard_action={top_action_type}")
    if report.rollback_count:
        parts.append(f"rollbacks={report.rollback_count}")
    if report.evaluation_count:
        parts.append(f"evaluations={report.evaluation_count}")
    return "AutoFOH session report: " + "; ".join(parts)


def write_session_report(
    report: AutoFOHSessionReport,
    path: Path | str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, ensure_ascii=True, indent=2, sort_keys=True)
    return path
