"""Non-blocking JSONL logging for source-grounded decisions and feedback."""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any, Dict, Iterable, Optional

from .models import DecisionTrace, FeedbackRecord


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
        return _normalize(value.__dict__)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return repr(value)
    if hasattr(value, "__dict__"):
        return _normalize(value.__dict__)
    return repr(value)


@dataclass
class SourceLoggerStats:
    enqueued: int = 0
    written: int = 0
    dropped: int = 0
    errors: int = 0
    last_error: str = ""


@dataclass
class SourceDecisionLogger:
    """Background JSONL logger.

    The logger is safe to keep on the side of live code because writes happen on
    a worker thread and queue overflow drops log rows instead of blocking audio
    or OSC paths.
    """

    path: Path | str
    queue_maxsize: int = 256
    stats: SourceLoggerStats = field(default_factory=SourceLoggerStats)

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser()
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(self.queue_maxsize)))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)
        self._thread = None

    def log_event(self, event_type: str, **payload: Any) -> bool:
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

    def log_decision(self, trace: DecisionTrace) -> bool:
        return self.log_event("source_decision", **trace.to_dict())

    def log_feedback(self, feedback: FeedbackRecord) -> bool:
        return self.log_event("source_feedback", **feedback.to_dict())

    def _worker(self) -> None:
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
                        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                        handle.flush()
                        self.stats.written += 1
                    except Exception as exc:
                        self.stats.errors += 1
                        self.stats.last_error = str(exc)
        except Exception as exc:
            self.stats.errors += 1
            self.stats.last_error = str(exc)


def iter_jsonl_events(path: Path | str) -> Iterable[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []

    def _iterator():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    return _iterator()
