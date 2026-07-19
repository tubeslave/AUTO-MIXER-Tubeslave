"""Always-on live feedback detection service.

Closes review finding C3 (2026-07-19): before this service, continuous
feedback suppression existed only while ``AutoSoundcheckEngine`` was running
(its ``_monitor_loop``). This monitor runs the proven ``FeedbackDetector``
against the server's single shared ``AudioCapture`` whenever capture is
active, and emits ``feedback_alert`` messages for the operator through the
server broadcast path.

Scope: detection + operator alerting only. The service performs **no console
writes** — every gated write path in ``server.py`` requires an operator-bound
context and fails closed otherwise (see
``Docs/adr/c3-always-on-feedback-detection.md``). Auto fader action is a
reserved config flag pending a dedicated Council decision.

Threading: engine-style daemon thread (the detector is CPU-bound numpy).
``AudioCapture`` is safe for external readers (internal ``_data_lock``,
copies on read); the detector is not thread-safe, so this service serializes
detector access with its own lock — the same pattern the engine uses with
``_monitor_iteration_lock``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from feedback_detector import FeedbackDetector, FeedbackDetectorConfig

logger = logging.getLogger(__name__)


@dataclass
class FeedbackLiveMonitorConfig:
    """Tuning for the always-on feedback monitor.

    Detector defaults mirror the soundcheck engine tuning
    (``autofoh.detectors.feedback_*`` fallbacks in auto_soundcheck_engine.py).
    """

    enabled: bool = True
    sample_rate: int = 48000
    signal_threshold_db: float = -50.0  # engine SIGNAL_THRESHOLD_DB
    poll_interval_sec: float = 0.1
    idle_interval_sec: float = 1.0
    fft_size: int = 2048
    persistence_frames: int = 12
    min_confidence: float = 0.75
    peak_height_db: float = -12.0
    peak_prominence_db: float = 10.0
    alert_freq_bucket_hz: float = 50.0
    alert_min_interval_sec: float = 5.0
    max_alerts_per_minute: int = 30
    exclude_channels: Tuple[int, ...] = ()
    # Reserved: no console writes in v1 regardless of this flag; see ADR.
    auto_apply: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackLiveMonitorConfig":
        if not isinstance(data, dict):
            return cls()
        cfg = cls()

        def _float(key: str, current: float) -> float:
            value = data.get(key, current)
            try:
                return float(value)
            except (TypeError, ValueError):
                return current

        def _int(key: str, current: int) -> int:
            value = data.get(key, current)
            try:
                return int(value)
            except (TypeError, ValueError):
                return current

        cfg.enabled = bool(data.get("enabled", cfg.enabled))
        cfg.sample_rate = _int("sample_rate", cfg.sample_rate)
        cfg.signal_threshold_db = _float("signal_threshold_db", cfg.signal_threshold_db)
        cfg.poll_interval_sec = max(0.02, _float("poll_interval_sec", cfg.poll_interval_sec))
        cfg.idle_interval_sec = max(0.1, _float("idle_interval_sec", cfg.idle_interval_sec))
        cfg.fft_size = max(256, _int("fft_size", cfg.fft_size))
        cfg.persistence_frames = max(1, _int("persistence_frames", cfg.persistence_frames))
        cfg.min_confidence = _float("min_confidence", cfg.min_confidence)
        cfg.peak_height_db = _float("peak_height_db", cfg.peak_height_db)
        cfg.peak_prominence_db = _float("peak_prominence_db", cfg.peak_prominence_db)
        cfg.alert_freq_bucket_hz = max(1.0, _float("alert_freq_bucket_hz", cfg.alert_freq_bucket_hz))
        cfg.alert_min_interval_sec = max(0.0, _float("alert_min_interval_sec", cfg.alert_min_interval_sec))
        cfg.max_alerts_per_minute = max(1, _int("max_alerts_per_minute", cfg.max_alerts_per_minute))
        raw_exclude = data.get("exclude_channels", ())
        if isinstance(raw_exclude, (list, tuple)):
            cfg.exclude_channels = tuple(
                int(ch) for ch in raw_exclude if isinstance(ch, (int, float)) and not isinstance(ch, bool)
            )
        cfg.auto_apply = bool(data.get("auto_apply", cfg.auto_apply))
        return cfg


class FeedbackLiveMonitor:
    """Background always-on feedback detection + operator alerting.

    Parameters
    ----------
    audio_capture_provider:
        Callable returning the server's current ``AudioCapture`` (or None).
        The monitor idles when no capture is available or running.
    config:
        ``FeedbackLiveMonitorConfig``.
    alert_sink:
        Callable accepting the alert message dict. Called from the monitor
        thread; the server wires it to ``loop.call_soon_threadsafe`` +
        ``asyncio.create_task(self.broadcast(...))``.
    """

    def __init__(
        self,
        *,
        audio_capture_provider: Callable[[], Optional[Any]],
        config: FeedbackLiveMonitorConfig,
        alert_sink: Callable[[Dict[str, Any]], None],
    ) -> None:
        self._capture_provider = audio_capture_provider
        self.config = config
        self._alert_sink = alert_sink
        self._detector = FeedbackDetector(
            config=FeedbackDetectorConfig(
                sample_rate=config.sample_rate,
                fft_size=config.fft_size,
                persistence_frames=config.persistence_frames,
                min_confidence=config.min_confidence,
                peak_height_db=config.peak_height_db,
                peak_prominence_db=config.peak_prominence_db,
                require_tonality_criteria=True,
            )
        )
        self._detector_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_alert_at: Dict[Tuple[int, str, int], float] = {}
        self._alert_timestamps: List[float] = []
        self._consecutive_errors = 0
        self._alerts_sent_total = 0
        self._started_at_epoch: Optional[float] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the monitor thread. Idempotent; no-op when disabled."""
        if not self.config.enabled:
            logger.info("Feedback live monitor disabled by config")
            return
        if self.running:
            return
        self._stop_event.clear()
        self._started_at_epoch = time.time()
        self._thread = threading.Thread(
            target=self._loop,
            name="feedback-live-monitor",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Feedback live monitor started (detection + alerts only; no console writes)"
        )

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.error(
                    "Feedback live monitor thread did not stop within %.1fs", timeout
                )
        self._thread = None

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "running": self.running,
            "auto_apply": False,
            "read_only": True,
            "alerts_sent_total": self._alerts_sent_total,
            "consecutive_errors": self._consecutive_errors,
            "started_at_epoch": self._started_at_epoch,
        }

    def run_once(self) -> int:
        """Run one detection iteration over all captured channels.

        Returns the number of alerts emitted. Exposed for deterministic tests;
        the background loop calls it repeatedly.
        """
        capture = self._capture_provider()
        if capture is None or getattr(capture, "running", False) is not True:
            return 0
        num_channels = int(getattr(capture, "num_channels", 0) or 0)
        if num_channels <= 0:
            return 0
        excluded = set(self.config.exclude_channels)
        alerts = 0
        for ch in range(1, num_channels + 1):
            if ch in excluded:
                continue
            try:
                peak = capture.get_peak(ch, self.config.fft_size * 2)
            except Exception:
                continue
            if peak < self.config.signal_threshold_db:
                continue
            samples = capture.get_buffer(ch, self.config.fft_size)
            if len(samples) < self.config.fft_size:
                continue
            with self._detector_lock:
                event = self._detector.process_audio(
                    ch, np.asarray(samples, dtype=np.float32)
                )
            if event is None:
                continue
            if self._emit_alert(ch, event):
                alerts += 1
        return alerts

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            capture = self._capture_provider()
            active = capture is not None and getattr(capture, "running", False) is True
            if active:
                try:
                    self.run_once()
                    self._consecutive_errors = 0
                except Exception:
                    self._consecutive_errors += 1
                    logger.exception(
                        "Feedback live monitor error (%s consecutive)",
                        self._consecutive_errors,
                    )
                    if self._consecutive_errors == 10:
                        logger.critical(
                            "Feedback live monitor failing persistently: "
                            "always-on feedback detection degraded, operator "
                            "attention required"
                        )
            interval = (
                self.config.poll_interval_sec if active else self.config.idle_interval_sec
            )
            self._stop_event.wait(interval)

    def _emit_alert(self, ch: int, event: Any) -> bool:
        now = time.monotonic()
        bucket = int(
            round(float(event.frequency_hz or 0.0) / max(1.0, self.config.alert_freq_bucket_hz))
        )
        key = (int(ch), str(event.action), bucket)
        last = self._last_alert_at.get(key)
        if last is not None and (now - last) < self.config.alert_min_interval_sec:
            return False
        window_start = now - 60.0
        self._alert_timestamps = [t for t in self._alert_timestamps if t >= window_start]
        if len(self._alert_timestamps) >= self.config.max_alerts_per_minute:
            logger.warning(
                "Feedback live monitor: global alert cap (%s/min) reached, "
                "suppressing alert for Ch %s",
                self.config.max_alerts_per_minute,
                ch,
            )
            return False
        self._last_alert_at[key] = now
        self._alert_timestamps.append(now)
        self._alerts_sent_total += 1
        message = {
            "type": "feedback_alert",
            "channel": int(ch),
            "frequency_hz": round(float(event.frequency_hz), 2),
            "magnitude_db": round(float(event.magnitude_db), 2),
            "recommended_action": str(event.action),
            "confidence": round(float(event.confidence), 3),
            # Detection + alert only: no console writes in v1 (see module docstring).
            "auto_action": "none",
            "read_only": True,
            "timestamp": time.time(),
        }
        logger.warning(
            "FEEDBACK live Ch %s: %s at %.0fHz (%.1fdB, confidence %.2f)",
            ch,
            event.action,
            float(event.frequency_hz),
            float(event.magnitude_db),
            float(event.confidence),
        )
        try:
            self._alert_sink(message)
        except Exception:
            logger.exception("Feedback live monitor: alert sink failed")
        return True
