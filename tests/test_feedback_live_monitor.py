"""Tests for the always-on feedback live monitor (review finding C3).

Synthetic-signal tests per project convention. The growing pure tone pattern
(12 x 2048 frames, bin 42, amplitude 0.02 -> 0.5) mirrors the proven
test_server.py::test_m110_feedback_recheck_blocks_growing_pure_tone fixture.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from feedback_detector import FeedbackEvent
from feedback_live_monitor import FeedbackLiveMonitor, FeedbackLiveMonitorConfig


SR = 48000
FFT = 2048


class CursorFakeCapture:
    """Minimal AudioCapture stand-in that advances frames like live input."""

    def __init__(self, frames_per_channel: Optional[Dict[int, List[np.ndarray]]] = None,
                 num_channels: int = 4) -> None:
        self.num_channels = num_channels
        self.sample_rate = SR
        self._running = True
        self._frames = frames_per_channel or {}
        self._cursor: Dict[int, int] = {}

    @property
    def running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._running = False

    def _current_frame(self, ch: int) -> np.ndarray:
        frames = self._frames.get(ch)
        if not frames:
            return np.zeros(FFT, dtype=np.float32)
        idx = min(self._cursor.get(ch, 0), len(frames) - 1)
        return frames[idx]

    def advance(self, ch: int) -> None:
        self._cursor[ch] = self._cursor.get(ch, 0) + 1

    def get_buffer(self, channel=None, num_samples=0, channel_id=None):
        ch = channel_id or channel or 1
        frame = self._current_frame(ch)
        out = frame[-num_samples:] if num_samples else frame
        return np.asarray(out, dtype=np.float32).copy()

    def get_peak(self, channel, num_samples=1024):
        data = self.get_buffer(channel, num_samples)
        if len(data) == 0:
            return -100.0
        peak = float(np.max(np.abs(data)))
        if peak <= 1e-9:
            return -100.0
        return 20.0 * float(np.log10(peak))


def growing_tone_frames(frames: int = 12, fft_size: int = FFT) -> List[np.ndarray]:
    frequency = SR * 42.0 / fft_size
    timeline = np.arange(fft_size, dtype=np.float64) / SR
    return [
        (amplitude * np.sin(2.0 * np.pi * frequency * timeline)).astype(np.float32)
        for amplitude in np.linspace(0.02, 0.5, frames)
    ]


def make_monitor(
    capture: Any,
    sink_messages: List[Dict[str, Any]],
    **overrides: Any,
) -> FeedbackLiveMonitor:
    config_kwargs: Dict[str, Any] = dict(
        persistence_frames=6,
        min_confidence=0.5,
        peak_height_db=-12.0,
        peak_prominence_db=10.0,
        alert_min_interval_sec=5.0,
        max_alerts_per_minute=30,
    )
    config_kwargs.update(overrides)
    config = FeedbackLiveMonitorConfig(**config_kwargs)
    return FeedbackLiveMonitor(
        audio_capture_provider=lambda: capture,
        config=config,
        alert_sink=sink_messages.append,
    )


def drive(monitor: FeedbackLiveMonitor, capture: CursorFakeCapture, ch: int, steps: int) -> int:
    total = 0
    for _ in range(steps):
        total += monitor.run_once()
        capture.advance(ch)
    return total


class TestDetection:
    def test_growing_pure_tone_emits_alert(self) -> None:
        frames = growing_tone_frames()
        capture = CursorFakeCapture({1: frames})
        messages: List[Dict[str, Any]] = []
        monitor = make_monitor(capture, messages)

        emitted = drive(monitor, capture, ch=1, steps=len(frames) + 2)

        assert emitted >= 1
        assert messages, "expected at least one feedback_alert"
        alert = messages[0]
        assert alert["type"] == "feedback_alert"
        assert alert["channel"] == 1
        expected_hz = SR * 42.0 / FFT
        assert abs(alert["frequency_hz"] - expected_hz) < 50.0
        assert alert["recommended_action"] in ("notch", "fader_reduce")
        assert alert["auto_action"] == "none"
        assert alert["read_only"] is True
        assert 0.0 <= alert["confidence"] <= 1.0
        assert monitor.status()["alerts_sent_total"] >= 1

    def test_silence_emits_nothing(self) -> None:
        capture = CursorFakeCapture({1: [np.zeros(FFT, dtype=np.float32)]})
        messages: List[Dict[str, Any]] = []
        monitor = make_monitor(capture, messages)
        assert drive(monitor, capture, ch=1, steps=5) == 0
        assert messages == []

    def test_steady_pinkish_noise_emits_nothing(self) -> None:
        rng = np.random.default_rng(11)
        noise = (rng.standard_normal(FFT) * 0.05).astype(np.float32)
        capture = CursorFakeCapture({1: [noise]})
        messages: List[Dict[str, Any]] = []
        monitor = make_monitor(capture, messages)
        assert drive(monitor, capture, ch=1, steps=8) == 0
        assert messages == []

    def test_no_capture_or_stopped_capture_idles(self) -> None:
        messages: List[Dict[str, Any]] = []
        monitor = make_monitor(None, messages)
        assert monitor.run_once() == 0
        capture = CursorFakeCapture({1: growing_tone_frames()})
        capture.stop()
        monitor = make_monitor(capture, messages)
        assert monitor.run_once() == 0
        assert messages == []

    def test_excluded_channel_is_skipped(self) -> None:
        frames = growing_tone_frames()
        capture = CursorFakeCapture({1: frames})
        messages: List[Dict[str, Any]] = []
        monitor = make_monitor(capture, messages, exclude_channels=(1,))
        assert drive(monitor, capture, ch=1, steps=len(frames) + 2) == 0
        assert messages == []


class StubDetector:
    """Detector stand-in that always emits an event (rate-limit tests)."""

    def __init__(self, action: str = "notch") -> None:
        self._action = action

    def process_audio(self, channel: int, samples: np.ndarray) -> Optional[FeedbackEvent]:
        return FeedbackEvent(
            channel=channel,
            frequency_hz=1000.0,
            magnitude_db=-6.0,
            action=self._action,
            confidence=0.9,
            timestamp=time.time(),
        )


def loud_capture() -> CursorFakeCapture:
    tone = (0.4 * np.sin(2.0 * np.pi * 1000.0 * np.arange(FFT) / SR)).astype(np.float32)
    return CursorFakeCapture({1: [tone], 2: [tone]})


class TestRateLimiting:
    def test_per_key_cooldown_suppresses_repeats(self) -> None:
        capture = loud_capture()
        messages: List[Dict[str, Any]] = []
        monitor = make_monitor(capture, messages, alert_min_interval_sec=60.0)
        monitor._detector = StubDetector()
        assert monitor.run_once() == 2  # ch1 + ch2 first alerts pass
        assert monitor.run_once() == 0  # same keys inside cooldown
        assert len(messages) == 2

    def test_global_cap_enforced(self) -> None:
        capture = loud_capture()
        messages: List[Dict[str, Any]] = []
        monitor = make_monitor(
            capture,
            messages,
            alert_min_interval_sec=0.0,
            max_alerts_per_minute=3,
        )
        monitor._detector = StubDetector()
        emitted = sum(monitor.run_once() for _ in range(5))
        assert emitted == 3
        assert len(messages) == 3

    def test_alert_keys_distinct_per_frequency_bucket(self) -> None:
        capture = loud_capture()
        messages: List[Dict[str, Any]] = []

        class SweepDetector:
            def __init__(self) -> None:
                self.freq = 1000.0

            def process_audio(self, channel, samples):
                self.freq += 500.0  # new bucket each call
                return FeedbackEvent(
                    channel=channel,
                    frequency_hz=self.freq,
                    magnitude_db=-6.0,
                    action="notch",
                    confidence=0.9,
                    timestamp=time.time(),
                )

        monitor = make_monitor(capture, messages, alert_min_interval_sec=60.0)
        monitor._detector = SweepDetector()
        assert monitor.run_once() == 2  # distinct buckets pass the cooldown
        assert len(messages) == 2


class TestLifecycle:
    def test_disabled_config_never_starts_thread(self) -> None:
        capture = loud_capture()
        monitor = make_monitor(capture, [], enabled=False)
        monitor.start()
        assert monitor.running is False
        assert monitor.status()["enabled"] is False

    def test_start_stop_thread_idempotent(self) -> None:
        capture = loud_capture()
        monitor = make_monitor(capture, [], poll_interval_sec=0.02, idle_interval_sec=0.05)
        monitor.start()
        monitor.start()  # second start must not spawn another thread
        first_thread = monitor._thread
        time.sleep(0.05)
        assert monitor.running is True
        assert monitor._thread is first_thread
        monitor.stop()
        assert monitor.running is False
        monitor.stop()  # idempotent

    def test_sink_exception_does_not_kill_loop(self) -> None:
        capture = loud_capture()

        def bad_sink(_message: Dict[str, Any]) -> None:
            raise RuntimeError("sink exploded")

        monitor = FeedbackLiveMonitor(
            audio_capture_provider=lambda: capture,
            config=FeedbackLiveMonitorConfig(alert_min_interval_sec=0.0),
            alert_sink=bad_sink,
        )
        monitor._detector = StubDetector()
        assert monitor.run_once() == 2  # alert counted even though sink raised
        assert monitor.status()["alerts_sent_total"] == 2


class TestConfigFromDict:
    def test_defaults_when_section_missing(self) -> None:
        cfg = FeedbackLiveMonitorConfig.from_dict({})
        assert cfg.enabled is True
        assert cfg.auto_apply is False
        assert cfg.signal_threshold_db == -50.0

    def test_yaml_like_section_parsed_and_clamped(self) -> None:
        cfg = FeedbackLiveMonitorConfig.from_dict(
            {
                "enabled": True,
                "signal_threshold_db": "-45.5",
                "poll_interval_sec": 0.0,  # clamped to >= 0.02
                "fft_size": 4096,
                "exclude_channels": [47, 48, "bogus"],
                "max_alerts_per_minute": 10,
            }
        )
        assert cfg.signal_threshold_db == -45.5
        assert cfg.poll_interval_sec == 0.02
        assert cfg.fft_size == 4096
        assert cfg.exclude_channels == (47, 48)
        assert cfg.max_alerts_per_minute == 10

    def test_garbage_values_fall_back_to_defaults(self) -> None:
        cfg = FeedbackLiveMonitorConfig.from_dict(
            {"enabled": True, "fft_size": "not-a-number", "exclude_channels": "nope"}
        )
        assert cfg.fft_size == 2048
        assert cfg.exclude_channels == ()
