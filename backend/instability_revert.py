"""Детектор нестабильности + план отката к baseline (revert-to-snapshot).

Live-контур с автоматическими коррекциями может стать нестабильным:
осцилляция уровня («прокачка» с периодом < 2 с — коррекции дерутся с
программой или друг с другом), либо серия feedback-срабатываний подряд.
В обоих случаях правильное действие — не следующая коррекция, а мгновенный
откат канала к baseline-снимку саундчека.

Proposal-only: модуль выдаёт план отката (адрес -> значение baseline);
применение идёт через существующий supervised manual-write путь. Baseline —
это rollback-снимок формата reports/*_original_*.json (address -> value),
который сохраняет каждый apply-инструмент проекта.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

DEFAULT_WINDOW_S = 12.0
DEFAULT_MAX_OSC_PERIOD_S = 2.0
DEFAULT_MIN_OSC_CYCLES = 3
DEFAULT_MIN_OSC_AMPLITUDE_DB = 2.0
DEFAULT_FEEDBACK_BURST_COUNT = 3
DEFAULT_FEEDBACK_BURST_WINDOW_S = 10.0


@dataclass
class InstabilityAssessment:
    channel: int
    oscillating: bool = False
    oscillation_period_s: Optional[float] = None
    oscillation_amplitude_db: Optional[float] = None
    feedback_burst: bool = False
    feedback_event_count: int = 0
    should_revert: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "oscillating": self.oscillating,
            "oscillation_period_s": self.oscillation_period_s,
            "oscillation_amplitude_db": self.oscillation_amplitude_db,
            "feedback_burst": self.feedback_burst,
            "feedback_event_count": self.feedback_event_count,
            "should_revert": self.should_revert,
            "reason": self.reason,
        }


@dataclass
class _ChannelHistory:
    levels: List[tuple] = field(default_factory=list)     # (t, level_db)
    feedback_events: List[float] = field(default_factory=list)  # t


class InstabilityDetector:
    """Следит за уровнями и feedback-событиями, решает «пора откатывать»."""

    def __init__(
        self,
        window_s: float = DEFAULT_WINDOW_S,
        max_oscillation_period_s: float = DEFAULT_MAX_OSC_PERIOD_S,
        min_oscillation_cycles: int = DEFAULT_MIN_OSC_CYCLES,
        min_oscillation_amplitude_db: float = DEFAULT_MIN_OSC_AMPLITUDE_DB,
        feedback_burst_count: int = DEFAULT_FEEDBACK_BURST_COUNT,
        feedback_burst_window_s: float = DEFAULT_FEEDBACK_BURST_WINDOW_S,
        time_provider=time.monotonic,
    ):
        self.window_s = window_s
        self.max_oscillation_period_s = max_oscillation_period_s
        self.min_oscillation_cycles = min_oscillation_cycles
        self.min_oscillation_amplitude_db = min_oscillation_amplitude_db
        self.feedback_burst_count = feedback_burst_count
        self.feedback_burst_window_s = feedback_burst_window_s
        self.time_provider = time_provider
        self._channels: Dict[int, _ChannelHistory] = {}

    def observe_level(self, channel: int, level_db: float, t: Optional[float] = None):
        t = float(self.time_provider()) if t is None else float(t)
        hist = self._channels.setdefault(channel, _ChannelHistory())
        hist.levels.append((t, float(level_db)))
        self._trim(hist, t)

    def observe_feedback_event(self, channel: int, t: Optional[float] = None):
        t = float(self.time_provider()) if t is None else float(t)
        hist = self._channels.setdefault(channel, _ChannelHistory())
        hist.feedback_events.append(t)
        self._trim(hist, t)

    def _trim(self, hist: _ChannelHistory, now: float):
        horizon = now - self.window_s
        hist.levels = [(t, v) for t, v in hist.levels if t >= horizon]
        fb_horizon = now - self.feedback_burst_window_s
        hist.feedback_events = [t for t in hist.feedback_events if t >= fb_horizon]

    def assess(self, channel: int) -> InstabilityAssessment:
        hist = self._channels.get(channel)
        result = InstabilityAssessment(channel=channel)
        if hist is None:
            return result

        result.feedback_event_count = len(hist.feedback_events)
        if result.feedback_event_count >= self.feedback_burst_count:
            result.feedback_burst = True

        osc = self._detect_oscillation(hist)
        if osc is not None:
            result.oscillating = True
            result.oscillation_period_s, result.oscillation_amplitude_db = osc

        result.should_revert = result.oscillating or result.feedback_burst
        if result.oscillating and result.feedback_burst:
            result.reason = "oscillation_and_feedback_burst"
        elif result.oscillating:
            result.reason = "level_oscillation_period_below_2s"
        elif result.feedback_burst:
            result.reason = "repeated_feedback_events"
        return result

    def _detect_oscillation(self, hist: _ChannelHistory):
        """Осцилляция: чередующиеся пересечения среднего уровня с периодом
        < max_oscillation_period_s и амплитудой >= min_oscillation_amplitude_db,
        не меньше min_oscillation_cycles полных циклов."""
        if len(hist.levels) < self.min_oscillation_cycles * 4:
            return None
        times = np.array([t for t, _ in hist.levels])
        levels = np.array([v for _, v in hist.levels])
        detrended = levels - np.mean(levels)
        amplitude = float(np.max(detrended) - np.min(detrended))
        if amplitude < self.min_oscillation_amplitude_db:
            return None

        signs = np.sign(detrended)
        signs[signs == 0] = 1
        crossings = np.where(np.diff(signs) != 0)[0]
        if len(crossings) < self.min_oscillation_cycles * 2:
            return None
        crossing_times = times[crossings]
        # период = 2 средних интервала между пересечениями нуля
        intervals = np.diff(crossing_times)
        if len(intervals) == 0:
            return None
        period = 2.0 * float(np.median(intervals))
        if period > self.max_oscillation_period_s or period <= 0.0:
            return None
        return period, amplitude

    def reset_channel(self, channel: int):
        self._channels.pop(channel, None)

    def reset_all(self):
        self._channels.clear()


def build_revert_plan(
    channel: int,
    baseline: Mapping[str, Any],
    assessment: Optional[InstabilityAssessment] = None,
) -> List[Dict[str, Any]]:
    """План отката канала к baseline-снимку (proposal-only, без OSC).

    baseline — mapping «адрес -> исходное значение» (формат rollback-снимков
    reports/*_original_*.json). Отбираются только адреса этого канала;
    None-значения (не прочитались при снятии) пропускаются.
    """
    prefix = f"/ch/{int(channel)}/"
    plan = []
    for addr, value in baseline.items():
        if not str(addr).startswith(prefix) or value is None:
            continue
        plan.append({
            "addr": str(addr),
            "value": value,
            "kind": "revert_to_baseline",
            "reason": (assessment.reason if assessment else "operator_requested"),
        })
    return plan
