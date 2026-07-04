"""Детектор нестабильности и план revert-to-baseline."""

import numpy as np
import pytest

from instability_revert import (
    InstabilityDetector,
    build_revert_plan,
)


def _feed_series(det: InstabilityDetector, channel: int, fn, duration_s=10.0, dt=0.25):
    t = 0.0
    while t <= duration_s:
        det.observe_level(channel, fn(t), t=t)
        t += dt


class TestOscillation:
    def test_fast_oscillation_triggers_revert(self):
        """«Прокачка» ±3 дБ с периодом 1 с — нестабильность."""
        det = InstabilityDetector()
        _feed_series(det, 3, lambda t: -20.0 + 3.0 * np.sin(2 * np.pi * t / 1.0))
        a = det.assess(3)
        assert a.oscillating
        assert a.should_revert
        assert a.oscillation_period_s == pytest.approx(1.0, rel=0.3)

    def test_slow_musical_dynamics_ignored(self):
        """Медленная динамика песни (период 8 с) — НЕ нестабильность."""
        det = InstabilityDetector()
        _feed_series(det, 3, lambda t: -20.0 + 4.0 * np.sin(2 * np.pi * t / 8.0))
        assert not det.assess(3).oscillating

    def test_monotonic_ramp_ignored(self):
        det = InstabilityDetector()
        _feed_series(det, 3, lambda t: -30.0 + t)
        assert not det.assess(3).should_revert

    def test_small_amplitude_ignored(self):
        """Осцилляция в пределах 2 дБ — рабочий шум, не прокачка."""
        det = InstabilityDetector()
        _feed_series(det, 3, lambda t: -20.0 + 0.7 * np.sin(2 * np.pi * t / 1.0))
        assert not det.assess(3).oscillating

    def test_steady_level_ignored(self):
        det = InstabilityDetector()
        _feed_series(det, 3, lambda t: -20.0)
        assert not det.assess(3).should_revert


class TestFeedbackBurst:
    def test_three_events_trigger(self):
        det = InstabilityDetector()
        for t in (1.0, 4.0, 8.0):
            det.observe_feedback_event(5, t=t)
        a = det.assess(5)
        assert a.feedback_burst
        assert a.should_revert
        assert a.reason == "repeated_feedback_events"

    def test_old_events_expire(self):
        det = InstabilityDetector(feedback_burst_window_s=10.0)
        det.observe_feedback_event(5, t=1.0)
        det.observe_feedback_event(5, t=2.0)
        det.observe_feedback_event(5, t=30.0)  # первые два уже вне окна
        assert not det.assess(5).feedback_burst

    def test_unknown_channel_safe(self):
        det = InstabilityDetector()
        assert not det.assess(99).should_revert


class TestRevertPlan:
    BASELINE = {
        "/ch/3/fdr": -3.6,
        "/ch/3/eq/on": 1,
        "/ch/3/eq/1g": -2.5,
        "/ch/3/in/set/trim": None,   # не прочиталось при снятии
        "/ch/4/fdr": 0.0,            # чужой канал
        "/bus/13/fdr": -6.0,         # не канал
    }

    def test_plan_only_own_channel_and_known_values(self):
        plan = build_revert_plan(3, self.BASELINE)
        addrs = {p["addr"] for p in plan}
        assert addrs == {"/ch/3/fdr", "/ch/3/eq/on", "/ch/3/eq/1g"}
        assert all(p["kind"] == "revert_to_baseline" for p in plan)

    def test_plan_carries_assessment_reason(self):
        det = InstabilityDetector()
        for t in (1.0, 2.0, 3.0):
            det.observe_feedback_event(3, t=t)
        plan = build_revert_plan(3, self.BASELINE, det.assess(3))
        assert plan[0]["reason"] == "repeated_feedback_events"

    def test_empty_baseline(self):
        assert build_revert_plan(3, {}) == []
