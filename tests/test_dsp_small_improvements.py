"""Мелкие DSP-улучшения: LCR-правило панорамы, FX pre-delay/decay от темпа."""

import numpy as np
import pytest

from auto_fx import AutoFXPlanner
from auto_panner_adaptive import (
    lf_energy_ratio,
    should_center_low_frequency,
)

SR = 48000


class TestLCRRule:
    def test_bass_centered(self):
        t = np.arange(SR) / SR
        bass = np.sin(2 * np.pi * 60.0 * t)
        assert should_center_low_frequency(lf_energy_ratio(bass, SR))

    def test_guitar_range_not_centered(self):
        t = np.arange(SR) / SR
        gtr = np.sin(2 * np.pi * 400.0 * t) + 0.5 * np.sin(2 * np.pi * 1200.0 * t)
        assert not should_center_low_frequency(lf_energy_ratio(gtr, SR))

    def test_ratio_bounded(self):
        rng = np.random.default_rng(1)
        assert 0.0 <= lf_energy_ratio(rng.normal(0, 0.3, SR), SR) <= 1.0

    def test_short_signal_safe(self):
        assert lf_energy_ratio(np.zeros(64), SR) == 0.0


class TestFXTempoTimes:
    def test_predelay_and_decay_follow_tempo(self):
        """120 BPM: plate pre-delay = 1/32 = 62.5 мс; decay = целая − pre."""
        plan = AutoFXPlanner(tempo_bpm=120.0).create_plan({})
        plate = next(b for b in plan.buses if b.name == "Vocal Plate")
        assert plate.params["predelay_ms"] == pytest.approx(62.5)
        assert plate.params["decay_s"] == pytest.approx(2.0 - 0.0625, abs=0.01)
        room = next(b for b in plan.buses if b.name == "Drum Room")
        assert room.params["predelay_ms"] == pytest.approx(25.0)  # кламп 1/64
        assert room.params["decay_s"] == pytest.approx(0.975, abs=0.01)

    def test_faster_tempo_shorter_tail(self):
        slow = AutoFXPlanner(tempo_bpm=80.0).create_plan({})
        fast = AutoFXPlanner(tempo_bpm=170.0).create_plan({})
        decay = lambda plan, name: next(
            b for b in plan.buses if b.name == name).params["decay_s"]
        assert decay(fast, "Vocal Plate") < decay(slow, "Vocal Plate")
        assert decay(fast, "Drum Room") <= decay(slow, "Drum Room")

    def test_clamps_hold_musical_range(self):
        for bpm in (40.0, 240.0):
            plan = AutoFXPlanner(tempo_bpm=bpm).create_plan({})
            plate = next(b for b in plan.buses if b.name == "Vocal Plate")
            assert 20.0 <= plate.params["predelay_ms"] <= 80.0
            assert 1.0 <= plate.params["decay_s"] <= 2.6
