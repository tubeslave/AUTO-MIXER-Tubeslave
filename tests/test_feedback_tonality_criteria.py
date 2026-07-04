"""Тональные критерии заводки PHPR/PNPR/IMSD (van Waterschoot & Moonen 2010).

Ключевой контракт: устойчивый растущий тон С ГАРМОНИКАМИ (нота инструмента)
не подтверждается как заводка; чистый растущий синус — подтверждается.
"""

import numpy as np
import pytest

from feedback_detector import (
    FeedbackDetector,
    FeedbackDetectorConfig,
    RingOutPlanner,
    compute_imsd_db,
    compute_phpr_db,
    compute_pnpr_db,
)

SR = 48000
FFT = 2048


def _spectrum_db(x: np.ndarray) -> np.ndarray:
    block = x[-FFT:] * np.hanning(FFT)
    return 20.0 * np.log10(np.abs(np.fft.rfft(block)) + 1e-10)


def _tone(freq: float, n: int, amp: float, harmonics: dict[int, float] | None = None) -> np.ndarray:
    t = np.arange(n) / SR
    x = amp * np.sin(2 * np.pi * freq * t)
    for mult, h_amp in (harmonics or {}).items():
        x += amp * h_amp * np.sin(2 * np.pi * freq * mult * t)
    return x.astype(np.float32)


def _feed_growing(detector: FeedbackDetector, freq: float, harmonics=None, frames: int = 12):
    """Скармливает детектору растущий тон, возвращает последнее событие."""
    event = None
    for i in range(frames):
        amp = 0.05 * (1.25 ** i)  # экспоненциальный разгон
        block = _tone(freq, FFT, min(amp, 0.9), harmonics)
        event = detector.process_audio(1, block) or event
    return event


class TestCriteriaFunctions:
    def test_phpr_high_for_pure_tone(self):
        db = _spectrum_db(_tone(1000.0, FFT, 0.5))
        peak_bin = int(round(1000.0 / (SR / FFT)))
        phpr = compute_phpr_db(db, peak_bin, SR / FFT)
        assert phpr is not None and phpr > 30.0

    def test_phpr_low_for_harmonic_note(self):
        db = _spectrum_db(_tone(1000.0, FFT, 0.5, harmonics={2: 0.4, 3: 0.25}))
        peak_bin = int(round(1000.0 / (SR / FFT)))
        phpr = compute_phpr_db(db, peak_bin, SR / FFT)
        assert phpr is not None and phpr < 15.0

    def test_phpr_none_when_harmonics_beyond_nyquist(self):
        db = _spectrum_db(_tone(15000.0, FFT, 0.5))
        peak_bin = int(round(15000.0 / (SR / FFT)))
        assert compute_phpr_db(db, peak_bin, SR / FFT) is None

    def test_pnpr_high_for_tone_low_for_noise(self):
        db_tone = _spectrum_db(_tone(1000.0, FFT, 0.5))
        peak_bin = int(round(1000.0 / (SR / FFT)))
        assert compute_pnpr_db(db_tone, peak_bin) > 12.0

        rng = np.random.default_rng(7)
        db_noise = _spectrum_db(rng.normal(0, 0.3, FFT).astype(np.float32))
        noisy_bin = int(np.argmax(db_noise[10:-10])) + 10
        assert compute_pnpr_db(db_noise, noisy_bin) < 12.0

    def test_imsd_exponential_growth(self):
        history = [(-40.0 + 2.0 * i) for i in range(8)]  # постоянный наклон +2 dB
        slope, dev = compute_imsd_db(history)
        assert slope == pytest.approx(2.0)
        assert dev == pytest.approx(0.0, abs=1e-9)

    def test_imsd_none_for_short_history(self):
        assert compute_imsd_db([-30.0, -28.0], min_frames=4) is None


class TestDetectorWithCriteria:
    def test_pure_growing_sine_confirmed(self):
        det = FeedbackDetector(config=FeedbackDetectorConfig(persistence_frames=4))
        event = _feed_growing(det, 3150.0)
        assert event is not None
        assert event.action == "notch"
        assert abs(event.frequency_hz - 3150.0) / 3150.0 < 0.02

    def test_harmonic_note_rejected(self):
        """Нота с гармониками (аккордеон/орган) не подтверждается как заводка."""
        det = FeedbackDetector(config=FeedbackDetectorConfig(persistence_frames=4))
        event = _feed_growing(det, 440.0, harmonics={2: 0.5, 3: 0.3})
        assert event is None

    def test_harmonic_note_confirmed_when_criteria_disabled(self):
        """Отключение критериев возвращает старое поведение (регрессия-защита)."""
        det = FeedbackDetector(config=FeedbackDetectorConfig(
            persistence_frames=4, require_tonality_criteria=False,
        ))
        event = _feed_growing(det, 440.0, harmonics={2: 0.5, 3: 0.3})
        assert event is not None


class TestRingOutPlanner:
    def test_ring_out_collects_fixed_notch_plan(self):
        planner = RingOutPlanner()
        for i in range(14):
            amp = 0.05 * (1.25 ** i)
            planner.observe_block(_tone(2000.0, FFT, min(amp, 0.9)))
        plan = planner.fixed_notch_plan()
        assert len(plan) == 1
        assert abs(plan[0]["frequency_hz"] - 2000.0) / 2000.0 < 0.02
        assert plan[0]["gain_db"] == -3.0
        assert plan[0]["kind"] == "fixed_ring_out_notch"

    def test_ring_out_ignores_harmonic_notes(self):
        planner = RingOutPlanner()
        for i in range(14):
            amp = 0.05 * (1.25 ** i)
            planner.observe_block(_tone(500.0, FFT, min(amp, 0.9), harmonics={2: 0.5, 3: 0.3}))
        assert planner.fixed_notch_plan() == []

    def test_reset(self):
        planner = RingOutPlanner()
        for i in range(14):
            planner.observe_block(_tone(2000.0, FFT, min(0.05 * (1.25 ** i), 0.9)))
        assert planner.ring_frequencies()
        planner.reset()
        assert planner.ring_frequencies() == []
