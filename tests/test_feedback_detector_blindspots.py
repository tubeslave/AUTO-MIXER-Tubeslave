"""Ф2 (задача #6): слепые зоны детектора заводки, найденные ревью 2026-07-29.

FB-D3 — установившаяся заводка (упёрлась в компрессию тракта, НЕ растёт)
        обязана детектироваться без требования is_growing().
FB-D4 — дрейфующая заводка (движение микрофона): трекинг не разваливается
        при уходе пика с бина-ключа.
FB-D6 — magnitude в dBFS: синус полной шкалы читается ~0 дБ.
FB-D7 — обязательный регресс правила CLAUDE.md: нота, представленная
        2-й гармоникой, НЕ флагуется (суб-гармоника f/2 её выдаёт).
"""

import numpy as np
import pytest


def _detector(**overrides):
    from feedback_detector import FeedbackDetector, FeedbackDetectorConfig
    cfg = FeedbackDetectorConfig(**overrides)
    return FeedbackDetector(config=cfg)


def _tone(freq, fs, n, amp, phase=0.0):
    t = np.arange(n) / fs
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


FS = 48000
N = 2048


class TestSteadyStateFeedback:
    def test_constant_level_squeal_detected(self):
        """Стационарный визг монитора: постоянный высокий уровень, 40 кадров."""
        det = _detector()
        event = None
        for _ in range(40):
            event = event or det.process_audio(1, _tone(3150.0, FS, N, 0.4))
        assert event is not None, (
            "установившаяся заводка не детектирована — is_growing() снова "
            "обязателен (ревью FB-D3)"
        )
        assert abs(event.frequency_hz - 3150.0) < 47.0
        assert event.action in ("notch", "fader_reduce")

    def test_quiet_steady_tone_not_flagged(self):
        """Тихий устойчивый тон (ниже steady-порога) — не заводка."""
        det = _detector()
        event = None
        for _ in range(40):
            event = event or det.process_audio(1, _tone(3150.0, FS, N, 0.005))
        assert event is None, "тихий тон (−46 dBFS) флагнут как заводка"


class TestDriftingFeedback:
    def test_slow_drift_keeps_tracking(self):
        """Дрейф ~5 центов/кадр: persistence копится, событие приходит."""
        det = _detector()
        event = None
        freq = 3150.0
        amp = 0.05
        for i in range(30):
            freq *= 2 ** (5.0 / 1200.0)  # +5 центов
            amp = min(0.6, amp * 1.25)   # растущая
            event = event or det.process_audio(1, _tone(freq, FS, N, amp))
        assert event is not None, (
            "дрейфующая заводка не детектирована — matched-треки снова "
            "получают stale-декремент (ревью FB-D4)"
        )


class TestDbfsUnits:
    def test_full_scale_sine_reads_near_zero_dbfs(self):
        det = _detector()
        det.process_audio(1, _tone(3150.0, FS, N, 1.0))
        state = det._get_or_create_state(1)
        assert state.tracked_peaks, "пик полной шкалы не найден"
        mag = max(p.magnitude_db for p in state.tracked_peaks.values())
        assert abs(mag) < 3.0, (
            f"синус 0 dBFS читается {mag:+.1f} дБ — единицы снова сырые FFT-дБ"
        )


class TestSubharmonicRule:
    """Правило CLAUDE.md: 2-я гармоника ноты сама проходит PHPR (выше неё
    нет 2f/3f) — обязательна проверка суб-гармоник f/2, f/3."""

    def test_note_second_harmonic_not_flagged(self):
        det = _detector()
        event = None
        for i in range(40):
            # нота 500 Гц: фундаментал + сильная растущая 2-я гармоника
            amp = min(0.5, 0.05 * (1.2 ** i))
            block = (_tone(500.0, FS, N, amp)
                     + _tone(1000.0, FS, N, amp * 0.9)
                     + _tone(1500.0, FS, N, amp * 0.3))
            event = event or det.process_audio(1, block)
        assert event is None, (
            f"нота с гармониками флагнута как заводка ({event.frequency_hz if event else 0:.0f} Гц)"
        )

    def test_pure_growing_sine_still_flagged(self):
        """Контроль: чистый растущий синус остаётся заводкой."""
        det = _detector()
        event = None
        for i in range(30):
            amp = min(0.6, 0.02 * (1.3 ** i))
            event = event or det.process_audio(1, _tone(3150.0, FS, N, amp))
        assert event is not None
