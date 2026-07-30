"""Синтетические тесты санитарного слоя входа (Ф8, DEC-0008)."""

import numpy as np
import pytest

from input_health import (
    ChannelHealth,
    analyze_channel,
    detect_clicks,
    detect_clipping,
    detect_dropouts,
    repair_clicks,
    robust_peak,
    synchronized_click_groups,
)

SR = 48000


def sine(freq=440.0, seconds=1.0, amp=0.3, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


class TestClicks:
    def test_clean_sine_has_no_clicks(self):
        assert detect_clicks(sine(), SR) == []

    def test_isolated_full_scale_click_detected(self):
        x = sine(amp=0.2)
        x[24000:24003] = 1.0
        events = detect_clicks(x, SR)
        assert len(events) == 1
        assert events[0].start == 24000
        assert events[0].peak_abs == pytest.approx(1.0)
        assert events[0].duration_ms(SR) < 1.0

    def test_loud_musical_peak_is_not_a_click(self):
        # громкий, но длинный всплеск (10 мс) — музыкальный пик, не клик
        x = sine(amp=0.2)
        burst = np.hanning(480) * 0.99
        x[24000:24480] = burst
        assert detect_clicks(x, SR) == []

    def test_click_in_loud_context_not_flagged(self):
        # выброс на фоне почти столь же громкого окружения — не изолирован
        x = sine(amp=0.9)
        x[24000] = 0.99
        assert detect_clicks(x, SR) == []

    def test_repair_removes_click_and_preserves_signal(self):
        x = sine(amp=0.2)
        x[24000:24002] = 1.0
        events = detect_clicks(x, SR)
        y = repair_clicks(x, events, SR)
        assert np.abs(y).max() < 0.5
        # вне зоны ремонта сигнал нетронут
        assert np.array_equal(y[:23800], x[:23800])
        assert np.array_equal(y[24500:], x[24500:])

    def test_robust_peak_ignores_click(self):
        x = sine(amp=0.25)
        x[10000] = 1.0
        assert robust_peak(x, SR) == pytest.approx(0.25, abs=0.02)

    def test_robust_peak_without_clicks_is_true_peak(self):
        x = sine(amp=0.4)
        assert robust_peak(x, SR) == pytest.approx(0.4, abs=0.01)


class TestClipping:
    def test_clean_signal_no_clipping(self):
        assert detect_clipping(sine(amp=0.9)) == (0, 0)

    def test_flat_ceiling_run_detected(self):
        x = sine(amp=0.5)
        x[1000:1200] = 1.0  # плато 200 сэмплов
        runs, samples = detect_clipping(x)
        assert runs == 1
        assert samples == 200

    def test_short_spike_is_not_clipping(self):
        x = sine(amp=0.5)
        x[1000:1002] = 1.0  # короче CLIP_MIN_RUN
        assert detect_clipping(x) == (0, 0)


class TestDropouts:
    def test_clean_signal_no_dropouts(self):
        assert detect_dropouts(sine(), SR) == 0

    def test_zero_run_inside_signal_detected(self):
        x = sine(amp=0.3)
        x[24000:24480] = 0.0  # 10 мс нулей внутри сигнала
        assert detect_dropouts(x, SR) == 1

    def test_silence_is_not_dropout(self):
        assert detect_dropouts(np.zeros(SR), SR) == 0


class TestChannelHealth:
    def test_healthy_channel(self):
        h = analyze_channel(sine(), SR)
        assert h.healthy and not h.silent and h.reasons == []

    def test_silent_channel(self):
        h = analyze_channel(np.zeros(SR) + 1e-6, SR)
        assert h.silent and "silent" in h.reasons

    def test_defective_channel_reports_reasons(self):
        x = sine(amp=0.3)
        x[10000] = 1.0
        x[20000:20480] = 0.0
        h = analyze_channel(x, SR)
        assert not h.healthy
        assert any(r.startswith("clicks") for r in h.reasons)
        assert any(r.startswith("dropout") for r in h.reasons)


class TestSynchronizedClicks:
    def test_recording_glitch_across_channels(self):
        healths = []
        for _ in range(4):
            x = sine(amp=0.2)
            x[24000:24002] = 1.0  # один и тот же момент во всех каналах
            healths.append(analyze_channel(x, SR))
        groups = synchronized_click_groups(healths, SR)
        assert len(groups) == 1
        assert len(groups[0]) == 4

    def test_unrelated_clicks_not_grouped(self):
        healths = []
        for i in range(3):
            x = sine(amp=0.2)
            x[5000 + i * 15000] = 1.0  # разнесены по времени
            healths.append(analyze_channel(x, SR))
        assert synchronized_click_groups(healths, SR) == []


class TestRealFixtureRegression:
    """Регрессия инцидента 2026-07-29: SONG REPA с глитчами записи."""

    def test_song_repa_glitches_detected(self):
        sf = pytest.importorskip("soundfile")
        from pathlib import Path
        src = Path(__file__).resolve().parents[1] / "offline_test_input" / "multitrack"
        if not src.is_dir():
            pytest.skip("multitrack fixture not present")
        healths = []
        for f in sorted(src.glob("*.wav"))[:6]:
            x, sr = sf.read(str(f))
            healths.append(analyze_channel(x, sr))
        groups = synchronized_click_groups(healths, 48000)
        # известные глитчи на ~1.06с и ~66.8с синхронны по каналам
        assert len(groups) >= 1
