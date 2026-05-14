"""Direct tests for SignalAnalyzer and the shared rolling LUFS helper."""

import numpy as np
import pytest

from auto_fader_v2.core.integrated_lufs import RollingIntegratedLufs
from signal_metrics import SignalAnalyzer


def _process_in_blocks(analyzer: SignalAnalyzer, samples: np.ndarray, block_size: int) -> None:
    """Feed a signal into the analyzer using realistic audio blocks."""
    for start in range(0, len(samples), block_size):
        analyzer.process(samples[start:start + block_size])


def test_signal_analyzer_reports_expected_level_and_spectral_metrics(sample_rate, block_size):
    analyzer = SignalAnalyzer(channel=7, sample_rate=sample_rate, block_size=block_size)
    duration_sec = 4.0
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / sample_rate
    tone = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)

    _process_in_blocks(analyzer, tone, block_size)
    metrics = analyzer.get_metrics()

    assert metrics.channel == 7
    assert metrics.level.peak_db == pytest.approx(-6.02, abs=0.5)
    assert metrics.level.rms_db == pytest.approx(-9.03, abs=0.5)
    assert metrics.level.lufs_momentary == pytest.approx(metrics.level.lufs_short_term, abs=0.5)
    assert metrics.level.lufs_integrated == pytest.approx(metrics.level.lufs_short_term, abs=0.5)
    assert metrics.level.true_peak_dbtp >= metrics.level.peak_db
    assert metrics.spectral.centroid_hz > 500.0
    assert metrics.spectral.rolloff_hz > 900.0
    assert metrics.spectral.flux >= 0.0
    assert set(metrics.spectral.band_energy) >= {
        "sub",
        "bass",
        "low_mid",
        "mid",
        "high_mid",
        "presence",
        "air",
    }


def test_signal_analyzer_detects_transients_and_dynamic_range(sample_rate, block_size):
    analyzer = SignalAnalyzer(channel=11, sample_rate=sample_rate, block_size=block_size)
    tone_t = np.arange(block_size, dtype=np.float32) / sample_rate

    blocks = [np.zeros(block_size, dtype=np.float32) for _ in range(8)]
    for amplitude in (0.1, 0.4, 0.8, 0.4, 0.1):
        blocks.append((amplitude * np.sin(2 * np.pi * 1000.0 * tone_t)).astype(np.float32))
    blocks.extend(np.zeros(block_size, dtype=np.float32) for _ in range(10))

    for block in blocks:
        analyzer.process(block)

    metrics = analyzer.get_metrics()

    assert metrics.dynamics.dynamic_range_db > 10.0
    assert metrics.dynamics.attack_time_ms > 0.0
    assert metrics.dynamics.decay_time_ms > 0.0
    assert metrics.dynamics.envelope_variance > 0.0
    assert metrics.dynamics.transient_density > 0.0
    assert metrics.dynamics.transient_strength_db > 3.0


def test_signal_analyzer_silence_stays_gated_and_reports_no_spectral_state(block_size):
    analyzer = SignalAnalyzer(channel=1, sample_rate=48_000, block_size=block_size)

    for _ in range(12):
        analyzer.process(np.zeros(block_size, dtype=np.float32))

    metrics = analyzer.get_metrics()

    assert metrics.level.lufs_integrated == -100.0
    assert metrics.level.lufs_short_term == -100.0
    assert metrics.level.true_peak_dbtp == -100.0
    assert metrics.spectral.centroid_hz == 0.0
    assert metrics.spectral.band_energy == {}
    assert metrics.dynamics.dynamic_range_db == 0.0


def test_signal_analyzer_low_signal_stays_below_integrated_gate(sample_rate, block_size):
    analyzer = SignalAnalyzer(channel=2, sample_rate=sample_rate, block_size=block_size)
    duration_sec = 2.0
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / sample_rate
    quiet_tone = (1e-4 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)

    _process_in_blocks(analyzer, quiet_tone, block_size)
    metrics = analyzer.get_metrics()

    assert metrics.level.peak_db == pytest.approx(-80.0, abs=1.0)
    assert metrics.level.true_peak_dbtp == pytest.approx(-80.0, abs=1.5)
    assert metrics.level.lufs_integrated == -100.0
    assert metrics.level.lufs_short_term < -70.0
    assert metrics.spectral.band_energy == {}


def test_signal_analyzer_trailing_silence_preserves_last_meaningful_spectrum(sample_rate, block_size):
    analyzer = SignalAnalyzer(channel=5, sample_rate=sample_rate, block_size=block_size)
    tone_t = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    tone = (0.5 * np.sin(2 * np.pi * 1000.0 * tone_t)).astype(np.float32)
    trailing_silence = np.zeros(sample_rate * 2, dtype=np.float32)

    _process_in_blocks(analyzer, np.concatenate([tone, trailing_silence]), block_size)
    metrics = analyzer.get_metrics()

    assert metrics.level.lufs_integrated > -15.0
    assert metrics.level.lufs_momentary < -50.0
    assert metrics.spectral.centroid_hz > 500.0
    assert metrics.spectral.band_energy["mid"] > 0.0


def test_signal_analyzer_spectral_ratios_distinguish_low_and_high_frequency_content(sample_rate, block_size):
    low = SignalAnalyzer(channel=21, sample_rate=sample_rate, block_size=block_size)
    high = SignalAnalyzer(channel=22, sample_rate=sample_rate, block_size=block_size)
    duration_sec = 2.0
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / sample_rate

    _process_in_blocks(low, (0.5 * np.sin(2 * np.pi * 80.0 * t)).astype(np.float32), block_size)
    _process_in_blocks(high, (0.5 * np.sin(2 * np.pi * 6000.0 * t)).astype(np.float32), block_size)

    low_metrics = low.get_metrics()
    high_metrics = high.get_metrics()

    assert high_metrics.spectral.centroid_hz > low_metrics.spectral.centroid_hz
    assert high_metrics.spectral.brightness > low_metrics.spectral.brightness
    assert low_metrics.spectral.warmth > high_metrics.spectral.warmth
    assert high_metrics.spectral.band_energy["presence"] > low_metrics.spectral.band_energy["presence"]


def test_signal_analyzer_reset_clears_accumulated_state(sample_rate, block_size):
    analyzer = SignalAnalyzer(channel=3, sample_rate=sample_rate, block_size=block_size)
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    tone = (0.25 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    _process_in_blocks(analyzer, tone, block_size)
    analyzer.reset()
    metrics = analyzer.get_metrics()

    assert metrics.level.peak_db == -100.0
    assert metrics.level.true_peak_dbtp == -100.0
    assert metrics.level.lufs_integrated == -100.0
    assert metrics.level.lufs_short_term == -100.0
    assert metrics.dynamics.dynamic_range_db == 0.0
    assert metrics.spectral.band_energy == {}


def test_rolling_integrated_lufs_uses_relative_gate_to_reject_quiet_outlier():
    meter = RollingIntegratedLufs(window_seconds=3.0)

    result = -100.0
    for idx, lufs in enumerate((-24.0, -24.0, -24.0, -45.0)):
        result = meter.update(channel_id=5, lufs_value=lufs, now=float(idx))

    assert result == pytest.approx(-24.0, abs=0.1)
    assert meter.get(5) == pytest.approx(-24.0, abs=0.1)


def test_rolling_integrated_lufs_prunes_expired_samples_from_window():
    meter = RollingIntegratedLufs(window_seconds=1.0)

    meter.update(channel_id=2, lufs_value=-18.0, now=0.0)
    meter.update(channel_id=2, lufs_value=-30.0, now=0.5)
    latest = meter.update(channel_id=2, lufs_value=-12.0, now=2.0)

    assert latest == pytest.approx(-12.0, abs=0.1)
    assert meter.get(2) == pytest.approx(-12.0, abs=0.1)


def test_rolling_integrated_lufs_returns_fallback_when_all_samples_fail_absolute_gate():
    meter = RollingIntegratedLufs(window_seconds=3.0)

    meter.update(channel_id=9, lufs_value=-80.0, now=0.0)
    meter.update(channel_id=9, lufs_value=-75.0, now=1.0)

    assert meter.get(9) == -100.0
    assert meter.get(channel_id=404, fallback=-55.0) == -55.0
