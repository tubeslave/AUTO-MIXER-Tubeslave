import numpy as np
import pytest

from auto_mastering import AutoMaster
from studio_mastering_metrics import StudioMasteringMeter


def sine(freq_hz=1000.0, amplitude=1.0, duration=1.0, sample_rate=48000, phase=0.0):
    sample_count = int(sample_rate * duration)
    t = np.arange(sample_count, dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t + phase)).astype(np.float32)


def test_builtin_master_reports_integrated_lufs_from_studio_meter():
    sample_rate = 48000
    master = AutoMaster(sample_rate=sample_rate, target_lufs=-18.0, true_peak_limit=-1.0)
    audio = sine(freq_hz=997.0, amplitude=0.08, duration=1.0, sample_rate=sample_rate)

    result = master.master(audio)
    measured = StudioMasteringMeter(sample_rate).measure(result.audio)

    assert result.success is True
    assert result.lufs == pytest.approx(measured.integrated_lufs, abs=0.02)
    assert result.peak_db == pytest.approx(measured.sample_peak_dbfs, abs=0.02)
    assert measured.true_peak_dbtp <= -0.98


def test_actual_target_path_catches_intersample_peak_missed_by_legacy_limit():
    sample_rate = 48000
    master = AutoMaster(sample_rate=sample_rate, target_lufs=-14.0, true_peak_limit=-1.0)
    meter = StudioMasteringMeter(sample_rate)
    audio = sine(
        freq_hz=12000.0,
        amplitude=0.88,
        duration=0.1,
        sample_rate=sample_rate,
        phase=3.0 * np.pi / 4.0,
    )
    before = meter.measure(audio)

    assert before.sample_peak_dbfs < -1.0
    assert before.true_peak_dbtp > -1.0

    legacy_limited = master._limit(audio)
    np.testing.assert_allclose(legacy_limited, audio, atol=0.0, rtol=0.0)

    conformed = master._match_target_loudness(audio, target_lufs=before.integrated_lufs)
    after = meter.measure(conformed)

    assert after.true_peak_dbtp <= -0.98
    assert np.max(np.abs(conformed)) < np.max(np.abs(audio))


def test_reference_fallback_finishes_true_peak_safe():
    sample_rate = 48000
    master = AutoMaster(sample_rate=sample_rate, target_lufs=-18.0, true_peak_limit=-1.0)
    audio = sine(
        freq_hz=12000.0,
        amplitude=0.88,
        duration=0.5,
        sample_rate=sample_rate,
        phase=3.0 * np.pi / 4.0,
    )
    reference = sine(freq_hz=997.0, amplitude=0.2, duration=0.5, sample_rate=sample_rate)

    result = master._master_fallback(audio, reference, sample_rate)
    measured = StudioMasteringMeter(sample_rate).measure(result)

    assert result.dtype == np.float32
    assert result.shape == audio.shape
    assert measured.true_peak_dbtp <= -0.98
