import numpy as np
import pytest

from studio_mastering_metrics import StudioMasteringMeter


def sine(freq_hz=1000.0, amplitude=1.0, duration=1.0, sample_rate=48000, phase=0.0):
    t = np.arange(int(sr := sample_rate * duration), dtype=np.float64) / sample_rate
    assert len(t) == int(sr)
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t + phase)).astype(np.float32)


def test_integrated_lufs_matches_full_scale_1khz_sine():
    meter = StudioMasteringMeter(48000)
    measured = meter.integrated_lufs(sine())
    assert measured == pytest.approx(-3.05, abs=0.35)


def test_integrated_lufs_matches_project_997hz_eval():
    meter = StudioMasteringMeter(48000)
    amplitude = 10.0 ** (-20.0 / 20.0)
    audio = sine(freq_hz=997.0, amplitude=amplitude, duration=3.0)
    measured = meter.integrated_lufs(audio)
    assert measured == pytest.approx(-23.0, abs=0.1)


def test_stereo_energy_is_about_three_lu_above_identical_mono():
    meter = StudioMasteringMeter(48000)
    mono = sine(amplitude=0.5)
    stereo = np.column_stack([mono, mono])
    delta = meter.integrated_lufs(stereo) - meter.integrated_lufs(mono)
    assert delta == pytest.approx(3.0103, abs=0.15)


def test_silence_returns_floor_values():
    meter = StudioMasteringMeter(48000)
    result = meter.measure(np.zeros(48000, dtype=np.float32))
    assert result.integrated_lufs <= -90.0
    assert result.true_peak_dbtp <= -90.0
    assert result.sample_peak_dbfs <= -90.0


def test_true_peak_matches_project_11025hz_eval():
    meter = StudioMasteringMeter(48000)
    expected_dbtp = -6.0
    amplitude = 10.0 ** (expected_dbtp / 20.0)
    audio = sine(freq_hz=11025.0, amplitude=amplitude, duration=1.0)
    measured = meter.true_peak_dbtp(audio)
    assert measured == pytest.approx(expected_dbtp, abs=0.3)


def test_true_peak_can_exceed_sample_peak():
    meter = StudioMasteringMeter(48000)
    audio = sine(
        freq_hz=12000.0,
        amplitude=0.88,
        duration=0.1,
        phase=3.0 * np.pi / 4.0,
    )
    result = meter.measure(audio)
    assert result.true_peak_dbtp > result.sample_peak_dbfs + 2.0


def test_true_peak_limiter_catches_intersample_overshoot():
    meter = StudioMasteringMeter(48000)
    audio = sine(
        freq_hz=12000.0,
        amplitude=0.88,
        duration=0.1,
        phase=3.0 * np.pi / 4.0,
    )
    before = meter.measure(audio)
    assert before.sample_peak_dbfs < -1.0
    assert before.true_peak_dbtp > -1.0

    limited, reduction_db = meter.limit_true_peak(audio, ceiling_dbtp=-1.0)
    after = meter.measure(limited)

    assert reduction_db > 0.0
    assert after.true_peak_dbtp <= -0.98
    assert np.max(np.abs(limited)) < np.max(np.abs(audio))


def test_true_peak_limiter_never_boosts_safe_audio():
    meter = StudioMasteringMeter(48000)
    audio = sine(amplitude=0.2)
    limited, reduction_db = meter.limit_true_peak(audio, ceiling_dbtp=-1.0)
    assert reduction_db == 0.0
    np.testing.assert_allclose(limited, audio, atol=0.0, rtol=0.0)


def test_true_peak_limiter_preserves_channels_first_layout():
    meter = StudioMasteringMeter(48000)
    mono = sine(
        freq_hz=12000.0,
        amplitude=0.88,
        duration=0.1,
        phase=3.0 * np.pi / 4.0,
    )
    channels_first = np.stack([mono, mono * 0.8], axis=0)
    limited, reduction_db = meter.limit_true_peak(channels_first, ceiling_dbtp=-1.0)
    assert reduction_db > 0.0
    assert limited.shape == channels_first.shape
    assert meter.true_peak_dbtp(limited) <= -0.98
