"""Regression tests for the shared offline/reference DSP analysis core."""

import numpy as np

from mix_agent.analysis.dsp_utils import sample_major, to_mono
from mix_agent.analysis.loudness import compute_level_metrics
from mix_agent.analysis.spectral import compute_spectral_metrics
from mix_agent.analysis.stereo import compute_stereo_metrics

SR = 48000


def _sine(freq: float, amp: float = 0.25, seconds: float = 1.0, phase: float = 0.0):
    t = np.arange(int(SR * seconds), dtype=np.float64) / SR
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def test_channel_first_audio_is_normalized_to_sample_major():
    stereo_cf = np.vstack([_sine(440, seconds=0.2), _sine(880, seconds=0.2)])
    data = sample_major(stereo_cf)
    assert data.shape == (int(0.2 * SR), 2)
    assert to_mono(stereo_cf).shape == (int(0.2 * SR),)


def test_integrated_loudness_tracks_known_gain_change():
    base = _sine(1000.0, amp=0.1, seconds=4.0)
    louder = base * np.float32(2.0)
    m1, limits1 = compute_level_metrics(base, SR)
    m2, limits2 = compute_level_metrics(louder, SR)
    delta = m2["integrated_lufs"] - m1["integrated_lufs"]
    assert 5.8 <= delta <= 6.2
    assert np.isfinite(m1["true_peak_dbtp"])
    assert np.isfinite(m2["true_peak_dbtp"])
    assert not any("unweighted" in item.lower() for item in limits1 + limits2)


def test_level_metrics_are_finite_on_nonfinite_input_and_silence():
    audio = np.zeros(SR, dtype=np.float32)
    audio[10] = np.nan
    audio[20] = np.inf
    audio[30] = -np.inf
    metrics, _ = compute_level_metrics(audio, SR)
    for key in ("peak_dbfs", "true_peak_dbtp", "rms_dbfs", "integrated_lufs", "dc_offset"):
        assert np.isfinite(float(metrics[key]))
    assert metrics["clip_count"] == 0


def test_spectral_metrics_use_late_material_not_only_first_frame():
    first = np.concatenate([_sine(500.0, seconds=0.5), _sine(120.0, seconds=1.5)])
    second = np.concatenate([_sine(500.0, seconds=0.5), _sine(8000.0, seconds=1.5)])
    a = compute_spectral_metrics(first, SR)
    b = compute_spectral_metrics(second, SR)
    assert a["spectral_centroid_hz"] < b["spectral_centroid_hz"]
    assert b["brightness_proxy"] > a["brightness_proxy"] + 0.25
    assert a["boominess_proxy"] > b["boominess_proxy"] + 0.25


def test_spectral_silence_has_zero_flatness_and_finite_metrics():
    metrics = compute_spectral_metrics(np.zeros(SR, dtype=np.float32), SR)
    assert metrics["spectral_flatness"] == 0.0
    assert metrics["spectral_centroid_hz"] == 0.0
    assert all(np.isfinite(float(v)) for v in metrics["band_energy_db"].values())


def test_stereo_frequency_width_uses_late_material():
    # First 0.5 s is fully mono.  The following 1.5 s is wide at 8 kHz.
    prefix = _sine(500.0, seconds=0.5)
    wide = _sine(8000.0, seconds=1.5)
    left = np.concatenate([prefix, wide])
    right = np.concatenate([prefix, -wide])
    stereo = np.column_stack([left, right])
    metrics = compute_stereo_metrics(stereo, SR)
    assert metrics["is_stereo"] is True
    assert metrics["frequency_dependent_width"]["air"] > 0.8
    assert metrics["phase_cancellation_risk"] is True


def test_identical_stereo_is_narrow_and_mono_compatible():
    tone = _sine(1000.0, seconds=1.0)
    metrics = compute_stereo_metrics(np.column_stack([tone, tone]), SR)
    assert metrics["inter_channel_correlation"] > 0.999
    assert metrics["stereo_width"] < 1e-5
    assert metrics["mono_fold_down_loss_db"] < 0.01
    assert metrics["phase_cancellation_risk"] is False
