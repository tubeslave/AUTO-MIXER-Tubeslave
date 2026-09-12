"""Extended synthetic invariants for phase, processor nodes and mastering.

No mixer client or network transport is instantiated by these tests.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from auto_mastering import AutoMaster
from auto_phase_gcc_phat import GCCPHATAnalyzer
from ml.processing_graph import CompressorNode, EQNode, FaderNode, GateNode, HPFNode, PanNode

SR = 48000


def _sine(freq: float, amp: float = 0.25, seconds: float = 1.0, phase: float = 0.0):
    t = np.arange(int(SR * seconds), dtype=np.float64) / SR
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float64)


def _rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(x)) + 1e-15))


def test_gcc_phat_known_delay_has_usable_quality_metrics():
    rng = np.random.default_rng(2026)
    ref = rng.standard_normal(8192).astype(np.float64)
    ref = np.convolve(ref, np.ones(5) / 5.0, mode="same")
    delay = 96
    target = np.zeros_like(ref)
    target[delay:] = ref[:-delay]
    result = GCCPHATAnalyzer(sample_rate=SR, fft_size=4096, max_delay_ms=10).compute_delay(ref, target)
    assert abs(result.delay_samples - delay) < 2.0
    assert result.correlation_peak > 0.5
    assert result.psr > 5.0
    assert 0.0 <= result.coherence <= 1.0
    assert 0.0 <= result.confidence <= 1.0


def test_gcc_phat_uses_representative_material_not_only_first_fft_frame():
    rng = np.random.default_rng(42)
    prefix = rng.standard_normal(4096).astype(np.float64) * 0.001
    body = rng.standard_normal(8192).astype(np.float64)
    delay = 72
    target_body = np.zeros_like(body)
    target_body[delay:] = body[:-delay]
    ref = np.concatenate([prefix, body])
    target = np.concatenate([prefix, target_body])
    result = GCCPHATAnalyzer(sample_rate=SR, fft_size=4096, max_delay_ms=10).compute_delay(ref, target)
    assert abs(abs(result.delay_samples) - delay) < 3.0


def test_hpf_attenuates_sub_cutoff_far_more_than_passband():
    node = HPFNode(cutoff_hz=100.0, order=2)
    low = _sine(25.0, seconds=2.0)
    high = _sine(1000.0, seconds=2.0)
    y_low = node.process(low, SR)
    y_high = node.process(high, SR)
    low_gain_db = 20 * np.log10(_rms(y_low[SR // 2 :]) / _rms(low[SR // 2 :]))
    high_gain_db = 20 * np.log10(_rms(y_high[SR // 2 :]) / _rms(high[SR // 2 :]))
    assert low_gain_db < -20.0
    assert high_gain_db > -1.0


def test_hpf_streaming_is_chunk_boundary_invariant_after_warmup():
    audio = np.ones(SR, dtype=np.float64) * 0.5
    whole = HPFNode(cutoff_hz=80.0, order=2).process(audio, SR)
    node = HPFNode(cutoff_hz=80.0, order=2)
    chunks = [node.process(audio[start:start + 1024], SR) for start in range(0, len(audio), 1024)]
    chunked = np.concatenate(chunks)
    # Stateful stream processing should agree except for numerical epsilon.
    assert _rms(whole - chunked) < 1e-4


def test_parametric_eq_center_gain_is_close_to_requested_gain():
    audio = _sine(1000.0, amp=0.1, seconds=2.0)
    node = EQNode(bands=[{"band_type": "peak", "frequency": 1000.0, "gain_db": 6.0, "q": 1.0}])
    out = node.process(audio, SR)
    gain_db = 20 * np.log10(_rms(out[SR // 2 :]) / _rms(audio[SR // 2 :]))
    assert 5.3 <= gain_db <= 6.7


def test_compressor_reduces_hot_program_and_preserves_finite_output():
    audio = _sine(1000.0, amp=0.8, seconds=2.0)
    node = CompressorNode(threshold_db=-18.0, ratio=4.0, attack_ms=5.0, release_ms=100.0)
    out = node.process(audio, SR)
    assert np.all(np.isfinite(out))
    assert _rms(out[SR:]) < _rms(audio[SR:]) * 0.7


def test_gate_and_fader_do_not_emit_nonfinite_values():
    audio = _sine(440.0, amp=0.05, seconds=0.25)
    audio[10] = np.nan
    gated = GateNode(threshold_db=-40.0).process(audio.copy(), SR)
    faded = FaderNode(gain_db=-6.0).process(np.nan_to_num(gated), SR)
    assert np.all(np.isfinite(faded))


def test_pan_node_uses_constant_power_law():
    audio = _sine(440.0, amp=0.2, seconds=0.5)
    center = PanNode(pan=0.0).process(audio, SR)
    left = PanNode(pan=-1.0).process(audio, SR)
    p_in = float(np.mean(audio ** 2))
    p_center = float(np.mean(center[:, 0] ** 2 + center[:, 1] ** 2))
    p_left = float(np.mean(left[:, 0] ** 2 + left[:, 1] ** 2))
    assert abs(p_center - p_in) / p_in < 0.01
    assert abs(p_left - p_in) / p_in < 0.01


def test_mastering_loudness_estimator_tracks_bs1770_reference():
    pyln = pytest.importorskip("pyloudnorm")
    # Low-frequency programme exposes the difference between plain RMS and K weighting.
    audio = _sine(60.0, amp=0.25, seconds=4.0).astype(np.float32)
    expected = float(pyln.Meter(SR).integrated_loudness(audio))
    measured = float(AutoMaster._estimate_lufs(audio))
    assert abs(measured - expected) < 1.0


def test_master_limiter_controls_reconstructed_true_peak():
    scipy_signal = pytest.importorskip("scipy.signal")
    ceiling_db = -1.0
    ceiling = 10 ** (ceiling_db / 20.0)
    rng = np.random.default_rng(12345)
    candidate = None
    for _ in range(400):
        x = rng.uniform(-0.88, 0.88, 256).astype(np.float32)
        tp = float(np.max(np.abs(scipy_signal.resample_poly(x, 4, 1))))
        if float(np.max(np.abs(x))) < ceiling and tp > ceiling * 1.05:
            candidate = x
            break
    assert candidate is not None, "deterministic search did not create an inter-sample peak fixture"
    limited = AutoMaster(target_lufs=-14.0, true_peak_limit=ceiling_db)._limit(candidate)
    reconstructed = scipy_signal.resample_poly(limited, 4, 1)
    final_tp = float(np.max(np.abs(reconstructed)))
    assert final_tp <= ceiling * 1.01
