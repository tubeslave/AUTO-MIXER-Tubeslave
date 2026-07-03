"""Focused tests for dry-run room-response analysis."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from system_measurement import MeasurementPosition, SystemMeasurement, analyze_room_response


def test_room_response_analysis_classifies_dry_environment():
    sample_rate = 48_000
    ir = _synthetic_ir(sample_rate, rt60_s=0.22, noise_floor_db=-62.0)

    result = analyze_room_response(ir, sample_rate=sample_rate)

    assert result.environment == "dry"
    assert result.avg_rt60_s < 0.6
    assert result.speech_clarity.clarity_score > 0.5
    assert result.confidence > 0.2


def test_room_response_analysis_classifies_reflective_hall():
    sample_rate = 48_000
    ir = _synthetic_ir(
        sample_rate,
        rt60_s=1.85,
        noise_floor_db=-54.0,
        resonance_hz=110.0,
        resonance_gain=0.06,
    )

    result = analyze_room_response(ir, sample_rate=sample_rate)

    assert result.environment == "reflective_hall"
    assert result.avg_rt60_s > 1.0
    assert result.confidence > 0.2
    assert result.speech_clarity.clarity_score < 0.8


def test_room_response_analysis_classifies_noisy_venue_when_noise_dominates():
    sample_rate = 48_000
    ir = _synthetic_ir(sample_rate, rt60_s=0.55, noise_floor_db=-18.0)
    coherence = np.full(16_384, 0.42, dtype=np.float64)

    result = analyze_room_response(
        ir,
        sample_rate=sample_rate,
        coherence=coherence,
        overall_quality=0.38,
    )

    assert result.environment == "noisy_venue"
    assert result.noise_floor_db > -30.0
    assert result.coherence_mean < 0.5


def test_system_measurement_analyze_emits_room_analysis_and_non_stub_coherence():
    sample_rate = 48_000
    measurement = SystemMeasurement(sample_rate=sample_rate, fft_size=16_384)
    ir_a = _synthetic_ir(sample_rate, rt60_s=0.42, noise_floor_db=-50.0, resonance_hz=180.0, resonance_gain=0.03)
    ir_b = _synthetic_ir(sample_rate, rt60_s=0.48, noise_floor_db=-48.0, resonance_hz=200.0, resonance_gain=0.04)

    measurement.recorded_responses = [
        MeasurementPosition(0, 0.0, 0.0, 1.5, mic_response=ir_a, impulse_response=ir_a, quality_score=0.9),
        MeasurementPosition(1, 1.0, 0.5, 1.5, mic_response=ir_b, impulse_response=ir_b, quality_score=0.85),
    ]

    result = measurement.analyze()

    assert result.room_analysis.environment in {
        "dry",
        "conference_room",
        "live_stage",
        "mixed",
    }
    assert len(result.room_analysis.band_decay) == 8
    assert len(result.magnitude_response.coherence) == measurement.fft_size // 2
    assert not np.allclose(result.magnitude_response.coherence, 0.95)


def _synthetic_ir(
    sample_rate: int,
    rt60_s: float,
    noise_floor_db: float,
    resonance_hz: float | None = None,
    resonance_gain: float = 0.0,
) -> np.ndarray:
    duration_s = max(1.25, rt60_s * 2.8)
    num_samples = int(sample_rate * duration_s)
    time = np.arange(num_samples, dtype=np.float64) / sample_rate
    decay = np.exp(-6.907755278982137 * time / max(rt60_s, 0.08))
    rng = np.random.default_rng(12345)

    ir = rng.normal(0.0, 0.08, num_samples) * decay
    ir[0] += 1.0
    for delay_ms, gain in ((18.0, 0.28), (43.0, 0.16), (74.0, 0.09)):
        index = min(num_samples - 1, int(sample_rate * delay_ms / 1000.0))
        ir[index] += gain

    if resonance_hz is not None and resonance_gain > 0.0:
        ir += resonance_gain * np.sin(2.0 * np.pi * resonance_hz * time) * decay

    noise_amp = 10 ** (noise_floor_db / 20.0)
    ir += rng.normal(0.0, noise_amp, num_samples)
    return ir.astype(np.float64)
