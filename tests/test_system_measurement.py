"""Tests for backend/system_measurement.py."""

import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from observation_mixer import ObservationMixerClient
from system_measurement import SystemMeasurementController, TargetBus


class FakeMixer:
    def __init__(self):
        self.is_connected = True
        self.state = {}

    def send(self, address, *args):
        if not args:
            return {"query": address}
        raise AssertionError("Writes should be intercepted by ObservationMixerClient")


def test_transfer_function_analysis_produces_real_coherence():
    controller = SystemMeasurementController(sample_rate=48000)

    duration = 4.0
    sample_rate = 48000
    samples = int(duration * sample_rate)
    rng = np.random.default_rng(42)
    reference = rng.normal(0.0, 0.25, samples).astype(np.float32)

    kernel = np.array([1.0, 0.55, 0.25], dtype=np.float32)
    measured = np.convolve(reference, kernel, mode='same')
    measured *= 0.8
    measured += rng.normal(0.0, 0.005, samples).astype(np.float32)

    result = controller.analyze_reference_measurement(reference, measured)

    assert result["num_corrections"] >= 0
    assert len(result["frequencies"]) == len(result["magnitude"]) == len(result["coherence"])
    assert max(result["coherence"]) <= 1.0
    assert min(result["coherence"]) >= 0.0
    assert controller.last_result is not None
    assert controller.last_result.overall_quality > 0.2


def test_apply_corrections_targets_main_and_matrix_eq():
    mixer = ObservationMixerClient(FakeMixer())
    controller = SystemMeasurementController(mixer_client=mixer, sample_rate=48000)
    controller.applied_corrections = [
        type("Correction", (), {"frequency": 125.0, "gain_db": -3.0, "q": 1.4})(),
        type("Correction", (), {"frequency": 2500.0, "gain_db": 2.5, "q": 1.8})(),
    ]

    assert controller.apply_corrections(TargetBus.MASTER, 1) is True
    assert mixer.state["/main/1/eq/on"] == 1
    assert mixer.state["/main/1/eq/1f"] == 125.0
    assert mixer.state["/main/1/eq/2g"] == 2.5

    assert controller.apply_corrections(TargetBus.MATRIX, 3) is True
    assert mixer.state["/mtx/3/eq/on"] == 1
    assert mixer.state["/mtx/3/eq/1q"] == 1.4

    assert controller.reset_corrections(TargetBus.MATRIX, 3) is True
    assert mixer.state["/mtx/3/eq/on"] == 0


def test_pink_noise_reference_curve_builds_safe_master_eq():
    controller = SystemMeasurementController(sample_rate=48000)

    sample_rate = 48000
    duration = 8.0
    samples = int(sample_rate * duration)
    rng = np.random.default_rng(7)
    white = rng.normal(0.0, 1.0, samples)
    freqs = np.fft.rfftfreq(samples, 1.0 / sample_rate)
    pink_scale = np.ones_like(freqs)
    pink_scale[1:] = 1.0 / np.sqrt(freqs[1:])
    reference = np.fft.irfft(np.fft.rfft(white) * pink_scale, n=samples).astype(np.float32)
    reference *= 0.25 / (np.max(np.abs(reference)) + 1e-12)

    log_freqs = np.log2(np.maximum(freqs, 1.0))
    low_bump = 7.0 * np.exp(-0.5 * ((log_freqs - math.log2(125.0)) / 0.28) ** 2)
    high_dip = -5.0 * np.exp(-0.5 * ((log_freqs - math.log2(8000.0)) / 0.38) ** 2)
    response_db = low_bump + high_dip
    measured = np.fft.irfft(
        np.fft.rfft(reference) * (10.0 ** (response_db / 20.0)),
        n=samples,
    ).astype(np.float32)
    measured += rng.normal(0.0, 0.0005, samples).astype(np.float32)

    result = controller.analyze_reference_measurement(
        reference,
        measured,
        correction_mode="pink_noise_reference",
        reference_curve="pink_noise_live_pa",
    )

    corrections = result["corrections"]
    low = min(corrections, key=lambda c: abs(c["frequency"] - 125.0))
    high = min(corrections, key=lambda c: abs(c["frequency"] - 8000.0))

    assert result["correction_mode"] == "pink_noise_reference"
    assert result["reference_curve"] == "pink_noise_live_pa"
    assert result["quality"] > 0.3
    assert low["frequency"] == 125.0
    assert low["gain_db"] < -2.0
    assert high["frequency"] == 8000.0
    assert high["gain_db"] > 0.5
    assert all(-6.0 <= correction["gain_db"] <= 3.0 for correction in corrections)
