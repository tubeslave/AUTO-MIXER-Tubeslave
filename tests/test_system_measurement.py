"""Tests for backend/system_measurement.py."""

import os
import sys

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
