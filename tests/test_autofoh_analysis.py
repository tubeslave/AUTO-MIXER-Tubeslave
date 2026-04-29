"""Tests for the first AutoFOH analysis utilities."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from autofoh_analysis import (
    apply_slope_compensation,
    build_fractional_octave_bands,
    build_stem_contribution_matrix,
    extract_analysis_features,
)


def _sine(frequency_hz: float, amplitude: float = 1.0, sample_rate: int = 48000, duration_sec: float = 0.25):
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * t)


def test_named_band_energy_detects_sine_wave_in_expected_band():
    features = extract_analysis_features(_sine(45.0))
    dominant_band = max(
        features.named_band_levels_db,
        key=features.named_band_levels_db.get,
    )
    assert dominant_band == "SUB"


def test_fractional_octave_analysis_places_1khz_tone_in_correct_band():
    bands = build_fractional_octave_bands(fraction=3)
    features = extract_analysis_features(_sine(1000.0), octave_fraction=3)
    dominant_band = max(bands, key=lambda band: features.octave_band_levels_db[band.name])

    assert dominant_band.low_hz <= 1000.0 <= dominant_band.high_hz


def test_slope_compensation_flattens_nominal_minus_4_5_db_per_octave_curve():
    freqs = np.array([100.0, 200.0, 400.0], dtype=np.float64)
    magnitudes_db = np.array([0.0, -4.5, -9.0], dtype=np.float64)
    compensated = apply_slope_compensation(freqs, magnitudes_db, slope_db_per_octave=4.5)

    assert np.allclose(compensated, compensated[0], atol=0.2)


def test_stem_contribution_matrix_identifies_dominant_stem_per_band():
    guitars = extract_analysis_features(_sine(320.0, amplitude=0.8))
    cymbals = extract_analysis_features(_sine(5000.0, amplitude=0.8))
    lead = extract_analysis_features(_sine(2200.0, amplitude=0.2))

    matrix = build_stem_contribution_matrix(
        {
            "GUITARS": guitars,
            "CYMBALS": cymbals,
            "LEAD": lead,
        }
    )

    assert matrix.dominant_stem("MUD") == "GUITARS"
    assert matrix.dominant_stem("HARSHNESS") == "CYMBALS"
