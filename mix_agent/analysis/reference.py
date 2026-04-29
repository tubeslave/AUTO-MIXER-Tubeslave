"""Reference-track comparison metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np

from mix_agent.models import ReferenceComparison

from .loudness import to_mono
from .spectral import BANDS, band_powers
from .stereo import compute_stereo_metrics


def _rms_normalize(audio: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    return np.asarray(audio, dtype=np.float32) / max(rms, 1e-12) * 0.1


def compare_reference(
    mix_audio: np.ndarray,
    reference_audio: np.ndarray | None,
    sample_rate: int,
) -> ReferenceComparison:
    """Compare mix to reference after analysis-only loudness normalization."""
    if reference_audio is None:
        return ReferenceComparison(enabled=False, limitations=["No reference track supplied."])

    mix = _rms_normalize(mix_audio)
    ref = _rms_normalize(reference_audio)
    mix_powers = band_powers(mix, sample_rate)
    ref_powers = band_powers(ref, sample_rate)
    diffs: Dict[str, float] = {}
    distance_terms = []
    for band in BANDS:
        mix_db = 10.0 * np.log10(mix_powers[band] + 1e-18)
        ref_db = 10.0 * np.log10(ref_powers[band] + 1e-18)
        diff = float(mix_db - ref_db)
        diffs[band] = round(diff, 3)
        distance_terms.append(min(abs(diff) / 12.0, 1.0))

    mix_stereo = compute_stereo_metrics(mix, sample_rate)
    ref_stereo = compute_stereo_metrics(ref, sample_rate)
    mix_active = to_mono(mix)
    ref_active = to_mono(ref)
    mix_dyn = 20.0 * np.log10(
        (np.percentile(np.abs(mix_active), 95) + 1e-12)
        / (np.percentile(np.abs(mix_active), 50) + 1e-12)
    )
    ref_dyn = 20.0 * np.log10(
        (np.percentile(np.abs(ref_active), 95) + 1e-12)
        / (np.percentile(np.abs(ref_active), 50) + 1e-12)
    )
    return ReferenceComparison(
        enabled=True,
        spectral_distance=round(float(np.mean(distance_terms)), 6),
        band_differences_db=diffs,
        stereo_width_difference=round(
            float(mix_stereo.get("stereo_width", 0.0) - ref_stereo.get("stereo_width", 0.0)),
            6,
        ),
        dynamic_range_difference_db=round(float(mix_dyn - ref_dyn), 3),
        loudness_matched=True,
        limitations=[],
    )
