"""
Analysis helpers for stem-aware AutoFOH feature extraction.

The current soundcheck engine remains channel-centric, so these helpers are
implemented as add-on utilities that can be integrated incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from autofoh_models import (
    AnalysisFeatures,
    FrequencyBand,
    MixIndexSet,
    NAMED_FREQUENCY_BANDS,
    TargetCorridor,
)

EPS = 1e-10


def build_fractional_octave_bands(
    fraction: int = 3,
    start_hz: float = 20.0,
    stop_hz: float = 20000.0,
) -> Sequence[FrequencyBand]:
    """Create fractional-octave bands using 1 kHz as the reference center."""
    if fraction <= 0:
        raise ValueError("fraction must be > 0")

    bands = []
    k = -64
    while k <= 64:
        center = 1000.0 * (2.0 ** (k / fraction))
        low = center / (2.0 ** (1.0 / (2.0 * fraction)))
        high = center * (2.0 ** (1.0 / (2.0 * fraction)))
        if high < start_hz:
            k += 1
            continue
        if low > stop_hz:
            break
        label = f"{center:.1f}Hz" if center < 1000.0 else f"{center / 1000.0:.2f}kHz"
        bands.append(FrequencyBand(label, max(low, start_hz), min(high, stop_hz)))
        k += 1
    return bands


def apply_slope_compensation(
    frequencies_hz: np.ndarray,
    magnitude_db: np.ndarray,
    slope_db_per_octave: float = 4.5,
    reference_hz: float = 100.0,
) -> np.ndarray:
    """Flatten an approximate -X dB/octave tilt by adding compensation."""
    freqs = np.maximum(np.asarray(frequencies_hz, dtype=np.float64), EPS)
    mags = np.asarray(magnitude_db, dtype=np.float64)
    compensation = slope_db_per_octave * np.log2(freqs / max(reference_hz, EPS))
    return mags + compensation


def _band_levels_from_power(
    frequencies_hz: np.ndarray,
    power_spectrum: np.ndarray,
    bands: Iterable[FrequencyBand],
) -> Dict[str, float]:
    levels: Dict[str, float] = {}
    for band in bands:
        mask = (frequencies_hz >= band.low_hz) & (frequencies_hz < band.high_hz)
        if not np.any(mask):
            levels[band.name] = -100.0
            continue
        band_power = float(np.sum(power_spectrum[mask]))
        levels[band.name] = 10.0 * np.log10(band_power + EPS)
    return levels


def calculate_mix_indexes(
    named_band_levels_db: Mapping[str, float],
    target_corridor: Optional[TargetCorridor] = None,
) -> MixIndexSet:
    """Compute named tonal indexes relative to a corridor or reference region."""
    target_corridor = target_corridor or TargetCorridor.default_intergenre()

    def db(name: str) -> float:
        return float(named_band_levels_db.get(name, -100.0))

    available_levels = [
        db(band.name)
        for band in NAMED_FREQUENCY_BANDS
        if band.name in named_band_levels_db
    ]
    spectral_center_db = float(np.mean(available_levels)) if available_levels else 0.0

    def centered(name: str) -> float:
        return db(name) - spectral_center_db

    def delta(name: str) -> float:
        return centered(name) - target_corridor.target_for_band(name)

    reference_body = np.mean([centered("BODY"), centered("MUD")])
    reference_low_mid = np.mean([centered("MUD"), centered("LOW_MID")])

    return MixIndexSet(
        sub_index=centered("SUB") - reference_body - target_corridor.target_for_band("SUB"),
        bass_index=centered("BASS") - reference_low_mid - target_corridor.target_for_band("BASS"),
        body_index=delta("BODY"),
        mud_index=delta("MUD"),
        presence_index=delta("PRESENCE"),
        harshness_index=delta("HARSHNESS"),
        sibilance_index=delta("SIBILANCE"),
        air_index=delta("AIR"),
    )


def extract_analysis_features(
    samples: np.ndarray,
    sample_rate: int = 48000,
    fft_size: int = 4096,
    octave_fraction: int = 3,
    slope_compensation_db_per_octave: float = 4.5,
    target_corridor: Optional[TargetCorridor] = None,
) -> AnalysisFeatures:
    """Extract the first AutoFOH feature slice from a block of audio."""
    data = np.asarray(samples, dtype=np.float32).flatten()
    if len(data) == 0:
        return AnalysisFeatures(confidence=0.0)

    if len(data) < fft_size:
        data = np.pad(data, (0, fft_size - len(data)))
    else:
        data = data[-fft_size:]

    window = np.hanning(fft_size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(data * window)) + EPS
    power_spectrum = spectrum ** 2
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)

    named_levels = _band_levels_from_power(freqs, power_spectrum, NAMED_FREQUENCY_BANDS)
    octave_levels = _band_levels_from_power(
        freqs,
        power_spectrum,
        build_fractional_octave_bands(fraction=octave_fraction),
    )

    compensated_levels = {}
    for band in NAMED_FREQUENCY_BANDS:
        level = named_levels.get(band.name, -100.0)
        compensation = slope_compensation_db_per_octave * np.log2(
            band.center_hz / 100.0
        )
        compensated_levels[band.name] = level + compensation

    rms = float(np.sqrt(np.mean(np.square(data)) + EPS))
    peak = float(np.max(np.abs(data)) + EPS)
    rms_db = 20.0 * np.log10(rms)
    peak_db = 20.0 * np.log10(peak)

    return AnalysisFeatures(
        rms_db=rms_db,
        peak_db=peak_db,
        crest_factor_db=peak_db - rms_db,
        named_band_levels_db=named_levels,
        octave_band_levels_db=octave_levels,
        slope_compensated_band_levels_db=compensated_levels,
        mix_indexes=calculate_mix_indexes(compensated_levels, target_corridor=target_corridor),
        confidence=1.0,
    )


@dataclass
class StemContributionMatrix:
    band_contributions: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def dominant_stem(self, band_name: str) -> Optional[str]:
        row = self.band_contributions.get(band_name, {})
        if not row:
            return None
        return max(row.items(), key=lambda item: item[1])[0]

    def contribution(self, band_name: str, stem_name: str) -> float:
        return float(self.band_contributions.get(band_name, {}).get(stem_name, 0.0))


def build_stem_contribution_matrix(
    stem_features: Mapping[str, AnalysisFeatures],
) -> StemContributionMatrix:
    """Estimate per-band stem contribution from per-stem named band levels."""
    contributions: Dict[str, Dict[str, float]] = {}
    for band in NAMED_FREQUENCY_BANDS:
        band_name = band.name
        stem_powers: Dict[str, float] = {}
        for stem_name, features in stem_features.items():
            level_db = features.named_band_levels_db.get(band_name, -100.0)
            stem_powers[stem_name] = 10.0 ** (level_db / 10.0)
        total_power = sum(stem_powers.values())
        if total_power <= EPS:
            contributions[band_name] = {stem_name: 0.0 for stem_name in stem_powers}
            continue
        contributions[band_name] = {
            stem_name: power / total_power
            for stem_name, power in stem_powers.items()
        }
    return StemContributionMatrix(band_contributions=contributions)
