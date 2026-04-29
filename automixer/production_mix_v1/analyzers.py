"""Deterministic analyzers for production mix candidates."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .config import ProductionMixConfig
from .models import AudioStem, amp_to_db, ensure_stereo


EPS = 1e-12


def analyze_audio(audio: np.ndarray, sample_rate: int, config: ProductionMixConfig) -> dict[str, Any]:
    """Analyze a stereo audio buffer with dependency-light metrics."""
    stereo = ensure_stereo(audio)
    mono = np.mean(stereo, axis=1) if stereo.size else np.zeros(0, dtype=np.float32)
    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    rms = float(np.sqrt(np.mean(stereo * stereo) + EPS)) if stereo.size else 0.0
    peak_db = amp_to_db(peak)
    rms_db = amp_to_db(rms)
    true_peak_dbfs = _estimate_true_peak_db(stereo)
    crest_factor_db = true_peak_dbfs - rms_db if rms_db > -119.0 else 0.0

    band_energy = _band_energy_db(mono, sample_rate, config.bands)
    band_ratios = _band_energy_ratios(mono, sample_rate, config.bands)
    spectral_centroid = _spectral_centroid_hz(mono, sample_rate)
    phase_correlation = _phase_correlation(stereo)

    dimensions = _dimension_scores(
        band_ratios=band_ratios,
        true_peak_dbfs=true_peak_dbfs,
        crest_factor_db=crest_factor_db,
        phase_correlation=phase_correlation,
    )

    return {
        "sample_rate": int(sample_rate),
        "samples": int(stereo.shape[0]),
        "peak_db": peak_db,
        "true_peak_dbfs": true_peak_dbfs,
        "rms_db": rms_db,
        "lufs_integrated_approx": rms_db - 0.691,
        "crest_factor_db": crest_factor_db,
        "phase_correlation": phase_correlation,
        "spectral_centroid_hz": spectral_centroid,
        "band_energy_db": band_energy,
        "band_ratios": band_ratios,
        "dimensions": dimensions,
    }


def analyze_session(
    stems: list[AudioStem],
    mix_audio: np.ndarray,
    config: ProductionMixConfig,
) -> dict[str, Any]:
    """Analyze mix bus and individual stems."""
    if stems:
        sample_rate = int(stems[0].sample_rate)
    else:
        sample_rate = 48000
    stem_payload = {}
    for stem in stems:
        metrics = analyze_audio(stem.audio, stem.sample_rate, config)
        stem_payload[str(stem.channel_id)] = {
            "name": stem.name,
            "role": stem.role,
            "channel_id": int(stem.channel_id),
            "metrics": metrics,
        }
    return {
        "mix": analyze_audio(mix_audio, sample_rate, config),
        "stems": stem_payload,
    }


def score_dimensions(analysis: Mapping[str, Any]) -> dict[str, float]:
    """Return normalized dimension scores from an analysis payload."""
    dimensions = dict(analysis.get("mix", analysis).get("dimensions", {}) or {})
    return {str(key): float(value) for key, value in dimensions.items()}


def critical_degradations(
    baseline_analysis: Mapping[str, Any],
    candidate_analysis: Mapping[str, Any],
    config: ProductionMixConfig,
) -> list[str]:
    """Detect degradations that should block a candidate."""
    limits = config.safety_limits
    baseline = baseline_analysis.get("mix", baseline_analysis)
    candidate = candidate_analysis.get("mix", candidate_analysis)
    base_dims = score_dimensions(baseline)
    cand_dims = score_dimensions(candidate)
    issues: list[str] = []

    if bool(limits.get("reject_if_vocal_clarity_drops", True)):
        if cand_dims.get("vocal_clarity", 0.0) + 0.03 < base_dims.get("vocal_clarity", 0.0):
            issues.append("vocal_clarity_drop")
    if bool(limits.get("reject_if_low_end_collapses", True)):
        if cand_dims.get("low_end_control", 0.0) + 0.05 < base_dims.get("low_end_control", 0.0):
            issues.append("low_end_control_drop")
    if bool(limits.get("reject_if_harshness_increases_too_much", True)):
        base_harsh = float(baseline.get("band_energy_db", {}).get("harshness", -120.0))
        cand_harsh = float(candidate.get("band_energy_db", {}).get("harshness", -120.0))
        if cand_harsh > base_harsh + 1.5:
            issues.append("harshness_increase")
    return issues


def weighted_rule_score(dimensions: Mapping[str, float], config: ProductionMixConfig) -> float:
    """Combine rule dimensions using pipeline weights."""
    total = 0.0
    weight_sum = 0.0
    for key, weight in config.weights.items():
        total += float(dimensions.get(key, 0.0)) * float(weight)
        weight_sum += float(weight)
    if weight_sum <= 0:
        return 0.0
    return float(total / weight_sum)


def _dimension_scores(
    *,
    band_ratios: Mapping[str, float],
    true_peak_dbfs: float,
    crest_factor_db: float,
    phase_correlation: float,
) -> dict[str, float]:
    mud = float(band_ratios.get("mud", 0.0) + band_ratios.get("boxiness", 0.0))
    intelligibility = float(band_ratios.get("intelligibility", 0.0))
    presence = float(band_ratios.get("presence", 0.0))
    harshness = float(band_ratios.get("harshness", 0.0))
    sub = float(band_ratios.get("sub", 0.0))
    punch = float(band_ratios.get("punch", 0.0))
    air = float(band_ratios.get("air", 0.0) + band_ratios.get("polish", 0.0))

    vocal_clarity = _clamp01(0.45 + 2.2 * (intelligibility + presence) - 1.6 * mud - 1.1 * harshness)
    spectral_balance = _clamp01(1.0 - abs(mud - 0.16) * 2.0 - abs(harshness - 0.10) * 1.5)
    dynamics_control = _clamp01(1.0 - abs(crest_factor_db - 12.0) / 18.0)
    low_end_control = _clamp01(1.0 - abs((sub + punch) - 0.24) * 2.4)
    translation = _clamp01(0.65 * max(0.0, phase_correlation) + 0.35 * (1.0 - harshness))
    safety_margin = _clamp01((-1.0 - true_peak_dbfs) / 12.0 + 0.5)
    musical_quality = _clamp01(
        0.24 * vocal_clarity
        + 0.22 * spectral_balance
        + 0.18 * dynamics_control
        + 0.16 * low_end_control
        + 0.12 * translation
        + 0.08 * max(0.0, min(1.0, air * 4.0))
    )
    return {
        "musical_quality": musical_quality,
        "vocal_clarity": vocal_clarity,
        "spectral_balance": spectral_balance,
        "dynamics_control": dynamics_control,
        "low_end_control": low_end_control,
        "translation_score": translation,
        "safety_margin": safety_margin,
    }


def _estimate_true_peak_db(stereo: np.ndarray) -> float:
    if stereo.size == 0:
        return -120.0
    if stereo.shape[0] < 2:
        return amp_to_db(float(np.max(np.abs(stereo))))
    factor = 4
    x_old = np.arange(stereo.shape[0], dtype=np.float64)
    x_new = np.linspace(0, stereo.shape[0] - 1, num=stereo.shape[0] * factor)
    peaks = []
    for channel in range(stereo.shape[1]):
        peaks.append(float(np.max(np.abs(np.interp(x_new, x_old, stereo[:, channel])))))
    return amp_to_db(max(peaks))


def _band_energy_db(
    mono: np.ndarray,
    sample_rate: int,
    bands: Mapping[str, tuple[float, float]],
) -> dict[str, float]:
    powers = _band_powers(mono, sample_rate, bands)
    return {name: float(10.0 * np.log10(power + EPS)) for name, power in powers.items()}


def _band_energy_ratios(
    mono: np.ndarray,
    sample_rate: int,
    bands: Mapping[str, tuple[float, float]],
) -> dict[str, float]:
    powers = _band_powers(mono, sample_rate, bands)
    total = sum(powers.values()) + EPS
    return {name: float(power / total) for name, power in powers.items()}


def _band_powers(
    mono: np.ndarray,
    sample_rate: int,
    bands: Mapping[str, tuple[float, float]],
) -> dict[str, float]:
    if mono.size == 0:
        return {name: 0.0 for name in bands}
    fft_size = int(2 ** np.ceil(np.log2(max(2048, min(65536, mono.size)))))
    windowed = mono[-fft_size:] if mono.size >= fft_size else np.pad(mono, (0, fft_size - mono.size))
    window = np.hanning(fft_size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(windowed * window)) ** 2
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    powers = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= float(low)) & (freqs < float(high))
        powers[str(name)] = float(np.sum(spectrum[mask])) if np.any(mask) else 0.0
    return powers


def _spectral_centroid_hz(mono: np.ndarray, sample_rate: int) -> float:
    if mono.size == 0:
        return 0.0
    fft_size = min(65536, max(2048, int(2 ** np.ceil(np.log2(mono.size)))))
    block = mono[-fft_size:] if mono.size >= fft_size else np.pad(mono, (0, fft_size - mono.size))
    spectrum = np.abs(np.fft.rfft(block * np.hanning(fft_size))) + EPS
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    return float(np.sum(freqs * spectrum) / np.sum(spectrum))


def _phase_correlation(stereo: np.ndarray) -> float:
    if stereo.ndim != 2 or stereo.shape[1] < 2 or stereo.shape[0] < 2:
        return 1.0
    left = stereo[:, 0]
    right = stereo[:, 1]
    if float(np.std(left)) < EPS or float(np.std(right)) < EPS:
        return 1.0
    return float(np.clip(np.corrcoef(left, right)[0, 1], -1.0, 1.0))


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))
