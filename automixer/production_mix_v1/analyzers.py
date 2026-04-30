"""Dependency-light audio analyzers for production_mix_v1."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .config import ProductionMixConfig
from .models import AudioStem, amp_to_db, ensure_stereo


EPS = 1e-12


def analyze_audio(audio: np.ndarray, sample_rate: int, config: ProductionMixConfig) -> dict[str, Any]:
    stereo = ensure_stereo(audio)
    mono = np.mean(stereo, axis=1) if stereo.size else np.zeros(0, dtype=np.float32)
    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    rms = float(np.sqrt(np.mean(stereo * stereo) + EPS)) if stereo.size else 0.0
    peak_db = amp_to_db(peak)
    rms_db = amp_to_db(rms)
    true_peak_dbfs = peak_db
    crest = true_peak_dbfs - rms_db if rms_db > -119.0 else 0.0
    band_energy = _band_energy_db(mono, sample_rate, config.bands)
    band_ratios = _band_energy_ratios(mono, sample_rate, config.bands)
    centroid = _spectral_centroid_hz(mono, sample_rate)
    phase = _phase_correlation(stereo)
    dimensions = _dimension_scores(
        band_ratios=band_ratios,
        true_peak_dbfs=true_peak_dbfs,
        crest_factor_db=crest,
        phase_correlation=phase,
    )
    metrics = {
        "sample_rate": int(sample_rate),
        "samples": int(stereo.shape[0]),
        "peak_db": peak_db,
        "true_peak_dbfs": true_peak_dbfs,
        "rms_db": rms_db,
        "lufs_integrated_approx": rms_db - 0.691,
        "crest_factor_db": crest,
        "phase_correlation": phase,
        "spectral_centroid_hz": centroid,
        "band_energy_db": band_energy,
        "band_ratios": band_ratios,
        "dimensions": dimensions,
    }
    metrics["analyzer_issues"] = analyzer_issues(metrics, role="unknown")
    return metrics


def analyze_session(stems: list[AudioStem], mix_audio: np.ndarray, config: ProductionMixConfig) -> dict[str, Any]:
    sample_rate = stems[0].sample_rate if stems else 48000
    stem_payload = {}
    for stem in stems:
        metrics = analyze_audio(stem.audio, stem.sample_rate, config)
        metrics["analyzer_issues"] = analyzer_issues(metrics, role=stem.role)
        stem_payload[str(stem.channel_id)] = {
            "name": stem.name,
            "role": stem.role,
            "channel_id": int(stem.channel_id),
            "metrics": metrics,
        }
    analysis = {"mix": analyze_audio(mix_audio, sample_rate, config), "stems": stem_payload}
    analysis["guitar_forwardness"] = guitar_forwardness_from_analysis(analysis, config)
    return analysis


def guitar_forwardness_from_audio(
    stems: list[AudioStem],
    config: ProductionMixConfig,
    *,
    processed_audio: Mapping[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    stem_payload: dict[str, Any] = {}
    for stem in stems:
        audio = processed_audio.get(int(stem.channel_id), stem.audio) if processed_audio else stem.audio
        metrics = analyze_audio(audio, stem.sample_rate, config)
        metrics["analyzer_issues"] = analyzer_issues(metrics, role=stem.role)
        stem_payload[str(stem.channel_id)] = {
            "name": stem.name,
            "role": stem.role,
            "channel_id": int(stem.channel_id),
            "metrics": metrics,
        }
    return guitar_forwardness_from_analysis({"stems": stem_payload}, config)


def guitar_forwardness_from_analysis(analysis: Mapping[str, Any], config: ProductionMixConfig) -> dict[str, Any]:
    guitar_700_1500 = 0.0
    guitar_1500_3000 = 0.0
    vocal_700_1500 = 0.0
    vocal_1500_3000 = 0.0
    guitar_count = 0
    vocal_count = 0
    for payload in dict(analysis.get("stems", {}) or {}).values():
        if not isinstance(payload, Mapping):
            continue
        role = str(payload.get("role", ""))
        metrics = dict(payload.get("metrics", {}) or {})
        energy = dict(metrics.get("band_energy_db", {}) or {})
        low_mid = _power_from_db(float(energy.get("intelligibility", -120.0)))
        presence = _power_from_db(float(energy.get("presence", -120.0)))
        if role == "electric_guitar":
            guitar_700_1500 += low_mid
            guitar_1500_3000 += presence
            guitar_count += 1
        elif role == "lead_vocal":
            vocal_700_1500 += low_mid
            vocal_1500_3000 += presence
            vocal_count += 1
    cfg = dict(config.reference_recipe.get("guitar_control", {}) or {})
    combined_guitar = guitar_700_1500 + guitar_1500_3000
    combined_vocal = vocal_700_1500 + vocal_1500_3000
    mid_ratio = combined_guitar / max(combined_vocal, EPS)
    low_mid_ratio = guitar_700_1500 / max(vocal_700_1500, EPS)
    presence_ratio = guitar_1500_3000 / max(vocal_1500_3000, EPS)
    mid_ratio_db = _power_to_db(mid_ratio)
    low_mid_ratio_db = _power_to_db(low_mid_ratio)
    presence_ratio_db = _power_to_db(presence_ratio)
    masking_index = _clamp01((max(low_mid_ratio_db + 3.0, presence_ratio_db) + 6.0) / 12.0)
    has_vocal = vocal_count > 0
    has_guitars = guitar_count > 0
    too_forward = bool(
        has_vocal
        and has_guitars
        and (
            presence_ratio_db >= float(cfg.get("presence_ratio_threshold_db", 2.0))
            or mid_ratio_db >= float(cfg.get("midrange_ratio_threshold_db", -3.0))
            or masking_index >= float(cfg.get("masking_index_threshold", 0.62))
        )
    )
    return {
        "lead_vocal_present": has_vocal,
        "electric_guitars_present": has_guitars,
        "guitar_count": guitar_count,
        "lead_vocal_count": vocal_count,
        "detect_guitars_too_forward": too_forward,
        "guitars_too_forward": too_forward,
        "guitar_vocal_masking_index": masking_index,
        "guitar_to_vocal_midrange_ratio": mid_ratio,
        "guitar_to_vocal_midrange_ratio_db": mid_ratio_db,
        "guitar_to_vocal_700_1500_ratio_db": low_mid_ratio_db,
        "guitar_to_vocal_1500_3000_ratio_db": presence_ratio_db,
        "guitar_energy_700_1500": _power_to_db(guitar_700_1500),
        "guitar_energy_1500_3000": _power_to_db(guitar_1500_3000),
        "vocal_energy_700_1500": _power_to_db(vocal_700_1500),
        "vocal_energy_1500_3000": _power_to_db(vocal_1500_3000),
    }


def score_dimensions(analysis: Mapping[str, Any]) -> dict[str, float]:
    return {str(k): float(v) for k, v in dict(analysis.get("mix", analysis).get("dimensions", {}) or {}).items()}


def weighted_rule_score(dimensions: Mapping[str, float], config: ProductionMixConfig) -> float:
    total = 0.0
    weight_sum = 0.0
    for key, weight in config.weights.items():
        total += float(dimensions.get(key, 0.0)) * float(weight)
        weight_sum += float(weight)
    return float(total / weight_sum) if weight_sum else 0.0


def analyzer_issues(metrics: Mapping[str, Any], *, role: str) -> list[str]:
    rms_db = float(metrics.get("rms_db", -120.0))
    centroid = float(metrics.get("spectral_centroid_hz", 0.0))
    crest = float(metrics.get("crest_factor_db", 0.0))
    ratios = dict(metrics.get("band_ratios", {}) or {})
    ratio_sum = sum(float(ratios.get(k, 0.0)) for k in ratios)
    issues: list[str] = []
    if rms_db > -60.0 and centroid == 0.0:
        issues.append("analyzer_failure_or_silence_mismatch")
    if rms_db > -60.0 and ratio_sum == 0.0:
        issues.append("analyzer_failure")
    if rms_db > -60.0 and ((centroid == 0.0 and abs(crest - abs(rms_db)) < 0.25) or crest > 45.0):
        issues.append("low_signal_or_metric_error")
    if role in {"lead_vocal", "backing_vocal", "music_stem", "playback"} and crest > 45.0:
        issues.append("low_signal_or_metric_error")
    return list(dict.fromkeys(issues))


def _dimension_scores(*, band_ratios: Mapping[str, float], true_peak_dbfs: float, crest_factor_db: float, phase_correlation: float) -> dict[str, float]:
    mud = float(band_ratios.get("mud", 0.0) + band_ratios.get("boxiness", 0.0))
    presence = float(band_ratios.get("intelligibility", 0.0) + band_ratios.get("presence", 0.0))
    harsh = float(band_ratios.get("harshness", 0.0))
    low = float(band_ratios.get("sub", 0.0) + band_ratios.get("punch", 0.0))
    vocal = _clamp01(0.45 + 2.0 * presence - 1.4 * mud - harsh)
    spectral = _clamp01(1.0 - abs(mud - 0.16) * 2.0 - abs(harsh - 0.10) * 1.5)
    dynamics = _clamp01(1.0 - abs(crest_factor_db - 12.0) / 24.0)
    low_end = _clamp01(1.0 - abs(low - 0.24) * 2.3)
    translation = _clamp01(0.65 * max(0.0, phase_correlation) + 0.35 * (1.0 - harsh))
    safety = _clamp01((-1.0 - true_peak_dbfs) / 12.0 + 0.5)
    musical = _clamp01(0.25 * vocal + 0.25 * spectral + 0.20 * dynamics + 0.15 * low_end + 0.15 * translation)
    return {
        "musical_quality": musical,
        "vocal_clarity": vocal,
        "spectral_balance": spectral,
        "dynamics_control": dynamics,
        "low_end_control": low_end,
        "translation_score": translation,
        "safety_margin": safety,
    }


def _band_energy_db(mono: np.ndarray, sample_rate: int, bands: Mapping[str, tuple[float, float]]) -> dict[str, float]:
    powers = _band_powers(mono, sample_rate, bands)
    return {name: float(10.0 * np.log10(power + EPS)) for name, power in powers.items()}


def _power_from_db(value_db: float) -> float:
    if value_db <= -119.0:
        return 0.0
    return float(10.0 ** (float(value_db) / 10.0))


def _power_to_db(value: float) -> float:
    return float(10.0 * np.log10(max(float(value), EPS)))


def _band_energy_ratios(mono: np.ndarray, sample_rate: int, bands: Mapping[str, tuple[float, float]]) -> dict[str, float]:
    powers = _band_powers(mono, sample_rate, bands)
    total = sum(powers.values()) + EPS
    return {name: float(power / total) for name, power in powers.items()}


def _band_powers(mono: np.ndarray, sample_rate: int, bands: Mapping[str, tuple[float, float]]) -> dict[str, float]:
    if mono.size == 0:
        return {name: 0.0 for name in bands}
    fft_size = int(2 ** np.ceil(np.log2(max(2048, min(65536, mono.size)))))
    block = _active_fft_block(mono, fft_size)
    spectrum = np.abs(np.fft.rfft(block * np.hanning(block.size))) ** 2
    freqs = np.fft.rfftfreq(block.size, 1.0 / float(sample_rate))
    return {
        name: float(np.sum(spectrum[(freqs >= low) & (freqs < high)]))
        for name, (low, high) in bands.items()
    }


def _spectral_centroid_hz(mono: np.ndarray, sample_rate: int) -> float:
    if mono.size == 0:
        return 0.0
    fft_size = int(2 ** np.ceil(np.log2(max(2048, min(65536, mono.size)))))
    block = _active_fft_block(mono, fft_size)
    spectrum = np.abs(np.fft.rfft(block * np.hanning(block.size)))
    total = float(np.sum(spectrum) + EPS)
    if total <= EPS * 2:
        return 0.0
    freqs = np.fft.rfftfreq(block.size, 1.0 / float(sample_rate))
    return float(np.sum(freqs * spectrum) / total)


def _active_fft_block(mono: np.ndarray, fft_size: int) -> np.ndarray:
    """Use the loudest analysis block, not the tail; rendered stems often end in silence."""
    if mono.size < fft_size:
        return np.pad(mono, (0, fft_size - mono.size))
    step = max(1, fft_size // 2)
    best_start = 0
    best_power = -1.0
    starts = list(range(0, mono.size - fft_size + 1, step))
    tail_start = mono.size - fft_size
    if not starts or starts[-1] != tail_start:
        starts.append(tail_start)
    for start in starts:
        block = mono[start : start + fft_size]
        power = float(np.mean(block * block))
        if power > best_power:
            best_power = power
            best_start = start
    return mono[best_start : best_start + fft_size]


def _phase_correlation(stereo: np.ndarray) -> float:
    if stereo.ndim != 2 or stereo.shape[1] < 2 or stereo.shape[0] < 2:
        return 1.0
    left, right = stereo[:, 0], stereo[:, 1]
    if float(np.std(left)) < EPS or float(np.std(right)) < EPS:
        return 1.0
    return float(np.clip(np.corrcoef(left, right)[0, 1], -1.0, 1.0))


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))
