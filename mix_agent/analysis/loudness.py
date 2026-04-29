"""Level, loudness and artifact-adjacent technical metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def to_mono(audio: np.ndarray) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        return data
    return np.mean(data, axis=1)


def amp_to_db(value: float) -> float:
    return 20.0 * np.log10(max(float(value), 1e-12))


def _window_rms_db(samples: np.ndarray, sample_rate: int, window_sec: float) -> List[float]:
    mono = to_mono(samples)
    window = max(1, int(sample_rate * window_sec))
    hop = max(1, window // 4)
    values: List[float] = []
    for start in range(0, max(1, len(mono) - window + 1), hop):
        chunk = mono[start:start + window]
        if len(chunk) == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
        values.append(-0.691 + 20.0 * np.log10(rms + 1e-12))
    return values


def _integrated_lufs(audio: np.ndarray, sample_rate: int) -> Tuple[float, str]:
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sample_rate)
        return float(meter.integrated_loudness(audio)), "pyloudnorm"
    except Exception:
        mono = to_mono(audio)
        rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
        return -0.691 + 20.0 * np.log10(rms + 1e-12), "rms_approximation"


def _true_peak(audio: np.ndarray) -> Tuple[float, str]:
    try:
        from scipy.signal import resample_poly

        upsampled = resample_poly(np.asarray(audio, dtype=np.float32), 4, 1, axis=0)
        return amp_to_db(float(np.max(np.abs(upsampled)))), "4x_resample_poly"
    except Exception:
        return amp_to_db(float(np.max(np.abs(audio)))), "sample_peak_fallback"


def _silence_regions(samples: np.ndarray, sample_rate: int, threshold_db: float = -60.0) -> Dict[str, Any]:
    mono = to_mono(samples)
    window = max(1, int(sample_rate * 0.25))
    hop = window
    regions = []
    in_region = False
    start_sec = 0.0
    for start in range(0, len(mono), hop):
        chunk = mono[start:start + window]
        rms_db = amp_to_db(float(np.sqrt(np.mean(np.square(chunk)) + 1e-12)))
        silent = rms_db < threshold_db
        if silent and not in_region:
            in_region = True
            start_sec = start / sample_rate
        if in_region and (not silent or start + window >= len(mono)):
            end_sec = min(len(mono), start + window) / sample_rate
            regions.append({"start_sec": round(start_sec, 3), "end_sec": round(end_sec, 3)})
            in_region = False
    total = sum(item["end_sec"] - item["start_sec"] for item in regions)
    return {"count": len(regions), "total_sec": round(total, 3), "regions": regions[:20]}


def _noise_floor(samples: np.ndarray, sample_rate: int) -> float:
    mono = to_mono(samples)
    window = max(1, int(sample_rate * 0.1))
    values = []
    for start in range(0, max(1, len(mono) - window + 1), window):
        chunk = mono[start:start + window]
        values.append(amp_to_db(float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))))
    if not values:
        return -120.0
    return float(np.percentile(values, 10))


def _click_pop_count(samples: np.ndarray) -> int:
    mono = to_mono(samples)
    if len(mono) < 4:
        return 0
    diff = np.diff(mono)
    median = float(np.median(np.abs(diff))) + 1e-12
    threshold = max(0.25, median * 12.0)
    return int(np.sum(np.abs(diff) > threshold))


def compute_level_metrics(audio: np.ndarray, sample_rate: int) -> Tuple[Dict[str, Any], List[str]]:
    """Compute technical level metrics for mono or stereo audio."""
    limitations: List[str] = []
    data = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(data)) + 1e-12)
    peak_db = amp_to_db(peak)
    rms = float(np.sqrt(np.mean(np.square(data)) + 1e-12))
    rms_db = amp_to_db(rms)
    true_peak_db, true_peak_method = _true_peak(data)
    lufs_integrated, lufs_method = _integrated_lufs(data, sample_rate)
    if lufs_method != "pyloudnorm":
        limitations.append("Integrated LUFS uses RMS approximation because pyloudnorm was unavailable.")
    if true_peak_method != "4x_resample_poly":
        limitations.append("True peak uses sample peak fallback because scipy resampling was unavailable.")

    momentary = _window_rms_db(data, sample_rate, 0.4)
    short_term = _window_rms_db(data, sample_rate, 3.0)
    lra = 0.0
    if len(short_term) >= 4:
        gated = np.asarray([value for value in short_term if value > -70.0], dtype=np.float32)
        if len(gated) >= 4:
            lra = float(np.percentile(gated, 95) - np.percentile(gated, 10))

    clip_threshold = 0.999
    clip_count = int(np.sum(np.abs(data) >= clip_threshold))
    metrics = {
        "peak_dbfs": round(peak_db, 3),
        "true_peak_dbtp": round(true_peak_db, 3),
        "true_peak_method": true_peak_method,
        "rms_dbfs": round(rms_db, 3),
        "integrated_lufs": round(float(lufs_integrated), 3),
        "integrated_lufs_method": lufs_method,
        "momentary_lufs": round(float(momentary[-1]), 3) if momentary else None,
        "short_term_lufs": round(float(short_term[-1]), 3) if short_term else None,
        "loudness_range_lu": round(lra, 3),
        "crest_factor_db": round(true_peak_db - rms_db, 3),
        "plr_db": round(true_peak_db - float(lufs_integrated), 3),
        "headroom_db": round(0.0 - true_peak_db, 3),
        "clip_count": clip_count,
        "inter_sample_peak_risk": bool(true_peak_db > peak_db + 0.5 or true_peak_db > -1.0),
        "dc_offset": round(float(np.mean(data)), 6),
        "noise_floor_dbfs": round(_noise_floor(data, sample_rate), 3),
        "silence": _silence_regions(data, sample_rate),
        "click_pop_estimate": _click_pop_count(data),
    }
    return metrics, limitations
