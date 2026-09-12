"""Level, loudness and artifact-adjacent technical metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .dsp_utils import (
    amp_to_db,
    integrated_loudness,
    sample_major,
    to_mono,
    true_peak,
    window_loudness,
)


def _integrated_lufs(audio: np.ndarray, sample_rate: int) -> Tuple[float, str]:
    value, method = integrated_loudness(audio, sample_rate)
    return value, "bs1770_gated" if method == "k_weighting" else method


def _true_peak(audio: np.ndarray) -> Tuple[float, str]:
    return true_peak(audio, oversample=4)


def _silence_regions(samples: np.ndarray, sample_rate: int, threshold_db: float = -60.0) -> Dict[str, Any]:
    mono = to_mono(samples)
    if mono.size == 0:
        return {"count": 0, "total_sec": 0.0, "regions": []}
    window = max(1, int(sample_rate * 0.25))
    hop = window
    regions = []
    in_region = False
    start_sec = 0.0
    for start in range(0, len(mono), hop):
        chunk = mono[start:start + window]
        rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64)) + 1e-12))
        rms_db = amp_to_db(rms)
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
    if mono.size == 0:
        return -120.0
    window = max(1, int(sample_rate * 0.1))
    values = []
    for start in range(0, len(mono), window):
        chunk = mono[start:start + window]
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64)) + 1e-12))
        values.append(amp_to_db(rms))
    if not values:
        return -120.0
    return float(np.percentile(values, 10))


def _click_pop_count(samples: np.ndarray) -> int:
    mono = to_mono(samples)
    if len(mono) < 4:
        return 0
    diff = np.diff(mono.astype(np.float64, copy=False))
    median = float(np.median(np.abs(diff))) + 1e-12
    threshold = max(0.25, median * 12.0)
    return int(np.sum(np.abs(diff) > threshold))


def _lra_from_short_term(short_term: List[float], integrated_lufs: float) -> float:
    """Approximate EBU-style LRA from 3 s loudness values.

    Apply the absolute gate and the usual LRA relative gate at integrated-20 LU,
    then use the 10th and 95th percentiles.  This is still a compact estimator,
    but unlike the previous implementation it operates on K-weighted windows.
    """
    values = np.asarray(
        [v for v in short_term if np.isfinite(v) and v > -70.0 and v >= integrated_lufs - 20.0],
        dtype=np.float64,
    )
    if values.size < 4:
        return 0.0
    return max(0.0, float(np.percentile(values, 95) - np.percentile(values, 10)))


def compute_level_metrics(audio: np.ndarray, sample_rate: int) -> Tuple[Dict[str, Any], List[str]]:
    """Compute finite technical level metrics for mono or stereo audio."""
    limitations: List[str] = []
    data = sample_major(audio)
    if data.size == 0:
        metrics = {
            "peak_dbfs": -240.0,
            "true_peak_dbtp": -240.0,
            "true_peak_method": "empty",
            "rms_dbfs": -240.0,
            "integrated_lufs": -100.0,
            "integrated_lufs_method": "empty",
            "momentary_lufs": None,
            "short_term_lufs": None,
            "loudness_range_lu": 0.0,
            "crest_factor_db": 0.0,
            "plr_db": 0.0,
            "headroom_db": 240.0,
            "clip_count": 0,
            "inter_sample_peak_risk": False,
            "dc_offset": 0.0,
            "noise_floor_dbfs": -120.0,
            "silence": {"count": 0, "total_sec": 0.0, "regions": []},
            "click_pop_estimate": 0,
        }
        limitations.append("Empty audio buffer: level metrics contain neutral/floor values.")
        return metrics, limitations

    peak = float(np.max(np.abs(data)))
    peak_db = amp_to_db(peak)
    rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64)) + 1e-12))
    rms_db = amp_to_db(rms)
    true_peak_db, true_peak_method = _true_peak(data)
    lufs_integrated, lufs_method = _integrated_lufs(data, sample_rate)
    if lufs_method != "bs1770_gated":
        limitations.append("Integrated loudness fell back from K-weighted BS.1770-style analysis.")
    if true_peak_method != "4x_resample_poly":
        limitations.append("True peak uses a fallback because scipy polyphase resampling was unavailable.")

    momentary, momentary_method = window_loudness(data, sample_rate, 0.4)
    short_term, short_method = window_loudness(data, sample_rate, 3.0)
    if momentary_method != "k_weighting" or short_method != "k_weighting":
        limitations.append("Momentary/short-term loudness uses an unweighted fallback.")
    lra = _lra_from_short_term(short_term, float(lufs_integrated))

    clip_threshold = 0.999
    clip_count = int(np.sum(np.abs(data) >= clip_threshold))
    crest = true_peak_db - rms_db if np.isfinite(true_peak_db) and np.isfinite(rms_db) else 0.0
    plr = true_peak_db - float(lufs_integrated) if np.isfinite(lufs_integrated) else 0.0
    dc_offset = float(np.mean(data, dtype=np.float64))

    metrics = {
        "peak_dbfs": round(float(peak_db), 3),
        "true_peak_dbtp": round(float(true_peak_db), 3),
        "true_peak_method": true_peak_method,
        "rms_dbfs": round(float(rms_db), 3),
        "integrated_lufs": round(float(lufs_integrated), 3),
        "integrated_lufs_method": lufs_method,
        "momentary_lufs": round(float(momentary[-1]), 3) if momentary else None,
        "short_term_lufs": round(float(short_term[-1]), 3) if short_term else None,
        "loudness_range_lu": round(float(lra), 3),
        "crest_factor_db": round(float(crest), 3),
        "plr_db": round(float(plr), 3),
        "headroom_db": round(float(0.0 - true_peak_db), 3),
        "clip_count": clip_count,
        "inter_sample_peak_risk": bool(true_peak_db > peak_db + 0.5 or true_peak_db > -1.0),
        "dc_offset": round(dc_offset, 6),
        "noise_floor_dbfs": round(_noise_floor(data, sample_rate), 3),
        "silence": _silence_regions(data, sample_rate),
        "click_pop_estimate": _click_pop_count(data),
    }
    return metrics, limitations
