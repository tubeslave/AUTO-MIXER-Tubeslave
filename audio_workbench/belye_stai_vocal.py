"""Frozen raw-to-processed vocal adapter for the delivered Belye Stai DSP mix.

This song/version-specific STUDIO module makes the original local vocal chain
reusable and exposes the first compression insert as an explicit replacement
point. The no-override path reproduces the legacy recipe; an override replaces
that first compressor in place rather than adding another compressor downstream.
No model inference, pitch correction, timing correction, clipping or hidden
makeup is performed here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numba import njit
from scipy import ndimage, signal

from .mixing.compression import CompressorConfig, LinkedCompressor, as_audio


@dataclass(frozen=True)
class BelyeStaiLeadVocalRecipe:
    sample_rate: int = 44100
    frames: int = 9_128_700
    highpass_hz: float = 110.0
    lowpass_hz: float = 14_500.0
    target_active_rms_dbfs: float = -22.6
    level_percentile: float = 65.0


@dataclass(frozen=True)
class VocalStages:
    pre_compression: np.ndarray
    processed: np.ndarray
    report: dict[str, Any]


def _db(x: np.ndarray | float) -> np.ndarray | float:
    return 20 * np.log10(np.maximum(x, 1e-12))


def _gain(x: np.ndarray, db_value: np.ndarray | float) -> np.ndarray:
    g = np.power(10, np.asarray(db_value) / 20).astype(np.float32)
    return (x * (g[:, None] if np.ndim(g) and x.ndim == 2 else g)).astype(np.float32)


def _filter(x: np.ndarray, sr: int, low: float | None = None,
            high: float | None = None, order: int = 2) -> np.ndarray:
    y = x
    if low:
        y = signal.sosfilt(signal.butter(order, low, btype="highpass", fs=sr, output="sos"), y, axis=0)
    if high:
        y = signal.sosfilt(signal.butter(order, high, btype="lowpass", fs=sr, output="sos"), y, axis=0)
    return np.asarray(y, dtype=np.float32)


def _bell(x: np.ndarray, sr: int, frequency_hz: float, gain_db: float,
          q: float = .85) -> np.ndarray:
    amplitude = 10 ** (gain_db / 40)
    omega = 2 * np.pi * frequency_hz / sr
    cosine = np.cos(omega)
    alpha = np.sin(omega) / (2 * q)
    b = np.array([1 + alpha * amplitude, -2 * cosine, 1 - alpha * amplitude])
    a = np.array([1 + alpha / amplitude, -2 * cosine, 1 - alpha / amplitude])
    return signal.lfilter(b / a[0], a / a[0], x, axis=0).astype(np.float32)


def _band(x: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    return signal.sosfiltfilt(
        signal.butter(2, [low, high], btype="bandpass", fs=sr, output="sos"),
        x, axis=0,
    ).astype(np.float32)


def _frame_db(x: np.ndarray, sr: int, hop_s: float = .01) -> np.ndarray:
    hop = max(1, round(sr * hop_s))
    usable = len(x) // hop * hop
    if usable == 0:
        raise ValueError("vocal source is too short for frame analysis")
    frames = x[:usable].reshape(-1, hop, *x.shape[1:])
    power = np.mean(frames.astype(np.float64) ** 2, axis=1)
    if power.ndim > 1:
        power = np.mean(power, axis=1)
    return 10 * np.log10(np.maximum(power, 1e-24))


def _curve(values: np.ndarray, hop_s: float, sr: int, n: int) -> np.ndarray:
    return np.interp(
        np.arange(n) / sr,
        (np.arange(len(values)) + .5) * hop_s,
        values,
    ).astype(np.float32)


def _active_level(x: np.ndarray, sr: int, percentile: float = 65) -> float:
    level = _frame_db(x, sr, .02)
    threshold = max(float(np.percentile(level, percentile)), float(np.max(level)) - 35)
    active = level >= threshold
    return float(10 * np.log10(np.mean(10 ** (level[active] / 10))))


def _level(x: np.ndarray, sr: int, target: float,
           percentile: float = 65) -> tuple[np.ndarray, dict]:
    before = _active_level(x, sr, percentile)
    adjustment = float(np.clip(target - before, -30, 30))
    y = _gain(x, adjustment)
    return y, {
        "measurement": "RMS of source-active 20 ms frames, not a universal LUFS prescription",
        "active_percentile": percentile,
        "active_rms_before_dbfs": before,
        "target_active_rms_dbfs": target,
        "gain_db": adjustment,
        "active_rms_after_dbfs": _active_level(y, sr, percentile),
    }


@njit(cache=True)
def _smooth_gr(request: np.ndarray, sr: int, attack_ms: float,
               release_ms: float) -> np.ndarray:
    attack = np.exp(-1 / (sr * attack_ms / 1000))
    release = np.exp(-1 / (sr * release_ms / 1000))
    out = np.empty(len(request), np.float32)
    previous = 0.0
    for i in range(len(request)):
        coefficient = attack if request[i] > previous else release
        previous = coefficient * previous + (1 - coefficient) * request[i]
        out[i] = previous
    return out


def _legacy_compressor(x: np.ndarray, sr: int, desired: float, ratio: float,
                       attack_ms: float, release_ms: float, cap_db: float,
                       knee_db: float = 5.) -> tuple[np.ndarray, dict, np.ndarray]:
    power = x.astype(np.float64) ** 2
    if power.ndim == 2:
        power = power.mean(1)
    alpha = np.exp(-1 / (sr * .003))
    power = signal.lfilter([1 - alpha], [1, -alpha], power)
    envelope = 10 * np.log10(np.maximum(power, 1e-20))
    step = max(1, round(sr * .01))
    sampled = envelope[::step]
    active = sampled[sampled >= np.percentile(sampled, 60)]
    threshold = float(np.percentile(active, 90) - desired / (1 - 1 / ratio))
    over = envelope - threshold
    request = np.where(
        over <= -knee_db / 2,
        0,
        np.where(
            over >= knee_db / 2,
            over * (1 - 1 / ratio),
            ((over + knee_db / 2) ** 2) / (2 * knee_db) * (1 - 1 / ratio),
        ),
    )
    request = np.clip(request, 0, cap_db)
    gr = _smooth_gr(request, sr, attack_ms, release_ms)
    y = _gain(x, -gr)
    return y, {
        "mode": "frozen_delivery_recipe",
        "ratio": ratio,
        "threshold_dbfs": threshold,
        "attack_ms": attack_ms,
        "release_ms": release_ms,
        "knee_db": knee_db,
        "hard_max_gr_db": cap_db,
        "observed_max_gr_db": float(gr.max()),
        "observed_p95_gr_db": float(np.percentile(gr[::step], 95)),
        "stereo_linked": True,
    }, gr


def _expand(x: np.ndarray, sr: int, low: float, high: float, *,
            threshold_q: float = 25, margin_db: float = 6., floor: float = -10.,
            release_s: float = .16, hold_s: float = .10) -> tuple[np.ndarray, dict]:
    envelope = _frame_db(_band(x, sr, low, high), sr)
    threshold = float(np.percentile(envelope, threshold_q) + margin_db)
    envelope = ndimage.maximum_filter1d(
        envelope, size=max(3, int(hold_s / .01) * 2 + 1)
    )
    delta = np.clip((envelope - threshold) * 1.2, floor, 0)
    delta = ndimage.gaussian_filter1d(delta, max(1, release_s / .01 / 3))
    return _gain(x, _curve(delta, .01, sr, len(x))), {
        "detector_band_hz": [low, high],
        "threshold_dbfs": threshold,
        "floor_db": floor,
        "release_seconds": release_s,
        "hold_preopen_seconds": hold_s,
        "min_gain_db": float(delta.min()),
        "hard_gate": False,
    }


def _ride(x: np.ndarray, sr: int, low: float = 180, high: float = 3800,
          depth_db: float = 2.) -> tuple[np.ndarray, dict]:
    detector = _frame_db(_band(x, sr, low, high), sr, .05)
    smoothed = ndimage.gaussian_filter1d(detector, 8)
    threshold = max(float(np.percentile(smoothed, 20) + 9),
                    float(np.percentile(smoothed, 75) - 18))
    active = smoothed > threshold
    target = float(np.median(smoothed[active])) if active.any() else float(np.median(smoothed))
    delta = np.where(active, np.clip((target - smoothed) * .50, -depth_db, depth_db), 0)
    delta = ndimage.gaussian_filter1d(delta, 6)
    return _gain(x, _curve(delta, .05, sr, len(x))), {
        "range_db": [float(delta.min()), float(delta.max())],
        "window_s": .4,
        "activity_threshold_dbfs": threshold,
    }


def _deess(x: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
    high = _band(x, sr, 4500, 10000)
    mid = _band(x, sr, 800, 3500)
    high_db = _frame_db(high, sr, .005)
    mid_db = _frame_db(mid, sr, .005)
    ratio = high_db - mid_db
    threshold = max(float(np.percentile(ratio, 82)), -10.)
    gr = np.clip((ratio - threshold) * .6, 0, 2.8)
    gr[high_db < np.percentile(high_db, 55)] = 0
    gr = ndimage.gaussian_filter1d(gr, 2)
    gain = 10 ** (_curve(-gr, .005, sr, len(x)) / 20)
    return (x + high * (gain - 1)).astype(np.float32), {
        "band_hz": [4500, 10000],
        "threshold_relative_db": threshold,
        "max_gr_db": float(gr.max()),
    }


def _validate_raw(raw: np.ndarray, recipe: BelyeStaiLeadVocalRecipe) -> np.ndarray:
    audio = as_audio(raw)
    if audio.ndim != 1:
        raise ValueError("Belye Stai lead vocal raw source must be mono")
    if len(audio) != recipe.frames:
        raise ValueError("lead vocal frame count differs from frozen recipe")
    if recipe.sample_rate != 44100:
        raise ValueError("lead vocal recipe sample rate differs from frozen delivery")
    return audio.astype(np.float32, copy=True)


def prepare_lead_vocal(raw: np.ndarray, *,
                       recipe: BelyeStaiLeadVocalRecipe | None = None
                       ) -> tuple[np.ndarray, dict]:
    """Render immutable raw lead vocal through EQ, expansion and slow ride only."""
    recipe = recipe or BelyeStaiLeadVocalRecipe()
    x = _validate_raw(raw, recipe)
    y = _filter(x, recipe.sample_rate, recipe.highpass_hz, recipe.lowpass_hz)
    y = _bell(y, recipe.sample_rate, 230, -2.4, .8)
    y = _bell(y, recipe.sample_rate, 700, -.7, .9)
    y = _bell(y, recipe.sample_rate, 2400, 2.0, .7)
    y, expansion = _expand(
        y, recipe.sample_rate, 250, 3500, threshold_q=18, margin_db=8.,
        floor=-8, release_s=.2, hold_s=.15,
    )
    y, rider = _ride(y, recipe.sample_rate, 220, 3800, 2.3)
    return y, {
        "schema": "belye-stai-lead-vocal-precompression-v1",
        "eq": [[230, -2.4, .8], [700, -.7, .9], [2400, 2.0, .7]],
        "highpass_hz": recipe.highpass_hz,
        "lowpass_hz": recipe.lowpass_hz,
        "expansion": expansion,
        "ride": rider,
        "input_unchanged": True,
    }


def render_lead_vocal(raw: np.ndarray, *,
                      first_stage_config: CompressorConfig | None = None,
                      recipe: BelyeStaiLeadVocalRecipe | None = None) -> VocalStages:
    """Render the full local lead-vocal chain with one explicit compressor insert.

    ``first_stage_config=None`` reproduces the frozen delivery compressor. Passing
    a config REPLACES that first compressor at the same insertion point. The frozen
    second compressor, de-esser and final active-RMS level stage remain unchanged.
    """
    recipe = recipe or BelyeStaiLeadVocalRecipe()
    source = _validate_raw(raw, recipe)
    pre, pre_report = prepare_lead_vocal(source, recipe=recipe)
    if first_stage_config is None:
        first, first_report, first_gr = _legacy_compressor(
            pre, recipe.sample_rate, 3.2, 3.2, 10, 105, 4.5
        )
        first_mode = "frozen_delivery_recipe"
    else:
        first_stage_config.validate(recipe.sample_rate)
        first, first_gr = LinkedCompressor(recipe.sample_rate, first_stage_config).process(pre)
        first_report = {
            "mode": "compression_director_replacement",
            "config": asdict(first_stage_config),
            "observed_max_gr_db": float(np.max(first_gr)) if len(first_gr) else 0.0,
            "observed_p95_gr_db": float(np.percentile(first_gr, 95)) if len(first_gr) else 0.0,
        }
        first_mode = "replacement"
    second, second_report, _ = _legacy_compressor(
        first, recipe.sample_rate, 1.3, 2.0, 30, 220, 2.5
    )
    processed, deess = _deess(second, recipe.sample_rate)
    processed, level = _level(
        processed, recipe.sample_rate, recipe.target_active_rms_dbfs,
        recipe.level_percentile,
    )
    report = {
        "schema": "belye-stai-lead-vocal-local-processor-v1",
        "source": "VALERA_VOX",
        "sample_rate": recipe.sample_rate,
        "frames": recipe.frames,
        "precompression": pre_report,
        "first_compressor": first_report,
        "first_compressor_mode": first_mode,
        "second_compressor": second_report,
        "deess": deess,
        "level": level,
        "input_unchanged": bool(np.array_equal(source, raw)),
        "neural_audio_used": False,
        "pitch_or_timing_correction": False,
        "requires_human_review": first_stage_config is not None,
        "requires_human_listening": first_stage_config is not None,
        "baseline_eligible": False,
    }
    return VocalStages(pre_compression=pre, processed=processed, report=report)
