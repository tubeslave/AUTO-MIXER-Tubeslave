"""Frozen Belye Stai overhead boundary for safe compression experiments.

No-change reproduces the delivered OH recipe: R calibration, linked HP/LP, EQ,
and source-active leveling. An optional linked stereo compressor is inserted
before the final frozen static gain; it is never stacked after the processed OH.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import numpy as np

from .belye_stai_vocal import _bell, _filter, _gain, _level
from .mixing.compression import CompressorConfig, LinkedCompressor, as_audio


@dataclass(frozen=True)
class BelyeStaiOverheadRecipe:
    sample_rate: int = 44100
    frames: int = 9_128_700


@dataclass(frozen=True)
class FrozenOverheadControls:
    left_raw_pcm_sha256: str
    right_raw_pcm_sha256: str
    pre_compression_sha256: str
    final_gain_db: float


@dataclass(frozen=True)
class OverheadStages:
    pre_compression: np.ndarray
    processed: np.ndarray
    controls: FrozenOverheadControls
    report: dict


def _digest(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype='<f4').tobytes()).hexdigest()


def _validate_mono(x: np.ndarray, recipe: BelyeStaiOverheadRecipe, label: str) -> np.ndarray:
    y = as_audio(x)
    if recipe.sample_rate != 44100 or isinstance(recipe.sample_rate, bool):
        raise ValueError("overhead recipe sample rate must be frozen at 44100 Hz")
    if isinstance(recipe.frames, bool) or not isinstance(recipe.frames, (int, np.integer)) or recipe.frames < 4410:
        raise ValueError("overhead recipe needs at least 100 ms")
    if y.ndim != 1 or len(y) != recipe.frames:
        raise ValueError(f"{label} must be mono and match frozen frame count")
    if float(np.max(np.abs(y))) < 1e-10:
        raise ValueError(f"silent {label} cannot define an overhead reference")
    return y.copy()


def prepare_overhead(left_raw: np.ndarray, right_raw: np.ndarray,
                     *, recipe: BelyeStaiOverheadRecipe | None = None) -> tuple[np.ndarray, dict]:
    recipe = recipe or BelyeStaiOverheadRecipe()
    left = _validate_mono(left_raw, recipe, "OHL")
    right = _validate_mono(right_raw, recipe, "OH_R")
    pair = np.column_stack([left, _gain(right, 3.0)]).astype(np.float32)
    pair = _filter(pair, recipe.sample_rate, 190, 13500)
    pair = _bell(pair, recipe.sample_rate, 6000, -1.2, .7)
    return pair, {
        "schema": "belye-stai-overhead-precompression-v1",
        "right_calibration_db": 3.0,
        "highpass_hz": 190,
        "lowpass_hz": 13500,
        "eq": [[6000, -1.2, .7]],
        "stereo_linked": True,
        "pre_compression_sha256": _digest(pair),
        "input_unchanged": True,
    }


def render_overhead(left_raw: np.ndarray, right_raw: np.ndarray,
                    *, compression_config: CompressorConfig | None = None,
                    frozen_controls: FrozenOverheadControls | None = None,
                    recipe: BelyeStaiOverheadRecipe | None = None) -> OverheadStages:
    recipe = recipe or BelyeStaiOverheadRecipe()
    left = _validate_mono(left_raw, recipe, "OHL")
    right = _validate_mono(right_raw, recipe, "OH_R")
    pre, prep = prepare_overhead(left, right, recipe=recipe)
    left_hash, right_hash, pre_hash = _digest(left), _digest(right), _digest(pre)

    if compression_config is None:
        if frozen_controls is not None:
            raise ValueError("frozen controls are only used with an explicit compressor candidate")
        processed, level = _level(pre, recipe.sample_rate, -34.0, 65)
        controls = FrozenOverheadControls(left_hash, right_hash, pre_hash, float(level["gain_db"]))
        comp = {"mode": "no_compressor_in_delivered_recipe", "observed_max_gr_db": 0.0}
        mode = "frozen_delivery_recipe"
    else:
        if not isinstance(frozen_controls, FrozenOverheadControls):
            raise ValueError("overhead compression candidate requires source-bound frozen controls")
        controls = frozen_controls
        if (controls.left_raw_pcm_sha256 != left_hash or controls.right_raw_pcm_sha256 != right_hash
                or controls.pre_compression_sha256 != pre_hash):
            raise ValueError("frozen overhead controls belong to a different source/preparation")
        if not np.isfinite(controls.final_gain_db) or not -30 <= controls.final_gain_db <= 30:
            raise ValueError("invalid frozen overhead output gain")
        compression_config.validate(recipe.sample_rate)
        compressed, gr = LinkedCompressor(recipe.sample_rate, compression_config).process(pre)
        processed = _gain(compressed, controls.final_gain_db)
        level = {"mode": "frozen_baseline_static_gain", "gain_db": controls.final_gain_db,
                 "level_reestimated": False, "audition_level_match_applied": False}
        comp = {"mode": "overhead_director_linked_replacement", "config": asdict(compression_config),
                "observed_max_gr_db": float(np.max(gr)), "whole_track_p95_gr_db": float(np.percentile(gr, 95))}
        mode = "compression_candidate"

    return OverheadStages(pre, processed, controls, {
        "schema": "belye-stai-overhead-local-processor-v1",
        "source": "OHL+OH_R",
        "sample_rate": recipe.sample_rate,
        "frames": recipe.frames,
        "left_source_sha256": left_hash,
        "right_source_sha256": right_hash,
        "processed_sha256": _digest(processed),
        "precompression": prep,
        "processor_mode": mode,
        "compressor": comp,
        "level": level,
        "frozen_controls": asdict(controls),
        "requires_full_session_rerender": compression_config is not None,
        "requires_human_review": compression_config is not None,
        "requires_human_listening": compression_config is not None,
        "baseline_eligible": False,
        "neural_audio_used": False,
        "stereo_width_processing": False,
        "time_or_polarity_changes": False,
    })
