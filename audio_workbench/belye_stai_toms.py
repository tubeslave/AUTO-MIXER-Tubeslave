"""Frozen Belye Stai tom inserts for source-specific compression experiments.

TOM_1, TOM_2 and FLOOR are independent mono close mics with the delivered
filter/EQ/expander preparation frozen before the compressor.  A replacement
occurs at the original compressor insert and keeps the source-bound final gain.
No replacement is a musical winner by construction; downstream session rerender
and human listening remain mandatory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any

import numpy as np

from .belye_stai_vocal import _bell, _expand, _filter, _gain, _legacy_compressor, _level
from .mixing.compression import CompressorConfig, LinkedCompressor, as_audio


@dataclass(frozen=True)
class TomSpec:
    highpass_hz: float
    pan: float


TOM_SPECS: dict[str, TomSpec] = {
    "TOM_1": TomSpec(highpass_hz=65.0, pan=-0.45),
    "TOM_2": TomSpec(highpass_hz=55.0, pan=0.12),
    "FLOOR": TomSpec(highpass_hz=45.0, pan=0.52),
}


@dataclass(frozen=True)
class BelyeStaiTomRecipe:
    sample_rate: int = 44_100
    frames: int = 9_128_700


@dataclass(frozen=True)
class FrozenTomControls:
    source: str
    raw_pcm_sha256: str
    pre_compression_sha256: str
    final_gain_db: float


@dataclass(frozen=True)
class TomStages:
    pre_compression: np.ndarray
    processed: np.ndarray
    controls: FrozenTomControls
    report: dict[str, Any]


def _digest(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype="<f4").tobytes()).hexdigest()


def _spec(source: str) -> TomSpec:
    try:
        return TOM_SPECS[str(source)]
    except KeyError as exc:
        raise ValueError(f"unsupported Belye Stai tom source: {source}") from exc


def _validate(raw: np.ndarray, source: str, recipe: BelyeStaiTomRecipe) -> np.ndarray:
    _spec(source)
    x = as_audio(raw)
    if recipe.sample_rate != 44_100 or isinstance(recipe.sample_rate, bool):
        raise ValueError("tom recipe sample rate must be the frozen 44100 Hz")
    if isinstance(recipe.frames, bool) or not isinstance(recipe.frames, (int, np.integer)) or recipe.frames < 4410:
        raise ValueError("tom recipe needs at least 100 ms of source frames")
    if x.ndim != 1 or len(x) != recipe.frames:
        raise ValueError(f"{source} must be mono and match frozen frame count")
    if not len(x) or np.max(np.abs(x)) < 1e-10:
        raise ValueError(f"silent {source} cannot define tom processing")
    return x.copy()


def prepare_tom(raw: np.ndarray, source: str, *,
                recipe: BelyeStaiTomRecipe | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """RAW close mic -> frozen filter/EQ/expander -> compressor boundary."""
    recipe = recipe or BelyeStaiTomRecipe()
    spec = _spec(source)
    x = _validate(raw, source, recipe)
    y = _filter(x, recipe.sample_rate, spec.highpass_hz, 10_500)
    y = _bell(y, recipe.sample_rate, 420, -2.0, .8)
    y, expansion = _expand(
        y, recipe.sample_rate, 80, 350,
        threshold_q=90, margin_db=-5.0, floor=-16.0,
        release_s=.22, hold_s=.08,
    )
    return y, {
        "schema": "belye-stai-tom-precompression-v1",
        "source": source,
        "highpass_hz": spec.highpass_hz,
        "lowpass_hz": 10_500,
        "eq": [[420, -2.0, .8]],
        "expansion": expansion,
        "pan_downstream": spec.pan,
        "pre_compression_sha256": _digest(y),
        "input_unchanged": True,
    }


def render_tom(raw: np.ndarray, source: str, *,
               compressor_config: CompressorConfig | None = None,
               frozen_controls: FrozenTomControls | None = None,
               recipe: BelyeStaiTomRecipe | None = None) -> TomStages:
    """Reproduce one delivered tom close mic or replace its compressor in place."""
    recipe = recipe or BelyeStaiTomRecipe()
    x = _validate(raw, source, recipe)
    raw_hash = _digest(x)
    pre, preparation = prepare_tom(x, source, recipe=recipe)
    pre_hash = _digest(pre)

    if compressor_config is None:
        if frozen_controls is not None:
            raise ValueError("frozen controls are only valid with an explicit tom compressor replacement")
        y, comp, _ = _legacy_compressor(pre, recipe.sample_rate, 2.5, 2.6, 18, 160, 4.0)
        y, level = _level(y, recipe.sample_rate, -27.0, 98)
        controls = FrozenTomControls(source, raw_hash, pre_hash, float(level["gain_db"]))
        mode = "frozen_delivery_recipe"
    else:
        if not isinstance(frozen_controls, FrozenTomControls):
            raise ValueError("tom replacement requires source-bound baseline controls")
        if frozen_controls.source != source or frozen_controls.raw_pcm_sha256 != raw_hash:
            raise ValueError("frozen tom controls belong to a different source")
        if frozen_controls.pre_compression_sha256 != pre_hash:
            raise ValueError("frozen tom controls belong to a different preparation")
        if not np.isfinite(frozen_controls.final_gain_db) or not -30 <= frozen_controls.final_gain_db <= 30:
            raise ValueError("invalid frozen tom final gain")
        compressor_config.validate(recipe.sample_rate)
        y, gr = LinkedCompressor(recipe.sample_rate, compressor_config).process(pre)
        comp = {
            "mode": "compression_director_replacement",
            "config": asdict(compressor_config),
            "observed_max_gr_db": float(np.max(gr)) if len(gr) else 0.0,
            "whole_track_p95_gr_db": float(np.percentile(gr, 95)) if len(gr) else 0.0,
        }
        y = _gain(y, frozen_controls.final_gain_db)
        level = {
            "mode": "frozen_baseline_static_gain",
            "gain_db": frozen_controls.final_gain_db,
            "level_reestimated": False,
            "audition_level_match_applied": False,
        }
        controls = frozen_controls
        mode = "replacement"

    return TomStages(pre, y, controls, {
        "schema": "belye-stai-tom-local-processor-v1",
        "source": source,
        "sample_rate": recipe.sample_rate,
        "frames": recipe.frames,
        "source_sha256": raw_hash,
        "processed_sha256": _digest(y),
        "precompression": preparation,
        "compressor_mode": mode,
        "compressor": comp,
        "level": level,
        "frozen_controls": asdict(controls),
        "compression_stage_count": 1,
        "expansion_frozen": True,
        "pan_downstream_frozen": True,
        "requires_full_session_rerender": compressor_config is not None,
        "requires_human_review": compressor_config is not None,
        "requires_human_listening": compressor_config is not None,
        "baseline_eligible": False,
        "neural_audio_used": False,
        "input_files_modified": False,
    })
