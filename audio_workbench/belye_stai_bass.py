"""Frozen Belye Stai bass insert for instrument-specific compressor experiments.

The original two-stage bass chain is reproduced on the no-override path. A new
first compressor replaces the original at the same point, never stacks after it.
For replacements the original second-stage threshold and final static gain are
explicitly frozen. Level matching for audition is a separate reported operation.
The unchanged vocal adapter is reused only for its frozen arithmetic helpers;
no vocal compressor settings, EQ, detector band or rider settings are borrowed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any

import numpy as np
from scipy import signal

from .belye_stai_vocal import (
    _bell, _filter, _gain, _legacy_compressor, _level, _ride, _smooth_gr,
)
from .mixing.compression import CompressorConfig, LinkedCompressor, as_audio


@dataclass(frozen=True)
class BelyeStaiBassRecipe:
    sample_rate: int = 44100
    frames: int = 9_128_700


@dataclass(frozen=True)
class FrozenBassControls:
    """Source-bound downstream values obtained from the no-change render."""
    raw_pcm_sha256: str
    pre_compression_sha256: str
    second_threshold_dbfs: float
    final_gain_db: float


@dataclass(frozen=True)
class BassStages:
    pre_compression: np.ndarray
    processed: np.ndarray
    controls: FrozenBassControls
    report: dict[str, Any]


def _digest(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype='<f4').tobytes()).hexdigest()


def _validate(raw: np.ndarray, recipe: BelyeStaiBassRecipe) -> np.ndarray:
    x = as_audio(raw)
    if recipe.sample_rate != 44100 or isinstance(recipe.sample_rate, bool):
        raise ValueError('bass recipe sample rate must be the frozen 44100 Hz')
    if (isinstance(recipe.frames, bool) or not isinstance(recipe.frames, (int, np.integer))
            or recipe.frames < 4410):
        raise ValueError('bass recipe needs at least 100 ms of source frames')
    if x.ndim != 1 or len(x) != recipe.frames:
        raise ValueError('bass raw source must be mono and match frozen frame count')
    if not len(x) or np.max(np.abs(x)) < 1e-10:
        raise ValueError('silent bass source cannot define a level reference')
    return x.copy()


def prepare_bass(raw: np.ndarray, *, recipe: BelyeStaiBassRecipe | None = None
                 ) -> tuple[np.ndarray, dict[str, Any]]:
    """RAW -> frozen bass EQ -> slow bass-band ride, before original compressor 1."""
    recipe = recipe or BelyeStaiBassRecipe()
    x = _validate(raw, recipe)
    y = _filter(x, recipe.sample_rate, 33, 6000)
    y = _bell(y, recipe.sample_rate, 175, -2.0, .7)
    y = _bell(y, recipe.sample_rate, 850, 1.2, .7)
    y, rider = _ride(y, recipe.sample_rate, 50, 1100, 2.0)
    return y, {
        'schema': 'belye-stai-bass-precompression-v1',
        'highpass_hz': 33, 'lowpass_hz': 6000,
        'eq': [[175, -2.0, .7], [850, 1.2, .7]],
        'ride': rider, 'ride_detector_band_hz': [50, 1100],
        'pre_compression_sha256': _digest(y), 'input_unchanged': True,
    }


def _second_stage(x: np.ndarray, sr: int, threshold: float
                  ) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact legacy arithmetic with a FIXED threshold and a newly computed envelope.

    We freeze compressor settings, not its gain-reduction trace. The second stage
    must still respond to the new first-stage output, but cannot retune itself.
    """
    alpha = np.exp(-1 / (sr * .003))
    power = signal.lfilter([1 - alpha], [1, -alpha], x.astype(np.float64) ** 2)
    over = 10 * np.log10(np.maximum(power, 1e-20)) - threshold
    knee, slope = 5.0, .5
    requested = np.where(over <= -knee / 2, 0, np.where(
        over >= knee / 2, over * slope, (over + knee / 2) ** 2 / (2 * knee) * slope))
    gr = _smooth_gr(np.clip(requested, 0, 2.8), sr, 35, 240)
    return _gain(x, -gr), {
        'mode': 'source_bound_frozen_threshold', 'threshold_dbfs': threshold,
        'ratio': 2.0, 'attack_ms': 35, 'release_ms': 240, 'knee_db': 5.0,
        'hard_max_gr_db': 2.8, 'observed_max_gr_db': float(gr.max()),
        'whole_track_p95_gr_db': float(np.percentile(gr, 95)),
        'threshold_reestimated': False, 'gain_trace_recomputed': True,
    }


def render_bass(raw: np.ndarray, *, first_stage_config: CompressorConfig | None = None,
                frozen_controls: FrozenBassControls | None = None,
                recipe: BelyeStaiBassRecipe | None = None) -> BassStages:
    """Reproduce the delivered bass or replace compressor 1 under frozen controls.

    Overrides require controls from this exact source's baseline render. The
    original EQ, slow ride and two-stage topology remain; no saturation is added.
    The returned source is not peak-limited or automatically promoted. Callers must
    rerender the full session and level-match A/B before musical acceptance.
    """
    recipe = recipe or BelyeStaiBassRecipe()
    x = _validate(raw, recipe)
    pre, preparation = prepare_bass(x, recipe=recipe)
    source_hash, pre_hash = _digest(x), _digest(pre)
    if first_stage_config is None:
        if frozen_controls is not None:
            raise ValueError('frozen controls are only used with an explicit first-stage override')
        first, one, _ = _legacy_compressor(pre, recipe.sample_rate, 4., 3.5, 9, 130, 5.)
        second, two, _ = _legacy_compressor(first, recipe.sample_rate, 1.5, 2., 35, 240, 2.8)
        y, level = _level(second, recipe.sample_rate, -26.8, 65)
        controls = FrozenBassControls(source_hash, pre_hash, two['threshold_dbfs'], level['gain_db'])
        mode = 'frozen_delivery_recipe'
    else:
        if not isinstance(frozen_controls, FrozenBassControls):
            raise ValueError('replacement requires source-bound baseline controls')
        controls = frozen_controls
        if controls.raw_pcm_sha256 != source_hash or controls.pre_compression_sha256 != pre_hash:
            raise ValueError('frozen bass controls belong to a different source or preparation')
        if (not np.isfinite(controls.second_threshold_dbfs)
                or not -160 <= controls.second_threshold_dbfs <= 24
                or not np.isfinite(controls.final_gain_db)
                or not -30 <= controls.final_gain_db <= 30):
            raise ValueError('invalid frozen downstream controls')
        first_stage_config.validate(recipe.sample_rate)
        first, gr = LinkedCompressor(recipe.sample_rate, first_stage_config).process(pre)
        one = {'mode': 'compression_director_replacement', 'config': asdict(first_stage_config),
               'observed_max_gr_db': float(gr.max()),
               'whole_track_p95_gr_db': float(np.percentile(gr, 95))}
        second, two = _second_stage(first, recipe.sample_rate, controls.second_threshold_dbfs)
        y = _gain(second, controls.final_gain_db)
        level = {'mode': 'frozen_baseline_static_gain', 'gain_db': controls.final_gain_db,
                 'level_reestimated': False, 'audition_level_match_applied': False}
        mode = 'replacement'
    return BassStages(pre, y, controls, {
        'schema': 'belye-stai-bass-local-processor-v1', 'source': 'BASS',
        'sample_rate': recipe.sample_rate, 'frames': recipe.frames,
        'source_sha256': source_hash, 'processed_sha256': _digest(y),
        'precompression': preparation, 'first_compressor_mode': mode,
        'first_compressor': one, 'second_compressor': two, 'level': level,
        'frozen_controls': asdict(controls), 'compression_stage_count': 2,
        'requires_full_session_rerender': first_stage_config is not None,
        'requires_human_review': first_stage_config is not None,
        'requires_human_listening': first_stage_config is not None,
        'baseline_eligible': False, 'input_unchanged': _digest(raw) == source_hash,
        'saturation': False, 'neural_audio_used': False,
        'pitch_or_timing_correction': False,
        'kick_sidechain': 'unchanged downstream session stage, not baked into local bass',
    })
