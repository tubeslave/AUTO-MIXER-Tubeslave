"""Frozen Belye Stai grouped kick insert for source-specific compression experiments.

KICK IN / KICK OUT remain a synchronized microphone group. The no-override path
reproduces the delivered local processing exactly. Compression replacements occur
only after the frozen phase/EQ/mic-balance stage and before the frozen final gain.
No candidate may be promoted without the downstream session rerender and listening.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any

import numpy as np

from .belye_stai_vocal import (
    _active_level, _bell, _filter, _gain, _legacy_compressor, _level,
)
from .mixing.compression import CompressorConfig, LinkedCompressor, as_audio


@dataclass(frozen=True)
class BelyeStaiKickRecipe:
    sample_rate: int = 44_100
    frames: int = 9_128_700
    kick_out_advance_samples: int = 84


@dataclass(frozen=True)
class FrozenKickControls:
    kick_in_pcm_sha256: str
    kick_out_pcm_sha256: str
    pre_compression_sha256: str
    out_relative_match_gain_db: float
    final_gain_db: float


@dataclass(frozen=True)
class KickStages:
    pre_compression: np.ndarray
    processed: np.ndarray
    controls: FrozenKickControls
    report: dict[str, Any]


def _digest(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype='<f4').tobytes()).hexdigest()


def _validate_mono(x: np.ndarray, name: str, recipe: BelyeStaiKickRecipe) -> np.ndarray:
    y = as_audio(x)
    if recipe.sample_rate != 44_100 or isinstance(recipe.sample_rate, bool):
        raise ValueError('kick recipe sample rate must be the frozen 44100 Hz')
    if isinstance(recipe.frames, bool) or not isinstance(recipe.frames, (int, np.integer)) or recipe.frames < 4410:
        raise ValueError('kick recipe needs at least 100 ms of source frames')
    if y.ndim != 1 or len(y) != recipe.frames:
        raise ValueError(f'{name} must be mono and match frozen frame count')
    if not len(y) or np.max(np.abs(y)) < 1e-10:
        raise ValueError(f'silent {name} cannot define grouped kick processing')
    return y.copy()


def _advance(x: np.ndarray, samples: int) -> np.ndarray:
    if samples < 0:
        raise ValueError('kick-out advance must be non-negative')
    if samples == 0:
        return x.copy()
    if samples >= len(x):
        raise ValueError('kick-out advance exceeds source duration')
    return np.concatenate([x[samples:], np.zeros(samples, np.float32)])


def prepare_kick(kick_in: np.ndarray, kick_out: np.ndarray, *,
                 recipe: BelyeStaiKickRecipe | None = None,
                 frozen_match_gain_db: float | None = None
                 ) -> tuple[np.ndarray, dict[str, Any], float]:
    """Raw pair -> frozen phase/EQ/mic balance -> grouped compressor boundary."""
    recipe = recipe or BelyeStaiKickRecipe()
    ki = _validate_mono(kick_in, 'KICK_IN', recipe)
    ko = _validate_mono(kick_out, 'KICK_OUT', recipe)
    ko = _advance(ko, recipe.kick_out_advance_samples)

    ki = _filter(ki, recipe.sample_rate, 28, 6500)
    for frequency, gain_db, q in ((65, 1.2, .8), (190, -2.0, .9), (2800, 1.0, .7)):
        ki = _bell(ki, recipe.sample_rate, frequency, gain_db, q)
    ko = _filter(ko, recipe.sample_rate, 28, 1400)
    ko = _bell(ko, recipe.sample_rate, 190, -2.5, .8)

    measured_match = (
        _active_level(ki, recipe.sample_rate, 92)
        - _active_level(ko, recipe.sample_rate, 92)
        - 5.0
    )
    match = measured_match if frozen_match_gain_db is None else float(frozen_match_gain_db)
    if not np.isfinite(match) or not -30 <= match <= 12:
        raise ValueError('invalid kick-out relative match gain')
    combined = (ki + _gain(ko, match)).astype(np.float32)
    return combined, {
        'schema': 'belye-stai-kick-precompression-v1',
        'phase': {'KICK_OUT': {'advance_samples': recipe.kick_out_advance_samples,
                               'milliseconds': recipe.kick_out_advance_samples / recipe.sample_rate * 1000}},
        'kick_in': {'highpass_hz': 28, 'lowpass_hz': 6500,
                    'eq': [[65, 1.2, .8], [190, -2.0, .9], [2800, 1.0, .7]]},
        'kick_out': {'highpass_hz': 28, 'lowpass_hz': 1400,
                     'eq': [[190, -2.5, .8]]},
        'out_relative_match_gain_db': match,
        'measured_out_relative_match_gain_db': float(measured_match),
        'match_reestimated': frozen_match_gain_db is None,
        'pre_compression_sha256': _digest(combined),
        'input_unchanged': True,
    }, match


def render_kick(kick_in: np.ndarray, kick_out: np.ndarray, *,
                compressor_config: CompressorConfig | None = None,
                frozen_controls: FrozenKickControls | None = None,
                recipe: BelyeStaiKickRecipe | None = None) -> KickStages:
    """Reproduce the delivered grouped kick or replace its sole compressor in place."""
    recipe = recipe or BelyeStaiKickRecipe()
    ki = _validate_mono(kick_in, 'KICK_IN', recipe)
    ko = _validate_mono(kick_out, 'KICK_OUT', recipe)
    in_hash, out_hash = _digest(ki), _digest(ko)

    if compressor_config is None:
        if frozen_controls is not None:
            raise ValueError('frozen controls are only valid with an explicit compressor replacement')
        pre, preparation, match = prepare_kick(ki, ko, recipe=recipe)
        y, comp, _ = _legacy_compressor(pre, recipe.sample_rate, 3.0, 3.0, 22, 115, 4.5)
        y, level = _level(y, recipe.sample_rate, -22.5, 92)
        controls = FrozenKickControls(in_hash, out_hash, _digest(pre), match, float(level['gain_db']))
        mode = 'frozen_delivery_recipe'
    else:
        if not isinstance(frozen_controls, FrozenKickControls):
            raise ValueError('kick replacement requires source-bound baseline controls')
        if frozen_controls.kick_in_pcm_sha256 != in_hash or frozen_controls.kick_out_pcm_sha256 != out_hash:
            raise ValueError('frozen kick controls belong to a different raw microphone pair')
        pre, preparation, _ = prepare_kick(
            ki, ko, recipe=recipe, frozen_match_gain_db=frozen_controls.out_relative_match_gain_db)
        if frozen_controls.pre_compression_sha256 != _digest(pre):
            raise ValueError('frozen kick controls belong to a different preparation')
        if not np.isfinite(frozen_controls.final_gain_db) or not -30 <= frozen_controls.final_gain_db <= 30:
            raise ValueError('invalid frozen kick final gain')
        compressor_config.validate(recipe.sample_rate)
        y, gr = LinkedCompressor(recipe.sample_rate, compressor_config).process(pre)
        comp = {
            'mode': 'compression_director_replacement',
            'config': asdict(compressor_config),
            'observed_max_gr_db': float(np.max(gr)) if len(gr) else 0.0,
            'whole_track_p95_gr_db': float(np.percentile(gr, 95)) if len(gr) else 0.0,
        }
        y = _gain(y, frozen_controls.final_gain_db)
        level = {'mode': 'frozen_baseline_static_gain', 'gain_db': frozen_controls.final_gain_db,
                 'level_reestimated': False, 'audition_level_match_applied': False}
        controls = frozen_controls
        mode = 'replacement'

    return KickStages(pre, y, controls, {
        'schema': 'belye-stai-kick-local-processor-v1',
        'source_group': ['KICK_IN', 'KICK_OUT'],
        'sample_rate': recipe.sample_rate,
        'frames': recipe.frames,
        'source_sha256': {'KICK_IN': in_hash, 'KICK_OUT': out_hash},
        'processed_sha256': _digest(y),
        'precompression': preparation,
        'compressor_mode': mode,
        'compressor': comp,
        'level': level,
        'frozen_controls': asdict(controls),
        'synchronized_mic_group': True,
        'compression_stage_count': 1,
        'requires_full_session_rerender': compressor_config is not None,
        'requires_human_review': compressor_config is not None,
        'requires_human_listening': compressor_config is not None,
        'baseline_eligible': False,
        'neural_audio_used': False,
        'input_files_modified': False,
    })
