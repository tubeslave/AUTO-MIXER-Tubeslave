"""Frozen Belye Stai grouped-snare insert for source-specific compression experiments.

SN_T / SN_B remain a synchronized microphone group.  The bottom-mic polarity,
EQ, expansion and relative balance are part of the frozen preparation stage.
The no-override path reproduces the delivered local SNARE processing exactly.
Compression replacements occur only at the original group-compressor insert and
keep the frozen final gain.  No candidate is a musical winner or baseline by
construction; full-session rerender and human listening remain mandatory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any

import numpy as np

from .belye_stai_vocal import (
    _active_level, _bell, _expand, _filter, _gain, _legacy_compressor, _level,
)
from .mixing.compression import CompressorConfig, LinkedCompressor, as_audio


@dataclass(frozen=True)
class BelyeStaiSnareRecipe:
    sample_rate: int = 44_100
    frames: int = 9_128_700


@dataclass(frozen=True)
class FrozenSnareControls:
    snare_top_pcm_sha256: str
    snare_bottom_pcm_sha256: str
    bottom_processed_sha256: str
    pre_compression_sha256: str
    bottom_relative_match_gain_db: float
    final_gain_db: float


@dataclass(frozen=True)
class SnareStages:
    pre_compression: np.ndarray
    processed: np.ndarray
    controls: FrozenSnareControls
    report: dict[str, Any]


def _digest(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype='<f4').tobytes()).hexdigest()


def _validate_mono(x: np.ndarray, name: str, recipe: BelyeStaiSnareRecipe) -> np.ndarray:
    y = as_audio(x)
    if recipe.sample_rate != 44_100 or isinstance(recipe.sample_rate, bool):
        raise ValueError('snare recipe sample rate must be the frozen 44100 Hz')
    if isinstance(recipe.frames, bool) or not isinstance(recipe.frames, (int, np.integer)) or recipe.frames < 4410:
        raise ValueError('snare recipe needs at least 100 ms of source frames')
    if y.ndim != 1 or len(y) != recipe.frames:
        raise ValueError(f'{name} must be mono and match frozen frame count')
    if not len(y) or np.max(np.abs(y)) < 1e-10:
        raise ValueError(f'silent {name} cannot define grouped snare processing')
    return y.copy()


def prepare_snare(snare_top: np.ndarray, snare_bottom: np.ndarray, *,
                  recipe: BelyeStaiSnareRecipe | None = None,
                  frozen_match_gain_db: float | None = None
                  ) -> tuple[np.ndarray, dict[str, Any], float, str]:
    """Raw pair -> frozen polarity/EQ/bottom expansion/mic balance -> compressor."""
    recipe = recipe or BelyeStaiSnareRecipe()
    st = _validate_mono(snare_top, 'SN_T', recipe)
    sb = _validate_mono(snare_bottom, 'SN_B', recipe)

    # Frozen delivered topology: bottom polarity flip before any local filtering.
    sb = -sb
    st = _filter(st, recipe.sample_rate, 85, 12_500)
    sb = _filter(sb, recipe.sample_rate, 220, 11_500)
    st = _bell(_bell(st, recipe.sample_rate, 380, -2.0), recipe.sample_rate, 2500, 2.0, .7)
    sb, expansion = _expand(
        sb, recipe.sample_rate, 600, 7000,
        threshold_q=25, margin_db=9, floor=-8, release_s=.16, hold_s=.10,
    )
    bottom_hash = _digest(sb)

    measured_match = (
        _active_level(st, recipe.sample_rate, 94)
        - _active_level(sb, recipe.sample_rate, 94)
        - 11.0
    )
    match = measured_match if frozen_match_gain_db is None else float(frozen_match_gain_db)
    if not np.isfinite(match) or not -30 <= match <= 12:
        raise ValueError('invalid snare-bottom relative match gain')
    combined = (st + _gain(sb, match)).astype(np.float32)
    return combined, {
        'schema': 'belye-stai-snare-precompression-v1',
        'polarity': {'SN_B': -1},
        'snare_top': {
            'highpass_hz': 85, 'lowpass_hz': 12_500,
            'eq': [[380, -2.0, 1.0], [2500, 2.0, .7]],
        },
        'snare_bottom': {
            'highpass_hz': 220, 'lowpass_hz': 11_500,
            'expansion': expansion,
            'processed_sha256': bottom_hash,
        },
        'bottom_relative_match_gain_db': match,
        'measured_bottom_relative_match_gain_db': float(measured_match),
        'match_reestimated': frozen_match_gain_db is None,
        'pre_compression_sha256': _digest(combined),
        'input_unchanged': True,
    }, match, bottom_hash


def render_snare(snare_top: np.ndarray, snare_bottom: np.ndarray, *,
                 compressor_config: CompressorConfig | None = None,
                 frozen_controls: FrozenSnareControls | None = None,
                 recipe: BelyeStaiSnareRecipe | None = None) -> SnareStages:
    """Reproduce delivered grouped snare or replace its sole compressor in place."""
    recipe = recipe or BelyeStaiSnareRecipe()
    st = _validate_mono(snare_top, 'SN_T', recipe)
    sb = _validate_mono(snare_bottom, 'SN_B', recipe)
    top_hash, bottom_raw_hash = _digest(st), _digest(sb)

    if compressor_config is None:
        if frozen_controls is not None:
            raise ValueError('frozen controls are only valid with an explicit compressor replacement')
        pre, preparation, match, bottom_processed_hash = prepare_snare(st, sb, recipe=recipe)
        y, comp, _ = _legacy_compressor(pre, recipe.sample_rate, 3.5, 3.0, 14, 110, 5.0)
        y, level = _level(y, recipe.sample_rate, -23.8, 94)
        controls = FrozenSnareControls(
            top_hash, bottom_raw_hash, bottom_processed_hash, _digest(pre), match, float(level['gain_db']))
        mode = 'frozen_delivery_recipe'
    else:
        if not isinstance(frozen_controls, FrozenSnareControls):
            raise ValueError('snare replacement requires source-bound baseline controls')
        if (frozen_controls.snare_top_pcm_sha256 != top_hash
                or frozen_controls.snare_bottom_pcm_sha256 != bottom_raw_hash):
            raise ValueError('frozen snare controls belong to a different raw microphone pair')
        pre, preparation, _, bottom_processed_hash = prepare_snare(
            st, sb, recipe=recipe,
            frozen_match_gain_db=frozen_controls.bottom_relative_match_gain_db,
        )
        if bottom_processed_hash != frozen_controls.bottom_processed_sha256:
            raise ValueError('frozen snare controls belong to different bottom-mic processing')
        if _digest(pre) != frozen_controls.pre_compression_sha256:
            raise ValueError('frozen snare controls belong to a different preparation')
        if not np.isfinite(frozen_controls.final_gain_db) or not -30 <= frozen_controls.final_gain_db <= 30:
            raise ValueError('invalid frozen snare final gain')
        compressor_config.validate(recipe.sample_rate)
        y, gr = LinkedCompressor(recipe.sample_rate, compressor_config).process(pre)
        comp = {
            'mode': 'compression_director_replacement',
            'config': asdict(compressor_config),
            'observed_max_gr_db': float(np.max(gr)) if len(gr) else 0.0,
            'whole_track_p95_gr_db': float(np.percentile(gr, 95)) if len(gr) else 0.0,
        }
        y = _gain(y, frozen_controls.final_gain_db)
        level = {
            'mode': 'frozen_baseline_static_gain',
            'gain_db': frozen_controls.final_gain_db,
            'level_reestimated': False,
            'audition_level_match_applied': False,
        }
        controls = frozen_controls
        mode = 'replacement'

    return SnareStages(pre, y, controls, {
        'schema': 'belye-stai-snare-local-processor-v1',
        'source_group': ['SN_T', 'SN_B'],
        'sample_rate': recipe.sample_rate,
        'frames': recipe.frames,
        'source_sha256': {'SN_T': top_hash, 'SN_B': bottom_raw_hash},
        'processed_sha256': _digest(y),
        'precompression': preparation,
        'compressor_mode': mode,
        'compressor': comp,
        'level': level,
        'frozen_controls': asdict(controls),
        'synchronized_mic_group': True,
        'bottom_polarity_frozen': True,
        'bottom_expansion_frozen': True,
        'compression_stage_count': 1,
        'requires_full_session_rerender': compressor_config is not None,
        'requires_human_review': compressor_config is not None,
        'requires_human_listening': compressor_config is not None,
        'baseline_eligible': False,
        'neural_audio_used': False,
        'input_files_modified': False,
    })
