"""Frozen Belye Stai guitar insert for no-change-first compression experiments."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
from typing import Any
import numpy as np
from .belye_stai_vocal import _bell, _filter, _gain, _legacy_compressor, _level
from .mixing.compression import CompressorConfig, LinkedCompressor, as_audio

@dataclass(frozen=True)
class BelyeStaiGuitarRecipe:
    sample_rate: int = 44_100
    frames: int = 9_128_700

@dataclass(frozen=True)
class FrozenGuitarControls:
    raw_pcm_sha256: str
    pre_compression_sha256: str
    final_gain_db: float

@dataclass(frozen=True)
class GuitarStages:
    pre_compression: np.ndarray
    processed: np.ndarray
    controls: FrozenGuitarControls
    report: dict[str, Any]

def _digest(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype='<f4').tobytes()).hexdigest()

def _validate(raw: np.ndarray, recipe: BelyeStaiGuitarRecipe) -> np.ndarray:
    x = as_audio(raw)
    if recipe.sample_rate != 44_100 or isinstance(recipe.sample_rate, bool):
        raise ValueError('guitar recipe sample rate must be the frozen 44100 Hz')
    if isinstance(recipe.frames, bool) or not isinstance(recipe.frames,(int,np.integer)) or recipe.frames < 4410:
        raise ValueError('guitar recipe needs at least 100 ms of source frames')
    if x.ndim != 1 or len(x) != recipe.frames:
        raise ValueError('GTR raw source must be mono and match frozen frame count')
    if not len(x) or float(np.max(np.abs(x))) < 1e-10:
        raise ValueError('silent GTR source cannot define guitar processing')
    return x.copy()

def prepare_guitar(raw: np.ndarray, *, recipe: BelyeStaiGuitarRecipe | None=None):
    recipe = recipe or BelyeStaiGuitarRecipe(); x=_validate(raw,recipe)
    y=_filter(x, recipe.sample_rate, 90, 9000)
    y=_bell(y, recipe.sample_rate, 310, -1.8, .8)
    y=_bell(y, recipe.sample_rate, 3000, -.8, .8)
    return y, {'schema':'belye-stai-guitar-precompression-v1','highpass_hz':90,'lowpass_hz':9000,
               'eq':[[310,-1.8,.8],[3000,-.8,.8]],'pre_compression_sha256':_digest(y),'input_unchanged':True}

def render_guitar(raw: np.ndarray, *, compressor_config: CompressorConfig|None=None,
                  frozen_controls: FrozenGuitarControls|None=None,
                  recipe: BelyeStaiGuitarRecipe|None=None) -> GuitarStages:
    recipe=recipe or BelyeStaiGuitarRecipe(); x=_validate(raw,recipe); raw_hash=_digest(x)
    pre,prep=prepare_guitar(x,recipe=recipe); pre_hash=_digest(pre)
    if compressor_config is None:
        if frozen_controls is not None: raise ValueError('frozen controls require an explicit guitar compressor replacement')
        y,comp,_=_legacy_compressor(pre,recipe.sample_rate,1.5,2.0,25,180,2.8)
        y,level=_level(y,recipe.sample_rate,-28.3,60)
        controls=FrozenGuitarControls(raw_hash,pre_hash,float(level['gain_db'])); mode='frozen_delivery_recipe'
    else:
        if not isinstance(frozen_controls,FrozenGuitarControls): raise ValueError('guitar replacement requires source-bound baseline controls')
        if frozen_controls.raw_pcm_sha256 != raw_hash or frozen_controls.pre_compression_sha256 != pre_hash:
            raise ValueError('frozen guitar controls belong to a different source or preparation')
        if not np.isfinite(frozen_controls.final_gain_db) or not -30 <= frozen_controls.final_gain_db <= 30:
            raise ValueError('invalid frozen guitar final gain')
        compressor_config.validate(recipe.sample_rate)
        y,gr=LinkedCompressor(recipe.sample_rate,compressor_config).process(pre)
        comp={'mode':'compression_director_replacement','config':asdict(compressor_config),
              'observed_max_gr_db':float(np.max(gr)) if len(gr) else 0.0,
              'whole_track_p95_gr_db':float(np.percentile(gr,95)) if len(gr) else 0.0}
        y=_gain(y,frozen_controls.final_gain_db)
        level={'mode':'frozen_baseline_static_gain','gain_db':frozen_controls.final_gain_db,
               'level_reestimated':False,'audition_level_match_applied':False}
        controls=frozen_controls; mode='replacement'
    return GuitarStages(pre,y,controls,{'schema':'belye-stai-guitar-local-processor-v1','source':'GTR',
        'sample_rate':recipe.sample_rate,'frames':recipe.frames,'source_sha256':raw_hash,'processed_sha256':_digest(y),
        'precompression':prep,'compressor_mode':mode,'compressor':comp,'level':level,'frozen_controls':asdict(controls),
        'compression_stage_count':1,'pan_downstream':-.23,'requires_full_session_rerender':compressor_config is not None,
        'requires_human_review':compressor_config is not None,'requires_human_listening':compressor_config is not None,
        'baseline_eligible':False,'neural_audio_used':False,'input_files_modified':False})
