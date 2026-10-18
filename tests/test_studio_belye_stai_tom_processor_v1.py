import numpy as np
import pytest

from audio_workbench.belye_stai_toms import (
    BelyeStaiTomRecipe, FrozenTomControls, TOM_SPECS, prepare_tom, render_tom,
)
from audio_workbench.mixing.compression import CompressorConfig


def source(seconds=1.2, sr=44100):
    t=np.arange(int(seconds*sr))/sr
    x=np.zeros_like(t,dtype=np.float32)
    for i,p in enumerate(np.arange(.10,seconds-.1,.24)):
        tt=np.maximum(t-p,0); on=t>=p
        x += (0.55+.05*np.sin(i))*np.sin(2*np.pi*130*tt)*np.exp(-tt*18)*on
    return x.astype(np.float32)


def recipe(x): return BelyeStaiTomRecipe(frames=len(x))


def test_all_tom_sources_render_and_report_frozen_preparation():
    x=source()
    for name,spec in TOM_SPECS.items():
        stages=render_tom(x,name,recipe=recipe(x))
        assert stages.processed.shape==x.shape
        assert stages.report['source']==name
        assert stages.report['precompression']['highpass_hz']==spec.highpass_hz
        assert stages.report['precompression']['pan_downstream']==spec.pan
        assert stages.report['compression_stage_count']==1
        assert not stages.report['baseline_eligible']


def test_replacement_requires_exact_source_bound_controls_and_frozen_gain():
    x=source(); r=recipe(x); base=render_tom(x,'TOM_1',recipe=r)
    cfg=CompressorConfig(threshold_dbfs=-35,ratio=2.6,attack_ms=22.5,release_ms=136,knee_db=5,max_gr_db=4,rms_ms=3)
    changed=render_tom(x,'TOM_1',compressor_config=cfg,frozen_controls=base.controls,recipe=r)
    assert changed.report['compressor_mode']=='replacement'
    assert changed.report['level']['gain_db']==base.controls.final_gain_db
    assert changed.report['requires_full_session_rerender'] and changed.report['requires_human_listening']
    with pytest.raises(ValueError):
        render_tom(x,'TOM_2',compressor_config=cfg,frozen_controls=base.controls,recipe=r)


def test_preparation_is_deterministic_and_invalid_source_fails_closed():
    x=source(); r=recipe(x)
    a,ra=prepare_tom(x,'FLOOR',recipe=r); b,rb=prepare_tom(x,'FLOOR',recipe=r)
    np.testing.assert_array_equal(a,b)
    assert ra['pre_compression_sha256']==rb['pre_compression_sha256']
    with pytest.raises(ValueError): prepare_tom(x,'RACK_TOM',recipe=r)


def test_mismatched_controls_and_invalid_recipe_fail_closed():
    x=source();r=recipe(x);base=render_tom(x,'TOM_1',recipe=r)
    cfg=CompressorConfig(threshold_dbfs=-35,ratio=2.6,attack_ms=22.5,release_ms=136,knee_db=5,max_gr_db=4,rms_ms=3)
    bad=FrozenTomControls('TOM_1','0'*64,base.controls.pre_compression_sha256,base.controls.final_gain_db)
    with pytest.raises(ValueError): render_tom(x,'TOM_1',compressor_config=cfg,frozen_controls=bad,recipe=r)
    with pytest.raises(ValueError): render_tom(x,'TOM_1',recipe=BelyeStaiTomRecipe(sample_rate=48000,frames=len(x)))
