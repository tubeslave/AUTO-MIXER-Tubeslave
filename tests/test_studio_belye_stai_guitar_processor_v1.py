import numpy as np
import pytest
from audio_workbench.belye_stai_guitar import BelyeStaiGuitarRecipe, prepare_guitar, render_guitar
from audio_workbench.belye_stai_vocal import _legacy_compressor, _level
from audio_workbench.mixing.compression import CompressorConfig

def source(n=44100,sr=44100):
    t=np.arange(n)/sr
    env=.08+.04*(.5+.5*np.sin(2*np.pi*2*t))
    return (env*(np.sin(2*np.pi*220*t)+.32*np.sin(2*np.pi*440*t))).astype(np.float32)

def test_no_change_exact_legacy_chain():
    r=BelyeStaiGuitarRecipe(frames=44100); x=source(); pre,_=prepare_guitar(x,recipe=r)
    expected,_,_=_legacy_compressor(pre,r.sample_rate,1.5,2.,25,180,2.8)
    expected,_=_level(expected,r.sample_rate,-28.3,60)
    got=render_guitar(x,recipe=r)
    assert np.array_equal(got.processed,expected)
    assert got.report['baseline_eligible'] is False

def test_replacement_is_source_bound():
    r=BelyeStaiGuitarRecipe(frames=44100); x=source(); base=render_guitar(x,recipe=r)
    cfg=CompressorConfig(threshold_dbfs=-24,ratio=2,attack_ms=30,release_ms=180,knee_db=5,max_gr_db=2.8,rms_ms=3)
    cand=render_guitar(x,recipe=r,compressor_config=cfg,frozen_controls=base.controls)
    assert cand.report['requires_full_session_rerender'] is True
    with pytest.raises(ValueError):
        render_guitar(x*.9,recipe=r,compressor_config=cfg,frozen_controls=base.controls)
