"""Static, temporal, stereo and evidence regression tests, no console access."""
from dataclasses import replace
import json

import numpy as np
import pytest

from audio_workbench.mixing.compression import (
    CompressorConfig, LinkedCompressor, _smooth_gr, static_reduction,
)
from audio_workbench.mixing import dynamics


def tone(sr=48000, seconds=1.0, amplitude=.2):
    return (amplitude * np.sin(2*np.pi*997*np.arange(int(sr*seconds))/sr)).astype('float32')


def test_known_hard_knee_static_ratio():
    cfg=CompressorConfig(threshold_dbfs=-24,ratio=4,knee_db=0,max_gr_db=30)
    np.testing.assert_allclose(static_reduction(np.array([-30,-24,-20,-12,0]),cfg),[0,0,3,9,18])


def test_soft_knee_boundary_continuity_and_monotonicity():
    cfg=CompressorConfig(threshold_dbfs=-24,ratio=4,knee_db=6,max_gr_db=30)
    levels=np.array([-27-1e-7,-27,-27+1e-7,-24,-21-1e-7,-21,-21+1e-7])
    gr=static_reduction(levels,cfg)
    assert np.all(np.diff(gr)>=0)
    assert gr[3]==pytest.approx(.5625)
    assert abs(gr[0]-gr[2])<1e-6 and abs(gr[-1]-gr[-3])<1e-6


@pytest.mark.parametrize('sr',[8000,44100,48000,96000])
def test_attack_release_have_declared_independent_time_constants(sr):
    for attack_ms,release_ms in [(1,10),(10,100),(100,500)]:
        na=round(sr*attack_ms/1000);nr=round(sr*release_ms/1000)
        a=np.exp(-1/(sr*attack_ms/1000));r=np.exp(-1/(sr*release_ms/1000))
        gr,_=_smooth_gr(np.full(na,12.),a,r,0.)
        assert gr[-1]==pytest.approx(12*(1-np.exp(-na/(sr*attack_ms/1000))),abs=2e-6)
        gr,_=_smooth_gr(np.zeros(nr),a,r,12.)
        assert gr[-1]==pytest.approx(12*np.exp(-nr/(sr*release_ms/1000)),abs=2e-6)


def test_python_and_optional_compiler_same_recurrence():
    if not hasattr(_smooth_gr,'py_func'):
        pytest.skip('optional compiler absent; Python path exercised by other tests')
    req=np.random.default_rng(31).uniform(0,8,4000)
    fast,fs=_smooth_gr(req,.97,.999,2.)
    slow,ss=_smooth_gr.py_func(req,.97,.999,2.)
    np.testing.assert_array_equal(fast,slow);assert fs==ss


def test_peak_detector_known_steady_reduction_and_latency():
    x=np.full(48000,10**(-12/20),dtype='float32')
    cfg=CompressorConfig(threshold_dbfs=-24,ratio=4,knee_db=0,max_gr_db=30,detector='peak')
    y,gr=LinkedCompressor(48000,cfg).process(x)
    assert gr[-1]==pytest.approx(9,abs=1e-5)
    assert np.all(y>0)  # no delayed audio samples / no lookahead
    assert y[-1]==pytest.approx(10**(-21/20),rel=1e-5)


def test_loud_burst_does_not_change_audio_before_it_arrives():
    cfg=CompressorConfig(threshold_dbfs=-24,detector='peak',knee_db=0)
    x=np.full(48000,.005,dtype='float32');x[24000:]=.8
    y,gr=LinkedCompressor(48000,cfg).process(x)
    np.testing.assert_array_equal(y[:24000],x[:24000])
    assert np.max(gr[:24000])==0 and gr[24001]>0


@pytest.mark.parametrize('detector',['rms','peak'])
def test_antiphase_stereo_has_identical_compression(detector):
    x=tone();cfg=CompressorConfig(threshold_dbfs=-30,detector=detector)
    same=np.column_stack([x,x]);opposite=np.column_stack([x,-x])
    a,ga=LinkedCompressor(48000,cfg).process(same)
    b,gb=LinkedCompressor(48000,cfg).process(opposite)
    np.testing.assert_array_equal(ga,gb)
    np.testing.assert_array_equal(a[:,0],b[:,0]);np.testing.assert_array_equal(a[:,1],-b[:,1])
    assert ga.max()>1


@pytest.mark.parametrize('detector',['rms','peak'])
def test_full_and_irregular_blocks_match_with_external_filtered_sidechain(detector):
    rng=np.random.default_rng(7);x=rng.normal(0,.15,(12123,2)).astype('float32');original=x.copy()
    sc=tone(seconds=len(x)/48000);cfg=CompressorConfig(threshold_dbfs=-30,sidechain_hpf_hz=80,detector=detector)
    whole,wgr=LinkedCompressor(48000,cfg).process(x,sc)
    proc=LinkedCompressor(48000,cfg);ys=[];gs=[]
    points=[0,1,5,111,3000,3001,4800,12123]
    for a,b in zip(points[:-1],points[1:]):
        y,g=proc.process(x[a:b],sc[a:b]);ys.append(y);gs.append(g)
    np.testing.assert_array_equal(whole,np.concatenate(ys))
    np.testing.assert_array_equal(wgr,np.concatenate(gs))
    np.testing.assert_array_equal(x,original)


def test_linked_stereo_and_gr_cap():
    x=tone(amplitude=2);stereo=np.column_stack([x,x*.25])
    y,gr=LinkedCompressor(48000,CompressorConfig(threshold_dbfs=-50,max_gr_db=3)).process(stereo)
    assert gr.max()<=3
    np.testing.assert_array_equal(y[:,1],y[:,0]*.25)
    assert np.all(np.abs(y)<=np.abs(stereo))


def test_sidechain_hpf_does_not_filter_program_and_reduces_sub_trigger():
    sr=48000;x=(.4*np.sin(2*np.pi*30*np.arange(sr)/sr)).astype('float32')
    cfg=CompressorConfig(threshold_dbfs=-24,max_gr_db=12)
    _,plain=LinkedCompressor(sr,cfg).process(x)
    y,filtered=LinkedCompressor(sr,replace(cfg,sidechain_hpf_hz=150)).process(x)
    assert filtered[sr//2:].mean()<plain[sr//2:].mean()-4
    np.testing.assert_allclose(y,x*10**(-filtered/20),atol=1e-7)


@pytest.mark.parametrize('shape',[(0,),(1,),(480,),(48000,),(0,2),(480,2)])
def test_silence_and_short_sources_do_not_fail(shape):
    x=np.zeros(shape,'float32');y,report=dynamics.apply(x,48000,'vocal')
    np.testing.assert_array_equal(x,y)
    assert report['max_gr_db']==0 and report['makeup_db']==0
    assert report['status']=='no_op'
    json.dumps(report,allow_nan=False)


def test_short_audible_tail_not_dropped():
    x=tone(seconds=.013);y,report=dynamics.apply(x,48000,'vocal')
    assert y.shape==x.shape and np.isfinite(y).all()
    assert report['active_frames']==1


@pytest.mark.parametrize('field,value',[('ratio',.8),('ratio',np.nan),('attack_ms',0),
 ('release_ms',-1),('knee_db',-1),('max_gr_db',np.inf),('sidechain_hpf_hz',25000),('detector','magic')])
def test_invalid_config_fails_closed(field,value):
    with pytest.raises(ValueError):LinkedCompressor(48000,replace(CompressorConfig(),**{field:value}))


@pytest.mark.parametrize('bad',[np.array([np.nan],dtype='float32'),np.zeros((2,100),'float32'),
 np.zeros(10,dtype='int16'),np.zeros((10,3),'float32'),np.array([1e99],dtype='float64')])
def test_invalid_pcm_fails_closed(bad):
    with pytest.raises(ValueError):LinkedCompressor(48000).process(bad)


@pytest.mark.parametrize('change',[{'bypass':True},{'ratio':1},{'max_gr_db':0}])
def test_explicit_bypass_is_exact_and_disables_implicit_rider_makeup(change):
    x=tone();cfg=replace(CompressorConfig(),**change)
    y,report=dynamics.apply(x,48000,'vocal',compressor_config=cfg)
    np.testing.assert_array_equal(y,x);assert report['makeup_db']==0


def test_adapter_stats_are_actual_pcm_not_gain_curve_predictions():
    x=tone();x[:24000]*=.15;original=x.copy()
    y,r=dynamics.apply(x,48000,'vocal',enable_ride=False)
    np.testing.assert_array_equal(x,original)
    p,counts=dynamics._frame_power(x,960);q,_=dynamics._frame_power(y,960)
    db=dynamics._db(p);active=db>=max(-100,float(max(db))-40,float(np.percentile(db,55)))
    expected=dynamics._spread(dynamics._db(q),active)
    assert r['after_spread_db']==pytest.approx(expected,abs=1e-8)
    assert abs(r['active_rms_delta_db'])<.01 or r['makeup_limited']
    assert r['max_gr_db']<=r['compressor']['max_gr_db']
    assert r['rider']['enabled'] is False
    assert r['baseline_eligible'] is False
    json.dumps(r,allow_nan=False)


def test_makeup_never_bypasses_peak_headroom():
    x=tone(amplitude=.8);x[12345]=1.0
    y,r=dynamics.apply(x,48000,'vocal',enable_ride=False,makeup_limit_db=6)
    from audio_workbench.mastering.analyzer import true_peak_dbtp
    if r['makeup_db']>0:assert true_peak_dbtp(y)<=-.999


def test_adapter_phase_invariance_and_frame_api():
    x=tone();x[24000:]*=.3
    a,ra=dynamics.apply(np.column_stack([x,x]),48000,'vocal')
    b,rb=dynamics.apply(np.column_stack([x,-x]),48000,'vocal')
    np.testing.assert_array_equal(a[:,0],b[:,0]);assert ra['max_gr_db']==rb['max_gr_db']
    frames=dynamics.analyze_frames(x,48000,'vocal')
    assert len(frames['db'])==50 and len(frames['gr_db'])==50


def test_reset_and_bad_sidechain_or_layout():
    p=LinkedCompressor(48000);x=tone()
    a,g=p.process(x);p.reset();b,h=p.process(x)
    np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError):p.process(np.column_stack([x,x]))
    with pytest.raises(ValueError):p.process(x,x[:-1])
