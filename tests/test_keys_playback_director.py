import numpy as np
import pytest

from audio_workbench.mixing.compression import CompressorConfig, LinkedCompressor
from audio_workbench.mixing.keys_playback_director import (
    KeysPlaybackPolicy,
    assess_against_baseline,
    baseline_actionability,
    dynamics_evidence,
    playback_section_ride,
    propose_compression_candidates,
)


def _stable_stereo(sr=8000, seconds=24, amp=.12):
    n = int(sr * seconds); t = np.arange(n) / sr
    l = amp * (np.sin(2*np.pi*220*t) + .35*np.sin(2*np.pi*1100*t))
    r = amp * (np.sin(2*np.pi*220*t + .15) + .35*np.sin(2*np.pi*1100*t + .45))
    for c in range(sr, n-sr, sr):
        m = min(160, n-c); env = np.exp(-np.arange(m)/(sr*.006))
        l[c:c+m] += .18*env; r[c:c+m] += .16*env
    return np.column_stack([l, r]).astype(np.float32)


def _dynamic_keys(sr=8000, seconds=24):
    n = int(sr*seconds); x=np.zeros((n,2),np.float32)
    amps=[.035,.16,.05,.21,.045,.14,.06,.19]
    for i,c in enumerate(range(sr//2,n-sr//2,sr//2)):
        a=amps[i%len(amps)]; length=min(int(.38*sr),n-c); q=np.arange(length)/sr
        env=np.exp(-q/0.24); tone=np.sin(2*np.pi*330*q)+.28*np.sin(2*np.pi*1650*q)
        click=np.exp(-q/.005)*np.sin(2*np.pi*2400*q)
        x[c:c+length,0]+=a*env*tone+.10*a/.16*click
        x[c:c+length,1]+=a*env*(np.sin(2*np.pi*330*q+.12)+.28*np.sin(2*np.pi*1650*q+.4))+.09*a/.16*click
    return x


def test_playback_ride_bounded():
    db=np.array([-30,-24,-28,-25,-29,-26],dtype=float);active=np.ones(len(db),dtype=bool)
    r=playback_section_ride(db,active,.6)
    assert np.max(np.abs(r))<=.601


def test_role_evidence_requires_explicit_stereo():
    mono=np.ones(8000,dtype=np.float32)*.1
    with pytest.raises(ValueError, match="explicit stereo"):
        dynamics_evidence(mono,mono,8000)


def test_stable_playback_is_no_change_first():
    x=_stable_stereo()
    action=baseline_actionability("playback",x,x,8000)
    director=propose_compression_candidates("playback",8000,CompressorConfig(),action)
    assert not action["actionable"]
    assert director["decision"]=="no_change"
    assert director["candidates"]==[]


def test_keys_with_large_local_spread_can_be_actionable_and_get_bounded_candidates():
    x=_dynamic_keys(); policy=KeysPlaybackPolicy(keys_actionable_local_spread_db=2.0)
    action=baseline_actionability("keys",x,x,8000,policy=policy)
    assert action["actionable"], action
    cfg=CompressorConfig(threshold_dbfs=-26,ratio=2.0,attack_ms=30,release_ms=180,knee_db=5,max_gr_db=3)
    proposal=propose_compression_candidates("keys",8000,cfg,action,policy=policy)
    assert proposal["decision"]=="evaluate_bounded_candidates"
    assert [c["id"] for c in proposal["candidates"]]==["preserve_attack","balanced","tighten_body"]
    assert all(c["compressor"]["threshold_dbfs"]==cfg.threshold_dbfs for c in proposal["candidates"])
    assert all(c["compressor"]["ratio"]==cfg.ratio for c in proposal["candidates"])


def test_playback_macro_programming_blocks_compressor_and_points_to_ride_review():
    x=_stable_stereo(seconds=24); y=x.copy(); q=len(y)//4
    for i,g in enumerate([.45,1.0,.55,1.25]): y[i*q:(i+1)*q]*=g
    policy=KeysPlaybackPolicy(playback_actionable_local_spread_db=0.1,
                              playback_max_macro_spread_for_compression_db=.8,
                              playback_section_ride_review_db=.8,
                              min_active_blocks=5, playback_min_transient_events=1)
    action=baseline_actionability("playback",y,y,8000,policy=policy)
    assert not action["actionable"]
    assert "playback_macro_dynamics_may_be_programmed" in action["failures"]
    assert action["alternate_action"]=="review_section_level_ride_not_compression"


def test_candidate_assessment_rejects_stereo_image_change():
    x=_dynamic_keys(); policy=KeysPlaybackPolicy(keys_actionable_local_spread_db=1.0,
                                                  keys_min_local_spread_improvement_db=0.0,
                                                  max_side_mid_change_db=.05,
                                                  max_correlation_change=.02)
    cand=x.copy(); cand[:,1]=cand[:,0]
    out=assess_against_baseline("keys",x,x,cand,8000,policy=policy)
    assert not out["technically_survives"]
    assert any("stereo_" in f for f in out["failures"])


def test_real_linked_candidate_keeps_stereo_link_and_returns_finite_evidence():
    x=_dynamic_keys(); cfg=CompressorConfig(threshold_dbfs=-30,ratio=2.3,attack_ms=30,release_ms=150,knee_db=5,max_gr_db=4)
    y,gr=LinkedCompressor(8000,cfg).process(x)
    e=dynamics_evidence(x,y,8000)
    assert np.isfinite(gr).all() and float(np.max(gr))>0
    assert np.isfinite([e["median_local_spread_db"],e["macro_spread_db"],e["median_attack_body_db"],e["side_mid_db"],e["stereo_correlation"]]).all()
