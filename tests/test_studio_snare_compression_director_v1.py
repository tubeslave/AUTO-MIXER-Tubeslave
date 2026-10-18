from dataclasses import replace
import numpy as np
import pytest

from audio_workbench.mixing.compression import CompressorConfig
from audio_workbench.mixing.snare_compression_director import (
    SnareGatePolicy, assess_against_baseline, propose_baseline_aware_candidates,
    snare_event_centers, snare_transient_evidence,
)


def fixture(seconds=8):
    sr = 44100
    t = np.arange(sr*seconds)/sr
    x = np.zeros_like(t, dtype=np.float64)
    centers=[]
    for i,p in enumerate(np.arange(.15, seconds-.15, .20)):
        centers.append(int(round(p*sr)))
        dt=np.maximum(t-p,0); on=t>=p
        amp=.20 if i%5==4 else (.34+.03*np.sin(i))
        x += amp*np.sin(2*np.pi*220*dt)*np.exp(-dt*28)*on
        x += .22*amp*np.sin(2*np.pi*2500*dt)*np.exp(-dt*65)*on
    x += .002*np.sin(2*np.pi*6000*t)
    return x.astype('float32'), sr


def test_event_detection_and_evidence_are_deterministic():
    x,sr=fixture(); a=snare_event_centers(x,sr); b=snare_event_centers(x,sr)
    assert len(a) >= 25
    np.testing.assert_array_equal(a,b)
    e=snare_transient_evidence(x,x,sr)
    assert e['event_count'] == len(a)
    assert e['body_level_spread_db'] > 0
    assert np.isfinite(e['ghost_hit_level_dbfs'])


def test_no_change_fails_improvement_gate():
    x,sr=fixture()
    result=assess_against_baseline(x,x,x.copy(),sr,policy=replace(SnareGatePolicy(),min_events_for_comparison=8))
    assert not result['technically_survives']
    assert 'snare_body_stability_not_improved' in result['failures']
    assert result['requires_human_listening']
    assert not result['baseline_eligible']


def test_body_stability_improvement_can_survive_without_attack_loss():
    x,sr=fixture(); centers=snare_event_centers(x,sr)
    y=x.copy()
    # Pull louder event bodies toward a common level while leaving attack windows untouched.
    body_levels=[]
    for c in centers:
        lo=c+int(.022*sr); hi=min(len(x),c+int(.085*sr))
        seg=x[lo:hi].astype(np.float64)
        body_levels.append(10*np.log10(max(float(np.mean(seg*seg)),1e-30)))
    target=float(np.median(body_levels))
    for c,level in zip(centers,body_levels):
        lo=c+int(.022*sr); hi=min(len(x),c+int(.085*sr))
        delta=float(np.clip((target-level)*.8,-2,2))
        y[lo:hi]*=10**(delta/20)
    result=assess_against_baseline(x,x,y,sr,policy=replace(
        SnareGatePolicy(), min_events_for_comparison=8,
        min_body_spread_improvement_db=.02, max_attack_body_loss_db=2.0))
    assert result['body_spread_delta_db'] < -.02
    assert 'quiet_snare_hits_reduced' not in result['failures']


def test_candidate_family_is_bounded_timing_only():
    sr=44100; cfg=CompressorConfig(threshold_dbfs=-37,ratio=3,attack_ms=14,release_ms=110,knee_db=5,max_gr_db=5,rms_ms=3)
    result=propose_baseline_aware_candidates(sr,cfg)
    assert result['reference']['id']=='no_change'
    assert [c['id'] for c in result['candidates']] == ['preserve_crack','tighter_body','longer_body']
    for c in result['candidates']:
        q=c['compressor']
        assert q['threshold_dbfs']==cfg.threshold_dbfs and q['ratio']==cfg.ratio
        assert q['knee_db']==cfg.knee_db and q['max_gr_db']==cfg.max_gr_db and q['rms_ms']==cfg.rms_ms
        assert c['change_scope']=='attack_release_only'
        assert not c['baseline_eligible']


def test_invalid_baseline_config_fails_closed():
    with pytest.raises(ValueError):
        propose_baseline_aware_candidates(44100,CompressorConfig(detector='peak'))
