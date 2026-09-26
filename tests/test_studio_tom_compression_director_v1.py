from dataclasses import replace
import numpy as np
import pytest

from audio_workbench.mixing.compression import CompressorConfig
from audio_workbench.mixing.tom_compression_director import (
    TomGatePolicy, assess_against_baseline, propose_baseline_aware_candidates,
    tom_event_centers, tom_transient_evidence,
)


def fixture(seconds=20):
    sr=44100;t=np.arange(sr*seconds)/sr;x=np.zeros_like(t,dtype=np.float64)
    for i,p in enumerate(np.arange(.2,seconds-.2,.32)):
        dt=np.maximum(t-p,0);on=t>=p;amp=.28+.08*np.sin(i*.7)
        x += amp*np.sin(2*np.pi*135*dt)*np.exp(-dt*15)*on
        x += .15*amp*np.sin(2*np.pi*2100*dt)*np.exp(-dt*55)*on
    x += .0015*np.sin(2*np.pi*700*t)
    return x.astype('float32'),sr


def baseline_cfg():
    return CompressorConfig(threshold_dbfs=-36,ratio=2.6,attack_ms=18,release_ms=160,knee_db=5,max_gr_db=4,rms_ms=3)


def test_event_detection_and_evidence_are_deterministic():
    x,sr=fixture();a=tom_event_centers(x,sr);b=tom_event_centers(x,sr)
    assert len(a)>=16; np.testing.assert_array_equal(a,b)
    e=tom_transient_evidence(x,x,sr); assert e['event_count']==len(a)
    for key in ('body_level_spread_db','median_attack_body_db','median_body_tail_db','quiet_hit_level_dbfs'):
        assert np.isfinite(e[key])


def test_no_change_fails_improvement_gate():
    x,sr=fixture(); result=assess_against_baseline(x,x,x.copy(),sr,policy=replace(TomGatePolicy(),min_events_for_comparison=8))
    assert not result['technically_survives']; assert 'tom_body_stability_not_improved' in result['failures']
    assert result['requires_human_listening'] and not result['baseline_eligible']


def test_body_stability_improvement_can_survive_when_decay_is_preserved():
    x,sr=fixture();centers=tom_event_centers(x,sr);y=x.copy(); levels=[]
    for c in centers:
        lo=c+int(.035*sr);hi=min(len(x),c+int(.120*sr));seg=x[lo:hi].astype(float);levels.append(10*np.log10(max(float(np.mean(seg*seg)),1e-30)))
    target=float(np.median(levels))
    for c,level in zip(centers,levels):
        # Apply same local trim through body and tail so decay shape is preserved.
        lo=c+int(.035*sr);hi=min(len(y),c+int(.260*sr));delta=float(np.clip((target-level)*.75,-1.5,1.5));y[lo:hi]*=10**(delta/20)
    result=assess_against_baseline(x,x,y,sr,policy=replace(TomGatePolicy(),min_events_for_comparison=8,min_body_spread_improvement_db=.01,max_attack_body_loss_db=2.0,max_decay_shape_change_db=1.0,max_between_hit_floor_increase_db=2.0))
    assert result['body_spread_delta_db']<-.01
    assert 'tom_decay_shape_changed' not in result['failures']


def test_tail_collapse_is_rejected_and_candidate_family_is_bounded():
    x,sr=fixture();y=x.copy()
    for c in tom_event_centers(x,sr):
        lo=c+int(.140*sr);hi=min(len(y),c+int(.260*sr));y[lo:hi]*=.55
    result=assess_against_baseline(x,x,y,sr,policy=replace(TomGatePolicy(),min_events_for_comparison=8,min_body_spread_improvement_db=-100))
    assert 'tom_decay_shape_changed' in result['failures']
    family=propose_baseline_aware_candidates(sr,baseline_cfg())
    assert [c['id'] for c in family['candidates']]==['preserve_attack','tighter_body','longer_decay']
    for c in family['candidates']:
        q=c['compressor']; assert q['threshold_dbfs']==baseline_cfg().threshold_dbfs and q['ratio']==baseline_cfg().ratio
        assert q['knee_db']==baseline_cfg().knee_db and q['max_gr_db']==baseline_cfg().max_gr_db
        assert c['change_scope']=='attack_release_only' and not c['baseline_eligible']


def test_invalid_baseline_fails_closed():
    with pytest.raises(ValueError): propose_baseline_aware_candidates(44100,CompressorConfig(detector='peak'))
