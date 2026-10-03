import json

import numpy as np

from audio_workbench.mixing.bass_compression_director import (
    BassBaselinePolicy,
    assess_against_baseline,
    bass_stability_evidence,
    propose_baseline_aware_candidates,
    render_candidate,
)
from audio_workbench.mixing.compression import CompressorConfig


def bass_events(sr=4000, seconds=8):
    x=np.zeros(sr*seconds,dtype='float32')
    rng=np.random.default_rng(7)
    for i,t in enumerate(np.arange(.5,seconds-.4,.48)):
        start=int(t*sr); n=min(int(.30*sr),len(x)-start)
        tt=np.arange(n)/sr
        amp=.18+.08*(i%4)+float(rng.uniform(-.01,.01))
        env=(1-np.exp(-tt/.008))*np.exp(-tt/.22)
        x[start:start+n]+=amp*env*np.sin(2*np.pi*82*tt).astype('float32')
    return x


def baseline_config():
    return CompressorConfig(
        threshold_dbfs=-24, ratio=3.5, attack_ms=9, release_ms=130,
        knee_db=5, max_gr_db=5, detector='rms', rms_ms=3,
    )


def test_baseline_aware_family_is_bounded_and_has_no_winner():
    x=bass_events(); p=propose_baseline_aware_candidates(x,4000,baseline_config(),4.2)
    assert p['schema']=='bass-compression-director-baseline-aware-v1'
    assert [c['id'] for c in p['candidates']]==['faster_recovery','longer_recovery','quicker_attack_longer_release']
    assert p['reference']['id']=='no_change' and p['reference']['compressor']==baseline_config().__dict__
    assert 'winner' not in p and 'ranking' not in p
    for c in p['candidates']:
        cfg=c['compressor']; rel=c['baseline_relative']
        assert .70 <= rel['attack_factor'] <= 1.30
        assert .55 <= rel['release_factor'] <= 1.45
        assert cfg['threshold_dbfs']==baseline_config().threshold_dbfs
        assert cfg['ratio']==baseline_config().ratio and cfg['knee_db']==baseline_config().knee_db
        assert cfg['max_gr_db']==5 and cfg['rms_ms']==baseline_config().rms_ms
        assert rel['threshold_delta_db']==0 and rel['ratio_delta']==0
        assert c['change_scope']=='attack_release_only'
        assert c['baseline_eligible'] is False and c['requires_human_listening'] is True
    json.dumps(p,allow_nan=False)


def test_render_hits_calibrated_target_and_preserves_input():
    x=bass_events(); before=x.copy()
    p=propose_baseline_aware_candidates(x,4000,baseline_config(),4.2)
    for c in p['candidates']:
        y,r=render_candidate(x,4000,c)
        assert y.shape==x.shape and np.isfinite(y).all()
        assert 0 <= r['active_p95_gr_db'] <= 5.0001
        assert r['max_gr_db'] <= 5.0001
    np.testing.assert_array_equal(x,before)


def test_stability_measurement_uses_fixed_detection_windows():
    x=bass_events(); base=x.copy(); cand=x.copy()
    centers=np.where(np.diff((np.abs(x)>.001).astype(int))==1)[0]
    # Reduce only alternate event bodies while leaving event starts alone.
    for k,c in enumerate(centers):
        if k%2:
            cand[c+120:c+480]*=.55
    a=bass_stability_evidence(x,base,4000)
    b=bass_stability_evidence(x,cand,4000)
    assert a['event_count']>=8 and b['event_count']==a['event_count']
    assert b['body_level_spread_db'] > a['body_level_spread_db']
    json.dumps(a,allow_nan=False)


def test_assessment_accepts_only_improved_body_with_protected_attack():
    x=bass_events(); base=x.copy(); cand=x.copy()
    centers=np.where(np.diff((np.abs(x)>.001).astype(int))==1)[0]
    # Normalize body amplitudes toward a common value without touching attack windows.
    for c in centers:
        a=c+120; b=min(c+460,len(cand))
        seg=cand[a:b]
        r=float(np.sqrt(np.mean(seg.astype('float64')**2)+1e-30))
        if r>1e-8:
            cand[a:b]*=np.clip(.075/r,.7,1.35)
    result=assess_against_baseline(x,base,cand,4000,
        policy=BassBaselinePolicy(min_body_spread_improvement_db=.01,max_attack_body_loss_db=20))
    assert result['technically_survives'] is True
    assert result['body_spread_delta_db'] < 0
    assert result['baseline_eligible'] is False


def test_assessment_rejects_worse_body_spread():
    x=bass_events(); base=x.copy(); cand=x.copy()
    centers=np.where(np.diff((np.abs(x)>.001).astype(int))==1)[0]
    for k,c in enumerate(centers):
        if k%2:
            cand[c+120:c+480]*=.45
    result=assess_against_baseline(x,base,cand,4000)
    assert result['technically_survives'] is False
    assert 'body_stability_not_improved' in result['failures']
    assert 'winner' not in result
