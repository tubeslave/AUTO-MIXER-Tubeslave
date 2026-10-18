import json
import numpy as np
import pytest

from audio_workbench.mixing.compression import LinkedCompressor
from audio_workbench.mixing.compression_director import (
    POLICIES, analyze_events, audition_pair_plan, config_from_candidate,
    level_match_evidence, propose_candidates, render_core_candidate,
)


def pulses(spacing=.5, attack_ms=3, decay_ms=35, sr=48000, seconds=4):
    x=np.zeros(int(sr*seconds),dtype='float32')
    for center in np.arange(.3,seconds-.2,spacing):
        peak=int(center*sr); attack=max(1,int(attack_ms*sr/1000)); start=max(0,peak-attack)
        x[start:peak]=np.linspace(0,.8,peak-start,endpoint=False,dtype='float32')
        end=min(len(x),peak+int(.14*sr)); x[peak:end]=.8*np.exp(-np.arange(end-peak)/(sr*decay_ms/1000)).astype('float32')
    return x


def candidate(proposal, ident):
    return next(x for x in proposal['candidates'] if x['id']==ident)


def test_candidate_set_is_bounded_serializable_and_never_machine_accepted():
    proposal=propose_candidates(pulses(),48000,'snare')
    assert [x['id'] for x in proposal['candidates']]==['preserve_transient','balanced','control']
    assert proposal['baseline_eligible'] is False and proposal['requires_human_review'] is True
    for item in proposal['candidates']:
        cfg=config_from_candidate(item);cfg.validate(48000);policy=POLICIES['snare']
        assert policy.attack_bounds_ms[0]<=cfg.attack_ms<=policy.attack_bounds_ms[1]
        assert policy.release_bounds_ms[0]<=cfg.release_ms<=policy.release_bounds_ms[1]
        assert item['baseline_eligible'] is False and item['requires_human_listening'] is True
    json.dumps(proposal,allow_nan=False)


def test_slow_macro_onsets_propose_slower_attack_than_sharp_onsets():
    sharp=candidate(propose_candidates(pulses(attack_ms=2),48000,'snare'),'balanced')
    slow=candidate(propose_candidates(pulses(attack_ms=35),48000,'snare'),'balanced')
    assert slow['compressor']['attack_ms']>sharp['compressor']['attack_ms']


def test_dense_events_propose_faster_release_than_sparse_events():
    dense=candidate(propose_candidates(pulses(spacing=.12),48000,'snare'),'balanced')
    sparse=candidate(propose_candidates(pulses(spacing=.55),48000,'snare'),'balanced')
    assert dense['compressor']['release_ms']<sparse['compressor']['release_ms']


def test_event_analysis_is_stereo_polarity_invariant():
    x=pulses();same=np.column_stack([x,x]);opposite=np.column_stack([x,-x])
    assert analyze_events(same,48000,'snare')==analyze_events(opposite,48000,'snare')


def test_sustained_source_uses_deterministic_sparse_fallback():
    sr=48000;t=np.arange(sr*2)/sr;x=(.2*np.sin(2*np.pi*220*t)).astype('float32')
    a=propose_candidates(x,sr,'vocal');b=propose_candidates(x,sr,'vocal')
    assert a==b and a['analysis']['status']=='sparse_event_fallback'


def test_silence_uses_fallback_without_nan_or_exception():
    proposal=propose_candidates(np.zeros(48000,'float32'),48000,'vocal')
    assert proposal['analysis']['status']=='no_active_audio'
    json.dumps(proposal,allow_nan=False)


def test_director_candidate_runs_through_accepted_linked_compressor_and_preserves_input():
    x=pulses();original=x.copy();item=candidate(propose_candidates(x,48000,'snare'),'balanced')
    y,gr=LinkedCompressor(48000,config_from_candidate(item)).process(x)
    np.testing.assert_array_equal(x,original)
    assert y.shape==x.shape and np.isfinite(y).all() and np.max(gr)>0
    assert np.max(gr)<=item['compressor']['max_gr_db']+1e-6


def test_level_match_without_export_ceiling_reports_exact_internal_float_gain():
    ref=pulses();cand=ref*np.float32(.5)
    report=level_match_evidence(ref,cand,48000)
    assert report['required_gain_db']==pytest.approx(6.0205999,abs=.001)
    assert report['applied_gain_db']==pytest.approx(report['required_gain_db'])
    assert abs(report['residual_match_error_db'])<1e-9
    assert report['headroom_limited'] is False and report['match_passed'] is True


def test_level_match_export_ceiling_exposes_conflict_instead_of_hiding_it():
    ref=pulses();cand=ref*np.float32(.5)
    report=level_match_evidence(ref,cand,48000,ceiling_dbtp=-9.)
    assert report['headroom_limited'] is True
    assert report['applied_gain_db']<report['required_gain_db']
    assert report['match_passed'] is False
    assert report['residual_match_error_db']<-.05


def test_level_match_rejects_shape_mismatch_and_handles_silence():
    with pytest.raises(ValueError):level_match_evidence(np.zeros(20,'float32'),np.zeros(21,'float32'),48000)
    report=level_match_evidence(np.zeros(20,'float32'),np.zeros(20,'float32'),48000)
    assert report['status']=='no_active_reference' and report['match_passed'] is None


def test_core_render_has_no_hidden_makeup_and_remains_human_gated():
    x=pulses();item=candidate(propose_candidates(x,48000,'snare'),'balanced')
    y,report=render_core_candidate(x,48000,item)
    assert report['max_gr_db']>0 and report['baseline_eligible'] is False
    assert report['requires_human_review'] and np.sqrt(np.mean(y.astype('float64')**2)) < np.sqrt(np.mean(x.astype('float64')**2))


def test_audition_pair_plan_matches_first_then_applies_common_trim():
    ref=pulses();cand=ref*np.float32(.5)
    plan=audition_pair_plan(ref,cand,48000,ceiling_dbtp=-6.)
    assert plan['match_passed'] is True
    assert plan['candidate_match_gain_db']==pytest.approx(6.0205999,abs=.001)
    assert plan['reference_gain_db']==pytest.approx(plan['common_trim_db'])
    assert plan['candidate_total_gain_db']==pytest.approx(plan['candidate_match_gain_db']+plan['common_trim_db'])
    assert plan['reference_true_peak_after_dbtp']<=-5.999 and plan['candidate_true_peak_after_dbtp']<=-5.999


def test_candidate_metadata_holds_machine_safe_change_for_human_listening():
    from audio_workbench.quality_loop import evaluate_candidate
    item=candidate(propose_candidates(pulses(),48000,'snare'),'balanced')
    result=evaluate_candidate({'confidence':{'cause':.9,'intervention':.9}},item,True,[],.95)
    assert result['accepted'] is False
    assert result['acceptance_state']=='pending_human_review'
