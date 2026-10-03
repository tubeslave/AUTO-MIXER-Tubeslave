import json
from dataclasses import replace

import numpy as np
import pytest

from audio_workbench.mixing.compression import CompressorConfig
from audio_workbench.mixing.compression_director_v11 import (
    calibrate_threshold, propose_calibrated_candidates,
    render_calibrated_core_candidate, timing_censor_evidence,
)


def pulses(spacing=.5, attack_ms=3, decay_ms=35, sr=48000, seconds=4):
    x=np.zeros(int(sr*seconds),dtype='float32')
    for center in np.arange(.3,seconds-.2,spacing):
        peak=int(center*sr); attack=max(1,int(attack_ms*sr/1000)); start=max(0,peak-attack)
        x[start:peak]=np.linspace(0,.8,peak-start,endpoint=False,dtype='float32')
        end=min(len(x),peak+int(.14*sr)); x[peak:end]=.8*np.exp(-np.arange(end-peak)/(sr*decay_ms/1000)).astype('float32')
    return x


def test_three_candidates_calibrate_actual_active_p95_without_changing_safety_shape():
    proposal=propose_calibrated_candidates(pulses(),48000,'snare')
    assert proposal['schema']=='compression-director-v1.1'
    assert proposal['baseline_eligible'] is False and proposal['requires_human_listening'] is True
    assert [x['id'] for x in proposal['candidates']]==['preserve_transient','balanced','control']
    for item in proposal['candidates']:
        ev=item['calibration']; assert ev['status']=='converged'
        assert abs(ev['error_db'])<=ev['tolerance_db']
        assert ev['max_gr_unchanged'] is True
        y,render=render_calibrated_core_candidate(pulses(),48000,item)
        assert y.shape==pulses().shape
        assert abs(render['target_error_db'])<=ev['tolerance_db']+.002
        assert render['baseline_eligible'] is False
    json.dumps(proposal,allow_nan=False)


def test_calibration_changes_threshold_only():
    x=pulses();cfg=CompressorConfig(threshold_dbfs=-18,ratio=2.2,attack_ms=17,release_ms=123,knee_db=4,max_gr_db=3,rms_ms=1)
    out,ev=calibrate_threshold(x,48000,cfg,2.0)
    assert ev['status']=='converged'
    for field in ('ratio','attack_ms','release_ms','knee_db','max_gr_db','detector','rms_ms','sidechain_hpf_hz','bypass'):
        assert getattr(out,field)==getattr(cfg,field)
    assert out.threshold_dbfs!=cfg.threshold_dbfs


def test_unreachable_target_is_explicit_and_never_raises_max_gr():
    cfg=CompressorConfig(threshold_dbfs=-18,ratio=4,max_gr_db=1,knee_db=0,attack_ms=2,release_ms=40,detector='peak')
    out,ev=calibrate_threshold(pulses(),48000,cfg,3.0)
    assert ev['status']=='unreachable_high'
    assert out.max_gr_db==1 and ev['max_gr_unchanged'] is True
    assert ev['actual_active_p95_gr_db']<=1.000001


def test_silence_is_insufficient_signal_and_json_safe():
    cfg=CompressorConfig();out,ev=calibrate_threshold(np.zeros(48000,'float32'),48000,cfg,2.0)
    assert ev['status']=='insufficient_signal' and out==cfg
    json.dumps(ev,allow_nan=False)


def test_invalid_calibration_requests_fail_closed():
    x=pulses();cfg=CompressorConfig()
    for value in (-1,np.nan,np.inf):
        with pytest.raises(ValueError):calibrate_threshold(x,48000,cfg,value)
    with pytest.raises(ValueError):calibrate_threshold(x,48000,cfg,2,tolerance_db=0)
    with pytest.raises(ValueError):calibrate_threshold(x,48000,cfg,2,max_iterations=0)


def test_timing_censoring_reports_window_limited_measurements():
    sr=48000;x=np.full(sr*6,.35,dtype='float32')
    width=int(.12*sr)
    for center in (1,3,5):
        c=int(center*sr); x[c-width:c]=np.linspace(.35,.7,width,endpoint=False,dtype='float32');x[c:c+width]=np.linspace(.7,.35,width,endpoint=False,dtype='float32')
    ev=timing_censor_evidence(x,sr,'vocal')
    assert ev['event_count']>=1
    assert ev['attack_censored_count']>0 or ev['recovery_censored_count']>0
    assert ev['attack_reason_counts'] or ev['recovery_reason_counts']
    json.dumps(ev,allow_nan=False)


def test_calibrated_proposal_is_deterministic_and_input_unchanged():
    x=pulses();original=x.copy()
    a=propose_calibrated_candidates(x,48000,'snare');b=propose_calibrated_candidates(x,48000,'snare')
    assert a==b;np.testing.assert_array_equal(x,original)


def test_human_review_gate_survives_v11():
    from audio_workbench.quality_loop import evaluate_candidate
    item=propose_calibrated_candidates(pulses(),48000,'snare')['candidates'][1]
    result=evaluate_candidate({'confidence':{'cause':.9,'intervention':.9}},item,True,[],.95)
    assert result['accepted'] is False and result['acceptance_state']=='pending_human_review'
