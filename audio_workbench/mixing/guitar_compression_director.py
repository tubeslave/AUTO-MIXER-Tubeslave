"""No-change-first rhythm-guitar compression gate for STUDIO/offline use."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import numpy as np
from scipy import ndimage, signal
from .compression import CompressorConfig, as_audio

@dataclass(frozen=True)
class GuitarGatePolicy:
    actionable_local_spread_db: float = 2.8
    min_local_spread_improvement_db: float = 0.10
    max_pick_attack_loss_db: float = 0.25
    max_macro_dynamics_change_db: float = 0.20
    min_active_blocks: int = 12
    min_pick_events: int = 24

def _mono(x):
    x=as_audio(x); return np.mean(x,axis=1).astype(np.float32) if x.ndim==2 else x

def _frame_rms_db(x,sr,frame_ms=40.,hop_ms=20.):
    x=_mono(x); frame=max(8,round(frame_ms*sr/1000)); hop=max(1,round(hop_ms*sr/1000))
    if len(x)<frame:return np.zeros(0),np.zeros(0)
    starts=np.arange(1+(len(x)-frame)//hop,dtype=np.int64)*hop
    p=x.astype(np.float64)**2; integ=np.concatenate([[0.],np.cumsum(p)])
    rms=np.sqrt(np.maximum((integ[starts+frame]-integ[starts])/frame,1e-30))
    return 20*np.log10(rms),(starts+frame/2)/sr

def _rms_db(x,start,end):
    start=max(0,int(start));end=min(len(x),int(end))
    if end-start<8:return None
    s=np.asarray(x[start:end],np.float64); p=np.mean(s*s,axis=1) if s.ndim==2 else s*s
    return 10*np.log10(max(float(np.mean(p)),1e-30))

def guitar_pick_centers(detection_source,sr):
    x=_mono(detection_source)
    if not len(x) or float(np.max(np.abs(x)))<1e-10:return np.zeros(0,dtype=np.int64)
    high=signal.sosfiltfilt(signal.butter(2,[700,7000],btype='bandpass',fs=sr,output='sos'),x)
    db,t=_frame_rms_db(high,sr,12.,4.)
    if not len(db):return np.zeros(0,dtype=np.int64)
    env=ndimage.gaussian_filter1d(db,1.0); delta=env-ndimage.gaussian_filter1d(env,6.0)
    threshold=max(float(np.percentile(delta,92)),1.5)
    hop_s=float(np.median(np.diff(t))) if len(t)>1 else .004
    peaks,_=signal.find_peaks(delta,height=threshold,prominence=.8,distance=max(1,round(.055/hop_s)))
    return np.asarray(np.round(t[peaks]*sr),dtype=np.int64)

def guitar_dynamics_evidence(detection_source,audio,sr):
    detection_source=as_audio(detection_source); audio=as_audio(audio)
    if len(detection_source)!=len(audio):raise ValueError('guitar detection source and measured audio must have equal frames')
    db,t=_frame_rms_db(audio,sr,40.,20.)
    if not len(db):
        return {'active_block_count':0,'pick_event_count':0,'median_local_spread_db':None,'macro_spread_db':None,'median_pick_attack_body_db':None}
    global_th=max(float(np.percentile(db,35)),float(np.max(db))-38.)
    block_ids=np.floor(t/2.0).astype(int); local=[]; block_means=[]
    for bid in np.unique(block_ids):
        vals=db[(block_ids==bid)&(db>=global_th)]
        if len(vals)>=30:
            local.append(float(np.percentile(vals,90)-np.percentile(vals,10))); block_means.append(float(np.mean(vals)))
    centers=guitar_pick_centers(detection_source,sr); ab=[]
    for c in centers:
        a=_rms_db(audio,c-int(.006*sr),c+int(.014*sr)); b=_rms_db(audio,c+int(.022*sr),c+int(.075*sr))
        if a is not None and b is not None and np.isfinite(a+b):ab.append(a-b)
    macro=float(np.percentile(block_means,90)-np.percentile(block_means,10)) if block_means else None
    return {'active_block_count':len(local),'pick_event_count':len(ab),
            'median_local_spread_db':float(np.median(local)) if local else None,'macro_spread_db':macro,
            'median_pick_attack_body_db':float(np.median(ab)) if ab else None,
            'measurement_scope':'2 s local active-frame spread + fixed high-band pick windows; technical proxy, not listening judgement'}

def baseline_actionability(detection_source,baseline_audio,sr,*,policy=None):
    policy=policy or GuitarGatePolicy(); e=guitar_dynamics_evidence(detection_source,baseline_audio,sr); failures=[]
    if e['active_block_count']<policy.min_active_blocks: failures.append('insufficient_active_guitar_blocks')
    if e['pick_event_count']<policy.min_pick_events: failures.append('insufficient_pick_events')
    if not failures and e['median_local_spread_db'] < policy.actionable_local_spread_db: failures.append('no_actionable_local_dynamics_problem')
    return {'schema':'guitar-compression-actionability-v1','evidence':e,'actionable':not failures,'failures':failures,
            'selection_policy':'no-change first; compressor changes require demonstrated local dynamics problem','requires_human_listening':True,'baseline_eligible':False}

def assess_against_baseline(detection_source,baseline_audio,candidate_audio,sr,*,policy=None):
    policy=policy or GuitarGatePolicy(); base=guitar_dynamics_evidence(detection_source,baseline_audio,sr); cand=guitar_dynamics_evidence(detection_source,candidate_audio,sr); failures=[]
    if min(base['active_block_count'],cand['active_block_count'])<policy.min_active_blocks or min(base['pick_event_count'],cand['pick_event_count'])<policy.min_pick_events:
        failures.append('insufficient_guitar_evidence'); deltas={'local_spread_delta_db':None,'pick_attack_body_delta_db':None,'macro_spread_delta_db':None}
    else:
        deltas={'local_spread_delta_db':float(cand['median_local_spread_db']-base['median_local_spread_db']),
                'pick_attack_body_delta_db':float(cand['median_pick_attack_body_db']-base['median_pick_attack_body_db']),
                'macro_spread_delta_db':float(cand['macro_spread_db']-base['macro_spread_db'])}
        if deltas['local_spread_delta_db'] > -policy.min_local_spread_improvement_db: failures.append('guitar_local_stability_not_improved')
        if deltas['pick_attack_body_delta_db'] < -policy.max_pick_attack_loss_db: failures.append('guitar_pick_attack_reduced')
        if abs(deltas['macro_spread_delta_db']) > policy.max_macro_dynamics_change_db: failures.append('guitar_macro_dynamics_changed')
    return {'schema':'guitar-compression-baseline-assessment-v1','baseline':base,'candidate':cand,**deltas,'technically_survives':not failures,'failures':failures,
            'requires_full_session_rerender':not failures,'requires_human_listening':True,'baseline_eligible':False}

def propose_candidates(sr,baseline_config,actionability,*,policy=None):
    policy=policy or GuitarGatePolicy();baseline_config.validate(sr)
    if not actionability.get('actionable',False):
        return {'schema':'guitar-compression-director-v1','role':'rhythm_guitar','reference':{'id':'no_change','compressor':asdict(baseline_config)},'policy':asdict(policy),'candidates':[],
                'decision':'no_change','reason':'no actionable local dynamics problem demonstrated','requires_human_listening':True,'baseline_eligible':False}
    variants=(('preserve_pick',1.28,1.0),('tighten_body',.80,.78),('longer_sustain',1.0,1.25)); out=[]
    for ident,af,rf in variants:
        cfg=replace(baseline_config,attack_ms=baseline_config.attack_ms*af,release_ms=baseline_config.release_ms*rf);cfg.validate(sr)
        out.append({'id':ident,'compressor':asdict(cfg),'change_scope':'attack_release_only','requires_full_session_rerender':True,'requires_human_listening':True,'baseline_eligible':False})
    return {'schema':'guitar-compression-director-v1','role':'rhythm_guitar','reference':{'id':'no_change','compressor':asdict(baseline_config)},'policy':asdict(policy),'candidates':out,'decision':'evaluate_bounded_candidates','requires_human_listening':True,'baseline_eligible':False}
