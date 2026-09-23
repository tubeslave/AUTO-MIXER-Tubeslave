#!/usr/bin/env python3
"""Fresh, deterministic Ptitsa mix using offline directors and controller mastering.

No previous mix recipe is loaded. Original recordings are immutable. Vocal model
cleanup remains bypassed until separately accepted; all outputs require listening.
"""
from __future__ import annotations
from dataclasses import asdict
import argparse, gc, hashlib, json, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from scipy import signal, ndimage

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audio_workbench.mixing import context, balance, dynamics, masking, masking_director, space, perceptual_critic, section_context, critic
from audio_workbench.mastering.analyzer import analyze
from audio_workbench.mastering.offline import deliver_master

SR=44100
N=11421900
NAMES=['BASS','FLOOR','GTR','HI_HAT','KEYS_L','KEYS_R','KICK_IN','KICK_OUT','NIKITA_VOX','OH_R','OHL','PB_L','PB_R','SN_B','SN_T','TOM_1','TOM_2','VALERA_VOX']
HPF={'BASS':32,'FLOOR':48,'GTR':85,'HI_HAT':250,'KEYS_L':100,'KEYS_R':100,'KICK_IN':28,'KICK_OUT':28,'NIKITA_VOX':110,'OH_R':150,'OHL':150,'PB_L':65,'PB_R':65,'SN_B':120,'SN_T':90,'TOM_1':70,'TOM_2':58,'VALERA_VOX':100}
# Role planes are targets, not copied faders. Layer planes are verified again at bus sums.
DRUM_PLANES={'KICK_IN':-27.,'KICK_OUT':-31.,'SN_T':-27.,'SN_B':-36.,'TOM_1':-33.,'TOM_2':-33.,'FLOOR':-33.,'OHL':-33.,'OH_R':-33.,'HI_HAT':-39.}
PAN={'KICK_IN':0.,'KICK_OUT':0.,'SN_T':0.,'SN_B':0.,'TOM_1':-.35,'TOM_2':.18,'FLOOR':.5,'OHL':-.85,'OH_R':.85,'HI_HAT':-.4,'GTR':-.25,'VALERA_VOX':0.,'NIKITA_VOX':.15,'BASS':0.}


def rms(x): return float(np.sqrt(np.mean(np.asarray(x,dtype='float64')**2)+1e-24))
def db_rms(x): return 20*np.log10(rms(x)+1e-15)
def lufs(x): return float(pyln.Meter(SR).integrated_loudness(x))
def gain(x,db): return (x*np.float32(10**(db/20))).astype('float32')
def frames(x,hop_s=.02):
    m=x.mean(1) if x.ndim==2 else x;h=int(SR*hop_s);q=m[:len(m)//h*h].reshape(-1,h)
    return np.sqrt(np.mean(q.astype('float64')**2,axis=1)+1e-20)
def interp_control(db,hop_s,n=N): return np.interp(np.arange(n,dtype='float64')/SR,(np.arange(len(db))+.5)*hop_s,db).astype('float32')
def matched(x,y):
    r=frames(x);h=int(SR*.02);active=np.repeat(r>np.percentile(r,55),h)
    active=np.pad(active,(0,len(x)-len(active)),constant_values=False)
    db=float(np.clip(20*np.log10((rms(x[active])+1e-15)/(rms(y[active])+1e-15)),-12,12))
    z=gain(y,db)
    return z,{'gain_db':db,'active_rms_delta_db':float(db_rms(z[active])-db_rms(x[active]))}
def plane(x,target):
    current=lufs(x);db=float(np.clip(target-current,-24,24));return gain(x,db),{'input_lufs':current,'target_lufs':target,'gain_db':db,'output_lufs':lufs(gain(x,db))}
def bell(x,hz,db,q=.8):
    A=10**(db/40);w=2*np.pi*hz/SR;alpha=np.sin(w)/(2*q);c=np.cos(w)
    b=np.array([1+alpha*A,-2*c,1-alpha*A]);a=np.array([1+alpha/A,-2*c,1-alpha/A])
    return signal.lfilter(b/a[0],a/a[0],x,axis=0).astype('float32')
def band(x,lo,hi): return signal.sosfiltfilt(signal.butter(2,[lo,hi],btype='bandpass',fs=SR,output='sos'),x,axis=0).astype('float32')
def deess(x):
    sib=frames(band(x,4800,10000),.01);body=frames(band(x,700,3500),.01)
    ratio=20*np.log10((sib+1e-12)/(body+1e-12));th=np.percentile(ratio,84)
    gr=ndimage.gaussian_filter1d(np.clip((ratio-th)*.5,0,2.0),2)
    curve=interp_control(-gr,.01,len(x));component=band(x,4800,10000)
    y=x+component*(np.power(10,curve/20)-1)
    return y.astype('float32'),{'max_gr_db':float(gr.max()),'p95_gr_db':float(np.percentile(gr,95))}
def phase_pair(reference,target,lo,hi):
    # Only propose a polarity switch when independent windows agree. Never move timing here.
    corr=[]
    for t in (30,60,100,160,205):
        a=band(reference[int(t*SR):int((t+2)*SR)],lo,hi);b=band(target[int(t*SR):int((t+2)*SR)],lo,hi)
        corr.append(float(np.corrcoef(a,b)[0,1]))
    invert=bool(np.median(corr)<-.25 and sum(v<-.15 for v in corr)>=4)
    return (-target if invert else target),{'window_correlation':corr,'polarity_inverted':invert,'timing_shift_samples':0}


def run(raw_dir,clean_dir,out,*,premaster_only=False):
    raw_dir=Path(raw_dir);clean_dir=Path(clean_dir);out=Path(out)
    out.mkdir(parents=True,exist_ok=False);(out/'buses').mkdir()
    report={'schema':'ptitsa-fresh-studio-v1','status':'pending_human_review','baseline_eligible':False,'previous_mix_recipe_loaded':False,'originals':{},'tracks':{},'groups':{},'live_control_used':False}
    roles=context.infer_roles(NAMES);tracks={}
    for name in NAMES:
        path=raw_dir/(name+'.flac');x,sr=sf.read(path,dtype='float32',always_2d=True)
        if sr!=SR or len(x)!=N or not np.isfinite(x).all():raise ValueError('invalid source '+name)
        report['originals'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
        mono=x.mean(1);del x;entry={'role':roles[name],'raw_lufs':lufs(mono)}
        clean_path=clean_dir/(name+'.wav')
        if clean_path.exists() and 'VOX' not in name:
            y,csr=sf.read(clean_path,dtype='float32',always_2d=True)
            if csr!=SR or len(y)!=len(mono): raise ValueError('cleanup is not aligned')
            y=y.mean(1);mono,match=matched(mono,y);del y
            if abs(match['active_rms_delta_db'])>.05: raise ValueError('cleanup level-matching gate failed')
            entry['cleanup']={'source':'saved_model_checkpoint','sha256':hashlib.sha256(clean_path.read_bytes()).hexdigest(),'level_match':match,'human_status':'not_separately_accepted'}
        else: entry['cleanup']={'source':'raw','reason':'vocal model intervention not accepted' if 'VOX' in name else 'no neural processing requested'}
        y=signal.sosfilt(signal.butter(2,HPF[name],btype='highpass',fs=SR,output='sos'),mono).astype('float32')
        entry['highpass_hz']=HPF[name];del mono
        # Broad, subtractive EQ only, conditioned by the actual low-mid/body balance.
        lo=band(y,180,450);mid=band(y,500,2200);ratio=20*np.log10((rms(lo)+1e-12)/(rms(mid)+1e-12));del lo,mid
        if name in ('GTR','VALERA_VOX','NIKITA_VOX','SN_T') and ratio>1.5:
            cut=-min(2.5,(ratio-1.5)*.25);y=bell(y,300 if 'VOX' in name else 350,cut)
            entry['eq']={'frequency_hz':300 if 'VOX' in name else 350,'gain_db':float(cut),'lowmid_body_ratio_db':float(ratio)}
        if name in ('TOM_1','TOM_2','FLOOR'):
            e=frames(band(y,70,450));edb=20*np.log10(e+1e-12);threshold=float(np.percentile(edb,88)-5)
            reduction=np.clip((threshold-edb)*1.2,0,15);reduction=ndimage.gaussian_filter1d(reduction,3)
            y=gain(y,interp_control(-reduction,.02));entry['soft_bleed_control']={'floor_db':-15,'threshold_dbfs':threshold,'no_hard_gate':True}
        if roles[name] in ('vocal','bass','kick','snare','toms','guitar','keys','playback','cymbals'):
            candidate,diag=dynamics.apply(y,SR,roles[name])
            reduction=diag['before_spread_db']-diag['after_spread_db']
            enabled=bool(reduction>=.05 and reduction<=5 and diag['before_spread_db']>3)
            entry['dynamics']={**diag,'enabled':enabled}
            if enabled:y,_=matched(y,candidate)
            del candidate
        if 'VOX' in name:y,entry['deess']=deess(y)
        tracks[name]=y.astype('float32');report['tracks'][name]=entry
        print('source processed',name,flush=True)
    for ref,tgt,lo,hi in [('KICK_IN','KICK_OUT',40,180),('SN_T','SN_B',100,2000)]:
        tracks[tgt],ev=phase_pair(tracks[ref],tracks[tgt],lo,hi);report['tracks'][tgt]['phase']=ev
    groups={};drums=np.zeros((N,2),dtype='float32');kick=np.zeros((N,2),dtype='float32')
    for name,target in DRUM_PLANES.items():
        m,ev=plane(tracks.pop(name),target);p=balance.equal_power_pan(m,PAN[name]);del m
        report['tracks'][name]['balance_plane']=ev;drums+=p
        if name.startswith('KICK'):kick+=p
    drums,report['groups']['drums']=plane(drums,-25)
    kick=gain(kick,report['groups']['drums']['gain_db']);groups['drums']=drums
    for name,group,target in [('BASS','bass',-25),('GTR','guitar',-25),('VALERA_VOX','lead',-22),('NIKITA_VOX','backing',-27)]:
        p=balance.equal_power_pan(tracks.pop(name),PAN[name]);p,ev=plane(p,target);groups[group]=p;report['groups'][group]=ev
    for left,right,group,target in [('KEYS_L','KEYS_R','keys',-30),('PB_L','PB_R','playback',-30)]:
        p=np.column_stack([tracks.pop(left),tracks.pop(right)]).astype('float32');p,ev=plane(p,target);groups[group]=p;report['groups'][group]=ev
    del tracks;gc.collect()
    # One source-driven section map. Do not invent verse/chorus labels or normalize sections.
    music=sum(groups[g] for g in ('drums','bass','guitar','keys','playback'))
    sections=context.section_curve(music,SR);d=np.asarray(sections['density']);report['arrangement']=sections
    for group,amount in [('drums',.45),('bass',.25),('guitar',.35),('keys',-.25),('playback',-.20)]:
        frame_db=ndimage.gaussian_filter1d(np.clip((d-.5)*amount,-.4,.4),2)
        groups[group]*=np.power(10,interp_control(frame_db,.5)[:,None]/20).astype('float32')
    del music
    # Low-end and vocal unmasking require measured overlap, not just instrument labels.
    report['masking']=[]
    for target,masker,band_name,target_role,masker_role in [('lead','guitar','presence','vocal','guitar'),('lead','keys','presence','vocal','keys'),('lead','playback','presence','vocal','playback'),('kick','bass','low_punch','kick','bass')]:
        tx=kick if target=='kick' else groups[target];mx=groups[masker];b=masking.BANDS[band_name]
        # Shorter representative windows bound analysis allocation; rendering stays full-length.
        indexes=[(20,32),(80,92),(145,157),(200,212)]
        ta=np.concatenate([tx[int(a*SR):int(z*SR)] for a,z in indexes]);ma=np.concatenate([mx[int(a*SR):int(z*SR)] for a,z in indexes])
        ev=masking.masking_evidence(ta,ma,SR,b);dec=masking_director.decision(target_role,masker_role,band_name,ev);del ta,ma
        if dec['apply']:
            env=frames(band(tx,b.lo,b.hi));lo,hi=np.percentile(env,[55,90]);activity=ndimage.gaussian_filter1d(np.clip((env-lo)/(hi-lo+1e-12),0,1),2)
            curve=interp_control(-dec['depth_db']*activity,.02)
            candidate=masking.apply_dynamic_band_cut(mx,curve,SR,b)
            # Verify the intervention did not buy clarity by a large level change.
            delta=float(db_rms(candidate)-db_rms(mx));dec['whole_track_rms_delta_db']=delta
            if abs(delta)<=.6:groups[masker]=candidate;dec['rendered']=True
            else:dec['rendered']=False;dec['rollback_reason']='masker_level_regression'
        report['masking'].append(dec)
    del kick;gc.collect()
    dry=sum(groups.values()).astype('float32')
    # Re-use deterministic Space Director room. Normalize IR energy, not output chunks.
    ir=space.common_room_ir(SR,rt60_s=.68,length_s=1.0);ir/=np.sqrt(np.sum(ir.astype('float64')**2,axis=0,keepdims=True)).astype('float32')
    send=groups['lead'].mean(1)*10**(-20/20)+groups['drums'].mean(1)*10**(-25/20)+groups['guitar'].mean(1)*10**(-24/20)
    predelay=int(.027*SR);send=np.pad(send,(predelay,0))[:N]
    wet=np.column_stack([signal.fftconvolve(send,ir[:,ch])[:N] for ch in range(2)]).astype('float32');del send
    wet_lufs=lufs(wet);wet_gain=float(np.clip(-35-wet_lufs,-18,6));wet=gain(wet,wet_gain)
    report['space']={'ir':'SpaceDirector.common_room_ir','rt60_s':.68,'predelay_ms':27,'wet_gain_db':wet_gain,'wet_lufs':lufs(wet)}
    groups['room']=wet;mix=(dry+wet).astype('float32');del dry
    fade_in=int(.008*SR);fade_out=int(.12*SR)
    mix[:fade_in]*=np.linspace(0,1,fade_in)[:,None];mix[-fade_out:]*=np.linspace(1,0,fade_out)[:,None]
    # Global headroom only. No independent segment or channel normalization after mixing.
    measurement=analyze(mix,SR,include_true_peak=True,include_loudness=True)
    headroom_db=min(0.,-3.0-measurement['true_peak_dbtp']);mix=gain(mix,headroom_db)
    for group in groups: groups[group]=gain(groups[group],headroom_db)
    report['global_headroom_gain_db']=float(headroom_db)
    snap=perceptual_critic.snapshot(mix,SR,vocal=groups['lead'],drums=groups['drums'],early_room=groups['room'],section_rms_db=sections['rms_db'])
    report['perceptual']={'snapshot':asdict(snap),'diagnoses':perceptual_critic.diagnose(snap),'scope':'engineered proxies, not a human listening verdict'}
    report['section_context']=section_context.describe_sections(mix,SR)
    metrics={'peak_dbfs':float(20*np.log10(np.max(np.abs(mix))+1e-15)),
             'vocal_to_music_db':critic.ratio_db(groups['lead']+groups['backing'],mix-groups['lead']-groups['backing']),
             'kick_to_bass_db':None}
    report['balance_diagnostics']=metrics
    for group,y in groups.items():
        sf.write(out/'buses'/(group+'.wav'),y,SR,subtype='FLOAT')
    report['premaster']=analyze(mix,SR,include_true_peak=True,include_loudness=True)
    sf.write(out/'Ptitsa_Fresh_Premaster.wav',mix,SR,subtype='PCM_24')
    (out/'mix_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('PREMASTER COMPLETE',report['premaster'],flush=True)
    del groups,mix,wet;gc.collect()
    if premaster_only:return report,None
    # The same new CLI delivery path is used in tests and in this real song render.
    master=deliver_master(out/'Ptitsa_Fresh_Premaster.wav',out/'master',name='Ptitsa_Fresh_Master',target_lufs=-15.5,ceiling_dbtp=-1.3)
    print('MASTER COMPLETE',master['status'],master['post_master'],flush=True)
    return report,master


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('raw_dir');p.add_argument('clean_dir');p.add_argument('output_dir');p.add_argument('--premaster-only',action='store_true');a=p.parse_args()
    run(a.raw_dir,a.clean_dir,a.output_dir,premaster_only=a.premaster_only)
