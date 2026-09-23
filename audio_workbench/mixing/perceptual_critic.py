from __future__ import annotations
from dataclasses import dataclass,asdict
import numpy as np
from scipy import signal

@dataclass(frozen=True)
class PerceptualSnapshot:
    foreground_db:float
    vocal_intelligibility:float
    punch_db:float
    harshness:float
    density:float
    depth_proxy:float
    width_db:float
    climax_lift_db:float

def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x,dtype="float64")**2)+1e-20))

def _band(x,sr,lo,hi):
    hi=min(hi,sr*.47)
    sos=signal.butter(3,[lo,hi],btype="bandpass",fs=sr,output="sos")
    return signal.sosfilt(sos,x,axis=0)

def _db_ratio(a,b):
    return float(20*np.log10((_rms(a)+1e-12)/(_rms(b)+1e-12)))

def snapshot(mix:np.ndarray,sr:int,vocal:np.ndarray|None=None,drums:np.ndarray|None=None,
             early_room:np.ndarray|None=None,section_rms_db:list[float]|None=None)->PerceptualSnapshot:
    if mix.ndim==1: mix=np.column_stack([mix,mix])
    mono=mix.mean(1)
    mid=(mix[:,0]+mix[:,1])*.5
    side=(mix[:,0]-mix[:,1])*.5

    # Harshness is a bounded spectral prominence proxy, not a taste score.
    harsh=_rms(_band(mono,sr,2500,6500))/(_rms(_band(mono,sr,250,2200))+1e-12)
    harsh=float(np.clip(harsh*2.2,0,1))

    # Density: fraction of short windows close to the upper RMS envelope.
    hop=max(1,int(.05*sr));n=len(mono)//hop
    if n:
        q=mono[:n*hop].reshape(n,hop)
        db=20*np.log10(np.sqrt(np.mean(q.astype("float64")**2,axis=1))+1e-12)
        density=float(np.mean(db>np.percentile(db,80)-6))
    else:density=0.

    if vocal is not None:
        if vocal.ndim>1:vocal=vocal.mean(1)
        nv=min(len(vocal),len(mono));v=vocal[:nv];m=mono[:nv]
        fg=_db_ratio(v,m-v)
        vb=_rms(_band(v,sr,1200,4500));mb=_rms(_band(m-v,sr,1200,4500))
        intellig=float(np.clip((20*np.log10((vb+1e-12)/(mb+1e-12))+18)/24,0,1))
    else:
        fg=0.;intellig=.5

    if drums is not None:
        if drums.ndim>1:drums=drums.mean(1)
        d=drums[:len(mono)]
        low=_band(d,sr,45,180)
        env=np.abs(signal.hilbert(low))
        punch=float(20*np.log10((np.percentile(env,99)+1e-12)/(np.percentile(env,75)+1e-12)))
    else:punch=0.

    if early_room is not None:
        er=early_room[:len(mix)]
        depth=float(np.clip((_rms(er)/(_rms(mix)+1e-12))*8,0,1))
    else:depth=.5

    width=_db_ratio(side,mid)
    climax=0.
    if section_rms_db and len(section_rms_db)>=3:
        a=np.asarray(section_rms_db,float)
        climax=float(np.percentile(a,90)-np.median(a))

    return PerceptualSnapshot(fg,intellig,punch,harsh,density,depth,width,climax)

def diagnose(s:PerceptualSnapshot)->list[dict]:
    """Return hypotheses, never an overall quality score."""
    h=[]
    if s.vocal_intelligibility<.42:
        h.append({"target":"vocal_intelligibility","direction":"increase","confidence":.78,
                  "bounded_action":"reduce competing 1.2-4.5 kHz or ride vocal <=0.7 dB"})
    if s.harshness>.72:
        h.append({"target":"harshness","direction":"decrease","confidence":.72,
                  "bounded_action":"dynamic 2.5-6.5 kHz control <=1.0 dB"})
    if s.punch_db<5.0:
        h.append({"target":"punch","direction":"increase","confidence":.68,
                  "bounded_action":"transient/density adjustment on drums <=0.8 dB"})
    if s.climax_lift_db<1.0:
        h.append({"target":"climax","direction":"increase","confidence":.60,
                  "bounded_action":"section orchestration/width/FX move, not broadband gain first"})
    return sorted(h,key=lambda x:x["confidence"],reverse=True)

def compare(before:PerceptualSnapshot,after:PerceptualSnapshot,target:str)->dict:
    b=asdict(before);a=asdict(after)
    key={"vocal_intelligibility":"vocal_intelligibility","harshness":"harshness",
         "punch":"punch_db","climax":"climax_lift_db"}[target]
    return {"target":target,"before":b[key],"after":a[key],"delta":a[key]-b[key],
            "collateral":{"density":a["density"]-b["density"],"width_db":a["width_db"]-b["width_db"],
                          "foreground_db":a["foreground_db"]-b["foreground_db"]}}
