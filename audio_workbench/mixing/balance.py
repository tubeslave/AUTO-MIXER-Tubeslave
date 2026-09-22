from __future__ import annotations
import numpy as np

ROLE_TARGETS={"kick":-25.5,"snare":-27.0,"toms":-31.0,"cymbals":-32.0,"bass":-28.0,
              "guitar":-30.0,"keys":-33.0,"playback":-31.0,"vocal":-27.0,"other":-34.0}
PAN_DEFAULT={"kick":0.0,"snare":0.0,"bass":0.0,"vocal":0.0,"guitar":-.35,
             "keys":.25,"playback":0.0,"toms":.20,"cymbals":0.0,"other":0.0}

def active_rms_db(x:np.ndarray)->float:
    m=np.mean(x,axis=1) if x.ndim>1 else x
    hop=max(64,min(2048,len(m)//200 if len(m)>200 else len(m)))
    n=len(m)//hop
    if n<2:return float(20*np.log10(np.sqrt(np.mean(m*m)+1e-20)))
    q=m[:n*hop].reshape(n,hop).astype("float64")
    r=np.sqrt(np.mean(q*q,axis=1)+1e-20);active=r>np.percentile(r,35)
    return float(20*np.log10(np.sqrt(np.mean(q[active]*q[active])+1e-20)))

def initial_gain_db(x:np.ndarray,role:str)->float:
    return float(np.clip(ROLE_TARGETS.get(role,-34)-active_rms_db(x),-18,18))

def equal_power_pan(mono:np.ndarray,pan:float)->np.ndarray:
    pan=float(np.clip(pan,-1,1));theta=(pan+1)*np.pi/4
    return np.column_stack([mono*np.cos(theta),mono*np.sin(theta)]).astype("float32")
