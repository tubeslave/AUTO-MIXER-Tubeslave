from __future__ import annotations
import numpy as np
from scipy import ndimage

ROLE_POLICY={
 "vocal":{"ratio":2.8,"max_gr_db":5.0,"attack_ms":18,"release_ms":110,"target_pctl":58},
 "bass":{"ratio":2.4,"max_gr_db":4.0,"attack_ms":28,"release_ms":140,"target_pctl":62},
 "guitar":{"ratio":1.7,"max_gr_db":2.5,"attack_ms":24,"release_ms":120,"target_pctl":68},
 "keys":{"ratio":1.5,"max_gr_db":2.0,"attack_ms":30,"release_ms":150,"target_pctl":70},
 "playback":{"ratio":1.35,"max_gr_db":1.5,"attack_ms":35,"release_ms":180,"target_pctl":72},
 "kick":{"ratio":2.0,"max_gr_db":3.0,"attack_ms":22,"release_ms":90,"target_pctl":70},
 "snare":{"ratio":2.0,"max_gr_db":3.0,"attack_ms":16,"release_ms":105,"target_pctl":70},
 "toms":{"ratio":1.8,"max_gr_db":2.5,"attack_ms":20,"release_ms":120,"target_pctl":72},
 "cymbals":{"ratio":1.2,"max_gr_db":1.0,"attack_ms":45,"release_ms":220,"target_pctl":80},
}

def short_term_db(x:np.ndarray,sr:int,window_ms:float=80,hop_ms:float=20)->np.ndarray:
    m=x.mean(1) if x.ndim>1 else x
    win=max(16,int(window_ms*sr/1000));hop=max(1,int(hop_ms*sr/1000))
    n=max(0,1+(len(m)-win)//hop)
    if n<=0:return np.array([],dtype=float)
    out=np.empty(n)
    for i in range(n):
        q=m[i*hop:i*hop+win].astype("float64")
        out[i]=20*np.log10(np.sqrt(np.mean(q*q)+1e-20))
    return out

def diagnose(x:np.ndarray,sr:int,role:str)->dict:
    db=short_term_db(x,sr)
    if len(db)==0:return {"role":role,"dynamic_range_db":0,"recommended":False}
    active=db>np.percentile(db,35);a=db[active]
    spread=float(np.percentile(a,90)-np.percentile(a,20))
    p=ROLE_POLICY.get(role,{"ratio":1.5,"max_gr_db":2,"attack_ms":25,"release_ms":140,"target_pctl":70})
    return {"role":role,"dynamic_range_db":spread,"recommended":spread>5.0,
            "threshold_db":float(np.percentile(a,p["target_pctl"])),"policy":p}

def compressor_curve(x:np.ndarray,sr:int,role:str)->tuple[np.ndarray,dict]:
    d=diagnose(x,sr,role);p=d["policy"]
    if not d["recommended"]:return np.ones(len(x),dtype="float32"),{**d,"max_gr_db":0.0}
    m=x.mean(1) if x.ndim>1 else x
    # 12 ms detector, smoothed with role attack/release.
    win=max(8,int(.012*sr))
    env=np.sqrt(ndimage.uniform_filter1d(m.astype("float64")**2,size=win)+1e-20)
    db=20*np.log10(env+1e-20);over=np.maximum(db-d["threshold_db"],0)
    target=np.minimum(over*(1-1/p["ratio"]),p["max_gr_db"])
    gr=np.zeros(len(target),dtype="float32")
    aa=np.exp(-1/(max(1,p["attack_ms"]*sr/1000)))
    rr=np.exp(-1/(max(1,p["release_ms"]*sr/1000)))
    for i in range(1,len(gr)):
        c=aa if target[i]>gr[i-1] else rr
        gr[i]=c*gr[i-1]+(1-c)*target[i]
    return np.power(10,-gr/20).astype("float32"),{**d,"max_gr_db":float(gr.max()),"p95_gr_db":float(np.percentile(gr,95))}

def apply(x:np.ndarray,sr:int,role:str)->tuple[np.ndarray,dict]:
    g,d=compressor_curve(x,sr,role)
    y=x*g[:,None] if x.ndim>1 else x*g
    # No automatic makeup. Static balance remains the Balance Director's job.
    return y.astype("float32"),d
