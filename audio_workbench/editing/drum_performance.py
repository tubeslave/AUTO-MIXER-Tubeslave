from __future__ import annotations
import numpy as np
from scipy import signal,ndimage

def _novelty(x,sr,lo,hi):
    if x.ndim>1:x=x.mean(1)
    sos=signal.butter(2,[lo,hi],btype="bandpass",fs=sr,output="sos")
    y=signal.sosfiltfilt(sos,x).astype("float32")
    e=np.abs(signal.hilbert(y));slow=ndimage.uniform_filter1d(e,size=max(3,int(.075*sr)))
    return np.maximum(e-slow,0)

def confirmed_hits(close:np.ndarray,secondary:np.ndarray,overheads:np.ndarray,sr:int,
                   lo:float,hi:float,percentile:float=97.5,min_gap_ms:float=90)->dict:
    """A hit must be supported by close mic plus secondary/overhead evidence."""
    nc=_novelty(close,sr,lo,hi);ns=_novelty(secondary,sr,lo,hi);no=_novelty(overheads,sr,lo,hi)
    peaks,_=signal.find_peaks(nc,height=np.percentile(nc,percentile),
                              distance=max(1,int(min_gap_ms*sr/1000)))
    tc=np.percentile(nc,percentile);ts=np.percentile(ns,94);to=np.percentile(no,94)
    radius=max(1,int(.012*sr));rows=[]
    for p in peaks:
        a=max(0,p-radius);b=min(len(nc),p+radius+1)
        sec=float(np.max(ns[a:b])/(ts+1e-12));oh=float(np.max(no[a:b])/(to+1e-12))
        close_score=float(nc[p]/(tc+1e-12))
        confidence=float(np.clip(.50*np.tanh(close_score/2)+.25*np.tanh(sec/2)+.25*np.tanh(oh/2),0,1))
        if (sec>=.35 or oh>=.35) and confidence>=.48:
            rows.append({"sample":int(p),"time_s":float(p/sr),"confidence":confidence,
                         "close_score":close_score,"secondary_score":sec,"overhead_score":oh})
    return {"hits":rows}

def hit_level(x:np.ndarray,p:int,sr:int,pre_ms:float=8,post_ms:float=85)->float:
    a=max(0,p-int(pre_ms*sr/1000));b=min(len(x),p+int(post_ms*sr/1000))
    q=x[a:b].astype("float64")
    return float(20*np.log10(np.sqrt(np.mean(q*q)+1e-20)))

def contextual_outliers(hits:list[dict],levels:list[float],section_s:float=20,
                        min_dev_db:float=4.0,max_move_db:float=1.5)->list[dict]:
    """Compare a hit only with nearby confirmed hits in the same broad song section."""
    t=np.array([h["time_s"] for h in hits]);l=np.asarray(levels);out=[]
    for i,h in enumerate(hits):
        m=(np.abs(t-t[i])<=section_s/2)&(np.arange(len(t))!=i)
        if np.sum(m)<6:continue
        local=l[m];med=float(np.median(local));mad=float(np.median(np.abs(local-med))+1e-9)
        dev=float(l[i]-med);z=.6745*dev/mad
        if abs(dev)>=min_dev_db and abs(z)>=2.8 and h["confidence"]>=.55:
            # partial correction; preserve accents
            move=float(np.clip(-dev*.45,-max_move_db,max_move_db))
            out.append({**h,"level_db":float(l[i]),"local_median_db":med,
                        "deviation_db":dev,"robust_z":float(z),"gain_db":move})
    return out
