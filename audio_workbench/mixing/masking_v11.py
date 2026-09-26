from __future__ import annotations
import numpy as np
from scipy import signal
BANDS=((55,110),(110,220),(220,450),(450,900),(900,1800),(1800,3600),(3600,7200),(7200,14000))
def band_profile(x,sr):
    if x.ndim>1:x=x.mean(1)
    f,p=signal.welch(x.astype("float64"),fs=sr,nperseg=min(8192,len(x)))
    vals=[]
    for lo,hi in BANDS:
        m=(f>=lo)&(f<min(hi,sr*.48));vals.append(np.trapz(p[m],f[m]) if np.any(m) else 0.)
    v=np.asarray(vals)+1e-20
    return v/np.sum(v)
def masking_candidates(anchor,competitor,sr,max_moves=2):
    a=band_profile(anchor,sr);b=band_profile(competitor,sr);overlap=np.sqrt(a*b);rows=[]
    for i,(lo,hi) in enumerate(BANDS):
        rows.append({"band":(lo,hi),"center_hz":float(np.sqrt(lo*hi)),"score":float(overlap[i]),
                     "anchor_share":float(a[i]),"competitor_share":float(b[i])})
    rows.sort(key=lambda r:r["score"],reverse=True);return rows[:max_moves]
def bounded_cut(score,max_cut_db=1.5):
    return float(-np.clip(score*18,0,max_cut_db))
