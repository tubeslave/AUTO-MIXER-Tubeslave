from __future__ import annotations
import numpy as np
from scipy import signal

def masking_profile(foreground:np.ndarray,background:np.ndarray,sr:int)->dict:
    """Find broad regions where a background bus competes with an active foreground bus."""
    fg=foreground.mean(1) if foreground.ndim>1 else foreground
    bg=background.mean(1) if background.ndim>1 else background
    f,pf=signal.welch(fg,fs=sr,nperseg=8192)
    _,pb=signal.welch(bg,fs=sr,nperseg=8192)
    score=10*np.log10((pb+1e-20)/(pf+1e-20))
    # Weight only frequencies materially represented in the foreground.
    fg_rel=10*np.log10((pf+1e-20)/(np.max(pf)+1e-20))
    valid=(f>=180)&(f<=6000)&(fg_rel>-28)
    return {"frequency_hz":f,"competition_db":score,"foreground_relative_db":fg_rel,"valid":valid}

def propose_bells(profile:dict,max_bands:int=2,max_cut_db:float=1.2)->list[dict]:
    f=profile["frequency_hz"];s=profile["competition_db"];v=profile["valid"]
    rows=[]
    for lo,hi in [(180,450),(450,1200),(1200,2800),(2800,6000)]:
        m=v&(f>=lo)&(f<hi)
        if not np.any(m):continue
        q=float(np.percentile(s[m],70))
        if q<1.5:continue
        fc=float(np.exp(np.mean(np.log(f[m]+1e-9))))
        cut=float(-np.clip((q-1.0)*.18,.25,max_cut_db))
        rows.append({"fc_hz":fc,"q":.8,"gain_db":cut,"competition_db":q})
    rows.sort(key=lambda r:r["competition_db"],reverse=True)
    return rows[:max_bands]
