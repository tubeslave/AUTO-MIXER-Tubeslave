from __future__ import annotations
import numpy as np
from scipy import signal,ndimage

def process(x:np.ndarray,sr:int,max_db:float=.9)->tuple[np.ndarray,dict]:
    # Diagnose 30–180 Hz sustain vs attack. Only act when low-end envelope is unusually sustained.
    sos=signal.butter(3,[30,180],btype="bandpass",fs=sr,output="sos")
    low=signal.sosfiltfilt(sos,x,axis=0).astype("float32")
    e=np.max(np.abs(low),axis=1)+1e-12
    fast=ndimage.maximum_filter1d(e,size=max(3,int(.015*sr)))
    slow=np.sqrt(ndimage.uniform_filter1d(e*e,size=max(3,int(.22*sr)))+1e-12)
    sustain=20*np.log10((slow+1e-12)/(fast+1e-12))
    q=float(np.percentile(sustain,80))
    # Positive sustain score means low end is relatively long/flat; reduce only that condition.
    amount=np.clip((q+5.0)*.18,0,max_db)
    if amount<.15:return x,{"active":False,"amount_db":0.0,"sustain_p80_db":q}
    mask=np.clip((sustain-(q-2.0))/4.0,0,1).astype("float32")
    mask=ndimage.gaussian_filter1d(mask,sigma=max(1,int(.025*sr)))
    y=x+low*(np.power(10,(-amount*mask)/20).astype("float32")[:,None]-1)
    return y.astype("float32"),{"active":True,"amount_db":float(amount),"sustain_p80_db":q}
