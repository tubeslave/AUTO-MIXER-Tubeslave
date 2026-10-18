from __future__ import annotations
import numpy as np
from scipy import signal,ndimage

def transient_times(x:np.ndarray,sr:int,lo:float=35,hi:float=8000,min_gap_ms:float=70)->np.ndarray:
    sos=signal.butter(2,[lo,hi],btype="bandpass",fs=sr,output="sos")
    y=signal.sosfiltfilt(sos,x).astype("float32")
    env=np.abs(signal.hilbert(y))
    slow=ndimage.uniform_filter1d(env,size=max(3,int(.08*sr)))
    novelty=np.maximum(env-slow,0)
    threshold=np.percentile(novelty,97)
    peaks,_=signal.find_peaks(novelty,height=threshold,distance=max(1,int(min_gap_ms*sr/1000)))
    return peaks

def timing_outliers(peaks:np.ndarray,sr:int,sigma:float=3.0)->dict:
    if len(peaks)<5:return {"outliers":[],"median_interval_ms":None}
    d=np.diff(peaks)/sr
    med=np.median(d);mad=np.median(np.abs(d-med))+1e-9
    z=.6745*(d-med)/mad
    idx=np.where(np.abs(z)>sigma)[0]+1
    return {"outliers":idx.tolist(),"median_interval_ms":float(med*1000),"mad_ms":float(mad*1000)}
