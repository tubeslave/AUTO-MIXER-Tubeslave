from __future__ import annotations
import numpy as np
from scipy import signal,ndimage

def event_levels(x:np.ndarray,sr:int,lo:float,hi:float,percentile:float=97,
                 min_gap_ms:float=70,window_ms:float=90)->dict:
    if x.ndim>1:x=x.mean(axis=1)
    sos=signal.butter(2,[lo,hi],btype="bandpass",fs=sr,output="sos")
    y=signal.sosfiltfilt(sos,x).astype("float32")
    env=np.abs(signal.hilbert(y));slow=ndimage.uniform_filter1d(env,size=max(3,int(.08*sr)))
    novelty=np.maximum(env-slow,0)
    peaks,_=signal.find_peaks(novelty,height=np.percentile(novelty,percentile),
                              distance=max(1,int(min_gap_ms*sr/1000)))
    half=max(1,int(window_ms*sr/2000));levels=[]
    for p in peaks:
        a=max(0,p-half);b=min(len(x),p+half)
        levels.append(20*np.log10(np.sqrt(np.mean(x[a:b].astype("float64")**2)+1e-20)))
    return {"peaks":peaks,"levels_db":np.asarray(levels)}

def robust_outliers(levels_db:np.ndarray,z_min:float=2.8)->np.ndarray:
    if len(levels_db)<8:return np.zeros(len(levels_db),dtype=bool)
    med=np.median(levels_db);mad=np.median(np.abs(levels_db-med))+1e-9
    z=.6745*(levels_db-med)/mad
    return np.abs(z)>=z_min

def bounded_event_gain(level_db:float,median_db:float,max_move_db:float=2.0)->float:
    # Partial correction only. Preserve performance accents.
    delta=(median_db-level_db)*.55
    return float(np.clip(delta,-max_move_db,max_move_db))

def click_candidates(x:np.ndarray,sr:int,z_threshold:float=20.)->list[int]:
    if x.ndim>1:x=x.mean(axis=1)
    d=np.abs(np.diff(x));med=np.median(d);mad=np.median(np.abs(d-med))+1e-12
    idx=np.where(d>med+z_threshold*mad)[0];out=[]
    for i in idx:
        if i<3 or i>=len(d)-3:continue
        local=max(float(d[i-3:i].max()),float(d[i+1:i+4].max()),1e-9)
        if d[i]>6*local:out.append(int(i))
    return out
