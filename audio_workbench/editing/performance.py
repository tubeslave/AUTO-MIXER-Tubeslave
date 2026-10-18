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

def robust_outliers(levels_db:np.ndarray,z_min:float=2.8,
                    min_events:int=6,min_scale_db:float=.25)->np.ndarray:
    """Flag only strong local level outliers while preserving short musical phrases.

    Six detected events are enough to identify an extreme defect, but passages shorter
    than that remain diagnose-only.  The robust-scale floor avoids treating tiny
    sub-dB performance variation as an outlier when MAD is close to zero.
    """
    levels=np.asarray(levels_db,dtype="float64")
    if len(levels)<min_events:
        return np.zeros(len(levels),dtype=bool)
    med=float(np.median(levels))
    mad=float(np.median(np.abs(levels-med)))
    scale=max(mad,float(min_scale_db),1e-9)
    z=.6745*(levels-med)/scale
    return np.abs(z)>=z_min

def bounded_event_gain(level_db:float,median_db:float,max_move_db:float=2.0)->float:
    # Partial correction only. Preserve performance accents.
    delta=(median_db-level_db)*.55
    return float(np.clip(delta,-max_move_db,max_move_db))

def click_candidates(x:np.ndarray,sr:int,z_threshold:float=20.)->list[int]:
    """Return sample-local discontinuity/click candidates.

    Candidate derivative samples are clustered before local comparison.  A one-sample
    impulse produces two adjacent large derivatives (rise and fall); treating each one
    independently makes the other look like legitimate local activity and suppresses
    the click.  Clustering fixes that failure while rejecting broader transient shapes.
    """
    x=np.asarray(x)
    if x.ndim>1:x=x.mean(axis=1)
    if len(x)<3:return []
    d=np.abs(np.diff(x));med=float(np.median(d));mad=float(np.median(np.abs(d-med)))+1e-12
    idx=np.flatnonzero(d>med+z_threshold*mad)
    if len(idx)==0:return []

    groups=[];start=prev=int(idx[0])
    for raw in idx[1:]:
        i=int(raw)
        if i==prev+1:
            prev=i;continue
        groups.append((start,prev));start=prev=i
    groups.append((start,prev))

    out=[]
    for start,end in groups:
        # True sample discontinuities are very narrow.  Wider supra-threshold runs are
        # more likely musical attacks/ramps and stay diagnose-only here.
        if end-start+1>3:continue
        left=d[max(0,start-3):start];right=d[end+1:min(len(d),end+4)]
        outside=np.concatenate((left,right)) if left.size or right.size else np.empty(0)
        local=max(float(outside.max()) if outside.size else 0.0,1e-9)
        peak=float(d[start:end+1].max())
        if peak>6*local:
            # Map derivative cluster back to the nearest affected audio sample.
            out.append(int((start+end+2)//2))
    return out
