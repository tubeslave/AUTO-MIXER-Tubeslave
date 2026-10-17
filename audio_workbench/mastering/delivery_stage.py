from __future__ import annotations
import numpy as np
from scipy import signal

def soft_clip_chunk(x:np.ndarray,drive_db:float=3.0,threshold:float=.82,oversample:int=4)->np.ndarray:
    """Delivery soft clipper. Does NOT normalize each chunk: macro dynamics must survive streaming."""
    up=signal.resample_poly(x,oversample,1,axis=0).astype("float32")
    up*=np.float32(10**(drive_db/20))
    a=np.abs(up);sg=np.sign(up);u=np.maximum(a-threshold,0)/(1-threshold)
    up=np.where(a>threshold,sg*(threshold+(1-threshold)*np.tanh(u)),up).astype("float32")
    return signal.resample_poly(up,1,oversample,axis=0)[:len(x)].astype("float32")

def global_ceiling_trim(true_peak_dbtp:float,ceiling_dbtp:float=-1.03)->float:
    """One constant trim after all chunks are assembled. Never apply per-chunk ceiling gain."""
    return min(0.0,ceiling_dbtp-true_peak_dbtp)
