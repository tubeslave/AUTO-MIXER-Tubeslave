from __future__ import annotations
import numpy as np
from scipy import signal

def best_alignment(reference:np.ndarray,target:np.ndarray,sr:int,max_ms:float=8.0)->dict:
    n=min(len(reference),len(target));r=reference[:n].astype("float64");t=target[:n].astype("float64")
    # transient/low-mid weighted correlation for multi-mic sources
    r=signal.lfilter([1,-1],[1],r);t=signal.lfilter([1,-1],[1],t)
    maxlag=max(1,int(sr*max_ms/1000))
    c=signal.correlate(t,r,mode="full",method="fft")
    lags=signal.correlation_lags(len(t),len(r),mode="full")
    m=(lags>=-maxlag)&(lags<=maxlag);c=c[m];lags=lags[m]
    i=int(np.argmax(np.abs(c)));lag=int(lags[i]);polarity=1 if c[i]>=0 else -1
    norm=np.sqrt(np.sum(r*r)*np.sum(t*t))+1e-20
    return {"lag_samples":lag,"lag_ms":1000*lag/sr,"polarity":polarity,
            "confidence":float(abs(c[i])/norm)}

def shift_preserve_length(x:np.ndarray,samples:int)->np.ndarray:
    y=np.zeros_like(x)
    if samples>0:y[samples:]=x[:-samples]
    elif samples<0:y[:samples]=x[-samples:]
    else:y[:]=x
    return y
