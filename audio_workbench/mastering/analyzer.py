from __future__ import annotations
import numpy as np
from scipy import signal

BANDS={"sub":(25,80),"low":(80,200),"lowmid":(200,500),"mid":(500,2000),"presence":(2000,5000),"air":(5000,16000)}

def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    modern = getattr(np, "trapezoid", None)
    if modern is not None:
        return float(modern(y, x))
    return float(np.trapz(y, x))

def true_peak_dbtp(x: np.ndarray, oversample: int = 4) -> float:
    x=np.asarray(x,dtype=np.float64)
    if x.ndim==1:x=x[:,None]
    if len(x)==0:return float("-inf")
    if oversample<=1:
        peak=float(np.max(np.abs(x))+1e-20)
    else:
        up=signal.resample_poly(x,oversample,1,axis=0)
        peak=float(np.max(np.abs(up))+1e-20)
    return float(20*np.log10(peak))

def integrated_lufs(x: np.ndarray, sr: int) -> tuple[float|None,str]:
    """Measure BS.1770-style integrated loudness via pyloudnorm when available.

    No RMS approximation is returned here: a missing standards-based meter must remain
    visible to the mastering safety gate instead of masquerading as LUFS evidence.
    """
    try:
        import pyloudnorm as pyln
        data=np.asarray(x,dtype=np.float64)
        if data.ndim==1:data=data[:,None]
        if len(data)<int(.4*sr):
            return None,"insufficient_duration"
        return float(pyln.Meter(sr).integrated_loudness(data)),"pyloudnorm"
    except Exception:
        return None,"unavailable"

def analyze(x: np.ndarray, sr: int, *, include_true_peak: bool=False, include_loudness: bool=False) -> dict:
    x=np.asarray(x,dtype=np.float32)
    if x.ndim==1:x=np.column_stack([x,x])
    mono=x.mean(1); rms=float(np.sqrt(np.mean(x.astype(np.float64)**2)+1e-20))
    peak=float(np.max(np.abs(x))+1e-20)
    f,p=signal.welch(mono,fs=sr,nperseg=min(8192,len(mono)))
    bands={}
    for k,(lo,hi) in BANDS.items():
        m=(f>=lo)&(f<hi); bands[k]=float(10*np.log10(_trapezoid(p[m],f[m])+1e-20))
    mid=(x[:,0]+x[:,1])*.5;side=(x[:,0]-x[:,1])*.5
    result={"rms_dbfs":float(20*np.log10(rms)),"sample_peak_dbfs":float(20*np.log10(peak)),
      "crest_db":float(20*np.log10(peak/rms)),"bands_db":bands,
      "side_mid_db":float(10*np.log10((np.mean(side*side)+1e-20)/(np.mean(mid*mid)+1e-20))),
      "correlation":float(np.corrcoef(x[:,0],x[:,1])[0,1])}
    if include_true_peak:
        result["true_peak_dbtp"]=true_peak_dbtp(x)
    if include_loudness:
        value,method=integrated_lufs(x,sr)
        result["integrated_lufs"]=value
        result["integrated_lufs_method"]=method
    return result
