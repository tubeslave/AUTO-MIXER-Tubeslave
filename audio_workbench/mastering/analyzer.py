from __future__ import annotations
import numpy as np
from scipy import signal

BANDS={"sub":(25,80),"low":(80,200),"lowmid":(200,500),"mid":(500,2000),"presence":(2000,5000),"air":(5000,16000)}

def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate a spectrum across NumPy 1.x/2.x API changes."""
    modern = getattr(np, "trapezoid", None)
    if modern is not None:
        return float(modern(y, x))
    return float(np.trapz(y, x))

def analyze(x: np.ndarray, sr: int) -> dict:
    x=np.asarray(x,dtype=np.float32)
    if x.ndim==1:x=np.column_stack([x,x])
    mono=x.mean(1); rms=float(np.sqrt(np.mean(x.astype(np.float64)**2)+1e-20))
    peak=float(np.max(np.abs(x))+1e-20)
    f,p=signal.welch(mono,fs=sr,nperseg=min(8192,len(mono)))
    bands={}
    for k,(lo,hi) in BANDS.items():
        m=(f>=lo)&(f<hi); bands[k]=float(10*np.log10(_trapezoid(p[m],f[m])+1e-20))
    mid=(x[:,0]+x[:,1])*.5;side=(x[:,0]-x[:,1])*.5
    return {"rms_dbfs":float(20*np.log10(rms)),"sample_peak_dbfs":float(20*np.log10(peak)),
      "crest_db":float(20*np.log10(peak/rms)),"bands_db":bands,
      "side_mid_db":float(10*np.log10((np.mean(side*side)+1e-20)/(np.mean(mid*mid)+1e-20))),
      "correlation":float(np.corrcoef(x[:,0],x[:,1])[0,1])}
