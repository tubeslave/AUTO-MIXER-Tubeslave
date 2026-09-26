from __future__ import annotations

import hashlib
from typing import Any
import numpy as np
import soundfile as sf
from scipy import signal

def _read(path: str) -> tuple[np.ndarray,int]:
    y,sr=sf.read(path,always_2d=True,dtype="float32")
    return y,sr

def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y*y)+1e-20))

def compare(a_path: str,b_path: str, loudness_match: bool=True) -> dict[str,Any]:
    a,sra=_read(a_path); b,srb=_read(b_path)
    if sra!=srb:
        raise ValueError("sample rates differ")
    if len(a) != len(b):
        raise ValueError(f"render lengths differ: {len(a)} != {len(b)} frames")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"channel counts differ: {a.shape[1]} != {b.shape[1]}")
    n=len(a); c=a.shape[1]
    gain_db=0.0
    if loudness_match:
        ra,rb=_rms(a),_rms(b)
        gain=ra/max(rb,1e-12)
        b=b*gain
        gain_db=float(20*np.log10(max(gain,1e-12)))
    d=b-a
    mono_a=a.mean(axis=1); mono_b=b.mean(axis=1)
    f,coh=signal.coherence(mono_a,mono_b,fs=sra,nperseg=min(8192,n))
    return {
      "duration_s":n/sra,"channels":c,"b_match_gain_db":gain_db,
      "difference_rms_dbfs":float(20*np.log10(max(_rms(d),1e-12))),
      "difference_peak_dbfs":float(20*np.log10(max(float(np.max(np.abs(d))),1e-12))),
      "waveform_correlation":float(np.corrcoef(mono_a,mono_b)[0,1]) if n>1 else 1.0,
      "mean_magnitude_squared_coherence":float(np.nanmean(coh)),
      "limitations":[
        "RMS matching is not a substitute for perceptual loudness matching",
        "difference size is not a quality score",
        "preference requires a separate listening/evaluation step"
      ]
    }

def blind_labels(a_path: str,b_path: str, salt: str) -> dict[str,str]:
    h=hashlib.sha256((salt+a_path+b_path).encode()).digest()[0]
    return {"A":a_path,"B":b_path} if h%2==0 else {"A":b_path,"B":a_path}
