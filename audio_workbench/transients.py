from __future__ import annotations

from typing import Any
import numpy as np
import soundfile as sf
from scipy import signal

def analyze_transients(path: str, frame_ms: float=10.0, hop_ms: float=5.0) -> dict[str,Any]:
    y,sr=sf.read(path,always_2d=True,dtype="float32")
    x=np.mean(y,axis=1)
    n=max(64,int(sr*frame_ms/1000)); hop=max(32,int(sr*hop_ms/1000))
    # rectified short-time RMS envelope
    power=signal.convolve(x*x,np.ones(n)/n,mode="same")
    env=np.sqrt(power+1e-20)
    sampled=env[::hop]
    flux=np.maximum(np.diff(sampled,prepend=sampled[0]),0)
    if len(flux)==0:
        return {"event_count":0,"events":[]}
    threshold=float(np.median(flux)+4*np.median(np.abs(flux-np.median(flux))))
    peaks,_=signal.find_peaks(flux,height=max(threshold,1e-8),distance=max(1,int(.03*sr/hop)))
    events=[]
    for p in peaks[:500]:
        t=p*hop/sr
        events.append({"time_s":float(t),"onset_strength":float(flux[p])})
    intervals=np.diff([e["time_s"] for e in events])
    return {
      "event_count":len(events),"events":events,
      "median_inter_event_s":float(np.median(intervals)) if len(intervals) else None,
      "threshold":threshold,
      "limitations":["event detector is generic; it does not know whether an onset is musically desirable"]
    }
