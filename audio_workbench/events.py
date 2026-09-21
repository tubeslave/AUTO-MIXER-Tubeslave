from __future__ import annotations
from typing import Any
import numpy as np
import soundfile as sf
from scipy import signal

def analyze_events(path: str, frame_ms: float=10, hop_ms: float=5) -> dict[str,Any]:
    y,sr=sf.read(path,always_2d=True,dtype="float32")
    x=np.mean(y,axis=1)
    n=max(64,int(sr*frame_ms/1000)); hop=max(32,int(sr*hop_ms/1000))
    power=signal.convolve(x*x,np.ones(n)/n,mode="same")
    env=np.sqrt(power+1e-20)[::hop]
    novelty=np.maximum(np.diff(env,prepend=env[0]),0)
    med=np.median(novelty); mad=np.median(np.abs(novelty-med))+1e-12
    threshold=med+4*mad
    peaks,_=signal.find_peaks(novelty,height=threshold,distance=max(1,int(.025*sr/hop)))
    events=[]
    for p in peaks[:1000]:
        center=p*hop
        attack=max(0,center-int(.03*sr)); body=min(len(x),center+int(.12*sr))
        seg=x[attack:body]
        if not len(seg): continue
        pk=float(np.max(np.abs(seg))); rms=float(np.sqrt(np.mean(seg*seg)+1e-20))
        events.append({"time_s":center/sr,"strength":float(novelty[p]),
                       "local_crest_db":float(20*np.log10(max(pk/max(rms,1e-12),1e-12)))})
    return {"event_count":len(events),"events":events,
            "limitations":["generic events are not source labels","event detection does not imply quality"]}
