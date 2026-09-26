from __future__ import annotations

from itertools import combinations
from typing import Any
import numpy as np
import soundfile as sf
from scipy import signal

BANDS = [(40,80),(80,160),(160,315),(315,630),(630,1250),(1250,2500),(2500,5000),(5000,10000),(10000,18000)]

def _read(path: str) -> tuple[np.ndarray,int]:
    y,sr = sf.read(path,always_2d=True,dtype="float32")
    return y.mean(axis=1),sr

def _frames(x: np.ndarray, n: int, hop: int):
    if len(x)<n:
        x=np.pad(x,(0,n-len(x)))
    count=1+(len(x)-n)//hop
    for i in range(count):
        yield x[i*hop:i*hop+n]

def _profile(path: str, frame_s: float=.1, hop_s: float=.05) -> tuple[np.ndarray,int]:
    x,sr=_read(path)
    n=max(256,int(frame_s*sr)); hop=max(128,int(hop_s*sr))
    win=np.hanning(n)
    freqs=np.fft.rfftfreq(n,1/sr)
    vals=[]
    for fr in _frames(x,n,hop):
        p=np.abs(np.fft.rfft(fr*win))**2
        vals.append([float(np.sum(p[(freqs>=lo)&(freqs<min(hi,sr/2))])) for lo,hi in BANDS])
    a=np.asarray(vals,dtype=float)+1e-20
    return 10*np.log10(a),sr

def masking_graph(tracks: list[dict[str,Any]], activity_floor_db: float=-45.0) -> dict[str,Any]:
    profiles={}
    min_frames=None
    for t in tracks:
        p,_=_profile(t["path"])
        profiles[t["name"]]=p
        min_frames=len(p) if min_frames is None else min(min_frames,len(p))
    edges=[]
    for a,b in combinations(tracks,2):
        pa=profiles[a["name"]][:min_frames]; pb=profiles[b["name"]][:min_frames]
        aa=pa-np.max(pa,axis=0,keepdims=True)
        bb=pb-np.max(pb,axis=0,keepdims=True)
        active=(aa>activity_floor_db)&(bb>activity_floor_db)
        closeness=np.exp(-np.abs(pa-pb)/12.0)
        score=float(np.mean(closeness*active))
        band_scores=np.mean(closeness*active,axis=0)
        j=int(np.argmax(band_scores))
        edges.append({
          "a":a["name"],"b":b["name"],"overlap_score":score,
          "strongest_band_hz":list(BANDS[j]),"strongest_band_score":float(band_scores[j])
        })
    edges.sort(key=lambda x:x["overlap_score"],reverse=True)
    return {
      "edges":edges,
      "limitations":[
        "overlap_score is a diagnostic energy-overlap proxy, not proof of audible masking",
        "musical priority and source audibility must be checked before any attenuation",
        "do not auto-EQ from this graph"
      ]
    }
