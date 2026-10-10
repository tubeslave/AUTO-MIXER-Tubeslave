from __future__ import annotations
from itertools import combinations
from typing import Any

import numpy as np
import soundfile as sf

BANDS=[(60,120),(120,250),(250,500),(500,1000),(1000,2000),(2000,4000),(4000,8000)]

def _read(path: str):
    y,sr=sf.read(path,always_2d=True,dtype="float32")
    return y.mean(axis=1),sr

def _band_activity(x: np.ndarray, sr: int, frame_s=.1, hop_s=.05) -> np.ndarray:
    n=max(256,int(frame_s*sr)); hop=max(128,int(hop_s*sr))
    if len(x)<n: x=np.pad(x,(0,n-len(x)))
    win=np.hanning(n)
    f=np.fft.rfftfreq(n,1/sr)
    rows=[]
    for start in range(0,len(x)-n+1,hop):
        p=np.abs(np.fft.rfft(x[start:start+n]*win))**2+1e-20
        rows.append([10*np.log10(np.sum(p[(f>=lo)&(f<min(hi,sr/2))])+1e-20) for lo,hi in BANDS])
    return np.asarray(rows,dtype=float)

def dynamic_masking_graph(tracks: list[dict[str,Any]], priorities: dict[str,float],
                          activity_floor_db: float=-45.0) -> dict[str,Any]:
    profiles={}
    min_frames=None
    for t in tracks:
        p=_band_activity(*_read(t["path"]))
        profiles[t["name"]]=p
        min_frames=len(p) if min_frames is None else min(min_frames,len(p))
    edges=[]
    for a,b in combinations(tracks,2):
        pa=profiles[a["name"]][:min_frames]; pb=profiles[b["name"]][:min_frames]
        aa=pa-np.max(pa,axis=0,keepdims=True); bb=pb-np.max(pb,axis=0,keepdims=True)
        active=(aa>activity_floor_db)&(bb>activity_floor_db)
        closeness=np.exp(-np.abs(pa-pb)/10.0)
        frame_scores=np.mean(closeness*active,axis=1)
        score=float(np.mean(frame_scores))
        band_scores=np.mean(closeness*active,axis=0)
        j=int(np.argmax(band_scores))
        pa_pr=float(priorities.get(a["name"],.5)); pb_pr=float(priorities.get(b["name"],.5))
        if pa_pr>pb_pr:
            protect,candidate=a["name"],b["name"]
        elif pb_pr>pa_pr:
            protect,candidate=b["name"],a["name"]
        else:
            protect,candidate=None,None
        edges.append({
            "a":a["name"],"b":b["name"],"score":score,
            "strongest_band_hz":list(BANDS[j]),
            "active_frame_fraction":float(np.mean(frame_scores>0.1)),
            "protect":protect,"candidate_for_movement":candidate,
            "priority_delta":abs(pa_pr-pb_pr),
        })
    edges.sort(key=lambda e:e["score"],reverse=True)
    return {"edges":edges,
            "policy":"masking is time-varying evidence; priority selects who to protect, not how much to attenuate",
            "limitations":["energy overlap is not a psychoacoustic proof of masking","equal-priority conflicts require further evidence"]}
