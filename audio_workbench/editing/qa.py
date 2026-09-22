from __future__ import annotations
import numpy as np

def compare_track(original:np.ndarray,edited:np.ndarray)->dict:
    n=min(len(original),len(edited));a=original[:n].astype("float64");b=edited[:n].astype("float64")
    peak_a=float(np.max(np.abs(a)));peak_b=float(np.max(np.abs(b)))
    rms_a=float(np.sqrt(np.mean(a*a)+1e-20));rms_b=float(np.sqrt(np.mean(b*b)+1e-20))
    return {
      "same_length":len(original)==len(edited),
      "peak_change_db":float(20*np.log10((peak_b+1e-12)/(peak_a+1e-12))),
      "rms_change_db":float(20*np.log10((rms_b+1e-12)/(rms_a+1e-12))),
      "new_clip_samples":int(np.sum(np.abs(b)>=1.0))-int(np.sum(np.abs(a)>=1.0)),
    }

def phase_pair(a:np.ndarray,b:np.ndarray)->float:
    n=min(len(a),len(b));x=a[:n].reshape(-1);y=b[:n].reshape(-1)
    if np.std(x)<1e-10 or np.std(y)<1e-10:return 0.
    return float(np.corrcoef(x,y)[0,1])

def decide(track_rows:list[dict],artifact_rows:list[dict])->dict:
    fail=[]
    for r in track_rows:
        if not r["same_length"]:fail.append(f'{r["track"]}:length')
        if r["new_clip_samples"]>0:fail.append(f'{r["track"]}:new_clipping')
        if abs(r["rms_change_db"])>3.0:fail.append(f'{r["track"]}:unexpected_global_level')
    for r in artifact_rows:
        if r.get("new_clicks",0)>0:fail.append(f'{r["track"]}:new_clicks')
    return {"accept":not fail,"failures":fail}
