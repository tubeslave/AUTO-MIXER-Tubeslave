from __future__ import annotations
import numpy as np

def level_stability(frame_db:np.ndarray)->dict:
    a=frame_db[np.isfinite(frame_db)]
    if len(a)<8:return {"spread_db":0.0,"p95_jump_db":0.0}
    spread=float(np.percentile(a,90)-np.percentile(a,20))
    jumps=np.abs(np.diff(a))
    return {"spread_db":spread,"p95_jump_db":float(np.percentile(jumps,95))}

def accept(before:dict,after:dict,max_spread_reduction_db:float=5.0)->dict:
    fail=[]
    reduction=before["spread_db"]-after["spread_db"]
    if reduction<0:fail.append("less_stable")
    if reduction>max_spread_reduction_db:fail.append("over_flattened")
    return {"accept":not fail,"spread_reduction_db":float(reduction),"failures":fail}
