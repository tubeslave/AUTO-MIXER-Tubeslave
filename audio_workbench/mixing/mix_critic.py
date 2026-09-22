from __future__ import annotations
import numpy as np

def section_spread(values:list[float])->float:
    a=np.asarray(values,dtype=float)
    return float(np.percentile(a,90)-np.percentile(a,10)) if len(a)>=3 else 0.

def _active_adjacent_jump(rows:list[dict])->float:
    """Never compare two active rows that were separated by skipped silence/inactive sections."""
    jumps=[]
    for i in range(1,len(rows)):
        if rows[i].get("active",True) and rows[i-1].get("active",True):
            jumps.append(abs(rows[i]["rms_db"]-rows[i-1]["rms_db"]))
    return max(jumps,default=0.)

def diagnostics(section_rows:list[dict],global_metrics:dict)->dict:
    active=[r for r in section_rows if r.get("active",True)]
    vocal=[r["vocal_music_db"] for r in active if "vocal_music_db" in r]
    kb=[r["kick_bass_db"] for r in active if "kick_bass_db" in r]
    width=[r["side_mid_db"] for r in active if "side_mid_db" in r]
    density=[r["density"] for r in active if "density" in r]
    dense=[w for w,d in zip(width,density) if d>=.7]
    sparse=[w for w,d in zip(width,density) if d<=.35]
    width_def=0.
    if dense and sparse:
        observed=float(np.median(dense)-np.median(sparse))
        width_def=max(0.,1.0-observed)
    return {**global_metrics,
      "vocal_section_spread_db":section_spread(vocal),
      "kick_bass_section_drift_db":section_spread(kb),
      "dense_section_width_deficit_db":width_def,
      "section_loudness_jump_db":_active_adjacent_jump(section_rows),
    }
