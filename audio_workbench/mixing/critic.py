from __future__ import annotations
import numpy as np

def score(metrics:dict)->dict:
    """Diagnostic guardrails, not a subjective winner score."""
    fail=[]
    if metrics.get("peak_dbfs",0)>-1.0: fail.append("insufficient_headroom")
    if metrics.get("vocal_to_music_db",-99)<-5: fail.append("vocal_buried")
    if metrics.get("vocal_to_music_db",99)>4: fail.append("vocal_detached")
    if metrics.get("kick_to_bass_db",-99)<-6: fail.append("kick_buried_by_bass")
    if metrics.get("kick_to_bass_db",99)>6: fail.append("bass_underweight")
    if metrics.get("section_gain_jump_db",0)>1.0: fail.append("section_loudness_jump")
    return {"accept":not fail,"failures":fail}

def ratio_db(a:np.ndarray,b:np.ndarray)->float:
    ra=np.sqrt(np.mean(a.astype("float64")**2)+1e-20)
    rb=np.sqrt(np.mean(b.astype("float64")**2)+1e-20)
    return float(20*np.log10((ra+1e-12)/(rb+1e-12)))
