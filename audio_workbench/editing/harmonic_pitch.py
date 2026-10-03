from __future__ import annotations
import numpy as np

def local_pitch_class_support(chroma:np.ndarray,times:np.ndarray,start_s:float,end_s:float)->np.ndarray:
    m=(times>=start_s-.12)&(times<=end_s+.12)
    if not np.any(m): return np.ones(12,dtype=float)/12
    v=np.median(chroma[:,m],axis=1).astype(float)
    return v/(np.sum(v)+1e-12)

def choose_contextual_target(observed_midi:float,support:np.ndarray,max_distance_semitones:int=2)->dict:
    """Choose among nearby chromatic notes using both intonation distance and local harmonic support."""
    center=int(round(observed_midi));rows=[]
    for n in range(center-max_distance_semitones,center+max_distance_semitones+1):
        pc=n%12;dist=abs(observed_midi-n)
        # Distance remains dominant; context can break ambiguous near-semitone cases.
        score=1.25*dist-0.55*np.log(float(support[pc])+1e-5)
        rows.append({"midi":n,"pitch_class":pc,"distance_semitones":float(dist),
                     "support":float(support[pc]),"score":float(score)})
    rows.sort(key=lambda r:r["score"])
    return {"chosen":rows[0],"alternatives":rows[1:]}

def correction_from_context(observed_midi:float,target_midi:int,support:float,
                            strength:float=.65,max_cents:float=35)->dict:
    error=(observed_midi-target_midi)*100
    # Require meaningful harmonic support before a larger correction.
    confidence_scale=float(np.clip((support-.03)/.12,.35,1.0))
    correction=float(np.clip(-error*strength*confidence_scale,-max_cents,max_cents))
    return {"error_cents":float(error),"correction_cents":correction,
            "context_confidence_scale":confidence_scale}
