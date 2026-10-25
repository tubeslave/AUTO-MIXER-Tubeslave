from __future__ import annotations
import numpy as np

def infer_straight_grid(event_times_s:np.ndarray,bpm_min=70.,bpm_max=180.,subdivision=4)->dict:
    """Fit a robust straight subdivision grid. Used only to detect candidates, never to quantize globally."""
    events=np.asarray(event_times_s,dtype=float)
    best=None
    for bpm in np.linspace(bpm_min,bpm_max,441):
        step=60./bpm/subdivision
        for phase in np.linspace(0,step,48,endpoint=False):
            d=((events-phase+step/2)%step)-step/2
            score=float(np.mean(np.exp(-(d/.035)**2)))
            if best is None or score>best[0]:best=(score,bpm,step,phase)
    return {"score":best[0],"bpm":best[1],"step_s":best[2],"phase_s":best[3]}

def isolated_outliers(event_times_s:np.ndarray,grid:dict,min_dev_ms=35,max_dev_ms=60,
                      neighbor_dev_ms=18,max_nudge_ms=12)->list[dict]:
    t=np.asarray(event_times_s,float);step=grid["step_s"];phase=grid["phase_s"]
    d=((t-phase+step/2)%step)-step/2;out=[]
    for i in range(1,len(t)-1):
        ms=abs(d[i])*1000
        if not(min_dev_ms<=ms<=max_dev_ms):continue
        if abs(d[i-1])*1000>neighbor_dev_ms or abs(d[i+1])*1000>neighbor_dev_ms:continue
        nudge=float(np.clip(-d[i]*1000,-max_nudge_ms,max_nudge_ms))
        out.append({"index":i,"time_s":float(t[i]),"grid_deviation_ms":float(d[i]*1000),"nudge_ms":nudge})
    return out
