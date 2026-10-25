from __future__ import annotations
import numpy as np
from scipy import ndimage

def section_automation(role:str,density:np.ndarray,sr:int,hop_s:float,n_samples:int)->np.ndarray:
    """Small role-aware rides. Section energy comes from the song, not hard-coded verse/chorus labels."""
    d=np.asarray(density,dtype="float32")
    # Dense sections: vocal anchor stable, drums/bass modestly forward, guitars expand without loudness jump.
    if role=="vocal": db=.35*(.5-d)
    elif role in ("kick","snare"): db=.55*(d-.45)
    elif role=="bass": db=.35*(d-.45)
    elif role=="guitar": db=.25*(d-.50)
    elif role in ("keys","playback"): db=.25*(.45-d)
    else: db=np.zeros_like(d)
    db=np.clip(db,-.6,.6)
    t=np.arange(len(db))*hop_s;st=np.arange(n_samples)/sr
    curve=np.interp(st,t,db,left=db[0],right=db[-1]).astype("float32")
    return ndimage.gaussian_filter1d(curve,sigma=max(1,int(.75*sr)))
