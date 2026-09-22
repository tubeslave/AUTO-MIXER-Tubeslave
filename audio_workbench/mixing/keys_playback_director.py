from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal,ndimage

@dataclass(frozen=True)
class KeysPlaybackPolicy:
    keys_lowmid_max_db:float=1.2
    playback_lowmid_max_db:float=.9
    playback_ride_max_db:float=.6
    keys_width_max_db:float=.7

def band_env(x:np.ndarray,sr:int,lo:float,hi:float,ms:float=45)->np.ndarray:
    if x.ndim>1:x=x.mean(1)
    y=signal.sosfiltfilt(signal.butter(2,[lo,hi],btype="bandpass",fs=sr,output="sos"),x)
    return np.sqrt(ndimage.uniform_filter1d(y*y,max(3,int(ms*sr/1000)))+1e-12)

def dynamic_lowmid_db(x:np.ndarray,sr:int,lo:float,hi:float,max_db:float)->np.ndarray:
    lm=band_env(x,sr,lo,hi);body=band_env(x,sr,600,2200)
    ratio=20*np.log10((lm+1e-9)/(body+1e-9))
    th=float(np.percentile(ratio,86))
    gr=np.clip((ratio-th)*.36,0,max_db)
    return -ndimage.gaussian_filter1d(gr.astype("float32"),sigma=max(1,int(.03*sr)))

def playback_section_ride(frame_db:np.ndarray,active:np.ndarray,max_db:float=.6)->np.ndarray:
    target=float(np.median(frame_db[active]))
    move=np.where(active,np.clip((target-frame_db)*.22,-max_db,max_db),0)
    return ndimage.gaussian_filter1d(move.astype("float32"),sigma=2)

def accept(metrics:dict)->dict:
    fail=[]
    if abs(metrics.get("mix_loudness_change_lu",0))>.35:fail.append("loudness_cheat")
    if abs(metrics.get("mix_lowmid_shift_db",0))>.45:fail.append("excess_lowmid_change")
    if metrics.get("side_energy_gain_db",0)>1.0:fail.append("excess_width")
    if metrics.get("playback_section_spread_reduction_db",0)>3.0:fail.append("playback_overflattened")
    return {"accept":not fail,"failures":fail}
