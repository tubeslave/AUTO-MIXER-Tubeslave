from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal,ndimage

@dataclass(frozen=True)
class GuitarPolicy:
    wall_layer_max_db:float=-16.0
    lowmid_control_max_db:float=1.2
    presence_control_max_db:float=1.0
    dense_width_max:float=.18

def lowmid_control_db(x:np.ndarray,sr:int,max_db:float=1.2)->np.ndarray:
    if x.ndim>1:x=x.mean(1)
    lm=signal.sosfilt(signal.butter(2,[140,420],btype="bandpass",fs=sr,output="sos"),x)
    body=signal.sosfilt(signal.butter(2,[500,1800],btype="bandpass",fs=sr,output="sos"),x)
    a=np.sqrt(ndimage.uniform_filter1d(lm*lm,int(.04*sr))+1e-12)
    b=np.sqrt(ndimage.uniform_filter1d(body*body,int(.04*sr))+1e-12)
    r=20*np.log10((a+1e-9)/(b+1e-9));th=float(np.percentile(r,84))
    return -ndimage.gaussian_filter1d(np.clip((r-th)*.4,0,max_db).astype("float32"),int(.025*sr))

def wall_layer(x:np.ndarray,sr:int,density:np.ndarray)->np.ndarray:
    """Low-level decorrelated stereo layer. Original guitar remains the anchor."""
    if x.ndim>1:x=x.mean(1)
    l=np.r_[np.zeros(int(.009*sr),dtype="float32"),x][0:len(x)]
    r=np.r_[np.zeros(int(.014*sr),dtype="float32"),x][0:len(x)]
    l=signal.sosfilt(signal.butter(2,[120,9000],btype="bandpass",fs=sr,output="sos"),l).astype("float32")
    r=signal.sosfilt(signal.butter(2,[120,9000],btype="bandpass",fs=sr,output="sos"),r).astype("float32")
    d=np.asarray(density,dtype="float32")
    amount=np.clip((d-.48)*1.25,0,1)*10**(-16/20)
    return np.column_stack([l*amount,r*amount]).astype("float32")

def accept(metrics:dict)->dict:
    fail=[]
    if metrics.get("guitar_rms_change_db",0)>1.0:fail.append("guitar_level_cheat")
    if metrics.get("mix_side_gain_db",0)>1.2:fail.append("too_wide")
    if metrics.get("mix_lowmid_shift_db",0)<-.7:fail.append("mix_thinned")
    if metrics.get("mix_presence_shift_db",0)>.7:fail.append("presence_harshness")
    return {"accept":not fail,"failures":fail}
