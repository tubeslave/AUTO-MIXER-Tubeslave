from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal,ndimage

@dataclass(frozen=True)
class BassPolicy:
    kick_duck_max_db:float=1.4
    kick_duck_release_ms:float=95.
    mid_control_max_db:float=1.5
    harmonic_drive_db:float=2.0
    harmonic_mix:float=.16

def envelope(x:np.ndarray,sr:int,lo:float,hi:float,window_ms:float=28)->np.ndarray:
    if x.ndim>1:x=x.mean(1)
    y=signal.sosfiltfilt(signal.butter(2,[lo,hi],btype="bandpass",fs=sr,output="sos"),x)
    return np.sqrt(ndimage.uniform_filter1d(y*y,size=max(3,int(window_ms*sr/1000)))+1e-12)

def kick_duck_db(kick:np.ndarray,sr:int,max_db:float=1.4)->np.ndarray:
    e=envelope(kick,sr,45,110,18)
    lo,hi=np.percentile(e,[62,94])
    a=np.clip((e-lo)/(hi-lo+1e-12),0,1)
    a=ndimage.maximum_filter1d(a,size=max(3,int(.025*sr)))
    a=ndimage.gaussian_filter1d(a,sigma=max(1,int(.028*sr)))
    return -max_db*a.astype("float32")

def mid_control_db(bass:np.ndarray,sr:int,max_db:float=1.5)->np.ndarray:
    mid=envelope(bass,sr,250,900,35);low=envelope(bass,sr,55,180,35)
    ratio=20*np.log10((mid+1e-9)/(low+1e-9));th=float(np.percentile(ratio,78))
    gr=np.clip((ratio-th)*.45,0,max_db)
    return -ndimage.gaussian_filter1d(gr.astype("float32"),sigma=max(1,int(.025*sr)))

def harmonic_layer(bass:np.ndarray,drive_db:float=2.0,mix:float=.16)->np.ndarray:
    g=10**(drive_db/20);sat=np.tanh(bass*g)/g
    layer=sat-bass
    return (bass+layer*mix).astype("float32")

def accept(metrics:dict)->dict:
    fail=[]
    if metrics.get("bass_rms_change_db",0)>1.0:fail.append("bass_level_cheat")
    if metrics.get("kick_band_shift_db",0)>.6:fail.append("low_end_rebalanced_too_far")
    if metrics.get("bass_mid_shift_db",0)<-1.8:fail.append("bass_hollowed")
    if metrics.get("small_speaker_band_gain_db",0)>1.2:fail.append("too_much_harmonic_audibility")
    return {"accept":not fail,"failures":fail}
