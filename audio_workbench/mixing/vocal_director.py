from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal,ndimage

@dataclass(frozen=True)
class VocalPolicy:
    lead_ride_db:float=1.25
    secondary_ride_db:float=1.0
    deess_max_db:float=3.0
    saturation_drive_db:float=1.2
    secondary_dense_gain_db:float=.55

def band_envelope(x:np.ndarray,sr:int,lo:float,hi:float,window_ms:float=25)->np.ndarray:
    if x.ndim>1:x=x.mean(1)
    y=signal.sosfiltfilt(signal.butter(2,[lo,hi],btype="bandpass",fs=sr,output="sos"),x)
    return np.sqrt(ndimage.uniform_filter1d(y*y,size=max(3,int(window_ms*sr/1000)))+1e-12)

def deess_curve_db(x:np.ndarray,sr:int,max_db:float=3.0)->np.ndarray:
    sib=band_envelope(x,sr,4800,min(10500,sr*.45))
    body=band_envelope(x,sr,700,3500)
    ratio=20*np.log10((sib+1e-9)/(body+1e-9))
    threshold=float(np.percentile(ratio,84))
    gr=np.clip((ratio-threshold)*.55,0,max_db)
    return ndimage.gaussian_filter1d(gr.astype("float32"),sigma=max(1,int(.012*sr)))

def slow_vocal_ride_db(x:np.ndarray,sr:int,max_db:float=1.25)->np.ndarray:
    if x.ndim>1:x=x.mean(1)
    hop=max(1,int(.05*sr));n=len(x)//hop
    q=x[:n*hop].reshape(n,hop).astype("float64")
    db=20*np.log10(np.sqrt(np.mean(q*q,axis=1))+1e-12)
    active=db>np.percentile(db,48)
    target=float(np.median(db[active]))
    move=np.where(active,np.clip((target-db)*.28,-max_db,max_db),0).astype("float32")
    move=ndimage.gaussian_filter1d(move,sigma=10)
    return np.interp(np.arange(len(x))/hop,np.arange(n)+.5,move,left=0,right=0).astype("float32")

def saturate_level_matched(x:np.ndarray,drive_db:float=1.2)->np.ndarray:
    g=10**(drive_db/20);y=np.tanh(x*g)/g
    ri=np.sqrt(np.mean(x.astype("float64")**2)+1e-20);ro=np.sqrt(np.mean(y.astype("float64")**2)+1e-20)
    return (y*(ri/(ro+1e-20))).astype("float32")

def accept(metrics:dict)->dict:
    fail=[]
    if metrics.get("lead_spread_reduction_db",0)>3:fail.append("lead_overflattened")
    if metrics.get("deess_p95_db",0)>3.05:fail.append("excess_deess")
    if abs(metrics.get("vocal_rms_change_db",0))>1.0:fail.append("vocal_level_cheat")
    if metrics.get("mix_spectral_max_shift_db",0)>.6:fail.append("excess_tonal_shift")
    return {"accept":not fail,"failures":fail}
