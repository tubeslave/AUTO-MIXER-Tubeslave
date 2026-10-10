from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal,ndimage

@dataclass(frozen=True)
class MaskingBand:
    name:str
    lo:float
    hi:float
    max_cut_db:float

BANDS={
 "low_punch":MaskingBand("low_punch",45,110,1.5),
 "low_mid":MaskingBand("low_mid",180,420,1.2),
 "presence":MaskingBand("presence",1200,3500,1.8),
 "upper_presence":MaskingBand("upper_presence",3000,6500,1.2),
}

def band_envelope(x:np.ndarray,sr:int,band:MaskingBand,window_ms:float=45)->np.ndarray:
    if x.ndim>1:x=x.mean(1)
    sos=signal.butter(2,[band.lo,band.hi],btype="bandpass",fs=sr,output="sos")
    y=signal.sosfiltfilt(sos,x).astype("float32")
    return np.sqrt(ndimage.uniform_filter1d(y*y,size=max(3,int(window_ms*sr/1000)))+1e-12)

def masking_evidence(target:np.ndarray,masker:np.ndarray,sr:int,band:MaskingBand)->dict:
    t=band_envelope(target,sr,band);m=band_envelope(masker,sr,band)
    ta=t>np.percentile(t,65);ma=m>np.percentile(m,55);co=ta&ma
    if np.sum(co)<sr*.25:return {"score":0.,"coactivity":0.,"median_ratio_db":0.}
    ratio=20*np.log10((m[co]+1e-12)/(t[co]+1e-12))
    coactivity=float(np.mean(co))
    med=float(np.median(ratio))
    # More evidence when masker is competitive with target during target activity.
    score=float(np.clip(coactivity*2.2,0,1)*np.clip((med+12)/18,0,1))
    return {"score":score,"coactivity":coactivity,"median_ratio_db":med}

def sidechain_cut_curve(target:np.ndarray,sr:int,band:MaskingBand,depth_db:float)->np.ndarray:
    e=band_envelope(target,sr,band)
    lo=np.percentile(e,55);hi=np.percentile(e,90)
    activity=np.clip((e-lo)/(hi-lo+1e-12),0,1)
    activity=ndimage.gaussian_filter1d(activity.astype("float32"),sigma=max(1,int(.035*sr)))
    return (-abs(depth_db)*activity).astype("float32")

def apply_dynamic_band_cut(masker:np.ndarray,curve_db:np.ndarray,sr:int,band:MaskingBand)->np.ndarray:
    sos=signal.butter(2,[band.lo,band.hi],btype="bandpass",fs=sr,output="sos")
    band_audio=signal.sosfiltfilt(sos,masker,axis=0).astype("float32")
    gain=np.power(10,curve_db/20).astype("float32")
    return (masker+band_audio*(gain[:,None]-1)).astype("float32")
