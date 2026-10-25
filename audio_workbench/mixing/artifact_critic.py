from __future__ import annotations
import numpy as np
from scipy import signal,ndimage

def _mono(x):
    return x.mean(1) if x.ndim>1 else x

def _band(x,sr,lo,hi):
    """Bandpass helper that is safe when the requested band is above Nyquist."""
    upper=min(float(hi),float(sr)*.47)
    lower=max(1.0,float(lo))
    if upper<=lower:
        return np.zeros_like(x)
    return signal.sosfilt(signal.butter(2,[lower,upper],btype="bandpass",fs=sr,output="sos"),x)

def detect(audio:np.ndarray,sr:int)->dict:
    """Reference-free artifact evidence. Returns evidence, not a quality score."""
    x=_mono(audio).astype("float32")
    # Clicks: isolated derivative spikes relative to local derivative energy.
    # 6x keeps the detector sensitive to sparse discontinuities in short fixtures while
    # remaining well above ordinary periodic sample-to-sample slope changes.
    d=np.abs(np.diff(x,prepend=x[0]))
    local=np.sqrt(ndimage.uniform_filter1d(d*d,max(3,int(.012*sr)))+1e-12)
    click_ratio=d/(local+1e-9)
    click_rate=float(np.mean(click_ratio>6.0))

    # Musical-noise proxy: narrow high-frequency peaks with unstable frame-to-frame occupancy.
    f,t,z=signal.stft(x,fs=sr,nperseg=1024,noverlap=768,boundary=None)
    mag=np.abs(z)+1e-10
    hf=(f>=3500)&(f<=min(12000,sr*.47))
    if np.any(hf) and mag.shape[1]>2:
        m=mag[hf]
        tonality=np.max(m,axis=0)/(np.mean(m,axis=0)+1e-10)
        flux=np.mean(np.abs(np.diff(np.log(m),axis=1)),axis=0)
        musical_noise=float(np.mean((tonality[1:]>8)&(flux>1.1)))
    else:musical_noise=0.

    # Pumping proxy: unusually periodic broadband envelope modulation in 1.5-8 Hz.
    env=np.sqrt(ndimage.uniform_filter1d(x*x,max(3,int(.025*sr)))+1e-12)
    dec=max(1,int(sr/200));e=env[::dec];fs_e=sr/dec
    e=e-np.mean(e)
    ff,pp=signal.periodogram(e,fs_e)
    band=(ff>=1.5)&(ff<=8)
    total=(ff>=.3)&(ff<=15)
    pumping=float(np.sum(pp[band])/(np.sum(pp[total])+1e-20)) if np.any(total) else 0.

    # HF tearing / separation fizz proxy. At low sample rates this band may not exist;
    # in that case the evidence is correctly zero rather than an invalid filter design.
    hi=_band(x,sr,6000,14000);body=_band(x,sr,300,3500)
    fizz=float(np.clip((20*np.log10((np.sqrt(np.mean(hi.astype("float64")**2))+1e-12)/
                                    (np.sqrt(np.mean(body.astype("float64")**2))+1e-12))+28)/20,0,1))
    return {"click_rate":click_rate,"musical_noise":musical_noise,"pumping":pumping,"hf_fizz":fizz}

def compare(before:dict,after:dict)->dict:
    keys=("click_rate","musical_noise","pumping","hf_fizz")
    delta={k:float(after[k]-before[k]) for k in keys}
    # Click rate is measured per sample, so an 0.08 absolute threshold would require an
    # implausible 8% of all samples to be clicks. Other bounded proxies keep the original
    # 0.08 regression tolerance.
    limits={"click_rate":.001,"musical_noise":.08,"pumping":.08,"hf_fizz":.08}
    regressions=[k for k in keys if delta[k]>limits[k]]
    return {"delta":delta,"accept":not regressions,"regressions":regressions}
