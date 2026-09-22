from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal

@dataclass(frozen=True)
class DrumPolicy:
    kick_reinforce_db:float=-17.0
    snare_reinforce_db:float=-19.0
    max_parallel_fraction:float=.22
    sample_pre_ms:float=8.0
    sample_post_ms:float=115.0

def transient_novelty(x:np.ndarray,sr:int,lo:float,hi:float)->np.ndarray:
    if x.ndim>1:x=x.mean(1)
    sos=signal.butter(2,[lo,hi],btype="bandpass",fs=sr,output="sos")
    y=signal.sosfiltfilt(sos,x)
    env=np.abs(signal.hilbert(y))
    slow=signal.sosfilt(signal.butter(1,18,btype="lowpass",fs=sr,output="sos"),env)
    return np.maximum(env-slow,0)

def main_hits(x:np.ndarray,sr:int,lo:float,hi:float,percentile:float,min_gap_ms:float)->np.ndarray:
    n=transient_novelty(x,sr,lo,hi)
    p,_=signal.find_peaks(n,height=np.percentile(n,percentile),distance=int(min_gap_ms*sr/1000))
    return p

def self_sample(x:np.ndarray,hits:np.ndarray,sr:int,pre_ms:float=8,post_ms:float=115,count:int=24)->np.ndarray:
    """Median stack of the source's own strongest hits. No external sample library."""
    if x.ndim==1:x=x[:,None]
    pre=int(pre_ms*sr/1000);post=int(post_ms*sr/1000);length=pre+post
    rows=[]
    peak=np.max(np.abs(x),axis=1)
    order=hits[np.argsort(peak[hits])[::-1]]
    for p in order:
        if p-pre>=0 and p+post<=len(x):rows.append(x[p-pre:p+post])
        if len(rows)>=count:break
    if not rows:return np.zeros((length,x.shape[1]),dtype="float32")
    s=np.median(np.stack(rows),axis=0).astype("float32")
    # Keep the sample's own envelope but guarantee click-free end.
    fade=max(8,int(.012*sr));s[-fade:]*=np.linspace(1,0,fade,dtype="float32")[:,None]
    return s

def reinforce(length:int,hits:np.ndarray,sample:np.ndarray,sr:int,gain_db:float)->np.ndarray:
    y=np.zeros((length,sample.shape[1]),dtype="float32");pre=int(.008*sr);g=10**(gain_db/20)
    for p in hits:
        a=p-pre;b=a+len(sample);sa=0;sb=len(sample)
        if a<0:sa=-a;a=0
        if b>length:sb-=b-length;b=length
        if b>a:y[a:b]+=sample[sa:sb]*g
    return y

def accept(metrics:dict)->dict:
    fail=[]
    if metrics.get("kick_attack_gain_db",0)>2.5:fail.append("kick_overenhanced")
    if metrics.get("snare_attack_gain_db",0)>2.5:fail.append("snare_overenhanced")
    if metrics.get("drum_rms_gain_db",0)>1.2:fail.append("drum_bus_level_cheat")
    if metrics.get("mix_spectral_max_shift_db",0)>.6:fail.append("excess_tonal_shift")
    return {"accept":not fail,"failures":fail}
