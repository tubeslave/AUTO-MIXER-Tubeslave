from __future__ import annotations
import numpy as np
from scipy import signal,ndimage
from .analyzer import _trapezoid

def _mono(x): return np.asarray(x,dtype=np.float32).mean(axis=1)

def _bp(x,sr,lo,hi):
    sos=signal.butter(3,[lo,hi],btype="bandpass",fs=sr,output="sos")
    return signal.sosfiltfilt(sos,_mono(x)).astype("float32")

def transient_critic(base,candidate,sr):
    # Onset novelty in drum-relevant bands. Compare strongest matched temporal events.
    results={}
    for name,lo,hi in [("kick",35,150),("snare_presence",900,6000)]:
        a=np.abs(signal.hilbert(_bp(base,sr,lo,hi)))
        b=np.abs(signal.hilbert(_bp(candidate,sr,lo,hi)))
        win=max(3,int(.035*sr))
        aa=ndimage.maximum_filter1d(a,size=win); bb=ndimage.maximum_filter1d(b,size=win)
        slowa=ndimage.uniform_filter1d(a,size=max(3,int(.18*sr)))
        slowb=ndimage.uniform_filter1d(b,size=max(3,int(.18*sr)))
        na=np.maximum(aa-slowa,0); nb=np.maximum(bb-slowb,0)
        threshold=np.percentile(na,97)
        peaks,_=signal.find_peaks(na,height=threshold,distance=max(1,int(.08*sr)))
        if len(peaks)==0:
            results[name]={"events":0,"median_attack_change_db":0.0}
            continue
        ratio=20*np.log10((nb[peaks]+1e-9)/(na[peaks]+1e-9))
        results[name]={"events":int(len(peaks)),"median_attack_change_db":float(np.median(ratio)),
                       "p10_attack_change_db":float(np.percentile(ratio,10))}
    return results

def low_end_punch_critic(base,candidate,sr):
    # Attack/body ratio in 35–180 Hz on base-defined low-frequency events.
    a=np.abs(signal.hilbert(_bp(base,sr,35,180)));b=np.abs(signal.hilbert(_bp(candidate,sr,35,180)))
    fasta=ndimage.maximum_filter1d(a,size=max(3,int(.025*sr)))
    fastb=ndimage.maximum_filter1d(b,size=max(3,int(.025*sr)))
    bodya=np.sqrt(ndimage.uniform_filter1d(a*a,size=max(3,int(.20*sr)))+1e-12)
    bodyb=np.sqrt(ndimage.uniform_filter1d(b*b,size=max(3,int(.20*sr)))+1e-12)
    pa=20*np.log10((fasta+1e-9)/(bodya+1e-9));pb=20*np.log10((fastb+1e-9)/(bodyb+1e-9))
    peaks,_=signal.find_peaks(fasta,height=np.percentile(fasta,97),distance=max(1,int(.09*sr)))
    d=pb[peaks]-pa[peaks] if len(peaks) else np.array([0.])
    return {"events":int(len(peaks)),"median_punch_change_db":float(np.median(d)),
            "p10_punch_change_db":float(np.percentile(d,10))}

def spectral_shift_critic(base,candidate,sr):
    # Level-insensitive broad-band spectral shift.
    bands=[("sub",30,80),("low",80,200),("lowmid",200,500),("mid",500,2000),
           ("presence",2000,5000),("air",5000,min(16000,sr*.45))]
    def energies(x):
        f,p=signal.welch(_mono(x),fs=sr,nperseg=min(8192,len(x)))
        vals={}
        for n,lo,hi in bands:
            m=(f>=lo)&(f<hi);vals[n]=10*np.log10(_trapezoid(p[m],f[m])+1e-20)
        anchor=np.mean(list(vals.values()))
        return {k:v-anchor for k,v in vals.items()}
    a,b=energies(base),energies(candidate)
    shift={k:float(b[k]-a[k]) for k in a}
    return {"band_shift_db":shift,"max_abs_shift_db":float(max(abs(v) for v in shift.values())),
            "rms_shift_db":float(np.sqrt(np.mean(np.square(list(shift.values())))))}

def evaluate(base,candidate,sr,transient_limit_db=-.75,punch_limit_db=-.6,spectral_limit_db=.8):
    t=transient_critic(base,candidate,sr);p=low_end_punch_critic(base,candidate,sr);s=spectral_shift_critic(base,candidate,sr)
    failures=[]
    for k,v in t.items():
        if v["median_attack_change_db"]<transient_limit_db: failures.append(f"{k}_attack_loss")
    if p["median_punch_change_db"]<punch_limit_db: failures.append("low_end_punch_loss")
    if s["max_abs_shift_db"]>spectral_limit_db: failures.append("spectral_shift")
    return {"transients":t,"low_end_punch":p,"spectral_shift":s,"accept":not failures,"failures":failures}
