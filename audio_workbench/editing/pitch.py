from __future__ import annotations
import numpy as np
from scipy import signal,ndimage

def hz_to_midi(f): return 69+12*np.log2(np.maximum(f,1e-9)/440.0)

def estimate_f0_autocorr(x:np.ndarray,sr:int,frame_ms=45,hop_ms=10,fmin=75.,fmax=700.) -> dict:
    """Dependency-light monophonic F0 tracker for vocal editing diagnosis."""
    if x.ndim>1:x=x.mean(axis=1)
    frame=max(64,int(frame_ms*sr/1000));hop=max(1,int(hop_ms*sr/1000))
    win=np.hanning(frame);f0=[];conf=[];times=[]
    lo=max(1,int(sr/fmax));hi=min(frame-2,int(sr/fmin))
    for p in range(0,len(x)-frame,hop):
        q=x[p:p+frame]*win;q=q-np.mean(q)
        rms=np.sqrt(np.mean(q*q)+1e-12)
        if rms<1e-4:f0.append(np.nan);conf.append(0.);times.append((p+frame/2)/sr);continue
        ac=signal.fftconvolve(q,q[::-1],mode="full")[frame-1:]
        ac/=ac[0]+1e-12
        seg=ac[lo:hi+1];peaks,_=signal.find_peaks(seg)
        if len(peaks)==0:f0.append(np.nan);conf.append(0.);times.append((p+frame/2)/sr);continue
        i=peaks[np.argmax(seg[peaks])]+lo;c=float(ac[i])
        f0.append(float(sr/i) if c>=.32 else np.nan);conf.append(c);times.append((p+frame/2)/sr)
    return {"time_s":np.asarray(times),"f0_hz":np.asarray(f0),"confidence":np.asarray(conf)}

def note_segments(track:dict,min_ms=120,confidence=.55)->list[dict]:
    t=track["time_s"];f=track["f0_hz"];c=track["confidence"];m=np.isfinite(f)&(c>=confidence)
    midi=np.full_like(f,np.nan,dtype=float);midi[m]=hz_to_midi(f[m])
    # Robust smoothing keeps vibrato but rejects octave glitches.
    valid=np.where(m,np.round(midi))[0]
    if not len(valid):return []
    labels=np.full(len(f),-1,int);lab=0
    for i in range(len(f)):
        if not m[i]:continue
        if i and labels[i-1]>=0 and abs(np.nanmedian(midi[max(0,i-3):i+1])-np.nanmedian(midi[max(0,i-4):i]))<.8:
            labels[i]=labels[i-1]
        else: labels[i]=lab;lab+=1
    out=[]
    for k in np.unique(labels[labels>=0]):
        idx=np.where(labels==k)[0]
        if len(idx)<2 or (t[idx[-1]]-t[idx[0]])*1000<min_ms:continue
        cents=(midi[idx]-np.round(np.median(midi[idx])))*100
        out.append({"start_s":float(t[idx[0]]),"end_s":float(t[idx[-1]]),
                    "target_midi":int(np.round(np.median(midi[idx]))),
                    "median_error_cents":float(np.median(cents)),
                    "p90_abs_error_cents":float(np.percentile(np.abs(cents),90)),
                    "confidence":float(np.median(c[idx]))})
    return out

def correction_plan(segments:list[dict],min_error_cents=18,max_correction_cents=35)->list[dict]:
    edits=[]
    for s in segments:
        e=s["median_error_cents"]
        if abs(e)<min_error_cents or s["confidence"]<.58:continue
        # Partial correction only. Never flatten frame-level vibrato.
        delta=float(np.clip(-e*.65,-max_correction_cents,max_correction_cents))
        edits.append({**s,"correction_cents":delta})
    return edits
