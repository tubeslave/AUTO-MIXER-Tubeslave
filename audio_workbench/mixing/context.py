from __future__ import annotations
import numpy as np
from scipy import ndimage

def section_curve(track_bus:np.ndarray,sr:int,hop_s:float=.5)->dict:
    """Arrangement-agnostic section map from energy, density and change points."""
    if track_bus.ndim>1: mono=track_bus.mean(1)
    else: mono=track_bus
    hop=max(1,int(hop_s*sr));n=len(mono)//hop
    q=mono[:n*hop].reshape(n,hop).astype("float64")
    rms=20*np.log10(np.sqrt(np.mean(q*q,axis=1))+1e-12)
    crest=20*np.log10(np.max(np.abs(q),axis=1)+1e-12)-rms
    er=(rms-np.percentile(rms,10))/(np.percentile(rms,90)-np.percentile(rms,10)+1e-9)
    cr=(np.percentile(crest,85)-crest)/(np.percentile(crest,85)-np.percentile(crest,15)+1e-9)
    density=ndimage.gaussian_filter1d(np.clip(.72*er+.28*cr,0,1).astype("float32"),2)
    novelty=np.abs(np.r_[0,np.diff(ndimage.gaussian_filter1d(density,3))])
    boundaries=np.where(novelty>np.percentile(novelty,92))[0]
    # Merge boundaries closer than 4 s.
    merged=[]
    for b in boundaries:
        if not merged or (b-merged[-1])*hop_s>=4: merged.append(int(b))
    return {"hop_s":hop_s,"density":density.tolist(),"boundaries_s":[float(b*hop_s) for b in merged],
            "rms_db":rms.tolist()}

def infer_roles(track_names:list[str])->dict[str,str]:
    roles={}
    for n in track_names:
        s=n.lower()
        if "kick" in s: r="kick"
        elif "sn_" in s or "snare" in s: r="snare"
        elif "tom" in s or "floor" in s: r="toms"
        elif "oh" in s or "hi_hat" in s: r="cymbals"
        elif "bass" in s: r="bass"
        elif "gtr" in s or "guitar" in s: r="guitar"
        elif "keys" in s: r="keys"
        elif "pb_" in s or "playback" in s: r="playback"
        elif "vox" in s or "vocal" in s: r="vocal"
        else: r="other"
        roles[n]=r
    return roles
