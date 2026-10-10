from __future__ import annotations
import numpy as np
from scipy import signal

def describe_sections(mix:np.ndarray,sr:int,section_seconds:float=4.0)->list[dict]:
    """Describe arrangement trajectory without assuming verse/chorus labels."""
    if mix.ndim==1:mix=np.column_stack([mix,mix])
    hop=max(1,int(section_seconds*sr));rows=[]
    for i in range(0,len(mix),hop):
        x=mix[i:i+hop]
        if len(x)<hop//2:continue
        mono=x.mean(1);mid=(x[:,0]+x[:,1])*.5;side=(x[:,0]-x[:,1])*.5
        rms=np.sqrt(np.mean(mono.astype("float64")**2)+1e-20)
        peak=np.max(np.abs(mono))+1e-12
        f,p=signal.welch(mono,sr,nperseg=min(4096,len(mono)))
        centroid=float(np.sum(f*p)/(np.sum(p)+1e-20))
        rows.append({"start_s":i/sr,"rms_db":float(20*np.log10(rms+1e-20)),
          "crest_db":float(20*np.log10(peak/(rms+1e-20))),
          "width_db":float(20*np.log10((np.sqrt(np.mean(side.astype("float64")**2))+1e-12)/
                                      (np.sqrt(np.mean(mid.astype("float64")**2))+1e-12))),
          "spectral_centroid_hz":centroid})
    if not rows:return rows
    loud=np.array([r["rms_db"] for r in rows]);bright=np.array([r["spectral_centroid_hz"] for r in rows])
    for r,l,b in zip(rows,loud,bright):
        r["energy_percentile"]=float(np.mean(loud<=l))
        r["brightness_percentile"]=float(np.mean(bright<=b))
        r["role"]="peak" if r["energy_percentile"]>=.8 else ("sparse" if r["energy_percentile"]<=.25 else "body")
    return rows

def climax_evidence(rows:list[dict])->dict:
    if not rows:return {"peak_lift_db":0.,"peak_width_lift_db":0.,"peak_count":0}
    peak=[r for r in rows if r["role"]=="peak"];body=[r for r in rows if r["role"]=="body"]
    if not peak or not body:return {"peak_lift_db":0.,"peak_width_lift_db":0.,"peak_count":len(peak)}
    return {"peak_lift_db":float(np.median([r["rms_db"] for r in peak])-np.median([r["rms_db"] for r in body])),
      "peak_width_lift_db":float(np.median([r["width_db"] for r in peak])-np.median([r["width_db"] for r in body])),
      "peak_count":len(peak)}
