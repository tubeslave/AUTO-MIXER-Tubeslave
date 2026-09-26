from __future__ import annotations
import numpy as np
from scipy import ndimage

def analyze_sections(x:np.ndarray,sr:int,hop_s:float=.5)->dict:
    """Infer broad density sections from the audio itself, no arrangement labels required."""
    hop=max(1,int(hop_s*sr));n=len(x)//hop
    q=x[:n*hop].reshape(n,hop,x.shape[1]).astype("float64")
    rms=20*np.log10(np.sqrt(np.mean(q*q,axis=(1,2)))+1e-12)
    # Robust relative energy plus short-term crest/density.
    peak=20*np.log10(np.max(np.abs(q),axis=(1,2))+1e-12)
    crest=peak-rms
    er=(rms-np.percentile(rms,15))/(np.percentile(rms,90)-np.percentile(rms,15)+1e-9)
    cr=(np.percentile(crest,85)-crest)/(np.percentile(crest,85)-np.percentile(crest,15)+1e-9)
    density=np.clip(.72*er+.28*cr,0,1)
    density=ndimage.gaussian_filter1d(density.astype("float32"),sigma=2)
    # Three broad states; boundaries are intentionally hysteretic/smoothed.
    state=np.where(density<.36,0,np.where(density>.68,2,1))
    segments=[];start=0
    for i in range(1,n):
        if state[i]!=state[start]:
            if (i-start)*hop_s>=2.0:
                segments.append({"start_s":start*hop_s,"end_s":i*hop_s,"state":int(state[start]),
                                 "density":float(np.mean(density[start:i]))})
                start=i
            else: state[i]=state[start]
    segments.append({"start_s":start*hop_s,"end_s":n*hop_s,"state":int(state[start]),
                     "density":float(np.mean(density[start:n]))})
    return {"hop_s":hop_s,"segments":segments,"density":density.tolist()}

def parameters_for_density(d:float)->dict:
    """Small deviations around v0.7 Quality Baseline. No section gets a different mastering aesthetic."""
    # Quiet/open sections: least processing. Dense sections: slightly more control, still bounded.
    return {
      "clarity_strength": float(.075 + .055*d),      # 0.075..0.130 around v0.7's 0.10
      "impact_max_gr_db": float(.55 + .35*d),        # 0.55..0.90 around 0.75
      "clip_drive_db": float(.28 + .24*d),           # 0.28..0.52 around 0.40
      "maximizer_drive_db": float(2.05 + .30*d),     # 2.05..2.35 around 2.20
    }

def automation_curve(section_analysis:dict,sr:int,n_samples:int)->dict[str,np.ndarray]:
    hop_s=section_analysis["hop_s"];d=np.asarray(section_analysis["density"],dtype="float32")
    t=np.arange(len(d))*hop_s
    sample_t=np.arange(n_samples)/sr
    ds=np.interp(sample_t,t,d,left=d[0],right=d[-1]).astype("float32")
    # 0.75 s smoothing prevents mastering settings from following individual hits.
    ds=ndimage.gaussian_filter1d(ds,sigma=max(1,int(.75*sr)))
    params={k:np.empty(n_samples,dtype="float32") for k in parameters_for_density(0)}
    for i,v in enumerate(ds):
        for k,q in parameters_for_density(float(v)).items():params[k][i]=q
    return params
