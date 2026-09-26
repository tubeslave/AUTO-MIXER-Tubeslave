from __future__ import annotations
import numpy as np
from scipy import ndimage

def process(x,sr,max_peak_control_db=1.25):
    # Microdynamic peak conditioning: fast envelope vs slow envelope, attenuate only excess attacks.
    mono=np.max(np.abs(x),axis=1)+1e-12
    fast=ndimage.maximum_filter1d(mono,size=max(3,int(.004*sr)))
    slow=np.sqrt(ndimage.uniform_filter1d(mono*mono,size=max(3,int(.08*sr)))+1e-12)
    ratio=20*np.log10((fast+1e-12)/(slow+1e-12))
    threshold=float(np.percentile(ratio,88))
    gr=np.clip((ratio-threshold)*.35,0,max_peak_control_db)
    gr=ndimage.gaussian_filter1d(gr.astype("float32"),sigma=max(1,int(.0015*sr)))
    y=x*np.power(10,-gr[:,None]/20).astype("float32")
    return y,{"threshold_crest_db":threshold,"max_gr_db":float(gr.max()),"p95_gr_db":float(np.percentile(gr,95))}
