from __future__ import annotations
import numpy as np
from scipy import signal,ndimage

def _limit_band(x,sr,ceiling,release_ms):
    env=ndimage.maximum_filter1d(np.max(np.abs(x),axis=1),size=max(3,int(.004*sr)))+1e-12
    req=np.minimum(1.0,ceiling/env)
    # release smoothing; attack remains effectively immediate through lookahead-like max filter
    alpha=np.exp(-1/(sr*release_ms/1000))
    g=np.ones(len(req),dtype="float32")
    for i in range(1,len(g)):
        g[i]=req[i] if req[i]<g[i-1] else alpha*g[i-1]+(1-alpha)*req[i]
    return x*g[:,None],g

def process(x,sr,ceiling_db=-1.0,drive_db=2.0):
    # IRC5-inspired architecture only: four independent band envelopes. Not an iZotope algorithm.
    driven=x*np.float32(10**(drive_db/20));edges=[25,120,900,5000,0.49*sr]
    bands=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        sos=signal.butter(3,[lo,hi],btype="bandpass",fs=sr,output="sos")
        bands.append(signal.sosfilt(sos,driven,axis=0).astype("float32"))
    ceiling=10**(ceiling_db/20)/4**.25
    rel=[180,120,75,45];out=np.zeros_like(x);stats=[]
    for b,r in zip(bands,rel):
        q,g=_limit_band(b,sr,ceiling,r);out+=q
        stats.append({"release_ms":r,"max_gr_db":float(-20*np.log10(max(g.min(),1e-8)))})
    # final safety envelope, still no hard clip
    q,g=_limit_band(out,sr,10**(ceiling_db/20),55)
    return q.astype("float32"),{"mode":"four_band_psychoacoustic_inspired","drive_db":drive_db,"ceiling_db":ceiling_db,
      "bands":stats,"final_max_gr_db":float(-20*np.log10(max(g.min(),1e-8)))}
