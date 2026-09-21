from __future__ import annotations
import numpy as np
from scipy import signal

def _bell(sr,f,q,g):
    A=10**(g/40);w=2*np.pi*f/sr;al=np.sin(w)/(2*q);c=np.cos(w)
    b=np.array([1+al*A,-2*c,1-al*A]);a=np.array([1+al/A,-2*c,1-al/A])
    return b/a[0],a/a[0]

def process(x,sr,analysis,max_db=1.25):
    b=analysis["bands_db"]; mid=b["mid"]
    # self-relative smooth target: suppress broad outliers, never impose a genre/reference curve
    rel=np.array([b["sub"]-mid,b["low"]-mid,b["lowmid"]-mid,0,b["presence"]-mid,b["air"]-mid])
    smooth=np.convolve(np.pad(rel,(1,1),mode="edge"),np.ones(3)/3,mode="valid")
    err=np.clip(smooth-rel,-max_db,max_db)
    freqs=[55,125,330,1000,3300,9000]
    y=x.copy()
    for f,g in zip(freqs,err):
        if abs(g)>.15:
            bb,aa=_bell(sr,f,.65,float(g));y=signal.lfilter(bb,aa,y,axis=0).astype("float32")
    return y,{"moves_db":dict(zip(map(str,freqs),map(float,err))),"max_db":max_db}
