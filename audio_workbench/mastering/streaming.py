from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal, ndimage
from .analyzer import analyze
from . import stabilizer, clarity, impact, clipper

def _vector_limiter(x: np.ndarray, sr: int, ceiling: float,
                    lookahead_ms: float=4.0, release_ms: float=80.0):
    env=np.max(np.abs(x),axis=1)+1e-12
    win=max(3,int(sr*lookahead_ms/1000))
    if win%2==0: win+=1
    env=ndimage.maximum_filter1d(env,size=win,mode="nearest")
    req=np.minimum(1.0,ceiling/env).astype("float32")
    # Fast downward action, smooth recovery. Vectorized approximation avoids Python/sample loops.
    req=ndimage.minimum_filter1d(req,size=win,mode="nearest")
    sigma=max(1,int(sr*release_ms/1000/5))
    smooth=ndimage.gaussian_filter1d(req,sigma=sigma,mode="nearest")
    g=np.minimum(req,smooth)
    return x*g[:,None],g

def four_band_maximizer(x: np.ndarray, sr: int, ceiling_db=-1.0, drive_db=2.0):
    driven=x*np.float32(10**(drive_db/20))
    cut=(120,900,5000)
    lows=[]
    for f in cut:
        sos=signal.butter(3,f,btype="lowpass",fs=sr,output="sos")
        lows.append(signal.sosfiltfilt(sos,driven,axis=0).astype("float32"))
    bands=[lows[0],lows[1]-lows[0],lows[2]-lows[1],driven-lows[2]]
    # Sum of bands reconstructs driven exactly before gain changes.
    rel=(180.,120.,75.,45.)
    # Allocate modest per-band headroom; final broadband stage enforces actual ceiling.
    band_ceiling=10**((ceiling_db+5.0)/20)
    out=np.zeros_like(x);stats=[]
    for b,r in zip(bands,rel):
        q,g=_vector_limiter(b,sr,band_ceiling,4.0,r);out+=q
        stats.append({"release_ms":r,"max_gr_db":float(-20*np.log10(max(float(g.min()),1e-8)))})
    q,g=_vector_limiter(out,sr,10**(ceiling_db/20),3.0,55.0)
    return q.astype("float32"),{"drive_db":drive_db,"ceiling_db":ceiling_db,
      "bands":stats,"final_max_gr_db":float(-20*np.log10(max(float(g.min()),1e-8)))}

def render_file(source: str, output: str, report_path: str|None=None,
                chunk_s: float=12.0, overlap_s: float=1.0,
                ceiling_db: float=-1.0) -> dict:
    info=sf.info(source);sr=info.samplerate;n=info.frames
    # One global analysis fixes tonal decisions for all chunks.
    with sf.SoundFile(source) as f:
        # analyzer is intentionally global for v0.1; stereo float32 is ~110 MB for 5 min/44.1k.
        full=f.read(dtype="float32",always_2d=True)
    before=analyze(full,sr)
    del full
    step=int(chunk_s*sr);ov=int(overlap_s*sr)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    writer=sf.SoundFile(output,"w",samplerate=sr,channels=2,subtype="FLOAT")
    events=[];pos=0;previous_tail=None
    with sf.SoundFile(source) as f:
        while pos<n:
            a=max(0,pos-ov);b=min(n,pos+step+ov);f.seek(a)
            x=f.read(b-a,dtype="float32",always_2d=True)
            y,s=stabilizer.process(x,sr,before); 
            y,c=clarity.process(y,sr,.18)
            y,i=impact.process(y,sr,1.25)
            y,cl=clipper.process(y,sr,.8,4)
            y,m=four_band_maximizer(y,sr,ceiling_db,2.0)
            lo=pos-a;hi=min(lo+step,n-pos)
            core=y[lo:lo+hi]
            # Padding is context only; cores are contiguous, so no double-render overlap.
            writer.write(core)
            if not events: events=[{"module":"stabilizer",**s},{"module":"clarity",**c},
              {"module":"impact",**i},{"module":"clipper",**cl},{"module":"maximizer",**m}]
            pos+=hi
    writer.close()
    z,_=sf.read(output,dtype="float32",always_2d=True);after=analyze(z,sr)
    report={"streaming":True,"chunk_s":chunk_s,"overlap_context_s":overlap_s,
      "before":before,"after":after,"events_first_chunk":events,
      "regression":{"crest_change_db":after["crest_db"]-before["crest_db"],
        "side_mid_change_db":after["side_mid_db"]-before["side_mid_db"],
        "correlation_change":after["correlation"]-before["correlation"]}}
    if report_path:Path(report_path).write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report
