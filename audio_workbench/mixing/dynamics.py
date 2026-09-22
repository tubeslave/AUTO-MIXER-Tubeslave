from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import ndimage

@dataclass(frozen=True)
class DynamicsProfile:
    ratio:float; max_gr_db:float; ride_max_db:float; ride_window_s:float; active_percentile:float

PROFILES={
 "vocal":DynamicsProfile(3.2,5.0,2.0,1.6,55),
 "bass":DynamicsProfile(3.0,4.0,1.5,1.8,50),
 "kick":DynamicsProfile(2.0,2.5,.5,1.0,78),
 "snare":DynamicsProfile(2.2,3.0,.5,1.0,78),
 "toms":DynamicsProfile(2.0,2.5,.5,1.1,75),
 "guitar":DynamicsProfile(1.8,2.5,1.2,2.0,50),
 "keys":DynamicsProfile(1.6,2.0,1.0,2.2,50),
 "playback":DynamicsProfile(1.5,1.5,.8,2.5,50),
 "cymbals":DynamicsProfile(1.3,1.0,.35,2.5,70),
}

def analyze_frames(x:np.ndarray,sr:int,role:str,hop_s:float=.02)->dict:
    m=x.mean(1) if x.ndim>1 else x;hop=max(1,int(sr*hop_s));n=len(m)//hop
    q=m[:n*hop].reshape(n,hop).astype("float64")
    db=20*np.log10(np.sqrt(np.mean(q*q,axis=1))+1e-12)
    p=PROFILES[role];active=db>np.percentile(db,p.active_percentile)
    target=float(np.median(db[active]))
    ride=np.where(active,np.clip((target-db)*.42,-p.ride_max_db,p.ride_max_db),0).astype("float32")
    ride=ndimage.gaussian_filter1d(ride,max(1,p.ride_window_s/(2.355*hop_s)))
    threshold=float(np.percentile((db+ride)[active],64 if role in ("vocal","bass") else 70))
    gr=np.where(active,np.clip(np.maximum(db+ride-threshold,0)*(1-1/p.ratio),0,p.max_gr_db),0).astype("float32")
    sigma={"vocal":1.5,"bass":2,"kick":1,"snare":1,"toms":1.2,"guitar":2,"keys":2.5,"playback":3,"cymbals":3}[role]
    gr=ndimage.gaussian_filter1d(gr,sigma)
    # Bounded median compensation preserves the Balance Director's static relationship.
    makeup=float(np.clip(-np.median((ride-gr)[active]),0,2.0))
    net=(ride-gr+makeup).astype("float32")
    return {"db":db,"active":active,"ride_db":ride,"gr_db":gr,"net_db":net,"makeup_db":makeup,
            "before_spread_db":float(np.percentile(db[active],90)-np.percentile(db[active],10)),
            "after_spread_db":float(np.percentile((db+net)[active],90)-np.percentile((db+net)[active],10)),
            "p95_gr_db":float(np.percentile(gr[active],95)),"max_gr_db":float(gr.max()),"hop_s":hop_s}

def apply(x:np.ndarray,sr:int,role:str)->tuple[np.ndarray,dict]:
    a=analyze_frames(x,sr,role);n=len(a["net_db"]);hop_s=a["hop_s"]
    frame_t=(np.arange(n)+.5)*hop_s;sample_t=np.arange(len(x))/sr
    gdb=np.interp(sample_t,frame_t,a["net_db"],left=a["net_db"][0],right=a["net_db"][-1]).astype("float32")
    y=x*np.power(10,gdb[:,None]/20).astype("float32") if x.ndim>1 else x*np.power(10,gdb/20).astype("float32")
    diag={k:v for k,v in a.items() if k not in ("db","active","ride_db","gr_db","net_db")}
    return y.astype("float32"),diag
