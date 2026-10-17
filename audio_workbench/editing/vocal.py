from __future__ import annotations
import numpy as np
from scipy import signal,ndimage

def phrase_activity(x:np.ndarray,sr:int,hop_ms:float=20.0)->dict:
    """Conservative vocal phrase detector. Returns a soft activity envelope, not a gate."""
    if x.ndim>1:x=x.mean(axis=1)
    hop=max(1,int(sr*hop_ms/1000));n=len(x)//hop
    q=x[:n*hop].reshape(n,hop).astype("float64")
    rms=20*np.log10(np.sqrt(np.mean(q*q,axis=1))+1e-12)
    # Vocal-band energy helps distinguish phrase energy from LF stage bleed.
    sos=signal.butter(2,[120,min(8000,sr*.45)],btype="bandpass",fs=sr,output="sos")
    v=signal.sosfiltfilt(sos,x).astype("float32")
    qv=v[:n*hop].reshape(n,hop).astype("float64")
    vrms=20*np.log10(np.sqrt(np.mean(qv*qv,axis=1))+1e-12)
    floor=float(np.percentile(vrms,20));top=float(np.percentile(vrms,90))
    norm=np.clip((vrms-(floor+6))/(max(top-floor-6,6)),0,1)
    active=norm>.18
    # Phrase hysteresis: close short gaps, remove tiny islands.
    active=ndimage.binary_closing(active,iterations=max(1,int(180/hop_ms)))
    active=ndimage.binary_opening(active,iterations=max(1,int(60/hop_ms)))
    soft=ndimage.gaussian_filter1d(active.astype("float32"),sigma=max(1,int(60/hop_ms)))
    return {"hop":hop,"rms_db":rms,"vocal_rms_db":vrms,"activity":soft,
            "noise_floor_db":floor,"active_fraction":float(np.mean(active))}

def cleanup_between_phrases(x:np.ndarray,sr:int,max_atten_db:float=8.0)->tuple[np.ndarray,dict]:
    a=phrase_activity(x,sr);hop=a["hop"];env=np.interp(np.arange(len(x))/hop,np.arange(len(a["activity"]))+.5,
        a["activity"],left=0,right=0).astype("float32")
    # Keep breaths/tails by using attenuation, never hard muting. Slow edges avoid chatter.
    env=ndimage.gaussian_filter1d(env,sigma=max(1,int(.025*sr)))
    gain_db=-max_atten_db*(1-env)
    gain=np.power(10,gain_db/20).astype("float32")
    y=x*gain[:,None] if x.ndim>1 else x*gain
    return y.astype("float32"),{"max_atten_db":max_atten_db,"active_fraction":a["active_fraction"],
      "noise_floor_db":a["noise_floor_db"]}

def phrase_level_automation(x:np.ndarray,sr:int,max_move_db:float=2.0)->tuple[np.ndarray,dict]:
    """Correct only large phrase-level outliers; no fast riding and no compression substitute."""
    a=phrase_activity(x,sr);hop=a["hop"];act=a["activity"]>.5;r=a["vocal_rms_db"]
    labels,n=ndimage.label(act);moves=[];curve=np.zeros(len(r),dtype="float32")
    vals=[]
    for lab in range(1,n+1):
        m=labels==lab
        if np.sum(m)*hop/sr<.18:continue
        vals.append((lab,float(np.median(r[m]))))
    if not vals:return x,{"moves":[]}
    target=float(np.median([v for _,v in vals]))
    for lab,v in vals:
        delta=float(np.clip(target-v,-max_move_db,max_move_db))
        if abs(delta)<.75:continue
        m=labels==lab;curve[m]=delta;moves.append({"phrase":int(lab),"level_db":v,"delta_db":delta})
    curve=ndimage.gaussian_filter1d(curve,sigma=max(1,int(80/(1000*hop/sr))))
    sample=np.interp(np.arange(len(x))/hop,np.arange(len(curve))+.5,curve,left=0,right=0)
    g=np.power(10,sample/20).astype("float32")
    y=x*g[:,None] if x.ndim>1 else x*g
    return y.astype("float32"),{"target_phrase_db":target,"moves":moves}
