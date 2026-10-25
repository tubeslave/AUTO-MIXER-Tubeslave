from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CleanupPolicy:
    floor_max_atten_db:float
    bleed_max_atten_db:float
    artifact_max_atten_db:float
    attack_ms:float
    release_ms:float

POLICIES={
 "vocal":CleanupPolicy(8,10,8,18,140),
 "drum_close":CleanupPolicy(7,9,6,8,95),
 "overhead":CleanupPolicy(3,4,3,30,180),
 "bass":CleanupPolicy(6,5,10,25,160),
 "guitar":CleanupPolicy(6,5,10,20,140),
}

def soft_mask(target:np.ndarray,interference:np.ndarray,margin_db:float=3.,max_atten_db:float=10.)->np.ndarray:
    """Continuous attenuation mask, never a binary gate."""
    t=np.maximum(target,1e-12);i=np.maximum(interference,1e-12)
    snr=20*np.log10(t/i)
    need=np.clip((margin_db-snr)*.45,0,max_atten_db)
    return -need.astype("float32")

def spectral_residual_mask(target_spec:np.ndarray,bleed_spec:np.ndarray,max_atten_db:float)->np.ndarray:
    """Wiener-like target/bleed ratio mask expressed as bounded attenuation."""
    p=np.abs(target_spec)**2;b=np.abs(bleed_spec)**2
    ratio=p/(p+b+1e-12)
    db=10*np.log10(np.clip(ratio,10**(-max_atten_db/10),1))
    return db.astype("float32")

def artifact_envelope(score:np.ndarray,max_atten_db:float=10.)->np.ndarray:
    """For fret/scrape/inter-note artifacts. Score is continuous 0..1."""
    return (-max_atten_db*np.clip(score,0,1)).astype("float32")
