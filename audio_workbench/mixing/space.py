from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal

@dataclass(frozen=True)
class SpaceProfile:
    depth:float
    room_send_db:float
    predelay_ms:float
    early_late:float

PROFILES={
 "lead_vocal":SpaceProfile(.10,-21,34,.42),
 "secondary_vocal":SpaceProfile(.28,-18,24,.38),
 "kick":SpaceProfile(.08,-30,8,.65),
 "snare":SpaceProfile(.25,-18,14,.45),
 "toms":SpaceProfile(.34,-17,10,.40),
 "bass":SpaceProfile(.08,-30,8,.65),
 "guitar":SpaceProfile(.42,-18,16,.36),
 "keys":SpaceProfile(.55,-17,12,.32),
 "playback":SpaceProfile(.62,-18,10,.30),
 "cymbals":SpaceProfile(.52,-20,8,.30),
}

def depth_profile(track_name:str,role:str)->SpaceProfile:
    if track_name=="18_VALERA_VOX.wav":return PROFILES["lead_vocal"]
    if track_name=="09_NIKITA_VOX.wav":return PROFILES["secondary_vocal"]
    return PROFILES.get(role,SpaceProfile(.4,-22,12,.4))

def common_room_ir(sr:int,rt60_s:float=.72,length_s:float=1.05)->np.ndarray:
    """Deterministic stereo room: sparse early reflections plus decorrelated late tail."""
    n=max(8,int(length_s*sr));t=np.arange(n)/sr
    decay=np.exp(-6.91*t/max(rt60_s,.1))
    rng=np.random.default_rng(7)
    ir=np.zeros((n,2),dtype="float32")
    for ms,a,pan in [(18,.72,-.55),(31,.58,.48),(47,.44,-.35),(69,.34,.58),(96,.24,-.20),(131,.18,.32)]:
        i=min(n-1,int(ms*sr/1000));ir[i,0]+=a*(1-pan*.35);ir[i,1]+=a*(1+pan*.35)
    noise=rng.normal(0,1,(n,2)).astype("float32")
    # Smoothed late field, high-passed enough to avoid low-end wash.
    sos=signal.butter(2,[180,min(9000,sr*.45)],btype="bandpass",fs=sr,output="sos")
    noise=signal.sosfilt(sos,noise,axis=0).astype("float32")
    ir+=noise*decay[:,None].astype("float32")*.035
    ir/=np.max(np.abs(ir))+1e-12
    return ir

def send_gain_db(profile:SpaceProfile,density:np.ndarray|float)->np.ndarray:
    d=np.asarray(density)
    # Dense sections gain space mostly through send, not a loudness jump.
    return profile.room_send_db + np.clip((d-.5)*1.8,-.7,.9)

def accept_space(metrics:dict)->dict:
    fail=[]
    if metrics.get("mix_loudness_change_lu",0)>.8:fail.append("space_raises_mix_too_much")
    if metrics.get("low_band_shift_db",0)>.45:fail.append("room_muddies_low_end")
    if metrics.get("side_energy_gain_db",0)>3.0:fail.append("excess_width")
    if metrics.get("wet_peak_dbfs",-99)>-3:fail.append("wet_bus_hot")
    return {"accept":not fail,"failures":fail}
