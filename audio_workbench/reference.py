from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.integrate import trapezoid

BANDS = {
    "sub": (20, 60), "bass": (60, 200), "low_mid": (200, 500),
    "mid": (500, 2000), "presence": (2000, 6000), "air": (6000, 20000),
}

def _read(path: str) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=True, dtype="float32")
    if not np.isfinite(y).all():
        raise ValueError("audio contains NaN/Inf")
    return y, sr

def _db(x: float) -> float:
    return float(20*np.log10(max(float(x), 1e-12)))

def _profile_array(y: np.ndarray, sr: int) -> dict[str, Any]:
    mono = y.mean(axis=1)
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y*y)+1e-20))
    f,p = signal.welch(mono, sr, nperseg=min(8192,len(mono)))
    total = trapezoid(p,f)+1e-20
    bands={}
    for name,(lo,hi) in BANDS.items():
        hi=min(hi,sr/2)
        mask=(f>=lo)&(f<hi)
        bands[name]=float(10*np.log10((trapezoid(p[mask],f[mask])+1e-20)/total))
    stereo=None
    if y.shape[1]==2:
        l,r=y[:,0],y[:,1]
        den=np.sqrt(np.sum(l*l)*np.sum(r*r))+1e-20
        mid=(l+r)*.5; side=(l-r)*.5
        stereo={"correlation":float(np.sum(l*r)/den),
                "side_mid_rms_db":_db(np.sqrt(np.mean(side*side)+1e-20)/
                                      max(np.sqrt(np.mean(mid*mid)+1e-20),1e-12))}
    return {"rms_dbfs":_db(rms),"sample_peak_dbfs":_db(peak),
            "crest_db":_db(peak/max(rms,1e-12)),
            "band_energy_db_relative":bands,"stereo":stereo}

def create_profile(reference_path: str, sections: list[dict[str,Any]] | None=None) -> dict[str,Any]:
    y,sr=_read(reference_path)
    duration=len(y)/sr
    profile={"reference_path":str(Path(reference_path).resolve()),"samplerate":sr,
             "channels":int(y.shape[1]),"duration_s":duration,
             "global":_profile_array(y,sr),"sections":[]}
    for s in sections or []:
        start=max(0,float(s["start_s"])); end=min(duration,float(s["end_s"]))
        if end<=start: continue
        profile["sections"].append({"name":s.get("name","section"),"start_s":start,"end_s":end,
                                    "features":_profile_array(y[int(start*sr):int(end*sr)],sr)})
    profile["limitations"]=[
        "profile describes production traits, not artistic correctness",
        "reference and candidate need not share arrangement, key, tempo or instrumentation",
        "relative spectrum is evidence; do not directly invert it into master EQ",
    ]
    return profile

def _delta(candidate: dict[str,Any], reference: dict[str,Any]) -> dict[str,Any]:
    cb=candidate["band_energy_db_relative"]; rb=reference["band_energy_db_relative"]
    out={"rms_db":candidate["rms_dbfs"]-reference["rms_dbfs"],
         "crest_db":candidate["crest_db"]-reference["crest_db"],
         "bands_db":{k:cb[k]-rb[k] for k in BANDS}}
    if candidate.get("stereo") and reference.get("stereo"):
        out["stereo"]={
            "correlation":candidate["stereo"]["correlation"]-reference["stereo"]["correlation"],
            "side_mid_rms_db":candidate["stereo"]["side_mid_rms_db"]-reference["stereo"]["side_mid_rms_db"]}
    return out

def compare_to_reference(candidate_path: str, profile: dict[str,Any],
                         candidate_sections: list[dict[str,Any]] | None=None) -> dict[str,Any]:
    y,sr=_read(candidate_path)
    if sr != int(profile["samplerate"]):
        raise ValueError("sample rates differ; resample explicitly before reference comparison")
    result={"global_delta":_delta(_profile_array(y,sr),profile["global"]),"section_deltas":[]}
    refs={s["name"]:s for s in profile.get("sections",[])}
    duration=len(y)/sr
    for s in candidate_sections or []:
        name=s.get("name","section")
        if name not in refs: continue
        start=max(0,float(s["start_s"])); end=min(duration,float(s["end_s"]))
        if end<=start: continue
        result["section_deltas"].append({"name":name,
            "delta":_delta(_profile_array(y[int(start*sr):int(end*sr)],sr),refs[name]["features"])})
    result["policy"]="deltas generate hypotheses only; no automatic master-EQ or dynamics correction"
    return result

def build_hypotheses(comparison: dict[str,Any], threshold_db: float=1.5) -> list[dict[str,Any]]:
    d=comparison["global_delta"]; out=[]
    for band,delta in d["bands_db"].items():
        if abs(delta)>=threshold_db:
            out.append({"domain":"tonal_balance","band":band,"delta_db":float(delta),
                        "observation":"candidate has more relative energy" if delta>0 else "candidate has less relative energy",
                        "next_test":"inspect contributing tracks/buses before proposing EQ"})
    if abs(d["crest_db"])>=1.5:
        out.append({"domain":"dynamics","delta_db":float(d["crest_db"]),
                    "observation":"candidate crest is higher" if d["crest_db"]>0 else "candidate crest is lower",
                    "next_test":"compare transient sources and bus compression before changing master dynamics"})
    if "stereo" in d and abs(d["stereo"]["side_mid_rms_db"])>=1.5:
        out.append({"domain":"stereo","delta_db":float(d["stereo"]["side_mid_rms_db"]),
                    "observation":"candidate is wider by S/M energy" if d["stereo"]["side_mid_rms_db"]>0 else "candidate is narrower by S/M energy",
                    "next_test":"inspect stereo contributors and mono compatibility before widening"})
    return out

def save_profile(profile: dict[str,Any], path: str) -> str:
    p=Path(path).expanduser().resolve(); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(profile,ensure_ascii=False,indent=2),encoding="utf-8")
    return str(p)
