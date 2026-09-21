from __future__ import annotations

from typing import Any
import numpy as np
import soundfile as sf
from scipy import signal

def _load_mono(path: str) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=True, dtype="float32")
    if not np.isfinite(y).all():
        raise ValueError("audio contains NaN/Inf")
    return y.mean(axis=1), sr

def section_features(path: str, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    y, sr = _load_mono(path)
    out = []
    for s in sections:
        start = max(0.0, float(s["start_s"]))
        end = min(len(y)/sr, float(s["end_s"]))
        if end <= start:
            continue
        x = y[int(start*sr):int(end*sr)]
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        rms = float(np.sqrt(np.mean(x*x)+1e-20)) if len(x) else 0.0
        crest = 20*np.log10(max(peak/max(rms,1e-12),1e-12))
        f,p = signal.welch(x, sr, nperseg=min(4096,len(x)))
        centroid = float(np.sum(f*p)/(np.sum(p)+1e-20))
        out.append({
            "name": s.get("name","section"), "start_s":start, "end_s":end,
            "rms_dbfs": float(20*np.log10(max(rms,1e-12))),
            "sample_peak_dbfs": float(20*np.log10(max(peak,1e-12))),
            "crest_db": float(crest), "spectral_centroid_hz":centroid
        })
    return out

def fixed_windows(path: str, window_s: float = 8.0, hop_s: float = 4.0) -> list[dict[str, Any]]:
    y, sr = _load_mono(path)
    duration = len(y)/sr
    sections = []
    t = 0.0
    i = 0
    while t < duration:
        sections.append({"name":f"window_{i:04d}","start_s":t,"end_s":min(t+window_s,duration)})
        if t + window_s >= duration:
            break
        t += hop_s
        i += 1
    return section_features(path, sections)

def find_outlier_windows(path: str, window_s: float = 8.0, hop_s: float = 4.0, top_k: int = 8) -> dict[str, Any]:
    rows = fixed_windows(path,window_s,hop_s)
    if len(rows) < 3:
        return {"windows":rows,"outliers":[]}
    keys = ["rms_dbfs","crest_db","spectral_centroid_hz"]
    a = np.array([[r[k] for k in keys] for r in rows],dtype=float)
    med = np.median(a,axis=0)
    mad = np.median(np.abs(a-med),axis=0)+1e-9
    z = np.abs((a-med)/(1.4826*mad))
    scores = np.max(z,axis=1)
    idx = np.argsort(scores)[::-1][:top_k]
    outliers = []
    for j in idx:
        r = dict(rows[int(j)])
        r["outlier_score"] = float(scores[int(j)])
        r["dominant_feature"] = keys[int(np.argmax(z[int(j)]))]
        outliers.append(r)
    return {"windows":rows,"outliers":outliers,"method":"robust MAD over RMS/crest/spectral-centroid; diagnostic only"}
