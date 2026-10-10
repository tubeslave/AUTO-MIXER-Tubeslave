from __future__ import annotations
from typing import Any

import numpy as np
import soundfile as sf

def _rms_db(x: np.ndarray) -> float:
    rms=float(np.sqrt(np.mean(x*x)+1e-20))
    return float(20*np.log10(max(rms,1e-12)))

def energy_curve(path: str, sections: list[dict[str,Any]]) -> dict[str,Any]:
    y,sr=sf.read(path,always_2d=True,dtype="float32")
    duration=len(y)/sr
    rows=[]
    for s in sections:
        start=max(0,float(s["start_s"])); end=min(duration,float(s["end_s"]))
        if end<=start: continue
        x=y[int(start*sr):int(end*sr)]
        rows.append({"name":s.get("name","section"),"start_s":start,"end_s":end,
                     "rms_dbfs":_rms_db(x)})
    if not rows:
        return {"sections":[],"normalized":[]}
    base=min(r["rms_dbfs"] for r in rows)
    normalized=[{"name":r["name"],"lift_db":r["rms_dbfs"]-base} for r in rows]
    return {"sections":rows,"normalized":normalized,
            "policy":"macro energy describes section relationships, not a requirement that later sections are always louder"}

def contrast_regression(before: dict[str,Any], after: dict[str,Any],
                        protected_pairs: list[dict[str,Any]], tolerance_db: float=.75) -> dict[str,Any]:
    b={x["name"]:x["lift_db"] for x in before.get("normalized",[])}
    a={x["name"]:x["lift_db"] for x in after.get("normalized",[])}
    issues=[]
    for p in protected_pairs:
        lo=p["from"]; hi=p["to"]; min_lift=float(p.get("min_lift_db",0))
        if lo not in a or hi not in a: continue
        before_delta=b.get(hi,0)-b.get(lo,0)
        after_delta=a[hi]-a[lo]
        if after_delta < min_lift-tolerance_db or after_delta < before_delta-tolerance_db:
            issues.append({"from":lo,"to":hi,"before_delta_db":before_delta,
                           "after_delta_db":after_delta,"min_lift_db":min_lift})
    return {"passed":not issues,"issues":issues}
