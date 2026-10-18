from __future__ import annotations
from typing import Any

DOMAINS=("drums","bass","guitars","vocals","space","stereo","macro","master")

def composite_reference(profiles: dict[str,dict[str,Any]],
                        assignments: dict[str,str]) -> dict[str,Any]:
    out={"schema_version":2,"domains":{},"conflicts":[]}
    for domain,source in assignments.items():
        if domain not in DOMAINS: raise ValueError(f"unknown domain: {domain}")
        if source not in profiles: raise KeyError(f"unknown reference: {source}")
        out["domains"][domain]={"source":source,"profile":profiles[source]}
    return out

def target_range(values: list[float], padding: float=.5) -> dict[str,float]:
    if not values: raise ValueError("no values")
    lo=min(map(float,values)); hi=max(map(float,values))
    return {"min":lo-padding,"max":hi+padding,"center":sum(map(float,values))/len(values)}

def compare_range(value: float, target: dict[str,float]) -> dict[str,Any]:
    v=float(value)
    if v<target["min"]: state="below"
    elif v>target["max"]: state="above"
    else: state="inside"
    return {"value":v,"state":state,"distance":0.0 if state=="inside" else
            (target["min"]-v if state=="below" else v-target["max"])}
