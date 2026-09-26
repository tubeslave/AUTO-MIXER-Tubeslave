from __future__ import annotations
from typing import Any

FIELDS=("observation","cause","intervention","evaluation")

def assess(confidence: dict[str,float], observer_disagreement: float=0.0) -> dict[str,Any]:
    vals={}
    for k in FIELDS:
        v=float(confidence.get(k,0))
        if not 0<=v<=1: raise ValueError(f"{k} confidence outside 0..1")
        vals[k]=v
    disagreement=max(0.0,min(1.0,float(observer_disagreement)))
    vals["evaluation_adjusted"]=vals["evaluation"]*(1-.5*disagreement)
    weakest=min(vals[k] for k in FIELDS)
    if vals["cause"]<.7:
        action="diagnostic_experiment"
    elif vals["evaluation_adjusted"]<.75:
        action="more_evidence_or_human_ab"
    elif vals["intervention"]<.7:
        action="compare_alternative_interventions"
    else:
        action="eligible_for_protected_gate"
    return {"confidence":vals,"weakest":weakest,"observer_disagreement":disagreement,
            "next_action":action}
