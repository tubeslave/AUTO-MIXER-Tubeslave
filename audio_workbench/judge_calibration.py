from __future__ import annotations
from typing import Any

CAPABILITIES=("level","tonal_balance","dynamics","stereo","clarity","harshness","depth","overall_preference")

def score_capability(trials: list[dict[str,Any]], capability: str,
                     min_accuracy: float=.8, max_false_preference: float=.1) -> dict[str,Any]:
    rows=[t for t in trials if t.get("capability")==capability]
    if not rows:
        return {"capability":capability,"enabled":False,"reason":"no_trials"}
    normal=[t for t in rows if not t.get("catch_trial",False)]
    catch=[t for t in rows if t.get("catch_trial",False)]
    correct=sum(bool(t.get("correct",False)) for t in normal)
    accuracy=correct/max(1,len(normal))
    false=sum(bool(t.get("confident_false_preference",False)) for t in catch)
    false_rate=false/max(1,len(catch)) if catch else 0.0
    enabled=bool(normal) and accuracy>=min_accuracy and false_rate<=max_false_preference
    return {"capability":capability,"enabled":enabled,"accuracy":accuracy,
            "false_preference_rate":false_rate,"trials":len(rows),
            "thresholds":{"min_accuracy":min_accuracy,"max_false_preference":max_false_preference}}

def authority_map(trials: list[dict[str,Any]]) -> dict[str,Any]:
    caps={c:score_capability(trials,c) for c in CAPABILITIES}
    return {"capabilities":caps,
            "policy":"only enabled capabilities may contribute decision weight; disabled outputs remain annotations"}

def filter_verdict(verdict: dict[str,Any], authority: dict[str,Any]) -> dict[str,Any]:
    axes=verdict.get("axes",{})
    kept={}
    for axis,value in axes.items():
        cap={"punch":"dynamics","low_end":"tonal_balance","stereo":"stereo",
             "harshness":"harshness","depth":"depth","clarity":"clarity"}.get(axis)
        if cap and authority["capabilities"].get(cap,{}).get("enabled"):
            kept[axis]=value
    overall_allowed=authority["capabilities"].get("overall_preference",{}).get("enabled",False)
    return {"axes":kept,
            "preference":verdict.get("preference") if overall_allowed else "uncertain",
            "confidence":float(verdict.get("confidence",0)) if overall_allowed else 0.0,
            "authority_filtered":True}
