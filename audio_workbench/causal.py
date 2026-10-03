from __future__ import annotations
from typing import Any

ALLOWED={"gain","eq_bell","compressor","bypass"}

def make_plan(observation: str, hypothesis: str, target: str,
              interventions: list[dict[str,Any]], expected_effect: str,
              protected_metrics: list[str], confidence: dict[str,float] | None=None) -> dict[str,Any]:
    if not observation.strip() or not hypothesis.strip() or not expected_effect.strip():
        raise ValueError("observation, hypothesis and expected_effect are required")
    candidates=[{"type":"bypass","params":{},"label":"no_change"}]
    for x in interventions:
        if x.get("type") not in ALLOWED-{"bypass"}:
            raise ValueError(f"unsupported intervention: {x.get('type')}")
        candidates.append(x)
    conf=confidence or {}
    for k,v in conf.items():
        if not 0<=float(v)<=1: raise ValueError(f"confidence outside 0..1: {k}")
    return {"observation":observation,"hypothesis":hypothesis,"target":target,
            "expected_effect":expected_effect,"protected_metrics":protected_metrics,
            "candidates":candidates,"confidence":conf,"status":"proposed",
            "policy":"candidate must improve target without protected regression; no_change is mandatory"}

def may_auto_accept(plan: dict[str,Any], target_improved: bool,
                    protected_regressions: list[str], evaluation_confidence: float) -> dict[str,Any]:
    cause=float(plan.get("confidence",{}).get("cause",0))
    intervention=float(plan.get("confidence",{}).get("intervention",0))
    ok=(target_improved and not protected_regressions and cause>=.7 and intervention>=.7
        and float(evaluation_confidence)>=.75)
    return {"allowed":ok,"target_improved":target_improved,
            "protected_regressions":protected_regressions,
            "reason":"accepted_by_gate" if ok else "needs_more_evidence_or_reject"}
