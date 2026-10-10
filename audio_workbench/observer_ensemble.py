from __future__ import annotations
from typing import Any
from .perceptual import aggregate

def disagreement(verdicts: list[dict[str,Any]]) -> float:
    strong=[v for v in verdicts if v.get("preference") in ("A","B") and float(v.get("confidence",0))>=.65]
    if len(strong)<2: return 0.0
    a=sum(1 for v in strong if v["preference"]=="A")
    b=len(strong)-a
    return 1.0-min(1.0,abs(a-b)/len(strong))

def evaluate(verdicts: list[dict[str,Any]]) -> dict[str,Any]:
    return {"aggregate":aggregate(verdicts),"disagreement":disagreement(verdicts),
            "policy":"observer agreement is evidence; it does not bypass protected technical/macro gates"}
