from __future__ import annotations
from typing import Any

from . import causal

def evaluate_candidate(plan: dict[str,Any], candidate: dict[str,Any],
                       target_improved: bool,
                       protected_regressions: list[str],
                       evaluation_confidence: float) -> dict[str,Any]:
    gate=causal.may_auto_accept(plan,target_improved,protected_regressions,evaluation_confidence)
    return {
        "candidate":candidate,
        "gate":gate,
        "accepted":bool(gate["allowed"]),
        "policy":"accept only when target improves, protected metrics hold, and confidence gates pass",
    }

def rank_next_problem(problems: list[dict[str,Any]]) -> dict[str,Any] | None:
    """Rank by importance * confidence * expected impact.
    This prioritizes what to inspect next, not which processing move is correct.
    """
    if not problems:
        return None
    scored=[]
    for p in problems:
        importance=float(p.get("importance",0.5))
        confidence=float(p.get("confidence",0.5))
        impact=float(p.get("expected_impact",0.5))
        uncertainty_penalty=1.0-float(p.get("uncertainty",0.0))
        score=importance*confidence*impact*max(0.0,uncertainty_penalty)
        q=dict(p); q["priority_score"]=score
        scored.append(q)
    return max(scored,key=lambda x:x["priority_score"])
