from __future__ import annotations
from typing import Any

def significance_gate(delta_rms_dbfs: float, affected_fraction: float,
                      perceptual_verdict: str | None=None,
                      min_delta_rms_dbfs: float=-50.0,
                      min_affected_fraction: float=.02) -> dict[str,Any]:
    """Stop spending iterations on vanishingly small changes.
    This is an engineering relevance gate, not a universal audibility model.
    """
    tiny = float(delta_rms_dbfs) < min_delta_rms_dbfs or float(affected_fraction) < min_affected_fraction
    human_tie = perceptual_verdict in ("tie","uncertain")
    meaningful = not tiny and not human_tie
    reason = "meaningful_candidate"
    if human_tie: reason="blind_ab_tie_or_uncertain"
    elif tiny: reason="effect_below_engineering_significance_floor"
    return {"meaningful":meaningful,"reason":reason,
            "delta_rms_dbfs":float(delta_rms_dbfs),"affected_fraction":float(affected_fraction),
            "thresholds":{"min_delta_rms_dbfs":min_delta_rms_dbfs,
                          "min_affected_fraction":min_affected_fraction},
            "policy":"below-threshold changes are not auto-optimized repeatedly; escalate to a larger causal hypothesis or stop"}

def update_problem_after_tie(problem: dict[str,Any]) -> dict[str,Any]:
    q=dict(problem)
    q["status"]="deprioritized"
    q["reason"]="blind_ab_no_meaningful_difference"
    q["expected_impact"]=min(float(q.get("expected_impact",.5)),.2)
    q["uncertainty"]=max(float(q.get("uncertainty",0)),.7)
    return q
