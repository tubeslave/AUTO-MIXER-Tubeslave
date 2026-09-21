from __future__ import annotations
from typing import Any

from .quality_loop import rank_next_problem, evaluate_candidate
from .uncertainty import assess

def next_iteration(problems: list[dict[str,Any]],
                   confidence: dict[str,float],
                   observer_disagreement: float=0.0) -> dict[str,Any]:
    problem=rank_next_problem(problems)
    if problem is None:
        return {"status":"stop","reason":"no_remaining_problem"}
    u=assess(confidence,observer_disagreement)
    return {"status":"investigate","problem":problem,"uncertainty":u,
            "next_action":u["next_action"]}

def stop_decision(problems: list[dict[str,Any]], accepted_improvements: int,
                  failed_recent_experiments: int, operator_stop: bool=False) -> dict[str,Any]:
    if operator_stop: return {"stop":True,"reason":"operator_stop"}
    meaningful=[p for p in problems if float(p.get("importance",0))>=.5 and float(p.get("confidence",0))>=.65]
    if not meaningful: return {"stop":True,"reason":"no_high_confidence_meaningful_problem"}
    if failed_recent_experiments>=3:
        return {"stop":True,"reason":"repeated_experiments_failed_to_improve"}
    return {"stop":False,"reason":"continue"}
