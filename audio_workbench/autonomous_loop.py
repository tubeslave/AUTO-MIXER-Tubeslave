from __future__ import annotations
from typing import Any

from .quality_loop import rank_next_problem, advance_perceptual_iteration
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


def resolve_candidate_iteration(plan: dict[str,Any], candidate: dict[str,Any],
                                critic_result: dict[str,Any], baseline_id: str,
                                evaluation_confidence: float) -> dict[str,Any]:
    """Turn Perceptual Critic evidence into an auditable iteration transition.

    The autonomous loop never mutates a baseline directly. It returns an explicit
    action: promote only an accepted candidate, hold subjective/uncertain candidates
    for listening, or roll back rejected/unsafe candidates.
    """
    result=advance_perceptual_iteration(
        plan,candidate,critic_result,baseline_id,evaluation_confidence
    )
    if result["accepted"]:
        status="accepted"
        next_action="promote_baseline"
    elif result["acceptance_state"]=="pending_human_review":
        status="pending_human_review"
        next_action="human_listening"
    elif result["rollback_candidate"]:
        status="rejected"
        next_action="rollback_candidate"
    else:
        status="blocked"
        next_action="collect_more_evidence"
    return {**result,"status":status,"next_action":next_action}


def stop_decision(problems: list[dict[str,Any]], accepted_improvements: int,
                  failed_recent_experiments: int, operator_stop: bool=False) -> dict[str,Any]:
    if operator_stop: return {"stop":True,"reason":"operator_stop"}
    meaningful=[p for p in problems if float(p.get("importance",0))>=.5 and float(p.get("confidence",0))>=.65]
    if not meaningful: return {"stop":True,"reason":"no_high_confidence_meaningful_problem"}
    if failed_recent_experiments>=3:
        return {"stop":True,"reason":"repeated_experiments_failed_to_improve"}
    return {"stop":False,"reason":"continue"}
