from __future__ import annotations
from typing import Any

from . import causal


def _human_review_status(candidate: dict[str, Any]) -> str:
    """Normalize an optional subjective-review decision.

    Machine evidence can validate a candidate, but subjective audio transforms may
    explicitly require a human listening decision before they become accepted.
    """
    review = candidate.get("human_review")
    if isinstance(review, dict):
        review = review.get("status")
    if review is None:
        return "pending"
    value = str(review).strip().lower()
    if value in {"accepted", "accept", "approved", "approve", "pass"}:
        return "accepted"
    if value in {"rejected", "reject", "failed", "fail"}:
        return "rejected"
    return "pending"


def evaluate_candidate(plan: dict[str, Any], candidate: dict[str, Any],
                       target_improved: bool,
                       protected_regressions: list[str],
                       evaluation_confidence: float) -> dict[str, Any]:
    machine_gate = causal.may_auto_accept(
        plan, target_improved, protected_regressions, evaluation_confidence
    )
    requires_human_review = bool(
        plan.get("requires_human_review") or candidate.get("requires_human_review")
    )
    human_review_status = _human_review_status(candidate)
    machine_allowed = bool(machine_gate["allowed"])

    if requires_human_review:
        if human_review_status == "rejected":
            accepted = False
            acceptance_state = "human_rejected"
        elif not machine_allowed:
            accepted = False
            acceptance_state = "machine_gate_failed"
        elif human_review_status != "accepted":
            accepted = False
            acceptance_state = "pending_human_review"
        else:
            accepted = True
            acceptance_state = "accepted"
    else:
        accepted = machine_allowed
        acceptance_state = "accepted" if accepted else "machine_gate_failed"

    return {
        "candidate": candidate,
        "gate": machine_gate,
        "accepted": accepted,
        "acceptance_state": acceptance_state,
        "human_review_required": requires_human_review,
        "human_review_status": human_review_status if requires_human_review else "not_required",
        "policy": (
            "machine gate must pass; candidates marked requires_human_review also need "
            "an explicit human listening acceptance"
        ),
    }


def rank_next_problem(problems: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Rank by importance * confidence * expected impact.
    This prioritizes what to inspect next, not which processing move is correct.
    """
    if not problems:
        return None
    scored = []
    for p in problems:
        importance = float(p.get("importance", 0.5))
        confidence = float(p.get("confidence", 0.5))
        impact = float(p.get("expected_impact", 0.5))
        uncertainty_penalty = 1.0 - float(p.get("uncertainty", 0.0))
        score = importance * confidence * impact * max(0.0, uncertainty_penalty)
        q = dict(p)
        q["priority_score"] = score
        scored.append(q)
    return max(scored, key=lambda x: x["priority_score"])
