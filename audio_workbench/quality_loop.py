from __future__ import annotations
from typing import Any

from . import causal


_CRITIC_DECISIONS = {"machine_safe", "pending_human_review", "rejected"}


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


def advance_perceptual_iteration(plan: dict[str, Any], candidate: dict[str, Any],
                                 critic_result: dict[str, Any], baseline_id: str,
                                 evaluation_confidence: float) -> dict[str, Any]:
    """Resolve one offline studio candidate without silently moving the baseline.

    Perceptual Critic v2 is authoritative for its three machine states. Rejected
    evidence is a veto. Pending evidence is held for listening. A machine-safe
    perceptual change is still held whenever the critic marks it as requiring human
    listening. Evidence and uncertainty are copied into the transition record so an
    autonomous loop cannot discard the reason for a decision.
    """
    decision = str(critic_result.get("machine_decision", "")).strip().lower()
    if decision not in _CRITIC_DECISIONS:
        raise ValueError(f"unsupported perceptual critic decision: {decision!r}")
    if not str(baseline_id).strip():
        raise ValueError("baseline_id is required")
    candidate_id = candidate.get("id")
    if candidate_id is None or not str(candidate_id).strip():
        raise ValueError("candidate.id is required for baseline transitions")

    evidence = list(critic_result.get("evidence") or [])
    uncertainty = dict(critic_result.get("uncertainty") or {})
    failures = list(critic_result.get("failures") or [])
    protected_regressions = list(critic_result.get("protected_regressions") or [])
    target_improved = "target_not_improved" not in failures

    iteration_candidate = dict(candidate)
    requires_human = bool(critic_result.get("requires_human_listening"))
    if requires_human or decision == "pending_human_review":
        iteration_candidate["requires_human_review"] = True

    evaluation = evaluate_candidate(
        plan,
        iteration_candidate,
        target_improved,
        protected_regressions,
        evaluation_confidence,
    )

    # A Perceptual Critic rejection is terminal for this candidate. Human approval
    # can resolve subjective/pending evidence, but cannot override a machine veto.
    if decision == "rejected":
        evaluation["accepted"] = False
        evaluation["acceptance_state"] = "critic_rejected"

    accepted = bool(evaluation["accepted"])
    state = str(evaluation["acceptance_state"])
    baseline_after = str(candidate_id) if accepted else str(baseline_id)
    rollback_candidate = state in {
        "critic_rejected", "human_rejected", "machine_gate_failed"
    }

    return {
        **evaluation,
        "critic_machine_decision": decision,
        "evidence": evidence,
        "uncertainty": uncertainty,
        "protected_regressions": protected_regressions,
        "baseline_before": str(baseline_id),
        "baseline_after": baseline_after,
        "promote_baseline": accepted,
        "rollback_candidate": rollback_candidate,
        "audit_policy": (
            "baseline changes only after the critic, causal machine gate, and any "
            "required human listening gate all allow the candidate"
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
