"""Decision engine for offline correction candidates."""

from __future__ import annotations

from typing import Any

from .action_schema import CandidateActionSet
from .score_normalizer import aggregate_scores, configured_weights
from ai_mixing_pipeline.numeric_safety import finite_float


class CorrectionDecisionEngine:
    """Select the best safe candidate or keep no_change."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.weights = configured_weights(self.config)
        self.min_score_improvement = finite_float(
            (self.config.get("safety", {}) or {}).get("min_score_improvement", 0.03)
        )
        self.proxy_weight_multiplier = finite_float(
            self.config.get("proxy_weight_multiplier", 1.0)
        )
        if self.min_score_improvement is None or self.min_score_improvement < 0.0:
            raise ValueError("min_score_improvement must be finite and non-negative")
        if self.proxy_weight_multiplier is None or self.proxy_weight_multiplier < 0.0:
            raise ValueError("proxy_weight_multiplier must be finite and non-negative")

    def choose_best(
        self,
        run_id: str,
        candidates: list[CandidateActionSet],
        critic_scores: dict[str, dict[str, dict[str, Any]]],
        safety_scores: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if not candidates:
            raise ValueError("Decision engine requires at least one candidate")
        no_change = next((candidate for candidate in candidates if candidate.is_no_change), candidates[0])
        breakdown: dict[str, Any] = {}
        final_scores: dict[str, float] = {}
        validated_safety: dict[str, bool] = {}
        for candidate in candidates:
            combined_scores = dict(critic_scores.get(candidate.candidate_id, {}))
            combined_scores["safety"] = safety_scores.get(candidate.candidate_id, {})
            aggregate = aggregate_scores(
                combined_scores,
                self.weights,
                proxy_weight_multiplier=self.proxy_weight_multiplier,
            )
            safety = safety_scores.get(candidate.candidate_id, {})
            safety_score = finite_float(safety.get("safety_score"))
            safety_passed = safety.get("passed") is True and safety_score is not None
            validated_safety[candidate.candidate_id] = safety_passed
            if not safety_passed and not candidate.is_no_change:
                aggregate["final_score"] -= 1.0
            breakdown[candidate.candidate_id] = aggregate
            final_scores[candidate.candidate_id] = float(aggregate["final_score"])

        selected_id = max(final_scores.items(), key=lambda item: item[1])[0]
        selected = next(candidate for candidate in candidates if candidate.candidate_id == selected_id)
        no_change_score = final_scores.get(no_change.candidate_id, 0.0)
        score_delta = final_scores[selected_id] - no_change_score
        decision = "accept"
        reason = "Best safe candidate exceeded no-change by the configured threshold."
        if selected.candidate_id == no_change.candidate_id or score_delta < self.min_score_improvement:
            selected = no_change
            selected_id = no_change.candidate_id
            score_delta = 0.0
            decision = "no_change"
            reason = f"Best improvement below min_score_improvement {self.min_score_improvement:.4f}; keeping no_change."
        if not validated_safety.get(selected_id, False):
            selected = no_change
            selected_id = no_change.candidate_id
            score_delta = 0.0
            decision = "reject"
            reason = "Best candidate failed Safety Governor; keeping no_change."

        return {
            "run_id": run_id,
            "selected_candidate_id": selected_id,
            "selected_actions": [action.to_dict() for action in selected.actions] if not selected.is_no_change else [],
            "final_score": final_scores.get(selected_id, 0.0),
            "score_delta_vs_no_change": score_delta,
            "safety_passed": validated_safety.get(selected_id, False),
            "decision": decision,
            "reason": reason,
            "critic_breakdown": breakdown,
            "warnings": list(safety_scores.get(selected_id, {}).get("warnings", [])),
            "final_scores": final_scores,
            "normalized_weights": breakdown.get(selected_id, {}).get("normalized_weights", {}),
        }
