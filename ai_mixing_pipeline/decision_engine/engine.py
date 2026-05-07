"""Weighted decision engine for AI mixing candidates."""

from __future__ import annotations

from typing import Any

from ai_mixing_pipeline.config import enabled_critic_weights
from ai_mixing_pipeline.models import DecisionResult, MixCandidate, SafetyResult


class DecisionEngine:
    """Select the best candidate from critic and safety scores."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.weights = enabled_critic_weights(self.config)

    def choose_best(
        self,
        candidates: list[MixCandidate],
        evaluations: dict[str, dict[str, dict[str, Any]]],
        safety_results: dict[str, SafetyResult],
    ) -> DecisionResult:
        """Return final scores and selected candidate id."""

        normalized_weights = self._normalize_available_weights(evaluations, safety_results)
        scores: dict[str, float] = {}
        explanations: dict[str, str] = {}
        no_change_id = candidates[0].candidate_id if candidates else "000_initial_mix"
        for candidate in candidates:
            candidate_scores = evaluations.get(candidate.candidate_id, {})
            safety = safety_results.get(candidate.candidate_id)
            final_score = 0.0
            parts = []
            for critic_name, weight in normalized_weights.items():
                if critic_name == "safety":
                    value = float(safety.safety_score if safety is not None else 0.0)
                else:
                    value = self._extract_delta(candidate_scores.get(critic_name, {}))
                final_score += weight * value
                parts.append(f"{critic_name}={value:.4f}*{weight:.3f}")
            if safety is not None and not safety.passed and candidate.candidate_id != no_change_id:
                final_score -= 1.0
                parts.append("safety_block=-1.0000")
            scores[candidate.candidate_id] = float(final_score)
            explanations[candidate.candidate_id] = "; ".join(parts)

        selected = no_change_id
        if scores:
            selected = max(scores.items(), key=lambda item: item[1])[0]
        if selected != no_change_id:
            no_change_score = scores.get(no_change_id, 0.0)
            improvement = scores.get(selected, 0.0) - no_change_score
            min_improvement = float((self.config.get("safety", {}) or {}).get("min_score_improvement", 0.03))
            if improvement < min_improvement:
                selected = no_change_id
                explanations[no_change_id] = (
                    explanations.get(no_change_id, "")
                    + f"; selected because best improvement {improvement:.4f} < {min_improvement:.4f}"
                )

        return DecisionResult(
            selected_candidate_id=selected,
            final_scores=scores,
            normalized_weights=normalized_weights,
            explanations=explanations,
            no_change_selected=selected == no_change_id,
        )

    def _normalize_available_weights(
        self,
        evaluations: dict[str, dict[str, dict[str, Any]]],
        safety_results: dict[str, SafetyResult],
    ) -> dict[str, float]:
        available: dict[str, float] = {}
        critic_names = set()
        for candidate_scores in evaluations.values():
            critic_names.update(candidate_scores.keys())
        for critic_name in critic_names:
            weight = float(self.weights.get(critic_name, 0.0))
            if weight <= 0.0:
                continue
            if self._critic_has_signal(critic_name, evaluations):
                available[critic_name] = weight
        if safety_results:
            available["safety"] = float(self.weights.get("safety", 0.05))
        total = sum(available.values())
        if total <= 0.0:
            return {"safety": 1.0}
        return {name: value / total for name, value in available.items()}

    @staticmethod
    def _critic_has_signal(
        critic_name: str,
        evaluations: dict[str, dict[str, dict[str, Any]]],
    ) -> bool:
        for candidate_scores in evaluations.values():
            result = candidate_scores.get(critic_name)
            if not result:
                continue
            if result.get("confidence", 0.0) > 0.0:
                return True
            if result.get("scores"):
                return True
        return False

    @staticmethod
    def _extract_delta(result: dict[str, Any]) -> float:
        delta = result.get("delta", {}) if isinstance(result, dict) else {}
        for key in ("overall", "quality_score", "technical_score", "semantic_alignment"):
            value = delta.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0
