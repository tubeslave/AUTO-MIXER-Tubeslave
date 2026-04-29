"""Combine manual and optimizer-proposed decision-layer candidates."""

from __future__ import annotations

from typing import Any

from .action_planner import DecisionActionPlanner
from .action_schema import CandidateActionSet, ensure_no_change_candidate
from .nevergrad_optimizer import NevergradActionOptimizer
from .optuna_optimizer import OptunaActionOptimizer


class CandidateGenerator:
    """Produce no-change, manual, and optional optimizer candidates."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.optimizer_status: dict[str, Any] = {}
        self._optimizer: Any | None = None

    def generate(
        self,
        channels: dict[str, str],
        technical_profile: dict[str, Any] | None = None,
        *,
        optimizer_name: str = "nevergrad",
        max_candidates: int = 20,
    ) -> list[CandidateActionSet]:
        planner = DecisionActionPlanner(self.config)
        candidates = planner.plan(channels, technical_profile)
        remaining = max(0, int(max_candidates) - len(candidates))
        action_space = {
            "channels": dict(channels),
            "safety": self.config.get("safety", {}),
            "action_space": self.config.get("action_space", {}),
        }
        seed = int((self.config.get("optimizer", {}) or {}).get("random_seed", 42))
        if optimizer_name == "optuna":
            self._optimizer = OptunaActionOptimizer(self.config.get("optimizer", {}), action_space, seed)
        else:
            self._optimizer = NevergradActionOptimizer(self.config.get("optimizer", {}), action_space, seed)
        optimizer_candidates = self._optimizer.ask_candidates(remaining)
        self.optimizer_status = self._optimizer.status.to_dict()
        candidates.extend(optimizer_candidates)
        seen: set[str] = set()
        unique: list[CandidateActionSet] = []
        for candidate in ensure_no_change_candidate(candidates):
            if candidate.candidate_id in seen:
                continue
            seen.add(candidate.candidate_id)
            unique.append(candidate)
        return unique[:max_candidates]

    def tell_results(self, scores: dict[str, float], metadata: dict[str, dict[str, Any]] | None = None) -> None:
        """Feed evaluated candidate scores back to the active optimizer."""

        if self._optimizer is None:
            return
        metadata = dict(metadata or {})
        for candidate_id, score in scores.items():
            if candidate_id.startswith(f"{self._optimizer.status.name}_"):
                self._optimizer.tell_result(candidate_id, float(score), metadata.get(candidate_id, {}))
        self.optimizer_status = self._optimizer.status.to_dict()
