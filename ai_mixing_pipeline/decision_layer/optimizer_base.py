"""Shared optimizer protocol for candidate action optimizers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .action_schema import CandidateActionSet


@dataclass
class OptimizerStatus:
    name: str
    available: bool
    warnings: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "warnings": list(self.warnings),
            "history": list(self.history),
        }


class ActionOptimizer(Protocol):
    status: OptimizerStatus

    def ask_candidates(self, n_candidates: int) -> list[CandidateActionSet]:
        ...

    def tell_result(self, candidate_id: str, score: float, metadata: dict[str, Any]) -> None:
        ...

    def recommend_best(self) -> CandidateActionSet | None:
        ...
