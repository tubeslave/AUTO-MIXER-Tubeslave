"""Reward-signal abstraction for future agent/RL training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return float(max(low, min(high, float(value))))


@dataclass
class RewardSignal:
    engineering_score: float = 0.0
    perceptual_score: float = 0.0
    safety_score: float = 1.0
    combined_score: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def combine(
        cls,
        engineering_score: float = 0.0,
        perceptual_score: float = 0.0,
        safety_score: float = 1.0,
        weights: Optional[Dict[str, float]] = None,
    ) -> "RewardSignal":
        weights = dict(weights or {})
        weights.setdefault("engineering", 0.4)
        weights.setdefault("perceptual", 0.4)
        weights.setdefault("safety", 0.2)
        total = sum(max(0.0, float(value)) for value in weights.values()) or 1.0

        engineering = _clamp(engineering_score)
        perceptual = _clamp(perceptual_score)
        safety = _clamp(safety_score)
        combined = (
            weights["engineering"] * engineering
            + weights["perceptual"] * perceptual
            + weights["safety"] * safety
        ) / total

        return cls(
            engineering_score=engineering,
            perceptual_score=perceptual,
            safety_score=safety,
            combined_score=_clamp(combined),
            weights=weights,
        )

    def to_dict(self) -> Dict[str, float | Dict[str, float]]:
        return {
            "engineering_score": float(self.engineering_score),
            "perceptual_score": float(self.perceptual_score),
            "safety_score": float(self.safety_score),
            "combined_score": float(self.combined_score),
            "weights": dict(self.weights),
        }
