"""Common AudioCritic interface and helpers."""

from __future__ import annotations

from typing import Any


def standard_critic_result(
    *,
    critic_name: str,
    role: str,
    scores: dict[str, Any] | None = None,
    delta: dict[str, Any] | None = None,
    confidence: float = 0.0,
    warnings: list[str] | None = None,
    explanation: str = "",
    model_available: bool = False,
    score_source: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the standard critic payload shape."""

    if score_source is None:
        score_source = "real_model" if model_available else ("proxy" if scores or delta else "unavailable")
    return {
        "critic_name": critic_name,
        "role": role,
        "scores": scores or {},
        "delta": delta or {},
        "confidence": float(max(0.0, min(1.0, confidence))),
        "warnings": list(warnings or []),
        "explanation": explanation,
        "model_available": bool(model_available),
        "score_source": str(score_source),
        "proxy_score": bool(score_source not in {"real_model", "unavailable"}),
        "metadata": metadata or {},
    }


def _numeric_scores(scores: dict[str, Any]) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for key, value in scores.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            numeric[key] = float(value)
    return numeric


class AudioCritic:
    """Base interface for all offline audio critics."""

    name: str = "audio_critic"
    role: str = "critic"

    def analyze(self, audio_path: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def compare(
        self,
        before_path: str,
        after_path: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Default before/after comparison from numeric analyze scores."""

        before = self.analyze(before_path, context=context)
        after = self.analyze(after_path, context=context)
        before_scores = _numeric_scores(before.get("scores", {}))
        after_scores = _numeric_scores(after.get("scores", {}))
        delta = {
            key: float(after_scores[key] - before_scores.get(key, 0.0))
            for key in after_scores
        }
        if "overall" not in delta:
            delta["overall"] = float(delta.get("quality_score", 0.0))
        warnings = list(before.get("warnings", [])) + list(after.get("warnings", []))
        confidence = min(float(before.get("confidence", 0.0)), float(after.get("confidence", 0.0)))
        model_available = bool(before.get("model_available")) and bool(after.get("model_available"))
        return standard_critic_result(
            critic_name=self.name,
            role=self.role,
            scores=after.get("scores", {}),
            delta=delta,
            confidence=confidence,
            warnings=warnings,
            explanation=f"Compared {self.name} before/after scores.",
            model_available=model_available,
            score_source="real_model" if model_available else "proxy",
            metadata={
                "before": before.get("scores", {}),
                "after": after.get("scores", {}),
            },
        )

    def unavailable_result(self, reason: str) -> dict[str, Any]:
        """Return a non-fatal unavailable result."""

        return standard_critic_result(
            critic_name=self.name,
            role=self.role,
            scores={},
            delta={"overall": 0.0},
            confidence=0.0,
            warnings=[reason],
            explanation=f"{self.name} is unavailable; pipeline continued.",
            model_available=False,
            score_source="unavailable",
        )
