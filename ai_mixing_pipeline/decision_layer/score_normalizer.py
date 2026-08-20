"""Normalize and aggregate critic scores for decision-layer candidates."""

from __future__ import annotations

from typing import Any

from ai_mixing_pipeline.numeric_safety import finite_float


DEFAULT_WEIGHTS = {
    "muq_eval": 0.30,
    "audiobox_aesthetics": 0.20,
    "mert": 0.15,
    "clap": 0.10,
    "essentia": 0.10,
    "panns_or_beats": 0.10,
    "safety": 0.05,
}

ALIASES = {
    "audiobox": "audiobox_aesthetics",
    "stem_critics": "mert",
    "identity_bleed": "panns_or_beats",
}


def normalize_score(name: str, raw_score: Any) -> float:
    value = finite_float(raw_score)
    if value is None:
        return 0.0
    if name == "safety":
        return max(0.0, min(1.0, value))
    return max(-1.0, min(1.0, value))


def normalize_delta(before: Any, after: Any) -> float:
    before_value = finite_float(before)
    after_value = finite_float(after)
    if before_value is None or after_value is None:
        return 0.0
    return max(-1.0, min(1.0, after_value - before_value))


def configured_weights(config: dict[str, Any] | None = None) -> dict[str, float]:
    config = dict(config or {})
    critics = dict(config.get("critics", {}) or {})
    weights = dict(DEFAULT_WEIGHTS)
    for key, value in critics.items():
        canonical = ALIASES.get(key, key)
        if isinstance(value, dict) and bool(value.get("enabled", True)):
            weight = finite_float(value.get("weight", weights.get(canonical, 0.0)))
            if weight is None or weight < 0.0:
                raise ValueError(f"critic weight for {canonical} must be finite and non-negative")
            weights[canonical] = weight
    return weights


def extract_signal(critic_name: str, result: dict[str, Any]) -> float | None:
    if not result:
        return None
    delta = result.get("delta", {}) or {}
    for key in ("overall", "quality_score", "technical_score", "semantic_alignment"):
        if isinstance(delta.get(key), (int, float)):
            return normalize_score(critic_name, delta[key])
    scores = result.get("scores", {}) or {}
    if critic_name == "panns_or_beats":
        present = [
            scores[key]
            for key in ("identity_confidence", "bleed_score", "activity_score")
            if key in scores
        ]
        values = [finite_float(value) for value in present]
        if any(value is None for value in values):
            return None
        if values:
            return max(0.0, min(1.0, sum(values) / len(values)))
    if isinstance(scores.get("overall"), (int, float)):
        return normalize_score(critic_name, scores["overall"])
    return None


def _weight_source(result: dict[str, Any]) -> str:
    if str(result.get("score_source", "")).lower() == "proxy":
        return "proxy"
    if result.get("model_available") is False:
        return "proxy"
    if result.get("model_available") is True:
        return "model"
    return "unknown"


def aggregate_scores(
    scores: dict[str, dict[str, Any]],
    weights: dict[str, float],
    *,
    proxy_weight_multiplier: float = 1.0,
) -> dict[str, Any]:
    available: dict[str, float] = {}
    values: dict[str, float] = {}
    effective_weights: dict[str, float] = {}
    weight_sources: dict[str, str] = {}
    proxy_multiplier = finite_float(proxy_weight_multiplier)
    proxy_weight_multiplier = max(0.0, proxy_multiplier) if proxy_multiplier is not None else 0.0
    for name, weight in weights.items():
        safe_weight = finite_float(weight)
        if safe_weight is None or safe_weight < 0.0:
            continue
        weight = safe_weight
        if name == "safety":
            safety = scores.get("safety", {})
            value = safety.get("safety_score")
            if isinstance(value, (int, float)):
                available[name] = weight
                effective_weights[name] = weight
                weight_sources[name] = "safety"
                values[name] = normalize_score(name, value)
            continue
        result = scores.get(name, {})
        value = extract_signal(name, result)
        if value is not None:
            source = _weight_source(result)
            effective_weight = weight * proxy_weight_multiplier if source == "proxy" else weight
            available[name] = effective_weight
            effective_weights[name] = effective_weight
            weight_sources[name] = source
            values[name] = value
    total = sum(available.values())
    if total <= 0.0:
        return {
            "final_score": 0.0,
            "normalized_weights": {},
            "effective_weights": effective_weights,
            "weight_sources": weight_sources,
            "values": values,
        }
    normalized = {name: weight / total for name, weight in available.items()}
    final_score = sum(values[name] * normalized[name] for name in normalized)
    if finite_float(final_score) is None:
        final_score = 0.0
    return {
        "final_score": float(final_score),
        "normalized_weights": normalized,
        "effective_weights": effective_weights,
        "weight_sources": weight_sources,
        "values": values,
    }
