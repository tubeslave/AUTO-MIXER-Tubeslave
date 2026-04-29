"""Normalize and aggregate critic scores for decision-layer candidates."""

from __future__ import annotations

from typing import Any


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
    try:
        value = float(raw_score)
    except (TypeError, ValueError):
        return 0.0
    if name == "safety":
        return max(0.0, min(1.0, value))
    return max(-1.0, min(1.0, value))


def normalize_delta(before: Any, after: Any) -> float:
    try:
        return max(-1.0, min(1.0, float(after) - float(before)))
    except (TypeError, ValueError):
        return 0.0


def configured_weights(config: dict[str, Any] | None = None) -> dict[str, float]:
    config = dict(config or {})
    critics = dict(config.get("critics", {}) or {})
    weights = dict(DEFAULT_WEIGHTS)
    for key, value in critics.items():
        canonical = ALIASES.get(key, key)
        if isinstance(value, dict) and bool(value.get("enabled", True)):
            weights[canonical] = float(value.get("weight", weights.get(canonical, 0.0)))
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
        values = [
            float(scores[key])
            for key in ("identity_confidence", "bleed_score", "activity_score")
            if isinstance(scores.get(key), (int, float))
        ]
        if values:
            return max(0.0, min(1.0, sum(values) / len(values)))
    if isinstance(scores.get("overall"), (int, float)):
        return normalize_score(critic_name, scores["overall"])
    return None


def _score_source(result: dict[str, Any]) -> str:
    source = str(result.get("score_source", "") or "").strip().lower()
    if source:
        return source
    if result.get("model_available") is True:
        return "real_model"
    if result.get("scores") or result.get("delta"):
        return "proxy"
    return "unavailable"


def aggregate_scores(
    scores: dict[str, dict[str, Any]],
    weights: dict[str, float],
    *,
    proxy_weight_multiplier: float = 0.25,
) -> dict[str, Any]:
    available: dict[str, float] = {}
    values: dict[str, float] = {}
    weight_sources: dict[str, str] = {}
    proxy_weights: dict[str, float] = {}
    excluded: dict[str, str] = {}
    proxy_weight_multiplier = max(0.0, min(1.0, float(proxy_weight_multiplier)))
    for name, weight in weights.items():
        if name == "safety":
            safety = scores.get("safety", {})
            value = safety.get("safety_score")
            if isinstance(value, (int, float)):
                available[name] = weight
                values[name] = normalize_score(name, value)
                weight_sources[name] = "safety"
            continue
        result = scores.get(name, {})
        source = _score_source(result)
        if source == "unavailable":
            excluded[name] = "unavailable"
            continue
        value = extract_signal(name, result)
        if value is not None:
            effective_weight = weight if source == "real_model" else weight * proxy_weight_multiplier
            if effective_weight <= 0.0:
                excluded[name] = f"{source}_weight_zero"
                continue
            available[name] = effective_weight
            values[name] = value
            weight_sources[name] = source
            if source != "real_model":
                proxy_weights[name] = effective_weight
    total = sum(available.values())
    if total <= 0.0:
        return {
            "final_score": 0.0,
            "normalized_weights": {},
            "values": values,
            "weight_sources": weight_sources,
            "proxy_weights": proxy_weights,
            "excluded_weights": excluded,
        }
    normalized = {name: weight / total for name, weight in available.items()}
    final_score = sum(values[name] * normalized[name] for name in normalized)
    return {
        "final_score": float(final_score),
        "normalized_weights": normalized,
        "effective_weights": available,
        "weight_sources": weight_sources,
        "proxy_weights": proxy_weights,
        "excluded_weights": excluded,
        "values": values,
    }
