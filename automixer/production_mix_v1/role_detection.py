"""Role detection with confidence gates and analyzer sanity evidence."""

from __future__ import annotations

from fnmatch import fnmatchcase
import re
from typing import Any, Mapping

import numpy as np

from .analyzers import analyze_audio, analyzer_issues
from .config import ProductionMixConfig
from .models import (
    ROLE_ACTION_NO_OP,
    ROLE_ACTION_ROLE_SPECIFIC,
    ROLE_ACTION_UNIVERSAL_SAFE,
    RoleDetectionResult,
)


def detect_role(
    *,
    filename: str,
    channel_name: str,
    audio: np.ndarray,
    sample_rate: int,
    config: ProductionMixConfig,
    overrides: Mapping[str, Any] | None = None,
) -> RoleDetectionResult:
    tokens = _tokens(filename or channel_name)
    metrics = analyze_audio(audio, sample_rate, config)
    override = _override_for(filename=filename, channel_name=channel_name, overrides=overrides or {})
    if override:
        role = str(override.get("role", "unknown"))
        confidence = float(override.get("confidence", 1.0))
        allowed = str(override.get("allowed_action_level", _allowed(role, confidence, False, config)))
        return _result(role, confidence, metrics, tokens, False, allowed, override)

    scores = _token_scores(tokens)
    if not scores:
        role = "unknown"
        confidence = 0.45
        ambiguous = True
    else:
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        role, confidence = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        ambiguous = confidence < 0.85 or confidence - second < 0.12
        if ambiguous:
            confidence = min(confidence, 0.84)
    allowed = _allowed(role, confidence, ambiguous, config)
    if analyzer_issues(metrics, role=role):
        allowed = ROLE_ACTION_UNIVERSAL_SAFE if confidence >= 0.60 else ROLE_ACTION_NO_OP
        ambiguous = True
    return _result(role, confidence, metrics, tokens, ambiguous, allowed, None, token_scores=scores)


def _result(
    role: str,
    confidence: float,
    metrics: Mapping[str, Any],
    tokens: list[str],
    ambiguous: bool,
    allowed: str,
    override: Mapping[str, Any] | None,
    *,
    token_scores: Mapping[str, float] | None = None,
) -> RoleDetectionResult:
    evidence = {
        "filename_tokens": tokens,
        "spectral_features": {
            "rms_db": round(float(metrics.get("rms_db", -120.0)), 3),
            "crest_factor_db": round(float(metrics.get("crest_factor_db", 0.0)), 3),
            "spectral_centroid_hz": round(float(metrics.get("spectral_centroid_hz", 0.0)), 1),
            "low_ratio": round(float(metrics.get("band_ratios", {}).get("sub", 0.0) + metrics.get("band_ratios", {}).get("punch", 0.0)), 4),
            "mid_ratio": round(float(metrics.get("band_ratios", {}).get("mud", 0.0) + metrics.get("band_ratios", {}).get("boxiness", 0.0) + metrics.get("band_ratios", {}).get("intelligibility", 0.0) + metrics.get("band_ratios", {}).get("presence", 0.0)), 4),
            "high_ratio": round(float(metrics.get("band_ratios", {}).get("harshness", 0.0) + metrics.get("band_ratios", {}).get("air", 0.0) + metrics.get("band_ratios", {}).get("polish", 0.0)), 4),
            "analyzer_issues": analyzer_issues(metrics, role=role),
        },
        "user_override": dict(override) if override else None,
    }
    if token_scores:
        evidence["token_scores"] = dict(token_scores)
    return RoleDetectionResult(role=role, confidence=float(confidence), evidence=evidence, is_ambiguous=bool(ambiguous), allowed_action_level=allowed)


def _allowed(role: str, confidence: float, ambiguous: bool, config: ProductionMixConfig) -> str:
    if role in {"playback", "playback_music", "music_stem"} and not bool(config.role_detection.get("playback_role_specific_enabled", False)):
        return ROLE_ACTION_UNIVERSAL_SAFE
    if confidence >= float(config.role_detection.get("role_specific_confidence", 0.85)) and not ambiguous:
        return ROLE_ACTION_ROLE_SPECIFIC
    if confidence >= float(config.role_detection.get("universal_safe_confidence", 0.60)):
        return ROLE_ACTION_UNIVERSAL_SAFE
    return ROLE_ACTION_NO_OP


def _tokens(name: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", name.lower()) if token]


def _token_scores(tokens: list[str]) -> dict[str, float]:
    token_set = set(tokens)
    scores: dict[str, float] = {}
    def add(role: str, score: float) -> None:
        scores[role] = max(scores.get(role, 0.0), score)
    if "playback" in token_set or "track" in token_set or "tracks" in token_set:
        add("music_stem", 0.74)
    if "room" in token_set:
        add("drum_room", 0.94)
    if "oh" in token_set or "overhead" in token_set or "overheads" in token_set:
        add("overheads", 0.95)
    if "ride" in token_set:
        add("ride", 0.96)
    if "hat" in token_set or "hihat" in token_set:
        add("hihat", 0.96)
    if "back" in token_set or "backs" in token_set or "bgv" in token_set:
        add("backing_vocal", 0.93)
    if "vox" in token_set or "vocal" in token_set or "lead" in token_set:
        add("lead_vocal", 0.88)
    if "kick" in token_set:
        add("kick", 0.97)
    if "snare" in token_set:
        add("snare", 0.96)
    if "tom" in token_set or "toms" in token_set:
        add("toms", 0.92)
    if "bass" in token_set:
        add("bass", 0.96)
    if "guitar" in token_set or "gtr" in token_set:
        add("electric_guitar", 0.92)
    if "accordion" in token_set or "keys" in token_set or "piano" in token_set:
        add("keys", 0.90)
    return scores


def _override_for(*, filename: str, channel_name: str, overrides: Mapping[str, Any]) -> dict[str, Any] | None:
    patterns = dict(overrides.get("patterns", overrides) or {})
    for pattern, payload in patterns.items():
        if any(_matches(str(pattern), value) for value in (filename, channel_name)):
            if isinstance(payload, str):
                return {"role": payload, "confidence": 1.0, "pattern": pattern}
            result = dict(payload or {})
            result.setdefault("confidence", 1.0)
            result.setdefault("pattern", pattern)
            return result
    return None


def _matches(pattern: str, value: str) -> bool:
    return pattern == value or fnmatchcase(value.lower(), pattern.lower())
