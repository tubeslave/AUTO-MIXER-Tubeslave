"""Role detection with confidence gates for `production_mix_v1`."""

from __future__ import annotations

from fnmatch import fnmatchcase
import re
from typing import Any, Mapping

import numpy as np

from .models import (
    ROLE_ACTION_NO_OP,
    ROLE_ACTION_ROLE_SPECIFIC,
    ROLE_ACTION_UNIVERSAL_SAFE,
    RoleDetectionResult,
    amp_to_db,
)


ROLE_SPECIFIC_THRESHOLD = 0.85
UNIVERSAL_SAFE_THRESHOLD = 0.60


def detect_role(
    *,
    filename: str,
    channel_name: str,
    audio: np.ndarray,
    sample_rate: int,
    overrides: Mapping[str, Any] | None = None,
) -> RoleDetectionResult:
    """Classify a source role from override, filename tokens, and light features."""
    tokens = _filename_tokens(filename or channel_name)
    spectral = spectral_features(audio, sample_rate)
    override = _override_for(filename=filename, channel_name=channel_name, overrides=overrides or {})
    if override is not None:
        role = str(override.get("role", "unknown"))
        confidence = float(override.get("confidence", 1.0))
        return _result(
            role=role,
            confidence=confidence,
            evidence={
                "filename_tokens": tokens,
                "spectral_features": spectral,
                "user_override": override,
            },
            is_ambiguous=False,
            forced_allowed_action_level=override.get("allowed_action_level"),
        )

    scored = _token_scores(tokens)
    if not scored:
        guessed_role = _spectral_hint(spectral)
        confidence = 0.52 if guessed_role != "unknown" else 0.45
        return _result(
            role=guessed_role,
            confidence=confidence,
            evidence={
                "filename_tokens": tokens,
                "spectral_features": spectral,
                "user_override": None,
                "reason": "no_strong_filename_token",
            },
            is_ambiguous=True,
        )

    ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
    role, confidence = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    is_ambiguous = bool(confidence < ROLE_SPECIFIC_THRESHOLD or confidence - second < 0.12)
    if is_ambiguous:
        confidence = min(confidence, 0.84)
    return _result(
        role=role,
        confidence=confidence,
        evidence={
            "filename_tokens": tokens,
            "spectral_features": spectral,
            "user_override": None,
            "token_scores": scored,
        },
        is_ambiguous=is_ambiguous,
    )


def spectral_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Return cheap, deterministic spectral evidence for reports and review."""
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if mono.size == 0:
        return {
            "rms_db": -120.0,
            "crest_factor_db": 0.0,
            "spectral_centroid_hz": 0.0,
            "low_ratio": 0.0,
            "mid_ratio": 0.0,
            "high_ratio": 0.0,
        }
    window = mono
    rms = float(np.sqrt(np.mean(window * window) + 1e-12))
    peak = float(np.max(np.abs(window)) + 1e-12)
    fft_size = int(2 ** np.ceil(np.log2(max(2048, min(65536, window.size)))))
    spectral_block = window[-fft_size:] if window.size >= fft_size else np.pad(window, (0, fft_size - window.size))
    spectrum = np.abs(np.fft.rfft(spectral_block * np.hanning(spectral_block.size))) ** 2
    freqs = np.fft.rfftfreq(spectral_block.size, 1.0 / float(sample_rate))
    total = float(np.sum(spectrum) + 1e-12)
    centroid = float(np.sum(freqs * spectrum) / total)
    return {
        "rms_db": round(amp_to_db(rms), 3),
        "crest_factor_db": round(amp_to_db(peak) - amp_to_db(rms), 3),
        "spectral_centroid_hz": round(centroid, 1),
        "low_ratio": round(float(np.sum(spectrum[(freqs >= 40) & (freqs < 180)]) / total), 4),
        "mid_ratio": round(float(np.sum(spectrum[(freqs >= 180) & (freqs < 3000)]) / total), 4),
        "high_ratio": round(float(np.sum(spectrum[(freqs >= 3000) & (freqs < 16000)]) / total), 4),
    }


def _result(
    *,
    role: str,
    confidence: float,
    evidence: dict[str, Any],
    is_ambiguous: bool,
    forced_allowed_action_level: Any | None = None,
) -> RoleDetectionResult:
    confidence = max(0.0, min(1.0, float(confidence)))
    if forced_allowed_action_level in {
        ROLE_ACTION_NO_OP,
        ROLE_ACTION_UNIVERSAL_SAFE,
        ROLE_ACTION_ROLE_SPECIFIC,
    }:
        allowed = str(forced_allowed_action_level)
    elif role in {"playback", "playback_music", "music_stem"}:
        allowed = ROLE_ACTION_UNIVERSAL_SAFE
    elif confidence >= ROLE_SPECIFIC_THRESHOLD and not is_ambiguous:
        allowed = ROLE_ACTION_ROLE_SPECIFIC
    elif confidence >= UNIVERSAL_SAFE_THRESHOLD:
        allowed = ROLE_ACTION_UNIVERSAL_SAFE
    else:
        allowed = ROLE_ACTION_NO_OP
    return RoleDetectionResult(
        role=role,
        confidence=confidence,
        evidence=evidence,
        is_ambiguous=bool(is_ambiguous),
        allowed_action_level=allowed,
    )


def _filename_tokens(name: str) -> list[str]:
    return [
        token
        for token in re.split(r"[^a-z0-9]+", name.lower())
        if token
    ]


def _token_scores(tokens: list[str]) -> dict[str, float]:
    token_set = set(tokens)
    text = " ".join(tokens)
    scores: dict[str, float] = {}

    def add(role: str, score: float) -> None:
        scores[role] = max(scores.get(role, 0.0), score)

    if "playback" in token_set or "tracks" in token_set or "track" in token_set:
        add("music_stem", 0.74)
    if "room" in token_set or ("dr" in token_set and "room" in text):
        add("drum_room", 0.94)
    if "ride" in token_set:
        add("ride", 0.96)
    if "hihat" in token_set or ("hi" in token_set and "hat" in token_set) or "hat" in token_set:
        add("hihat", 0.96)
    if "oh" in token_set or "overhead" in token_set or "overheads" in token_set:
        add("overheads", 0.95)
    if "backs" in token_set or "back" in token_set or "bgv" in token_set:
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
    if "music" in token_set or "stem" in token_set:
        add("music_stem", 0.74)
    return scores


def _spectral_hint(features: Mapping[str, float]) -> str:
    low = float(features.get("low_ratio", 0.0))
    high = float(features.get("high_ratio", 0.0))
    centroid = float(features.get("spectral_centroid_hz", 0.0))
    if low > 0.55 and centroid < 180.0:
        return "low_frequency_source"
    if high > 0.55 and centroid > 4500.0:
        return "bright_percussion_or_noise"
    return "unknown"


def _override_for(
    *,
    filename: str,
    channel_name: str,
    overrides: Mapping[str, Any],
) -> dict[str, Any] | None:
    patterns = dict(overrides.get("patterns", overrides) or {})
    candidates = [filename, channel_name]
    for pattern, payload in patterns.items():
        if any(_matches(str(pattern), candidate) for candidate in candidates):
            if isinstance(payload, str):
                return {"role": payload, "confidence": 1.0, "pattern": pattern}
            result = dict(payload or {})
            result.setdefault("confidence", 1.0)
            result.setdefault("pattern", pattern)
            return result
    return None


def _matches(pattern: str, value: str) -> bool:
    return pattern == value or fnmatchcase(value.lower(), pattern.lower())
