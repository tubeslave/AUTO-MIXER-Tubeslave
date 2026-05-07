"""Audiobox Aesthetics critic adapter with graceful fallback."""

from __future__ import annotations

from typing import Any

from ai_mixing_pipeline.audio_utils import measure_audio_file, signal_quality_score

from .base import AudioCritic, standard_critic_result


class AudioboxAestheticsCritic(AudioCritic):
    """Secondary independent quality/aesthetic critic."""

    name = "audiobox_aesthetics"
    role = "secondary_quality_critic"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})

    def analyze(self, audio_path: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not bool(self.config.get("enabled", True)):
            return self.unavailable_result("Audiobox Aesthetics critic disabled in config.")
        metrics = measure_audio_file(audio_path)
        base = signal_quality_score(metrics)
        spectral = metrics.get("spectral", {})
        level = metrics.get("level", {})
        mud = float(spectral.get("muddiness_proxy", 0.0) or 0.0)
        harsh = float(spectral.get("harshness_proxy", 0.0) or 0.0)
        headroom = float(level.get("headroom_db", 0.0) or 0.0)
        pleasantness = max(0.0, min(1.0, base - 0.35 * max(0.0, harsh - 0.18)))
        cleanliness = max(0.0, min(1.0, base - 0.35 * max(0.0, mud - 0.16)))
        naturalness = max(0.0, min(1.0, base - 0.04 * max(0.0, 1.0 - headroom)))
        overall = float((pleasantness + cleanliness + naturalness) / 3.0)
        return standard_critic_result(
            critic_name=self.name,
            role=self.role,
            scores={
                "overall": overall,
                "pleasantness": pleasantness,
                "cleanliness": cleanliness,
                "naturalness": naturalness,
            },
            confidence=0.35,
            warnings=["Audiobox Aesthetics model unavailable; using deterministic aesthetic proxy."],
            explanation="Aesthetic fallback checks headroom, clipping, harshness, and low-mid mud.",
            model_available=False,
            metadata={"metrics": metrics},
        )
