"""CLAP / LAION-CLAP semantic critic adapter with fallback scoring."""

from __future__ import annotations

from typing import Any

from ai_mixing_pipeline.audio_utils import measure_audio_file

from .base import AudioCritic, standard_critic_result

NEGATIVE_PROMPT_TOKENS = ("muddy", "boomy", "harsh", "overcompressed", "noisy", "feedback")


class CLAPSemanticCritic(AudioCritic):
    """Audio-text semantic critic using prompts as an advisory signal."""

    name = "clap"
    role = "semantic_audio_text_critic"

    DEFAULT_PROMPTS = [
        "clean lead vocal",
        "balanced live rock mix",
        "punchy kick drum",
        "defined bass guitar",
        "muddy low mids",
        "harsh guitar",
        "overcompressed mix",
    ]

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.prompts = list(self.config.get("prompts") or self.DEFAULT_PROMPTS)
        self._model_available = False
        try:
            __import__("laion_clap")
            self._model_available = False
        except Exception:
            self._model_available = False

    def analyze(self, audio_path: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not bool(self.config.get("enabled", True)):
            return self.unavailable_result("CLAP critic disabled in config.")
        metrics = measure_audio_file(audio_path)
        prompt_scores = {
            prompt: self._fallback_prompt_score(prompt, metrics)
            for prompt in self.prompts
        }
        weighted = []
        for prompt, score in prompt_scores.items():
            if any(token in prompt.lower() for token in NEGATIVE_PROMPT_TOKENS):
                weighted.append(1.0 - score)
            else:
                weighted.append(score)
        overall = float(sum(weighted) / max(1, len(weighted)))
        return standard_critic_result(
            critic_name=self.name,
            role=self.role,
            scores={
                "overall": overall,
                "semantic_alignment": overall,
                "prompt_similarity": prompt_scores,
            },
            confidence=0.30,
            warnings=["CLAP/LAION-CLAP unavailable; using prompt-specific technical proxy."],
            explanation="Semantic fallback maps prompt descriptors to spectral, level, and dynamics proxies.",
            model_available=False,
            metadata={"metrics": metrics},
        )

    @staticmethod
    def _fallback_prompt_score(prompt: str, metrics: dict[str, Any]) -> float:
        prompt = prompt.lower()
        spectral = metrics.get("spectral", {})
        level = metrics.get("level", {})
        dynamics = metrics.get("dynamics", {})
        stereo = metrics.get("stereo", {})
        mud = float(spectral.get("muddiness_proxy", 0.0) or 0.0)
        harsh = float(spectral.get("harshness_proxy", 0.0) or 0.0)
        boom = float(spectral.get("boominess_proxy", 0.0) or 0.0)
        bright = float(spectral.get("brightness_proxy", 0.0) or 0.0)
        crest = float(level.get("crest_factor_db", 0.0) or 0.0)
        headroom = float(level.get("headroom_db", 0.0) or 0.0)
        pumping = float(dynamics.get("compression_pumping_proxy", 0.0) or 0.0)
        width = float(stereo.get("stereo_width", 0.0) or 0.0)

        clean = max(0.0, min(1.0, 1.0 - mud * 1.4 - harsh * 0.8))
        punch = max(0.0, min(1.0, crest / 18.0))
        balanced = max(0.0, min(1.0, 0.8 * clean + 0.2 * min(headroom / 6.0, 1.0)))
        if "clean" in prompt or "clear" in prompt:
            return clean
        if "muddy" in prompt or "low mid" in prompt:
            return max(0.0, min(1.0, mud * 3.0))
        if "punchy" in prompt or "kick" in prompt:
            return punch
        if "defined bass" in prompt:
            return max(0.0, min(1.0, 1.0 - abs(boom - 0.28) * 2.0))
        if "boomy" in prompt:
            return max(0.0, min(1.0, boom * 2.0))
        if "harsh" in prompt:
            return max(0.0, min(1.0, harsh * 3.0 + bright * 0.4))
        if "overcompressed" in prompt:
            return max(0.0, min(1.0, pumping / 5.0 + max(0.0, 8.0 - crest) / 12.0))
        if "wide" in prompt:
            return max(0.0, min(1.0, width * 2.0))
        if "balanced" in prompt:
            return balanced
        return balanced
