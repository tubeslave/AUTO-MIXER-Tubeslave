"""Bridge decision-layer renders to existing audio critics."""

from __future__ import annotations

from typing import Any

from ai_mixing_pipeline.critics import AudioboxAestheticsCritic, CLAPSemanticCritic, MuQEvalCritic
from ai_mixing_pipeline.stem_critics import MERTStemCritic
from ai_mixing_pipeline.technical_analyzers import EssentiaTechnicalAnalyzer, IdentityBleedCritic


class CriticBridge:
    """Call available critics and return graceful fallback payloads."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        critics = self.config.get("critics", {}) or {}
        self.critics = [
            MuQEvalCritic(critics.get("muq_eval", {})),
            AudioboxAestheticsCritic(critics.get("audiobox_aesthetics", critics.get("audiobox", {}))),
            MERTStemCritic(critics.get("mert", critics.get("stem_critics", {}))),
            CLAPSemanticCritic(critics.get("clap", {})),
            EssentiaTechnicalAnalyzer(critics.get("essentia", {})),
            IdentityBleedCritic(critics.get("panns_or_beats", critics.get("identity_bleed", {}))),
        ]
        self.critics = [critic for critic in self.critics if bool(getattr(critic, "config", {}).get("enabled", True))]

    def evaluate_render(self, audio_path: str, context: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for critic in self.critics:
            try:
                results[critic.name] = critic.analyze(audio_path, context=context)
            except Exception as exc:
                results[critic.name] = critic.unavailable_result(str(exc))
        return results

    def compare(self, before_path: str, after_path: str, context: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for critic in self.critics:
            try:
                results[critic.name] = critic.compare(before_path, after_path, context=context)
            except Exception as exc:
                results[critic.name] = critic.unavailable_result(str(exc))
        return results

    @staticmethod
    def module_status(evaluations: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
        status: dict[str, Any] = {}
        for by_critic in evaluations.values():
            for name, result in by_critic.items():
                item = status.setdefault(name, {"available": False, "participated": True, "warnings": [], "role": result.get("role", "")})
                item["available"] = bool(item["available"] or result.get("model_available", False))
                for warning in result.get("warnings", []):
                    if warning and warning not in item["warnings"]:
                        item["warnings"].append(warning)
        return status
