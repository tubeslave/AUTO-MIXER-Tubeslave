"""PANNs/BEATs identity, event, bleed, and noise detector adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ai_mixing_pipeline.audio_utils import measure_audio_file, read_audio
from ai_mixing_pipeline.critics.base import AudioCritic, standard_critic_result


class IdentityBleedCritic(AudioCritic):
    """Channel identity and bleed/noise detector."""

    name = "panns_or_beats"
    role = "channel_identity_and_bleed_detector"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self._model_available = False
        try:
            __import__("panns_inference")
            self._model_available = False
        except Exception:
            self._model_available = False

    def analyze(self, audio_path: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not bool(self.config.get("enabled", True)):
            return self.unavailable_result("PANNs/BEATs detector disabled in config.")
        context = dict(context or {})
        expected_role = str(context.get("expected_role") or context.get("role") or "").lower()
        channel_name = str(context.get("channel_name") or Path(audio_path).stem)
        inferred = self._infer_role(channel_name)
        metrics = measure_audio_file(audio_path)
        audio, _ = read_audio(audio_path)
        mono = np.mean(audio, axis=1)
        rms = float(np.sqrt(np.mean(mono * mono) + 1e-12)) if mono.size else 0.0
        active_ratio = float(np.mean(np.abs(mono) > max(1e-4, rms * 0.25))) if mono.size else 0.0
        noise_floor = float(metrics.get("level", {}).get("noise_floor_dbfs", -120.0) or -120.0)
        identity = 0.65
        if expected_role:
            identity = 0.9 if inferred == expected_role or expected_role in inferred else 0.35
        elif inferred != "unknown":
            identity = 0.75
        silence_score = 1.0 if rms > 1e-5 and active_ratio > 0.03 else 0.15
        noise_score = max(0.0, min(1.0, (abs(noise_floor) - 30.0) / 50.0))
        bleed_score = 0.85 if inferred != "unknown" else 0.55
        overall = float((identity + silence_score + noise_score + bleed_score) / 4.0)
        return standard_critic_result(
            critic_name=self.name,
            role=self.role,
            scores={
                "overall": overall,
                "identity_confidence": identity,
                "activity_score": silence_score,
                "noise_score": noise_score,
                "bleed_score": bleed_score,
                "channel_silence_risk": 1.0 - silence_score,
            },
            confidence=0.35,
            warnings=["PANNs/BEATs unavailable; using filename and signal-statistics fallback."],
            explanation="Identity fallback checks filename role, activity, noise floor, and silence risk.",
            model_available=False,
            score_source="proxy",
            metadata={
                "inferred_role": inferred,
                "expected_role": expected_role,
                "metrics": metrics,
            },
        )

    def compare(
        self,
        before_path: str,
        after_path: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        before = self.analyze(before_path, context=context)
        after = self.analyze(after_path, context=context)
        warnings = list(before.get("warnings", [])) + list(after.get("warnings", []))
        if not bool(before.get("model_available")) or not bool(after.get("model_available")):
            warnings.append(
                "PANNs/BEATs real model unavailable; identity/bleed comparison is neutral and not penalized."
            )
            return standard_critic_result(
                critic_name=self.name,
                role=self.role,
                scores={
                    "overall": 0.0,
                    "identity_confidence": 0.0,
                    "activity_score": 0.0,
                    "noise_score": 0.0,
                    "bleed_score": 0.0,
                    "channel_silence_risk": 0.0,
                },
                delta={
                    "overall": 0.0,
                    "identity_confidence": 0.0,
                    "activity_score": 0.0,
                    "noise_score": 0.0,
                    "bleed_score": 0.0,
                    "channel_silence_risk": 0.0,
                },
                confidence=0.0,
                warnings=warnings,
                explanation="Neutral fallback: no real identity/bleed model was available for before/after scoring.",
                model_available=False,
                score_source="unavailable",
                metadata={"before": before.get("metadata", {}), "after": after.get("metadata", {})},
            )
        return super().compare(before_path, after_path, context=context)

    @staticmethod
    def _infer_role(name: str) -> str:
        try:
            from mix_agent.analysis.loader import infer_stem_role

            return infer_stem_role(name)
        except Exception:
            label = name.lower()
            for role in ("kick", "snare", "bass", "vocal", "guitar", "drums", "keys"):
                if role in label:
                    return role
            return "unknown"
