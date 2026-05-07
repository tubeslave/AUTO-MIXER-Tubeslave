"""MERT embedding critic adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ai_mixing_pipeline.audio_utils import measure_audio, read_audio, signal_quality_score
from ai_mixing_pipeline.critics.base import AudioCritic, standard_critic_result


class MERTStemCritic(AudioCritic):
    """Stem embedding critic with optional MERT and lightweight fallback."""

    name = "mert"
    role = "stem_embedding_critic"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self._backend = None
        self._backend_error = ""
        if not bool(self.config.get("enabled", True)):
            return
        try:
            try:
                from backend.perceptual.embedding_backend import create_embedding_backend
            except Exception:
                from perceptual.embedding_backend import create_embedding_backend

            backend_config = {
                "backend": self.config.get("backend", "mert"),
                "model_name": self.config.get("model_name", "m-a-p/MERT-v1-95M"),
                "sample_rate": int(self.config.get("sample_rate", 24000)),
                "window_seconds": float(self.config.get("max_window_seconds", self.config.get("window_seconds", 5.0))),
                "local_files_only": bool(self.config.get("local_files_only", True)),
                "fallback_to_lightweight": bool(self.config.get("fallback_to_lightweight", True)),
                "device": self.config.get("device", "auto"),
            }
            self._backend = create_embedding_backend(backend_config)
        except Exception as exc:
            self._backend_error = str(exc)

    def analyze(self, audio_path: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not bool(self.config.get("enabled", True)):
            return self.unavailable_result("MERT critic disabled in config.")
        context = dict(context or {})
        audio, sample_rate = read_audio(audio_path)
        metrics = measure_audio(audio, sample_rate)
        quality = signal_quality_score(metrics)
        warnings: list[str] = []
        model_available = False
        embedding = np.zeros(1, dtype=np.float32)
        backend_name = "unavailable"
        if self._backend is not None:
            try:
                embedding = self._backend.extract(
                    audio,
                    sample_rate,
                    channel_name=context.get("channel_name"),
                    instrument_type=context.get("instrument_type") or context.get("role"),
                )
                backend_name = getattr(self._backend, "name", "embedding")
                model_available = backend_name == "mert"
                if not model_available:
                    warnings.append("MERT model unavailable; lightweight embedding fallback used.")
            except Exception as exc:
                warnings.append(f"MERT embedding failed; fallback vector used: {exc}")
        else:
            warnings.append(f"MERT backend unavailable: {self._backend_error}")

        if context.get("embedding_dir"):
            output = Path(context["embedding_dir"]).expanduser()
            output.mkdir(parents=True, exist_ok=True)
            np.save(output / f"{Path(audio_path).stem}.mert_embedding.npy", embedding)

        norm = float(np.linalg.norm(embedding)) if embedding.size else 0.0
        return standard_critic_result(
            critic_name=self.name,
            role=self.role,
            scores={
                "overall": quality,
                "stem_embedding_quality": quality,
                "embedding_norm": norm,
                "vocal_clarity": self._vocal_clarity_proxy(metrics),
                "kick_punch": self._kick_punch_proxy(metrics),
                "bass_definition": self._bass_definition_proxy(metrics),
                "mix_cleanliness": quality,
                "live_mix_readiness": quality,
            },
            confidence=0.45 if model_available else 0.30,
            warnings=warnings,
            explanation="MERT adapter stores/compares embeddings; custom trained heads are placeholders.",
            model_available=model_available,
            metadata={
                "backend": backend_name,
                "embedding_size": int(embedding.size),
                "metrics": metrics,
            },
        )

    def compare(
        self,
        before_path: str,
        after_path: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = dict(context or {})
        before = self.analyze(before_path, context=context)
        after = self.analyze(after_path, context=context)
        delta = {}
        for key, value in after.get("scores", {}).items():
            if isinstance(value, (int, float)) and key in before.get("scores", {}):
                delta[key] = float(value) - float(before["scores"][key])
        delta.setdefault("overall", delta.get("stem_embedding_quality", 0.0))
        warnings = list(before.get("warnings", [])) + list(after.get("warnings", []))
        return standard_critic_result(
            critic_name=self.name,
            role=self.role,
            scores=after.get("scores", {}),
            delta=delta,
            confidence=min(float(before.get("confidence", 0.0)), float(after.get("confidence", 0.0))),
            warnings=warnings,
            explanation="Compared MERT/custom-head proxy scores before and after.",
            model_available=bool(before.get("model_available")) and bool(after.get("model_available")),
            metadata={"before": before.get("metadata", {}), "after": after.get("metadata", {})},
        )

    @staticmethod
    def _vocal_clarity_proxy(metrics: dict[str, Any]) -> float:
        spectral = metrics.get("spectral", {})
        mud = float(spectral.get("muddiness_proxy", 0.0) or 0.0)
        harsh = float(spectral.get("harshness_proxy", 0.0) or 0.0)
        presence = float((spectral.get("band_energy_ratios", {}) or {}).get("presence", 0.0) or 0.0)
        return float(max(0.0, min(1.0, 0.65 + presence - mud * 1.2 - harsh * 0.4)))

    @staticmethod
    def _kick_punch_proxy(metrics: dict[str, Any]) -> float:
        dynamics = metrics.get("dynamics", {})
        level = metrics.get("level", {})
        transient = float(dynamics.get("transient_strength_db", 0.0) or 0.0)
        crest = float(level.get("crest_factor_db", 0.0) or 0.0)
        return float(max(0.0, min(1.0, transient / 12.0 + crest / 30.0)))

    @staticmethod
    def _bass_definition_proxy(metrics: dict[str, Any]) -> float:
        spectral = metrics.get("spectral", {})
        boom = float(spectral.get("boominess_proxy", 0.0) or 0.0)
        mud = float(spectral.get("muddiness_proxy", 0.0) or 0.0)
        return float(max(0.0, min(1.0, 1.0 - abs(boom - 0.28) * 1.5 - mud * 0.5)))
