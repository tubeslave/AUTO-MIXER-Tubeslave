"""MuQ-Eval critic adapter."""

from __future__ import annotations

from typing import Any

from ai_mixing_pipeline.audio_utils import read_audio, signal_quality_score, measure_audio

from .base import AudioCritic, standard_critic_result


class MuQEvalCritic(AudioCritic):
    """Chief music critic using MuQ-Eval when available."""

    name = "muq_eval"
    role = "chief_music_critic"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self._service = None
        self._service_error = ""
        if not bool(self.config.get("enabled", True)):
            return
        try:
            try:
                from backend.evaluation.muq_eval_service import MuQEvalService
            except Exception:
                from evaluation import MuQEvalService

            service_config = {
                "enabled": True,
                "fallback_enabled": True,
                "log_scores": False,
                "window_sec": float(self.config.get("max_window_seconds", self.config.get("window_sec", 10))),
                "sample_rate": int(self.config.get("sample_rate", 24000)),
                "local_files_only": bool(self.config.get("local_files_only", True)),
                "model_repo_id": self.config.get("model_repo_id", "zhudi2825/MuQ-Eval-A1"),
                "device": self.config.get("device", self.config.get("device_preference", ["cpu"])[0] if isinstance(self.config.get("device_preference"), list) else "auto"),
                "min_seconds_between_quality_decisions": 0,
            }
            self._service = MuQEvalService(service_config)
        except Exception as exc:
            self._service_error = str(exc)

    def analyze(self, audio_path: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not bool(self.config.get("enabled", True)):
            return self.unavailable_result("MuQ-Eval critic disabled in config.")

        audio, sample_rate = read_audio(audio_path)
        if self._service is not None:
            try:
                result = self._service.evaluate(audio, sample_rate)
                warnings = []
                if result.model_status != "available":
                    warnings.append("MuQ-Eval model unavailable; fallback quality heuristic used.")
                return standard_critic_result(
                    critic_name=self.name,
                    role=self.role,
                    scores={
                        "overall": float(result.quality_score),
                        "quality_score": float(result.quality_score),
                        "muq_quality_score": float(result.quality_score),
                    },
                    delta={},
                    confidence=float(result.confidence),
                    warnings=warnings,
                    explanation=result.musical_impression,
                    model_available=result.model_status == "available",
                    metadata={"technical_artifacts": result.technical_artifacts},
                )
            except Exception as exc:
                self._service_error = str(exc)

        metrics = measure_audio(audio, sample_rate)
        score = signal_quality_score(metrics)
        return standard_critic_result(
            critic_name=self.name,
            role=self.role,
            scores={"overall": score, "quality_score": score, "muq_quality_score": score},
            delta={},
            confidence=0.25,
            warnings=[
                "MuQ-Eval service unavailable; using local technical fallback.",
                self._service_error,
            ],
            explanation="Fallback proxy estimates musical quality from headroom, clipping, tonal balance, and dynamics.",
            model_available=False,
            metadata={"metrics": metrics},
        )
