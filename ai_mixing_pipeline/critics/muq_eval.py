"""MuQ-Eval critic adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
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
                "muq_eval_root": self.config.get("muq_eval_root", self._default_muq_eval_root()),
                "device": self.config.get("device", self.config.get("device_preference", ["cpu"])[0] if isinstance(self.config.get("device_preference"), list) else "auto"),
                "min_seconds_between_quality_decisions": 0,
            }
            self._service_config = service_config
            self._service = MuQEvalService(service_config)
        except Exception as exc:
            self._service_error = str(exc)
            self._service_config = {}

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
                    score_source="real_model" if result.model_status == "available" else "proxy",
                    metadata={
                        "technical_artifacts": result.technical_artifacts,
                        "diagnostics": self.diagnostics(inference_attempted=True, model_status=result.model_status),
                    },
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
            score_source="proxy",
            metadata={"metrics": metrics, "diagnostics": self.diagnostics(inference_attempted=False)},
        )

    def diagnostics(self, *, inference_attempted: bool = False, model_status: str | None = None) -> dict[str, Any]:
        """Return load/inference diagnostics without failing the offline pipeline."""

        configured = dict(getattr(self, "_service_config", {}) or {})
        service = self._service
        checkpoint_config = None
        checkpoint_model = None
        checkpoint_error = ""
        if service is not None and hasattr(service, "_resolve_checkpoint_paths"):
            try:
                cfg, model = service._resolve_checkpoint_paths()  # noqa: SLF001 - diagnostic wrapper
                checkpoint_config = str(cfg) if cfg else None
                checkpoint_model = str(model) if model else None
            except Exception as exc:
                checkpoint_error = str(exc)
        imports = {
            "torch": importlib.util.find_spec("torch") is not None,
            "omegaconf": importlib.util.find_spec("omegaconf") is not None,
            "huggingface_hub": importlib.util.find_spec("huggingface_hub") is not None,
            "muq_eval_src_model": importlib.util.find_spec("src.model") is not None,
        }
        return {
            "model_status": model_status or getattr(service, "model_status", "unavailable") if service else "unavailable",
            "model_error": getattr(service, "_model_error", "") if service else self._service_error,
            "service_error": self._service_error,
            "model_repo_id": configured.get("model_repo_id", self.config.get("model_repo_id", "zhudi2825/MuQ-Eval-A1")),
            "local_files_only": bool(configured.get("local_files_only", self.config.get("local_files_only", True))),
            "configured_sample_rate": int(configured.get("sample_rate", self.config.get("sample_rate", 24000))),
            "configured_window_sec": float(configured.get("window_sec", self.config.get("max_window_seconds", 10))),
            "device_requested": configured.get("device", self.config.get("device", "auto")),
            "device_resolved": getattr(service, "_device", "unavailable") if service else "unavailable",
            "checkpoint_config_path": checkpoint_config,
            "checkpoint_model_path": checkpoint_model,
            "checkpoint_error": checkpoint_error,
            "imports": imports,
            "inference_attempted": bool(inference_attempted),
        }

    @staticmethod
    def _default_muq_eval_root() -> str:
        root = Path(__file__).resolve().parents[2] / "external" / "MuQ-Eval"
        return str(root) if (root / "src" / "model.py").exists() else ""
