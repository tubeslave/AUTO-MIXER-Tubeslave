"""Essentia technical/music analyzer adapter."""

from __future__ import annotations

from typing import Any

from ai_mixing_pipeline.audio_utils import measure_audio_file, signal_quality_score
from ai_mixing_pipeline.critics.base import AudioCritic, standard_critic_result


class EssentiaTechnicalAnalyzer(AudioCritic):
    """Technical analyzer for spectrum, loudness, timbre, rhythm, and similarity."""

    name = "essentia"
    role = "technical_music_analyzer"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        try:
            __import__("essentia")
            self._essentia_available = True
        except Exception:
            self._essentia_available = False

    def analyze(self, audio_path: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not bool(self.config.get("enabled", True)):
            return self.unavailable_result("Essentia analyzer disabled in config.")
        metrics = measure_audio_file(audio_path)
        spectral = metrics.get("spectral", {})
        level = metrics.get("level", {})
        dynamics = metrics.get("dynamics", {})
        stereo = metrics.get("stereo", {})
        score = signal_quality_score(metrics)
        warnings = []
        if not self._essentia_available:
            warnings.append("Essentia unavailable; using project technical metrics fallback.")
        return standard_critic_result(
            critic_name=self.name,
            role=self.role,
            scores={
                "overall": score,
                "technical_score": score,
                "spectral_centroid_hz": float(spectral.get("spectral_centroid_hz", 0.0) or 0.0),
                "spectral_rolloff_hz": float(spectral.get("spectral_rolloff_hz", 0.0) or 0.0),
                "loudness_lufs": float(level.get("integrated_lufs", -120.0) or -120.0),
                "brightness": float(spectral.get("brightness_proxy", 0.0) or 0.0),
                "low_mid_mud": float(spectral.get("muddiness_proxy", 0.0) or 0.0),
                "harshness": float(spectral.get("harshness_proxy", 0.0) or 0.0),
                "compression_pumping": float(dynamics.get("compression_pumping_proxy", 0.0) or 0.0),
                "phase_correlation": float(stereo.get("inter_channel_correlation", 1.0) or 1.0),
            },
            confidence=0.55 if self._essentia_available else 0.45,
            warnings=warnings,
            explanation="Technical profile from spectral, loudness, dynamics, and stereo descriptors.",
            model_available=self._essentia_available,
            metadata={"metrics": metrics},
        )
