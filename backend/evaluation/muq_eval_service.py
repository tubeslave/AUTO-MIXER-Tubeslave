"""MuQ-Eval quality layer for master-mix A/B validation.

The service is deliberately advisory. It never talks to the mixer and never
emits OSC. Callers may use its decision to block or weaken a correction, but
the actual console write must still go through the existing safety controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, float(value))))


def _as_mono(audio: np.ndarray) -> np.ndarray:
    """Return a finite mono float32 signal from mono/stereo/multichannel input."""

    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 0:
        return np.zeros(0, dtype=np.float32)
    if array.ndim == 1:
        mono = array
    elif array.ndim == 2:
        if array.shape[0] <= array.shape[1] and array.shape[0] <= 16:
            mono = np.mean(array, axis=0)
        else:
            mono = np.mean(array, axis=1)
    else:
        mono = array.reshape(-1)
    return np.nan_to_num(mono.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Small dependency-free resampler used before optional MuQ-Eval inference."""

    if source_rate <= 0 or target_rate <= 0 or source_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = audio.size / float(source_rate)
    target_size = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.interp(target_x, source_x, audio).astype(np.float32)


@dataclass
class MuQEvalConfig:
    """Configuration for the optional MuQ-Eval reward layer."""

    enabled: bool = True
    device: str = "auto"
    window_sec: float = 10.0
    hop_sec: float = 5.0
    sample_rate: int = 24000
    min_improvement_threshold: float = 0.03
    rollback_on_quality_drop: bool = True
    fallback_enabled: bool = True
    log_scores: bool = True
    shadow_mode: bool = True
    log_path: str = "logs/muq_eval_decisions.jsonl"
    training_log_path: str = "logs/muq_eval_rewards.jsonl"
    model_repo_id: str = "zhudi2825/MuQ-Eval-A1"
    local_files_only: bool = True
    muq_eval_root: str = ""
    checkpoint_config_path: str = ""
    checkpoint_model_path: str = ""
    min_confidence: float = 0.35
    max_gain_change_db_per_step: float = 1.0
    max_eq_gain_change_db_per_step: float = 1.0
    max_compressor_threshold_change_db_per_step: float = 2.0
    min_seconds_between_quality_decisions: float = 5.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, config: Optional[Dict[str, Any]] = None) -> "MuQEvalConfig":
        """Build config from either a raw section or a `{muq_eval: ...}` mapping."""

        config = dict(config or {})
        if isinstance(config.get("muq_eval"), dict):
            config = dict(config["muq_eval"])
        known = {name for name in cls.__dataclass_fields__ if name != "extra"}
        values = {key: config[key] for key in known if key in config}
        extra = {key: value for key, value in config.items() if key not in known}
        result = cls(**values)
        result.extra = extra
        return result


@dataclass
class MuQEvalResult:
    """Per-window MuQ-Eval or fallback quality result."""

    quality_score: float
    musical_impression: str
    technical_artifacts: Dict[str, Any]
    confidence: float
    timestamp: float
    audio_window_sec: float
    model_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_score": float(self.quality_score),
            "musical_impression": self.musical_impression,
            "technical_artifacts": self.technical_artifacts,
            "confidence": float(self.confidence),
            "timestamp": float(self.timestamp),
            "audio_window_sec": float(self.audio_window_sec),
            "model_status": self.model_status,
        }


@dataclass
class MuQValidationDecision:
    """A/B decision for a proposed correction."""

    timestamp: float
    session_id: str
    current_scene: str
    proposed_action: Dict[str, Any]
    score_before: Optional[MuQEvalResult]
    score_after: Optional[MuQEvalResult]
    delta: float
    accepted: bool
    rejection_reason: str
    confidence: float
    reward: float
    safety_penalty: float = 0.0
    excessive_change_penalty: float = 0.0
    osc_commands: list[Dict[str, Any]] = field(default_factory=list)
    shadow_mode: bool = True

    @property
    def should_block_osc(self) -> bool:
        return (not self.shadow_mode) and (not self.accepted)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "session_id": self.session_id,
            "current_scene": self.current_scene,
            "proposed_action": self.proposed_action,
            "score_before": self.score_before.to_dict() if self.score_before else None,
            "score_after": self.score_after.to_dict() if self.score_after else None,
            "delta": float(self.delta),
            "accepted": bool(self.accepted),
            "rejection_reason": self.rejection_reason,
            "confidence": float(self.confidence),
            "reward": float(self.reward),
            "safety_penalty": float(self.safety_penalty),
            "excessive_change_penalty": float(self.excessive_change_penalty),
            "osc_commands": self.osc_commands if self.accepted else [],
            "shadow_mode": bool(self.shadow_mode),
        }


class MuQEvalService:
    """Evaluate master-mix windows with MuQ-Eval or a deterministic fallback."""

    def __init__(self, config: Optional[Dict[str, Any] | MuQEvalConfig] = None):
        self.config = config if isinstance(config, MuQEvalConfig) else MuQEvalConfig.from_mapping(config)
        self._model: Any = None
        self._torch: Any = None
        self._device: str = "cpu"
        self._model_status: str = "unavailable"
        self._model_error: str = ""
        self._fallback_logged = False
        self._last_quality_decision_at = 0.0

        if self.config.enabled:
            self._try_load_model()

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def model_status(self) -> str:
        return self._model_status

    def evaluate(
        self,
        audio_fragment: np.ndarray,
        sample_rate: int,
        *,
        timestamp: Optional[float] = None,
    ) -> MuQEvalResult:
        """Score a 5-10 second master-mix fragment on a normalized 0..1 scale."""

        timestamp = float(time.time() if timestamp is None else timestamp)
        prepared = self._prepare_audio(audio_fragment, sample_rate)
        window_sec = prepared.size / float(max(1, self.config.sample_rate))

        if not self.enabled:
            return MuQEvalResult(
                quality_score=0.0,
                musical_impression="disabled",
                technical_artifacts={"note": "MuQ-Eval layer disabled"},
                confidence=0.0,
                timestamp=timestamp,
                audio_window_sec=window_sec,
                model_status="disabled",
            )

        if self._model is not None:
            try:
                raw_score = self._score_with_model(prepared)
                normalized = self.normalize_score(raw_score)
                return MuQEvalResult(
                    quality_score=normalized,
                    musical_impression=self._impression_from_score(normalized),
                    technical_artifacts={"muq_raw_mi_score": raw_score},
                    confidence=self._model_confidence(prepared, window_sec),
                    timestamp=timestamp,
                    audio_window_sec=window_sec,
                    model_status="available",
                )
            except Exception as exc:
                self._model_status = "unavailable"
                self._model_error = str(exc)
                logger.warning("MuQ-Eval inference failed; using fallback: %s", exc)

        if not self.config.fallback_enabled:
            return MuQEvalResult(
                quality_score=0.0,
                musical_impression="unavailable",
                technical_artifacts={"model_error": self._model_error or "MuQ-Eval unavailable"},
                confidence=0.0,
                timestamp=timestamp,
                audio_window_sec=window_sec,
                model_status="unavailable",
            )

        if not self._fallback_logged:
            logger.warning("MuQ-Eval unavailable; using fallback quality heuristic")
            self._fallback_logged = True
        score, artifacts, confidence = self._fallback_quality(prepared)
        return MuQEvalResult(
            quality_score=score,
            musical_impression=self._impression_from_artifacts(score, artifacts),
            technical_artifacts=artifacts,
            confidence=confidence,
            timestamp=timestamp,
            audio_window_sec=window_sec,
            model_status="unavailable",
        )

    @staticmethod
    def normalize_score(raw_score: float) -> float:
        """Normalize MuQ-Eval A1 `MI` scores from 1..5 into 0..1."""

        return _clamp((float(raw_score) - 1.0) / 4.0)

    def validate_correction(
        self,
        *,
        before_audio: Optional[np.ndarray],
        sample_rate: int,
        proposed_action: Any,
        after_audio: Optional[np.ndarray] = None,
        session_id: str = "",
        current_scene: str = "",
        osc_commands: Optional[list[Dict[str, Any]]] = None,
        safety_penalty: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> MuQValidationDecision:
        """A/B-score a proposed correction and return an advisory decision."""

        now = float(time.time() if timestamp is None else timestamp)
        action_payload = self._json_safe(proposed_action)
        excessive_penalty = self._excessive_change_penalty(action_payload)

        if before_audio is None:
            return self._decision(
                now=now,
                session_id=session_id,
                current_scene=current_scene,
                proposed_action=action_payload,
                score_before=None,
                score_after=None,
                delta=0.0,
                accepted=False,
                rejection_reason="missing_before_audio",
                confidence=0.0,
                safety_penalty=safety_penalty,
                excessive_change_penalty=excessive_penalty,
                osc_commands=osc_commands,
            )

        min_interval = max(0.0, float(self.config.min_seconds_between_quality_decisions))
        if (
            self._last_quality_decision_at > 0.0
            and now - self._last_quality_decision_at < min_interval
        ):
            before = self.evaluate(before_audio, sample_rate, timestamp=now)
            return self._decision(
                now=now,
                session_id=session_id,
                current_scene=current_scene,
                proposed_action=action_payload,
                score_before=before,
                score_after=None,
                delta=0.0,
                accepted=False,
                rejection_reason="quality_decision_rate_limited",
                confidence=before.confidence,
                safety_penalty=safety_penalty,
                excessive_change_penalty=excessive_penalty,
                osc_commands=osc_commands,
            )

        self._last_quality_decision_at = now
        before = self.evaluate(before_audio, sample_rate, timestamp=now)

        if after_audio is None:
            accepted = bool(self.config.shadow_mode)
            reason = "shadow_mode_no_candidate_audio" if accepted else "candidate_audio_missing"
            return self._decision(
                now=now,
                session_id=session_id,
                current_scene=current_scene,
                proposed_action=action_payload,
                score_before=before,
                score_after=None,
                delta=0.0,
                accepted=accepted,
                rejection_reason=reason,
                confidence=before.confidence,
                safety_penalty=safety_penalty,
                excessive_change_penalty=excessive_penalty,
                osc_commands=osc_commands,
            )

        after = self.evaluate(after_audio, sample_rate, timestamp=now)
        delta = float(after.quality_score - before.quality_score)
        confidence = min(float(before.confidence), float(after.confidence))
        reward = delta - float(safety_penalty) - excessive_penalty

        accepted = True
        reason = ""
        if confidence < float(self.config.min_confidence):
            accepted = False
            reason = "quality_estimate_unstable"
        elif excessive_penalty > 0.0:
            accepted = False
            reason = "excessive_change"
        elif delta < 0.0 and self.config.rollback_on_quality_drop:
            accepted = False
            reason = "quality_drop"
        elif delta < float(self.config.min_improvement_threshold):
            accepted = False
            reason = "below_improvement_threshold"

        decision = self._decision(
            now=now,
            session_id=session_id,
            current_scene=current_scene,
            proposed_action=action_payload,
            score_before=before,
            score_after=after,
            delta=delta,
            accepted=accepted,
            rejection_reason=reason,
            confidence=confidence,
            safety_penalty=safety_penalty,
            excessive_change_penalty=excessive_penalty,
            osc_commands=osc_commands,
            reward=reward,
        )
        return decision

    def log_decision(self, decision: MuQValidationDecision) -> None:
        """Append a validation decision and reward row to JSONL logs."""

        if not self.config.log_scores:
            return
        record = decision.to_dict()
        self._write_jsonl(Path(self.config.log_path), record)
        self._write_jsonl(
            Path(self.config.training_log_path),
            {
                "timestamp": record["timestamp"],
                "session_id": record["session_id"],
                "proposed_action": record["proposed_action"],
                "delta_quality_score": record["delta"],
                "safety_penalty": record["safety_penalty"],
                "excessive_change_penalty": record["excessive_change_penalty"],
                "reward": record["reward"],
                "accepted": record["accepted"],
                "model_status": (
                    record.get("score_after") or record.get("score_before") or {}
                ).get("model_status"),
            },
        )

    def _try_load_model(self) -> None:
        root = self.config.muq_eval_root or os.environ.get("MUQ_EVAL_ROOT", "")
        if root:
            root_path = Path(root).expanduser().resolve()
            if str(root_path) not in sys.path:
                sys.path.insert(0, str(root_path))

        try:
            import torch
            from omegaconf import OmegaConf
            from src.model import MusicQualityModel
        except Exception as exc:
            self._model_status = "unavailable"
            self._model_error = str(exc)
            logger.info("MuQ-Eval model code unavailable: %s", exc)
            return

        try:
            config_path, model_path = self._resolve_checkpoint_paths()
            if not config_path or not model_path:
                raise RuntimeError("checkpoint files are not available locally")
            self._torch = torch
            self._device = self._resolve_device(torch)
            model_config = OmegaConf.load(str(config_path))
            model = MusicQualityModel(model_config)
            state = torch.load(str(model_path), map_location=self._device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            model.to(self._device)
            model.eval()
            self._model = model
            self._model_status = "available"
            logger.info("MuQ-Eval loaded: checkpoint=%s device=%s", model_path, self._device)
        except Exception as exc:
            self._model = None
            self._model_status = "unavailable"
            self._model_error = str(exc)
            logger.info("MuQ-Eval checkpoint unavailable: %s", exc)

    def _resolve_checkpoint_paths(self) -> tuple[Optional[Path], Optional[Path]]:
        config_path = Path(self.config.checkpoint_config_path).expanduser() if self.config.checkpoint_config_path else None
        model_path = Path(self.config.checkpoint_model_path).expanduser() if self.config.checkpoint_model_path else None
        if config_path and model_path and config_path.exists() and model_path.exists():
            return config_path, model_path

        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            logger.debug("huggingface_hub unavailable for MuQ-Eval cache lookup: %s", exc)
            return None, None

        try:
            cfg = Path(
                hf_hub_download(
                    repo_id=self.config.model_repo_id,
                    filename="config.yaml",
                    local_files_only=bool(self.config.local_files_only),
                )
            )
            model = Path(
                hf_hub_download(
                    repo_id=self.config.model_repo_id,
                    filename="model_state_dict.pt",
                    local_files_only=bool(self.config.local_files_only),
                )
            )
            return cfg, model
        except Exception as exc:
            logger.debug("MuQ-Eval checkpoint cache lookup failed: %s", exc)
            return None, None

    def _resolve_device(self, torch_module: Any) -> str:
        requested = str(self.config.device or "auto").strip().lower()
        if requested and requested != "auto":
            return requested
        if torch_module.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(torch_module, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _prepare_audio(self, audio_fragment: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = _as_mono(audio_fragment)
        audio = _resample_linear(audio, int(sample_rate), int(self.config.sample_rate))
        max_samples = int(max(1.0, self.config.window_sec) * self.config.sample_rate)
        if audio.size > max_samples:
            audio = audio[-max_samples:]
        if audio.size == 0:
            return np.zeros(0, dtype=np.float32)
        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            audio = audio / peak
        return audio.astype(np.float32, copy=False)

    def _score_with_model(self, prepared_audio: np.ndarray) -> float:
        if self._model is None or self._torch is None:
            raise RuntimeError("MuQ-Eval model is not loaded")
        tensor = self._torch.as_tensor(prepared_audio, dtype=self._torch.float32, device=self._device)
        with self._torch.inference_mode():
            output = self._model(tensor.unsqueeze(0))
        if isinstance(output, dict):
            value = output.get("MI")
        else:
            value = getattr(output, "MI", None)
        if value is None:
            raise RuntimeError("MuQ-Eval output did not include MI score")
        if hasattr(value, "detach"):
            value = value.detach().float().mean().cpu().item()
        return float(value)

    def _fallback_quality(self, audio: np.ndarray) -> tuple[float, Dict[str, Any], float]:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        rms = float(np.sqrt(np.mean(audio * audio) + 1e-12)) if audio.size else 0.0
        peak_db = 20.0 * math.log10(max(peak, 1e-10))
        rms_db = 20.0 * math.log10(max(rms, 1e-10))
        crest_db = peak_db - rms_db if rms > 1e-9 else 0.0
        clipping_ratio = float(np.mean(np.abs(audio) >= 0.99)) if audio.size else 0.0
        active_ratio = float(np.mean(np.abs(audio) > max(1e-4, rms * 0.1))) if audio.size else 0.0

        band = self._band_energy_ratios(audio)
        low_ratio = band.get("low_end_ratio", 0.0)
        harsh_ratio = band.get("harshness_ratio", 0.0)
        imbalance = band.get("spectral_imbalance", 0.0)

        clipping_penalty = min(0.45, clipping_ratio * 25.0 + max(0.0, peak - 0.98) * 4.0)
        low_penalty = min(0.25, max(0.0, low_ratio - 0.48) * 0.9)
        harsh_penalty = min(0.25, max(0.0, harsh_ratio - 0.32) * 0.9)
        imbalance_penalty = min(0.25, max(0.0, imbalance - 0.35) * 0.65)
        crest_penalty = min(0.25, abs(crest_db - 12.0) / 48.0)
        loudness_penalty = min(0.25, max(0.0, rms_db + 8.0) / 24.0 + max(0.0, -38.0 - rms_db) / 48.0)
        inactivity_penalty = min(0.25, max(0.0, 0.08 - active_ratio) * 2.0)

        total_penalty = (
            clipping_penalty
            + low_penalty
            + harsh_penalty
            + imbalance_penalty
            + crest_penalty
            + loudness_penalty
            + inactivity_penalty
        )
        score = _clamp(1.0 - total_penalty)
        confidence = _clamp(
            0.25
            + min(0.25, audio.size / max(1.0, self.config.window_sec * self.config.sample_rate) * 0.25)
            + min(0.25, active_ratio * 0.4)
            + (0.15 if score > 0.15 else 0.0)
        )
        artifacts = {
            "fallback": True,
            "peak_db": peak_db,
            "rms_db": rms_db,
            "crest_factor_db": crest_db,
            "approx_lufs": rms_db,
            "active_ratio": active_ratio,
            "clipping_ratio": clipping_ratio,
            "low_end_ratio": low_ratio,
            "harshness_2_5khz_ratio": harsh_ratio,
            "spectral_imbalance": imbalance,
            "clipping_penalty": clipping_penalty,
            "excessive_low_end_penalty": low_penalty,
            "harshness_penalty": harsh_penalty,
            "spectral_imbalance_penalty": imbalance_penalty,
            "crest_factor_penalty": crest_penalty,
            "lufs_rms_penalty": loudness_penalty,
            "inactivity_penalty": inactivity_penalty,
            "model_error": self._model_error,
        }
        return score, artifacts, confidence

    def _band_energy_ratios(self, audio: np.ndarray) -> Dict[str, float]:
        if audio.size < 128:
            return {"low_end_ratio": 0.0, "harshness_ratio": 0.0, "spectral_imbalance": 1.0}
        fft_size = min(8192, max(512, 2 ** int(math.floor(math.log2(audio.size)))))
        window = np.hanning(fft_size).astype(np.float32)
        frame = audio[-fft_size:] * window
        spectrum = np.abs(np.fft.rfft(frame)).astype(np.float64)
        power = spectrum * spectrum + 1e-12
        freqs = np.fft.rfftfreq(fft_size, 1.0 / float(self.config.sample_rate))
        total = float(np.sum(power)) + 1e-12

        def ratio(low_hz: float, high_hz: float) -> float:
            mask = (freqs >= low_hz) & (freqs < high_hz)
            if not np.any(mask):
                return 0.0
            return float(np.sum(power[mask]) / total)

        edges = np.array([40, 80, 160, 320, 640, 1250, 2500, 5000, 10000, 12000], dtype=float)
        bands = []
        for low, high in zip(edges[:-1], edges[1:]):
            bands.append(ratio(float(low), float(high)))
        band_values = np.asarray(bands, dtype=np.float64)
        imbalance = float(np.std(np.log10(band_values + 1e-9)))
        return {
            "low_end_ratio": ratio(20.0, 160.0),
            "harshness_ratio": ratio(2000.0, 5000.0),
            "spectral_imbalance": imbalance,
        }

    def _model_confidence(self, audio: np.ndarray, window_sec: float) -> float:
        active = float(np.mean(np.abs(audio) > 1e-4)) if audio.size else 0.0
        duration_score = min(1.0, window_sec / max(1.0, self.config.window_sec))
        return _clamp(0.45 + 0.25 * active + 0.25 * duration_score)

    def _excessive_change_penalty(self, action_payload: Dict[str, Any]) -> float:
        action_type = str(
            action_payload.get("action_type")
            or action_payload.get("type")
            or action_payload.get("__class__")
            or ""
        ).lower()
        delta = self._extract_action_delta(action_payload)
        if delta <= 0.0:
            return 0.0
        if "eq" in action_type:
            limit = self.config.max_eq_gain_change_db_per_step
        elif "compressor" in action_type:
            limit = self.config.max_compressor_threshold_change_db_per_step
        else:
            limit = self.config.max_gain_change_db_per_step
        excess = max(0.0, delta - max(0.0, float(limit)))
        return _clamp(excess / max(1.0, float(limit)), 0.0, 1.0)

    def _extract_action_delta(self, payload: Dict[str, Any]) -> float:
        candidates = []
        for key in (
            "delta_db",
            "gain_change_db",
            "eq_gain_change_db",
            "threshold_change_db",
        ):
            if key in payload:
                candidates.append(abs(float(payload.get(key) or 0.0)))
        previous = payload.get("previous_state")
        if isinstance(previous, dict):
            for key in ("target_db", "gain_db", "threshold_db"):
                if key in payload and key in previous:
                    candidates.append(abs(float(payload[key]) - float(previous[key])))
        return max(candidates) if candidates else 0.0

    def _decision(
        self,
        *,
        now: float,
        session_id: str,
        current_scene: str,
        proposed_action: Dict[str, Any],
        score_before: Optional[MuQEvalResult],
        score_after: Optional[MuQEvalResult],
        delta: float,
        accepted: bool,
        rejection_reason: str,
        confidence: float,
        safety_penalty: float,
        excessive_change_penalty: float,
        osc_commands: Optional[list[Dict[str, Any]]],
        reward: Optional[float] = None,
    ) -> MuQValidationDecision:
        if reward is None:
            reward = float(delta) - float(safety_penalty) - float(excessive_change_penalty)
        decision = MuQValidationDecision(
            timestamp=now,
            session_id=session_id,
            current_scene=current_scene,
            proposed_action=proposed_action,
            score_before=score_before,
            score_after=score_after,
            delta=float(delta),
            accepted=bool(accepted),
            rejection_reason=rejection_reason,
            confidence=float(confidence),
            reward=float(reward),
            safety_penalty=float(safety_penalty),
            excessive_change_penalty=float(excessive_change_penalty),
            osc_commands=list(osc_commands or []),
            shadow_mode=bool(self.config.shadow_mode),
        )
        self.log_decision(decision)
        return decision

    @staticmethod
    def _impression_from_score(score: float) -> str:
        if score >= 0.78:
            return "strong musical quality"
        if score >= 0.58:
            return "usable musical quality"
        if score >= 0.35:
            return "questionable musical quality"
        return "poor musical quality"

    @staticmethod
    def _impression_from_artifacts(score: float, artifacts: Dict[str, Any]) -> str:
        if artifacts.get("clipping_penalty", 0.0) > 0.1:
            return "clipping risk"
        if artifacts.get("excessive_low_end_penalty", 0.0) > 0.08:
            return "excessive low-end"
        if artifacts.get("harshness_penalty", 0.0) > 0.08:
            return "harsh upper-mid content"
        return MuQEvalService._impression_from_score(score)

    @staticmethod
    def _write_jsonl(path: Path, payload: Dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(MuQEvalService._json_safe(payload), ensure_ascii=True) + "\n")
        except Exception as exc:
            logger.warning("MuQ-Eval JSONL logging failed for %s: %s", path, exc)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return {
                "shape": list(value.shape),
                "mean": float(np.mean(value)) if value.size else 0.0,
                "std": float(np.std(value)) if value.size else 0.0,
            }
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if is_dataclass(value):
            payload = {"action_type": value.__class__.__name__}
            payload.update(value.__dict__)
            return MuQEvalService._json_safe(payload)
        if isinstance(value, dict):
            return {str(key): MuQEvalService._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [MuQEvalService._json_safe(item) for item in value]
        if hasattr(value, "__dict__"):
            payload = {"action_type": value.__class__.__name__}
            payload.update(value.__dict__)
            return MuQEvalService._json_safe(payload)
        return str(value)
