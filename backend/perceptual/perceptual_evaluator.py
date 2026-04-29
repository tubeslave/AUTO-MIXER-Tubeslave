"""Perceptual shadow evaluator based on audio embeddings."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, is_dataclass
import json
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

try:
    from output_paths import ai_logs_path
except ImportError:  # pragma: no cover - package import fallback
    from backend.output_paths import ai_logs_path

from .embedding_backend import create_embedding_backend
from .metrics import cosine_distance, embedding_mse, fad_like_distance
from .reference_store import ReferenceStore
from .reward import RewardSignal

logger = logging.getLogger(__name__)


@dataclass
class PerceptualConfig:
    enabled: bool = False
    mode: str = "shadow"
    backend: str = "lightweight"
    model_name: str = "m-a-p/MERT-v1-95M"
    sample_rate: int = 24000
    window_seconds: float = 5.0
    hop_seconds: float = 2.0
    evaluate_channels: bool = True
    evaluate_mix_bus: bool = True
    max_cpu_percent: float = 25.0
    log_scores: bool = True
    block_osc_when_score_worse: bool = False
    log_path: str = str(ai_logs_path("perceptual_decisions.jsonl"))
    queue_maxsize: int = 128
    async_evaluation: bool = True
    improvement_threshold: float = 0.03
    min_confidence_for_verdict: float = 0.2
    local_files_only: bool = False
    fallback_to_lightweight: bool = True
    device: str = "auto"
    reference_embedding_cache_size: int = 8
    max_candidate_scores: int = 2
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, config: Optional[Dict[str, Any]] = None) -> "PerceptualConfig":
        config = dict(config or {})
        if isinstance(config.get("perceptual"), dict):
            config = dict(config["perceptual"])
        known = {field_name for field_name in cls.__dataclass_fields__ if field_name != "extra"}
        values = {key: config[key] for key in known if key in config}
        if "decision_log_path" in config and "log_path" not in values:
            values["log_path"] = config["decision_log_path"]
        extra = {key: value for key, value in config.items() if key not in known}
        result = cls(**values)
        result.extra = extra
        return result

    def backend_config(self) -> Dict[str, Any]:
        payload = dict(self.extra)
        payload.update(
            {
                "backend": self.backend,
                "model_name": self.model_name,
                "sample_rate": self.sample_rate,
                "window_seconds": self.window_seconds,
                "local_files_only": self.local_files_only,
                "fallback_to_lightweight": self.fallback_to_lightweight,
                "device": self.device,
            }
        )
        return payload


@dataclass
class PerceptualEvaluationResult:
    timestamp: float
    channel: Optional[str]
    instrument: Optional[str]
    action: Dict[str, Any]
    backend: str
    score_before: float
    score_after: float
    perceptual_score: float
    delta_score: float
    mse: float
    cosine_distance: float
    verdict: str
    confidence: float
    reward_signal: RewardSignal
    osc_sent: bool
    features_before: Dict[str, Any] = field(default_factory=dict)
    features_after: Dict[str, Any] = field(default_factory=dict)
    reference_used: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "channel": self.channel,
            "instrument": self.instrument,
            "action": self.action,
            "backend": self.backend,
            "score_before": float(self.score_before),
            "score_after": float(self.score_after),
            "perceptual_score": float(self.perceptual_score),
            "delta_score": float(self.delta_score),
            "mse": float(self.mse),
            "cosine_distance": float(self.cosine_distance),
            "verdict": self.verdict,
            "confidence": float(self.confidence),
            "features_before": self.features_before,
            "features_after": self.features_after,
            "reward_signal": self.reward_signal.to_dict(),
            "reference_used": bool(self.reference_used),
            "osc_sent": bool(self.osc_sent),
            "notes": self.notes,
        }


class PerceptualEvaluator:
    """Embedding-based evaluator for shadow/offline mix-quality experiments."""

    def __init__(self, config: Optional[Dict[str, Any] | PerceptualConfig] = None):
        self.config = config if isinstance(config, PerceptualConfig) else PerceptualConfig.from_mapping(config)
        self.backend = create_embedding_backend(self.config.backend_config())
        self.reference_store = ReferenceStore(self.config.extra.get("reference_store_path"))
        self.log_path = Path(self.config.log_path).expanduser()
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(self.config.queue_maxsize)))
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._reference_embedding_cache: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()

        if self.config.block_osc_when_score_worse:
            logger.warning(
                "perceptual.block_osc_when_score_worse is ignored in shadow mode; OSC behavior is unchanged"
            )

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def mode(self) -> str:
        return str(self.config.mode)

    def start(self) -> None:
        if not self.enabled or not self.config.async_evaluation:
            return
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._worker_thread is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._worker_thread.join(timeout=timeout)
            self._worker_thread = None

    def extract_embedding(
        self,
        audio_buffer: np.ndarray,
        sample_rate: int,
        channel_name: Optional[str] = None,
        instrument_type: Optional[str] = None,
    ) -> np.ndarray:
        return self.backend.extract(
            audio_buffer,
            sample_rate,
            channel_name=channel_name,
            instrument_type=instrument_type,
        )

    def compare_embeddings(
        self,
        before_embedding: np.ndarray,
        after_embedding: np.ndarray,
    ) -> Dict[str, float]:
        mse = embedding_mse(before_embedding, after_embedding)
        cosine = cosine_distance(before_embedding, after_embedding)
        return {
            "embedding_mse": mse,
            "cosine_distance": cosine,
        }

    def score_change(
        self,
        before_audio: np.ndarray,
        after_audio: np.ndarray,
        sample_rate: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> PerceptualEvaluationResult:
        context = dict(context or {})
        channel = context.get("channel")
        instrument = context.get("instrument") or context.get("instrument_type")
        action = self._action_payload(context.get("action", {}))
        osc_sent = bool(context.get("osc_sent", True))

        before_embedding = self.extract_embedding(before_audio, sample_rate, channel, instrument)
        after_embedding = self.extract_embedding(after_audio, sample_rate, channel, instrument)
        metrics = self.compare_embeddings(before_embedding, after_embedding)

        reference_embedding = self._resolve_reference_embedding(context, channel, instrument)
        reference_used = reference_embedding is not None
        if reference_embedding is not None:
            before_distance = self._combined_embedding_distance(before_embedding, reference_embedding)
            after_distance = self._combined_embedding_distance(after_embedding, reference_embedding)
            score_before = 1.0 / (1.0 + before_distance)
            score_after = 1.0 / (1.0 + after_distance)
            notes = "scored against reference embedding"
        else:
            score_before = self._audio_quality_score(before_audio)
            score_after = self._audio_quality_score(after_audio)
            notes = "no reference embedding; using lightweight signal-quality proxy"

        delta_score = float(score_after - score_before)
        confidence = self._confidence_score(
            before_audio=before_audio,
            after_audio=after_audio,
            delta_score=delta_score,
            reference_used=reference_used,
            sample_rate=sample_rate,
        )
        verdict = self._verdict(delta_score, confidence)
        reward_signal = RewardSignal.combine(
            engineering_score=float(context.get("engineering_score", 0.0)),
            perceptual_score=delta_score,
            safety_score=float(context.get("safety_score", 1.0 if osc_sent else 0.0)),
            weights=context.get("reward_weights"),
        )

        return PerceptualEvaluationResult(
            timestamp=float(context.get("timestamp", time.time())),
            channel=str(channel) if channel is not None else None,
            instrument=str(instrument) if instrument is not None else None,
            action=action,
            backend=self.backend.name,
            score_before=float(score_before),
            score_after=float(score_after),
            perceptual_score=delta_score,
            delta_score=delta_score,
            mse=float(metrics["embedding_mse"]),
            cosine_distance=float(metrics["cosine_distance"]),
            verdict=verdict,
            confidence=confidence,
            reward_signal=reward_signal,
            osc_sent=osc_sent,
            features_before=self._feature_summary(before_audio, before_embedding),
            features_after=self._feature_summary(after_audio, after_embedding),
            reference_used=reference_used,
            notes=notes,
        )

    def evaluate_mix_snapshot(
        self,
        mix_audio: np.ndarray,
        stems: Optional[Dict[str, np.ndarray]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = dict(context or {})
        sample_rate = int(context.get("sample_rate", self.config.sample_rate))
        mix_embedding = self.extract_embedding(
            mix_audio,
            sample_rate,
            channel_name="mix_bus",
            instrument_type="mix",
        )
        payload: Dict[str, Any] = {
            "timestamp": time.time(),
            "backend": self.backend.name,
            "mix": self._feature_summary(mix_audio, mix_embedding),
            "quality_score": self._audio_quality_score(mix_audio),
        }

        if stems:
            payload["stems"] = {}
            for stem_name, stem_audio in stems.items():
                stem_embedding = self.extract_embedding(
                    stem_audio,
                    sample_rate,
                    channel_name=str(stem_name),
                    instrument_type=str(stem_name),
                )
                payload["stems"][str(stem_name)] = self._feature_summary(stem_audio, stem_embedding)

        reference_embedding = self._resolve_reference_embedding(context, "mix_bus", "mix")
        if reference_embedding is not None:
            payload["reference_distance"] = self._combined_embedding_distance(
                mix_embedding,
                reference_embedding,
            )
        return payload

    def score_candidate_batch(
        self,
        before_audio: np.ndarray,
        candidate_audios: list[np.ndarray],
        sample_rate: int,
        contexts: Optional[list[Dict[str, Any]]] = None,
        prefilter_scores: Optional[list[float]] = None,
    ) -> list[Dict[str, Any]]:
        """Score a bounded subset of candidates for offline selection.

        Heavy embedding backends should not be run over every speculative mix
        variant. When `prefilter_scores` are supplied, only the top
        `max_candidate_scores` candidates are evaluated.
        """

        contexts = list(contexts or [{} for _ in candidate_audios])
        if len(contexts) < len(candidate_audios):
            contexts.extend({} for _ in range(len(candidate_audios) - len(contexts)))

        indexes = list(range(len(candidate_audios)))
        max_scores = max(1, int(self.config.max_candidate_scores))
        if prefilter_scores is not None and len(prefilter_scores) == len(candidate_audios):
            indexes.sort(key=lambda idx: float(prefilter_scores[idx]), reverse=True)
            indexes = indexes[:max_scores]

        results: list[Dict[str, Any]] = []
        for index in indexes:
            result = self.score_change(
                before_audio,
                candidate_audios[index],
                sample_rate,
                context=contexts[index],
            )
            results.append({"index": index, "result": result})
        return results

    def record_shadow_decision(
        self,
        before_audio: np.ndarray,
        after_audio: np.ndarray,
        sample_rate: int,
        context: Optional[Dict[str, Any]] = None,
        osc_sent: bool = True,
    ) -> Optional[PerceptualEvaluationResult]:
        if not self.enabled:
            return None
        context = dict(context or {})
        context["osc_sent"] = bool(osc_sent)
        try:
            result = self.score_change(before_audio, after_audio, sample_rate, context=context)
            if self.config.log_scores:
                self.log_decision(result)
            logger.info(
                "Perceptual shadow: channel=%s instrument=%s verdict=%s delta=%.4f "
                "mse=%.4f cosine=%.4f confidence=%.2f",
                result.channel,
                result.instrument,
                result.verdict,
                result.delta_score,
                result.mse,
                result.cosine_distance,
                result.confidence,
            )
            return result
        except Exception as exc:
            logger.warning("Perceptual shadow evaluation failed: %s", exc)
            return None

    def submit_shadow_evaluation(
        self,
        before_audio: np.ndarray,
        after_audio: np.ndarray,
        sample_rate: int,
        context: Optional[Dict[str, Any]] = None,
        osc_sent: bool = True,
    ) -> bool:
        if not self.enabled:
            return False
        payload = (
            np.asarray(before_audio, dtype=np.float32).copy(),
            np.asarray(after_audio, dtype=np.float32).copy(),
            int(sample_rate),
            dict(context or {}),
            bool(osc_sent),
        )
        if not self.config.async_evaluation:
            self.record_shadow_decision(*payload)
            return True
        self.start()
        try:
            self._queue.put_nowait(payload)
            return True
        except queue.Full:
            logger.warning("Perceptual shadow queue full; dropping evaluation")
            return False

    def log_decision(self, result: PerceptualEvaluationResult) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = result.to_dict()
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._json_safe(record), ensure_ascii=True) + "\n")

    def _worker(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                continue
            before_audio, after_audio, sample_rate, context, osc_sent = item
            self.record_shadow_decision(
                before_audio,
                after_audio,
                sample_rate,
                context=context,
                osc_sent=osc_sent,
            )

    def _resolve_reference_embedding(
        self,
        context: Dict[str, Any],
        channel: Optional[str],
        instrument: Optional[str],
    ) -> Optional[np.ndarray]:
        if "reference_embedding" in context and context["reference_embedding"] is not None:
            return np.asarray(context["reference_embedding"], dtype=np.float32).reshape(-1)
        if "reference_audio" in context and context["reference_audio"] is not None:
            reference_audio = np.asarray(context["reference_audio"], dtype=np.float32)
            reference_sr = int(
                context.get(
                    "reference_sample_rate",
                    context.get("sample_rate", self.config.sample_rate),
                )
            )
            cache_key = self._reference_audio_cache_key(
                reference_audio,
                reference_sr,
                channel,
                instrument,
                context,
            )
            cached = self._reference_embedding_cache.get(cache_key)
            if cached is not None:
                self._reference_embedding_cache.move_to_end(cache_key)
                return cached.copy()
            embedding = self.extract_embedding(
                reference_audio,
                reference_sr,
                channel_name=channel,
                instrument_type=instrument,
            )
            self._store_reference_embedding(cache_key, embedding)
            return embedding
        context_id = context.get("reference_context") or context.get("context_id")
        return self.reference_store.get_for_context(channel, instrument, context_id)

    def _reference_audio_cache_key(
        self,
        reference_audio: np.ndarray,
        sample_rate: int,
        channel: Optional[str],
        instrument: Optional[str],
        context: Dict[str, Any],
    ) -> tuple[Any, ...]:
        explicit_key = context.get("reference_cache_key")
        if explicit_key is not None:
            source_key = ("explicit", str(explicit_key))
        else:
            source_key = (
                "array",
                id(reference_audio),
                tuple(reference_audio.shape),
                str(reference_audio.dtype),
            )
        return (
            self.backend.name,
            self.config.model_name,
            int(sample_rate),
            str(channel or ""),
            str(instrument or ""),
            source_key,
        )

    def _store_reference_embedding(self, key: tuple[Any, ...], embedding: np.ndarray) -> None:
        max_items = max(0, int(self.config.reference_embedding_cache_size))
        if max_items <= 0:
            return
        self._reference_embedding_cache[key] = np.asarray(embedding, dtype=np.float32).copy()
        self._reference_embedding_cache.move_to_end(key)
        while len(self._reference_embedding_cache) > max_items:
            self._reference_embedding_cache.popitem(last=False)

    def _combined_embedding_distance(self, first: np.ndarray, second: np.ndarray) -> float:
        return 0.5 * embedding_mse(first, second) + 0.5 * cosine_distance(first, second)

    def _confidence_score(
        self,
        before_audio: np.ndarray,
        after_audio: np.ndarray,
        delta_score: float,
        reference_used: bool,
        sample_rate: int,
    ) -> float:
        before = np.asarray(before_audio, dtype=np.float32).reshape(-1)
        after = np.asarray(after_audio, dtype=np.float32).reshape(-1)
        duration_sec = min(before.size, after.size) / float(max(1, int(sample_rate)))
        active = float(
            min(
                1.0,
                0.5 * (np.mean(np.abs(before) > 1e-4) + np.mean(np.abs(after) > 1e-4)),
            )
        ) if before.size and after.size else 0.0
        backend_bonus = 0.25 if self.backend.name == "mert" else 0.1
        reference_bonus = 0.25 if reference_used else 0.0
        duration_bonus = min(0.25, duration_sec / max(1.0, self.config.window_seconds) * 0.25)
        delta_bonus = min(0.15, abs(float(delta_score)) * 2.0)
        confidence = 0.15 + backend_bonus + reference_bonus + duration_bonus + delta_bonus
        confidence *= max(0.2, active)
        return float(max(0.0, min(1.0, confidence)))

    def _verdict(self, delta_score: float, confidence: float) -> str:
        if confidence < self.config.min_confidence_for_verdict:
            return "neutral"
        threshold = float(self.config.improvement_threshold)
        if delta_score > threshold:
            return "improved"
        if delta_score < -threshold:
            return "worse"
        return "neutral"

    def _audio_quality_score(self, audio_buffer: np.ndarray) -> float:
        audio = np.asarray(audio_buffer, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return 0.0
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
        rms_db = 20.0 * np.log10(rms + 1e-10)
        peak_db = 20.0 * np.log10(peak + 1e-10)
        crest_db = peak_db - rms_db if rms > 1e-9 else 0.0
        clipping_ratio = float(np.mean(np.abs(audio) >= 0.99))
        active_ratio = float(np.mean(np.abs(audio) > 1e-4))

        level_score = 1.0 - min(1.0, abs(rms_db + 20.0) / 50.0)
        crest_score = 1.0 - min(1.0, abs(crest_db - 12.0) / 30.0)
        headroom_score = 1.0 - min(1.0, max(0.0, peak - 0.95) * 10.0 + clipping_ratio * 20.0)
        activity_score = min(1.0, active_ratio * 2.0)
        return float(max(0.0, min(1.0, 0.35 * level_score + 0.25 * crest_score + 0.25 * headroom_score + 0.15 * activity_score)))

    def _feature_summary(self, audio_buffer: np.ndarray, embedding: np.ndarray) -> Dict[str, Any]:
        audio = np.asarray(audio_buffer, dtype=np.float32).reshape(-1)
        if audio.size:
            peak = float(np.max(np.abs(audio)))
            rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
        else:
            peak = 0.0
            rms = 0.0
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        return {
            "audio_samples": int(audio.size),
            "audio_rms": float(rms),
            "audio_peak": float(peak),
            "audio_quality_proxy": self._audio_quality_score(audio),
            "embedding_dim": int(emb.size),
            "embedding_mean": float(np.mean(emb)) if emb.size else 0.0,
            "embedding_std": float(np.std(emb)) if emb.size else 0.0,
            "embedding_norm": float(np.linalg.norm(emb)) if emb.size else 0.0,
        }

    def _action_payload(self, action: Any) -> Dict[str, Any]:
        if action is None:
            return {}
        if isinstance(action, dict):
            return self._json_safe(action)
        if is_dataclass(action):
            return self._json_safe(action.__dict__)
        if hasattr(action, "__dict__"):
            return self._json_safe(action.__dict__)
        return {"value": str(action)}

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
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
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(item) for item in value]
        if is_dataclass(value):
            return self._json_safe(value.__dict__)
        if hasattr(value, "__dict__"):
            return self._json_safe(value.__dict__)
        return str(value)

    @staticmethod
    def fad_like_distance(reference_embeddings: np.ndarray, candidate_embeddings: np.ndarray) -> float:
        return fad_like_distance(reference_embeddings, candidate_embeddings)
