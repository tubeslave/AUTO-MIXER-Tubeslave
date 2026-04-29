"""Audio embedding backends for perceptual shadow evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _to_mono(audio_buffer: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio_buffer, dtype=np.float32)
    if audio.ndim == 0:
        return np.zeros(0, dtype=np.float32)
    if audio.ndim == 1:
        mono = audio
    elif audio.ndim == 2:
        if audio.shape[0] <= audio.shape[1] and audio.shape[0] <= 16:
            mono = np.mean(audio, axis=0)
        else:
            mono = np.mean(audio, axis=1)
    else:
        mono = audio.reshape(-1)
    mono = np.nan_to_num(mono.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return mono


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0 or source_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = audio.size / float(source_rate)
    target_size = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.interp(target_x, source_x, audio).astype(np.float32)


@dataclass
class EmbeddingBackend:
    """Abstract embedding backend."""

    name: str
    sample_rate: int = 24000

    def extract(
        self,
        audio_buffer: np.ndarray,
        sample_rate: int,
        channel_name: Optional[str] = None,
        instrument_type: Optional[str] = None,
    ) -> np.ndarray:
        raise NotImplementedError


class LightweightEmbeddingBackend(EmbeddingBackend):
    """Fast deterministic embedding from numpy spectral and temporal features."""

    def __init__(
        self,
        sample_rate: int = 24000,
        window_seconds: float = 5.0,
        fft_size: int = 2048,
        hop_size: int = 512,
        band_count: int = 24,
    ):
        super().__init__(name="lightweight", sample_rate=sample_rate)
        self.window_seconds = float(window_seconds)
        self.fft_size = int(max(256, fft_size))
        self.hop_size = int(max(64, hop_size))
        self.band_count = int(max(8, band_count))

    def extract(
        self,
        audio_buffer: np.ndarray,
        sample_rate: int,
        channel_name: Optional[str] = None,
        instrument_type: Optional[str] = None,
    ) -> np.ndarray:
        audio = _to_mono(audio_buffer)
        audio = _resample_linear(audio, int(sample_rate), self.sample_rate)
        max_samples = int(max(1.0, self.window_seconds) * self.sample_rate)
        if audio.size > max_samples:
            audio = audio[-max_samples:]
        if audio.size < self.fft_size:
            audio = np.pad(audio, (0, self.fft_size - audio.size))

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        rms = float(np.sqrt(np.mean(audio * audio) + 1e-12)) if audio.size else 0.0
        peak_db = 20.0 * np.log10(peak + 1e-10)
        rms_db = 20.0 * np.log10(rms + 1e-10)
        crest_db = peak_db - rms_db if rms > 1e-9 else 0.0
        zero_crossing_rate = float(np.mean(np.abs(np.diff(np.signbit(audio))))) if audio.size > 1 else 0.0
        clipping_ratio = float(np.mean(np.abs(audio) >= 0.99)) if audio.size else 0.0
        active_ratio = float(np.mean(np.abs(audio) > max(1e-4, rms * 0.1))) if audio.size else 0.0

        window = np.hanning(self.fft_size).astype(np.float32)
        frames = []
        last_start = max(0, audio.size - self.fft_size)
        for start in range(0, last_start + 1, self.hop_size):
            frame = audio[start:start + self.fft_size]
            if frame.size < self.fft_size:
                frame = np.pad(frame, (0, self.fft_size - frame.size))
            frames.append(frame * window)
        if not frames:
            frames.append(audio[-self.fft_size:] * window)

        spectrum = np.abs(np.fft.rfft(np.stack(frames, axis=0), axis=1)).astype(np.float64)
        power = spectrum * spectrum + 1e-12
        freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)
        total_power = np.sum(power, axis=1, keepdims=True) + 1e-12

        centroid = np.sum(power * freqs, axis=1, keepdims=True) / total_power
        bandwidth = np.sqrt(
            np.sum(power * (freqs.reshape(1, -1) - centroid) ** 2, axis=1, keepdims=True)
            / total_power
        )
        cumulative = np.cumsum(power, axis=1)
        rolloff_idx = np.argmax(cumulative >= 0.85 * total_power, axis=1)
        rolloff = freqs[np.clip(rolloff_idx, 0, freqs.size - 1)]
        flatness = np.exp(np.mean(np.log(power), axis=1)) / (np.mean(power, axis=1) + 1e-12)

        band_edges = np.geomspace(20.0, max(40.0, self.sample_rate / 2.0), self.band_count + 1)
        band_features = []
        for low, high in zip(band_edges[:-1], band_edges[1:]):
            mask = (freqs >= low) & (freqs < high)
            if not np.any(mask):
                band_energy = np.zeros(power.shape[0], dtype=np.float64)
            else:
                band_energy = np.mean(power[:, mask], axis=1)
            band_db = 10.0 * np.log10(band_energy + 1e-12)
            band_features.append(np.clip((band_db + 100.0) / 100.0, 0.0, 1.5))
        band_matrix = np.stack(band_features, axis=1)

        envelope_frame = np.sqrt(np.mean(np.stack(frames, axis=0) ** 2, axis=1) + 1e-12)
        envelope_db = 20.0 * np.log10(envelope_frame + 1e-10)

        global_features = np.array(
            [
                np.clip((rms_db + 100.0) / 100.0, 0.0, 1.5),
                np.clip((peak_db + 100.0) / 100.0, 0.0, 1.5),
                np.clip(crest_db / 40.0, 0.0, 1.5),
                np.clip(zero_crossing_rate, 0.0, 1.0),
                np.clip(float(np.mean(centroid)) / (self.sample_rate / 2.0), 0.0, 1.0),
                np.clip(float(np.mean(rolloff)) / (self.sample_rate / 2.0), 0.0, 1.0),
                np.clip(float(np.mean(bandwidth)) / (self.sample_rate / 2.0), 0.0, 1.0),
                np.clip(float(np.mean(flatness)), 0.0, 1.0),
                np.clip(active_ratio, 0.0, 1.0),
                np.clip(clipping_ratio * 10.0, 0.0, 1.0),
                np.clip(float(np.std(envelope_db)) / 40.0, 0.0, 1.0),
                np.clip(audio.size / float(max_samples), 0.0, 1.0),
            ],
            dtype=np.float32,
        )

        embedding = np.concatenate(
            [
                global_features,
                np.mean(band_matrix, axis=0).astype(np.float32),
                np.std(band_matrix, axis=0).astype(np.float32),
                np.percentile(band_matrix, 10, axis=0).astype(np.float32),
                np.percentile(band_matrix, 90, axis=0).astype(np.float32),
            ]
        )
        return np.nan_to_num(embedding.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


class MERTEmbeddingBackend(EmbeddingBackend):
    """Optional HuggingFace MERT-like embedding backend."""

    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-95M",
        sample_rate: int = 24000,
        window_seconds: float = 5.0,
        device: Optional[str] = None,
        local_files_only: bool = False,
    ):
        super().__init__(name="mert", sample_rate=sample_rate)
        self.model_name = model_name
        self.local_files_only = bool(local_files_only)
        self.window_seconds = float(max(1.0, window_seconds))

        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel
        except Exception as exc:
            raise RuntimeError("torch/transformers are not available") from exc

        self.torch = torch
        self.device = self._resolve_device(torch, device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name,
            local_files_only=self.local_files_only,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=self.local_files_only,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("Loaded MERT backend model=%s device=%s", self.model_name, self.device)

    @staticmethod
    def _resolve_device(torch_module: Any, requested_device: Optional[str]) -> str:
        device = str(requested_device or "auto").strip().lower()
        if device and device != "auto":
            return device
        if torch_module.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(torch_module, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def extract(
        self,
        audio_buffer: np.ndarray,
        sample_rate: int,
        channel_name: Optional[str] = None,
        instrument_type: Optional[str] = None,
    ) -> np.ndarray:
        audio = _to_mono(audio_buffer)
        audio = _resample_linear(audio, int(sample_rate), self.sample_rate)
        max_samples = int(self.window_seconds * self.sample_rate)
        if audio.size > max_samples:
            audio = audio[-max_samples:]
        if audio.size == 0:
            return np.zeros(1, dtype=np.float32)
        max_abs = float(np.max(np.abs(audio)))
        if max_abs > 1.0:
            audio = audio / max_abs

        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None and hasattr(outputs, "hidden_states") and outputs.hidden_states:
            hidden = outputs.hidden_states[-1]
        if hidden is None:
            raise RuntimeError("MERT model did not return hidden states")
        embedding = hidden.mean(dim=1).detach().cpu().numpy()[0]
        return np.nan_to_num(embedding.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def create_embedding_backend(config: Optional[Dict[str, Any]] = None) -> EmbeddingBackend:
    """Create the requested backend and fall back to lightweight on failure."""

    config = config or {}
    backend_name = str(config.get("backend", "lightweight")).strip().lower()
    sample_rate = int(config.get("sample_rate", 24000))
    window_seconds = float(config.get("window_seconds", 5.0))
    fallback_to_lightweight = bool(config.get("fallback_to_lightweight", True))

    if backend_name in {"mert", "mert-like", "mert_like"}:
        try:
            return MERTEmbeddingBackend(
                model_name=str(config.get("model_name", "m-a-p/MERT-v1-95M")),
                sample_rate=sample_rate,
                window_seconds=window_seconds,
                device=config.get("device"),
                local_files_only=bool(config.get("local_files_only", False)),
            )
        except Exception as exc:
            if not fallback_to_lightweight:
                raise RuntimeError(
                    "MERT backend unavailable and fallback_to_lightweight is disabled"
                ) from exc
            logger.warning(
                "MERT backend unavailable (%s); falling back to lightweight perceptual backend",
                exc,
            )

    return LightweightEmbeddingBackend(
        sample_rate=sample_rate,
        window_seconds=window_seconds,
        fft_size=int(config.get("fft_size", 2048)),
        hop_size=int(config.get("hop_size", 512)),
        band_count=int(config.get("band_count", 24)),
    )
