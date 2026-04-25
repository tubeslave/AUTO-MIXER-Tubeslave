"""Realtime-safe experimental target-source extractors."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import time
from typing import Callable, Deque, Dict, Optional

import numpy as np

from .config import normalize_instrument
from .models import TSEChunk, TSEDiagnostics, TSEResult

logger = logging.getLogger(__name__)


class TargetSourceExtractor:
    """Interface for chunk-wise target source extraction."""

    processing_mode = "bypass"

    def process_chunk(self, chunk: TSEChunk) -> TSEResult:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class BypassExtractor(TargetSourceExtractor):
    processing_mode = "bypass"

    def process_chunk(self, chunk: TSEChunk) -> TSEResult:
        start = time.perf_counter()
        audio = np.asarray(chunk.audio, dtype=np.float32).reshape(-1)
        latency_ms = (time.perf_counter() - start) * 1000.0
        diagnostics = TSEDiagnostics(
            processing_time_ms=latency_ms,
            realtime_budget_ms=1000.0 * max(1, chunk.chunk_size) / max(1, chunk.sample_rate),
        )
        return TSEResult(
            original_audio=audio.copy(),
            estimated_target_audio=audio.copy(),
            bleed_residual=np.zeros_like(audio),
            bleed_score=0.0,
            confidence=1.0,
            latency_ms=latency_ms,
            processing_mode=self.processing_mode,
            diagnostics=diagnostics,
        )


INSTRUMENT_BANDS: Dict[str, tuple[float, float]] = {
    "vocal": (90.0, 12000.0),
    "kick": (35.0, 5000.0),
    "snare": (90.0, 12000.0),
    "tom": (55.0, 7000.0),
    "bass": (35.0, 5000.0),
    "guitar": (80.0, 10000.0),
    "cymbals": (300.0, 18000.0),
    "unknown": (40.0, 18000.0),
}


class SimpleSpectralBleedReducer(TargetSourceExtractor):
    """Lightweight spectral gating for analysis-only target estimates."""

    processing_mode = "spectral_gate"

    def __init__(self, min_mask: float = 0.35, smoothing: float = 0.75):
        self.min_mask = float(max(0.05, min(1.0, min_mask)))
        self.smoothing = float(max(0.0, min(0.98, smoothing)))
        self._prev_mask: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_mask = None

    def process_chunk(self, chunk: TSEChunk) -> TSEResult:
        start = time.perf_counter()
        original = np.asarray(chunk.audio, dtype=np.float32).reshape(-1)
        budget_ms = 1000.0 * max(1, chunk.chunk_size) / max(1, chunk.sample_rate)
        if original.size < 16 or float(np.max(np.abs(original))) < 1e-8:
            latency_ms = (time.perf_counter() - start) * 1000.0
            diagnostics = TSEDiagnostics(
                processing_time_ms=latency_ms,
                realtime_budget_ms=budget_ms,
                notes={"reason": "empty_or_silent"},
            )
            return TSEResult(
                original_audio=original.copy(),
                estimated_target_audio=original.copy(),
                bleed_residual=np.zeros_like(original),
                bleed_score=0.0,
                confidence=0.5,
                latency_ms=latency_ms,
                processing_mode=self.processing_mode,
                diagnostics=diagnostics,
            )

        n_fft = 1
        while n_fft < max(256, original.size):
            n_fft *= 2
        padded = np.zeros(n_fft, dtype=np.float32)
        padded[:original.size] = original
        window = np.hanning(n_fft).astype(np.float32)
        spectrum = np.fft.rfft(padded * window)
        magnitude = np.abs(spectrum)
        power = magnitude * magnitude + 1e-12
        freqs = np.fft.rfftfreq(n_fft, 1.0 / max(1, chunk.sample_rate))

        instrument = normalize_instrument(chunk.instrument_type)
        low_hz, high_hz = INSTRUMENT_BANDS.get(instrument, INSTRUMENT_BANDS["unknown"])
        passband = (freqs >= low_hz) & (freqs <= high_hz)
        total_energy = float(np.sum(power))
        in_band_energy = float(np.sum(power[passband])) if np.any(passband) else total_energy
        out_band_energy = max(0.0, total_energy - in_band_energy)
        out_band_ratio = out_band_energy / max(total_energy, 1e-12)

        local_floor = np.percentile(magnitude, 35)
        tonal_floor = np.percentile(magnitude[passband], 20) if np.any(passband) else local_floor
        threshold = max(local_floor, tonal_floor) * 1.5
        mask = np.where(passband & (magnitude >= threshold), 1.0, self.min_mask).astype(np.float32)
        if self._prev_mask is not None and self._prev_mask.shape == mask.shape:
            mask = self.smoothing * self._prev_mask + (1.0 - self.smoothing) * mask
        self._prev_mask = mask.copy()

        estimated_full = np.fft.irfft(spectrum * mask, n=n_fft).astype(np.float32)
        estimated = estimated_full[:original.size]
        residual = (original - estimated).astype(np.float32)

        original_rms = float(np.sqrt(np.mean(original * original) + 1e-12))
        residual_rms = float(np.sqrt(np.mean(residual * residual) + 1e-12))
        residual_ratio = residual_rms / max(original_rms, 1e-12)
        bleed_score = float(max(0.0, min(1.0, 0.55 * out_band_ratio + 0.45 * residual_ratio)))
        confidence = float(max(0.0, min(1.0, 0.82 - 0.55 * bleed_score)))

        latency_ms = (time.perf_counter() - start) * 1000.0
        diagnostics = TSEDiagnostics(
            processing_time_ms=latency_ms,
            realtime_budget_ms=budget_ms,
            mask_strength=float(1.0 - np.mean(mask)),
            notes={
                "instrument": instrument,
                "passband_low_hz": low_hz,
                "passband_high_hz": high_hz,
                "out_band_ratio": out_band_ratio,
            },
        )
        return TSEResult(
            original_audio=original.copy(),
            estimated_target_audio=estimated,
            bleed_residual=residual,
            bleed_score=bleed_score,
            confidence=confidence,
            latency_ms=latency_ms,
            processing_mode=self.processing_mode,
            diagnostics=diagnostics,
        )


@dataclass
class StreamingTSEExtractor(TargetSourceExtractor):
    """Chunk-wise streaming scaffold with historical context and crossfades.

    ``model_backend`` may later be an ONNX/PyTorch/CoreML callable accepting
    ``(current_chunk, historical_context, metadata)`` and returning a target
    chunk. Without a backend, this class behaves as a model stub over the
    lightweight spectral reducer.
    """

    sample_rate: int = 48000
    chunk_size: int = 480
    lookback_chunks: int = 8
    max_latency_ms: float = 20.0
    model_backend: Optional[Callable[[np.ndarray, np.ndarray, Dict[str, object]], np.ndarray]] = None
    fallback_extractor: Optional[TargetSourceExtractor] = None

    processing_mode = "model_stub"

    def __post_init__(self):
        self.historical_context: Deque[np.ndarray] = deque(maxlen=max(0, int(self.lookback_chunks)))
        self._overlap_tail: Optional[np.ndarray] = None
        if self.fallback_extractor is None:
            self.fallback_extractor = SimpleSpectralBleedReducer()

    def reset(self) -> None:
        self.historical_context.clear()
        self._overlap_tail = None
        if self.fallback_extractor is not None:
            self.fallback_extractor.reset()

    def process_chunk(self, chunk: TSEChunk) -> TSEResult:
        start = time.perf_counter()
        original = np.asarray(chunk.audio, dtype=np.float32).reshape(-1)
        context = (
            np.concatenate(list(self.historical_context)).astype(np.float32)
            if self.historical_context
            else np.zeros(0, dtype=np.float32)
        )
        diagnostics_notes: Dict[str, object] = {"model_backend": self.model_backend is not None}

        if self.model_backend is not None:
            try:
                estimated = np.asarray(
                    self.model_backend(
                        original,
                        context,
                        {
                            "channel_id": chunk.channel_id,
                            "channel_name": chunk.channel_name,
                            "instrument_type": chunk.instrument_type,
                            "sample_rate": chunk.sample_rate,
                        },
                    ),
                    dtype=np.float32,
                ).reshape(-1)
            except Exception as exc:
                logger.warning("Streaming TSE external model failed: %s", exc)
                estimated = self.fallback_extractor.process_chunk(chunk).estimated_target_audio
                diagnostics_notes["external_model_error"] = str(exc)
        else:
            estimated = self.fallback_extractor.process_chunk(chunk).estimated_target_audio

        if estimated.size != original.size:
            fixed = np.zeros_like(original)
            n = min(fixed.size, estimated.size)
            fixed[:n] = estimated[:n]
            estimated = fixed

        estimated = self._crossfade_boundary(estimated)
        residual = (original - estimated).astype(np.float32)
        original_rms = float(np.sqrt(np.mean(original * original) + 1e-12))
        residual_rms = float(np.sqrt(np.mean(residual * residual) + 1e-12))
        bleed_score = float(max(0.0, min(1.0, residual_rms / max(original_rms, 1e-12))))
        confidence = float(max(0.0, min(1.0, 0.78 - 0.45 * bleed_score)))

        self.historical_context.append(original.copy())
        latency_ms = (time.perf_counter() - start) * 1000.0
        diagnostics = TSEDiagnostics(
            processing_time_ms=latency_ms,
            realtime_budget_ms=min(
                self.max_latency_ms,
                1000.0 * max(1, chunk.chunk_size) / max(1, chunk.sample_rate),
            ),
            historical_chunks=len(self.historical_context),
            notes=diagnostics_notes,
        )
        return TSEResult(
            original_audio=original.copy(),
            estimated_target_audio=estimated,
            bleed_residual=residual,
            bleed_score=bleed_score,
            confidence=confidence,
            latency_ms=latency_ms,
            processing_mode="external_model" if self.model_backend else self.processing_mode,
            diagnostics=diagnostics,
        )

    def _crossfade_boundary(self, estimated: np.ndarray) -> np.ndarray:
        if estimated.size == 0:
            return estimated
        overlap = min(max(8, int(0.0025 * self.sample_rate)), estimated.size // 2)
        if overlap <= 0:
            return estimated
        output = estimated.copy()
        if self._overlap_tail is not None and self._overlap_tail.size >= overlap:
            fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            fade_out = 1.0 - fade_in
            output[:overlap] = self._overlap_tail[-overlap:] * fade_out + output[:overlap] * fade_in
        self._overlap_tail = output[-overlap:].copy()
        return output
