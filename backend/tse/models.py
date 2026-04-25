"""Data models for experimental Target Source Extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class TSEChunk:
    audio: np.ndarray
    sample_rate: int
    chunk_size: int
    channel_id: Optional[int] = None
    channel_name: Optional[str] = None
    instrument_type: Optional[str] = None


@dataclass
class TSEDiagnostics:
    processing_time_ms: float = 0.0
    realtime_budget_ms: float = 0.0
    fallback_to_original: bool = False
    historical_chunks: int = 0
    mask_strength: float = 0.0
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "processing_time_ms": float(self.processing_time_ms),
            "realtime_budget_ms": float(self.realtime_budget_ms),
            "fallback_to_original": bool(self.fallback_to_original),
            "historical_chunks": int(self.historical_chunks),
            "mask_strength": float(self.mask_strength),
            "notes": dict(self.notes),
        }


@dataclass
class TSEResult:
    original_audio: np.ndarray
    estimated_target_audio: np.ndarray
    bleed_residual: Optional[np.ndarray] = None
    bleed_score: float = 0.0
    confidence: float = 1.0
    latency_ms: float = 0.0
    processing_mode: str = "bypass"
    diagnostics: TSEDiagnostics = field(default_factory=TSEDiagnostics)

    def audio_for_analysis(self, min_confidence: float = 0.65) -> np.ndarray:
        if self.confidence < min_confidence:
            return self.original_audio
        return self.estimated_target_audio


@dataclass
class TSEStats:
    chunks_processed: int = 0
    fallback_to_original_count: int = 0
    budget_overrun_count: int = 0
    latency_total_ms: float = 0.0
    confidence_total: float = 0.0
    bleed_score_total: float = 0.0

    def update(self, result: TSEResult, max_latency_ms: float) -> None:
        self.chunks_processed += 1
        self.latency_total_ms += float(result.latency_ms)
        self.confidence_total += float(result.confidence)
        self.bleed_score_total += float(result.bleed_score)
        if result.diagnostics.fallback_to_original:
            self.fallback_to_original_count += 1
        if result.latency_ms > max_latency_ms:
            self.budget_overrun_count += 1

    @property
    def average_latency_ms(self) -> float:
        if self.chunks_processed <= 0:
            return 0.0
        return self.latency_total_ms / self.chunks_processed

    @property
    def average_confidence(self) -> float:
        if self.chunks_processed <= 0:
            return 0.0
        return self.confidence_total / self.chunks_processed

    @property
    def average_bleed_score(self) -> float:
        if self.chunks_processed <= 0:
            return 0.0
        return self.bleed_score_total / self.chunks_processed

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "chunks_processed": int(self.chunks_processed),
            "fallback_to_original_count": int(self.fallback_to_original_count),
            "budget_overrun_count": int(self.budget_overrun_count),
            "average_latency_ms": float(self.average_latency_ms),
            "average_confidence": float(self.average_confidence),
            "average_bleed_score": float(self.average_bleed_score),
        }
