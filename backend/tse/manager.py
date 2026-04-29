"""Channel manager for experimental Target Source Extraction."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from .config import TSEConfig, normalize_instrument
from .extractors import BypassExtractor, SimpleSpectralBleedReducer, StreamingTSEExtractor, TargetSourceExtractor
from .models import TSEChunk, TSEResult, TSEStats

logger = logging.getLogger(__name__)


class TSEManager:
    """Owns per-channel TSE extractor instances and telemetry."""

    def __init__(self, config: TSEConfig | Dict[str, Any] | None = None):
        self.config = config if isinstance(config, TSEConfig) else TSEConfig.from_mapping(config)
        self._extractors: Dict[int, TargetSourceExtractor] = {}
        self._stats_by_channel: Dict[int, TSEStats] = {}
        self._last_log_at = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def get_stats(self, channel_id: int) -> TSEStats:
        return self._stats_by_channel.setdefault(int(channel_id), TSEStats())

    def process_channel(
        self,
        channel_id: int,
        audio: np.ndarray,
        sample_rate: int,
        chunk_size: int,
        channel_name: Optional[str] = None,
        instrument_type: Optional[str] = None,
    ) -> TSEResult:
        if not self.enabled or not self.config.instrument_enabled(instrument_type):
            return BypassExtractor().process_chunk(
                TSEChunk(
                    audio=audio,
                    sample_rate=sample_rate,
                    chunk_size=chunk_size,
                    channel_id=channel_id,
                    channel_name=channel_name,
                    instrument_type=instrument_type,
                )
            )

        extractor = self._extractors.get(channel_id)
        if extractor is None:
            extractor = self._build_extractor(sample_rate)
            self._extractors[channel_id] = extractor

        chunk = TSEChunk(
            audio=audio,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            channel_id=channel_id,
            channel_name=channel_name,
            instrument_type=normalize_instrument(instrument_type),
        )
        result = extractor.process_chunk(chunk)
        if (
            self.config.fallback_to_original
            and result.confidence < self.config.min_confidence_for_control
        ):
            result.estimated_target_audio = result.original_audio.copy()
            result.diagnostics.fallback_to_original = True

        stats = self.get_stats(channel_id)
        stats.update(result, self.config.max_latency_ms)
        self._maybe_log_stats(channel_id, result, stats)
        return result

    def reset_channel(self, channel_id: int) -> None:
        extractor = self._extractors.get(channel_id)
        if extractor is not None:
            extractor.reset()

    def _build_extractor(self, sample_rate: int) -> TargetSourceExtractor:
        mode = str(self.config.mode).lower()
        if mode == "bypass":
            return BypassExtractor()
        if mode == "spectral_gate":
            return SimpleSpectralBleedReducer()
        if mode in {"model_stub", "external_model"}:
            chunk_size = max(1, int(sample_rate * self.config.chunk_size_ms / 1000.0))
            return StreamingTSEExtractor(
                sample_rate=sample_rate,
                chunk_size=chunk_size,
                lookback_chunks=self.config.lookback_chunks,
                max_latency_ms=self.config.max_latency_ms,
                fallback_extractor=SimpleSpectralBleedReducer(),
            )
        logger.warning("Unknown TSE mode '%s'; using bypass", mode)
        return BypassExtractor()

    def _maybe_log_stats(self, channel_id: int, result: TSEResult, stats: TSEStats) -> None:
        now = time.monotonic()
        if result.latency_ms > self.config.max_latency_ms:
            logger.warning(
                "TSE channel %s processing over budget: %.2fms > %.2fms",
                channel_id,
                result.latency_ms,
                self.config.max_latency_ms,
            )
        if now - self._last_log_at < self.config.log_interval_sec:
            return
        self._last_log_at = now
        logger.info(
            "TSE stats ch=%s mode=%s avg_latency=%.2fms avg_conf=%.2f avg_bleed=%.2f fallbacks=%s",
            channel_id,
            result.processing_mode,
            stats.average_latency_ms,
            stats.average_confidence,
            stats.average_bleed_score,
            stats.fallback_to_original_count,
        )
