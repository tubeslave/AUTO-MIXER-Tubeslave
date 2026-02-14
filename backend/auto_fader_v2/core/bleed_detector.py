"""
Bleed detector: temporal (hit simultaneity) + spectral (instrument band dominance).

Returns BleedInfo with bleed_ratio 0..1 and source channel. Never excludes channels -
compensation is applied externally by BleedCompensator.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Band names matching ChannelMetrics fields
BAND_KEYS = ['sub', 'bass', 'low_mid', 'mid', 'high_mid', 'high', 'air']

# Channels with these instrument types are excluded from being bleed sources
# (playback/backing track is full-range and loud - causes false positives)
EXCLUDED_BLEED_SOURCES = {'playback'}

# Characteristic bands per instrument (primary frequency ranges for bleed detection)
INSTRUMENT_BANDS = {
    'kick': ['sub', 'bass'],
    'snare': ['low_mid', 'mid', 'high_mid', 'high'],
    'tom': ['bass', 'low_mid', 'mid'],
    'hihat': ['mid', 'high_mid', 'high', 'air'],
    'ride': ['mid', 'high_mid', 'high', 'air'],
    'overhead': ['mid', 'high_mid', 'high', 'air'],
    'room': ['low_mid', 'mid', 'high_mid', 'high'],
    'bass': ['sub', 'bass', 'low_mid'],
    'drums': ['sub', 'bass', 'low_mid', 'mid', 'high_mid', 'high'],
    'unknown': BAND_KEYS,
}


@dataclass
class BleedInfo:
    """Result of bleed detection for one channel."""
    channel_id: int
    bleed_source_channel: Optional[int]
    bleed_ratio: float  # 0.0 = no bleed, 1.0 = full bleed
    confidence: float   # 0.0-1.0
    method_used: str   # 'spectral', 'temporal', 'combined'


def _get_band_energies(metrics) -> Dict[str, float]:
    """Extract band energies from ChannelMetrics as dict."""
    return {
        'sub': getattr(metrics, 'band_energy_sub', -100),
        'bass': getattr(metrics, 'band_energy_bass', -100),
        'low_mid': getattr(metrics, 'band_energy_low_mid', -100),
        'mid': getattr(metrics, 'band_energy_mid', -100),
        'high_mid': getattr(metrics, 'band_energy_high_mid', -100),
        'high': getattr(metrics, 'band_energy_high', -100),
        'air': getattr(metrics, 'band_energy_air', -100),
    }


class BleedDetector:
    """
    Detects bleed from other channels using spectral and temporal analysis.
    Never excludes channels - always returns BleedInfo for compensation.
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.spectral_enabled = config.get('spectral', {}).get('enabled', True)
        self.spectral_dominance_threshold = config.get('spectral', {}).get('dominance_threshold', 0.7)
        self.temporal_enabled = config.get('temporal', {}).get('enabled', True)
        self.temporal_max_delay_ms = config.get('temporal', {}).get('max_delay_ms', 20)
        self.temporal_correlation_threshold = config.get('temporal', {}).get('correlation_threshold', 0.6)
        self.instrument_types: Dict[int, str] = {}
        # Temporal: envelope history for peak detection (channel_id -> deque of (time, rms))
        self._envelope_history: Dict[int, deque] = {}
        self._history_maxlen = 30
        self._peak_cooldown_sec = 0.05

    def configure(self, instrument_types: Dict[int, str]):
        """Set instrument type per channel for band selection."""
        self.instrument_types = dict(instrument_types)

    def reset(self):
        """Clear temporal history."""
        self._envelope_history.clear()

    def detect_bleed(
        self,
        channel_id: int,
        current_lufs: float,
        spectral_centroid: float,
        all_channel_levels: Dict[int, float],
        all_channel_centroids: Dict[int, float],
        all_channel_metrics: Optional[Dict[int, object]] = None,
    ) -> BleedInfo:
        """
        Detect bleed for one channel. Returns BleedInfo with bleed_ratio.
        Channel is never excluded - caller applies compensation.
        """
        all_channel_metrics = all_channel_metrics or {}

        spectral_ratio = 0.0
        spectral_source = None
        spectral_conf = 0.0

        temporal_ratio = 0.0
        temporal_source = None
        temporal_conf = 0.0

        if self.spectral_enabled and all_channel_metrics:
            spectral_ratio, spectral_source, spectral_conf = self._spectral_detect(
                channel_id,
                current_lufs,
                all_channel_levels,
                all_channel_metrics,
            )

        if self.temporal_enabled:
            temporal_ratio, temporal_source, temporal_conf = self._temporal_detect(
                channel_id,
                current_lufs,
                all_channel_levels,
                all_channel_centroids,
                all_channel_metrics,
            )

        # Combine results: take max ratio from either method, prefer higher confidence
        if spectral_conf >= temporal_conf and spectral_source is not None:
            source = spectral_source
            ratio = spectral_ratio
            confidence = spectral_conf
            method = 'spectral'
        elif temporal_source is not None:
            source = temporal_source
            ratio = temporal_ratio
            confidence = temporal_conf
            method = 'temporal'
        else:
            # Combined: average if both detected same source
            if spectral_source == temporal_source and spectral_source is not None:
                ratio = (spectral_ratio + temporal_ratio) / 2
                confidence = (spectral_conf + temporal_conf) / 2
                method = 'combined'
                source = spectral_source
            else:
                source = spectral_source or temporal_source
                ratio = max(spectral_ratio, temporal_ratio)
                confidence = max(spectral_conf, temporal_conf)
                method = 'spectral' if spectral_source else 'temporal'

        return BleedInfo(
            channel_id=channel_id,
            bleed_source_channel=source,
            bleed_ratio=min(1.0, max(0.0, ratio)),
            confidence=confidence,
            method_used=method,
        )

    def _spectral_detect(
        self,
        channel_id: int,
        current_lufs: float,
        all_channel_levels: Dict[int, float],
        all_channel_metrics: Dict[int, object],
    ) -> tuple:
        """
        Spectral bleed: in source instrument's characteristic bands, if source
        dominates and target has similar shape but lower level -> bleed.
        Returns (bleed_ratio, source_channel, confidence).
        """
        if channel_id not in all_channel_metrics:
            return 0.0, None, 0.0

        target_metrics = all_channel_metrics[channel_id]
        target_bands = _get_band_energies(target_metrics)
        target_instrument = self.instrument_types.get(channel_id, 'unknown')
        target_band_keys = INSTRUMENT_BANDS.get(target_instrument, BAND_KEYS)

        best_ratio = 0.0
        best_source = None
        best_conf = 0.0

        for src_ch, src_level in all_channel_levels.items():
            if src_ch == channel_id:
                continue
            if src_ch not in all_channel_metrics:
                continue
            src_instrument = self.instrument_types.get(src_ch, 'unknown')
            if src_instrument in EXCLUDED_BLEED_SOURCES:
                continue
            if src_level <= current_lufs - 3:  # Source must be notably louder (3 dB)
                continue

            src_metrics = all_channel_metrics[src_ch]
            src_bands = _get_band_energies(src_metrics)
            # src_instrument already fetched above for exclusion check
            src_band_keys = INSTRUMENT_BANDS.get(src_instrument, BAND_KEYS)

            # In source's characteristic bands: how much does source dominate?
            bleed_energy_linear = 0.0
            total_energy_linear = 0.0

            for band in src_band_keys:
                if band not in target_bands or band not in src_bands:
                    continue
                t_db = target_bands[band]
                s_db = src_bands[band]
                if s_db < -90:
                    continue
                t_lin = 10 ** (t_db / 20)
                s_lin = 10 ** (s_db / 20)
                total_energy_linear += t_lin
                # Dominance: if source much louder in this band, target likely bleed
                if s_db > t_db + 2:  # Source >2dB louder in this band
                    dominance = 1.0 - min(1.0, t_lin / (s_lin + 1e-10))
                    if dominance >= 1.0 - self.spectral_dominance_threshold:
                        bleed_energy_linear += t_lin

            if total_energy_linear < 1e-10:
                continue

            ratio = bleed_energy_linear / (total_energy_linear + 1e-10)
            # Confidence based on level difference and ratio
            level_diff_db = src_level - current_lufs
            conf = min(1.0, ratio * 1.2 + level_diff_db / 20)

            if ratio > best_ratio and conf > 0.15:
                best_ratio = ratio
                best_source = src_ch
                best_conf = conf

        return best_ratio, best_source, best_conf

    def _temporal_detect(
        self,
        channel_id: int,
        current_lufs: float,
        all_channel_levels: Dict[int, float],
        all_channel_centroids: Dict[int, float],
        all_channel_metrics: Dict[int, object],
    ) -> tuple:
        """
        Temporal bleed: if target channel peaks shortly after source and source
        is louder, likely bleed. Uses RMS history for simple peak correlation.
        Returns (bleed_ratio, source_channel, confidence).
        """
        now = time.time()
        rms = all_channel_metrics.get(channel_id)
        rms_val = getattr(rms, 'rms_level', current_lufs) if rms else current_lufs

        # Update envelope history
        if channel_id not in self._envelope_history:
            self._envelope_history[channel_id] = deque(maxlen=self._history_maxlen)
        self._envelope_history[channel_id].append((now, rms_val, current_lufs))

        # Centroid+level detection works without history (batch Auto Balance calls once per channel)
        # History would be used for peak correlation in streaming mode; skip early return

        # Simple: if this channel's level is lower than another and centroids similar
        # (indicating same source), temporal bleed. Without envelope peaks we use
        # centroid + level correlation as proxy.
        target_centroid = all_channel_centroids.get(channel_id, 0)

        best_ratio = 0.0
        best_source = None
        best_conf = 0.0

        for src_ch, src_level in all_channel_levels.items():
            if src_ch == channel_id:
                continue
            if self.instrument_types.get(src_ch, 'unknown') in EXCLUDED_BLEED_SOURCES:
                continue
            if src_level <= current_lufs - 2:  # Source at least 2 dB louder
                continue

            src_centroid = all_channel_centroids.get(src_ch, 0)
            centroid_diff_hz = abs(target_centroid - src_centroid)
            # Similar centroid + source louder = possible bleed (2000 Hz allows drum/cymbal overlap)
            if centroid_diff_hz < 2000 and src_level > current_lufs:
                level_ratio = 10 ** ((current_lufs - src_level) / 20)  # target/source linear
                ratio = min(1.0, 1.0 - level_ratio)  # more bleed when target << source
                conf = min(1.0, (src_level - current_lufs) / 12)  # 12dB diff = high conf
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_source = src_ch
                    best_conf = conf

        return best_ratio, best_source, best_conf
