"""
Bleed detector: temporal (hit simultaneity) + spectral (instrument band dominance).

Returns BleedInfo with bleed_ratio 0..1 and source channel. Never excludes channels -
compensation is applied externally by BleedCompensator.
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Band names matching ChannelMetrics fields
BAND_KEYS = ['sub', 'bass', 'low_mid', 'mid', 'high_mid', 'high', 'air']
DRUM_INSTRUMENTS = {'kick', 'snare', 'tom', 'hihat', 'ride', 'overhead', 'room', 'drums'}
VOCAL_INSTRUMENTS = {'vocal'}

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

# Expected spectral "shape" for instruments. Values are relative weights, not dB.
INSTRUMENT_TARGET_CURVES = {
    'kick': {'sub': 1.0, 'bass': 0.9, 'low_mid': 0.3, 'mid': 0.1, 'high_mid': 0.05, 'high': 0.02, 'air': 0.0},
    'snare': {'sub': 0.1, 'bass': 0.3, 'low_mid': 0.8, 'mid': 1.0, 'high_mid': 0.8, 'high': 0.5, 'air': 0.2},
    'tom': {'sub': 0.35, 'bass': 1.0, 'low_mid': 0.85, 'mid': 0.45, 'high_mid': 0.18, 'high': 0.06, 'air': 0.0},
    'hihat': {'sub': 0.0, 'bass': 0.02, 'low_mid': 0.1, 'mid': 0.45, 'high_mid': 0.95, 'high': 1.0, 'air': 0.75},
    'ride': {'sub': 0.0, 'bass': 0.05, 'low_mid': 0.15, 'mid': 0.55, 'high_mid': 0.95, 'high': 0.9, 'air': 0.6},
    'overhead': {'sub': 0.05, 'bass': 0.12, 'low_mid': 0.25, 'mid': 0.55, 'high_mid': 0.9, 'high': 1.0, 'air': 0.8},
    'room': {'sub': 0.3, 'bass': 0.45, 'low_mid': 0.7, 'mid': 0.8, 'high_mid': 0.65, 'high': 0.5, 'air': 0.3},
    'bass': {'sub': 1.0, 'bass': 0.95, 'low_mid': 0.45, 'mid': 0.12, 'high_mid': 0.04, 'high': 0.0, 'air': 0.0},
    'guitar': {'sub': 0.05, 'bass': 0.18, 'low_mid': 0.5, 'mid': 0.95, 'high_mid': 0.85, 'high': 0.55, 'air': 0.15},
    'vocal': {'sub': 0.03, 'bass': 0.16, 'low_mid': 0.65, 'mid': 1.0, 'high_mid': 0.72, 'high': 0.35, 'air': 0.12},
    'unknown': {'sub': 0.3, 'bass': 0.3, 'low_mid': 0.3, 'mid': 0.3, 'high_mid': 0.3, 'high': 0.3, 'air': 0.3},
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


def _normalize_instrument_type(name: Optional[str]) -> str:
    if not name:
        return 'unknown'
    normalized = str(name).strip().lower().replace('-', '_').replace(' ', '_')
    aliases = {
        'tom_hi': 'tom',
        'tom_mid': 'tom',
        'tom_lo': 'tom',
        'toms': 'tom',
        'hi_hat': 'hihat',
        'overheads': 'overhead',
        'overhead_l': 'overhead',
        'overhead_r': 'overhead',
        'lead_vocal': 'vocal',
        'leadvocal': 'vocal',
        'back_vocal': 'vocal',
        'backvocal': 'vocal',
        'bgv': 'vocal',
        'electric_guitar': 'guitar',
        'acoustic_guitar': 'guitar',
    }
    return aliases.get(normalized, normalized)


def _bands_to_linear_shape(bands: Dict[str, float]) -> Dict[str, float]:
    linear = {band: (10 ** (bands.get(band, -100.0) / 20.0) if bands.get(band, -100.0) > -90.0 else 0.0)
              for band in BAND_KEYS}
    total = sum(linear.values())
    if total <= 1e-10:
        return {band: 0.0 for band in BAND_KEYS}
    return {band: linear[band] / total for band in BAND_KEYS}


def _curve_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    dot = sum(a[band] * b[band] for band in BAND_KEYS)
    norm_a = math.sqrt(sum(a[band] ** 2 for band in BAND_KEYS))
    norm_b = math.sqrt(sum(b[band] ** 2 for band in BAND_KEYS))
    if norm_a <= 1e-10 or norm_b <= 1e-10:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _band_max(bands: Dict[str, float], band_keys: List[str]) -> float:
    values = [bands.get(band, -100.0) for band in band_keys]
    if not values:
        return -100.0
    return max(values)


def _mean_band_delta(
    src_bands: Dict[str, float],
    target_bands: Dict[str, float],
    band_keys: List[str],
) -> float:
    if not band_keys:
        return 0.0
    deltas = [
        float(src_bands.get(band, -100.0) - target_bands.get(band, -100.0))
        for band in band_keys
    ]
    return float(sum(deltas) / max(len(deltas), 1))


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
        # Level-based: тихий фон = bleed, громкий сигнал = own source (для всех инструментов)
        self.ambient_bleed_threshold_lufs = float(
            config.get('ambient_bleed_threshold_lufs', -25.0)
        )
        self.high_level_own_source_lufs = float(
            config.get('high_level_own_source_lufs', -18.0)
        )
        self.high_level_bleed_scale = float(
            config.get('high_level_bleed_scale', 0.5)
        )
        self.ambient_bleed_ratio = float(config.get('ambient_bleed_ratio', 0.95))
        self.ambient_bleed_ratio_vocal = float(config.get('ambient_bleed_ratio_vocal', 0.75))
        self.instrument_specific_enabled = bool(config.get('instrument_specific_enabled', True))
        self.drums_bleed_min_ratio = float(config.get('drums_bleed_min_ratio', 0.45))
        self.drums_bleed_min_confidence = float(config.get('drums_bleed_min_confidence', 0.40))
        self.vocal_bleed_min_ratio = float(config.get('vocal_bleed_min_ratio', 0.35))
        self.vocal_bleed_min_confidence = float(config.get('vocal_bleed_min_confidence', 0.30))
        self.vocal_low_band_excess_db = float(config.get('vocal_low_band_excess_db', 4.0))
        self.vocal_own_curve_similarity = float(config.get('vocal_own_curve_similarity', 0.72))
        self.instrument_types: Dict[int, str] = {}
        # Temporal: envelope history for peak detection (channel_id -> deque of (time, rms))
        self._envelope_history: Dict[int, deque] = {}
        self._history_maxlen = 30
        self._peak_cooldown_sec = 0.05
        self._adaptive_controls: Dict[int, float] = {}

    def configure(self, instrument_types: Dict[int, str]):
        """Set instrument type per channel for band selection."""
        self.instrument_types = {
            int(channel_id): _normalize_instrument_type(instrument)
            for channel_id, instrument in dict(instrument_types).items()
        }

    def reset(self):
        """Clear temporal history."""
        self._envelope_history.clear()
        self._adaptive_controls.clear()

    def adaptive_gate(
        self,
        channel_id: int,
        channel_level_rms: float,
        reference_level_rms: float,
        alpha: float = 0.3,
    ) -> float:
        """
        Adaptive control update using external reference (Eq. 4.2 style).

        c_m[n+1] = c_m[n]                         if x_RMS,m <= r_RMS
                 = alpha * c'_m + (1-alpha)c_m    otherwise
        """
        current = self._adaptive_controls.get(channel_id, 0.0)
        if channel_level_rms <= reference_level_rms:
            return current

        # c'_m: instantaneous estimate proportional to level-over-reference
        instant = max(0.0, channel_level_rms - reference_level_rms)
        a = max(0.0, min(1.0, alpha))
        updated = (a * instant) + ((1.0 - a) * current)
        self._adaptive_controls[channel_id] = updated
        return updated

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
        target_instrument = _normalize_instrument_type(self.instrument_types.get(channel_id, 'unknown'))

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

        # Instrument-specific refinement for drums and vocals.
        if (
            self.instrument_specific_enabled
            and source is not None
            and channel_id in all_channel_metrics
            and source in all_channel_metrics
        ):
            ratio, confidence = self._refine_instrument_specific_ratio(
                channel_id=channel_id,
                source_id=source,
                target_instrument=target_instrument,
                current_lufs=current_lufs,
                all_channel_levels=all_channel_levels,
                all_channel_metrics=all_channel_metrics,
                ratio=ratio,
                confidence=confidence,
            )

        # Level-based: тихий фон = bleed, громкий = own source (для всех инструментов)
        if current_lufs < self.ambient_bleed_threshold_lufs:
            ratio = (
                self.ambient_bleed_ratio_vocal
                if target_instrument in VOCAL_INSTRUMENTS
                else self.ambient_bleed_ratio
            )
            source = None
            method = 'level_ambient'
        elif current_lufs >= self.high_level_own_source_lufs:
            ratio = ratio * self.high_level_bleed_scale

        return BleedInfo(
            channel_id=channel_id,
            bleed_source_channel=source,
            bleed_ratio=min(1.0, max(0.0, ratio)),
            confidence=confidence,
            method_used=method or 'spectral',
        )

    def _refine_instrument_specific_ratio(
        self,
        channel_id: int,
        source_id: int,
        target_instrument: str,
        current_lufs: float,
        all_channel_levels: Dict[int, float],
        all_channel_metrics: Dict[int, object],
        ratio: float,
        confidence: float,
    ) -> tuple:
        target_metrics = all_channel_metrics.get(channel_id)
        source_metrics = all_channel_metrics.get(source_id)
        if target_metrics is None or source_metrics is None:
            return ratio, confidence

        source_instrument = _normalize_instrument_type(self.instrument_types.get(source_id, 'unknown'))
        target_bands = _get_band_energies(target_metrics)
        source_bands = _get_band_energies(source_metrics)
        target_shape = _bands_to_linear_shape(target_bands)
        source_shape = _bands_to_linear_shape(source_bands)
        target_curve = INSTRUMENT_TARGET_CURVES.get(target_instrument, INSTRUMENT_TARGET_CURVES['unknown'])

        own_similarity = _curve_similarity(target_shape, target_curve)
        cross_similarity = _curve_similarity(target_shape, source_shape)
        source_level = all_channel_levels.get(source_id, current_lufs)
        level_gap_db = float(source_level - current_lufs)

        if target_instrument in DRUM_INSTRUMENTS:
            source_band_keys = INSTRUMENT_BANDS.get(source_instrument, BAND_KEYS)
            mean_source_dominance = _mean_band_delta(source_bands, target_bands, source_band_keys)
            if mean_source_dominance >= 2.0 and level_gap_db >= 2.0:
                ratio += 0.20
                confidence += 0.20

            # If target still looks like its own instrument curve, avoid over-penalizing.
            if own_similarity >= 0.80 and cross_similarity < 0.65 and level_gap_db < 4.0:
                ratio *= 0.60
                confidence *= 0.75

            if ratio < self.drums_bleed_min_ratio or confidence < self.drums_bleed_min_confidence:
                ratio *= 0.50

        elif target_instrument in VOCAL_INSTRUMENTS:
            low_band = _band_max(target_bands, ['sub', 'bass'])
            vocal_mid_band = _band_max(target_bands, ['mid', 'high_mid'])
            low_excess_db = low_band - vocal_mid_band
            source_is_rhythm = source_instrument in DRUM_INSTRUMENTS or source_instrument in {'bass', 'guitar'}

            if source_is_rhythm and level_gap_db >= 2.0 and low_excess_db >= self.vocal_low_band_excess_db:
                ratio = max(ratio, 0.65)
                confidence = max(confidence, 0.60)

            # Preserve own vocal where vocal curve and mid presence are strong.
            if (
                own_similarity >= self.vocal_own_curve_similarity
                and vocal_mid_band >= low_band
                and current_lufs >= self.high_level_own_source_lufs - 4.0
            ):
                ratio *= 0.45
                confidence *= 0.70

            if ratio < self.vocal_bleed_min_ratio or confidence < self.vocal_bleed_min_confidence:
                ratio *= 0.50

        return min(1.0, max(0.0, ratio)), min(1.0, max(0.0, confidence))

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
        target_instrument = _normalize_instrument_type(self.instrument_types.get(channel_id, 'unknown'))
        target_band_keys = INSTRUMENT_BANDS.get(target_instrument, BAND_KEYS)
        target_shape = _bands_to_linear_shape(target_bands)
        target_curve = INSTRUMENT_TARGET_CURVES.get(target_instrument, INSTRUMENT_TARGET_CURVES['unknown'])
        own_curve_similarity = _curve_similarity(target_shape, target_curve)

        best_ratio = 0.0
        best_source = None
        best_conf = 0.0

        for src_ch, src_level in all_channel_levels.items():
            if src_ch == channel_id:
                continue
            if src_ch not in all_channel_metrics:
                continue
            src_instrument = _normalize_instrument_type(self.instrument_types.get(src_ch, 'unknown'))
            if src_instrument in EXCLUDED_BLEED_SOURCES:
                continue
            if src_level <= current_lufs - 3:  # Source must be notably louder (3 dB)
                continue

            src_metrics = all_channel_metrics[src_ch]
            src_bands = _get_band_energies(src_metrics)
            src_band_keys = INSTRUMENT_BANDS.get(src_instrument, BAND_KEYS)
            src_shape = _bands_to_linear_shape(src_bands)
            source_curve = INSTRUMENT_TARGET_CURVES.get(src_instrument, INSTRUMENT_TARGET_CURVES['unknown'])
            src_curve_similarity = _curve_similarity(src_shape, source_curve)
            cross_similarity = _curve_similarity(target_shape, src_shape)
            target_to_source_curve = _curve_similarity(target_shape, source_curve)

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

            dominance_ratio = bleed_energy_linear / (total_energy_linear + 1e-10)
            shape_mismatch = max(0.0, target_to_source_curve - own_curve_similarity)
            ratio = min(
                1.0,
                (dominance_ratio * 0.45) +
                (cross_similarity * 0.30) +
                (shape_mismatch * 0.25)
            )

            level_diff_db = src_level - current_lufs
            conf = min(
                1.0,
                (ratio * 0.6) +
                (min(1.0, level_diff_db / 18.0) * 0.2) +
                (max(0.0, src_curve_similarity - own_curve_similarity) * 0.2)
            )

            if ratio > best_ratio and conf > 0.15:
                best_ratio = ratio
                best_source = src_ch
                best_conf = conf

        return best_ratio, best_source, best_conf

    def _delayed_repetition_score(self, channel_id: int, source_id: int) -> tuple:
        target_history = list(self._envelope_history.get(channel_id, ()))
        source_history = list(self._envelope_history.get(source_id, ()))
        if len(target_history) < 3 or len(source_history) < 3:
            return 0.0, 0.0

        max_delay_sec = self.temporal_max_delay_ms / 1000.0
        matches = 0
        weighted_score = 0.0
        considered = 0

        for target_time, target_rms, _ in target_history[-8:]:
            best_match_score = 0.0
            for source_time, source_rms, _ in source_history:
                delay = target_time - source_time
                if delay < 0 or delay > max_delay_sec:
                    continue
                considered += 1
                if source_rms <= target_rms:
                    continue
                level_advantage = min(1.0, max(0.0, (source_rms - target_rms) / 12.0))
                delay_score = max(0.0, 1.0 - (delay / max(max_delay_sec, 1e-6)))
                candidate = (level_advantage * 0.65) + (delay_score * 0.35)
                if candidate > best_match_score:
                    best_match_score = candidate
            if best_match_score > 0.0:
                matches += 1
                weighted_score += best_match_score

        if matches == 0:
            return 0.0, 0.0
        match_ratio = matches / max(min(len(target_history[-8:]), len(source_history[-8:])), 1)
        score = weighted_score / matches
        confidence = min(1.0, (score * 0.7) + (match_ratio * 0.3))
        return score, confidence

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
            if src_level <= current_lufs or centroid_diff_hz >= 3000:
                continue

            delayed_score, delayed_conf = self._delayed_repetition_score(channel_id, src_ch)
            if delayed_conf < self.temporal_correlation_threshold:
                continue

            level_ratio = 10 ** ((current_lufs - src_level) / 20)  # target/source linear
            level_component = min(1.0, 1.0 - level_ratio)
            centroid_component = max(0.0, 1.0 - (centroid_diff_hz / 3000.0))
            ratio = min(
                1.0,
                (delayed_score * 0.55) +
                (level_component * 0.25) +
                (centroid_component * 0.20)
            )
            conf = min(
                1.0,
                (delayed_conf * 0.7) +
                (min(1.0, (src_level - current_lufs) / 12.0) * 0.2) +
                (centroid_component * 0.1)
            )
            if ratio > best_ratio:
                best_ratio = ratio
                best_source = src_ch
                best_conf = conf

        return best_ratio, best_source, best_conf
