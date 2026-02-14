"""
Bleed compensator: converts raw level to "own signal" level by subtracting
estimated bleed contribution. Channel stays in processing - only the metric
used for balance is adjusted.
"""

import math
from typing import Optional

from .bleed_detector import BleedInfo

# K-weighting approx weights per band (relative importance for LUFS)
BAND_WEIGHTS = {
    'sub': 0.15,
    'bass': 0.35,
    'low_mid': 0.25,
    'mid': 0.4,
    'high_mid': 0.5,
    'high': 0.3,
    'air': 0.15,
}
BAND_KEYS = list(BAND_WEIGHTS.keys())


def _get_band_energies(metrics) -> dict:
    """Extract band energies from ChannelMetrics."""
    return {
        'sub': getattr(metrics, 'band_energy_sub', -100),
        'bass': getattr(metrics, 'band_energy_bass', -100),
        'low_mid': getattr(metrics, 'band_energy_low_mid', -100),
        'mid': getattr(metrics, 'band_energy_mid', -100),
        'high_mid': getattr(metrics, 'band_energy_high_mid', -100),
        'high': getattr(metrics, 'band_energy_high', -100),
        'air': getattr(metrics, 'band_energy_air', -100),
    }


def get_compensated_level(
    raw_lufs: float,
    bleed_info: BleedInfo,
    channel_metrics: Optional[object] = None,
    source_metrics: Optional[object] = None,
    compensation_factor_db: float = 6.0,
    compensation_mode: str = 'band_level',
) -> float:
    """
    Get compensated level (own signal only) from raw level and bleed info.

    Args:
        raw_lufs: Measured LUFS (including bleed)
        bleed_info: Bleed detection result
        channel_metrics: ChannelMetrics for target channel (for band-level mode)
        source_metrics: ChannelMetrics for source channel (for band-level mode)
        compensation_factor_db: dB to subtract per full bleed at ratio=1 (linear mode)
        compensation_mode: 'band_level' (use band energies) or 'linear' (use ratio only)

    Returns:
        Compensated LUFS (own signal estimate) for balance calculations
    """
    if bleed_info.bleed_ratio <= 0 or bleed_info.bleed_source_channel is None:
        return raw_lufs

    if compensation_mode == 'band_level' and channel_metrics and source_metrics:
        return _band_level_compensate(
            raw_lufs,
            bleed_info,
            channel_metrics,
            source_metrics,
        )

    # Linear fallback: compensated = raw - (ratio * factor_db)
    bleed_db = bleed_info.bleed_ratio * compensation_factor_db
    return raw_lufs - bleed_db


def _band_level_compensate(
    raw_lufs: float,
    bleed_info: BleedInfo,
    channel_metrics: object,
    source_metrics: object,
    dominance_threshold: float = 0.7,
) -> float:
    """
    Band-level compensation: for each band where source dominates,
    reduce that band's contribution to the level estimate.
    Returns compensated LUFS.
    """
    target_bands = _get_band_energies(channel_metrics)
    source_bands = _get_band_energies(source_metrics)

    total_weight = 0.0
    weighted_sum_linear = 0.0
    gated_weighted_sum_linear = 0.0

    for band in BAND_KEYS:
        t_db = target_bands.get(band, -100)
        s_db = source_bands.get(band, -100)
        w = BAND_WEIGHTS.get(band, 0.2)

        t_lin = 10 ** (t_db / 20) if t_db > -90 else 0
        s_lin = 10 ** (s_db / 20) if s_db > -90 else 0

        total_weight += w
        weighted_sum_linear += w * t_lin

        # If source dominates in this band, reduce target band contribution
        if s_lin > 1e-10:
            dominance = t_lin / (s_lin + 1e-10)
            if dominance < dominance_threshold:
                # Gate: reduce contribution proportional to dominance
                gated = t_lin * (dominance / dominance_threshold)
                gated_weighted_sum_linear += w * gated
            else:
                gated_weighted_sum_linear += w * t_lin
        else:
            gated_weighted_sum_linear += w * t_lin

    if total_weight < 1e-10 or weighted_sum_linear < 1e-10:
        return raw_lufs

    # Ratio of gated to original weighted energy
    energy_ratio = gated_weighted_sum_linear / (weighted_sum_linear + 1e-10)

    # Convert to dB reduction
    if energy_ratio <= 0:
        return raw_lufs - 24  # Max 24dB reduction
    reduction_db = -20 * math.log10(energy_ratio + 1e-10)
    return raw_lufs - min(reduction_db, 24.0)
