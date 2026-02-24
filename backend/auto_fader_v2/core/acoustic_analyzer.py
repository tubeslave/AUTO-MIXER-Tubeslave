"""Acoustic analysis - extracts features from ChannelMetrics for ML/display."""

from dataclasses import dataclass
from typing import Any


@dataclass
class AcousticFeatures:
    """Extracted acoustic features from channel metrics."""
    lufs: float = -60.0
    spectral_centroid: float = 0.0
    spectral_flatness: float = 0.0
    spectral_rolloff: float = 0.0
    rms: float = -60.0
    band_energy: dict = None

    def __post_init__(self):
        if self.band_energy is None:
            self.band_energy = {}


class AcousticAnalyzer:
    """Analyzes ChannelMetrics and returns AcousticFeatures."""

    def analyze(self, metrics: Any) -> AcousticFeatures:
        """Extract acoustic features from ChannelMetrics."""
        band_energy = {
            'sub': getattr(metrics, 'band_energy_sub', -100),
            'bass': getattr(metrics, 'band_energy_bass', -100),
            'low_mid': getattr(metrics, 'band_energy_low_mid', -100),
            'mid': getattr(metrics, 'band_energy_mid', -100),
            'high_mid': getattr(metrics, 'band_energy_high_mid', -100),
            'high': getattr(metrics, 'band_energy_high', -100),
            'air': getattr(metrics, 'band_energy_air', -100),
        }

        # Document-aligned coarse bands:
        # LF 20-250, LMF 250-1k, UMF 1-4k, HF 4-20k.
        band_energy['lf'] = max(band_energy['sub'], band_energy['bass'])
        band_energy['lmf'] = band_energy['low_mid']
        band_energy['umf'] = max(band_energy['mid'], band_energy['high_mid'])
        band_energy['hf'] = max(band_energy['high'], band_energy['air'])

        return AcousticFeatures(
            lufs=getattr(metrics, 'lufs_momentary', -60.0),
            spectral_centroid=getattr(metrics, 'spectral_centroid', 0.0),
            spectral_flatness=getattr(metrics, 'spectral_flatness', 0.0),
            spectral_rolloff=getattr(metrics, 'spectral_rolloff', 0.0),
            rms=getattr(metrics, 'rms_level', -60.0),
            band_energy=band_energy,
        )
