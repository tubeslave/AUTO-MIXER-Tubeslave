"""
Simplified partial loudness estimator for real-time use.

This is not a full Glasberg/Moore model. It provides a lightweight approximation:
- derive masking threshold from mix band energies
- estimate audible part of channel above threshold
"""

from typing import Dict
import math


class SimplifiedPartialLoudness:
    """Approximate partial loudness from band energies."""

    def __init__(self, masking_margin_db: float = 6.0):
        self.masking_margin_db = masking_margin_db

    def estimate_partial_loudness(
        self,
        channel_spectrum,
        mix_spectrum,
    ) -> float:
        """
        Estimate audible channel loudness against a masker spectrum.

        Parameters are assumed linear magnitude arrays with equal shape.
        Returns LUFS-like value in dB.
        """
        if channel_spectrum is None or mix_spectrum is None:
            return -100.0
        if len(channel_spectrum) == 0 or len(channel_spectrum) != len(mix_spectrum):
            return -100.0

        audible_energy = 0.0
        for ch_val, mix_val in zip(channel_spectrum, mix_spectrum):
            threshold = float(mix_val) * (10.0 ** (self.masking_margin_db / 20.0))
            if ch_val > threshold:
                audible_energy += (float(ch_val) - threshold) ** 2

        if audible_energy <= 1e-12:
            return -100.0
        return 10.0 * math.log10(audible_energy)

    def estimate_from_band_energies(
        self,
        channel_band_energy: Dict[str, float],
        mix_band_energy: Dict[str, float],
    ) -> float:
        """Band-energy variant used by controller integration."""
        if not channel_band_energy:
            return -100.0
        audible_linear = 0.0
        for band, ch_db in channel_band_energy.items():
            mix_db = mix_band_energy.get(band, -100.0)
            threshold_db = mix_db + self.masking_margin_db
            if ch_db > threshold_db:
                audible_linear += 10.0 ** ((ch_db - threshold_db) / 10.0)
        if audible_linear <= 1e-12:
            return -100.0
        return 10.0 * math.log10(audible_linear)
