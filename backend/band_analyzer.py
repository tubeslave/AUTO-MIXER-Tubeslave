"""
Band energy analyzer for bleed detection.

Computes band_energy_* (sub, bass, low_mid, mid, high_mid, high, air) from
audio samples. Compatible with BleedDetector._get_band_energies().
"""

import numpy as np
from typing import Dict, Any


# Band ranges (Hz) matching BleedDetector / signal_analysis
BAND_RANGES = {
    'sub': (20, 60),
    'bass': (60, 250),
    'low_mid': (250, 500),
    'mid': (500, 2000),
    'high_mid': (2000, 4000),
    'high': (4000, 8000),
    'air': (8000, 20000),
}


class BandMetrics:
    """Adapter: band_energy dict -> object with band_energy_* attributes."""

    def __init__(self, band_energy: Dict[str, float]):
        self.band_energy_sub = float(band_energy.get('sub', -100.0))
        self.band_energy_bass = float(band_energy.get('bass', -100.0))
        self.band_energy_low_mid = float(band_energy.get('low_mid', -100.0))
        self.band_energy_mid = float(band_energy.get('mid', -100.0))
        self.band_energy_high_mid = float(band_energy.get('high_mid', -100.0))
        self.band_energy_high = float(band_energy.get('high', -100.0))
        self.band_energy_air = float(band_energy.get('air', -100.0))


def compute_band_energy(samples: np.ndarray, sample_rate: int = 48000) -> Dict[str, float]:
    """
    Compute band energy in dB for each band from samples.

    Uses FFT. Returns dict with keys: sub, bass, low_mid, mid, high_mid, high, air.
    """
    fft_size = min(4096, max(2048, len(samples)))
    if len(samples) < fft_size:
        pad = np.zeros(fft_size - len(samples), dtype=samples.dtype)
        block = np.concatenate([samples, pad])
    else:
        block = samples[-fft_size:].astype(np.float64)

    window = np.hanning(len(block))
    spectrum = np.abs(np.fft.rfft(block * window)) + 1e-12
    freqs = np.fft.rfftfreq(len(block), 1.0 / sample_rate)

    band_energy = {}
    for name, (lo_hz, hi_hz) in BAND_RANGES.items():
        mask = (freqs >= lo_hz) & (freqs < hi_hz)
        if np.any(mask):
            e = np.sum(spectrum[mask] ** 2)
            band_energy[name] = float(20.0 * np.log10(e + 1e-12))
        else:
            band_energy[name] = -100.0

    return band_energy


def samples_to_band_metrics(samples: np.ndarray, sample_rate: int = 48000) -> BandMetrics:
    """Convert samples to BandMetrics object for BleedDetector."""
    band_energy = compute_band_energy(samples, sample_rate)
    return BandMetrics(band_energy)
