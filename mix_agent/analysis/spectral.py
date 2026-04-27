"""Spectral and tonal-balance feature extraction."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .loudness import to_mono

BANDS = {
    "sub": (20.0, 60.0),
    "bass": (60.0, 250.0),
    "low_mid": (250.0, 500.0),
    "mid": (500.0, 2000.0),
    "high_mid": (2000.0, 4000.0),
    "presence": (4000.0, 8000.0),
    "air": (8000.0, 20000.0),
}


def magnitude_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 4096,
    hop: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return frequency bins and a magnitude spectrogram."""
    mono = to_mono(audio)
    if len(mono) < n_fft:
        mono = np.pad(mono, (0, n_fft - len(mono)))
    frames = []
    window = np.hanning(n_fft).astype(np.float32)
    for start in range(0, len(mono) - n_fft + 1, hop):
        frame = mono[start:start + n_fft] * window
        frames.append(np.abs(np.fft.rfft(frame)) + 1e-12)
    if not frames:
        frames.append(np.abs(np.fft.rfft(mono[:n_fft] * window)) + 1e-12)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    return freqs, np.asarray(frames, dtype=np.float32).T


def band_powers(audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
    freqs, mags = magnitude_spectrogram(audio, sample_rate)
    power = np.mean(np.square(mags), axis=1)
    result: Dict[str, float] = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        result[name] = float(np.sum(power[mask]) + 1e-18)
    return result


def compute_spectral_metrics(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """Compute spectral descriptors and broad tonal proxies."""
    freqs, mags = magnitude_spectrogram(audio, sample_rate)
    mean_mag = np.mean(mags, axis=1) + 1e-12
    power = np.square(mean_mag)
    total = float(np.sum(power) + 1e-18)

    centroid = float(np.sum(freqs * mean_mag) / np.sum(mean_mag))
    cumsum = np.cumsum(mean_mag)
    rolloff_idx = int(np.searchsorted(cumsum, 0.85 * cumsum[-1]))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mean_mag) / np.sum(mean_mag)))
    flatness = float(np.exp(np.mean(np.log(mean_mag))) / (np.mean(mean_mag) + 1e-12))

    valid = (freqs >= 50.0) & (freqs <= min(18000.0, sample_rate / 2.0))
    slope = 0.0
    if np.sum(valid) > 4:
        slope = float(np.polyfit(np.log2(freqs[valid]), 20.0 * np.log10(mean_mag[valid]), 1)[0])

    flux = 0.0
    if mags.shape[1] > 1:
        norm = mags / (np.linalg.norm(mags, axis=0, keepdims=True) + 1e-12)
        flux = float(np.mean(np.sqrt(np.sum(np.square(np.diff(norm, axis=1)), axis=0))))

    bands_db: Dict[str, float] = {}
    band_ratios: Dict[str, float] = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        value = float(np.sum(power[mask]) + 1e-18)
        bands_db[name] = round(10.0 * np.log10(value), 3)
        band_ratios[name] = round(value / total, 6)

    harshness = band_ratios.get("high_mid", 0.0) + 0.6 * band_ratios.get("presence", 0.0)
    muddiness = band_ratios.get("low_mid", 0.0)
    boominess = band_ratios.get("bass", 0.0) + band_ratios.get("sub", 0.0)
    brightness = band_ratios.get("presence", 0.0) + band_ratios.get("air", 0.0)
    warmth = band_ratios.get("bass", 0.0) + band_ratios.get("low_mid", 0.0)

    return {
        "spectral_centroid_hz": round(centroid, 3),
        "spectral_rolloff_hz": round(float(freqs[min(rolloff_idx, len(freqs) - 1)]), 3),
        "spectral_bandwidth_hz": round(bandwidth, 3),
        "spectral_flatness": round(flatness, 6),
        "spectral_flux": round(flux, 6),
        "spectral_slope_db_per_octave": round(slope, 3),
        "band_energy_db": bands_db,
        "band_energy_ratios": band_ratios,
        "tonal_balance_curve": bands_db,
        "harshness_proxy": round(float(harshness), 6),
        "muddiness_proxy": round(float(muddiness), 6),
        "boominess_proxy": round(float(boominess), 6),
        "brightness_proxy": round(float(brightness), 6),
        "warmth_proxy": round(float(warmth), 6),
    }
