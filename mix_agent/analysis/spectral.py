"""Spectral and tonal-balance feature extraction."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .dsp_utils import EPS, iter_frames, power_to_db, to_mono

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
    """Return frequency bins and a finite magnitude spectrogram.

    The final partial frame is zero-padded instead of silently discarded, so
    descriptors remain sensitive to material near the end of a clip.
    """
    mono = to_mono(audio)
    n_fft = max(32, int(n_fft))
    hop = max(1, int(hop))
    window = np.hanning(n_fft).astype(np.float64)
    frames = []
    for frame in iter_frames(mono, n_fft, hop, pad_end=True):
        windowed = frame.astype(np.float64, copy=False) * window
        mag = np.abs(np.fft.rfft(windowed))
        frames.append(np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0))
    if not frames:
        frames = [np.zeros(n_fft // 2 + 1, dtype=np.float64)]
    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sample_rate))
    return freqs.astype(np.float64), np.asarray(frames, dtype=np.float64).T


def _mean_power(mags: np.ndarray) -> np.ndarray:
    """Average frame power, not square of an averaged magnitude."""
    if mags.size == 0:
        return np.zeros(mags.shape[0] if mags.ndim else 0, dtype=np.float64)
    return np.mean(np.square(mags, dtype=np.float64), axis=1)


def band_powers(audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
    freqs, mags = magnitude_spectrogram(audio, sample_rate)
    power = _mean_power(mags)
    result: Dict[str, float] = {}
    nyquist = sample_rate / 2.0
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < min(hi, nyquist + 1.0))
        result[name] = float(np.sum(power[mask])) if np.any(mask) else 0.0
    return result


def compute_spectral_metrics(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """Compute full-window spectral descriptors and broad tonal proxies."""
    freqs, mags = magnitude_spectrogram(audio, sample_rate)
    power = _mean_power(mags)
    magnitude = np.sqrt(np.maximum(power, 0.0))
    audible = (freqs >= 20.0) & (freqs <= min(20000.0, sample_rate / 2.0))
    total = float(np.sum(power[audible])) if np.any(audible) else float(np.sum(power))

    bands_db: Dict[str, float] = {}
    band_ratios: Dict[str, float] = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < min(hi, sample_rate / 2.0 + 1.0))
        value = float(np.sum(power[mask])) if np.any(mask) else 0.0
        bands_db[name] = round(power_to_db(value), 3)
        band_ratios[name] = round(value / max(total, EPS), 6) if total > EPS else 0.0

    if total <= EPS or float(np.sum(magnitude)) <= EPS:
        centroid = 0.0
        rolloff_hz = 0.0
        bandwidth = 0.0
        flatness = 0.0
        slope = 0.0
    else:
        denom = float(np.sum(magnitude))
        centroid = float(np.sum(freqs * magnitude) / denom)
        cumsum = np.cumsum(power)
        rolloff_idx = int(np.searchsorted(cumsum, 0.85 * cumsum[-1]))
        rolloff_hz = float(freqs[min(rolloff_idx, len(freqs) - 1)])
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / denom))

        positive = magnitude[audible]
        positive = positive[positive > EPS]
        if positive.size >= 4:
            flatness = float(np.exp(np.mean(np.log(positive))) / (np.mean(positive) + EPS))
        else:
            flatness = 0.0

        valid = audible & (freqs >= 50.0) & (magnitude > EPS)
        slope = 0.0
        if np.sum(valid) > 4:
            slope = float(
                np.polyfit(
                    np.log2(freqs[valid]),
                    20.0 * np.log10(np.maximum(magnitude[valid], EPS)),
                    1,
                )[0]
            )

    flux = 0.0
    if mags.shape[1] > 1:
        norms = np.linalg.norm(mags, axis=0, keepdims=True)
        norm = mags / np.maximum(norms, EPS)
        delta = np.diff(norm, axis=1)
        flux = float(np.mean(np.sqrt(np.sum(np.square(delta), axis=0))))

    harshness = band_ratios.get("high_mid", 0.0) + 0.6 * band_ratios.get("presence", 0.0)
    muddiness = band_ratios.get("low_mid", 0.0)
    boominess = band_ratios.get("bass", 0.0) + band_ratios.get("sub", 0.0)
    brightness = band_ratios.get("presence", 0.0) + band_ratios.get("air", 0.0)
    warmth = band_ratios.get("bass", 0.0) + band_ratios.get("low_mid", 0.0)

    return {
        "spectral_centroid_hz": round(float(centroid), 3),
        "spectral_rolloff_hz": round(float(rolloff_hz), 3),
        "spectral_bandwidth_hz": round(float(bandwidth), 3),
        "spectral_flatness": round(float(flatness), 6),
        "spectral_flux": round(float(flux), 6),
        "spectral_slope_db_per_octave": round(float(slope), 3),
        "band_energy_db": bands_db,
        "band_energy_ratios": band_ratios,
        "tonal_balance_curve": bands_db,
        "harshness_proxy": round(float(harshness), 6),
        "muddiness_proxy": round(float(muddiness), 6),
        "boominess_proxy": round(float(boominess), 6),
        "brightness_proxy": round(float(brightness), 6),
        "warmth_proxy": round(float(warmth), 6),
    }
