"""
EQ normalization: correct a channel's spectral profile toward a target curve.

Computes the deviation between actual and target spectral profiles,
then fits a 4-band parametric EQ to minimize that deviation using
scipy.optimize.minimize.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import minimize

    HAS_SCIPY_OPT = True
except ImportError:
    HAS_SCIPY_OPT = False


@dataclass
class EQBand:
    """Single parametric EQ band."""
    freq: float     # Center frequency in Hz
    gain_db: float  # Gain in dB
    Q: float        # Q factor (bandwidth)


# Default target spectral profiles per instrument type
# Each profile is (frequencies_hz, magnitudes_db) describing the
# ideal spectral shape relative to 1kHz = 0 dB
REFERENCE_PROFILES = {
    "kick": {
        "freqs": np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]),
        "mags": np.array([0.0, 6.0, 3.0, -2.0, -6.0, -8.0, -4.0, -6.0, -12.0, -18.0]),
    },
    "snare": {
        "freqs": np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]),
        "mags": np.array([-12.0, -6.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, -2.0, -6.0]),
    },
    "vocals": {
        "freqs": np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]),
        "mags": np.array([-24.0, -18.0, -6.0, -1.0, 0.0, 0.0, 3.0, 2.0, -1.0, -6.0]),
    },
    "bass_guitar": {
        "freqs": np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]),
        "mags": np.array([2.0, 6.0, 4.0, 2.0, -1.0, -3.0, -6.0, -10.0, -16.0, -24.0]),
    },
    "electric_guitar": {
        "freqs": np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]),
        "mags": np.array([-18.0, -12.0, -3.0, 1.0, 2.0, 0.0, 2.0, 3.0, 0.0, -6.0]),
    },
    "acoustic_guitar": {
        "freqs": np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]),
        "mags": np.array([-18.0, -8.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, -1.0, -6.0]),
    },
    "keys": {
        "freqs": np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]),
        "mags": np.array([-12.0, -6.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -2.0, -6.0]),
    },
    "hihat": {
        "freqs": np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]),
        "mags": np.array([-30.0, -24.0, -18.0, -12.0, -6.0, -3.0, 0.0, 2.0, 3.0, 0.0]),
    },
    "overheads": {
        "freqs": np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]),
        "mags": np.array([-12.0, -6.0, -2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, -2.0]),
    },
    "brass": {
        "freqs": np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]),
        "mags": np.array([-24.0, -12.0, -4.0, 0.0, 2.0, 0.0, 3.0, 1.0, -2.0, -8.0]),
    },
    "strings": {
        "freqs": np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]),
        "mags": np.array([-18.0, -10.0, -3.0, 0.0, 1.0, 0.0, 1.0, 0.0, -3.0, -8.0]),
    },
}


def compute_spectral_profile(audio, sr=48000, n_fft=4096):
    """
    Compute the averaged spectral profile of an audio signal.

    Args:
        audio: 1D numpy array of audio samples
        sr: sample rate
        n_fft: FFT size

    Returns:
        (frequencies, magnitudes_db): tuple of 1D numpy arrays
            frequencies: Hz values for each bin
            magnitudes_db: average magnitude in dB for each bin
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    hop = n_fft // 2
    window = np.hanning(n_fft)
    n_frames = max(1, (len(audio) - n_fft) // hop)

    accum = np.zeros(n_fft // 2 + 1)
    for i in range(n_frames):
        start = i * hop
        frame = audio[start: start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        windowed = frame * window
        spectrum = np.abs(np.fft.rfft(windowed))
        accum += spectrum ** 2

    avg_mag = np.sqrt(accum / max(n_frames, 1))
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    magnitudes_db = 20.0 * np.log10(avg_mag + 1e-10)

    return frequencies, magnitudes_db


def _parametric_eq_response(freqs, bands, sr=48000):
    """
    Compute the combined frequency response of parametric EQ bands.

    Each band is a peaking EQ filter. The combined response is the
    sum of individual band gains.

    Args:
        freqs: 1D array of frequency points (Hz)
        bands: list of (center_freq, gain_db, Q)
        sr: sample rate

    Returns:
        response_db: 1D array of gain in dB at each frequency
    """
    response = np.zeros_like(freqs, dtype=np.float64)

    for center_freq, gain_db, Q in bands:
        if abs(gain_db) < 0.01:
            continue
        # Peaking EQ response approximation
        # H(f) = gain_db * (1 / (1 + Q^2 * ((f/fc) - (fc/f))^2))
        # This gives a bell-shaped response centered at fc
        ratio = freqs / (center_freq + 1e-6)
        deviation = Q * (ratio - 1.0 / (ratio + 1e-6))
        bell = 1.0 / (1.0 + deviation ** 2)
        response += gain_db * bell

    return response


def _cost_function(params, actual_freqs, deviation_db, n_bands=4):
    """
    Cost function for EQ fitting optimization.

    Params layout: [freq0, gain0, Q0, freq1, gain1, Q1, ...]

    Args:
        params: flattened parameter array
        actual_freqs: frequency points
        deviation_db: target deviation to correct (actual - target)
        n_bands: number of EQ bands

    Returns:
        cost: scalar cost value
    """
    bands = []
    for i in range(n_bands):
        freq = params[i * 3]
        gain = params[i * 3 + 1]
        Q = params[i * 3 + 2]
        bands.append((freq, gain, Q))

    eq_response = _parametric_eq_response(actual_freqs, bands)

    # The EQ should correct the deviation: eq_response ≈ -deviation
    # So the residual is deviation + eq_response (should be zero)
    residual = deviation_db + eq_response
    cost = np.mean(residual ** 2)

    # Regularization: penalize extreme gains and very narrow Q
    gain_penalty = sum(abs(p[1]) for p in bands) * 0.01
    q_penalty = sum(max(0, 0.3 - p[2]) for p in bands) * 1.0

    return cost + gain_penalty + q_penalty


def compute_correction(
    actual_profile: Tuple[np.ndarray, np.ndarray],
    target_profile: Tuple[np.ndarray, np.ndarray],
    n_bands: int = 4,
    max_gain_db: float = 12.0,
    sr: int = 48000,
) -> List[EQBand]:
    """
    Compute parametric EQ correction to match actual profile to target.

    Fits n_bands of parametric EQ to minimize the spectral deviation
    between actual and target profiles.

    Args:
        actual_profile: (frequencies, magnitudes_db) from compute_spectral_profile
        target_profile: (frequencies, magnitudes_db) target spectral shape
        n_bands: number of EQ bands to fit (default 4)
        max_gain_db: maximum allowed gain per band
        sr: sample rate

    Returns:
        list of EQBand objects describing the correction EQ
    """
    actual_freqs, actual_mags = actual_profile
    target_freqs, target_mags = target_profile

    # Interpolate target to match actual frequency points
    target_interp = np.interp(actual_freqs, target_freqs, target_mags)

    # Normalize both to 0 dB at 1kHz
    idx_1k_actual = np.argmin(np.abs(actual_freqs - 1000.0))
    idx_1k_target = np.argmin(np.abs(actual_freqs - 1000.0))
    actual_norm = actual_mags - actual_mags[idx_1k_actual]
    target_norm = target_interp - target_interp[idx_1k_target]

    # Deviation: positive means actual is louder than target (need cut)
    deviation_db = actual_norm - target_norm

    # Focus on audible range only (20 Hz - 20 kHz)
    mask = (actual_freqs >= 20.0) & (actual_freqs <= 20000.0)
    fit_freqs = actual_freqs[mask]
    fit_deviation = deviation_db[mask]

    if len(fit_freqs) == 0:
        return [EQBand(freq=1000.0, gain_db=0.0, Q=1.0) for _ in range(n_bands)]

    if HAS_SCIPY_OPT:
        # Initial guesses: spread bands across log-frequency range
        init_freqs = np.logspace(
            np.log10(80), np.log10(12000), n_bands
        )
        initial_params = []
        for i in range(n_bands):
            initial_params.extend([init_freqs[i], 0.0, 1.0])  # freq, gain, Q

        # Bounds: freq 20-20kHz, gain -max..+max, Q 0.3-10
        bounds = []
        for i in range(n_bands):
            bounds.append((20.0, 20000.0))       # freq
            bounds.append((-max_gain_db, max_gain_db))  # gain
            bounds.append((0.3, 10.0))            # Q

        result = minimize(
            _cost_function,
            initial_params,
            args=(fit_freqs, fit_deviation, n_bands),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-8},
        )

        bands = []
        for i in range(n_bands):
            freq = result.x[i * 3]
            gain = result.x[i * 3 + 1]
            Q = result.x[i * 3 + 2]
            bands.append(EQBand(freq=float(freq), gain_db=float(-gain), Q=float(Q)))

        # Sort by frequency
        bands.sort(key=lambda b: b.freq)
        return bands
    else:
        # Fallback: simple peak-finding approach without scipy
        return _fallback_correction(fit_freqs, fit_deviation, n_bands, max_gain_db)


def _fallback_correction(freqs, deviation_db, n_bands=4, max_gain_db=12.0):
    """
    Simple EQ correction without scipy optimization.

    Finds the n_bands largest deviations and creates correction bands.
    """
    # Smooth deviation to avoid fitting noise
    kernel_size = max(3, len(deviation_db) // 50)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(deviation_db, kernel, mode="same")

    # Find peaks (largest deviations) in the smoothed curve
    bands = []
    used_indices = set()

    for _ in range(n_bands):
        # Find the index with largest absolute deviation not near used indices
        best_idx = -1
        best_val = 0.0
        for i in range(len(smoothed)):
            # Skip if too close to an already-used band
            too_close = False
            for ui in used_indices:
                if abs(i - ui) < len(smoothed) // (n_bands * 2):
                    too_close = True
                    break
            if too_close:
                continue

            if abs(smoothed[i]) > abs(best_val):
                best_val = smoothed[i]
                best_idx = i

        if best_idx < 0:
            bands.append(EQBand(freq=1000.0, gain_db=0.0, Q=1.0))
            continue

        used_indices.add(best_idx)
        freq = float(freqs[best_idx])
        # Correction gain is negative of deviation
        gain = float(np.clip(-best_val, -max_gain_db, max_gain_db))
        # Estimate Q from deviation shape (wider deviation = lower Q)
        half_width = 0
        threshold = abs(best_val) * 0.5
        for offset in range(1, len(smoothed)):
            left = max(0, best_idx - offset)
            right = min(len(smoothed) - 1, best_idx + offset)
            if abs(smoothed[left]) < threshold and abs(smoothed[right]) < threshold:
                half_width = offset
                break
        if half_width == 0:
            half_width = len(smoothed) // 4
        # Q approximation from bandwidth
        if half_width > 0 and best_idx > 0:
            bw_ratio = freqs[min(best_idx + half_width, len(freqs) - 1)] / (
                freqs[max(best_idx - half_width, 0)] + 1e-6
            )
            Q = max(0.3, min(10.0, 1.0 / (np.log2(bw_ratio + 1e-6) + 1e-6)))
        else:
            Q = 1.0

        bands.append(EQBand(freq=freq, gain_db=gain, Q=Q))

    bands.sort(key=lambda b: b.freq)
    return bands


def get_reference_profile(instrument_type):
    """
    Get the reference spectral profile for an instrument type.

    Args:
        instrument_type: string instrument name

    Returns:
        (frequencies, magnitudes_db) or None if not found
    """
    profile = REFERENCE_PROFILES.get(instrument_type)
    if profile is None:
        return None
    return profile["freqs"].copy(), profile["mags"].copy()


def compute_channel_eq(audio, sr=48000, instrument_type=None, n_bands=4):
    """
    Convenience function: compute EQ correction for a channel.

    Args:
        audio: 1D numpy array
        sr: sample rate
        instrument_type: string instrument type for target profile
        n_bands: number of EQ bands

    Returns:
        list of EQBand objects, or empty list if no correction needed
    """
    actual_profile = compute_spectral_profile(audio, sr)

    if instrument_type and instrument_type in REFERENCE_PROFILES:
        target_freqs = REFERENCE_PROFILES[instrument_type]["freqs"]
        target_mags = REFERENCE_PROFILES[instrument_type]["mags"]
        target_profile = (target_freqs, target_mags)
    else:
        # Default: roughly flat response (pink noise reference)
        target_freqs = np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        target_mags = np.array([3.0, 2.0, 1.0, 0.5, 0.0, 0.0, -0.5, -1.0, -2.0, -3.0])
        target_profile = (target_freqs, target_mags)

    bands = compute_correction(actual_profile, target_profile, n_bands=n_bands, sr=sr)
    return bands
