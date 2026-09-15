"""Shared numerical primitives for offline/reference DSP analysis.

The functions in this module are deliberately side-effect free.  They do not
know about mixer clients, OSC/MIDI, console state, or automation policy.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np

EPS = 1e-12


def sample_major(audio: np.ndarray) -> np.ndarray:
    """Return finite float32 audio with shape ``(samples, channels)`` or mono 1-D.

    The project normally uses sample-major arrays, but imported stems can be
    channel-first.  A small first dimension (<= 8) with a much larger second
    dimension is treated as channel-first.
    """
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 0:
        data = data.reshape(1)
    if data.ndim > 2:
        data = data.reshape(data.shape[0], -1)
    if data.ndim == 2 and data.shape[0] <= 8 and data.shape[1] > data.shape[0] * 4:
        data = data.T
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Return a finite mono signal without changing sample count."""
    data = sample_major(audio)
    if data.ndim == 1:
        return data
    if data.shape[1] == 0:
        return np.zeros(data.shape[0], dtype=np.float32)
    return np.mean(data, axis=1, dtype=np.float32)


def channels(audio: np.ndarray) -> np.ndarray:
    """Return sample-major 2-D audio, promoting mono to one channel."""
    data = sample_major(audio)
    if data.ndim == 1:
        return data[:, None]
    return data


def amp_to_db(value: float, floor_db: float = -240.0) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        return float(floor_db)
    return float(max(floor_db, 20.0 * np.log10(value)))


def power_to_db(value: float, floor_db: float = -240.0) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        return float(floor_db)
    return float(max(floor_db, 10.0 * np.log10(value)))


def iter_frames(
    samples: np.ndarray,
    frame_size: int,
    hop_size: int,
    *,
    pad_end: bool = True,
) -> Iterator[np.ndarray]:
    """Yield deterministic 1-D frames and optionally include the tail."""
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    frame_size = max(1, int(frame_size))
    hop_size = max(1, int(hop_size))
    if mono.size == 0:
        if pad_end:
            yield np.zeros(frame_size, dtype=np.float32)
        return
    if mono.size <= frame_size:
        frame = np.zeros(frame_size, dtype=np.float32)
        frame[: mono.size] = mono
        yield frame
        return
    starts = list(range(0, mono.size - frame_size + 1, hop_size))
    for start in starts:
        yield mono[start:start + frame_size]
    last_end = starts[-1] + frame_size if starts else 0
    if pad_end and last_end < mono.size:
        start = starts[-1] + hop_size if starts else 0
        if start < mono.size:
            tail = mono[start:start + frame_size]
            frame = np.zeros(frame_size, dtype=np.float32)
            frame[: tail.size] = tail
            yield frame


def _k_weight_coefficients(sample_rate: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Design the two BS.1770 K-weighting biquads used by the project."""
    fs = float(sample_rate)
    # Pre-filter / high shelf.
    f0 = 1681.974450955533
    gain_db = 3.999843853973347
    q = 0.7071752369554196
    k = np.tan(np.pi * f0 / fs)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf_b = np.array([
        (vh + vb * k / q + k * k) / a0,
        2.0 * (k * k - vh) / a0,
        (vh - vb * k / q + k * k) / a0,
    ], dtype=np.float64)
    shelf_a = np.array([
        1.0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / q + k * k) / a0,
    ], dtype=np.float64)

    # RLB high-pass.
    f0 = 38.13547087602444
    q = 0.5003270373238773
    k = np.tan(np.pi * f0 / fs)
    a0 = 1.0 + k / q + k * k
    hp_b = np.array([1.0 / a0, -2.0 / a0, 1.0 / a0], dtype=np.float64)
    hp_a = np.array([
        1.0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / q + k * k) / a0,
    ], dtype=np.float64)
    return shelf_b, shelf_a, hp_b, hp_a


def k_weight(audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, str]:
    """Apply K-weighting independently to every channel.

    SciPy is preferred because it provides stable IIR filtering.  If unavailable,
    the original finite audio is returned and the method string makes the
    approximation explicit to callers.
    """
    data = channels(audio).astype(np.float64, copy=False)
    try:
        from scipy.signal import lfilter

        b1, a1, b2, a2 = _k_weight_coefficients(sample_rate)
        out = np.empty_like(data, dtype=np.float64)
        for ch in range(data.shape[1]):
            stage1 = lfilter(b1, a1, data[:, ch])
            out[:, ch] = lfilter(b2, a2, stage1)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), "k_weighting"
    except Exception:
        return data, "unweighted_fallback"


def loudness_from_weighted(weighted: np.ndarray) -> float:
    """Convert K-weighted channel mean-square energy to LUFS."""
    data = channels(weighted).astype(np.float64, copy=False)
    if data.size == 0:
        return -100.0
    energy = float(np.sum(np.mean(np.square(data), axis=0)))
    if not np.isfinite(energy) or energy <= EPS:
        return -100.0
    return float(-0.691 + 10.0 * np.log10(energy))


def window_loudness(audio: np.ndarray, sample_rate: int, window_sec: float) -> Tuple[list[float], str]:
    """Return overlapping K-weighted loudness windows (75% overlap)."""
    weighted, method = k_weight(audio, sample_rate)
    frame = max(1, int(round(float(window_sec) * sample_rate)))
    hop = max(1, frame // 4)
    if weighted.shape[0] == 0:
        return [], method
    values: list[float] = []
    # Preserve channels while framing on the sample axis.
    starts = list(range(0, max(1, weighted.shape[0] - frame + 1), hop))
    if not starts:
        starts = [0]
    for start in starts:
        chunk = weighted[start:start + frame]
        if chunk.shape[0] < frame:
            padded = np.zeros((frame, weighted.shape[1]), dtype=np.float64)
            padded[: chunk.shape[0]] = chunk
            chunk = padded
        values.append(loudness_from_weighted(chunk))
    last_end = starts[-1] + frame
    if weighted.shape[0] > frame and last_end < weighted.shape[0]:
        start = starts[-1] + hop
        if start < weighted.shape[0]:
            chunk = weighted[start:start + frame]
            padded = np.zeros((frame, weighted.shape[1]), dtype=np.float64)
            padded[: chunk.shape[0]] = chunk
            values.append(loudness_from_weighted(padded))
    return values, method


def integrated_loudness(audio: np.ndarray, sample_rate: int) -> Tuple[float, str]:
    """BS.1770-style integrated loudness with absolute and relative gating."""
    blocks, method = window_loudness(audio, sample_rate, 0.4)
    finite = np.asarray([v for v in blocks if np.isfinite(v) and v > -70.0], dtype=np.float64)
    if finite.size == 0:
        return -100.0, method
    energies = np.power(10.0, (finite + 0.691) / 10.0)
    ungated_energy = float(np.mean(energies))
    ungated_lufs = float(-0.691 + 10.0 * np.log10(max(ungated_energy, EPS)))
    relative_gate = ungated_lufs - 10.0
    kept = finite[finite >= relative_gate]
    if kept.size == 0:
        return ungated_lufs, method
    kept_energy = np.power(10.0, (kept + 0.691) / 10.0)
    result = float(-0.691 + 10.0 * np.log10(max(float(np.mean(kept_energy)), EPS)))
    return result, method


def true_peak(audio: np.ndarray, oversample: int = 4) -> Tuple[float, str]:
    """Offline reconstructed peak using polyphase oversampling when available."""
    data = sample_major(audio)
    if data.size == 0:
        return -100.0, "empty"
    try:
        from scipy.signal import resample_poly

        reconstructed = resample_poly(data, int(oversample), 1, axis=0)
        peak = float(np.max(np.abs(reconstructed)))
        return amp_to_db(peak), f"{int(oversample)}x_resample_poly"
    except Exception:
        return amp_to_db(float(np.max(np.abs(data)))), "sample_peak_fallback"
