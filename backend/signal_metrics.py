"""Finite streaming signal metrics for automatic soundcheck.

This module keeps the public API used by ``auto_soundcheck_engine`` while
replacing the legacy metric internals with deterministic, finite calculations.
It performs analysis only and contains no mixer/OSC/MIDI write path.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from lufs_gain_staging import KWeightingFilter, LUFSMeter
    HAS_LUFS_METERS = True
except ImportError:
    KWeightingFilter = None
    LUFSMeter = None
    HAS_LUFS_METERS = False

EPS = 1e-12

FREQ_BANDS = {
    "sub": (20.0, 60.0),
    "bass": (60.0, 250.0),
    "low_mid": (250.0, 500.0),
    "mid": (500.0, 2000.0),
    "high_mid": (2000.0, 4000.0),
    "presence": (4000.0, 8000.0),
    "air": (8000.0, 20000.0),
}


def _finite_mono(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 0:
        data = data.reshape(1)
    if data.ndim > 1:
        if data.shape[0] <= 8 and data.shape[1] > data.shape[0] * 4:
            data = data.T
        data = np.mean(data, axis=1, dtype=np.float32)
    return np.nan_to_num(data.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)


def _amp_db(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        return -100.0
    return float(max(-100.0, 20.0 * np.log10(value)))


def _k_weight(samples: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
    data = _finite_mono(samples)
    if data.size == 0:
        return data
    if HAS_LUFS_METERS and KWeightingFilter is not None:
        return np.nan_to_num(KWeightingFilter(sample_rate).process(data), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        from scipy.signal import lfilter

        if sample_rate == 48000:
            b1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
            a1 = np.array([1.0, -1.69065929318241, 0.73248077421585])
            b2 = np.array([1.0, -2.0, 1.0])
            a2 = np.array([1.0, -1.99004745483398, 0.99007225036621])
            return np.nan_to_num(lfilter(b2, a2, lfilter(b1, a1, data)))
    except Exception:
        pass
    return data


def _true_peak_with_overlap(samples: np.ndarray, previous_tail: np.ndarray) -> Tuple[float, np.ndarray]:
    data = _finite_mono(samples)
    if data.size == 0:
        return -100.0, previous_tail
    overlap = np.concatenate([previous_tail, data]) if previous_tail.size else data
    try:
        from scipy.signal import resample_poly

        reconstructed = resample_poly(overlap, 4, 1)
        peak_db = _amp_db(float(np.max(np.abs(reconstructed))))
    except Exception:
        peak_db = _amp_db(float(np.max(np.abs(overlap))))
    tail_len = min(64, data.size)
    return peak_db, data[-tail_len:].copy()


def _gated_integrated_lufs(values: List[float]) -> float:
    blocks = np.asarray([x for x in values if np.isfinite(x) and x > -70.0], dtype=np.float64)
    if blocks.size == 0:
        return -100.0
    energies = np.power(10.0, blocks / 10.0)
    ungated = float(10.0 * np.log10(max(float(np.mean(energies)), EPS)))
    kept = blocks[blocks >= ungated - 10.0]
    if kept.size == 0:
        return ungated
    return float(10.0 * np.log10(max(float(np.mean(np.power(10.0, kept / 10.0))), EPS)))


def _append_ring(blocks: deque, samples: np.ndarray, current_count: int, max_count: int) -> int:
    block = np.asarray(samples, dtype=np.float32).copy()
    blocks.append(block)
    current_count += block.size
    while blocks and current_count > max_count:
        removed = blocks.popleft()
        current_count -= removed.size
    return current_count


@dataclass
class LevelMetrics:
    peak_db: float = -100.0
    true_peak_dbtp: float = -100.0
    rms_db: float = -100.0
    lufs_momentary: float = -100.0
    lufs_short_term: float = -100.0
    lufs_integrated: float = -100.0
    crest_factor_db: float = 0.0
    loudness_range_lu: float = 0.0


@dataclass
class DynamicsMetrics:
    dynamic_range_db: float = 0.0
    attack_time_ms: float = 0.0
    decay_time_ms: float = 0.0
    sustain_level_db: float = -100.0
    release_time_ms: float = 0.0
    envelope_variance: float = 0.0
    transient_density: float = 0.0
    transient_strength_db: float = 0.0
    transient_regularity: float = 0.0
    peak_to_rms_ratio: float = 0.0


@dataclass
class SpectralMetrics:
    centroid_hz: float = 0.0
    rolloff_hz: float = 0.0
    flatness: float = 0.0
    spectral_tilt_db: float = 0.0
    brightness: float = 0.0
    warmth: float = 0.0
    mud_ratio: float = 0.0
    presence_ratio: float = 0.0
    flux: float = 0.0
    band_energy: Dict[str, float] = field(default_factory=dict)


@dataclass
class InterChannelMetrics:
    channel_a: int = 0
    channel_b: int = 0
    cross_correlation: float = 0.0
    delay_samples: int = 0
    delay_ms: float = 0.0
    coherence: float = 0.0
    spectral_similarity: float = 0.0
    level_difference_db: float = 0.0
    phase_inverted: bool = False


@dataclass
class ChannelMetrics:
    channel: int
    timestamp: float = 0.0
    level: LevelMetrics = field(default_factory=LevelMetrics)
    dynamics: DynamicsMetrics = field(default_factory=DynamicsMetrics)
    spectral: SpectralMetrics = field(default_factory=SpectralMetrics)

    def to_dict(self) -> Dict:
        result = {"channel": self.channel, "timestamp": self.timestamp}
        for name, obj in (("level", self.level), ("dynamics", self.dynamics), ("spectral", self.spectral)):
            for key, value in obj.__dict__.items():
                if isinstance(value, dict):
                    for band, band_value in value.items():
                        result[f"{name}_{key}_{band}"] = round(float(band_value), 2)
                elif isinstance(value, float):
                    result[f"{name}_{key}"] = round(value, 2)
                else:
                    result[f"{name}_{key}"] = value
        return result


class SignalAnalyzer:
    """Streaming per-channel analyzer with bounded memory and finite outputs."""

    def __init__(self, channel: int, sample_rate: int = 48000, block_size: int = 1024):
        self.channel = int(channel)
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self._lufs_meter = LUFSMeter(sample_rate) if HAS_LUFS_METERS and LUFSMeter is not None else None
        self._stream_k_filter = KWeightingFilter(sample_rate) if HAS_LUFS_METERS and KWeightingFilter is not None else None
        self._history: deque[np.ndarray] = deque()
        self._history_samples = 0
        self._max_history_samples = max(self.sample_rate * 3, self.block_size * 4)
        self._k_history: deque[np.ndarray] = deque()
        self._k_history_samples = 0
        self._lufs_blocks: List[float] = []
        self._rms_linear: deque[float] = deque(maxlen=256)
        self._level_history: deque[float] = deque(maxlen=512)
        self._max_peak = -100.0
        self._max_true_peak = -100.0
        self._momentary_lufs = -100.0
        self._time_sec = 0.0
        self._tp_tail = np.zeros(0, dtype=np.float32)
        self._transient_times: deque[float] = deque(maxlen=256)
        self._transient_strengths: deque[float] = deque(maxlen=256)
        self._last_level_db = -100.0
        self._prev_norm_spectrum: Optional[np.ndarray] = None
        self._flux = 0.0

    def reset(self):
        if self._lufs_meter is not None:
            self._lufs_meter.reset()
        if self._stream_k_filter is not None:
            self._stream_k_filter.reset()
        self._history.clear()
        self._history_samples = 0
        self._k_history.clear()
        self._k_history_samples = 0
        self._lufs_blocks.clear()
        self._rms_linear.clear()
        self._level_history.clear()
        self._max_peak = -100.0
        self._max_true_peak = -100.0
        self._momentary_lufs = -100.0
        self._time_sec = 0.0
        self._tp_tail = np.zeros(0, dtype=np.float32)
        self._transient_times.clear()
        self._transient_strengths.clear()
        self._last_level_db = -100.0
        self._prev_norm_spectrum = None
        self._flux = 0.0

    def process(self, samples: np.ndarray):
        data = _finite_mono(samples)
        if data.size == 0:
            return
        self._time_sec += data.size / float(self.sample_rate)
        self._history_samples = _append_ring(
            self._history, data, self._history_samples, self._max_history_samples
        )

        peak_db = _amp_db(float(np.max(np.abs(data))))
        self._max_peak = max(self._max_peak, peak_db)
        rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64)) + EPS))
        self._rms_linear.append(rms)
        rms_db = _amp_db(rms)

        if self._lufs_meter is not None:
            value = float(self._lufs_meter.process(data))
            self._momentary_lufs = value if np.isfinite(value) else -100.0
        else:
            weighted = _k_weight(data, self.sample_rate)
            ms = float(np.mean(np.square(weighted, dtype=np.float64)) + EPS)
            self._momentary_lufs = float(-0.691 + 10.0 * np.log10(ms))
        if self._momentary_lufs > -70.0:
            self._lufs_blocks.append(self._momentary_lufs)

        if self._stream_k_filter is not None:
            weighted_block = np.nan_to_num(
                self._stream_k_filter.process(data), nan=0.0, posinf=0.0, neginf=0.0
            )
        else:
            weighted_block = _k_weight(data, self.sample_rate)
        self._k_history_samples = _append_ring(
            self._k_history,
            weighted_block,
            self._k_history_samples,
            max(self.sample_rate * 3, self.block_size),
        )

        tp_db, self._tp_tail = _true_peak_with_overlap(data, self._tp_tail)
        self._max_true_peak = max(self._max_true_peak, tp_db)

        env_db = max(self._momentary_lufs, rms_db)
        if np.isfinite(env_db) and env_db > -100.0:
            self._level_history.append(env_db)
            rise = env_db - self._last_level_db
            if self._last_level_db > -90.0 and rise > 3.0:
                self._transient_times.append(self._time_sec)
                self._transient_strengths.append(min(30.0, rise))
            self._last_level_db = env_db
        cutoff = self._time_sec - 3.0
        while self._transient_times and self._transient_times[0] < cutoff:
            self._transient_times.popleft()
            if self._transient_strengths:
                self._transient_strengths.popleft()

        self._update_flux(data)

    def _update_flux(self, data: np.ndarray):
        n_fft = max(2048, self.block_size)
        block = np.zeros(n_fft, dtype=np.float64)
        take = min(n_fft, data.size)
        block[-take:] = data[-take:]
        spectrum = np.abs(np.fft.rfft(block * np.hanning(n_fft)))
        norm = float(np.linalg.norm(spectrum))
        current = spectrum / max(norm, EPS)
        if self._prev_norm_spectrum is not None and self._prev_norm_spectrum.size == current.size:
            self._flux = float(np.linalg.norm(current - self._prev_norm_spectrum))
        else:
            self._flux = 0.0
        self._prev_norm_spectrum = current

    def _history_array(self) -> np.ndarray:
        if not self._history:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(list(self._history)).astype(np.float32, copy=False)

    def _spectral_metrics(self, audio: np.ndarray) -> SpectralMetrics:
        result = SpectralMetrics(flux=float(self._flux))
        if audio.size == 0:
            return result
        n_fft = max(2048, self.block_size)
        hop = n_fft // 2
        window = np.hanning(n_fft)
        powers = []
        if audio.size < n_fft:
            padded = np.zeros(n_fft, dtype=np.float64)
            padded[: audio.size] = audio
            audio = padded
        starts = list(range(0, audio.size - n_fft + 1, hop)) or [0]
        for start in starts:
            frame = audio[start:start + n_fft]
            if frame.size < n_fft:
                padded = np.zeros(n_fft, dtype=np.float64)
                padded[: frame.size] = frame
                frame = padded
            spec = np.fft.rfft(frame.astype(np.float64) * window)
            powers.append(np.square(np.abs(spec)))
        power = np.mean(np.asarray(powers), axis=0)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)
        total = float(np.sum(power))
        if total <= EPS:
            return result
        magnitude = np.sqrt(power)
        mag_sum = float(np.sum(magnitude))
        result.centroid_hz = float(np.sum(freqs * magnitude) / max(mag_sum, EPS))
        cumsum = np.cumsum(power)
        idx = int(np.searchsorted(cumsum, 0.85 * cumsum[-1]))
        result.rolloff_hz = float(freqs[min(idx, freqs.size - 1)])
        positive = magnitude[magnitude > EPS]
        if positive.size >= 4:
            result.flatness = float(np.exp(np.mean(np.log(positive))) / max(float(np.mean(positive)), EPS))
        valid = (freqs >= 50.0) & (freqs <= min(18000.0, self.sample_rate / 2.0)) & (magnitude > EPS)
        if np.sum(valid) > 4:
            result.spectral_tilt_db = float(
                np.polyfit(np.log2(freqs[valid]), 20.0 * np.log10(magnitude[valid]), 1)[0]
            )
        for name, (lo, hi) in FREQ_BANDS.items():
            mask = (freqs >= lo) & (freqs < min(hi, self.sample_rate / 2.0 + 1.0))
            energy = float(np.sum(power[mask])) if np.any(mask) else 0.0
            result.band_energy[name] = float(10.0 * np.log10(max(energy, EPS)))
        def ratio(lo: float, hi: float) -> float:
            mask = (freqs >= lo) & (freqs < hi)
            return float(np.sum(power[mask]) / total) if np.any(mask) else 0.0
        result.brightness = ratio(4000.0, self.sample_rate / 2.0 + 1.0)
        result.warmth = ratio(200.0, 800.0)
        result.mud_ratio = ratio(200.0, 500.0)
        result.presence_ratio = ratio(2000.0, 5000.0)
        return result

    def get_metrics(self) -> ChannelMetrics:
        metrics = ChannelMetrics(channel=self.channel, timestamp=time.time())
        level = metrics.level
        level.peak_db = float(self._max_peak)
        level.true_peak_dbtp = float(self._max_true_peak)
        if self._rms_linear:
            mean_power = float(np.mean(np.square(np.asarray(self._rms_linear, dtype=np.float64))))
            level.rms_db = _amp_db(np.sqrt(max(mean_power, EPS)))
        level.lufs_momentary = float(self._momentary_lufs)
        if self._k_history:
            weighted = np.concatenate(list(self._k_history)).astype(np.float64, copy=False)
            if weighted.size:
                ms = float(np.mean(np.square(weighted)) + EPS)
                level.lufs_short_term = float(-0.691 + 10.0 * np.log10(ms))
        level.lufs_integrated = _gated_integrated_lufs(self._lufs_blocks)
        if level.rms_db > -90.0 and level.true_peak_dbtp > -90.0:
            level.crest_factor_db = max(0.0, level.true_peak_dbtp - level.rms_db)
        gated = np.asarray([x for x in self._lufs_blocks if x > -70.0], dtype=np.float64)
        if gated.size >= 10:
            relative = level.lufs_integrated - 20.0
            kept = gated[gated >= relative]
            if kept.size >= 4:
                level.loudness_range_lu = max(0.0, float(np.percentile(kept, 95) - np.percentile(kept, 10)))

        dynamics = metrics.dynamics
        history = np.asarray(self._level_history, dtype=np.float64)
        if history.size:
            dynamics.dynamic_range_db = max(0.0, float(np.percentile(history, 95) - np.percentile(history, 10)))
            dynamics.envelope_variance = float(np.std(history))
            dynamics.sustain_level_db = float(np.median(history[-min(50, history.size):]))
        elapsed = min(3.0, max(self._time_sec, EPS))
        dynamics.transient_density = len(self._transient_times) / elapsed
        if self._transient_strengths:
            dynamics.transient_strength_db = float(np.mean(self._transient_strengths))
        if len(self._transient_times) >= 3:
            intervals = np.diff(np.asarray(self._transient_times, dtype=np.float64))
            mean_interval = float(np.mean(intervals))
            if mean_interval > EPS:
                cv = float(np.std(intervals) / mean_interval)
                dynamics.transient_regularity = float(1.0 / (1.0 + cv))
        if level.peak_db > -90.0 and level.rms_db > -90.0:
            dynamics.peak_to_rms_ratio = max(0.0, level.peak_db - level.rms_db)

        metrics.spectral = self._spectral_metrics(self._history_array())
        return metrics


def _align_for_lag(a: np.ndarray, b: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        return a[lag:], b[:-lag] if lag < len(b) else b[:0]
    if lag < 0:
        shift = -lag
        return a[:-shift] if shift < len(a) else a[:0], b[shift:]
    return a, b


def _welch_coherence(a: np.ndarray, b: np.ndarray, block: int = 2048) -> float:
    min_len = min(a.size, b.size)
    if min_len < 64:
        return 0.0
    block = min(block, min_len)
    hop = max(1, block // 2)
    window = np.hanning(block)
    pxx = None
    pyy = None
    pxy = None
    count = 0
    for start in range(0, min_len - block + 1, hop):
        A = np.fft.rfft(a[start:start + block] * window)
        B = np.fft.rfft(b[start:start + block] * window)
        aa = np.abs(A) ** 2
        bb = np.abs(B) ** 2
        ab = A * np.conj(B)
        pxx = aa if pxx is None else pxx + aa
        pyy = bb if pyy is None else pyy + bb
        pxy = ab if pxy is None else pxy + ab
        count += 1
    if count == 0 or pxx is None or pyy is None or pxy is None:
        return 0.0
    coherence = np.abs(pxy) ** 2 / np.maximum(pxx * pyy, EPS)
    energy = pxx + pyy
    valid = energy > np.max(energy) * 1e-6 if np.max(energy) > 0 else np.zeros_like(energy, dtype=bool)
    if not np.any(valid):
        return 0.0
    return float(np.clip(np.mean(coherence[valid]), 0.0, 1.0))


def compare_channels(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    sample_rate: int = 48000,
    ch_a: int = 0,
    ch_b: int = 0,
) -> InterChannelMetrics:
    """Compare two channels using GCC-PHAT, aligned correlation and Welch coherence."""
    result = InterChannelMetrics(channel_a=ch_a, channel_b=ch_b)
    a = _finite_mono(samples_a)
    b = _finite_mono(samples_b)
    min_len = min(a.size, b.size)
    if min_len < 1024:
        return result
    a = a[:min_len].astype(np.float64, copy=False)
    b = b[:min_len].astype(np.float64, copy=False)

    fft_size = 1
    while fft_size < 2 * min_len:
        fft_size <<= 1
    A = np.fft.rfft(a, n=fft_size)
    B = np.fft.rfft(b, n=fft_size)
    cross = A * np.conj(B)
    phat = cross / np.maximum(np.abs(cross), EPS)
    gcc = np.fft.irfft(phat, n=fft_size)
    gcc = np.concatenate((gcc[-fft_size // 2 :], gcc[: fft_size // 2 + 1]))
    center = fft_size // 2
    max_delay = min(center - 1, int(sample_rate * 0.020))
    region = gcc[center - max_delay : center + max_delay + 1]
    local_index = int(np.argmax(np.abs(region)))
    lag = local_index - max_delay
    result.delay_samples = int(lag)
    result.delay_ms = float(abs(lag) / sample_rate * 1000.0)

    aa, bb = _align_for_lag(a, b, lag)
    n = min(aa.size, bb.size)
    if n >= 32:
        aa = aa[:n]
        bb = bb[:n]
        std_a = float(np.std(aa))
        std_b = float(np.std(bb))
        if std_a > EPS and std_b > EPS:
            corr = float(np.corrcoef(aa, bb)[0, 1])
            result.cross_correlation = float(np.clip(corr if np.isfinite(corr) else 0.0, -1.0, 1.0))
            result.phase_inverted = result.cross_correlation < -0.3

    result.coherence = _welch_coherence(a, b)

    window = np.hanning(min_len)
    Sa = np.abs(np.fft.rfft(a * window))
    Sb = np.abs(np.fft.rfft(b * window))
    norm_a = float(np.linalg.norm(Sa))
    norm_b = float(np.linalg.norm(Sb))
    if norm_a > EPS and norm_b > EPS:
        result.spectral_similarity = float(np.clip(np.dot(Sa, Sb) / (norm_a * norm_b), 0.0, 1.0))

    rms_a = float(np.sqrt(np.mean(np.square(a)) + EPS))
    rms_b = float(np.sqrt(np.mean(np.square(b)) + EPS))
    result.level_difference_db = _amp_db(rms_a) - _amp_db(rms_b)
    return result
