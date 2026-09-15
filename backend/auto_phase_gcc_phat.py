"""Robust GCC-PHAT time alignment for AutoMixer.

The analyzer is deliberately split into measurement and actuation. Measurement
is hardware independent and can be regression-tested with synthetic signals;
AutoPhaseAligner keeps the existing bounded mixer-client hook.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import math
import time
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)
EPS = 1e-12


@dataclass
class DelayMeasurement:
    delay_ms: float
    delay_samples: float
    correlation_peak: float
    psr: float
    snr_db: float
    confidence: float
    coherence: float

    def is_valid(self, min_correlation: float = 0.5, min_psr: float = 5.0) -> bool:
        values = (
            self.delay_ms,
            self.delay_samples,
            self.correlation_peak,
            self.psr,
            self.snr_db,
            self.confidence,
            self.coherence,
        )
        return (
            all(np.isfinite(v) for v in values)
            and self.correlation_peak >= min_correlation
            and self.psr >= min_psr
            and self.confidence > 0.6
        )


def _sanitize(signal: np.ndarray) -> np.ndarray:
    data = np.asarray(signal, dtype=np.float64).reshape(-1)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def _next_pow_two(value: int) -> int:
    return 1 << max(1, int(value - 1).bit_length())


def _aligned_views(ref: np.ndarray, tgt: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Return overlapping samples; positive lag means the target arrives late."""
    n = min(ref.size, tgt.size)
    ref = ref[:n]
    tgt = tgt[:n]
    if lag > 0:
        if lag >= n:
            return ref[:0], tgt[:0]
        return ref[:-lag], tgt[lag:]
    if lag < 0:
        shift = -lag
        if shift >= n:
            return ref[:0], tgt[:0]
        return ref[shift:], tgt[:-shift]
    return ref, tgt


def _normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 32 or b.size < 32:
        return 0.0
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if denom <= EPS:
        return 0.0
    return float(np.clip(abs(float(np.dot(a, b))) / denom, 0.0, 1.0))


def _welch_coherence(ref: np.ndarray, tgt: np.ndarray, frame_size: int, hop: int) -> float:
    n = min(ref.size, tgt.size)
    if n < 64:
        return 0.0
    frame = min(max(64, frame_size), n)
    hop = max(1, min(hop, frame))
    window = np.hanning(frame)
    pxx = None
    pyy = None
    pxy = None
    count = 0
    starts = list(range(0, max(1, n - frame + 1), hop))
    if not starts or starts[-1] != n - frame:
        starts.append(max(0, n - frame))
    for start in starts:
        xr = ref[start:start + frame]
        yr = tgt[start:start + frame]
        if xr.size != frame or yr.size != frame:
            continue
        X = np.fft.rfft(xr * window)
        Y = np.fft.rfft(yr * window)
        xx = np.abs(X) ** 2
        yy = np.abs(Y) ** 2
        xy = X * np.conj(Y)
        pxx = xx if pxx is None else pxx + xx
        pyy = yy if pyy is None else pyy + yy
        pxy = xy if pxy is None else pxy + xy
        count += 1
    if count == 0:
        return 0.0
    coh = np.abs(pxy) ** 2 / (pxx * pyy + EPS)
    finite = coh[np.isfinite(coh)]
    if finite.size == 0:
        return 0.0
    return float(np.clip(np.median(finite), 0.0, 1.0))


class GCCPHATAnalyzer:
    """Estimate inter-channel delay from all representative material available."""

    def __init__(
        self,
        sample_rate: int = 48000,
        fft_size: int = 4096,
        hop_size: int = 2048,
        max_delay_ms: float = 50.0,
    ):
        self.sample_rate = int(sample_rate)
        self.fft_size = int(max(256, fft_size))
        self.hop_size = int(max(64, hop_size))
        self.max_delay_samples = int(max(0.0, max_delay_ms) * self.sample_rate / 1000.0)
        self.buffer_size = self.fft_size * 8
        self.ref_buffer = np.zeros(self.buffer_size, dtype=np.float64)
        self.tgt_buffer = np.zeros(self.buffer_size, dtype=np.float64)
        self.buffer_idx = 0
        self.valid_samples = 0
        self.delay_history: deque[float] = deque(maxlen=10)
        self.confidence_history: deque[float] = deque(maxlen=10)

    def add_frames(self, ref_frame: np.ndarray, tgt_frame: np.ndarray) -> None:
        ref = _sanitize(ref_frame)
        tgt = _sanitize(tgt_frame)
        n = min(ref.size, tgt.size)
        if n == 0:
            return
        ref = ref[:n]
        tgt = tgt[:n]
        if n >= self.buffer_size:
            self.ref_buffer[:] = ref[-self.buffer_size:]
            self.tgt_buffer[:] = tgt[-self.buffer_size:]
            self.buffer_idx = 0
            self.valid_samples = self.buffer_size
            return
        first = min(n, self.buffer_size - self.buffer_idx)
        self.ref_buffer[self.buffer_idx:self.buffer_idx + first] = ref[:first]
        self.tgt_buffer[self.buffer_idx:self.buffer_idx + first] = tgt[:first]
        remaining = n - first
        if remaining:
            self.ref_buffer[:remaining] = ref[first:]
            self.tgt_buffer[:remaining] = tgt[first:]
        self.buffer_idx = (self.buffer_idx + n) % self.buffer_size
        self.valid_samples = min(self.buffer_size, self.valid_samples + n)

    def _buffer_view(self, buffer: np.ndarray) -> np.ndarray:
        if self.valid_samples <= 0:
            return buffer[:0]
        if self.valid_samples < self.buffer_size:
            return buffer[:self.valid_samples].copy()
        if self.buffer_idx == 0:
            return buffer.copy()
        return np.concatenate((buffer[self.buffer_idx:], buffer[:self.buffer_idx]))

    def compute_delay(
        self,
        ref_signal: Optional[np.ndarray] = None,
        tgt_signal: Optional[np.ndarray] = None,
    ) -> DelayMeasurement:
        ref = _sanitize(ref_signal) if ref_signal is not None else self._buffer_view(self.ref_buffer)
        tgt = _sanitize(tgt_signal) if tgt_signal is not None else self._buffer_view(self.tgt_buffer)
        n = min(ref.size, tgt.size)
        if n < max(256, min(self.fft_size, 1024)):
            return DelayMeasurement(0.0, 0.0, 0.0, 0.0, -120.0, 0.0, 0.0)
        ref = ref[:n]
        tgt = tgt[:n]

        ref_rms = math.sqrt(float(np.mean(ref * ref)) + EPS)
        tgt_rms = math.sqrt(float(np.mean(tgt * tgt)) + EPS)
        if ref_rms < 1e-8 or tgt_rms < 1e-8:
            return DelayMeasurement(0.0, 0.0, 0.0, 0.0, -120.0, 0.0, 0.0)

        nfft = _next_pow_two(2 * n - 1)
        R = np.fft.rfft(ref, n=nfft)
        T = np.fft.rfft(tgt, n=nfft)
        cross = T * np.conj(R)
        phat = cross / np.maximum(np.abs(cross), EPS)
        gcc = np.fft.irfft(phat, n=nfft)
        gcc = np.concatenate((gcc[-(n - 1):], gcc[:n]))
        lags = np.arange(-(n - 1), n, dtype=np.int64)

        max_delay = min(self.max_delay_samples, n - 1)
        allowed = np.abs(lags) <= max_delay
        region = gcc[allowed]
        region_lags = lags[allowed]
        if region.size == 0:
            return DelayMeasurement(0.0, 0.0, 0.0, 0.0, -120.0, 0.0, 0.0)

        magnitude = np.abs(region)
        peak_index = int(np.argmax(magnitude))
        lag = float(region_lags[peak_index])
        peak_mag = float(magnitude[peak_index])
        if 0 < peak_index < magnitude.size - 1:
            left, center, right = float(magnitude[peak_index - 1]), peak_mag, float(magnitude[peak_index + 1])
            denom = left - 2.0 * center + right
            if abs(denom) > EPS:
                lag += float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))

        integer_lag = int(round(lag))
        aligned_ref, aligned_tgt = _aligned_views(ref, tgt, integer_lag)
        correlation = _normalized_correlation(aligned_ref, aligned_tgt)

        excluded = magnitude.copy()
        radius = max(3, int(round(0.00025 * self.sample_rate)))
        lo = max(0, peak_index - radius)
        hi = min(excluded.size, peak_index + radius + 1)
        excluded[lo:hi] = 0.0
        sidelobe = float(np.max(excluded)) if excluded.size else 0.0
        psr = float(np.clip(20.0 * math.log10((peak_mag + EPS) / (sidelobe + EPS)), 0.0, 120.0))

        if aligned_ref.size >= 32:
            gain = float(np.dot(aligned_tgt, aligned_ref) / (np.dot(aligned_ref, aligned_ref) + EPS))
            residual = aligned_tgt - gain * aligned_ref
            signal_power = float(np.mean((gain * aligned_ref) ** 2))
            noise_power = float(np.mean(residual ** 2)) + EPS
            snr_db = float(np.clip(10.0 * math.log10((signal_power + EPS) / noise_power), -120.0, 120.0))
        else:
            snr_db = -120.0

        coherence = _welch_coherence(aligned_ref, aligned_tgt, self.fft_size, self.hop_size)
        psr_score = float(np.clip(psr / 12.0, 0.0, 1.0))
        snr_score = float(np.clip((snr_db + 6.0) / 36.0, 0.0, 1.0))
        confidence = float(np.clip(0.50 * correlation + 0.25 * psr_score + 0.15 * coherence + 0.10 * snr_score, 0.0, 1.0))

        self.delay_history.append(lag)
        self.confidence_history.append(confidence)
        if len(self.delay_history) >= 3 and float(np.mean(self.confidence_history)) > 0.65:
            median = float(np.median(np.asarray(self.delay_history, dtype=np.float64)))
            if abs(lag - median) > max(2.0, 0.00025 * self.sample_rate):
                lag = median

        delay_ms = float(lag * 1000.0 / self.sample_rate)
        return DelayMeasurement(delay_ms, float(lag), correlation, psr, snr_db, confidence, coherence)

    def reset(self) -> None:
        self.ref_buffer.fill(0.0)
        self.tgt_buffer.fill(0.0)
        self.buffer_idx = 0
        self.valid_samples = 0
        self.delay_history.clear()
        self.confidence_history.clear()


class AutoPhaseAligner:
    """Bounded controller around GCCPHATAnalyzer."""

    def __init__(self, mixer_client=None, sample_rate: int = 48000, fft_size: int = 4096):
        self.mixer_client = mixer_client
        self.sample_rate = int(sample_rate)
        self.analyzer = GCCPHATAnalyzer(sample_rate=self.sample_rate, fft_size=fft_size)
        self.reference_channel: Optional[int] = None
        self.channels_to_align: List[int] = []
        self.channel_analyzers: Dict[int, GCCPHATAnalyzer] = {}
        self.is_running = False
        self.measurements: Dict[int, DelayMeasurement] = {}
        self.last_update_time: Dict[int, float] = {}
        self.min_update_interval_ms = 100
        self.correlation_threshold = 0.5
        self.psr_threshold = 5.0

    def set_reference_channel(self, channel: int) -> None:
        self.reference_channel = channel

    def add_channel(self, channel: int) -> None:
        if channel not in self.channels_to_align:
            self.channels_to_align.append(channel)
            self.channel_analyzers[channel] = GCCPHATAnalyzer(sample_rate=self.sample_rate)

    def remove_channel(self, channel: int) -> None:
        if channel in self.channels_to_align:
            self.channels_to_align.remove(channel)
            self.channel_analyzers.pop(channel, None)

    def process_audio(self, channel: int, ref_audio: np.ndarray, tgt_audio: np.ndarray) -> None:
        analyzer = self.channel_analyzers.get(channel)
        if analyzer is None:
            return
        analyzer.add_frames(ref_audio, tgt_audio)
        now_ms = time.time() * 1000.0
        last = self.last_update_time.get(channel)
        if last is not None and now_ms - last < self.min_update_interval_ms:
            return
        measurement = analyzer.compute_delay()
        self.measurements[channel] = measurement
        if measurement.is_valid(self.correlation_threshold, self.psr_threshold):
            if self.mixer_client is not None and abs(measurement.delay_ms) > 0.1:
                self._send_delay_command(channel, measurement)
        self.last_update_time[channel] = now_ms

    def _send_delay_command(self, channel: int, measurement: DelayMeasurement) -> None:
        try:
            delay_ms = round(float(measurement.delay_ms) * 50.0) / 50.0
            if delay_ms > 0.0 and np.isfinite(delay_ms):
                self.mixer_client.set_channel_delay(channel, delay_ms)
        except Exception as exc:
            logger.error("Failed to send delay command: %s", exc)

    def get_status(self) -> Dict:
        return {
            "reference_channel": self.reference_channel,
            "channels": self.channels_to_align,
            "measurements": {
                ch: {
                    "delay_ms": m.delay_ms,
                    "confidence": m.confidence,
                    "correlation": m.correlation_peak,
                    "valid": m.is_valid(self.correlation_threshold, self.psr_threshold),
                }
                for ch, m in self.measurements.items()
            },
        }

    def reset(self) -> None:
        self.analyzer.reset()
        for analyzer in self.channel_analyzers.values():
            analyzer.reset()
        self.measurements.clear()
        self.last_update_time.clear()
