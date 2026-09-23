"""Deterministic offline reference DSP. Never connects to a mixer.

The contract is aligned sample-major buffers tapped AFTER input trim/polarity/
input delay and BEFORE HPF/EQ/dynamics. It models a direct, complete stereo sum,
not arbitrary console routing or proprietary compressor algorithms.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import numpy as np
import pyloudnorm as pyln
from scipy.signal import butter, lfilter, resample_poly, sosfilt, welch

from autofoh_safety import ChannelEQMove, ChannelFaderMove, CompressorAdjust, HighPassAdjust

BANDS = {"50_100": (50, 100), "100_200": (100, 200), "200_500": (200, 500),
         "500_1000": (500, 1000), "1000_2500": (1000, 2500),
         "2500_5000": (2500, 5000), "5000_8000": (5000, 8000),
         "8000_12000": (8000, 12000)}
MID = ("100_200", "200_500", "500_1000", "1000_2500", "2500_5000")
EPS = 1e-12


def bounded(value: float, low: float, high: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Invalid {name}: {value}")
    return value


@dataclass(frozen=True)
class EQBand:
    freq_hz: float = 1000.0
    gain_db: float = 0.0
    q: float = 1.0


@dataclass(frozen=True)
class Compressor:
    threshold_db: float = -20.0
    ratio: float = 4.0
    attack_ms: float = 10.0
    release_ms: float = 100.0
    makeup_db: float = 0.0


@dataclass
class RenderChannel:
    channel_id: int
    fader_db: float = 0.0
    eq_enabled: bool = False
    eq_bands: dict[int, EQBand] = field(default_factory=dict)
    hpf_hz: float | None = None
    hpf_order: int = 2
    compressor: Compressor | None = None
    muted: bool = False
    role: str = "unknown"
    pan: float = 0.0


@dataclass
class RenderResult:
    audio: np.ndarray
    channels: dict[int, np.ndarray]
    features: dict[str, float]
    true_peak_dbtp: float
    channel_true_peaks: dict[int, float]


def true_peak(audio: np.ndarray) -> float:
    """4x polyphase reconstructed peak estimate, without a sample-peak fallback."""
    data = np.asarray(audio, dtype=np.float64)
    if not data.size or not np.isfinite(data).all():
        raise ValueError("Invalid true-peak input")
    peak = max(float(np.max(np.abs(data))),
               float(np.max(np.abs(resample_poly(data, 4, 1, axis=0)))))
    return float(20.0 * np.log10(max(peak, EPS)))


def _eq(data: np.ndarray, band: EQBand, sr: int) -> np.ndarray:
    f = bounded(band.freq_hz, 20.0, sr * 0.49, "EQ frequency")
    g = bounded(band.gain_db, -24.0, 24.0, "EQ gain")
    q = bounded(band.q, 0.05, 30.0, "EQ Q")
    if g == 0.0:
        return data.copy()
    amplitude = 10.0 ** (g / 40.0)
    w = 2.0 * np.pi * f / sr
    alpha = np.sin(w) / (2.0 * q)
    b = np.array([1 + alpha * amplitude, -2 * np.cos(w), 1 - alpha * amplitude])
    a = np.array([1 + alpha / amplitude, -2 * np.cos(w), 1 - alpha / amplitude])
    return lfilter(b / a[0], a / a[0], data, axis=0)


def _compress(data: np.ndarray, comp: Compressor, sr: int) -> np.ndarray:
    threshold = bounded(comp.threshold_db, -80, 0, "threshold")
    ratio = bounded(comp.ratio, 1, 20, "ratio")
    attack = math.exp(-1.0 / (bounded(comp.attack_ms, 0.1, 500, "attack") * sr / 1000))
    release = math.exp(-1.0 / (bounded(comp.release_ms, 1, 5000, "release") * sr / 1000))
    makeup = bounded(comp.makeup_db, -12, 12, "makeup")
    envelope = 0.0
    gains = np.empty(len(data), dtype=np.float64)
    # Linked peak detector, feed-forward hard knee, no auto makeup or clipping.
    for i, level in enumerate(np.max(np.abs(data), axis=1)):
        coeff = attack if level > envelope else release
        envelope = coeff * envelope + (1 - coeff) * float(level)
        over = max(0.0, 20 * math.log10(max(envelope, EPS)) - threshold)
        gains[i] = 10 ** ((makeup - over * (1 - 1 / ratio)) / 20)
    return data * gains[:, None]


def measure(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Stereo-power LTAS (no anti-phase cancellation), gated LUFS and crest."""
    data = np.asarray(audio, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if (data.ndim != 2 or data.shape[1] not in (1, 2) or len(data) < sr * 0.4
            or not np.isfinite(data).all()):
        raise ValueError("Measurement needs >=400 ms of finite mono/stereo audio")
    freqs, psd = welch(data, fs=sr, nperseg=min(8192, len(data)), axis=0)
    power = np.mean(psd, axis=1)
    compensated = power * 10 ** ((4.5 * np.log2(np.maximum(freqs, 1) / 100)) / 10)
    levels = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            raise ValueError("Sample rate cannot resolve target features")
        levels[name] = 10 * np.log10(max(float(np.mean(compensated[mask])), EPS))
    mid = float(np.median([levels[name] for name in MID]))
    features = {name: float(value - mid) for name, value in levels.items()}
    rms = float(np.sqrt(np.mean(data * data)))
    loudness = float(pyln.Meter(sr).integrated_loudness(data))
    features["lufs"] = loudness if np.isfinite(loudness) else -120.0
    features["crest_db"] = true_peak(data) - 20 * math.log10(max(rms, EPS))
    return features


class ShadowRenderer:
    def __init__(self, sample_rate: int):
        if isinstance(sample_rate, bool) or int(sample_rate) != sample_rate:
            raise ValueError("Invalid sample rate")
        self.sample_rate = int(bounded(sample_rate, 32000, 192000, "sample rate"))

    def render(self, buffers: Mapping[int, np.ndarray], states: Mapping[int, RenderChannel]) -> RenderResult:
        if not buffers or set(buffers) != set(states):
            raise ValueError("Complete buffer/state correspondence is required")
        outputs: dict[int, np.ndarray] = {}
        peaks: dict[int, float] = {}
        n = None
        for channel_id in sorted(buffers):
            state = states[channel_id]
            if channel_id != state.channel_id:
                raise ValueError("Channel identity mismatch")
            data = np.array(buffers[channel_id], dtype=np.float64, copy=True)
            if data.ndim == 1:
                data = data[:, None]
            if (data.ndim != 2 or data.shape[1] not in (1, 2) or not np.isfinite(data).all()
                    or len(data) < self.sample_rate * 0.5):
                raise ValueError("Finite aligned sample-major audio >=500 ms is required")
            if np.max(np.abs(data)) > 1.0:
                raise ValueError("Input exceeds digital full scale")
            if n is not None and len(data) != n:
                raise ValueError("Unaligned buffer lengths; no implicit truncation")
            n = len(data)
            bounded(state.fader_db, -144, 0, "fader")
            bounded(state.pan, -1, 1, "pan")
            if state.hpf_hz is not None:
                cutoff = bounded(state.hpf_hz, 20, self.sample_rate * 0.49, "HPF")
                if state.hpf_order not in (1, 2, 3, 4, 6, 8):
                    raise ValueError("Unsupported HPF order")
                sos = butter(state.hpf_order, cutoff, fs=self.sample_rate, btype="high", output="sos")
                data = sosfilt(sos, data, axis=0)
            # Validate disabled-band readback too: enabling EQ enables the full bank.
            for band in state.eq_bands.values():
                processed = _eq(data, band, self.sample_rate)
                if state.eq_enabled:
                    data = processed
            if state.compressor is not None:
                data = _compress(data, state.compressor, self.sample_rate)
            if not np.isfinite(data).all():
                raise ValueError("Non-finite DSP result")
            peaks[channel_id] = true_peak(data)
            data *= 0.0 if state.muted else 10 ** (state.fader_db / 20)
            if data.shape[1] == 1:
                angle = (state.pan + 1) * np.pi / 4
                data = data * np.array([np.cos(angle), np.sin(angle)])
            elif state.pan != 0:
                raise ValueError("Stereo balance/pan model not specified")
            outputs[channel_id] = data
        mix = np.sum(list(outputs.values()), axis=0)
        if not np.isfinite(mix).all():
            raise ValueError("Non-finite sum")
        features = measure(mix, self.sample_rate)
        lead_power, other_power = 0.0, 0.0
        for channel_id, data in outputs.items():
            f, p = welch(data, fs=self.sample_rate, nperseg=min(8192, len(data)), axis=0)
            energy = float(np.sum(p[(f >= 1500) & (f < 4000)]))
            if states[channel_id].role == "lead_vocal":
                lead_power += energy
            else:
                other_power += energy
        if lead_power > EPS:
            features["lead_mask_db"] = float(10 * np.log10(max(other_power, EPS) / lead_power))
        return RenderResult(mix, outputs, features, true_peak(mix), peaks)

    def propose(self, states: Mapping[int, RenderChannel], action: Any) -> tuple[dict, dict]:
        """Replace actual settings, not cascade a guessed delta filter."""
        updated = deepcopy(dict(states))
        channel_id = getattr(action, "channel_id", None)
        if channel_id not in updated:
            raise ValueError("Unknown/unmodelled action target")
        state = updated[channel_id]
        changes: dict[str, float] = {}
        if isinstance(action, ChannelFaderMove):
            changes["fader"] = (action.target_db - state.fader_db) / 0.25
            state.fader_db = action.target_db
        elif isinstance(action, ChannelEQMove):
            if action.band not in state.eq_bands:
                raise ValueError("Missing physical EQ band readback")
            old = state.eq_bands[action.band]
            prefix = f"eq{action.band}:"
            changes[prefix + "gain"] = (action.gain_db - old.gain_db) / 0.25
            changes[prefix + "frequency"] = math.log2(action.freq_hz / old.freq_hz) / 0.02
            changes[prefix + "q"] = math.log2(action.q / old.q) / 0.02
            state.eq_bands[action.band] = EQBand(action.freq_hz, action.gain_db, action.q)
            # set_eq_band on both supported consoles does NOT enable the EQ bank.
        elif isinstance(action, HighPassAdjust):
            old = state.hpf_hz or 20.0
            changes["hpf"] = math.log2(action.freq_hz / old) / 0.02
            changes["hpf_enable"] = float(action.enabled) - float(state.hpf_hz is not None)
            state.hpf_hz = action.freq_hz if action.enabled else None
        elif isinstance(action, CompressorAdjust):
            old = state.compressor or Compressor()
            for name in ("threshold_db", "ratio", "attack_ms", "release_ms", "makeup_db"):
                a, b = float(getattr(action, name)), float(getattr(old, name))
                changes[name] = ((a - b) / 0.25 if name.endswith("_db")
                                 else math.log2(a / b) / 0.05)
            changes["compressor_enable"] = float(action.enabled) - float(state.compressor is not None)
            state.compressor = (Compressor(action.threshold_db, action.ratio, action.attack_ms,
                                           action.release_ms, action.makeup_db) if action.enabled else None)
        else:
            raise ValueError(f"Unsupported renderer action: {type(action).__name__}")
        return updated, changes
