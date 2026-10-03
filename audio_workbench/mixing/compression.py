"""Causal, linked STUDIO compressor. No rider, makeup, clipping or console I/O.

Attack/release are e-folding constants of the requested-GR smoother in dB
(63.2% of a step in one tau), not the end-to-end response of an RMS detector.
The optional Numba compiler accelerates exactly the same recurrence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

try:
    from numba import njit
except ImportError:  # Correct, slower implementation without an extra dependency.
    def njit(*args, **kwargs):
        return lambda function: function


@dataclass(frozen=True)
class CompressorConfig:
    threshold_dbfs: float = -18.0
    ratio: float = 3.0
    attack_ms: float = 15.0
    release_ms: float = 150.0
    knee_db: float = 6.0
    max_gr_db: float = 5.0
    detector: str = "rms"
    rms_ms: float = 3.0
    sidechain_hpf_hz: float = 0.0
    bypass: bool = False

    def validate(self, sr: int) -> None:
        if isinstance(sr, bool) or not isinstance(sr, (int, np.integer)) or sr < 1000:
            raise ValueError("sample rate must be an integer >= 1000 Hz")
        limits = {
            "threshold_dbfs": (-160, 24), "ratio": (1, 1000),
            "attack_ms": (0.01, 5000), "release_ms": (0.01, 10000),
            "knee_db": (0, 24), "max_gr_db": (0, 60), "rms_ms": (0.01, 1000),
            "sidechain_hpf_hz": (0, 0.45 * sr),
        }
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"invalid {name}: {value}")
        if self.detector not in {"rms", "peak"}:
            raise ValueError("detector must be rms or peak")
        if not isinstance(self.bypass, bool):
            raise ValueError("bypass must be bool")


def as_audio(x: np.ndarray) -> np.ndarray:
    """Validate explicit mono or samples x (1|2) float PCM; never guess layout."""
    x = np.asarray(x)
    if x.dtype.kind != "f" or not np.isfinite(x).all():
        raise ValueError("audio must be finite floating-point PCM")
    if x.ndim != 1 and not (x.ndim == 2 and x.shape[1] in (1, 2)):
        raise ValueError("expected mono or samples x one/two channels")
    with np.errstate(over="ignore"):
        y = x.astype(np.float32, copy=False)
    if not np.isfinite(y).all():
        raise ValueError("audio overflows float32")
    return y


def static_reduction(level_db: np.ndarray, config: CompressorConfig) -> np.ndarray:
    """Positive requested GR, continuous quadratic soft knee, explicitly capped."""
    delta = np.asarray(level_db, dtype=np.float64) - config.threshold_dbfs
    slope = 1.0 - 1.0 / config.ratio
    gr = np.maximum(delta, 0.0) * slope
    if config.knee_db > 0:
        width = config.knee_db
        mask = np.abs(delta) < width / 2
        gr = np.where(mask, slope * (delta + width / 2) ** 2 / (2 * width), gr)
    return np.clip(gr, 0, config.max_gr_db)


@njit(cache=True)
def _smooth_gr(request: np.ndarray, attack: float, release: float,
               previous: float) -> tuple[np.ndarray, float]:
    gr = np.empty(len(request), dtype=np.float32)
    for i in range(len(request)):
        coefficient = attack if request[i] > previous else release
        previous = coefficient * previous + (1 - coefficient) * request[i]
        gr[i] = previous
    return gr, previous


class LinkedCompressor:
    """Stateful processor: preserve one instance across all blocks of a source.

    RMS detector uses average channel POWER, peak uses maximum channel magnitude.
    One gain curve is applied to all program channels. Detector/HPF/GR state all
    survive block boundaries. External sidechain requires the same frame count.
    """

    def __init__(self, sr: int, config: CompressorConfig | None = None):
        self.config = config or CompressorConfig()
        self.config.validate(sr)
        self.sr = int(sr)
        self._attack = float(np.exp(-1 / (sr * self.config.attack_ms / 1000)))
        self._release = float(np.exp(-1 / (sr * self.config.release_ms / 1000)))
        self._rms_alpha = float(np.exp(-1 / (sr * self.config.rms_ms / 1000)))
        hz = self.config.sidechain_hpf_hz
        self._sos = signal.butter(2, hz, btype="highpass", fs=sr, output="sos") if hz else None
        self.reset()

    def reset(self) -> None:
        self._gr_state = 0.0
        self._power_state = np.zeros(1, dtype=np.float64)
        self._filter_state = None
        self._layout = None

    def process(self, x: np.ndarray, sidechain: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
        x = as_audio(x)
        sc = x if sidechain is None else as_audio(sidechain)
        if len(sc) != len(x):
            raise ValueError("sidechain frame count differs from program")
        if not len(x):
            return x.copy(), np.zeros(0, dtype=np.float32)
        layout = (x.shape[1:] if x.ndim == 2 else (), sc.shape[1:] if sc.ndim == 2 else ())
        if self._layout is not None and layout != self._layout:
            raise ValueError("channel layout changed: reset before a new source")
        self._layout = layout
        cfg = self.config
        if cfg.bypass or cfg.ratio == 1 or cfg.max_gr_db == 0:
            return x.copy(), np.zeros(len(x), dtype=np.float32)
        detected = sc[:, None] if sc.ndim == 1 else sc
        detected = detected.astype(np.float64)
        if self._sos is not None:
            if self._filter_state is None:
                self._filter_state = np.zeros((len(self._sos), 2, detected.shape[1]))
            detected, self._filter_state = signal.sosfilt(
                self._sos, detected, axis=0, zi=self._filter_state)
        if cfg.detector == "rms":
            power = np.mean(detected ** 2, axis=1)
            power, self._power_state = signal.lfilter(
                [1 - self._rms_alpha], [1, -self._rms_alpha], power, zi=self._power_state)
            level = 10 * np.log10(np.maximum(power, 1e-30))
        else:
            level = 20 * np.log10(np.maximum(np.max(np.abs(detected), axis=1), 1e-15))
        gr, self._gr_state = _smooth_gr(
            static_reduction(level, cfg), self._attack, self._release, self._gr_state)
        gain = np.power(10.0, -gr / 20).astype(np.float32)
        out = x * (gain[:, None] if x.ndim == 2 else gain)
        return out.astype(np.float32, copy=False), gr
