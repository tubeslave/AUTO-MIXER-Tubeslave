"""Stateful modular audio processing graph used by offline/ML experiments.

The original node implementations recreated filter/envelope state on every
``process`` call, so the same signal produced different output depending on host
buffer boundaries.  This version preserves state explicitly, sanitizes invalid
samples, and keeps the existing public classes/parameter API.

This module processes numpy audio only.  It has no mixer, OSC or MIDI write path.
"""

from __future__ import annotations

import abc
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.signal import butter, lfilter, sosfilt, sosfilt_zi
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

logger = logging.getLogger(__name__)
EPS = 1e-12


def _finite(audio: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(audio, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


class ProcessingNode(abc.ABC):
    def __init__(self, name: str = "", bypass: bool = False):
        self.name = name or self.__class__.__name__
        self.bypass = bool(bypass)

    @abc.abstractmethod
    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        raise NotImplementedError

    def reset_state(self) -> None:
        """Reset streaming state without changing parameters."""

    def get_params(self) -> Dict[str, float]:
        return {}

    def set_params(self, params: Dict[str, float]) -> None:
        del params

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, bypass={self.bypass})"


class HPFNode(ProcessingNode):
    """Stateful Butterworth high-pass filter."""

    def __init__(self, cutoff_hz: float = 80.0, order: int = 2, **kwargs: Any):
        super().__init__(**kwargs)
        self.cutoff_hz = float(cutoff_hz)
        self.order = int(order)
        self._design_key: Optional[Tuple[int, float, int]] = None
        self._sos: Optional[np.ndarray] = None
        self._zi: Optional[np.ndarray] = None
        self._fallback_prev_x = 0.0
        self._fallback_prev_y = 0.0

    def reset_state(self) -> None:
        self._zi = None
        self._fallback_prev_x = 0.0
        self._fallback_prev_y = 0.0

    def _ensure_filter(self, sr: int) -> bool:
        nyquist = float(sr) * 0.5
        if self.cutoff_hz <= 0.0 or self.cutoff_hz >= nyquist:
            return False
        key = (int(sr), round(float(self.cutoff_hz), 9), int(self.order))
        if key != self._design_key:
            self._design_key = key
            self.reset_state()
            if HAS_SCIPY:
                self._sos = butter(self.order, self.cutoff_hz / nyquist, btype="high", output="sos")
            else:
                self._sos = None
        return True

    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        data = _finite(audio)
        if self.bypass or data.size == 0 or not self._ensure_filter(sr):
            return data.copy()
        if HAS_SCIPY and self._sos is not None:
            shape = (self._sos.shape[0], 2) + data.shape[1:]
            if self._zi is None or self._zi.shape != shape:
                self._zi = np.zeros(shape, dtype=np.float64)
            out, self._zi = sosfilt(self._sos, data, axis=0, zi=self._zi)
            return np.nan_to_num(out)

        rc = 1.0 / (2.0 * math.pi * self.cutoff_hz)
        dt = 1.0 / float(sr)
        alpha = rc / (rc + dt)
        out = np.empty_like(data)
        prev_x = self._fallback_prev_x
        prev_y = self._fallback_prev_y
        for i, x in enumerate(data):
            y = alpha * (prev_y + float(x) - prev_x)
            out[i] = y
            prev_x, prev_y = float(x), y
        self._fallback_prev_x, self._fallback_prev_y = prev_x, prev_y
        return out

    def get_params(self) -> Dict[str, float]:
        return {"cutoff_hz": self.cutoff_hz, "order": float(self.order)}

    def set_params(self, params: Dict[str, float]) -> None:
        changed = False
        if "cutoff_hz" in params:
            value = max(20.0, min(2000.0, float(params["cutoff_hz"])))
            changed |= value != self.cutoff_hz
            self.cutoff_hz = value
        if "order" in params:
            value = max(1, min(8, int(params["order"])))
            changed |= value != self.order
            self.order = value
        if changed:
            self._design_key = None
            self.reset_state()


class GateNode(ProcessingNode):
    """Stateful gate with peak envelope, attack, hold and release."""

    def __init__(
        self,
        threshold_db: float = -50.0,
        attack_ms: float = 0.5,
        hold_ms: float = 50.0,
        release_ms: float = 100.0,
        range_db: float = -80.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.threshold_db = float(threshold_db)
        self.attack_ms = float(attack_ms)
        self.hold_ms = float(hold_ms)
        self.release_ms = float(release_ms)
        self.range_db = float(range_db)
        self._gain = 10.0 ** (self.range_db / 20.0)
        self._hold_remaining = 0

    def reset_state(self) -> None:
        self._gain = 10.0 ** (self.range_db / 20.0)
        self._hold_remaining = 0

    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        data = _finite(audio)
        if self.bypass or data.size == 0:
            return data.copy()
        if data.ndim != 1:
            # Use linked detector across channels, then apply same gain.
            detector = np.max(np.abs(data), axis=1)
        else:
            detector = np.abs(data)
        threshold = 10.0 ** (self.threshold_db / 20.0)
        floor_gain = 10.0 ** (self.range_db / 20.0)
        attack_samples = max(1.0, self.attack_ms * sr / 1000.0)
        release_samples = max(1.0, self.release_ms * sr / 1000.0)
        attack_coeff = math.exp(-1.0 / attack_samples)
        release_coeff = math.exp(-1.0 / release_samples)
        hold_samples = max(0, int(round(self.hold_ms * sr / 1000.0)))
        gains = np.empty(detector.size, dtype=np.float64)
        gain = float(np.clip(self._gain, floor_gain, 1.0))
        hold = int(self._hold_remaining)
        for i, env in enumerate(detector):
            if env >= threshold:
                hold = hold_samples
                target = 1.0
            elif hold > 0:
                hold -= 1
                target = 1.0
            else:
                target = floor_gain
            coeff = attack_coeff if target > gain else release_coeff
            gain = target + coeff * (gain - target)
            gain = float(np.clip(gain, floor_gain, 1.0))
            gains[i] = gain
        self._gain = gain
        self._hold_remaining = hold
        if data.ndim == 1:
            return data * gains
        return data * gains[:, None]

    def get_params(self) -> Dict[str, float]:
        return {
            "threshold_db": self.threshold_db,
            "attack_ms": self.attack_ms,
            "hold_ms": self.hold_ms,
            "release_ms": self.release_ms,
            "range_db": self.range_db,
        }

    def set_params(self, params: Dict[str, float]) -> None:
        if "threshold_db" in params:
            self.threshold_db = max(-96.0, min(0.0, float(params["threshold_db"])))
        if "attack_ms" in params:
            self.attack_ms = max(0.01, min(100.0, float(params["attack_ms"])))
        if "hold_ms" in params:
            self.hold_ms = max(0.0, min(2000.0, float(params["hold_ms"])))
        if "release_ms" in params:
            self.release_ms = max(1.0, min(5000.0, float(params["release_ms"])))
        if "range_db" in params:
            self.range_db = max(-96.0, min(0.0, float(params["range_db"])))
        self._gain = float(np.clip(self._gain, 10.0 ** (self.range_db / 20.0), 1.0))


class EQNode(ProcessingNode):
    """Stateful RBJ-style parametric EQ."""

    @dataclass
    class Band:
        band_type: str = "peak"
        frequency: float = 1000.0
        gain_db: float = 0.0
        q: float = 1.0

    def __init__(self, bands: Optional[List[Dict[str, Any]]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.bands: List[EQNode.Band] = []
        for item in bands or []:
            self.bands.append(
                EQNode.Band(
                    band_type=str(item.get("band_type", "peak")),
                    frequency=float(item.get("frequency", 1000.0)),
                    gain_db=float(item.get("gain_db", 0.0)),
                    q=float(item.get("q", 1.0)),
                )
            )
        self._states: Dict[Tuple[int, int], np.ndarray] = {}
        self._coeff_keys: Dict[int, Tuple[Any, ...]] = {}
        self._coeffs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def reset_state(self) -> None:
        self._states.clear()

    @staticmethod
    def _coefficients(band: "EQNode.Band", sr: int) -> Tuple[np.ndarray, np.ndarray]:
        freq = float(np.clip(band.frequency, 10.0, sr * 0.49))
        q = max(0.1, float(band.q))
        A = 10.0 ** (float(band.gain_db) / 40.0)
        omega = 2.0 * math.pi * freq / sr
        sin_o = math.sin(omega)
        cos_o = math.cos(omega)
        alpha = sin_o / (2.0 * q)
        kind = band.band_type
        if kind == "peak":
            b0, b1, b2 = 1 + alpha * A, -2 * cos_o, 1 - alpha * A
            a0, a1, a2 = 1 + alpha / A, -2 * cos_o, 1 - alpha / A
        else:
            sqrt_a = math.sqrt(max(A, EPS))
            if kind == "low_shelf":
                b0 = A * ((A + 1) - (A - 1) * cos_o + 2 * sqrt_a * alpha)
                b1 = 2 * A * ((A - 1) - (A + 1) * cos_o)
                b2 = A * ((A + 1) - (A - 1) * cos_o - 2 * sqrt_a * alpha)
                a0 = (A + 1) + (A - 1) * cos_o + 2 * sqrt_a * alpha
                a1 = -2 * ((A - 1) + (A + 1) * cos_o)
                a2 = (A + 1) + (A - 1) * cos_o - 2 * sqrt_a * alpha
            elif kind == "high_shelf":
                b0 = A * ((A + 1) + (A - 1) * cos_o + 2 * sqrt_a * alpha)
                b1 = -2 * A * ((A - 1) + (A + 1) * cos_o)
                b2 = A * ((A + 1) + (A - 1) * cos_o - 2 * sqrt_a * alpha)
                a0 = (A + 1) - (A - 1) * cos_o + 2 * sqrt_a * alpha
                a1 = 2 * ((A - 1) - (A + 1) * cos_o)
                a2 = (A + 1) - (A - 1) * cos_o - 2 * sqrt_a * alpha
            else:
                return np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        return (
            np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64),
            np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64),
        )

    def _apply_channel(self, data: np.ndarray, sr: int, channel_index: int) -> np.ndarray:
        out = data.astype(np.float64, copy=True)
        for band_index, band in enumerate(self.bands):
            if abs(band.gain_db) < 0.001:
                continue
            key = (sr, band.band_type, round(band.frequency, 8), round(band.gain_db, 8), round(band.q, 8))
            if self._coeff_keys.get(band_index) != key:
                self._coeff_keys[band_index] = key
                self._coeffs[band_index] = self._coefficients(band, sr)
                # Parameter jumps intentionally reset this band to avoid using
                # state generated by a different transfer function.
                for state_key in [k for k in self._states if k[0] == band_index]:
                    self._states.pop(state_key, None)
            b, a = self._coeffs[band_index]
            state_key = (band_index, channel_index)
            zi = self._states.get(state_key, np.zeros(2, dtype=np.float64))
            if HAS_SCIPY:
                out, zf = lfilter(b, a, out, zi=zi)
            else:
                z1, z2 = float(zi[0]), float(zi[1])
                result = np.empty_like(out)
                for i, x in enumerate(out):
                    y = b[0] * x + z1
                    z1 = b[1] * x - a[1] * y + z2
                    z2 = b[2] * x - a[2] * y
                    result[i] = y
                out, zf = result, np.array([z1, z2])
            self._states[state_key] = np.asarray(zf, dtype=np.float64)
        return np.nan_to_num(out)

    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        data = _finite(audio)
        if self.bypass or not self.bands or data.size == 0:
            return data.copy()
        if data.ndim == 1:
            return self._apply_channel(data, sr, 0)
        return np.column_stack([self._apply_channel(data[:, ch], sr, ch) for ch in range(data.shape[1])])

    def get_params(self) -> Dict[str, float]:
        params: Dict[str, float] = {}
        for i, band in enumerate(self.bands):
            params[f"band{i}_freq"] = band.frequency
            params[f"band{i}_gain"] = band.gain_db
            params[f"band{i}_q"] = band.q
        return params

    def set_params(self, params: Dict[str, float]) -> None:
        for i, band in enumerate(self.bands):
            if f"band{i}_freq" in params:
                band.frequency = max(20.0, min(20000.0, float(params[f"band{i}_freq"])))
            if f"band{i}_gain" in params:
                band.gain_db = max(-24.0, min(24.0, float(params[f"band{i}_gain"])))
            if f"band{i}_q" in params:
                band.q = max(0.1, min(30.0, float(params[f"band{i}_q"])))


class CompressorNode(ProcessingNode):
    """Stateful linked peak compressor with soft knee."""

    def __init__(
        self,
        threshold_db: float = -20.0,
        ratio: float = 4.0,
        attack_ms: float = 5.0,
        release_ms: float = 50.0,
        knee_db: float = 6.0,
        makeup_db: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.threshold_db = float(threshold_db)
        self.ratio = float(ratio)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)
        self.knee_db = float(knee_db)
        self.makeup_db = float(makeup_db)
        self._envelope_db = -120.0

    def reset_state(self) -> None:
        self._envelope_db = -120.0

    def _gain_reduction(self, level_db: float) -> float:
        ratio = max(1.0, self.ratio)
        over = level_db - self.threshold_db
        half = max(0.0, self.knee_db) * 0.5
        if self.knee_db > 0.0 and -half < over < half:
            x = over + half
            return (1.0 - 1.0 / ratio) * x * x / (2.0 * self.knee_db)
        if over > half or (self.knee_db <= 0.0 and over > 0.0):
            return over * (1.0 - 1.0 / ratio)
        return 0.0

    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        data = _finite(audio)
        if self.bypass or data.size == 0:
            return data.copy()
        detector = np.max(np.abs(data), axis=1) if data.ndim > 1 else np.abs(data)
        attack = math.exp(-1.0 / max(1.0, self.attack_ms * sr / 1000.0))
        release = math.exp(-1.0 / max(1.0, self.release_ms * sr / 1000.0))
        env = float(self._envelope_db)
        gains = np.empty(detector.size, dtype=np.float64)
        for i, level in enumerate(detector):
            level_db = max(-120.0, 20.0 * math.log10(max(float(level), EPS)))
            coeff = attack if level_db > env else release
            env = level_db + coeff * (env - level_db)
            reduction = max(0.0, self._gain_reduction(env))
            gains[i] = 10.0 ** ((self.makeup_db - reduction) / 20.0)
        self._envelope_db = env
        if data.ndim == 1:
            return data * gains
        return data * gains[:, None]

    def get_params(self) -> Dict[str, float]:
        return {
            "threshold_db": self.threshold_db,
            "ratio": self.ratio,
            "attack_ms": self.attack_ms,
            "release_ms": self.release_ms,
            "knee_db": self.knee_db,
            "makeup_db": self.makeup_db,
        }

    def set_params(self, params: Dict[str, float]) -> None:
        if "threshold_db" in params:
            self.threshold_db = max(-60.0, min(0.0, float(params["threshold_db"])))
        if "ratio" in params:
            self.ratio = max(1.0, min(20.0, float(params["ratio"])))
        if "attack_ms" in params:
            self.attack_ms = max(0.01, min(200.0, float(params["attack_ms"])))
        if "release_ms" in params:
            self.release_ms = max(1.0, min(5000.0, float(params["release_ms"])))
        if "knee_db" in params:
            self.knee_db = max(0.0, min(24.0, float(params["knee_db"])))
        if "makeup_db" in params:
            self.makeup_db = max(-12.0, min(24.0, float(params["makeup_db"])))


class FaderNode(ProcessingNode):
    def __init__(self, gain_db: float = 0.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.gain_db = float(gain_db)

    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        del sr
        data = _finite(audio)
        if self.bypass:
            return data.copy()
        return data * (10.0 ** (self.gain_db / 20.0))

    def get_params(self) -> Dict[str, float]:
        return {"gain_db": self.gain_db}

    def set_params(self, params: Dict[str, float]) -> None:
        if "gain_db" in params:
            self.gain_db = max(-96.0, min(24.0, float(params["gain_db"])))


class PanNode(ProcessingNode):
    """Constant-power mono panner."""

    def __init__(self, pan: float = 0.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.pan = float(pan)

    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        del sr
        data = _finite(audio)
        if data.ndim > 1:
            # Already stereo/multichannel: leave topology intact.
            return data.copy()
        if self.bypass:
            return np.column_stack([data / math.sqrt(2.0), data / math.sqrt(2.0)])
        angle = (float(np.clip(self.pan, -1.0, 1.0)) + 1.0) * math.pi / 4.0
        return np.column_stack([data * math.cos(angle), data * math.sin(angle)])

    def get_params(self) -> Dict[str, float]:
        return {"pan": self.pan}

    def set_params(self, params: Dict[str, float]) -> None:
        if "pan" in params:
            self.pan = max(-1.0, min(1.0, float(params["pan"])))


class BusSendNode(ProcessingNode):
    def __init__(self, send_level_db: float = -10.0, bus_name: str = "bus1", **kwargs: Any):
        super().__init__(**kwargs)
        self.send_level_db = float(send_level_db)
        self.bus_name = str(bus_name)
        self.last_send: Optional[np.ndarray] = None

    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        del sr
        data = _finite(audio)
        if self.bypass:
            self.last_send = np.zeros_like(data)
        else:
            self.last_send = data * (10.0 ** (self.send_level_db / 20.0))
        return data.copy()

    def reset_state(self) -> None:
        self.last_send = None

    def get_send(self) -> Optional[np.ndarray]:
        return self.last_send

    def get_params(self) -> Dict[str, float]:
        return {"send_level_db": self.send_level_db}

    def set_params(self, params: Dict[str, float]) -> None:
        if "send_level_db" in params:
            self.send_level_db = max(-96.0, min(10.0, float(params["send_level_db"])))


class ProcessingGraph:
    """Ordered channel-strip graph with explicit streaming-state reset."""

    def __init__(self, nodes: Optional[List[ProcessingNode]] = None):
        self.nodes = list(nodes) if nodes is not None else self._default_chain()

    @staticmethod
    def _default_chain() -> List[ProcessingNode]:
        return [
            HPFNode(cutoff_hz=80.0, name="hpf"),
            GateNode(threshold_db=-50.0, name="gate"),
            EQNode(
                bands=[
                    {"band_type": "low_shelf", "frequency": 100.0, "gain_db": 0.0, "q": 0.7},
                    {"band_type": "peak", "frequency": 400.0, "gain_db": 0.0, "q": 1.0},
                    {"band_type": "peak", "frequency": 1000.0, "gain_db": 0.0, "q": 1.0},
                    {"band_type": "peak", "frequency": 4000.0, "gain_db": 0.0, "q": 1.0},
                    {"band_type": "high_shelf", "frequency": 10000.0, "gain_db": 0.0, "q": 0.7},
                ],
                name="eq",
            ),
            CompressorNode(threshold_db=-20.0, ratio=4.0, name="comp"),
            FaderNode(gain_db=0.0, name="fader"),
            PanNode(pan=0.0, name="pan"),
            BusSendNode(send_level_db=-96.0, bus_name="fx1", name="bus_send"),
        ]

    def reset_state(self) -> None:
        for node in self.nodes:
            node.reset_state()

    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        signal = _finite(audio)
        for node in self.nodes:
            signal = node.process(signal, sr)
        return np.nan_to_num(signal)

    def get_node(self, name: str) -> Optional[ProcessingNode]:
        return next((node for node in self.nodes if node.name == name), None)

    def get_params(self) -> Dict[str, Dict[str, float]]:
        return {node.name: node.get_params() for node in self.nodes}

    def set_params(self, params: Dict[str, Dict[str, float]]) -> None:
        for node in self.nodes:
            if node.name in params:
                node.set_params(params[node.name])

    def get_params_flat(self) -> np.ndarray:
        values: List[float] = []
        for node in self.nodes:
            for key in sorted(node.get_params()):
                values.append(float(node.get_params()[key]))
        return np.asarray(values, dtype=np.float64)

    def set_params_flat(self, flat_params: np.ndarray) -> None:
        values = np.asarray(flat_params, dtype=np.float64).reshape(-1)
        index = 0
        for node in self.nodes:
            current = node.get_params()
            update: Dict[str, float] = {}
            for key in sorted(current):
                if index >= values.size:
                    break
                update[key] = float(values[index])
                index += 1
            node.set_params(update)

    def gradient_interface(self):
        params = self.get_params_flat()
        if HAS_TORCH:
            return torch.tensor(params, dtype=torch.float64, requires_grad=True), self
        return params, self

    @staticmethod
    def _mono(output: np.ndarray) -> np.ndarray:
        data = np.asarray(output, dtype=np.float64)
        return np.mean(data, axis=1) if data.ndim > 1 else data

    def _loss(self, params: np.ndarray, input_audio: np.ndarray, target_audio: np.ndarray, sr: int) -> float:
        self.set_params_flat(params)
        self.reset_state()
        output = self._mono(self.process(input_audio, sr))
        target = self._mono(target_audio)
        n = min(output.size, target.size)
        if n == 0:
            return float("inf")
        return float(np.mean(np.square(output[:n] - target[:n])))

    def optimize(
        self,
        target_audio: np.ndarray,
        input_audio: np.ndarray,
        sr: int = 48000,
        lr: float = 0.01,
        steps: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        # Deterministic coordinate descent is used even when torch is present.
        # The graph contains numpy/scipy operators, so pretending gradients flow
        # through a detached numpy render is misleading.
        params = self.get_params_flat()
        best = params.copy()
        best_loss = self._loss(best, input_audio, target_audio, sr)
        step_size = max(float(lr), 1e-5)
        for _ in range(max(0, int(steps))):
            improved = False
            for idx in range(params.size):
                origin = params[idx]
                for direction in (1.0, -1.0):
                    candidate = params.copy()
                    candidate[idx] = origin + direction * step_size
                    loss = self._loss(candidate, input_audio, target_audio, sr)
                    if loss < best_loss:
                        best_loss = loss
                        best = candidate.copy()
                        params = candidate
                        improved = True
                        break
                if not improved:
                    params[idx] = origin
            if not improved:
                step_size *= 0.5
                if step_size < 1e-8:
                    break
        self.set_params_flat(best)
        self.reset_state()
        return self.get_params()

    # Compatibility names retained for callers from earlier revisions.
    def _optimize_numpy(self, target_audio, input_audio, sr, lr, steps):
        return self.optimize(target_audio, input_audio, sr, lr, steps)

    def _optimize_torch(self, target_audio, input_audio, sr, lr, steps):
        return self.optimize(target_audio, input_audio, sr, lr, steps)
