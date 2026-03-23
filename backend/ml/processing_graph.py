"""
Processing Graph - Modular audio processing chain
===================================================
Defines a directed processing graph of audio nodes (HPF, Gate, EQ, Compressor,
Fader, Pan, BusSend) with configurable parameters.  Each node processes audio
in numpy.  An optional gradient-based interface (via PyTorch) allows optimizing
the graph parameters to match a target output.
"""

import abc
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from scipy.signal import butter, sosfilt, lfilter

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


# ============================================================================
# Base processing node
# ============================================================================


class ProcessingNode(abc.ABC):
    """
    Abstract base class for all processing nodes in the graph.
    Each node takes mono audio (numpy float64) and returns processed audio.
    """

    def __init__(self, name: str = "", bypass: bool = False):
        self.name = name or self.__class__.__name__
        self.bypass = bypass

    @abc.abstractmethod
    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        """Process audio through this node.

        Args:
            audio: Mono audio signal as float64 numpy array.
            sr: Sample rate in Hz.

        Returns:
            Processed audio as float64 numpy array.
        """
        ...

    def get_params(self) -> Dict[str, float]:
        """Return current parameters as a dict."""
        return {}

    def set_params(self, params: Dict[str, float]) -> None:
        """Set parameters from a dict."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, bypass={self.bypass})"


# ============================================================================
# Concrete processing nodes
# ============================================================================


class HPFNode(ProcessingNode):
    """High-pass filter node using a second-order Butterworth filter."""

    def __init__(self, cutoff_hz: float = 80.0, order: int = 2, **kwargs: Any):
        super().__init__(**kwargs)
        self.cutoff_hz = cutoff_hz
        self.order = order

    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        if self.bypass or self.cutoff_hz <= 0:
            return audio

        nyquist = sr / 2.0
        normalized_cutoff = self.cutoff_hz / nyquist
        if normalized_cutoff >= 1.0 or normalized_cutoff <= 0.0:
            return audio

        if HAS_SCIPY:
            sos = butter(self.order, normalized_cutoff, btype="high", output="sos")
            return sosfilt(sos, audio)
        else:
            # Manual single-pole HPF fallback
            rc = 1.0 / (2.0 * math.pi * self.cutoff_hz)
            dt = 1.0 / sr
            alpha = rc / (rc + dt)
            output = np.zeros_like(audio)
            if len(audio) > 0:
                output[0] = audio[0]
            for i in range(1, len(audio)):
                output[i] = alpha * (output[i - 1] + audio[i] - audio[i - 1])
            return output

    def get_params(self) -> Dict[str, float]:
        return {"cutoff_hz": self.cutoff_hz, "order": float(self.order)}

    def set_params(self, params: Dict[str, float]) -> None:
        if "cutoff_hz" in params:
            self.cutoff_hz = max(20.0, min(2000.0, params["cutoff_hz"]))
        if "order" in params:
            self.order = max(1, min(8, int(params["order"])))


class GateNode(ProcessingNode):
    """Noise gate with threshold, attack, hold, and release."""

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
        self.threshold_db = threshold_db
        self.attack_ms = attack_ms
        self.hold_ms = hold_ms
        self.release_ms = release_ms
        self.range_db = range_db

    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        if self.bypass:
            return audio

        threshold_linear = 10.0 ** (self.threshold_db / 20.0)
        range_linear = 10.0 ** (self.range_db / 20.0)

        attack_samples = max(1, int(self.attack_ms * sr / 1000.0))
        hold_samples = max(1, int(self.hold_ms * sr / 1000.0))
        release_samples = max(1, int(self.release_ms * sr / 1000.0))

        # Compute envelope
        envelope = np.abs(audio)

        # Smooth envelope with a small window
        window = max(1, int(0.002 * sr))  # 2ms
        if window > 1 and len(envelope) > window:
            kernel = np.ones(window) / window
            envelope = np.convolve(envelope, kernel, mode="same")

        output = audio.copy()
        gate_gain = range_linear  # Start closed
        hold_counter = 0

        for i in range(len(audio)):
            if envelope[i] >= threshold_linear:
                hold_counter = hold_samples
                # Attack: ramp up
                target = 1.0
                coeff = 1.0 / attack_samples
            elif hold_counter > 0:
                hold_counter -= 1
                target = 1.0
                coeff = 0.0  # Hold steady
            else:
                # Release: ramp down
                target = range_linear
                coeff = 1.0 / release_samples

            if coeff > 0:
                gate_gain += coeff * (target - gate_gain)

            gate_gain = max(range_linear, min(1.0, gate_gain))
            output[i] = audio[i] * gate_gain

        return output

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
            self.threshold_db = max(-96.0, min(0.0, params["threshold_db"]))
        if "attack_ms" in params:
            self.attack_ms = max(0.01, min(100.0, params["attack_ms"]))
        if "hold_ms" in params:
            self.hold_ms = max(0.0, min(2000.0, params["hold_ms"]))
        if "release_ms" in params:
            self.release_ms = max(1.0, min(5000.0, params["release_ms"]))
        if "range_db" in params:
            self.range_db = max(-96.0, min(0.0, params["range_db"]))


class EQNode(ProcessingNode):
    """
    Parametric EQ node with configurable bands.
    Each band is a biquad filter (peak, low_shelf, or high_shelf).
    """

    @dataclass
    class Band:
        band_type: str = "peak"  # "peak", "low_shelf", "high_shelf"
        frequency: float = 1000.0
        gain_db: float = 0.0
        q: float = 1.0

    def __init__(self, bands: Optional[List[Dict[str, Any]]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.bands: List[EQNode.Band] = []
        if bands:
            for b in bands:
                self.bands.append(
                    EQNode.Band(
                        band_type=b.get("band_type", "peak"),
                        frequency=b.get("frequency", 1000.0),
                        gain_db=b.get("gain_db", 0.0),
                        q=b.get("q", 1.0),
                    )
                )

    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        if self.bypass or not self.bands:
            return audio

        output = audio.copy()
        for band in self.bands:
            if abs(band.gain_db) < 0.01:
                continue
            output = self._apply_biquad(output, band, sr)
        return output

    def _apply_biquad(self, audio: np.ndarray, band: "EQNode.Band", sr: int) -> np.ndarray:
        """Apply a biquad filter for a single EQ band."""
        A = 10.0 ** (band.gain_db / 40.0)
        omega = 2.0 * math.pi * band.frequency / sr
        sin_omega = math.sin(omega)
        cos_omega = math.cos(omega)
        alpha = sin_omega / (2.0 * max(0.1, band.q))

        if band.band_type == "peak":
            b0 = 1.0 + alpha * A
            b1 = -2.0 * cos_omega
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * cos_omega
            a2 = 1.0 - alpha / A
        elif band.band_type == "low_shelf":
            sqrt_a = math.sqrt(max(0.001, A))
            b0 = A * ((A + 1) - (A - 1) * cos_omega + 2 * sqrt_a * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * cos_omega)
            b2 = A * ((A + 1) - (A - 1) * cos_omega - 2 * sqrt_a * alpha)
            a0 = (A + 1) + (A - 1) * cos_omega + 2 * sqrt_a * alpha
            a1 = -2 * ((A - 1) + (A + 1) * cos_omega)
            a2 = (A + 1) + (A - 1) * cos_omega - 2 * sqrt_a * alpha
        elif band.band_type == "high_shelf":
            sqrt_a = math.sqrt(max(0.001, A))
            b0 = A * ((A + 1) + (A - 1) * cos_omega + 2 * sqrt_a * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cos_omega)
            b2 = A * ((A + 1) + (A - 1) * cos_omega - 2 * sqrt_a * alpha)
            a0 = (A + 1) - (A - 1) * cos_omega + 2 * sqrt_a * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cos_omega)
            a2 = (A + 1) - (A - 1) * cos_omega - 2 * sqrt_a * alpha
        else:
            return audio

        # Normalize
        b = np.array([b0 / a0, b1 / a0, b2 / a0])
        a = np.array([1.0, a1 / a0, a2 / a0])

        if HAS_SCIPY:
            return lfilter(b, a, audio)
        else:
            # Manual Direct Form II transposed
            output = np.zeros_like(audio)
            z1, z2 = 0.0, 0.0
            for i in range(len(audio)):
                x = audio[i]
                y = b[0] * x + z1
                z1 = b[1] * x - a[1] * y + z2
                z2 = b[2] * x - a[2] * y
                output[i] = y
            return output

    def get_params(self) -> Dict[str, float]:
        params = {}
        for i, band in enumerate(self.bands):
            params[f"band{i}_freq"] = band.frequency
            params[f"band{i}_gain"] = band.gain_db
            params[f"band{i}_q"] = band.q
        return params

    def set_params(self, params: Dict[str, float]) -> None:
        for i, band in enumerate(self.bands):
            key_freq = f"band{i}_freq"
            key_gain = f"band{i}_gain"
            key_q = f"band{i}_q"
            if key_freq in params:
                band.frequency = max(20.0, min(20000.0, params[key_freq]))
            if key_gain in params:
                band.gain_db = max(-24.0, min(24.0, params[key_gain]))
            if key_q in params:
                band.q = max(0.1, min(30.0, params[key_q]))


class CompressorNode(ProcessingNode):
    """
    Dynamic range compressor with threshold, ratio, knee, attack, release,
    and makeup gain.
    """

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
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.knee_db = knee_db
        self.makeup_db = makeup_db

    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        if self.bypass:
            return audio

        eps = 1e-10
        attack_coeff = math.exp(-1.0 / max(1, self.attack_ms * sr / 1000.0))
        release_coeff = math.exp(-1.0 / max(1, self.release_ms * sr / 1000.0))

        output = np.zeros_like(audio)
        envelope_db = -96.0
        half_knee = self.knee_db / 2.0

        for i in range(len(audio)):
            # Level detection
            level = abs(audio[i])
            if level > eps:
                level_db = 20.0 * math.log10(level)
            else:
                level_db = -96.0

            # Envelope follower
            if level_db > envelope_db:
                envelope_db = attack_coeff * envelope_db + (1.0 - attack_coeff) * level_db
            else:
                envelope_db = release_coeff * envelope_db + (1.0 - release_coeff) * level_db

            # Gain computation with soft knee
            overshoot = envelope_db - self.threshold_db

            if self.knee_db > 0.01 and abs(overshoot) < half_knee:
                # Soft knee region
                knee_factor = (overshoot + half_knee) / self.knee_db
                gain_reduction = (1.0 - 1.0 / self.ratio) * (overshoot + half_knee) ** 2 / (2.0 * self.knee_db)
            elif overshoot > 0:
                # Above threshold
                gain_reduction = overshoot * (1.0 - 1.0 / self.ratio)
            else:
                gain_reduction = 0.0

            # Apply gain reduction + makeup
            total_gain_db = -gain_reduction + self.makeup_db
            gain_linear = 10.0 ** (total_gain_db / 20.0)
            output[i] = audio[i] * gain_linear

        return output

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
            self.threshold_db = max(-60.0, min(0.0, params["threshold_db"]))
        if "ratio" in params:
            self.ratio = max(1.0, min(20.0, params["ratio"]))
        if "attack_ms" in params:
            self.attack_ms = max(0.01, min(200.0, params["attack_ms"]))
        if "release_ms" in params:
            self.release_ms = max(1.0, min(5000.0, params["release_ms"]))
        if "knee_db" in params:
            self.knee_db = max(0.0, min(24.0, params["knee_db"]))
        if "makeup_db" in params:
            self.makeup_db = max(-12.0, min(24.0, params["makeup_db"]))


class FaderNode(ProcessingNode):
    """Simple gain / fader node."""

    def __init__(self, gain_db: float = 0.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.gain_db = gain_db

    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        if self.bypass:
            return audio
        gain_linear = 10.0 ** (self.gain_db / 20.0)
        return audio * gain_linear

    def get_params(self) -> Dict[str, float]:
        return {"gain_db": self.gain_db}

    def set_params(self, params: Dict[str, float]) -> None:
        if "gain_db" in params:
            self.gain_db = max(-96.0, min(24.0, params["gain_db"]))


class PanNode(ProcessingNode):
    """
    Panning node: takes mono input and returns stereo (N, 2).
    Uses constant-power panning law.
    """

    def __init__(self, pan: float = 0.0, **kwargs: Any):
        """
        Args:
            pan: Pan position from -1.0 (full left) to 1.0 (full right).
        """
        super().__init__(**kwargs)
        self.pan = pan

    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        if self.bypass:
            # Return stereo centered
            return np.column_stack([audio, audio])

        # Constant power panning
        # pan: -1 = left, 0 = center, 1 = right
        angle = (self.pan + 1.0) * math.pi / 4.0  # 0..pi/2
        left_gain = math.cos(angle)
        right_gain = math.sin(angle)

        left = audio * left_gain
        right = audio * right_gain

        return np.column_stack([left, right])

    def get_params(self) -> Dict[str, float]:
        return {"pan": self.pan}

    def set_params(self, params: Dict[str, float]) -> None:
        if "pan" in params:
            self.pan = max(-1.0, min(1.0, params["pan"]))


class BusSendNode(ProcessingNode):
    """
    Bus send node: creates a copy of the signal at a specified level.
    Returns the original signal unchanged; the send signal is stored
    for retrieval.
    """

    def __init__(self, send_level_db: float = -10.0, bus_name: str = "bus1", **kwargs: Any):
        super().__init__(**kwargs)
        self.send_level_db = send_level_db
        self.bus_name = bus_name
        self.last_send: Optional[np.ndarray] = None

    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        if self.bypass:
            self.last_send = np.zeros_like(audio)
            return audio

        send_gain = 10.0 ** (self.send_level_db / 20.0)
        self.last_send = audio * send_gain
        return audio  # Pass through unchanged

    def get_send(self) -> Optional[np.ndarray]:
        """Retrieve the last computed send signal."""
        return self.last_send

    def get_params(self) -> Dict[str, float]:
        return {"send_level_db": self.send_level_db}

    def set_params(self, params: Dict[str, float]) -> None:
        if "send_level_db" in params:
            self.send_level_db = max(-96.0, min(10.0, params["send_level_db"]))


# ============================================================================
# Processing Graph
# ============================================================================


class ProcessingGraph:
    """
    A chain of ProcessingNodes forming a complete channel strip.

    Default chain: HPF -> Gate -> EQ -> Compressor -> Fader -> Pan -> BusSend

    Supports gradient-based optimization when PyTorch is available.
    """

    def __init__(self, nodes: Optional[List[ProcessingNode]] = None):
        """
        Args:
            nodes: List of ProcessingNode instances. If None, creates a
                   default channel strip chain.
        """
        if nodes is not None:
            self.nodes = nodes
        else:
            self.nodes = self._default_chain()

    @staticmethod
    def _default_chain() -> List[ProcessingNode]:
        """Create the default processing chain: HPF->Gate->EQ->Comp->Fader->Pan->BusSend."""
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

    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        """
        Run audio through the entire processing chain.

        Args:
            audio: Input mono audio (float64).
            sr: Sample rate.

        Returns:
            Processed audio. May be stereo (N, 2) if a PanNode is in the chain.
        """
        signal = audio.astype(np.float64)
        for node in self.nodes:
            signal = node.process(signal, sr)
        return signal

    def get_node(self, name: str) -> Optional[ProcessingNode]:
        """Find a node by name."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_params(self) -> Dict[str, Dict[str, float]]:
        """Get all parameters from all nodes."""
        return {node.name: node.get_params() for node in self.nodes}

    def set_params(self, params: Dict[str, Dict[str, float]]) -> None:
        """Set parameters for nodes by name."""
        for node in self.nodes:
            if node.name in params:
                node.set_params(params[node.name])

    def get_params_flat(self) -> np.ndarray:
        """
        Flatten all parameters into a single numpy array.
        Used as the interface for gradient-based optimization.

        Returns:
            1D numpy array of all parameter values.
        """
        values = []
        for node in self.nodes:
            p = node.get_params()
            for key in sorted(p.keys()):
                values.append(p[key])
        return np.array(values, dtype=np.float64)

    def set_params_flat(self, flat_params: np.ndarray) -> None:
        """
        Set all parameters from a flattened array.

        Args:
            flat_params: 1D array matching the shape from get_params_flat().
        """
        idx = 0
        for node in self.nodes:
            p = node.get_params()
            new_params = {}
            for key in sorted(p.keys()):
                if idx < len(flat_params):
                    new_params[key] = float(flat_params[idx])
                    idx += 1
            node.set_params(new_params)

    def gradient_interface(self) -> Tuple[np.ndarray, "ProcessingGraph"]:
        """
        Return (params_tensor, self) for optimization.

        When PyTorch is available, returns a torch tensor with requires_grad.
        Otherwise returns a numpy array.
        """
        params = self.get_params_flat()
        if HAS_TORCH:
            tensor = torch.tensor(params, dtype=torch.float64, requires_grad=True)
            return tensor, self
        else:
            return params, self

    def optimize(
        self,
        target_audio: np.ndarray,
        input_audio: np.ndarray,
        sr: int = 48000,
        lr: float = 0.01,
        steps: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        """
        Optimize processing parameters to make the graph output match a target.

        Uses gradient descent when PyTorch is available; otherwise falls back to
        coordinate descent (Nelder-Mead-style perturbation).

        Args:
            target_audio: Target output audio (mono, float64).
            input_audio: Input audio to process.
            sr: Sample rate.
            lr: Learning rate (for gradient descent) or step size (for coordinate descent).
            steps: Number of optimization iterations.

        Returns:
            Optimized parameters dict.
        """
        if HAS_TORCH:
            return self._optimize_torch(target_audio, input_audio, sr, lr, steps)
        else:
            return self._optimize_numpy(target_audio, input_audio, sr, lr, steps)

    def _optimize_torch(
        self,
        target_audio: np.ndarray,
        input_audio: np.ndarray,
        sr: int,
        lr: float,
        steps: int,
    ) -> Dict[str, Dict[str, float]]:
        """Gradient-based optimization using PyTorch."""
        target = torch.tensor(target_audio[:, 0] if target_audio.ndim == 2 else target_audio, dtype=torch.float64)

        params_np = self.get_params_flat()
        params_tensor = torch.tensor(params_np, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([params_tensor], lr=lr)

        best_loss = float("inf")
        best_params = params_np.copy()

        for step in range(steps):
            optimizer.zero_grad()

            # Set params from tensor (detach for numpy processing)
            self.set_params_flat(params_tensor.detach().numpy())

            # Forward pass through graph
            output = self.process(input_audio, sr)
            if output.ndim == 2:
                output_mono = output[:, 0]
            else:
                output_mono = output

            # Compute loss: MSE + spectral loss
            min_len = min(len(output_mono), len(target))
            output_t = torch.tensor(output_mono[:min_len], dtype=torch.float64)
            target_t = target[:min_len]

            loss = torch.nn.functional.mse_loss(output_t, target_t)

            # Numerical gradient via finite differences
            grad = torch.zeros_like(params_tensor)
            delta = 1e-4
            base_loss_val = loss.item()

            for i in range(len(params_tensor)):
                perturbed = params_tensor.detach().clone()
                perturbed[i] += delta
                self.set_params_flat(perturbed.numpy())
                out_p = self.process(input_audio, sr)
                if out_p.ndim == 2:
                    out_p = out_p[:, 0]
                out_pt = torch.tensor(out_p[:min_len], dtype=torch.float64)
                loss_p = torch.nn.functional.mse_loss(out_pt, target_t).item()
                grad[i] = (loss_p - base_loss_val) / delta

            # Update params
            with torch.no_grad():
                params_tensor -= lr * grad

            current_loss = base_loss_val
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = params_tensor.detach().numpy().copy()

            if step % 20 == 0:
                logger.debug(f"Optimization step {step}: loss={current_loss:.6f}")

        self.set_params_flat(best_params)
        logger.info(f"Optimization complete: best loss={best_loss:.6f}")
        return self.get_params()

    def _optimize_numpy(
        self,
        target_audio: np.ndarray,
        input_audio: np.ndarray,
        sr: int,
        lr: float,
        steps: int,
    ) -> Dict[str, Dict[str, float]]:
        """Coordinate descent optimization using only numpy."""
        target_mono = target_audio[:, 0] if target_audio.ndim == 2 else target_audio

        params = self.get_params_flat()
        best_params = params.copy()

        def compute_loss(p: np.ndarray) -> float:
            self.set_params_flat(p)
            output = self.process(input_audio, sr)
            out_mono = output[:, 0] if output.ndim == 2 else output
            min_len = min(len(out_mono), len(target_mono))
            return float(np.mean((out_mono[:min_len] - target_mono[:min_len]) ** 2))

        best_loss = compute_loss(params)
        delta = lr

        for step in range(steps):
            improved = False
            for i in range(len(params)):
                # Try positive perturbation
                params[i] += delta
                loss_plus = compute_loss(params)
                if loss_plus < best_loss:
                    best_loss = loss_plus
                    best_params = params.copy()
                    improved = True
                    continue

                # Try negative perturbation
                params[i] -= 2 * delta
                loss_minus = compute_loss(params)
                if loss_minus < best_loss:
                    best_loss = loss_minus
                    best_params = params.copy()
                    improved = True
                    continue

                # Revert
                params[i] += delta

            if not improved:
                delta *= 0.5
                if delta < 1e-8:
                    break

            if step % 20 == 0:
                logger.debug(f"Optimization step {step}: loss={best_loss:.6f}, delta={delta:.6f}")

        self.set_params_flat(best_params)
        logger.info(f"Numpy optimization complete: best loss={best_loss:.6f}")
        return self.get_params()
