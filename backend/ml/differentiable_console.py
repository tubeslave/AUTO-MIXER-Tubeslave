"""
Differentiable Mixing Console for gradient-based optimization.

Implements a fully differentiable audio processing chain:
- Parametric EQ (biquad filters computed from freq/gain/Q)
- Compressor (soft-knee gain reduction)
- Gain and pan controls

All operations use differentiable math so gradients flow through
the entire console for end-to-end training.
"""

import numpy as np
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    class DifferentiableBiquadEQ(nn.Module):
        """
        Differentiable parametric EQ using biquad filter coefficients.

        Computes biquad coefficients from (frequency, gain_dB, Q) parameters
        and applies the filter in the frequency domain for differentiability.
        """

        def __init__(self, num_bands=4, sr=48000):
            super().__init__()
            self.num_bands = num_bands
            self.sr = sr

        def _compute_biquad_coeffs(self, freq, gain_db, q):
            """
            Compute biquad peaking EQ coefficients from parameters.

            Args:
                freq: (batch, num_bands) center frequencies in Hz
                gain_db: (batch, num_bands) gain in dB
                q: (batch, num_bands) Q factor

            Returns:
                b0, b1, b2, a0, a1, a2: coefficient tensors (batch, num_bands)
            """
            A = torch.pow(10.0, gain_db / 40.0)  # amplitude
            w0 = 2.0 * math.pi * freq / self.sr
            alpha = torch.sin(w0) / (2.0 * q)

            b0 = 1.0 + alpha * A
            b1 = -2.0 * torch.cos(w0)
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * torch.cos(w0)
            a2 = 1.0 - alpha / A

            return b0, b1, b2, a0, a1, a2

        def _apply_biquad_frequency_domain(self, audio, b0, b1, b2, a0, a1, a2):
            """
            Apply biquad filter in frequency domain (differentiable).

            Computes H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
            on the DFT of the signal.

            Args:
                audio: (batch, samples)
                b0..a2: (batch,) scalar coefficients for one band
            Returns:
                filtered: (batch, samples)
            """
            n_fft = audio.shape[-1]
            # Create frequency axis z = exp(j*2*pi*k/N)
            freqs = torch.arange(n_fft // 2 + 1, device=audio.device, dtype=audio.dtype)
            w = 2.0 * math.pi * freqs / n_fft

            # z^-1 = exp(-jw), z^-2 = exp(-j2w)
            z_inv = torch.exp(-1j * w.to(torch.complex64))
            z_inv2 = torch.exp(-2j * w.to(torch.complex64))

            # Compute transfer function for each batch element
            b0 = b0.unsqueeze(-1).to(torch.complex64)
            b1 = b1.unsqueeze(-1).to(torch.complex64)
            b2 = b2.unsqueeze(-1).to(torch.complex64)
            a0 = a0.unsqueeze(-1).to(torch.complex64)
            a1 = a1.unsqueeze(-1).to(torch.complex64)
            a2 = a2.unsqueeze(-1).to(torch.complex64)

            H_num = b0 + b1 * z_inv + b2 * z_inv2
            H_den = a0 + a1 * z_inv + a2 * z_inv2
            H = H_num / (H_den + 1e-8)

            # Apply in frequency domain
            X = torch.fft.rfft(audio)
            Y = X * H
            filtered = torch.fft.irfft(Y, n=n_fft)

            return filtered

        def forward(self, audio, eq_params):
            """
            Apply parametric EQ to audio.

            Args:
                audio: (batch, samples) mono audio
                eq_params: dict with keys:
                    'freq': (batch, num_bands) Hz
                    'gain_db': (batch, num_bands) dB
                    'q': (batch, num_bands) Q factor
            Returns:
                filtered_audio: (batch, samples)
            """
            freq = eq_params["freq"]
            gain_db = eq_params["gain_db"]
            q = eq_params["q"].clamp(min=0.1)  # prevent division by zero

            b0, b1, b2, a0, a1, a2 = self._compute_biquad_coeffs(freq, gain_db, q)

            result = audio
            for band in range(self.num_bands):
                result = self._apply_biquad_frequency_domain(
                    result,
                    b0[:, band], b1[:, band], b2[:, band],
                    a0[:, band], a1[:, band], a2[:, band],
                )
            return result

    class DifferentiableCompressor(nn.Module):
        """
        Differentiable compressor with soft-knee gain reduction.

        Uses a smooth approximation of the compressor characteristic
        to maintain differentiability.
        """

        def __init__(self, sr=48000, frame_size=256):
            super().__init__()
            self.sr = sr
            self.frame_size = frame_size

        def _compute_envelope(self, audio, attack_samples, release_samples):
            """
            Differentiable envelope follower using exponential smoothing.

            Args:
                audio: (batch, samples)
                attack_samples: (batch, 1) attack time in samples
                release_samples: (batch, 1) release time in samples
            Returns:
                envelope: (batch, samples)
            """
            # Compute per-frame RMS
            batch, n_samples = audio.shape
            n_frames = n_samples // self.frame_size
            trimmed = audio[:, : n_frames * self.frame_size]
            frames = trimmed.reshape(batch, n_frames, self.frame_size)
            rms = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-8)

            # Smooth attack/release coefficients
            attack_coeff = torch.exp(-1.0 / (attack_samples / self.frame_size + 1e-8))
            release_coeff = torch.exp(-1.0 / (release_samples / self.frame_size + 1e-8))

            # Envelope follower (differentiable via smooth interpolation)
            envelope = torch.zeros_like(rms)
            envelope[:, 0] = rms[:, 0]
            for i in range(1, n_frames):
                # Smooth blend between attack and release
                is_attack = torch.sigmoid(100.0 * (rms[:, i] - envelope[:, i - 1]))
                coeff = is_attack * attack_coeff.squeeze(-1) + (
                    1.0 - is_attack
                ) * release_coeff.squeeze(-1)
                envelope[:, i] = coeff * envelope[:, i - 1] + (1.0 - coeff) * rms[:, i]

            # Upsample envelope back to sample rate
            envelope_upsampled = envelope.unsqueeze(-1).expand(
                batch, n_frames, self.frame_size
            ).reshape(batch, n_frames * self.frame_size)

            # Pad to original length
            if envelope_upsampled.shape[-1] < n_samples:
                pad_len = n_samples - envelope_upsampled.shape[-1]
                envelope_upsampled = F.pad(
                    envelope_upsampled, (0, pad_len), mode="constant",
                    value=0.0,
                )
                # Fill padded region with last value
                envelope_upsampled[:, -pad_len:] = envelope_upsampled[
                    :, -(pad_len + 1) : -(pad_len + 1) + 1
                ].expand(batch, pad_len)

            return envelope_upsampled

        def forward(self, audio, comp_params):
            """
            Apply differentiable compression to audio.

            Args:
                audio: (batch, samples)
                comp_params: dict with keys:
                    'threshold_db': (batch, 1) threshold in dB
                    'ratio': (batch, 1) compression ratio (e.g. 4.0 = 4:1)
                    'attack_ms': (batch, 1) attack time in ms
                    'release_ms': (batch, 1) release time in ms
                    'knee_db': (batch, 1) soft knee width in dB (optional, default 6)
                    'makeup_db': (batch, 1) makeup gain in dB (optional, default 0)
            Returns:
                compressed_audio: (batch, samples)
            """
            threshold_db = comp_params["threshold_db"]
            ratio = comp_params["ratio"].clamp(min=1.0)
            attack_ms = comp_params["attack_ms"].clamp(min=0.1)
            release_ms = comp_params["release_ms"].clamp(min=1.0)
            knee_db = comp_params.get("knee_db", torch.full_like(threshold_db, 6.0))
            makeup_db = comp_params.get("makeup_db", torch.zeros_like(threshold_db))

            attack_samples = (attack_ms / 1000.0) * self.sr
            release_samples = (release_ms / 1000.0) * self.sr

            envelope = self._compute_envelope(audio, attack_samples, release_samples)
            envelope_db = 20.0 * torch.log10(envelope + 1e-8)

            # Soft-knee gain computation (differentiable)
            half_knee = knee_db / 2.0
            below_knee = envelope_db - threshold_db + half_knee
            above_knee = envelope_db - threshold_db - half_knee

            # Region below knee: no compression
            # Region in knee: quadratic transition
            # Region above knee: full compression
            # Using smooth approximation with softplus
            gain_reduction_db = torch.where(
                envelope_db < (threshold_db - half_knee),
                torch.zeros_like(envelope_db),
                torch.where(
                    envelope_db > (threshold_db + half_knee),
                    # Full compression above knee
                    (1.0 - 1.0 / ratio) * (envelope_db - threshold_db),
                    # Soft knee region: quadratic
                    (1.0 - 1.0 / ratio)
                    * (envelope_db - threshold_db + half_knee) ** 2
                    / (2.0 * knee_db + 1e-8),
                ),
            )

            # Apply gain reduction + makeup
            gain_db = -gain_reduction_db + makeup_db
            gain_linear = torch.pow(10.0, gain_db / 20.0)
            compressed = audio * gain_linear

            return compressed

    class DifferentiableGainPan(nn.Module):
        """Differentiable gain and stereo pan control."""

        def __init__(self):
            super().__init__()

        def forward(self, audio, gain_db, pan):
            """
            Apply gain and pan to audio.

            Args:
                audio: (batch, samples) mono input
                gain_db: (batch, 1) gain in dB
                pan: (batch, 1) pan position -1 (left) to +1 (right)
            Returns:
                stereo_audio: (batch, 2, samples)
            """
            gain_linear = torch.pow(10.0, gain_db / 20.0)
            gained = audio * gain_linear

            # Constant-power panning law
            pan_norm = (pan + 1.0) / 2.0  # 0 to 1
            left_gain = torch.cos(pan_norm * math.pi / 2.0)
            right_gain = torch.sin(pan_norm * math.pi / 2.0)

            left = gained * left_gain
            right = gained * right_gain

            return torch.stack([left, right], dim=1)

    class DifferentiableMixingConsole(nn.Module):
        """
        Full differentiable mixing console for gradient-based mix optimization.

        Signal chain: Input -> EQ -> Compressor -> Gain/Pan -> Output
        All parameters are differentiable, enabling end-to-end training.
        """

        def __init__(self, num_channels=32, num_eq_bands=4, sr=48000):
            super().__init__()
            self.num_channels = num_channels
            self.num_eq_bands = num_eq_bands
            self.sr = sr

            self.eq = DifferentiableBiquadEQ(num_bands=num_eq_bands, sr=sr)
            self.compressor = DifferentiableCompressor(sr=sr)
            self.gain_pan = DifferentiableGainPan()

        def forward(self, audio, params):
            """
            Process multi-channel audio through the mixing console.

            Args:
                audio: (batch, num_channels, samples) multi-channel input
                params: dict of parameter tensors:
                    'eq_freq': (batch, num_channels, num_eq_bands) Hz
                    'eq_gain': (batch, num_channels, num_eq_bands) dB
                    'eq_q': (batch, num_channels, num_eq_bands)
                    'comp_threshold': (batch, num_channels, 1) dB
                    'comp_ratio': (batch, num_channels, 1)
                    'comp_attack': (batch, num_channels, 1) ms
                    'comp_release': (batch, num_channels, 1) ms
                    'gain_db': (batch, num_channels, 1) dB
                    'pan': (batch, num_channels, 1) -1..+1
            Returns:
                mix: (batch, 2, samples) stereo mix output
            """
            batch, n_ch, n_samples = audio.shape

            # Accumulate stereo mix
            mix = torch.zeros(batch, 2, n_samples, device=audio.device, dtype=audio.dtype)

            for ch in range(n_ch):
                ch_audio = audio[:, ch, :]  # (batch, samples)

                # EQ
                eq_params = {
                    "freq": params["eq_freq"][:, ch, :],
                    "gain_db": params["eq_gain"][:, ch, :],
                    "q": params["eq_q"][:, ch, :],
                }
                ch_audio = self.eq(ch_audio, eq_params)

                # Compressor
                comp_params = {
                    "threshold_db": params["comp_threshold"][:, ch, :],
                    "ratio": params["comp_ratio"][:, ch, :],
                    "attack_ms": params["comp_attack"][:, ch, :],
                    "release_ms": params["comp_release"][:, ch, :],
                }
                ch_audio = self.compressor(ch_audio, comp_params)

                # Gain and pan to stereo
                stereo = self.gain_pan(
                    ch_audio,
                    params["gain_db"][:, ch, :],
                    params["pan"][:, ch, :],
                )

                mix = mix + stereo

            return mix

        def create_default_params(self, batch_size, num_channels=None, device=None):
            """Create default mixing parameters (unity gain, center pan, flat EQ)."""
            n_ch = num_channels or self.num_channels
            dev = device or "cpu"

            return {
                "eq_freq": torch.tensor(
                    [100.0, 500.0, 2000.0, 8000.0], device=dev
                ).unsqueeze(0).unsqueeze(0).expand(batch_size, n_ch, self.num_eq_bands),
                "eq_gain": torch.zeros(batch_size, n_ch, self.num_eq_bands, device=dev),
                "eq_q": torch.ones(batch_size, n_ch, self.num_eq_bands, device=dev) * 0.707,
                "comp_threshold": torch.full((batch_size, n_ch, 1), -20.0, device=dev),
                "comp_ratio": torch.full((batch_size, n_ch, 1), 2.0, device=dev),
                "comp_attack": torch.full((batch_size, n_ch, 1), 10.0, device=dev),
                "comp_release": torch.full((batch_size, n_ch, 1), 100.0, device=dev),
                "gain_db": torch.zeros(batch_size, n_ch, 1, device=dev),
                "pan": torch.zeros(batch_size, n_ch, 1, device=dev),
            }

else:
    # Numpy-only fallback (non-differentiable, for evaluation / inference)

    class DifferentiableMixingConsole:
        """
        Numpy fallback mixing console (non-differentiable).
        Provides the same signal chain for evaluation without PyTorch.
        """

        def __init__(self, num_channels=32, num_eq_bands=4, sr=48000):
            self.num_channels = num_channels
            self.num_eq_bands = num_eq_bands
            self.sr = sr

        def _apply_biquad(self, audio, freq, gain_db, q):
            """Apply a single biquad peaking EQ band in frequency domain."""
            from scipy.signal import sosfilt, iirpeak

            if abs(gain_db) < 0.01:
                return audio
            try:
                w0 = freq / (self.sr / 2.0)
                w0 = np.clip(w0, 0.001, 0.999)
                b, a = iirpeak(w0, q)
                # Scale by gain
                A = 10.0 ** (gain_db / 40.0)
                b = b * A
                from scipy.signal import lfilter
                return lfilter(b, a, audio)
            except Exception:
                return audio

        def _apply_compression(self, audio, threshold_db, ratio, attack_ms, release_ms):
            """Apply simple compression using envelope following."""
            if ratio <= 1.0:
                return audio

            # RMS envelope
            frame_size = max(1, int(self.sr * 0.005))  # 5ms frames
            n_frames = len(audio) // frame_size
            if n_frames == 0:
                return audio

            envelope = np.zeros(len(audio))
            for i in range(n_frames):
                start = i * frame_size
                end = start + frame_size
                rms = np.sqrt(np.mean(audio[start:end] ** 2) + 1e-8)
                envelope[start:end] = rms

            # Remainder
            if n_frames * frame_size < len(audio):
                start = n_frames * frame_size
                rms = np.sqrt(np.mean(audio[start:] ** 2) + 1e-8)
                envelope[start:] = rms

            envelope_db = 20.0 * np.log10(envelope + 1e-8)

            # Gain reduction
            over_threshold = np.maximum(envelope_db - threshold_db, 0.0)
            gain_reduction_db = over_threshold * (1.0 - 1.0 / ratio)
            gain_linear = 10.0 ** (-gain_reduction_db / 20.0)

            return audio * gain_linear

        def _apply_gain_pan(self, audio, gain_db, pan):
            """Apply gain and constant-power pan, return (left, right)."""
            gain_linear = 10.0 ** (gain_db / 20.0)
            gained = audio * gain_linear
            pan_norm = (pan + 1.0) / 2.0  # 0..1
            left_gain = np.cos(pan_norm * np.pi / 2.0)
            right_gain = np.sin(pan_norm * np.pi / 2.0)
            return gained * left_gain, gained * right_gain

        def forward(self, audio, params):
            """
            Process multi-channel audio.

            Args:
                audio: (num_channels, samples) numpy array
                params: dict with numpy arrays for EQ, comp, gain, pan
            Returns:
                mix: (2, samples) stereo numpy array
            """
            n_ch, n_samples = audio.shape
            mix = np.zeros((2, n_samples))

            for ch in range(n_ch):
                ch_audio = audio[ch].copy()

                # Apply EQ bands
                for band in range(self.num_eq_bands):
                    freq = params["eq_freq"][ch, band]
                    gain = params["eq_gain"][ch, band]
                    q = params["eq_q"][ch, band]
                    ch_audio = self._apply_biquad(ch_audio, freq, gain, q)

                # Apply compression
                ch_audio = self._apply_compression(
                    ch_audio,
                    params["comp_threshold"][ch, 0],
                    params["comp_ratio"][ch, 0],
                    params["comp_attack"][ch, 0],
                    params["comp_release"][ch, 0],
                )

                # Apply gain and pan
                left, right = self._apply_gain_pan(
                    ch_audio,
                    params["gain_db"][ch, 0],
                    params["pan"][ch, 0],
                )

                mix[0] += left
                mix[1] += right

            return mix

        def __call__(self, audio, params):
            return self.forward(audio, params)

        def create_default_params(self, num_channels=None):
            """Create default mixing parameters as numpy arrays."""
            n_ch = num_channels or self.num_channels
            return {
                "eq_freq": np.tile([100.0, 500.0, 2000.0, 8000.0], (n_ch, 1)),
                "eq_gain": np.zeros((n_ch, self.num_eq_bands)),
                "eq_q": np.full((n_ch, self.num_eq_bands), 0.707),
                "comp_threshold": np.full((n_ch, 1), -20.0),
                "comp_ratio": np.full((n_ch, 1), 2.0),
                "comp_attack": np.full((n_ch, 1), 10.0),
                "comp_release": np.full((n_ch, 1), 100.0),
                "gain_db": np.zeros((n_ch, 1)),
                "pan": np.zeros((n_ch, 1)),
            }
