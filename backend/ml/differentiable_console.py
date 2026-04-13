"""
Differentiable mixing console -- allows gradient-based optimization of mix parameters.
Based on Steinmetz et al. "Automatic Multitrack Mixing with a Differentiable Mixing Console" (2020).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class DifferentiableGain(nn.Module):
    """Differentiable gain stage with dB parameterization."""

    def __init__(self, n_channels: int, init_db: float = -6.0):
        super().__init__()
        self.gain_db = nn.Parameter(torch.full((n_channels,), init_db))

    def forward(self, x):
        gain_linear = 10 ** (self.gain_db / 20.0)
        return x * gain_linear.unsqueeze(-1)


class DifferentiablePan(nn.Module):
    """Differentiable stereo panner using constant-power law."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.pan = nn.Parameter(torch.zeros(n_channels))  # -1 to 1

    def forward(self, x):
        pan_clamped = torch.clamp(self.pan, -1, 1)
        angle = (pan_clamped + 1) * np.pi / 4
        left_gain = torch.cos(angle).unsqueeze(-1)
        right_gain = torch.sin(angle).unsqueeze(-1)
        if x.dim() == 2:
            left = x * left_gain
            right = x * right_gain
            return torch.stack([left, right], dim=1)
        return x


class DifferentiableEQ(nn.Module):
    """Differentiable parametric EQ using biquad filter coefficients."""

    def __init__(self, n_channels: int, n_bands: int = 4, sample_rate: int = 48000):
        super().__init__()
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.sample_rate = sample_rate
        default_freqs = torch.tensor([100.0, 400.0, 2000.0, 8000.0][:n_bands])
        self.freq = nn.Parameter(
            default_freqs.unsqueeze(0).expand(n_channels, -1).clone()
        )
        self.gain_db = nn.Parameter(torch.zeros(n_channels, n_bands))
        self.q = nn.Parameter(torch.ones(n_channels, n_bands) * 0.707)

    def compute_biquad(self, freq, gain_db, q):
        w0 = 2 * np.pi * freq / self.sample_rate
        A = 10 ** (gain_db / 40.0)
        alpha = torch.sin(w0) / (2 * q)
        b0 = 1 + alpha * A
        b1 = -2 * torch.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * torch.cos(w0)
        a2 = 1 - alpha / A
        return torch.stack([b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0], dim=-1)

    def forward(self, x):
        # Apply EQ in frequency domain for differentiability
        n_fft = x.shape[-1]
        X = torch.fft.rfft(x)
        freqs = torch.linspace(0, self.sample_rate / 2, X.shape[-1], device=x.device)
        for band in range(self.n_bands):
            fc = torch.clamp(self.freq[:, band], 20, self.sample_rate / 2 - 1)
            g = self.gain_db[:, band]
            q = torch.clamp(self.q[:, band], 0.1, 10.0)
            magnitude = 10 ** (g / 20.0)
            bw = fc / q
            response = 1.0 + (magnitude.unsqueeze(-1) - 1.0) / (
                1.0 + ((freqs.unsqueeze(0) - fc.unsqueeze(-1)) / (bw.unsqueeze(-1) / 2 + 1e-8)) ** 2
            )
            X = X * response
        return torch.fft.irfft(X, n=n_fft)


class DifferentiableCompressor(nn.Module):
    """Differentiable dynamics compressor."""

    def __init__(self, n_channels: int, sample_rate: int = 48000):
        super().__init__()
        self.sample_rate = sample_rate
        self.threshold_db = nn.Parameter(torch.full((n_channels,), -20.0))
        self.ratio = nn.Parameter(torch.full((n_channels,), 4.0))
        self.attack_ms = nn.Parameter(torch.full((n_channels,), 10.0))
        self.release_ms = nn.Parameter(torch.full((n_channels,), 100.0))

    def forward(self, x):
        eps = 1e-8
        x_db = 20 * torch.log10(torch.abs(x) + eps)
        threshold = self.threshold_db.unsqueeze(-1)
        ratio = torch.clamp(self.ratio.unsqueeze(-1), 1.0, 20.0)
        over = F.relu(x_db - threshold)
        gain_reduction_db = over * (1 - 1 / ratio)
        gain_linear = 10 ** (-gain_reduction_db / 20.0)
        return x * gain_linear


class DifferentiableMixingConsole(nn.Module):
    """Complete differentiable mixing console chain: EQ -> Compressor -> Gain -> Pan -> Sum."""

    def __init__(self, n_channels: int, sample_rate: int = 48000, n_eq_bands: int = 4):
        super().__init__()
        self.n_channels = n_channels
        self.eq = DifferentiableEQ(n_channels, n_eq_bands, sample_rate)
        self.compressor = DifferentiableCompressor(n_channels, sample_rate)
        self.gain = DifferentiableGain(n_channels)
        self.pan = DifferentiablePan(n_channels)

    def forward(self, channels):
        processed = []
        for i in range(min(self.n_channels, len(channels))):
            x = channels[i]
            if x.dim() == 1:
                x = x.unsqueeze(0)
            # Process through per-channel slice of EQ, compressor, gain
            n_fft = x.shape[-1]
            X = torch.fft.rfft(x)
            freqs = torch.linspace(
                0, self.eq.sample_rate / 2, X.shape[-1], device=x.device
            )
            for band in range(self.eq.n_bands):
                fc = torch.clamp(self.eq.freq[i, band], 20, self.eq.sample_rate / 2 - 1)
                g = self.eq.gain_db[i, band]
                q = torch.clamp(self.eq.q[i, band], 0.1, 10.0)
                magnitude = 10 ** (g / 20.0)
                bw = fc / q
                response = 1.0 + (magnitude - 1.0) / (
                    1.0 + ((freqs - fc) / (bw / 2 + 1e-8)) ** 2
                )
                X = X * response
            x = torch.fft.irfft(X, n=n_fft)

            # Per-channel compressor
            eps = 1e-8
            x_db = 20 * torch.log10(torch.abs(x) + eps)
            threshold = self.compressor.threshold_db[i]
            ratio = torch.clamp(self.compressor.ratio[i], 1.0, 20.0)
            over = F.relu(x_db - threshold)
            gain_reduction_db = over * (1 - 1 / ratio)
            gain_linear = 10 ** (-gain_reduction_db / 20.0)
            x = x * gain_linear

            # Per-channel gain
            gain_lin = 10 ** (self.gain.gain_db[i] / 20.0)
            x = x * gain_lin

            processed.append(x)
        mix = sum(processed)
        return mix, processed

    def get_parameters_dict(self):
        return {
            'gain_db': self.gain.gain_db.detach().cpu().numpy().tolist(),
            'pan': self.pan.pan.detach().cpu().numpy().tolist(),
            'eq_freq': self.eq.freq.detach().cpu().numpy().tolist(),
            'eq_gain': self.eq.gain_db.detach().cpu().numpy().tolist(),
            'eq_q': self.eq.q.detach().cpu().numpy().tolist(),
            'threshold': self.compressor.threshold_db.detach().cpu().numpy().tolist(),
            'ratio': self.compressor.ratio.detach().cpu().numpy().tolist(),
        }
