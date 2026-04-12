"""
Differentiable mixing console -- allows gradient-based optimization of mix parameters.
Based on Steinmetz et al. "Automatic Multitrack Mixing with a Differentiable Mixing Console" (2020).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List


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
        """Mix a list of per-channel waveforms.

        Each tensor is shaped ``(batch, n_samples)`` or ``(n_samples,)``. All
        tensors are stacked along the channel dimension (rows) so that EQ,
        dynamics, and gain use one row per mixer channel. Pan is applied per
        channel; the mix bus is the sum of stereo images, downmixed to mono.
        """
        n = min(self.n_channels, len(channels))
        rows: List[torch.Tensor] = []
        for i in range(n):
            x = channels[i]
            if x.dim() == 1:
                x = x.unsqueeze(0)
            rows.append(x)
        x = torch.cat(rows, dim=0)
        x = self.eq(x)
        x = self.compressor(x)
        x = self.gain(x)
        # Pan expects (n_channels, n_samples): one pan parameter per row.
        stereo = self.pan(x)
        processed = [stereo[i : i + 1] for i in range(stereo.shape[0])]
        # Sum stereo images across channels, then mono downmix (L+R)/2.
        summed_lr = stereo.sum(dim=0)
        mix = ((summed_lr[0] + summed_lr[1]) / 2.0).unsqueeze(0)
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
