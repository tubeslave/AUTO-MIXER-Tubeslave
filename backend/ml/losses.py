"""
Multi-resolution STFT loss for differentiable mixing.
Based on Engel et al. "DDSP" and Steinmetz & Reiss "auraloss".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class STFTLoss(nn.Module):
    """Single-resolution STFT loss (spectral convergence + log magnitude)."""

    def __init__(self, fft_size=1024, hop_size=256, win_size=1024):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer('window', torch.hann_window(win_size))

    def stft(self, x):
        return torch.stft(x, self.fft_size, self.hop_size, self.win_size,
                          self.window, return_complex=True)

    def forward(self, predicted, target):
        pred_stft = self.stft(predicted)
        tgt_stft = self.stft(target)
        pred_mag = torch.abs(pred_stft)
        tgt_mag = torch.abs(tgt_stft)
        # Spectral convergence
        sc_loss = torch.norm(tgt_mag - pred_mag, p='fro') / (torch.norm(tgt_mag, p='fro') + 1e-8)
        # Log magnitude loss
        log_loss = F.l1_loss(torch.log(pred_mag + 1e-8), torch.log(tgt_mag + 1e-8))
        return sc_loss + log_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss combining multiple FFT sizes."""

    def __init__(self, fft_sizes=(512, 1024, 2048), hop_sizes=(128, 256, 512),
                 win_sizes=(512, 1024, 2048)):
        super().__init__()
        self.losses = nn.ModuleList([
            STFTLoss(f, h, w) for f, h, w in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, predicted, target):
        total = 0.0
        for loss_fn in self.losses:
            total = total + loss_fn(predicted, target)
        return total / len(self.losses)


class LoudnessLoss(nn.Module):
    """Perceptual loudness loss using A-weighting approximation."""

    def __init__(self, sample_rate=48000, block_size=1024):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size

    def a_weight(self, frequencies):
        f2 = frequencies ** 2
        weights = (12194**2 * f2**2) / (
            (f2 + 20.6**2) *
            torch.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) *
            (f2 + 12194**2)
        )
        return weights

    def forward(self, predicted, target):
        pred_spec = torch.fft.rfft(predicted.reshape(-1, self.block_size))
        tgt_spec = torch.fft.rfft(target.reshape(-1, self.block_size))
        freqs = torch.linspace(0, self.sample_rate / 2, pred_spec.shape[-1],
                               device=predicted.device)
        weights = self.a_weight(freqs + 1e-8)
        pred_weighted = torch.abs(pred_spec) * weights
        tgt_weighted = torch.abs(tgt_spec) * weights
        pred_loudness = torch.mean(pred_weighted ** 2, dim=-1)
        tgt_loudness = torch.mean(tgt_weighted ** 2, dim=-1)
        return F.mse_loss(torch.log(pred_loudness + 1e-8),
                          torch.log(tgt_loudness + 1e-8))


class MixConsistencyLoss(nn.Module):
    """Ensures mix sum equals sum of processed channels."""

    def forward(self, channel_outputs, mix_output):
        channel_sum = sum(channel_outputs)
        return F.mse_loss(channel_sum, mix_output)
