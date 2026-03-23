"""
Gain and pan predictor using squeeze-and-excitation network.
Based on Martinez Ramirez et al. "Deep Learning for Intelligent Mixing" and
Steinmetz et al. "Automatic Multitrack Mixing with a Differentiable Mixing Console".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y


class GainPanPredictorNet(nn.Module):
    """Predicts gain and pan values from audio features."""

    def __init__(self, n_channels: int = 32, feature_dim: int = 128):
        super().__init__()
        self.n_channels = n_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            SEBlock(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            SEBlock(64),
            nn.Conv1d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.channel_context = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.gain_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.pan_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, channel_audios: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_ch, audio_len = channel_audios.shape
        x = channel_audios.view(batch_size * n_ch, 1, audio_len)
        features = self.encoder(x).squeeze(-1)
        features = features.view(batch_size, n_ch, -1)
        features = self.channel_context(features)
        gains = self.gain_head(features).squeeze(-1)
        gains = gains * 48 - 60  # Scale to dB range [-60, -12]
        pans = self.pan_head(features).squeeze(-1)
        return gains, pans


class GainPanPredictor:
    """High-level gain/pan prediction interface."""

    def __init__(self, n_channels: int = 32,
                 model_path: Optional[str] = None):
        self.n_channels = n_channels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GainPanPredictorNet(n_channels=n_channels).to(self.device)
        self.model.eval()
        if model_path:
            self.load_model(model_path)

    def load_model(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def predict(self, channel_audios: List[np.ndarray],
                block_size: int = 8192) -> Dict[str, List[float]]:
        n_ch = len(channel_audios)
        max_len = max(len(a) for a in channel_audios) if channel_audios else block_size
        target_len = min(max_len, block_size)
        padded = np.zeros((1, n_ch, target_len), dtype=np.float32)
        for i, audio in enumerate(channel_audios):
            length = min(len(audio), target_len)
            padded[0, i, :length] = audio[:length]
        x = torch.from_numpy(padded).to(self.device)
        with torch.no_grad():
            gains, pans = self.model(x)
        return {
            'gains_db': gains[0].cpu().numpy().tolist(),
            'pans': pans[0].cpu().numpy().tolist(),
        }
