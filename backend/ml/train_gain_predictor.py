"""
Training script for the gain/pan predictor.
Generates synthetic multitrack training data.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import os
from typing import Optional

from .gain_pan_predictor import GainPanPredictorNet
from .losses import MultiResolutionSTFTLoss

logger = logging.getLogger(__name__)


class SyntheticMixDataset(Dataset):
    """Synthetic multitrack dataset for training gain/pan predictor."""

    def __init__(self, n_samples: int = 500, n_channels: int = 8,
                 audio_len: int = 8192, sample_rate: int = 48000):
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.audio_len = audio_len
        self.sample_rate = sample_rate
        self.data = []
        self.target_gains = []
        self.target_pans = []
        for _ in range(n_samples):
            channels = np.random.randn(n_channels, audio_len).astype(np.float32) * 0.1
            for ch in range(n_channels):
                freq = np.random.uniform(60, 4000)
                t = np.linspace(0, audio_len / sample_rate, audio_len)
                channels[ch] += 0.3 * np.sin(
                    2 * np.pi * freq * t
                ).astype(np.float32)
            gains = np.random.uniform(-40, -6, n_channels).astype(np.float32)
            pans = np.random.uniform(-0.8, 0.8, n_channels).astype(np.float32)
            self.data.append(channels)
            self.target_gains.append(gains)
            self.target_pans.append(pans)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (torch.from_numpy(self.data[idx]),
                torch.from_numpy(self.target_gains[idx]),
                torch.from_numpy(self.target_pans[idx]))


def train_gain_predictor(
    output_path: str = 'models/gain_pan_predictor.pt',
    n_epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    n_channels: int = 8,
    n_samples: int = 500,
    device: Optional[str] = None,
):
    """Train the gain/pan predictor."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logger.info(f"Training gain/pan predictor on {device}")
    dataset = SyntheticMixDataset(n_samples=n_samples, n_channels=n_channels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size],
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    model = GainPanPredictorNet(n_channels=n_channels).to(device)
    gain_criterion = nn.MSELoss()
    pan_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs,
    )
    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for batch_audio, batch_gains, batch_pans in train_loader:
            batch_audio = batch_audio.to(device)
            batch_gains = batch_gains.to(device)
            batch_pans = batch_pans.to(device)
            optimizer.zero_grad()
            pred_gains, pred_pans = model(batch_audio)
            loss = (gain_criterion(pred_gains, batch_gains) +
                    0.5 * pan_criterion(pred_pans, batch_pans))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_audio, batch_gains, batch_pans in val_loader:
                batch_audio = batch_audio.to(device)
                batch_gains = batch_gains.to(device)
                batch_pans = batch_pans.to(device)
                pred_gains, pred_pans = model(batch_audio)
                loss = (gain_criterion(pred_gains, batch_gains) +
                        0.5 * pan_criterion(pred_pans, batch_pans))
                val_loss += loss.item()
        avg_val_loss = val_loss / max(len(val_loader), 1)
        logger.info(
            f"Epoch {epoch+1}/{n_epochs}: "
            f"train_loss={total_loss/len(train_loader):.4f} "
            f"val_loss={avg_val_loss:.4f}"
        )
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(
                os.path.dirname(output_path)
                if os.path.dirname(output_path) else '.',
                exist_ok=True,
            )
            torch.save(model.state_dict(), output_path)
            logger.info(f"Saved best model (val_loss={avg_val_loss:.4f})")
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return model


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train_gain_predictor()
