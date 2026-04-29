"""
Training script for the gain/pan predictor.
Generates synthetic multitrack training data.
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import os
from typing import List, Optional, Sequence, Tuple

from .gain_pan_predictor import GainPanPredictorNet
from .losses import MultiResolutionSTFTLoss

logger = logging.getLogger(__name__)


def _normalize_list_shape(values: Sequence, *, target_channels: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1 and array.size == target_channels:
        return array.astype(np.float32)

    if array.ndim > 1:
        flattened = array.reshape(-1)
        if flattened.size < target_channels:
            padded = np.zeros((target_channels,), dtype=np.float32)
            padded[: flattened.size] = flattened[:target_channels]
            return padded
        return flattened[:target_channels]

    padded = np.zeros((target_channels,), dtype=np.float32)
    if array.size:
        padded[:1] = float(array.reshape(-1)[0])
    return padded


def _load_dataset_from_file(path: str, *, n_channels: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    path = os.path.abspath(path)
    _, ext = os.path.splitext(path.lower())
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    if ext == ".npz":
        with np.load(path, allow_pickle=True) as raw:
            channel_data = raw["channel_audios"] if "channel_audios" in raw.files else raw["channels"] if "channels" in raw.files else None
            if channel_data is None:
                raise ValueError("NPZ must include 'channel_audios' or 'channels'")
            if "gains" not in raw.files:
                raise ValueError("NPZ must include 'gains'")
            if "pans" not in raw.files:
                raise ValueError("NPZ must include 'pans'")
            gains = raw["gains"]
            pans = raw["pans"]
    elif ext == ".jsonl":
        channel_data = []
        gains = []
        pans = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                channel_data.append(row.get("channels") or row.get("channel_audios"))
                gains.append(row.get("gains"))
                pans.append(row.get("pans"))
    else:
        raise ValueError(f"Unsupported dataset format: {ext}")

    if isinstance(channel_data, np.ndarray):
        channel_data = channel_data.tolist()
    if isinstance(gains, np.ndarray):
        gains = gains.tolist()
    if isinstance(pans, np.ndarray):
        pans = pans.tolist()
    if len(channel_data) != len(gains) or len(channel_data) != len(pans):
        raise ValueError("channel/gain/pan datasets have different lengths")

    for channels, gain_values, pan_values in zip(channel_data, gains, pans):
        channel_matrix = np.asarray(channels, dtype=np.float32)
        if channel_matrix.ndim == 1:
            channel_matrix = channel_matrix.reshape(1, -1)
        if channel_matrix.ndim != 2:
            raise ValueError("Each sample must contain a 2D (channels, audio_len) array")
        if channel_matrix.shape[0] < n_channels:
            pad = np.zeros((n_channels - channel_matrix.shape[0], channel_matrix.shape[1]), dtype=np.float32)
            channel_matrix = np.vstack([channel_matrix, pad])
        channel_matrix = channel_matrix[:n_channels]

        target_gains = _normalize_list_shape(gain_values, target_channels=n_channels)
        target_pans = _normalize_list_shape(pan_values, target_channels=n_channels)
        samples.append((channel_matrix.astype(np.float32), target_gains, target_pans))

    if not samples:
        raise ValueError(f"External dataset is empty: {path}")

    return samples


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


class ExternalMixDataset(Dataset):
    """Dataset loaded from external data file."""

    def __init__(self, samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        channels, gains, pans = self.samples[idx]
        return (torch.from_numpy(channels),
                torch.from_numpy(gains),
                torch.from_numpy(pans))


def train_gain_predictor(
    output_path: str = 'models/gain_pan_predictor.pt',
    n_epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    n_channels: int = 8,
    n_samples: int = 500,
    dataset_path: Optional[str] = None,
    device: Optional[str] = None,
):
    """Train the gain/pan predictor."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logger.info(f"Training gain/pan predictor on {device}")
    if dataset_path:
        logger.info("Loading external dataset from %s", dataset_path)
        samples = _load_dataset_from_file(dataset_path, n_channels=n_channels)
        dataset = ExternalMixDataset(samples)
    else:
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
