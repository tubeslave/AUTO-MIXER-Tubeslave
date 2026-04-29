"""Training script for the channel classifier.

Generates synthetic training data by default and trains the CNN model.
Supports loading external datasets from JSONL/NPZ when `dataset_path` is
provided.
"""
import json
import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from .channel_classifier import ChannelClassifierNet, INSTRUMENT_CLASSES, NUM_CLASSES

logger = logging.getLogger(__name__)


def _label_to_index(label: str) -> int:
    if label in INSTRUMENT_CLASSES:
        return INSTRUMENT_CLASSES.index(label)
    return INSTRUMENT_CLASSES.index("unknown")


def _to_numpy_array(payload: Sequence, *, expected_mels: int, expected_frames: int) -> np.ndarray:
    arr = np.asarray(payload, dtype=np.float32)
    if arr.ndim != 2:
        arr = np.asarray(arr).reshape(1, -1)
    if arr.shape != (expected_mels, expected_frames):
        if arr.size < expected_mels * expected_frames:
            padded = np.zeros((expected_mels * expected_frames,), dtype=np.float32)
            padded[:arr.size] = arr.ravel()[: expected_mels * expected_frames]
            arr = padded.reshape(expected_mels, expected_frames)
        else:
            arr = arr.reshape(expected_mels, expected_frames)
    return arr


def _load_dataset_from_file(path: str, *, n_mels: int, n_frames: int) -> List[tuple]:
    path = os.path.abspath(path)
    _, ext = os.path.splitext(path.lower())
    features: List[np.ndarray] = []
    labels: List[int] = []

    if ext == ".npz":
        with np.load(path, allow_pickle=True) as raw:
            data = raw["data"] if "data" in raw.files else None
            feat = raw["features"] if "features" in raw.files else None
            lab = raw["labels"] if "labels" in raw.files else None
            if data is None and feat is None:
                raise ValueError("NPZ must include 'data' or 'features'")
            if lab is None:
                raise ValueError("NPZ must include 'labels'")
            sample_data = data if data is not None else feat
            sample_labels = lab
    elif ext == ".jsonl":
        sample_data = []
        sample_labels = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                sample_data.append(row.get("spectrogram") or row.get("features"))
                sample_labels.append(row.get("label", "unknown"))
    else:
        raise ValueError(f"Unsupported dataset format: {ext}")

    if isinstance(sample_data, np.ndarray):
        sample_data = sample_data.tolist()
    if isinstance(sample_labels, np.ndarray):
        sample_labels = sample_labels.tolist()
    if len(sample_data) != len(sample_labels):
        raise ValueError("Dataset payloads and labels have different lengths")

    for idx, row in enumerate(sample_data):
        arr = _to_numpy_array(row, expected_mels=n_mels, expected_frames=n_frames)
        if len(sample_labels) <= idx:
            label_value = "unknown"
        else:
            label_value = sample_labels[idx]
        if isinstance(label_value, str):
            class_idx = _label_to_index(label_value)
        else:
            class_idx = int(label_value)
            if class_idx < 0 or class_idx >= NUM_CLASSES:
                class_idx = NUM_CLASSES - 1
        features.append(arr)
        labels.append(class_idx)

    if not features:
        raise ValueError(f"External dataset is empty: {path}")

    return [(feature, label) for feature, label in zip(features, labels)]


class SyntheticInstrumentDataset(Dataset):
    """Generates synthetic training spectrograms for each instrument class."""

    def __init__(self, n_samples_per_class: int = 100, n_mels: int = 64,
                 n_frames: int = 64, sample_rate: int = 48000):
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.sample_rate = sample_rate
        self.data = []
        self.labels = []
        spectral_profiles = {
            'kick': {'peak_freq': 60, 'bandwidth': 40, 'transient': True},
            'snare': {'peak_freq': 200, 'bandwidth': 300, 'transient': True},
            'hi_hat': {'peak_freq': 8000, 'bandwidth': 6000, 'transient': True},
            'toms': {'peak_freq': 150, 'bandwidth': 100, 'transient': True},
            'overheads': {'peak_freq': 4000, 'bandwidth': 8000, 'transient': False},
            'room_mics': {'peak_freq': 1000, 'bandwidth': 4000, 'transient': False},
            'bass_guitar': {'peak_freq': 80, 'bandwidth': 200, 'transient': False},
            'electric_guitar': {'peak_freq': 1500, 'bandwidth': 3000, 'transient': False},
            'acoustic_guitar': {'peak_freq': 1000, 'bandwidth': 4000, 'transient': False},
            'keys_piano': {'peak_freq': 500, 'bandwidth': 4000, 'transient': True},
            'synth': {'peak_freq': 2000, 'bandwidth': 6000, 'transient': False},
            'organ': {'peak_freq': 400, 'bandwidth': 3000, 'transient': False},
            'lead_vocal': {'peak_freq': 2500, 'bandwidth': 2000, 'transient': False},
            'backing_vocal': {'peak_freq': 2000, 'bandwidth': 2500, 'transient': False},
            'choir': {'peak_freq': 1500, 'bandwidth': 3000, 'transient': False},
            'brass': {'peak_freq': 1000, 'bandwidth': 3000, 'transient': False},
            'woodwind': {'peak_freq': 800, 'bandwidth': 2000, 'transient': False},
            'strings': {'peak_freq': 600, 'bandwidth': 3000, 'transient': False},
            'percussion': {'peak_freq': 3000, 'bandwidth': 5000, 'transient': True},
            'dj_playback': {'peak_freq': 1000, 'bandwidth': 8000, 'transient': False},
            'click_track': {'peak_freq': 1000, 'bandwidth': 100, 'transient': True},
            'ambient_mic': {'peak_freq': 500, 'bandwidth': 6000, 'transient': False},
            'audience': {'peak_freq': 1000, 'bandwidth': 6000, 'transient': False},
            'unknown': {'peak_freq': 1000, 'bandwidth': 4000, 'transient': False},
        }
        mel_freqs = np.linspace(0, sample_rate // 2, n_mels)
        for cls_idx, cls_name in enumerate(INSTRUMENT_CLASSES):
            profile = spectral_profiles.get(cls_name, spectral_profiles['unknown'])
            for _ in range(n_samples_per_class):
                spec = np.random.randn(n_mels, n_frames) * 0.1
                peak = profile['peak_freq'] + np.random.randn() * profile['bandwidth'] * 0.1
                bw = profile['bandwidth'] * (0.8 + np.random.rand() * 0.4)
                for m in range(n_mels):
                    dist = abs(mel_freqs[m] - peak) / (bw + 1e-8)
                    spec[m, :] += np.exp(-0.5 * dist ** 2) * (2 + np.random.rand())
                if profile['transient']:
                    n_transients = np.random.randint(2, 8)
                    for _ in range(n_transients):
                        t = np.random.randint(0, n_frames)
                        spec[:, t] += np.random.rand(n_mels) * 1.5
                spec += np.random.randn(n_mels, n_frames) * 0.05
                spec = (spec - spec.mean()) / (spec.std() + 1e-8)
                self.data.append(spec.astype(np.float32))
                self.labels.append(cls_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).unsqueeze(0), self.labels[idx]


class ExternalInstrumentDataset(Dataset):
    """Dataset loaded from external internet or local files."""

    def __init__(self, records: List[tuple]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        spec, label = self.records[idx]
        return torch.from_numpy(spec).unsqueeze(0), label


def train_classifier(
    output_path: str = 'models/channel_classifier.pt',
    n_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    n_samples_per_class: int = 100,
    n_mels: int = 64,
    n_frames: int = 64,
    dataset_path: Optional[str] = None,
    device: Optional[str] = None,
):
    """Train the channel classifier model."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logger.info(f"Training on {device}")

    if dataset_path:
        logger.info("Loading external dataset from %s", dataset_path)
        records = _load_dataset_from_file(
            dataset_path, n_mels=n_mels, n_frames=n_frames,
        )
        dataset = ExternalInstrumentDataset(records)
    else:
        dataset = SyntheticInstrumentDataset(
            n_samples_per_class=n_samples_per_class,
            n_mels=n_mels,
            n_frames=n_frames,
        )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size],
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    model = ChannelClassifierNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5,
    )
    best_val_acc = 0.0
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = (torch.tensor(batch_y, device=device)
                       if not isinstance(batch_y, torch.Tensor)
                       else batch_y.to(device))
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)
        train_acc = correct / max(total, 1)
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = (torch.tensor(batch_y, device=device)
                           if not isinstance(batch_y, torch.Tensor)
                           else batch_y.to(device))
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_y)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_loss)
        logger.info(
            f"Epoch {epoch+1}/{n_epochs}: "
            f"train_loss={total_loss/len(train_loader):.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(
                os.path.dirname(output_path)
                if os.path.dirname(output_path) else '.',
                exist_ok=True,
            )
            torch.save(model.state_dict(), output_path)
            logger.info(f"Saved best model (val_acc={val_acc:.3f})")
    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.3f}")
    return model


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train_classifier()
