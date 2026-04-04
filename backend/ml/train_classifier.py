"""
Training script for the channel classifier.
Generates synthetic training data and trains the CNN model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import os
from typing import Optional

from .channel_classifier import ChannelClassifierNet, INSTRUMENT_CLASSES, NUM_CLASSES

logger = logging.getLogger(__name__)


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


def train_classifier(
    output_path: str = 'models/channel_classifier.pt',
    n_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    n_samples_per_class: int = 100,
    device: Optional[str] = None,
    disk_jsonl: Optional[str] = None,
):
    """Train the channel classifier model.

    If disk_jsonl is set, loads mel spectrograms from JSONL shards
    (see docs/AGENT_TRAINING_DATA.md). Otherwise uses synthetic data.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logger.info(f"Training on {device}")
    if disk_jsonl:
        from .training_dataset_io import DiskInstrumentMelDataset

        class_to_idx = {c: i for i, c in enumerate(INSTRUMENT_CLASSES)}
        dataset = DiskInstrumentMelDataset(disk_jsonl, class_to_idx)
        logger.info("Using disk dataset %s (%s samples)", disk_jsonl, len(dataset))
    else:
        dataset = SyntheticInstrumentDataset(
            n_samples_per_class=n_samples_per_class,
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
    import argparse

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description='Train channel classifier')
    p.add_argument(
        '--disk-jsonl',
        type=str,
        default=None,
        help='JSONL with {class, mel_npy} rows (see docs/AGENT_TRAINING_DATA.md)',
    )
    p.add_argument('--output', type=str, default='models/channel_classifier.pt')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    train_classifier(
        output_path=args.output,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        disk_jsonl=args.disk_jsonl,
    )
