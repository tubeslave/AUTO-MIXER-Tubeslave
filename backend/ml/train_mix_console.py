"""
Training script for the differentiable mixing console.
Uses multi-resolution STFT loss to optimize mix parameters against reference mixes.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import os
from typing import Optional, List

from .differentiable_console import DifferentiableMixingConsole
from .losses import MultiResolutionSTFTLoss, LoudnessLoss, MixConsistencyLoss

logger = logging.getLogger(__name__)


def generate_synthetic_multitracks(n_samples: int, n_channels: int,
                                   audio_len: int,
                                   sample_rate: int = 48000):
    """Generate synthetic multitrack audio and reference mixes for training."""
    multitracks = []
    references = []

    # Instrument-like frequency ranges for each channel type
    freq_ranges = [
        (40, 120),    # kick/bass
        (100, 400),   # snare body
        (200, 800),   # low-mid instruments
        (400, 2000),  # mid instruments
        (800, 4000),  # vocals/guitars
        (2000, 8000), # presence/cymbals
        (4000, 16000),# air/brightness
        (60, 3000),   # full-range
    ]

    for _ in range(n_samples):
        channels = []
        t = np.linspace(0, audio_len / sample_rate, audio_len,
                        dtype=np.float32)
        for ch in range(n_channels):
            freq_range = freq_ranges[ch % len(freq_ranges)]
            freq = np.random.uniform(freq_range[0], freq_range[1])
            amplitude = np.random.uniform(0.05, 0.3)
            signal = amplitude * np.sin(2 * np.pi * freq * t)
            # Add harmonics
            for h in range(2, 5):
                harm_amp = amplitude / (h * 2)
                signal += harm_amp * np.sin(2 * np.pi * freq * h * t)
            # Add noise floor
            signal += np.random.randn(audio_len).astype(np.float32) * 0.005
            channels.append(signal.astype(np.float32))

        # Create a reference mix with random but reasonable gains
        gains_db = np.random.uniform(-24, -6, n_channels)
        gains_linear = 10 ** (gains_db / 20.0)
        ref_mix = np.zeros(audio_len, dtype=np.float32)
        for ch_idx, ch_signal in enumerate(channels):
            ref_mix += ch_signal * gains_linear[ch_idx]

        # Normalize reference to prevent clipping
        peak = np.max(np.abs(ref_mix))
        if peak > 0.9:
            ref_mix = ref_mix * (0.9 / peak)

        multitracks.append(np.stack(channels))
        references.append(ref_mix)

    return multitracks, references


def train_mix_console(
    output_path: str = 'models/mix_console.pt',
    n_epochs: int = 50,
    lr: float = 1e-2,
    n_channels: int = 8,
    audio_len: int = 16384,
    n_samples: int = 200,
    sample_rate: int = 48000,
    device: Optional[str] = None,
):
    """Train the differentiable mixing console parameters against reference mixes."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logger.info(f"Training differentiable mixing console on {device}")

    # Generate training data
    logger.info("Generating synthetic multitrack data...")
    multitracks, references = generate_synthetic_multitracks(
        n_samples, n_channels, audio_len, sample_rate,
    )

    # Split into train/val
    split = int(0.8 * n_samples)
    train_multi, val_multi = multitracks[:split], multitracks[split:]
    train_refs, val_refs = references[:split], references[split:]

    # Initialize console and losses
    console = DifferentiableMixingConsole(
        n_channels=n_channels, sample_rate=sample_rate,
    ).to(device)
    stft_loss = MultiResolutionSTFTLoss().to(device)
    loudness_loss = LoudnessLoss(sample_rate=sample_rate).to(device)
    consistency_loss = MixConsistencyLoss().to(device)

    optimizer = optim.Adam(console.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        console.train()
        epoch_loss = 0.0

        # Shuffle training data
        indices = np.random.permutation(len(train_multi))

        for idx in indices:
            channels_np = train_multi[idx]
            ref_np = train_refs[idx]

            # Convert to tensors
            channel_list = [
                torch.from_numpy(channels_np[ch]).to(device)
                for ch in range(n_channels)
            ]
            ref_tensor = torch.from_numpy(ref_np).unsqueeze(0).to(device)

            optimizer.zero_grad()

            mix, processed = console(channel_list)

            # Compute losses
            mix_for_loss = mix if mix.dim() == 2 else mix.unsqueeze(0)
            loss_stft = stft_loss(mix_for_loss, ref_tensor)
            loss_loud = loudness_loss(mix_for_loss, ref_tensor)
            loss_consist = consistency_loss(processed, mix)

            total_loss = loss_stft + 0.1 * loss_loud + 0.01 * loss_consist
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(console.parameters(), 5.0)
            optimizer.step()

            epoch_loss += total_loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_multi)

        # Validation
        console.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for idx in range(len(val_multi)):
                channels_np = val_multi[idx]
                ref_np = val_refs[idx]
                channel_list = [
                    torch.from_numpy(channels_np[ch]).to(device)
                    for ch in range(n_channels)
                ]
                ref_tensor = torch.from_numpy(ref_np).unsqueeze(0).to(device)

                mix, processed = console(channel_list)
                mix_for_loss = mix if mix.dim() == 2 else mix.unsqueeze(0)
                loss_stft = stft_loss(mix_for_loss, ref_tensor)
                val_loss_total += loss_stft.item()

        avg_val_loss = val_loss_total / max(len(val_multi), 1)

        logger.info(
            f"Epoch {epoch+1}/{n_epochs}: "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(
                os.path.dirname(output_path)
                if os.path.dirname(output_path) else '.',
                exist_ok=True,
            )
            torch.save(console.state_dict(), output_path)
            logger.info(f"Saved best console model (val_loss={avg_val_loss:.4f})")

    # Log final parameter summary
    params = console.get_parameters_dict()
    logger.info(f"Final console parameters: gain_db={params['gain_db']}")
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return console


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train_mix_console()
