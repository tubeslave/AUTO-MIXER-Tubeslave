"""
Training script for the differentiable mixing console.
Uses multi-resolution STFT loss to optimize mix parameters against reference mixes.
"""
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import os
from typing import List, Optional, Tuple

from .differentiable_console import DifferentiableMixingConsole
from .losses import MultiResolutionSTFTLoss, LoudnessLoss, MixConsistencyLoss
from .npz_archive_safety import load_validated_numeric_npz, open_regular_binary_file

logger = logging.getLogger(__name__)

MAX_MIX_CONSOLE_EPOCHS = 200
MAX_MIX_CONSOLE_CHANNELS = 64
MAX_MIX_CONSOLE_AUDIO_LEN = 262_144
MAX_MIX_CONSOLE_SAMPLES = 4096
MAX_MIX_CONSOLE_DATASET_ELEMENTS = 40_000_000
MAX_MIX_CONSOLE_SAMPLE_ELEMENTS = 8_000_000
MAX_MIX_CONSOLE_TRAINING_EXAMPLES = 200_000
MAX_MIX_CONSOLE_JSONL_BYTES = 32 * 1024 * 1024
MAX_MIX_CONSOLE_JSONL_LINE_BYTES = 1024 * 1024
MIN_MIX_CONSOLE_LEARNING_RATE = 1e-8
MAX_MIX_CONSOLE_LEARNING_RATE = 1.0


def _bounded_int(name: str, value: int, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in range {minimum}..{maximum}")
    return value


def _validate_mix_console_training_request(
    *,
    n_epochs: int,
    lr: float,
    n_channels: int,
    audio_len: int,
    n_samples: int,
    sample_rate: int,
) -> None:
    _bounded_int("n_epochs", n_epochs, minimum=1, maximum=MAX_MIX_CONSOLE_EPOCHS)
    _bounded_int("n_channels", n_channels, minimum=1, maximum=MAX_MIX_CONSOLE_CHANNELS)
    _bounded_int("audio_len", audio_len, minimum=32, maximum=MAX_MIX_CONSOLE_AUDIO_LEN)
    _bounded_int("n_samples", n_samples, minimum=2, maximum=MAX_MIX_CONSOLE_SAMPLES)
    _bounded_int("sample_rate", sample_rate, minimum=8000, maximum=384000)
    if isinstance(lr, bool) or not isinstance(lr, (int, float)):
        raise ValueError("lr must be a finite numeric value")
    if not math.isfinite(float(lr)) or not MIN_MIX_CONSOLE_LEARNING_RATE <= float(lr) <= MAX_MIX_CONSOLE_LEARNING_RATE:
        raise ValueError(
            f"lr must be in range {MIN_MIX_CONSOLE_LEARNING_RATE}..{MAX_MIX_CONSOLE_LEARNING_RATE}"
        )
    if n_samples * (n_channels + 1) * audio_len > MAX_MIX_CONSOLE_DATASET_ELEMENTS:
        raise ValueError("mix-console synthetic dataset allocation limit exceeded")
    if n_epochs * n_samples > MAX_MIX_CONSOLE_TRAINING_EXAMPLES:
        raise ValueError("mix-console training work limit exceeded")


def _load_dataset_from_file(
    path: str,
    n_channels: int,
    *,
    max_dataset_bytes: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    _bounded_int(
        "n_channels",
        n_channels,
        minimum=1,
        maximum=MAX_MIX_CONSOLE_CHANNELS,
    )
    if max_dataset_bytes is not None and (
        type(max_dataset_bytes) is not int or max_dataset_bytes <= 0
    ):
        raise ValueError("max_dataset_bytes must be a positive integer")
    effective_max_bytes = min(
        MAX_MIX_CONSOLE_JSONL_BYTES,
        max_dataset_bytes or MAX_MIX_CONSOLE_JSONL_BYTES,
    )
    path = os.path.abspath(path)
    _, ext = os.path.splitext(path.lower())
    if ext == ".npz":
        with load_validated_numeric_npz(
            path,
            required_key_groups=(("multitracks",), ("references",)),
            max_compressed_bytes=effective_max_bytes,
        ) as raw:
            multitracks = raw["multitracks"]
            references = raw["references"]
    elif ext == ".jsonl":
        multitracks = []
        references = []
        with open_regular_binary_file(
            path,
            error_message="mix-console dataset must be a regular file",
        ) as f:
            opened_stat = os.fstat(f.fileno())
            if opened_stat.st_size > effective_max_bytes:
                raise ValueError("mix-console JSONL byte limit exceeded")
            bytes_read = 0
            while True:
                raw_line = f.readline(MAX_MIX_CONSOLE_JSONL_LINE_BYTES + 1)
                if not raw_line:
                    break
                bytes_read += len(raw_line)
                if bytes_read > effective_max_bytes:
                    raise ValueError("mix-console JSONL byte limit exceeded")
                if len(raw_line) > MAX_MIX_CONSOLE_JSONL_LINE_BYTES:
                    raise ValueError("mix-console JSONL line limit exceeded")
                if len(multitracks) >= MAX_MIX_CONSOLE_SAMPLES:
                    raise ValueError("mix-console dataset sample limit exceeded")
                try:
                    row = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError("invalid mix-console JSONL row") from exc
                if not isinstance(row, dict):
                    raise ValueError("mix-console JSONL row must be an object")
                multitracks.append(row.get("multitracks") or row.get("channels"))
                references.append(row.get("references") or row.get("reference"))
    else:
        raise ValueError(f"Unsupported dataset format: {ext}")

    if len(multitracks) != len(references):
        raise ValueError("multitracks/references length mismatch")
    if len(multitracks) == 0:
        raise ValueError(f"External dataset is empty: {path}")
    if len(multitracks) > MAX_MIX_CONSOLE_SAMPLES:
        raise ValueError("mix-console dataset sample limit exceeded")

    parsed_multitracks: List[np.ndarray] = []
    parsed_references: List[np.ndarray] = []
    materialized_elements = 0
    for tracks, ref in zip(multitracks, references):
        tracks_np = np.asarray(tracks, dtype=np.float32)
        if tracks_np.ndim == 1:
            tracks_np = tracks_np.reshape(1, -1)
        if tracks_np.ndim != 2:
            raise ValueError("Each mix-console sample must be a 2D channels/audio array")
        ref_np = np.asarray(ref, dtype=np.float32).reshape(-1)
        if ref_np.size != tracks_np.shape[1]:
            raise ValueError("mix-console reference length mismatch")
        sample_elements = (n_channels * tracks_np.shape[1]) + ref_np.size
        if sample_elements > MAX_MIX_CONSOLE_SAMPLE_ELEMENTS:
            raise ValueError("mix-console sample element limit exceeded")
        materialized_elements += sample_elements
        if materialized_elements > MAX_MIX_CONSOLE_DATASET_ELEMENTS:
            raise ValueError("mix-console materialized element limit exceeded")
        if tracks_np.shape[0] < n_channels:
            pad = np.zeros((n_channels - tracks_np.shape[0], tracks_np.shape[1]), dtype=np.float32)
            tracks_np = np.vstack([tracks_np, pad])
        tracks_np = tracks_np[:n_channels]
        parsed_multitracks.append(tracks_np.astype(np.float32, copy=False))
        parsed_references.append(ref_np.astype(np.float32, copy=False))

    return parsed_multitracks, parsed_references


def generate_synthetic_multitracks(n_samples: int, n_channels: int,
                                   audio_len: int,
                                   sample_rate: int = 48000):
    """Generate synthetic multitrack audio and reference mixes for training."""
    _validate_mix_console_training_request(
        n_epochs=1,
        lr=1e-2,
        n_channels=n_channels,
        audio_len=audio_len,
        n_samples=n_samples,
        sample_rate=sample_rate,
    )
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
    dataset_path: Optional[str] = None,
    device: Optional[str] = None,
    max_dataset_bytes: Optional[int] = None,
):
    """Train the differentiable mixing console parameters against reference mixes."""
    _validate_mix_console_training_request(
        n_epochs=n_epochs,
        lr=lr,
        n_channels=n_channels,
        audio_len=audio_len,
        n_samples=n_samples,
        sample_rate=sample_rate,
    )
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logger.info(f"Training differentiable mixing console on {device}")

    if dataset_path:
        logger.info("Loading external multitracks from %s", dataset_path)
        multitracks, references = _load_dataset_from_file(
            dataset_path,
            n_channels,
            max_dataset_bytes=max_dataset_bytes,
        )
        n_samples = len(multitracks)
    else:
        logger.info("Generating synthetic multitrack data...")
        multitracks, references = generate_synthetic_multitracks(
            n_samples, n_channels, audio_len, sample_rate,
        )

    if len(multitracks) < 2 or len(multitracks) > MAX_MIX_CONSOLE_SAMPLES:
        raise ValueError(
            f"mix-console dataset size must be in range 2..{MAX_MIX_CONSOLE_SAMPLES}"
        )
    materialized_elements = 0
    for tracks, reference in zip(multitracks, references):
        if tracks.ndim != 2 or tracks.shape[0] != n_channels:
            raise ValueError("mix-console sample channel shape mismatch")
        if reference.ndim != 1 or reference.shape[0] != tracks.shape[1]:
            raise ValueError("mix-console reference length mismatch")
        materialized_elements += int(tracks.size) + int(reference.size)
        if materialized_elements > MAX_MIX_CONSOLE_DATASET_ELEMENTS:
            raise ValueError("mix-console dataset allocation limit exceeded")
    if n_epochs * len(multitracks) > MAX_MIX_CONSOLE_TRAINING_EXAMPLES:
        raise ValueError("mix-console training work limit exceeded")
    n_samples = len(multitracks)

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
