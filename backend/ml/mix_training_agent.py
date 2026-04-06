"""
Mix training agent: batched optimization of DifferentiableMixingConsole with
metrics logging, checkpoints, validation, and hold-out evaluation.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .differentiable_console import DifferentiableMixingConsole
from .losses import LoudnessLoss, MixConsistencyLoss, MultiResolutionSTFTLoss
from .mix_training_buffer import MixSample, MixTrainingBuffer
from .train_mix_console import generate_synthetic_multitracks

logger = logging.getLogger(__name__)


def _mix_to_mono_reference_shape(mix: torch.Tensor) -> torch.Tensor:
    """Collapse console bus to shape (1, T) for mono reference STFT / loudness losses."""
    x = mix
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 2:
        if x.shape[0] == 2:
            return x.mean(dim=0, keepdim=True)
        return x.mean(dim=0, keepdim=True)
    if x.dim() == 3:
        return x.mean(dim=(0, 1), keepdim=False).unsqueeze(0)
    raise ValueError(f"Unexpected mix tensor shape {tuple(x.shape)}")


@dataclass
class MixTrainingConfig:
    """Hyperparameters for a mix training run."""

    n_channels: int = 8
    audio_len: int = 16384
    sample_rate: int = 48000
    n_samples: int = 200
    batch_size: int = 8
    n_epochs: int = 50
    lr: float = 1e-2
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    holdout_ratio: float = 0.15
    split_seed: int = 42
    grad_clip: float = 5.0
    loudness_weight: float = 0.1
    consistency_weight: float = 0.01
    checkpoint_every: int = 1
    eval_holdout_every: int = 1


class MixTrainingAgent:
    """Trains a differentiable console using buffered data and mini-batches."""

    def __init__(
        self,
        config: MixTrainingConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            self.device = device

        self.buffer = MixTrainingBuffer(
            config.n_channels,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            holdout_ratio=config.holdout_ratio,
            seed=config.split_seed,
        )
        self.console = DifferentiableMixingConsole(
            n_channels=config.n_channels,
            sample_rate=config.sample_rate,
        ).to(self.device)
        self._stft_loss = MultiResolutionSTFTLoss().to(self.device)
        self._loudness_loss = LoudnessLoss(
            sample_rate=config.sample_rate,
        ).to(self.device)
        self._consistency_loss = MixConsistencyLoss().to(self.device)
        self._optimizer = optim.Adam(self.console.parameters(), lr=config.lr)
        self._scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer,
            T_max=max(config.n_epochs, 1),
        )
        self._next_epoch = 0
        self._best_val_loss = float("inf")
        self._rng = np.random.RandomState(config.split_seed)

    def load_synthetic_corpus(self) -> None:
        """Fill buffer with synthetic multitracks (same generator as train_mix_console)."""
        self.buffer.clear()
        multi, refs = generate_synthetic_multitracks(
            self.config.n_samples,
            self.config.n_channels,
            self.config.audio_len,
            self.config.sample_rate,
        )
        for mt, ref in zip(multi, refs):
            self.buffer.add(mt, ref)
        logger.info(
            "Loaded %d synthetic samples (%d ch, len=%d)",
            len(self.buffer),
            self.config.n_channels,
            self.config.audio_len,
        )

    def add_samples(self, samples: List[MixSample]) -> None:
        """Append externally built samples (e.g. decoded stems)."""
        for s in samples:
            self.buffer.add(s.multitrack, s.reference)

    def _forward_loss(
        self,
        sample: MixSample,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single-example forward and loss components."""
        n_ch = self.config.n_channels
        channel_list = [
            torch.from_numpy(sample.multitrack[ch]).to(self.device)
            for ch in range(n_ch)
        ]
        ref_tensor = torch.from_numpy(sample.reference).unsqueeze(0).to(
            self.device,
        )
        mix, processed = self.console(channel_list)
        mix_for_loss = _mix_to_mono_reference_shape(mix)
        loss_stft = self._stft_loss(mix_for_loss, ref_tensor)
        loss_loud = self._loudness_loss(mix_for_loss, ref_tensor)
        loss_consist = self._consistency_loss(processed, mix)
        total = (
            loss_stft
            + self.config.loudness_weight * loss_loud
            + self.config.consistency_weight * loss_consist
        )
        metrics = {
            "stft": float(loss_stft.detach().item()),
            "loudness": float(loss_loud.detach().item()),
            "consistency": float(loss_consist.detach().item()),
            "total": float(total.detach().item()),
        }
        return total, metrics

    def _run_split(
        self,
        split: str,
        *,
        train: bool,
        batch_size: int,
    ) -> Dict[str, float]:
        """Average metrics over one pass of a split (val/holdout: no grad)."""
        if train:
            self.console.train()
        else:
            self.console.eval()
        sums: Dict[str, float] = {
            "stft": 0.0,
            "loudness": 0.0,
            "consistency": 0.0,
            "total": 0.0,
        }
        n_batches = 0
        n_examples = 0

        for batch_idx in self.buffer.batch_indices(
            split,
            batch_size,
            shuffle=train,
            rng=self._rng,
        ):
            n_batches += 1
            if train:
                self._optimizer.zero_grad()
            batch_total = 0.0
            batch_metrics = {k: 0.0 for k in sums}

            for i in batch_idx:
                sample = self.buffer.get_sample(i)
                if train:
                    loss, m = self._forward_loss(sample)
                    contrib = loss / max(len(batch_idx), 1)
                    contrib.backward()
                    batch_total += float(loss.detach().item())
                else:
                    with torch.no_grad():
                        loss, m = self._forward_loss(sample)
                        batch_total += float(loss.detach().item())
                for k in sums:
                    batch_metrics[k] += m[k]
                n_examples += 1

            if train:
                torch.nn.utils.clip_grad_norm_(
                    self.console.parameters(),
                    self.config.grad_clip,
                )
                self._optimizer.step()

            for k in sums:
                sums[k] += batch_metrics[k] / max(len(batch_idx), 1)

        denom = max(n_batches, 1)
        return {k: sums[k] / denom for k in sums}

    def train_epoch(self) -> Dict[str, float]:
        """One training epoch (shuffled batches) plus LR step."""
        out = self._run_split(
            "train",
            train=True,
            batch_size=self.config.batch_size,
        )
        self._scheduler.step()
        return out

    @torch.no_grad()
    def evaluate_val(self) -> Dict[str, float]:
        self.console.eval()
        return self._run_split(
            "val",
            train=False,
            batch_size=self.config.batch_size,
        )

    @torch.no_grad()
    def evaluate_holdout(self) -> Dict[str, float]:
        self.console.eval()
        return self._run_split(
            "holdout",
            train=False,
            batch_size=self.config.batch_size,
        )

    def _append_metrics_jsonl(self, path: str, row: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def save_checkpoint(
        self,
        path: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "next_epoch": self._next_epoch,
            "best_val_loss": self._best_val_loss,
            "model_state_dict": self.console.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "scheduler_state_dict": self._scheduler.state_dict(),
            "config": asdict(self.config),
        }
        if extra:
            payload["extra"] = extra
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(payload, path)
        logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: str) -> None:
        try:
            data = torch.load(
                path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            data = torch.load(path, map_location=self.device)
        self.console.load_state_dict(data["model_state_dict"])
        if "optimizer_state_dict" in data:
            self._optimizer.load_state_dict(data["optimizer_state_dict"])
        if "scheduler_state_dict" in data:
            self._scheduler.load_state_dict(data["scheduler_state_dict"])
        if "next_epoch" in data:
            self._next_epoch = int(data["next_epoch"])
        else:
            self._next_epoch = int(data.get("epoch", -1)) + 1
        self._best_val_loss = float(data.get("best_val_loss", float("inf")))
        logger.info(
            "Loaded checkpoint next_epoch=%d best_val=%.6f",
            self._next_epoch,
            self._best_val_loss,
        )

    def run(
        self,
        *,
        checkpoint_dir: str,
        resume_path: Optional[str] = None,
    ) -> DifferentiableMixingConsole:
        """Full training loop with best-of-val checkpoint and hold-out metrics."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_path = os.path.join(checkpoint_dir, "mix_console_best.pt")
        last_path = os.path.join(checkpoint_dir, "mix_console_last.pt")
        metrics_path = os.path.join(checkpoint_dir, "metrics.jsonl")

        if resume_path and os.path.isfile(resume_path):
            self.load_checkpoint(resume_path)

        for ep in range(self._next_epoch, self.config.n_epochs):
            train_m = self.train_epoch()
            val_m = self.evaluate_val()
            hold_m: Dict[str, float] = {}
            if (ep + 1) % max(self.config.eval_holdout_every, 1) == 0:
                hold_m = self.evaluate_holdout()

            row: Dict[str, Any] = {
                "epoch": ep + 1,
                "train": train_m,
                "val": val_m,
            }
            if hold_m:
                row["holdout"] = hold_m
            row["lr"] = self._scheduler.get_last_lr()[0]
            self._append_metrics_jsonl(metrics_path, row)

            logger.info(
                "Epoch %d/%d train=%.6f val=%.6f (stft val=%.6f)",
                ep + 1,
                self.config.n_epochs,
                train_m["total"],
                val_m["total"],
                val_m["stft"],
            )
            if hold_m:
                logger.info(
                    "  holdout total=%.6f stft=%.6f",
                    hold_m["total"],
                    hold_m["stft"],
                )

            self._next_epoch = ep + 1

            if val_m["total"] < self._best_val_loss:
                self._best_val_loss = val_m["total"]
                self.save_checkpoint(
                    best_path,
                    extra={"selection": "best_val_total"},
                )

            if (ep + 1) % max(self.config.checkpoint_every, 1) == 0:
                self.save_checkpoint(
                    last_path,
                    extra={"selection": "last"},
                )

        if self._best_val_loss < float("inf") and os.path.isfile(best_path):
            self.load_checkpoint(best_path)
        logger.info("Training finished. Best val total loss: %.6f", self._best_val_loss)
        return self.console


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mix training agent (batched console training)")
    p.add_argument(
        "--checkpoint-dir",
        default="backend/models/mix_training",
        help="Directory for checkpoints and metrics.jsonl",
    )
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--n-channels", type=int, default=8)
    p.add_argument("--audio-len", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save last.pt every N epochs",
    )
    p.add_argument(
        "--holdout-every",
        type=int,
        default=1,
        help="Evaluate hold-out every N epochs",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    cfg = MixTrainingConfig(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        n_channels=args.n_channels,
        audio_len=args.audio_len,
        lr=args.lr,
        split_seed=args.split_seed,
        checkpoint_every=args.checkpoint_every,
        eval_holdout_every=args.holdout_every,
    )
    agent = MixTrainingAgent(cfg)
    agent.load_synthetic_corpus()
    agent.run(checkpoint_dir=args.checkpoint_dir, resume_path=args.resume)


if __name__ == "__main__":
    main()
