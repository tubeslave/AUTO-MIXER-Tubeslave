"""
In-memory buffer of multitrack + reference pairs with train / val / hold-out splits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MixSample:
    """One training example: per-channel mono waveforms and a mono reference mix."""

    multitrack: np.ndarray  # shape (n_channels, n_samples), float32
    reference: np.ndarray  # shape (n_samples,), float32


class MixTrainingBuffer:
    """Stores mix samples and exposes disjoint train / validation / hold-out index sets."""

    def __init__(
        self,
        n_channels: int,
        *,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        holdout_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        total_r = train_ratio + val_ratio + holdout_ratio
        if abs(total_r - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + holdout_ratio must sum to 1.0")
        self.n_channels = n_channels
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._holdout_ratio = holdout_ratio
        self._seed = seed
        self._samples: List[MixSample] = []
        self._train_idx: List[int] = []
        self._val_idx: List[int] = []
        self._holdout_idx: List[int] = []
        self._splits_dirty = True

    def __len__(self) -> int:
        return len(self._samples)

    def clear(self) -> None:
        self._samples.clear()
        self._train_idx.clear()
        self._val_idx.clear()
        self._holdout_idx.clear()
        self._splits_dirty = True

    def add(self, multitrack: np.ndarray, reference: np.ndarray) -> None:
        """Append one sample; splits are recomputed on next access."""
        if multitrack.ndim != 2:
            raise ValueError("multitrack must be 2-D (n_channels, n_samples)")
        if multitrack.shape[0] != self.n_channels:
            raise ValueError(
                f"multitrack has {multitrack.shape[0]} channels, expected {self.n_channels}",
            )
        if reference.ndim != 1:
            raise ValueError("reference must be 1-D (n_samples,)")
        if reference.shape[0] != multitrack.shape[1]:
            raise ValueError("reference length must match multitrack n_samples")
        mt = np.asarray(multitrack, dtype=np.float32)
        ref = np.asarray(reference, dtype=np.float32)
        self._samples.append(MixSample(multitrack=mt, reference=ref))
        self._splits_dirty = True

    def _rebuild_splits(self) -> None:
        n = len(self._samples)
        if n == 0:
            self._train_idx = []
            self._val_idx = []
            self._holdout_idx = []
            self._splits_dirty = False
            return
        rng = np.random.RandomState(self._seed)
        perm = rng.permutation(n)
        if n == 1:
            only = [int(perm[0])]
            self._train_idx = only
            self._val_idx = only
            self._holdout_idx = only
            self._splits_dirty = False
            return
        if n == 2:
            self._train_idx = [int(perm[0])]
            self._val_idx = [int(perm[1])]
            self._holdout_idx = [int(perm[1])]
            self._splits_dirty = False
            return
        n_train = max(1, int(n * self._train_ratio))
        n_val = max(1, int(n * self._val_ratio))
        n_hold = max(1, n - n_train - n_val)
        while n_train + n_val + n_hold > n:
            if n_train > 1 and n_train >= n_val and n_train >= n_hold:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                n_hold -= 1
        while n_train + n_val + n_hold < n:
            n_train += 1
        cut1 = n_train
        cut2 = n_train + n_val
        self._train_idx = perm[:cut1].tolist()
        self._val_idx = perm[cut1:cut2].tolist()
        self._holdout_idx = perm[cut2:].tolist()
        self._splits_dirty = False

    def ensure_splits(self) -> None:
        if self._splits_dirty:
            self._rebuild_splits()

    @property
    def train_indices(self) -> List[int]:
        self.ensure_splits()
        return list(self._train_idx)

    @property
    def val_indices(self) -> List[int]:
        self.ensure_splits()
        return list(self._val_idx)

    @property
    def holdout_indices(self) -> List[int]:
        self.ensure_splits()
        return list(self._holdout_idx)

    def get_sample(self, index: int) -> MixSample:
        return self._samples[index]

    def batch_indices(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
        rng: np.random.RandomState,
    ) -> Iterator[List[int]]:
        """Yield lists of sample indices for one split."""
        self.ensure_splits()
        if split == "train":
            idxs = list(self._train_idx)
        elif split == "val":
            idxs = list(self._val_idx)
        elif split == "holdout":
            idxs = list(self._holdout_idx)
        else:
            raise ValueError("split must be 'train', 'val', or 'holdout'")
        if not idxs:
            return
        order = list(idxs)
        if shuffle:
            rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            yield order[start : start + batch_size]
