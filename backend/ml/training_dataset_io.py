"""
Disk dataset I/O for backend/ml training scripts.

Specification: docs/AGENT_TRAINING_DATA.md
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore


def iter_jsonl_dicts(path: str) -> Iterator[Dict[str, Any]]:
    """Yield one JSON object per non-empty line (events v1 or classifier rows)."""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("JSONL parse error %s:%s: %s", path, line_no, e)


def load_dataset_manifest(path: str) -> Dict[str, Any]:
    """Load manifest.json (see docs/schemas/dataset_manifest_v1.schema.json)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(base_dir: str, rel_path: str) -> str:
    return os.path.normpath(os.path.join(base_dir, rel_path))


class DiskInstrumentMelDataset(Dataset):  # type: ignore[misc]
    """
    JSONL shard: each line {"class": "<INSTRUMENT_CLASSES name>", "mel_npy": "rel/path.npy"}

    The .npy array must be float32, shape (n_mels, n_frames) or (1, n_mels, n_frames).
    """

    def __init__(
        self,
        jsonl_path: str,
        class_to_idx: Dict[str, int],
        base_dir: Optional[str] = None,
    ):
        if torch is None:
            raise RuntimeError("DiskInstrumentMelDataset requires PyTorch")
        super().__init__()
        self._jsonl_path = jsonl_path
        self._base = base_dir or os.path.dirname(os.path.abspath(jsonl_path))
        self._class_to_idx = class_to_idx
        self._rows: List[Tuple[str, int]] = []
        for row in _read_classifier_jsonl(jsonl_path):
            cls_name = row.get("class") or row.get("label")
            mel_rel = row.get("mel_npy")
            if not cls_name or not mel_rel:
                logger.warning("skip row missing class/mel_npy: %s", row)
                continue
            if cls_name not in class_to_idx:
                logger.warning("unknown class %s, skip", cls_name)
                continue
            mel_path = _resolve_path(self._base, mel_rel)
            if not os.path.isfile(mel_path):
                logger.warning("missing mel file: %s", mel_path)
                continue
            self._rows.append((mel_path, class_to_idx[cls_name]))
        if not self._rows:
            raise ValueError(f"no valid rows in {jsonl_path}")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int):
        mel_path, label = self._rows[idx]
        arr = np.load(mel_path)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        x = torch.from_numpy(arr.astype(np.float32, copy=False))
        return x, label


def _read_classifier_jsonl(path: str) -> List[Dict[str, Any]]:
    return list(iter_jsonl_dicts(path))
