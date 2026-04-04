"""Tests for backend/ml/training_dataset_io.py."""
import json
import os

import numpy as np
import pytest

pytest.importorskip("torch")

from ml.channel_classifier import INSTRUMENT_CLASSES
from ml.training_dataset_io import DiskInstrumentMelDataset, iter_jsonl_dicts


def test_iter_jsonl_dicts(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    rows = list(iter_jsonl_dicts(str(p)))
    assert rows == [{"a": 1}, {"b": 2}]


def test_disk_instrument_mel_dataset(tmp_path):
    mel = np.random.randn(64, 64).astype(np.float32)
    mel_dir = tmp_path / "mels"
    mel_dir.mkdir()
    mel_path = mel_dir / "x.npy"
    np.save(str(mel_path), mel)

    jl = tmp_path / "train.jsonl"
    with open(jl, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"class": INSTRUMENT_CLASSES[0], "mel_npy": "mels/x.npy"}
            )
            + "\n"
        )

    class_to_idx = {c: i for i, c in enumerate(INSTRUMENT_CLASSES)}
    ds = DiskInstrumentMelDataset(str(jl), class_to_idx, base_dir=str(tmp_path))
    assert len(ds) == 1
    x, y = ds[0]
    assert tuple(x.shape) == (1, 64, 64)
    assert y == 0
