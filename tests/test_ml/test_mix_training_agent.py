"""Tests for mix training buffer and batched training agent."""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from ml.mix_training_agent import MixTrainingAgent, MixTrainingConfig
from ml.mix_training_buffer import MixSample, MixTrainingBuffer


def _fake_sample(n_ch: int, n_samples: int, seed: int) -> MixSample:
    rng = np.random.RandomState(seed)
    mt = rng.randn(n_ch, n_samples).astype(np.float32) * 0.1
    ref = mt.sum(axis=0).astype(np.float32)
    return MixSample(multitrack=mt, reference=ref)


class TestMixTrainingBuffer:
    def test_splits_disjoint_and_cover_all(self) -> None:
        buf = MixTrainingBuffer(3, seed=0)
        for i in range(30):
            s = _fake_sample(3, 64, i)
            buf.add(s.multitrack, s.reference)
        buf.ensure_splits()
        tr, va, ho = set(buf.train_indices), set(buf.val_indices), set(buf.holdout_indices)
        assert len(tr | va | ho) == 30
        assert tr.isdisjoint(va)
        assert tr.isdisjoint(ho)
        assert va.isdisjoint(ho)

    def test_add_wrong_shape_raises(self) -> None:
        buf = MixTrainingBuffer(2)
        with pytest.raises(ValueError):
            buf.add(np.zeros((3, 10), np.float32), np.zeros(10, np.float32))


class TestMixTrainingAgent:
    @pytest.fixture
    def tiny_config(self) -> MixTrainingConfig:
        return MixTrainingConfig(
            n_channels=4,
            audio_len=2048,
            sample_rate=48000,
            n_samples=16,
            batch_size=4,
            n_epochs=2,
            lr=1e-2,
            train_ratio=0.5,
            val_ratio=0.25,
            holdout_ratio=0.25,
            split_seed=1,
            checkpoint_every=1,
            eval_holdout_every=1,
        )

    def test_run_creates_checkpoints_and_metrics(
        self,
        tiny_config: MixTrainingConfig,
    ) -> None:
        agent = MixTrainingAgent(tiny_config, device=torch.device("cpu"))
        agent.load_synthetic_corpus()
        with tempfile.TemporaryDirectory() as tmp:
            agent.run(checkpoint_dir=tmp)
            assert os.path.isfile(os.path.join(tmp, "mix_console_best.pt"))
            assert os.path.isfile(os.path.join(tmp, "mix_console_last.pt"))
            mpath = os.path.join(tmp, "metrics.jsonl")
            assert os.path.isfile(mpath)
            lines = open(mpath, encoding="utf-8").read().strip().splitlines()
            assert len(lines) == tiny_config.n_epochs
            row = json.loads(lines[0])
            assert "train" in row and "val" in row and "holdout" in row
            assert "stft" in row["val"]

    def test_resume_continues_epoch_counter(
        self,
        tiny_config: MixTrainingConfig,
    ) -> None:
        agent = MixTrainingAgent(tiny_config, device=torch.device("cpu"))
        agent.load_synthetic_corpus()
        with tempfile.TemporaryDirectory() as tmp:
            agent.run(checkpoint_dir=tmp)
            last = os.path.join(tmp, "mix_console_last.pt")
            cfg2 = MixTrainingConfig(
                **{**tiny_config.__dict__, "n_epochs": tiny_config.n_epochs * 2},
            )
            agent2 = MixTrainingAgent(cfg2, device=torch.device("cpu"))
            agent2.load_synthetic_corpus()
            agent2.run(checkpoint_dir=tmp, resume_path=last)
            mpath = os.path.join(tmp, "metrics.jsonl")
            nlines = len(open(mpath, encoding="utf-8").read().strip().splitlines())
            assert nlines == cfg2.n_epochs
