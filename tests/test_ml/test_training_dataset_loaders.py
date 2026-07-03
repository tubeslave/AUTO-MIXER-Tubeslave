"""Safety tests for external ML training dataset loaders."""

import os
import sys

import numpy as np
import pytest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_training_npz_loaders_accept_numeric_arrays(tmp_path):
    from ml.train_classifier import _load_dataset_from_file as load_classifier
    from ml.train_gain_predictor import _load_dataset_from_file as load_gain_pan
    from ml.train_mix_console import _load_dataset_from_file as load_mix_console

    classifier_path = tmp_path / "classifier_numeric.npz"
    np.savez(
        classifier_path,
        data=np.ones((2, 4, 5), dtype=np.float32),
        labels=np.array(["kick", "unknown"]),
    )
    classifier_rows = load_classifier(str(classifier_path), n_mels=4, n_frames=5)
    assert len(classifier_rows) == 2
    assert classifier_rows[0][0].shape == (4, 5)

    gain_path = tmp_path / "gain_numeric.npz"
    np.savez(
        gain_path,
        channel_audios=np.ones((2, 2, 8), dtype=np.float32),
        gains=np.zeros((2, 2), dtype=np.float32),
        pans=np.zeros((2, 2), dtype=np.float32),
    )
    gain_rows = load_gain_pan(str(gain_path), n_channels=2)
    assert len(gain_rows) == 2
    assert gain_rows[0][0].shape == (2, 8)

    console_path = tmp_path / "console_numeric.npz"
    np.savez(
        console_path,
        multitracks=np.ones((2, 2, 8), dtype=np.float32),
        references=np.ones((2, 8), dtype=np.float32),
    )
    multitracks, references = load_mix_console(str(console_path), n_channels=2)
    assert len(multitracks) == 2
    assert multitracks[0].shape == (2, 8)
    assert references[0].shape == (8,)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_training_npz_loaders_reject_object_arrays(tmp_path):
    from ml.train_classifier import _load_dataset_from_file as load_classifier
    from ml.train_gain_predictor import _load_dataset_from_file as load_gain_pan
    from ml.train_mix_console import _load_dataset_from_file as load_mix_console

    classifier_path = tmp_path / "classifier_object.npz"
    np.savez(
        classifier_path,
        data=np.array([np.ones((4, 5), dtype=np.float32)], dtype=object),
        labels=np.array(["kick"], dtype=object),
    )
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_classifier(str(classifier_path), n_mels=4, n_frames=5)

    gain_path = tmp_path / "gain_object.npz"
    np.savez(
        gain_path,
        channel_audios=np.array([np.ones((2, 8), dtype=np.float32)], dtype=object),
        gains=np.array([[0.0, 0.0]], dtype=np.float32),
        pans=np.array([[0.0, 0.0]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_gain_pan(str(gain_path), n_channels=2)

    console_path = tmp_path / "console_object.npz"
    np.savez(
        console_path,
        multitracks=np.array([np.ones((2, 8), dtype=np.float32)], dtype=object),
        references=np.ones((1, 8), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_mix_console(str(console_path), n_channels=2)
