"""Safety tests for external ML training dataset loaders."""

import os
import sys
import io
import zipfile

import numpy as np
import pytest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))


def _npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def _npy_header_only(*, shape, dtype=np.float32):
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
            "fortran_order": False,
            "shape": tuple(shape),
        },
    )
    return stream.getvalue()


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


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_classifier_npz_rejects_text_features_and_oversized_text_labels(tmp_path):
    from ml.train_classifier import _load_dataset_from_file as load_classifier

    text_features_path = tmp_path / "classifier_text_features.npz"
    np.savez(
        text_features_path,
        data=np.full((1, 4, 5), "not-a-number"),
        labels=np.array(["kick"]),
    )
    with pytest.raises(ValueError, match="plain numeric dtype"):
        load_classifier(str(text_features_path), n_mels=4, n_frames=5)

    oversized_labels_path = tmp_path / "classifier_oversized_labels.npz"
    np.savez(
        oversized_labels_path,
        data=np.ones((1, 4, 5), dtype=np.float32),
        labels=np.array(["x" * 65]),
    )
    with pytest.raises(ValueError, match="plain numeric dtype"):
        load_classifier(str(oversized_labels_path), n_mels=4, n_frames=5)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_gain_pan_npz_rejects_header_allocation_bomb_before_numpy_load(tmp_path, monkeypatch):
    from ml import npz_archive_safety
    from ml.train_gain_predictor import _load_dataset_from_file as load_gain_pan

    path = tmp_path / "gain_header_bomb.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "channel_audios.npy",
            _npy_header_only(shape=(1, 2, npz_archive_safety.MAX_NPZ_ARRAY_ELEMENTS)),
        )
        archive.writestr("gains.npy", _npy_bytes(np.zeros((1, 2), dtype=np.float32)))
        archive.writestr("pans.npy", _npy_bytes(np.zeros((1, 2), dtype=np.float32)))

    called = False

    def fail_if_numpy_load_runs(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("np.load must not run for an unsafe archive")

    monkeypatch.setattr(npz_archive_safety.np, "load", fail_if_numpy_load_runs)

    with pytest.raises(ValueError, match="element limit exceeded|byte limit exceeded"):
        load_gain_pan(str(path), n_channels=2)
    assert called is False

    zero_product_bomb = tmp_path / "gain_zero_product_bomb.npz"
    with zipfile.ZipFile(zero_product_bomb, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "channel_audios.npy",
            _npy_header_only(shape=(0, npz_archive_safety.MAX_NPZ_ARRAY_ELEMENTS + 1)),
        )
        archive.writestr("gains.npy", _npy_bytes(np.zeros((0, 2), dtype=np.float32)))
        archive.writestr("pans.npy", _npy_bytes(np.zeros((0, 2), dtype=np.float32)))
    with pytest.raises(ValueError, match="dimension limit exceeded"):
        load_gain_pan(str(zero_product_bomb), n_channels=2)
    assert called is False


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_gain_pan_npz_rejects_member_count_and_uncompressed_budgets(tmp_path, monkeypatch):
    from ml import npz_archive_safety
    from ml.train_gain_predictor import _load_dataset_from_file as load_gain_pan

    many_members = tmp_path / "many_members.npz"
    arrays = {
        "channel_audios": np.ones((1, 2, 8), dtype=np.float32),
        "gains": np.zeros((1, 2), dtype=np.float32),
        "pans": np.zeros((1, 2), dtype=np.float32),
        **{f"extra_{index}": np.zeros((1,), dtype=np.float32) for index in range(6)},
    }
    np.savez(many_members, **arrays)
    with pytest.raises(ValueError, match="member count limit exceeded"):
        load_gain_pan(str(many_members), n_channels=2)

    oversized = tmp_path / "uncompressed_budget.npz"
    np.savez_compressed(
        oversized,
        channel_audios=np.zeros((1, 2, 128), dtype=np.float32),
        gains=np.zeros((1, 2), dtype=np.float32),
        pans=np.zeros((1, 2), dtype=np.float32),
    )
    monkeypatch.setattr(npz_archive_safety, "MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES", 256)
    with pytest.raises(ValueError, match="total uncompressed byte limit exceeded"):
        load_gain_pan(str(oversized), n_channels=2)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_gain_pan_npz_rejects_compressed_size_and_unsafe_members(tmp_path, monkeypatch):
    from ml import npz_archive_safety
    from ml.train_gain_predictor import _load_dataset_from_file as load_gain_pan

    valid = tmp_path / "compressed_size.npz"
    np.savez(
        valid,
        channel_audios=np.ones((1, 2, 8), dtype=np.float32),
        gains=np.zeros((1, 2), dtype=np.float32),
        pans=np.zeros((1, 2), dtype=np.float32),
    )
    with monkeypatch.context() as patch_context:
        patch_context.setattr(npz_archive_safety, "MAX_NPZ_COMPRESSED_BYTES", 64)
        with pytest.raises(ValueError, match="compressed file size limit exceeded"):
            load_gain_pan(str(valid), n_channels=2)

    unsafe_name = tmp_path / "unsafe_name.npz"
    with zipfile.ZipFile(unsafe_name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("nested/channel_audios.npy", _npy_bytes(np.ones((1, 2, 8))))
        archive.writestr("gains.npy", _npy_bytes(np.zeros((1, 2))))
        archive.writestr("pans.npy", _npy_bytes(np.zeros((1, 2))))
    with pytest.raises(ValueError, match="Unsafe NPZ member name"):
        load_gain_pan(str(unsafe_name), n_channels=2)

    duplicate = tmp_path / "duplicate.npz"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("channel_audios.npy", _npy_bytes(np.ones((1, 2, 8))))
            archive.writestr("gains.npy", _npy_bytes(np.zeros((1, 2))))
            archive.writestr("gains.npy", _npy_bytes(np.zeros((1, 2))))
            archive.writestr("pans.npy", _npy_bytes(np.zeros((1, 2))))
    with pytest.raises(ValueError, match="Duplicate or empty NPZ array key"):
        load_gain_pan(str(duplicate), n_channels=2)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_gain_pan_npz_rejects_payload_size_mismatch_and_unsafe_channel_expansion(tmp_path):
    from ml.train_gain_predictor import _load_dataset_from_file as load_gain_pan

    mismatch = tmp_path / "payload_mismatch.npz"
    with zipfile.ZipFile(mismatch, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("channel_audios.npy", _npy_header_only(shape=(1, 2, 8)))
        archive.writestr("gains.npy", _npy_bytes(np.zeros((1, 2), dtype=np.float32)))
        archive.writestr("pans.npy", _npy_bytes(np.zeros((1, 2), dtype=np.float32)))
    with pytest.raises(ValueError, match="payload size mismatch"):
        load_gain_pan(str(mismatch), n_channels=2)

    expansion = tmp_path / "unsafe_expansion.npz"
    np.savez(
        expansion,
        channel_audios=np.ones((1, 1, 130_000), dtype=np.float32),
        gains=np.zeros((1, 1), dtype=np.float32),
        pans=np.zeros((1, 1), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="sample element limit exceeded"):
        load_gain_pan(str(expansion), n_channels=64)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mix_console_npz_rejects_header_bomb_before_numpy_load(tmp_path, monkeypatch):
    from ml import npz_archive_safety
    from ml.train_mix_console import _load_dataset_from_file as load_mix_console

    path = tmp_path / "console_header_bomb.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "multitracks.npy",
            _npy_header_only(shape=(1, 2, npz_archive_safety.MAX_NPZ_ARRAY_ELEMENTS)),
        )
        archive.writestr(
            "references.npy",
            _npy_bytes(np.zeros((1, 2), dtype=np.float32)),
        )

    called = False

    def fail_if_numpy_load_runs(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("np.load must not run for an unsafe mix-console archive")

    monkeypatch.setattr(npz_archive_safety.np, "load", fail_if_numpy_load_runs)

    with pytest.raises(ValueError, match="element limit exceeded|byte limit exceeded"):
        load_mix_console(str(path), n_channels=2)
    assert called is False


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mix_console_loader_rejects_padding_and_materialization_budgets(
    tmp_path,
    monkeypatch,
):
    from ml import train_mix_console as training

    expansion = tmp_path / "console_expansion.npz"
    np.savez(
        expansion,
        multitracks=np.ones((1, 1, 130_000), dtype=np.float32),
        references=np.ones((1, 130_000), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="sample element limit exceeded"):
        training._load_dataset_from_file(str(expansion), n_channels=64)

    cumulative = tmp_path / "console_materialized.npz"
    np.savez(
        cumulative,
        multitracks=np.ones((2, 2, 32), dtype=np.float32),
        references=np.ones((2, 32), dtype=np.float32),
    )
    monkeypatch.setattr(training, "MAX_MIX_CONSOLE_DATASET_ELEMENTS", 128)
    with pytest.raises(ValueError, match="materialized element limit exceeded"):
        training._load_dataset_from_file(str(cumulative), n_channels=2)
