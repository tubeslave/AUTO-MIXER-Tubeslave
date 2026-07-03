import io
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from ml.agent_training_service import AgentTrainingService


DATASET_ROW = '{"spectrogram": [[0.1]], "label": 1}\n'


def _service(tmp_path: Path) -> AgentTrainingService:
    return AgentTrainingService(
        {
            "artifacts_dir": str(tmp_path / "models"),
            "dataset_dir": str(tmp_path / "datasets"),
            "state_file": str(tmp_path / "training_state.json"),
            "study": {"output_dir": str(tmp_path / "knowledge")},
        },
        repo_root=str(tmp_path),
    )


def test_agent_training_service_extracts_safe_zip_dataset(tmp_path: Path):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("nested/features.jsonl", DATASET_ROW)

    prepared = _service(tmp_path)._prepare_dataset_path("channel_classifier", archive)

    assert prepared.name == "features.jsonl"
    assert prepared.parent == tmp_path / ".tmp_dataset"
    assert prepared.read_text(encoding="utf-8") == DATASET_ROW


def test_agent_training_service_extracts_safe_tar_dataset(tmp_path: Path):
    archive = tmp_path / "dataset.tar"
    payload = DATASET_ROW.encode("utf-8")
    info = tarfile.TarInfo("nested/features.jsonl")
    info.size = len(payload)
    with tarfile.open(archive, "w") as tf:
        tf.addfile(info, io.BytesIO(payload))

    prepared = _service(tmp_path)._prepare_dataset_path("channel_classifier", archive)

    assert prepared.name == "features.jsonl"
    assert prepared.parent == tmp_path / ".tmp_dataset"
    assert prepared.read_text(encoding="utf-8") == DATASET_ROW


def test_agent_training_service_rejects_zip_member_traversal(tmp_path: Path):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("nested/features.jsonl", DATASET_ROW)
        zf.writestr("../evil.jsonl", DATASET_ROW)

    with pytest.raises(ValueError, match="Unsafe archive entry path"):
        _service(tmp_path)._prepare_dataset_path("channel_classifier", archive)

    assert not (tmp_path / "evil.jsonl").exists()


def test_agent_training_service_rejects_tar_member_traversal(tmp_path: Path):
    archive = tmp_path / "dataset.tar"
    payload = DATASET_ROW.encode("utf-8")
    info = tarfile.TarInfo("../evil.jsonl")
    info.size = len(payload)
    with tarfile.open(archive, "w") as tf:
        tf.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="Unsafe archive entry path"):
        _service(tmp_path)._prepare_dataset_path("channel_classifier", archive)

    assert not (tmp_path / "evil.jsonl").exists()


def test_agent_training_service_rejects_archive_links(tmp_path: Path):
    zip_archive = tmp_path / "zip_symlink.zip"
    zip_info = zipfile.ZipInfo("nested/link.jsonl")
    zip_info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(zip_archive, "w") as zf:
        zf.writestr(zip_info, "nested/features.jsonl")

    with pytest.raises(ValueError, match="Unsafe archive entry type"):
        _service(tmp_path)._prepare_dataset_path("channel_classifier", zip_archive)

    tar_archive = tmp_path / "tar_symlink.tar"
    tar_info = tarfile.TarInfo("nested/link.jsonl")
    tar_info.type = tarfile.SYMTYPE
    tar_info.linkname = "nested/features.jsonl"
    with tarfile.open(tar_archive, "w") as tf:
        tf.addfile(tar_info)

    with pytest.raises(ValueError, match="Unsafe archive entry type"):
        _service(tmp_path)._prepare_dataset_path("channel_classifier", tar_archive)
