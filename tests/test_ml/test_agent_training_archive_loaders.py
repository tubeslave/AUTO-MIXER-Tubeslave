import gzip
import io
import stat
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ml.agent_training_service import (
    AgentTrainingService,
    REMOTE_MANIFEST_TARGET_FIELDS,
)


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


@pytest.mark.parametrize("archive_kind", ["zip", "tar", "gzip"])
def test_agent_training_service_bounds_decompressed_archive_bytes(
    tmp_path: Path,
    archive_kind: str,
):
    payload = b"x" * (1024 * 1024 + 1)
    if archive_kind == "zip":
        archive = tmp_path / "dataset.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("features.jsonl", payload)
    elif archive_kind == "tar":
        archive = tmp_path / "dataset.tar"
        info = tarfile.TarInfo("features.jsonl")
        info.size = len(payload)
        with tarfile.open(archive, "w") as tf:
            tf.addfile(info, io.BytesIO(payload))
    else:
        archive = tmp_path / "features.jsonl.gz"
        with gzip.open(archive, "wb") as gz:
            gz.write(payload)

    service = _service(tmp_path)
    service.config["max_dataset_bytes"] = 1024 * 1024
    with pytest.raises(RuntimeError, match="archive member exceeds max_dataset_bytes"):
        service._prepare_dataset_path("channel_classifier", archive)


def test_agent_training_study_fetch_rejects_redirect_without_following(tmp_path: Path):
    service = _service(tmp_path)
    response = MagicMock(status_code=302)

    with patch(
        "ml.agent_training_service.socket.getaddrinfo",
        return_value=[(2, 1, 6, "", ("93.184.216.34", 443))],
    ), patch(
        "ml.agent_training_service.requests.get",
        return_value=response,
    ) as request_get:
        with pytest.raises(RuntimeError, match="training_study_redirect_forbidden"):
            service._fetch_study_text(
                "https://arxiv.org/resource",
                timeout=5.0,
                max_bytes=1024,
            )

    assert request_get.call_args.kwargs["allow_redirects"] is False


@pytest.mark.asyncio
async def test_agent_training_service_manifest_cannot_override_model_output_path(
    tmp_path: Path,
):
    safe_output = "models/channel_classifier.pt"
    outside_output = tmp_path.parent / "manifest-controlled-model.pt"
    service = AgentTrainingService(
        {
            "artifacts_dir": str(tmp_path / "models"),
            "dataset_dir": str(tmp_path / "datasets"),
            "state_file": str(tmp_path / "training_state.json"),
            "study": {"output_dir": str(tmp_path / "knowledge")},
            "targets": {
                "channel_classifier": {
                    "enabled": True,
                    "output_path": safe_output,
                    "train_without_dataset": True,
                },
            },
        },
        repo_root=str(tmp_path),
    )
    service._fetch_manifest_with_retries = AsyncMock(
        return_value={
            "targets": {
                "channel_classifier": {
                    "output_path": str(outside_output),
                    "n_epochs": 1,
                },
            },
        }
    )
    service._resolve_dataset = AsyncMock(
        return_value=(None, None, True, None, {}),
    )
    captured_config = {}

    def capture_training_config(**kwargs):
        captured_config.update(kwargs["config"])

    service._train_target = capture_training_config

    await service._run_once("security-path-test", True, "security-regression")

    assert captured_config["output_path"] == safe_output
    assert captured_config["n_epochs"] == 20
    assert not outside_output.exists()


@pytest.mark.asyncio
async def test_agent_training_manifest_cannot_override_gain_pan_work_budget(tmp_path: Path):
    service = AgentTrainingService(
        {
            "enabled": True,
            "artifacts_dir": str(tmp_path / "models"),
            "dataset_dir": str(tmp_path / "datasets"),
            "state_file": str(tmp_path / "training_state.json"),
            "study": {"output_dir": str(tmp_path / "knowledge")},
            "targets": {
                "gain_pan_predictor": {
                    "enabled": True,
                    "train_without_dataset": True,
                    "n_epochs": 2,
                    "batch_size": 2,
                    "n_channels": 2,
                    "n_samples": 4,
                },
            },
        },
        repo_root=str(tmp_path),
    )
    service._fetch_manifest_with_retries = AsyncMock(
        return_value={
            "targets": {
                "gain_pan_predictor": {
                    "n_epochs": 10**9,
                    "batch_size": 10**9,
                    "n_channels": 10**9,
                    "n_samples": 10**9,
                    "lr": float("nan"),
                },
            },
        }
    )
    service._resolve_dataset = AsyncMock(return_value=(None, None, True, None, {}))
    captured_config = {}

    def capture_training_config(**kwargs):
        captured_config.update(kwargs["config"])

    service._train_target = capture_training_config

    await service._run_once("security-path-test", True, "security-regression")

    assert captured_config["n_epochs"] == 2
    assert captured_config["batch_size"] == 2
    assert captured_config["n_channels"] == 2
    assert captured_config["n_samples"] == 4
    assert captured_config["lr"] == pytest.approx(1e-3)


@pytest.mark.asyncio
async def test_agent_training_manifest_cannot_override_mix_console_work_budget(tmp_path: Path):
    service = AgentTrainingService(
        {
            "enabled": True,
            "artifacts_dir": str(tmp_path / "models"),
            "dataset_dir": str(tmp_path / "datasets"),
            "state_file": str(tmp_path / "training_state.json"),
            "study": {"output_dir": str(tmp_path / "knowledge")},
            "targets": {
                "mix_console": {
                    "enabled": True,
                    "train_without_dataset": True,
                    "n_epochs": 2,
                    "n_channels": 2,
                    "audio_len": 64,
                    "n_samples": 4,
                    "sample_rate": 48000,
                },
            },
        },
        repo_root=str(tmp_path),
    )
    service._fetch_manifest_with_retries = AsyncMock(
        return_value={
            "targets": {
                "mix_console": {
                    "n_epochs": 10**9,
                    "n_channels": 10**9,
                    "audio_len": 10**9,
                    "n_samples": 10**9,
                    "sample_rate": 10**9,
                    "lr": float("nan"),
                },
            },
        }
    )
    service._resolve_dataset = AsyncMock(return_value=(None, None, True, None, {}))
    captured_config = {}

    def capture_training_config(**kwargs):
        captured_config.update(kwargs["config"])

    service._train_target = capture_training_config

    await service._run_once("security-path-test", True, "security-regression")

    assert captured_config["n_epochs"] == 2
    assert captured_config["n_channels"] == 2
    assert captured_config["audio_len"] == 64
    assert captured_config["n_samples"] == 4
    assert captured_config["sample_rate"] == 48000
    assert captured_config["lr"] == pytest.approx(1e-2)


@pytest.mark.asyncio
async def test_agent_training_service_start_requires_local_enablement(tmp_path: Path):
    service = _service(tmp_path)

    result = await service.start_once(
        force=True,
        reason="ingress",
        manifest_url="https://169.254.169.254/manifest.json",
    )

    assert result == {
        "started": False,
        "reason": "training disabled by local config",
    }
    assert service._active_task is None


def test_agent_training_url_policy_rejects_private_and_unapproved_hosts(tmp_path: Path):
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="domain_not_allowed"):
        service._validate_training_url(
            "https://169.254.169.254/dataset.jsonl",
            {"huggingface.co"},
        )

    with patch(
        "ml.agent_training_service.socket.getaddrinfo",
        return_value=[(2, 1, 6, "", ("127.0.0.1", 443))],
    ):
        with pytest.raises(ValueError, match="public_ip_required"):
            service._validate_training_url(
                "https://huggingface.co/dataset.jsonl",
                {"huggingface.co"},
            )


def test_agent_training_manifest_cannot_enable_or_expand_resource_budget(tmp_path: Path):
    service = _service(tmp_path)
    local_target = dict(service.config["targets"]["channel_classifier"])
    manifest_target = {
        "enabled": True,
        "train_without_dataset": True,
        "n_epochs": 10**9,
        "batch_size": 10**9,
        "output_path": "/tmp/manifest-model.pt",
        "dataset_id": "trusted-id",
    }

    merged = dict(local_target)
    merged.update(
        {
            key: value
            for key, value in manifest_target.items()
            if key in REMOTE_MANIFEST_TARGET_FIELDS
        }
    )

    assert merged["enabled"] is False
    assert merged["train_without_dataset"] is False
    assert merged["n_epochs"] == local_target["n_epochs"]
    assert merged["batch_size"] == local_target["batch_size"]
    assert merged["output_path"] == local_target["output_path"]
    assert merged["dataset_id"] == "trusted-id"
