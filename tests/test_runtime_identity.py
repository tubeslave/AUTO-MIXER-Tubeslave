from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import backend.runtime_identity as runtime_identity


def _fake_git(_root: Path, *arguments: str) -> bytes:
    if arguments == ("rev-parse", "HEAD"):
        return b"a" * 40 + b"\n"
    if arguments == ("rev-parse", "HEAD^{tree}"):
        return b"b" * 40 + b"\n"
    if arguments[0] in {"status", "diff"}:
        return b""
    raise AssertionError(arguments)


def test_source_runtime_manifest_binds_critical_file_bytes(tmp_path, monkeypatch):
    source = tmp_path / "backend" / "critical.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        runtime_identity,
        "RUNTIME_CRITICAL_SOURCE_PATHS",
        ("backend/critical.py",),
    )
    monkeypatch.setattr(runtime_identity, "_git_bytes", _fake_git)

    first = runtime_identity.capture_source_runtime_components(tmp_path)
    source.write_text("value = 2\n", encoding="utf-8")
    second = runtime_identity.capture_source_runtime_components(tmp_path)

    assert first["source"]["worktree_dirty"] is False
    assert first["code"]["files"][0]["path"] == "backend/critical.py"
    assert first["code"]["files_sha256"] != second["code"]["files_sha256"]
    assert first["source"]["source_content_sha256"] == first["code"]["files_sha256"]


def test_source_runtime_manifest_rejects_symlink(tmp_path, monkeypatch):
    target = tmp_path / "target.py"
    target.write_text("value = 1\n", encoding="utf-8")
    link = tmp_path / "backend" / "critical.py"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    monkeypatch.setattr(
        runtime_identity,
        "RUNTIME_CRITICAL_SOURCE_PATHS",
        ("backend/critical.py",),
    )
    monkeypatch.setattr(runtime_identity, "_git_bytes", _fake_git)

    with pytest.raises(RuntimeError, match="non-symlink"):
        runtime_identity.capture_source_runtime_components(tmp_path)


def test_runtime_manifest_hash_excludes_only_request_challenge():
    identity = {
        "schema": runtime_identity.RUNTIME_IDENTITY_SCHEMA,
        "process": {"pid": 1},
    }
    digest = runtime_identity.runtime_identity_manifest_sha256(identity)
    challenged = {**copy.deepcopy(identity), "request_challenge_nonce": "c" * 32}

    assert runtime_identity.runtime_identity_manifest_sha256(challenged) == digest
    challenged["process"]["pid"] = 2
    assert runtime_identity.runtime_identity_manifest_sha256(challenged) != digest
    assert len(digest) == hashlib.sha256().digest_size * 2


def test_packaged_runtime_manifest_round_trip_binds_build_source(
    tmp_path, monkeypatch
):
    source = tmp_path / "backend" / "critical.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        runtime_identity,
        "RUNTIME_CRITICAL_SOURCE_PATHS",
        ("backend/critical.py",),
    )
    monkeypatch.setattr(runtime_identity, "_git_bytes", _fake_git)
    components = runtime_identity.capture_source_runtime_components(tmp_path)
    manifest = runtime_identity.build_packaged_runtime_source_manifest(components)
    manifest_path = tmp_path / runtime_identity.PACKAGED_RUNTIME_SOURCE_MANIFEST_FILENAME
    manifest_path.write_bytes(runtime_identity.canonical_json_bytes(manifest) + b"\n")

    loaded, binding = runtime_identity.load_packaged_runtime_source_components(
        manifest_path
    )

    assert loaded["source"]["observation"] == "packaged_build_manifest"
    assert loaded["source"]["head"] == "a" * 40
    assert loaded["code"]["files"][0]["sha256"] == hashlib.sha256(
        b"value = 1\n"
    ).hexdigest()
    assert binding == {
        "mode": "packaged_build_manifest",
        "packaged_manifest_schema": (
            runtime_identity.PACKAGED_RUNTIME_SOURCE_MANIFEST_SCHEMA
        ),
        "packaged_manifest_name": manifest_path.name,
        "packaged_manifest_size": manifest_path.stat().st_size,
        "packaged_manifest_file_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "packaged_manifest_sha256": manifest["manifest_sha256"],
        "bound_before_process_start": True,
        "proof_eligible": False,
    }


@pytest.mark.parametrize("mutation", ["file_hash", "manifest_hash", "empty_files"])
def test_packaged_runtime_manifest_rejects_tampering(
    tmp_path, monkeypatch, mutation
):
    source = tmp_path / "backend" / "critical.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        runtime_identity,
        "RUNTIME_CRITICAL_SOURCE_PATHS",
        ("backend/critical.py",),
    )
    monkeypatch.setattr(runtime_identity, "_git_bytes", _fake_git)
    manifest = runtime_identity.build_packaged_runtime_source_manifest(
        runtime_identity.capture_source_runtime_components(tmp_path)
    )
    if mutation == "file_hash":
        manifest["code"]["files"][0]["sha256"] = "0" * 64
    elif mutation == "manifest_hash":
        manifest["manifest_sha256"] = "0" * 64
    else:
        manifest["code"]["files"] = []
    manifest_path = tmp_path / runtime_identity.PACKAGED_RUNTIME_SOURCE_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="manifest"):
        runtime_identity.load_packaged_runtime_source_components(manifest_path)
