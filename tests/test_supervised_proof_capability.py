from __future__ import annotations

import json
import os
import secrets
import subprocess
from pathlib import Path

import pytest
import supervised_proof_capability as capability_module

from supervised_proof_capability import (
    M110_AUTHORITY_FD_ENV,
    M110_RUNTIME_CRITICAL_PATHS,
    build_m110_prestart_manifest,
    consume_m110_attended_launch_authority,
    sign_m110_request,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["/usr/bin/git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _committed_runtime_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    for relative_path in M110_RUNTIME_CRITICAL_PATHS:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative_path}\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "m110@example.invalid")
    _git(root, "config", "user.name", "M110 Test")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "fixture")
    return root


def _consume_packet(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    packet: bytes,
):
    read_fd, write_fd = os.pipe()
    os.write(write_fd, packet)
    os.close(write_fd)
    monkeypatch.setenv(M110_AUTHORITY_FD_ENV, str(read_fd))
    return consume_m110_attended_launch_authority(repo_root)


def test_inherited_fd_manifest_and_hmac_are_process_private_and_single_use(
    tmp_path,
    monkeypatch,
):
    root = _committed_runtime_repo(tmp_path)
    launch_nonce = "a" * 32
    secret = secrets.token_bytes(32)
    manifest = build_m110_prestart_manifest(root, launch_nonce=launch_nonce)
    raw = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    binding, authority = _consume_packet(
        monkeypatch,
        root,
        len(raw).to_bytes(4, "big") + raw + secret,
    )

    assert binding["mode"] == "m110_attended_runner"
    assert binding["tracked_tree_clean"] is True
    assert binding["m110_attended_eligible"] is True
    assert secret.hex() not in json.dumps(binding)
    assert "authority_secret_sha256" in binding
    assert M110_AUTHORITY_FD_ENV not in os.environ
    request = sign_m110_request(
        {
            "type": "capture_wing_master_route_proof",
            "request_nonce": "b" * 32,
        },
        secret=secret,
    )
    assert authority is not None
    assert authority.verify_and_tombstone_request(request) == (
        True,
        "m110_authority_request_verified",
    )
    assert authority.verify_and_tombstone_request(request) == (
        False,
        "m110_request_nonce_replayed",
    )


def test_unrelated_untracked_report_does_not_become_source_authority(
    tmp_path,
):
    root = _committed_runtime_repo(tmp_path)
    report = root / "reports" / "generated.json"
    report.parent.mkdir()
    report.write_text("{}", encoding="utf-8")

    manifest = build_m110_prestart_manifest(root, launch_nonce="c" * 32)

    assert manifest["launch_nonce"] == "c" * 32
    assert report.relative_to(root).as_posix() not in {
        row["path"] for row in manifest["files"]
    }


def test_untracked_python_shadow_file_blocks_manifest(tmp_path):
    root = _committed_runtime_repo(tmp_path)
    (root / "backend" / "numpy.py").write_text("raise SystemExit\n")

    with pytest.raises(RuntimeError, match="unsafe untracked path"):
        build_m110_prestart_manifest(root, launch_nonce="9" * 32)


@pytest.mark.parametrize(
    "relative_path",
    [
        "backend/json.pyc",
        "backend/wing_client.so",
        "backend/handlers/connection_handlers.so",
        "backend/osc/enhanced_osc_client.so",
        "backend/sitecustomize.py",
        "tools/json.pyc",
    ],
)
def test_ignored_import_shadow_artifact_blocks_manifest(tmp_path, relative_path):
    root = _committed_runtime_repo(tmp_path)
    (root / ".gitignore").write_text(
        "*.py\n*.pyw\n*.py[cod]\n*.so\n",
        encoding="utf-8",
    )
    _git(root, "add", ".gitignore")
    _git(root, "commit", "-q", "-m", "ignore runtime artifacts")
    shadow = root / relative_path
    shadow.write_bytes(b"ignored executable shadow")

    with pytest.raises(RuntimeError, match="ignored import-shadow artifact"):
        build_m110_prestart_manifest(root, launch_nonce="7" * 32)


def test_ignored_non_import_build_cache_does_not_block_manifest(tmp_path):
    root = _committed_runtime_repo(tmp_path)
    (root / ".gitignore").write_text("backend/build/\n", encoding="utf-8")
    _git(root, "add", ".gitignore")
    _git(root, "commit", "-q", "-m", "ignore build output")
    cache = root / "backend" / "build" / "server" / "localpycs" / "struct.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"build output")

    manifest = build_m110_prestart_manifest(root, launch_nonce="4" * 32)

    assert manifest["launch_nonce"] == "4" * 32


def test_untracked_report_symlink_blocks_manifest(tmp_path):
    root = _committed_runtime_repo(tmp_path)
    target = root / "target.json"
    target.write_text("{}")
    report = root / "reports" / "generated.json"
    report.parent.mkdir()
    report.symlink_to(target)

    with pytest.raises(RuntimeError, match="unsafe untracked path|non-symlink"):
        build_m110_prestart_manifest(root, launch_nonce="8" * 32)


def test_dirty_tracked_source_fails_before_authority_issuance(tmp_path):
    root = _committed_runtime_repo(tmp_path)
    path = root / M110_RUNTIME_CRITICAL_PATHS[0]
    path.write_text("dirty\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="tracked source tree is dirty"):
        build_m110_prestart_manifest(root, launch_nonce="d" * 32)


def test_manifest_trailing_bytes_fail_closed(tmp_path, monkeypatch):
    root = _committed_runtime_repo(tmp_path)
    secret = secrets.token_bytes(32)
    manifest = build_m110_prestart_manifest(root, launch_nonce="e" * 32)
    raw = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    with pytest.raises(RuntimeError, match="trailing bytes"):
        _consume_packet(
            monkeypatch,
            root,
            len(raw).to_bytes(4, "big") + raw + secret + b"x",
        )


def test_regular_file_cannot_replace_one_shot_authority_pipe(
    tmp_path,
    monkeypatch,
):
    root = _committed_runtime_repo(tmp_path)
    packet_file = tmp_path / "authority.bin"
    packet_file.write_bytes(b"not-a-pipe")
    fd = os.open(packet_file, os.O_RDONLY)
    monkeypatch.setenv(M110_AUTHORITY_FD_ENV, str(fd))

    with pytest.raises(RuntimeError, match="one-shot pipe"):
        consume_m110_attended_launch_authority(root)


def test_self_declared_manifest_bytes_cannot_hide_source_drift(
    tmp_path,
    monkeypatch,
):
    root = _committed_runtime_repo(tmp_path)
    secret = secrets.token_bytes(32)
    manifest = build_m110_prestart_manifest(root, launch_nonce="f" * 32)
    raw = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    (root / M110_RUNTIME_CRITICAL_PATHS[-1]).write_text(
        "changed after manifest\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="tracked source tree is dirty"):
        _consume_packet(
            monkeypatch,
            root,
            len(raw).to_bytes(4, "big") + raw + secret,
        )


def test_consumed_authority_permanently_closes_on_real_source_drift(
    tmp_path,
    monkeypatch,
):
    root = _committed_runtime_repo(tmp_path)
    secret = secrets.token_bytes(32)
    manifest = build_m110_prestart_manifest(root, launch_nonce="6" * 32)
    raw = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    _, authority = _consume_packet(
        monkeypatch,
        root,
        len(raw).to_bytes(4, "big") + raw + secret,
    )
    assert authority is not None
    assert authority.source_snapshot_status() == (
        True,
        "m110_live_source_snapshot_current",
    )

    (root / M110_RUNTIME_CRITICAL_PATHS[0]).write_text(
        "changed after launch\n",
        encoding="utf-8",
    )

    assert authority.source_snapshot_status() == (
        False,
        "m110_live_source_snapshot_drift",
    )
    request = sign_m110_request(
        {"type": "unit", "request_nonce": "5" * 32},
        secret=secret,
    )
    assert authority.verify_and_tombstone_request(request) == (
        False,
        "m110_launch_authority_closed",
    )


def test_nonce_capacity_fails_closed_without_reenabling_old_nonce(monkeypatch):
    monkeypatch.setattr(capability_module, "M110_MAX_TOMBSTONED_NONCES", 2)
    secret = secrets.token_bytes(32)
    authority = capability_module.M110LaunchAuthority(
        launch_nonce="1" * 32,
        manifest_sha256="2" * 64,
        head="3" * 40,
        tree="4" * 40,
        authority_secret_sha256="5" * 64,
        _secret=bytearray(secret),
        _repo_root=Path.cwd(),
    )
    authority.source_snapshot_status = lambda: (
        True,
        "m110_live_source_snapshot_current",
    )

    def request(nonce):
        return sign_m110_request(
            {"type": "unit", "request_nonce": nonce},
            secret=secret,
        )

    first = request("a" * 32)
    assert authority.verify_and_tombstone_request(first)[0] is True
    assert authority.verify_and_tombstone_request(request("b" * 32))[0] is True
    assert authority.verify_and_tombstone_request(request("c" * 32)) == (
        False,
        "m110_request_nonce_capacity_exhausted",
    )
    assert authority.verify_and_tombstone_request(first) == (
        False,
        "m110_request_nonce_replayed",
    )


def test_closed_launch_authority_rejects_even_zero_key_signature():
    secret = secrets.token_bytes(32)
    authority = capability_module.M110LaunchAuthority(
        launch_nonce="1" * 32,
        manifest_sha256="2" * 64,
        head="3" * 40,
        tree="4" * 40,
        authority_secret_sha256="5" * 64,
        _secret=bytearray(secret),
        _repo_root=Path.cwd(),
    )
    authority.close()
    request = sign_m110_request(
        {"type": "unit", "request_nonce": "d" * 32},
        secret=b"\x00" * 32,
    )

    assert authority.verify_and_tombstone_request(request) == (
        False,
        "m110_launch_authority_closed",
    )
