#!/usr/bin/env python3
"""Attended M110 proof-owned backend and persistent control-session runner.

The runner itself is WebSocket-only and never imports a mixer transport.  Its
isolated child backend may open OSC/UDP only after the committed CP05 hardware
gate, exact operator authorization and fixed RFC1918 unicast validation pass.
Full write execution remains blocked until the later capability checkpoints.
"""

from __future__ import annotations

import sys

# A normal ``python tools/...`` launch puts the repository's ``tools``
# directory on sys.path before this file can inspect it.  Require CPython's
# isolated safe-path mode before importing even standard-library helpers; the
# same file also serves as the child's stdlib-only bootstrap below.
if __name__ == "__main__" and (
    not sys.flags.isolated or not sys.flags.no_site
):
    print(
        "M110 attended runner requires: python -I -S " + __file__,
        file=sys.stderr,
    )
    raise SystemExit(2)

import argparse
import atexit
import asyncio
import hashlib
import importlib.abc
import importlib.machinery
import ipaddress
import json
import math
import os
import re
import runpy
import secrets
import shutil
import site
import stat
import subprocess
import sysconfig
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence


EARLY_M110_RUNTIME_CRITICAL_PATHS = (
    "Docs/reports/M110_CP05_SOFTWARE_VALIDATION.json",
    "backend/audio_capture.py",
    "backend/decision_queue_wire_contract.py",
    "backend/handlers/connection_handlers.py",
    "backend/handlers/product_state_handlers.py",
    "backend/m110_cp05_hardware_gate.py",
    "backend/maintenance_dsp_reset.py",
    "backend/master_capture_route_proof.py",
    "backend/operator_proposal_queue.py",
    "backend/operator_proposal_result_store.py",
    "backend/osc/enhanced_osc_client.py",
    "backend/runtime_identity.py",
    "backend/server.py",
    "backend/soundcheck_transaction_plan.py",
    "backend/supervised_console_transaction.py",
    "backend/supervised_family_proof_plan.py",
    "backend/supervised_proof_capability.py",
    "backend/wing_capture_hardware_identity.py",
    "backend/wing_client.py",
    "backend/wing_supervised_console_adapters.py",
    "ops/m110/CP05_HARDWARE_GATE.json",
    "tools/wing_attended_soundcheck_runner.py",
    "tools/wing_supervised_family_proof_runner.py",
)
EARLY_ALLOWED_UNTRACKED_REPORT = re.compile(
    r"reports/[A-Za-z0-9][A-Za-z0-9._-]*\.json"
)
EARLY_IMPORT_ARTIFACT_SUFFIXES = (
    ".py",
    ".pyw",
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
)
EARLY_IMPORT_ROOTS = (".", "backend", "tools")
EARLY_IMPORT_SCAN_EXCLUDED = {
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


def _early_git(repo_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=5,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"m110_bootstrap_git_{arguments[0]}_failed")
    return completed.stdout


def _early_validate_source_before_repo_imports(
    repo_root: Path,
) -> tuple[set[str], dict[str, Any]]:
    """Use stdlib only to reject source drift and executable shadows."""

    dirty = subprocess.run(
        ["/usr/bin/git", "diff", "--quiet", "HEAD", "--"],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=5,
    )
    if dirty.returncode != 0:
        raise RuntimeError("m110_bootstrap_tracked_source_tree_dirty")
    initial_head = _early_git(repo_root, "rev-parse", "HEAD").decode().strip()
    initial_tree = _early_git(repo_root, "rev-parse", "HEAD^{tree}").decode().strip()
    tracked = {
        item.decode("utf-8")
        for item in _early_git(repo_root, "ls-files", "-z").split(b"\0")
        if item
    }
    for raw_path in _early_git(
        repo_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    ).split(b"\0"):
        if not raw_path:
            continue
        relative_path = raw_path.decode("utf-8")
        if EARLY_ALLOWED_UNTRACKED_REPORT.fullmatch(relative_path) is None:
            raise RuntimeError(
                f"m110_bootstrap_unsafe_untracked_path:{relative_path}"
            )
        report = repo_root / relative_path
        metadata = report.lstat()
        if report.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                f"m110_bootstrap_untracked_report_invalid:{relative_path}"
            )
    critical_rows: list[dict[str, Any]] = []
    for relative_path in EARLY_M110_RUNTIME_CRITICAL_PATHS:
        if relative_path not in tracked:
            raise RuntimeError(
                f"m110_bootstrap_critical_path_untracked:{relative_path}"
            )
        path = repo_root / relative_path
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                f"m110_bootstrap_critical_path_invalid:{relative_path}"
            )
        value = path.read_bytes()
        critical_rows.append(
            {
                "path": relative_path,
                "size": len(value),
                "sha256": f"sha256:{hashlib.sha256(value).hexdigest()}",
            }
        )

    for import_root_relative in EARLY_IMPORT_ROOTS:
        import_root = (
            repo_root
            if import_root_relative == "."
            else repo_root / import_root_relative
        )
        for entry in import_root.iterdir():
            if entry.name in EARLY_IMPORT_SCAN_EXCLUDED:
                continue
            candidates: list[Path] = []
            if entry.is_file() and entry.name.endswith(
                EARLY_IMPORT_ARTIFACT_SUFFIXES
            ):
                candidates.append(entry)
            elif entry.is_dir() and entry.name.isidentifier():
                for child in entry.iterdir():
                    if (
                        child.name.startswith("__init__")
                        and child.is_file()
                        and child.name.endswith(EARLY_IMPORT_ARTIFACT_SUFFIXES)
                    ):
                        candidates.append(child)
            for candidate in candidates:
                relative_path = candidate.relative_to(repo_root).as_posix()
                if relative_path not in tracked:
                    raise RuntimeError(
                        f"m110_bootstrap_ignored_import_shadow:{relative_path}"
                    )
        if import_root_relative not in {"backend", "tools"}:
            continue
        for current_root, directory_names, file_names in os.walk(import_root):
            directory_names[:] = [
                name
                for name in directory_names
                if name not in EARLY_IMPORT_SCAN_EXCLUDED
            ]
            current = Path(current_root)
            for file_name in file_names:
                if not file_name.endswith(EARLY_IMPORT_ARTIFACT_SUFFIXES):
                    continue
                candidate = current / file_name
                relative_path = candidate.relative_to(repo_root).as_posix()
                if relative_path not in tracked:
                    raise RuntimeError(
                        f"m110_bootstrap_ignored_import_shadow:{relative_path}"
                    )
    final_head = _early_git(repo_root, "rev-parse", "HEAD").decode().strip()
    final_tree = _early_git(repo_root, "rev-parse", "HEAD^{tree}").decode().strip()
    final_dirty = subprocess.run(
        ["/usr/bin/git", "diff", "--quiet", "HEAD", "--"],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=5,
    )
    if (
        final_dirty.returncode != 0
        or final_head != initial_head
        or final_tree != initial_tree
    ):
        raise RuntimeError("m110_bootstrap_source_changed_during_validation")
    return tracked, {
        "head": initial_head,
        "tree": initial_tree,
        "files": critical_rows,
    }


class _CommittedRepoImportGuard(importlib.abc.MetaPathFinder):
    """Reject any later repository import not backed by a tracked file."""

    def __init__(self, repo_root: Path, tracked_paths: set[str]):
        self._repo_root = repo_root
        self._tracked_paths = frozenset(tracked_paths)

    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        origin = getattr(spec, "origin", None) if spec is not None else None
        if not isinstance(origin, str) or origin in {"built-in", "frozen"}:
            return spec
        raw_origin = Path(os.path.abspath(origin))
        if raw_origin != self._repo_root and self._repo_root not in raw_origin.parents:
            return spec
        try:
            resolved_origin = raw_origin.resolve(strict=True)
        except OSError as exc:
            raise ImportError(f"m110_import_origin_unreadable:{fullname}") from exc
        if (
            resolved_origin != self._repo_root
            and self._repo_root not in resolved_origin.parents
        ):
            raise ImportError(f"m110_import_origin_escaped_checkout:{fullname}")
        relative_path = resolved_origin.relative_to(self._repo_root).as_posix()
        if relative_path not in self._tracked_paths:
            raise ImportError(
                f"m110_import_origin_not_committed:{fullname}:{relative_path}"
            )
        if resolved_origin.is_symlink() or not resolved_origin.is_file():
            raise ImportError(f"m110_import_origin_invalid:{fullname}")
        return spec


def _install_early_import_guard(repo_root: Path, tracked_paths: set[str]) -> None:
    sys.meta_path.insert(0, _CommittedRepoImportGuard(repo_root, tracked_paths))


def _add_external_dependency_paths_without_sitecustomize() -> None:
    """Expose interpreter-managed packages without executing site hooks."""

    paths = sysconfig.get_paths()
    candidates: list[str] = []
    for key in ("purelib", "platlib"):
        candidate = paths.get(key)
        if isinstance(candidate, str) and candidate:
            candidates.append(candidate)
    user_site = site.getusersitepackages()
    if isinstance(user_site, str) and user_site:
        candidates.append(user_site)
    elif isinstance(user_site, (list, tuple)):
        candidates.extend(str(value) for value in user_site if value)
    for candidate in candidates:
        if candidate not in sys.path:
            sys.path.append(candidate)


def _install_private_empty_pycache_prefix() -> Path:
    """Prevent repository-local bytecode from participating in code loading."""

    root = Path(tempfile.mkdtemp(prefix="automixer-m110-pycache-"))
    os.chmod(root, 0o700)
    sys.pycache_prefix = str(root)
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root


REPO_ROOT = Path(__file__).resolve().parents[1]
_EARLY_SOURCE_BINDING: dict[str, Any] | None = None
if __name__ == "__main__":
    _early_tracked_paths, _EARLY_SOURCE_BINDING = (
        _early_validate_source_before_repo_imports(REPO_ROOT)
    )
    _install_early_import_guard(REPO_ROOT, _early_tracked_paths)
    _install_private_empty_pycache_prefix()
    _add_external_dependency_paths_without_sitecustomize()
    if sys.argv[1:] == ["--_m110-dependency-smoke"]:
        for module_name in (
            "numpy",
            "requests",
            "sounddevice",
            "websockets",
            "yaml",
        ):
            __import__(module_name)
        raise SystemExit(0)
    if sys.argv[1:] == ["--_m110-backend-bootstrap"]:
        backend_script = REPO_ROOT / "backend" / "server.py"
        sys.path.insert(0, str(REPO_ROOT))
        sys.path.insert(0, str(REPO_ROOT / "backend"))
        sys.argv = [str(backend_script)]
        runpy.run_path(str(backend_script), run_name="__main__")
        raise SystemExit(0)

import websockets


BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from master_capture_route_proof import (  # noqa: E402
    MASTER_CAPTURE_ROUTE_ADDRESSES,
    MASTER_CAPTURE_ROUTE_EXPECTED,
    MASTER_CAPTURE_ROUTE_REQUEST_CONFIRMATION,
    validate_master_capture_route_request,
    verify_master_capture_route_artifact,
)
from m110_cp05_hardware_gate import (  # noqa: E402
    verify_m110_cp05_hardware_gate,
)
from supervised_proof_capability import (  # noqa: E402
    M110_ALLOWED_UNTRACKED_REPORT,
    M110_IGNORED_IMPORT_ARTIFACT_SUFFIXES,
    M110_IMPORT_SCAN_EXCLUDED_DIRECTORIES,
    M110_PRESTART_MANIFEST_SCHEMA,
    M110_RUNTIME_CRITICAL_PATHS,
    M110_RUNTIME_IMPORT_ROOTS,
    sign_m110_request,
)
from runtime_identity import (  # noqa: E402
    RUNTIME_IDENTITY_SCHEMA,
    runtime_identity_manifest_sha256,
)

if tuple(M110_RUNTIME_CRITICAL_PATHS) != EARLY_M110_RUNTIME_CRITICAL_PATHS:
    raise RuntimeError("m110_bootstrap_critical_path_contract_drift")


ATTENDED_EXECUTION_CONFIRMATION = "EXECUTE_M110_ATTENDED_WING_SOUNDCHECK"
M110_RUNNER_CLIENT_ID = "m110-attended-runner"
DEFAULT_ARTIFACT_ROOT = Path.home() / ".automixer" / "runtime" / "m110_attended"
DEFAULT_ROUTE_ARTIFACT_ROOT = (
    Path.home() / ".automixer" / "runtime" / "m110_master_route_proofs"
)


def prepare_private_artifact_root(
    artifact_root: Path,
    *,
    repo_root: Path,
) -> Path:
    """Create only missing private directories; never chmod an existing path."""

    root = repo_root.expanduser().resolve()
    candidate = Path(
        os.path.abspath(os.path.expanduser(str(artifact_root)))
    )
    forbidden_exact = {
        Path(candidate.anchor),
        Path.home().absolute(),
        Path(tempfile.gettempdir()).absolute(),
        root,
    }
    if candidate in forbidden_exact or root in candidate.parents:
        raise RuntimeError("m110_runner_artifact_root_scope_invalid")

    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                metadata = current.lstat()
            else:
                metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError("m110_runner_artifact_root_symlink_forbidden")
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("m110_runner_artifact_root_component_invalid")

    metadata = candidate.lstat()
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise RuntimeError("m110_runner_artifact_root_mode_must_be_0700")
    getuid = getattr(os, "getuid", None)
    if callable(getuid) and metadata.st_uid != getuid():
        raise RuntimeError("m110_runner_artifact_root_owner_mismatch")
    return candidate


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def assert_m110_cp05_hardware_gate(repo_root: Path) -> dict[str, Any]:
    """Require committed CP05 evidence before any real-WING route command."""
    return verify_m110_cp05_hardware_gate(
        repo_root=repo_root,
        critical_paths=M110_RUNTIME_CRITICAL_PATHS,
    )


def _git(repo_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=5,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git_{arguments[0]}_failed")
    return completed.stdout


def build_runner_prestart_manifest(
    repo_root: Path,
    *,
    launch_nonce: str,
) -> dict[str, Any]:
    dirty = subprocess.run(
        ["/usr/bin/git", "diff", "--quiet", "HEAD", "--"],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=5,
    )
    if dirty.returncode != 0:
        raise RuntimeError("m110_tracked_source_tree_dirty")
    untracked = _git(
        repo_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    for raw_path in untracked.split(b"\0"):
        if not raw_path:
            continue
        relative_path = raw_path.decode("utf-8")
        if M110_ALLOWED_UNTRACKED_REPORT.fullmatch(relative_path) is None:
            raise RuntimeError(f"m110_unsafe_untracked_path:{relative_path}")
        report_path = repo_root / relative_path
        metadata = report_path.lstat()
        if report_path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                f"m110_untracked_report_invalid:{relative_path}"
            )
    tracked = {
        raw_path.decode("utf-8")
        for raw_path in _git(repo_root, "ls-files", "-z").split(b"\0")
        if raw_path
    }
    for import_root_relative in M110_RUNTIME_IMPORT_ROOTS:
        import_root = (
            repo_root
            if import_root_relative == "."
            else repo_root / import_root_relative
        )
        for entry in import_root.iterdir():
            if entry.name in M110_IMPORT_SCAN_EXCLUDED_DIRECTORIES:
                continue
            candidates: list[Path] = []
            if entry.is_file() and entry.name.endswith(
                M110_IGNORED_IMPORT_ARTIFACT_SUFFIXES
            ):
                candidates.append(entry)
            elif entry.is_dir() and entry.name.isidentifier():
                for child in entry.iterdir():
                    if (
                        child.name.startswith("__init__")
                        and child.is_file()
                        and child.name.endswith(
                            M110_IGNORED_IMPORT_ARTIFACT_SUFFIXES
                        )
                    ):
                        candidates.append(child)
            for candidate in candidates:
                relative_path = candidate.relative_to(repo_root).as_posix()
                if relative_path not in tracked:
                    raise RuntimeError(
                        f"m110_ignored_import_shadow:{relative_path}"
                    )
        if import_root_relative not in {"backend", "tools"}:
            continue
        for current_root, directory_names, file_names in os.walk(import_root):
            directory_names[:] = [
                name
                for name in directory_names
                if name not in M110_IMPORT_SCAN_EXCLUDED_DIRECTORIES
            ]
            current = Path(current_root)
            for file_name in file_names:
                if not file_name.endswith(M110_IGNORED_IMPORT_ARTIFACT_SUFFIXES):
                    continue
                candidate = current / file_name
                relative_path = candidate.relative_to(repo_root).as_posix()
                if relative_path not in tracked:
                    raise RuntimeError(
                        f"m110_ignored_import_shadow:{relative_path}"
                    )
    files = []
    for relative_path in sorted(M110_RUNTIME_CRITICAL_PATHS):
        _git(repo_root, "ls-files", "--error-unmatch", "--", relative_path)
        path = repo_root / relative_path
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"m110_critical_path_invalid:{relative_path}")
        value = path.read_bytes()
        files.append(
            {
                "path": relative_path,
                "size": len(value),
                "sha256": f"sha256:{hashlib.sha256(value).hexdigest()}",
            }
        )
    return {
        "schema": M110_PRESTART_MANIFEST_SCHEMA,
        "mode": "m110_attended_runner",
        "head": _git(repo_root, "rev-parse", "HEAD").decode().strip(),
        "tree": _git(repo_root, "rev-parse", "HEAD^{tree}").decode().strip(),
        "launch_nonce": launch_nonce,
        "files": files,
    }


def assert_manifest_matches_early_source_binding(
    manifest: Mapping[str, Any],
    early_binding: Mapping[str, Any],
) -> None:
    candidate = {
        "head": manifest.get("head"),
        "tree": manifest.get("tree"),
        "files": manifest.get("files"),
    }
    expected = {
        "head": early_binding.get("head"),
        "tree": early_binding.get("tree"),
        "files": early_binding.get("files"),
    }
    if _canonical_json_bytes(candidate) != _canonical_json_bytes(expected):
        raise RuntimeError("m110_runner_source_changed_after_early_bootstrap")


def build_authority_pipe_packet(
    manifest: Mapping[str, Any],
    *,
    secret: bytes,
) -> bytes:
    if len(secret) != 32:
        raise ValueError("m110_authority_secret_must_be_32_bytes")
    raw_manifest = _canonical_json_bytes(manifest)
    if not 0 < len(raw_manifest) <= 64 * 1024:
        raise ValueError("m110_prestart_manifest_size_invalid")
    return len(raw_manifest).to_bytes(4, "big") + raw_manifest + secret


def build_route_capture_request(
    *,
    request_id: str,
    approval_id: str,
    request_nonce: str,
    secret: bytes,
) -> dict[str, Any]:
    return sign_m110_request(
        {
            "type": "capture_wing_master_route_proof",
            "contract_version": 1,
            "request_id": request_id,
            "approved": True,
            "approval_id": approval_id,
            "confirmation": MASTER_CAPTURE_ROUTE_REQUEST_CONFIRMATION,
            "request_nonce": request_nonce,
        },
        secret=secret,
    )


def _write_all(fd: int, value: bytes) -> None:
    view = memoryview(value)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise RuntimeError("m110_runner_pipe_write_made_no_progress")
        view = view[written:]


@dataclass
class ProofOwnedBackendProcess:
    process: subprocess.Popen
    repo_root: Path
    manifest: dict[str, Any]
    secret: bytearray = field(repr=False)
    token: bytearray = field(repr=False)
    ws_url: str
    log_handle: BinaryIO

    def close(self) -> None:
        """Terminate the exact child and erase parent-held authority material."""
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5.0)
        finally:
            self.log_handle.close()
            for value in (self.secret, self.token):
                for index in range(len(value)):
                    value[index] = 0

    def __enter__(self) -> "ProofOwnedBackendProcess":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def spawn_proof_owned_backend(
    *,
    repo_root: Path,
    ws_port: int,
    artifact_root: Path,
    backend_command: Sequence[str] | None = None,
) -> ProofOwnedBackendProcess:
    """Spawn one backend with two distinct inherited one-shot pipes."""

    root = repo_root.expanduser().resolve()
    if backend_command is None and root != REPO_ROOT.resolve():
        raise RuntimeError("m110_runner_repo_root_must_match_runner_checkout")
    if backend_command is None and (
        not sys.flags.isolated or not sys.flags.no_site
    ):
        raise RuntimeError("m110_runner_isolated_no_site_process_required")
    if not 1024 <= int(ws_port) <= 65535:
        raise ValueError("m110_runner_ws_port_invalid")
    artifact_root = prepare_private_artifact_root(
        artifact_root,
        repo_root=root,
    )
    manifest = build_runner_prestart_manifest(
        root,
        launch_nonce=secrets.token_hex(16),
    )
    if backend_command is None:
        if _EARLY_SOURCE_BINDING is None:
            raise RuntimeError("m110_runner_early_source_binding_required")
        assert_manifest_matches_early_source_binding(
            manifest,
            _EARLY_SOURCE_BINDING,
        )
    secret = bytearray(secrets.token_bytes(32))
    token = bytearray(secrets.token_urlsafe(32).encode("ascii"))
    if len(token) != 43:
        raise RuntimeError("m110_runner_websocket_token_size_invalid")
    authority_packet = build_authority_pipe_packet(
        manifest,
        secret=bytes(secret),
    )
    authority_read, authority_write = os.pipe()
    token_read, token_write = os.pipe()
    process: subprocess.Popen | None = None
    log_handle: BinaryIO | None = None
    try:
        if authority_read == token_read:
            raise RuntimeError("m110_runner_inherited_pipe_identity_collision")
        command = list(
            backend_command
            or (
                sys.executable,
                "-I",
                "-S",
                str(root / "tools" / "wing_attended_soundcheck_runner.py"),
                "--_m110-backend-bootstrap",
            )
        )
        log_path = artifact_root / (
            f"proof-owned-backend-{manifest['launch_nonce']}.log"
        )
        log_handle = log_path.open("xb")
        child_environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", str(Path.home())),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "PYTHONPATH": f"{root / 'backend'}:{root}",
            "AUTOMIXER_M110_AUTHORITY_FD": str(authority_read),
            "AUTOMIXER_WS_AUTH_TOKEN_FD": str(token_read),
            "AUTOMIXER_WS_CLIENT_ID": M110_RUNNER_CLIENT_ID,
            "AUTOMIXER_WS_HOST": "127.0.0.1",
            "AUTOMIXER_WS_PORT": str(int(ws_port)),
            "AUTOMIXER_OPERATOR_MODE": "analyze",
            "AUTOMIXER_WING_OBSERVATION_MODE": "true",
            "AUTOMIXER_WING_RECORD_TELEMETRY": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
        }
        process = subprocess.Popen(
            command,
            cwd=root,
            env=child_environment,
            pass_fds=(authority_read, token_read),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        os.close(authority_read)
        authority_read = -1
        os.close(token_read)
        token_read = -1
        _write_all(token_write, bytes(token))
        os.close(token_write)
        token_write = -1
        # AutoMixerServer consumes the bearer pipe before the M110 authority
        # pipe. Feed them in that order so a future near-limit manifest cannot
        # fill its pipe while the child is still waiting for the token EOF.
        _write_all(authority_write, authority_packet)
        os.close(authority_write)
        authority_write = -1
        return ProofOwnedBackendProcess(
            process=process,
            repo_root=root,
            manifest=manifest,
            secret=secret,
            token=token,
            ws_url=f"ws://127.0.0.1:{int(ws_port)}",
            log_handle=log_handle,
        )
    except BaseException:
        for fd in (authority_read, authority_write, token_read, token_write):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        if log_handle is not None:
            log_handle.close()
        for value in (secret, token):
            for index in range(len(value)):
                value[index] = 0
        raise


def validate_challenged_runtime_identity(
    *,
    identity: Mapping[str, Any],
    challenge_nonce: str,
    manifest: Mapping[str, Any],
    secret: bytes,
    child_pid: int,
    ws_port: int,
) -> dict[str, Any]:
    """Validate that the responding backend is the exact proof-owned child."""

    candidate = json.loads(json.dumps(identity, allow_nan=False))
    returned_challenge = candidate.pop("request_challenge_nonce", None)
    launch = candidate.get("launch_binding") or {}
    source = candidate.get("source") or {}
    expected_manifest_sha = hashlib.sha256(
        _canonical_json_bytes(manifest)
    ).hexdigest()
    checks = {
        "challenge_matches": returned_challenge == challenge_nonce,
        "runtime_schema_present": candidate.get("schema")
        == RUNTIME_IDENTITY_SCHEMA,
        "source_runtime": candidate.get("runtime_kind") == "source",
        "child_pid_matches": candidate.get("process", {}).get("pid") == child_pid,
        "websocket_port_matches": candidate.get("websocket", {}).get("port")
        == ws_port,
        "loopback_only": candidate.get("websocket", {}).get("loopback_only")
        is True,
        "launch_mode_matches": launch.get("mode") == "m110_attended_runner",
        "launch_nonce_matches": launch.get("launch_nonce")
        == manifest.get("launch_nonce"),
        "head_matches": launch.get("head") == manifest.get("head")
        and source.get("head") == manifest.get("head"),
        "tree_matches": launch.get("tree") == manifest.get("tree")
        and source.get("tree") == manifest.get("tree"),
        "manifest_sha_matches": launch.get("prestart_manifest_sha256")
        == expected_manifest_sha,
        "authority_secret_sha_matches": launch.get("authority_secret_sha256")
        == hashlib.sha256(secret).hexdigest(),
        "tracked_tree_clean": launch.get("tracked_tree_clean") is True,
        "bound_before_start": launch.get("bound_before_process_start") is True,
        "runtime_manifest_self_consistent": candidate.get("manifest_sha256")
        == runtime_identity_manifest_sha256(candidate),
        "no_live_authority": candidate.get("live_authority_granted") is False,
        "read_only_identity": candidate.get("read_only_identity") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "verified": not failed,
        "reason": (
            "proof_owned_backend_runtime_identity_verified"
            if not failed
            else "proof_owned_backend_runtime_identity_failed"
        ),
        "checks": checks,
        "failed_checks": failed,
        "runtime_instance_nonce": candidate.get("instance_nonce"),
        "runtime_manifest_sha256": candidate.get("manifest_sha256"),
    }


def _strict_websocket_object(raw: Any) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate_json_key")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non_finite_json_constant:{value}")

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError("websocket_text_frame_required")
    value = json.loads(
        raw,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("websocket_json_object_required")
    return value


M110_ROUTE_TERMINAL_TYPES = frozenset(
    {
        "authentication_status",
        "runtime_identity",
        "connection_status",
        "audio_capture_status",
        "wing_master_route_proof_result",
        "control_authentication_required",
    }
)
M110_RFC1918_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)


def validate_m110_wing_ip(value: str) -> str:
    try:
        parsed = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError("m110_wing_ip_invalid") from exc
    octets = str(parsed).split(".") if parsed.version == 4 else []
    if (
        parsed.version != 4
        or not any(parsed in network for network in M110_RFC1918_NETWORKS)
        or parsed.is_unspecified
        or parsed.is_loopback
        or parsed.is_multicast
        or not octets
        or octets[-1] in {"0", "255"}
    ):
        raise ValueError("m110_wing_ip_not_rfc1918_unicast_host")
    return str(parsed)


@dataclass
class ProofOwnedBackendSession:
    """One backend + one authenticated WebSocket kept alive across M110 stages."""

    backend: ProofOwnedBackendProcess
    websocket: Any = field(repr=False)
    identity_result: dict[str, Any]
    wing_connected: bool = False
    capture_started: bool = False
    route_established: bool = False
    route_result: dict[str, Any] | None = None
    closed: bool = False
    teardown_result: dict[str, Any] | None = None
    cp05_gate_result: dict[str, Any] | None = None
    _close_task: asyncio.Task | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @classmethod
    async def open(
        cls,
        backend: ProofOwnedBackendProcess,
        *,
        timeout_seconds: float = 15.0,
    ) -> "ProofOwnedBackendSession":
        deadline = time.monotonic() + timeout_seconds
        last_error = "backend_not_ready"
        websocket = None
        while time.monotonic() < deadline:
            if backend.process.poll() is not None:
                raise RuntimeError(
                    f"proof_owned_backend_exited:{backend.process.returncode}"
                )
            try:
                websocket = await websockets.connect(
                    backend.ws_url,
                    open_timeout=1.0,
                    close_timeout=1.0,
                    max_size=1024 * 1024,
                )
                break
            except (OSError, asyncio.TimeoutError, websockets.WebSocketException) as exc:
                last_error = type(exc).__name__
                await asyncio.sleep(0.1)
        if websocket is None:
            raise RuntimeError(f"proof_owned_backend_start_timeout:{last_error}")

        session = cls(backend=backend, websocket=websocket, identity_result={})
        try:
            await session._send_json(
                {
                    "type": "authenticate",
                    "token": bytes(backend.token).decode("ascii"),
                    "client_id": M110_RUNNER_CLIENT_ID,
                }
            )
            auth = await session._receive_terminal(
                "authentication_status",
                timeout_seconds=2.0,
            )
            if auth != {"type": "authentication_status", "authenticated": True}:
                raise RuntimeError("proof_owned_backend_authentication_failed")
            challenge = secrets.token_hex(16)
            await session._send_json(
                {
                    "type": "get_runtime_identity",
                    "challenge_nonce": challenge,
                }
            )
            identity = await session._receive_terminal(
                "runtime_identity",
                timeout_seconds=2.0,
            )
            result = validate_challenged_runtime_identity(
                identity=identity,
                challenge_nonce=challenge,
                manifest=backend.manifest,
                secret=bytes(backend.secret),
                child_pid=backend.process.pid,
                ws_port=int(backend.ws_url.rsplit(":", 1)[1]),
            )
            if result["verified"] is not True:
                raise RuntimeError(
                    "proof_owned_backend_identity_failed:"
                    + ",".join(result["failed_checks"])
                )
            session.identity_result = result
            return session
        except BaseException:
            try:
                await websocket.close(
                    code=1000,
                    reason="m110_session_open_failed",
                )
            finally:
                session.closed = True
            raise

    async def _send_json(self, payload: Mapping[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("proof_owned_backend_session_closed")
        await self.websocket.send(_canonical_json_bytes(payload).decode("utf-8"))

    async def _recv_object(self, timeout_seconds: float) -> dict[str, Any]:
        raw = await asyncio.wait_for(
            self.websocket.recv(),
            timeout=timeout_seconds,
        )
        return _strict_websocket_object(raw)

    @staticmethod
    def _reject_unexpected_terminal(
        payload: Mapping[str, Any],
        *,
        expected_type: str,
    ) -> None:
        message_type = payload.get("type")
        if message_type in M110_ROUTE_TERMINAL_TYPES:
            raise RuntimeError(
                f"unexpected_route_terminal:{message_type}:expected:{expected_type}"
            )
        if payload.get("status") == "blocked" or payload.get("success") is False:
            raise RuntimeError(
                f"interleaved_backend_failure:{message_type or 'missing_type'}"
            )

    async def _reject_immediate_duplicate_terminal(
        self,
        expected_type: str,
        *,
        grace_seconds: float = 0.025,
    ) -> None:
        deadline = time.monotonic() + grace_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                payload = await self._recv_object(remaining)
            except asyncio.TimeoutError:
                return
            if payload.get("type") == expected_type:
                raise RuntimeError(
                    f"duplicate_route_terminal:{expected_type}"
                )
            self._reject_unexpected_terminal(
                payload,
                expected_type=expected_type,
            )

    async def _receive_terminal(
        self,
        expected_type: str,
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            payload = await self._recv_object(remaining)
            if payload.get("type") == expected_type:
                await self._reject_immediate_duplicate_terminal(expected_type)
                return payload
            self._reject_unexpected_terminal(
                payload,
                expected_type=expected_type,
            )

    @staticmethod
    def _bounded_terminal_failure_context(payload: Mapping[str, Any]) -> str:
        values: list[str] = []
        reason = payload.get("reason")
        if isinstance(reason, str) and re.fullmatch(
            r"[a-z0-9_]{1,128}",
            reason,
        ):
            values.append(f"reason={reason}")
        blockers = payload.get("blockers")
        if isinstance(blockers, list):
            safe_blockers = sorted(
                {
                    value
                    for value in blockers[:32]
                    if isinstance(value, str)
                    and re.fullmatch(r"[a-z0-9_]{1,128}", value)
                }
            )
            if safe_blockers:
                values.append("blockers=" + ",".join(safe_blockers))
        error = payload.get("error")
        if isinstance(error, str) and re.fullmatch(
            r"[A-Za-z0-9_:-]{1,128}",
            error,
        ):
            values.append(f"error={error}")
        return ";".join(values) or "terminal_contract_mismatch"

    @staticmethod
    def _validate_route_result(
        *,
        result: Mapping[str, Any],
        request: Mapping[str, Any],
        audio_device_index: int,
        route_artifact_root: Path,
        identity_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        proof_id = result.get("proof_id")
        artifact = result.get("artifact")
        peak = result.get("conservative_true_peak_dbtp")
        scalar_checks = {
            "type": result.get("type") == "wing_master_route_proof_result",
            "issued": result.get("status") == "issued",
            "success": result.get("success") is True,
            "accepted": result.get("accepted") is True,
            "reason": result.get("reason")
            == "process_private_master_capture_route_authority_issued",
            "proof_id": isinstance(proof_id, str)
            and re.fullmatch(r"[0-9a-f]{32}", proof_id) is not None,
            "artifact": isinstance(artifact, Mapping),
            "device_index": isinstance(
                result.get("capture_device_index"),
                int,
            )
            and not isinstance(result.get("capture_device_index"), bool)
            and result.get("capture_device_index") == audio_device_index,
            "capture_device": isinstance(result.get("capture_device"), str)
            and bool(result.get("capture_device")),
            "console_serial": isinstance(result.get("console_serial"), str)
            and re.fullmatch(
                r"[A-Z0-9]{8,32}",
                result.get("console_serial", ""),
            )
            is not None,
            "firmware": isinstance(result.get("firmware_version"), str)
            and re.fullmatch(
                r"3\.0\.5(?:[-+ ][A-Za-z0-9._+-]+)?",
                result.get("firmware_version", ""),
            )
            is not None,
            "hardware_uid": isinstance(result.get("capture_hardware_uid"), str)
            and result.get("capture_hardware_uid", "").startswith("usb:"),
            "usb_serial": isinstance(result.get("capture_usb_serial"), str)
            and bool(result.get("capture_usb_serial")),
            "master_channels": result.get("master_channels") == [47, 48],
            "peak": isinstance(peak, (int, float))
            and not isinstance(peak, bool)
            and math.isfinite(float(peak))
            and float(peak) <= -1.0,
            "zero_console_writes": isinstance(
                result.get("console_mutation_write_attempt_count"),
                int,
            )
            and not isinstance(
                result.get("console_mutation_write_attempt_count"),
                bool,
            )
            and result.get("console_mutation_write_attempt_count") == 0,
            "no_live_authority": result.get("live_authority_granted") is False,
            "no_readiness_closure": result.get("readiness_closure_allowed")
            is False,
        }
        failed = [name for name, passed in scalar_checks.items() if not passed]
        if failed:
            raise RuntimeError(
                "wing_master_route_result_contract_failed:" + ",".join(failed)
            )
        assert isinstance(artifact, Mapping)
        verification = verify_master_capture_route_artifact(
            root=route_artifact_root,
            path=str(artifact.get("path") or ""),
            expected_sha256=str(artifact.get("sha256") or ""),
            expected_proof_id=str(proof_id),
            expected_evidence_sha256=str(
                artifact.get("evidence_sha256") or ""
            ),
        )
        if verification.get("verified") is not True:
            raise RuntimeError(
                "wing_master_route_artifact_verification_failed:"
                + str(verification.get("reason") or "unknown")
            )
        evidence = verification.get("evidence")
        evidence_request = (
            evidence.get("request") if isinstance(evidence, Mapping) else None
        )
        runtime = (
            evidence.get("runtime") if isinstance(evidence, Mapping) else None
        )
        console = (
            evidence.get("console") if isinstance(evidence, Mapping) else None
        )
        capture = (
            evidence.get("capture") if isinstance(evidence, Mapping) else None
        )
        route = evidence.get("route") if isinstance(evidence, Mapping) else None
        safety = (
            evidence.get("safety") if isinstance(evidence, Mapping) else None
        )
        hardware = (
            capture.get("hardware_identity")
            if isinstance(capture, Mapping)
            else None
        )

        def finite_number(value: Any) -> bool:
            return bool(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            )

        def exact_integer(value: Any, expected: int | None = None) -> bool:
            valid = bool(isinstance(value, int) and not isinstance(value, bool))
            return valid and (expected is None or value == expected)

        def route_value_equal(left: Any, right: Any) -> bool:
            if isinstance(left, str) or isinstance(right, str):
                return isinstance(left, str) and isinstance(right, str) and (
                    left.strip().upper() == right.strip().upper()
                )
            return bool(
                finite_number(left)
                and finite_number(right)
                and float(left) == float(right)
            )

        def route_pass_verified(candidate: Any) -> bool:
            if not isinstance(candidate, Mapping):
                return False
            values = candidate.get("values")
            readbacks = candidate.get("readbacks")
            if not isinstance(values, Mapping) or not isinstance(
                readbacks,
                Mapping,
            ):
                return False
            if set(values) != set(MASTER_CAPTURE_ROUTE_ADDRESSES) or set(
                readbacks
            ) != set(MASTER_CAPTURE_ROUTE_ADDRESSES):
                return False
            if not (
                candidate.get("verified") is True
                and candidate.get("reason")
                == "fixed_master_route_pass_verified"
                and candidate.get("addresses")
                == list(MASTER_CAPTURE_ROUTE_ADDRESSES)
                and candidate.get("errors") == []
                and candidate.get("confirmed_round_trips_required") == 2
            ):
                return False
            for address in MASTER_CAPTURE_ROUTE_ADDRESSES:
                row = readbacks.get(address)
                if not isinstance(row, Mapping) or not (
                    row.get("status") == "received"
                    and row.get("address") == address
                    and exact_integer(row.get("confirmed_round_trips"))
                    and row.get("confirmed_round_trips") >= 2
                    and route_value_equal(
                        row.get("value"),
                        values.get(address),
                    )
                ):
                    return False
            return bool(
                values.get("/$syscfg/$serial")
                == result.get("console_serial")
                and values.get("/$syscfg/$firmware")
                == result.get("firmware_version")
                and all(
                    route_value_equal(values.get(address), expected)
                    for address, expected in MASTER_CAPTURE_ROUTE_EXPECTED.items()
                )
            )

        pass_1 = route.get("pass_1") if isinstance(route, Mapping) else None
        pass_2 = route.get("pass_2") if isinstance(route, Mapping) else None
        route_passes_equal = bool(
            isinstance(pass_1, Mapping)
            and isinstance(pass_2, Mapping)
            and isinstance(pass_1.get("values"), Mapping)
            and isinstance(pass_2.get("values"), Mapping)
            and all(
                route_value_equal(
                    pass_1["values"].get(address),
                    pass_2["values"].get(address),
                )
                for address in MASTER_CAPTURE_ROUTE_ADDRESSES
            )
        )

        sample_counts = (
            capture.get("sample_counts")
            if isinstance(capture, Mapping)
            else None
        )
        sample_peaks = (
            capture.get("sample_peaks_dbfs")
            if isinstance(capture, Mapping)
            else None
        )
        start_count = (
            capture.get("captured_sample_count_before")
            if isinstance(capture, Mapping)
            else None
        )
        required_count = (
            capture.get("required_fresh_sample_count")
            if isinstance(capture, Mapping)
            else None
        )
        snapshot_count = (
            capture.get("captured_sample_count_snapshot")
            if isinstance(capture, Mapping)
            else None
        )
        after_count = (
            capture.get("captured_sample_count_after")
            if isinstance(capture, Mapping)
            else None
        )
        peaks_valid = bool(
            isinstance(sample_peaks, Mapping)
            and set(sample_peaks) == {"47", "48"}
            and all(
                finite_number(sample_peaks.get(channel))
                and float(sample_peaks[channel]) > -90.0
                for channel in ("47", "48")
            )
        )
        calculated_peak = (
            max(float(sample_peaks[channel]) for channel in ("47", "48"))
            + 3.0
            if peaks_valid
            else None
        )
        capture_peak = (
            capture.get("conservative_true_peak_dbtp")
            if isinstance(capture, Mapping)
            else None
        )
        capture_semantics_valid = bool(
            isinstance(capture, Mapping)
            and capture.get("verified") is True
            and capture.get("reason") == "fresh_real_master_capture_verified"
            and capture.get("source_type") == "sounddevice"
            and exact_integer(capture.get("capture_instance_id"))
            and capture.get("capture_instance_id") > 0
            and exact_integer(capture.get("capture_stream_instance_id"))
            and capture.get("capture_stream_instance_id") > 0
            and exact_integer(start_count)
            and start_count >= 0
            and exact_integer(required_count, 12_000)
            and exact_integer(snapshot_count)
            and snapshot_count >= start_count + required_count
            and exact_integer(after_count)
            and after_count >= snapshot_count
            and isinstance(sample_counts, Mapping)
            and sample_counts == {"47": 12_000, "48": 12_000}
            and peaks_valid
            and capture.get("true_peak_allowance_db") == 3.0
            and capture.get("headroom_ceiling_dbtp") == -1.0
            and finite_number(capture_peak)
            and calculated_peak is not None
            and math.isclose(
                float(capture_peak),
                calculated_peak,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and float(capture_peak) <= -1.0
            and finite_number(result.get("conservative_true_peak_dbtp"))
            and math.isclose(
                float(capture_peak),
                float(result.get("conservative_true_peak_dbtp")),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )

        usb = hardware.get("usb") if isinstance(hardware, Mapping) else None
        coreaudio = (
            hardware.get("coreaudio")
            if isinstance(hardware, Mapping)
            else None
        )
        hardware_binding = (
            {
                key: hardware.get(key)
                for key in (
                    "schema",
                    "device_name",
                    "device_index",
                    "console_serial",
                    "coreaudio",
                    "usb",
                    "serial_binding",
                )
            }
            if isinstance(hardware, Mapping)
            else {}
        )
        expected_hardware_sha = hashlib.sha256(
            _canonical_json_bytes(hardware_binding)
        ).hexdigest()
        console_serial = str(result.get("console_serial") or "").upper()
        usb_serial = str(
            usb.get("serial") if isinstance(usb, Mapping) else ""
        ).upper()
        serial_binding = (
            "exact"
            if usb_serial == console_serial
            else "chassis_serial_plus_three_digit_usb_interface"
            if re.fullmatch(re.escape(console_serial) + r"[0-9]{3}", usb_serial)
            else None
        )
        location_id = (
            usb.get("location_id") if isinstance(usb, Mapping) else None
        )
        expected_hardware_uid = (
            f"usb:1397:050b:{usb_serial}:{location_id:08x}"
            if exact_integer(location_id) and location_id >= 0
            else None
        )
        hardware_semantics_valid = bool(
            isinstance(hardware, Mapping)
            and hardware.get("verified") is True
            and hardware.get("reason")
            == "physical_wing_usb_capture_bound_to_console"
            and hardware.get("schema")
            == "automixer.wing_capture_hardware_identity.v1"
            and hardware.get("device_name") == result.get("capture_device")
            and exact_integer(
                hardware.get("device_index"),
                audio_device_index,
            )
            and hardware.get("console_serial") == console_serial
            and coreaudio
            == {
                "manufacturer": "BEHRINGER",
                "transport": "usb",
                "input_channels": 48,
                "sample_rate_hz": 48_000,
            }
            and isinstance(usb, Mapping)
            and exact_integer(usb.get("vendor_id"), 5015)
            and exact_integer(usb.get("product_id"), 1291)
            and usb_serial
            == str(result.get("capture_usb_serial") or "").upper()
            and serial_binding is not None
            and hardware.get("serial_binding") == serial_binding
            and hardware.get("hardware_uid") == expected_hardware_uid
            and hardware.get("hardware_uid")
            == result.get("capture_hardware_uid")
            and hardware.get("binding_sha256") == expected_hardware_sha
        )
        evidence_checks = {
            "request_binding": isinstance(evidence_request, Mapping)
            and evidence_request.get("request_id") == request.get("request_id")
            and evidence_request.get("approval_id") == request.get("approval_id")
            and evidence_request.get("request_nonce") == request.get("request_nonce"),
            "runtime_binding": isinstance(runtime, Mapping)
            and runtime.get("instance_nonce")
            == identity_result.get("runtime_instance_nonce")
            and runtime.get("manifest_sha256")
            == identity_result.get("runtime_manifest_sha256")
            and runtime.get("tracked_tree_clean") is True,
            "console_binding": isinstance(console, Mapping)
            and console.get("serial") == result.get("console_serial")
            and console.get("firmware") == result.get("firmware_version"),
            "capture_binding": isinstance(capture, Mapping)
            and exact_integer(capture.get("device_index"), audio_device_index)
            and capture.get("device_name") == result.get("capture_device")
            and exact_integer(capture.get("channel_count"), 48)
            and exact_integer(capture.get("sample_rate_hz"), 48_000)
            and capture.get("master_channels") == [47, 48],
            "capture_semantics": capture_semantics_valid,
            "hardware_binding": hardware_semantics_valid,
            "fixed_route": isinstance(route, Mapping)
            and route.get("addresses") == list(MASTER_CAPTURE_ROUTE_ADDRESSES)
            and route.get("expected") == dict(MASTER_CAPTURE_ROUTE_EXPECTED)
            and route.get("consistent") is True
            and route.get("logical_query_count") == 12
            and route.get("confirmed_round_trips_required") == 2
            and route.get("complete_passes_required") == 2
            and route_pass_verified(pass_1)
            and route_pass_verified(pass_2)
            and route_passes_equal,
            "safety_binding": isinstance(safety, Mapping)
            and safety.get("authenticated_control_session") is True
            and safety.get("operator_mode") == "analyze"
            and safety.get("observation_only") is True
            and safety.get("console_writer_scope_acquired") is True
            and safety.get("wal_integrity_ok") is True
            and exact_integer(safety.get("recovery_transaction_count"), 0)
            and exact_integer(safety.get("emergency_generation"))
            and safety.get("emergency_generation") >= 0
            and safety.get("emergency_generation_stable") is True
            and exact_integer(
                safety.get("console_mutation_write_attempt_count"),
                0,
            )
            and safety.get("normal_live_allowlist_changed") is False
            and safety.get("live_authority_granted") is False
            and safety.get("readiness_closure_allowed") is False,
            "expiry_binding": finite_number(result.get("issued_at_epoch"))
            and finite_number(result.get("expires_at_epoch"))
            and float(result["issued_at_epoch"])
            == verification.get("issued_at_epoch")
            and float(result["expires_at_epoch"])
            == verification.get("expires_at_epoch")
            and float(result["issued_at_epoch"]) <= time.time()
            and 0.0
            < float(result["expires_at_epoch"])
            - float(result["issued_at_epoch"])
            <= 30.0 * 60.0
            and float(result["expires_at_epoch"]) > time.time(),
        }
        evidence_failed = [
            name for name, passed in evidence_checks.items() if not passed
        ]
        if evidence_failed:
            raise RuntimeError(
                "wing_master_route_evidence_binding_failed:"
                + ",".join(evidence_failed)
            )
        return {
            "verified": True,
            "reason": "runner_independently_verified_master_route_artifact",
            "path": verification["path"],
            "sha256": verification["sha256"],
            "proof_id": verification["proof_id"],
            "evidence_sha256": verification["evidence_sha256"],
            "request_binding_verified": True,
            "hardware_binding_verified": True,
            "live_authority_granted": False,
        }

    async def establish_master_route(
        self,
        *,
        wing_ip: str,
        audio_device_index: int,
        approval_id: str,
        request_id: str | None = None,
        request_nonce: str | None = None,
        route_artifact_root: Path = DEFAULT_ROUTE_ARTIFACT_ROOT,
    ) -> dict[str, Any]:
        """Fail closed: a route-stage error tears down this entire authority."""

        try:
            return await self._establish_master_route(
                wing_ip=wing_ip,
                audio_device_index=audio_device_index,
                approval_id=approval_id,
                request_id=request_id,
                request_nonce=request_nonce,
                route_artifact_root=route_artifact_root,
            )
        except BaseException:
            await self.close()
            raise

    async def _establish_master_route(
        self,
        *,
        wing_ip: str,
        audio_device_index: int,
        approval_id: str,
        request_id: str | None = None,
        request_nonce: str | None = None,
        route_artifact_root: Path = DEFAULT_ROUTE_ARTIFACT_ROOT,
    ) -> dict[str, Any]:
        """Establish read-only WING/capture/route proof without closing ownership."""

        self.cp05_gate_result = assert_m110_cp05_hardware_gate(
            self.backend.repo_root
        )
        validated_wing_ip = validate_m110_wing_ip(wing_ip)
        if (
            isinstance(audio_device_index, bool)
            or not isinstance(audio_device_index, int)
            or audio_device_index < 0
        ):
            raise ValueError("m110_audio_device_index_invalid")
        request = build_route_capture_request(
            request_id=request_id or f"m110-route-{secrets.token_hex(8)}",
            approval_id=approval_id,
            request_nonce=request_nonce or secrets.token_hex(16),
            secret=bytes(self.backend.secret),
        )
        request_errors = validate_master_capture_route_request(request)
        if request_errors:
            raise ValueError(
                "m110_route_request_authorization_invalid:"
                + ",".join(request_errors)
            )
        await self._send_json(
            {
                "type": "connect_wing",
                "ip": validated_wing_ip,
                "send_port": 2223,
                "receive_port": 2223,
                "observation_only": True,
                "record_telemetry": False,
            }
        )
        connection = await self._receive_terminal(
            "connection_status",
            timeout_seconds=10.0,
        )
        if not (
            connection.get("connected") is True
            and connection.get("mode") == "wing"
            and connection.get("ip") == validated_wing_ip
            and connection.get("observation_only") is True
            and connection.get("record_telemetry") is False
        ):
            raise RuntimeError(
                "m110_observation_only_wing_connect_failed:"
                + self._bounded_terminal_failure_context(connection)
            )
        self.wing_connected = True

        await self._send_json(
            {
                "type": "start_audio_capture_preview",
                "device_id": audio_device_index,
                "num_channels": 48,
                "sample_rate": 48_000,
                "buffer_seconds": 8.0,
            }
        )
        capture = await self._receive_terminal(
            "audio_capture_status",
            timeout_seconds=10.0,
        )
        if not (
            capture.get("running") is True
            and capture.get("preview") is True
            and capture.get("owned_by_requester") is True
            and capture.get("source_type") == "sounddevice"
            and capture.get("real_input") is True
            and capture.get("device_index") == audio_device_index
            and isinstance(capture.get("device_name"), str)
            and bool(capture.get("device_name"))
            and capture.get("num_channels") == 48
            and capture.get("sample_rate") == 48_000
        ):
            raise RuntimeError(
                "m110_real_48_channel_capture_start_failed:"
                + self._bounded_terminal_failure_context(capture)
            )
        self.capture_started = True

        await self._send_json(request)
        route_result = await self._receive_terminal(
            "wing_master_route_proof_result",
            timeout_seconds=20.0,
        )
        if route_result.get("status") == "blocked":
            raise RuntimeError(
                "m110_master_route_proof_blocked:"
                + self._bounded_terminal_failure_context(route_result)
            )
        artifact_verification = self._validate_route_result(
            result=route_result,
            request=request,
            audio_device_index=audio_device_index,
            route_artifact_root=route_artifact_root,
            identity_result=self.identity_result,
        )
        self.route_established = True
        self.route_result = dict(route_result)
        return {
            "verified": True,
            "reason": "persistent_proof_owned_master_route_session_established",
            "connection_status": connection,
            "audio_capture_status": capture,
            "route_result": route_result,
            "artifact_verification": artifact_verification,
            "cp05_hardware_gate": dict(self.cp05_gate_result),
            "same_websocket_retained": True,
            "backend_process_retained": self.backend.process.poll() is None,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
        }

    async def close(self) -> dict[str, Any]:
        if self.closed and self.teardown_result is not None:
            return dict(self.teardown_result)
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_impl())
        close_task = self._close_task
        try:
            return await asyncio.shield(close_task)
        except asyncio.CancelledError:
            # Drain the shielded teardown before propagating caller
            # cancellation.  A nested finally in _close_impl terminates the
            # exact child even if this task itself is cancelled again.
            try:
                await asyncio.shield(close_task)
            finally:
                if not self.closed:
                    self.closed = True
                    self.backend.close()
            raise

    async def _close_impl(self) -> dict[str, Any]:
        if self.closed:
            return dict(self.teardown_result or {})
        errors: list[str] = []
        capture_stopped = not self.capture_started
        mixer_disconnected = not self.wing_connected
        try:
            if self.capture_started:
                try:
                    await self._send_json({"type": "stop_audio_capture_preview"})
                    stopped = await self._receive_terminal(
                        "audio_capture_status",
                        timeout_seconds=3.0,
                    )
                    capture_stopped = bool(
                        stopped.get("running") is False
                        and stopped.get("preview") is True
                        and stopped.get("owned_by_requester") is True
                    )
                    if not capture_stopped:
                        errors.append("owned_audio_capture_stop_unverified")
                except Exception as exc:
                    errors.append(f"owned_audio_capture_stop_failed:{type(exc).__name__}")
            if self.wing_connected:
                try:
                    await self._send_json({"type": "disconnect_wing"})
                    disconnected = await self._receive_terminal(
                        "connection_status",
                        timeout_seconds=3.0,
                    )
                    mixer_disconnected = disconnected.get("connected") is False
                    if not mixer_disconnected:
                        errors.append("wing_disconnect_unverified")
                except Exception as exc:
                    errors.append(f"wing_disconnect_failed:{type(exc).__name__}")
        finally:
            try:
                await self.websocket.close(
                    code=1000,
                    reason="m110_attended_session_complete",
                )
            except Exception as exc:
                errors.append(f"websocket_close_failed:{type(exc).__name__}")
            finally:
                # No await is permitted between this flag and exact child
                # termination: it is the cancellation-proof safety backstop.
                self.closed = True
                self.backend.close()
        backend_terminated = self.backend.process.poll() is not None
        if not backend_terminated:
            errors.append("proof_owned_backend_not_terminated")
        self.teardown_result = {
            "verified": not errors and capture_stopped and mixer_disconnected,
            "capture_stopped": capture_stopped,
            "mixer_disconnected": mixer_disconnected,
            "websocket_closed": True,
            "backend_process_terminated": backend_terminated,
            "route_authority_retained": False,
            "errors": errors,
        }
        return dict(self.teardown_result)

    async def __aenter__(self) -> "ProofOwnedBackendSession":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()


async def open_proof_owned_backend_session(
    *,
    repo_root: Path,
    ws_port: int,
    artifact_root: Path,
) -> ProofOwnedBackendSession:
    backend = spawn_proof_owned_backend(
        repo_root=repo_root,
        ws_port=ws_port,
        artifact_root=artifact_root,
    )
    try:
        return await ProofOwnedBackendSession.open(backend)
    except BaseException:
        backend.close()
        raise


async def probe_proof_owned_backend_identity(
    backend: ProofOwnedBackendProcess,
    *,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    session = await ProofOwnedBackendSession.open(
        backend,
        timeout_seconds=timeout_seconds,
    )
    try:
        return dict(session.identity_result)
    finally:
        await session.close()


def run_proof_owned_backend_identity_probe(
    *,
    repo_root: Path,
    ws_port: int,
    artifact_root: Path,
) -> dict[str, Any]:
    with spawn_proof_owned_backend(
        repo_root=repo_root,
        ws_port=ws_port,
        artifact_root=artifact_root,
    ) as backend:
        return asyncio.run(probe_proof_owned_backend_identity(backend))


async def run_proof_owned_master_route_probe(
    *,
    repo_root: Path,
    ws_port: int,
    artifact_root: Path,
    wing_ip: str,
    audio_device_index: int,
    approval_id: str,
) -> dict[str, Any]:
    """Run HW01 only, retaining one session until verified teardown."""

    cp05_gate = assert_m110_cp05_hardware_gate(repo_root)
    session = await open_proof_owned_backend_session(
        repo_root=repo_root,
        ws_port=ws_port,
        artifact_root=artifact_root,
    )
    route: dict[str, Any] | None = None
    try:
        route = await session.establish_master_route(
            wing_ip=wing_ip,
            audio_device_index=audio_device_index,
            approval_id=approval_id,
        )
    finally:
        teardown = await session.close()
    if teardown.get("verified") is not True:
        raise RuntimeError(
            "proof_owned_master_route_teardown_failed:"
            + ",".join(teardown.get("errors") or [])
        )
    return {
        "verified": True,
        "reason": "hw01_route_probe_verified_then_torn_down",
        "identity": session.identity_result,
        "cp05_hardware_gate": cp05_gate,
        "route": route,
        "teardown": teardown,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
    }


def run_proof_owned_master_route_probe_sync(**kwargs: Any) -> dict[str, Any]:
    return asyncio.run(run_proof_owned_master_route_probe(**kwargs))


def write_private_summary(
    *,
    artifact_root: Path,
    filename: str,
    payload: Mapping[str, Any],
) -> Path:
    """Atomically replace one private regular summary without following links."""

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", filename) is None:
        raise ValueError("m110_summary_filename_invalid")
    root = artifact_root.absolute()
    root_metadata = root.lstat()
    if root.is_symlink() or not stat.S_ISDIR(root_metadata.st_mode):
        raise RuntimeError("m110_summary_root_invalid")
    if stat.S_IMODE(root_metadata.st_mode) != 0o700:
        raise RuntimeError("m110_summary_root_mode_must_be_0700")
    try:
        existing = (root / filename).lstat()
    except FileNotFoundError:
        existing = None
    if existing is not None and (
        stat.S_ISLNK(existing.st_mode)
        or not stat.S_ISREG(existing.st_mode)
    ):
        raise RuntimeError("m110_summary_existing_target_invalid")

    raw = _canonical_json_bytes(payload) + b"\n"
    directory_fd = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    temporary_name = f".{filename}.{secrets.token_hex(8)}.tmp"
    fd: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=directory_fd,
        )
        _write_all(fd, raw)
        os.fchmod(fd, 0o600)
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(
            temporary_name,
            filename,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    except BaseException:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(directory_fd)
    return root / filename


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00",
        "Z",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verify-launch-only", action="store_true")
    parser.add_argument("--verify-route-only", action="store_true")
    parser.add_argument("--ws-port", type=int, default=18765)
    parser.add_argument("--approval-id", default="")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--wing-ip", default="")
    parser.add_argument("--audio-device-index", type=int)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        args.artifact_root = prepare_private_artifact_root(
            args.artifact_root,
            repo_root=args.repo_root,
        )
    except Exception as exc:
        print(
            f"m110_runner_artifact_root_rejected:{type(exc).__name__}:{exc}",
            file=sys.stderr,
        )
        return 2
    failures: list[str] = []
    launch_probe: dict[str, Any] | None = None
    route_probe: dict[str, Any] | None = None
    contacts_backend = False
    contacts_wing = False
    selected_modes = sum(
        bool(value)
        for value in (
            args.execute,
            args.verify_launch_only,
            args.verify_route_only,
        )
    )
    if selected_modes > 1:
        failures.append("runner_modes_are_mutually_exclusive")
    if args.execute:
        if not args.approval_id:
            failures.append("operator_approval_id_required")
        if args.confirmation != ATTENDED_EXECUTION_CONFIRMATION:
            failures.append("attended_execution_confirmation_mismatch")
        failures.append("m110_remaining_capability_checkpoints_not_committed")
    if args.verify_route_only:
        if re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}",
            args.approval_id,
        ) is None:
            failures.append("operator_approval_id_required")
        if args.confirmation != MASTER_CAPTURE_ROUTE_REQUEST_CONFIRMATION:
            failures.append("route_capture_confirmation_mismatch")
        if not args.wing_ip:
            failures.append("wing_ip_required")
        else:
            try:
                args.wing_ip = validate_m110_wing_ip(args.wing_ip)
            except ValueError as exc:
                failures.append(str(exc))
        if args.audio_device_index is None:
            failures.append("audio_device_index_required")
        elif args.audio_device_index < 0:
            failures.append("audio_device_index_invalid")
        if not failures:
            try:
                assert_m110_cp05_hardware_gate(args.repo_root)
            except Exception as exc:
                failures.append(f"m110_cp05_hardware_gate_failed:{exc}")
    if args.verify_launch_only and not failures:
        contacts_backend = True
        try:
            launch_probe = run_proof_owned_backend_identity_probe(
                repo_root=args.repo_root,
                ws_port=args.ws_port,
                artifact_root=args.artifact_root,
            )
        except Exception as exc:
            failures.append(
                f"proof_owned_backend_launch_failed:{type(exc).__name__}"
            )
    if args.verify_route_only and not failures:
        contacts_backend = True
        contacts_wing = True
        try:
            route_probe = run_proof_owned_master_route_probe_sync(
                repo_root=args.repo_root,
                ws_port=args.ws_port,
                artifact_root=args.artifact_root,
                wing_ip=args.wing_ip,
                audio_device_index=args.audio_device_index,
                approval_id=args.approval_id,
            )
            launch_probe = dict(route_probe.get("identity") or {})
        except Exception as exc:
            failures.append(
                f"proof_owned_master_route_failed:{type(exc).__name__}:{exc}"
            )
    summary = {
        "schema": "automixer.m110_attended_runner_summary.v1",
        "timestamp_utc": utc_now(),
        "status": (
            "verified"
            if (launch_probe is not None or route_probe is not None) and not failures
            else "blocked"
            if args.execute or failures
            else "planned"
        ),
        "reason": (
            "proof_owned_backend_launch_verified"
            if launch_probe is not None and route_probe is None and not failures
            else "hw01_route_probe_verified_then_torn_down"
            if route_probe is not None and not failures
            else "route_execution_blocked"
            if args.verify_route_only
            else "proof_owned_backend_launch_blocked"
            if args.verify_launch_only
            else "execution_authorization_failed"
            if args.execute
            else "plan_only_execute_flag_required"
        ),
        "authorization_failures": failures,
        "required_confirmation": ATTENDED_EXECUTION_CONFIRMATION,
        "route_confirmation": MASTER_CAPTURE_ROUTE_REQUEST_CONFIRMATION,
        "contacts_backend": contacts_backend,
        "contacts_wing": contacts_wing,
        "opens_udp_or_osc": contacts_wing,
        "runner_opens_udp_or_osc": False,
        "proof_owned_backend_opens_udp_or_osc": contacts_wing,
        "live_write": False,
        "accepted": False,
        "readiness_closure_allowed": False,
        "launch_probe": launch_probe,
        "route_probe": route_probe,
    }
    try:
        write_private_summary(
            artifact_root=args.artifact_root,
            filename="plan-summary.json",
            payload=summary,
        )
    except Exception as exc:
        print(
            f"m110_runner_summary_write_failed:{type(exc).__name__}:{exc}",
            file=sys.stderr,
        )
        return 2
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
