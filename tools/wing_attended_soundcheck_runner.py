#!/usr/bin/env python3
"""Attended M110 proof-owned backend and persistent control-session runner.

The runner itself is WebSocket-only and never imports a mixer transport.  Its
isolated child backend may open OSC/UDP only after the committed CP05 hardware
gate, exact operator authorization and fixed RFC1918 unicast validation pass.
Full write execution exists only as the attended, CP05-bound, one-shot
route -> reset -> M006 restore -> staged soundcheck -> HW05 audit workflow.
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
import fcntl
import hashlib
import importlib.abc
import importlib.machinery
import ipaddress
import json
import math
import os
import pwd
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
from typing import Any, Awaitable, BinaryIO, Callable, Mapping, Sequence


EARLY_M110_RUNTIME_CRITICAL_PATHS = (
    "Docs/reports/M110_CP05_SOFTWARE_VALIDATION.json",
    "backend/audio_capture.py",
    "backend/auto_soundcheck_engine.py",
    "backend/contracts/m110_m006_restore_20260711.json",
    "backend/decision_queue_wire_contract.py",
    "backend/handlers/connection_handlers.py",
    "backend/handlers/product_state_handlers.py",
    "backend/handlers/soundcheck_handlers.py",
    "backend/m110_cp05_hardware_gate.py",
    "backend/m110_manual_write_wal.py",
    "backend/m110_m006_restore.py",
    "backend/m110_processor_enable.py",
    "backend/maintenance_dsp_reset.py",
    "backend/master_capture_route_proof.py",
    "backend/operator_proposal_queue.py",
    "backend/operator_proposal_result_store.py",
    "backend/osc/enhanced_osc_client.py",
    "backend/runtime_identity.py",
    "backend/server.py",
    "backend/soundcheck_mix_readiness.py",
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
    # The later manifest is path-sorted by contract.  Keep the independent
    # stdlib-only bootstrap binding in the same canonical order so an
    # unchanged checkout cannot reject itself before the child log exists.
    for relative_path in sorted(EARLY_M110_RUNTIME_CRITICAL_PATHS):
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
    is_m110_wing_firmware_version,
    validate_master_capture_route_request,
    verify_master_capture_route_artifact,
)
from m110_cp05_hardware_gate import (  # noqa: E402
    verify_m110_cp05_hardware_gate,
)
from m110_m006_restore import (  # noqa: E402
    M110_M006_RESTORE_STAGES,
    M110_M006_SOURCE_EVENT_SHA256,
    M110_M006_SOURCE_TRANSACTION_ID,
    m110_m006_restore_bundle_selector,
    m110_m006_restore_stage_operations,
)
from maintenance_dsp_reset import (  # noqa: E402
    M110_RESET_ADDRESSES,
    M110_RESET_BUNDLE_KIND,
    M110_RESET_OPERATION_DEFINITIONS,
)
from supervised_proof_capability import (  # noqa: E402
    M110_CAPABILITY_CONFIRMATIONS,
    M110_ALLOWED_UNTRACKED_REPORT,
    M110_IGNORED_IMPORT_ARTIFACT_SUFFIXES,
    M110_IMPORT_SCAN_EXCLUDED_DIRECTORIES,
    M110_PRESTART_MANIFEST_SCHEMA,
    M110_M006_RESTORE_BUNDLE_KIND,
    M110_RUNTIME_CRITICAL_PATHS,
    M110_RUNTIME_IMPORT_ROOTS,
    M110_SOUNDCHECK_ATOMIC_GROUP_BUNDLE_KIND,
    M110_SOUNDCHECK_BUNDLE_KINDS,
    M110_SOUNDCHECK_SINGLE_BUNDLE_KIND,
    sign_m110_request,
    validate_m110_arm_request,
    validate_m110_capability_issue_request,
)
from runtime_identity import (  # noqa: E402
    RUNTIME_IDENTITY_SCHEMA,
    runtime_identity_manifest_sha256,
)

if tuple(M110_RUNTIME_CRITICAL_PATHS) != EARLY_M110_RUNTIME_CRITICAL_PATHS:
    raise RuntimeError("m110_bootstrap_critical_path_contract_drift")


ATTENDED_EXECUTION_CONFIRMATION = "EXECUTE_M110_ATTENDED_WING_SOUNDCHECK"
M110_ATTENDED_CONTROL_CONFIRMATION = "ARM_M110_ATTENDED_ONE_SHOT_CONTROL"
M110_RESET_CONTAINMENT_CONFIRMATION = (
    "CONTAIN_ALL_PA_MONITOR_OUTPUTS_AND_MUTE_MAIN_1_FOR_M110_RESET"
)
M110_SOUNDCHECK_UNMUTE_CONFIRMATION = (
    "UNMUTE_MAIN_1_FOR_M110_SOUNDCHECK"
)
M110_RUNNER_CLIENT_ID = "m110-attended-runner"
M110_CAPTURE_READY_TIMEOUT_SECONDS = 1.0
M110_CAPTURE_READY_POLL_INTERVAL_SECONDS = 0.1
M110_TERMINAL_DUPLICATE_GRACE_SECONDS = 0.025
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


def build_m110_attended_control_request(
    *,
    approval_id: str,
    request_nonce: str,
    secret: bytes,
) -> dict[str, Any]:
    """Build the exact signed one-shot control-arm envelope."""

    request = sign_m110_request(
        {
            "type": "arm_m110_attended_control",
            "contract_version": 1,
            "approval_id": approval_id,
            "confirmation": M110_ATTENDED_CONTROL_CONFIRMATION,
            "request_nonce": request_nonce,
        },
        secret=secret,
    )
    valid, reason = validate_m110_arm_request(request)
    if not valid:
        raise ValueError(f"m110_arm_request_builder_invalid:{reason}")
    return request


def build_m110_capability_request(
    *,
    capability_kind: str,
    proposal_id: str,
    review_contract_version: int,
    review_content_digest: str,
    operation_digest: str,
    approval_id: str,
    request_nonce: str,
    secret: bytes,
) -> dict[str, Any]:
    """Build the exact signed capability-issue envelope for one bundle."""

    if capability_kind not in M110_CAPABILITY_CONFIRMATIONS:
        raise ValueError("m110_capability_kind_not_supported_by_runner")
    if review_contract_version != 2 or isinstance(
        review_contract_version,
        bool,
    ):
        raise ValueError("m110_capability_review_version_invalid")
    request = sign_m110_request(
        {
            "type": "issue_supervised_capability",
            "contract_version": 1,
            "capability_kind": capability_kind,
            "proposal_id": proposal_id,
            "expected_review_contract_version": review_contract_version,
            "expected_review_content_digest": review_content_digest,
            "expected_operation_digest": operation_digest,
            "approval_id": approval_id,
            "confirmation": M110_CAPABILITY_CONFIRMATIONS[capability_kind],
            "request_nonce": request_nonce,
        },
        secret=secret,
    )
    valid, reason = validate_m110_capability_issue_request(request)
    if not valid:
        raise ValueError(f"m110_capability_request_builder_invalid:{reason}")
    return request


def build_m110_restore_only_capability_request(
    *,
    proposal_id: str,
    review_contract_version: int,
    review_content_digest: str,
    operation_digest: str,
    approval_id: str,
    request_nonce: str,
    secret: bytes,
) -> dict[str, Any]:
    """Preserve the legacy proof builder as an exact narrow wrapper."""

    return build_m110_capability_request(
        capability_kind="restore_only_proof",
        proposal_id=proposal_id,
        review_contract_version=review_contract_version,
        review_content_digest=review_content_digest,
        operation_digest=operation_digest,
        approval_id=approval_id,
        request_nonce=request_nonce,
        secret=secret,
    )


def _write_all(fd: int, value: bytes) -> None:
    view = memoryview(value)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise RuntimeError("m110_runner_pipe_write_made_no_progress")
        view = view[written:]


def _m110_console_writer_lock_path() -> Path:
    """Return the same stable per-user writer lock used by AutoMixerServer."""

    return (
        Path(pwd.getpwuid(os.getuid()).pw_dir)
        / ".automixer"
        / "runtime"
        / "console_writer.lock"
    )


def _open_m110_writer_lock_pin(writer_lock_path: Path) -> int:
    """Open and pin the exact regular-file inode used for writer ownership."""

    writer_lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_metadata = writer_lock_path.parent.lstat()
    if (
        writer_lock_path.parent.is_symlink()
        or not stat.S_ISDIR(parent_metadata.st_mode)
    ):
        raise RuntimeError("console_writer_lock_parent_invalid")
    os.chmod(writer_lock_path.parent, 0o700)
    open_flags = os.O_CREAT | os.O_RDWR
    open_flags |= getattr(os, "O_CLOEXEC", 0)
    open_flags |= getattr(os, "O_NOFOLLOW", 0)
    lock_fd = os.open(writer_lock_path, open_flags, 0o600)
    try:
        os.fchmod(lock_fd, 0o600)
        fd_metadata = os.fstat(lock_fd)
        path_metadata = writer_lock_path.lstat()
        if (
            not stat.S_ISREG(fd_metadata.st_mode)
            or not stat.S_ISREG(path_metadata.st_mode)
            or (fd_metadata.st_dev, fd_metadata.st_ino)
            != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise RuntimeError("console_writer_lock_identity_invalid")
        return lock_fd
    except BaseException:
        os.close(lock_fd)
        raise


def _terminate_exact_child_safely(
    process: subprocess.Popen,
    *,
    writer_lock_path: Path,
    writer_lock_fd: int | None = None,
    graceful_timeout_seconds: float = 10.0,
    poll_interval_seconds: float = 0.1,
    progress_callback: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Terminate a child without SIGKILL while console-write ownership is live.

    SIGTERM lets the backend latch shutdown and drain its in-flight write.  If
    the child does not exit, SIGKILL is permitted only while this parent owns
    the exact process-wide console writer lock.  A lock/open error therefore
    withholds SIGKILL and keeps waiting for the exact child to exit.
    """

    started = time.monotonic()
    result: dict[str, Any] = {
        "terminated": False,
        "clean_shutdown": False,
        "forced": False,
        "returncode": process.poll(),
        "sigterm_sent": False,
        "writer_lock_path": str(writer_lock_path),
        "writer_lock_busy_observed": False,
        "writer_lock_acquired_before_forced_kill": False,
        "writer_lock_identity_verified": False,
        "writer_lock_identity_changed": False,
        "sigkill_withheld": False,
        "wait_seconds": 0.0,
        "errors": [],
    }
    lock_fd: int | None = None
    lock_identity: tuple[int, int] | None = None
    try:
        lock_fd = (
            os.dup(writer_lock_fd)
            if writer_lock_fd is not None
            else _open_m110_writer_lock_pin(writer_lock_path)
        )
        lock_metadata = os.fstat(lock_fd)
        if not stat.S_ISREG(lock_metadata.st_mode):
            raise RuntimeError("console_writer_lock_identity_invalid")
        lock_identity = (lock_metadata.st_dev, lock_metadata.st_ino)
    except (OSError, RuntimeError) as exc:
        if lock_fd is not None:
            os.close(lock_fd)
            lock_fd = None
        result["errors"].append(
            "writer_lock_open_failed:" + type(exc).__name__
        )
        result["sigkill_withheld"] = True
    if result["returncode"] is None:
        try:
            process.terminate()
            result["sigterm_sent"] = True
        except Exception as exc:
            result["errors"].append(
                "sigterm_failed:" + type(exc).__name__
            )
        try:
            process.wait(timeout=float(graceful_timeout_seconds))
        except subprocess.TimeoutExpired:
            pass
        except Exception as exc:
            result["errors"].append(
                "graceful_wait_failed:" + type(exc).__name__
            )

    lock_held = False
    next_progress = time.monotonic() + 30.0
    try:
        while process.poll() is None:
            identity_current = False
            if lock_fd is not None and lock_identity is not None:
                try:
                    path_metadata = writer_lock_path.lstat()
                    identity_current = bool(
                        stat.S_ISREG(path_metadata.st_mode)
                        and (path_metadata.st_dev, path_metadata.st_ino)
                        == lock_identity
                    )
                except OSError:
                    identity_current = False
                if not identity_current:
                    result["writer_lock_identity_changed"] = True
                    if "writer_lock_identity_changed" not in result["errors"]:
                        result["errors"].append(
                            "writer_lock_identity_changed"
                        )
                    result["sigkill_withheld"] = True
                else:
                    result["writer_lock_identity_verified"] = True
            if lock_fd is not None and identity_current and not lock_held:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    lock_held = True
                    result[
                        "writer_lock_acquired_before_forced_kill"
                    ] = True
                except BlockingIOError:
                    result["writer_lock_busy_observed"] = True
                    result["sigkill_withheld"] = True
                except OSError as exc:
                    error = "writer_lock_acquire_failed:" + type(exc).__name__
                    if error not in result["errors"]:
                        result["errors"].append(error)
                    result["sigkill_withheld"] = True
            if lock_held and identity_current and process.poll() is None:
                try:
                    process.kill()
                    result["forced"] = True
                    result["sigkill_withheld"] = False
                except Exception as exc:
                    error = "sigkill_failed:" + type(exc).__name__
                    if error not in result["errors"]:
                        result["errors"].append(error)
                    result["sigkill_withheld"] = True
                    time.sleep(max(0.01, poll_interval_seconds))
                    continue
                while process.poll() is None:
                    try:
                        process.wait(timeout=max(0.01, poll_interval_seconds))
                    except subprocess.TimeoutExpired:
                        continue
                break
            now = time.monotonic()
            if progress_callback is not None and now >= next_progress:
                try:
                    progress_callback(
                        {
                            "event": "m110_backend_shutdown_waiting",
                            "writer_lock_busy": result[
                                "writer_lock_busy_observed"
                            ],
                            "sigkill_withheld": True,
                            "wait_seconds": now - started,
                        }
                    )
                except Exception as exc:
                    error = (
                        "shutdown_progress_callback_failed:"
                        + type(exc).__name__
                    )
                    if error not in result["errors"]:
                        result["errors"].append(error)
                next_progress = now + 30.0
            time.sleep(max(0.01, poll_interval_seconds))
    finally:
        if lock_fd is not None:
            if lock_held:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    returncode = process.poll()
    result.update(
        {
            "terminated": returncode is not None,
            "clean_shutdown": bool(
                returncode == 0 and result["forced"] is False
            ),
            "returncode": returncode,
            "wait_seconds": time.monotonic() - started,
        }
    )
    return result


@dataclass
class ProofOwnedBackendProcess:
    process: subprocess.Popen
    repo_root: Path
    manifest: dict[str, Any]
    secret: bytearray = field(repr=False)
    token: bytearray = field(repr=False)
    ws_url: str
    log_handle: BinaryIO
    writer_lock_path: Path = field(
        default_factory=_m110_console_writer_lock_path
    )
    writer_lock_fd: int | None = field(default=None, repr=False)
    shutdown_result: dict[str, Any] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def close(self) -> dict[str, Any]:
        """Terminate the exact child and erase parent-held authority material."""
        if self.shutdown_result is not None:
            return dict(self.shutdown_result)
        try:
            shutdown_result = _terminate_exact_child_safely(
                self.process,
                writer_lock_path=self.writer_lock_path,
                writer_lock_fd=self.writer_lock_fd,
            )
        finally:
            if self.writer_lock_fd is not None:
                os.close(self.writer_lock_fd)
                self.writer_lock_fd = None
            self.log_handle.close()
            for value in (self.secret, self.token):
                for index in range(len(value)):
                    value[index] = 0
        self.shutdown_result = dict(shutdown_result)
        return dict(self.shutdown_result)

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
    writer_lock_path = _m110_console_writer_lock_path()
    writer_lock_fd: int | None = None
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
        # Pin the writer-lock inode before the child can acquire it.  Teardown
        # can then prove that a same-path replacement is not the lock held by
        # the live writer and must never authorize SIGKILL.
        writer_lock_fd = _open_m110_writer_lock_pin(writer_lock_path)
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
            writer_lock_path=writer_lock_path,
            writer_lock_fd=writer_lock_fd,
        )
    except BaseException:
        for fd in (authority_read, authority_write, token_read, token_write):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if process is not None and process.poll() is None:
            _terminate_exact_child_safely(
                process,
                writer_lock_path=writer_lock_path,
                writer_lock_fd=writer_lock_fd,
                graceful_timeout_seconds=5.0,
            )
        if writer_lock_fd is not None:
            os.close(writer_lock_fd)
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


M110_CONTROL_TERMINAL_TYPES = frozenset(
    {
        "authentication_status",
        "runtime_identity",
        "connection_status",
        "audio_capture_status",
        "wing_master_route_proof_result",
        "m110_attended_control_result",
        "m110_main_output_checkpoint_result",
        "m110_maintenance_reset_proposal_result",
        "m110_m006_restore_proposal_result",
        "m110_processor_enable_proposal_result",
        "m110_soundcheck_child_proposal_result",
        "m110_soundcheck_replan_result",
        "m110_soundcheck_analysis_result",
        "m110_soundcheck_analysis_stop_result",
        "m110_restore_proof_prepare_result",
        "supervised_capability_result",
        "operator_proposal_apply_result",
        "operator_proposal_apply_blocked",
        "control_authentication_required",
    }
)
M110_RFC1918_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)

OperatorInputCallback = Callable[[str], Awaitable[str]]
TerminalEmitterCallback = Callable[[Mapping[str, Any]], Any]


async def _emit_attended_terminal(
    emitter: TerminalEmitterCallback,
    payload: Mapping[str, Any],
) -> None:
    emitted = emitter(dict(payload))
    if asyncio.iscoroutine(emitted) or hasattr(emitted, "__await__"):
        await emitted


async def _require_exact_operator_literal(
    operator_input: OperatorInputCallback,
    *,
    prompt: str,
    expected: str,
    deadline_epoch: float | None = None,
) -> dict[str, Any]:
    if deadline_epoch is not None and time.time() >= deadline_epoch:
        raise RuntimeError("m110_operator_checkpoint_deadline_expired")
    response = await operator_input(prompt)
    captured_at = time.time()
    if deadline_epoch is not None and captured_at >= deadline_epoch:
        raise RuntimeError("m110_operator_checkpoint_deadline_expired")
    if response != expected:
        raise RuntimeError("m110_operator_checkpoint_confirmation_mismatch")
    return {
        "verified": True,
        "expected_literal": expected,
        "operator_response_matched": True,
        "captured_at_epoch": captured_at,
        "deadline_epoch": deadline_epoch,
    }


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
    control_armed: bool = False
    control_approval_id: str | None = None
    analysis_started: bool = False
    closed: bool = False
    teardown_result: dict[str, Any] | None = None
    cp05_gate_result: dict[str, Any] | None = None
    _close_task: asyncio.Task | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _readiness_invalidation_latched: bool = field(
        default=False,
        init=False,
        repr=False,
    )
    _capture_status_get_in_flight: bool = field(
        default=False,
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

    def _reject_unexpected_terminal(
        self,
        payload: Mapping[str, Any],
        *,
        expected_type: str,
    ) -> None:
        message_type = payload.get("type")
        if (
            message_type == "auto_soundcheck_status"
            and payload.get("status") == "readiness_invalidated"
        ):
            self._readiness_invalidation_latched = True
            raise RuntimeError("m110_soundcheck_readiness_invalidated")
        if message_type in M110_CONTROL_TERMINAL_TYPES:
            raise RuntimeError(
                f"unexpected_control_terminal:{message_type}:expected:{expected_type}"
            )
        if payload.get("status") == "blocked" or payload.get("success") is False:
            raise RuntimeError(
                f"interleaved_backend_failure:{message_type or 'missing_type'}"
            )

    async def _reject_immediate_duplicate_terminal(
        self,
        expected_type: str,
        *,
        grace_seconds: float = M110_TERMINAL_DUPLICATE_GRACE_SECONDS,
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
                    f"duplicate_control_terminal:{expected_type}"
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
        if self._readiness_invalidation_latched:
            raise RuntimeError("m110_soundcheck_readiness_invalidated")
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            payload = await self._recv_object(remaining)
            if (
                payload.get("type") == "auto_soundcheck_status"
                and payload.get("status") == "readiness_invalidated"
            ):
                self._reject_unexpected_terminal(
                    payload,
                    expected_type=expected_type,
                )
            if payload.get("type") == expected_type:
                await self._reject_immediate_duplicate_terminal(expected_type)
                if self._readiness_invalidation_latched:
                    raise RuntimeError("m110_soundcheck_readiness_invalidated")
                return payload
            self._reject_unexpected_terminal(
                payload,
                expected_type=expected_type,
            )

    @staticmethod
    def _is_expected_disconnect_readiness_invalidation(
        payload: Mapping[str, Any],
    ) -> bool:
        terminal_error = payload.get("error")
        return bool(
            payload.get("type") == "auto_soundcheck_status"
            and payload.get("status") == "readiness_invalidated"
            and payload.get("running") is False
            and payload.get("is_running") is False
            and payload.get("message")
            == "Mixer disconnected; soundcheck readiness invalidated."
            and isinstance(payload.get("soundcheck_readiness"), Mapping)
            and payload.get("success") is not False
            and (terminal_error is None or terminal_error == "")
        )

    async def _receive_disconnect_terminal(
        self,
        *,
        timeout_seconds: float,
    ) -> tuple[dict[str, Any], bool]:
        """Receive one disconnect ack and its optional exact invalidation."""

        deadline = time.monotonic() + timeout_seconds
        disconnect: dict[str, Any] | None = None
        invalidation_observed = False
        grace_deadline: float | None = None
        while True:
            active_deadline = (
                grace_deadline if grace_deadline is not None else deadline
            )
            remaining = active_deadline - time.monotonic()
            if remaining <= 0:
                if disconnect is not None:
                    return disconnect, invalidation_observed
                raise asyncio.TimeoutError
            try:
                payload = await self._recv_object(remaining)
            except asyncio.TimeoutError:
                if disconnect is not None:
                    return disconnect, invalidation_observed
                raise

            if (
                disconnect is not None
                and self._is_expected_disconnect_readiness_invalidation(payload)
            ):
                if invalidation_observed:
                    raise RuntimeError(
                        "duplicate_disconnect_readiness_invalidation"
                    )
                invalidation_observed = True
                self._readiness_invalidation_latched = True
                continue

            if payload.get("type") == "connection_status":
                if disconnect is not None:
                    raise RuntimeError(
                        "duplicate_control_terminal:connection_status"
                    )
                terminal_error = payload.get("error")
                if not (
                    payload.get("connected") is False
                    and payload.get("status") != "blocked"
                    and payload.get("success") is not False
                    and (terminal_error is None or terminal_error == "")
                ):
                    raise RuntimeError(
                        "m110_disconnect_terminal_not_disconnected"
                    )
                disconnect = dict(payload)
                grace_deadline = min(
                    deadline,
                    time.monotonic()
                    + M110_TERMINAL_DUPLICATE_GRACE_SECONDS,
                )
                continue

            self._reject_unexpected_terminal(
                payload,
                expected_type="connection_status",
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
    def _m110_capture_identity_failures(
        status: Mapping[str, Any],
        *,
        audio_device_index: int,
        expected_device_name: str | None = None,
    ) -> list[str]:
        device_name = status.get("device_name")
        checks = {
            "running": status.get("running") is True,
            "error_absent": status.get("error") is None
            or status.get("error") == "",
            "source_type": status.get("source_type") == "sounddevice",
            "device_name": isinstance(device_name, str)
            and bool(device_name)
            and re.search(
                r"(?:^|[^a-z0-9])wing(?:[^a-z0-9]|$)",
                device_name,
                flags=re.IGNORECASE,
            )
            is not None
            and (
                expected_device_name is None
                or device_name == expected_device_name
            ),
            "device_index": type(status.get("device_index")) is int
            and status.get("device_index") == audio_device_index,
            "num_channels": type(status.get("num_channels")) is int
            and status.get("num_channels") == 48,
            "sample_rate": type(status.get("sample_rate")) is int
            and status.get("sample_rate") == 48_000,
            "real_input_boolean": type(status.get("real_input")) is bool,
            "sample_count": type(status.get("captured_sample_count")) is int
            and status.get("captured_sample_count") >= 0,
            "optional_preview": "preview" not in status
            or status.get("preview") is True,
            "optional_owner": "owned_by_requester" not in status
            or status.get("owned_by_requester") is True,
        }
        return [name for name, passed in checks.items() if not passed]

    @staticmethod
    def _m110_capture_readiness_state(
        status: Mapping[str, Any],
        *,
        initial_sample_count: int,
    ) -> str:
        real_input = status.get("real_input")
        sample_count = status.get("captured_sample_count")
        callback_age = status.get("callback_age_sec")
        if real_input is False:
            if callback_age is None and sample_count == initial_sample_count == 0:
                return "pending_first_callback"
            return "invalid_not_ready_state"
        if real_input is not True:
            return "invalid_real_input_state"
        if not (
            isinstance(callback_age, (int, float))
            and not isinstance(callback_age, bool)
            and math.isfinite(float(callback_age))
            and 0.0 <= float(callback_age) <= 1.0
        ):
            return "invalid_callback_freshness"
        if not (
            type(sample_count) is int
            and sample_count > initial_sample_count
        ):
            return "missing_sample_progression"
        return "ready"

    async def _await_m110_capture_ready(
        self,
        *,
        initial_status: Mapping[str, Any],
        audio_device_index: int,
    ) -> dict[str, Any]:
        device_name = str(initial_status["device_name"])
        initial_sample_count = int(initial_status["captured_sample_count"])
        deadline = time.monotonic() + M110_CAPTURE_READY_TIMEOUT_SECONDS
        poll_count = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= M110_TERMINAL_DUPLICATE_GRACE_SECONDS:
                raise RuntimeError(
                    "m110_real_48_channel_capture_not_ready_within_1s"
                )
            sleep_seconds = min(
                M110_CAPTURE_READY_POLL_INTERVAL_SECONDS,
                max(
                    0.0,
                    remaining - M110_TERMINAL_DUPLICATE_GRACE_SECONDS,
                ),
            )
            if sleep_seconds > 0.0:
                await asyncio.sleep(sleep_seconds)
            remaining = deadline - time.monotonic()
            if remaining <= M110_TERMINAL_DUPLICATE_GRACE_SECONDS:
                raise RuntimeError(
                    "m110_real_48_channel_capture_not_ready_within_1s"
                )
            try:
                self._capture_status_get_in_flight = True
                await asyncio.wait_for(
                    self._send_json({"type": "get_audio_capture_status"}),
                    timeout=(
                        remaining - M110_TERMINAL_DUPLICATE_GRACE_SECONDS
                    ),
                )
                remaining = deadline - time.monotonic()
                if remaining <= M110_TERMINAL_DUPLICATE_GRACE_SECONDS:
                    raise asyncio.TimeoutError
                status = await self._receive_terminal(
                    "audio_capture_status",
                    timeout_seconds=(
                        remaining - M110_TERMINAL_DUPLICATE_GRACE_SECONDS
                    ),
                )
                self._capture_status_get_in_flight = False
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    "m110_real_48_channel_capture_not_ready_within_1s"
                ) from exc
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "m110_real_48_channel_capture_not_ready_within_1s"
                )
            poll_count += 1
            identity_failures = self._m110_capture_identity_failures(
                status,
                audio_device_index=audio_device_index,
                expected_device_name=device_name,
            )
            if identity_failures:
                raise RuntimeError(
                    "m110_real_48_channel_capture_status_drift:"
                    + ",".join(identity_failures)
                )
            readiness = self._m110_capture_readiness_state(
                status,
                initial_sample_count=initial_sample_count,
            )
            if readiness == "ready":
                return {
                    **dict(initial_status),
                    **dict(status),
                    "preview": True,
                    "owned_by_requester": True,
                    "device_id": audio_device_index,
                    "readiness_poll_count": poll_count,
                }
            if readiness != "pending_first_callback":
                raise RuntimeError(
                    "m110_real_48_channel_capture_status_invalid:"
                    + readiness
                )

    async def _receive_m110_owned_capture_stop_terminal(
        self,
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Correlate STOP despite one possible late response to the last GET."""

        deadline = time.monotonic() + timeout_seconds
        late_get_status_seen = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise asyncio.TimeoutError
            payload = await self._recv_object(remaining)
            if payload.get("type") != "audio_capture_status":
                self._reject_unexpected_terminal(
                    payload,
                    expected_type="audio_capture_status",
                )
                continue
            if (
                payload.get("running") is False
                and payload.get("preview") is True
                and payload.get("owned_by_requester") is True
            ):
                self._capture_status_get_in_flight = False
                return payload
            # The poll loop has at most one outstanding read-only GET.  If it
            # is cancelled after send, its generic status can precede the STOP
            # response and carries neither preview nor owner fields.
            if (
                self._capture_status_get_in_flight
                and not late_get_status_seen
                and payload.get("running") is True
                and "preview" not in payload
                and "owned_by_requester" not in payload
                and (
                    payload.get("error") is None
                    or payload.get("error") == ""
                )
            ):
                late_get_status_seen = True
                self._capture_status_get_in_flight = False
                continue
            raise RuntimeError("m110_owned_capture_stop_terminal_invalid")

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
            "firmware": is_m110_wing_firmware_version(
                result.get("firmware_version")
            ),
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
            and isinstance(coreaudio, Mapping)
            and set(coreaudio)
            == {
                "manufacturer",
                "transport",
                "input_channels",
                "output_channels",
                "sample_rate_hz",
            }
            and coreaudio.get("manufacturer") == "BEHRINGER"
            and coreaudio.get("transport") == "usb"
            and exact_integer(coreaudio.get("input_channels"), 48)
            and exact_integer(coreaudio.get("output_channels"), 48)
            and exact_integer(coreaudio.get("sample_rate_hz"), 48_000)
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
        ):
            raise RuntimeError(
                "m110_real_48_channel_capture_start_failed:"
                + self._bounded_terminal_failure_context(capture)
            )
        # Ownership is established even if the first CoreAudio callback has
        # not arrived yet.  Latch teardown before any later identity/freshness
        # check so every failure explicitly stops this preview.
        self.capture_started = True
        if not (
            capture.get("error") is None
            or capture.get("error") == ""
        ):
            raise RuntimeError(
                "m110_real_48_channel_capture_start_failed:"
                + self._bounded_terminal_failure_context(capture)
            )
        identity_failures = self._m110_capture_identity_failures(
            capture,
            audio_device_index=audio_device_index,
        )
        if identity_failures:
            raise RuntimeError(
                "m110_real_48_channel_capture_start_failed:identity="
                + ",".join(identity_failures)
            )
        initial_sample_count = int(capture["captured_sample_count"])
        readiness = self._m110_capture_readiness_state(
            capture,
            initial_sample_count=(
                0 if capture.get("real_input") is True else initial_sample_count
            ),
        )
        if readiness == "pending_first_callback":
            capture = await self._await_m110_capture_ready(
                initial_status=capture,
                audio_device_index=audio_device_index,
            )
        elif readiness == "ready":
            capture = {**capture, "readiness_poll_count": 0}
        else:
            raise RuntimeError(
                "m110_real_48_channel_capture_start_failed:"
                + readiness
            )

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

    async def verify_m110_main_output_checkpoint(
        self,
        *,
        checkpoint: str,
        timeout_seconds: float = 15.0,
    ) -> dict[str, Any]:
        """Verify one server-fixed Main state; never accept an OSC address."""

        if checkpoint not in {"reset_muted", "soundcheck_unmuted"}:
            raise ValueError("m110_main_output_checkpoint_invalid")
        if not self.control_armed or not self.route_established:
            raise RuntimeError("m110_attended_control_must_be_armed")
        await self._send_json(
            {
                "type": "verify_m110_main_output_checkpoint",
                "contract_version": 1,
                "checkpoint": checkpoint,
            }
        )
        terminal = await self._receive_terminal(
            "m110_main_output_checkpoint_result",
            timeout_seconds=float(timeout_seconds),
        )
        self._assert_no_capability_secret(terminal)
        expected_reason = (
            "m110_reset_muted_containment_verified"
            if checkpoint == "reset_muted"
            else "m110_soundcheck_unmuted_audio_verified"
        )
        before = terminal.get("main_mute_before")
        after = terminal.get("main_mute_after")
        if not (
            terminal.get("type") == "m110_main_output_checkpoint_result"
            and terminal.get("status") == "verified"
            and terminal.get("success") is True
            and terminal.get("verified") is True
            and terminal.get("reason") == expected_reason
            and terminal.get("checkpoint") == checkpoint
            and terminal.get("audio_status") == "passed"
            and isinstance(terminal.get("audio_reason"), str)
            and bool(terminal.get("audio_reason"))
            and isinstance(before, Mapping)
            and before.get("verified") is True
            and isinstance(after, Mapping)
            and after.get("verified") is True
            and terminal.get("console_mutation_write_attempt_count") == 0
            and not isinstance(
                terminal.get("console_mutation_write_attempt_count"),
                bool,
            )
            and terminal.get("live_mixer_writes") is False
            and terminal.get("live_authority_granted") is False
            and terminal.get("readiness_closure_allowed") is False
        ):
            raise RuntimeError(
                "m110_main_output_checkpoint_failed:"
                + self._bounded_terminal_failure_context(terminal)
            )
        return {
            "verified": True,
            "checkpoint": checkpoint,
            "reason": expected_reason,
            "terminal": terminal,
            "console_mutation_write_attempt_count": 0,
        }

    @staticmethod
    def _require_sha256(value: Any, *, field_name: str) -> str:
        if not (
            isinstance(value, str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None
        ):
            raise RuntimeError(f"m110_{field_name}_invalid")
        return value

    @staticmethod
    def _assert_no_capability_secret(payload: Any) -> None:
        """Reject a terminal payload that contains any token-shaped material."""

        nodes = 0

        def visit(value: Any, depth: int) -> None:
            nonlocal nodes
            nodes += 1
            if nodes > 4096 or depth > 16:
                raise RuntimeError("m110_terminal_secret_scan_bounds_exceeded")
            if isinstance(value, str):
                if "m110cap_v1_" in value:
                    raise RuntimeError("m110_capability_token_leaked_to_terminal")
                return
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if key == "supervised_capability_token":
                        raise RuntimeError(
                            "m110_capability_token_field_leaked_to_terminal"
                        )
                    visit(item, depth + 1)
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item, depth + 1)

        visit(payload, 0)

    @classmethod
    def _validate_fixed_reset_proposal_terminal(
        cls,
        terminal: Mapping[str, Any],
        *,
        role: str,
    ) -> dict[str, Any]:
        cls._assert_no_capability_secret(terminal)
        if role not in {"proof", "commit"}:
            raise ValueError("m110_fixed_reset_role_invalid")
        required_kind = (
            "restore_only_proof"
            if role == "proof"
            else "maintenance_dsp_reset_commit_once"
        )
        proposal_id = terminal.get("proposal_id")
        review_digest = cls._require_sha256(
            terminal.get("review_content_digest"),
            field_name="review_content_digest",
        )
        operation_digest = cls._require_sha256(
            terminal.get("operation_digest"),
            field_name="operation_digest",
        )
        review_preimage_sha256 = cls._require_sha256(
            terminal.get("review_preimage_sha256"),
            field_name="review_preimage_sha256",
        )
        review_operations = terminal.get("review_operations")
        expected_review_operations = [
            {
                "operation_id": str(row["operation_id"]),
                "family_id": str(row["family_id"]),
                "write_kind": str(row["write_kind"]),
                "channel": int(row["channel"]),
                "parameter": str(row["parameter"]),
                "address": str(row["address"]),
                "target_value": row["target_value"],
            }
            for row in M110_RESET_OPERATION_DEFINITIONS
        ]
        reset_review_valid = bool(
            isinstance(review_operations, list)
            and review_operations == expected_review_operations
        )
        if not (
            terminal.get("type") == "m110_maintenance_reset_proposal_result"
            and terminal.get("status") == "created"
            and terminal.get("success") is True
            and terminal.get("reason")
            == "m110_maintenance_reset_proposal_created"
            and isinstance(proposal_id, str)
            and re.fullmatch(
                rf"m110-maintenance-reset-{role}-[0-9a-f]{{32}}",
                proposal_id,
            )
            is not None
            and type(terminal.get("review_contract_version")) is int
            and terminal.get("review_contract_version") == 2
            and terminal.get("required_capability_kind") == required_kind
            and terminal.get("bundle_kind") == M110_RESET_BUNDLE_KIND
            and terminal.get("operation_count") == len(M110_RESET_ADDRESSES)
            and not isinstance(terminal.get("operation_count"), bool)
            and terminal.get("scope_addresses") == list(M110_RESET_ADDRESSES)
            and reset_review_valid
            and terminal.get("matching_restored_proof_receipt")
            is (role == "commit")
            and terminal.get("token_issued") is False
            and terminal.get("live_authority_granted") is False
            and terminal.get("readiness_closure_allowed") is False
        ):
            raise RuntimeError(
                "m110_fixed_reset_proposal_invalid:"
                + cls._bounded_terminal_failure_context(terminal)
            )
        return {
            "role": role,
            "proposal_id": proposal_id,
            "review_contract_version": 2,
            "review_content_digest": review_digest,
            "operation_digest": operation_digest,
            "review_preimage_sha256": review_preimage_sha256,
            "required_capability_kind": required_kind,
            "bundle_kind": M110_RESET_BUNDLE_KIND,
            "scope_addresses": list(M110_RESET_ADDRESSES),
            "review_operations": [
                dict(item) for item in review_operations
            ],
        }

    @classmethod
    def _validate_fixed_reset_apply_terminal(
        cls,
        terminal: Mapping[str, Any],
        *,
        created: Mapping[str, Any],
        capability_kind: str,
    ) -> None:
        cls._assert_no_capability_secret(terminal)
        proof_run = capability_kind == "restore_only_proof"
        if capability_kind not in {
            "restore_only_proof",
            "maintenance_dsp_reset_commit_once",
        }:
            raise ValueError("m110_fixed_reset_capability_kind_invalid")
        expected_status = "proof_passed" if proof_run else "applied"
        expected_wal = "proof_restored" if proof_run else "committed"
        expected_operation_ids = [
            str(item.get("operation_id"))
            for item in created.get("review_operations") or []
            if isinstance(item, Mapping)
            and isinstance(item.get("operation_id"), str)
            and item.get("operation_id")
        ]
        expected_operation_id_set = set(expected_operation_ids)
        applied_operation_ids = terminal.get("applied_operation_ids")
        rollback_operation_ids = terminal.get("rollback_operation_ids")
        already_satisfied_operation_ids = terminal.get(
            "already_satisfied_operation_ids"
        )
        operation_identifier_lists_valid = bool(
            len(expected_operation_ids) == len(M110_RESET_ADDRESSES)
            and len(expected_operation_ids)
            == len(expected_operation_id_set)
            and type(applied_operation_ids) is list
            and type(rollback_operation_ids) is list
            and type(already_satisfied_operation_ids) is list
            and all(
                type(operation_id) is str and bool(operation_id)
                for operation_id in (
                    *applied_operation_ids,
                    *rollback_operation_ids,
                    *already_satisfied_operation_ids,
                )
            )
            and len(applied_operation_ids)
            == len(set(applied_operation_ids))
            and len(rollback_operation_ids)
            == len(set(rollback_operation_ids))
            and len(already_satisfied_operation_ids)
            == len(set(already_satisfied_operation_ids))
            and set(applied_operation_ids).isdisjoint(
                already_satisfied_operation_ids
            )
        )
        exact_operation_coverage = bool(
            operation_identifier_lists_valid
            and set(applied_operation_ids).union(
                already_satisfied_operation_ids
            )
            == expected_operation_id_set
            and (
                set(rollback_operation_ids) == set(applied_operation_ids)
                if proof_run
                else rollback_operation_ids == []
            )
        )
        all_targets_already_satisfied = bool(
            exact_operation_coverage
            and applied_operation_ids == []
            and rollback_operation_ids == []
            and set(already_satisfied_operation_ids)
            == expected_operation_id_set
        )
        expected_rollback = bool(
            proof_run and not all_targets_already_satisfied
        )
        expected_rollback_status = (
            "not_required_already_satisfied"
            if proof_run and all_targets_already_satisfied
            else "verified"
            if proof_run
            else "not_attempted"
        )
        result_digest = cls._require_sha256(
            terminal.get("result_digest"),
            field_name="fixed_reset_result_digest",
        )
        for field_name in (
            "review_preimage_sha256",
            "before_manifest_sha256",
            "after_manifest_sha256",
            "verification_cut_reset_surface_sha256",
            "verification_cut_preservation_manifest_sha256",
        ):
            cls._require_sha256(
                terminal.get(field_name),
                field_name=field_name,
            )
        proposal = terminal.get("proposal")
        if not (
            terminal.get("type") == "operator_proposal_apply_result"
            and exact_operation_coverage
            and type(terminal.get("contract_version")) is int
            and terminal.get("contract_version") == 2
            and terminal.get("status") == expected_status
            and terminal.get("success") is True
            and terminal.get("proposal_id") == created.get("proposal_id")
            and terminal.get("proposal_status") == expected_status
            and terminal.get("apply_status") == expected_status
            and terminal.get("accepted") is False
            and terminal.get("owner_review_required") is True
            and terminal.get("live_authority_granted") is False
            and terminal.get("readiness_closure_allowed") is False
            and terminal.get("proof_run") is proof_run
            and terminal.get("restore_after_verify") is proof_run
            and terminal.get("capability_kind") == capability_kind
            and terminal.get("operation_digest")
            == created.get("operation_digest")
            and terminal.get("bundle_kind") == M110_RESET_BUNDLE_KIND
            and terminal.get("review_preimage_sha256")
            == created.get("review_preimage_sha256")
            and terminal.get("proof_target_verified") is proof_run
            and terminal.get("no_console_send_proven")
            is all_targets_already_satisfied
            and terminal.get("physical_verified") is True
            and terminal.get("physical_state_restored") is proof_run
            and terminal.get("terminal_after_console_send")
            is (not all_targets_already_satisfied)
            and terminal.get("write_state_unknown") is False
            and terminal.get("rollback_attempted") is expected_rollback
            and terminal.get("rollback_succeeded") is expected_rollback
            and terminal.get("rollback_state_unknown") is False
            and terminal.get("manual_review_required") is False
            and terminal.get("emergency_stop_activated") is False
            and terminal.get("emergency_stop_required") is False
            and terminal.get("wal_terminal") == expected_wal
            and terminal.get("rollback_status")
            == expected_rollback_status
            and terminal.get("audio_recheck_passed") is True
            and terminal.get("no_outside_mutation_attempt_proven") is True
            and terminal.get("current_state_guaranteed_at_response") is False
            and terminal.get("fresh_readback_required_before_next_stage")
            is True
            and terminal.get("attended_no_touch_required_during_execution")
            is True
            and terminal.get(
                "later_console_event_invalidates_readiness_freshness"
            )
            is True
            and terminal.get(
                "verification_cut_preservation_manifest_sha256"
            )
            == terminal.get("after_manifest_sha256")
            and terminal.get("result_detail_available") is True
            and terminal.get("result_evidence_pinned") is False
            and (
                proof_run
                or terminal.get("parent_commit_recorded") is True
            )
            and isinstance(proposal, Mapping)
            and proposal.get("id") == created.get("proposal_id")
            and proposal.get("status") == expected_status
            and proposal.get("review_contract_version") == 2
            and proposal.get("review_content_digest")
            == created.get("review_content_digest")
            and proposal.get("result_digest") == result_digest
            and proposal.get("result_detail_available") is True
            and proposal.get("result_evidence_pinned") is False
        ):
            raise RuntimeError(
                "m110_fixed_reset_apply_terminal_invalid:"
                + cls._bounded_terminal_failure_context(terminal)
            )

    @classmethod
    def _validate_m006_restore_proposal_terminal(
        cls,
        terminal: Mapping[str, Any],
        *,
        stage: str,
        role: str,
    ) -> dict[str, Any]:
        """Accept only the immutable server-owned M006 inverse stage."""

        cls._assert_no_capability_secret(terminal)
        if stage not in M110_M006_RESTORE_STAGES or role not in {
            "proof",
            "commit",
        }:
            raise ValueError("m110_m006_restore_stage_role_invalid")
        required_kind = (
            "restore_only_proof"
            if role == "proof"
            else "maintenance_dsp_reset_commit_once"
        )
        expected_operations = [
            dict(item)
            for item in m110_m006_restore_stage_operations()[stage]
        ]
        expected_scope = {
            str(address)
            for operation in expected_operations
            for address in dict(
                operation["metadata"]["m110_expected_current_values"]
            )
        }
        proposal_id = terminal.get("proposal_id")
        review_digest = cls._require_sha256(
            terminal.get("review_content_digest"),
            field_name="review_content_digest",
        )
        operation_digest = cls._require_sha256(
            terminal.get("operation_digest"),
            field_name="operation_digest",
        )
        scope_addresses = terminal.get("scope_addresses")
        review_operations = terminal.get("review_operations")
        snapshots = terminal.get("snapshot_summaries")
        already_restored_count = terminal.get(
            "already_restored_address_count"
        )
        postimage_count = terminal.get("postimage_address_count")
        expected_operation_ids = [
            item["operation_id"] for item in expected_operations
        ]
        if not (
            terminal.get("type") == "m110_m006_restore_proposal_result"
            and terminal.get("status") == "created"
            and terminal.get("success") is True
            and terminal.get("reason")
            == "m110_m006_restore_proposal_created"
            and isinstance(proposal_id, str)
            and re.fullmatch(
                rf"m110-m006-restore-{re.escape(stage)}-{role}-[0-9a-f]{{32}}",
                proposal_id,
            )
            is not None
            and terminal.get("stage") == stage
            and type(terminal.get("review_contract_version")) is int
            and terminal.get("review_contract_version") == 2
            and terminal.get("review_preimage_sha256") == ""
            and terminal.get("required_capability_kind") == required_kind
            and terminal.get("bundle_kind")
            == M110_M006_RESTORE_BUNDLE_KIND
            and terminal.get("operation_count") == len(expected_operations)
            and not isinstance(terminal.get("operation_count"), bool)
            and review_operations == expected_operations
            and isinstance(scope_addresses, list)
            and len(scope_addresses) == len(expected_scope)
            and len(scope_addresses) == len(set(scope_addresses))
            and set(scope_addresses) == expected_scope
            and isinstance(snapshots, list)
            and [item.get("operation_id") for item in snapshots]
            == expected_operation_ids
            and all(
                isinstance(item, Mapping)
                and item.get("physical_verified") is True
                for item in snapshots
            )
            and isinstance(already_restored_count, int)
            and not isinstance(already_restored_count, bool)
            and already_restored_count >= 0
            and isinstance(postimage_count, int)
            and not isinstance(postimage_count, bool)
            and postimage_count >= 0
            and already_restored_count + postimage_count
            == len(expected_scope)
            and terminal.get("matching_restored_proof_receipt")
            is (role == "commit")
            and terminal.get("token_issued") is False
            and terminal.get("live_authority_granted") is False
            and terminal.get("readiness_closure_allowed") is False
        ):
            raise RuntimeError(
                "m110_m006_restore_proposal_invalid:"
                + cls._bounded_terminal_failure_context(terminal)
            )
        return {
            "state": "created",
            "lane": "m006_restore",
            "stage": stage,
            "role": role,
            "proposal_id": proposal_id,
            "review_contract_version": 2,
            "review_content_digest": review_digest,
            "operation_digest": operation_digest,
            "review_preimage_sha256": "",
            "required_capability_kind": required_kind,
            "bundle_kind": M110_M006_RESTORE_BUNDLE_KIND,
            "operation_count": len(expected_operations),
            "scope_addresses": list(scope_addresses),
            "review_operations": expected_operations,
            "source_transaction_id": M110_M006_SOURCE_TRANSACTION_ID,
            "source_event_sha256": M110_M006_SOURCE_EVENT_SHA256,
        }

    async def _create_fixed_reset_proposal(
        self,
        *,
        role: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        await self._send_json(
            {
                "type": f"create_m110_maintenance_reset_{role}_proposal",
                "contract_version": 1,
                "bundle_selector": "fixed_channels_1_16_dsp_bypass",
            }
        )
        terminal = await self._receive_terminal(
            "m110_maintenance_reset_proposal_result",
            timeout_seconds=min(float(timeout_seconds), 900.0),
        )
        return self._validate_fixed_reset_proposal_terminal(
            terminal,
            role=role,
        )

    async def _create_m006_restore_proposal(
        self,
        *,
        stage: str,
        role: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        if stage not in M110_M006_RESTORE_STAGES or role not in {
            "proof",
            "commit",
        }:
            raise ValueError("m110_m006_restore_stage_role_invalid")
        await self._send_json(
            {
                "type": f"create_m110_m006_restore_{role}_proposal",
                "contract_version": 1,
                "bundle_selector": m110_m006_restore_bundle_selector(stage),
            }
        )
        terminal = await self._receive_terminal(
            "m110_m006_restore_proposal_result",
            timeout_seconds=float(timeout_seconds),
        )
        return self._validate_m006_restore_proposal_terminal(
            terminal,
            stage=stage,
            role=role,
        )

    async def _require_bundle_operator_review(
        self,
        *,
        created: Mapping[str, Any],
        capability_kind: str,
        approval_id: str,
        operator_input: OperatorInputCallback,
        terminal_emitter: TerminalEmitterCallback,
    ) -> dict[str, Any]:
        operation_digest = str(created.get("operation_digest") or "")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", operation_digest) is None:
            raise RuntimeError("m110_operator_review_digest_invalid")
        review_event = {
            "event": "m110_operator_bundle_review_required",
            "proposal_id": created.get("proposal_id"),
            "capability_kind": capability_kind,
            "approval_id": approval_id,
            "lane": created.get("lane", "maintenance_reset"),
            "role": created.get("role"),
            "stage": created.get("stage"),
            "bundle_kind": created.get("bundle_kind"),
            "source_transaction_id": created.get(
                "source_transaction_id"
            ),
            "source_event_sha256": created.get("source_event_sha256"),
            "review_content_digest": created.get("review_content_digest"),
            "operation_digest": operation_digest,
            "operation_count": created.get("operation_count"),
            "scope_addresses": list(created.get("scope_addresses") or []),
            "review_operations": [
                dict(item)
                for item in created.get("review_operations") or []
                if isinstance(item, Mapping)
            ],
            "source_operation_ids": list(
                created.get("source_operation_ids") or []
            ),
            "requires_exact_digest_entry": True,
            "live_authority_granted": False,
        }
        await _emit_attended_terminal(terminal_emitter, review_event)
        acknowledgement = await _require_exact_operator_literal(
            operator_input,
            prompt=(
                "Review the exact M110 bundle above. To approve this one "
                "capability, enter its full operation_digest: "
            ),
            expected=operation_digest,
        )
        return {
            **acknowledgement,
            "proposal_id": created.get("proposal_id"),
            "capability_kind": capability_kind,
            "operation_digest": operation_digest,
        }

    async def _prepare_issue_apply_fixed_reset(
        self,
        *,
        created: Mapping[str, Any],
        capability_kind: str,
        approval_id: str,
        operator_input: OperatorInputCallback,
        terminal_emitter: TerminalEmitterCallback,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        proposal_id = str(created["proposal_id"])
        await self._send_json(
            {
                "type": "prepare_m110_restore_only_proof",
                "contract_version": 1,
                "proposal_id": proposal_id,
                "bundle_selector": "reviewed_supervised_transaction",
            }
        )
        prepared = await self._receive_terminal(
            "m110_restore_proof_prepare_result",
            timeout_seconds=min(float(timeout_seconds), 20.0),
        )
        if not (
            prepared.get("status") == "prepared"
            and prepared.get("success") is True
            and prepared.get("reason") == "m110_fixed_reset_bundle_current"
            and prepared.get("proposal_id") == proposal_id
            and type(prepared.get("review_contract_version")) is int
            and prepared.get("review_contract_version") == 2
            and prepared.get("review_content_digest")
            == created.get("review_content_digest")
            and prepared.get("operation_digest")
            == created.get("operation_digest")
            and prepared.get("bundle_kind") == M110_RESET_BUNDLE_KIND
            and prepared.get("review_preimage_sha256")
            == created.get("review_preimage_sha256")
            and prepared.get("required_capability_kind") == capability_kind
            and prepared.get("operation_count") == len(M110_RESET_ADDRESSES)
            and not isinstance(prepared.get("operation_count"), bool)
            and prepared.get("scope_addresses") == list(M110_RESET_ADDRESSES)
            and prepared.get("token_issued") is False
            and prepared.get("live_authority_granted") is False
        ):
            raise RuntimeError(
                "m110_fixed_reset_prepare_failed:"
                + self._bounded_terminal_failure_context(prepared)
            )

        operator_checkpoint = await self._require_bundle_operator_review(
            created=created,
            capability_kind=capability_kind,
            approval_id=approval_id,
            operator_input=operator_input,
            terminal_emitter=terminal_emitter,
        )

        issue_request = build_m110_capability_request(
            capability_kind=capability_kind,
            proposal_id=proposal_id,
            review_contract_version=2,
            review_content_digest=str(created["review_content_digest"]),
            operation_digest=str(created["operation_digest"]),
            approval_id=approval_id,
            request_nonce=secrets.token_hex(16),
            secret=bytes(self.backend.secret),
        )
        await self._send_json(issue_request)
        issued = await self._receive_terminal(
            "supervised_capability_result",
            timeout_seconds=min(float(timeout_seconds), 20.0),
        )
        raw_token = issued.pop("supervised_capability_token", None)
        idempotency_key = issued.get("idempotency_key")
        latest_start_epoch = issued.get("latest_start_epoch")
        idempotency_pattern = (
            r"m110-proof:[0-9a-f]{32}"
            if capability_kind == "restore_only_proof"
            else r"m110-reset-commit:[0-9a-f]{32}"
        )
        now = time.time()
        if not (
            issued.get("status") == "issued"
            and issued.get("success") is True
            and issued.get("reason") == "m110_capability_issued"
            and issued.get("capability_kind") == capability_kind
            and issued.get("proposal_id") == proposal_id
            and issued.get("review_contract_version") == 2
            and issued.get("review_content_digest")
            == created.get("review_content_digest")
            and issued.get("operation_digest")
            == created.get("operation_digest")
            and issued.get("approval_id") == approval_id
            and isinstance(idempotency_key, str)
            and re.fullmatch(idempotency_pattern, idempotency_key) is not None
            and isinstance(latest_start_epoch, (int, float))
            and not isinstance(latest_start_epoch, bool)
            and math.isfinite(float(latest_start_epoch))
            and now < float(latest_start_epoch) <= now + 31.0
            and isinstance(raw_token, str)
            and re.fullmatch(r"m110cap_v1_[0-9a-f]{64}", raw_token) is not None
            and issued.get("single_use") is True
            and issued.get("stores_plaintext_token") is False
            and issued.get("live_authority_granted") is False
        ):
            raw_token = None
            raise RuntimeError(
                "m110_fixed_reset_capability_issue_failed:"
                + self._bounded_terminal_failure_context(issued)
            )

        token = bytearray(raw_token.encode("ascii"))
        raw_token = None
        apply_request = {
            "type": "apply_operator_proposal",
            "proposal_id": proposal_id,
            "approved": True,
            "approval_id": approval_id,
            "idempotency_key": idempotency_key,
            "expected_review_contract_version": 2,
            "expected_review_content_digest": created[
                "review_content_digest"
            ],
            "expected_operation_digest": created["operation_digest"],
            "supervised_capability_token": token.decode("ascii"),
        }
        try:
            await self._send_json(apply_request)
        finally:
            apply_request.pop("supervised_capability_token", None)
            for index in range(len(token)):
                token[index] = 0
        terminal = await self._receive_terminal(
            "operator_proposal_apply_result",
            timeout_seconds=float(timeout_seconds),
        )
        self._validate_fixed_reset_apply_terminal(
            terminal,
            created=created,
            capability_kind=capability_kind,
        )
        await _emit_attended_terminal(
            terminal_emitter,
            {
                "event": "m110_capability_terminal",
                "proposal_id": proposal_id,
                "capability_kind": capability_kind,
                "operation_digest": created["operation_digest"],
                "terminal": dict(terminal),
            },
        )
        return {
            "verified": True,
            "proposal_id": proposal_id,
            "capability_kind": capability_kind,
            "operation_digest": created["operation_digest"],
            "review_preimage_sha256": created["review_preimage_sha256"],
            "scope_addresses": list(M110_RESET_ADDRESSES),
            "terminal": terminal,
            "operator_checkpoint": operator_checkpoint,
            "token_retained": False,
        }

    async def execute_fixed_m110_maintenance_reset(
        self,
        *,
        approval_id: str,
        operator_input: OperatorInputCallback,
        terminal_emitter: TerminalEmitterCallback,
        timeout_seconds: float = 900.0,
    ) -> dict[str, Any]:
        """Proof, recapture and permanently bypass the fixed 1-16 DSP set."""

        try:
            return await self._execute_fixed_m110_maintenance_reset(
                approval_id=approval_id,
                operator_input=operator_input,
                terminal_emitter=terminal_emitter,
                timeout_seconds=timeout_seconds,
            )
        except BaseException:
            await self.close()
            raise

    async def _execute_fixed_m110_maintenance_reset(
        self,
        *,
        approval_id: str,
        operator_input: OperatorInputCallback,
        terminal_emitter: TerminalEmitterCallback,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        if not self.route_established or self.route_result is None:
            raise RuntimeError("m110_master_route_must_precede_capability")
        if self.closed:
            raise RuntimeError("proof_owned_backend_session_closed")
        if not self.control_armed:
            raise RuntimeError("m110_attended_control_must_be_armed")
        if approval_id != self.control_approval_id:
            raise RuntimeError("m110_attended_control_approval_mismatch")
        if not (
            isinstance(timeout_seconds, (int, float))
            and not isinstance(timeout_seconds, bool)
            and math.isfinite(float(timeout_seconds))
            and 0.0 < float(timeout_seconds) <= 900.0
        ):
            raise ValueError("m110_capability_timeout_invalid")

        proof_created = await self._create_fixed_reset_proposal(
            role="proof",
            timeout_seconds=timeout_seconds,
        )
        proof = await self._prepare_issue_apply_fixed_reset(
            created=proof_created,
            capability_kind="restore_only_proof",
            approval_id=approval_id,
            operator_input=operator_input,
            terminal_emitter=terminal_emitter,
            timeout_seconds=timeout_seconds,
        )
        commit_created = await self._create_fixed_reset_proposal(
            role="commit",
            timeout_seconds=timeout_seconds,
        )
        if not (
            commit_created.get("operation_digest")
            == proof_created.get("operation_digest")
            and commit_created.get("review_preimage_sha256")
            == proof_created.get("review_preimage_sha256")
            and commit_created.get("scope_addresses")
            == proof_created.get("scope_addresses")
        ):
            raise RuntimeError("m110_fixed_reset_proof_commit_binding_drift")
        commit = await self._prepare_issue_apply_fixed_reset(
            created=commit_created,
            capability_kind="maintenance_dsp_reset_commit_once",
            approval_id=approval_id,
            operator_input=operator_input,
            terminal_emitter=terminal_emitter,
            timeout_seconds=timeout_seconds,
        )
        return {
            "verified": True,
            "reason": "m110_fixed_maintenance_reset_committed",
            "operation_digest": commit_created["operation_digest"],
            "review_preimage_sha256": commit_created[
                "review_preimage_sha256"
            ],
            "scope_addresses": list(M110_RESET_ADDRESSES),
            "proof": proof,
            "commit": commit,
            "same_websocket_retained": True,
            "backend_process_retained": self.backend.process.poll() is None,
            "token_retained": False,
            "reset_applied": True,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
        }

    async def execute_m110_m006_exact_preimage_restore(
        self,
        *,
        approval_id: str,
        operator_input: OperatorInputCallback,
        terminal_emitter: TerminalEmitterCallback,
        timeout_seconds: float = 900.0,
    ) -> dict[str, Any]:
        """Restore all 77 M006-changed addresses while Main remains muted."""

        try:
            stages: list[dict[str, Any]] = []
            for stage in M110_M006_RESTORE_STAGES:
                proof_created = await self._create_m006_restore_proposal(
                    stage=stage,
                    role="proof",
                    timeout_seconds=timeout_seconds,
                )
                proof = await self._prepare_issue_apply_lane(
                    created=proof_created,
                    capability_kind="restore_only_proof",
                    approval_id=approval_id,
                    operator_input=operator_input,
                    terminal_emitter=terminal_emitter,
                    timeout_seconds=timeout_seconds,
                )
                commit_created = await self._create_m006_restore_proposal(
                    stage=stage,
                    role="commit",
                    timeout_seconds=timeout_seconds,
                )
                if not (
                    commit_created.get("operation_digest")
                    == proof_created.get("operation_digest")
                    and commit_created.get("bundle_kind")
                    == proof_created.get("bundle_kind")
                    and commit_created.get("stage") == stage
                    and commit_created.get("scope_addresses")
                    == proof_created.get("scope_addresses")
                    and commit_created.get("review_operations")
                    == proof_created.get("review_operations")
                    and commit_created.get("source_transaction_id")
                    == proof_created.get("source_transaction_id")
                    and commit_created.get("source_event_sha256")
                    == proof_created.get("source_event_sha256")
                ):
                    raise RuntimeError(
                        "m110_m006_restore_proof_commit_binding_drift"
                    )
                commit = await self._prepare_issue_apply_lane(
                    created=commit_created,
                    capability_kind="maintenance_dsp_reset_commit_once",
                    approval_id=approval_id,
                    operator_input=operator_input,
                    terminal_emitter=terminal_emitter,
                    timeout_seconds=timeout_seconds,
                )
                stages.append(
                    {
                        "stage": stage,
                        "operation_digest": proof_created[
                            "operation_digest"
                        ],
                        "scope_addresses": list(
                            proof_created["scope_addresses"]
                        ),
                        "proof": proof,
                        "commit": commit,
                    }
                )
            return {
                "verified": True,
                "reason": "m110_m006_exact_preimage_restored",
                "source_transaction_id": M110_M006_SOURCE_TRANSACTION_ID,
                "source_event_sha256": M110_M006_SOURCE_EVENT_SHA256,
                "stages": stages,
                "stage_count": len(stages),
                "m006_restore_complete": True,
                "same_websocket_retained": True,
                "backend_process_retained": self.backend.process.poll()
                is None,
                "token_retained": False,
                "live_authority_granted": False,
                "readiness_closure_allowed": False,
            }
        except BaseException:
            await self.close()
            raise

    @classmethod
    def _validate_m110_soundcheck_final_audit_terminal(
        cls,
        terminal: Mapping[str, Any],
    ) -> dict[str, Any]:
        verified_state_sha256 = cls._require_sha256(
            terminal.get("verified_state_sha256"),
            field_name="verified_state_sha256",
        )
        ledger_sha256 = cls._require_sha256(
            terminal.get("committed_surface_ledger_sha256"),
            field_name="committed_surface_ledger_sha256",
        )
        if not (
            terminal.get("type") == "m110_soundcheck_final_audit_result"
            and type(terminal.get("contract_version")) is int
            and terminal.get("contract_version") == 1
            and terminal.get("status") == "verified"
            and terminal.get("success") is True
            and terminal.get("verified") is True
            and terminal.get("reason")
            == "m110_show_mix_full_reconciliation_verified"
            and isinstance(terminal.get("run_id"), str)
            and bool(terminal.get("run_id"))
            and terminal.get("mix_readiness") == "show_mix_ready"
            and isinstance(terminal.get("audited_address_count"), int)
            and not isinstance(terminal.get("audited_address_count"), bool)
            and terminal.get("audited_address_count") > 0
            and terminal.get("audit_pass_count") == 2
            and terminal.get("operator_listening_required") is True
            and terminal.get("console_mutation_write_attempt_count") == 0
            and terminal.get("live_mixer_writes") is False
            and terminal.get("live_authority_granted") is False
            and terminal.get("readiness_closure_allowed") is False
        ):
            raise RuntimeError(
                "m110_soundcheck_final_audit_terminal_invalid:"
                + cls._bounded_terminal_failure_context(terminal)
            )
        return {
            **dict(terminal),
            "verified_state_sha256": verified_state_sha256,
            "committed_surface_ledger_sha256": ledger_sha256,
        }

    async def run_m110_soundcheck_final_audit(
        self,
        *,
        timeout_seconds: float = 900.0,
    ) -> dict[str, Any]:
        if not self.route_established or self.route_result is None:
            raise RuntimeError("m110_master_route_must_precede_final_audit")
        if self.closed:
            raise RuntimeError("proof_owned_backend_session_closed")
        if not self.control_armed:
            raise RuntimeError("m110_attended_control_must_be_armed")
        if not (
            isinstance(timeout_seconds, (int, float))
            and not isinstance(timeout_seconds, bool)
            and math.isfinite(float(timeout_seconds))
            and 0.0 < float(timeout_seconds) <= 1800.0
        ):
            raise ValueError("m110_final_audit_timeout_invalid")
        await self._send_json(
            {
                "type": "run_m110_soundcheck_final_audit",
                "contract_version": 1,
                "audit_selector": "active_soundcheck_full_reconciliation",
            }
        )
        terminal = await self._receive_terminal(
            "m110_soundcheck_final_audit_result",
            timeout_seconds=float(timeout_seconds),
        )
        return self._validate_m110_soundcheck_final_audit_terminal(terminal)

    async def run_m110_soundcheck_analysis(
        self,
        *,
        timeout_seconds: float = 900.0,
    ) -> dict[str, Any]:
        try:
            return await self._run_m110_soundcheck_analysis(
                timeout_seconds=timeout_seconds,
            )
        except BaseException:
            await self.close()
            raise

    async def _run_m110_soundcheck_analysis(
        self,
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Derive and register one trusted manifest from the retained capture."""

        if not self.control_armed:
            raise RuntimeError("m110_attended_control_must_be_armed")
        await self._send_json(
            {
                "type": "start_m110_soundcheck_analysis",
                "contract_version": 1,
                "analysis_selector": "channels_1_16_from_owned_capture",
            }
        )
        self.analysis_started = True
        terminal = await self._receive_terminal(
            "m110_soundcheck_analysis_result",
            timeout_seconds=float(timeout_seconds),
        )
        if not (
            terminal.get("status") == "complete"
            and terminal.get("success") is True
            and terminal.get("reason")
            == "m110_soundcheck_manifest_registered"
            and isinstance(terminal.get("run_id"), str)
            and bool(terminal.get("run_id"))
            and isinstance(terminal.get("plan_revision"), int)
            and not isinstance(terminal.get("plan_revision"), bool)
            and terminal.get("plan_revision") >= 1
            and terminal.get("shared_capture_retained") is True
            and terminal.get("live_authority_granted") is False
            and terminal.get("readiness_closure_allowed") is False
        ):
            raise RuntimeError(
                "m110_soundcheck_analysis_failed:"
                + self._bounded_terminal_failure_context(terminal)
            )
        await self._send_json({"type": "stop_m110_soundcheck_analysis"})
        stopped = await self._receive_terminal(
            "m110_soundcheck_analysis_stop_result",
            timeout_seconds=15.0,
        )
        if not (
            stopped.get("status") == "stopped"
            and stopped.get("success") is True
            and stopped.get("shared_capture_retained") is True
        ):
            raise RuntimeError(
                "m110_soundcheck_analysis_stop_failed:"
                + self._bounded_terminal_failure_context(stopped)
            )
        self.analysis_started = False
        return {
            "verified": True,
            "reason": "m110_soundcheck_manifest_registered",
            "run_id": terminal["run_id"],
            "plan_revision": terminal["plan_revision"],
            "imported_count": terminal.get("imported_count"),
            "soundcheck_readiness": terminal.get("soundcheck_readiness"),
            "shared_capture_retained": True,
        }

    async def advance_m110_soundcheck_replan(
        self,
        *,
        timeout_seconds: float = 60.0,
    ) -> dict[str, Any]:
        """Request only the next server-selected bounded split revision."""

        await self._send_json(
            {
                "type": "advance_m110_soundcheck_replan",
                "contract_version": 1,
                "replan_selector": "next_bounded_split_revision",
            }
        )
        terminal = await self._receive_terminal(
            "m110_soundcheck_replan_result",
            timeout_seconds=float(timeout_seconds),
        )
        self._assert_no_capability_secret(terminal)
        common_safe = bool(
            terminal.get("server_owned_selection") is True
            and terminal.get("client_targets_accepted") is False
            and terminal.get("console_mutation_write_attempt_count") == 0
            and not isinstance(
                terminal.get("console_mutation_write_attempt_count"),
                bool,
            )
            and terminal.get("live_mixer_writes") is False
            and terminal.get("live_authority_granted") is False
            and terminal.get("readiness_closure_allowed") is False
        )
        if terminal.get("status") == "advanced":
            run_id = terminal.get("run_id")
            from_revision = terminal.get("from_plan_revision")
            to_revision = terminal.get("to_plan_revision")
            safe_alternative_count = terminal.get(
                "safe_alternative_operation_count"
            )
            if not (
                common_safe
                and terminal.get("success") is True
                and terminal.get("reason")
                == "m110_bounded_split_revision_advanced"
                and isinstance(run_id, str)
                and bool(run_id)
                and isinstance(from_revision, int)
                and not isinstance(from_revision, bool)
                and from_revision >= 1
                and isinstance(to_revision, int)
                and not isinstance(to_revision, bool)
                and to_revision == from_revision + 1
                and isinstance(terminal.get("split_operation_count"), int)
                and not isinstance(
                    terminal.get("split_operation_count"),
                    bool,
                )
                and terminal.get("split_operation_count") >= 0
                and isinstance(safe_alternative_count, int)
                and not isinstance(safe_alternative_count, bool)
                and (
                    safe_alternative_count == 0
                    or safe_alternative_count >= 2
                )
                and terminal.get("split_operation_count")
                + safe_alternative_count
                >= 1
                and terminal.get("replan_basis_refreshed") is True
                and terminal.get(
                    "new_operator_approval_required_for_each_child"
                )
                is True
            ):
                raise RuntimeError(
                    "m110_soundcheck_replan_terminal_invalid:"
                    + self._bounded_terminal_failure_context(terminal)
                )
            return {
                "state": "advanced",
                "success": True,
                "run_id": run_id,
                "from_plan_revision": from_revision,
                "to_plan_revision": to_revision,
                "split_operation_count": terminal[
                    "split_operation_count"
                ],
                "safe_alternative_operation_count": safe_alternative_count,
            }
        if terminal.get("status") == "deferred":
            if not (
                common_safe
                and terminal.get("success") is False
                and terminal.get("retryable") is True
                and terminal.get("retry_class")
                == "fresh_soundcheck_analysis_required"
                and terminal.get("fresh_analysis_required") is True
            ):
                raise RuntimeError(
                    "m110_soundcheck_replan_deferred_invalid:"
                    + self._bounded_terminal_failure_context(terminal)
                )
            return {
                "state": "deferred",
                "success": False,
                "reason": terminal.get("reason"),
                "retry_class": terminal.get("retry_class"),
            }
        if terminal.get("status") == "blocked":
            if not (
                common_safe
                and terminal.get("success") is False
                and isinstance(terminal.get("reason"), str)
                and bool(terminal.get("reason"))
            ):
                raise RuntimeError(
                    "m110_soundcheck_replan_blocked_invalid:"
                    + self._bounded_terminal_failure_context(terminal)
                )
            return {
                "state": "blocked",
                "success": False,
                "reason": terminal.get("reason"),
            }
        raise RuntimeError(
            "m110_soundcheck_replan_terminal_invalid:"
            + self._bounded_terminal_failure_context(terminal)
        )

    @classmethod
    def _validate_lane_proposal_terminal(
        cls,
        terminal: Mapping[str, Any],
        *,
        lane: str,
        role: str,
    ) -> dict[str, Any]:
        cls._assert_no_capability_secret(terminal)
        if lane not in {"processor", "soundcheck"} or role not in {
            "proof",
            "commit",
        }:
            raise ValueError("m110_lane_role_invalid")
        expected_type = (
            "m110_processor_enable_proposal_result"
            if lane == "processor"
            else "m110_soundcheck_child_proposal_result"
        )
        if terminal.get("type") != expected_type:
            raise RuntimeError("m110_lane_proposal_terminal_type_invalid")
        if terminal.get("status") == "deferred":
            retry_class = terminal.get("retry_class")
            if not (
                terminal.get("success") is False
                and terminal.get("retryable") is True
                and retry_class
                in {
                    "other_lane_progress_required",
                    "fresh_soundcheck_analysis_required",
                    "fresh_preflight_required",
                }
                and terminal.get("candidate_available") is False
                and terminal.get("live_authority_granted") is False
                and terminal.get("readiness_closure_allowed") is False
                and isinstance(terminal.get("reason"), str)
                and bool(terminal.get("reason"))
            ):
                raise RuntimeError(
                    "m110_lane_deferred_terminal_invalid:"
                    + cls._bounded_terminal_failure_context(terminal)
                )
            return {
                "state": "deferred",
                "success": False,
                "reason": terminal["reason"],
                "retry_class": retry_class,
            }
        if terminal.get("status") == "blocked":
            if not (
                terminal.get("success") is False
                and isinstance(terminal.get("reason"), str)
                and bool(terminal.get("reason"))
                and terminal.get("live_authority_granted") is False
                and terminal.get("readiness_closure_allowed") is False
            ):
                raise RuntimeError(
                    "m110_lane_blocked_terminal_invalid:"
                    + cls._bounded_terminal_failure_context(terminal)
                )
            return {
                "state": "blocked",
                "success": False,
                "reason": terminal["reason"],
                "blockers": list(terminal.get("blockers") or []),
            }
        if terminal.get("status") in {"complete", "incomplete"}:
            status = str(terminal.get("status"))
            residuals = terminal.get("residuals") or []
            reason = str(terminal.get("reason") or "")
            replan_available = terminal.get("replan_available", False)
            replan_split_count = terminal.get(
                "replan_split_operation_count",
                0,
            )
            replan_safe_count = terminal.get(
                "replan_safe_alternative_operation_count",
                0,
            )
            replan_contract_valid = bool(
                lane == "processor"
                or (
                    isinstance(replan_available, bool)
                    and isinstance(replan_split_count, int)
                    and not isinstance(replan_split_count, bool)
                    and replan_split_count >= 0
                    and isinstance(replan_safe_count, int)
                    and not isinstance(replan_safe_count, bool)
                    and replan_safe_count >= 0
                    and (
                        (
                            not replan_available
                            and terminal.get("replan_from_plan_revision")
                            is None
                            and terminal.get("replan_to_plan_revision") is None
                            and replan_split_count == 0
                            and replan_safe_count == 0
                        )
                        or (
                            replan_available
                            and isinstance(
                                terminal.get("replan_from_plan_revision"),
                                int,
                            )
                            and not isinstance(
                                terminal.get("replan_from_plan_revision"),
                                bool,
                            )
                            and isinstance(
                                terminal.get("replan_to_plan_revision"),
                                int,
                            )
                            and not isinstance(
                                terminal.get("replan_to_plan_revision"),
                                bool,
                            )
                            and terminal.get("replan_to_plan_revision")
                            == terminal.get("replan_from_plan_revision") + 1
                            and (
                                replan_safe_count == 0
                                or replan_safe_count >= 2
                            )
                            and replan_split_count + replan_safe_count >= 1
                        )
                    )
                )
            )
            expected_complete_reason = (
                "m110_no_unresolved_processor_candidate"
                if lane == "processor"
                else "m110_no_unresolved_soundcheck_child"
            )
            allowed_incomplete_reasons = (
                {"m110_processor_has_unresolved_local_residuals"}
                if lane == "processor"
                else {
                    "m110_soundcheck_has_unresolved_local_residuals",
                    "m110_soundcheck_final_readiness_not_verified",
                }
            )
            if not (
                terminal.get("success") is (status == "complete")
                and terminal.get("candidate_available") is False
                and terminal.get("live_authority_granted") is False
                and terminal.get("readiness_closure_allowed") is False
                and isinstance(terminal.get("run_id"), str)
                and bool(terminal.get("run_id"))
                and isinstance(residuals, list)
                and all(isinstance(item, Mapping) for item in residuals)
                and replan_contract_valid
                and (
                    reason == expected_complete_reason
                    if status == "complete"
                    else reason in allowed_incomplete_reasons
                )
                and (status == "complete" or bool(residuals) or lane == "soundcheck")
            ):
                raise RuntimeError(
                    "m110_lane_completion_terminal_invalid:"
                    + cls._bounded_terminal_failure_context(terminal)
                )
            return {
                "state": status,
                "success": status == "complete",
                "reason": reason,
                "run_id": terminal.get("run_id"),
                "residuals": list(residuals),
                "soundcheck_readiness": terminal.get(
                    "soundcheck_readiness"
                ),
                "replan_available": (
                    replan_available if lane == "soundcheck" else False
                ),
                "replan_from_plan_revision": terminal.get(
                    "replan_from_plan_revision"
                ),
                "replan_to_plan_revision": terminal.get(
                    "replan_to_plan_revision"
                ),
                "replan_split_operation_count": replan_split_count,
                "replan_safe_alternative_operation_count": (
                    replan_safe_count
                ),
            }
        required_kind = (
            "restore_only_proof"
            if role == "proof"
            else "processor_enable_commit_once"
            if lane == "processor"
            else "soundcheck_child_commit_once"
        )
        bundle_kind = terminal.get("bundle_kind")
        operation_count = terminal.get("operation_count")
        valid_bundle = bool(
            bundle_kind == "processor_enable_bundle"
            and operation_count == 1
            if lane == "processor"
            else bundle_kind in M110_SOUNDCHECK_BUNDLE_KINDS
            and isinstance(operation_count, int)
            and not isinstance(operation_count, bool)
            and (
                operation_count == 1
                if bundle_kind == M110_SOUNDCHECK_SINGLE_BUNDLE_KIND
                else 2 <= operation_count <= 64
            )
        )
        proposal_id = terminal.get("proposal_id")
        review_digest = cls._require_sha256(
            terminal.get("review_content_digest"),
            field_name="review_content_digest",
        )
        operation_digest = cls._require_sha256(
            terminal.get("operation_digest"),
            field_name="operation_digest",
        )
        scope_addresses = terminal.get("scope_addresses")
        review_operations = terminal.get("review_operations")
        review_operations_valid = bool(
            isinstance(review_operations, list)
            and len(review_operations) == operation_count
            and all(
                isinstance(item, Mapping)
                and isinstance(item.get("operation_id"), str)
                and bool(item.get("operation_id"))
                and isinstance(item.get("family_id"), str)
                and bool(item.get("family_id"))
                and isinstance(item.get("write_kind"), str)
                and bool(item.get("write_kind"))
                and "target_value" in item
                for item in review_operations
            )
        )
        review_operation_ids = (
            [str(item.get("operation_id") or "") for item in review_operations]
            if isinstance(review_operations, list)
            else []
        )
        raw_source_operation_ids = terminal.get("source_operation_ids")
        source_operation_ids = (
            [str(value or "") for value in raw_source_operation_ids]
            if isinstance(raw_source_operation_ids, list)
            else []
        )
        source_operation_ids_valid = bool(
            source_operation_ids
            and all(source_operation_ids)
            and len(source_operation_ids) == len(set(source_operation_ids))
        )
        source_provenance_valid = bool(
            source_operation_ids_valid
            and (
                (
                    lane == "soundcheck"
                    and source_operation_ids == review_operation_ids
                    and terminal.get("source_operation_id")
                    == source_operation_ids[0]
                )
                or (
                    lane == "processor"
                    and isinstance(terminal.get("candidate_id"), str)
                    and re.fullmatch(
                        r"m110-processor-(?:hpf|eq|gate|compressor)-ch"
                        r"(?:[1-9]|1[0-6])-[0-9a-f]{16}",
                        terminal.get("candidate_id"),
                    )
                    is not None
                )
            )
        )
        expected_proposal_id = (
            rf"m110-processor-{role}-[0-9a-f]{{32}}"
            if lane == "processor"
            else rf"m110-soundcheck-child-{role}-[0-9a-f]{{32}}"
        )
        if not (
            terminal.get("status") == "created"
            and terminal.get("success") is True
            and isinstance(proposal_id, str)
            and re.fullmatch(expected_proposal_id, proposal_id) is not None
            and type(terminal.get("review_contract_version")) is int
            and terminal.get("review_contract_version") == 2
            and terminal.get("required_capability_kind") == required_kind
            and valid_bundle
            and isinstance(scope_addresses, list)
            and bool(scope_addresses)
            and len(scope_addresses) == len(set(scope_addresses))
            and all(
                isinstance(address, str) and address.startswith("/")
                for address in scope_addresses
            )
            and review_operations_valid
            and source_provenance_valid
            and terminal.get("matching_restored_proof_receipt")
            is (role == "commit")
            and terminal.get("token_issued") is False
            and terminal.get("live_authority_granted") is False
            and terminal.get("readiness_closure_allowed") is False
        ):
            raise RuntimeError(
                "m110_lane_proposal_invalid:"
                + cls._bounded_terminal_failure_context(terminal)
            )
        return {
            "state": "created",
            "lane": lane,
            "role": role,
            "proposal_id": proposal_id,
            "review_contract_version": 2,
            "review_content_digest": review_digest,
            "operation_digest": operation_digest,
            "review_preimage_sha256": str(
                terminal.get("review_preimage_sha256") or ""
            ),
            "required_capability_kind": required_kind,
            "bundle_kind": bundle_kind,
            "operation_count": operation_count,
            "scope_addresses": list(scope_addresses),
            "review_operations": [
                dict(item) for item in review_operations
            ],
            "candidate_id": terminal.get("candidate_id"),
            "source_operation_ids": source_operation_ids,
        }

    async def _create_lane_proposal(
        self,
        *,
        lane: str,
        role: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        request_type = (
            f"create_m110_processor_enable_{role}_proposal"
            if lane == "processor"
            else f"create_m110_soundcheck_child_{role}_proposal"
        )
        selector = (
            "next_required_soundcheck_processor"
            if lane == "processor"
            else "next_eligible_soundcheck_child"
        )
        expected_type = (
            "m110_processor_enable_proposal_result"
            if lane == "processor"
            else "m110_soundcheck_child_proposal_result"
        )
        await self._send_json(
            {
                "type": request_type,
                "contract_version": 1,
                "bundle_selector": selector,
            }
        )
        terminal = await self._receive_terminal(
            expected_type,
            timeout_seconds=float(timeout_seconds),
        )
        return self._validate_lane_proposal_terminal(
            terminal,
            lane=lane,
            role=role,
        )

    @classmethod
    def _validate_lane_apply_terminal(
        cls,
        terminal: Mapping[str, Any],
        *,
        created: Mapping[str, Any],
        capability_kind: str,
    ) -> None:
        cls._assert_no_capability_secret(terminal)
        proof_run = capability_kind == "restore_only_proof"
        expected_status = "proof_passed" if proof_run else "applied"
        expected_wal = "proof_restored" if proof_run else "committed"
        proposal = terminal.get("proposal")
        result_digest = cls._require_sha256(
            terminal.get("result_digest"),
            field_name="lane_result_digest",
        )
        proof_receipt_id = terminal.get("proof_receipt_id")
        expected_operation_id_list = [
            str(item.get("operation_id"))
            for item in created.get("review_operations") or []
            if isinstance(item, Mapping)
            and isinstance(item.get("operation_id"), str)
            and item.get("operation_id")
        ]
        expected_operation_ids = set(expected_operation_id_list)
        applied_operation_ids = terminal.get("applied_operation_ids")
        rollback_operation_ids = terminal.get("rollback_operation_ids")
        already_satisfied_operation_ids = terminal.get(
            "already_satisfied_operation_ids"
        )
        operation_identifier_lists_valid = bool(
            expected_operation_ids
            and len(expected_operation_id_list)
            == len(created.get("review_operations") or [])
            and len(expected_operation_id_list)
            == len(expected_operation_ids)
            and type(applied_operation_ids) is list
            and type(rollback_operation_ids) is list
            and type(already_satisfied_operation_ids) is list
            and all(
                type(operation_id) is str and bool(operation_id)
                for operation_id in (
                    *applied_operation_ids,
                    *rollback_operation_ids,
                    *already_satisfied_operation_ids,
                )
            )
            and len(applied_operation_ids)
            == len(set(applied_operation_ids))
            and len(rollback_operation_ids)
            == len(set(rollback_operation_ids))
            and len(already_satisfied_operation_ids)
            == len(set(already_satisfied_operation_ids))
            and set(applied_operation_ids).isdisjoint(
                already_satisfied_operation_ids
            )
        )
        exact_operation_coverage = bool(
            operation_identifier_lists_valid
            and set(applied_operation_ids).union(
                already_satisfied_operation_ids
            )
            == expected_operation_ids
            and (
                set(rollback_operation_ids) == set(applied_operation_ids)
                if proof_run
                else rollback_operation_ids == []
            )
        )
        all_targets_already_satisfied = bool(
            exact_operation_coverage
            and applied_operation_ids == []
            and rollback_operation_ids == []
            and set(already_satisfied_operation_ids)
            == expected_operation_ids
        )
        expected_rollback = bool(proof_run and not all_targets_already_satisfied)
        expected_rollback_status = (
            "not_required_already_satisfied"
            if proof_run and all_targets_already_satisfied
            else "verified"
            if proof_run
            else "not_attempted"
        )
        if not (
            terminal.get("type") == "operator_proposal_apply_result"
            and exact_operation_coverage
            and terminal.get("contract_version") == 2
            and terminal.get("status") == expected_status
            and terminal.get("success") is True
            and terminal.get("proposal_id") == created.get("proposal_id")
            and terminal.get("proposal_status") == expected_status
            and terminal.get("apply_status") == expected_status
            and terminal.get("accepted") is (not proof_run)
            and terminal.get("owner_review_required") is proof_run
            and terminal.get("live_authority_granted") is False
            and terminal.get("readiness_closure_allowed") is False
            and terminal.get("proof_run") is proof_run
            and terminal.get("restore_after_verify") is proof_run
            and terminal.get("capability_kind") == capability_kind
            and terminal.get("operation_digest")
            == created.get("operation_digest")
            and terminal.get("bundle_kind") == created.get("bundle_kind")
            and terminal.get("review_preimage_sha256")
            == (created.get("review_preimage_sha256") or None)
            and terminal.get("proof_target_verified") is proof_run
            and terminal.get("no_console_send_proven")
            is all_targets_already_satisfied
            and terminal.get("physical_verified") is True
            and terminal.get("audio_recheck_passed") is True
            and terminal.get("terminal_after_console_send")
            is (not all_targets_already_satisfied)
            and terminal.get("wal_terminal") == expected_wal
            and terminal.get("write_state_unknown") is False
            and terminal.get("rollback_state_unknown") is False
            and terminal.get("manual_review_required") is False
            and terminal.get("emergency_stop_activated") is False
            and terminal.get("emergency_stop_required") is False
            and terminal.get("physical_state_restored") is proof_run
            and terminal.get("rollback_attempted") is expected_rollback
            and terminal.get("rollback_succeeded") is expected_rollback
            and terminal.get("rollback_status") == expected_rollback_status
            and (
                proof_run
                or terminal.get("parent_commit_recorded") is True
            )
            and (
                proof_receipt_id is None
                if proof_run
                else isinstance(proof_receipt_id, str)
                and re.fullmatch(r"m110proof_v1_[0-9a-f]{64}", proof_receipt_id)
                is not None
            )
            and terminal.get("result_detail_available") is True
            and terminal.get("result_evidence_pinned") is False
            and isinstance(proposal, Mapping)
            and proposal.get("id") == created.get("proposal_id")
            and proposal.get("status") == expected_status
            and proposal.get("review_content_digest")
            == created.get("review_content_digest")
            and proposal.get("result_digest") == result_digest
        ):
            raise RuntimeError(
                "m110_lane_apply_terminal_invalid:"
                + cls._bounded_terminal_failure_context(terminal)
            )

    async def _prepare_issue_apply_lane(
        self,
        *,
        created: Mapping[str, Any],
        capability_kind: str,
        approval_id: str,
        operator_input: OperatorInputCallback,
        terminal_emitter: TerminalEmitterCallback,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        proposal_id = str(created["proposal_id"])
        await self._send_json(
            {
                "type": "prepare_m110_restore_only_proof",
                "contract_version": 1,
                "proposal_id": proposal_id,
                "bundle_selector": "reviewed_supervised_transaction",
            }
        )
        prepared = await self._receive_terminal(
            "m110_restore_proof_prepare_result",
            timeout_seconds=min(float(timeout_seconds), 30.0),
        )
        expected_reason = {
            "processor": "m110_processor_bundle_current",
            "soundcheck": "m110_soundcheck_child_bundle_current",
            "m006_restore": "m110_m006_restore_bundle_current",
        }.get(created.get("lane"))
        if not (
            prepared.get("status") == "prepared"
            and prepared.get("success") is True
            and prepared.get("reason") == expected_reason
            and prepared.get("proposal_id") == proposal_id
            and prepared.get("review_contract_version") == 2
            and prepared.get("review_content_digest")
            == created.get("review_content_digest")
            and prepared.get("operation_digest")
            == created.get("operation_digest")
            and prepared.get("bundle_kind") == created.get("bundle_kind")
            and prepared.get("required_capability_kind") == capability_kind
            and prepared.get("operation_count")
            == created.get("operation_count")
            and prepared.get("scope_addresses")
            == created.get("scope_addresses")
            and prepared.get("token_issued") is False
            and prepared.get("live_authority_granted") is False
        ):
            raise RuntimeError(
                "m110_lane_prepare_failed:"
                + self._bounded_terminal_failure_context(prepared)
            )
        operator_checkpoint = await self._require_bundle_operator_review(
            created=created,
            capability_kind=capability_kind,
            approval_id=approval_id,
            operator_input=operator_input,
            terminal_emitter=terminal_emitter,
        )
        issue_request = build_m110_capability_request(
            capability_kind=capability_kind,
            proposal_id=proposal_id,
            review_contract_version=2,
            review_content_digest=str(created["review_content_digest"]),
            operation_digest=str(created["operation_digest"]),
            approval_id=approval_id,
            request_nonce=secrets.token_hex(16),
            secret=bytes(self.backend.secret),
        )
        await self._send_json(issue_request)
        issued = await self._receive_terminal(
            "supervised_capability_result",
            timeout_seconds=min(float(timeout_seconds), 30.0),
        )
        raw_token = issued.pop("supervised_capability_token", None)
        idempotency_key = issued.get("idempotency_key")
        latest_start_epoch = issued.get("latest_start_epoch")
        idempotency_pattern = {
            "restore_only_proof": r"m110-proof:[0-9a-f]{32}",
            "maintenance_dsp_reset_commit_once": (
                r"m110-reset-commit:[0-9a-f]{32}"
            ),
            "processor_enable_commit_once": r"m110-processor-commit:[0-9a-f]{32}",
            "soundcheck_child_commit_once": r"m110-soundcheck-commit:[0-9a-f]{32}",
        }.get(capability_kind)
        if not (
            issued.get("status") == "issued"
            and issued.get("success") is True
            and issued.get("reason") == "m110_capability_issued"
            and issued.get("capability_kind") == capability_kind
            and issued.get("proposal_id") == proposal_id
            and issued.get("review_content_digest")
            == created.get("review_content_digest")
            and issued.get("operation_digest")
            == created.get("operation_digest")
            and issued.get("approval_id") == approval_id
            and isinstance(idempotency_key, str)
            and isinstance(idempotency_pattern, str)
            and re.fullmatch(idempotency_pattern, idempotency_key) is not None
            and isinstance(latest_start_epoch, (int, float))
            and not isinstance(latest_start_epoch, bool)
            and math.isfinite(float(latest_start_epoch))
            and time.time() < float(latest_start_epoch) <= time.time() + 31.0
            and isinstance(raw_token, str)
            and re.fullmatch(r"m110cap_v1_[0-9a-f]{64}", raw_token) is not None
            and issued.get("single_use") is True
            and issued.get("stores_plaintext_token") is False
            and issued.get("live_authority_granted") is False
        ):
            raw_token = None
            raise RuntimeError(
                "m110_lane_capability_issue_failed:"
                + self._bounded_terminal_failure_context(issued)
            )
        token = bytearray(raw_token.encode("ascii"))
        raw_token = None
        apply_request = {
            "type": "apply_operator_proposal",
            "proposal_id": proposal_id,
            "approved": True,
            "approval_id": approval_id,
            "idempotency_key": idempotency_key,
            "expected_review_contract_version": 2,
            "expected_review_content_digest": created[
                "review_content_digest"
            ],
            "expected_operation_digest": created["operation_digest"],
            "supervised_capability_token": token.decode("ascii"),
        }
        try:
            await self._send_json(apply_request)
        finally:
            apply_request.pop("supervised_capability_token", None)
            for index in range(len(token)):
                token[index] = 0
        terminal = await self._receive_terminal(
            "operator_proposal_apply_result",
            timeout_seconds=float(timeout_seconds),
        )
        self._validate_lane_apply_terminal(
            terminal,
            created=created,
            capability_kind=capability_kind,
        )
        await _emit_attended_terminal(
            terminal_emitter,
            {
                "event": "m110_capability_terminal",
                "proposal_id": proposal_id,
                "capability_kind": capability_kind,
                "operation_digest": created["operation_digest"],
                "terminal": dict(terminal),
            },
        )
        return {
            "verified": True,
            "proposal_id": proposal_id,
            "capability_kind": capability_kind,
            "operation_digest": created["operation_digest"],
            "bundle_kind": created["bundle_kind"],
            "terminal": terminal,
            "operator_checkpoint": operator_checkpoint,
            "token_retained": False,
        }

    async def execute_m110_lane_step(
        self,
        *,
        lane: str,
        approval_id: str,
        operator_input: OperatorInputCallback,
        terminal_emitter: TerminalEmitterCallback,
        timeout_seconds: float = 180.0,
    ) -> dict[str, Any]:
        """Execute at most one exact proof/commit pair for one lane."""

        proof_created = await self._create_lane_proposal(
            lane=lane,
            role="proof",
            timeout_seconds=timeout_seconds,
        )
        if proof_created.get("state") != "created":
            return dict(proof_created)
        digest = str(proof_created["operation_digest"])
        proof = await self._prepare_issue_apply_lane(
            created=proof_created,
            capability_kind="restore_only_proof",
            approval_id=approval_id,
            operator_input=operator_input,
            terminal_emitter=terminal_emitter,
            timeout_seconds=timeout_seconds,
        )
        commit_created = await self._create_lane_proposal(
            lane=lane,
            role="commit",
            timeout_seconds=timeout_seconds,
        )
        if not (
            commit_created.get("state") == "created"
            and commit_created.get("operation_digest") == digest
            and commit_created.get("bundle_kind")
            == proof_created.get("bundle_kind")
            and commit_created.get("scope_addresses")
            == proof_created.get("scope_addresses")
            and commit_created.get("source_operation_ids")
            == proof_created.get("source_operation_ids")
            and commit_created.get("review_operations")
            == proof_created.get("review_operations")
            and (
                lane != "processor"
                or commit_created.get("candidate_id")
                == proof_created.get("candidate_id")
            )
        ):
            raise RuntimeError("m110_lane_proof_commit_binding_drift")
        commit_kind = (
            "processor_enable_commit_once"
            if lane == "processor"
            else "soundcheck_child_commit_once"
        )
        commit = await self._prepare_issue_apply_lane(
            created=commit_created,
            capability_kind=commit_kind,
            approval_id=approval_id,
            operator_input=operator_input,
            terminal_emitter=terminal_emitter,
            timeout_seconds=timeout_seconds,
        )
        return {
            "state": "committed",
            "success": True,
            "lane": lane,
            "operation_digest": digest,
            "source_operation_ids": proof_created.get(
                "source_operation_ids"
            ),
            "proof": proof,
            "commit": commit,
        }

    async def execute_m110_lane_until_terminal(
        self,
        *,
        lane: str,
        approval_id: str,
        operator_input: OperatorInputCallback,
        terminal_emitter: TerminalEmitterCallback,
        timeout_seconds: float = 180.0,
        maximum_children: int = 256,
    ) -> dict[str, Any]:
        """Proof and permanently commit each server-selected lane child."""

        commits: list[dict[str, Any]] = []
        seen_digests: set[str] = set()
        for _index in range(maximum_children + 1):
            step = await self.execute_m110_lane_step(
                lane=lane,
                approval_id=approval_id,
                operator_input=operator_input,
                terminal_emitter=terminal_emitter,
                timeout_seconds=timeout_seconds,
            )
            if step.get("state") in {
                "complete",
                "incomplete",
                "deferred",
                "blocked",
            }:
                return {
                    "verified": bool(
                        step.get("state") == "complete"
                        and step.get("success") is True
                    ),
                    "status": step.get("state"),
                    "reason": step.get("reason"),
                    "retry_class": step.get("retry_class"),
                    "run_id": step.get("run_id"),
                    "residuals": step.get("residuals", []),
                    "soundcheck_readiness": step.get(
                        "soundcheck_readiness"
                    ),
                    "commit_count": len(commits),
                    "commits": commits,
                }
            if step.get("state") != "committed":
                raise RuntimeError("m110_lane_step_terminal_invalid")
            digest = str(step["operation_digest"])
            if digest in seen_digests:
                raise RuntimeError("m110_lane_candidate_repeated_after_commit")
            seen_digests.add(digest)
            commits.append(step)
        raise RuntimeError("m110_lane_child_limit_exceeded")

    async def execute_m110_interleaved_soundcheck(
        self,
        *,
        approval_id: str,
        operator_input: OperatorInputCallback,
        terminal_emitter: TerminalEmitterCallback,
        timeout_seconds: float = 180.0,
        maximum_commits: int = 512,
        maximum_reanalyses: int = 16,
        maximum_replans: int = 64,
    ) -> dict[str, Any]:
        """Advance processor and musical lanes without cross-stage deadlock."""

        commits: dict[str, list[dict[str, Any]]] = {
            "processor": [],
            "soundcheck": [],
        }
        analyses: list[dict[str, Any]] = []
        replans: list[dict[str, Any]] = []
        seen_replan_revisions: set[tuple[str, int, int]] = set()
        last_replan_revision_by_run: dict[str, int] = {}
        seen_digests: set[str] = set()
        previous_no_progress_signature: tuple[Any, ...] | None = None
        for _round in range(
            maximum_commits + maximum_reanalyses + maximum_replans + 2
        ):
            states: dict[str, dict[str, Any]] = {}
            made_progress = False
            for lane in ("processor", "soundcheck"):
                step = await self.execute_m110_lane_step(
                    lane=lane,
                    approval_id=approval_id,
                    operator_input=operator_input,
                    terminal_emitter=terminal_emitter,
                    timeout_seconds=timeout_seconds,
                )
                states[lane] = step
                if step.get("state") == "blocked":
                    raise RuntimeError(
                        f"m110_{lane}_lane_blocked:"
                        + str(step.get("reason") or "unknown")
                    )
                if step.get("state") != "committed":
                    continue
                digest = str(step.get("operation_digest") or "")
                if not digest or digest in seen_digests:
                    raise RuntimeError(
                        "m110_interleaved_candidate_repeated_after_commit"
                    )
                seen_digests.add(digest)
                commits[lane].append(step)
                made_progress = True
                if sum(len(items) for items in commits.values()) > maximum_commits:
                    raise RuntimeError("m110_interleaved_commit_limit_exceeded")
            if made_progress:
                previous_no_progress_signature = None
                continue

            retry_classes = {
                str(state.get("retry_class") or "")
                for state in states.values()
                if state.get("state") == "deferred"
            }
            if "fresh_soundcheck_analysis_required" in retry_classes:
                if len(analyses) >= maximum_reanalyses:
                    raise RuntimeError("m110_soundcheck_reanalysis_limit_exceeded")
                analyses.append(
                    await self.run_m110_soundcheck_analysis(
                        timeout_seconds=timeout_seconds
                    )
                )
                previous_no_progress_signature = None
                continue
            if "fresh_preflight_required" in retry_classes:
                raise RuntimeError("m110_fresh_preflight_made_no_progress")

            soundcheck_state = states.get("soundcheck", {})
            if (
                soundcheck_state.get("state") == "incomplete"
                and soundcheck_state.get("replan_available") is True
            ):
                if len(replans) >= maximum_replans:
                    raise RuntimeError(
                        "m110_soundcheck_replan_limit_exceeded"
                    )
                replan = await self.advance_m110_soundcheck_replan(
                    timeout_seconds=min(float(timeout_seconds), 60.0)
                )
                if replan.get("state") == "advanced":
                    replan_run_id = str(replan.get("run_id") or "")
                    soundcheck_run_id = str(
                        soundcheck_state.get("run_id") or ""
                    )
                    if (
                        not replan_run_id
                        or not soundcheck_run_id
                        or replan_run_id != soundcheck_run_id
                    ):
                        raise RuntimeError(
                            "m110_soundcheck_replan_run_binding_mismatch"
                        )
                    expected_replan_binding = {
                        "from_plan_revision": soundcheck_state.get(
                            "replan_from_plan_revision"
                        ),
                        "to_plan_revision": soundcheck_state.get(
                            "replan_to_plan_revision"
                        ),
                    }
                    actual_replan_binding = {
                        key: replan.get(key)
                        for key in expected_replan_binding
                    }
                    if actual_replan_binding != expected_replan_binding:
                        raise RuntimeError(
                            "m110_soundcheck_replan_hint_binding_mismatch"
                        )
                    from_revision = int(replan["from_plan_revision"])
                    to_revision = int(replan["to_plan_revision"])
                    revision_pair = (
                        replan_run_id,
                        from_revision,
                        to_revision,
                    )
                    previous_to_revision = last_replan_revision_by_run.get(
                        replan_run_id
                    )
                    if (
                        previous_to_revision is not None
                        and from_revision != previous_to_revision
                    ):
                        raise RuntimeError(
                            "m110_soundcheck_replan_revision_discontinuous"
                        )
                    if revision_pair in seen_replan_revisions:
                        raise RuntimeError(
                            "m110_soundcheck_replan_revision_repeated"
                        )
                    seen_replan_revisions.add(revision_pair)
                    last_replan_revision_by_run[replan_run_id] = to_revision
                    replans.append(replan)
                    previous_no_progress_signature = None
                    continue
                if (
                    replan.get("state") == "deferred"
                    and replan.get("retry_class")
                    == "fresh_soundcheck_analysis_required"
                ):
                    if len(analyses) >= maximum_reanalyses:
                        raise RuntimeError(
                            "m110_soundcheck_reanalysis_limit_exceeded"
                        )
                    analyses.append(
                        await self.run_m110_soundcheck_analysis(
                            timeout_seconds=timeout_seconds
                        )
                    )
                    previous_no_progress_signature = None
                    continue
                raise RuntimeError(
                    "m110_soundcheck_replan_blocked:"
                    + str(replan.get("reason") or "unknown")
                )

            if all(
                state.get("state") in {"complete", "incomplete"}
                for state in states.values()
            ):
                soundcheck = states["soundcheck"]
                readiness = soundcheck.get("soundcheck_readiness")
                verified = bool(
                    all(state.get("state") == "complete" for state in states.values())
                    and isinstance(readiness, Mapping)
                    and readiness.get("mix_readiness") == "show_mix_ready"
                )
                return {
                    "verified": verified,
                    "status": "complete" if verified else "incomplete",
                    "reason": (
                        "m110_interleaved_show_mix_ready"
                        if verified
                        else "m110_interleaved_soundcheck_incomplete"
                    ),
                    "lane_states": states,
                    "processor": states["processor"],
                    "soundcheck": soundcheck,
                    "soundcheck_readiness": readiness,
                    "commit_count": sum(
                        len(items) for items in commits.values()
                    ),
                    "commits": commits,
                    "reanalyses": analyses,
                    "replans": replans,
                }

            signature = tuple(
                (
                    lane,
                    states[lane].get("state"),
                    states[lane].get("reason"),
                    states[lane].get("retry_class"),
                )
                for lane in ("processor", "soundcheck")
            )
            if signature == previous_no_progress_signature:
                raise RuntimeError("m110_interleaved_scheduler_deadlock")
            previous_no_progress_signature = signature
        raise RuntimeError("m110_interleaved_scheduler_limit_exceeded")

    @classmethod
    def _validate_restore_only_proof_terminal(
        cls,
        terminal: Mapping[str, Any],
        *,
        proposal_id: str,
        review_digest: str,
        operation_digest: str,
    ) -> None:
        cls._assert_no_capability_secret(terminal)
        terminal_result_digest = cls._require_sha256(
            terminal.get("result_digest"),
            field_name="proof_result_digest",
        )
        terminal_proposal = terminal.get("proposal")
        if not (
            type(terminal.get("contract_version")) is int
            and terminal.get("contract_version") == 2
            and terminal.get("status") == "proof_passed"
            and terminal.get("success") is True
            and terminal.get("proposal_id") == proposal_id
            and terminal.get("proposal_status") == "proof_passed"
            and terminal.get("apply_status") == "proof_passed"
            and terminal.get("accepted") is False
            and terminal.get("owner_review_required") is True
            and terminal.get("live_authority_granted") is False
            and terminal.get("readiness_closure_allowed") is False
            and terminal.get("proof_run") is True
            and terminal.get("restore_after_verify") is True
            and terminal.get("capability_kind") == "restore_only_proof"
            and terminal.get("operation_digest") == operation_digest
            and terminal.get("proof_target_verified") is True
            and terminal.get("no_console_send_proven") is False
            and terminal.get("physical_verified") is True
            and terminal.get("physical_state_restored") is True
            and terminal.get("terminal_after_console_send") is True
            and terminal.get("write_state_unknown") is False
            and terminal.get("rollback_attempted") is True
            and terminal.get("rollback_succeeded") is True
            and terminal.get("rollback_status") == "verified"
            and terminal.get("rollback_state_unknown") is False
            and terminal.get("manual_review_required") is False
            and terminal.get("emergency_stop_activated") is False
            and terminal.get("emergency_stop_required") is False
            and terminal.get("wal_terminal") == "proof_restored"
            and terminal.get("audio_recheck_passed") is True
            and terminal.get("result_detail_available") is True
            and terminal.get("result_evidence_pinned") is False
            and isinstance(terminal_proposal, Mapping)
            and terminal_proposal.get("id") == proposal_id
            and terminal_proposal.get("status") == "proof_passed"
            and terminal_proposal.get("review_contract_version") == 2
            and terminal_proposal.get("review_content_digest")
            == review_digest
            and terminal_proposal.get("result_digest")
            == terminal_result_digest
            and terminal_proposal.get("result_detail_available") is True
            and terminal_proposal.get("result_evidence_pinned") is False
        ):
            raise RuntimeError(
                "m110_restore_only_proof_terminal_invalid:"
                + cls._bounded_terminal_failure_context(terminal)
            )

    async def prove_pending_restore_only_proposal(
        self,
        *,
        proposal_id: str,
        approval_id: str,
        timeout_seconds: float = 45.0,
    ) -> dict[str, Any]:
        """Bind and consume one restore-only capability on this armed session.

        The proposal must already be a server-reviewed, one-operation
        supervised transaction created by the later reset/soundcheck stage.
        Any protocol failure tears down the whole proof-owned authority.
        """

        try:
            return await self._prove_pending_restore_only_proposal(
                proposal_id=proposal_id,
                approval_id=approval_id,
                timeout_seconds=timeout_seconds,
            )
        except BaseException:
            await self.close()
            raise

    async def arm_m110_attended_control(
        self,
        *,
        approval_id: str,
        timeout_seconds: float = 20.0,
    ) -> dict[str, Any]:
        """Arm the one-shot capability control without widening normal mode."""

        try:
            return await self._arm_m110_attended_control(
                approval_id=approval_id,
                timeout_seconds=timeout_seconds,
            )
        except BaseException:
            await self.close()
            raise

    async def _arm_m110_attended_control(
        self,
        *,
        approval_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        if not self.route_established or self.route_result is None:
            raise RuntimeError("m110_master_route_must_precede_capability")
        if self.control_armed:
            raise RuntimeError("m110_attended_control_already_armed")
        if not (
            isinstance(approval_id, str)
            and 1 <= len(approval_id.encode("utf-8")) <= 128
            and all(0x21 <= ord(character) <= 0x7E for character in approval_id)
        ):
            raise ValueError("m110_approval_id_invalid")
        if not (
            isinstance(timeout_seconds, (int, float))
            and not isinstance(timeout_seconds, bool)
            and math.isfinite(float(timeout_seconds))
            and 0.0 < float(timeout_seconds) <= 120.0
        ):
            raise ValueError("m110_capability_timeout_invalid")
        arm_request = build_m110_attended_control_request(
            approval_id=approval_id,
            request_nonce=secrets.token_hex(16),
            secret=bytes(self.backend.secret),
        )
        await self._send_json(arm_request)
        armed = await self._receive_terminal(
            "m110_attended_control_result",
            timeout_seconds=float(timeout_seconds),
        )
        route_proof_id = self.route_result.get("proof_id")
        if not (
            armed.get("status") == "armed"
            and armed.get("success") is True
            and armed.get("reason")
            == "m110_supervised_capability_only_armed"
            and armed.get("route_proof_id") == route_proof_id
            and armed.get("supervised_capability_only") is True
            and armed.get("normal_operator_mode_unchanged") is True
            and armed.get("normal_live_allowlist_changed") is False
            and armed.get("live_authority_granted") is False
            and armed.get("readiness_closure_allowed") is False
        ):
            raise RuntimeError(
                "m110_attended_control_arm_failed:"
                + self._bounded_terminal_failure_context(armed)
            )
        self.control_armed = True
        self.control_approval_id = approval_id
        return {
            "verified": True,
            "reason": "m110_supervised_capability_only_armed",
            "route_proof_id": route_proof_id,
            "normal_operator_mode_unchanged": True,
            "normal_live_allowlist_changed": False,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
        }

    async def _prove_pending_restore_only_proposal(
        self,
        *,
        proposal_id: str,
        approval_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        if not self.route_established or self.route_result is None:
            raise RuntimeError("m110_master_route_must_precede_capability")
        if self.closed:
            raise RuntimeError("proof_owned_backend_session_closed")
        if not self.control_armed:
            raise RuntimeError("m110_attended_control_must_be_armed")
        if approval_id != self.control_approval_id:
            raise RuntimeError("m110_attended_control_approval_mismatch")
        for field_name, value in (
            ("proposal_id", proposal_id),
            ("approval_id", approval_id),
        ):
            if not (
                isinstance(value, str)
                and 1 <= len(value.encode("utf-8")) <= 128
                and all(0x21 <= ord(character) <= 0x7E for character in value)
            ):
                raise ValueError(f"m110_{field_name}_invalid")
        if not (
            isinstance(timeout_seconds, (int, float))
            and not isinstance(timeout_seconds, bool)
            and math.isfinite(float(timeout_seconds))
            and 0.0 < float(timeout_seconds) <= 120.0
        ):
            raise ValueError("m110_capability_timeout_invalid")

        await self._send_json(
            {
                "type": "prepare_m110_restore_only_proof",
                "contract_version": 1,
                "proposal_id": proposal_id,
                "bundle_selector": "reviewed_supervised_transaction",
            }
        )
        prepared = await self._receive_terminal(
            "m110_restore_proof_prepare_result",
            timeout_seconds=min(float(timeout_seconds), 20.0),
        )
        review_digest = self._require_sha256(
            prepared.get("review_content_digest"),
            field_name="review_content_digest",
        )
        operation_digest = self._require_sha256(
            prepared.get("operation_digest"),
            field_name="operation_digest",
        )
        scope_addresses = prepared.get("scope_addresses")
        if not (
            prepared.get("status") == "prepared"
            and prepared.get("success") is True
            and prepared.get("reason") == "m110_restore_proof_bundle_current"
            and prepared.get("proposal_id") == proposal_id
            and type(prepared.get("review_contract_version")) is int
            and prepared.get("review_contract_version") == 2
            and prepared.get("operation_count") == 1
            and not isinstance(prepared.get("operation_count"), bool)
            and isinstance(scope_addresses, list)
            and bool(scope_addresses)
            and all(
                isinstance(address, str) and address.startswith("/")
                for address in scope_addresses
            )
            and prepared.get("token_issued") is False
            and prepared.get("live_authority_granted") is False
        ):
            raise RuntimeError(
                "m110_restore_proof_prepare_failed:"
                + self._bounded_terminal_failure_context(prepared)
            )

        issue_request = build_m110_restore_only_capability_request(
            proposal_id=proposal_id,
            review_contract_version=2,
            review_content_digest=review_digest,
            operation_digest=operation_digest,
            approval_id=approval_id,
            request_nonce=secrets.token_hex(16),
            secret=bytes(self.backend.secret),
        )
        await self._send_json(issue_request)
        issued = await self._receive_terminal(
            "supervised_capability_result",
            timeout_seconds=min(float(timeout_seconds), 20.0),
        )
        raw_token = issued.pop("supervised_capability_token", None)
        idempotency_key = issued.get("idempotency_key")
        latest_start_epoch = issued.get("latest_start_epoch")
        if not (
            issued.get("status") == "issued"
            and issued.get("success") is True
            and issued.get("reason") == "m110_capability_issued"
            and issued.get("capability_kind") == "restore_only_proof"
            and issued.get("proposal_id") == proposal_id
            and issued.get("review_contract_version") == 2
            and issued.get("review_content_digest") == review_digest
            and issued.get("operation_digest") == operation_digest
            and issued.get("approval_id") == approval_id
            and isinstance(idempotency_key, str)
            and re.fullmatch(r"m110-proof:[0-9a-f]{32}", idempotency_key)
            is not None
            and isinstance(latest_start_epoch, (int, float))
            and not isinstance(latest_start_epoch, bool)
            and math.isfinite(float(latest_start_epoch))
            and time.time() < float(latest_start_epoch) <= time.time() + 31.0
            and isinstance(raw_token, str)
            and re.fullmatch(r"m110cap_v1_[0-9a-f]{64}", raw_token) is not None
            and issued.get("single_use") is True
            and issued.get("stores_plaintext_token") is False
            and issued.get("live_authority_granted") is False
        ):
            raw_token = None
            raise RuntimeError(
                "m110_restore_proof_capability_issue_failed:"
                + self._bounded_terminal_failure_context(issued)
            )

        token = bytearray(raw_token.encode("ascii"))
        raw_token = None
        apply_request = {
            "type": "apply_operator_proposal",
            "proposal_id": proposal_id,
            "approved": True,
            "approval_id": approval_id,
            "idempotency_key": idempotency_key,
            "expected_review_contract_version": 2,
            "expected_review_content_digest": review_digest,
            "expected_operation_digest": operation_digest,
            "supervised_capability_token": token.decode("ascii"),
        }
        try:
            await self._send_json(apply_request)
        finally:
            apply_request.pop("supervised_capability_token", None)
            for index in range(len(token)):
                token[index] = 0

        terminal = await self._receive_terminal(
            "operator_proposal_apply_result",
            timeout_seconds=float(timeout_seconds),
        )
        self._validate_restore_only_proof_terminal(
            terminal,
            proposal_id=proposal_id,
            review_digest=review_digest,
            operation_digest=operation_digest,
        )
        return {
            "verified": True,
            "reason": "restore_only_proof_verified_and_restored",
            "proposal_id": proposal_id,
            "review_contract_version": 2,
            "review_content_digest": review_digest,
            "operation_digest": operation_digest,
            "scope_addresses": list(scope_addresses),
            "proof_terminal": terminal,
            "same_websocket_retained": True,
            "backend_process_retained": self.backend.process.poll() is None,
            "token_retained": False,
            "accepted": False,
            "owner_review_required": True,
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
        prior_readiness_invalidation_latched = bool(
            self._readiness_invalidation_latched
        )
        disconnect_invalidation_observed = False
        capture_stopped = not self.capture_started
        mixer_disconnected = not self.wing_connected
        backend_shutdown: dict[str, Any] | None = None
        try:
            if self.analysis_started:
                try:
                    await self._send_json(
                        {"type": "stop_m110_soundcheck_analysis"}
                    )
                    analysis_stopped = await self._receive_terminal(
                        "m110_soundcheck_analysis_stop_result",
                        timeout_seconds=3.0,
                    )
                    if not (
                        analysis_stopped.get("status") == "stopped"
                        and analysis_stopped.get("success") is True
                        and analysis_stopped.get("shared_capture_retained")
                        is True
                    ):
                        errors.append("m110_soundcheck_analysis_stop_unverified")
                    else:
                        self.analysis_started = False
                except Exception as exc:
                    errors.append(
                        "m110_soundcheck_analysis_stop_failed:"
                        + type(exc).__name__
                    )
            if self.capture_started:
                try:
                    await self._send_json({"type": "stop_audio_capture_preview"})
                    stopped = await self._receive_m110_owned_capture_stop_terminal(
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
                    (
                        disconnected,
                        disconnect_invalidation_observed,
                    ) = await self._receive_disconnect_terminal(
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
                self.control_armed = False
                self.control_approval_id = None
                backend_shutdown = self.backend.close()
        backend_terminated = bool(
            isinstance(backend_shutdown, Mapping)
            and backend_shutdown.get("terminated") is True
        )
        if not backend_terminated:
            errors.append("proof_owned_backend_not_terminated")
        if not (
            isinstance(backend_shutdown, Mapping)
            and backend_shutdown.get("clean_shutdown") is True
            and backend_shutdown.get("forced") is False
        ):
            errors.append("proof_owned_backend_shutdown_unverified")
        self.teardown_result = {
            "verified": not errors and capture_stopped and mixer_disconnected,
            "capture_stopped": capture_stopped,
            "mixer_disconnected": mixer_disconnected,
            "websocket_closed": True,
            "backend_process_terminated": backend_terminated,
            "backend_shutdown": dict(backend_shutdown or {}),
            "route_authority_retained": False,
            "readiness_invalidation_observed": (
                disconnect_invalidation_observed
            ),
            "prior_readiness_invalidation_latched": (
                prior_readiness_invalidation_latched
            ),
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


async def run_proof_owned_full_soundcheck(
    *,
    repo_root: Path,
    ws_port: int,
    artifact_root: Path,
    wing_ip: str,
    audio_device_index: int,
    approval_id: str,
    operator_input: OperatorInputCallback,
    terminal_emitter: TerminalEmitterCallback,
) -> dict[str, Any]:
    """Run route, reset, analysis, processor and musical child checkpoints."""

    cp05_gate = assert_m110_cp05_hardware_gate(repo_root)
    session = await open_proof_owned_backend_session(
        repo_root=repo_root,
        ws_port=ws_port,
        artifact_root=artifact_root,
    )
    route = None
    arm = None
    reset_containment_ack = None
    reset_muted_checkpoint = None
    reset = None
    m006_restore = None
    soundcheck_unmute_ack = None
    soundcheck_unmuted_checkpoint = None
    analysis = None
    lanes = None
    processors = None
    soundcheck = None
    final_unmuted_checkpoint = None
    final_audit_before_listening = None
    listening_ack = None
    final_audit_after_listening = None
    try:
        route = await session.establish_master_route(
            wing_ip=wing_ip,
            audio_device_index=audio_device_index,
            approval_id=approval_id,
        )
        arm = await session.arm_m110_attended_control(
            approval_id=approval_id,
        )
        await _emit_attended_terminal(
            terminal_emitter,
            {
                "event": "m110_physical_operator_checkpoint_required",
                "checkpoint": "reset_muted",
                "instruction": (
                    "Physically contain every audible PA/monitor output path "
                    "and mute WING Main 1. Software independently verifies "
                    "Main 1 only; non-Main output containment is operator-attested."
                ),
                "expected_literal": M110_RESET_CONTAINMENT_CONFIRMATION,
                "software_verified_output_scope": ["/main/1/mute"],
                "operator_attested_external_output_containment_required": True,
            },
        )
        reset_containment_ack = await _require_exact_operator_literal(
            operator_input,
            prompt=(
                "After all PA/monitor outputs are physically contained and "
                "Main 1 is muted, enter the exact confirmation literal: "
            ),
            expected=M110_RESET_CONTAINMENT_CONFIRMATION,
        )
        reset_containment_ack[
            "operator_attested_external_output_containment"
        ] = True
        reset_muted_checkpoint = (
            await session.verify_m110_main_output_checkpoint(
                checkpoint="reset_muted",
            )
        )
        reset = await session.execute_fixed_m110_maintenance_reset(
            approval_id=approval_id,
            operator_input=operator_input,
            terminal_emitter=terminal_emitter,
        )
        m006_restore = (
            await session.execute_m110_m006_exact_preimage_restore(
                approval_id=approval_id,
                operator_input=operator_input,
                terminal_emitter=terminal_emitter,
            )
        )
        await _emit_attended_terminal(
            terminal_emitter,
            {
                "event": "m110_physical_operator_checkpoint_required",
                "checkpoint": "soundcheck_unmuted",
                "instruction": (
                    "Keep the signal sources active, restore the intended "
                    "audible output path, and unmute WING Main 1."
                ),
                "expected_literal": M110_SOUNDCHECK_UNMUTE_CONFIRMATION,
            },
        )
        soundcheck_unmute_ack = await _require_exact_operator_literal(
            operator_input,
            prompt=(
                "After Main 1 is unmuted and the intended soundcheck output "
                "path is active, enter the exact confirmation literal: "
            ),
            expected=M110_SOUNDCHECK_UNMUTE_CONFIRMATION,
        )
        soundcheck_unmuted_checkpoint = (
            await session.verify_m110_main_output_checkpoint(
                checkpoint="soundcheck_unmuted",
            )
        )
        analysis = await session.run_m110_soundcheck_analysis()
        lanes = await session.execute_m110_interleaved_soundcheck(
            approval_id=approval_id,
            operator_input=operator_input,
            terminal_emitter=terminal_emitter,
        )
        processors = lanes.get("processor")
        soundcheck = lanes.get("soundcheck")
        final_unmuted_checkpoint = (
            await session.verify_m110_main_output_checkpoint(
                checkpoint="soundcheck_unmuted",
            )
        )
        final_audit_before_listening = (
            await session.run_m110_soundcheck_final_audit()
        )
        verified_state_sha256 = final_audit_before_listening[
            "verified_state_sha256"
        ]
        listening_literal = (
            "CONFIRM_M110_HW05_LISTENING_" + verified_state_sha256
        )
        listening_deadline_epoch = time.time() + 300.0
        await _emit_attended_terminal(
            terminal_emitter,
            {
                "event": "m110_physical_operator_checkpoint_required",
                "checkpoint": "hw05_verified_listening",
                "instruction": (
                    "Listen to the current full mix on the intended output "
                    "path. Confirm only if balance, dynamics, spatial image, "
                    "FX and feedback safety are physically acceptable."
                ),
                "verified_state_sha256": verified_state_sha256,
                "expected_literal": listening_literal,
                "deadline_epoch": listening_deadline_epoch,
                "operator_listening_required": True,
            },
        )
        listening_ack = await _require_exact_operator_literal(
            operator_input,
            prompt=(
                "After physically listening to this exact verified mix, "
                "enter the SHA-bound confirmation literal: "
            ),
            expected=listening_literal,
            deadline_epoch=listening_deadline_epoch,
        )
        listening_ack["verified_state_sha256"] = verified_state_sha256
        final_audit_after_listening = (
            await session.run_m110_soundcheck_final_audit()
        )
        if not (
            final_audit_after_listening.get("verified_state_sha256")
            == verified_state_sha256
            and final_audit_after_listening.get(
                "committed_surface_ledger_sha256"
            )
            == final_audit_before_listening.get(
                "committed_surface_ledger_sha256"
            )
            and final_audit_after_listening.get("run_id")
            == final_audit_before_listening.get("run_id")
        ):
            raise RuntimeError("m110_hw05_verified_state_changed_after_listening")
    finally:
        teardown = await session.close()
    if teardown.get("verified") is not True:
        raise RuntimeError(
            "m110_full_soundcheck_teardown_failed:"
            + ",".join(teardown.get("errors") or [])
        )
    readiness = (
        soundcheck.get("soundcheck_readiness")
        if isinstance(soundcheck, Mapping)
        else None
    )
    verified = bool(
        isinstance(lanes, Mapping)
        and lanes.get("verified") is True
        and isinstance(m006_restore, Mapping)
        and m006_restore.get("m006_restore_complete") is True
        and isinstance(readiness, Mapping)
        and readiness.get("mix_readiness") == "show_mix_ready"
        and isinstance(final_audit_before_listening, Mapping)
        and final_audit_before_listening.get("verified") is True
        and isinstance(listening_ack, Mapping)
        and listening_ack.get("operator_response_matched") is True
        and isinstance(final_audit_after_listening, Mapping)
        and final_audit_after_listening.get("verified_state_sha256")
        == final_audit_before_listening.get("verified_state_sha256")
        and teardown.get("verified") is True
    )
    return {
        "verified": verified,
        "status": "show_mix_ready" if verified else "soundcheck_incomplete",
        "reason": (
            "m110_attended_show_mix_hw05_verified_and_torn_down"
            if verified
            else str(
                (soundcheck or {}).get("reason")
                or "m110_final_show_mix_readiness_not_verified"
            )
        ),
        "identity": session.identity_result,
        "cp05_hardware_gate": cp05_gate,
        "route": route,
        "arm": arm,
        "reset_containment_operator_ack": reset_containment_ack,
        "reset_muted_checkpoint": reset_muted_checkpoint,
        "reset": reset,
        "m006_restore": m006_restore,
        "soundcheck_unmute_operator_ack": soundcheck_unmute_ack,
        "soundcheck_unmuted_checkpoint": soundcheck_unmuted_checkpoint,
        "final_soundcheck_unmuted_checkpoint": final_unmuted_checkpoint,
        "analysis": analysis,
        "lanes": lanes,
        "processors": processors,
        "soundcheck": soundcheck,
        "soundcheck_readiness": readiness,
        "final_audit_before_listening": final_audit_before_listening,
        "hw05_operator_listening_ack": listening_ack,
        "final_audit_after_listening": final_audit_after_listening,
        "teardown": teardown,
        "live_authority_granted": False,
        "readiness_closure_allowed": verified,
    }


def run_proof_owned_full_soundcheck_sync(**kwargs: Any) -> dict[str, Any]:
    return asyncio.run(run_proof_owned_full_soundcheck(**kwargs))


async def interactive_operator_input(prompt: str) -> str:
    """Read one attended checkpoint without blocking the event loop."""

    def read() -> str:
        try:
            return input(prompt)
        except EOFError as exc:
            raise RuntimeError("m110_operator_checkpoint_eof") from exc
        except KeyboardInterrupt as exc:
            raise RuntimeError("m110_operator_checkpoint_interrupted") from exc

    return await asyncio.to_thread(read)


def print_attended_terminal(payload: Mapping[str, Any]) -> None:
    print(
        json.dumps(dict(payload), sort_keys=True, separators=(",", ":")),
        flush=True,
    )


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
    full_execution: dict[str, Any] | None = None
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
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            failures.append("attended_execute_requires_interactive_tty")
        if not args.approval_id:
            failures.append("operator_approval_id_required")
        if args.confirmation != ATTENDED_EXECUTION_CONFIRMATION:
            failures.append("attended_execution_confirmation_mismatch")
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
    if args.execute and not failures:
        contacts_backend = True
        contacts_wing = True
        try:
            full_execution = run_proof_owned_full_soundcheck_sync(
                repo_root=args.repo_root,
                ws_port=args.ws_port,
                artifact_root=args.artifact_root,
                wing_ip=args.wing_ip,
                audio_device_index=args.audio_device_index,
                approval_id=args.approval_id,
                operator_input=interactive_operator_input,
                terminal_emitter=print_attended_terminal,
            )
            launch_probe = dict(full_execution.get("identity") or {})
            route_probe = dict(full_execution.get("route") or {})
            if full_execution.get("verified") is not True:
                failures.append(
                    "m110_attended_soundcheck_incomplete:"
                    + str(full_execution.get("reason") or "unknown")
                )
        except Exception as exc:
            failures.append(
                f"m110_attended_soundcheck_failed:{type(exc).__name__}:{exc}"
            )
    summary = {
        "schema": "automixer.m110_attended_runner_summary.v1",
        "timestamp_utc": utc_now(),
        "status": (
            "show_mix_ready"
            if full_execution is not None
            and full_execution.get("verified") is True
            and not failures
            else
            "verified"
            if (launch_probe is not None or route_probe is not None) and not failures
            else "blocked"
            if args.execute or failures
            else "planned"
        ),
        "reason": (
            "m110_attended_show_mix_committed"
            if full_execution is not None
            and full_execution.get("verified") is True
            and not failures
            else
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
        "reset_containment_confirmation": (
            M110_RESET_CONTAINMENT_CONFIRMATION
        ),
        "soundcheck_unmute_confirmation": (
            M110_SOUNDCHECK_UNMUTE_CONFIRMATION
        ),
        "interactive_tty_required": bool(args.execute),
        "contacts_backend": contacts_backend,
        "contacts_wing": contacts_wing,
        "opens_udp_or_osc": contacts_wing,
        "runner_opens_udp_or_osc": False,
        "proof_owned_backend_opens_udp_or_osc": contacts_wing,
        "live_write": bool(args.execute and contacts_wing),
        "accepted": bool(
            full_execution is not None
            and full_execution.get("verified") is True
        ),
        "readiness_closure_allowed": bool(
            full_execution is not None
            and full_execution.get("readiness_closure_allowed") is True
        ),
        "launch_probe": launch_probe,
        "route_probe": route_probe,
        "execution": full_execution,
    }
    try:
        write_private_summary(
            artifact_root=args.artifact_root,
            filename=(
                "execution-summary.json"
                if args.execute
                else "plan-summary.json"
            ),
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
