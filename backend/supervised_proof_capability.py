"""Process-private attended launch authority and one-shot capability contracts.

M110 live authority never comes from a file or environment digest.  The
attended parent passes one exact manifest and a random secret through an
inherited, one-shot file descriptor.  This module consumes that packet before
the server opens any mixer transport and retains the secret only in memory.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import select
import stat
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


M110_PRESTART_MANIFEST_SCHEMA = "automixer.m110_attended_prestart_manifest.v1"
M110_REQUEST_HMAC_DOMAIN = b"automixer.m110.request.v1\n"
M110_AUTHORITY_FD_ENV = "AUTOMIXER_M110_AUTHORITY_FD"
M110_AUTHORITY_PIPE_TIMEOUT_SECONDS = 5.0
M110_MAX_MANIFEST_BYTES = 64 * 1024
M110_MAX_TOMBSTONED_NONCES = 4096
M110_ALLOWED_UNTRACKED_REPORT = re.compile(
    r"reports/[A-Za-z0-9][A-Za-z0-9._-]*\.json"
)
M110_IGNORED_IMPORT_ARTIFACT_SUFFIXES = (
    ".py",
    ".pyw",
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
)
M110_RUNTIME_IMPORT_ROOTS = (".", "backend", "tools")
M110_IMPORT_SCAN_EXCLUDED_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
M110_RUNTIME_CRITICAL_PATHS = (
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


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _valid_lower_hex(value: Any, length: int) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError("json_object_required")

    def reject_float(item: Any) -> None:
        if isinstance(item, float):
            raise ValueError("json_float_forbidden")
        if isinstance(item, dict):
            for nested in item.values():
                reject_float(nested)
        elif isinstance(item, list):
            for nested in item:
                reject_float(nested)

    reject_float(value)
    return value


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
        raise RuntimeError(
            f"M110 git {' '.join(arguments)} failed ({completed.returncode})"
        )
    return completed.stdout


def _tracked_tree_is_clean(repo_root: Path) -> bool:
    completed = subprocess.run(
        ["/usr/bin/git", "diff", "--quiet", "HEAD", "--"],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=5,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError("M110 could not determine tracked tree cleanliness")
    return completed.returncode == 0


def _assert_untracked_paths_safe(repo_root: Path) -> None:
    raw = _git(repo_root, "ls-files", "--others", "--exclude-standard", "-z")
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            relative_path = item.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("M110 untracked path is not UTF-8") from exc
        if M110_ALLOWED_UNTRACKED_REPORT.fullmatch(relative_path) is None:
            raise RuntimeError(
                f"M110 unsafe untracked path present: {relative_path}"
            )
        path = repo_root / relative_path
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                f"M110 allowed report must be a regular non-symlink: {relative_path}"
            )


def _assert_no_ignored_import_shadow_artifacts(repo_root: Path) -> None:
    """Reject ignored executable files that Python can load from import roots.

    The normal untracked scan intentionally respects ``.gitignore`` so that
    caches and local environments do not invalidate a checkout.  Python can,
    however, import a sourceless bytecode or extension module placed directly
    on ``PYTHONPATH`` even when Git ignores it.  Scan the two runtime import
    roots, plus direct package initializers, without traversing venvs or cache
    trees.
    """

    tracked = {
        item.decode("utf-8")
        for item in _git(repo_root, "ls-files", "-z").split(b"\0")
        if item
    }

    def is_loadable_ignored_file(path: Path) -> bool:
        return path.is_file() and path.name.endswith(
            M110_IGNORED_IMPORT_ARTIFACT_SUFFIXES
        )

    for import_root_relative in M110_RUNTIME_IMPORT_ROOTS:
        import_root = (
            repo_root
            if import_root_relative == "."
            else repo_root / import_root_relative
        )
        try:
            entries = list(import_root.iterdir())
        except OSError as exc:
            raise RuntimeError("M110 runtime import root is unreadable") from exc
        for entry in entries:
            if entry.name in M110_IMPORT_SCAN_EXCLUDED_DIRECTORIES:
                continue
            candidates: list[Path] = []
            try:
                if is_loadable_ignored_file(entry):
                    candidates.append(entry)
                elif entry.is_dir() and entry.name.isidentifier():
                    for child in entry.iterdir():
                        if child.name.startswith("__init__") and (
                            is_loadable_ignored_file(child)
                        ):
                            candidates.append(child)
            except OSError as exc:
                raise RuntimeError(
                    f"M110 import-shadow path is unreadable: {entry.name}"
                ) from exc
            for candidate in candidates:
                relative_path = candidate.relative_to(repo_root).as_posix()
                if relative_path not in tracked:
                    raise RuntimeError(
                        "M110 ignored import-shadow artifact present: "
                        f"{relative_path}"
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
                        "M110 ignored import-shadow artifact present: "
                        f"{relative_path}"
                    )


def _critical_file_row(repo_root: Path, relative_path: str) -> dict[str, Any]:
    if relative_path.startswith("/") or ".." in Path(relative_path).parts:
        raise RuntimeError("M110 critical path is not canonical")
    _git(repo_root, "ls-files", "--error-unmatch", "--", relative_path)
    path = repo_root / relative_path
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"M110 critical source missing: {relative_path}") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(
            f"M110 critical source must be a regular non-symlink: {relative_path}"
        )
    value = path.read_bytes()
    return {
        "path": relative_path,
        "size": len(value),
        "sha256": f"sha256:{hashlib.sha256(value).hexdigest()}",
    }


def build_m110_prestart_manifest(
    repo_root: str | Path,
    *,
    launch_nonce: str,
) -> dict[str, Any]:
    """Independently capture the exact clean source identity required by M110."""

    root = Path(repo_root).resolve()
    if not _valid_lower_hex(launch_nonce, 32):
        raise RuntimeError("M110 launch nonce must be 32 lowercase hex")
    if not _tracked_tree_is_clean(root):
        raise RuntimeError("M110 tracked source tree is dirty")
    _assert_untracked_paths_safe(root)
    _assert_no_ignored_import_shadow_artifacts(root)
    files = [
        _critical_file_row(root, relative_path)
        for relative_path in sorted(M110_RUNTIME_CRITICAL_PATHS)
    ]
    return {
        "schema": M110_PRESTART_MANIFEST_SCHEMA,
        "mode": "m110_attended_runner",
        "head": _git(root, "rev-parse", "HEAD").decode("ascii").strip(),
        "tree": _git(root, "rev-parse", "HEAD^{tree}").decode("ascii").strip(),
        "launch_nonce": launch_nonce,
        "files": files,
    }


def _read_pipe_exact(fd: int, size: int, *, deadline: float) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        wait = deadline - time.monotonic()
        if wait <= 0:
            raise RuntimeError("Timed out waiting for M110 authority pipe")
        readable, _, _ = select.select([fd], [], [], wait)
        if not readable:
            raise RuntimeError("Timed out waiting for M110 authority pipe")
        chunk = os.read(fd, remaining)
        if not chunk:
            raise RuntimeError("M110 authority pipe ended early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@dataclass
class M110LaunchAuthority:
    """Non-serializable in-memory proof that requests come from this launcher."""

    launch_nonce: str
    manifest_sha256: str
    head: str
    tree: str
    authority_secret_sha256: str
    _secret: bytearray = field(repr=False)
    _repo_root: Path = field(repr=False)
    _used_nonces: OrderedDict[str, None] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def source_snapshot_status(self) -> tuple[bool, str]:
        """Recompute the bounded launch manifest and permanently fail on drift."""

        with self._lock:
            if self._closed:
                return False, "m110_launch_authority_closed"
        try:
            current = build_m110_prestart_manifest(
                self._repo_root,
                launch_nonce=self.launch_nonce,
            )
            current_raw = _canonical_json_bytes(current)
            current_sha256 = hashlib.sha256(current_raw).hexdigest()
            current_matches = bool(
                current_sha256 == self.manifest_sha256
                and current.get("head") == self.head
                and current.get("tree") == self.tree
            )
        except Exception:
            current_matches = False
        if not current_matches:
            self.close()
            return False, "m110_live_source_snapshot_drift"
        with self._lock:
            if self._closed:
                return False, "m110_launch_authority_closed"
        return True, "m110_live_source_snapshot_current"

    def verify_and_tombstone_request(
        self,
        request: Mapping[str, Any],
    ) -> tuple[bool, str]:
        source_current, source_reason = self.source_snapshot_status()
        if not source_current:
            return False, source_reason
        nonce = request.get("request_nonce")
        proof = request.get("authority_proof")
        if not _valid_lower_hex(nonce, 32):
            return False, "m110_request_nonce_invalid"
        if not (
            isinstance(proof, str)
            and re.fullmatch(r"hmac-sha256:[0-9a-f]{64}", proof)
        ):
            return False, "m110_authority_proof_invalid"
        canonical = dict(request)
        canonical.pop("authority_proof", None)
        try:
            request_bytes = _canonical_json_bytes(canonical)
        except (TypeError, ValueError):
            return False, "m110_request_not_canonicalizable"
        with self._lock:
            if self._closed:
                return False, "m110_launch_authority_closed"
            if nonce in self._used_nonces:
                return False, "m110_request_nonce_replayed"
            expected = "hmac-sha256:" + hmac.new(
                bytes(self._secret),
                M110_REQUEST_HMAC_DOMAIN + request_bytes,
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(expected, proof):
                return False, "m110_authority_proof_mismatch"
            if len(self._used_nonces) >= M110_MAX_TOMBSTONED_NONCES:
                return False, "m110_request_nonce_capacity_exhausted"
            self._used_nonces[nonce] = None
        return True, "m110_authority_request_verified"

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for index in range(len(self._secret)):
                self._secret[index] = 0
            self._used_nonces.clear()
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def consume_m110_attended_launch_authority(
    repo_root: str | Path,
) -> tuple[dict[str, Any], Optional[M110LaunchAuthority]]:
    """Consume, verify and close the inherited M110 launch packet exactly once."""

    fd_text = os.environ.pop(M110_AUTHORITY_FD_ENV, None)
    if fd_text is None:
        return (
            {
                "mode": "unbound",
                "bound_before_process_start": False,
                "m110_attended_eligible": False,
            },
            None,
        )
    if not re.fullmatch(r"[3-9]|[1-9][0-9]{1,2}", str(fd_text)):
        raise RuntimeError("AUTOMIXER_M110_AUTHORITY_FD is invalid")
    fd = int(fd_text)
    deadline = time.monotonic() + M110_AUTHORITY_PIPE_TIMEOUT_SECONDS
    secret = bytearray()
    try:
        if not stat.S_ISFIFO(os.fstat(fd).st_mode):
            raise RuntimeError("M110 authority descriptor must be a one-shot pipe")
        manifest_size = int.from_bytes(
            _read_pipe_exact(fd, 4, deadline=deadline),
            "big",
            signed=False,
        )
        if not 1 <= manifest_size <= M110_MAX_MANIFEST_BYTES:
            raise RuntimeError("M110 prestart manifest size is invalid")
        raw_manifest = _read_pipe_exact(fd, manifest_size, deadline=deadline)
        secret.extend(_read_pipe_exact(fd, 32, deadline=deadline))
        wait = deadline - time.monotonic()
        if wait <= 0:
            raise RuntimeError("Timed out waiting for M110 authority pipe EOF")
        readable, _, _ = select.select([fd], [], [], wait)
        if not readable:
            raise RuntimeError("Timed out waiting for M110 authority pipe EOF")
        if os.read(fd, 1):
            raise RuntimeError("M110 authority pipe has trailing bytes")
    except OSError as exc:
        for index in range(len(secret)):
            secret[index] = 0
        raise RuntimeError("Could not consume M110 authority pipe") from exc
    except BaseException:
        for index in range(len(secret)):
            secret[index] = 0
        raise
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

    try:
        supplied_manifest = _strict_json_object(raw_manifest)
        if raw_manifest != _canonical_json_bytes(supplied_manifest):
            raise RuntimeError("M110 prestart manifest bytes are not canonical")
        if set(supplied_manifest) != {
            "schema",
            "mode",
            "head",
            "tree",
            "launch_nonce",
            "files",
        }:
            raise RuntimeError("M110 prestart manifest fields mismatch")
        if (
            supplied_manifest.get("schema") != M110_PRESTART_MANIFEST_SCHEMA
            or supplied_manifest.get("mode") != "m110_attended_runner"
            or not _valid_lower_hex(supplied_manifest.get("head"), 40)
            or not _valid_lower_hex(supplied_manifest.get("tree"), 40)
            or not _valid_lower_hex(supplied_manifest.get("launch_nonce"), 32)
        ):
            raise RuntimeError("M110 prestart manifest identity is invalid")
        files = supplied_manifest.get("files")
        if not isinstance(files, list):
            raise RuntimeError("M110 prestart manifest files are invalid")
        expected_paths = sorted(M110_RUNTIME_CRITICAL_PATHS)
        if [row.get("path") for row in files if isinstance(row, dict)] != expected_paths:
            raise RuntimeError("M110 prestart manifest critical paths mismatch")
        for row in files:
            if (
                not isinstance(row, dict)
                or set(row) != {"path", "size", "sha256"}
                or not isinstance(row.get("size"), int)
                or isinstance(row.get("size"), bool)
                or row["size"] < 0
                or not (
                    isinstance(row.get("sha256"), str)
                    and re.fullmatch(r"sha256:[0-9a-f]{64}", row["sha256"])
                )
            ):
                raise RuntimeError("M110 prestart manifest file row is invalid")
        expected_manifest = build_m110_prestart_manifest(
            repo_root,
            launch_nonce=supplied_manifest["launch_nonce"],
        )
        if raw_manifest != _canonical_json_bytes(expected_manifest):
            raise RuntimeError("M110 prestart manifest does not match current source")
    except Exception:
        for index in range(len(secret)):
            secret[index] = 0
        raise

    manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
    secret_sha256 = hashlib.sha256(secret).hexdigest()
    authority = M110LaunchAuthority(
        launch_nonce=supplied_manifest["launch_nonce"],
        manifest_sha256=manifest_sha256,
        head=supplied_manifest["head"],
        tree=supplied_manifest["tree"],
        authority_secret_sha256=secret_sha256,
        _secret=secret,
        _repo_root=Path(repo_root).resolve(),
    )
    binding = {
        "mode": "m110_attended_runner",
        "launch_nonce": authority.launch_nonce,
        "prestart_manifest_schema": M110_PRESTART_MANIFEST_SCHEMA,
        "prestart_manifest_size": len(raw_manifest),
        "prestart_manifest_sha256": manifest_sha256,
        "head": authority.head,
        "tree": authority.tree,
        "tracked_tree_clean": True,
        "authority_secret_sha256": secret_sha256,
        "bound_before_process_start": True,
        "m110_attended_eligible": True,
        "live_authority_granted": False,
    }
    return binding, authority


def sign_m110_request(
    request: Mapping[str, Any],
    *,
    secret: bytes,
) -> dict[str, Any]:
    """Runner helper: sign a strict request without mutating its input."""

    signed = dict(request)
    signed.pop("authority_proof", None)
    signed["authority_proof"] = "hmac-sha256:" + hmac.new(
        secret,
        M110_REQUEST_HMAC_DOMAIN + _canonical_json_bytes(signed),
        hashlib.sha256,
    ).hexdigest()
    return signed
