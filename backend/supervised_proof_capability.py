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
import secrets
import select
import stat
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


M110_PRESTART_MANIFEST_SCHEMA = "automixer.m110_attended_prestart_manifest.v1"
M110_REQUEST_HMAC_DOMAIN = b"automixer.m110.request.v1\n"
M110_AUTHORITY_FD_ENV = "AUTOMIXER_M110_AUTHORITY_FD"
M110_AUTHORITY_PIPE_TIMEOUT_SECONDS = 5.0
M110_MAX_MANIFEST_BYTES = 64 * 1024
M110_MAX_TOMBSTONED_NONCES = 4096
M110_CAPABILITY_TOKEN_PREFIX = "m110cap_v1_"
M110_CAPABILITY_TTL_SECONDS = 30.0
M110_MAX_ACTIVE_CAPABILITIES = 8
M110_MAX_CAPABILITY_TOMBSTONES = 4096
M110_CAPABILITY_KINDS = (
    "restore_only_proof",
    "maintenance_dsp_reset_commit_once",
    "processor_enable_commit_once",
    "soundcheck_child_commit_once",
)
M110_CAPABILITY_CONFIRMATIONS = {
    "restore_only_proof": "ISSUE_RESTORE_ONLY_PROOF_CAPABILITY",
    "maintenance_dsp_reset_commit_once": (
        "ISSUE_MAINTENANCE_DSP_RESET_COMMIT_ONCE"
    ),
    "processor_enable_commit_once": "ISSUE_PROCESSOR_ENABLE_COMMIT_ONCE",
    "soundcheck_child_commit_once": "ISSUE_SOUNDCHECK_CHILD_COMMIT_ONCE",
}
M110_CAPABILITY_ISSUE_REQUEST_FIELDS = frozenset(
    {
        "type",
        "contract_version",
        "capability_kind",
        "proposal_id",
        "expected_review_contract_version",
        "expected_review_content_digest",
        "expected_operation_digest",
        "approval_id",
        "confirmation",
        "request_nonce",
        "authority_proof",
    }
)
M110_ARM_REQUEST_FIELDS = frozenset(
    {
        "type",
        "contract_version",
        "approval_id",
        "confirmation",
        "request_nonce",
        "authority_proof",
    }
)
M110_CAPABILITY_APPLY_REQUEST_FIELDS = frozenset(
    {
        "type",
        "proposal_id",
        "approved",
        "approval_id",
        "idempotency_key",
        "expected_review_contract_version",
        "expected_review_content_digest",
        "expected_operation_digest",
        "supervised_capability_token",
    }
)
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


def validate_m110_capability_issue_request(
    request: Mapping[str, Any],
) -> tuple[bool, str]:
    """Validate the fixed public issuance envelope without granting authority."""

    if type(request) is not dict:
        return False, "m110_capability_issue_request_object_required"
    if set(request) != M110_CAPABILITY_ISSUE_REQUEST_FIELDS:
        return False, "m110_capability_issue_request_fields_mismatch"
    kind = request.get("capability_kind")
    if (
        request.get("type") != "issue_supervised_capability"
        or request.get("contract_version") != 1
        or isinstance(request.get("contract_version"), bool)
        or kind not in M110_CAPABILITY_KINDS
    ):
        return False, "m110_capability_issue_contract_invalid"
    for field_name in ("proposal_id", "approval_id"):
        value = request.get(field_name)
        if not (
            isinstance(value, str)
            and 1 <= len(value.encode("utf-8")) <= 128
            and all(0x21 <= ord(character) <= 0x7E for character in value)
        ):
            return False, f"m110_capability_issue_{field_name}_invalid"
    if request.get("expected_review_contract_version") != 2 or isinstance(
        request.get("expected_review_contract_version"), bool
    ):
        return False, "m110_capability_issue_review_version_invalid"
    for field_name in (
        "expected_review_content_digest",
        "expected_operation_digest",
    ):
        value = request.get(field_name)
        if not (
            isinstance(value, str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", value)
        ):
            return False, f"m110_capability_issue_{field_name}_invalid"
    if request.get("confirmation") != M110_CAPABILITY_CONFIRMATIONS[kind]:
        return False, "m110_capability_issue_confirmation_mismatch"
    if not _valid_lower_hex(request.get("request_nonce"), 32):
        return False, "m110_capability_issue_request_nonce_invalid"
    proof = request.get("authority_proof")
    if not (
        isinstance(proof, str)
        and re.fullmatch(r"hmac-sha256:[0-9a-f]{64}", proof)
    ):
        return False, "m110_capability_issue_authority_proof_invalid"
    return True, "m110_capability_issue_request_valid"


def validate_m110_arm_request(
    request: Mapping[str, Any],
) -> tuple[bool, str]:
    if type(request) is not dict:
        return False, "m110_arm_request_object_required"
    if set(request) != M110_ARM_REQUEST_FIELDS:
        return False, "m110_arm_request_fields_mismatch"
    if (
        request.get("type") != "arm_m110_attended_control"
        or request.get("contract_version") != 1
        or isinstance(request.get("contract_version"), bool)
        or request.get("confirmation")
        != "ARM_M110_ATTENDED_ONE_SHOT_CONTROL"
    ):
        return False, "m110_arm_contract_invalid"
    approval_id = request.get("approval_id")
    if not (
        isinstance(approval_id, str)
        and 1 <= len(approval_id.encode("utf-8")) <= 128
        and all(0x21 <= ord(character) <= 0x7E for character in approval_id)
    ):
        return False, "m110_arm_approval_id_invalid"
    if not _valid_lower_hex(request.get("request_nonce"), 32):
        return False, "m110_arm_request_nonce_invalid"
    if not (
        isinstance(request.get("authority_proof"), str)
        and re.fullmatch(
            r"hmac-sha256:[0-9a-f]{64}",
            request["authority_proof"],
        )
    ):
        return False, "m110_arm_authority_proof_invalid"
    return True, "m110_arm_request_valid"


def validate_m110_capability_apply_request(
    request: Mapping[str, Any],
) -> tuple[bool, str]:
    if type(request) is not dict:
        return False, "m110_capability_apply_request_object_required"
    if set(request) != M110_CAPABILITY_APPLY_REQUEST_FIELDS:
        return False, "m110_capability_apply_request_fields_mismatch"
    if request.get("type") != "apply_operator_proposal":
        return False, "m110_capability_apply_type_invalid"
    if request.get("approved") is not True:
        return False, "m110_capability_apply_approval_required"
    for field_name in ("proposal_id", "approval_id", "idempotency_key"):
        value = request.get(field_name)
        if not (
            isinstance(value, str)
            and 1 <= len(value.encode("utf-8")) <= 512
            and "\x00" not in value
        ):
            return False, f"m110_capability_apply_{field_name}_invalid"
    if request.get("expected_review_contract_version") != 2 or isinstance(
        request.get("expected_review_contract_version"), bool
    ):
        return False, "m110_capability_apply_review_version_invalid"
    for field_name in (
        "expected_review_content_digest",
        "expected_operation_digest",
    ):
        if not _valid_sha256_digest(request.get(field_name)):
            return False, f"m110_capability_apply_{field_name}_invalid"
    if SupervisedCapabilityStore._token_digest(
        request.get("supervised_capability_token")
    ) is None:
        return False, "m110_capability_apply_token_invalid"
    return True, "m110_capability_apply_request_valid"


_M110_CONTROL_FACTORY_MARKER = object()


@dataclass(frozen=True, slots=True, init=False)
class M110AttendedControlAuthority:
    runtime_instance_nonce: str
    runtime_manifest_sha256: str
    launch_manifest_sha256: str
    source_head: str
    source_tree: str
    route_proof_id: str
    route_artifact_sha256: str
    connection_generation: int
    transport_generation: int
    emergency_generation: int
    owner_connection_id: int
    approval_id: str
    armed_at_epoch: float
    _owner_ref: Any = field(repr=False, compare=False)
    _route_ref: Any = field(repr=False, compare=False)
    _transport_ref: Any = field(repr=False, compare=False)
    _capture_ref: Any = field(repr=False, compare=False)
    _factory_marker: object = field(repr=False, compare=False)

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("M110AttendedControlAuthority is backend-factory-only")

    def is_authentic(self) -> bool:
        return self._factory_marker is _M110_CONTROL_FACTORY_MARKER


def new_m110_attended_control_authority(
    **values: Any,
) -> M110AttendedControlAuthority:
    required = {
        "runtime_instance_nonce",
        "runtime_manifest_sha256",
        "launch_manifest_sha256",
        "source_head",
        "source_tree",
        "route_proof_id",
        "route_artifact_sha256",
        "connection_generation",
        "transport_generation",
        "emergency_generation",
        "owner_connection_id",
        "approval_id",
        "armed_at_epoch",
        "_owner_ref",
        "_route_ref",
        "_transport_ref",
        "_capture_ref",
    }
    if set(values) != required:
        raise ValueError("m110_control_authority_fields_mismatch")
    authority = object.__new__(M110AttendedControlAuthority)
    for name, value in values.items():
        object.__setattr__(authority, name, value)
    object.__setattr__(
        authority,
        "_factory_marker",
        _M110_CONTROL_FACTORY_MARKER,
    )
    if not authority.is_authentic():
        raise RuntimeError("m110_control_authority_factory_failed")
    return authority


def _valid_sha256_digest(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", value)
    )


def _valid_binding_text(value: Any, *, maximum_bytes: int = 512) -> bool:
    return bool(
        isinstance(value, str)
        and value
        and len(value.encode("utf-8")) <= maximum_bytes
        and "\x00" not in value
    )


@dataclass(frozen=True, slots=True)
class M110CapabilityBinding:
    """Exact immutable authority facts re-derived at issue and consume."""

    capability_kind: str
    runtime_instance_nonce: str
    runtime_manifest_sha256: str
    launch_manifest_sha256: str
    source_head: str
    source_tree: str
    connection_generation: int
    transport_generation: int
    console_serial: str
    firmware_version: str
    route_proof_id: str
    route_artifact_sha256: str
    capture_hardware_uid: str
    emergency_generation: int
    owner_connection_id: int
    proposal_id: str
    review_contract_version: int
    review_content_digest: str
    operation_digest: str
    canonical_bundle_bytes: bytes = field(repr=False)
    approval_id: str = ""
    idempotency_key: str = ""
    scope_addresses: tuple[str, ...] = ()
    target_fingerprints: tuple[tuple[str, str], ...] = ()
    _owner_ref: Any = field(default=None, repr=False, compare=False)
    _route_ref: Any = field(default=None, repr=False, compare=False)
    _transport_ref: Any = field(default=None, repr=False, compare=False)
    _capture_ref: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.capability_kind not in M110_CAPABILITY_KINDS:
            raise ValueError("m110_capability_binding_kind_invalid")
        for value, length, reason in (
            (self.runtime_instance_nonce, 32, "runtime_nonce"),
            (self.source_head, 40, "source_head"),
            (self.source_tree, 40, "source_tree"),
        ):
            if not _valid_lower_hex(value, length):
                raise ValueError(f"m110_capability_binding_{reason}_invalid")
        for value, reason in (
            (self.runtime_manifest_sha256, "runtime_manifest"),
            (self.launch_manifest_sha256, "launch_manifest"),
            (self.route_artifact_sha256, "route_artifact"),
            (self.review_content_digest, "review_content"),
            (self.operation_digest, "operation"),
        ):
            normalized = (
                value if str(value).startswith("sha256:") else f"sha256:{value}"
            )
            if not _valid_sha256_digest(normalized):
                raise ValueError(f"m110_capability_binding_{reason}_digest_invalid")
        for value, reason in (
            (self.console_serial, "console_serial"),
            (self.firmware_version, "firmware_version"),
            (self.route_proof_id, "route_proof_id"),
            (self.capture_hardware_uid, "capture_hardware_uid"),
            (self.proposal_id, "proposal_id"),
            (self.approval_id, "approval_id"),
            (self.idempotency_key, "idempotency_key"),
        ):
            if not _valid_binding_text(value):
                raise ValueError(f"m110_capability_binding_{reason}_invalid")
        for value, reason in (
            (self.connection_generation, "connection_generation"),
            (self.transport_generation, "transport_generation"),
            (self.emergency_generation, "emergency_generation"),
            (self.owner_connection_id, "owner_connection_id"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"m110_capability_binding_{reason}_invalid")
        if self.review_contract_version != 2 or isinstance(
            self.review_contract_version, bool
        ):
            raise ValueError("m110_capability_binding_review_version_invalid")
        if not isinstance(self.canonical_bundle_bytes, bytes) or not (
            self.canonical_bundle_bytes
        ):
            raise ValueError("m110_capability_binding_bundle_bytes_invalid")
        actual_operation_digest = "sha256:" + hashlib.sha256(
            self.canonical_bundle_bytes
        ).hexdigest()
        if not hmac.compare_digest(actual_operation_digest, self.operation_digest):
            raise ValueError("m110_capability_binding_operation_digest_mismatch")
        if (
            not self.scope_addresses
            or len(self.scope_addresses) > 64
            or len(self.scope_addresses) != len(set(self.scope_addresses))
            or any(
                not isinstance(address, str)
                or not re.fullmatch(r"/[A-Za-z0-9_$.-]+(?:/[A-Za-z0-9_$.-]+)+", address)
                for address in self.scope_addresses
            )
        ):
            raise ValueError("m110_capability_binding_scope_invalid")
        if len(self.target_fingerprints) != len(self.scope_addresses):
            raise ValueError("m110_capability_binding_targets_incomplete")
        target_addresses = tuple(address for address, _ in self.target_fingerprints)
        if target_addresses != self.scope_addresses or any(
            not _valid_binding_text(target, maximum_bytes=256)
            for _address, target in self.target_fingerprints
        ):
            raise ValueError("m110_capability_binding_targets_invalid")
        if (
            self._owner_ref is None
            or self._route_ref is None
            or self._transport_ref is None
            or self._capture_ref is None
        ):
            raise ValueError("m110_capability_binding_object_identity_required")

    def matches_exactly(self, other: Any) -> bool:
        if type(other) is not M110CapabilityBinding:
            return False
        scalar_fields = (
            "capability_kind",
            "runtime_instance_nonce",
            "runtime_manifest_sha256",
            "launch_manifest_sha256",
            "source_head",
            "source_tree",
            "connection_generation",
            "transport_generation",
            "console_serial",
            "firmware_version",
            "route_proof_id",
            "route_artifact_sha256",
            "capture_hardware_uid",
            "emergency_generation",
            "owner_connection_id",
            "proposal_id",
            "review_contract_version",
            "review_content_digest",
            "operation_digest",
            "canonical_bundle_bytes",
            "approval_id",
            "idempotency_key",
            "scope_addresses",
            "target_fingerprints",
        )
        return bool(
            all(getattr(self, name) == getattr(other, name) for name in scalar_fields)
            and self._owner_ref is other._owner_ref
            and self._route_ref is other._route_ref
            and self._transport_ref is other._transport_ref
            and self._capture_ref is other._capture_ref
        )


@dataclass(frozen=True, slots=True)
class M110IssuedCapability:
    """Plaintext token handoff. The store never retains this object or token."""

    capability_kind: str
    token: str = field(repr=False)
    issued_at_epoch: float
    latest_start_epoch: float
    idempotency_key: str
    operation_digest: str


_M110_GRANT_FACTORY_MARKER = object()
_M110_GRANT_EXECUTION_STATE_FACTORY_MARKER = object()


class _M110GrantExecutionState:
    __slots__ = ("_phase", "_terminal", "_lock", "_factory_marker")

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("m110_grant_execution_state_is_factory_only")

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("m110_grant_execution_state_is_immutable")

    def begin(self) -> bool:
        if self._factory_marker is not _M110_GRANT_EXECUTION_STATE_FACTORY_MARKER:
            return False
        with self._lock:
            if self._phase != "issued":
                return False
            object.__setattr__(self, "_phase", "executing")
            return True

    def mark_terminal(self, terminal: str) -> bool:
        with self._lock:
            if self._phase not in {"issued", "executing"}:
                return False
            object.__setattr__(self, "_phase", "terminal")
            object.__setattr__(self, "_terminal", str(terminal))
            return True

    def status(self) -> str:
        with self._lock:
            return str(self._phase)

    def __copy__(self):
        raise TypeError("m110_grant_execution_state_copy_forbidden")

    def __deepcopy__(self, _memo):
        raise TypeError("m110_grant_execution_state_copy_forbidden")

    def __reduce__(self):
        raise TypeError("m110_grant_execution_state_pickle_forbidden")


def _new_m110_grant_execution_state() -> _M110GrantExecutionState:
    state = object.__new__(_M110GrantExecutionState)
    object.__setattr__(state, "_phase", "issued")
    object.__setattr__(state, "_terminal", None)
    object.__setattr__(state, "_lock", threading.Lock())
    object.__setattr__(
        state,
        "_factory_marker",
        _M110_GRANT_EXECUTION_STATE_FACTORY_MARKER,
    )
    return state


class _M110CapabilityGrantBase:
    __slots__ = (
        "binding",
        "consumed_at_epoch",
        "consumed_at_monotonic",
        "_factory_marker",
        "_execution_state",
        "_execution_lock",
    )
    CAPABILITY_KIND = ""

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("m110_capability_grant_is_immutable")

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} proposal_id="
            f"{self.binding.proposal_id!r} operation_digest="
            f"{self.binding.operation_digest!r}>"
        )

    def is_authentic(self) -> bool:
        return bool(
            self._factory_marker is _M110_GRANT_FACTORY_MARKER
            and self.binding.capability_kind == self.CAPABILITY_KIND
        )

    def begin_execution(self) -> bool:
        if not self.is_authentic():
            return False
        if type(self._execution_state) is not _M110GrantExecutionState:
            return False
        return self._execution_state.begin()

    def mark_terminal(self, terminal: str) -> bool:
        if type(self._execution_state) is not _M110GrantExecutionState:
            return False
        return self._execution_state.mark_terminal(terminal)

    def execution_status(self) -> str:
        if type(self._execution_state) is not _M110GrantExecutionState:
            return "invalid"
        return self._execution_state.status()

    def __copy__(self):
        raise TypeError("m110_capability_grant_copy_forbidden")

    def __deepcopy__(self, _memo):
        raise TypeError("m110_capability_grant_copy_forbidden")

    def __reduce__(self):
        raise TypeError("m110_capability_grant_pickle_forbidden")


class RestoreOnlyProofGrant(_M110CapabilityGrantBase):
    CAPABILITY_KIND = "restore_only_proof"


class MaintenanceDspResetCommitOnceGrant(_M110CapabilityGrantBase):
    CAPABILITY_KIND = "maintenance_dsp_reset_commit_once"


class ProcessorEnableCommitOnceGrant(_M110CapabilityGrantBase):
    CAPABILITY_KIND = "processor_enable_commit_once"


class SoundcheckChildCommitOnceGrant(_M110CapabilityGrantBase):
    CAPABILITY_KIND = "soundcheck_child_commit_once"


_M110_GRANT_TYPES = {
    grant_type.CAPABILITY_KIND: grant_type
    for grant_type in (
        RestoreOnlyProofGrant,
        MaintenanceDspResetCommitOnceGrant,
        ProcessorEnableCommitOnceGrant,
        SoundcheckChildCommitOnceGrant,
    )
}


_M110_RESTORED_PROOF_RECEIPT_FACTORY_MARKER = object()


@dataclass(frozen=True, slots=True, init=False)
class M110RestoredProofReceipt:
    """Process-private proof truth minted only after both durable terminals."""

    receipt_id: str
    proposal_id: str
    review_contract_version: int
    review_content_digest: str
    operation_digest: str
    result_digest: str
    transaction_id: str
    runtime_instance_nonce: str
    source_head: str
    source_tree: str
    route_proof_id: str
    route_artifact_sha256: str
    owner_connection_id: int
    created_at_epoch: float
    created_at_monotonic: float
    _owner_ref: Any = field(repr=False, compare=False)
    _control_ref: Any = field(repr=False, compare=False)
    _route_ref: Any = field(repr=False, compare=False)
    _factory_marker: object = field(repr=False, compare=False)

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("M110RestoredProofReceipt is backend-factory-only")

    def is_authentic(self) -> bool:
        return bool(
            self._factory_marker
            is _M110_RESTORED_PROOF_RECEIPT_FACTORY_MARKER
        )

    def __copy__(self):
        raise TypeError("m110_restored_proof_receipt_copy_forbidden")

    def __deepcopy__(self, _memo):
        raise TypeError("m110_restored_proof_receipt_copy_forbidden")

    def __reduce__(self):
        raise TypeError("m110_restored_proof_receipt_pickle_forbidden")


def new_m110_restored_proof_receipt(**values: Any) -> M110RestoredProofReceipt:
    required = {
        "proposal_id",
        "review_contract_version",
        "review_content_digest",
        "operation_digest",
        "result_digest",
        "transaction_id",
        "runtime_instance_nonce",
        "source_head",
        "source_tree",
        "route_proof_id",
        "route_artifact_sha256",
        "owner_connection_id",
        "created_at_epoch",
        "created_at_monotonic",
        "_owner_ref",
        "_control_ref",
        "_route_ref",
    }
    if set(values) != required:
        raise ValueError("m110_restored_proof_receipt_fields_mismatch")
    for field_name in (
        "proposal_id",
        "transaction_id",
        "route_proof_id",
    ):
        if not _valid_binding_text(values[field_name]):
            raise ValueError(
                f"m110_restored_proof_receipt_{field_name}_invalid"
            )
    for field_name in (
        "review_content_digest",
        "operation_digest",
        "result_digest",
        "route_artifact_sha256",
    ):
        if not _valid_sha256_digest(values[field_name]):
            raise ValueError(
                f"m110_restored_proof_receipt_{field_name}_invalid"
            )
    for field_name, length in (
        ("runtime_instance_nonce", 32),
        ("source_head", 40),
        ("source_tree", 40),
    ):
        if not _valid_lower_hex(values[field_name], length):
            raise ValueError(
                f"m110_restored_proof_receipt_{field_name}_invalid"
            )
    if values["review_contract_version"] != 2 or isinstance(
        values["review_contract_version"], bool
    ):
        raise ValueError("m110_restored_proof_receipt_review_version_invalid")
    if not (
        isinstance(values["owner_connection_id"], int)
        and not isinstance(values["owner_connection_id"], bool)
        and values["owner_connection_id"] >= 0
    ):
        raise ValueError("m110_restored_proof_receipt_owner_invalid")
    for field_name in ("created_at_epoch", "created_at_monotonic"):
        value = values[field_name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"m110_restored_proof_receipt_{field_name}_invalid"
            )
    if any(
        values[field_name] is None
        for field_name in ("_owner_ref", "_control_ref", "_route_ref")
    ):
        raise ValueError("m110_restored_proof_receipt_object_identity_required")
    receipt = object.__new__(M110RestoredProofReceipt)
    object.__setattr__(
        receipt,
        "receipt_id",
        "m110proof_v1_" + secrets.token_hex(32),
    )
    for field_name, value in values.items():
        object.__setattr__(receipt, field_name, value)
    object.__setattr__(
        receipt,
        "_factory_marker",
        _M110_RESTORED_PROOF_RECEIPT_FACTORY_MARKER,
    )
    return receipt


def _new_m110_grant(
    binding: M110CapabilityBinding,
    *,
    consumed_at_epoch: float,
    consumed_at_monotonic: float,
) -> _M110CapabilityGrantBase:
    grant_type = _M110_GRANT_TYPES[binding.capability_kind]
    grant = object.__new__(grant_type)
    object.__setattr__(grant, "binding", binding)
    object.__setattr__(grant, "consumed_at_epoch", consumed_at_epoch)
    object.__setattr__(grant, "consumed_at_monotonic", consumed_at_monotonic)
    object.__setattr__(grant, "_factory_marker", _M110_GRANT_FACTORY_MARKER)
    object.__setattr__(
        grant,
        "_execution_state",
        _new_m110_grant_execution_state(),
    )
    object.__setattr__(grant, "_execution_lock", threading.Lock())
    return grant


@dataclass(frozen=True, slots=True)
class _M110CapabilityRecord:
    token_digest: str
    binding: M110CapabilityBinding
    issued_at_epoch: float
    issued_at_monotonic: float
    latest_start_epoch: float
    latest_start_monotonic: float


class SupervisedCapabilityStore:
    """Bounded digest-only store for exact one-shot attended capabilities."""

    def __init__(
        self,
        *,
        epoch_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        self._epoch_clock = epoch_clock
        self._monotonic_clock = monotonic_clock
        self._token_factory = token_factory or (lambda: secrets.token_hex(32))
        self._records: dict[str, _M110CapabilityRecord] = {}
        self._owner_active: dict[int, str] = {}
        self._tombstones: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()
        self._closed = False
        self._issuance_disabled = False

    @staticmethod
    def _token_digest(token: Any) -> str | None:
        if not (
            isinstance(token, str)
            and re.fullmatch(r"m110cap_v1_[0-9a-f]{64}", token)
        ):
            return None
        return hashlib.sha256(token.encode("ascii")).hexdigest()

    def _tombstone_locked(self, digest: str, reason: str) -> None:
        if digest in self._tombstones:
            return
        if len(self._tombstones) >= M110_MAX_CAPABILITY_TOMBSTONES:
            self._issuance_disabled = True
            return
        self._tombstones[digest] = reason

    def _remove_record_locked(self, digest: str, *, reason: str) -> None:
        record = self._records.pop(digest, None)
        if record is None:
            return
        owner_id = record.binding.owner_connection_id
        if self._owner_active.get(owner_id) == digest:
            self._owner_active.pop(owner_id, None)
        self._tombstone_locked(digest, reason)

    def issue(
        self,
        binding: M110CapabilityBinding,
    ) -> tuple[M110IssuedCapability | None, str]:
        if type(binding) is not M110CapabilityBinding:
            return None, "m110_capability_binding_required"
        now_epoch = float(self._epoch_clock())
        now_monotonic = float(self._monotonic_clock())
        with self._lock:
            if self._closed:
                return None, "m110_capability_store_closed"
            if self._issuance_disabled or len(self._tombstones) >= (
                M110_MAX_CAPABILITY_TOMBSTONES
            ):
                return None, "m110_capability_tombstone_capacity_exhausted"
            existing = self._owner_active.get(binding.owner_connection_id)
            active_count_after_replacement = len(self._records) - bool(existing)
            if active_count_after_replacement >= M110_MAX_ACTIVE_CAPABILITIES:
                return None, "m110_capability_global_capacity_exhausted"
            if existing:
                self._remove_record_locked(
                    existing,
                    reason="replaced_by_new_owner_capability",
                )
                if self._issuance_disabled:
                    return None, "m110_capability_tombstone_capacity_exhausted"
            token = ""
            digest = ""
            for _attempt in range(8):
                candidate = self._token_factory()
                if not (
                    isinstance(candidate, str)
                    and re.fullmatch(r"[0-9a-f]{64}", candidate)
                ):
                    self._issuance_disabled = True
                    return None, "m110_capability_token_factory_invalid"
                token = M110_CAPABILITY_TOKEN_PREFIX + candidate
                digest = hashlib.sha256(token.encode("ascii")).hexdigest()
                if digest not in self._records and digest not in self._tombstones:
                    break
            else:
                self._issuance_disabled = True
                return None, "m110_capability_token_collision_exhausted"
            record = _M110CapabilityRecord(
                token_digest=digest,
                binding=binding,
                issued_at_epoch=now_epoch,
                issued_at_monotonic=now_monotonic,
                latest_start_epoch=now_epoch + M110_CAPABILITY_TTL_SECONDS,
                latest_start_monotonic=(
                    now_monotonic + M110_CAPABILITY_TTL_SECONDS
                ),
            )
            self._records[digest] = record
            self._owner_active[binding.owner_connection_id] = digest
        return (
            M110IssuedCapability(
                capability_kind=binding.capability_kind,
                token=token,
                issued_at_epoch=record.issued_at_epoch,
                latest_start_epoch=record.latest_start_epoch,
                idempotency_key=binding.idempotency_key,
                operation_digest=binding.operation_digest,
            ),
            "m110_capability_issued",
        )

    def _validate_record_locked(
        self,
        token: Any,
        current_binding: M110CapabilityBinding,
    ) -> tuple[_M110CapabilityRecord | None, str, str | None]:
        digest = self._token_digest(token)
        if digest is None:
            return None, "m110_capability_token_invalid", None
        if digest in self._tombstones:
            return None, "m110_capability_token_replayed_or_revoked", digest
        record = self._records.get(digest)
        if record is None:
            return None, "m110_capability_token_unknown", digest
        if float(self._monotonic_clock()) >= record.latest_start_monotonic:
            self._remove_record_locked(digest, reason="expired_before_start")
            return None, "m110_capability_latest_start_expired", digest
        if not record.binding.matches_exactly(current_binding):
            return None, "m110_capability_binding_mismatch", digest
        return record, "m110_capability_current", digest

    def preview(
        self,
        token: Any,
        current_binding: M110CapabilityBinding,
    ) -> tuple[bool, str]:
        if type(current_binding) is not M110CapabilityBinding:
            return False, "m110_capability_binding_required"
        with self._lock:
            if self._closed:
                return False, "m110_capability_store_closed"
            record, reason, _digest = self._validate_record_locked(
                token,
                current_binding,
            )
            return record is not None, reason

    def consume(
        self,
        token: Any,
        current_binding: M110CapabilityBinding,
    ) -> tuple[_M110CapabilityGrantBase | None, str]:
        if type(current_binding) is not M110CapabilityBinding:
            return None, "m110_capability_binding_required"
        with self._lock:
            if self._closed:
                return None, "m110_capability_store_closed"
            record, reason, digest = self._validate_record_locked(
                token,
                current_binding,
            )
            if record is None or digest is None:
                return None, reason
            consumed_at_epoch = float(self._epoch_clock())
            consumed_at_monotonic = float(self._monotonic_clock())
            self._remove_record_locked(digest, reason="consumed")
            grant = _new_m110_grant(
                record.binding,
                consumed_at_epoch=consumed_at_epoch,
                consumed_at_monotonic=consumed_at_monotonic,
            )
        return grant, "m110_capability_consumed"

    def revoke_owner(self, owner_ref: Any, *, reason: str) -> int:
        with self._lock:
            digests = [
                digest
                for digest, record in self._records.items()
                if record.binding._owner_ref is owner_ref
            ]
            for digest in digests:
                self._remove_record_locked(digest, reason=reason)
            return len(digests)

    def revoke_proposal(self, proposal_id: str, *, reason: str) -> int:
        """Revoke only unconsumed tokens bound to one exact proposal."""

        if not isinstance(proposal_id, str) or not proposal_id:
            return 0
        with self._lock:
            digests = [
                digest
                for digest, record in self._records.items()
                if record.binding.proposal_id == proposal_id
            ]
            for digest in digests:
                self._remove_record_locked(digest, reason=reason)
            return len(digests)

    def revoke_all(self, *, reason: str) -> int:
        with self._lock:
            digests = list(self._records)
            for digest in digests:
                self._remove_record_locked(digest, reason=reason)
            return len(digests)

    def active_count(self) -> int:
        with self._lock:
            return len(self._records)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_count": len(self._records),
                "active_limit": M110_MAX_ACTIVE_CAPABILITIES,
                "tombstone_count": len(self._tombstones),
                "tombstone_limit": M110_MAX_CAPABILITY_TOMBSTONES,
                "issuance_enabled": bool(
                    not self._closed and not self._issuance_disabled
                ),
                "stores_plaintext_tokens": False,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for digest in list(self._records):
                self._remove_record_locked(digest, reason="store_closed")
            self._closed = True
