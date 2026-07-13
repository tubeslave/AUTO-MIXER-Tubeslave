"""Backend-owned CP05 gate for every M110 physical-console contact."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


M110_CP05_GATE_RELATIVE_PATH = "ops/m110/CP05_HARDWARE_GATE.json"
M110_CP05_EVIDENCE_RELATIVE_PATH = (
    "Docs/reports/M110_CP05_SOFTWARE_VALIDATION.json"
)
M110_CP05_MARKER_PATHS = frozenset(
    {
        M110_CP05_GATE_RELATIVE_PATH,
        M110_CP05_EVIDENCE_RELATIVE_PATH,
    }
)
M110_CP05_REQUIRED_CHECKS = frozenset(
    {
        "focused_tests",
        "frontend_build_and_tests",
        "full_backend_suite",
        "independent_review",
        "no_wing_release_check",
        "released_development_lease",
        "static_scan",
    }
)
MAX_GATE_BYTES = 64 * 1024


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non_finite_json_constant:{value}")

    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("json_object_required")
    return value


def _validate_relative_path(relative_path: str) -> tuple[str, ...]:
    pure = PurePosixPath(relative_path)
    if (
        not relative_path
        or pure.is_absolute()
        or "\\" in relative_path
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != relative_path
    ):
        raise RuntimeError("m110_cp05_source_path_invalid")
    return pure.parts


def _read_fixed_regular_bytes(
    root: Path,
    relative_path: str,
    *,
    max_bytes: int = MAX_GATE_BYTES,
) -> bytes:
    parts = _validate_relative_path(relative_path)
    current = root
    for part in parts:
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError("m110_cp05_source_symlink_forbidden")
    metadata = current.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("m110_cp05_source_not_regular")
    if not 0 < metadata.st_size <= max_bytes:
        raise RuntimeError("m110_cp05_source_size_invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(current, flags)
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (metadata.st_dev, metadata.st_ino)
        ):
            raise RuntimeError("m110_cp05_source_identity_changed")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, 8192)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise RuntimeError("m110_cp05_source_size_invalid")
        closed_over = os.fstat(fd)
        if (
            closed_over.st_dev,
            closed_over.st_ino,
            closed_over.st_size,
        ) != (opened.st_dev, opened.st_ino, opened.st_size):
            raise RuntimeError("m110_cp05_source_changed_during_read")
    finally:
        os.close(fd)
    return b"".join(chunks)


def _read_canonical_json(root: Path, relative_path: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_fixed_regular_bytes(root, relative_path)
    value = _strict_json_object(raw)
    if raw != canonical_json_bytes(value) + b"\n":
        raise RuntimeError("m110_cp05_source_not_canonical")
    return value, raw


def compute_m110_cp05_source_binding(
    *,
    repo_root: str | Path,
    critical_paths: Sequence[str],
) -> dict[str, Any]:
    """Digest every committed tree entry except the two marker files."""

    root = Path(repo_root).expanduser().resolve()
    if (
        len(set(critical_paths)) != len(critical_paths)
        or not M110_CP05_MARKER_PATHS.issubset(set(critical_paths))
    ):
        raise RuntimeError("m110_cp05_critical_path_contract_invalid")
    clean = subprocess.run(
        ["/usr/bin/git", "diff", "--quiet", "HEAD", "--"],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    if clean.returncode != 0:
        raise RuntimeError("m110_cp05_tracked_tree_not_clean")
    completed = subprocess.run(
        [
            "/usr/bin/git",
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            "HEAD",
        ],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0 or not 0 < len(completed.stdout) <= 16 * 1024 * 1024:
        raise RuntimeError("m110_cp05_git_tree_unavailable")
    rows = []
    for raw_entry in completed.stdout.split(b"\0"):
        if not raw_entry:
            continue
        try:
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
            relative_path = raw_path.decode("utf-8")
        except (UnicodeError, ValueError) as exc:
            raise RuntimeError("m110_cp05_git_tree_entry_invalid") from exc
        _validate_relative_path(relative_path)
        if relative_path in M110_CP05_MARKER_PATHS:
            continue
        if not (
            re.fullmatch(r"[0-7]{6}", mode)
            and object_type in {"blob", "commit"}
            and re.fullmatch(r"[0-9a-f]{40,64}", object_id)
        ):
            raise RuntimeError("m110_cp05_git_tree_entry_invalid")
        rows.append(
            {
                "mode": mode,
                "object_id": object_id,
                "object_type": object_type,
                "path": relative_path,
            }
        )
    if not rows:
        raise RuntimeError("m110_cp05_git_tree_empty")
    manifest = {
        "schema": "automixer.m110_cp05_git_tree_binding.v1",
        "entries": rows,
    }
    digest = "sha256:" + hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    return {
        "schema": manifest["schema"],
        "sha256": digest,
        "tracked_entry_count": len(rows),
    }


def verify_m110_cp05_hardware_gate(
    *,
    repo_root: str | Path,
    critical_paths: Sequence[str],
) -> dict[str, Any]:
    """Validate CP05 checks and bind them to the exact current critical source."""

    root = Path(repo_root).expanduser().resolve()
    gate, _ = _read_canonical_json(root, M110_CP05_GATE_RELATIVE_PATH)
    if set(gate) != {
        "schema",
        "checkpoint",
        "hardware_enabled",
        "software_evidence_sha256",
    } or not (
        gate.get("schema") == "automixer.m110_cp05_hardware_gate.v1"
        and gate.get("checkpoint") == "M110-CP05"
        and gate.get("hardware_enabled") is True
        and isinstance(gate.get("software_evidence_sha256"), str)
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            gate["software_evidence_sha256"],
        )
    ):
        raise RuntimeError("m110_cp05_hardware_gate_closed")
    evidence, evidence_raw = _read_canonical_json(
        root,
        M110_CP05_EVIDENCE_RELATIVE_PATH,
    )
    evidence_sha = "sha256:" + hashlib.sha256(evidence_raw).hexdigest()
    if not hmac.compare_digest(gate["software_evidence_sha256"], evidence_sha):
        raise RuntimeError("m110_cp05_software_evidence_digest_mismatch")
    if set(evidence) != {
        "schema",
        "status",
        "checks",
        "source_binding_sha256",
    } or not (
        evidence.get("schema")
        == "automixer.m110_cp05_software_validation.v1"
        and evidence.get("status") == "passed"
        and isinstance(evidence.get("checks"), Mapping)
        and set(evidence["checks"]) == M110_CP05_REQUIRED_CHECKS
        and all(
            evidence["checks"].get(name) is True
            for name in M110_CP05_REQUIRED_CHECKS
        )
        and isinstance(evidence.get("source_binding_sha256"), str)
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            evidence["source_binding_sha256"],
        )
    ):
        raise RuntimeError("m110_cp05_software_evidence_not_passed")
    binding = compute_m110_cp05_source_binding(
        repo_root=root,
        critical_paths=critical_paths,
    )
    if not hmac.compare_digest(
        evidence["source_binding_sha256"],
        binding["sha256"],
    ):
        raise RuntimeError("m110_cp05_source_binding_mismatch")
    return {
        "verified": True,
        "reason": "m110_cp05_hardware_gate_verified",
        "software_evidence_sha256": evidence_sha,
        "source_binding_sha256": binding["sha256"],
        "source_binding_tracked_entry_count": binding[
            "tracked_entry_count"
        ],
    }
