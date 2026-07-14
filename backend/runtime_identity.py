"""Immutable source/runtime identity helpers for attended safety proofs."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any, Dict


RUNTIME_IDENTITY_SCHEMA = "automixer.backend_runtime_identity.v2"
RUNTIME_SOURCE_SCHEMA = "automixer.backend_runtime_source.v1"
RUNTIME_CODE_SCOPE = "automixer.live_safety_critical.v1"
RUNTIME_CONTENT_SCOPE = "automixer.production_source_inputs.v2"
PACKAGED_RUNTIME_SOURCE_MANIFEST_SCHEMA = (
    "automixer.packaged_runtime_source_manifest.v1"
)
PACKAGED_RUNTIME_SOURCE_MANIFEST_FILENAME = (
    "automixer-runtime-source-manifest.json"
)
PACKAGED_RUNTIME_SOURCE_OBSERVATION = "packaged_build_manifest"
_MAX_PACKAGED_RUNTIME_SOURCE_MANIFEST_BYTES = 1024 * 1024
RUNTIME_CRITICAL_SOURCE_PATHS = (
    "backend/auth.py",
    "backend/audio_capture.py",
    "backend/auto_soundcheck_engine.py",
    "backend/contracts/m110_m006_restore_20260711.json",
    "backend/decision_queue_wire_contract.py",
    "backend/handlers/__init__.py",
    "backend/handlers/automation_handlers.py",
    "backend/handlers/connection_handlers.py",
    "backend/handlers/mixer_handlers.py",
    "backend/handlers/product_state_handlers.py",
    "backend/handlers/soundcheck_handlers.py",
    "backend/m110_cp05_hardware_gate.py",
    "backend/m110_manual_write_wal.py",
    "backend/m110_m006_restore.py",
    "backend/m110_processor_enable.py",
    "backend/maintenance_dsp_reset.py",
    "backend/master_capture_route_proof.py",
    "backend/mixer_client_base.py",
    "backend/operator_mode_policy.py",
    "backend/operator_proposal_queue.py",
    "backend/operator_proposal_result_store.py",
    "backend/osc/enhanced_osc_client.py",
    "backend/runtime_identity.py",
    "backend/server.py",
    "backend/soundcheck_mix_readiness.py",
    "backend/soundcheck_transaction_plan.py",
    "backend/supervised_console_parameter_registry.py",
    "backend/supervised_console_transaction.py",
    "backend/supervised_family_proof_plan.py",
    "backend/supervised_proof_capability.py",
    "backend/wing_capture_hardware_identity.py",
    "backend/wing_client.py",
    "backend/wing_supervised_console_adapters.py",
    "backend/ws_transport.py",
    "tools/wing_attended_soundcheck_runner.py",
    "tools/wing_supervised_family_proof_runner.py",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _git_bytes(repo_root: Path, *arguments: str) -> bytes:
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
            f"git {' '.join(arguments)} failed ({completed.returncode}): "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return completed.stdout


def _critical_file_identity(repo_root: Path, relative_path: str) -> Dict[str, Any]:
    path = repo_root / relative_path
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"runtime identity source is missing: {relative_path}") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"runtime identity source must be a regular non-symlink file: {relative_path}")
    value = path.read_bytes()
    return {
        "path": relative_path,
        "exists": True,
        "mode": stat.S_IMODE(metadata.st_mode),
        "size": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
    }


def capture_source_runtime_components(repo_root: str | Path) -> Dict[str, Any]:
    """Capture current Git and safety-critical source bytes without mutation."""

    root = Path(repo_root).resolve()
    status = _git_bytes(root, "status", "--porcelain=v1", "-uall")
    tracked_diff = _git_bytes(root, "diff", "HEAD", "--binary")
    files = [
        _critical_file_identity(root, relative_path)
        for relative_path in sorted(RUNTIME_CRITICAL_SOURCE_PATHS)
    ]
    files_sha256 = canonical_json_sha256(files)
    source = {
        "schema": RUNTIME_SOURCE_SCHEMA,
        "observation": "git_process_start",
        "head": _git_bytes(root, "rev-parse", "HEAD").decode().strip(),
        "tree": _git_bytes(root, "rev-parse", "HEAD^{tree}").decode().strip(),
        "worktree_dirty": bool(status.strip()),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "source_content_sha256": files_sha256,
        "content_scope": RUNTIME_CONTENT_SCOPE,
    }
    code = {
        "scope": RUNTIME_CODE_SCOPE,
        "files": files,
        "files_sha256": files_sha256,
    }
    return {"source": source, "code": code}


def _packaged_source_manifest_sha256(manifest: Dict[str, Any]) -> str:
    canonical = dict(manifest)
    canonical.pop("manifest_sha256", None)
    return canonical_json_sha256(canonical)


def _validate_packaged_runtime_components(components: Dict[str, Any]) -> None:
    source = components.get("source")
    code = components.get("code")
    if not isinstance(source, dict) or not isinstance(code, dict):
        raise RuntimeError("Packaged runtime source manifest components are invalid")
    files = code.get("files")
    if not isinstance(files, list):
        raise RuntimeError("Packaged runtime source manifest file list is invalid")
    expected_paths = sorted(RUNTIME_CRITICAL_SOURCE_PATHS)
    if [row.get("path") for row in files if isinstance(row, dict)] != expected_paths:
        raise RuntimeError("Packaged runtime source manifest critical paths mismatch")
    for row in files:
        if (
            not isinstance(row, dict)
            or row.get("exists") is not True
            or not isinstance(row.get("mode"), int)
            or not 0 <= row["mode"] <= 0o777
            or not isinstance(row.get("size"), int)
            or row["size"] < 0
            or not valid_lower_hex(row.get("sha256"), 64)
        ):
            raise RuntimeError("Packaged runtime source manifest file identity is invalid")
    files_sha256 = canonical_json_sha256(files)
    if (
        source.get("schema") != RUNTIME_SOURCE_SCHEMA
        or source.get("observation") != PACKAGED_RUNTIME_SOURCE_OBSERVATION
        or not valid_lower_hex(source.get("head"), 40)
        or not valid_lower_hex(source.get("tree"), 40)
        or not isinstance(source.get("worktree_dirty"), bool)
        or not valid_lower_hex(source.get("status_sha256"), 64)
        or not valid_lower_hex(source.get("tracked_diff_sha256"), 64)
        or source.get("source_content_sha256") != files_sha256
        or source.get("content_scope") != RUNTIME_CONTENT_SCOPE
        or code.get("scope") != RUNTIME_CODE_SCOPE
        or code.get("files_sha256") != files_sha256
    ):
        raise RuntimeError("Packaged runtime source manifest content is invalid")


def build_packaged_runtime_source_manifest(
    components: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the immutable source manifest embedded in a packaged backend."""

    packaged_components = json.loads(json.dumps(components))
    source = packaged_components.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("Packaged runtime source manifest source is invalid")
    source["observation"] = PACKAGED_RUNTIME_SOURCE_OBSERVATION
    _validate_packaged_runtime_components(packaged_components)
    manifest = {
        "schema": PACKAGED_RUNTIME_SOURCE_MANIFEST_SCHEMA,
        "source": packaged_components["source"],
        "code": packaged_components["code"],
    }
    manifest["manifest_sha256"] = _packaged_source_manifest_sha256(manifest)
    return manifest


def load_packaged_runtime_source_components(
    manifest_path: str | Path,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load and validate the build-time manifest from a frozen backend bundle."""

    path = Path(manifest_path)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError("Packaged runtime source manifest is missing") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(
            "Packaged runtime source manifest must be a regular non-symlink file"
        )
    if metadata.st_size <= 0 or metadata.st_size > _MAX_PACKAGED_RUNTIME_SOURCE_MANIFEST_BYTES:
        raise RuntimeError("Packaged runtime source manifest size is invalid")
    value = path.read_bytes()
    try:
        manifest = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Packaged runtime source manifest JSON is invalid") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("Packaged runtime source manifest root is invalid")
    if (
        manifest.get("schema") != PACKAGED_RUNTIME_SOURCE_MANIFEST_SCHEMA
        or not valid_lower_hex(manifest.get("manifest_sha256"), 64)
        or manifest.get("manifest_sha256")
        != _packaged_source_manifest_sha256(manifest)
    ):
        raise RuntimeError("Packaged runtime source manifest binding is invalid")
    components = {
        "source": manifest.get("source"),
        "code": manifest.get("code"),
    }
    _validate_packaged_runtime_components(components)
    binding = {
        "mode": "packaged_build_manifest",
        "packaged_manifest_schema": PACKAGED_RUNTIME_SOURCE_MANIFEST_SCHEMA,
        "packaged_manifest_name": path.name,
        "packaged_manifest_size": len(value),
        "packaged_manifest_file_sha256": hashlib.sha256(value).hexdigest(),
        "packaged_manifest_sha256": manifest["manifest_sha256"],
        "bound_before_process_start": True,
        "proof_eligible": False,
    }
    return json.loads(json.dumps(components)), binding


def runtime_identity_manifest_sha256(identity: Dict[str, Any]) -> str:
    canonical = dict(identity)
    canonical.pop("manifest_sha256", None)
    canonical.pop("request_challenge_nonce", None)
    return canonical_json_sha256(canonical)


def valid_lower_hex(value: Any, length: int) -> bool:
    text = str(value or "")
    return len(text) == length and all(character in "0123456789abcdef" for character in text)


def consume_proof_launch_binding_environment() -> Dict[str, Any]:
    """Consume a runner-provided pre-process binding without exposing raw env."""

    names = {
        "launch_nonce": "AUTOMIXER_M005_LAUNCH_NONCE",
        "prestart_provenance_schema": "AUTOMIXER_M005_PRESTART_PROVENANCE_SCHEMA",
        "prestart_provenance_sha256": "AUTOMIXER_M005_PRESTART_PROVENANCE_SHA256",
        "prestart_provenance_size": "AUTOMIXER_M005_PRESTART_PROVENANCE_SIZE",
    }
    values = {field: os.environ.pop(name, None) for field, name in names.items()}
    supplied = any(value is not None for value in values.values())
    valid = (
        supplied
        and valid_lower_hex(values["launch_nonce"], 32)
        and values["prestart_provenance_schema"]
        == "automixer.m005_current_release_provenance.v1"
        and valid_lower_hex(values["prestart_provenance_sha256"], 64)
    )
    try:
        size = int(values["prestart_provenance_size"] or 0)
    except (TypeError, ValueError):
        size = 0
    valid = valid and size > 0
    if supplied and not valid:
        raise RuntimeError("Invalid M005 process-start launch binding")
    return {
        "mode": "proof_runner" if valid else "unbound",
        "launch_nonce": values["launch_nonce"] if valid else None,
        "prestart_provenance_schema": (
            values["prestart_provenance_schema"] if valid else None
        ),
        "prestart_provenance_size": size if valid else None,
        "prestart_provenance_sha256": (
            values["prestart_provenance_sha256"] if valid else None
        ),
        "bound_before_process_start": bool(valid),
        "proof_eligible": bool(valid),
    }
