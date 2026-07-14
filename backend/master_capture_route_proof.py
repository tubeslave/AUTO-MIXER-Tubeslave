"""Fixed, backend-owned Main -> USB 47/48 route proof for M110."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import stat
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


MASTER_CAPTURE_ROUTE_PROOF_SCHEMA = (
    "automixer.wing_master_capture_route_proof.v1"
)
MASTER_CAPTURE_ROUTE_REQUEST_CONFIRMATION = (
    "CAPTURE_WING_MAIN_USB_47_48_ROUTE_PROOF"
)
MASTER_CAPTURE_ROUTE_REQUEST_FIELDS = frozenset(
    {
        "type",
        "contract_version",
        "request_id",
        "approved",
        "approval_id",
        "confirmation",
        "request_nonce",
        "authority_proof",
    }
)
MASTER_CAPTURE_ROUTE_ADDRESSES = (
    "/$syscfg/$serial",
    "/$syscfg/$firmware",
    "/io/out/USB/47/grp",
    "/io/out/USB/47/in",
    "/io/out/USB/48/grp",
    "/io/out/USB/48/in",
)
MASTER_CAPTURE_ROUTE_EXPECTED = {
    "/io/out/USB/47/grp": "MAIN",
    "/io/out/USB/47/in": 0,
    "/io/out/USB/48/grp": "MAIN",
    "/io/out/USB/48/in": 1,
}
MASTER_CAPTURE_ROUTE_TTL_SECONDS = 30 * 60
MASTER_CAPTURE_ROUTE_ARTIFACT_MAX_BYTES = 64 * 1024
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_M110_WING_FIRMWARE_VERSION = re.compile(
    r"3\.0\.5(?:[-+ ][A-Za-z0-9._+-]+(?::release)?)?"
)


def is_m110_wing_firmware_version(value: Any) -> bool:
    """Accept only the verified fw3.0.5 family and exact release tag."""

    return bool(
        isinstance(value, str)
        and _M110_WING_FIRMWARE_VERSION.fullmatch(value) is not None
    )


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

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError("json_object_required")
    return value


def validate_master_capture_route_request(
    request: Mapping[str, Any],
) -> tuple[str, ...]:
    errors: list[str] = []
    if set(request) != MASTER_CAPTURE_ROUTE_REQUEST_FIELDS:
        errors.append("route_capture_request_fields_mismatch")
    if request.get("type") != "capture_wing_master_route_proof":
        errors.append("route_capture_request_type_mismatch")
    version = request.get("contract_version")
    if isinstance(version, bool) or version != 1:
        errors.append("route_capture_contract_version_mismatch")
    if request.get("approved") is not True:
        errors.append("route_capture_operator_approval_required")
    for field in ("request_id", "approval_id"):
        value = request.get(field)
        if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
            errors.append(f"route_capture_{field}_invalid")
    if request.get("confirmation") != MASTER_CAPTURE_ROUTE_REQUEST_CONFIRMATION:
        errors.append("route_capture_confirmation_mismatch")
    nonce = request.get("request_nonce")
    if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
        errors.append("route_capture_request_nonce_invalid")
    proof = request.get("authority_proof")
    if not (
        isinstance(proof, str)
        and re.fullmatch(r"hmac-sha256:[0-9a-f]{64}", proof)
    ):
        errors.append("route_capture_authority_proof_invalid")
    return tuple(errors)


def route_value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str):
        return bool(
            expected
            and isinstance(actual, str)
            and actual.strip().upper() == expected.upper()
        )
    return bool(
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and float(actual).is_integer()
        and int(actual) == expected
    )


@dataclass(frozen=True)
class MasterCaptureRouteAuthority:
    proof_id: str
    issued_at_epoch: float
    expires_at_epoch: float
    runtime_instance_nonce: str
    runtime_manifest_sha256: str
    launch_manifest_sha256: str
    connection_generation: int
    transport_generation: int
    transport_instance_id: int
    owner_connection_id: int
    emergency_generation: int
    console_serial: str
    firmware_version: str
    capture_device: str
    capture_device_index: int
    capture_channel_count: int
    capture_sample_rate_hz: int
    capture_instance_id: int
    capture_hardware_uid: str
    capture_hardware_binding_sha256: str
    capture_usb_serial: str
    capture_usb_location_id: int
    master_channels: tuple[int, int]
    artifact_path: str
    artifact_sha256: str
    evidence_sha256: str
    _transport_ref: Any = field(repr=False, compare=False)
    _capture_ref: Any = field(repr=False, compare=False)
    _capture_stream_ref: Any = field(repr=False, compare=False)
    _owner_ref: Any = field(repr=False, compare=False)

    def verified_view(self) -> dict[str, Any]:
        return {
            "verified": True,
            "reason": "process_private_master_capture_route_authority_verified",
            "proof_id": self.proof_id,
            "issued_at_epoch": self.issued_at_epoch,
            "expires_at_epoch": self.expires_at_epoch,
            "runtime_instance_nonce": self.runtime_instance_nonce,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "launch_manifest_sha256": self.launch_manifest_sha256,
            "connection_generation": self.connection_generation,
            "transport_generation": self.transport_generation,
            "transport_instance_id": self.transport_instance_id,
            "owner_connection_id": self.owner_connection_id,
            "emergency_generation": self.emergency_generation,
            "console_serial": self.console_serial,
            "firmware_version": self.firmware_version,
            "capture_device": self.capture_device,
            "capture_device_index": self.capture_device_index,
            "capture_channel_count": self.capture_channel_count,
            "capture_sample_rate_hz": self.capture_sample_rate_hz,
            "capture_instance_id": self.capture_instance_id,
            "capture_hardware_uid": self.capture_hardware_uid,
            "capture_hardware_binding_sha256": (
                self.capture_hardware_binding_sha256
            ),
            "capture_usb_serial": self.capture_usb_serial,
            "capture_usb_location_id": self.capture_usb_location_id,
            "master_channels": list(self.master_channels),
            "path": self.artifact_path,
            "sha256": self.artifact_sha256,
            "evidence_sha256": self.evidence_sha256,
            "live_authority_source": "same_process_non_serializable_object",
        }


def _ensure_private_artifact_root(root: Path) -> None:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = root.lstat()
    if root.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("M110 route artifact root must be a non-symlink directory")
    os.chmod(root, 0o700)


def write_master_capture_route_artifact(
    *,
    root: str | Path,
    proof_id: str,
    issued_at_epoch: float,
    expires_at_epoch: float,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_root = Path(root).expanduser()
    _ensure_private_artifact_root(artifact_root)
    evidence_copy = json.loads(json.dumps(evidence, allow_nan=False))
    evidence_sha256 = hashlib.sha256(canonical_json_bytes(evidence_copy)).hexdigest()
    payload = {
        "schema": MASTER_CAPTURE_ROUTE_PROOF_SCHEMA,
        "accepted": True,
        "proof_id": proof_id,
        "issued_at_utc": datetime.fromtimestamp(
            issued_at_epoch,
            timezone.utc,
        ).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "issued_at_epoch": issued_at_epoch,
        "expires_at_epoch": expires_at_epoch,
        "evidence_sha256": f"sha256:{evidence_sha256}",
        "evidence": evidence_copy,
    }
    raw = canonical_json_bytes(payload)
    if not 0 < len(raw) <= MASTER_CAPTURE_ROUTE_ARTIFACT_MAX_BYTES:
        raise RuntimeError("M110 route artifact size is invalid")
    if re.fullmatch(r"[0-9a-f]{32}", proof_id) is None:
        raise RuntimeError("M110 route proof id is invalid")
    path = artifact_root / f"{proof_id}.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = None
    created_by_this_call = False
    try:
        fd = os.open(path, flags, 0o600)
        created_by_this_call = True
        os.fchmod(fd, 0o600)
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise RuntimeError("M110 route artifact write made no progress")
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = None
        directory_fd = os.open(
            artifact_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if fd is not None:
            os.close(fd)
        if created_by_this_call:
            try:
                path.unlink()
            except OSError:
                pass
        raise
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
        "evidence_sha256": f"sha256:{evidence_sha256}",
        "schema": MASTER_CAPTURE_ROUTE_PROOF_SCHEMA,
    }


def verify_master_capture_route_artifact(
    *,
    root: str | Path,
    path: str | Path,
    expected_sha256: str,
    expected_proof_id: str,
    expected_evidence_sha256: str,
) -> dict[str, Any]:
    """Independently verify persisted audit evidence without granting authority."""

    artifact_root = Path(root).expanduser().absolute()
    artifact_path = Path(path).expanduser().absolute()
    try:
        root_metadata = artifact_root.lstat()
        if artifact_root.is_symlink() or not stat.S_ISDIR(root_metadata.st_mode):
            raise ValueError("artifact_root_not_private_directory")
        if stat.S_IMODE(root_metadata.st_mode) != 0o700:
            raise ValueError("artifact_root_mode_mismatch")
        if artifact_path.parent != artifact_root:
            raise ValueError("artifact_path_outside_fixed_root")
        if artifact_path.name != f"{expected_proof_id}.json":
            raise ValueError("artifact_filename_mismatch")
        metadata = artifact_path.lstat()
        if artifact_path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("artifact_not_regular_file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ValueError("artifact_mode_mismatch")
        if not 0 < metadata.st_size <= MASTER_CAPTURE_ROUTE_ARTIFACT_MAX_BYTES:
            raise ValueError("artifact_size_invalid")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(artifact_path, flags)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or (
                opened.st_dev,
                opened.st_ino,
            ) != (metadata.st_dev, metadata.st_ino):
                raise ValueError("artifact_identity_changed")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(fd, 8192)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > MASTER_CAPTURE_ROUTE_ARTIFACT_MAX_BYTES:
                    raise ValueError("artifact_size_invalid")
            closed_over = os.fstat(fd)
            if (
                closed_over.st_dev,
                closed_over.st_ino,
                closed_over.st_size,
            ) != (opened.st_dev, opened.st_ino, opened.st_size):
                raise ValueError("artifact_changed_during_read")
        finally:
            os.close(fd)
        raw = b"".join(chunks)
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if not hmac.compare_digest(actual_sha256, expected_sha256):
            raise ValueError("artifact_sha256_mismatch")
        payload = _strict_json_object(raw)
        if raw != canonical_json_bytes(payload):
            raise ValueError("artifact_not_canonical_json")
        if set(payload) != {
            "schema",
            "accepted",
            "proof_id",
            "issued_at_utc",
            "issued_at_epoch",
            "expires_at_epoch",
            "evidence_sha256",
            "evidence",
        }:
            raise ValueError("artifact_fields_mismatch")
        if payload.get("schema") != MASTER_CAPTURE_ROUTE_PROOF_SCHEMA:
            raise ValueError("artifact_schema_mismatch")
        if payload.get("accepted") is not True:
            raise ValueError("artifact_not_accepted_evidence")
        if payload.get("proof_id") != expected_proof_id:
            raise ValueError("artifact_proof_id_mismatch")
        issued = payload.get("issued_at_epoch")
        expires = payload.get("expires_at_epoch")
        if not (
            isinstance(issued, (int, float))
            and not isinstance(issued, bool)
            and math.isfinite(float(issued))
            and isinstance(expires, (int, float))
            and not isinstance(expires, bool)
            and math.isfinite(float(expires))
            and float(expires) > float(issued)
        ):
            raise ValueError("artifact_time_bounds_invalid")
        evidence = payload.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("artifact_evidence_object_required")
        evidence_sha256 = "sha256:" + hashlib.sha256(
            canonical_json_bytes(evidence)
        ).hexdigest()
        if not hmac.compare_digest(
            str(payload.get("evidence_sha256") or ""),
            evidence_sha256,
        ) or not hmac.compare_digest(
            expected_evidence_sha256,
            evidence_sha256,
        ):
            raise ValueError("artifact_evidence_sha256_mismatch")
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        return {
            "verified": False,
            "reason": str(exc) or type(exc).__name__,
        }
    return {
        "verified": True,
        "reason": "master_capture_route_artifact_verified_as_audit_only",
        "path": str(artifact_path),
        "sha256": expected_sha256,
        "proof_id": expected_proof_id,
        "issued_at_epoch": float(issued),
        "expires_at_epoch": float(expires),
        "evidence_sha256": expected_evidence_sha256,
        # Return the exact object decoded from the descriptor-bound read so
        # callers do not reopen the pathname and introduce a verification/use
        # race merely to bind request and hardware fields.
        "evidence": evidence,
        "live_authority_granted": False,
    }


def discard_master_capture_route_artifact(
    *,
    root: str | Path,
    path: str | Path,
) -> None:
    """Remove only a just-created artifact within the fixed private root."""

    artifact_root = Path(root).expanduser().absolute()
    artifact_path = Path(path).expanduser().absolute()
    if artifact_path.parent != artifact_root:
        raise RuntimeError("M110 artifact discard path escaped fixed root")
    metadata = artifact_path.lstat()
    if artifact_path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("M110 artifact discard target is not a regular file")
    artifact_path.unlink()
    directory_fd = os.open(
        artifact_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def new_route_proof_id() -> str:
    return uuid.uuid4().hex


def route_authority_expiry(issued_at_epoch: float | None = None) -> tuple[float, float]:
    issued = float(time.time() if issued_at_epoch is None else issued_at_epoch)
    return issued, issued + MASTER_CAPTURE_ROUTE_TTL_SECONDS
