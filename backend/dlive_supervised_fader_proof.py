"""Restore-only attended dLive input-fader proof control plane.

This module is intentionally separate from the normal server path.  It owns
one exact, downward-only input-fader proof, a per-user console writer lock, a
hash-chained durable WAL, fresh physical readback, compare-and-swap rollback,
and teardown.  It grants no standing dLive write authority.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import stat
import time
from typing import Any, Callable, Mapping
import uuid

from dlive_client import (
    _DLIVE_SUPERVISED_WRITE_SEAL,
    DLiveClient,
    build_nrpn,
    db_to_fader_value,
    validate_dlive_connection_config,
    FADER_0DB,
    FADER_PARAMETER_ID,
    FADER_NEG_INF,
)
from dlive_midi_capture_evidence import DLIVE_MIDI_CAPTURE_REVIEW_SCHEMA


DLIVE_FADER_PROOF_CONFIRMATION = "APPLY_AND_ROLLBACK_DLIVE_FADER_PROOF"
DLIVE_FADER_PROOF_CONFIRMATION_SCHEMA = (
    "automixer.dlive_fader_proof_confirmation_binding.v1"
)
DLIVE_FADER_PROOF_SCHEMA = "automixer.dlive_fader_restore_proof.v1"
DLIVE_FADER_PROOF_WAL_SCHEMA = "automixer.dlive_fader_proof_wal.v1"
MAX_APPROVAL_LIFETIME_SECONDS = 300.0
MAX_FADER_CUT_DB = 6.0
DLIVE_READONLY_PROBE_SCHEMA = "automixer.dlive_attended_readonly_probe.v1"

TERMINAL_WAL_PHASES = frozenset(
    {
        "proof_passed",
        "blocked_no_send",
        "restored_after_unverified_apply",
        "manual_review_required",
        "recovery_restored",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _per_user_runtime_root() -> Path:
    """Match the canonical server runtime root without trusting ``HOME``."""

    return Path(pwd.getpwuid(os.getuid()).pw_dir) / ".automixer" / "runtime"


def _reject_non_test_runtime_override(
    wal_path: Path | None,
    writer_lock_path: Path | None,
) -> None:
    if (wal_path is not None or writer_lock_path is not None) and not os.environ.get(
        "PYTEST_CURRENT_TEST"
    ):
        raise ValueError("dlive_runtime_path_override_allowed_only_under_pytest")


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field}_must_be_finite_number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}_must_be_finite_number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field}_must_be_finite_number")
    return number


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _same_fader_value(left_db: Any, right_db: Any) -> bool:
    try:
        return db_to_fader_value(float(left_db)) == db_to_fader_value(float(right_db))
    except (TypeError, ValueError):
        return False


def _contains_sensitive_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in ("password", "secret", "token")):
                return True
            if _contains_sensitive_key(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_sensitive_key(item) for item in value)
    return False


def build_dlive_fader_proof_confirmation(
    approval: "DLiveFaderProofApproval",
) -> str:
    """Bind the literal operator confirmation to the exact planned proof."""

    endpoint = validate_dlive_connection_config(
        host=approval.host,
        endpoint=approval.endpoint,
        tls=approval.tls,
        port=approval.port,
        midi_base_channel=approval.midi_base_channel,
        user_profile=approval.user_profile,
    )
    readonly_sha256 = str(approval.readonly_artifact_sha256 or "").lower()
    if readonly_sha256.startswith("sha256:"):
        readonly_sha256 = readonly_sha256.removeprefix("sha256:")
    binding = {
        "schema": DLIVE_FADER_PROOF_CONFIRMATION_SCHEMA,
        "operation": DLIVE_FADER_PROOF_CONFIRMATION,
        "approval_id": approval.approval_id,
        "endpoint": endpoint,
        "expected_interface": approval.expected_interface,
        "input_channel": approval.input_channel,
        "expected_name": approval.expected_name,
        "operator_attestation": {
            "model": approval.operator_attested_model,
            "firmware": approval.operator_attested_firmware,
            "show": approval.operator_attested_show,
        },
        "target_db": float(approval.target_db),
        "target_fader_raw": db_to_fader_value(float(approval.target_db)),
        "max_cut_db": float(approval.max_cut_db),
        "readonly_artifact": {
            "path": approval.readonly_artifact_path,
            "sha256": f"sha256:{readonly_sha256}",
        },
        "security_attestation": {
            "state_attested": approval.security_state_attested,
            "mode": approval.security_mode_attested,
            "changed_for_setup": approval.security_changed_for_setup,
            "reboot_after_change_confirmed": (
                approval.reboot_after_security_change_confirmed
            ),
        },
        "wire_evidence": {
            "path": approval.wire_evidence_path,
            "sha256": approval.normalized_wire_evidence_sha256(),
        },
    }
    digest = hashlib.sha256(_canonical_json_bytes(binding)).hexdigest()
    return f"{DLIVE_FADER_PROOF_CONFIRMATION}:sha256:{digest}"


@dataclass(frozen=True)
class DLiveFaderProofApproval:
    approval_id: str
    issued_at_epoch: float
    expires_at_epoch: float
    host: str
    endpoint: str
    port: int | None
    tls: bool
    midi_base_channel: int
    user_profile: int
    expected_interface: str
    input_channel: int
    expected_name: str
    operator_attested_model: str
    operator_attested_firmware: str
    operator_attested_show: str
    target_db: float
    max_cut_db: float
    pa_outputs_contained: bool
    require_channel_muted: bool
    receive_midi_confirmed_enabled: bool
    readonly_artifact_path: str
    readonly_artifact_sha256: str
    confirmation: str
    security_state_attested: bool = False
    security_mode_attested: str = ""
    security_changed_for_setup: bool = False
    reboot_after_security_change_confirmed: bool = False
    wire_evidence_path: str = ""
    wire_evidence_sha256: str = ""

    def normalized_wire_evidence_sha256(self) -> str:
        digest = str(self.wire_evidence_sha256 or "").lower()
        if digest.startswith("sha256:"):
            digest = digest.removeprefix("sha256:")
        return f"sha256:{digest}"

    def validate(self, *, now_epoch: float | None = None) -> dict[str, Any]:
        now = time.time() if now_epoch is None else float(now_epoch)
        issued = _finite_number(self.issued_at_epoch, "issued_at_epoch")
        expires = _finite_number(self.expires_at_epoch, "expires_at_epoch")
        if not self.approval_id or self.approval_id != self.approval_id.strip():
            raise ValueError("approval_id_required")
        if issued > now + 2.0 or expires <= now:
            raise ValueError("operator_approval_expired_or_not_yet_valid")
        if expires <= issued or expires - issued > MAX_APPROVAL_LIFETIME_SECONDS:
            raise ValueError("operator_approval_lifetime_exceeds_300_seconds")
        validated = validate_dlive_connection_config(
            host=self.host,
            endpoint=self.endpoint,
            tls=self.tls,
            port=self.port,
            midi_base_channel=self.midi_base_channel,
            user_profile=self.user_profile,
        )
        if not self.expected_interface or not self.expected_interface.replace("_", "").isalnum():
            raise ValueError("expected_route_interface_invalid")
        if isinstance(self.input_channel, bool) or not isinstance(self.input_channel, int):
            raise ValueError("input_channel_must_be_integer")
        if not 1 <= self.input_channel <= 128:
            raise ValueError("input_channel_out_of_range_1_128")
        for field, value in (
            ("expected_name", self.expected_name),
            ("operator_attested_model", self.operator_attested_model),
            ("operator_attested_firmware", self.operator_attested_firmware),
            ("operator_attested_show", self.operator_attested_show),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field}_required")
        target = _finite_number(self.target_db, "target_db")
        maximum_cut = _finite_number(self.max_cut_db, "max_cut_db")
        if target < -54.0 or target > 0.0:
            raise ValueError("target_db_out_of_restore_proof_range")
        if db_to_fader_value(target) == FADER_NEG_INF:
            raise ValueError("target_db_must_not_map_to_negative_infinity")
        if maximum_cut <= 0.0 or maximum_cut > MAX_FADER_CUT_DB:
            raise ValueError("max_cut_db_out_of_range_0_6")
        if self.pa_outputs_contained is not True:
            raise ValueError("pa_outputs_must_be_physically_contained")
        if self.require_channel_muted is not True:
            raise ValueError("proof_requires_muted_input_channel")
        if self.receive_midi_confirmed_enabled is not True:
            raise ValueError("proof_requires_receive_midi_enabled")
        if self.security_state_attested is not True:
            raise ValueError("proof_requires_effective_security_state_attestation")
        if self.tls is not True:
            raise ValueError("proof_requires_tls_authenticated_transport")
        required_security_mode = "tls_authenticated"
        if self.security_mode_attested != required_security_mode:
            raise ValueError(
                f"proof_security_mode_mismatch:expected_{required_security_mode}"
            )
        if not isinstance(self.security_changed_for_setup, bool):
            raise ValueError("security_changed_for_setup_must_be_boolean")
        if not isinstance(self.reboot_after_security_change_confirmed, bool):
            raise ValueError("reboot_after_security_change_confirmed_must_be_boolean")
        if (
            self.security_changed_for_setup
            and not self.reboot_after_security_change_confirmed
        ):
            raise ValueError("proof_requires_reboot_after_security_change")
        if (
            not isinstance(self.readonly_artifact_path, str)
            or not self.readonly_artifact_path
            or self.readonly_artifact_path != self.readonly_artifact_path.strip()
            or not Path(self.readonly_artifact_path).is_absolute()
        ):
            raise ValueError("readonly_artifact_absolute_path_required")
        readonly_sha256 = str(self.readonly_artifact_sha256 or "").lower()
        if readonly_sha256.startswith("sha256:"):
            readonly_sha256 = readonly_sha256.removeprefix("sha256:")
        if len(readonly_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in readonly_sha256
        ):
            raise ValueError("readonly_artifact_sha256_invalid")
        if (
            not isinstance(self.wire_evidence_path, str)
            or not self.wire_evidence_path
            or self.wire_evidence_path != self.wire_evidence_path.strip()
            or not Path(self.wire_evidence_path).is_absolute()
        ):
            raise ValueError("wire_evidence_absolute_path_required")
        wire_sha256 = self.normalized_wire_evidence_sha256().removeprefix("sha256:")
        if len(wire_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in wire_sha256
        ):
            raise ValueError("wire_evidence_sha256_invalid")
        required_confirmation = build_dlive_fader_proof_confirmation(self)
        if self.confirmation != required_confirmation:
            raise ValueError("artifact_target_bound_operator_confirmation_required")
        return {
            "schema": "automixer.dlive_fader_proof_approval.v1",
            "approval_id": self.approval_id,
            "issued_at_epoch": issued,
            "expires_at_epoch": expires,
            "endpoint": validated,
            "expected_interface": self.expected_interface,
            "input_channel": self.input_channel,
            "expected_name": self.expected_name,
            "operator_attestation": {
                "model": self.operator_attested_model,
                "firmware": self.operator_attested_firmware,
                "show": self.operator_attested_show,
                "machine_verified": False,
            },
            "target_db": target,
            "max_cut_db": maximum_cut,
            "pa_outputs_contained": True,
            "require_channel_muted": True,
            "receive_midi_confirmed_enabled": True,
            "security_attestation": {
                "state_attested": True,
                "mode": self.security_mode_attested,
                "changed_for_setup": self.security_changed_for_setup,
                "reboot_after_change_confirmed": (
                    self.reboot_after_security_change_confirmed
                ),
            },
            "readonly_artifact": {
                "path": self.readonly_artifact_path,
                "sha256": f"sha256:{readonly_sha256}",
            },
            "wire_evidence": {
                "path": self.wire_evidence_path,
                "sha256": f"sha256:{wire_sha256}",
            },
            "operator_confirmation_recorded": True,
            "operator_confirmation_binding_schema": (
                DLIVE_FADER_PROOF_CONFIRMATION_SCHEMA
            ),
            "operator_confirmation_sha256": _sha256_bytes(
                required_confirmation.encode("utf-8")
            ),
            "standing_live_authority": False,
        }


_AUTHORITY_SEAL = object()


def _validate_bounded_fader_cut(
    preimage_db: Any,
    target_db: Any,
    max_cut_db: Any,
) -> dict[str, Any]:
    """Validate the exact 7-bit downward proof move without transport."""

    old = _finite_number(preimage_db, "preimage_db")
    target = _finite_number(target_db, "target_db")
    maximum_cut = _finite_number(max_cut_db, "max_cut_db")
    old_raw = db_to_fader_value(old)
    target_raw = db_to_fader_value(target)
    if old_raw > FADER_0DB:
        raise ValueError("preimage_above_zero_db_not_supported")
    if target_raw > old_raw:
        raise ValueError("fader_increase_not_allowed")
    if target_raw == old_raw:
        raise ValueError("proof_target_equals_preimage")
    # Apply the operator's requested dB bound without tolerance. Exact
    # transport/readback comparisons remain in raw 7-bit space.
    if old - target > maximum_cut:
        raise ValueError("fader_cut_exceeds_operator_bound")
    return {
        "preimage_db": old,
        "preimage_fader_raw": old_raw,
        "target_db": target,
        "target_fader_raw": target_raw,
        "cut_db": old - target,
        "max_cut_db": maximum_cut,
    }


class _DLiveFaderProofAuthority:
    """Process-private apply-once/restore-once authority."""

    def __init__(
        self,
        seal: object,
        *,
        client: DLiveClient,
        approval: DLiveFaderProofApproval,
        now_fn: Callable[[], float],
    ) -> None:
        if seal is not _AUTHORITY_SEAL:
            raise TypeError("dlive_fader_proof_authority_is_process_private")
        self._client = client
        self._dlive_transport_seal = _DLIVE_SUPERVISED_WRITE_SEAL
        self._approval = approval
        self._now_fn = now_fn
        self._preimage_db: float | None = None
        self._state = "issued"

    @property
    def state(self) -> str:
        return self._state

    def bind_preimage(self, preimage_db: float) -> None:
        if self._state != "issued":
            raise RuntimeError("dlive_authority_preimage_already_bound")
        cut = _validate_bounded_fader_cut(
            preimage_db,
            self._approval.target_db,
            self._approval.max_cut_db,
        )
        self._preimage_db = cut["preimage_db"]
        self._state = "preimage_bound"

    def consume_fader_write(
        self,
        *,
        client: DLiveClient,
        channel: int,
        value_db: float,
        phase: str,
    ) -> bool:
        if client is not self._client or channel != self._approval.input_channel:
            return False
        if phase == "apply":
            if self._state != "preimage_bound":
                return False
            if self._now_fn() >= float(self._approval.expires_at_epoch):
                self._state = "expired"
                return False
            if not _same_fader_value(value_db, self._approval.target_db):
                return False
            self._state = "apply_send_attempted"
            return True
        if phase == "rollback":
            if self._state not in {"apply_send_attempted", "apply_verified"}:
                return False
            if self._preimage_db is None or not _same_fader_value(value_db, self._preimage_db):
                return False
            self._state = "rollback_send_attempted"
            return True
        return False

    def mark_apply_verified(self) -> None:
        if self._state == "apply_send_attempted":
            self._state = "apply_verified"

    def mark_restored(self) -> None:
        if self._state in {"apply_send_attempted", "apply_verified", "rollback_send_attempted"}:
            self._state = "restored"

    def revoke(self) -> None:
        self._state = "revoked"
        self._client = None  # type: ignore[assignment]


class _DLiveRollbackRecoveryAuthority:
    """One-shot exact-preimage authority reconstructed from a valid WAL."""

    def __init__(
        self,
        seal: object,
        *,
        client: DLiveClient,
        channel: int,
        preimage_db: float,
    ) -> None:
        if seal is not _AUTHORITY_SEAL:
            raise TypeError("dlive_rollback_recovery_authority_is_process_private")
        self._client = client
        self._dlive_transport_seal = _DLIVE_SUPERVISED_WRITE_SEAL
        self._channel = channel
        self._preimage_db = preimage_db
        self._used = False

    def consume_fader_write(
        self,
        *,
        client: DLiveClient,
        channel: int,
        value_db: float,
        phase: str,
    ) -> bool:
        if (
            self._used
            or client is not self._client
            or channel != self._channel
            or phase != "rollback"
            or not _same_fader_value(value_db, self._preimage_db)
        ):
            return False
        self._used = True
        return True

    def revoke(self) -> None:
        self._used = True
        self._client = None  # type: ignore[assignment]


class DLiveProofWal:
    """Small hash-chained, file-and-directory-fsynced JSONL WAL."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.records = self._load_and_validate()

    @staticmethod
    def _ensure_directory(path: Path) -> None:
        absolute = Path(path).expanduser().absolute()
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            candidate = current / part
            try:
                metadata = os.lstat(candidate)
            except FileNotFoundError:
                os.mkdir(candidate, 0o700)
                DLiveProofWal._fsync_directory(current)
                metadata = os.lstat(candidate)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise OSError("dlive_runtime_path_component_not_real_directory")
            current = candidate
        os.chmod(absolute, 0o700)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _load_and_validate(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        metadata = os.lstat(self.path)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise ValueError("dlive_wal_path_not_regular_file")
        records: list[dict[str, Any]] = []
        previous: str | None = None
        for expected_sequence, raw in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("dlive_wal_invalid_json") from exc
            if not isinstance(record, dict) or record.get("schema") != DLIVE_FADER_PROOF_WAL_SCHEMA:
                raise ValueError("dlive_wal_schema_invalid")
            claimed = record.get("record_sha256")
            unhashed = {key: value for key, value in record.items() if key != "record_sha256"}
            if (
                record.get("sequence") != expected_sequence
                or record.get("previous_record_sha256") != previous
                or claimed != _sha256_bytes(_canonical_json_bytes(unhashed))
            ):
                raise ValueError("dlive_wal_hash_chain_invalid")
            records.append(record)
            previous = claimed
        return records

    def incomplete_transaction_ids(self) -> tuple[str, ...]:
        latest: dict[str, str] = {}
        for record in self.records:
            latest[str(record["transaction_id"])] = str(record["phase"])
        return tuple(sorted(key for key, phase in latest.items() if phase not in TERMINAL_WAL_PHASES))

    def consumed_approval_ids(self) -> frozenset[str]:
        return frozenset(
            str(record.get("payload", {}).get("approval_id"))
            for record in self.records
            if record.get("phase") == "prepared"
            and record.get("payload", {}).get("approval_id")
        )

    def append(
        self,
        *,
        transaction_id: str,
        phase: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not transaction_id or not phase or _contains_sensitive_key(payload):
            raise ValueError("dlive_wal_event_invalid_or_sensitive")
        self._ensure_directory(self.path.parent)
        sequence = len(self.records) + 1
        event = {
            "schema": DLIVE_FADER_PROOF_WAL_SCHEMA,
            "sequence": sequence,
            "timestamp_utc": _utc_now(),
            "transaction_id": transaction_id,
            "phase": phase,
            "previous_record_sha256": (
                self.records[-1]["record_sha256"] if self.records else None
            ),
            "payload": dict(payload),
        }
        event["record_sha256"] = _sha256_bytes(_canonical_json_bytes(event))
        record_bytes = _canonical_json_bytes(event) + b"\n"
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags, 0o600)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise OSError("dlive_wal_file_not_private_regular_file")
            os.fchmod(descriptor, 0o600)
            written = os.write(descriptor, record_bytes)
            if written != len(record_bytes):
                raise OSError("dlive_wal_short_write")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._fsync_directory(self.path.parent)
        self.records.append(event)
        return {
            "path": str(self.path),
            "sequence": sequence,
            "phase": phase,
            "record_sha256": event["record_sha256"],
        }


class DLiveConsoleWriterLock:
    """Non-blocking per-user lock shared with the WING server."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._descriptor: int | None = None

    def acquire(self) -> bool:
        DLiveProofWal._ensure_directory(self.path.parent)
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags, 0o600)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise OSError("dlive_writer_lock_not_private_regular_file")
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(descriptor)
            return False
        self._descriptor = descriptor
        return True

    def release(self) -> None:
        if self._descriptor is None:
            return
        try:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._descriptor)
            self._descriptor = None


def _read_received(client: Any, address: str, timeout_sec: float) -> dict[str, Any]:
    result = client.query_parameter(address, timeout_sec=timeout_sec)
    if not isinstance(result, Mapping):
        raise RuntimeError("dlive_readback_response_not_object")
    normalized = dict(result)
    if (
        normalized.get("status") != "received"
        or normalized.get("address") != address
        or normalized.get("confirmed_round_trips") != 2
    ):
        raise RuntimeError(f"fresh_double_readback_failed:{address}")
    return normalized


def _packet_evidence(client: DLiveClient, channel: int, value_db: float) -> dict[str, Any]:
    midi_channel, selector = client._selector(channel, "input")
    packet = build_nrpn(
        midi_channel,
        selector,
        FADER_PARAMETER_ID,
        db_to_fader_value(value_db),
    )
    return {
        "packet_hex": packet.hex(),
        "packet_sha256": _sha256_bytes(packet),
        "midi_channel": midi_channel,
        "selector": selector,
        "fader_raw": db_to_fader_value(value_db),
    }


def _validate_bound_readonly_artifact(
    approval: DLiveFaderProofApproval,
    public_approval: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact operator-bound GET-only artifact without transport."""

    supplied = Path(approval.readonly_artifact_path)
    try:
        metadata = os.lstat(supplied)
    except OSError as exc:
        raise ValueError("readonly_artifact_unavailable") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
    ):
        raise ValueError("readonly_artifact_not_private_regular_owned_file")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise ValueError("readonly_artifact_unavailable") from exc
    if resolved != supplied:
        raise ValueError("readonly_artifact_symlink_or_noncanonical_path")
    expected_root = _per_user_runtime_root() / "dlive_readonly_probes"
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            resolved.relative_to(expected_root)
        except ValueError as exc:
            raise ValueError("readonly_artifact_outside_private_runtime_root") from exc

    encoded = resolved.read_bytes()
    actual_sha256 = _sha256_bytes(encoded)
    expected_sha256 = public_approval["readonly_artifact"]["sha256"]
    if actual_sha256 != expected_sha256:
        raise ValueError("readonly_artifact_sha256_mismatch")
    try:
        artifact = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("readonly_artifact_invalid_json") from exc
    if not isinstance(artifact, Mapping):
        raise ValueError("readonly_artifact_not_object")

    endpoint = public_approval["endpoint"]
    exact_fields = {
        "schema": DLIVE_READONLY_PROBE_SCHEMA,
        "status": "readback_observed",
        "endpoint": endpoint["endpoint"],
        "host": endpoint["host"],
        "port": endpoint["port"],
        "tls": endpoint["tls"],
        "midi_base_channel": endpoint["midi_base_channel"],
        "user_profile": endpoint["user_profile"],
        "input_channel": approval.input_channel,
        "expected_route_interface": approval.expected_interface,
        "observed_route_interface": approval.expected_interface,
        "console_contacted": True,
        "mutation_write_attempt_count": 0,
        "live_write": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "midi_get_request_count": 6,
    }
    for field, expected in exact_fields.items():
        if artifact.get(field) != expected:
            raise ValueError(f"readonly_artifact_binding_mismatch:{field}")
    if artifact.get("operator_attestation") != public_approval["operator_attestation"]:
        raise ValueError("readonly_artifact_binding_mismatch:operator_attestation")
    teardown = artifact.get("teardown")
    if not isinstance(teardown, Mapping) or teardown.get("disconnect_requested") is not True:
        raise ValueError("readonly_artifact_teardown_unverified")
    if teardown.get("client_connected_after_teardown") is not False:
        raise ValueError("readonly_artifact_teardown_unverified")

    readbacks = artifact.get("readbacks")
    if not isinstance(readbacks, Mapping):
        raise ValueError("readonly_artifact_readbacks_missing")
    channel = approval.input_channel
    values: dict[str, Any] = {}
    for field in ("name", "fdr", "mute"):
        address = f"/ch/{channel}/{field}"
        readback = readbacks.get(address)
        if (
            not isinstance(readback, Mapping)
            or readback.get("status") != "received"
            or readback.get("address") != address
            or readback.get("confirmed_round_trips") != 2
        ):
            raise ValueError(f"readonly_artifact_readback_invalid:{address}")
        values[field] = readback.get("value")
    if values["name"] != approval.expected_name:
        raise ValueError("readonly_artifact_channel_name_mismatch")
    if values["mute"] is not True:
        raise ValueError("readonly_artifact_input_not_muted")
    source_fader_db = _finite_number(values["fdr"], "readonly_artifact_fader_db")
    return {
        "path": str(resolved),
        "sha256": actual_sha256,
        "probe_id": artifact.get("probe_id"),
        "finished_at_utc": artifact.get("finished_at_utc"),
        "source_fader_db": source_fader_db,
        "source_fader_raw": db_to_fader_value(source_fader_db),
    }


def _validate_bound_wire_evidence(
    approval: DLiveFaderProofApproval,
    public_approval: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an accepted operator-owned capture review without transport."""

    supplied = Path(approval.wire_evidence_path)
    try:
        metadata = os.lstat(supplied)
    except OSError as exc:
        raise ValueError("wire_evidence_unavailable") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
    ):
        raise ValueError("wire_evidence_not_private_regular_owned_file")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise ValueError("wire_evidence_unavailable") from exc
    if resolved != supplied:
        raise ValueError("wire_evidence_symlink_or_noncanonical_path")
    expected_root = _per_user_runtime_root() / "dlive_midi_capture_reviews"
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            resolved.relative_to(expected_root)
        except ValueError as exc:
            raise ValueError("wire_evidence_outside_private_runtime_root") from exc

    encoded = resolved.read_bytes()
    actual_sha256 = _sha256_bytes(encoded)
    if actual_sha256 != public_approval["wire_evidence"]["sha256"]:
        raise ValueError("wire_evidence_sha256_mismatch")
    try:
        artifact = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("wire_evidence_invalid_json") from exc
    if not isinstance(artifact, Mapping):
        raise ValueError("wire_evidence_not_object")

    endpoint = public_approval["endpoint"]
    exact_fields = {
        "schema": DLIVE_MIDI_CAPTURE_REVIEW_SCHEMA,
        "status": "accepted",
        "operator_owned": True,
        "acquired_by_automixer": False,
        "interface_opened_by_automixer": False,
        "direction": "client_to_selected_endpoint",
        "observation_point": "selected_endpoint_ingress",
        "exact_wire_match": True,
        "wire_delivery_observed": True,
        "console_acceptance_proven": False,
        "physical_fader_movement_proven": False,
        "console_contacted_by_automixer": False,
        "midi_sent_by_automixer": False,
        "mutation_write_attempt_count": 0,
        "live_write": False,
        "execution_allowed": False,
        "live_authority_granted": False,
        "standing_live_authority": False,
    }
    for field, expected in exact_fields.items():
        if artifact.get(field) != expected:
            raise ValueError(f"wire_evidence_binding_mismatch:{field}")
    expected_endpoint = {
        field: endpoint[field]
        for field in (
            "host",
            "endpoint",
            "port",
            "tls",
            "midi_base_channel",
            "user_profile",
        )
    }
    if artifact.get("endpoint") != expected_endpoint:
        raise ValueError("wire_evidence_binding_mismatch:endpoint")
    packet_client = DLiveClient(
        ip=approval.host,
        endpoint=approval.endpoint,
        port=approval.port,
        tls=approval.tls,
        midi_base_channel=approval.midi_base_channel,
        user_profile=approval.user_profile,
        auto_reconnect=False,
    )
    expected_packet = _packet_evidence(
        packet_client,
        approval.input_channel,
        approval.target_db,
    )
    operation = artifact.get("operation")
    expected_operation = {
        "family": "input_fader",
        "input_channel": approval.input_channel,
        "target_db": float(approval.target_db),
        "target_fader_raw": expected_packet["fader_raw"],
        "midi_channel": expected_packet["midi_channel"],
        "selector": expected_packet["selector"],
        "parameter_id": FADER_PARAMETER_ID,
    }
    if operation != expected_operation:
        raise ValueError("wire_evidence_binding_mismatch:operation")
    if artifact.get("expected_packet_hex") != expected_packet["packet_hex"]:
        raise ValueError("wire_evidence_binding_mismatch:expected_packet_hex")
    if artifact.get("captured_packet_hex") != expected_packet["packet_hex"]:
        raise ValueError("wire_evidence_binding_mismatch:captured_packet_hex")
    if artifact.get("packet_sha256") != expected_packet["packet_sha256"]:
        raise ValueError("wire_evidence_binding_mismatch:packet_sha256")
    return {
        "path": str(resolved),
        "sha256": actual_sha256,
        "capture_id": artifact.get("capture_id"),
        "captured_at_utc": artifact.get("captured_at_utc"),
        "capture_tool": artifact.get("capture_tool"),
        "source_client": artifact.get("source_client"),
        "packet_hex": expected_packet["packet_hex"],
        "packet_sha256": expected_packet["packet_sha256"],
        "exact_wire_match": True,
        "wire_delivery_observed": True,
        "console_acceptance_proven": False,
        "physical_fader_movement_proven": False,
    }


def validate_dlive_fader_proof_plan(
    approval: DLiveFaderProofApproval,
) -> dict[str, Any]:
    """Validate the exact artifact/target plan without route lookup or TCP."""

    now = time.time()
    required_confirmation = build_dlive_fader_proof_confirmation(approval)
    validation_copy = replace(
        approval,
        issued_at_epoch=now,
        expires_at_epoch=now + 60.0,
        pa_outputs_contained=True,
        require_channel_muted=True,
        receive_midi_confirmed_enabled=True,
        confirmation=required_confirmation,
    )
    public_approval = validation_copy.validate(now_epoch=now)
    bound_artifact = _validate_bound_readonly_artifact(
        validation_copy,
        public_approval,
    )
    bound_wire_evidence = _validate_bound_wire_evidence(
        validation_copy,
        public_approval,
    )
    cut = _validate_bounded_fader_cut(
        bound_artifact["source_fader_db"],
        validation_copy.target_db,
        validation_copy.max_cut_db,
    )
    packet_client = DLiveClient(
        ip=validation_copy.host,
        endpoint=validation_copy.endpoint,
        port=validation_copy.port,
        tls=validation_copy.tls,
        midi_base_channel=validation_copy.midi_base_channel,
        user_profile=validation_copy.user_profile,
        auto_reconnect=False,
    )
    configured_ceiling_db = _finite_number(
        packet_client._fader_ceiling_db,
        "configured_fader_ceiling_db",
    )
    if not _same_fader_value(
        validation_copy.target_db,
        min(float(validation_copy.target_db), configured_ceiling_db, 0.0),
    ):
        raise ValueError("proof_target_above_configured_fader_ceiling")
    return {
        "schema": "automixer.dlive_fader_proof_plan_validation.v1",
        "status": "ready_for_operator_confirmation",
        "reason": "exact_artifact_and_bounded_target_validated_without_transport",
        "endpoint": public_approval["endpoint"],
        "expected_interface": validation_copy.expected_interface,
        "input_channel": validation_copy.input_channel,
        "expected_name": validation_copy.expected_name,
        "operator_attestation": public_approval["operator_attestation"],
        "readonly_artifact": bound_artifact,
        "wire_evidence": bound_wire_evidence,
        "security_attestation": public_approval["security_attestation"],
        "bounded_cut": cut,
        "configured_fader_ceiling_db": configured_ceiling_db,
        "apply_packet": _packet_evidence(
            packet_client,
            validation_copy.input_channel,
            validation_copy.target_db,
        ),
        "rollback_packet": _packet_evidence(
            packet_client,
            validation_copy.input_channel,
            bound_artifact["source_fader_db"],
        ),
        "required_confirmation": required_confirmation,
        "approval_id": validation_copy.approval_id,
        "operator_confirmation_recorded": False,
        "physical_conditions_attested": False,
        "receive_midi_confirmed_enabled": False,
        "route_checked": False,
        "console_contacted": False,
        "hardware_contacted": False,
        "midi_sent": False,
        "mutation_write_attempt_count": 0,
        "live_write": False,
        "execution_allowed": False,
        "live_authority_granted": False,
        "standing_live_authority": False,
    }


def run_dlive_fader_restore_proof(
    approval: DLiveFaderProofApproval,
    *,
    password: str = "",
    tls_ca_file: str | None = None,
    tls_server_hostname: str | None = None,
    timeout_sec: float = 0.75,
    client_factory: Callable[..., Any] = DLiveClient,
    route_interface_resolver: Callable[[str], str | None],
    wal_path: Path | None = None,
    writer_lock_path: Path | None = None,
    now_fn: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Execute one attended cut/readback/rollback proof and return a receipt."""

    transaction_id = str(uuid.uuid4())
    receipt: dict[str, Any] = {
        "schema": DLIVE_FADER_PROOF_SCHEMA,
        "transaction_id": transaction_id,
        "started_at_utc": _utc_now(),
        "status": "blocked",
        "reason": "preflight_not_completed",
        "hardware_contacted": False,
        "midi_get_request_count": 0,
        "mutation_write_attempt_count": 0,
        "live_write": False,
        "standing_live_authority": False,
        "scoped_restore_only_authority_granted": False,
        "write_state_unknown": False,
        "readbacks": {},
        "route_checks": [],
        "wal_receipts": [],
    }
    client: Any = None
    connected = False
    authority: _DLiveFaderProofAuthority | None = None
    writer_lock: DLiveConsoleWriterLock | None = None
    apply_verified = False
    prepared_persisted = False
    apply_send_boundary_persisted = False
    preimage_db: float | None = None
    bound_readonly_artifact: dict[str, Any] | None = None

    def append_wal(wal: DLiveProofWal, phase: str, payload: Mapping[str, Any]) -> None:
        receipt["wal_receipts"].append(
            wal.append(transaction_id=transaction_id, phase=phase, payload=payload)
        )

    def route_is_expected(stage: str) -> bool:
        observed = route_interface_resolver(approval.host)
        receipt["route_checks"].append(
            {"stage": stage, "expected": approval.expected_interface, "observed": observed}
        )
        receipt["observed_route_interface"] = observed
        return observed == approval.expected_interface

    def read(address: str) -> dict[str, Any]:
        result = _read_received(client, address, timeout_sec)
        receipt["midi_get_request_count"] += 2
        receipt["readbacks"].setdefault(address, []).append(result)
        return result

    def restore_if_needed(wal: DLiveProofWal, *, trigger: str) -> bool:
        nonlocal authority
        if preimage_db is None or authority is None:
            return False
        address = f"/ch/{approval.input_channel}/fdr"
        try:
            current = read(address).get("value")
        except BaseException as exc:
            receipt["write_state_unknown"] = True
            receipt["recovery_error"] = type(exc).__name__
            return False
        if _same_fader_value(current, preimage_db):
            authority.mark_restored()
            return True
        if not _same_fader_value(current, approval.target_db):
            receipt["rollback_cas_mismatch_db"] = current
            receipt["write_state_unknown"] = True
            return False
        append_wal(
            wal,
            "rollback_send_boundary",
            {
                "trigger": trigger,
                "preimage_db": preimage_db,
                "packet": _packet_evidence(client, approval.input_channel, preimage_db),
                "send_may_have_occurred": True,
            },
        )
        if not route_is_expected("rollback_after_durable_boundary"):
            append_wal(
                wal,
                "rollback_send_result",
                {"sent": False, "reason": "dlive_route_interface_changed"},
            )
            receipt["write_state_unknown"] = True
            return False
        # Re-read after the durable boundary and immediately before transport.
        # This narrows the unavoidable network CAS window and refuses to
        # overwrite an operator or second system that changed the value.
        current_at_send = read(address).get("value")
        if _same_fader_value(current_at_send, preimage_db):
            append_wal(
                wal,
                "rollback_send_result",
                {"sent": False, "reason": "preimage_already_restored_after_boundary"},
            )
            authority.mark_restored()
            return True
        if not _same_fader_value(current_at_send, approval.target_db):
            append_wal(
                wal,
                "rollback_send_result",
                {
                    "sent": False,
                    "reason": "rollback_cas_target_changed_after_boundary",
                    "observed_db": current_at_send,
                    "write_state_unknown": True,
                },
            )
            receipt["rollback_cas_mismatch_db"] = current_at_send
            receipt["write_state_unknown"] = True
            return False
        receipt["mutation_write_attempt_count"] += 1
        receipt["live_write"] = True
        rollback_sent = client._supervised_fader_write(
            approval.input_channel,
            preimage_db,
            authority=authority,
            phase="rollback",
        )
        append_wal(
            wal,
            "rollback_send_result",
            {"sent": bool(rollback_sent), "write_state_unknown": not bool(rollback_sent)},
        )
        restored = read(address).get("value")
        if _same_fader_value(restored, preimage_db):
            authority.mark_restored()
            return True
        receipt["write_state_unknown"] = True
        receipt["rollback_mismatch_db"] = restored
        return False

    try:
        _reject_non_test_runtime_override(wal_path, writer_lock_path)
        public_approval = approval.validate(now_epoch=now_fn())
        receipt["approval"] = public_approval
        receipt["approval_sha256"] = _sha256_bytes(_canonical_json_bytes(public_approval))
        bound_readonly_artifact = _validate_bound_readonly_artifact(
            approval,
            public_approval,
        )
        receipt["readonly_artifact"] = bound_readonly_artifact
        bound_wire_evidence = _validate_bound_wire_evidence(
            approval,
            public_approval,
        )
        receipt["wire_evidence"] = bound_wire_evidence
        if not route_is_expected("initial_preflight"):
            receipt["reason"] = "dlive_route_interface_mismatch"
            return receipt

        runtime_root = _per_user_runtime_root()
        effective_wal_path = (
            wal_path or runtime_root / "dlive" / "fader_restore_proof_wal.jsonl"
        )
        writer_lock = DLiveConsoleWriterLock(
            writer_lock_path or runtime_root / "console_writer.lock"
        )
        if not writer_lock.acquire():
            receipt["reason"] = "console_writer_conflict"
            return receipt
        if not route_is_expected("after_writer_lock"):
            receipt["reason"] = "dlive_route_interface_mismatch"
            return receipt
        # Reload only after global exclusion. A prior writer may have appended
        # and released between route preflight and lock acquisition.
        wal = DLiveProofWal(effective_wal_path)
        incomplete = wal.incomplete_transaction_ids()
        if incomplete:
            receipt["reason"] = "dlive_wal_recovery_required"
            receipt["incomplete_transaction_ids"] = list(incomplete)
            return receipt
        if approval.approval_id in wal.consumed_approval_ids():
            receipt["reason"] = "operator_approval_already_consumed"
            return receipt

        client = client_factory(
            ip=approval.host,
            endpoint=approval.endpoint,
            port=approval.port,
            tls=approval.tls,
            tls_ca_file=tls_ca_file,
            tls_server_hostname=tls_server_hostname,
            midi_base_channel=approval.midi_base_channel,
            user_profile=approval.user_profile,
            password=password,
            auto_reconnect=False,
        )
        connected = bool(client.connect(timeout=timeout_sec))
        receipt["hardware_contacted"] = connected
        if not connected:
            receipt["reason"] = "dlive_connect_failed"
            return receipt

        channel = approval.input_channel
        name = read(f"/ch/{channel}/name").get("value")
        fader = read(f"/ch/{channel}/fdr").get("value")
        muted = read(f"/ch/{channel}/mute").get("value")
        if name != approval.expected_name:
            receipt["reason"] = "operator_expected_channel_name_mismatch"
            return receipt
        if muted is not True:
            receipt["reason"] = "proof_input_channel_not_muted"
            return receipt
        preimage_db = _finite_number(fader, "physical_preimage_db")
        if not _same_fader_value(
            preimage_db,
            bound_readonly_artifact["source_fader_db"],
        ):
            receipt["reason"] = "readonly_artifact_fader_preimage_changed"
            return receipt
        configured_ceiling_db = _finite_number(
            getattr(client, "_fader_ceiling_db", 0.0),
            "configured_fader_ceiling_db",
        )
        receipt["configured_fader_ceiling_db"] = configured_ceiling_db
        if not _same_fader_value(
            approval.target_db,
            min(float(approval.target_db), configured_ceiling_db, 0.0),
        ):
            receipt["reason"] = "proof_target_above_configured_fader_ceiling"
            return receipt
        authority = _DLiveFaderProofAuthority(
            _AUTHORITY_SEAL,
            client=client,
            approval=approval,
            now_fn=now_fn,
        )
        authority.bind_preimage(preimage_db)
        receipt["scoped_restore_only_authority_granted"] = True
        prepared = {
            "approval_id": approval.approval_id,
            "approval_sha256": receipt["approval_sha256"],
            "readonly_artifact": bound_readonly_artifact,
            "wire_evidence": bound_wire_evidence,
            "security_attestation": public_approval["security_attestation"],
            "endpoint": public_approval["endpoint"],
            "expected_interface": approval.expected_interface,
            "input_channel": channel,
            "expected_name": approval.expected_name,
            "operator_attestation": public_approval["operator_attestation"],
            "channel_muted": True,
            "preimage_db": preimage_db,
            "target_db": approval.target_db,
            "max_cut_db": approval.max_cut_db,
            "apply_packet": _packet_evidence(client, channel, approval.target_db),
            "rollback_packet": _packet_evidence(client, channel, preimage_db),
            "standing_live_authority": False,
        }
        append_wal(wal, "prepared", prepared)
        prepared_persisted = True

        if now_fn() >= approval.expires_at_epoch:
            append_wal(wal, "blocked_no_send", {"reason": "operator_approval_expired_before_apply"})
            receipt["reason"] = "operator_approval_expired_before_apply"
            return receipt
        append_wal(
            wal,
            "apply_send_boundary",
            {
                "apply_packet": prepared["apply_packet"],
                "send_may_have_occurred": True,
                "write_state_unknown": True,
            },
        )
        apply_send_boundary_persisted = True
        if not route_is_expected("apply_after_durable_boundary"):
            append_wal(
                wal,
                "apply_send_result",
                {"sent": False, "reason": "dlive_route_interface_changed"},
            )
            append_wal(
                wal,
                "blocked_no_send",
                {"reason": "dlive_route_interface_changed"},
            )
            receipt["reason"] = "dlive_route_interface_changed"
            return receipt
        final_name = read(f"/ch/{channel}/name").get("value")
        final_muted = read(f"/ch/{channel}/mute").get("value")
        final_preimage = read(f"/ch/{channel}/fdr").get("value")
        if (
            final_name != approval.expected_name
            or final_muted is not True
            or not _same_fader_value(final_preimage, preimage_db)
        ):
            append_wal(
                wal,
                "apply_send_result",
                {
                    "sent": False,
                    "reason": "apply_identity_mute_or_preimage_cas_changed",
                    "name_matches": final_name == approval.expected_name,
                    "muted": final_muted is True,
                    "preimage_matches": _same_fader_value(final_preimage, preimage_db),
                },
            )
            append_wal(
                wal,
                "blocked_no_send",
                {"reason": "apply_identity_mute_or_preimage_cas_changed"},
            )
            receipt["reason"] = "apply_identity_mute_or_preimage_cas_changed"
            return receipt
        receipt["mutation_write_attempt_count"] += 1
        receipt["live_write"] = True
        apply_sent = client._supervised_fader_write(
            channel,
            approval.target_db,
            authority=authority,
            phase="apply",
        )
        append_wal(
            wal,
            "apply_send_result",
            {"sent": bool(apply_sent), "write_state_unknown": not bool(apply_sent)},
        )
        applied_value = read(f"/ch/{channel}/fdr").get("value")
        apply_verified = bool(_same_fader_value(applied_value, approval.target_db))
        if apply_verified:
            authority.mark_apply_verified()
            append_wal(wal, "apply_verified", {"readback_db": applied_value})
        restored = restore_if_needed(wal, trigger="normal_restore_after_verify")
        if restored and apply_verified:
            receipt["status"] = "proof_restored_pending_teardown"
            receipt["reason"] = "apply_and_rollback_verified_pending_teardown"
            receipt["write_state_unknown"] = False
        elif restored:
            receipt["status"] = "restored_after_unverified_apply_pending_teardown"
            receipt["reason"] = "target_not_verified_original_state_restored_pending_teardown"
            receipt["write_state_unknown"] = False
        else:
            append_wal(
                wal,
                "manual_review_required",
                {"reason": "rollback_not_physically_verified", "write_state_unknown": True},
            )
            receipt["status"] = "manual_review_required"
            receipt["reason"] = "rollback_not_physically_verified"
            receipt["write_state_unknown"] = True
        return receipt
    except BaseException as exc:
        receipt["exception_type"] = type(exc).__name__
        if receipt["mutation_write_attempt_count"]:
            receipt["reason"] = "proof_exception_recovery_attempted"
        else:
            detail = str(exc).strip()
            receipt["reason"] = detail or "proof_preapply_exception"
            receipt["write_state_unknown"] = False
        receipt["write_state_unknown"] = bool(receipt["mutation_write_attempt_count"])
        try:
            if "wal" in locals() and receipt["mutation_write_attempt_count"]:
                restored = restore_if_needed(wal, trigger="exception_or_emergency")
                if restored:
                    receipt["status"] = "restored_after_unverified_apply_pending_teardown"
                else:
                    append_wal(
                        wal,
                        "manual_review_required",
                        {
                            "reason": receipt["reason"],
                            "exception_type": type(exc).__name__,
                            "rollback_physical_verified": False,
                            "write_state_unknown": True,
                        },
                    )
                    receipt["status"] = "manual_review_required"
                receipt["write_state_unknown"] = not restored
            elif "wal" in locals() and apply_send_boundary_persisted:
                append_wal(
                    wal,
                    "apply_send_result",
                    {
                        "sent": False,
                        "reason": "exception_or_emergency_before_transport",
                        "exception_type": type(exc).__name__,
                        "write_state_unknown": False,
                    },
                )
                append_wal(
                    wal,
                    "blocked_no_send",
                    {
                        "reason": "exception_or_emergency_before_transport",
                        "exception_type": type(exc).__name__,
                    },
                )
                receipt["status"] = "blocked"
                receipt["reason"] = "exception_or_emergency_before_transport"
                receipt["write_state_unknown"] = False
            elif "wal" in locals() and prepared_persisted:
                append_wal(
                    wal,
                    "blocked_no_send",
                    {
                        "reason": "exception_or_emergency_before_apply_boundary",
                        "exception_type": type(exc).__name__,
                    },
                )
                receipt["status"] = "blocked"
                receipt["reason"] = "exception_or_emergency_before_apply_boundary"
                receipt["write_state_unknown"] = False
        except BaseException as recovery_exc:
            receipt["status"] = "manual_review_required"
            receipt["recovery_exception_type"] = type(recovery_exc).__name__
            receipt["write_state_unknown"] = True
        return receipt
    finally:
        if authority is not None:
            authority.revoke()
        if connected and client is not None:
            try:
                client.disconnect()
            except Exception:
                receipt["disconnect_failed"] = True
        receipt["scoped_restore_only_authority_granted"] = False
        receipt["client_connected_after_teardown"] = bool(
            getattr(client, "is_connected", False)
        ) if client is not None else False
        pending_status = receipt.get("status")
        if pending_status in {
            "proof_restored_pending_teardown",
            "restored_after_unverified_apply_pending_teardown",
        }:
            teardown_verified = bool(
                not receipt.get("disconnect_failed")
                and receipt["client_connected_after_teardown"] is False
            )
            try:
                if teardown_verified:
                    if pending_status == "proof_restored_pending_teardown":
                        append_wal(
                            wal,
                            "proof_passed",
                            {
                                "apply_physical_verified": True,
                                "rollback_physical_verified": True,
                                "teardown_verified": True,
                                "standing_live_authority": False,
                            },
                        )
                        receipt["status"] = "proof_passed"
                        receipt["reason"] = "apply_readback_exact_rollback_and_teardown_verified"
                    else:
                        append_wal(
                            wal,
                            "restored_after_unverified_apply",
                            {
                                "apply_physical_verified": False,
                                "rollback_physical_verified": True,
                                "teardown_verified": True,
                            },
                        )
                        receipt["status"] = "restored_after_unverified_apply"
                        receipt["reason"] = "target_not_verified_original_state_and_teardown_restored"
                else:
                    append_wal(
                        wal,
                        "manual_review_required",
                        {
                            "reason": "proof_teardown_unverified",
                            "rollback_physical_verified": True,
                            "teardown_verified": False,
                            "write_state_unknown": False,
                        },
                    )
                    receipt["status"] = "manual_review_required"
                    receipt["reason"] = "proof_teardown_unverified"
            except BaseException as exc:
                receipt["status"] = "manual_review_required"
                receipt["reason"] = "proof_terminal_wal_or_teardown_unverified"
                receipt["terminal_wal_exception_type"] = type(exc).__name__
        if writer_lock is not None:
            writer_lock.release()
        receipt["finished_at_utc"] = _utc_now()


def recover_dlive_fader_restore_proof(
    transaction_id: str,
    *,
    pa_outputs_contained: bool,
    input_confirmed_muted: bool,
    password: str = "",
    tls_ca_file: str | None = None,
    tls_server_hostname: str | None = None,
    timeout_sec: float = 0.75,
    client_factory: Callable[..., Any] = DLiveClient,
    route_interface_resolver: Callable[[str], str | None],
    wal_path: Path | None = None,
    writer_lock_path: Path | None = None,
) -> dict[str, Any]:
    """Reconcile one incomplete proof and restore only its exact preimage.

    Recovery never replays the proof target. It either observes the original
    value, restores the exact target left by the interrupted transaction, or
    stops for manual review on any identity/CAS mismatch.
    """

    receipt: dict[str, Any] = {
        "schema": "automixer.dlive_fader_restore_recovery.v1",
        "transaction_id": str(transaction_id or ""),
        "started_at_utc": _utc_now(),
        "status": "blocked",
        "reason": "recovery_preflight_not_completed",
        "hardware_contacted": False,
        "midi_get_request_count": 0,
        "mutation_write_attempt_count": 0,
        "live_write": False,
        "standing_live_authority": False,
        "write_state_unknown": True,
        "wal_receipts": [],
        "readbacks": {},
        "route_checks": [],
    }
    if not transaction_id:
        receipt["reason"] = "transaction_id_required"
        receipt["write_state_unknown"] = False
        receipt["finished_at_utc"] = _utc_now()
        return receipt
    try:
        _reject_non_test_runtime_override(wal_path, writer_lock_path)
    except ValueError as exc:
        receipt["reason"] = str(exc)
        receipt["write_state_unknown"] = False
        receipt["finished_at_utc"] = _utc_now()
        return receipt
    if pa_outputs_contained is not True or input_confirmed_muted is not True:
        receipt["reason"] = "recovery_requires_physical_output_containment_and_muted_input"
        receipt["finished_at_utc"] = _utc_now()
        return receipt

    runtime_root = _per_user_runtime_root()
    effective_wal_path = wal_path or runtime_root / "dlive" / "fader_restore_proof_wal.jsonl"
    lock = DLiveConsoleWriterLock(
        writer_lock_path or runtime_root / "console_writer.lock"
    )
    client: Any = None
    connected = False
    authority: _DLiveRollbackRecoveryAuthority | None = None
    try:
        if not lock.acquire():
            receipt["reason"] = "console_writer_conflict"
            return receipt
        wal = DLiveProofWal(effective_wal_path)
        transaction_records = [
            record for record in wal.records
            if record.get("transaction_id") == transaction_id
        ]
        if not transaction_records:
            receipt["reason"] = "dlive_wal_transaction_not_found"
            receipt["write_state_unknown"] = False
            return receipt
        if transaction_records[-1].get("phase") in TERMINAL_WAL_PHASES:
            receipt["reason"] = "dlive_wal_transaction_already_terminal"
            receipt["write_state_unknown"] = False
            return receipt
        prepared_records = [
            record for record in transaction_records
            if record.get("phase") == "prepared"
        ]
        if len(prepared_records) != 1:
            receipt["reason"] = "dlive_wal_prepared_record_missing_or_ambiguous"
            return receipt
        prepared = prepared_records[0].get("payload")
        if not isinstance(prepared, Mapping):
            receipt["reason"] = "dlive_wal_prepared_payload_invalid"
            return receipt
        endpoint = prepared.get("endpoint")
        if not isinstance(endpoint, Mapping):
            receipt["reason"] = "dlive_wal_endpoint_binding_missing"
            return receipt
        required = {
            "host", "endpoint", "tls", "port", "midi_base_channel", "user_profile"
        }
        if not required.issubset(endpoint):
            receipt["reason"] = "dlive_wal_endpoint_binding_incomplete"
            return receipt
        channel = prepared.get("input_channel")
        expected_name = prepared.get("expected_name")
        operator_attestation = prepared.get("operator_attestation")
        if not isinstance(operator_attestation, Mapping):
            receipt["reason"] = "dlive_wal_operator_attestation_missing"
            return receipt
        receipt["operator_attestation"] = dict(operator_attestation)
        preimage_db = _finite_number(prepared.get("preimage_db"), "wal_preimage_db")
        target_db = _finite_number(prepared.get("target_db"), "wal_target_db")
        expected_interface = str(prepared.get("expected_interface") or "")
        observed_interface = route_interface_resolver(str(endpoint["host"]))
        receipt["route_checks"].append(
            {"stage": "before_connect", "expected": expected_interface, "observed": observed_interface}
        )
        receipt["observed_route_interface"] = observed_interface
        if observed_interface != expected_interface:
            receipt["reason"] = "dlive_route_interface_mismatch"
            return receipt

        client = client_factory(
            ip=endpoint["host"],
            endpoint=endpoint["endpoint"],
            port=endpoint["port"],
            tls=endpoint["tls"],
            tls_ca_file=tls_ca_file,
            tls_server_hostname=tls_server_hostname,
            midi_base_channel=endpoint["midi_base_channel"],
            user_profile=endpoint["user_profile"],
            password=password,
            auto_reconnect=False,
        )
        connected = bool(client.connect(timeout=timeout_sec))
        receipt["hardware_contacted"] = connected
        if not connected:
            receipt["reason"] = "dlive_connect_failed"
            return receipt

        def read(address: str) -> dict[str, Any]:
            result = _read_received(client, address, timeout_sec)
            receipt["midi_get_request_count"] += 2
            receipt["readbacks"].setdefault(address, []).append(result)
            return result

        name = read(f"/ch/{channel}/name").get("value")
        muted = read(f"/ch/{channel}/mute").get("value")
        current_db = read(f"/ch/{channel}/fdr").get("value")
        if name != expected_name:
            receipt["reason"] = "recovery_channel_identity_mismatch"
            wal_receipt = wal.append(
                transaction_id=transaction_id,
                phase="manual_review_required",
                payload={"reason": receipt["reason"], "write_state_unknown": True},
            )
            receipt["wal_receipts"].append(wal_receipt)
            receipt["status"] = "manual_review_required"
            return receipt
        if muted is not True:
            receipt["reason"] = "recovery_input_channel_not_muted"
            return receipt
        if _same_fader_value(current_db, preimage_db):
            receipt["status"] = "recovery_restored_pending_teardown"
            receipt["reason"] = "preimage_restored_pending_teardown"
            receipt["rollback_send_required"] = False
            receipt["write_state_unknown"] = False
            return receipt
        if not _same_fader_value(current_db, target_db):
            receipt["reason"] = "recovery_cas_target_changed"
            wal_receipt = wal.append(
                transaction_id=transaction_id,
                phase="manual_review_required",
                payload={
                    "reason": receipt["reason"],
                    "observed_db": current_db,
                    "write_state_unknown": True,
                },
            )
            receipt["wal_receipts"].append(wal_receipt)
            receipt["status"] = "manual_review_required"
            return receipt

        authority = _DLiveRollbackRecoveryAuthority(
            _AUTHORITY_SEAL,
            client=client,
            channel=int(channel),
            preimage_db=preimage_db,
        )
        boundary = wal.append(
            transaction_id=transaction_id,
            phase="recovery_rollback_send_boundary",
            payload={
                "packet": _packet_evidence(client, int(channel), preimage_db),
                "send_may_have_occurred": True,
                "write_state_unknown": True,
            },
        )
        receipt["wal_receipts"].append(boundary)
        observed_interface = route_interface_resolver(str(endpoint["host"]))
        receipt["route_checks"].append(
            {"stage": "rollback_after_durable_boundary", "expected": expected_interface, "observed": observed_interface}
        )
        receipt["observed_route_interface"] = observed_interface
        if observed_interface != expected_interface:
            terminal = wal.append(
                transaction_id=transaction_id,
                phase="manual_review_required",
                payload={
                    "reason": "dlive_route_interface_changed",
                    "write_state_unknown": True,
                },
            )
            receipt["wal_receipts"].append(terminal)
            receipt["status"] = "manual_review_required"
            receipt["reason"] = "dlive_route_interface_changed"
            return receipt
        final_name = read(f"/ch/{channel}/name").get("value")
        final_muted = read(f"/ch/{channel}/mute").get("value")
        final_current_db = read(f"/ch/{channel}/fdr").get("value")
        if final_name != expected_name or final_muted is not True:
            terminal = wal.append(
                transaction_id=transaction_id,
                phase="manual_review_required",
                payload={
                    "reason": "recovery_identity_or_mute_changed_after_boundary",
                    "write_state_unknown": True,
                },
            )
            receipt["wal_receipts"].append(terminal)
            receipt["status"] = "manual_review_required"
            receipt["reason"] = "recovery_identity_or_mute_changed_after_boundary"
            return receipt
        if _same_fader_value(final_current_db, preimage_db):
            receipt["status"] = "recovery_restored_pending_teardown"
            receipt["reason"] = "preimage_restored_after_boundary_pending_teardown"
            receipt["rollback_send_required"] = False
            receipt["write_state_unknown"] = False
            return receipt
        if not _same_fader_value(final_current_db, target_db):
            terminal = wal.append(
                transaction_id=transaction_id,
                phase="manual_review_required",
                payload={
                    "reason": "recovery_cas_target_changed_after_boundary",
                    "observed_db": final_current_db,
                    "write_state_unknown": True,
                },
            )
            receipt["wal_receipts"].append(terminal)
            receipt["status"] = "manual_review_required"
            receipt["reason"] = "recovery_cas_target_changed_after_boundary"
            return receipt
        receipt["mutation_write_attempt_count"] = 1
        receipt["live_write"] = True
        sent = client._supervised_fader_write(
            int(channel),
            preimage_db,
            authority=authority,
            phase="rollback",
        )
        send_result = wal.append(
            transaction_id=transaction_id,
            phase="recovery_rollback_send_result",
            payload={"sent": bool(sent), "write_state_unknown": not bool(sent)},
        )
        receipt["wal_receipts"].append(send_result)
        restored_db = read(f"/ch/{channel}/fdr").get("value")
        if _same_fader_value(restored_db, preimage_db):
            receipt["status"] = "recovery_restored_pending_teardown"
            receipt["reason"] = "exact_preimage_rollback_verified_pending_teardown"
            receipt["rollback_send_required"] = True
            receipt["write_state_unknown"] = False
        else:
            terminal = wal.append(
                transaction_id=transaction_id,
                phase="manual_review_required",
                payload={
                    "reason": "recovery_rollback_not_physically_verified",
                    "observed_db": restored_db,
                    "write_state_unknown": True,
                },
            )
            receipt["wal_receipts"].append(terminal)
            receipt["status"] = "manual_review_required"
            receipt["reason"] = "recovery_rollback_not_physically_verified"
        return receipt
    except BaseException as exc:
        receipt["status"] = "manual_review_required"
        receipt["reason"] = "recovery_exception_manual_review_required"
        receipt["exception_type"] = type(exc).__name__
        receipt["write_state_unknown"] = True
        return receipt
    finally:
        if authority is not None:
            authority.revoke()
        if connected and client is not None:
            try:
                client.disconnect()
            except Exception:
                receipt["disconnect_failed"] = True
        receipt["client_connected_after_teardown"] = bool(
            getattr(client, "is_connected", False)
        ) if client is not None else False
        if receipt.get("status") == "recovery_restored_pending_teardown":
            teardown_verified = bool(
                not receipt.get("disconnect_failed")
                and receipt["client_connected_after_teardown"] is False
            )
            try:
                if teardown_verified:
                    terminal = wal.append(
                        transaction_id=transaction_id,
                        phase="recovery_restored",
                        payload={
                            "rollback_send_required": bool(
                                receipt.get("rollback_send_required")
                            ),
                            "preimage_physical_verified": True,
                            "teardown_verified": True,
                            "standing_live_authority": False,
                        },
                    )
                    receipt["wal_receipts"].append(terminal)
                    receipt["status"] = "recovery_restored"
                    receipt["reason"] = (
                        "exact_preimage_rollback_and_teardown_verified"
                        if receipt.get("rollback_send_required")
                        else "preimage_and_teardown_already_physically_restored"
                    )
                else:
                    terminal = wal.append(
                        transaction_id=transaction_id,
                        phase="manual_review_required",
                        payload={
                            "reason": "recovery_teardown_unverified",
                            "preimage_physical_verified": True,
                            "teardown_verified": False,
                            "write_state_unknown": False,
                        },
                    )
                    receipt["wal_receipts"].append(terminal)
                    receipt["status"] = "manual_review_required"
                    receipt["reason"] = "recovery_teardown_unverified"
            except BaseException as exc:
                receipt["status"] = "manual_review_required"
                receipt["reason"] = "recovery_terminal_wal_or_teardown_unverified"
                receipt["terminal_wal_exception_type"] = type(exc).__name__
        lock.release()
        receipt["finished_at_utc"] = _utc_now()
