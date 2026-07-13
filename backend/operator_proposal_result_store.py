"""Durable, byte-bounded evidence store for operator proposal results.

The store is deliberately independent from mixer execution.  A caller must
reserve the worst-case result slot before a possible console send, then commit
the exact terminal result through the reservation.  User-controlled proposal
identifiers never enter filesystem paths.
"""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import secrets
import shutil
import stat
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from decision_queue_wire_contract import (
    canonical_json_bytes,
    is_sha256_digest,
    sha256_digest,
    terminal_result_proves_verified_apply,
)


MAX_RESULT_ARTIFACT_BYTES = 8_388_608
MAX_RESULT_ARTIFACTS = 1_024
MAX_RESULT_STORE_BYTES = 268_435_456
MIN_RESULT_STORE_FREE_BYTES = 536_870_912


def default_result_store_root() -> Path:
    """Return a user-private mutable root outside source/app bundles."""

    home_value = os.getenv("HOME")
    home = (
        Path(home_value).expanduser()
        if home_value and Path(home_value).expanduser().is_absolute()
        else Path.home()
    )
    if sys.platform == "darwin":
        state_root = home / "Library" / "Application Support"
    else:
        xdg_state = os.getenv("XDG_STATE_HOME")
        state_root = (
            Path(xdg_state).expanduser()
            if xdg_state and Path(xdg_state).expanduser().is_absolute()
            else home / ".local" / "state"
        )
    return state_root / "AUTO-MIXER" / "operator_proposal_queue" / "results"


DEFAULT_RESULT_STORE_ROOT = default_result_store_root()

_SAFE_ARTIFACT_NAME = re.compile(r"^[0-9a-f]{64}\.json$")
_SAFE_RESERVATION_NAME = re.compile(r"^\.reservation\.([0-9a-f]{32})$")
_SAFE_TEMP_NAME = re.compile(r"^\.[0-9a-f]{64}\.[0-9a-f]{16}\.tmp$")
_SAFE_RESERVATION_TEMP_NAME = re.compile(
    r"^\.reservation\.([0-9a-f]{32})\.([0-9a-f]{16})\.tmp$"
)
_STORE_LOCK_NAME = ".store.lock"
_MAX_DIRECTORY_ENTRIES = 4_096
_MAX_RESULT_NODES = 100_000
_MAX_RESULT_DEPTH = 64
_MAX_RESERVATION_MARKER_BYTES = 4_096
_RESERVATION_PHASE_RESERVED = "reserved_no_send"
_RESERVATION_PHASE_SEND_MAY_HAVE_OCCURRED = "send_may_have_occurred"
_RESERVATION_PHASE_PERSISTENCE_FAILED = "persistence_failed_after_send"
_RESERVATION_PHASES = frozenset(
    {
        _RESERVATION_PHASE_RESERVED,
        _RESERVATION_PHASE_SEND_MAY_HAVE_OCCURRED,
        _RESERVATION_PHASE_PERSISTENCE_FAILED,
    }
)
_PROCESS_INSTANCE_ID = secrets.token_hex(16)
_RESULT_SAFETY_BOOLEAN_FIELDS = frozenset(
    {
        "write_state_unknown",
        "rollback_state_unknown",
        "physical_state_restored",
        "manual_review_required",
        "no_console_send_proven",
        "physical_verified",
        "audio_recheck_passed",
        "audio_recheck_required",
        "terminal_after_console_send",
        "result_persistence_failed",
    }
)

_ROOT_LOCKS_GUARD = threading.Lock()
_ROOT_LOCKS: Dict[str, threading.RLock] = {}


def _shared_root_lock(root: Path) -> threading.RLock:
    key = os.path.abspath(os.fspath(root))
    with _ROOT_LOCKS_GUARD:
        lock = _ROOT_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _ROOT_LOCKS[key] = lock
        return lock


class ResultEvidenceError(RuntimeError):
    """Typed fail-closed result evidence error."""

    def __init__(
        self,
        code: str,
        *,
        hazards: tuple[Dict[str, Any], ...] = (),
    ) -> None:
        super().__init__(code)
        self.code = code
        self.hazards = tuple(dict(item) for item in hazards)


@dataclass(frozen=True)
class ResultEvidenceRecord:
    """Bounded reference to one digest-addressed result artifact."""

    digest: str
    byte_count: int
    artifact_name: str
    pinned: bool

    def to_reference(self) -> Dict[str, Any]:
        return {
            "result_digest": self.digest,
            "result_bytes": self.byte_count,
            "result_detail_available": True,
            "result_evidence_pinned": self.pinned,
        }


@dataclass(frozen=True)
class ResultEvidenceReservationBinding:
    """Private frozen identity for one durable reservation marker."""

    token: str
    proposal_id: str
    review_content_digest: str


@dataclass(frozen=True)
class _Artifact:
    name: str
    digest: str
    byte_count: int
    pinned: bool
    fatal_codes: tuple[str, ...]
    conservative_recovery: bool
    modified_ns: int


@dataclass(frozen=True)
class _Reservation:
    name: str
    token: str
    reserved_bytes: int
    proposal_id: str
    review_content_digest: str
    phase: str
    pid: int
    process_instance_id: str


def _result_proves_exact_safe_no_send(result: Any) -> bool:
    """Inspect only scalar safety fields; malformed shapes fail possible-send."""

    if not isinstance(result, dict):
        return False
    try:
        if any(
            key in result and not isinstance(dict.get(result, key), bool)
            for key in _RESULT_SAFETY_BOOLEAN_FIELDS
        ):
            return False
        terminal_after_send = dict.get(result, "terminal_after_console_send")
        persistence_failed = dict.get(result, "result_persistence_failed")
        physical_verified = dict.get(result, "physical_verified")
        wal_terminal = dict.get(result, "wal_terminal")
        return bool(
            dict.get(result, "no_console_send_proven") is True
            and dict.get(result, "write_state_unknown") is False
            and dict.get(result, "rollback_state_unknown") is False
            and dict.get(result, "manual_review_required") is False
            and (
                "physical_state_restored" not in result
                or dict.get(result, "physical_state_restored") is True
            )
            and (
                "terminal_after_console_send" not in result
                or terminal_after_send is False
            )
            and (
                "result_persistence_failed" not in result
                or persistence_failed is False
            )
            and dict.get(result, "status")
            not in {
                "applied",
                "sent_unverified",
                "rolled_back",
                "manual_review_required",
            }
            and (
                "physical_verified" not in result
                or physical_verified is False
            )
            and (
                "wal_terminal" not in result
                or wal_terminal
                in {
                    None,
                    "",
                    "not_applicable",
                    "not_prepared",
                    "aborted_before_send",
                }
            )
        )
    except (AttributeError, TypeError):
        return False


def result_requires_pinned_evidence(result: Dict[str, Any]) -> bool:
    """Return true unless automatic eviction is physically proven safe."""

    if any(
        key in result and not isinstance(result.get(key), bool)
        for key in _RESULT_SAFETY_BOOLEAN_FIELDS
    ):
        return True
    for key in (
        "write_state_unknown",
        "rollback_state_unknown",
        "manual_review_required",
    ):
        if result.get(key) is not False:
            return True
    if (
        "physical_state_restored" in result
        and result.get("physical_state_restored") is not True
    ):
        return True
    known_no_send = _result_proves_exact_safe_no_send(result)
    known_restored = bool(
        result.get("physical_state_restored") is True
        and result.get("no_console_send_proven") is False
        and result.get("terminal_after_console_send") is True
        and result.get("wal_terminal") == "rolled_back"
        and result.get("status") in {"blocked", "rolled_back"}
    )
    known_verified_apply = terminal_result_proves_verified_apply(result)
    return not (known_no_send or known_restored or known_verified_apply)


def _validate_result_shape(value: Any) -> None:
    """Reject cycles and resource-exhausting JSON shapes without recursion."""

    active: set[int] = set()
    stack = [(value, 0, False)]
    nodes = 0
    while stack:
        item, depth, leaving = stack.pop()
        if leaving:
            active.discard(id(item))
            continue
        nodes += 1
        if nodes > _MAX_RESULT_NODES:
            raise ResultEvidenceError(
                "operator_proposal_result_evidence_node_limit_exceeded"
            )
        if depth > _MAX_RESULT_DEPTH:
            raise ResultEvidenceError(
                "operator_proposal_result_evidence_depth_limit_exceeded"
            )
        if not isinstance(item, (dict, list, tuple)):
            continue
        identity = id(item)
        if identity in active:
            raise ResultEvidenceError(
                "operator_proposal_result_evidence_cycle_detected"
            )
        active.add(identity)
        stack.append((item, depth, True))
        children = item.values() if isinstance(item, dict) else item
        stack.extend((child, depth + 1, False) for child in children)


class ResultEvidenceStore:
    """Digest-addressed result store with pessimistic pre-send reservations."""

    def __init__(
        self,
        root: Optional[Path] = None,
        *,
        max_artifact_bytes: int = MAX_RESULT_ARTIFACT_BYTES,
        max_artifacts: int = MAX_RESULT_ARTIFACTS,
        max_total_bytes: int = MAX_RESULT_STORE_BYTES,
        min_free_bytes: int = MIN_RESULT_STORE_FREE_BYTES,
        disk_usage: Callable[[Path], Any] = shutil.disk_usage,
        process_instance_id: Optional[str] = None,
    ) -> None:
        selected_root = (
            Path(root) if root is not None else default_result_store_root()
        )
        self.root = Path(os.path.abspath(os.fspath(selected_root)))
        self.max_artifact_bytes = int(max_artifact_bytes)
        self.max_artifacts = int(max_artifacts)
        self.max_total_bytes = int(max_total_bytes)
        self.min_free_bytes = int(min_free_bytes)
        self._disk_usage = disk_usage
        self._lock = _shared_root_lock(self.root)
        self._uses_native_disk_usage = disk_usage is shutil.disk_usage
        self._process_instance_id = str(
            process_instance_id or _PROCESS_INSTANCE_ID
        )
        self._local_recovery_required = False

    @property
    def active_reservation_count(self) -> int:
        with self._locked_root() as root_fd:
            entries = list(os.listdir(root_fd))
            return len(self._reservations(root_fd, entries))

    @staticmethod
    def canonical_result(result: Dict[str, Any]) -> tuple[bytes, str, bool]:
        if not isinstance(result, dict):
            raise ResultEvidenceError("operator_proposal_result_evidence_invalid")
        _validate_result_shape(result)
        try:
            content = canonical_json_bytes(result)
        except (RecursionError, TypeError, ValueError) as exc:
            raise ResultEvidenceError(
                "operator_proposal_result_evidence_not_canonical_json"
            ) from exc
        return content, sha256_digest(content), result_requires_pinned_evidence(result)

    @staticmethod
    def _artifact_name(digest: str) -> str:
        raw = str(digest or "")
        if len(raw) != 71 or not raw.startswith("sha256:"):
            raise ResultEvidenceError("operator_proposal_result_digest_invalid")
        name = f"{raw[7:]}.json"
        if not _SAFE_ARTIFACT_NAME.fullmatch(name):
            raise ResultEvidenceError("operator_proposal_result_digest_invalid")
        return name

    @staticmethod
    def _fsync_directory_fd(descriptor: int, _path: Path) -> None:
        os.fsync(descriptor)

    def _open_root_fd(self, *, create: bool = True) -> int:
        """Open/create every component without following a symlink."""

        if self.root.anchor != os.sep:
            raise ResultEvidenceError(
                "operator_proposal_result_evidence_store_unsafe_root"
            )
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        opened = []
        current_path = Path(os.sep)
        first_created_parent_index: Optional[int] = None
        try:
            current = os.open(os.sep, flags)
            opened.append(current)
            for part in self.root.parts[1:]:
                try:
                    child = os.open(part, flags, dir_fd=current)
                except FileNotFoundError:
                    if not create:
                        raise ResultEvidenceError(
                            "operator_proposal_result_evidence_store_unavailable"
                        )
                    os.mkdir(part, mode=0o700, dir_fd=current)
                    if first_created_parent_index is None:
                        first_created_parent_index = len(opened) - 1
                    child = os.open(part, flags, dir_fd=current)
                except OSError as exc:
                    raise ResultEvidenceError(
                        "operator_proposal_result_evidence_store_unsafe_root"
                    ) from exc
                info = os.fstat(child)
                child_path = current_path / part
                if not stat.S_ISDIR(info.st_mode):
                    os.close(child)
                    raise ResultEvidenceError(
                        "operator_proposal_result_evidence_store_unsafe_root"
                    )
                if child_path != self.root and info.st_mode & 0o022:
                    os.close(child)
                    raise ResultEvidenceError(
                        "operator_proposal_result_evidence_store_insecure_ancestor"
                    )
                opened.append(child)
                current = child
                current_path = child_path
            root_info = os.fstat(opened[-1])
            if root_info.st_uid != os.geteuid() or stat.S_IMODE(root_info.st_mode) != 0o700:
                raise ResultEvidenceError(
                    "operator_proposal_result_evidence_store_insecure_root"
                )
            if first_created_parent_index is not None:
                for index in range(len(opened) - 1, first_created_parent_index - 1, -1):
                    path = Path(os.sep).joinpath(*self.root.parts[1:index + 1])
                    self._fsync_directory_fd(opened[index], path)
            root_fd = opened.pop()
            return root_fd
        except ResultEvidenceError:
            raise
        except OSError as exc:
            raise ResultEvidenceError(
                "operator_proposal_result_evidence_directory_fsync_failed"
            ) from exc
        finally:
            for descriptor in reversed(opened):
                os.close(descriptor)

    def _root_path_matches_fd(self, root_fd: int) -> bool:
        try:
            current_fd = self._open_root_fd(create=False)
        except ResultEvidenceError:
            return False
        try:
            expected = os.fstat(root_fd)
            current = os.fstat(current_fd)
            return (expected.st_dev, expected.st_ino) == (current.st_dev, current.st_ino)
        finally:
            os.close(current_fd)

    @contextmanager
    def _locked_root(self):
        with self._lock:
            root_fd = self._open_root_fd()
            lock_fd: Optional[int] = None
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            try:
                lock_fd = os.open(
                    _STORE_LOCK_NAME,
                    flags,
                    0o600,
                    dir_fd=root_fd,
                )
                info = os.fstat(lock_fd)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid()
                    or info.st_nlink != 1
                    or stat.S_IMODE(info.st_mode) != 0o600
                ):
                    raise ResultEvidenceError(
                        "operator_proposal_result_evidence_store_unsafe_lock"
                    )
                os.fsync(lock_fd)
                self._fsync_directory_fd(root_fd, self.root)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                yield root_fd
            except ResultEvidenceError:
                raise
            except OSError as exc:
                raise ResultEvidenceError(
                    "operator_proposal_result_evidence_store_unavailable"
                ) from exc
            finally:
                if lock_fd is not None:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    finally:
                        os.close(lock_fd)
                os.close(root_fd)

    @staticmethod
    def _classify_result_safety(
        content: bytes,
    ) -> tuple[bool, tuple[str, ...], bool]:
        try:
            value = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return True, (), True
        if not isinstance(value, dict):
            return True, (), True
        pinned = result_requires_pinned_evidence(value)
        fatal_codes = []
        if value.get("write_state_unknown") is True:
            fatal_codes.append("terminal_write_state_unknown")
        if (
            value.get("rollback_state_unknown") is True
            or value.get("physical_state_restored") is False
        ):
            fatal_codes.append("terminal_rollback_state_unknown")
        if value.get("manual_review_required") is True:
            fatal_codes.append("terminal_manual_review_after_send")
        return pinned, tuple(fatal_codes), bool(pinned and not fatal_codes)

    @staticmethod
    def _decode_pinned(content: bytes) -> bool:
        pinned, _fatal_codes, _conservative_recovery = (
            ResultEvidenceStore._classify_result_safety(content)
        )
        return pinned

    @staticmethod
    def _artifact_recovery_hazards(
        artifacts: Dict[str, _Artifact],
    ) -> tuple[Dict[str, Any], ...]:
        hazards = []
        for artifact in sorted(
            artifacts.values(),
            key=lambda item: item.digest,
        ):
            for code in artifact.fatal_codes:
                hazards.append(
                    {
                        "code": code,
                        "proposal_id": None,
                        "review_content_digest": None,
                        "result_digest": artifact.digest,
                        "phase": "retained_result_hazard",
                    }
                )
            if artifact.conservative_recovery:
                hazards.append(
                    {
                        "code": "operator_proposal_result_recovery_required",
                        "proposal_id": None,
                        "review_content_digest": None,
                        "result_digest": artifact.digest,
                        "phase": "retained_result_conservative_recovery",
                    }
                )
        return tuple(hazards)

    @staticmethod
    def _possible_send_reservation_hazards(
        reservations: Dict[str, _Reservation],
        *,
        exclude_token: Optional[str] = None,
    ) -> tuple[Dict[str, Any], ...]:
        hazards = []
        for reservation in sorted(
            reservations.values(),
            key=lambda item: item.token,
        ):
            if reservation.token == exclude_token:
                continue
            if reservation.phase == _RESERVATION_PHASE_RESERVED:
                continue
            hazards.append(
                {
                    "code": "operator_proposal_result_persistence_failed",
                    "proposal_id": reservation.proposal_id,
                    "review_content_digest": (
                        reservation.review_content_digest
                    ),
                    "result_digest": None,
                    "phase": reservation.phase,
                }
            )
        return tuple(hazards)

    def _local_apply_hazards(self) -> tuple[Dict[str, Any], ...]:
        if not self._local_recovery_required:
            return ()
        return (
            {
                "code": "operator_proposal_result_persistence_failed",
                "proposal_id": None,
                "review_content_digest": None,
                "result_digest": None,
                "phase": "local_recovery_required",
            },
        )

    def current_apply_hazards(self) -> tuple[Dict[str, Any], ...]:
        """Read the cross-process apply fence without reaping markers."""

        with self._locked_root() as root_fd:
            artifacts, reservations = self._scan_store(root_fd)
            return (
                *self._artifact_recovery_hazards(artifacts),
                *self._possible_send_reservation_hazards(reservations),
                *self._local_apply_hazards(),
            )

    def _read_regular_file(
        self,
        root_fd: int,
        name: str,
        *,
        max_bytes: int,
        missing_code: str,
        unsafe_code: str,
        fsync_before_close: bool = False,
    ) -> tuple[bytes, os.stat_result]:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor: Optional[int] = None
        try:
            descriptor = os.open(name, flags, dir_fd=root_fd)
        except FileNotFoundError as exc:
            raise ResultEvidenceError(missing_code) from exc
        except OSError as exc:
            raise ResultEvidenceError(unsafe_code) from exc
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                raise ResultEvidenceError(unsafe_code)
            if info.st_size > max_bytes:
                raise ResultEvidenceError(
                    "operator_proposal_result_evidence_store_corrupt"
                )
            chunks = []
            remaining = max_bytes + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(65_536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            content = b"".join(chunks)
            if len(content) > max_bytes:
                raise ResultEvidenceError(
                    "operator_proposal_result_evidence_store_corrupt"
                )
            after = os.fstat(descriptor)
            if (
                (info.st_dev, info.st_ino, info.st_size)
                != (after.st_dev, after.st_ino, after.st_size)
            ):
                raise ResultEvidenceError(
                    "operator_proposal_result_evidence_store_corrupt"
                )
            if fsync_before_close:
                os.fsync(descriptor)
            return content, after
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _reservations(self, root_fd: int, entries: list[str]) -> Dict[str, _Reservation]:
        reservations: Dict[str, _Reservation] = {}
        for name in entries:
            match = _SAFE_RESERVATION_NAME.fullmatch(name)
            if match is None:
                continue
            content, _info = self._read_regular_file(
                root_fd,
                name,
                max_bytes=_MAX_RESERVATION_MARKER_BYTES,
                missing_code="operator_proposal_result_evidence_reservation_invalid",
                unsafe_code="operator_proposal_result_evidence_reservation_unsafe",
            )
            try:
                payload = json.loads(content.decode("utf-8"))
                token = str(payload.get("token") or "")
                reserved_bytes = int(payload.get("reserved_bytes"))
                proposal_id = str(payload.get("proposal_id") or "")
                review_content_digest = str(
                    payload.get("review_content_digest") or ""
                )
                phase = str(payload.get("phase") or "")
                pid = int(payload.get("pid"))
                process_instance_id = str(
                    payload.get("process_instance_id") or ""
                )
            except (AttributeError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ResultEvidenceError(
                    "operator_proposal_result_evidence_reservation_corrupt"
                ) from exc
            if (
                token != match.group(1)
                or reserved_bytes <= 0
                or not proposal_id
                or not is_sha256_digest(review_content_digest)
                or phase not in _RESERVATION_PHASES
                or pid <= 0
                or not re.fullmatch(r"[0-9a-f]{32}", process_instance_id)
            ):
                raise ResultEvidenceError(
                    "operator_proposal_result_evidence_reservation_corrupt"
                )
            reservations[token] = _Reservation(
                name=name,
                token=token,
                reserved_bytes=reserved_bytes,
                proposal_id=proposal_id,
                review_content_digest=review_content_digest,
                phase=phase,
                pid=pid,
                process_instance_id=process_instance_id,
            )
        return reservations

    def _inventory(
        self,
        root_fd: int,
        entries: list[str],
    ) -> Dict[str, _Artifact]:
        artifacts: Dict[str, _Artifact] = {}
        for name in entries:
            if not _SAFE_ARTIFACT_NAME.fullmatch(name):
                continue
            content, info = self._read_regular_file(
                root_fd,
                name,
                max_bytes=self.max_artifact_bytes,
                missing_code="operator_proposal_result_detail_not_retained",
                unsafe_code="operator_proposal_result_evidence_unsafe_target",
            )
            expected_hex = name[:-5]
            if hashlib.sha256(content).hexdigest() != expected_hex:
                raise ResultEvidenceError(
                    "operator_proposal_result_evidence_store_corrupt"
                )
            digest = f"sha256:{expected_hex}"
            pinned, fatal_codes, conservative_recovery = (
                self._classify_result_safety(content)
            )
            artifacts[digest] = _Artifact(
                name=name,
                digest=digest,
                byte_count=len(content),
                pinned=pinned,
                fatal_codes=fatal_codes,
                conservative_recovery=conservative_recovery,
                modified_ns=info.st_mtime_ns,
            )
        return artifacts

    def _scan_store(
        self,
        root_fd: int,
    ) -> tuple[Dict[str, _Artifact], Dict[str, _Reservation]]:
        try:
            entries = list(os.listdir(root_fd))
        except OSError as exc:
            raise ResultEvidenceError(
                "operator_proposal_result_evidence_store_unavailable"
            ) from exc
        if len(entries) > _MAX_DIRECTORY_ENTRIES:
            raise ResultEvidenceError(
                "operator_proposal_result_evidence_directory_entry_limit_exceeded"
            )
        cleaned_temp = False
        for name in entries:
            if (
                _SAFE_TEMP_NAME.fullmatch(name)
                or _SAFE_RESERVATION_TEMP_NAME.fullmatch(name)
            ):
                try:
                    info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                except OSError as exc:
                    raise ResultEvidenceError(
                        "operator_proposal_result_evidence_store_unavailable"
                    ) from exc
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid()
                    or info.st_nlink != 1
                    or stat.S_IMODE(info.st_mode) != 0o600
                ):
                    raise ResultEvidenceError(
                        "operator_proposal_result_evidence_unsafe_temp"
                    )
                os.unlink(name, dir_fd=root_fd)
                cleaned_temp = True
                continue
            if (
                name == _STORE_LOCK_NAME
                or _SAFE_ARTIFACT_NAME.fullmatch(name)
                or _SAFE_RESERVATION_NAME.fullmatch(name)
            ):
                continue
            raise ResultEvidenceError(
                "operator_proposal_result_evidence_unexpected_store_entry"
            )
        if cleaned_temp:
            self._fsync_directory_fd(root_fd, self.root)
            entries = list(os.listdir(root_fd))
        return self._inventory(root_fd, entries), self._reservations(root_fd, entries)

    def _free_bytes(self, root_fd: int) -> int:
        try:
            if self._uses_native_disk_usage:
                values = os.fstatvfs(root_fd)
                return int(values.f_bavail * values.f_frsize)
            return int(self._disk_usage(self.root).free)
        except (OSError, TypeError, ValueError, AttributeError) as exc:
            raise ResultEvidenceError(
                "operator_proposal_result_evidence_capacity_unavailable"
            ) from exc

    def _capacity_satisfied(
        self,
        artifacts: Dict[str, _Artifact],
        reservations: Dict[str, _Reservation],
        *,
        additional_count: int,
        additional_bytes: int,
        root_fd: int,
    ) -> bool:
        reserved_bytes = sum(item.reserved_bytes for item in reservations.values())
        return bool(
            len(artifacts) + len(reservations) + additional_count
            <= self.max_artifacts
            and sum(item.byte_count for item in artifacts.values())
            + reserved_bytes
            + additional_bytes
            <= self.max_total_bytes
            and self._free_bytes(root_fd) - reserved_bytes - additional_bytes
            >= self.min_free_bytes
        )

    def _evict_until_capacity(
        self,
        artifacts: Dict[str, _Artifact],
        reservations: Dict[str, _Reservation],
        *,
        additional_count: int,
        additional_bytes: int,
        root_fd: int,
    ) -> Dict[str, _Artifact]:
        while not self._capacity_satisfied(
            artifacts,
            reservations,
            additional_count=additional_count,
            additional_bytes=additional_bytes,
            root_fd=root_fd,
        ):
            candidates = sorted(
                (item for item in artifacts.values() if not item.pinned),
                key=lambda item: (item.modified_ns, item.digest),
            )
            if not candidates:
                raise ResultEvidenceError(
                    "operator_proposal_result_evidence_capacity_unavailable"
                )
            victim = candidates[0]
            try:
                os.unlink(victim.name, dir_fd=root_fd)
                self._fsync_directory_fd(root_fd, self.root)
            except OSError as exc:
                raise ResultEvidenceError(
                    "operator_proposal_result_evidence_eviction_failed"
                ) from exc
            artifacts.pop(victim.digest, None)
        return artifacts

    @staticmethod
    def _reservation_name(token: str) -> Optional[str]:
        return f".reservation.{token}" if re.fullmatch(r"[0-9a-f]{32}", token) else None

    def _write_new_file(
        self,
        root_fd: int,
        name: str,
        content: bytes,
    ) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor: Optional[int] = None
        created = False
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=root_fd)
            created = True
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short result evidence write")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            self._fsync_directory_fd(root_fd, self.root)
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
                descriptor = None
            if created:
                try:
                    os.unlink(name, dir_fd=root_fd)
                    self._fsync_directory_fd(root_fd, self.root)
                except FileNotFoundError:
                    pass
                except OSError:
                    # The caller never receives this reservation token.  Keep a
                    # local recovery latch; a later process also reaps the
                    # foreign reserved-no-send marker before capacity checks.
                    self._local_recovery_required = True
            raise ResultEvidenceError(
                "operator_proposal_result_evidence_persistence_failed"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _reservation_payload(
        self,
        *,
        token: str,
        proposal_id: str,
        review_content_digest: str,
        phase: str,
        failure_reason: Optional[str] = None,
    ) -> bytes:
        payload = {
            "schema": "automixer.operator_proposal.result_reservation.v2",
            "token": token,
            "reserved_bytes": self.max_artifact_bytes,
            "proposal_id": str(proposal_id),
            "review_content_digest": str(review_content_digest),
            "phase": phase,
            "pid": os.getpid(),
            "process_instance_id": self._process_instance_id,
        }
        if failure_reason:
            payload["failure_reason"] = str(failure_reason)[:256]
        return canonical_json_bytes(payload)

    def _replace_reservation_marker(
        self,
        root_fd: int,
        reservation: _Reservation,
        *,
        phase: str,
        failure_reason: Optional[str] = None,
    ) -> None:
        if phase not in _RESERVATION_PHASES:
            raise ResultEvidenceError(
                "operator_proposal_result_evidence_reservation_phase_invalid"
            )
        temporary = (
            f".reservation.{reservation.token}.{secrets.token_hex(8)}.tmp"
        )
        content = self._reservation_payload(
            token=reservation.token,
            proposal_id=reservation.proposal_id,
            review_content_digest=reservation.review_content_digest,
            phase=phase,
            failure_reason=failure_reason,
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor: Optional[int] = None
        try:
            descriptor = os.open(temporary, flags, 0o600, dir_fd=root_fd)
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short result evidence reservation write")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(
                temporary,
                reservation.name,
                src_dir_fd=root_fd,
                dst_dir_fd=root_fd,
            )
            self._fsync_directory_fd(root_fd, self.root)
        except OSError as exc:
            self._local_recovery_required = True
            raise ResultEvidenceError(
                "operator_proposal_result_evidence_reservation_transition_failed"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=root_fd)
            except FileNotFoundError:
                pass
            except OSError:
                self._local_recovery_required = True

    def reserve(
        self,
        *,
        proposal_id: str,
        review_content_digest: str,
    ) -> str:
        """Reserve one worst-case artifact before any possible console send."""

        bound_proposal_id = str(proposal_id or "").strip()
        bound_review_digest = str(review_content_digest or "").strip()
        if not bound_proposal_id or not is_sha256_digest(bound_review_digest):
            raise ResultEvidenceError(
                "operator_proposal_result_evidence_reservation_binding_required"
            )
        with self._locked_root() as root_fd:
            artifacts, reservations = self._scan_store(root_fd)
            apply_hazards = (
                *self._artifact_recovery_hazards(artifacts),
                *self._possible_send_reservation_hazards(reservations),
                *self._local_apply_hazards(),
            )
            if apply_hazards:
                raise ResultEvidenceError(
                    "operator_proposal_result_recovery_required",
                    hazards=apply_hazards,
                )
            self._evict_until_capacity(
                artifacts,
                reservations,
                additional_count=1,
                additional_bytes=self.max_artifact_bytes,
                root_fd=root_fd,
            )
            token = secrets.token_hex(16)
            while token in reservations:
                token = secrets.token_hex(16)
            marker_name = str(self._reservation_name(token))
            marker = self._reservation_payload(
                token=token,
                proposal_id=bound_proposal_id,
                review_content_digest=bound_review_digest,
                phase=_RESERVATION_PHASE_RESERVED,
            )
            self._write_new_file(root_fd, marker_name, marker)
            if not self._root_path_matches_fd(root_fd):
                try:
                    os.unlink(marker_name, dir_fd=root_fd)
                    self._fsync_directory_fd(root_fd, self.root)
                except OSError:
                    pass
                raise ResultEvidenceError(
                    "operator_proposal_result_evidence_store_unsafe_root"
                )
            return token

    @staticmethod
    def _validate_reservation_binding(
        reservation: _Reservation,
        binding: ResultEvidenceReservationBinding,
    ) -> None:
        if (
            not isinstance(binding, ResultEvidenceReservationBinding)
            or not binding.token
            or not binding.proposal_id
            or not is_sha256_digest(binding.review_content_digest)
        ):
            raise ResultEvidenceError(
                "operator_proposal_result_evidence_reservation_binding_required"
            )
        if (
            reservation.token != binding.token
            or reservation.proposal_id != binding.proposal_id
            or reservation.review_content_digest
            != binding.review_content_digest
        ):
            raise ResultEvidenceError(
                "operator_proposal_result_evidence_reservation_binding_mismatch"
            )

    def mark_send_may_have_occurred(
        self,
        binding: ResultEvidenceReservationBinding,
    ) -> None:
        """Durably cross the no-send boundary before a physical send."""

        token = binding.token if isinstance(
            binding,
            ResultEvidenceReservationBinding,
        ) else ""
        marker_name = self._reservation_name(token)
        with self._locked_root() as root_fd:
            artifacts, reservations = self._scan_store(root_fd)
            reservation = reservations.get(token)
            if marker_name is None or reservation is None:
                raise ResultEvidenceError(
                    "operator_proposal_result_evidence_reservation_invalid"
                )
            self._validate_reservation_binding(reservation, binding)
            apply_hazards = (
                *self._artifact_recovery_hazards(artifacts),
                *self._possible_send_reservation_hazards(
                    reservations,
                    exclude_token=reservation.token,
                ),
                *self._local_apply_hazards(),
            )
            if apply_hazards:
                raise ResultEvidenceError(
                    "operator_proposal_result_recovery_required",
                    hazards=apply_hazards,
                )
            if reservation.phase == _RESERVATION_PHASE_RESERVED:
                self._replace_reservation_marker(
                    root_fd,
                    reservation,
                    phase=_RESERVATION_PHASE_SEND_MAY_HAVE_OCCURRED,
                )
                return
            if reservation.phase == _RESERVATION_PHASE_SEND_MAY_HAVE_OCCURRED:
                return
            raise ResultEvidenceError(
                "operator_proposal_result_evidence_recovery_required"
            )

    def recovery_hazards(self) -> tuple[Dict[str, Any], ...]:
        """Expose retained unsafe results and every possible-send marker."""

        hazards = []
        with self._locked_root() as root_fd:
            artifacts, reservations = self._scan_store(root_fd)
            hazards.extend(self._artifact_recovery_hazards(artifacts))
            reaped = False
            for reservation in reservations.values():
                if reservation.phase != _RESERVATION_PHASE_RESERVED:
                    hazards.append(
                        {
                            "code": "operator_proposal_result_persistence_failed",
                            "proposal_id": reservation.proposal_id,
                            "review_content_digest": reservation.review_content_digest,
                            "phase": reservation.phase,
                        }
                    )
                    continue
                if reservation.process_instance_id == self._process_instance_id:
                    continue
                try:
                    os.unlink(reservation.name, dir_fd=root_fd)
                    reaped = True
                except OSError as exc:
                    self._local_recovery_required = True
                    raise ResultEvidenceError(
                        "operator_proposal_result_evidence_recovery_required"
                    ) from exc
            if reaped:
                self._fsync_directory_fd(root_fd, self.root)
        if self._local_recovery_required:
            hazards.append(
                {
                    "code": "operator_proposal_result_persistence_failed",
                    "proposal_id": None,
                    "review_content_digest": None,
                    "phase": "local_recovery_required",
                }
            )
        return tuple(hazards)

    def release(self, binding: ResultEvidenceReservationBinding) -> None:
        token = binding.token if isinstance(
            binding,
            ResultEvidenceReservationBinding,
        ) else ""
        marker_name = self._reservation_name(token)
        if marker_name is None:
            return
        with self._locked_root() as root_fd:
            _artifacts, reservations = self._scan_store(root_fd)
            reservation = reservations.get(token)
            if reservation is None:
                return
            self._validate_reservation_binding(reservation, binding)
            if reservation.phase != _RESERVATION_PHASE_RESERVED:
                raise ResultEvidenceError(
                    "operator_proposal_result_evidence_recovery_required"
                )
            os.unlink(marker_name, dir_fd=root_fd)
            self._fsync_directory_fd(root_fd, self.root)

    def _write_atomic(self, root_fd: int, target_name: str, content: bytes) -> None:
        try:
            existing, _info = self._read_regular_file(
                root_fd,
                target_name,
                max_bytes=self.max_artifact_bytes,
                missing_code="operator_proposal_result_detail_not_retained",
                unsafe_code="operator_proposal_result_evidence_unsafe_target",
                fsync_before_close=True,
            )
            if existing != content:
                raise ResultEvidenceError(
                    "operator_proposal_result_evidence_digest_collision"
                )
            self._fsync_directory_fd(root_fd, self.root)
            return
        except ResultEvidenceError as exc:
            if exc.code != "operator_proposal_result_detail_not_retained":
                raise

        temporary = f".{target_name[:-5]}.{secrets.token_hex(8)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor: Optional[int] = None
        try:
            descriptor = os.open(temporary, flags, 0o600, dir_fd=root_fd)
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short result evidence write")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(
                temporary,
                target_name,
                src_dir_fd=root_fd,
                dst_dir_fd=root_fd,
            )
            target_fd = os.open(
                target_name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_fd,
            )
            try:
                os.fsync(target_fd)
            finally:
                os.close(target_fd)
            self._fsync_directory_fd(root_fd, self.root)
            if not self._root_path_matches_fd(root_fd):
                raise ResultEvidenceError(
                    "operator_proposal_result_evidence_store_unsafe_root"
                )
        except ResultEvidenceError:
            raise
        except OSError as exc:
            raise ResultEvidenceError(
                "operator_proposal_result_evidence_persistence_failed"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=root_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass

    def commit(
        self,
        binding: ResultEvidenceReservationBinding,
        result: Dict[str, Any],
    ) -> ResultEvidenceRecord:
        """Persist terminal evidence and consume exactly one reservation."""

        token = binding.token if isinstance(
            binding,
            ResultEvidenceReservationBinding,
        ) else ""
        marker_name = self._reservation_name(token)
        with self._locked_root() as root_fd:
            entries = list(os.listdir(root_fd))
            reservations = self._reservations(root_fd, entries)
            if marker_name is None or token not in reservations:
                raise ResultEvidenceError(
                    "operator_proposal_result_evidence_reservation_invalid"
                )
            reservation = reservations[token]
            # Binding mismatch is a caller/integrity fault, not a persistence
            # failure.  Check it before the try/except that advances marker
            # phase so neither reservation is consumed or mutated.
            self._validate_reservation_binding(reservation, binding)
            safe_no_send_failure = bool(
                reservation.phase == _RESERVATION_PHASE_RESERVED
                and _result_proves_exact_safe_no_send(result)
            )
            try:
                content, digest, pinned = self.canonical_result(result)
                if len(content) > self.max_artifact_bytes:
                    raise ResultEvidenceError(
                        "operator_proposal_result_evidence_too_large"
                    )
                self._scan_store(root_fd)
                target_name = self._artifact_name(digest)
                self._write_atomic(root_fd, target_name, content)
                record = ResultEvidenceRecord(
                    digest=digest,
                    byte_count=len(content),
                    artifact_name=target_name,
                    pinned=pinned,
                )
            except Exception as exc:
                if not safe_no_send_failure:
                    try:
                        self._replace_reservation_marker(
                            root_fd,
                            reservation,
                            phase=_RESERVATION_PHASE_PERSISTENCE_FAILED,
                            failure_reason=(
                                exc.code
                                if isinstance(exc, ResultEvidenceError)
                                else type(exc).__name__
                            ),
                        )
                    except ResultEvidenceError:
                        self._local_recovery_required = True
                else:
                    try:
                        os.unlink(marker_name, dir_fd=root_fd)
                        self._fsync_directory_fd(root_fd, self.root)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        self._local_recovery_required = True
                raise
            else:
                try:
                    os.unlink(marker_name, dir_fd=root_fd)
                    self._fsync_directory_fd(root_fd, self.root)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    raise ResultEvidenceError(
                        "operator_proposal_result_evidence_persistence_failed"
                    ) from exc
                return record

    def persist_no_send_result(self, result: Dict[str, Any]) -> ResultEvidenceRecord:
        """Persist evidence for a path already proven to have made no send."""

        if result.get("no_console_send_proven") is not True:
            raise ResultEvidenceError(
                "operator_proposal_result_evidence_reservation_required"
            )
        result = dict(result)
        result.setdefault("write_state_unknown", False)
        result.setdefault("rollback_state_unknown", False)
        result.setdefault("manual_review_required", False)
        token = self.reserve(
            proposal_id="persisted-no-send-result",
            review_content_digest=sha256_digest(canonical_json_bytes(result)),
        )
        return self.commit(
            ResultEvidenceReservationBinding(
                token=token,
                proposal_id="persisted-no-send-result",
                review_content_digest=sha256_digest(
                    canonical_json_bytes(result)
                ),
            ),
            result,
        )

    def load(self, digest: str) -> bytes:
        """Load only a digest-bound regular artifact and reverify its bytes."""

        with self._locked_root() as root_fd:
            target_name = self._artifact_name(digest)
            content, _info = self._read_regular_file(
                root_fd,
                target_name,
                max_bytes=self.max_artifact_bytes,
                missing_code="operator_proposal_result_detail_not_retained",
                unsafe_code="operator_proposal_result_evidence_unsafe_target",
            )
            expected = str(digest)[7:]
            if hashlib.sha256(content).hexdigest() != expected:
                raise ResultEvidenceError(
                    "operator_proposal_result_evidence_store_corrupt"
                )
            return content
