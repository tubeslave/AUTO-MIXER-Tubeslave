"""Opt-in WING Rack laboratory auto-write session.

This module is intentionally separate from the production server. One explicit
arm call opens a bounded local lab session; accepted decisions then execute
without per-decision confirmation. Every value is read before write, written to
a durable audit log, freshly read back, and retained for reverse rollback.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
from functools import wraps
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Protocol
import uuid


LAB_PROFILE = "wing_lab_autowrite"
LAB_SCHEMA = "automixer.wing_lab_autowrite.v1"
ENABLE_ENV = "AUTOMIXER_WING_LAB_AUTOWRITE"
ALLOWED_FAMILIES = frozenset(
    {
        "channel_fader",
        "bus_fader",
        "master_fader",
        "input_gain",
        "channel_eq",
        "channel_gate",
        "channel_compressor",
        "channel_deesser",
        "channel_pan",
        "channel_phase",
        "channel_delay",
        "fx",
        "channel_send",
    }
)
EXCLUDED_FAMILIES = frozenset(
    {"mute", "unmute", "routing", "scene", "snapshot", "preset"}
)


class WingLabTransport(Protocol):
    is_connected: bool

    def connect(self, timeout: float = 5.0) -> bool: ...
    def disconnect(self) -> bool: ...
    def send(self, address: str, *values: Any) -> bool: ...
    def query_parameter(self, address: str, timeout: float = 0.5) -> Mapping[str, Any]: ...

    def install_mutation_boundary(
        self, boundary: Callable[[str, tuple[Any, ...]], bool]
    ) -> object: ...
    def remove_mutation_boundary(self, token: object) -> bool: ...
    def mutation_boundary_matches(self, token: object) -> bool: ...
    def flush_pending_writes(self, timeout: float = 2.0) -> bool: ...


TransportFactory = Callable[[str, int], WingLabTransport]


@dataclass(frozen=True)
class CompiledDecision:
    decision_id: str
    family: str
    reason: str
    confidence: float
    address: str
    value: Any
    tolerance: float
    preconditions: tuple[tuple[str, Any], ...] = ()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _finite(value: Any, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field}_must_be_number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}_must_be_number") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{field}_out_of_range")
    return parsed


def _integer(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field}_must_be_integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field}_out_of_range")
    return value


def _switch(value: Any, field: str) -> int:
    if value is True or value == 1:
        return 1
    if value is False or value == 0:
        return 0
    raise ValueError(f"{field}_must_be_boolean")


def _eq_suffix(decision: Mapping[str, Any]) -> tuple[str, Any, float]:
    parameter = str(decision.get("parameter") or "").strip()
    if parameter == "on":
        return "on", _switch(decision.get("value"), "value"), 0.0
    band = decision.get("band")
    suffixes = {
        "low_shelf_gain": ("lg", -15.0, 15.0, 0.1),
        "low_shelf_frequency": ("lf", 20.0, 2000.0, 2.0),
        "low_shelf_q": ("lq", 0.44, 10.0, 0.02),
        "high_shelf_gain": ("hg", -15.0, 15.0, 0.1),
        "high_shelf_frequency": ("hf", 50.0, 20000.0, 2.0),
        "high_shelf_q": ("hq", 0.44, 10.0, 0.02),
    }
    if parameter in suffixes:
        suffix, minimum, maximum, tolerance = suffixes[parameter]
        return suffix, _finite(decision.get("value"), "value", minimum, maximum), tolerance
    if parameter in {"gain", "frequency", "q"}:
        band_number = _integer(band, "band", 1, 4)
        limits = {
            "gain": ("g", -15.0, 15.0, 0.1),
            "frequency": ("f", 20.0, 20000.0, 2.0),
            "q": ("q", 0.44, 10.0, 0.02),
        }
        token, minimum, maximum, tolerance = limits[parameter]
        return f"{band_number}{token}", _finite(
            decision.get("value"), "value", minimum, maximum
        ), tolerance
    raise ValueError("channel_eq_parameter_not_allowed")


def compile_wing_lab_decision(decision: Mapping[str, Any]) -> CompiledDecision:
    if not isinstance(decision, Mapping):
        raise ValueError("decision_must_be_object")
    decision_id = str(decision.get("id") or "").strip()
    family = str(decision.get("family") or "").strip()
    if not decision_id:
        raise ValueError("decision_id_required")
    if decision.get("accepted") is not True:
        raise ValueError("decision_must_be_algorithm_accepted")
    reason = str(decision.get("reason") or "").strip()
    if not reason or len(reason) > 1000:
        raise ValueError("decision_reason_required")
    confidence = _finite(decision.get("confidence"), "confidence", 0.0, 1.0)
    if family in EXCLUDED_FAMILIES or family not in ALLOWED_FAMILIES:
        raise ValueError("decision_family_not_allowed_in_lab")

    address: str
    value: Any
    tolerance = 0.1
    preconditions: tuple[tuple[str, Any], ...] = ()
    if family == "channel_fader":
        channel = _integer(decision.get("channel"), "channel", 1, 40)
        address = f"/ch/{channel}/fdr"
        value = _finite(decision.get("value"), "value", -144.0, 0.0)
    elif family == "bus_fader":
        bus = _integer(decision.get("bus"), "bus", 1, 16)
        address = f"/bus/{bus}/fdr"
        value = _finite(decision.get("value"), "value", -144.0, 0.0)
    elif family == "master_fader":
        main = _integer(decision.get("main"), "main", 1, 4)
        address = f"/main/{main}/fdr"
        value = _finite(decision.get("value"), "value", -144.0, 0.0)
    elif family == "input_gain":
        channel = _integer(decision.get("channel"), "channel", 1, 40)
        address = f"/ch/{channel}/in/set/trim"
        value = _finite(decision.get("value"), "value", -18.0, 18.0)
    elif family == "channel_eq":
        channel = _integer(decision.get("channel"), "channel", 1, 40)
        suffix, value, tolerance = _eq_suffix(decision)
        address = f"/ch/{channel}/eq/{suffix}"
    elif family in {"channel_gate", "channel_compressor"}:
        channel = _integer(decision.get("channel"), "channel", 1, 40)
        parameter = str(decision.get("parameter") or "").strip()
        base = "gate" if family == "channel_gate" else "dyn"
        specs = (
            {
                "on": ("on", "switch", 0.0, 1.0, 0.0),
                "threshold": ("thr", "number", -80.0, 0.0, 0.1),
                "range": ("range", "number", 3.0, 60.0, 0.1),
                "attack": ("att", "number", 0.0, 120.0, 0.5),
                "hold": ("hld", "number", 0.0, 200.0, 0.5),
                "release": ("rel", "number", 4.0, 4000.0, 1.0),
            }
            if family == "channel_gate"
            else {
                "on": ("on", "switch", 0.0, 1.0, 0.0),
                "threshold": ("thr", "number", -60.0, 0.0, 0.1),
                "attack": ("att", "number", 0.0, 120.0, 0.5),
                "release": ("rel", "number", 4.0, 4000.0, 1.0),
                "makeup": ("gain", "number", -6.0, 12.0, 0.1),
                "mix": ("mix", "number", 0.0, 100.0, 0.5),
            }
        )
        if parameter == "ratio" and family == "channel_compressor":
            allowed = {
                "1.1", "1.2", "1.3", "1.5", "1.7", "2.0", "2.5",
                "3.0", "3.5", "4.0", "5.0", "6.0", "8.0", "10",
                "20", "50", "100",
            }
            value = str(decision.get("value") or "")
            if value not in allowed:
                raise ValueError("compressor_ratio_not_allowed")
            address = f"/ch/{channel}/dyn/ratio"
            tolerance = 0.0
        else:
            if parameter not in specs:
                raise ValueError(f"{family}_parameter_not_allowed")
            suffix, kind, minimum, maximum, tolerance = specs[parameter]
            value = (
                _switch(decision.get("value"), "value")
                if kind == "switch"
                else _finite(decision.get("value"), "value", minimum, maximum)
            )
            address = f"/ch/{channel}/{base}/{suffix}"
    elif family == "channel_deesser":
        channel = _integer(decision.get("channel"), "channel", 1, 40)
        parameter = str(decision.get("parameter") or "").strip()
        specs = {
            "on": ("on", "switch", 0.0, 1.0, 0.0),
            "frequency": ("f", "number", 800.0, 8000.0, 2.0),
            "range": ("range", "number", 3.0, 12.0, 0.1),
            "mode": ("mode", "mode", 0.0, 0.0, 0.0),
        }
        if parameter not in specs:
            raise ValueError("channel_deesser_parameter_not_allowed")
        suffix, kind, minimum, maximum, tolerance = specs[parameter]
        if kind == "switch":
            value = _switch(decision.get("value"), "value")
        elif kind == "mode":
            value = str(decision.get("value") or "").upper()
            if value not in {"FULL", "HF"}:
                raise ValueError("channel_deesser_mode_not_allowed")
        else:
            value = _finite(decision.get("value"), "value", minimum, maximum)
        address = f"/ch/{channel}/gate/{suffix}"
        preconditions = ((f"/ch/{channel}/gate/mdl", "DS902"),)
    elif family == "channel_pan":
        channel = _integer(decision.get("channel"), "channel", 1, 40)
        address = f"/ch/{channel}/pan"
        value = _finite(decision.get("value"), "value", -100.0, 100.0)
        tolerance = 1.0
    elif family == "channel_phase":
        channel = _integer(decision.get("channel"), "channel", 1, 40)
        address = f"/ch/{channel}/in/set/inv"
        value = _switch(decision.get("value"), "value")
        tolerance = 0.0
    elif family == "channel_delay":
        channel = _integer(decision.get("channel"), "channel", 1, 40)
        address = f"/ch/{channel}/in/set/dly"
        value = _finite(decision.get("value"), "value", 0.5, 500.0)
        tolerance = 0.5
    elif family == "channel_send":
        channel = _integer(decision.get("channel"), "channel", 1, 40)
        send = _integer(decision.get("send"), "send", 1, 16)
        address = f"/ch/{channel}/send/{send}/lvl"
        value = _finite(decision.get("value"), "value", -144.0, 0.0)
    else:
        slot = _integer(decision.get("slot"), "slot", 1, 16)
        parameter = str(decision.get("parameter") or "").strip()
        if parameter == "mix":
            address = f"/fx/{slot}/fxmix"
            value = _finite(decision.get("value"), "value", 0.0, 100.0)
            tolerance = 0.5
        elif parameter == "parameter":
            index = _integer(decision.get("index"), "index", 1, 32)
            minimum = _finite(decision.get("minimum"), "minimum", -100000.0, 100000.0)
            maximum = _finite(decision.get("maximum"), "maximum", -100000.0, 100000.0)
            if minimum >= maximum:
                raise ValueError("fx_parameter_bounds_invalid")
            address = f"/fx/{slot}/{index}"
            value = _finite(decision.get("value"), "value", minimum, maximum)
            tolerance = _finite(decision.get("tolerance", 0.1), "tolerance", 0.0, 10.0)
        else:
            raise ValueError("fx_parameter_not_allowed")

    return CompiledDecision(
        decision_id=decision_id,
        family=family,
        reason=reason,
        confidence=confidence,
        address=address,
        value=value,
        tolerance=tolerance,
        preconditions=preconditions,
    )


def _matches(actual: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= tolerance
        except (TypeError, ValueError):
            return False
    return actual == expected


def _serialized(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._session_lock:
            if self._active_call_thread == threading.get_ident():
                raise RuntimeError("wing_lab_reentrant_session_call")
            self._active_call_thread = threading.get_ident()
            try:
                return method(self, *args, **kwargs)
            finally:
                self._active_call_thread = None
    return locked


class WingLabAutowriteSession:
    def __init__(
        self,
        transport_factory: TransportFactory,
        *,
        runtime_root: str | Path | None = None,
        enabled: bool | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._factory = transport_factory
        self._runtime_root = Path(runtime_root or Path.home() / ".automixer/runtime")
        self._enabled = (
            os.environ.get(ENABLE_ENV) == "1" if enabled is None else bool(enabled)
        )
        self._clock = clock
        self._transport: WingLabTransport | None = None
        self._lock_fd: int | None = None
        self._session_id: str | None = None
        self._expires_at = 0.0
        self._max_corrections = 0
        self._applied: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._previous_hash = "sha256:" + "0" * 64
        self._wal_path: Path | None = None
        self._emergency_stopped = False
        self._session_lock = threading.RLock()
        self._active_call_thread: int | None = None
        self._boundary_token: object | None = None
        self._grant: tuple[str, Any, bool] | None = None
        self._peer: tuple[str, int] | None = None

    def status(self) -> dict[str, Any]:
        return {
            "schema": LAB_SCHEMA,
            "profile": LAB_PROFILE,
            "enabled": self._enabled,
            "armed": self._session_id is not None,
            "session_id": self._session_id,
            "expires_at_epoch": self._expires_at or None,
            "applied_count": len(self._applied),
            "remaining_capacity": max(0, self._max_corrections - len(self._applied)),
            "emergency_stopped": self._emergency_stopped,
            "allowed_families": sorted(ALLOWED_FAMILIES),
            "excluded_families": sorted(EXCLUDED_FAMILIES),
            "per_decision_confirmation_required": False,
            "production_policy_changed": False,
        }

    def _append_event(self, phase: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._wal_path is None:
            raise RuntimeError("wing_lab_wal_not_open")
        event = {
            "schema": LAB_SCHEMA,
            "session_id": self._session_id,
            "phase": phase,
            "timestamp_epoch": self._clock(),
            "previous_event_sha256": self._previous_hash,
            "payload": dict(payload),
        }
        event_hash = "sha256:" + hashlib.sha256(_canonical_bytes(event)).hexdigest()
        event["event_sha256"] = event_hash
        encoded = _canonical_bytes(event) + b"\n"
        descriptor = os.open(self._wal_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("wing_lab_wal_short_write")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory_fd = os.open(self._wal_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self._previous_hash = event_hash
        self._events.append(event)
        return event

    def _fresh_value(self, address: str) -> Any:
        if self._transport is None:
            raise RuntimeError("wing_lab_transport_not_connected")
        receipt = self._transport.query_parameter(address, timeout=0.75)
        if not (
            isinstance(receipt, Mapping)
            and receipt.get("status") == "received"
            and receipt.get("confirmed_round_trips") == 2
            and (receipt.get("peer_ip"), receipt.get("peer_port")) == self._peer
        ):
            raise RuntimeError(f"wing_lab_fresh_readback_failed:{address}")
        return receipt.get("value")

    def _allows_write(self, address: str, values: tuple[Any, ...]) -> bool:
        # Called under transport locks, including by the flusher: never acquire
        # the session lock or perform I/O here.
        grant = self._grant
        if grant is None or self._session_id is None or len(values) != 1:
            return False
        expected_address, expected_value, restoring = grant
        return bool(
            address == expected_address
            and type(values[0]) is type(expected_value)
            and values[0] == expected_value
            and (restoring or (
                not self._emergency_stopped and self._clock() < self._expires_at
            ))
        )

    def _send_owned(self, applied: Mapping[str, Any], *, restoring: bool = False) -> None:
        transport = self._transport
        if transport is None or self._boundary_token is None:
            raise RuntimeError("wing_lab_transport_not_owned")
        if not transport.mutation_boundary_matches(self._boundary_token):
            raise RuntimeError("wing_lab_transport_ownership_lost")
        self._grant = None
        # Old queued commands must be denied before issuing another capability.
        if not transport.flush_pending_writes(timeout=2.0):
            raise RuntimeError("wing_lab_pending_writes_not_drained")
        if not restoring:
            self._ensure_active()
        elif (not any(row is applied for row in self._applied)
              or applied.get("physically_restored") is True):
            raise RuntimeError("wing_lab_unowned_rollback")
        value = applied["preimage"] if restoring else applied["target"]
        self._grant = (applied["address"], value, restoring)
        try:
            if not transport.send(applied["address"], value):
                raise RuntimeError(f"wing_lab_send_failed:{applied['address']}")
            if not transport.flush_pending_writes(timeout=2.0):
                raise RuntimeError("wing_lab_pending_writes_not_drained")
        finally:
            self._grant = None

    @_serialized
    def arm(
        self,
        *,
        host: str,
        port: int = 2223,
        console_model: str,
        duration_sec: int = 900,
        max_corrections: int = 128,
    ) -> dict[str, Any]:
        if not self._enabled:
            raise PermissionError(f"wing_lab_disabled_set_{ENABLE_ENV}_to_1")
        if self._session_id is not None or self._transport is not None or self._lock_fd is not None:
            raise RuntimeError("wing_lab_session_already_armed")
        self._emergency_stopped = False
        if console_model.strip().lower() != "wing rack":
            raise ValueError("wing_lab_console_model_must_be_wing_rack")
        if not isinstance(host, str) or not host.strip() or len(host) > 255:
            raise ValueError("wing_lab_host_invalid")
        try:
            host_ip = ipaddress.ip_address(host.strip())
        except ValueError as exc:
            raise ValueError("wing_lab_host_must_be_private_ipv4") from exc
        private_networks = (
            ipaddress.ip_network("10.0.0.0/8"),
            ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("192.168.0.0/16"),
        )
        if host_ip.version != 4 or not any(host_ip in network for network in private_networks):
            raise ValueError("wing_lab_host_must_be_private_ipv4")
        port = _integer(port, "port", 1, 65535)
        duration_sec = _integer(duration_sec, "duration_sec", 30, 1800)
        max_corrections = _integer(max_corrections, "max_corrections", 1, 256)

        lab_root = self._runtime_root / "wing_lab_autowrite"
        lab_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(lab_root, 0o700)
        lock_path = self._runtime_root / "console_writer.lock"
        lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self._lock_fd)
            self._lock_fd = None
            raise RuntimeError("wing_lab_console_writer_lock_busy") from exc

        try:
            for previous in lab_root.glob("*.jsonl"):
                try:
                    rows = [json.loads(line) for line in previous.read_text().splitlines()]
                    previous_hash = "sha256:" + "0" * 64
                    for row in rows:
                        body = {k: v for k, v in row.items() if k != "event_sha256"}
                        actual_hash = "sha256:" + hashlib.sha256(_canonical_bytes(body)).hexdigest()
                        if (row.get("previous_event_sha256") != previous_hash
                                or row.get("event_sha256") != actual_hash):
                            raise ValueError("invalid_journal_chain")
                        previous_hash = actual_hash
                    if (not rows or rows[-1].get("phase") != "teardown_verified"
                            or rows[-1].get("payload", {}).get("transport_disconnected") is not True):
                        raise ValueError("journal_not_closed")
                except (OSError, ValueError, TypeError, AttributeError) as exc:
                    raise RuntimeError("wing_lab_previous_session_requires_recovery") from exc
            transport = self._factory(host.strip(), port)
            # Install before connect. Do not disconnect or revoke a transport
            # that already belongs to another writer.
            token = transport.install_mutation_boundary(self._allows_write)
            self._transport = transport
            self._boundary_token = token
            self._peer = (host.strip(), port)
            if not transport.connect(timeout=5.0):
                raise RuntimeError("wing_lab_wing_connection_failed")
            handshake_info = str(getattr(transport, "handshake_info", "") or "")
            normalized_handshake = handshake_info.upper()
            if "WING" not in normalized_handshake or "RACK" not in normalized_handshake:
                raise RuntimeError("wing_lab_wing_rack_identity_not_observed")
            if self._emergency_stopped:
                raise RuntimeError("wing_lab_arm_cancelled")
        except BaseException:
            self._teardown()
            raise
        self._events = []
        self._previous_hash = "sha256:" + "0" * 64
        self._session_id = f"wing-lab-{uuid.uuid4()}"
        self._expires_at = self._clock() + duration_sec
        self._max_corrections = max_corrections
        self._wal_path = lab_root / f"{self._session_id}.jsonl"
        try:
            self._append_event("armed", {
                "host": host.strip(),
                "port": port,
                "console_model": "WING Rack",
                "duration_sec": duration_sec,
                "max_corrections": max_corrections,
                "one_time_session_arm": True,
                "per_decision_confirmation_required": False,
            })
        except BaseException:
            self._teardown()
            raise
        return self.status()

    def _ensure_active(self) -> None:
        if self._session_id is None or self._transport is None:
            raise RuntimeError("wing_lab_session_not_armed")
        if self._emergency_stopped:
            raise RuntimeError("wing_lab_emergency_stopped")
        if self._clock() >= self._expires_at:
            self._emergency_stopped = True
            raise RuntimeError("wing_lab_session_expired")

    @_serialized
    def apply_decisions(self, decisions: Any) -> dict[str, Any]:
        try:
            self._ensure_active()
        except RuntimeError:
            if self._session_id is not None and self._emergency_stopped:
                self._finish_stop(reason="inactive_batch", emergency=True)
            raise
        if not isinstance(decisions, list) or not decisions:
            raise ValueError("wing_lab_decisions_must_be_nonempty_list")
        if len(self._applied) + len(decisions) > self._max_corrections:
            raise RuntimeError("wing_lab_correction_budget_exceeded")
        compiled = [compile_wing_lab_decision(row) for row in decisions]
        existing_ids = {row["decision_id"] for row in self._applied}
        compiled_ids = [row.decision_id for row in compiled]
        if len(set(compiled_ids)) != len(compiled_ids) or existing_ids.intersection(
            compiled_ids
        ):
            raise ValueError("wing_lab_duplicate_decision_id")

        batch_results: list[dict[str, Any]] = []
        try:
            for operation in compiled:
                self._ensure_active()
                for address, expected in operation.preconditions:
                    observed = self._fresh_value(address)
                    if observed != expected:
                        raise RuntimeError(
                            f"wing_lab_precondition_failed:{address}:{observed!r}"
                        )
                preimage = self._fresh_value(operation.address)
                if preimage is None or isinstance(preimage, (dict, list, tuple)):
                    raise RuntimeError(
                        f"wing_lab_preimage_missing_or_invalid:{operation.address}"
                    )
                self._append_event(
                    "prepared",
                    {
                        "decision_id": operation.decision_id,
                        "family": operation.family,
                        "reason": operation.reason,
                        "confidence": operation.confidence,
                        "address": operation.address,
                        "preimage": preimage,
                        "target": operation.value,
                    },
                )
                applied = {
                    "decision_id": operation.decision_id,
                    "family": operation.family,
                    "reason": operation.reason,
                    "confidence": operation.confidence,
                    "address": operation.address,
                    "preimage": preimage,
                    "target": operation.value,
                    "readback": None,
                    "tolerance": operation.tolerance,
                }
                # A send can deliver and then raise. Register recovery before
                # granting any write, not after send or post-send WAL succeeds.
                self._applied.append(applied)
                self._send_owned(applied)
                self._append_event(
                    "write_attempted",
                    {"decision_id": operation.decision_id, "address": operation.address},
                )
                observed = self._fresh_value(operation.address)
                if not _matches(observed, operation.value, operation.tolerance):
                    raise RuntimeError(
                        f"wing_lab_readback_mismatch:{operation.address}:{observed!r}"
                    )
                applied["readback"] = observed
                batch_results.append(dict(applied))
                self._append_event("verified", applied)
                self._ensure_active()
        except BaseException as exc:
            self._grant = None
            self._emergency_stopped = True
            rollback = self._rollback_all(reason="batch_failure")
            try:
                self._append_event("emergency_stopped", {"reason": str(exc), "rollback": rollback})
            finally:
                if not rollback["failures"]:
                    self._teardown()
            raise

        return {
            "schema": LAB_SCHEMA,
            "status": "applied_and_verified",
            "session_id": self._session_id,
            "decisions": batch_results,
            "per_decision_confirmation_requested": False,
            "corrections_written": len(batch_results),
        }

    def _rollback_all(self, *, reason: str) -> dict[str, Any]:
        restored: list[str] = []
        failures: list[str] = []
        blocked_addresses: set[str] = set()
        if self._transport is None and self._applied:
            return {"restored_decision_ids": [], "failures": ["transport_missing"]}
        if self._transport is not None:
            for applied in list(reversed(self._applied)):
                address = applied["address"]
                if address in blocked_addresses:
                    failures.append(f"{applied['decision_id']}:newer_restore_pending")
                    continue
                try:
                    if not applied.get("physically_restored"):
                        current = self._fresh_value(address)
                        if not _matches(current, applied["preimage"], applied["tolerance"]):
                            if not _matches(current, applied["target"], applied["tolerance"]):
                                raise RuntimeError("concurrent_console_change")
                            try:
                                self._append_event("rollback_prepared", {**applied, "reason": reason})
                            except Exception:
                                # The durable original prepared record owns this
                                # preimage even if the journal is now unavailable.
                                pass
                            self._send_owned(applied, restoring=True)
                        observed = self._fresh_value(address)
                        if not _matches(observed, applied["preimage"], applied["tolerance"]):
                            raise RuntimeError(f"readback_mismatch:{observed!r}")
                        # Audit retries must not replay intermediate values over
                        # the original baseline restored by an older operation.
                        applied["physically_restored"] = True
                        applied["restored_at_epoch"] = self._clock()
                    self._append_event("rollback_verified", {**applied, "reason": reason})
                    restored.append(applied["decision_id"])
                    self._applied.remove(applied)
                except Exception as exc:
                    if not applied.get("physically_restored"):
                        blocked_addresses.add(address)
                    failures.append(f"{applied['decision_id']}:{exc}")
        return {"restored_decision_ids": restored, "failures": failures}

    def stop(self, *, restore: bool = True) -> dict[str, Any]:
        self._grant = None
        self._emergency_stopped = True
        if self._active_call_thread == threading.get_ident():
            return {"schema": LAB_SCHEMA, "status": "stop_requested"}
        return self._stop(restore=restore)

    @_serialized
    def _stop(self, *, restore: bool) -> dict[str, Any]:
        if not restore and self._applied:
            raise PermissionError("wing_lab_stop_without_restore_forbidden")
        return self._finish_stop(reason="session_stop", emergency=False)

    def emergency_stop(self, *, reason: str = "operator_emergency_stop") -> dict[str, Any]:
        # Latch before waiting for the batch lock so the flusher and next
        # operation immediately lose forward authority.
        self._emergency_stopped = True
        self._grant = None
        if self._active_call_thread == threading.get_ident():
            return {"schema": LAB_SCHEMA, "status": "stop_requested"}
        return self._emergency_stop(reason=reason)

    @_serialized
    def _emergency_stop(self, *, reason: str) -> dict[str, Any]:
        return self._finish_stop(reason=reason, emergency=True)

    def _finish_stop(self, *, reason: str, emergency: bool) -> dict[str, Any]:
        if self._session_id is None and self._transport is None:
            return {"schema": LAB_SCHEMA, "status": "already_stopped"}
        rollback = self._rollback_all(reason=reason)
        self._emergency_stopped = True
        phase = "emergency_stopped" if emergency else "stopped"
        teardown = {}
        try:
            if self._wal_path is not None:
                self._append_event(phase, {"reason": reason, "rollback": rollback})
        finally:
            if not rollback["failures"]:
                teardown = self._teardown()
        return {
            "schema": LAB_SCHEMA,
            "status": "recovery_required" if rollback["failures"] else phase,
            "rollback": rollback,
            **teardown,
        }

    def trace(self) -> dict[str, Any]:
        return {
            "schema": LAB_SCHEMA,
            "session_id": self._session_id,
            "events": list(self._events),
            "wal_path": str(self._wal_path) if self._wal_path else None,
        }

    def _release_lock(self) -> None:
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None

    def _teardown(self) -> dict[str, Any]:
        self._grant = None
        disconnected = True
        if self._transport is not None:
            disconnected = bool(self._transport.disconnect())
            if not disconnected:
                raise RuntimeError("wing_lab_disconnect_unverified")
            if self._boundary_token is not None:
                if not self._transport.remove_mutation_boundary(self._boundary_token):
                    raise RuntimeError("wing_lab_boundary_release_unverified")
                self._boundary_token = None
        if self._session_id is not None and self._wal_path is not None:
            self._append_event("teardown_verified", {"transport_disconnected": disconnected})
        self._boundary_token = None
        self._transport = None
        self._release_lock()
        self._session_id = None
        self._expires_at = 0.0
        self._max_corrections = 0
        self._applied.clear()
        return {"transport_disconnected": disconnected, "writer_lock_released": True}
