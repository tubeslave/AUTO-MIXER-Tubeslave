"""Fixed M110 DSP-bypass reset contract and physical preimage capture.

This module deliberately owns a finite maintenance surface.  It does not add a
normal live write kind and it never calls the legacy broad channel reset.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import inspect
import json
import math
import threading
import time
from typing import Any, Callable, Mapping

from supervised_family_proof_plan import (
    build_canonical_operation_bundle_from_prepared,
    canonical_bundle_scope,
    canonical_decimal_string,
    canonical_operation_bundle_bytes,
    validate_canonical_operation_bundle,
)
from supervised_proof_capability import (
    M110_RESTORED_PROOF_RECEIPT_LATEST_START_SECONDS,
    M110RestoredProofReceipt,
    MaintenanceDspResetCommitOnceGrant,
    RestoreOnlyProofGrant,
)


M110_RESET_CHANNELS = tuple(range(1, 17))
M110_RESET_SUFFIX_TARGETS = (
    ("eq/on", 0),
    ("flt/lc", 0),
    ("gate/on", 0),
    ("dyn/on", 0),
)
M110_RESET_ADDRESSES = tuple(
    f"/ch/{channel}/{suffix}"
    for channel in M110_RESET_CHANNELS
    for suffix, _target in M110_RESET_SUFFIX_TARGETS
)
M110_RESET_TARGETS = {
    f"/ch/{channel}/{suffix}": target
    for channel in M110_RESET_CHANNELS
    for suffix, target in M110_RESET_SUFFIX_TARGETS
}
M110_RESET_FAMILY_ID = "maintenance_dsp_reset"
M110_RESET_WRITE_KIND = "processor_bypass"
M110_RESET_BUNDLE_KIND = "maintenance_dsp_reset_bundle"
M110_RESET_ADAPTER_REVISION = "m110-maintenance-dsp-reset.v1"
M110_RESET_PROTOCOL_REVISION = "wing-osc-fw3.0.5"
M110_RESET_PREIMAGE_SCHEMA = "automixer.m110_dsp_reset_preimage.v1"
M110_RESET_MANIFEST_SCHEMA = "automixer.m110_preservation_manifest.v1"
M110_RESET_SWEEP_DEADLINE_SECONDS = 120.0
M110_RESET_QUERY_TIMEOUT_SECONDS = 0.25
M110_RESET_REVIEW_PREIMAGE_SCHEMA = (
    "automixer.m110_dsp_reset_review_preimage.v1"
)
class M110SafetyCallCancelled(asyncio.CancelledError):
    """Cancellation delivered only after a blocking safety call has drained."""

    def __init__(self, *, outcome: Any = None, error: BaseException | None = None):
        super().__init__("m110_blocking_safety_call_completed_after_cancellation")
        self.outcome = outcome
        self.error = error


class M110DurableHookFailure(RuntimeError):
    """Preserve caller-cancellation truth when a drained hook also fails."""

    def __init__(self, error: BaseException, *, cancellation_requested: bool):
        super().__init__("m110_durable_hook_failed_after_drain")
        self.error = error
        self.cancellation_requested = bool(cancellation_requested)


async def _safety_thread_call(function: Callable[..., Any], *args: Any) -> Any:
    """Do not let cancellation race a still-running physical read or send."""

    task = asyncio.create_task(asyncio.to_thread(function, *args))
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
            continue
        except BaseException:
            if cancellation is None:
                raise
            break
    outcome = None
    error = None
    try:
        outcome = task.result()
    except BaseException as exc:  # pragma: no branch - one terminal outcome
        error = exc
    if cancellation is not None:
        raise M110SafetyCallCancelled(outcome=outcome, error=error) from cancellation
    if error is not None:
        raise error
    return outcome


@dataclass(frozen=True, slots=True)
class M110PreservationSpec:
    address: str
    value_kind: str
    tolerance: str


def _preservation_specs() -> tuple[M110PreservationSpec, ...]:
    rows: list[M110PreservationSpec] = []
    for channel in M110_RESET_CHANNELS:
        base = f"/ch/{channel}"
        rows.extend(
            (
                M110PreservationSpec(f"{base}/in/set/trim", "decimal", "0.05"),
                M110PreservationSpec(f"{base}/fdr", "decimal", "0.05"),
                M110PreservationSpec(f"{base}/mute", "binary", "0"),
                M110PreservationSpec(f"{base}/pan", "decimal", "0.1"),
                M110PreservationSpec(f"{base}/in/set/inv", "binary", "0"),
                M110PreservationSpec(f"{base}/in/set/dly", "decimal", "0.01"),
                M110PreservationSpec(f"{base}/in/set/dlyon", "binary", "0"),
                M110PreservationSpec(f"{base}/in/conn/grp", "text", "0"),
                M110PreservationSpec(f"{base}/in/conn/in", "integer", "0"),
                M110PreservationSpec(f"{base}/in/conn/altgrp", "text", "0"),
                M110PreservationSpec(f"{base}/in/conn/altin", "integer", "0"),
            )
        )
    rows.extend(
        (
            M110PreservationSpec("/main/1/fdr", "decimal", "0.05"),
            M110PreservationSpec("/main/1/mute", "binary", "0"),
            M110PreservationSpec("/$syscfg/$serial", "text", "0"),
            M110PreservationSpec("/$syscfg/$firmware", "text", "0"),
            M110PreservationSpec("/io/out/USB/47/grp", "text", "0"),
            M110PreservationSpec("/io/out/USB/47/in", "integer", "0"),
            M110PreservationSpec("/io/out/USB/48/grp", "text", "0"),
            M110PreservationSpec("/io/out/USB/48/in", "integer", "0"),
        )
    )
    return tuple(sorted(rows, key=lambda row: row.address))


M110_PRESERVATION_SPECS = _preservation_specs()
M110_PRESERVATION_ADDRESSES = tuple(
    row.address for row in M110_PRESERVATION_SPECS
)
if len(M110_RESET_ADDRESSES) != 64:  # pragma: no cover - import invariant
    raise RuntimeError("m110_reset_address_count_invalid")
if len(M110_PRESERVATION_ADDRESSES) != 184:  # pragma: no cover
    raise RuntimeError("m110_preservation_address_count_invalid")
if len(set(M110_PRESERVATION_ADDRESSES)) != 184:  # pragma: no cover
    raise RuntimeError("m110_preservation_addresses_not_unique")
if set(M110_RESET_ADDRESSES).intersection(M110_PRESERVATION_ADDRESSES):
    raise RuntimeError("m110_reset_preservation_surface_overlap")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _audit_safe_value(
    value: Any,
    *,
    _seen: set[int] | None = None,
    _depth: int = 0,
) -> Any:
    """Project hostile values into small, total and JSON-safe evidence."""

    max_depth = 4
    max_items = 32
    max_text_chars = 512
    seen = _seen if _seen is not None else set()
    try:
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, str):
            if len(value) <= max_text_chars:
                return value
            return {
                "type": "string",
                "character_count": len(value),
                "prefix": value[:max_text_chars],
                "truncated": True,
            }
        if isinstance(value, float):
            return value if math.isfinite(value) else {"type": "non_finite_float"}
        if isinstance(value, bytes):
            prefix = value[:4096]
            return {
                "type": "bytes",
                "size": len(value),
                "prefix_sha256": _sha256(prefix),
                "digest_scope_bytes": len(prefix),
                "truncated": len(value) > len(prefix),
            }
        if isinstance(value, (Mapping, list, tuple)):
            identity = id(value)
            if identity in seen:
                return {"type": "cycle", "container": type(value).__name__}
            if _depth >= max_depth:
                return {
                    "type": "depth_limit",
                    "container": type(value).__name__,
                }
            seen.add(identity)
            if isinstance(value, Mapping):
                projected: dict[str, Any] = {}
                item_count = 0
                for key, item in value.items():
                    if item_count >= max_items:
                        projected["__audit_truncated__"] = True
                        break
                    try:
                        key_text = str(key)
                    except BaseException:
                        key_text = f"<{type(key).__name__}>"
                    if len(key_text) > max_text_chars:
                        key_text = key_text[:max_text_chars] + "..."
                    projected[key_text] = _audit_safe_value(
                        item,
                        _seen=seen,
                        _depth=_depth + 1,
                    )
                    item_count += 1
                return projected
            projected_items = []
            for index, item in enumerate(value):
                if index >= max_items:
                    projected_items.append({"type": "items_truncated"})
                    break
                projected_items.append(
                    _audit_safe_value(
                        item,
                        _seen=seen,
                        _depth=_depth + 1,
                    )
                )
            return projected_items
        return {"type": type(value).__name__}
    except BaseException as exc:
        return {
            "type": "audit_projection_error",
            "value_type": type(value).__name__,
            "error_type": type(exc).__name__,
        }


def _normalize_binary(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number) and number in {0.0, 1.0}:
            return int(number)
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    raise ValueError("m110_binary_value_invalid")


def _normalize_integer(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("m110_integer_value_invalid")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("m110_integer_value_invalid") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError("m110_integer_value_invalid")
    return int(number)


def _normalize_manifest_value(value: Any, kind: str) -> Any:
    if kind == "binary":
        return _normalize_binary(value)
    if kind == "integer":
        return _normalize_integer(value)
    if kind == "decimal":
        return canonical_decimal_string(value)
    if kind == "text":
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ValueError("m110_text_value_invalid")
        return value
    raise ValueError("m110_manifest_value_kind_invalid")


def fixed_m110_reset_operation_definitions() -> tuple[dict[str, Any], ...]:
    """Return the exact channel-major 64-operation canonical reset surface."""

    rows = []
    parameter_by_suffix = {
        "eq/on": "eq_enabled",
        "flt/lc": "hpf_enabled",
        "gate/on": "gate_enabled",
        "dyn/on": "compressor_enabled",
    }
    for channel in M110_RESET_CHANNELS:
        for suffix, target in M110_RESET_SUFFIX_TARGETS:
            address = f"/ch/{channel}/{suffix}"
            rows.append(
                {
                    "operation_id": (
                        f"m110-reset-ch{channel:02d}-"
                        + suffix.replace("/", "-")
                    ),
                    "family_id": M110_RESET_FAMILY_ID,
                    "write_kind": M110_RESET_WRITE_KIND,
                    "channel": channel,
                    "parameter": parameter_by_suffix[suffix],
                    "address": address,
                    "target_value": target,
                    "tolerance": 0.0,
                }
            )
    return tuple(rows)


M110_RESET_OPERATION_DEFINITIONS = fixed_m110_reset_operation_definitions()


def fixed_m110_reset_prepared_operations() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "addresses": (row["address"],),
            "write_count": 1,
            "writes": (
                {
                    "address": row["address"],
                    "target_value": row["target_value"],
                    "tolerance": row["tolerance"],
                },
            ),
            "audible_state_reads": (),
        }
        for row in M110_RESET_OPERATION_DEFINITIONS
    )


def fixed_m110_reset_bundle_operations() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            key: row[key]
            for key in (
                "operation_id",
                "family_id",
                "write_kind",
                "channel",
                "parameter",
            )
        }
        for row in M110_RESET_OPERATION_DEFINITIONS
    )


def build_fixed_m110_reset_canonical_bundle(
    safety_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the only canonical 64-operation M110 reset bundle."""

    return build_canonical_operation_bundle_from_prepared(
        bundle_kind=M110_RESET_BUNDLE_KIND,
        operations=fixed_m110_reset_bundle_operations(),
        prepared_operations=fixed_m110_reset_prepared_operations(),
        safety_binding=dict(safety_binding),
    )


def _validate_query_result(
    result: Any,
    *,
    address: str,
    value_kind: str,
) -> tuple[Any, dict[str, Any]]:
    evidence = dict(result) if isinstance(result, Mapping) else {
        "status": "invalid_response",
        "response_type": type(result).__name__,
    }
    confirmed = evidence.get("confirmed_round_trips")
    if not (
        isinstance(result, Mapping)
        and result.get("status") == "received"
        and result.get("address") == address
        and isinstance(confirmed, int)
        and not isinstance(confirmed, bool)
        and confirmed >= 2
    ):
        raise ValueError(f"m110_physical_query_unverified:{address}")
    try:
        value = _normalize_manifest_value(result.get("value"), value_kind)
    except ValueError as exc:
        raise ValueError(f"m110_physical_value_invalid:{address}") from exc
    return value, evidence


async def _call_query(
    query_parameter: Callable[..., Any],
    address: str,
    timeout: float,
) -> Any:
    result = await _safety_thread_call(query_parameter, address, timeout)
    if inspect.isawaitable(result):
        result = await result
    return result


async def _capture_rows(
    query_parameter: Callable[..., Any],
    specs: tuple[M110PreservationSpec, ...],
    *,
    timeout: float,
    sweep_deadline_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for spec in specs:
        elapsed = time.monotonic() - started
        remaining = sweep_deadline_seconds - elapsed
        if remaining <= 0:
            raise TimeoutError("m110_physical_sweep_deadline_exceeded")
        # WingClient.query_parameter can wait for two phases.  Bound each phase
        # to half of the remaining sweep so the 120-second contract cannot be
        # exceeded by the final query.
        query_timeout = min(float(timeout), remaining / 2.0)
        if query_timeout <= 0:
            raise TimeoutError("m110_physical_sweep_deadline_exceeded")
        result = await _call_query(
            query_parameter,
            spec.address,
            query_timeout,
        )
        if time.monotonic() - started > sweep_deadline_seconds:
            raise TimeoutError("m110_physical_sweep_deadline_exceeded")
        value, query_evidence = _validate_query_result(
            result,
            address=spec.address,
            value_kind=spec.value_kind,
        )
        rows.append(
            {
                "address": spec.address,
                "value": value,
                "value_kind": spec.value_kind,
                "tolerance": spec.tolerance,
            }
        )
        evidence.append(query_evidence)
    return rows, evidence


def _reset_specs() -> tuple[M110PreservationSpec, ...]:
    return tuple(
        M110PreservationSpec(address, "binary", "0")
        for address in M110_RESET_ADDRESSES
    )


M110_RESET_PREIMAGE_SPECS = _reset_specs()


def preservation_manifest_document(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(rows) != len(M110_PRESERVATION_SPECS):
        raise ValueError("m110_preservation_manifest_row_count_invalid")
    if [row.get("address") for row in rows] != list(
        M110_PRESERVATION_ADDRESSES
    ):
        raise ValueError("m110_preservation_manifest_address_order_invalid")
    return {
        "schema": M110_RESET_MANIFEST_SCHEMA,
        "row_count": len(rows),
        "rows": [dict(row) for row in rows],
    }


def compare_preservation_manifest_rows(
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if len(before_rows) != 184 or len(after_rows) != 184:
        raise ValueError("m110_preservation_manifest_row_count_invalid")
    mismatches = []
    for spec, before, after in zip(
        M110_PRESERVATION_SPECS,
        before_rows,
        after_rows,
    ):
        if before.get("address") != spec.address or after.get("address") != spec.address:
            raise ValueError("m110_preservation_manifest_address_order_invalid")
        matches = False
        if spec.value_kind == "decimal":
            try:
                left = Decimal(str(before.get("value")))
                right = Decimal(str(after.get("value")))
                tolerance = Decimal(spec.tolerance)
                matches = bool(
                    left.is_finite()
                    and right.is_finite()
                    and abs(left - right) <= tolerance
                )
            except (InvalidOperation, TypeError, ValueError):
                matches = False
        else:
            matches = before.get("value") == after.get("value")
        if not matches:
            mismatches.append(
                {
                    "address": spec.address,
                    "before": before.get("value"),
                    "after": after.get("value"),
                    "tolerance": spec.tolerance,
                }
            )
    return tuple(mismatches)


async def capture_m110_reset_preimage(
    query_parameter: Callable[..., Any],
    *,
    expected_identity_values: Mapping[str, Any],
    timeout: float = M110_RESET_QUERY_TIMEOUT_SECONDS,
    sweep_deadline_seconds: float = M110_RESET_SWEEP_DEADLINE_SECONDS,
    total_deadline_seconds: float | None = None,
) -> dict[str, Any]:
    """Capture the exact reset preimage and the finite preserve manifest."""

    if not callable(query_parameter):
        raise TypeError("m110_query_parameter_callable_required")
    if total_deadline_seconds is not None and not (
        isinstance(total_deadline_seconds, (int, float))
        and not isinstance(total_deadline_seconds, bool)
        and math.isfinite(float(total_deadline_seconds))
        and float(total_deadline_seconds) > 0.0
    ):
        raise ValueError("m110_total_capture_deadline_invalid")
    capture_started = time.monotonic()

    def next_sweep_deadline() -> float:
        if total_deadline_seconds is None:
            return float(sweep_deadline_seconds)
        remaining = float(total_deadline_seconds) - (
            time.monotonic() - capture_started
        )
        if remaining <= 0.0:
            raise TimeoutError("m110_total_capture_deadline_exceeded")
        return min(float(sweep_deadline_seconds), remaining)

    reset_rows, reset_evidence = await _capture_rows(
        query_parameter,
        M110_RESET_PREIMAGE_SPECS,
        timeout=timeout,
        sweep_deadline_seconds=next_sweep_deadline(),
    )
    manifest_rows, manifest_evidence = await _capture_rows(
        query_parameter,
        M110_PRESERVATION_SPECS,
        timeout=timeout,
        sweep_deadline_seconds=next_sweep_deadline(),
    )
    if (
        total_deadline_seconds is not None
        and time.monotonic() - capture_started
        > float(total_deadline_seconds)
    ):
        raise TimeoutError("m110_total_capture_deadline_exceeded")
    identity_addresses = {
        "/$syscfg/$serial",
        "/$syscfg/$firmware",
        "/io/out/USB/47/grp",
        "/io/out/USB/47/in",
        "/io/out/USB/48/grp",
        "/io/out/USB/48/in",
    }
    if set(expected_identity_values) != identity_addresses:
        raise ValueError("m110_expected_identity_fields_mismatch")
    manifest_by_address = {row["address"]: row["value"] for row in manifest_rows}
    spec_by_address = {row.address: row for row in M110_PRESERVATION_SPECS}
    for address in sorted(identity_addresses):
        expected = _normalize_manifest_value(
            expected_identity_values[address],
            spec_by_address[address].value_kind,
        )
        if manifest_by_address.get(address) != expected:
            raise ValueError(f"m110_preservation_identity_mismatch:{address}")

    reset_document = {
        "schema": M110_RESET_PREIMAGE_SCHEMA,
        "row_count": len(reset_rows),
        "rows": reset_rows,
    }
    manifest_document = preservation_manifest_document(manifest_rows)
    reset_bytes = _canonical_json_bytes(reset_document)
    manifest_bytes = _canonical_json_bytes(manifest_document)
    return {
        "schema": M110_RESET_PREIMAGE_SCHEMA,
        "reset_row_count": len(reset_rows),
        "reset_rows": reset_rows,
        "reset_preimage_sha256": _sha256(reset_bytes),
        "preservation_manifest": manifest_document,
        "preservation_manifest_sha256": _sha256(manifest_bytes),
        "reset_query_evidence": reset_evidence,
        "preservation_query_evidence": manifest_evidence,
        "captured_at_epoch": time.time(),
        "live_authority_granted": False,
    }


def fixed_m110_reset_targets() -> dict[str, int]:
    return dict(M110_RESET_TARGETS)


async def capture_m110_preservation_manifest(
    query_parameter: Callable[..., Any],
    *,
    expected_identity_values: Mapping[str, Any],
    timeout: float = M110_RESET_QUERY_TIMEOUT_SECONDS,
    sweep_deadline_seconds: float = M110_RESET_SWEEP_DEADLINE_SECONDS,
) -> dict[str, Any]:
    """Capture only the finite preservation surface for a post-reset pass."""

    rows, evidence = await _capture_rows(
        query_parameter,
        M110_PRESERVATION_SPECS,
        timeout=timeout,
        sweep_deadline_seconds=sweep_deadline_seconds,
    )
    document = preservation_manifest_document(rows)
    identity_addresses = {
        "/$syscfg/$serial",
        "/$syscfg/$firmware",
        "/io/out/USB/47/grp",
        "/io/out/USB/47/in",
        "/io/out/USB/48/grp",
        "/io/out/USB/48/in",
    }
    if set(expected_identity_values) != identity_addresses:
        raise ValueError("m110_expected_identity_fields_mismatch")
    by_address = {row["address"]: row["value"] for row in rows}
    specs = {row.address: row for row in M110_PRESERVATION_SPECS}
    for address in sorted(identity_addresses):
        expected = _normalize_manifest_value(
            expected_identity_values[address],
            specs[address].value_kind,
        )
        if by_address.get(address) != expected:
            raise ValueError(f"m110_preservation_identity_mismatch:{address}")
    return {
        "preservation_manifest": document,
        "preservation_manifest_sha256": _sha256(_canonical_json_bytes(document)),
        "preservation_query_evidence": evidence,
        "captured_at_epoch": time.time(),
        "live_authority_granted": False,
    }


def m110_reset_review_preimage_document(
    capture: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and freeze the physical rows used by operator review."""

    reset_rows = capture.get("reset_rows")
    manifest = capture.get("preservation_manifest")
    if not isinstance(reset_rows, list) or len(reset_rows) != 64:
        raise ValueError("m110_review_reset_row_count_invalid")
    normalized_reset_rows = []
    for spec, row in zip(M110_RESET_PREIMAGE_SPECS, reset_rows):
        if not isinstance(row, Mapping) or row.get("address") != spec.address:
            raise ValueError("m110_review_reset_address_order_invalid")
        normalized_reset_rows.append(
            {
                "address": spec.address,
                "value": _normalize_binary(row.get("value")),
                "value_kind": "binary",
                "tolerance": "0",
            }
        )
    if not isinstance(manifest, Mapping):
        raise ValueError("m110_review_preservation_manifest_invalid")
    manifest_rows = manifest.get("rows")
    if not isinstance(manifest_rows, list) or len(manifest_rows) != 184:
        raise ValueError("m110_review_preservation_row_count_invalid")
    normalized_manifest_rows = []
    for spec, row in zip(M110_PRESERVATION_SPECS, manifest_rows):
        if not isinstance(row, Mapping) or row.get("address") != spec.address:
            raise ValueError("m110_review_preservation_address_order_invalid")
        normalized_manifest_rows.append(
            {
                "address": spec.address,
                "value": _normalize_manifest_value(
                    row.get("value"),
                    spec.value_kind,
                ),
                "value_kind": spec.value_kind,
                "tolerance": spec.tolerance,
            }
        )
    reset_document = {
        "schema": M110_RESET_PREIMAGE_SCHEMA,
        "row_count": 64,
        "rows": normalized_reset_rows,
    }
    manifest_document = preservation_manifest_document(normalized_manifest_rows)
    document = {
        "schema": M110_RESET_REVIEW_PREIMAGE_SCHEMA,
        "reset_preimage": reset_document,
        "reset_preimage_sha256": _sha256(_canonical_json_bytes(reset_document)),
        "preservation_manifest": manifest_document,
        "preservation_manifest_sha256": _sha256(
            _canonical_json_bytes(manifest_document)
        ),
    }
    document["review_preimage_sha256"] = _sha256(_canonical_json_bytes(document))
    return document


class M110DeliveredMutationObserver:
    """Record delivered state transitions, ignoring equal query replies."""

    def __init__(
        self,
        wing_client: Any,
        *,
        allowed_addresses: tuple[str, ...] = M110_RESET_ADDRESSES,
        initial_values: Mapping[str, Any] | None = None,
        max_events: int = 4096,
    ) -> None:
        self._wing_client = wing_client
        self._allowed = frozenset(allowed_addresses)
        self._max_events = int(max_events)
        self._lock = threading.RLock()
        self._active = False
        self._events: list[dict[str, Any]] = []
        self._values = dict(initial_values or {})
        self._required_reset_values: dict[str, int] | None = None
        self._observation_count = 0
        self._overflow = False
        self._unsubscribe_failed = False

    @staticmethod
    def _delivered_value(args: tuple[Any, ...]) -> Any:
        if len(args) >= 3:
            return args[2]
        if len(args) == 1:
            return args[0]
        return list(args)

    @staticmethod
    def _values_equal(left: Any, right: Any) -> bool:
        if isinstance(left, bool) or isinstance(right, bool):
            return left is right
        try:
            left_number = Decimal(str(left))
            right_number = Decimal(str(right))
            if left_number.is_finite() and right_number.is_finite():
                return left_number == right_number
        except (InvalidOperation, TypeError, ValueError):
            pass
        return left == right

    def _callback(self, address: str, *args: Any) -> None:
        with self._lock:
            if not self._active:
                return
            self._observation_count += 1
            address = str(address)
            value = self._delivered_value(tuple(args))
            if address not in self._values:
                self._values[address] = value
                if address not in self._allowed:
                    if len(self._events) >= self._max_events:
                        self._overflow = True
                        return
                    self._events.append(
                        {
                            "address": address,
                            "previous": None,
                            "value": _audit_safe_value(value),
                            "argument_count": len(args),
                            "allowed_reset_address": False,
                            "unexpected": True,
                            "baseline_unknown": True,
                            "observed_at_monotonic": time.monotonic(),
                        }
                    )
                return
            previous = self._values[address]
            if self._values_equal(previous, value):
                return
            self._values[address] = value
            if len(self._events) >= self._max_events:
                self._overflow = True
                return
            required = self._required_reset_values
            unexpected = address not in self._allowed
            if address in self._allowed and required is not None:
                try:
                    unexpected = (
                        _normalize_binary(value) != required[address]
                    )
                except (KeyError, ValueError):
                    unexpected = True
            self._events.append(
                {
                    "address": address,
                    "previous": _audit_safe_value(previous),
                    "value": _audit_safe_value(value),
                    "argument_count": len(args),
                    "allowed_reset_address": address in self._allowed,
                    "unexpected": unexpected,
                    "observed_at_monotonic": time.monotonic(),
                }
            )

    def require_reset_values(self, values: Mapping[str, Any]) -> None:
        if set(values) != set(M110_RESET_ADDRESSES):
            raise ValueError("m110_observer_required_reset_surface_invalid")
        normalized = {
            address: _normalize_binary(values[address])
            for address in M110_RESET_ADDRESSES
        }
        with self._lock:
            if not self._active:
                raise RuntimeError("m110_observer_not_active")
            self._required_reset_values = normalized

    def start(self) -> None:
        subscribe = getattr(self._wing_client, "subscribe", None)
        if not callable(subscribe):
            raise RuntimeError("m110_wildcard_observer_unavailable")
        with self._lock:
            if self._active:
                raise RuntimeError("m110_wildcard_observer_already_active")
            self._active = True
        try:
            subscribe("*", self._callback)
        except BaseException:
            with self._lock:
                self._active = False
            raise

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            events = tuple(dict(row) for row in self._events)
            return {
                "active": self._active,
                "observation_count": self._observation_count,
                "event_count": len(events),
                "events": events,
                "overflow": self._overflow,
                "unsubscribe_failed": self._unsubscribe_failed,
                "unexpected_events": tuple(
                    row for row in events if row.get("unexpected") is True
                ),
            }

    def close(self) -> dict[str, Any]:
        unsubscribe = getattr(self._wing_client, "unsubscribe", None)
        with self._lock:
            was_active = self._active
        try:
            removed = bool(
                was_active
                and callable(unsubscribe)
                and unsubscribe("*", self._callback) is True
            )
        except BaseException:
            removed = False
        with self._lock:
            # Stay active while unsubscribe executes so transitions delivered
            # from another thread inside that boundary remain observable.
            self._active = False
            if was_active and not removed:
                self._unsubscribe_failed = True
        return self.snapshot()


class M110MaintenanceMutationGate:
    """Authorize only executor-owned reset or exact-preimage rollback sends."""

    def __init__(
        self,
        wing_client: Any,
        preimage: Mapping[str, int],
        *,
        authority_current: Callable[[], bool],
        max_events: int = 4096,
    ) -> None:
        if set(preimage) != set(M110_RESET_ADDRESSES):
            raise ValueError("m110_mutation_gate_preimage_surface_invalid")
        if not isinstance(max_events, int) or isinstance(max_events, bool) or max_events < 1:
            raise ValueError("m110_mutation_gate_max_events_invalid")
        if not callable(authority_current):
            raise TypeError("m110_mutation_gate_authority_callable_required")
        self._wing_client = wing_client
        self._authority_current = authority_current
        self._preimage = {
            address: _normalize_binary(preimage[address])
            for address in M110_RESET_ADDRESSES
        }
        self._lock = threading.RLock()
        self._local = threading.local()
        self._token: object | None = None
        self._active = False
        self._sealed = False
        self._max_events = max_events
        self._total_event_count = 0
        self._overflow = False
        self._remove_failed = False
        self._boundary_state_unknown = False
        self._recovery_boundary_installed = False
        self._events: list[dict[str, Any]] = []

    def _record_event_locked(self, event: Mapping[str, Any]) -> None:
        self._total_event_count += 1
        if len(self._events) < self._max_events:
            self._events.append(dict(event))
        else:
            self._overflow = True

    def _boundary(self, address: str, values: tuple[Any, ...]) -> bool:
        permit = getattr(self._local, "permit", None)
        with self._lock:
            phase = permit.get("phase") if isinstance(permit, Mapping) else None
            authority_current = True
            if phase == "forward":
                try:
                    authority_current = self._authority_current() is True
                except BaseException:
                    authority_current = False
            allowed = bool(
                self._active
                and not self._sealed
                and isinstance(permit, Mapping)
                and permit.get("address") == address
                and permit.get("values") == values
                and address in M110_RESET_TARGETS
                and authority_current
            )
            rejection_reason = None
            if not allowed:
                rejection_reason = (
                    "forward_authority_not_current"
                    if phase == "forward" and not authority_current
                    else "mutation_not_owned_or_boundary_closed"
                )
            self._local.boundary_rejection_reason = rejection_reason
            self._record_event_locked(
                {
                    "event": "mutation_boundary",
                    "address": str(address),
                    "values": _audit_safe_value(values),
                    "phase": phase,
                    "authority_current": authority_current,
                    "allowed": allowed,
                    "rejection_reason": rejection_reason,
                    "owned_permit": isinstance(permit, Mapping),
                    "observed_at_monotonic": time.monotonic(),
                }
            )
            if allowed:
                self._local.permit = None
            return allowed

    def start(self) -> None:
        install = getattr(self._wing_client, "install_mutation_boundary", None)
        if not callable(install):
            raise RuntimeError("m110_mutation_boundary_unavailable")
        with self._lock:
            if self._active:
                raise RuntimeError("m110_mutation_boundary_already_active")
            self._active = True
            self._sealed = False
        try:
            token = install(self._boundary)
        except BaseException:
            with self._lock:
                self._active = False
            raise
        with self._lock:
            self._token = token

    def _blocking_send(self, phase: str, address: str, value: int) -> bool:
        if phase == "forward":
            expected = M110_RESET_TARGETS.get(address)
        elif phase == "rollback":
            expected = self._preimage.get(address)
        else:
            raise ValueError("m110_mutation_phase_invalid")
        normalized = _normalize_binary(value)
        if expected is None or normalized != expected:
            raise ValueError("m110_mutation_target_outside_contract")
        with self._lock:
            if not self._active or self._sealed:
                raise RuntimeError("m110_mutation_boundary_not_active")
        permit = {
            "phase": phase,
            "address": address,
            "values": (normalized,),
        }
        self._local.permit = permit
        self._local.boundary_rejection_reason = None
        try:
            outcome = self._wing_client.send(address, normalized)
        finally:
            self._local.permit = None
        boundary_rejection_reason = getattr(
            self._local,
            "boundary_rejection_reason",
            None,
        )
        self._local.boundary_rejection_reason = None
        with self._lock:
            self._record_event_locked(
                {
                    "event": "transport_result",
                    "address": address,
                    "values": [normalized],
                    "phase": phase,
                    "transport_ok": outcome is True,
                    "boundary_rejection_reason": boundary_rejection_reason,
                    "observed_at_monotonic": time.monotonic(),
                }
            )
        return outcome is True

    def last_transport_rejection_reason(
        self,
        phase: str,
        address: str,
    ) -> str | None:
        with self._lock:
            for event in reversed(self._events):
                if (
                    event.get("event") == "transport_result"
                    and event.get("phase") == phase
                    and event.get("address") == address
                ):
                    reason = event.get("boundary_rejection_reason")
                    return str(reason) if reason else None
        return None

    async def send(self, phase: str, address: str, value: int) -> bool:
        return bool(
            await _safety_thread_call(
                self._blocking_send,
                phase,
                address,
                value,
            )
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            events = tuple(dict(row) for row in self._events)
            denied = tuple(
                row
                for row in events
                if row.get("event") == "mutation_boundary"
                and row.get("allowed") is not True
            )
            return {
                "active": self._active,
                "sealed": self._sealed,
                "event_count": self._total_event_count,
                "retained_event_count": len(events),
                "max_events": self._max_events,
                "overflow": self._overflow,
                "remove_failed": self._remove_failed,
                "boundary_state_unknown": self._boundary_state_unknown,
                "recovery_boundary_installed": (
                    self._recovery_boundary_installed
                ),
                "events": events,
                "denied_attempts": denied,
                "no_outside_mutation_attempt_proven": (
                    not denied
                    and not self._overflow
                    and not self._boundary_state_unknown
                ),
            }

    def seal(self) -> dict[str, Any]:
        """Block every mutation while durable terminal state is written."""

        with self._lock:
            if not self._active:
                raise RuntimeError("m110_mutation_boundary_not_active")
            self._sealed = True
        return self.snapshot()

    def resume_rollback(self) -> None:
        """Re-open only executor-owned permits after a terminal/cleanup fault."""

        with self._lock:
            if not self._active:
                raise RuntimeError("m110_mutation_boundary_not_active")
            self._sealed = False

    def close(self) -> dict[str, Any]:
        remove = getattr(self._wing_client, "remove_mutation_boundary", None)
        matches = getattr(self._wing_client, "mutation_boundary_matches", None)
        install = getattr(self._wing_client, "install_mutation_boundary", None)
        with self._lock:
            token = self._token
        remove_exception_type = None
        try:
            removed = bool(
                token is not None
                and callable(remove)
                and remove(token) is True
            )
        except BaseException as exc:
            removed = False
            remove_exception_type = type(exc).__name__
        still_owned: bool | None = None
        if not removed and token is not None and callable(matches):
            try:
                still_owned = matches(token) is True
            except BaseException:
                still_owned = None
        recovery_token: object | None = None
        recovery_exception_type = None
        if not removed and still_owned is not True and callable(install):
            try:
                recovery_token = install(self._boundary)
            except BaseException as exc:
                recovery_exception_type = type(exc).__name__
        with self._lock:
            if removed:
                self._active = False
                self._sealed = True
                self._token = None
                self._remove_failed = False
                self._boundary_state_unknown = False
                self._recovery_boundary_installed = False
            elif still_owned is True:
                self._active = True
                self._sealed = True
                self._remove_failed = True
                self._boundary_state_unknown = False
                self._recovery_boundary_installed = False
                self._record_event_locked(
                    {
                        "event": "mutation_boundary_remove_failed",
                        "observed_at_monotonic": time.monotonic(),
                        "remove_exception_type": remove_exception_type,
                        "original_boundary_still_owned": True,
                    }
                )
            elif recovery_token is not None:
                self._active = True
                self._sealed = True
                self._token = recovery_token
                self._remove_failed = True
                self._boundary_state_unknown = False
                self._recovery_boundary_installed = True
                self._record_event_locked(
                    {
                        "event": "mutation_boundary_recovery_installed",
                        "observed_at_monotonic": time.monotonic(),
                        "remove_exception_type": remove_exception_type,
                        "original_boundary_still_owned": still_owned,
                    }
                )
            else:
                self._active = False
                self._sealed = True
                self._token = None
                self._remove_failed = True
                self._boundary_state_unknown = True
                self._recovery_boundary_installed = False
                self._record_event_locked(
                    {
                        "event": "mutation_boundary_remove_failed",
                        "observed_at_monotonic": time.monotonic(),
                        "remove_exception_type": remove_exception_type,
                        "original_boundary_still_owned": still_owned,
                        "recovery_exception_type": recovery_exception_type,
                    }
                )
        return self.snapshot()


@dataclass(frozen=True, slots=True)
class M110ResetHooks:
    authority_current: Callable[[], bool]
    persist_prepared: Callable[[Mapping[str, Any]], Any]
    mark_send_may_have_occurred: Callable[[], Any]
    persist_terminal: Callable[[str, Mapping[str, Any]], Any]
    audio_recheck: Callable[[], Any]
    emergency_stop: Callable[[str], Any]


@dataclass(slots=True)
class M110MaintenanceResetResult:
    transaction_id: str
    status: str = "running"
    reason: str = "maintenance_reset_started"
    applied_operation_ids: list[str] = field(default_factory=list)
    rollback_operation_ids: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    wal_terminal: str = "not_prepared"
    physical_verified: bool = False
    audio_recheck_passed: bool = False
    physical_state_restored: bool = False
    write_state_unknown: bool = False
    rollback_state_unknown: bool = False
    manual_review_required: bool = False
    emergency_stop_activated: bool = False
    emergency_stop_activation_failed: bool = False
    live_authority_granted: bool = False
    before_manifest_sha256: str = ""
    after_manifest_sha256: str = ""
    verification_cut_epoch: float | None = None
    verification_cut_reset_surface_sha256: str = ""
    verification_cut_preservation_manifest_sha256: str = ""
    physical_truth_scope: str = "not_verified_at_terminal_cut"
    current_state_guaranteed_at_response: bool = False
    attended_no_touch_required_during_execution: bool = True
    fresh_readback_required_before_next_stage: bool = True
    later_console_event_invalidates_readiness_freshness: bool = True
    preservation_mismatches: tuple[dict[str, Any], ...] = ()
    observer_audits: list[dict[str, Any]] = field(default_factory=list)
    mutation_audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _drain_task(task: asyncio.Task[Any]) -> Any:
    """Shield recovery from repeated caller cancellation until it is terminal."""

    cancellation_seen = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancellation_seen = True
            continue
    return task.result(), cancellation_seen


async def _call_durable_hook(
    function: Callable[..., Any],
    *args: Any,
) -> tuple[Any, bool]:
    """Drain an async durable side effect and report caller cancellation."""

    value = function(*args)
    if not inspect.isawaitable(value):
        return value, False
    task = asyncio.ensure_future(value)
    cancellation_seen = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancellation_seen = True
            continue
        except BaseException:
            break
    try:
        return task.result(), cancellation_seen
    except BaseException as exc:
        if cancellation_seen:
            raise M110DurableHookFailure(
                exc,
                cancellation_requested=True,
            ) from exc
        raise


class M110MaintenanceDspResetExecutor:
    """Dedicated finite executor for M110 reset proof and permanent commit."""

    def __init__(
        self,
        wing_client: Any,
        hooks: M110ResetHooks,
        *,
        query_timeout: float = M110_RESET_QUERY_TIMEOUT_SECONDS,
        sweep_deadline_seconds: float = M110_RESET_SWEEP_DEADLINE_SECONDS,
    ) -> None:
        self.wing_client = wing_client
        self.hooks = hooks
        self.query_timeout = float(query_timeout)
        self.sweep_deadline_seconds = float(sweep_deadline_seconds)
        self._lock = asyncio.Lock()

    @staticmethod
    def _audio_passed(value: Any) -> bool:
        status = getattr(value, "status", None)
        if isinstance(status, str):
            return status == "passed"
        if isinstance(value, Mapping):
            return value.get("status") == "passed"
        return False

    def _authority_current(self) -> bool:
        try:
            return self.hooks.authority_current() is True
        except BaseException:
            return False

    @staticmethod
    def _validate_fixed_bundle_binding(binding: Any) -> None:
        if binding.bundle_kind != M110_RESET_BUNDLE_KIND:
            raise ValueError("m110_reset_grant_bundle_kind_invalid")
        try:
            decoded = json.loads(binding.canonical_bundle_bytes.decode("utf-8"))
            bundle = validate_canonical_operation_bundle(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("m110_reset_grant_bundle_invalid") from exc
        if canonical_operation_bundle_bytes(bundle) != binding.canonical_bundle_bytes:
            raise ValueError("m110_reset_grant_bundle_not_canonical")
        if bundle.get("bundle_kind") != M110_RESET_BUNDLE_KIND:
            raise ValueError("m110_reset_grant_bundle_kind_invalid")
        scope, targets = canonical_bundle_scope(bundle)
        expected_targets = tuple(
            (
                address,
                json.dumps(
                    "0",
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            )
            for address in M110_RESET_ADDRESSES
        )
        if (
            scope != M110_RESET_ADDRESSES
            or targets != expected_targets
            or binding.scope_addresses != scope
            or binding.target_fingerprints != targets
        ):
            raise ValueError("m110_reset_grant_scope_invalid")
        safety_binding = bundle["safety_binding"]
        if (
            safety_binding.get("adapter_revision")
            != M110_RESET_ADAPTER_REVISION
            or safety_binding.get("protocol_revision")
            != M110_RESET_PROTOCOL_REVISION
        ):
            raise ValueError("m110_reset_grant_adapter_protocol_invalid")
        binding_fields = {
            field_name: getattr(binding, field_name)
            for field_name in (
                "runtime_instance_nonce",
                "runtime_manifest_sha256",
                "source_head",
                "source_tree",
                "console_serial",
                "firmware_version",
                "route_proof_id",
                "route_artifact_sha256",
                "capture_hardware_uid",
                "connection_generation",
                "transport_generation",
                "emergency_generation",
            )
        }
        if safety_binding != {
            **binding_fields,
            "adapter_revision": safety_binding.get("adapter_revision"),
            "protocol_revision": safety_binding.get("protocol_revision"),
        }:
            raise ValueError("m110_reset_grant_safety_binding_mismatch")
        for expected, actual in zip(
            M110_RESET_OPERATION_DEFINITIONS,
            bundle["operations"],
        ):
            if any(
                actual.get(field_name) != expected[field_name]
                for field_name in (
                    "operation_id",
                    "family_id",
                    "write_kind",
                    "channel",
                    "parameter",
                )
            ):
                raise ValueError("m110_reset_grant_operation_identity_invalid")
            if actual.get("writes") != [
                {
                    "address": expected["address"],
                    "target_value": "0",
                    "tolerance": "0",
                }
            ] or actual.get("audible_state_reads") != []:
                raise ValueError("m110_reset_grant_operation_surface_invalid")

    @staticmethod
    def _validate_restored_receipt(binding: Any, receipt: Any) -> None:
        if not (
            type(receipt) is M110RestoredProofReceipt
            and receipt.is_authentic()
            and receipt.bundle_kind == M110_RESET_BUNDLE_KIND
            and receipt.canonical_bundle_bytes == binding.canonical_bundle_bytes
            and receipt.operation_digest == binding.operation_digest
            and receipt.review_preimage_sha256
            == binding.review_preimage_sha256
            and receipt.receipt_id == binding.proof_receipt_id
            and binding._proof_receipt_ref is receipt
            and receipt._owner_ref is binding._owner_ref
            and receipt._route_ref is binding._route_ref
            and receipt._control_ref is binding._control_ref
            and receipt._transport_ref is binding._transport_ref
            and receipt._capture_ref is binding._capture_ref
            and receipt.owner_connection_id == binding.owner_connection_id
            and receipt.runtime_instance_nonce == binding.runtime_instance_nonce
            and receipt.source_head == binding.source_head
            and receipt.source_tree == binding.source_tree
            and receipt.route_proof_id == binding.route_proof_id
            and receipt.route_artifact_sha256 == binding.route_artifact_sha256
        ):
            raise ValueError("m110_reset_restored_proof_receipt_invalid")
        age = time.monotonic() - float(receipt.created_at_monotonic)
        if not (
            math.isfinite(age)
            and 0.0 <= age < M110_RESTORED_PROOF_RECEIPT_LATEST_START_SECONDS
        ):
            raise ValueError("m110_reset_restored_proof_receipt_expired")

    @staticmethod
    def _claim_execution_grant(
        capability_grant: Any,
    ) -> tuple[bool, object]:
        """Atomically burn an authentic typed grant before any await point."""

        if type(capability_grant) is RestoreOnlyProofGrant:
            proof_run = True
        elif type(capability_grant) is MaintenanceDspResetCommitOnceGrant:
            proof_run = False
        else:
            raise TypeError("m110_reset_typed_capability_grant_required")
        if not capability_grant.is_authentic():
            raise ValueError("m110_reset_capability_grant_inauthentic")
        owner_token = capability_grant._begin_owned_execution()
        if owner_token is None:
            raise ValueError("m110_reset_capability_grant_not_executable")
        return proof_run, owner_token

    def _validate_execution_grant(
        self,
        capability_grant: Any,
        restored_proof_receipt: Any,
        approved_preimage: Mapping[str, Any],
        *,
        proof_run: bool,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if proof_run and restored_proof_receipt is not None:
            raise ValueError("m110_reset_proof_receipt_not_allowed_for_proof")
        binding = capability_grant.binding
        if binding._transport_ref is not self.wing_client:
            raise ValueError("m110_reset_grant_transport_identity_mismatch")
        self._validate_fixed_bundle_binding(binding)
        reviewed_preimage = m110_reset_review_preimage_document(
            approved_preimage
        )
        if (
            reviewed_preimage["review_preimage_sha256"]
            != binding.review_preimage_sha256
        ):
            raise ValueError("m110_reset_reviewed_preimage_binding_mismatch")
        if not proof_run:
            self._validate_restored_receipt(binding, restored_proof_receipt)
        safe_wal_binding = {
            "proposal_id": binding.proposal_id,
            "approval_id": binding.approval_id,
            "idempotency_key": binding.idempotency_key,
            "review_content_digest": binding.review_content_digest,
            "operation_digest": binding.operation_digest,
            "bundle_kind": binding.bundle_kind,
        }
        if not proof_run:
            safe_wal_binding["proof_receipt_id"] = binding.proof_receipt_id
        expected_identity_values = {
            "/$syscfg/$serial": binding.console_serial,
            "/$syscfg/$firmware": binding.firmware_version,
            "/io/out/USB/47/grp": "MAIN",
            "/io/out/USB/47/in": 0,
            "/io/out/USB/48/grp": "MAIN",
            "/io/out/USB/48/in": 1,
        }
        return safe_wal_binding, expected_identity_values, reviewed_preimage

    async def _query_binary(self, address: str) -> tuple[int, dict[str, Any]]:
        result = await _call_query(
            self.wing_client.query_parameter,
            address,
            self.query_timeout,
        )
        value, evidence = _validate_query_result(
            result,
            address=address,
            value_kind="binary",
        )
        return int(value), evidence

    @staticmethod
    def _observer_failed(audit: Mapping[str, Any]) -> bool:
        return bool(
            audit.get("overflow") is True
            or audit.get("unsubscribe_failed") is True
            or audit.get("unexpected_events")
        )

    @staticmethod
    def _observer_has_unrestored_surface_risk(
        audit: Mapping[str, Any],
    ) -> bool:
        if (
            audit.get("overflow") is True
            or audit.get("unsubscribe_failed") is True
        ):
            return True
        return any(
            isinstance(event, Mapping)
            and event.get("allowed_reset_address") is not True
            for event in (audit.get("unexpected_events") or ())
        )

    @staticmethod
    def _gate_failed(audit: Mapping[str, Any]) -> bool:
        return bool(
            audit.get("overflow") is True
            or audit.get("remove_failed") is True
            or audit.get("boundary_state_unknown") is True
            or audit.get("denied_attempts")
        )

    async def _verify_reset_surface(
        self,
        result: M110MaintenanceResetResult,
        *,
        expected: Mapping[str, int],
        phase: str,
    ) -> tuple[bool, bool]:
        mismatches = []
        cancellation_requested = False
        for address in M110_RESET_ADDRESSES:
            try:
                actual, evidence = await self._query_binary(address)
            except BaseException as exc:
                cancellation_requested = bool(
                    cancellation_requested
                    or isinstance(
                        exc,
                        (asyncio.CancelledError, M110SafetyCallCancelled),
                    )
                )
                mismatches.append(
                    {
                        "address": address,
                        "reason": f"{phase}_read_failed",
                        "exception_type": type(exc).__name__,
                    }
                )
                continue
            if actual != expected[address]:
                mismatches.append(
                    {
                        "address": address,
                        "reason": f"{phase}_mismatch",
                        "expected": expected[address],
                        "actual": actual,
                        "query": evidence,
                    }
                )
        result.events.append({"phase": phase, "mismatches": mismatches})
        return not mismatches, cancellation_requested

    async def _rollback(
        self,
        result: M110MaintenanceResetResult,
        gate: M110MaintenanceMutationGate,
        preimage: Mapping[str, int],
        attempted_addresses: list[str],
    ) -> bool:
        failures: list[dict[str, Any]] = []
        restored_ids: list[str] = []
        operation_by_address = {
            row["address"]: row["operation_id"]
            for row in M110_RESET_OPERATION_DEFINITIONS
        }
        for address in reversed(attempted_addresses):
            try:
                current, compare = await self._query_binary(address)
            except BaseException as exc:
                failures.append(
                    {
                        "address": address,
                        "reason": "rollback_cas_read_failed",
                        "exception_type": type(exc).__name__,
                    }
                )
                continue
            old = preimage[address]
            target = M110_RESET_TARGETS[address]
            if current == old:
                restored_ids.append(operation_by_address[address])
                continue
            if current != target:
                failures.append(
                    {
                        "address": address,
                        "reason": "rollback_cas_target_changed",
                        "expected_target": target,
                        "actual": current,
                        "query": compare,
                    }
                )
                continue
            try:
                sent = await gate.send("rollback", address, old)
            except BaseException as exc:
                sent = False
                failures.append(
                    {
                        "address": address,
                        "reason": "rollback_send_exception",
                        "exception_type": type(exc).__name__,
                    }
                )
            if not sent:
                if not any(row.get("address") == address for row in failures):
                    failures.append(
                        {"address": address, "reason": "rollback_send_failed"}
                    )
                continue
            try:
                restored, verify = await self._query_binary(address)
            except BaseException as exc:
                failures.append(
                    {
                        "address": address,
                        "reason": "rollback_verify_read_failed",
                        "exception_type": type(exc).__name__,
                    }
                )
                continue
            if restored != old:
                failures.append(
                    {
                        "address": address,
                        "reason": "rollback_physical_mismatch",
                        "expected": old,
                        "actual": restored,
                        "query": verify,
                    }
                )
                continue
            restored_ids.append(operation_by_address[address])

        final_mismatches = []
        for address in M110_RESET_ADDRESSES:
            try:
                current, evidence = await self._query_binary(address)
            except BaseException as exc:
                final_mismatches.append(
                    {
                        "address": address,
                        "reason": "rollback_final_read_failed",
                        "exception_type": type(exc).__name__,
                    }
                )
                continue
            if current != preimage[address]:
                final_mismatches.append(
                    {
                        "address": address,
                        "reason": "rollback_final_mismatch",
                        "expected": preimage[address],
                        "actual": current,
                        "query": evidence,
                    }
                )
        result.rollback_operation_ids = restored_ids
        result.events.append(
            {
                "phase": "rollback",
                "failures": failures,
                "final_mismatches": final_mismatches,
            }
        )
        success = not failures and not final_mismatches
        result.physical_state_restored = success
        result.rollback_state_unknown = not success
        return success

    async def execute(
        self,
        *,
        transaction_id: str,
        approved_preimage: Mapping[str, Any],
        expected_identity_values: Mapping[str, Any],
        capability_grant: Any,
        restored_proof_receipt: M110RestoredProofReceipt | None = None,
    ) -> M110MaintenanceResetResult:
        proof_run, grant_owner_token = self._claim_execution_grant(
            capability_grant
        )

        def forward_authority_current() -> bool:
            return bool(
                capability_grant.execution_status() == "executing"
                and self._authority_current()
            )

        try:
            safe_wal_binding, bound_identity_values, approved_review = (
                self._validate_execution_grant(
                capability_grant,
                restored_proof_receipt,
                approved_preimage,
                proof_run=proof_run,
                )
            )
            if dict(expected_identity_values) != bound_identity_values:
                raise ValueError("m110_reset_expected_identity_binding_mismatch")
            async with self._lock:
                if not proof_run:
                    self._validate_restored_receipt(
                        capability_grant.binding,
                        restored_proof_receipt,
                    )
                result = await self._execute_locked(
                    transaction_id=transaction_id,
                    approved_review=approved_review,
                    expected_identity_values=bound_identity_values,
                    proof_run=proof_run,
                    safe_wal_binding=safe_wal_binding,
                    forward_authority_current=forward_authority_current,
                )
        except BaseException:
            capability_grant._mark_owned_terminal(
                "executor_exception",
                grant_owner_token,
            )
            raise
        if capability_grant._mark_owned_terminal(
            result.status,
            grant_owner_token,
        ) is not True:
            raise RuntimeError("m110_reset_capability_terminalization_failed")
        return result

    async def _execute_locked(
        self,
        *,
        transaction_id: str,
        approved_review: Mapping[str, Any],
        expected_identity_values: Mapping[str, Any],
        proof_run: bool,
        safe_wal_binding: Mapping[str, Any],
        forward_authority_current: Callable[[], bool],
    ) -> M110MaintenanceResetResult:
        transaction_id = str(transaction_id or "").strip()
        if not transaction_id:
            raise ValueError("m110_transaction_id_required")
        review = dict(approved_review)
        result = M110MaintenanceResetResult(transaction_id=transaction_id)
        preimage = {
            row["address"]: int(row["value"])
            for row in review["reset_preimage"]["rows"]
        }
        gate = M110MaintenanceMutationGate(
            self.wing_client,
            preimage,
            authority_current=forward_authority_current,
        )
        observer: M110DeliveredMutationObserver | None = None
        attempted_addresses: list[str] = []
        prepared = False
        boundary_marked = False
        forward_success = False
        failure_reason = ""
        cancellation_requested = False
        before_rows = review["preservation_manifest"]["rows"]
        after_rows: list[dict[str, Any]] | None = None
        full_surface_restoration_unknown = False

        try:
            gate.start()
            if not forward_authority_current():
                failure_reason = "m110_write_authority_not_current"
            else:
                current = await capture_m110_reset_preimage(
                    self.wing_client.query_parameter,
                    expected_identity_values=expected_identity_values,
                    timeout=self.query_timeout,
                    sweep_deadline_seconds=self.sweep_deadline_seconds,
                )
                current_review = m110_reset_review_preimage_document(current)
                if current_review["reset_preimage_sha256"] != review[
                    "reset_preimage_sha256"
                ]:
                    failure_reason = "m110_review_reset_preimage_changed"
                elif compare_preservation_manifest_rows(
                    before_rows,
                    current_review["preservation_manifest"]["rows"],
                ):
                    failure_reason = "m110_review_preservation_manifest_changed"
                else:
                    # Use the immediate physical recapture as the actual CAS and
                    # rollback preimage; it is equal to the reviewed binary rows.
                    preimage = {
                        row["address"]: int(row["value"])
                        for row in current_review["reset_preimage"]["rows"]
                    }
                    before_rows = current_review["preservation_manifest"]["rows"]
                    result.before_manifest_sha256 = current_review[
                        "preservation_manifest_sha256"
                    ]
                    observer_initial_values = {
                        **dict(getattr(self.wing_client, "state", {}) or {}),
                        **preimage,
                        **{
                            row["address"]: row["value"]
                            for row in before_rows
                        },
                    }
                    observer = M110DeliveredMutationObserver(
                        self.wing_client,
                        initial_values=observer_initial_values,
                    )
                    observer.start()
                    observer.require_reset_values(M110_RESET_TARGETS)
                    prepared_payload = {
                        **safe_wal_binding,
                        "scope": "supervised_transaction",
                        "maintenance_dsp_reset": True,
                        "proof_run": bool(proof_run),
                        "restore_after_verify": bool(proof_run),
                        "capability_kind": (
                            "restore_only_proof"
                            if proof_run
                            else "maintenance_dsp_reset_commit_once"
                        ),
                        "review_preimage_sha256": review[
                            "review_preimage_sha256"
                        ],
                        "reset_preimage_sha256": current_review[
                            "reset_preimage_sha256"
                        ],
                        "reset_preimage": current_review["reset_preimage"],
                        "before_manifest_sha256": result.before_manifest_sha256,
                        "before_manifest": current_review[
                            "preservation_manifest"
                        ],
                        "allowed_reset_surface_sha256": _sha256(
                            _canonical_json_bytes(list(M110_RESET_ADDRESSES))
                        ),
                        "expected_operation_ids": [
                            row["operation_id"]
                            for row in M110_RESET_OPERATION_DEFINITIONS
                        ],
                        "operations": [
                            {
                                "operation_id": row["operation_id"],
                                "family_id": row["family_id"],
                                "write_kind": row["write_kind"],
                                "channel": row["channel"],
                                "parameter": row["parameter"],
                                "writes": [
                                    {
                                        "address": row["address"],
                                        "target_value": 0,
                                        "tolerance": 0.0,
                                    }
                                ],
                                "preimage_values": {
                                    row["address"]: preimage[row["address"]]
                                },
                            }
                            for row in M110_RESET_OPERATION_DEFINITIONS
                        ],
                    }
                    prepared_path, durable_cancelled = await _call_durable_hook(
                        self.hooks.persist_prepared,
                        prepared_payload,
                    )
                    cancellation_requested = (
                        cancellation_requested or durable_cancelled
                    )
                    if not prepared_path:
                        failure_reason = "m110_reset_wal_prepare_failed"
                    else:
                        prepared = True
                        if durable_cancelled:
                            failure_reason = "m110_reset_cancelled"
                        else:
                            boundary_value, boundary_cancelled = (
                                await _call_durable_hook(
                                    self.hooks.mark_send_may_have_occurred
                                )
                            )
                            boundary_marked = bool(boundary_value)
                            cancellation_requested = (
                                cancellation_requested or boundary_cancelled
                            )
                        if cancellation_requested:
                            failure_reason = "m110_reset_cancelled"
                        elif not boundary_marked:
                            failure_reason = (
                                "m110_result_send_boundary_persist_failed"
                            )

            if not failure_reason:
                for row in M110_RESET_OPERATION_DEFINITIONS:
                    address = row["address"]
                    if observer and self._observer_failed(observer.snapshot()):
                        failure_reason = "m110_delivered_outside_event_detected"
                        break
                    if self._gate_failed(gate.snapshot()):
                        failure_reason = "m110_outside_mutation_attempt_detected"
                        break
                    if not forward_authority_current():
                        failure_reason = "m110_write_authority_expired_before_send"
                        break
                    current_value, cas_evidence = await self._query_binary(address)
                    if current_value != preimage[address]:
                        failure_reason = f"m110_forward_cas_changed:{address}"
                        result.events.append(
                            {
                                "phase": "forward_cas",
                                "address": address,
                                "expected": preimage[address],
                                "actual": current_value,
                                "query": cas_evidence,
                            }
                        )
                        break
                    if observer and self._observer_failed(observer.snapshot()):
                        failure_reason = "m110_delivered_outside_event_detected"
                        break
                    if self._gate_failed(gate.snapshot()):
                        failure_reason = "m110_outside_mutation_attempt_detected"
                        break
                    if not forward_authority_current():
                        failure_reason = "m110_write_authority_expired_before_send"
                        break
                    attempted_addresses.append(address)
                    try:
                        sent = await gate.send("forward", address, 0)
                    except M110SafetyCallCancelled as exc:
                        cancellation_requested = True
                        result.write_state_unknown = exc.error is not None
                        failure_reason = f"m110_forward_send_cancelled:{address}"
                        break
                    if not sent:
                        rejection_reason = gate.last_transport_rejection_reason(
                            "forward",
                            address,
                        )
                        if rejection_reason == "forward_authority_not_current":
                            attempted_addresses.pop()
                            failure_reason = (
                                "m110_write_authority_expired_before_send"
                            )
                        else:
                            failure_reason = f"m110_forward_send_failed:{address}"
                        break
                    actual, verify_evidence = await self._query_binary(address)
                    if actual != 0:
                        failure_reason = f"m110_forward_verify_mismatch:{address}"
                        result.events.append(
                            {
                                "phase": "forward_verify",
                                "address": address,
                                "expected": 0,
                                "actual": actual,
                                "query": verify_evidence,
                            }
                        )
                        break
                    result.applied_operation_ids.append(row["operation_id"])
                    observer_audit = observer.snapshot() if observer else {}
                    if self._observer_failed(observer_audit):
                        failure_reason = "m110_delivered_outside_event_detected"
                        break
                    if self._gate_failed(gate.snapshot()):
                        failure_reason = "m110_outside_mutation_attempt_detected"
                        break
                else:
                    if observer is None:
                        failure_reason = "m110_delivered_observer_missing"
                    else:
                        observer.require_reset_values(M110_RESET_TARGETS)
                    if not failure_reason:
                        forward_surface_ok, verify_cancelled = (
                            await self._verify_reset_surface(
                                result,
                                expected=M110_RESET_TARGETS,
                                phase="forward_final_verify",
                            )
                        )
                        cancellation_requested = bool(
                            cancellation_requested or verify_cancelled
                        )
                        if verify_cancelled:
                            failure_reason = "m110_reset_cancelled"
                        elif not forward_surface_ok:
                            failure_reason = (
                                "m110_forward_final_surface_mismatch"
                            )
                    if (
                        not failure_reason
                        and not forward_authority_current()
                    ):
                        failure_reason = "m110_write_authority_expired_after_writes"
                    if not failure_reason:
                        audio = await _maybe_await(self.hooks.audio_recheck())
                        result.audio_recheck_passed = self._audio_passed(audio)
                        result.events.append(
                            {
                                "phase": "audio_recheck",
                                "result": (
                                    asdict(audio)
                                    if hasattr(audio, "__dataclass_fields__")
                                    else dict(audio)
                                    if isinstance(audio, Mapping)
                                    else {"type": type(audio).__name__}
                                ),
                            }
                        )
                        if not result.audio_recheck_passed:
                            failure_reason = "m110_audio_recheck_failed"
                        elif observer and self._observer_failed(observer.snapshot()):
                            failure_reason = "m110_delivered_outside_event_detected"
                        elif self._gate_failed(gate.snapshot()):
                            failure_reason = (
                                "m110_outside_mutation_attempt_detected"
                            )
                        else:
                            forward_success = True
                            result.physical_verified = True
        except (asyncio.CancelledError, M110SafetyCallCancelled):
            cancellation_requested = True
            failure_reason = failure_reason or "m110_reset_cancelled"
        except BaseException as exc:
            if isinstance(exc, M110DurableHookFailure):
                cancellation_requested = bool(
                    cancellation_requested or exc.cancellation_requested
                )
            failure_reason = failure_reason or (
                f"m110_reset_exception:{type(exc).__name__}"
            )
            result.events.append(
                {
                    "phase": "executor_exception",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

        rollback_required = bool(attempted_addresses) and (
            proof_run or not forward_success
        )
        rollback_success = False

        async def perform_rollback() -> bool:
            nonlocal cancellation_requested, failure_reason
            if observer is not None and observer.snapshot().get("active") is True:
                try:
                    observer.require_reset_values(preimage)
                except BaseException as exc:
                    failure_reason = failure_reason or (
                        f"m110_rollback_observer_state_failed:{type(exc).__name__}"
                    )
            try:
                rollback_value, cancelled = await _drain_task(
                    asyncio.create_task(
                        self._rollback(result, gate, preimage, attempted_addresses)
                    )
                )
                cancellation_requested = cancellation_requested or cancelled
                if cancelled:
                    failure_reason = failure_reason or "m110_reset_cancelled"
                return rollback_value is True
            except BaseException as exc:
                result.events.append(
                    {
                        "phase": "rollback_executor_exception",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                return False

        if rollback_required:
            rollback_success = await perform_rollback()
        elif not attempted_addresses:
            rollback_success = True

        # The transition-aware observer stays active during this pass. Equal
        # GET replies are ignored; actual changes remain visible.
        after_manifest: dict[str, Any] | None = None
        if prepared:
            try:
                after = await capture_m110_preservation_manifest(
                    self.wing_client.query_parameter,
                    expected_identity_values=expected_identity_values,
                    timeout=self.query_timeout,
                    sweep_deadline_seconds=self.sweep_deadline_seconds,
                )
                after_manifest = after["preservation_manifest"]
                after_rows = after_manifest["rows"]
                result.after_manifest_sha256 = after[
                    "preservation_manifest_sha256"
                ]
                result.preservation_mismatches = (
                    compare_preservation_manifest_rows(before_rows, after_rows)
                )
            except BaseException as exc:
                if isinstance(
                    exc,
                    (asyncio.CancelledError, M110SafetyCallCancelled),
                ):
                    cancellation_requested = True
                result.preservation_mismatches = (
                    {
                        "reason": "m110_after_manifest_capture_failed",
                        "exception_type": type(exc).__name__,
                    },
                )
            if result.preservation_mismatches:
                failure_reason = failure_reason or "m110_preservation_manifest_changed"

        preterminal_surface_ok = True
        if attempted_addresses:
            terminal_expected = (
                preimage if rollback_required or proof_run else M110_RESET_TARGETS
            )
            preterminal_surface_ok, preterminal_cancelled = (
                await self._verify_reset_surface(
                result,
                expected=terminal_expected,
                phase="preterminal_final_verify",
                )
            )
            cancellation_requested = bool(
                cancellation_requested or preterminal_cancelled
            )
            if preterminal_cancelled:
                failure_reason = failure_reason or "m110_reset_cancelled"
            if not preterminal_surface_ok:
                failure_reason = failure_reason or (
                    "m110_preterminal_reset_surface_mismatch"
                )
                rollback_required = True
                rollback_success = await perform_rollback()

        if forward_success and not forward_authority_current():
            failure_reason = "m110_write_authority_expired_before_terminal"

        live_observer_audit = observer.snapshot() if observer is not None else {}
        gate_audit = gate.snapshot()
        if observer is not None and self._observer_failed(live_observer_audit):
            failure_reason = failure_reason or "m110_delivered_observer_audit_failed"
        if self._gate_failed(gate_audit):
            failure_reason = failure_reason or "m110_outside_mutation_attempt_detected"

        if (
            forward_success
            and not proof_run
            and not rollback_required
            and (
                failure_reason
                or result.preservation_mismatches
                or self._observer_failed(live_observer_audit)
                or self._gate_failed(gate_audit)
            )
        ):
            rollback_required = True
            rollback_success = await perform_rollback()

        try:
            sealed_gate = gate.seal()
        except BaseException as exc:
            sealed_gate = gate.snapshot()
            failure_reason = failure_reason or (
                f"m110_mutation_boundary_seal_failed:{type(exc).__name__}"
            )
        result.mutation_audit = sealed_gate

        sealed_observer_audit = (
            observer.snapshot() if observer is not None else {}
        )
        full_surface_restoration_unknown = bool(
            result.preservation_mismatches
            or self._observer_has_unrestored_surface_risk(
                sealed_observer_audit
            )
        )
        if full_surface_restoration_unknown:
            result.physical_state_restored = False
            result.rollback_state_unknown = True

        # Observer and mutation boundary remain installed through the staged
        # verification passes. The observer is detached only after the final
        # 64+184 sweep; the sealed mutation boundary remains until after the
        # success WAL attempt. Failures before that attempt may be compensated.
        clean_observer = not self._observer_failed(
            sealed_observer_audit
        )
        clean_preservation = not result.preservation_mismatches
        clean_gate = bool(
            not self._gate_failed(sealed_gate)
            and sealed_gate.get("sealed") is True
        )
        gate_remove_failed = False
        if cancellation_requested:
            failure_reason = failure_reason or "m110_reset_cancelled"
        if (
            proof_run
            and forward_success
            and rollback_success
            and clean_observer
            and clean_preservation
            and clean_gate
            and not cancellation_requested
            and not failure_reason
        ):
            result.status = "proof_passed"
            result.reason = "m110_reset_proof_applied_and_restored"
            terminal_phase = "proof_restored"
        elif (
            not proof_run
            and forward_success
            and not rollback_required
            and clean_observer
            and clean_preservation
            and clean_gate
            and not failure_reason
        ):
            result.status = "applied"
            result.reason = "m110_maintenance_dsp_reset_committed"
            terminal_phase = "committed"
        elif not attempted_addresses:
            result.status = "blocked"
            result.reason = failure_reason or "m110_reset_blocked_before_send"
            terminal_phase = "aborted_before_send" if prepared else "not_prepared"
        elif (
            cancellation_requested
            and failure_reason.startswith("m110_forward_send_cancelled")
        ):
            result.status = "manual_review_required"
            result.reason = failure_reason
            result.manual_review_required = True
            result.rollback_state_unknown = not rollback_success
            terminal_phase = "manual_review_required"
        elif rollback_success and clean_observer and clean_preservation and not gate_remove_failed:
            result.status = "rolled_back"
            result.reason = failure_reason or "m110_reset_rolled_back"
            terminal_phase = "rolled_back"
        else:
            result.status = "manual_review_required"
            result.reason = failure_reason or "m110_reset_state_unverified"
            result.manual_review_required = True
            result.rollback_state_unknown = not rollback_success
            terminal_phase = "manual_review_required"

        current_task = asyncio.current_task()
        initial_cancelling = (
            current_task.cancelling() if current_task is not None else 0
        )
        if initial_cancelling:
            cancellation_requested = True
        candidate_status = result.status
        candidate_reason = result.reason
        candidate_phase = terminal_phase
        finalizer_cancellation = False
        emergency_attempted = False

        def record_cancellation(value: bool) -> None:
            nonlocal cancellation_requested, finalizer_cancellation
            if value:
                cancellation_requested = True
                finalizer_cancellation = True

        def task_cancellation_pending() -> bool:
            return bool(
                current_task is not None
                and current_task.cancelling() > initial_cancelling
            )

        async def consume_pending_task_cancellation() -> bool:
            """Observe a sync-hook cancel request without losing truth."""

            if not task_cancellation_pending():
                return False
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                return True
            return True

        def mark_manual(
            reason: str,
            *,
            full_surface_unknown: bool = False,
        ) -> None:
            nonlocal full_surface_restoration_unknown
            result.status = "manual_review_required"
            result.reason = str(reason)
            result.manual_review_required = True
            result.physical_verified = False
            if full_surface_unknown:
                full_surface_restoration_unknown = True
                result.physical_state_restored = False
                result.rollback_state_unknown = True
            elif attempted_addresses and not result.physical_state_restored:
                result.rollback_state_unknown = True

        async def persist_phase(
            phase: str,
            payload: Mapping[str, Any],
        ) -> tuple[bool, bool, str]:
            cancelled_at_call = task_cancellation_pending()
            record_cancellation(cancelled_at_call)
            try:
                receipt, cancelled = await _call_durable_hook(
                    self.hooks.persist_terminal,
                    phase,
                    payload,
                )
                post_hook_cancelled = (
                    await consume_pending_task_cancellation()
                )
                cancelled = bool(
                    cancelled
                    or cancelled_at_call
                    or post_hook_cancelled
                )
                record_cancellation(cancelled)
                if isinstance(receipt, Mapping):
                    path = receipt.get("path")
                    payload_sha256 = str(
                        receipt.get("payload_sha256") or ""
                    )
                else:
                    path = receipt
                    payload_sha256 = ""
                return bool(path), bool(cancelled), payload_sha256
            except BaseException as exc:
                cancelled = bool(
                    cancelled_at_call
                    or task_cancellation_pending()
                    or isinstance(
                        exc,
                        (asyncio.CancelledError, M110SafetyCallCancelled),
                    )
                    or (
                        isinstance(exc, M110DurableHookFailure)
                        and exc.cancellation_requested
                    )
                )
                record_cancellation(cancelled)
                result.events.append(
                    {
                        "phase": "terminal_wal_exception",
                        "wal_phase": phase,
                        "exception_type": type(exc).__name__,
                        "cancellation_requested": cancelled,
                    }
                )
                return False, cancelled, ""

        async def verify_surface(
            *,
            expected: Mapping[str, int],
            phase: str,
        ) -> bool:
            try:
                verification, cancelled = await _drain_task(
                    asyncio.create_task(
                        self._verify_reset_surface(
                            result,
                            expected=expected,
                            phase=phase,
                        )
                    )
                )
                record_cancellation(cancelled)
                ok, inner_cancelled = verification
                record_cancellation(inner_cancelled)
                return bool(ok and not inner_cancelled and not cancelled)
            except BaseException as exc:
                record_cancellation(
                    isinstance(
                        exc,
                        (asyncio.CancelledError, M110SafetyCallCancelled),
                    )
                )
                result.events.append(
                    {
                        "phase": phase,
                        "mismatches": [
                            {
                                "reason": f"{phase}_executor_failed",
                                "exception_type": type(exc).__name__,
                            }
                        ],
                    }
                )
                return False

        async def refresh_manifest(phase: str) -> bool:
            nonlocal after_manifest, after_rows
            try:
                captured, cancelled = await _drain_task(
                    asyncio.create_task(
                        capture_m110_preservation_manifest(
                            self.wing_client.query_parameter,
                            expected_identity_values=expected_identity_values,
                            timeout=self.query_timeout,
                            sweep_deadline_seconds=self.sweep_deadline_seconds,
                        )
                    )
                )
                record_cancellation(cancelled)
                after_manifest = captured["preservation_manifest"]
                after_rows = after_manifest["rows"]
                result.after_manifest_sha256 = captured[
                    "preservation_manifest_sha256"
                ]
                result.preservation_mismatches = (
                    compare_preservation_manifest_rows(
                        before_rows,
                        after_rows,
                    )
                )
                result.events.append(
                    {
                        "phase": phase,
                        "preservation_mismatches": list(
                            result.preservation_mismatches
                        ),
                    }
                )
                return bool(
                    not cancelled and not result.preservation_mismatches
                )
            except BaseException as exc:
                record_cancellation(
                    isinstance(
                        exc,
                        (asyncio.CancelledError, M110SafetyCallCancelled),
                    )
                )
                after_manifest = None
                after_rows = None
                result.after_manifest_sha256 = ""
                result.preservation_mismatches = (
                    {
                        "reason": f"{phase}_capture_failed",
                        "exception_type": type(exc).__name__,
                    },
                )
                result.events.append(
                    {
                        "phase": phase,
                        "preservation_mismatches": list(
                            result.preservation_mismatches
                        ),
                    }
                )
                return False

        async def restore_preimage(phase: str) -> bool:
            nonlocal rollback_required, rollback_success, failure_reason
            if not attempted_addresses:
                return True
            rollback_required = True
            try:
                gate.resume_rollback()
                if (
                    observer is not None
                    and observer.snapshot().get("active") is True
                ):
                    observer.require_reset_values(preimage)
                rollback_success = await perform_rollback()
            except BaseException as exc:
                rollback_success = False
                result.events.append(
                    {
                        "phase": f"{phase}_rollback_exception",
                        "exception_type": type(exc).__name__,
                    }
                )
            finally:
                try:
                    gate.seal()
                except BaseException as exc:
                    rollback_success = False
                    result.events.append(
                        {
                            "phase": f"{phase}_reseal_exception",
                            "exception_type": type(exc).__name__,
                        }
                    )
            surface_ok = await verify_surface(
                expected=preimage,
                phase=f"{phase}_restored_surface_verify",
            )
            rollback_success = bool(rollback_success and surface_ok)
            result.physical_state_restored = rollback_success
            result.rollback_state_unknown = not rollback_success
            return rollback_success

        async def activate_emergency() -> None:
            nonlocal emergency_attempted
            if emergency_attempted:
                return
            emergency_attempted = True
            record_cancellation(task_cancellation_pending())
            try:
                emergency_value, emergency_cancelled = (
                    await _call_durable_hook(
                        self.hooks.emergency_stop,
                        f"m110_maintenance_reset_unverified:{transaction_id}",
                    )
                )
                post_emergency_cancelled = (
                    await consume_pending_task_cancellation()
                )
                emergency_cancelled = bool(
                    emergency_cancelled or post_emergency_cancelled
                )
                record_cancellation(emergency_cancelled)
                emergency_ok = bool(
                    emergency_value is True
                    or (
                        isinstance(emergency_value, Mapping)
                        and emergency_value.get("success") is True
                    )
                )
                result.emergency_stop_activated = emergency_ok
                result.emergency_stop_activation_failed = not emergency_ok
            except BaseException as exc:
                record_cancellation(
                    bool(
                        task_cancellation_pending()
                        or isinstance(
                            exc,
                            (
                                asyncio.CancelledError,
                                M110SafetyCallCancelled,
                            ),
                        )
                        or (
                            isinstance(exc, M110DurableHookFailure)
                            and exc.cancellation_requested
                        )
                    )
                )
                result.emergency_stop_activated = False
                result.emergency_stop_activation_failed = True
                result.events.append(
                    {
                        "phase": "emergency_stop_exception",
                        "exception_type": type(exc).__name__,
                    }
                )

        def terminal_payload(
            *,
            status_override: str | None = None,
            reason_override: str | None = None,
        ) -> dict[str, Any]:
            observer_audit = (
                observer.snapshot() if observer is not None else {}
            )
            mutation_audit = gate.snapshot()
            status_value = status_override or result.status
            reason_value = reason_override or result.reason
            is_pending = status_value in {
                "verification_pending",
                "verified_state_pending_cleanup",
            }
            is_manual = status_value == "manual_review_required"
            physical_verified = bool(
                result.physical_verified and not is_pending and not is_manual
            )
            proof_target_verified = bool(
                result.status == "proof_passed"
                and result.physical_verified
                and result.audio_recheck_passed
                and result.physical_state_restored
                and not is_pending
                and not is_manual
            )
            return {
                **safe_wal_binding,
                "scope": "supervised_transaction",
                "maintenance_dsp_reset": True,
                "result_status": status_value,
                "result_reason": reason_value,
                "verification_candidate_status": candidate_status,
                "verification_candidate_reason": candidate_reason,
                "verification_candidate_phase": candidate_phase,
                "proof_run": bool(proof_run),
                "restore_after_verify": bool(proof_run),
                "capability_kind": (
                    "restore_only_proof"
                    if proof_run
                    else "maintenance_dsp_reset_commit_once"
                ),
                "applied_operation_ids": list(result.applied_operation_ids),
                "rollback_operation_ids": list(result.rollback_operation_ids),
                "physical_verified": physical_verified,
                "audio_recheck_passed": result.audio_recheck_passed,
                "proof_target_verified": proof_target_verified,
                "physical_state_restored": result.physical_state_restored,
                "terminal_after_console_send": bool(attempted_addresses),
                "rollback_attempted": bool(
                    rollback_required or result.rollback_operation_ids
                ),
                "rollback_succeeded": bool(
                    result.physical_state_restored
                    and (
                        rollback_required
                        or result.rollback_operation_ids
                    )
                ),
                "write_state_unknown": result.write_state_unknown,
                "rollback_state_unknown": result.rollback_state_unknown,
                "manual_review_required": is_manual,
                "emergency_stop_activated": (
                    result.emergency_stop_activated
                ),
                "emergency_stop_activation_failed": (
                    result.emergency_stop_activation_failed
                ),
                "emergency_stop_required": is_manual,
                "accepted": False,
                "owner_review_required": True,
                "live_authority_granted": False,
                "readiness_closure_allowed": False,
                "before_manifest_sha256": result.before_manifest_sha256,
                "after_manifest_sha256": result.after_manifest_sha256,
                "physical_truth_scope": result.physical_truth_scope,
                "verification_cut_epoch": result.verification_cut_epoch,
                "verification_cut_reset_surface_sha256": (
                    result.verification_cut_reset_surface_sha256
                ),
                "verification_cut_preservation_manifest_sha256": (
                    result.verification_cut_preservation_manifest_sha256
                ),
                "current_state_guaranteed_at_response": (
                    result.current_state_guaranteed_at_response
                ),
                "attended_no_touch_required_during_execution": (
                    result.attended_no_touch_required_during_execution
                ),
                "fresh_readback_required_before_next_stage": (
                    result.fresh_readback_required_before_next_stage
                ),
                "later_console_event_invalidates_readiness_freshness": (
                    result.later_console_event_invalidates_readiness_freshness
                ),
                "before_manifest": {
                    "schema": M110_RESET_MANIFEST_SCHEMA,
                    "row_count": len(before_rows),
                    "rows": before_rows,
                },
                "after_manifest": after_manifest,
                "preservation_mismatches": list(
                    result.preservation_mismatches
                ),
                "no_outside_mutation_attempt_proven": bool(
                    not is_manual
                    and not self._gate_failed(mutation_audit)
                ),
                "delivered_observer_clean": bool(
                    not self._observer_failed(observer_audit)
                ),
                "delivered_observer_audit": observer_audit,
                "delivered_observer_audit_sha256": _sha256(
                    _canonical_json_bytes(observer_audit)
                ),
                "mutation_audit": mutation_audit,
                "mutation_audit_sha256": _sha256(
                    _canonical_json_bytes(mutation_audit)
                ),
                "cancellation_requested": cancellation_requested,
            }

        async def persist_manual_terminal() -> bool:
            await activate_emergency()
            if not prepared:
                return False
            baseline_observer = (
                observer.snapshot() if observer is not None else {}
            )
            baseline_gate = gate.snapshot()
            baseline_manifest_sha = result.after_manifest_sha256
            baseline_mismatches = tuple(result.preservation_mismatches)
            first_manual_payload = terminal_payload()
            first_path, first_cancelled, _first_digest = await persist_phase(
                "manual_review_required",
                first_manual_payload,
            )

            reset_ok = True
            if attempted_addresses:
                reset_ok = await verify_surface(
                    expected=preimage,
                    phase="manual_terminal_reset_surface_verify",
                )
                if not reset_ok:
                    reset_ok = await restore_preimage(
                        "manual_terminal_compensation"
                    )
            manifest_ok = await refresh_manifest(
                "manual_terminal_preservation_verify"
            )
            current_observer = (
                observer.snapshot() if observer is not None else {}
            )
            current_gate = gate.snapshot()
            observer_changed = bool(
                current_observer.get("event_count", 0)
                != baseline_observer.get("event_count", 0)
            )
            gate_changed = bool(
                current_gate.get("event_count", 0)
                != baseline_gate.get("event_count", 0)
            )
            manifest_changed = bool(
                result.after_manifest_sha256 != baseline_manifest_sha
                or tuple(result.preservation_mismatches)
                != baseline_mismatches
            )
            if self._observer_has_unrestored_surface_risk(
                current_observer
            ):
                mark_manual(
                    result.reason,
                    full_surface_unknown=True,
                )
            correction_required = bool(
                not first_path
                or (
                    first_cancelled
                    and first_manual_payload.get("cancellation_requested")
                    is not True
                )
                or not reset_ok
                or not manifest_ok
                or observer_changed
                or gate_changed
                or manifest_changed
            )
            if not correction_required:
                return first_path
            correction_path, correction_cancelled, _correction_digest = (
                await persist_phase(
                "manual_review_required",
                terminal_payload(),
                )
            )
            if correction_cancelled:
                mark_manual(
                    result.reason,
                    full_surface_unknown=(
                        full_surface_restoration_unknown
                    ),
                )
            return bool(correction_path)

        async def persist_no_write_incident(
            reason: str,
            *,
            linearized_terminal_phase: str,
            success_terminal_attempted: bool,
            success_terminal_durable_confirmed: bool,
        ) -> bool:
            """Persist a post-cut incident without mutating console state."""

            nonlocal gate, emergency_attempted
            result.status = "manual_review_required"
            result.reason = str(reason)
            result.manual_review_required = True
            result.physical_verified = False
            current_gate = gate.snapshot()
            gate_guaranteed = bool(
                current_gate.get("active") is True
                and current_gate.get("sealed") is True
                and current_gate.get("boundary_state_unknown") is not True
            )
            if not gate_guaranteed:
                replacement = M110MaintenanceMutationGate(
                    self.wing_client,
                    preimage,
                    authority_current=forward_authority_current,
                )
                try:
                    replacement.start()
                    replacement.seal()
                    gate = replacement
                    gate_guaranteed = True
                    result.events.append(
                        {
                            "phase": "post_terminal_no_write_gate_installed",
                            "audit": gate.snapshot(),
                        }
                    )
                except BaseException as exc:
                    try:
                        replacement.close()
                    except BaseException:
                        pass
                    result.events.append(
                        {
                            "phase": (
                                "post_terminal_no_write_gate_install_failed"
                            ),
                            "exception_type": type(exc).__name__,
                        }
                    )
            if gate_guaranteed:
                await activate_emergency()
            else:
                emergency_attempted = True
                result.emergency_stop_activated = False
                result.emergency_stop_activation_failed = True
                result.events.append(
                    {
                        "phase": "post_terminal_callbacks_suppressed",
                        "reason": "sealed_no_write_gate_unavailable",
                        "emergency_hook_called": False,
                        "terminal_hook_called": False,
                    }
                )
                return False

            def build_incident_payload() -> dict[str, Any]:
                incident_payload = terminal_payload()
                incident_payload.update({
                    "post_linearization_incident": bool(
                        success_terminal_attempted
                    ),
                    "linearized_terminal_phase": str(
                        linearized_terminal_phase
                    ),
                    "linearized_state_preserved_by_executor": True,
                    "success_terminal_attempted": bool(
                        success_terminal_attempted
                    ),
                    "success_terminal_durable_confirmed": bool(
                        success_terminal_durable_confirmed
                    ),
                    "compensation_attempted": False,
                    "rollback_attempted_after_linearization": False,
                })
                return incident_payload

            first_incident_payload = build_incident_payload()
            path, cancelled, _incident_digest = await persist_phase(
                "manual_review_required",
                first_incident_payload,
            )
            if (
                cancelled
                and first_incident_payload.get("cancellation_requested")
                is not True
            ):
                (
                    correction_path,
                    _correction_cancelled,
                    _correction_digest,
                ) = await persist_phase(
                        "manual_review_required",
                        build_incident_payload(),
                )
                return bool(correction_path)
            return bool(path)

        baseline_observer_events = int(
            sealed_observer_audit.get("event_count", 0)
        )
        baseline_gate_events = int(sealed_gate.get("event_count", 0))

        if (
            prepared
            and result.status in {"applied", "proof_passed"}
        ):
            pending_path, pending_cancelled, _pending_digest = await persist_phase(
                "verification_pending",
                terminal_payload(
                    status_override="verification_pending",
                    reason_override="m110_final_verification_pending",
                ),
            )
            if not pending_path or pending_cancelled:
                if result.status == "applied" and attempted_addresses:
                    await restore_preimage(
                        "verification_pending_failure"
                    )
                mark_manual(
                    (
                        "m110_reset_cancelled"
                        if pending_cancelled
                        else "m110_reset_terminal_wal_failed"
                    )
                )

        if (
            result.status in {"applied", "proof_passed"}
            and not forward_authority_current()
        ):
            restored = (
                await restore_preimage("final_authority_expired")
                if result.status == "applied"
                else result.physical_state_restored
            )
            if restored:
                result.status = "rolled_back"
                result.reason = (
                    "m110_write_authority_expired_before_terminal"
                )
                result.manual_review_required = False
                result.rollback_state_unknown = False
            else:
                mark_manual(
                    "m110_write_authority_expired_before_terminal"
                )

        final_reset_ok = True
        final_cancel_before = finalizer_cancellation
        if attempted_addresses:
            expected_surface = (
                M110_RESET_TARGETS
                if result.status == "applied"
                else preimage
            )
            final_reset_ok = await verify_surface(
                expected=expected_surface,
                phase="final_verification_reset_surface",
            )
        final_cancelled_now = bool(
            finalizer_cancellation and not final_cancel_before
        )
        if not final_reset_ok or final_cancelled_now:
            if attempted_addresses:
                await restore_preimage(
                    "final_verification_reset_failure"
                )
            mark_manual(
                (
                    "m110_reset_cancelled"
                    if final_cancelled_now
                    else "m110_final_reset_surface_mismatch"
                )
            )

        final_manifest_ok = True
        if prepared:
            manifest_cancel_before = finalizer_cancellation
            final_manifest_ok = await refresh_manifest(
                "final_verification_preservation_manifest"
            )
            manifest_cancelled_now = bool(
                finalizer_cancellation and not manifest_cancel_before
            )
            if not final_manifest_ok or manifest_cancelled_now:
                if result.status == "applied" and attempted_addresses:
                    await restore_preimage(
                        "final_preservation_failure"
                    )
                mark_manual(
                    (
                        "m110_reset_cancelled"
                        if manifest_cancelled_now
                        else "m110_final_preservation_manifest_changed"
                    ),
                    full_surface_unknown=bool(
                        result.preservation_mismatches
                    ),
                )

        preterminal_observer = (
            observer.snapshot() if observer is not None else {}
        )
        preterminal_gate = gate.snapshot()
        result.observer_audits.append(dict(preterminal_observer))
        late_observer_event = bool(
            int(preterminal_observer.get("event_count", 0))
            > baseline_observer_events
        )
        late_gate_event = bool(
            int(preterminal_gate.get("event_count", 0))
            > baseline_gate_events
        )
        if (
            result.status in {"applied", "proof_passed"}
            and (
                self._observer_failed(preterminal_observer)
                or self._gate_failed(preterminal_gate)
                or late_observer_event
                or late_gate_event
            )
        ):
            if attempted_addresses:
                await restore_preimage(
                    "preterminal_safety_compensation"
                )
            mark_manual(
                "m110_post_terminal_safety_audit_failed",
                full_surface_unknown=(
                    self._observer_has_unrestored_surface_risk(
                        preterminal_observer
                    )
                ),
            )
        elif (
            result.status not in {
                "manual_review_required",
                "rolled_back",
            }
            and (late_observer_event or late_gate_event)
        ):
            mark_manual(
                "m110_post_terminal_safety_audit_failed",
                full_surface_unknown=(
                    self._observer_has_unrestored_surface_risk(
                        preterminal_observer
                    )
                ),
            )

        terminal_phase = "not_prepared"
        phase_by_status = {
            "applied": "committed",
            "proof_passed": "proof_restored",
            "rolled_back": "rolled_back",
            "blocked": "aborted_before_send",
        }
        final_phase = phase_by_status.get(
            result.status,
            "manual_review_required",
        )

        if result.status == "manual_review_required":
            terminal_phase = (
                "manual_review_required"
                if await persist_manual_terminal()
                else "terminal_record_failed"
            )
        elif prepared and result.status != "blocked":
            candidate_observer_before = (
                observer.snapshot() if observer is not None else {}
            )
            candidate_gate_before = gate.snapshot()
            candidate_manifest_before = result.after_manifest_sha256
            candidate_mismatches_before = tuple(
                result.preservation_mismatches
            )
            candidate_path, candidate_cancelled, _candidate_digest = (
                await persist_phase(
                "verified_state_pending_cleanup",
                terminal_payload(
                    status_override="verified_state_pending_cleanup",
                    reason_override=(
                        "m110_verified_state_waiting_for_guard_cleanup"
                    ),
                ),
                )
            )
            if not candidate_path or candidate_cancelled:
                if result.status == "applied" and attempted_addresses:
                    await restore_preimage(
                        "cleanup_candidate_wal_failure"
                    )
                mark_manual(
                    (
                        "m110_reset_cancelled"
                        if candidate_cancelled
                        else "m110_reset_terminal_wal_failed"
                    )
                )
            else:
                post_candidate_reset_ok = True
                if attempted_addresses:
                    post_candidate_expected = (
                        M110_RESET_TARGETS
                        if result.status == "applied"
                        else preimage
                    )
                    post_candidate_reset_ok = await verify_surface(
                        expected=post_candidate_expected,
                        phase="post_candidate_reset_surface_verify",
                    )
                post_candidate_manifest_ok = await refresh_manifest(
                    "post_candidate_preservation_manifest"
                )
                candidate_observer_after = (
                    observer.snapshot() if observer is not None else {}
                )
                candidate_gate_after = gate.snapshot()
                result.observer_audits.append(
                    dict(candidate_observer_after)
                )
                candidate_observer_changed = bool(
                    candidate_observer_after.get("event_count", 0)
                    != candidate_observer_before.get("event_count", 0)
                )
                candidate_gate_changed = bool(
                    candidate_gate_after.get("event_count", 0)
                    != candidate_gate_before.get("event_count", 0)
                )
                candidate_manifest_changed = bool(
                    result.after_manifest_sha256
                    != candidate_manifest_before
                    or tuple(result.preservation_mismatches)
                    != candidate_mismatches_before
                )
                candidate_authority_expired = bool(
                    result.status in {"applied", "proof_passed"}
                    and not forward_authority_current()
                )
                if (
                    not post_candidate_reset_ok
                    or not post_candidate_manifest_ok
                    or candidate_observer_changed
                    or candidate_gate_changed
                    or candidate_manifest_changed
                    or candidate_authority_expired
                ):
                    candidate_surface_risk = bool(
                        not post_candidate_reset_ok
                        or self._observer_has_unrestored_surface_risk(
                            candidate_observer_after
                        )
                    )
                    if attempted_addresses and (
                        result.status == "applied"
                        or candidate_surface_risk
                    ):
                        await restore_preimage(
                            "post_candidate_compensation"
                        )
                    mark_manual(
                        (
                            "m110_write_authority_expired_before_terminal"
                            if candidate_authority_expired
                            else "m110_post_terminal_safety_audit_failed"
                        ),
                        full_surface_unknown=bool(
                            not post_candidate_manifest_ok
                            or candidate_manifest_changed
                            or self._observer_has_unrestored_surface_risk(
                                candidate_observer_after
                            )
                        ),
                    )

            if result.status == "manual_review_required":
                terminal_phase = (
                    "manual_review_required"
                    if await persist_manual_terminal()
                    else "terminal_record_failed"
                )

        # Keep the delivered-state observer active during the last physical
        # 64+184 sweep, then detach it while the mutation boundary remains
        # active and sealed. The attended no-touch contract covers the full
        # Apply/Proof-to-visible-terminal interval, including the small gap
        # between observer detach and the success WAL linearization point.
        if prepared and result.status != "manual_review_required":
            observer_before_sweep = (
                observer.snapshot() if observer is not None else {}
            )
            gate_before_sweep = gate.snapshot()
            detach_reset_ok = True
            if attempted_addresses:
                detach_expected = (
                    M110_RESET_TARGETS
                    if result.status == "applied"
                    else preimage
                )
                detach_reset_ok = await verify_surface(
                    expected=detach_expected,
                    phase="final_observed_reset_surface_verify",
                )
            detach_manifest_ok = await refresh_manifest(
                "final_observed_preservation_manifest"
            )
            observer_after_sweep = (
                observer.snapshot() if observer is not None else {}
            )
            observer_closed = (
                observer.close() if observer is not None else {}
            )
            result.observer_audits.append(dict(observer_after_sweep))
            result.observer_audits.append(dict(observer_closed))
            observer_sweep_changed = bool(
                observer_after_sweep.get("event_count", 0)
                != observer_before_sweep.get("event_count", 0)
            )
            observer_close_failed = bool(
                observer_after_sweep.get("active") is True
                and observer_closed.get("unsubscribe_failed") is True
            )
            observer_close_changed = bool(
                observer_closed.get("event_count", 0)
                != observer_after_sweep.get("event_count", 0)
            )
            gate_after_detach = gate.snapshot()
            detach_gate_changed = bool(
                gate_after_detach.get("event_count", 0)
                != gate_before_sweep.get("event_count", 0)
                or gate_after_detach.get("overflow") is True
                and gate_before_sweep.get("overflow") is not True
            )
            detach_authority_expired = bool(
                result.status in {"applied", "proof_passed"}
                and not forward_authority_current()
            )
            task_cancelled_now = bool(
                current_task is not None
                and current_task.cancelling() > initial_cancelling
            )
            record_cancellation(task_cancelled_now)
            detach_cancelled = bool(
                cancellation_requested
                and result.status not in {"blocked", "rolled_back"}
            )
            detach_fault = bool(
                self._observer_failed(observer_before_sweep)
                or self._observer_failed(observer_after_sweep)
                or self._observer_failed(observer_closed)
                or observer_sweep_changed
                or observer_close_failed
                or observer_close_changed
                or not detach_reset_ok
                or not detach_manifest_ok
                or detach_gate_changed
                or detach_authority_expired
                or detach_cancelled
                or gate_after_detach.get("active") is not True
                or gate_after_detach.get("sealed") is not True
            )
            if detach_fault:
                if result.status == "applied" and attempted_addresses:
                    await restore_preimage(
                        "observer_detach_compensation"
                    )
                mark_manual(
                    (
                        "m110_write_authority_expired_before_terminal"
                        if detach_authority_expired
                        else "m110_reset_cancelled"
                        if detach_cancelled
                        else "m110_delivered_observer_unsubscribe_failed"
                        if observer_close_failed
                        else "m110_observer_detach_safety_audit_failed"
                    ),
                    full_surface_unknown=bool(
                        observer_close_failed
                        or not detach_manifest_ok
                        or self._observer_has_unrestored_surface_risk(
                            observer_closed
                        )
                    ),
                )
                terminal_phase = (
                    "manual_review_required"
                    if await persist_manual_terminal()
                    else "terminal_record_failed"
                )
            elif attempted_addresses:
                verification_values = (
                    M110_RESET_TARGETS
                    if result.status == "applied"
                    else preimage
                )
                result.verification_cut_epoch = time.time()
                result.verification_cut_reset_surface_sha256 = _sha256(
                    _canonical_json_bytes(
                        [
                            {
                                "address": address,
                                "value": verification_values[address],
                            }
                            for address in M110_RESET_ADDRESSES
                        ]
                    )
                )
                result.verification_cut_preservation_manifest_sha256 = (
                    result.after_manifest_sha256
                )
                result.physical_truth_scope = (
                    "verified_as_of_m110_verification_cut"
                )

        success_terminal_attempted = False
        success_terminal_durable_confirmed = False
        linearized_terminal_phase = ""
        if prepared and result.status != "manual_review_required":
            final_phase = phase_by_status.get(
                result.status,
                "manual_review_required",
            )
            strict_success_terminal = result.status in {
                "applied",
                "proof_passed",
            }
            final_gate_before = gate.snapshot()
            final_cancel_precheck = bool(
                strict_success_terminal
                and current_task is not None
                and current_task.cancelling() > initial_cancelling
            )
            record_cancellation(final_cancel_precheck)
            final_authority_precheck_expired = bool(
                strict_success_terminal
                and not forward_authority_current()
            )
            final_gate_precheck_failed = bool(
                (
                    strict_success_terminal
                    and self._gate_failed(final_gate_before)
                )
                or final_gate_before.get("active") is not True
                or final_gate_before.get("sealed") is not True
            )
            final_payload = terminal_payload()
            observer_audit = final_payload.get(
                "delivered_observer_audit",
                {},
            )
            mutation_audit = final_payload.get("mutation_audit", {})
            expected_verification_values = (
                M110_RESET_TARGETS
                if result.status == "applied"
                else preimage
            )
            expected_cut_reset_sha256 = _sha256(
                _canonical_json_bytes(
                    [
                        {
                            "address": address,
                            "value": expected_verification_values[address],
                        }
                        for address in M110_RESET_ADDRESSES
                    ]
                )
            )
            expected_operation_ids = [
                str(row["operation_id"])
                for row in M110_RESET_OPERATION_DEFINITIONS
            ]
            before_manifest_payload = final_payload.get("before_manifest")
            after_manifest_payload = final_payload.get("after_manifest")
            strict_payload_clean = bool(
                not strict_success_terminal
                or (
                    final_payload.get("cancellation_requested") is False
                    and final_payload.get("physical_verified") is True
                    and final_payload.get("audio_recheck_passed") is True
                    and final_payload.get("manual_review_required") is False
                    and final_payload.get("write_state_unknown") is False
                    and final_payload.get("rollback_state_unknown") is False
                    and isinstance(
                        final_payload.get("verification_cut_epoch"),
                        (int, float),
                    )
                    and not isinstance(
                        final_payload.get("verification_cut_epoch"),
                        bool,
                    )
                    and math.isfinite(
                        float(final_payload["verification_cut_epoch"])
                    )
                    and final_payload.get("physical_truth_scope")
                    == "verified_as_of_m110_verification_cut"
                    and final_payload.get(
                        "current_state_guaranteed_at_response"
                    )
                    is False
                    and final_payload.get(
                        "fresh_readback_required_before_next_stage"
                    )
                    is True
                    and final_payload.get(
                        "attended_no_touch_required_during_execution"
                    )
                    is True
                    and final_payload.get(
                        "later_console_event_invalidates_readiness_freshness"
                    )
                    is True
                    and final_payload.get(
                        "verification_cut_reset_surface_sha256"
                    )
                    == expected_cut_reset_sha256
                    and final_payload.get(
                        "verification_cut_preservation_manifest_sha256"
                    )
                    == final_payload.get("after_manifest_sha256")
                    and final_payload.get("preservation_mismatches") == []
                    and final_payload.get(
                        "no_outside_mutation_attempt_proven"
                    )
                    is True
                    and final_payload.get("delivered_observer_clean") is True
                    and isinstance(observer_audit, Mapping)
                    and observer_audit.get("active") is False
                    and observer_audit.get("unsubscribe_failed") is False
                    and observer_audit.get("overflow") is False
                    and not observer_audit.get("unexpected_events")
                    and isinstance(mutation_audit, Mapping)
                    and mutation_audit.get("active") is True
                    and mutation_audit.get("sealed") is True
                    and mutation_audit.get("remove_failed") is False
                    and mutation_audit.get("overflow") is False
                    and not mutation_audit.get("denied_attempts")
                    and isinstance(before_manifest_payload, Mapping)
                    and before_manifest_payload.get("row_count") == 184
                    and final_payload.get("before_manifest_sha256")
                    == _sha256(
                        _canonical_json_bytes(before_manifest_payload)
                    )
                    and isinstance(after_manifest_payload, Mapping)
                    and after_manifest_payload.get("row_count") == 184
                    and final_payload.get("after_manifest_sha256")
                    == _sha256(
                        _canonical_json_bytes(after_manifest_payload)
                    )
                    and final_payload.get("applied_operation_ids")
                    == expected_operation_ids
                    and (
                        result.status != "proof_passed"
                        or (
                            final_payload.get("proof_target_verified") is True
                            and final_payload.get("physical_state_restored")
                            is True
                            and final_payload.get("rollback_attempted") is True
                            and final_payload.get("rollback_succeeded") is True
                            and final_payload.get("rollback_operation_ids")
                            == list(reversed(expected_operation_ids))
                        )
                    )
                )
            )
            if (
                final_gate_precheck_failed
                or final_cancel_precheck
                or final_authority_precheck_expired
                or not strict_payload_clean
            ):
                if result.status == "applied" and attempted_addresses:
                    await restore_preimage(
                        "pre_terminal_invariant_compensation"
                    )
                mark_manual(
                    (
                        "m110_reset_cancelled"
                        if final_cancel_precheck
                        else "m110_write_authority_expired_before_terminal"
                        if final_authority_precheck_expired
                        else "m110_preterminal_success_invariant_failed"
                    ),
                )
                terminal_phase = (
                    "manual_review_required"
                    if await persist_manual_terminal()
                    else "terminal_record_failed"
                )
            else:
                success_terminal_attempted = strict_success_terminal
                linearized_terminal_phase = (
                    final_phase if strict_success_terminal else ""
                )
                final_payload_for_persist = json.loads(
                    _canonical_json_bytes(final_payload)
                )
                final_payload_digest_before = _sha256(
                    _canonical_json_bytes(final_payload_for_persist)
                )
                (
                    final_path,
                    final_cancelled,
                    final_receipt_digest,
                ) = await persist_phase(
                    final_phase,
                    final_payload_for_persist,
                )
                try:
                    final_payload_mutated = bool(
                        _sha256(
                            _canonical_json_bytes(final_payload_for_persist)
                        )
                        != final_payload_digest_before
                    )
                except BaseException:
                    final_payload_mutated = True
                final_receipt_bound = bool(
                    not strict_success_terminal
                    or final_receipt_digest
                    == final_payload_digest_before
                )
                success_terminal_durable_confirmed = bool(
                    strict_success_terminal
                    and final_path
                    and final_receipt_bound
                )
                post_persist_cancelled = bool(
                    strict_success_terminal
                    and current_task is not None
                    and current_task.cancelling() > initial_cancelling
                )
                record_cancellation(post_persist_cancelled)
                final_cancelled = bool(
                    final_cancelled or post_persist_cancelled
                )
                final_gate_after = gate.snapshot()
                final_gate_changed = bool(
                    final_gate_after.get("event_count", 0)
                    != final_gate_before.get("event_count", 0)
                    or (
                        final_gate_after.get("overflow") is True
                        and final_gate_before.get("overflow") is not True
                    )
                )
                final_authority_expired = bool(
                    strict_success_terminal
                    and not forward_authority_current()
                )
                post_terminal_fault = bool(
                    not final_path
                    or final_cancelled
                    or final_payload_mutated
                    or not final_receipt_bound
                    or final_gate_changed
                    or (
                        strict_success_terminal
                        and self._gate_failed(final_gate_after)
                    )
                    or final_authority_expired
                    or final_gate_after.get("active") is not True
                    or final_gate_after.get("sealed") is not True
                )
                if post_terminal_fault:
                    incident_reason = (
                        "m110_reset_cancelled"
                        if final_cancelled
                        else "m110_reset_terminal_wal_failed"
                        if not final_path
                        else "m110_success_terminal_payload_mutated"
                        if final_payload_mutated
                        else "m110_success_terminal_receipt_unbound"
                        if not final_receipt_bound
                        else "m110_write_authority_expired_before_terminal"
                        if final_authority_expired
                        else "m110_post_terminal_safety_audit_failed"
                    )
                    terminal_phase = (
                        "manual_review_required"
                        if await persist_no_write_incident(
                            incident_reason,
                            linearized_terminal_phase=(
                                linearized_terminal_phase
                            ),
                            success_terminal_attempted=(
                                success_terminal_attempted
                            ),
                            success_terminal_durable_confirmed=(
                                success_terminal_durable_confirmed
                            ),
                        )
                        else "terminal_record_failed"
                    )
                else:
                    terminal_phase = final_phase

        # A manual path may have skipped the pre-terminal detach. Keep the gate
        # sealed while removing the observer so no programmatic write can race.
        if observer is not None and observer.snapshot().get("active") is True:
            observer_before_close = observer.snapshot()
            observer_closed = observer.close()
            result.observer_audits.append(dict(observer_closed))
            if (
                observer_closed.get("unsubscribe_failed") is True
                or observer_closed.get("event_count", 0)
                != observer_before_close.get("event_count", 0)
            ):
                mark_manual(
                    "m110_delivered_observer_unsubscribe_failed",
                    full_surface_unknown=self._observer_has_unrestored_surface_risk(
                        observer_closed
                    ),
                )
                terminal_phase = (
                    "manual_review_required"
                    if await persist_manual_terminal()
                    else "terminal_record_failed"
                )

        gate_before_close = gate.snapshot()
        gate_preclose_failed = bool(
            success_terminal_attempted
            and result.status != "manual_review_required"
            and self._gate_failed(gate_before_close)
        )
        if gate_before_close.get("active") is True:
            closed_gate = gate.close()
        else:
            closed_gate = gate_before_close
        result.mutation_audit = dict(closed_gate)
        gate_remove_failed = bool(
            closed_gate.get("active") is True
            or closed_gate.get("remove_failed") is True
        )
        gate_close_changed = bool(
            closed_gate.get("event_count", 0)
            != gate_before_close.get("event_count", 0)
            or closed_gate.get("overflow") is True
            and gate_before_close.get("overflow") is not True
        )
        post_close_authority_expired = bool(
            success_terminal_attempted
            and result.status != "manual_review_required"
            and not forward_authority_current()
        )
        if (
            gate_remove_failed
            or gate_close_changed
            or gate_preclose_failed
            or post_close_authority_expired
        ):
            close_reason = (
                "m110_write_authority_expired_after_terminal"
                if post_close_authority_expired
                else "m110_mutation_boundary_remove_failed"
                if gate_remove_failed
                else "m110_mutation_boundary_close_audit_failed"
            )
            result.events.append(
                {
                    "phase": "mutation_boundary_close_fault",
                    "audit": dict(closed_gate),
                    "gate_left_fail_closed": bool(
                        closed_gate.get("active") is True
                        and closed_gate.get("sealed") is True
                        and closed_gate.get("boundary_state_unknown")
                        is not True
                    ),
                    "compensation_attempted": False,
                }
            )
            terminal_phase = (
                "manual_review_required"
                if await persist_no_write_incident(
                    close_reason,
                    linearized_terminal_phase=linearized_terminal_phase,
                    success_terminal_attempted=success_terminal_attempted,
                    success_terminal_durable_confirmed=(
                        success_terminal_durable_confirmed
                    ),
                )
                else "terminal_record_failed"
            )
        result.mutation_audit = dict(gate.snapshot())
        if result.manual_review_required and not emergency_attempted:
            await activate_emergency()
        if full_surface_restoration_unknown:
            result.physical_state_restored = False
            result.rollback_state_unknown = True
        result.mutation_audit = dict(gate.snapshot())
        result.wal_terminal = terminal_phase
        return result
