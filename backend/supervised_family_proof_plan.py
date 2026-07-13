"""Canonical M110 proof/commit operation-bundle helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence


CANONICAL_ATTENDED_OPERATION_BUNDLE_SCHEMA = (
    "automixer.canonical_attended_operation_bundle.v1"
)
CANONICAL_ATTENDED_BUNDLE_KINDS = (
    "single_soundcheck_operation",
    "maintenance_dsp_reset_bundle",
    "processor_enable_bundle",
)
M110_CANONICAL_OPERATION_FIELDS = frozenset(
    {
        "operation_id",
        "family_id",
        "write_kind",
        "channel",
        "parameter",
        "writes",
        "audible_state_reads",
    }
)
M110_CANONICAL_SAFETY_BINDING_FIELDS = frozenset(
    {
        "adapter_revision",
        "protocol_revision",
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
    }
)


def canonical_decimal_string(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("boolean_is_not_a_canonical_decimal")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non_finite_canonical_decimal")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("invalid_canonical_decimal") from exc
    if not decimal.is_finite():
        raise ValueError("non_finite_canonical_decimal")
    if decimal == 0:
        return "0"
    text = format(decimal, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _canonical_target_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, Decimal)):
        return canonical_decimal_string(value)
    if isinstance(value, str) and value and "\x00" not in value:
        return value
    raise ValueError("unsupported_canonical_target_value")


def _canonical_digest(value: Any, *, reason: str) -> str:
    if not (
        isinstance(value, str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", value)
    ):
        raise ValueError(reason)
    return value


def _canonical_nonempty_text(value: Any, *, reason: str, limit: int = 512) -> str:
    if not (
        isinstance(value, str)
        and value
        and "\x00" not in value
        and len(value.encode("utf-8")) <= limit
    ):
        raise ValueError(reason)
    return value


def canonicalize_attended_safety_binding(
    safety_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if type(safety_binding) is not dict or set(safety_binding) != (
        M110_CANONICAL_SAFETY_BINDING_FIELDS
    ):
        raise ValueError("attended_safety_binding_fields_mismatch")
    result = {}
    for field_name in (
        "adapter_revision",
        "protocol_revision",
        "console_serial",
        "firmware_version",
        "route_proof_id",
        "capture_hardware_uid",
    ):
        result[field_name] = _canonical_nonempty_text(
            safety_binding[field_name],
            reason=f"attended_safety_binding_{field_name}_invalid",
        )
    for field_name, length in (
        ("runtime_instance_nonce", 32),
        ("source_head", 40),
        ("source_tree", 40),
    ):
        value = safety_binding[field_name]
        if not (
            isinstance(value, str)
            and re.fullmatch(rf"[0-9a-f]{{{length}}}", value)
        ):
            raise ValueError(f"attended_safety_binding_{field_name}_invalid")
        result[field_name] = value
    for field_name in (
        "runtime_manifest_sha256",
        "route_artifact_sha256",
    ):
        result[field_name] = _canonical_digest(
            safety_binding[field_name],
            reason=f"attended_safety_binding_{field_name}_invalid",
        )
    for field_name in (
        "connection_generation",
        "transport_generation",
        "emergency_generation",
    ):
        value = safety_binding[field_name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"attended_safety_binding_{field_name}_invalid")
        result[field_name] = value
    return result


def canonicalize_attended_operation(
    operation: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(operation, Mapping) or not isinstance(prepared, Mapping):
        raise ValueError("attended_operation_mapping_required")
    operation_id = _canonical_nonempty_text(
        operation.get("operation_id"),
        reason="attended_operation_id_invalid",
    )
    family_id = _canonical_nonempty_text(
        operation.get("family_id"),
        reason="attended_operation_family_invalid",
    )
    write_kind = _canonical_nonempty_text(
        operation.get("write_kind"),
        reason="attended_operation_write_kind_invalid",
    )
    parameter = operation.get("parameter")
    if not isinstance(parameter, str) or len(parameter.encode("utf-8")) > 512:
        raise ValueError("attended_operation_parameter_invalid")
    channel = operation.get("channel")
    if channel is not None and (
        not isinstance(channel, int)
        or isinstance(channel, bool)
        or not 1 <= channel <= 48
    ):
        raise ValueError("attended_operation_channel_invalid")

    def canonical_specs(
        raw_specs: Any,
        *,
        target_field: str,
    ) -> list[dict[str, Any]]:
        if not isinstance(raw_specs, (list, tuple)):
            raise ValueError("attended_operation_specs_invalid")
        rows = []
        seen_addresses = set()
        for raw in raw_specs:
            if not isinstance(raw, Mapping) or set(raw) != {
                "address",
                target_field,
                "tolerance",
            }:
                raise ValueError("attended_operation_spec_fields_mismatch")
            address = raw.get("address")
            if not (
                isinstance(address, str)
                and re.fullmatch(
                    r"/[A-Za-z0-9_$.-]+(?:/[A-Za-z0-9_$.-]+)+",
                    address,
                )
                and address not in seen_addresses
            ):
                raise ValueError("attended_operation_address_invalid")
            seen_addresses.add(address)
            rows.append(
                {
                    "address": address,
                    target_field: _canonical_target_value(raw.get(target_field)),
                    "tolerance": canonical_decimal_string(raw.get("tolerance")),
                }
            )
        return rows

    writes = canonical_specs(prepared.get("writes"), target_field="target_value")
    if not writes:
        raise ValueError("attended_operation_writes_required")
    audible_state_reads = canonical_specs(
        prepared.get("audible_state_reads", ()),
        target_field="expected_value",
    )
    expected_addresses = tuple(row["address"] for row in writes)
    supplied_addresses = tuple(prepared.get("addresses") or ())
    if supplied_addresses != expected_addresses:
        raise ValueError("attended_operation_prepared_addresses_mismatch")
    if prepared.get("write_count") != len(writes) or isinstance(
        prepared.get("write_count"), bool
    ):
        raise ValueError("attended_operation_write_count_mismatch")
    return {
        "operation_id": operation_id,
        "family_id": family_id,
        "write_kind": write_kind,
        "channel": channel,
        "parameter": parameter,
        "writes": writes,
        "audible_state_reads": audible_state_reads,
    }


def canonical_operation_bundle_bytes(bundle: Mapping[str, Any]) -> bytes:
    validated = validate_canonical_operation_bundle(bundle)
    return json.dumps(
        validated,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_operation_bundle_digest(bundle: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        canonical_operation_bundle_bytes(bundle)
    ).hexdigest()


def build_canonical_operation_bundle(
    *,
    bundle_kind: str,
    operations: Sequence[Mapping[str, Any]],
    safety_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if bundle_kind not in CANONICAL_ATTENDED_BUNDLE_KINDS:
        raise ValueError("unsupported_attended_bundle_kind")
    if not operations:
        raise ValueError("attended_bundle_operations_required")
    if bundle_kind == "single_soundcheck_operation" and len(operations) != 1:
        raise ValueError("single_soundcheck_operation_must_be_singular")
    bundle = {
        "schema": CANONICAL_ATTENDED_OPERATION_BUNDLE_SCHEMA,
        "bundle_kind": bundle_kind,
        "operations": [dict(operation) for operation in operations],
        "safety_binding": canonicalize_attended_safety_binding(
            dict(safety_binding)
        ),
    }
    return validate_canonical_operation_bundle(bundle)


def build_canonical_operation_bundle_from_prepared(
    *,
    bundle_kind: str,
    operations: Sequence[Mapping[str, Any]],
    prepared_operations: Sequence[Mapping[str, Any]],
    safety_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if len(operations) != len(prepared_operations):
        raise ValueError("attended_bundle_prepared_operation_count_mismatch")
    canonical_operations = [
        canonicalize_attended_operation(operation, prepared)
        for operation, prepared in zip(operations, prepared_operations)
    ]
    return build_canonical_operation_bundle(
        bundle_kind=bundle_kind,
        operations=canonical_operations,
        safety_binding=safety_binding,
    )


def validate_canonical_operation_bundle(
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    if type(bundle) is not dict or set(bundle) != {
        "schema",
        "bundle_kind",
        "operations",
        "safety_binding",
    }:
        raise ValueError("attended_bundle_fields_mismatch")
    if bundle.get("schema") != CANONICAL_ATTENDED_OPERATION_BUNDLE_SCHEMA:
        raise ValueError("attended_bundle_schema_invalid")
    kind = bundle.get("bundle_kind")
    if kind not in CANONICAL_ATTENDED_BUNDLE_KINDS:
        raise ValueError("unsupported_attended_bundle_kind")
    operations = bundle.get("operations")
    if not isinstance(operations, list) or not 1 <= len(operations) <= 64:
        raise ValueError("attended_bundle_operation_count_invalid")
    if kind == "single_soundcheck_operation" and len(operations) != 1:
        raise ValueError("single_soundcheck_operation_must_be_singular")
    operation_ids = []
    addresses = []
    for operation in operations:
        if type(operation) is not dict or set(operation) != (
            M110_CANONICAL_OPERATION_FIELDS
        ):
            raise ValueError("attended_bundle_operation_fields_mismatch")
        operation_ids.append(operation.get("operation_id"))
        if not (
            isinstance(operation.get("operation_id"), str)
            and operation.get("operation_id")
            and isinstance(operation.get("family_id"), str)
            and operation.get("family_id")
            and isinstance(operation.get("write_kind"), str)
            and operation.get("write_kind")
            and isinstance(operation.get("parameter"), str)
        ):
            raise ValueError("attended_bundle_operation_identity_invalid")
        channel = operation.get("channel")
        if channel is not None and (
            not isinstance(channel, int)
            or isinstance(channel, bool)
            or not 1 <= channel <= 48
        ):
            raise ValueError("attended_bundle_operation_channel_invalid")
        writes = operation.get("writes")
        reads = operation.get("audible_state_reads")
        if not isinstance(writes, list) or not writes or not isinstance(reads, list):
            raise ValueError("attended_bundle_operation_specs_invalid")
        for row, target_field in (
            *((row, "target_value") for row in writes),
            *((row, "expected_value") for row in reads),
        ):
            if type(row) is not dict or set(row) != {
                "address",
                target_field,
                "tolerance",
            }:
                raise ValueError("attended_bundle_spec_fields_mismatch")
            address = row.get("address")
            if not (
                isinstance(address, str)
                and re.fullmatch(
                    r"/[A-Za-z0-9_$.-]+(?:/[A-Za-z0-9_$.-]+)+",
                    address,
                )
            ):
                raise ValueError("attended_bundle_address_invalid")
            addresses.append(address)
            target = row.get(target_field)
            if not isinstance(target, (str, bool)) or (
                isinstance(target, str) and not target
            ):
                raise ValueError("attended_bundle_target_invalid")
            tolerance = row.get("tolerance")
            if not isinstance(tolerance, str) or canonical_decimal_string(
                tolerance
            ) != tolerance:
                raise ValueError("attended_bundle_tolerance_invalid")
    if len(operation_ids) != len(set(operation_ids)):
        raise ValueError("attended_bundle_duplicate_operation_id")
    if len(addresses) != len(set(addresses)):
        raise ValueError("attended_bundle_duplicate_address")
    return {
        "schema": CANONICAL_ATTENDED_OPERATION_BUNDLE_SCHEMA,
        "bundle_kind": kind,
        "operations": [dict(operation) for operation in operations],
        "safety_binding": canonicalize_attended_safety_binding(
            dict(bundle.get("safety_binding") or {})
        ),
    }


def canonical_bundle_scope(
    bundle: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    validated = validate_canonical_operation_bundle(bundle)
    addresses = []
    targets = []
    for operation in validated["operations"]:
        for write in operation["writes"]:
            address = write["address"]
            target = json.dumps(
                write["target_value"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            addresses.append(address)
            targets.append((address, target))
    return tuple(addresses), tuple(targets)
