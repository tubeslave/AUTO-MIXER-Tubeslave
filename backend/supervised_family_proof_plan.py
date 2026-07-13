"""Canonical M110 proof/commit operation-bundle helpers."""

from __future__ import annotations

import hashlib
import json
import math
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


def canonical_operation_bundle_bytes(
    bundle: Mapping[str, Any],
) -> bytes:
    return json.dumps(
        bundle,
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
    return {
        "schema": CANONICAL_ATTENDED_OPERATION_BUNDLE_SCHEMA,
        "bundle_kind": bundle_kind,
        "operations": [dict(operation) for operation in operations],
        "safety_binding": dict(safety_binding),
    }
