"""Finite exact-preimage restoration contract for the verified M006 write.

The attended M110 reset first bypasses channel DSP.  This module then builds
three small, immutable transactions that restore the non-enable parameters
changed by the real M006 transaction while keeping every processor bypassed.
It never accepts client addresses or targets.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from supervised_console_transaction import ConsoleTransactionOperation
from supervised_proof_capability import M110_M006_RESTORE_BUNDLE_KIND
from wing_supervised_console_adapters import parameter_writes_for_operation


M110_M006_FIXTURE_RELATIVE_PATH = (
    "backend/contracts/m110_m006_restore_20260711.json"
)
M110_M006_FIXTURE_SHA256 = (
    "e75ced24084dfe1526c707122d837bcc51c1fd0829313faf6abce7f44840b274"
)
M110_M006_SOURCE_EVENT_SHA256 = (
    "0ce7efb80f19e38f0a5e926467f77845b05ebee9820c78e7ec362b2443128d51"
)
M110_M006_SOURCE_TRANSACTION_ID = "soundcheck:normal:13c43cc98a3c487c"
M110_M006_RESTORE_ADAPTER_REVISION = "m110-m006-exact-preimage-restore.v1"
M110_M006_RESTORE_PROTOCOL_REVISION = "wing-osc-fw3.0.5"
M110_M006_RESTORE_STAGES = (
    "foundation_level",
    "tonal_dynamics",
    "spatial_balance",
)
M110_M006_RESTORE_BUNDLE_SELECTOR_PREFIX = "m006_exact_preimage_"
_PROCESSOR_ENABLE_SUFFIXES = (
    "/eq/on",
    "/flt/lc",
    "/gate/on",
    "/dyn/on",
)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("m110_m006_fixture_duplicate_json_key")
        value[key] = item
    return value


def load_m110_m006_fixture() -> dict[str, Any]:
    """Load the tracked immutable evidence fixture and verify its identity."""

    path = Path(__file__).resolve().parents[1] / M110_M006_FIXTURE_RELATIVE_PATH
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != M110_M006_FIXTURE_SHA256:
        raise ValueError("m110_m006_fixture_sha256_mismatch")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("m110_m006_fixture_json_invalid") from exc
    if not (
        type(payload) is dict
        and set(payload) == {"operation_count", "operations", "schema", "source"}
        and payload.get("schema") == "m006_exact_transaction_fixture/v1"
        and payload.get("operation_count") == 51
        and isinstance(payload.get("operations"), list)
        and len(payload["operations"]) == 51
        and isinstance(payload.get("source"), Mapping)
        and payload["source"].get("event_line_sha256")
        == M110_M006_SOURCE_EVENT_SHA256
        and payload["source"].get("transaction_id")
        == M110_M006_SOURCE_TRANSACTION_ID
    ):
        raise ValueError("m110_m006_fixture_contract_invalid")
    operation_ids = [
        str(item.get("operation_id") or "")
        for item in payload["operations"]
        if isinstance(item, Mapping)
    ]
    if len(operation_ids) != 51 or len(set(operation_ids)) != 51 or not all(
        operation_ids
    ):
        raise ValueError("m110_m006_fixture_operation_identity_invalid")
    return payload


def _original_operation(raw: Mapping[str, Any]) -> ConsoleTransactionOperation:
    return ConsoleTransactionOperation(
        operation_id=str(raw["operation_id"]),
        family_id=str(raw["family_id"]),
        write_kind=str(raw["write_kind"]),
        target_value=raw["target_value"],
        channel=int(raw["channel"]),
        parameter=str(raw["parameter"]),
        metadata=dict(raw.get("metadata") or {}),
    )


def _restore_target(raw: Mapping[str, Any]) -> Any:
    channel = int(raw["channel"])
    base = f"/ch/{channel}"
    snapshots = dict(raw["snapshot_values"])
    write_kind = str(raw["write_kind"])
    if write_kind in {"gain", "trim"}:
        return snapshots[f"{base}/in/set/trim"]
    if write_kind == "fader":
        return snapshots[f"{base}/fdr"]
    if write_kind == "pan":
        return snapshots[f"{base}/pan"]
    if write_kind == "hpf":
        return {
            "enabled": False,
            "freq_hz": snapshots[f"{base}/flt/lcf"],
        }
    if write_kind == "eq":
        band = int(dict(raw["target_value"])["band"])
        return {
            "band": band,
            "freq_hz": snapshots[f"{base}/eq/{band}f"],
            "gain_db": snapshots[f"{base}/eq/{band}g"],
            "q": snapshots[f"{base}/eq/{band}q"],
        }
    if write_kind == "gate":
        return {
            "threshold_db": snapshots[f"{base}/gate/thr"],
            "range_db": snapshots[f"{base}/gate/range"],
            "attack_ms": snapshots[f"{base}/gate/att"],
            "hold_ms": snapshots[f"{base}/gate/hld"],
            "release_ms": snapshots[f"{base}/gate/rel"],
            "enabled": False,
        }
    if write_kind == "compressor":
        return {
            "threshold_db": snapshots[f"{base}/dyn/thr"],
            "ratio": snapshots[f"{base}/dyn/ratio"],
            "attack_ms": snapshots[f"{base}/dyn/att"],
            "release_ms": snapshots[f"{base}/dyn/rel"],
            "makeup_db": snapshots[f"{base}/dyn/gain"],
            "enabled": False,
        }
    raise ValueError(f"m110_m006_restore_write_kind_unsupported:{write_kind}")


def _restore_stage(raw: Mapping[str, Any]) -> str:
    write_kind = str(raw["write_kind"])
    if write_kind in {"gain", "trim", "hpf", "fader"}:
        return "foundation_level"
    if write_kind in {"eq", "gate", "compressor"}:
        return "tonal_dynamics"
    if write_kind == "pan":
        return "spatial_balance"
    raise ValueError(f"m110_m006_restore_stage_unsupported:{write_kind}")


def _restore_operation(raw: Mapping[str, Any], stage: str) -> dict[str, Any]:
    original = _original_operation(raw)
    restore = ConsoleTransactionOperation(
        operation_id=f"m110:m006-restore:{original.operation_id}",
        family_id=original.family_id,
        write_kind=original.write_kind,
        target_value=_restore_target(raw),
        channel=original.channel,
        parameter=original.parameter,
        metadata={"m110_fixed_preimage_restore": True},
    )
    postimage = {
        write.address: write.value
        for write in parameter_writes_for_operation(original)
    }
    restore_writes = parameter_writes_for_operation(restore)
    restore_addresses = {write.address for write in restore_writes}
    if set(postimage) != restore_addresses:
        raise ValueError("m110_m006_restore_address_surface_changed")
    expected_current_values = {
        address: (
            0
            if address.endswith(_PROCESSOR_ENABLE_SUFFIXES)
            else postimage[address]
        )
        for address in sorted(postimage)
    }
    def values_equal(left: Any, right: Any, tolerance: float) -> bool:
        try:
            left_number = float(left)
            right_number = float(right)
        except (TypeError, ValueError):
            return str(left).strip().upper() == str(right).strip().upper()
        return bool(
            math.isfinite(left_number)
            and math.isfinite(right_number)
            and abs(left_number - right_number) <= tolerance
        )

    changed_addresses = [
        write.address
        for write in restore_writes
        if not write.address.endswith(_PROCESSOR_ENABLE_SUFFIXES)
        and not values_equal(
            postimage[write.address],
            write.value,
            write.tolerance,
        )
    ]
    return {
        "operation_id": restore.operation_id,
        "family_id": restore.family_id,
        "write_kind": restore.write_kind,
        "channel": restore.channel,
        "parameter": restore.parameter,
        "target_value": restore.target_value,
        "metadata": {
            "m110_fixed_preimage_restore": True,
            "m110_restore_stage": stage,
            "m110_source_operation_id": original.operation_id,
            "m110_source_transaction_id": M110_M006_SOURCE_TRANSACTION_ID,
            "m110_source_event_sha256": M110_M006_SOURCE_EVENT_SHA256,
            "m110_expected_current_values": expected_current_values,
            "m110_changed_addresses": changed_addresses,
            "atomic_group_commit_required": True,
            "atomic_group_id": f"m110:m006-restore:{stage}",
            "musically_required": False,
            "required_for": "maintenance_reset",
            "live_mixer_writes": False,
        },
    }


def m110_m006_restore_stage_operations() -> dict[str, tuple[dict[str, Any], ...]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        stage: [] for stage in M110_M006_RESTORE_STAGES
    }
    for raw in load_m110_m006_fixture()["operations"]:
        if not isinstance(raw, Mapping):
            raise ValueError("m110_m006_fixture_operation_invalid")
        stage = _restore_stage(raw)
        grouped[stage].append(_restore_operation(raw, stage))
    result = {
        stage: tuple(grouped[stage]) for stage in M110_M006_RESTORE_STAGES
    }
    if [len(result[stage]) for stage in M110_M006_RESTORE_STAGES] != [23, 14, 14]:
        raise ValueError("m110_m006_restore_stage_operation_count_invalid")
    return result


def build_m110_m006_restore_plan(
    *,
    stage: str,
    role: str,
    proposal_nonce: str,
) -> dict[str, Any]:
    if stage not in M110_M006_RESTORE_STAGES:
        raise ValueError("m110_m006_restore_stage_invalid")
    if role not in {"proof", "commit"}:
        raise ValueError("m110_m006_restore_role_invalid")
    if not proposal_nonce:
        raise ValueError("m110_m006_restore_proposal_nonce_required")
    operations = [
        json.loads(json.dumps(item, allow_nan=False))
        for item in m110_m006_restore_stage_operations()[stage]
    ]
    return {
        "schema_version": "supervised_transaction_plan/v1",
        "transaction_id": f"m110:m006-restore:{stage}:{role}:{proposal_nonce}",
        "reason": (
            "Prove and restore the exact M006 preimage while Main is muted"
            if role == "proof"
            else "Commit the exact proved M006 preimage while Main is muted"
        ),
        "operations": operations,
        "atomic_group_commit_required": True,
        "m110_fixed_preimage_restore": True,
        "m110_restore_stage": stage,
        "m110_source_transaction_id": M110_M006_SOURCE_TRANSACTION_ID,
        "m110_source_event_sha256": M110_M006_SOURCE_EVENT_SHA256,
    }


def m110_m006_restore_bundle_selector(stage: str) -> str:
    if stage not in M110_M006_RESTORE_STAGES:
        raise ValueError("m110_m006_restore_stage_invalid")
    return M110_M006_RESTORE_BUNDLE_SELECTOR_PREFIX + stage


def parse_m110_m006_restore_bundle_selector(value: Any) -> str | None:
    if not isinstance(value, str) or not value.startswith(
        M110_M006_RESTORE_BUNDLE_SELECTOR_PREFIX
    ):
        return None
    stage = value[len(M110_M006_RESTORE_BUNDLE_SELECTOR_PREFIX) :]
    return stage if stage in M110_M006_RESTORE_STAGES else None
