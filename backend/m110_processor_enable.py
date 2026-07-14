"""Server-owned complete processor blocks for the attended M110 lane.

The normal supervised adapters deliberately refuse processing enable changes.
This module converts trusted soundcheck children into one bounded synthetic
operation whose full audible parameter surface is written while bypassed and
whose enable address is always the final write.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


M110_PROCESSOR_BUNDLE_KIND = "processor_enable_bundle"
M110_PROCESSOR_ADAPTER_REVISION = "m110-processor-enable.v1"
M110_PROCESSOR_PROTOCOL_REVISION = "wing-osc-fw3.0.5"

_PROCESSOR_WRITE_KINDS = {"hpf", "eq", "gate", "compressor"}
_EQ_DEFAULTS = {
    1: {"band": 1, "freq_hz": 120.0, "gain_db": 0.0, "q": 1.0},
    2: {"band": 2, "freq_hz": 400.0, "gain_db": 0.0, "q": 1.0},
    3: {"band": 3, "freq_hz": 1600.0, "gain_db": 0.0, "q": 1.0},
    4: {"band": 4, "freq_hz": 6400.0, "gain_db": 0.0, "q": 1.0},
}


@dataclass(frozen=True, slots=True)
class M110ProcessorCandidate:
    """One immutable channel/processor block derived from a trusted manifest."""

    candidate_id: str
    run_id: str
    plan_revision: int
    plan_fingerprint: str
    processor_kind: str
    channel: int
    stage: str
    stage_order: int
    required_for: str
    source_operation_ids: tuple[str, ...]
    source_transaction_ids: tuple[str, ...]
    source_plans: tuple[Mapping[str, Any], ...]
    synthetic_plan: Mapping[str, Any]


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _source_operation(plan: Mapping[str, Any]) -> Mapping[str, Any] | None:
    operations = plan.get("operations")
    if not isinstance(operations, list) or len(operations) != 1:
        return None
    operation = operations[0]
    return operation if isinstance(operation, Mapping) else None


def _processor_kind(operation: Mapping[str, Any]) -> str | None:
    family_id = str(operation.get("family_id") or "")
    write_kind = str(operation.get("write_kind") or "")
    if write_kind not in _PROCESSOR_WRITE_KINDS:
        return None
    expected_family = {
        "hpf": "eq_hpf",
        "eq": "eq_hpf",
        "gate": "gate_expander",
        "compressor": "dynamics_compressor",
    }[write_kind]
    if family_id != expected_family:
        return None
    target = operation.get("target_value")
    if not isinstance(target, Mapping):
        return None
    # Channel EQ recommendations intentionally omit an enable flag: the
    # ordinary adapter treats EQ enable as retained audible state.  M110 starts
    # from a reset/bypassed console, so every channel-EQ child must instead be
    # absorbed into the complete block that enables EQ last.  The other three
    # processor actions carry an explicit enable intent and only belong here
    # when that intent is True.
    if write_kind != "eq" and target.get("enabled") is not True:
        return None
    return write_kind


def _outcome_committed(
    outcomes: Mapping[str, Mapping[str, Any]],
    operation_id: str,
) -> bool:
    outcome = outcomes.get(operation_id, {})
    return bool(
        isinstance(outcome, Mapping)
        and outcome.get("status") == "applied"
        and outcome.get("physical_verified") is True
        and outcome.get("audio_recheck_passed") is True
        and outcome.get("wal_terminal") == "committed"
    )


def _outcome_selection_terminal(
    outcomes: Mapping[str, Mapping[str, Any]],
    operation_id: str,
) -> bool:
    outcome = outcomes.get(operation_id, {})
    return bool(
        _outcome_committed(outcomes, operation_id)
        or (
            isinstance(outcome, Mapping)
            and outcome.get("disposition") == "already_satisfied"
            and outcome.get("physical_verified") is True
            and outcome.get("audible_state_verified") is True
            and outcome.get("final_reconciliation_verified") is True
        )
        or (
            isinstance(outcome, Mapping)
            and outcome.get("m110_terminal_local_residual") is True
        )
    )


def _processor_target(
    processor_kind: str,
    operations: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if processor_kind == "eq":
        bands = {band: dict(values) for band, values in _EQ_DEFAULTS.items()}
        for operation in operations:
            target = operation.get("target_value")
            if not isinstance(target, Mapping):
                raise ValueError("m110_eq_source_target_invalid")
            band = target.get("band")
            if isinstance(band, bool) or not isinstance(band, int) or band not in bands:
                raise ValueError("m110_eq_source_band_invalid")
            if bands[band].get("_planned") is True:
                raise ValueError("m110_eq_source_band_conflict")
            bands[band] = {
                "band": band,
                "freq_hz": target.get("freq_hz"),
                "gain_db": target.get("gain_db"),
                "q": target.get("q"),
                "_planned": True,
            }
        canonical_bands = []
        for band in range(1, 5):
            values = dict(bands[band])
            values.pop("_planned", None)
            canonical_bands.append(values)
        return {
            "m110_processor_block": "eq",
            "model": "STD",
            "mix_pct": 100.0,
            "low_shelf": {
                "freq_hz": 80.0,
                "gain_db": 0.0,
                "q": 1.0,
                "type": "PEQ",
            },
            "bands": canonical_bands,
            "high_shelf": {
                "freq_hz": 12000.0,
                "gain_db": 0.0,
                "q": 1.0,
                "type": "PEQ",
            },
            "enabled": True,
        }
    if len(operations) != 1:
        raise ValueError("m110_non_eq_processor_must_be_singular")
    source = operations[0].get("target_value")
    if not isinstance(source, Mapping):
        raise ValueError("m110_processor_source_target_invalid")
    if processor_kind == "hpf":
        return {
            "m110_processor_block": "hpf",
            "freq_hz": source.get("freq_hz"),
            # WING fw3.0.5 exposes /flt/lcs as an enum string, not a float.
            "slope_db_per_octave": "12",
            "enabled": True,
        }
    if processor_kind == "gate":
        return {
            "m110_processor_block": "gate",
            "model": "GATE",
            "mix_pct": 100.0,
            "gain_db": 0.0,
            "threshold_db": source.get("threshold_db"),
            "range_db": source.get("range_db"),
            "attack_ms": source.get("attack_ms"),
            "hold_ms": source.get("hold_ms"),
            "release_ms": source.get("release_ms"),
            "accent_pct": 0.0,
            "ratio": "GATE",
            "sidechain": {
                "type": "OFF",
                "freq_hz": 1000.0,
                "q": 1.0,
                "source": "SELF",
                "tap": "IN",
            },
            "enabled": True,
        }
    if processor_kind == "compressor":
        return {
            "m110_processor_block": "compressor",
            "model": "COMP",
            "mix_pct": 100.0,
            "threshold_db": source.get("threshold_db"),
            "ratio": source.get("ratio"),
            "knee": 0,
            "detector": "RMS",
            "attack_ms": source.get("attack_ms"),
            "hold_ms": 1.0,
            "release_ms": source.get("release_ms"),
            "envelope": "LOG",
            "auto": False,
            "makeup_db": source.get("makeup_db", 0.0),
            "crossover": {
                "depth_db": 0.0,
                "type": "OFF",
                "freq_hz": 1000.0,
                "q": 1.0,
            },
            "sidechain": {
                "type": "OFF",
                "freq_hz": 1000.0,
                "q": 1.0,
                "source": "SELF",
                "tap": "IN",
            },
            "enabled": True,
        }
    raise ValueError("m110_processor_kind_invalid")


def collect_m110_processor_candidates(
    manifest: Mapping[str, Any],
    *,
    outcomes: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[M110ProcessorCandidate, ...]:
    """Return deterministic unresolved channel processor blocks.

    Only trusted manifest data is accepted by callers. This function never
    accepts an OSC address or target from the attended runner.
    """

    if not isinstance(manifest, Mapping):
        raise ValueError("m110_soundcheck_manifest_required")
    run_id = str(manifest.get("run_id") or "").strip()
    revision = manifest.get("plan_revision")
    fingerprint = str(manifest.get("plan_fingerprint") or "").strip()
    if (
        not run_id
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
        or not fingerprint
    ):
        raise ValueError("m110_soundcheck_manifest_identity_invalid")
    current_outcomes = outcomes if isinstance(outcomes, Mapping) else {}
    grouped: dict[tuple[int, str], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for raw_plan in manifest.get("plans") or []:
        if not isinstance(raw_plan, Mapping):
            raise ValueError("m110_soundcheck_child_plan_invalid")
        operation = _source_operation(raw_plan)
        if operation is None:
            continue
        kind = _processor_kind(operation)
        if kind is None:
            continue
        channel = operation.get("channel")
        operation_id = str(operation.get("operation_id") or "").strip()
        if (
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or not 1 <= channel <= 16
            or not operation_id
        ):
            raise ValueError("m110_processor_source_identity_invalid")
        key = (channel, kind)
        if kind != "eq" and key in grouped:
            raise ValueError("m110_processor_source_duplicate")
        grouped.setdefault(key, []).append((raw_plan, operation))

    candidates: list[M110ProcessorCandidate] = []
    for (channel, kind), members in grouped.items():
        # Keep the whole planned EQ surface whenever any member is unresolved.
        # Rebuilding an EQ block from only the unresolved bands would
        # accidentally neutralize a band committed earlier in the same run.
        if all(
            _outcome_selection_terminal(
                current_outcomes,
                str(operation.get("operation_id") or ""),
            )
            for _plan, operation in members
        ) or any(
            isinstance(
                current_outcomes.get(
                    str(operation.get("operation_id") or ""),
                    {},
                ),
                Mapping,
            )
            and current_outcomes.get(
                str(operation.get("operation_id") or ""),
                {},
            ).get("m110_terminal_local_residual")
            is True
            for _plan, operation in members
        ):
            continue
        members.sort(key=lambda item: str(item[1].get("operation_id") or ""))
        plans = tuple(copy.deepcopy(item[0]) for item in members)
        operations = tuple(copy.deepcopy(item[1]) for item in members)
        source_operation_ids = tuple(
            str(operation.get("operation_id") or "") for operation in operations
        )
        source_transaction_ids = tuple(
            str(plan.get("transaction_id") or "") for plan in plans
        )
        if any(not value for value in source_transaction_ids):
            raise ValueError("m110_processor_source_transaction_invalid")
        stage_orders = []
        stages = []
        required_for_values = []
        for plan, operation in zip(plans, operations):
            metadata = operation.get("metadata")
            metadata = metadata if isinstance(metadata, Mapping) else {}
            order = plan.get("stage_order", metadata.get("stage_order"))
            if isinstance(order, bool) or not isinstance(order, int):
                raise ValueError("m110_processor_stage_order_invalid")
            stage_orders.append(order)
            stages.append(str(plan.get("stage") or metadata.get("stage") or ""))
            required_for_values.append(
                str(metadata.get("required_for") or "show_mix")
            )
        stage_order = min(stage_orders)
        stage = next((value for value in stages if value), "tonal_dynamics")
        required_for = (
            "safe_core"
            if "safe_core" in required_for_values
            else "show_mix"
        )
        target = _processor_target(kind, operations)
        identity = {
            "run_id": run_id,
            "plan_revision": revision,
            "plan_fingerprint": fingerprint,
            "processor_kind": kind,
            "channel": channel,
            "source_operation_ids": source_operation_ids,
            "target": target,
        }
        identity_digest = _canonical_digest(identity)
        candidate_id = f"m110-processor-{kind}-ch{channel}-{identity_digest[:16]}"
        operation_id = f"m110:processor:{kind}:ch{channel}:{identity_digest[:16]}"
        synthetic_operation = {
            "operation_id": operation_id,
            "family_id": {
                "hpf": "eq_hpf",
                "eq": "eq_hpf",
                "gate": "gate_expander",
                "compressor": "dynamics_compressor",
            }[kind],
            "write_kind": kind,
            "channel": channel,
            "parameter": f"m110_complete_{kind}_block",
            "target_value": target,
            "metadata": {
                "source": "m110_server_owned_processor_enable",
                "m110_processor_enable_bundle": True,
                "m110_processor_kind": kind,
                "source_operation_ids": list(source_operation_ids),
                "stage": stage,
                "stage_order": stage_order,
                "required_for": required_for,
                "musically_required": True,
                "absolute_target": True,
                "live_mixer_writes": False,
                "transport_policy": "m110_capability_only",
            },
        }
        synthetic_plan = {
            "transaction_id": f"m110:processor:{identity_digest[:24]}",
            "reason": (
                f"M110 complete {kind} block for channel {channel}; enable last"
            ),
            "run_id": run_id,
            "plan_revision": revision,
            "plan_fingerprint": fingerprint,
            "stage": stage,
            "stage_order": stage_order,
            "logical_correction_ids": list(source_operation_ids),
            "operations": [synthetic_operation],
        }
        candidates.append(
            M110ProcessorCandidate(
                candidate_id=candidate_id,
                run_id=run_id,
                plan_revision=revision,
                plan_fingerprint=fingerprint,
                processor_kind=kind,
                channel=channel,
                stage=stage,
                stage_order=stage_order,
                required_for=required_for,
                source_operation_ids=source_operation_ids,
                source_transaction_ids=source_transaction_ids,
                source_plans=plans,
                synthetic_plan=synthetic_plan,
            )
        )
    candidates.sort(
        key=lambda candidate: (
            candidate.stage_order,
            candidate.channel,
            {"hpf": 0, "gate": 1, "eq": 2, "compressor": 3}[
                candidate.processor_kind
            ],
            candidate.candidate_id,
        )
    )
    return tuple(candidates)


def is_m110_processor_source_operation(operation: Mapping[str, Any]) -> bool:
    return bool(isinstance(operation, Mapping) and _processor_kind(operation))
