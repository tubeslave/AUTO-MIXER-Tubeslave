"""Build proposal-only supervised transaction plans from soundcheck actions.

This module is deliberately pure: it does not import a mixer client, register
console adapters, or send transport messages.  Its only job is to translate
typed AutoFOH decisions into deterministic absolute operations that can later
be reviewed and executed by the supervised transaction coordinator.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from autofoh_safety import (
    BusCompressorAdjust,
    BusCompressorMakeupAdjust,
    BusEQMove,
    BusFaderMove,
    ChannelEQMove,
    ChannelFaderMove,
    ChannelGainMove,
    ChannelMuteMove,
    CompressorAdjust,
    CompressorMakeupAdjust,
    DCAFaderMove,
    DelayAdjust,
    EmergencyFeedbackNotch,
    GateAdjust,
    HighPassAdjust,
    MasterFaderMove,
    NoOp,
    PanAdjust,
    PolarityAdjust,
    SendLevelAdjust,
    TypedCorrectionAction,
)
from compressor_adaptation import ratio_float_to_wing
from supervised_console_parameter_registry import (
    get_supervised_console_family,
)
from supervised_console_transaction import (
    ConsoleTransactionOperation,
    ConsoleTransactionPlan,
)


FAMILY_ORDER = {
    "gain_trim": 10,
    "eq_hpf": 20,
    "gate_expander": 30,
    "dynamics_compressor": 40,
    "pan_phase_delay": 50,
    "fader": 60,
    "fx_sends_inserts": 70,
    "bus_group_dca_matrix": 80,
    "master_output": 90,
    "mute": 100,
    "routing": 110,
    "scene_preset": 120,
}

MANIFEST_SCHEMA_VERSION = "soundcheck_batch_manifest/v2"
STAGE_ORDER = {
    "foundation_level": 10,
    "source_control": 20,
    "phase_alignment_review": 30,
    "tonal_dynamics": 40,
    "group_balance": 50,
    "spatial_fx": 60,
    "destructive_high_risk": 70,
}


def _stage_for_operation(operation: ConsoleTransactionOperation) -> str:
    if operation.write_kind in {"gain", "trim", "hpf", "fader"}:
        return "foundation_level"
    if operation.write_kind in {"gate", "expander"}:
        return "source_control"
    if operation.write_kind in {"polarity", "delay"}:
        return "phase_alignment_review"
    if operation.write_kind in {"eq", "compressor", "dynamics", "deesser"}:
        return "tonal_dynamics"
    if operation.write_kind in {"bus", "group", "dca", "matrix"}:
        return "group_balance"
    if operation.write_kind in {"pan", "fx", "insert", "send", "return"}:
        return "spatial_fx"
    return "destructive_high_risk"


def _required_for_stage(stage: str) -> str:
    if stage in {
        "foundation_level",
        "source_control",
        "phase_alignment_review",
    }:
        return "safe_core"
    return "show_mix"


def _action_identity(action: TypedCorrectionAction) -> str:
    return json.dumps(
        {
            "action_type": action.action_type,
            "payload": asdict(action),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _logical_correction_id(action: TypedCorrectionAction) -> str:
    digest = hashlib.sha256(_action_identity(action).encode("utf-8")).hexdigest()[:16]
    return f"soundcheck:{action.action_type.lower()}:{digest}"


def soundcheck_manifest_content_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash immutable musical intent, excluding refreshable preflight data."""

    operation_fields = (
        "operation_id",
        "family_id",
        "write_kind",
        "target_value",
        "channel",
        "parameter",
        "metadata",
    )
    plan_fields = (
        "transaction_id",
        "reason",
        "stage",
        "stage_order",
        "required_for",
        "logical_correction_ids",
        "run_id",
        "plan_revision",
        "separate_approval_required",
        "coupled_safe_alternative_group",
        "atomic_group_commit_required",
        "approved",
        "execution_allowed",
        "live_mixer_writes",
        "transport_policy",
        "value_semantics",
    )
    canonical_plans = []
    for raw_plan in manifest.get("plans") or []:
        if not isinstance(raw_plan, Mapping):
            canonical_plans.append(raw_plan)
            continue
        canonical_plan = {
            key: raw_plan.get(key)
            for key in plan_fields
            if key in raw_plan
        }
        canonical_plan["operations"] = [
            (
                {
                    key: raw_operation.get(key)
                    for key in operation_fields
                    if key in raw_operation
                }
                if isinstance(raw_operation, Mapping)
                else raw_operation
            )
            for raw_operation in raw_plan.get("operations") or []
        ]
        canonical_plans.append(canonical_plan)
    payload = {
        "schema_version": manifest.get("schema_version"),
        "run_id": manifest.get("run_id"),
        "plan_revision": manifest.get("plan_revision"),
        "plans": canonical_plans,
        "unsupported_actions": manifest.get("unsupported_actions") or [],
        "readiness_residuals": manifest.get("readiness_residuals") or [],
        "source_coverage": manifest.get("source_coverage") or {},
        "last_master_mix_verdict": manifest.get("last_master_mix_verdict") or {},
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def bind_soundcheck_manifest_content_sha256(
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach one canonical digest to the parent and every child view."""

    digest = soundcheck_manifest_content_sha256(manifest)
    manifest["manifest_content_sha256"] = digest
    for collection_name in (
        "plans",
        "normal_plans",
        "separate_approval_plans",
    ):
        for plan in manifest.get(collection_name) or []:
            if isinstance(plan, dict):
                plan["manifest_content_sha256"] = digest
    return manifest


@dataclass(frozen=True)
class SoundcheckTransactionPlanBundle:
    """Reviewed-plan surface with explicit live-authority boundaries."""

    normal_plans: Tuple[ConsoleTransactionPlan, ...]
    separate_approval_plans: Tuple[ConsoleTransactionPlan, ...]
    unsupported_actions: Tuple[Mapping[str, Any], ...]
    live_mixer_writes: bool = False
    execution_allowed: bool = False
    transport_policy: str = "proposal_only"
    value_semantics: str = "absolute_target"
    run_id: str = ""
    plan_revision: int = 1
    plan_fingerprint: str = ""
    schema_version: str = MANIFEST_SCHEMA_VERSION

    @property
    def plans(self) -> Tuple[ConsoleTransactionPlan, ...]:
        return tuple(
            sorted(
                self.normal_plans + self.separate_approval_plans,
                key=lambda plan: _operation_sort_key(plan.operations[0]),
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        def operation_payload(operation: ConsoleTransactionOperation) -> Dict[str, Any]:
            payload = asdict(operation)
            payload["metadata"] = dict(operation.metadata)
            contract = get_supervised_console_family(operation.family_id)
            payload["family_contract"] = contract.to_dict() if contract else None
            return payload

        def plan_payload(
            plan: ConsoleTransactionPlan,
            *,
            separate_approval: bool,
        ) -> Dict[str, Any]:
            operation = plan.operations[0]
            metadata = dict(operation.metadata)
            return {
                "transaction_id": plan.transaction_id,
                "reason": plan.reason,
                "operations": [operation_payload(item) for item in plan.operations],
                "stage": metadata.get("stage"),
                "stage_order": metadata.get("stage_order"),
                "required_for": metadata.get("required_for"),
                "logical_correction_ids": [
                    str(item.metadata.get("logical_correction_id") or item.operation_id)
                    for item in plan.operations
                ],
                "run_id": self.run_id,
                "plan_revision": self.plan_revision,
                "plan_fingerprint": self.plan_fingerprint,
                "separate_approval_required": separate_approval,
                "approved": False,
                "execution_allowed": False,
                "live_mixer_writes": False,
                "transport_policy": "proposal_only",
                "value_semantics": "absolute_target",
            }

        normal = [
            plan_payload(plan, separate_approval=False) for plan in self.normal_plans
        ]
        separate = [
            plan_payload(plan, separate_approval=True)
            for plan in self.separate_approval_plans
        ]
        separate_transaction_ids = {
            plan.transaction_id for plan in self.separate_approval_plans
        }
        # Preserve the canonical semantic operation order.  Sorting child
        # transaction hashes here used to randomize same-stage dependencies
        # such as trim -> fader and EQ -> compressor.
        all_plans = [
            plan_payload(
                plan,
                separate_approval=(
                    plan.transaction_id in separate_transaction_ids
                ),
            )
            for plan in self.plans
        ]
        stage_counts: Dict[str, Dict[str, int]] = {
            stage: {"plan_count": 0, "operation_count": 0}
            for stage in STAGE_ORDER
        }
        for plan in self.plans:
            stage = str(plan.operations[0].metadata.get("stage") or "destructive_high_risk")
            stage_counts.setdefault(stage, {"plan_count": 0, "operation_count": 0})
            stage_counts[stage]["plan_count"] += 1
            stage_counts[stage]["operation_count"] += len(plan.operations)
        payload = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "batch_id": f"{self.run_id}:r{self.plan_revision}",
            "plan_revision": self.plan_revision,
            "plan_fingerprint": self.plan_fingerprint,
            "plans": all_plans,
            "normal_plans": normal,
            "separate_approval_plans": separate,
            "unsupported_actions": [dict(item) for item in self.unsupported_actions],
            "plan_count": len(normal) + len(separate),
            "operation_count": sum(
                len(plan.operations) for plan in self.plans
            ),
            "logical_correction_count": (
                sum(
                    max(
                        1,
                        len(
                            operation.metadata.get(
                                "merged_logical_correction_ids",
                                (),
                            )
                        ),
                    )
                    for plan in self.plans
                    for operation in plan.operations
                )
                + len(self.unsupported_actions)
            ),
            "stage_order": dict(STAGE_ORDER),
            "stages": stage_counts,
            "rollback_boundary": "semantic_child_transaction",
            "stage_atomicity": False,
            "live_mixer_writes": self.live_mixer_writes,
            "execution_allowed": self.execution_allowed,
            "transport_policy": self.transport_policy,
            "value_semantics": self.value_semantics,
        }
        return bind_soundcheck_manifest_content_sha256(payload)


def _semantic_target(action: TypedCorrectionAction) -> tuple[str, str, int | None, str, Any]:
    if isinstance(action, ChannelGainMove):
        return "gain_trim", "gain", action.channel_id, "input_gain_db", action.target_db
    if isinstance(action, ChannelEQMove):
        if float(action.gain_db) > 0.0:
            raise ValueError("live_channel_eq_boost_is_not_allowed")
        return "eq_hpf", "eq", action.channel_id, f"eq_band_{action.band}", {
            "band": action.band,
            "freq_hz": action.freq_hz,
            "gain_db": action.gain_db,
            "q": action.q,
        }
    if isinstance(action, BusEQMove):
        if float(action.gain_db) > 0.0:
            raise ValueError("live_bus_eq_boost_is_not_allowed")
        return "bus_group_dca_matrix", "bus", None, f"bus_{action.bus_id}_eq_band_{action.band}", {
            "bus_id": action.bus_id,
            "band": action.band,
            "freq_hz": action.freq_hz,
            "gain_db": action.gain_db,
            "q": action.q,
        }
    if isinstance(action, HighPassAdjust):
        return "eq_hpf", "hpf", action.channel_id, "high_pass_filter", {
            "freq_hz": action.freq_hz,
            "enabled": action.enabled,
        }
    if isinstance(action, GateAdjust):
        return "gate_expander", "gate", action.channel_id, "gate", {
            "threshold_db": action.threshold_db,
            "range_db": action.range_db,
            "attack_ms": action.attack_ms,
            "hold_ms": action.hold_ms,
            "release_ms": action.release_ms,
            "enabled": action.enabled,
        }
    if isinstance(action, CompressorAdjust):
        return "dynamics_compressor", "compressor", action.channel_id, "compressor", {
            "threshold_db": action.threshold_db,
            # WING exposes a discrete ratio enum.  Bind the reviewed absolute
            # target to the value that can actually be written and read back;
            # otherwise a proposal for (for example) 7:1 could only commit 6:1.
            "ratio": float(ratio_float_to_wing(float(action.ratio))),
            "attack_ms": action.attack_ms,
            "release_ms": action.release_ms,
            "makeup_db": action.makeup_db,
            "enabled": action.enabled,
        }
    if isinstance(action, BusCompressorAdjust):
        return "bus_group_dca_matrix", "bus", None, f"bus_{action.bus_id}_compressor", {
            "bus_id": action.bus_id,
            "threshold_db": action.threshold_db,
            "ratio": action.ratio,
            "attack_ms": action.attack_ms,
            "release_ms": action.release_ms,
            "makeup_db": action.makeup_db,
            "enabled": action.enabled,
        }
    if isinstance(action, CompressorMakeupAdjust):
        return "dynamics_compressor", "compressor", action.channel_id, "compressor_makeup_db", action.makeup_db
    if isinstance(action, BusCompressorMakeupAdjust):
        return "bus_group_dca_matrix", "bus", None, f"bus_{action.bus_id}_compressor_makeup_db", {
            "bus_id": action.bus_id,
            "makeup_db": action.makeup_db,
        }
    if isinstance(action, PolarityAdjust):
        return "pan_phase_delay", "polarity", action.channel_id, "polarity_inverted", action.inverted
    if isinstance(action, DelayAdjust):
        return "pan_phase_delay", "delay", action.channel_id, "input_delay", {
            "delay_ms": action.delay_ms,
            "enabled": action.enabled,
        }
    if isinstance(action, PanAdjust):
        return "pan_phase_delay", "pan", action.channel_id, "pan", action.pan
    if isinstance(action, ChannelFaderMove):
        return "fader", "fader", action.channel_id, "channel_fader_db", action.target_db
    if isinstance(action, BusFaderMove):
        return "bus_group_dca_matrix", "bus", None, f"bus_{action.bus_id}_fader_db", {
            "bus_id": action.bus_id,
            "target_db": action.target_db,
        }
    if isinstance(action, DCAFaderMove):
        if float(action.target_db) <= -120.0:
            raise ValueError("inactive_dca_target_is_negative_infinity")
        return "bus_group_dca_matrix", "dca", None, f"dca_{action.dca_id}_fader_db", {
            "dca_id": action.dca_id,
            "target_db": action.target_db,
        }
    if isinstance(action, SendLevelAdjust):
        return "fx_sends_inserts", "send", action.channel_id, f"send_bus_{action.send_bus}_level_db", {
            "send_bus": action.send_bus,
            "level_db": action.level_db,
        }
    if isinstance(action, MasterFaderMove):
        return "master_output", "master", None, f"main_{action.main_id}_fader_db", {
            "main_id": action.main_id,
            "target_db": action.target_db,
        }
    if isinstance(action, ChannelMuteMove):
        return "mute", "mute", action.channel_id, "channel_muted", action.muted
    if isinstance(action, EmergencyFeedbackNotch):
        if float(action.gain_db) > 0.0:
            raise ValueError("feedback_notch_boost_is_not_allowed")
        return "eq_hpf", "eq", action.channel_id, f"emergency_notch_band_{action.band}", {
            "band": action.band,
            "freq_hz": action.freq_hz,
            "gain_db": action.gain_db,
            "q": action.q,
            "ttl_seconds": action.ttl_seconds,
            "emergency": True,
        }
    raise TypeError(f"unsupported soundcheck action: {action.__class__.__name__}")


def operation_from_soundcheck_action(
    action: TypedCorrectionAction,
) -> ConsoleTransactionOperation:
    """Translate one typed action into one atomic semantic operation."""
    family_id, write_kind, channel, parameter, target_value = _semantic_target(action)
    operation_id = _logical_correction_id(action)
    provisional = ConsoleTransactionOperation(
        operation_id=operation_id,
        family_id=family_id,
        write_kind=write_kind,
        target_value=target_value,
        channel=channel,
        parameter=parameter,
    )
    stage = _stage_for_operation(provisional)
    audible_state_requirements: tuple[str, ...] = ()
    if write_kind == "eq" or (
        family_id == "bus_group_dca_matrix" and "_eq_band_" in parameter
    ):
        audible_state_requirements = ("eq_enabled_physical_readback",)
    elif (
        family_id == "dynamics_compressor"
        and write_kind == "compressor"
        and not isinstance(target_value, Mapping)
    ) or (
        family_id == "bus_group_dca_matrix"
        and parameter.endswith("_compressor_makeup_db")
    ):
        audible_state_requirements = (
            "compressor_enabled_physical_readback",
        )
    return ConsoleTransactionOperation(
        operation_id=operation_id,
        family_id=family_id,
        write_kind=write_kind,
        target_value=target_value,
        channel=channel,
        parameter=parameter,
        metadata={
            "source": "auto_soundcheck_engine",
            "action_type": action.action_type,
            "reason": action.reason,
            "logical_correction_id": operation_id,
            "stage": stage,
            "stage_order": STAGE_ORDER[stage],
            "required_for": _required_for_stage(stage),
            "musically_required": True,
            "assessment_scope": "operation",
            "preflight_disposition": "pending_physical_preflight",
            "audible_state_requirements": list(audible_state_requirements),
            "absolute_target": True,
            "live_mixer_writes": False,
            "transport_policy": "proposal_only",
        },
    )


def _operation_sort_key(operation: ConsoleTransactionOperation) -> tuple[Any, ...]:
    return (
        int(operation.metadata.get("stage_order") or 999),
        FAMILY_ORDER.get(operation.family_id, 999),
        operation.family_id,
        operation.channel if operation.channel is not None else 10_000,
        operation.parameter,
        operation.operation_id,
    )


def _semantic_conflict_key(operation: ConsoleTransactionOperation) -> tuple[Any, ...]:
    parameter = operation.parameter
    if operation.family_id == "eq_hpf" and operation.write_kind == "eq":
        target = operation.target_value
        band = target.get("band") if isinstance(target, Mapping) else None
        parameter = f"eq_band_{band}"
    elif operation.family_id == "dynamics_compressor":
        parameter = "compressor_chain"
    elif operation.family_id == "bus_group_dca_matrix" and "_compressor" in parameter:
        parameter = parameter.replace("_compressor_makeup_db", "_compressor")
    return (
        operation.family_id,
        operation.write_kind,
        operation.channel,
        parameter,
    )


def _semantic_target_signature(operation: ConsoleTransactionOperation) -> str:
    """Canonical equality surface for independent recommendations."""

    return json.dumps(
        operation.target_value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _merge_agreeing_operations(
    operations: Sequence[ConsoleTransactionOperation],
) -> ConsoleTransactionOperation:
    """Collapse identical semantic targets while preserving their provenance."""

    ordered = sorted(operations, key=lambda item: item.operation_id)
    canonical = ordered[0]
    metadata = dict(canonical.metadata)
    metadata["merged_logical_correction_ids"] = [
        operation.operation_id for operation in ordered
    ]
    metadata["merged_reasons"] = sorted(
        {
            str(operation.metadata.get("reason") or "").strip()
            for operation in ordered
            if str(operation.metadata.get("reason") or "").strip()
        }
    )
    metadata["merged_recommendation_count"] = len(ordered)
    return ConsoleTransactionOperation(
        operation_id=canonical.operation_id,
        family_id=canonical.family_id,
        write_kind=canonical.write_kind,
        target_value=canonical.target_value,
        channel=canonical.channel,
        parameter=canonical.parameter,
        metadata=metadata,
    )


def _transaction_id(
    prefix: str,
    operations: Sequence[ConsoleTransactionOperation],
    *,
    run_id: str,
    plan_revision: int,
) -> str:
    identity = "|".join(operation.operation_id for operation in operations)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    run_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:10]
    return f"soundcheck:{prefix}:r{plan_revision}:{run_digest}:{digest}"


def _manifest_fingerprint(
    operations: Sequence[ConsoleTransactionOperation],
    unsupported: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "operation_ids": [operation.operation_id for operation in operations],
        "unsupported": [
            {
                "logical_correction_id": item.get("logical_correction_id"),
                "reason": item.get("reason"),
                "status": item.get("status"),
            }
            for item in unsupported
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_soundcheck_transaction_plans(
    actions: Iterable[TypedCorrectionAction],
    *,
    reason: str = "AutoSoundcheck reviewed correction plan",
    run_id: str | None = None,
    plan_revision: int = 1,
) -> SoundcheckTransactionPlanBundle:
    """Build a parent manifest with one atomic child per semantic correction."""
    if (
        isinstance(plan_revision, bool)
        or not isinstance(plan_revision, int)
        or plan_revision < 1
    ):
        raise ValueError("plan_revision_must_be_positive_integer")
    normal_operations = []
    separate_operations = []
    unsupported = []

    for action in actions:
        if isinstance(action, NoOp):
            logical_id = _logical_correction_id(action)
            unsupported.append(
                {
                    "logical_correction_id": logical_id,
                    "action_type": action.action_type,
                    "reason": action.reason,
                    "message": action.message,
                    "status": "not_actionable",
                    "scope": "operation",
                    "disposition": "operator_review",
                    "stage": "foundation_level",
                    "required_for": "safe_core",
                    "musically_required": True,
                }
            )
            continue
        try:
            operation = operation_from_soundcheck_action(action)
        except (TypeError, ValueError) as exc:
            logical_id = _logical_correction_id(action)
            unsupported.append(
                {
                    "logical_correction_id": logical_id,
                    "action_type": getattr(action, "action_type", action.__class__.__name__),
                    "channel": getattr(action, "channel_id", None),
                    "reason": str(exc),
                    "status": "unsupported",
                    "scope": "operation",
                    "disposition": "operator_review",
                    "stage": "tonal_dynamics",
                    "required_for": "show_mix",
                    "musically_required": True,
                }
            )
            continue
        contract = get_supervised_console_family(operation.family_id)
        if contract is None:
            unsupported.append(
                {
                    "logical_correction_id": operation.operation_id,
                    "action_type": action.action_type,
                    "channel": operation.channel,
                    "reason": "family_contract_missing",
                    "status": "unsupported",
                    "scope": "operation",
                    "disposition": "forbidden",
                    "stage": operation.metadata.get("stage"),
                    "required_for": operation.metadata.get("required_for"),
                    "musically_required": True,
                }
            )
            continue
        if contract.separate_approval:
            separate_operations.append(operation)
        else:
            normal_operations.append(operation)

    def deduplicate(operations: Sequence[ConsoleTransactionOperation]):
        unique: Dict[str, ConsoleTransactionOperation] = {}
        for operation in operations:
            unique.setdefault(operation.operation_id, operation)
        return list(unique.values())

    normal_operations = deduplicate(normal_operations)
    separate_operations = deduplicate(separate_operations)
    grouped_targets: Dict[tuple[Any, ...], list[ConsoleTransactionOperation]] = {}
    for operation in normal_operations + separate_operations:
        grouped_targets.setdefault(_semantic_conflict_key(operation), []).append(operation)
    resolved_operations: list[ConsoleTransactionOperation] = []
    for semantic_group in grouped_targets.values():
        target_groups: Dict[str, list[ConsoleTransactionOperation]] = {}
        for operation in semantic_group:
            target_groups.setdefault(
                _semantic_target_signature(operation),
                [],
            ).append(operation)
        if len(target_groups) == 1:
            resolved_operations.append(
                _merge_agreeing_operations(semantic_group)
                if len(semantic_group) > 1
                else semantic_group[0]
            )
            continue
        for operation in semantic_group:
            unsupported.append(
                {
                    "logical_correction_id": operation.operation_id,
                    "action_type": operation.metadata.get("action_type"),
                    "channel": operation.channel,
                    "reason": "conflicting_semantic_target",
                    "status": "unsupported",
                    "scope": "operation",
                    "disposition": "operator_review",
                    "stage": operation.metadata.get("stage"),
                    "required_for": operation.metadata.get("required_for"),
                    "musically_required": True,
                }
            )

    normal_operations = []
    separate_operations = []
    for operation in resolved_operations:
        contract = get_supervised_console_family(operation.family_id)
        if contract is not None and contract.separate_approval:
            separate_operations.append(operation)
        else:
            normal_operations.append(operation)

    normal_operations.sort(key=_operation_sort_key)
    separate_operations.sort(key=_operation_sort_key)
    unsupported_by_id: Dict[str, Mapping[str, Any]] = {}
    for item in unsupported:
        logical_id = str(item.get("logical_correction_id") or "")
        unsupported_by_id.setdefault(logical_id, item)
    unsupported = list(unsupported_by_id.values())
    unsupported.sort(key=lambda item: str(item.get("logical_correction_id") or ""))
    all_operations = tuple(
        sorted(normal_operations + separate_operations, key=_operation_sort_key)
    )
    logical_ids = [operation.operation_id for operation in all_operations] + [
        str(item.get("logical_correction_id") or "") for item in unsupported
    ]
    if any(not item for item in logical_ids):
        raise ValueError("logical_correction_id_required")
    plan_fingerprint = _manifest_fingerprint(all_operations, unsupported)
    resolved_run_id = str(run_id or f"derived-{plan_fingerprint[:16]}").strip()
    if not resolved_run_id:
        raise ValueError("run_id_required")

    normal_plans = tuple(
        ConsoleTransactionPlan(
                transaction_id=_transaction_id(
                    str(operation.metadata.get("stage") or "normal"),
                    (operation,),
                    run_id=resolved_run_id,
                    plan_revision=plan_revision,
                ),
                operations=(operation,),
                reason=(
                    f"{reason}; stage={operation.metadata.get('stage')}; "
                    f"logical_correction={operation.operation_id}"
                ),
            )
        for operation in normal_operations
    )

    separate_plans = tuple(
        ConsoleTransactionPlan(
            transaction_id=_transaction_id(
                str(operation.metadata.get("stage") or operation.family_id),
                (operation,),
                run_id=resolved_run_id,
                plan_revision=plan_revision,
            ),
            operations=(operation,),
            reason=(
                f"{reason}; separate approval for {operation.family_id}; "
                f"logical_correction={operation.operation_id}"
            ),
        )
        for operation in separate_operations
    )

    return SoundcheckTransactionPlanBundle(
        normal_plans=normal_plans,
        separate_approval_plans=separate_plans,
        unsupported_actions=tuple(unsupported),
        run_id=resolved_run_id,
        plan_revision=plan_revision,
        plan_fingerprint=plan_fingerprint,
    )
