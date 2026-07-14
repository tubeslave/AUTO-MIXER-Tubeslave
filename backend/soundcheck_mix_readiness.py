"""Pure soundcheck preflight assessment and mix-readiness aggregation.

This module never opens a transport and never grants live authority. It turns
fresh snapshots and terminal child results into typed residuals and an honest
run-level readiness contract. The canonical transaction coordinator remains the
only execution kernel.
"""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import time
from typing import Any, Dict, Iterable, Mapping, Sequence

from compressor_adaptation import ratio_float_to_wing
from soundcheck_transaction_plan import (
    FAMILY_ORDER,
    STAGE_ORDER,
    bind_soundcheck_manifest_content_sha256,
)
from supervised_console_parameter_registry import get_supervised_console_family
from supervised_console_transaction import ConsoleTransactionOperation


READINESS_SCHEMA_VERSION = "soundcheck_readiness/v1"
ASSESSMENT_SCHEMA_VERSION = "soundcheck_preflight_assessment/v1"
SAFE_CORE_STAGES = (
    "foundation_level",
    "source_control",
    "phase_alignment_review",
)
SHOW_MIX_STAGES = (
    "tonal_dynamics",
    "group_balance",
    "spatial_fx",
)
SAFE_FADER_TRANSLATION_SOURCE_MAX_AGE_SEC = 30.0

DISPOSITION_APPLY = "apply"
DISPOSITION_ALREADY_SATISFIED = "already_satisfied"
DISPOSITION_SPLIT_STEPS = "split_steps"
DISPOSITION_OPERATOR_REVIEW = "operator_review"
DISPOSITION_FORBIDDEN = "forbidden"
DISPOSITION_BLOCKED_GLOBAL = "blocked_global"

SPLIT_REASONS = frozenset(
    {
        "gain_down_delta_exceeds_6db",
        "fader_cut_delta_exceeded",
        "send_cut_delta_exceeds_6db",
        "eq_cut_delta_exceeds_3db",
        "compressor_makeup_cut_delta_exceeds_3db",
        "pan_delta_exceeds_10_percent",
        "delay_delta_exceeds_5ms",
        "hpf_frequency_delta_exceeds_one_octave",
        "compressor_ratio_delta_exceeds_2",
        "dynamics_threshold_delta_exceeds_3db",
    }
)

OPERATOR_REVIEW_REASONS = frozenset(
    {
        "gain_up_requires_separate_headroom_and_source_proof",
        "fader_boost_not_allowed",
        "send_level_increase_not_allowed",
        "relative_eq_boost_not_allowed",
        "compressor_makeup_increase_not_allowed",
        "unmute_requires_dedicated_operator_action",
        "processing_enable_state_change_requires_separate_approval",
    }
)

GLOBAL_BLOCKER_REASONS = frozenset(
    {
        "mixer_not_connected",
        "operator_mode_blocks_live_write",
        "wing_observation_only_active",
        "physical_readback_transport_unavailable",
        "physical_snapshot_unverified",
        "physical_snapshot_stale",
        "emergency_stop_active",
        "emergency_generation_changed",
        "writer_conflict",
        "durable_write_journal_unavailable",
        "durable_terminal_record_failed",
        "rollback_state_unknown",
        "master_capture_route_proof_required",
        "stereo_master_capture_channels_required",
        "master_true_peak_unavailable_or_stale",
        "master_true_peak_headroom_insufficient",
        "feedback_state_unavailable_or_stale",
        "feedback_risk_detected",
    }
)

RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "destructive": 3}

ALTERNATIVE_SEARCH_BY_REASON = {
    "gain_up_requires_separate_headroom_and_source_proof": (
        "same_channel_fader_with_constructive_lane_proof",
        "lower_competing_sources_after_trusted_reanalysis",
    ),
    "fader_boost_not_allowed": (
        "lower_competing_sources_after_trusted_reanalysis",
        "constructive_fader_lane_with_fresh_main_and_feedback_proof",
    ),
    "send_level_increase_not_allowed": (
        "reduce_dry_competitors_after_trusted_reanalysis",
        "separate_constructive_send_lane",
    ),
    "relative_eq_boost_not_allowed": (
        "complementary_cut_on_verified_masking_source",
        "bounded_balance_change_after_trusted_reanalysis",
    ),
    "compressor_makeup_increase_not_allowed": (
        "fader_balance_after_gain_reduction_measurement",
        "lower_competing_sources_after_trusted_reanalysis",
    ),
}


@dataclass(frozen=True)
class SoundcheckOperationAssessment:
    operation_id: str
    scope: str
    disposition: str
    reason: str
    stage: str
    required_for: str
    physical_verified: bool = False
    audible_state_verified: bool = False
    executable: bool = False
    required_evidence: tuple[str, ...] = ()
    replacement_of: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["required_evidence"] = list(self.required_evidence)
        return payload


def plan_preflight_assessment(
    plan: Mapping[str, Any],
    assessments_by_operation: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Return a conservative plan-level view of operation preflight truth.

    Ordinary soundcheck children contain one operation. A coupled safe
    alternative is the sole multi-operation exception and is applyable only
    when every member is either applyable or already physically satisfied and
    at least one member still needs a write.
    """

    raw_operations = list(plan.get("operations") or [])
    mapped_operations = [
        operation
        for operation in raw_operations
        if isinstance(operation, Mapping)
    ]
    operation_ids = [
        str(operation.get("operation_id") or "").strip()
        for operation in mapped_operations
    ]
    identity_invalid = bool(
        len(raw_operations) != len(mapped_operations)
        or any(not operation_id for operation_id in operation_ids)
        or len(set(operation_ids)) != len(operation_ids)
    )
    members = [
        dict(assessments_by_operation.get(operation_id) or {})
        for operation_id in operation_ids
    ]
    if len(operation_ids) == 1 and not identity_invalid:
        return members[0]

    coupled_errors = coupled_safe_alternative_plan_errors(plan)
    coupled = bool(
        len(operation_ids) >= 2
        and not identity_invalid
        and not coupled_errors
        and plan.get("coupled_safe_alternative_group") is True
        and plan.get("atomic_group_commit_required") is True
    )
    missing_ids = [
        operation_id
        for operation_id, member in zip(operation_ids, members)
        if not member
    ]
    dispositions = {
        str(member.get("disposition") or "")
        for member in members
        if member
    }
    all_satisfied = bool(
        members
        and not missing_ids
        and dispositions == {DISPOSITION_ALREADY_SATISFIED}
    )
    applyable = bool(
        coupled
        and not missing_ids
        and dispositions.issubset(
            {DISPOSITION_APPLY, DISPOSITION_ALREADY_SATISFIED}
        )
        and DISPOSITION_APPLY in dispositions
    )
    if identity_invalid:
        disposition = DISPOSITION_BLOCKED_GLOBAL
        reason = "transaction_operation_identity_invalid"
    elif not operation_ids:
        disposition = DISPOSITION_BLOCKED_GLOBAL
        reason = "missing_transaction_operations"
    elif not coupled:
        disposition = DISPOSITION_BLOCKED_GLOBAL
        reason = "multi_operation_preflight_requires_atomic_group"
    elif missing_ids:
        disposition = DISPOSITION_BLOCKED_GLOBAL
        reason = "coupled_group_preflight_incomplete"
    elif all_satisfied:
        disposition = DISPOSITION_ALREADY_SATISFIED
        reason = "physical_target_already_satisfied"
    elif applyable:
        disposition = DISPOSITION_APPLY
        reason = "coupled_group_preflight_applyable"
    else:
        priority = (
            DISPOSITION_BLOCKED_GLOBAL,
            DISPOSITION_FORBIDDEN,
            DISPOSITION_OPERATOR_REVIEW,
            DISPOSITION_SPLIT_STEPS,
        )
        disposition = next(
            (candidate for candidate in priority if candidate in dispositions),
            DISPOSITION_OPERATOR_REVIEW,
        )
        reason = "coupled_group_member_not_applyable"

    captured_times = [
        float(member["captured_at_epoch"])
        for member in members
        if isinstance(member.get("captured_at_epoch"), (int, float))
        and not isinstance(member.get("captured_at_epoch"), bool)
    ]
    max_ages = [
        float(member["max_age_sec"])
        for member in members
        if isinstance(member.get("max_age_sec"), (int, float))
        and not isinstance(member.get("max_age_sec"), bool)
    ]
    digest_payload = [
        {
            "operation_id": operation_id,
            "disposition": member.get("disposition"),
            "reason": member.get("reason"),
            "physical_verified": member.get("physical_verified") is True,
            "audible_state_verified": member.get("audible_state_verified") is True,
            "captured_at_epoch": member.get("captured_at_epoch"),
            "max_age_sec": member.get("max_age_sec"),
        }
        for operation_id, member in zip(operation_ids, members)
    ]
    return {
        "operation_id": str(plan.get("transaction_id") or ""),
        "scope": "coupled_group",
        "disposition": disposition,
        "reason": reason,
        "physical_verified": bool(
            members
            and not missing_ids
            and all(member.get("physical_verified") is True for member in members)
        ),
        "audible_state_verified": bool(
            members
            and not missing_ids
            and all(member.get("audible_state_verified") is True for member in members)
        ),
        "executable": applyable,
        "captured_at_epoch": min(captured_times) if captured_times else None,
        "max_age_sec": min(max_ages) if max_ages else None,
        "member_operation_ids": operation_ids,
        "member_assessments": members,
        "member_assessment_digest": hashlib.sha256(
            json.dumps(
                digest_payload,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest(),
        "atomic_group_commit_required": coupled,
        "coupled_group_contract_errors": coupled_errors,
    }


def classify_transition_reason(reason: str | None) -> tuple[str, str, tuple[str, ...]]:
    """Map one adapter transition reason to scope, disposition and evidence."""
    normalized = str(reason or "").strip()
    if not normalized:
        return "operation", DISPOSITION_APPLY, ()
    global_reason = bool(
        normalized in GLOBAL_BLOCKER_REASONS
        or normalized.startswith("emergency_generation_changed")
        or normalized.startswith("physical_snapshot_")
        or normalized.startswith("master_true_peak_")
        or normalized.startswith("master_capture_route_")
        or normalized.startswith("stereo_master_capture_")
        or normalized.startswith("durable_")
        or normalized.startswith("rollback_state_unknown")
        or normalized.startswith("writer_conflict")
        or normalized.startswith("proposal_import_incomplete")
        or normalized in {
            "master_audio_signal_missing",
            "feedback_state_unavailable_or_stale",
            "feedback_risk_detected",
        }
    )
    if global_reason:
        return (
            "global",
            DISPOSITION_BLOCKED_GLOBAL,
            ("fresh_global_preflight",),
        )
    if normalized in SPLIT_REASONS:
        return (
            "operation",
            DISPOSITION_SPLIT_STEPS,
            (
                "original_physical_baseline",
                "cumulative_session_budget",
                "fresh_snapshot_between_steps",
                "fresh_audio_reanalysis_between_steps",
                "new_plan_revision_and_approval",
            ),
        )
    if normalized in OPERATOR_REVIEW_REASONS:
        evidence = ["separate_operator_review", "new_plan_revision_and_approval"]
        if normalized in {
            "gain_up_requires_separate_headroom_and_source_proof",
            "fader_boost_not_allowed",
            "send_level_increase_not_allowed",
            "compressor_makeup_increase_not_allowed",
        }:
            evidence.extend(
                [
                    "fresh_physical_main_47_48_route",
                    "windowed_master_headroom",
                    "trusted_source_identity",
                    "fresh_feedback_clear_state",
                ]
            )
        return "operation", DISPOSITION_OPERATOR_REVIEW, tuple(evidence)
    if (
        normalized.startswith("invalid_")
        or normalized.endswith("_out_of_range")
        or normalized.startswith("unsupported_")
        or normalized in {
            "apply_snapshot_incomplete",
            "channel_required",
            "family_adapter_identity_mismatch",
            "write_kind_family_mismatch",
        }
    ):
        return "operation", DISPOSITION_FORBIDDEN, ("corrected_absolute_target",)
    return "operation", DISPOSITION_OPERATOR_REVIEW, ("typed_operator_review",)


def _numbers_equal(expected: Any, actual: Any, tolerance: float) -> bool:
    try:
        left = float(expected)
        right = float(actual)
    except (TypeError, ValueError):
        return str(expected).strip().upper() == str(actual).strip().upper()
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance


def _snapshot_audible_state_verified(
    operation: ConsoleTransactionOperation,
    snapshot: Mapping[str, Any],
) -> bool:
    requirements = list(
        dict(operation.metadata).get("audible_state_requirements") or []
    )
    if not requirements:
        return True
    if snapshot.get("audible_state_verified") is True:
        return True
    values = dict(snapshot.get("values") or {})
    checks = []
    if "eq_enabled_physical_readback" in requirements:
        checks.append(
            any(
                str(address).endswith("/eq/on")
                and value in {1, 1.0, True, "1"}
                for address, value in values.items()
            )
        )
    if "compressor_enabled_physical_readback" in requirements:
        checks.append(
            any(
                str(address).endswith("/dyn/on")
                and value in {1, 1.0, True, "1"}
                for address, value in values.items()
            )
        )
    return bool(checks and all(checks))


def soundcheck_manifest_logical_id_errors(
    manifest: Mapping[str, Any],
) -> list[str]:
    """Reject ambiguous logical identities across every manifest category."""

    identifiers: list[str] = []
    plan_identity_errors: list[str] = []
    for plan in manifest.get("plans") or []:
        if not isinstance(plan, Mapping):
            continue
        operation_ids = [
            str(raw.get("operation_id") or "").strip()
            for raw in plan.get("operations") or []
            if isinstance(raw, Mapping)
        ]
        logical_ids = [
            str(value or "").strip()
            for value in plan.get("logical_correction_ids") or []
        ]
        if operation_ids != logical_ids:
            plan_identity_errors.append(
                "plan_logical_ids_must_equal_ordered_operation_ids:"
                + str(plan.get("transaction_id") or "missing")
            )
        plan_identity_errors.extend(
            coupled_safe_alternative_plan_errors(
                plan,
                source_coverage=(
                    manifest.get("source_coverage")
                    if isinstance(manifest.get("source_coverage"), Mapping)
                    else None
                ),
            )
        )
        for raw in plan.get("operations") or []:
            if isinstance(raw, Mapping):
                identifiers.append(str(raw.get("operation_id") or "").strip())
    for key in ("unsupported_actions", "readiness_residuals"):
        for raw in manifest.get(key) or []:
            if isinstance(raw, Mapping):
                identifiers.append(
                    str(raw.get("logical_correction_id") or "").strip()
                )
    errors = list(plan_identity_errors)
    if any(not item for item in identifiers):
        errors.append("logical_correction_id_required")
    duplicates = sorted(
        identifier
        for identifier, count in Counter(identifiers).items()
        if identifier and count > 1
    )
    errors.extend(
        f"duplicate_logical_correction_id:{identifier}"
        for identifier in duplicates
    )
    return errors


def coupled_safe_alternative_plan_errors(
    plan: Mapping[str, Any],
    *,
    snapshots_by_operation: Mapping[str, Mapping[str, Any]] | None = None,
    source_coverage: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate the sole trusted multi-operation soundcheck child shape."""

    if not (
        plan.get("coupled_safe_alternative_group") is True
        or plan.get("atomic_group_commit_required") is True
    ):
        return []
    transaction_id = str(plan.get("transaction_id") or "").strip()
    operations = [
        dict(operation)
        for operation in plan.get("operations") or []
        if isinstance(operation, Mapping)
    ]
    errors: list[str] = []
    operation_ids = [
        str(operation.get("operation_id") or "").strip()
        for operation in operations
    ]
    logical_ids = [
        str(value or "").strip()
        for value in plan.get("logical_correction_ids") or []
    ]
    channels = [operation.get("channel") for operation in operations]
    if not (
        plan.get("coupled_safe_alternative_group") is True
        and plan.get("atomic_group_commit_required") is True
    ):
        errors.append("coupled_group_flags_incomplete")
    if len(operations) < 2:
        errors.append("coupled_group_requires_multiple_operations")
    if (
        not operation_ids
        or any(not operation_id for operation_id in operation_ids)
        or len(set(operation_ids)) != len(operation_ids)
    ):
        errors.append("coupled_group_operation_ids_invalid")
    if logical_ids != operation_ids:
        errors.append("coupled_group_logical_ids_mismatch")
    if not transaction_id:
        errors.append("coupled_group_transaction_id_required")
    if (
        any(
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or not 1 <= channel <= 48
            for channel in channels
        )
        or len(set(channels)) != len(channels)
    ):
        errors.append("coupled_group_selected_channels_invalid")
    if isinstance(source_coverage, Mapping):
        selected_channels = {
            int(channel)
            for channel in source_coverage.get("selected_channel_ids") or []
            if isinstance(channel, int)
            and not isinstance(channel, bool)
            and 1 <= channel <= 48
        }
        if set(channels) != selected_channels:
            errors.append("coupled_group_must_cover_full_selected_source_vector")

    trusted_ids: set[str] = set()
    evidence_digests: set[str] = set()
    translations: set[float] = set()
    support_sets: set[tuple[str, ...]] = set()
    replacement_roots: list[str] = []
    fader_intents = (
        dict(source_coverage.get("per_source_fader_intent") or {})
        if isinstance(source_coverage, Mapping)
        and isinstance(source_coverage.get("per_source_fader_intent"), Mapping)
        else {}
    )
    for operation in operations:
        metadata = (
            dict(operation.get("metadata") or {})
            if isinstance(operation.get("metadata"), Mapping)
            else {}
        )
        if not (
            operation.get("family_id") == "fader"
            and operation.get("write_kind") == "fader"
            and operation.get("parameter") == "channel_fader_db"
        ):
            errors.append("coupled_group_member_must_be_channel_fader")
        if not (
            metadata.get("atomic_group_commit_required") is True
            and metadata.get("dependency_group_id") == transaction_id
            and metadata.get("coupled_safe_alternative_transaction_id")
            == transaction_id
        ):
            errors.append("coupled_group_member_binding_invalid")
        if metadata.get("plan_revision") != plan.get("plan_revision"):
            errors.append("coupled_group_member_plan_revision_mismatch")
        if metadata.get("safe_alternative_strategy") != (
            "global_fader_translation_v1"
        ):
            errors.append("coupled_group_strategy_invalid")
        role = str(metadata.get("member_role") or "")
        if role == "replacement":
            replacement_root = str(
                metadata.get("replacement_of") or ""
            ).strip()
            if not replacement_root:
                errors.append("coupled_group_replacement_provenance_missing")
            else:
                replacement_roots.append(replacement_root)
            if metadata.get("supporting_hold_current_operation") is not False:
                errors.append("coupled_group_replacement_support_flag_invalid")
        elif role == "hold_current_support":
            if metadata.get("replacement_of"):
                errors.append("coupled_group_hold_must_not_claim_replacement")
            hold_readback_epoch = metadata.get(
                "hold_readback_captured_at_epoch"
            )
            if not (
                metadata.get("supporting_hold_current_operation") is True
                and metadata.get("hold_current_baseline") is True
                and isinstance(hold_readback_epoch, (int, float))
                and not isinstance(hold_readback_epoch, bool)
                and math.isfinite(float(hold_readback_epoch))
            ):
                errors.append("coupled_group_hold_readback_binding_invalid")
        else:
            errors.append("coupled_group_member_role_invalid")
        trusted_id = str(metadata.get("trusted_reanalysis_id") or "").strip()
        digest = str(metadata.get("analysis_evidence_digest") or "").strip()
        if (
            not trusted_id
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            errors.append("coupled_group_analysis_binding_missing")
        trusted_ids.add(trusted_id)
        evidence_digests.add(digest)
        supports = tuple(
            sorted(
                str(value or "").strip()
                for value in metadata.get("supports_replacement_of") or []
                if str(value or "").strip()
            )
        )
        if not supports:
            errors.append("coupled_group_supported_boost_set_missing")
        support_sets.add(supports)
        try:
            translation = float(metadata.get("global_fader_translation_db"))
            original_final = float(
                metadata.get("pre_translation_final_target_value")
            )
            translated_final = float(metadata.get("final_target_value"))
            target = float(operation.get("target_value"))
        except (TypeError, ValueError):
            errors.append("coupled_group_numeric_contract_invalid")
            continue
        if not all(
            math.isfinite(value)
            for value in (
                translation,
                original_final,
                translated_final,
                target,
            )
        ):
            errors.append("coupled_group_numeric_contract_invalid")
            continue
        translations.add(translation)
        if translation >= 0.0:
            errors.append("coupled_group_translation_must_be_down_only")
        if abs((original_final + translation) - translated_final) > 0.05:
            errors.append("coupled_group_translation_equation_mismatch")
        if not (-90.0 < translated_final <= 0.0 and -90.0 < target <= 0.0):
            errors.append("coupled_group_target_outside_audible_safe_range")
        if isinstance(source_coverage, Mapping):
            intent = fader_intents.get(str(operation.get("channel")))
            expected_original = None
            if isinstance(intent, Mapping):
                expected_original = (
                    intent.get("analysis_baseline_db")
                    if role == "hold_current_support"
                    else intent.get("absolute_target_db")
                )
            if not (
                isinstance(intent, Mapping)
                and intent.get("mode")
                == (
                    "hold_current"
                    if role == "hold_current_support"
                    else "absolute_target"
                )
                and intent.get("analysis_run_id") == plan.get("run_id")
                and intent.get("analysis_captured_at_epoch")
                == source_coverage.get("captured_at_epoch")
                and metadata.get("source_analysis_captured_at_epoch")
                == source_coverage.get("captured_at_epoch")
                and isinstance(expected_original, (int, float))
                and not isinstance(expected_original, bool)
                and math.isfinite(float(expected_original))
                and abs(float(expected_original) - original_final) <= 0.05
            ):
                errors.append("coupled_group_source_intent_binding_mismatch")
        if snapshots_by_operation is not None:
            snapshot = snapshots_by_operation.get(
                str(operation.get("operation_id") or ""),
                {},
            )
            address = f"/ch/{operation.get('channel')}/fdr"
            try:
                current = float(dict(snapshot.get("values") or {})[address])
            except (KeyError, TypeError, ValueError):
                errors.append("coupled_group_dynamic_snapshot_missing")
            else:
                expected_step = max(translated_final, current - 6.0)
                if target > current + 0.05:
                    errors.append("coupled_group_member_increase_forbidden")
                if abs(target - expected_step) > 0.05:
                    errors.append("coupled_group_bounded_step_equation_mismatch")
    if len(trusted_ids) != 1 or len(evidence_digests) != 1:
        errors.append("coupled_group_analysis_binding_inconsistent")
    if len(translations) != 1:
        errors.append("coupled_group_translation_inconsistent")
    if len(support_sets) != 1:
        errors.append("coupled_group_supported_boost_set_inconsistent")
    if len(replacement_roots) != len(set(replacement_roots)):
        errors.append("coupled_group_replacement_roots_not_unique")
    if len(support_sets) == 1 and not set(next(iter(support_sets))).issubset(
        set(replacement_roots)
    ):
        errors.append("coupled_group_supported_boost_set_not_replaced")
    return sorted(set(errors))


def assess_operation_snapshot(
    operation: ConsoleTransactionOperation,
    snapshot: Mapping[str, Any],
    adapter: Any,
) -> SoundcheckOperationAssessment:
    """Classify one operation from an already collected physical snapshot."""
    metadata = dict(operation.metadata)
    stage = str(metadata.get("stage") or "destructive_high_risk")
    required_for = str(metadata.get("required_for") or "show_mix")
    if not isinstance(snapshot, Mapping) or snapshot.get("physical_verified") is not True:
        return SoundcheckOperationAssessment(
            operation.operation_id,
            "global",
            DISPOSITION_BLOCKED_GLOBAL,
            "physical_snapshot_unverified",
            stage,
            required_for,
            required_evidence=("fresh_two_round_trip_physical_snapshot",),
        )
    try:
        prepared = adapter.prepare(operation)
    except Exception as exc:
        reason = str(exc) or type(exc).__name__
        scope, disposition, evidence = classify_transition_reason(reason)
        return SoundcheckOperationAssessment(
            operation.operation_id,
            scope,
            disposition,
            reason,
            stage,
            required_for,
            physical_verified=True,
            audible_state_verified=_snapshot_audible_state_verified(operation, snapshot),
            executable=False,
            required_evidence=evidence,
        )

    writes = tuple(prepared.get("writes") or ())
    values = dict(snapshot.get("values") or {})
    validate_snapshot = getattr(adapter, "validate_snapshot", None)
    transition_reason = (
        validate_snapshot(operation, snapshot) if callable(validate_snapshot) else None
    )
    # A matching numeric target is not an audible result when the physical
    # processor is off (for example EQ gain already equals the target while
    # /eq/on=0).  Validate auxiliary enable-state and snapshot completeness
    # before the already-satisfied shortcut so the operator sees the real
    # residual and no readiness row is falsely resolved.
    if transition_reason:
        scope, disposition, evidence = classify_transition_reason(transition_reason)
        return SoundcheckOperationAssessment(
            operation.operation_id,
            scope,
            disposition,
            str(transition_reason),
            stage,
            required_for,
            physical_verified=True,
            audible_state_verified=_snapshot_audible_state_verified(operation, snapshot),
            executable=disposition == DISPOSITION_APPLY,
            required_evidence=evidence,
        )

    if writes and all(
        write.get("address") in values
        and _numbers_equal(
            write.get("target_value"),
            values.get(write.get("address")),
            float(write.get("tolerance") or 0.0),
        )
        for write in writes
    ):
        return SoundcheckOperationAssessment(
            operation.operation_id,
            "operation",
            DISPOSITION_ALREADY_SATISFIED,
            "physical_target_already_satisfied",
            stage,
            required_for,
            physical_verified=True,
            audible_state_verified=_snapshot_audible_state_verified(operation, snapshot),
            executable=False,
            required_evidence=("physical_snapshot",),
        )

    scope, disposition, evidence = classify_transition_reason(transition_reason)
    return SoundcheckOperationAssessment(
        operation.operation_id,
        scope,
        disposition,
        str(transition_reason or "transition_preflight_passed"),
        stage,
        required_for,
        physical_verified=True,
        audible_state_verified=_snapshot_audible_state_verified(operation, snapshot),
        executable=disposition == DISPOSITION_APPLY,
        required_evidence=evidence,
    )


def _operation_from_payload(payload: Mapping[str, Any]) -> ConsoleTransactionOperation:
    return ConsoleTransactionOperation(
        operation_id=str(payload.get("operation_id") or ""),
        family_id=str(payload.get("family_id") or ""),
        write_kind=str(payload.get("write_kind") or ""),
        target_value=payload.get("target_value"),
        channel=payload.get("channel"),
        parameter=str(payload.get("parameter") or ""),
        metadata=(
            dict(payload.get("metadata") or {})
            if isinstance(payload.get("metadata"), Mapping)
            else {}
        ),
    )


def assess_soundcheck_manifest(
    manifest: Mapping[str, Any],
    *,
    snapshots_by_operation: Mapping[str, Mapping[str, Any]],
    adapters: Mapping[str, Any],
    global_blockers: Iterable[str] = (),
) -> Dict[str, Any]:
    """Assess a complete manifest without sending any console write."""
    operations: list[tuple[ConsoleTransactionOperation, str]] = []
    for plan in manifest.get("plans") or []:
        if not isinstance(plan, Mapping):
            continue
        transaction_id = str(plan.get("transaction_id") or "")
        for raw in plan.get("operations") or []:
            if isinstance(raw, Mapping):
                operations.append((_operation_from_payload(raw), transaction_id))

    blockers = [str(item) for item in global_blockers if str(item).strip()]
    for operation, _transaction_id in operations:
        snapshot = snapshots_by_operation.get(operation.operation_id)
        if not isinstance(snapshot, Mapping) or snapshot.get("physical_verified") is not True:
            blockers.append(f"physical_snapshot_unverified:{operation.operation_id}")
        if operation.family_id not in adapters:
            blockers.append(f"family_adapter_unavailable:{operation.family_id}")
    blockers = sorted(set(blockers))

    assessments: list[SoundcheckOperationAssessment] = []
    transaction_by_operation = {
        operation.operation_id: transaction_id
        for operation, transaction_id in operations
    }
    if blockers:
        for operation, _transaction_id in operations:
            metadata = dict(operation.metadata)
            assessments.append(
                SoundcheckOperationAssessment(
                    operation.operation_id,
                    "global",
                    DISPOSITION_BLOCKED_GLOBAL,
                    blockers[0],
                    str(metadata.get("stage") or "destructive_high_risk"),
                    str(metadata.get("required_for") or "show_mix"),
                    required_evidence=("fresh_global_preflight",),
                )
            )
    else:
        for operation, _transaction_id in operations:
            assessments.append(
                assess_operation_snapshot(
                    operation,
                    snapshots_by_operation[operation.operation_id],
                    adapters[operation.family_id],
                )
            )

    unsupported = []
    for raw in manifest.get("unsupported_actions") or []:
        if not isinstance(raw, Mapping):
            continue
        payload = dict(raw)
        payload.setdefault("scope", "operation")
        payload.setdefault("disposition", DISPOSITION_OPERATOR_REVIEW)
        unsupported.append(payload)

    disposition_counts = Counter(item.disposition for item in assessments)
    disposition_counts.update(str(item.get("disposition")) for item in unsupported)
    local_reason_counts = Counter(
        item.reason
        for item in assessments
        if item.scope != "global"
        and item.disposition
        in {DISPOSITION_SPLIT_STEPS, DISPOSITION_OPERATOR_REVIEW, DISPOSITION_FORBIDDEN}
    )
    replan_requests = [
        {
            "operation_id": item.operation_id,
            "reason": item.reason,
            "disposition": item.disposition,
            "next_plan_revision": int(manifest.get("plan_revision") or 1) + 1,
            "replacement_constraints": {
                "trusted_reanalysis_required": True,
                "new_operator_approval_required": True,
                "maximum_substitution_hops": 2,
                "risk_escalation_allowed": False,
                "new_live_write_kind_allowed": False,
            },
        }
        for item in assessments
        if item.disposition
        in {DISPOSITION_SPLIT_STEPS, DISPOSITION_OPERATOR_REVIEW}
    ]
    return {
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "run_id": str(manifest.get("run_id") or ""),
        "plan_revision": int(manifest.get("plan_revision") or 1),
        "plan_fingerprint": str(manifest.get("plan_fingerprint") or ""),
        "global_blockers": blockers,
        "assessments": [item.to_dict() for item in assessments],
        "unsupported_actions": unsupported,
        "logical_correction_count": int(
            manifest.get("logical_correction_count")
            or (len(assessments) + len(unsupported))
        ),
        "physical_operation_assessment_count": len(assessments),
        "locally_admissible_count": disposition_counts[DISPOSITION_APPLY]
        + disposition_counts[DISPOSITION_ALREADY_SATISFIED],
        "physical_change_candidate_count": disposition_counts[DISPOSITION_APPLY],
        "already_satisfied_count": disposition_counts[DISPOSITION_ALREADY_SATISFIED],
        "local_residual_count": sum(
            disposition_counts[key]
            for key in (
                DISPOSITION_SPLIT_STEPS,
                DISPOSITION_OPERATOR_REVIEW,
                DISPOSITION_FORBIDDEN,
            )
        ),
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "local_reason_counts": dict(sorted(local_reason_counts.items())),
        "executable_child_transaction_ids": [
            transaction_by_operation[item.operation_id]
            for item in assessments
            if item.executable
        ],
        "replan_requests": replan_requests,
        "live_authority_changed": False,
    }


def validate_replacement(
    original: Mapping[str, Any],
    replacement: Mapping[str, Any],
    *,
    effective_live_write_kinds: Iterable[str],
    trusted_reanalysis_evidence: Mapping[str, str] | None = None,
    current_plan_revision: int,
) -> Dict[str, Any]:
    """Validate provenance/risk bounds for a planner-generated alternative."""
    reasons: list[str] = []
    original_id = str(original.get("operation_id") or "")
    metadata = (
        dict(replacement.get("metadata") or {})
        if isinstance(replacement.get("metadata"), Mapping)
        else {}
    )
    if not original_id:
        reasons.append("original_operation_id_required")
    if str(metadata.get("replacement_of") or "") != original_id:
        reasons.append("replacement_provenance_missing")
    hop = metadata.get("substitution_hop")
    if isinstance(hop, bool) or not isinstance(hop, int) or not 1 <= hop <= 2:
        reasons.append("substitution_hop_out_of_bounds")
    analysis_id = str(metadata.get("trusted_reanalysis_id") or "").strip()
    evidence_digest = str(metadata.get("analysis_evidence_digest") or "").strip()
    if not analysis_id:
        reasons.append("trusted_reanalysis_required")
    if not evidence_digest:
        reasons.append("analysis_evidence_digest_required")
    trusted_evidence = dict(trusted_reanalysis_evidence or {})
    if analysis_id and trusted_evidence.get(analysis_id) != evidence_digest:
        reasons.append("trusted_reanalysis_evidence_mismatch")
    if metadata.get("absolute_target") is not True:
        reasons.append("absolute_target_required")
    replacement_id = str(replacement.get("operation_id") or "").strip()
    if not replacement_id or replacement_id == original_id:
        reasons.append("distinct_replacement_operation_id_required")
    if "target_value" not in replacement or replacement.get("target_value") is None:
        reasons.append("replacement_target_required")
    write_kind = str(replacement.get("write_kind") or "")
    if write_kind not in {str(item) for item in effective_live_write_kinds}:
        reasons.append("replacement_write_kind_not_effectively_allowed")
    original_contract = get_supervised_console_family(str(original.get("family_id") or ""))
    replacement_contract = get_supervised_console_family(
        str(replacement.get("family_id") or "")
    )
    if original_contract is None or replacement_contract is None:
        reasons.append("replacement_family_contract_missing")
    else:
        if write_kind not in replacement_contract.write_kinds:
            reasons.append("replacement_write_kind_family_mismatch")
        if RISK_ORDER.get(replacement_contract.risk_class, 99) > RISK_ORDER.get(
            original_contract.risk_class,
            99,
        ):
            reasons.append("replacement_risk_escalation_not_allowed")
    next_revision = metadata.get("plan_revision")
    if (
        isinstance(next_revision, bool)
        or not isinstance(next_revision, int)
        or next_revision <= int(current_plan_revision)
    ):
        reasons.append("new_plan_revision_required")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "replacement_of": original_id or None,
        "new_operator_approval_required": True,
        "live_authority_changed": False,
    }


def _target_progresses_toward_final(
    baseline: Any,
    target: Any,
    final_target: Any,
) -> bool:
    """Return true only for a non-regressing, measurable bounded step."""

    if all(
        isinstance(value, Mapping)
        for value in (baseline, target, final_target)
    ):
        baseline_map = dict(baseline)
        target_map = dict(target)
        final_map = dict(final_target)
        if set(target_map) != set(final_map) or not set(final_map).issubset(
            baseline_map
        ):
            return False
        progressed = False
        for key, final_value in final_map.items():
            baseline_value = baseline_map.get(key)
            target_value = target_map.get(key)
            if (
                isinstance(baseline_value, (int, float))
                and not isinstance(baseline_value, bool)
                and isinstance(target_value, (int, float))
                and not isinstance(target_value, bool)
                and isinstance(final_value, (int, float))
                and not isinstance(final_value, bool)
                and all(
                    math.isfinite(float(value))
                    for value in (
                        baseline_value,
                        target_value,
                        final_value,
                    )
                )
            ):
                before = abs(float(baseline_value) - float(final_value))
                after = abs(float(target_value) - float(final_value))
                if after > before + 1e-9:
                    return False
                progressed = progressed or after < before - 1e-9
                continue
            if baseline_value == final_value:
                if target_value != final_value:
                    return False
            elif target_value == final_value:
                progressed = True
            elif target_value != baseline_value:
                return False
        return progressed
    if (
        isinstance(baseline, (int, float))
        and not isinstance(baseline, bool)
        and isinstance(target, (int, float))
        and not isinstance(target, bool)
        and isinstance(final_target, (int, float))
        and not isinstance(final_target, bool)
        and all(
            math.isfinite(float(value))
            for value in (baseline, target, final_target)
        )
    ):
        before = abs(float(baseline) - float(final_target))
        after = abs(float(target) - float(final_target))
        return after < before - 1e-9
    return baseline != final_target and target == final_target


def soundcheck_replan_lineage_errors(
    current_manifest: Mapping[str, Any],
    draft: Mapping[str, Any],
) -> list[str]:
    """Reject replans that drop or forge logical correction lineage."""

    revised = draft.get("revised_manifest")
    if not isinstance(revised, Mapping):
        return ["revised_manifest_missing"]

    def indexed_operations(
        manifest: Mapping[str, Any],
    ) -> tuple[dict[str, Dict[str, Any]], dict[str, Mapping[str, Any]], list[str]]:
        by_id: dict[str, Dict[str, Any]] = {}
        plan_by_id: dict[str, Mapping[str, Any]] = {}
        duplicates: list[str] = []
        for plan in manifest.get("plans") or []:
            if not isinstance(plan, Mapping):
                continue
            for operation in plan.get("operations") or []:
                if not isinstance(operation, Mapping):
                    continue
                operation_id = str(operation.get("operation_id") or "").strip()
                if not operation_id or operation_id in by_id:
                    duplicates.append(operation_id or "missing")
                    continue
                by_id[operation_id] = dict(operation)
                plan_by_id[operation_id] = plan
        return by_id, plan_by_id, duplicates

    old_by_id, old_plan_by_id, old_duplicates = indexed_operations(
        current_manifest
    )
    new_by_id, new_plan_by_id, new_duplicates = indexed_operations(revised)
    errors: list[str] = []
    if old_duplicates:
        errors.append("current_operation_identity_invalid")
    if new_duplicates:
        errors.append("revised_operation_identity_invalid")
    if not old_by_id:
        errors.append("current_operation_lineage_missing")

    safe_details = [
        item
        for item in draft.get("safe_alternatives") or []
        if isinstance(item, Mapping)
    ]
    safe_ids = {
        str(value or "").strip()
        for detail in safe_details
        for value in detail.get("replacement_operation_ids") or []
        if str(value or "").strip()
    }
    expected_analysis_id = str(
        draft.get("trusted_reanalysis_id") or ""
    ).strip()
    expected_analysis_digest = str(
        draft.get("analysis_evidence_digest") or ""
    ).strip()
    trusted_evidence = (
        {expected_analysis_id: expected_analysis_digest}
        if expected_analysis_id and expected_analysis_digest
        else {}
    )
    current_revision = current_manifest.get("plan_revision")
    next_revision = (
        current_revision + 1
        if isinstance(current_revision, int)
        and not isinstance(current_revision, bool)
        else None
    )
    preserved_ids: set[str] = set()
    replacement_parents: dict[str, str] = {}
    split_replacement_ids: set[str] = set()

    for operation_id, operation in new_by_id.items():
        metadata = (
            dict(operation.get("metadata") or {})
            if isinstance(operation.get("metadata"), Mapping)
            else {}
        )
        plan = new_plan_by_id.get(operation_id, {})
        coupled_member = bool(
            isinstance(plan, Mapping)
            and plan.get("coupled_safe_alternative_group") is True
            and plan.get("atomic_group_commit_required") is True
        )
        if operation_id in old_by_id:
            if operation != old_by_id[operation_id]:
                errors.append(
                    f"preserved_operation_mutated:{operation_id}"
                )
            preserved_ids.add(operation_id)
            continue

        role = str(metadata.get("member_role") or "")
        parent_id = str(metadata.get("replacement_of") or "").strip()
        if role == "hold_current_support" and not parent_id:
            if operation_id not in safe_ids or not coupled_member:
                errors.append(
                    f"unbound_supporting_operation:{operation_id}"
                )
            hold_split_index = metadata.get("split_step_index")
            if (
                isinstance(hold_split_index, bool)
                or not isinstance(hold_split_index, int)
                or not 1 <= hold_split_index <= 8
            ):
                errors.append(
                    f"replacement_split_index_invalid:{operation_id}"
                )
            if not _target_progresses_toward_final(
                metadata.get("split_baseline_target_value"),
                operation.get("target_value"),
                metadata.get("final_target_value"),
            ):
                errors.append(
                    f"replacement_target_made_no_progress:{operation_id}"
                )
            continue
        if not parent_id or parent_id not in old_by_id:
            errors.append(f"unknown_replan_operation:{operation_id}")
            continue
        if parent_id in replacement_parents:
            errors.append(f"duplicate_replacement_parent:{parent_id}")
        replacement_parents[parent_id] = operation_id
        original = old_by_id[parent_id]
        original_metadata = (
            dict(original.get("metadata") or {})
            if isinstance(original.get("metadata"), Mapping)
            else {}
        )
        expected_root = str(
            original_metadata.get("root_correction_id") or parent_id
        )
        if str(metadata.get("root_correction_id") or "") != expected_root:
            errors.append(f"replacement_root_drift:{operation_id}")
        if metadata.get("plan_revision") != next_revision:
            errors.append(f"replacement_revision_drift:{operation_id}")
        if metadata.get("substitution_hop") != 1:
            errors.append(f"replacement_hop_invalid:{operation_id}")
        replacement_split_index = metadata.get("split_step_index")
        if (
            isinstance(replacement_split_index, bool)
            or not isinstance(replacement_split_index, int)
            or not 1 <= replacement_split_index <= 8
        ):
            errors.append(f"replacement_split_index_invalid:{operation_id}")
        validation = validate_replacement(
            original,
            operation,
            effective_live_write_kinds={
                str(operation.get("write_kind") or "")
            },
            trusted_reanalysis_evidence=trusted_evidence,
            current_plan_revision=(
                current_revision
                if isinstance(current_revision, int)
                and not isinstance(current_revision, bool)
                else 0
            ),
        )
        if validation.get("accepted") is not True:
            errors.extend(
                f"replacement_invalid:{operation_id}:{reason}"
                for reason in validation.get("reasons") or []
            )
        allowed_metadata_drift = {
            "logical_correction_id",
            "replacement_of",
            "root_correction_id",
            "substitution_hop",
            "trusted_reanalysis_id",
            "analysis_evidence_digest",
            "source_analysis_captured_at_epoch",
            "absolute_target",
            "plan_revision",
            "final_target_value",
            "split_baseline_target_value",
            "split_step_index",
            "split_reason",
            "pre_translation_final_target_value",
            "safe_alternative_strategy",
            "global_fader_translation_db",
            "constructive_lane_used",
            "supporting_hold_current_operation",
            "member_role",
            "supports_replacement_of",
            "atomic_group_commit_required",
            "dependency_group_id",
            "coupled_safe_alternative_transaction_id",
            "completed_atomic_group_decoupling",
        }
        if any(
            metadata.get(field_name) != original_metadata.get(field_name)
            for field_name in set(original_metadata).union(metadata)
            if field_name not in allowed_metadata_drift
        ):
            errors.append(f"replacement_metadata_safety_drift:{operation_id}")
        target_progressed = _target_progresses_toward_final(
            metadata.get("split_baseline_target_value"),
            operation.get("target_value"),
            metadata.get("final_target_value"),
        )
        safe_noop_replacement = bool(
            operation_id in safe_ids
            and metadata.get("split_baseline_target_value")
            == operation.get("target_value")
            == metadata.get("final_target_value")
        )
        old_plan = old_plan_by_id.get(parent_id, {})
        completed_atomic_noop = bool(
            metadata.get("completed_atomic_group_decoupling") is True
            and isinstance(old_plan, Mapping)
            and old_plan.get("coupled_safe_alternative_group") is True
            and old_plan.get("atomic_group_commit_required") is True
            and new_plan_by_id.get(operation_id, {}).get(
                "coupled_safe_alternative_group"
            )
            is not True
            and metadata.get("split_baseline_target_value")
            == operation.get("target_value")
            == metadata.get("final_target_value")
        )
        if (
            not target_progressed
            and not safe_noop_replacement
            and not completed_atomic_noop
        ):
            errors.append(f"replacement_target_made_no_progress:{operation_id}")

        if operation_id in safe_ids:
            if role != "replacement" or not coupled_member:
                errors.append(
                    f"safe_replacement_group_binding_invalid:{operation_id}"
                )
            continue

        split_replacement_ids.add(operation_id)
        if any(
            operation.get(field_name) != original.get(field_name)
            for field_name in (
                "family_id",
                "write_kind",
                "channel",
                "parameter",
            )
        ):
            errors.append(f"split_operation_surface_drift:{operation_id}")
        old_split_index = original_metadata.get("split_step_index", 0)
        if (
            isinstance(old_split_index, bool)
            or not isinstance(old_split_index, int)
            or metadata.get("split_step_index") != old_split_index + 1
        ):
            errors.append(f"split_step_index_invalid:{operation_id}")
        old_final_target = original_metadata.get(
            "final_target_value",
            original.get("target_value"),
        )
        if metadata.get("final_target_value") != old_final_target:
            errors.append(f"split_final_target_drift:{operation_id}")

    overlap = preserved_ids.intersection(replacement_parents)
    errors.extend(
        f"preserved_and_replaced_parent:{operation_id}"
        for operation_id in sorted(overlap)
    )
    covered_old_ids = preserved_ids.union(replacement_parents)
    for operation_id in sorted(set(old_by_id).difference(covered_old_ids)):
        errors.append(f"operation_lineage_dropped:{operation_id}")

    seen_atomic_plans: set[int] = set()
    for parent_id, old_plan in old_plan_by_id.items():
        if not (
            isinstance(old_plan, Mapping)
            and old_plan.get("coupled_safe_alternative_group") is True
            and old_plan.get("atomic_group_commit_required") is True
            and id(old_plan) not in seen_atomic_plans
        ):
            continue
        seen_atomic_plans.add(id(old_plan))
        parent_ids = [
            str(operation.get("operation_id") or "")
            for operation in old_plan.get("operations") or []
            if isinstance(operation, Mapping)
        ]
        child_ids = [
            replacement_parents.get(operation_id, "")
            for operation_id in parent_ids
        ]
        if not (
            parent_ids
            and all(child_ids)
            and all(
                dict(new_by_id[child_id].get("metadata") or {}).get(
                    "completed_atomic_group_decoupling"
                )
                is True
                for child_id in child_ids
            )
            and all(
                new_plan_by_id[child_id].get(
                    "coupled_safe_alternative_group"
                )
                is not True
                and new_plan_by_id[child_id].get(
                    "atomic_group_commit_required"
                )
                is not True
                and len(new_plan_by_id[child_id].get("operations") or []) == 1
                for child_id in child_ids
            )
        ):
            errors.append(
                "atomic_group_lineage_not_safely_decoupled:"
                + str(old_plan.get("transaction_id") or parent_id)
            )

    actual_safe_ids = safe_ids.intersection(new_by_id)
    if safe_ids != actual_safe_ids:
        errors.append("safe_alternative_operation_missing")
    if draft.get("safe_alternative_operation_count") != len(actual_safe_ids):
        errors.append("safe_alternative_operation_count_mismatch")
    if draft.get("split_operation_count") != len(split_replacement_ids):
        errors.append("split_operation_count_mismatch")
    if revised.get("unsupported_actions") != current_manifest.get(
        "unsupported_actions"
    ):
        errors.append("unsupported_action_lineage_drift")
    current_logical_count = int(
        current_manifest.get("logical_correction_count")
        or (
            len(old_by_id)
            + len(current_manifest.get("unsupported_actions") or [])
        )
    )
    if revised.get("logical_correction_count") != current_logical_count:
        errors.append("logical_correction_count_drift")
    if revised.get("plan_count") != len(
        [plan for plan in revised.get("plans") or [] if isinstance(plan, Mapping)]
    ):
        errors.append("revised_plan_count_mismatch")
    if revised.get("operation_count") != len(new_by_id):
        errors.append("revised_operation_count_mismatch")

    old_residuals = [
        dict(item)
        for item in current_manifest.get("readiness_residuals") or []
        if isinstance(item, Mapping)
    ]
    new_residuals = [
        dict(item)
        for item in revised.get("readiness_residuals") or []
        if isinstance(item, Mapping)
    ]
    for child_id, child in new_by_id.items():
        child_metadata = (
            dict(child.get("metadata") or {})
            if isinstance(child.get("metadata"), Mapping)
            else {}
        )
        if "final_target_value" not in child_metadata:
            continue
        final_target = child_metadata.get(
            "final_target_value",
            child.get("target_value"),
        )
        expected_logical_id = (
            "soundcheck:deferred:"
            + str(child_metadata.get("root_correction_id") or child_id)
        )
        continuation_rows = [
            item
            for item in new_residuals
            if item.get("reason") == "split_target_remaining"
            and item.get("current_step_operation_id") == child_id
        ]
        if child.get("target_value") != final_target:
            if not (
                len(continuation_rows) == 1
                and continuation_rows[0].get("logical_correction_id")
                == expected_logical_id
                and continuation_rows[0].get("final_target_value")
                == final_target
                and continuation_rows[0].get("status") == "pending"
                and continuation_rows[0].get("scope") == "operation"
                and continuation_rows[0].get("disposition")
                == "operator_review"
                and continuation_rows[0].get("stage")
                == child_metadata.get("stage")
                and continuation_rows[0].get("required_for")
                == child_metadata.get("required_for")
                and continuation_rows[0].get("channel")
                == child.get("channel")
                and continuation_rows[0].get("musically_required") is True
                and continuation_rows[0].get(
                    "new_plan_revision_and_approval_required"
                )
                is True
            ):
                errors.append(
                    f"split_continuation_residual_invalid:{child_id}"
                )
        elif continuation_rows:
            errors.append(f"completed_split_has_residual:{child_id}")
    for residual in old_residuals:
        if residual in new_residuals:
            continue
        parent_id = str(
            residual.get("current_step_operation_id") or ""
        ).strip()
        child_id = replacement_parents.get(parent_id)
        if (
            residual.get("reason") != "split_target_remaining"
            or not child_id
        ):
            errors.append(
                "readiness_residual_lineage_dropped:"
                + str(residual.get("logical_correction_id") or parent_id)
            )
            continue
        child = new_by_id[child_id]
        child_metadata = (
            dict(child.get("metadata") or {})
            if isinstance(child.get("metadata"), Mapping)
            else {}
        )
        final_target = child_metadata.get(
            "final_target_value",
            child.get("target_value"),
        )
        if child.get("target_value") == final_target:
            continue
        if not any(
            item.get("reason") == "split_target_remaining"
            and item.get("logical_correction_id")
            == residual.get("logical_correction_id")
            and item.get("current_step_operation_id") == child_id
            and item.get("final_target_value") == final_target
            for item in new_residuals
        ):
            errors.append(
                "split_residual_lineage_invalid:"
                + str(residual.get("logical_correction_id") or parent_id)
            )

    return sorted(set(errors))


def _validate_supporting_down_only_fader(
    operation: Mapping[str, Any],
    *,
    current_fader_db: float,
    analysis_id: str,
    analysis_digest: str,
    current_plan_revision: int,
) -> Dict[str, Any]:
    """Validate a new hold-current member without pretending it replaces a root."""

    metadata = (
        dict(operation.get("metadata") or {})
        if isinstance(operation.get("metadata"), Mapping)
        else {}
    )
    reasons = []
    target = operation.get("target_value")
    if (
        operation.get("family_id") != "fader"
        or operation.get("write_kind") != "fader"
    ):
        reasons.append("supporting_operation_must_be_fader")
    if metadata.get("member_role") != "hold_current_support":
        reasons.append("supporting_member_role_required")
    if metadata.get("replacement_of"):
        reasons.append("supporting_operation_must_not_claim_replacement")
    if metadata.get("absolute_target") is not True:
        reasons.append("absolute_target_required")
    if metadata.get("trusted_reanalysis_id") != analysis_id:
        reasons.append("trusted_reanalysis_required")
    if metadata.get("analysis_evidence_digest") != analysis_digest:
        reasons.append("trusted_reanalysis_evidence_mismatch")
    if (
        isinstance(metadata.get("plan_revision"), bool)
        or not isinstance(metadata.get("plan_revision"), int)
        or metadata["plan_revision"] <= current_plan_revision
    ):
        reasons.append("new_plan_revision_required")
    try:
        target_value = float(target)
        current_value = float(current_fader_db)
    except (TypeError, ValueError):
        reasons.append("finite_supporting_target_required")
    else:
        if not math.isfinite(target_value) or not math.isfinite(current_value):
            reasons.append("finite_supporting_target_required")
        elif not (
            -144.0 <= current_value <= 0.0
            and -144.0 <= target_value <= 0.0
        ):
            reasons.append("supporting_fader_target_out_of_range")
        elif target_value > current_value + 1e-9:
            reasons.append("supporting_fader_increase_forbidden")
        elif current_value - target_value > 6.0 + 1e-9:
            reasons.append("supporting_fader_cut_exceeds_6db")
    return {"accepted": not reasons, "reasons": reasons}


def _bounded_numeric_step(current: float, target: float, maximum_delta: float) -> float:
    delta = target - current
    if abs(delta) <= maximum_delta:
        return target
    return current + math.copysign(maximum_delta, delta)


def _bounded_split_target(
    operation: ConsoleTransactionOperation,
    snapshot: Mapping[str, Any],
    adapter: Any,
    reason: str,
) -> Any:
    """Create exactly one adapter-valid step toward an absolute final target."""

    metadata = dict(operation.metadata)
    final_target = metadata.get("final_target_value", operation.target_value)
    prepared = adapter.prepare(operation)
    values = dict(snapshot.get("values") or {})
    writes = list(prepared.get("writes") or [])
    current_target = operation.target_value

    def current_address(suffix: str, contains: str = "") -> tuple[str, float]:
        for write in writes:
            address = str(write.get("address") or "")
            if address.endswith(suffix) and (not contains or contains in address):
                return address, float(values[address])
        raise ValueError(f"split_snapshot_address_missing:{suffix}")

    compressor_split_reasons = {
        "dynamics_threshold_delta_exceeds_3db",
        "compressor_ratio_delta_exceeds_2",
        "compressor_makeup_cut_delta_exceeds_3db",
    }
    if (
        operation.family_id == "dynamics_compressor"
        and isinstance(final_target, Mapping)
        and isinstance(current_target, Mapping)
        and reason in compressor_split_reasons
    ):
        # Compressor parameters share one semantic operation.  Advancing only
        # the first rejected field while resetting every other field to the
        # absolute final target causes threshold/ratio/makeup to oscillate
        # forever across revisions.  Preserve the previous bounded target and
        # advance every constrained field from the same fresh snapshot.
        result = dict(current_target)
        constrained_fields = (
            ("/dyn/thr", "threshold_db", 3.0),
            ("/dyn/ratio", "ratio", 2.0),
            ("/dyn/gain", "makeup_db", 3.0),
        )
        for suffix, field, maximum_delta in constrained_fields:
            if field not in final_target:
                continue
            _address, current = current_address(suffix)
            bounded_value = _bounded_numeric_step(
                current,
                float(final_target[field]),
                maximum_delta,
            )
            if field == "ratio":
                bounded_value = float(ratio_float_to_wing(bounded_value))
            result[field] = bounded_value
        return result

    if reason == "gain_down_delta_exceeds_6db":
        _address, current = current_address("/in/set/trim")
        return _bounded_numeric_step(current, float(final_target), 6.0)
    if reason == "fader_cut_delta_exceeded":
        _address, current = current_address("/fdr")
        limit = 6.0 if operation.family_id == "fader" else 3.0
        if isinstance(final_target, Mapping):
            result = dict(current_target)
            result["target_db"] = _bounded_numeric_step(
                current,
                float(final_target["target_db"]),
                limit,
            )
            return result
        return _bounded_numeric_step(current, float(final_target), limit)
    if reason == "send_cut_delta_exceeds_6db":
        _address, current = current_address("/lvl", "/send/")
        result = dict(current_target)
        result["level_db"] = _bounded_numeric_step(
            current,
            float(final_target["level_db"]),
            6.0,
        )
        return result
    if reason == "eq_cut_delta_exceeds_3db":
        _address, current = current_address("g", "/eq/")
        result = dict(current_target)
        result["gain_db"] = _bounded_numeric_step(
            current,
            float(final_target["gain_db"]),
            3.0,
        )
        return result
    if reason == "compressor_makeup_cut_delta_exceeds_3db":
        _address, current = current_address("/dyn/gain")
        if isinstance(final_target, Mapping):
            result = dict(current_target)
            result["makeup_db"] = _bounded_numeric_step(
                current,
                float(final_target["makeup_db"]),
                3.0,
            )
            return result
        return _bounded_numeric_step(current, float(final_target), 3.0)
    if reason == "pan_delta_exceeds_10_percent":
        _address, current = current_address("/pan")
        return _bounded_numeric_step(current, float(final_target), 10.0)
    if reason == "delay_delta_exceeds_5ms":
        _address, current = current_address("/in/set/dly")
        result = dict(current_target)
        result["delay_ms"] = _bounded_numeric_step(
            current,
            float(final_target["delay_ms"]),
            5.0,
        )
        return result
    if reason == "hpf_frequency_delta_exceeds_one_octave":
        _address, current = current_address("/flt/lcf")
        result = dict(current_target)
        target = float(final_target["freq_hz"])
        result["freq_hz"] = min(target, current * 2.0) if target > current else max(
            target,
            current / 2.0,
        )
        return result
    if reason == "compressor_ratio_delta_exceeds_2":
        _address, current = current_address("/dyn/ratio")
        result = dict(current_target)
        result["ratio"] = float(
            ratio_float_to_wing(
                _bounded_numeric_step(
                    current,
                    float(final_target["ratio"]),
                    2.0,
                )
            )
        )
        return result
    if reason == "dynamics_threshold_delta_exceeds_3db":
        suffix = "/gate/thr" if operation.family_id == "gate_expander" else "/dyn/thr"
        _address, current = current_address(suffix)
        result = dict(current_target)
        result["threshold_db"] = _bounded_numeric_step(
            current,
            float(final_target["threshold_db"]),
            3.0,
        )
        return result
    raise ValueError(f"unsupported_split_reason:{reason}")


def _split_baseline_target(
    operation: ConsoleTransactionOperation,
    snapshot: Mapping[str, Any],
    adapter: Any,
    reason: str,
) -> Any:
    """Project the fresh physical snapshot into the operation target shape."""

    prepared = adapter.prepare(operation)
    values = dict(snapshot.get("values") or {})
    writes = list(prepared.get("writes") or [])

    def current_address(suffix: str, contains: str = "") -> float:
        for write in writes:
            address = str(write.get("address") or "")
            if address.endswith(suffix) and (not contains or contains in address):
                return float(values[address])
        raise ValueError(f"split_snapshot_address_missing:{suffix}")

    target = operation.target_value
    if reason == "gain_down_delta_exceeds_6db":
        return current_address("/in/set/trim")
    if reason == "fader_cut_delta_exceeded":
        current = current_address("/fdr")
        if isinstance(target, Mapping):
            baseline = dict(target)
            baseline["target_db"] = current
            return baseline
        return current
    if reason == "send_cut_delta_exceeds_6db":
        baseline = dict(target)
        baseline["level_db"] = current_address("/lvl", "/send/")
        return baseline
    if reason == "eq_cut_delta_exceeds_3db":
        baseline = dict(target)
        baseline["gain_db"] = current_address("g", "/eq/")
        return baseline
    if reason == "pan_delta_exceeds_10_percent":
        return current_address("/pan")
    if reason == "delay_delta_exceeds_5ms":
        baseline = dict(target)
        baseline["delay_ms"] = current_address("/in/set/dly")
        return baseline
    if reason == "hpf_frequency_delta_exceeds_one_octave":
        baseline = dict(target)
        baseline["freq_hz"] = current_address("/flt/lcf")
        return baseline
    if reason in {
        "dynamics_threshold_delta_exceeds_3db",
        "compressor_ratio_delta_exceeds_2",
        "compressor_makeup_cut_delta_exceeds_3db",
    }:
        if not isinstance(target, Mapping):
            if reason == "compressor_makeup_cut_delta_exceeds_3db":
                return current_address("/dyn/gain")
            raise ValueError("split_mapping_target_required")
        baseline = dict(target)
        if operation.family_id == "dynamics_compressor":
            for suffix, field in (
                ("/dyn/thr", "threshold_db"),
                ("/dyn/ratio", "ratio"),
                ("/dyn/gain", "makeup_db"),
            ):
                if field in baseline:
                    baseline[field] = current_address(suffix)
            return baseline
        if reason == "dynamics_threshold_delta_exceeds_3db":
            baseline["threshold_db"] = current_address("/gate/thr")
            return baseline
        if reason == "compressor_makeup_cut_delta_exceeds_3db":
            baseline["makeup_db"] = current_address("/dyn/gain")
            return baseline
        if reason == "compressor_ratio_delta_exceeds_2":
            baseline["ratio"] = current_address("/dyn/ratio")
            return baseline
    raise ValueError(f"unsupported_split_reason:{reason}")


def build_soundcheck_replan_draft(
    manifest: Mapping[str, Any],
    assessment: Mapping[str, Any],
    *,
    snapshots_by_operation: Mapping[str, Mapping[str, Any]],
    adapters: Mapping[str, Any],
    include_safe_alternatives: bool = True,
) -> Dict[str, Any]:
    """Build a no-write next revision for bounded steps and operator alternatives."""

    assessment_by_id = {
        str(item.get("operation_id") or ""): item
        for item in assessment.get("assessments") or []
        if isinstance(item, Mapping)
    }
    split_operations: Dict[str, Dict[str, Any]] = {}
    safe_alternative_operations: Dict[str, Dict[str, Any]] = {}
    safe_alternative_supporting_operations: list[Dict[str, Any]] = []
    safe_alternative_details: list[Dict[str, Any]] = []
    next_revision = int(manifest.get("plan_revision") or 1) + 1
    analysis_payload = {
        "run_id": manifest.get("run_id"),
        "plan_revision": manifest.get("plan_revision"),
        "plan_fingerprint": manifest.get("plan_fingerprint"),
        "assessments": list(assessment_by_id.values()),
        "source_coverage": assessment.get("source_coverage") or {},
        "main_47_48_evidence": assessment.get("main_47_48_evidence") or {},
        "selected_source_fader_holds": (
            assessment.get("selected_source_fader_holds") or {}
        ),
        "selected_source_audible_paths": (
            assessment.get("selected_source_audible_paths") or {}
        ),
        "physical_snapshots": snapshots_by_operation,
    }
    analysis_digest = hashlib.sha256(
        json.dumps(
            analysis_payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    analysis_id = (
        f"soundcheck-preflight:{manifest.get('run_id')}:"
        f"r{manifest.get('plan_revision')}:{analysis_digest[:12]}"
    )
    deferred_residuals = []

    def evidence_fresh(
        payload: Mapping[str, Any],
        max_age_cap: float | None = None,
    ) -> bool:
        captured_at = payload.get("captured_at_epoch")
        max_age = payload.get("max_age_sec")
        effective_max_age = (
            min(float(max_age), float(max_age_cap))
            if isinstance(max_age, (int, float))
            and not isinstance(max_age, bool)
            and max_age_cap is not None
            else max_age
        )
        return bool(
            isinstance(captured_at, (int, float))
            and not isinstance(captured_at, bool)
            and isinstance(max_age, (int, float))
            and not isinstance(max_age, bool)
            and isinstance(effective_max_age, (int, float))
            and not isinstance(effective_max_age, bool)
            and float(effective_max_age) >= 0.0
            and 0.0
            <= time.time() - float(captured_at)
            <= float(effective_max_age)
        )

    # Trusted balance reanalysis: if every selected source has one fresh
    # absolute fader target, a requested boost can be represented exactly as a
    # common downward translation of the whole final target vector. Headroom
    # only increases and no constructive write is introduced. A bounded first
    # step may be transitional; the entire member vector is still atomic. Each
    # translated child still needs a new explicit
    # operator approval and the normal physical preflight/CAS path.
    source_coverage = (
        dict(assessment.get("source_coverage") or {})
        if isinstance(assessment.get("source_coverage"), Mapping)
        else {}
    )
    source_analysis_epoch = source_coverage.get("captured_at_epoch")
    if (
        isinstance(source_analysis_epoch, bool)
        or not isinstance(source_analysis_epoch, (int, float))
        or not math.isfinite(float(source_analysis_epoch))
    ):
        source_analysis_epoch = None
    source_analysis_evidence_current = bool(
        source_coverage.get("verified") is True
        and evidence_fresh(
            source_coverage,
            SAFE_FADER_TRANSLATION_SOURCE_MAX_AGE_SEC,
        )
    )
    main_evidence = (
        dict(assessment.get("main_47_48_evidence") or {})
        if isinstance(assessment.get("main_47_48_evidence"), Mapping)
        else {}
    )
    selected_channels = {
        int(channel)
        for channel in source_coverage.get("selected_channel_ids") or []
        if isinstance(channel, int) and not isinstance(channel, bool) and channel > 0
    }
    detailed_stage_coverage = (
        dict(source_coverage.get("per_source_stage_coverage") or {})
        if isinstance(source_coverage.get("per_source_stage_coverage"), Mapping)
        else {}
    )
    foundation_coverage_complete = bool(
        (source_coverage.get("stage_coverage") or {}).get("foundation_level")
        is True
    )
    if selected_channels:
        foundation_coverage_complete = bool(
            foundation_coverage_complete
            and detailed_stage_coverage
            and all(
                isinstance(detailed_stage_coverage.get(str(channel)), Mapping)
                and detailed_stage_coverage[str(channel)].get("foundation_level")
                in {"analyzed", "explicitly_not_required"}
                for channel in selected_channels
            )
        )
    main_ready = bool(
        evidence_fresh(main_evidence)
        and all(
            main_evidence.get(key) is True
            for key in ("route_verified", "fresh", "signal_present", "headroom_safe")
        )
    )
    source_ready = bool(
        source_coverage.get("verified") is True
        and source_analysis_evidence_current
        and foundation_coverage_complete
    )
    fader_rows: list[
        tuple[
            Mapping[str, Any],
            ConsoleTransactionOperation,
            float,
            float,
            bool,
        ]
    ] = []
    fader_channels: set[int] = set()
    fader_contract_complete = True
    fader_contract_reasons: list[str] = []
    fader_intents = (
        dict(source_coverage.get("per_source_fader_intent") or {})
        if isinstance(
            source_coverage.get("per_source_fader_intent"),
            Mapping,
        )
        else {}
    )
    source_paths = (
        dict(assessment.get("selected_source_audible_paths") or {})
        if isinstance(
            assessment.get("selected_source_audible_paths"),
            Mapping,
        )
        else {}
    )
    for channel in sorted(selected_channels):
        path = source_paths.get(str(channel))
        if not (
            isinstance(path, Mapping)
            and path.get("physical_verified") is True
            and path.get("two_pass_consistent") is True
            and path.get("matches_analysis_baseline") is True
            and path.get("matches_analysis_main_send") is True
            and path.get("channel_main_path_locally_enabled") is True
            and path.get("downstream_dca_path_verified") is True
            and evidence_fresh(
                path,
                SAFE_FADER_TRANSLATION_SOURCE_MAX_AGE_SEC,
            )
            and isinstance(path.get("current_fader_db"), (int, float))
            and not isinstance(path.get("current_fader_db"), bool)
            and -90.0 < float(path.get("current_fader_db")) <= 0.0
        ):
            fader_contract_complete = False
            path_reason = (
                str(path.get("reason") or "unverified")
                if isinstance(path, Mapping)
                else "missing"
            )
            fader_contract_reasons.append(
                f"selected_source_audible_path_unverified:{channel}:{path_reason}"
            )
    for plan in manifest.get("plans") or []:
        if not isinstance(plan, Mapping):
            continue
        for raw in plan.get("operations") or []:
            if not isinstance(raw, Mapping):
                continue
            operation = _operation_from_payload(raw)
            if operation.family_id != "fader" or operation.write_kind != "fader":
                continue
            snapshot = snapshots_by_operation.get(operation.operation_id, {})
            adapter = adapters.get(operation.family_id)
            try:
                prepared = adapter.prepare(operation) if adapter is not None else {}
                write = next(
                    item
                    for item in prepared.get("writes") or []
                    if str(item.get("address") or "").endswith("/fdr")
                )
                current = float(dict(snapshot.get("values") or {})[write["address"]])
                final_target = float(
                    dict(operation.metadata).get(
                        "final_target_value",
                        operation.target_value,
                    )
                )
            except (KeyError, StopIteration, TypeError, ValueError, AttributeError):
                fader_contract_complete = False
                continue
            if (
                snapshot.get("physical_verified") is not True
                or operation.channel is None
                or operation.channel in fader_channels
            ):
                fader_contract_complete = False
                continue
            intent = fader_intents.get(str(operation.channel))
            intent_target = (
                intent.get("absolute_target_db")
                if isinstance(intent, Mapping)
                else None
            )
            intent_baseline = (
                intent.get("analysis_baseline_db")
                if isinstance(intent, Mapping)
                else None
            )
            if not (
                isinstance(intent, Mapping)
                and intent.get("mode") == "absolute_target"
                and intent.get("signal_verified") is True
                and intent.get("foundation_analyzed") is True
                and intent.get("channel_main_path_locally_enabled") is True
                and intent.get("downstream_dca_path_verified") is True
                and intent.get("analyzed_source_path_enabled") is True
                and intent.get("analysis_run_id") == manifest.get("run_id")
                and intent.get("analysis_captured_at_epoch")
                == source_coverage.get("captured_at_epoch")
                and isinstance(intent_target, (int, float))
                and not isinstance(intent_target, bool)
                and math.isfinite(float(intent_target))
                and abs(float(intent_target) - final_target) <= 0.05
            ):
                fader_contract_complete = False
                fader_contract_reasons.append(
                    f"selected_source_fader_intent_unverified:{operation.channel}"
                )
                continue
            if not (
                isinstance(intent_baseline, (int, float))
                and not isinstance(intent_baseline, bool)
                and math.isfinite(float(intent_baseline))
                and -144.0 <= float(intent_baseline) <= 0.0
                and abs(float(intent_baseline) - current) <= 0.25
            ):
                fader_contract_complete = False
                fader_contract_reasons.append(
                    f"selected_source_fader_baseline_mismatch:{operation.channel}"
                )
                continue
            fader_channels.add(operation.channel)
            fader_rows.append((raw, operation, current, final_target, False))

    hold_readbacks = (
        dict(assessment.get("selected_source_fader_holds") or {})
        if isinstance(
            assessment.get("selected_source_fader_holds"),
            Mapping,
        )
        else {}
    )
    for channel in sorted(selected_channels - fader_channels):
        hold = hold_readbacks.get(str(channel))
        current = (
            hold.get("current_fader_db")
            if isinstance(hold, Mapping)
            else None
        )
        if not (
            isinstance(hold, Mapping)
            and hold.get("physical_verified") is True
            and hold.get("two_pass_consistent") is True
            and hold.get("matches_analysis_baseline") is True
            and evidence_fresh(hold)
            and isinstance(current, (int, float))
            and not isinstance(current, bool)
            and math.isfinite(float(current))
            and -144.0 <= float(current) <= 0.0
        ):
            fader_contract_complete = False
            fader_contract_reasons.append(
                f"selected_source_hold_readback_unverified:{channel}"
            )
            continue
        intent = fader_intents.get(str(channel))
        baseline = (
            intent.get("analysis_baseline_db")
            if isinstance(intent, Mapping)
            else None
        )
        if not (
            isinstance(intent, Mapping)
            and intent.get("mode") == "hold_current"
            and intent.get("signal_verified") is True
            and intent.get("foundation_analyzed") is True
            and intent.get("channel_main_path_locally_enabled") is True
            and intent.get("downstream_dca_path_verified") is True
            and intent.get("analyzed_source_path_enabled") is True
            and intent.get("analysis_run_id") == manifest.get("run_id")
            and intent.get("analysis_captured_at_epoch")
            == source_coverage.get("captured_at_epoch")
            and isinstance(baseline, (int, float))
            and not isinstance(baseline, bool)
            and math.isfinite(float(baseline))
            and -144.0 <= float(baseline) <= 0.0
            and abs(float(baseline) - float(current)) <= 0.25
        ):
            fader_contract_complete = False
            fader_contract_reasons.append(
                f"selected_source_fader_baseline_mismatch:{channel}"
            )
            continue
        operation_id = "soundcheck:hold-fader:" + hashlib.sha256(
            json.dumps(
                {
                    "run_id": manifest.get("run_id"),
                    "channel": channel,
                    "current_fader_db": float(current),
                    "analysis_evidence_digest": analysis_digest,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        metadata = {
            "source": "soundcheck_safe_alternative_reanalysis",
            "logical_correction_id": operation_id,
            "root_correction_id": operation_id,
            "stage": "foundation_level",
            "stage_order": STAGE_ORDER["foundation_level"],
            "required_for": "safe_core",
            "musically_required": True,
            "absolute_target": True,
            "hold_current_baseline": True,
            "hold_readback_captured_at_epoch": hold.get(
                "captured_at_epoch"
            ),
        }
        raw = {
            "operation_id": operation_id,
            "family_id": "fader",
            "write_kind": "fader",
            "target_value": float(current),
            "channel": channel,
            "parameter": "channel_fader_db",
            "metadata": metadata,
        }
        fader_rows.append(
            (
                raw,
                _operation_from_payload(raw),
                float(current),
                float(current),
                True,
            )
        )
        fader_channels.add(channel)

    blocked_fader_boost_ids = {
        operation_id
        for operation_id, item in assessment_by_id.items()
        if item.get("disposition") == DISPOSITION_OPERATOR_REVIEW
        and item.get("reason") == "fader_boost_not_allowed"
    }
    boost_deltas = [
        final_target - current
        for _raw, operation, current, final_target, _supporting in fader_rows
        if operation.operation_id in blocked_fader_boost_ids
        and final_target > current
    ]
    safe_alternative_blockers = []
    if not source_ready:
        safe_alternative_blockers.append("source_reanalysis_evidence_incomplete")
    if not main_ready:
        safe_alternative_blockers.append("main_47_48_evidence_incomplete")
    if assessment.get("global_blockers"):
        safe_alternative_blockers.append("global_blocker_active")
    if not fader_contract_complete:
        safe_alternative_blockers.append("fader_vector_contract_incomplete")
        safe_alternative_blockers.extend(fader_contract_reasons)
    if len(selected_channels) < 2:
        safe_alternative_blockers.append("multiple_selected_sources_required")
    if fader_channels != selected_channels:
        safe_alternative_blockers.append(
            "selected_fader_vector_mismatch:"
            f"expected={sorted(selected_channels)}:"
            f"actual={sorted(fader_channels)}"
        )
    if not boost_deltas:
        safe_alternative_blockers.append("blocked_fader_boost_missing")
    if (
        include_safe_alternatives
        and source_ready
        and main_ready
        and not assessment.get("global_blockers")
        and fader_contract_complete
        and len(selected_channels) >= 2
        and fader_channels == selected_channels
        and boost_deltas
    ):
        translation_db = max(boost_deltas)
        candidate_replacements: Dict[str, Dict[str, Any]] = {}
        candidate_supporting: list[Dict[str, Any]] = []
        candidate_deferred = []
        for (
            raw,
            operation,
            current,
            original_final_target,
            supporting,
        ) in fader_rows:
            translated_final_target = original_final_target - translation_db
            if translated_final_target <= -90.0:
                safe_alternative_blockers.append(
                    "translated_selected_source_below_audible_floor:"
                    f"{operation.channel}"
                )
                candidate_replacements = {}
                candidate_supporting = []
                candidate_deferred = []
                break
            next_target = max(translated_final_target, current - 6.0)
            if next_target <= -90.0:
                safe_alternative_blockers.append(
                    "translated_selected_source_step_below_audible_floor:"
                    f"{operation.channel}"
                )
                candidate_replacements = {}
                candidate_supporting = []
                candidate_deferred = []
                break
            replacement_id = "soundcheck:alternative:" + hashlib.sha256(
                json.dumps(
                    {
                        "replacement_of": operation.operation_id,
                        "target_value": next_target,
                        "translated_final_target": translated_final_target,
                        "plan_revision": next_revision,
                        "analysis_evidence_digest": analysis_digest,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:16]
            metadata = {
                **dict(operation.metadata),
                "logical_correction_id": replacement_id,
                **(
                    {}
                    if supporting
                    else {"replacement_of": operation.operation_id}
                ),
                "root_correction_id": dict(operation.metadata).get(
                    "root_correction_id",
                    operation.operation_id,
                ),
                "substitution_hop": 1,
                "trusted_reanalysis_id": analysis_id,
                "analysis_evidence_digest": analysis_digest,
                "source_analysis_captured_at_epoch": source_analysis_epoch,
                "absolute_target": True,
                "plan_revision": next_revision,
                "pre_translation_final_target_value": original_final_target,
                "final_target_value": translated_final_target,
                "split_baseline_target_value": current,
                "split_step_index": int(
                    dict(operation.metadata).get("split_step_index") or 0
                )
                + 1,
                "safe_alternative_strategy": "global_fader_translation_v1",
                "global_fader_translation_db": -translation_db,
                "constructive_lane_used": False,
                "supporting_hold_current_operation": supporting,
                "member_role": (
                    "hold_current_support"
                    if supporting
                    else "replacement"
                ),
                "supports_replacement_of": sorted(
                    blocked_fader_boost_ids
                ),
            }
            replacement = {
                "operation_id": replacement_id,
                "family_id": operation.family_id,
                "write_kind": operation.write_kind,
                "target_value": next_target,
                "channel": operation.channel,
                "parameter": operation.parameter,
                "metadata": metadata,
            }
            validation = (
                _validate_supporting_down_only_fader(
                    replacement,
                    current_fader_db=current,
                    analysis_id=analysis_id,
                    analysis_digest=analysis_digest,
                    current_plan_revision=int(
                        manifest.get("plan_revision") or 1
                    ),
                )
                if supporting
                else validate_replacement(
                    raw,
                    replacement,
                    effective_live_write_kinds={"fader"},
                    trusted_reanalysis_evidence={
                        analysis_id: analysis_digest
                    },
                    current_plan_revision=int(
                        manifest.get("plan_revision") or 1
                    ),
                )
            )
            if (
                validation.get("accepted") is not True
                or next_target > current + 1e-9
            ):
                candidate_replacements = {}
                candidate_supporting = []
                candidate_deferred = []
                break
            if supporting:
                candidate_supporting.append(replacement)
            else:
                candidate_replacements[operation.operation_id] = replacement
            if next_target != translated_final_target:
                candidate_deferred.append(
                    {
                        "logical_correction_id": (
                            f"soundcheck:deferred:{metadata['root_correction_id']}"
                        ),
                        "reason": "split_target_remaining",
                        "status": "pending",
                        "scope": "operation",
                        "disposition": "operator_review",
                        "stage": metadata.get("stage"),
                        "required_for": metadata.get("required_for"),
                        "channel": operation.channel,
                        "musically_required": True,
                        "current_step_operation_id": replacement_id,
                        "final_target_value": translated_final_target,
                        "new_plan_revision_and_approval_required": True,
                    }
                )
        if candidate_replacements and (
            len(candidate_replacements) + len(candidate_supporting)
            == len(fader_rows)
        ):
            coupled_operations = sorted(
                [
                    *candidate_replacements.values(),
                    *candidate_supporting,
                ],
                key=lambda item: (
                    int(item.get("channel") or 999),
                    str(item.get("operation_id") or ""),
                ),
            )
            coupled_digest = hashlib.sha256(
                "|".join(
                    str(item.get("operation_id") or "")
                    for item in coupled_operations
                ).encode("utf-8")
            ).hexdigest()[:16]
            coupled_transaction_id = (
                "soundcheck:replan:foundation_balance_group:"
                f"r{next_revision}:{coupled_digest}"
            )
            for item in coupled_operations:
                item["metadata"] = {
                    **dict(item.get("metadata") or {}),
                    "coupled_safe_alternative_transaction_id": (
                        coupled_transaction_id
                    ),
                    "dependency_group_id": coupled_transaction_id,
                    "atomic_group_commit_required": True,
                }
            safe_alternative_operations.update(candidate_replacements)
            safe_alternative_supporting_operations.extend(
                candidate_supporting
            )
            deferred_residuals.extend(candidate_deferred)
            safe_alternative_details.append(
                {
                    "strategy": "global_fader_translation_v1",
                    "blocked_operation_ids": sorted(blocked_fader_boost_ids),
                    "replacement_operation_ids": sorted(
                        item["operation_id"]
                        for item in [
                            *candidate_replacements.values(),
                            *candidate_supporting,
                        ]
                    ),
                    "translation_db": -translation_db,
                    "final_target_vector_preserved": True,
                    "relative_balance_preserved": not candidate_deferred,
                    "current_step_transitional": bool(candidate_deferred),
                    "atomic_group_prevents_partial_member_application": True,
                    "atomic_group_commit_required": True,
                    "transaction_id": coupled_transaction_id,
                    "headroom_direction": "non_increasing",
                    "constructive_lane_used": False,
                    "trusted_reanalysis_id": analysis_id,
                    "hold_current_channel_ids": sorted(
                        operation["channel"]
                        for operation in candidate_supporting
                    ),
                }
            )

    operator_requests = []
    for item in assessment_by_id.values():
        reason = str(item.get("reason") or "")
        if item.get("disposition") == DISPOSITION_OPERATOR_REVIEW:
            candidates = list(ALTERNATIVE_SEARCH_BY_REASON.get(reason, ()))
            safe_available = bool(
                item.get("operation_id") in safe_alternative_operations
                or (
                    reason == "fader_boost_not_allowed"
                    and safe_alternative_details
                )
            )
            operator_requests.append(
                {
                    "operation_id": item.get("operation_id"),
                    "reason": reason,
                    "candidate_strategies": candidates,
                    "safe_automatic_replacement_available": safe_available,
                    "selected_safe_strategy": (
                        "global_fader_translation_v1" if safe_available else None
                    ),
                    "operator_review_required": True,
                    "trusted_reanalysis_required": bool(candidates),
                }
            )

    for plan in manifest.get("plans") or []:
        if not isinstance(plan, Mapping):
            continue
        for raw in plan.get("operations") or []:
            if not isinstance(raw, Mapping):
                continue
            operation = _operation_from_payload(raw)
            if operation.operation_id in safe_alternative_operations:
                continue
            item = assessment_by_id.get(operation.operation_id, {})
            reason = str(item.get("reason") or "")
            snapshot = snapshots_by_operation.get(operation.operation_id, {})
            adapter = adapters.get(operation.family_id)
            if not isinstance(snapshot, Mapping) or adapter is None:
                continue
            final_target = dict(operation.metadata).get(
                "final_target_value",
                operation.target_value,
            )
            previous_source_analysis_epoch = dict(operation.metadata).get(
                "source_analysis_captured_at_epoch"
            )
            split_step_index = int(
                dict(operation.metadata).get("split_step_index") or 0
            )
            if split_step_index > 0 and not (
                isinstance(previous_source_analysis_epoch, (int, float))
                and not isinstance(previous_source_analysis_epoch, bool)
                and source_analysis_epoch is not None
                and source_analysis_evidence_current
                and float(source_analysis_epoch)
                > float(previous_source_analysis_epoch)
            ):
                operator_requests.append(
                    {
                        "operation_id": operation.operation_id,
                        "reason": "fresh_source_reanalysis_required_between_split_steps",
                        "candidate_strategies": [
                            "rerun_source_analysis_from_current_console_state"
                        ],
                        "safe_automatic_replacement_available": False,
                        "selected_safe_strategy": None,
                        "operator_review_required": True,
                        "trusted_reanalysis_required": True,
                    }
                )
                continue
            direct_final_step = False
            if item.get("disposition") != DISPOSITION_SPLIT_STEPS:
                if not (
                    item.get("disposition") == DISPOSITION_ALREADY_SATISFIED
                    and "final_target_value" in dict(operation.metadata)
                    and operation.target_value != final_target
                ):
                    continue
                final_operation = ConsoleTransactionOperation(
                    operation_id=operation.operation_id,
                    family_id=operation.family_id,
                    write_kind=operation.write_kind,
                    target_value=final_target,
                    channel=operation.channel,
                    parameter=operation.parameter,
                    metadata=dict(operation.metadata),
                )
                final_reason = adapter.validate_snapshot(final_operation, snapshot)
                _scope, final_disposition, _evidence = classify_transition_reason(
                    final_reason
                )
                if final_disposition == DISPOSITION_APPLY:
                    direct_final_step = True
                    reason = "bounded_final_step"
                elif final_disposition == DISPOSITION_SPLIT_STEPS:
                    reason = str(final_reason)
                else:
                    continue
            try:
                next_target = (
                    final_target
                    if direct_final_step
                    else _bounded_split_target(
                        operation,
                        snapshot,
                        adapter,
                        reason,
                    )
                )
                split_baseline_target = (
                    operation.target_value
                    if direct_final_step
                    else _split_baseline_target(
                        operation,
                        snapshot,
                        adapter,
                        reason,
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
            replacement_id = "soundcheck:split:" + hashlib.sha256(
                json.dumps(
                    {
                        "replacement_of": operation.operation_id,
                        "target_value": next_target,
                        "plan_revision": next_revision,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()[:16]
            metadata = {
                **dict(operation.metadata),
                "logical_correction_id": replacement_id,
                "replacement_of": operation.operation_id,
                "root_correction_id": dict(operation.metadata).get(
                    "root_correction_id",
                    operation.operation_id,
                ),
                # A bounded continuation is not a musical substitution.  Each
                # revision validates one immediate parent hop while the
                # separate split index provides a hard anti-loop bound.
                "substitution_hop": 1,
                "split_step_index": int(
                    dict(operation.metadata).get("split_step_index") or 0
                )
                + 1,
                "trusted_reanalysis_id": analysis_id,
                "analysis_evidence_digest": analysis_digest,
                "source_analysis_captured_at_epoch": source_analysis_epoch,
                "absolute_target": True,
                "plan_revision": next_revision,
                "final_target_value": final_target,
                "split_baseline_target_value": split_baseline_target,
                "split_reason": reason,
            }
            replacement = {
                "operation_id": replacement_id,
                "family_id": operation.family_id,
                "write_kind": operation.write_kind,
                "target_value": next_target,
                "channel": operation.channel,
                "parameter": operation.parameter,
                "metadata": metadata,
            }
            if metadata["split_step_index"] > 8:
                continue
            validation = validate_replacement(
                raw,
                replacement,
                effective_live_write_kinds={operation.write_kind},
                trusted_reanalysis_evidence={analysis_id: analysis_digest},
                current_plan_revision=int(manifest.get("plan_revision") or 1),
            )
            if validation.get("accepted") is not True:
                continue
            split_operations[operation.operation_id] = replacement
            if next_target != final_target:
                deferred_residuals.append(
                    {
                        "logical_correction_id": (
                            f"soundcheck:deferred:{metadata['root_correction_id']}"
                        ),
                        "reason": "split_target_remaining",
                        "status": "pending",
                        "scope": "operation",
                        "disposition": "operator_review",
                        "stage": metadata.get("stage"),
                        "required_for": metadata.get("required_for"),
                        "channel": operation.channel,
                        "musically_required": True,
                        "current_step_operation_id": replacement_id,
                        "final_target_value": final_target,
                        "new_plan_revision_and_approval_required": True,
                    }
                )

    atomic_group_metadata_keys = {
        "atomic_group_commit_required",
        "dependency_group_id",
        "coupled_safe_alternative_transaction_id",
        "safe_alternative_strategy",
        "global_fader_translation_db",
        "constructive_lane_used",
        "supporting_hold_current_operation",
        "member_role",
        "supports_replacement_of",
    }
    decoupled_atomic_transaction_ids: set[str] = set()
    for plan in manifest.get("plans") or []:
        if not (
            isinstance(plan, Mapping)
            and plan.get("coupled_safe_alternative_group") is True
            and plan.get("atomic_group_commit_required") is True
        ):
            continue
        group_operations = [
            dict(operation)
            for operation in plan.get("operations") or []
            if isinstance(operation, Mapping)
        ]
        group_ids = [
            str(operation.get("operation_id") or "")
            for operation in group_operations
        ]
        group_can_decouple = bool(
            group_ids
            and all(
                assessment_by_id.get(operation_id, {}).get("disposition")
                == DISPOSITION_ALREADY_SATISFIED
                for operation_id in group_ids
            )
            and all(
                operation_id in split_operations
                or dict(operation.get("metadata") or {}).get(
                    "final_target_value",
                    operation.get("target_value"),
                )
                == operation.get("target_value")
                for operation_id, operation in zip(group_ids, group_operations)
            )
            and all(
                int(dict(operation.get("metadata") or {}).get(
                    "split_step_index"
                ) or 0) < 8
                for operation in group_operations
            )
        )
        if not group_can_decouple:
            continue
        decoupled_atomic_transaction_ids.add(
            str(plan.get("transaction_id") or "")
        )
        for operation_id, operation in zip(group_ids, group_operations):
            original_metadata = dict(operation.get("metadata") or {})
            if operation_id in split_operations:
                replacement = copy.deepcopy(split_operations[operation_id])
                replacement_metadata = dict(
                    replacement.get("metadata") or {}
                )
            else:
                final_target = original_metadata.get(
                    "final_target_value",
                    operation.get("target_value"),
                )
                replacement_id = "soundcheck:decoupled:" + hashlib.sha256(
                    json.dumps(
                        {
                            "replacement_of": operation_id,
                            "target_value": operation.get("target_value"),
                            "plan_revision": next_revision,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    ).encode("utf-8")
                ).hexdigest()[:16]
                replacement_metadata = dict(original_metadata)
                replacement_metadata.update(
                    {
                        "logical_correction_id": replacement_id,
                        "replacement_of": operation_id,
                        "root_correction_id": original_metadata.get(
                            "root_correction_id",
                            operation_id,
                        ),
                        "substitution_hop": 1,
                        "trusted_reanalysis_id": analysis_id,
                        "analysis_evidence_digest": analysis_digest,
                        "source_analysis_captured_at_epoch": (
                            source_analysis_epoch
                        ),
                        "absolute_target": True,
                        "plan_revision": next_revision,
                        "final_target_value": final_target,
                        "split_baseline_target_value": operation.get(
                            "target_value"
                        ),
                        "split_step_index": int(
                            original_metadata.get("split_step_index") or 0
                        )
                        + 1,
                        "split_reason": (
                            "atomic_safe_translation_completed"
                        ),
                    }
                )
                replacement = {
                    **dict(operation),
                    "operation_id": replacement_id,
                    "metadata": replacement_metadata,
                }
            for field_name in atomic_group_metadata_keys:
                replacement_metadata.pop(field_name, None)
            replacement_metadata[
                "completed_atomic_group_decoupling"
            ] = True
            replacement["metadata"] = replacement_metadata
            split_operations[operation_id] = replacement

    replacement_operations = {
        **split_operations,
        **safe_alternative_operations,
    }
    if not replacement_operations and not safe_alternative_supporting_operations:
        return {
            "status": "operator_review_required" if operator_requests else "not_required",
            "draft_available": False,
            "operator_requests": operator_requests,
            "safe_alternative_blockers": safe_alternative_blockers,
            "live_authority_changed": False,
        }

    revised_plans = []
    for plan in manifest.get("plans") or []:
        if not isinstance(plan, Mapping):
            continue
        if str(plan.get("transaction_id") or "") in (
            decoupled_atomic_transaction_ids
        ):
            for raw in plan.get("operations") or []:
                if not isinstance(raw, Mapping):
                    continue
                operation_id = str(raw.get("operation_id") or "")
                replacement = replacement_operations.get(operation_id)
                if not isinstance(replacement, Mapping):
                    continue
                replacement_id = str(
                    replacement.get("operation_id") or ""
                )
                stage = str(
                    dict(replacement.get("metadata") or {}).get("stage")
                    or plan.get("stage")
                    or "foundation_level"
                )
                digest = hashlib.sha256(
                    replacement_id.encode("utf-8")
                ).hexdigest()[:16]
                revised_plans.append(
                    {
                        **{
                            key: value
                            for key, value in dict(plan).items()
                            if key
                            not in {
                                "preflight_assessment",
                                "transaction_id",
                                "operations",
                                "logical_correction_ids",
                                "coupled_safe_alternative_group",
                                "atomic_group_commit_required",
                                "reason",
                            }
                        },
                        "transaction_id": (
                            f"soundcheck:replan:{stage}:"
                            f"r{next_revision}:{digest}"
                        ),
                        "reason": (
                            "Continuation after verified atomic safe "
                            "translation."
                        ),
                        "operations": [dict(replacement)],
                        "logical_correction_ids": [replacement_id],
                        "plan_revision": next_revision,
                        "approved": False,
                        "execution_allowed": False,
                        "live_mixer_writes": False,
                        "transport_policy": "proposal_only",
                    }
                )
            continue
        raw_operations = []
        for raw in plan.get("operations") or []:
            if not isinstance(raw, Mapping):
                continue
            operation_id = str(raw.get("operation_id") or "")
            if operation_id in safe_alternative_operations:
                continue
            raw_operations.append(
                replacement_operations.get(operation_id, dict(raw))
            )
        if not raw_operations:
            continue
        operation_ids = [str(raw.get("operation_id") or "") for raw in raw_operations]
        stage = str(plan.get("stage") or "destructive_high_risk")
        digest = hashlib.sha256(
            "|".join(operation_ids).encode("utf-8")
        ).hexdigest()[:16]
        revised_plans.append(
            {
                **{
                    key: value
                    for key, value in dict(plan).items()
                    if key not in {"preflight_assessment", "transaction_id"}
                },
                "transaction_id": f"soundcheck:replan:{stage}:r{next_revision}:{digest}",
                "operations": raw_operations,
                "logical_correction_ids": operation_ids,
                "plan_revision": next_revision,
                "approved": False,
                "execution_allowed": False,
                "live_mixer_writes": False,
                "transport_policy": "proposal_only",
            }
        )
    if safe_alternative_details:
        coupled_operations = sorted(
            [
                *safe_alternative_operations.values(),
                *safe_alternative_supporting_operations,
            ],
            key=lambda item: (
                int(item.get("channel") or 999),
                str(item.get("operation_id") or ""),
            ),
        )
        coupled_transaction_id = str(
            safe_alternative_details[0].get("transaction_id") or ""
        )
        revised_plans.append(
            {
                "transaction_id": coupled_transaction_id,
                "reason": (
                    "Trusted common down-only fader translation; every member "
                    "must commit or roll back as one coupled balance group."
                ),
                "operations": [dict(item) for item in coupled_operations],
                "stage": "foundation_level",
                "stage_order": STAGE_ORDER["foundation_level"],
                "required_for": "safe_core",
                "logical_correction_ids": [
                    str(item.get("operation_id") or "")
                    for item in coupled_operations
                ],
                "run_id": manifest.get("run_id"),
                "plan_revision": next_revision,
                "separate_approval_required": False,
                "coupled_safe_alternative_group": True,
                "atomic_group_commit_required": True,
                "approved": False,
                "execution_allowed": False,
                "live_mixer_writes": False,
                "transport_policy": "proposal_only",
                "value_semantics": "absolute_target",
            }
        )
    revised_plans.sort(
        key=lambda plan: (
            int(plan.get("stage_order") or 999),
            FAMILY_ORDER.get(
                str((plan.get("operations") or [{}])[0].get("family_id") or ""),
                999,
            ),
            int(
                (plan.get("operations") or [{}])[0].get("channel")
                or 999
            ),
            str(
                (plan.get("operations") or [{}])[0].get("parameter")
                or ""
            ),
            str(
                (plan.get("operations") or [{}])[0].get("operation_id")
                or ""
            ),
        )
    )
    replaced_step_ids = set(replacement_operations)
    residuals = [
        dict(item)
        for item in manifest.get("readiness_residuals") or []
        if isinstance(item, Mapping)
        and not (
            str(item.get("reason") or "") == "split_target_remaining"
            and str(item.get("current_step_operation_id") or "")
            in replaced_step_ids
        )
    ] + deferred_residuals
    fingerprint_payload = {
        "plans": revised_plans,
        "unsupported_actions": manifest.get("unsupported_actions") or [],
        "readiness_residuals": residuals,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    for plan in revised_plans:
        plan["plan_fingerprint"] = fingerprint
    revised_manifest = {
        **dict(manifest),
        "batch_id": f"{manifest.get('run_id')}:r{next_revision}",
        "plan_revision": next_revision,
        "plan_fingerprint": fingerprint,
        "plans": revised_plans,
        "normal_plans": [
            plan
            for plan in revised_plans
            if plan.get("separate_approval_required") is not True
        ],
        "separate_approval_plans": [
            plan
            for plan in revised_plans
            if plan.get("separate_approval_required") is True
        ],
        "readiness_residuals": residuals,
        "plan_count": len(revised_plans),
        "operation_count": sum(
            len(plan.get("operations") or []) for plan in revised_plans
        ),
        "logical_correction_count": int(
            manifest.get("logical_correction_count")
            or (
                sum(
                    len(plan.get("operations") or [])
                    for plan in manifest.get("plans") or []
                    if isinstance(plan, Mapping)
                )
                + len(manifest.get("unsupported_actions") or [])
            )
        ),
        "supporting_physical_operation_count": len(
            safe_alternative_supporting_operations
        ),
        "execution_allowed": False,
        "live_mixer_writes": False,
        "transport_policy": "proposal_only",
    }
    bind_soundcheck_manifest_content_sha256(revised_manifest)
    draft_id = "soundcheck-replan:" + fingerprint[:20]
    return {
        "status": "draft_available",
        "draft_available": True,
        "draft_id": draft_id,
        "run_id": manifest.get("run_id"),
        "from_plan_revision": manifest.get("plan_revision"),
        "to_plan_revision": next_revision,
        "split_operation_count": len(split_operations),
        "safe_alternative_operation_count": (
            len(safe_alternative_operations)
            + len(safe_alternative_supporting_operations)
        ),
        "safe_alternatives": safe_alternative_details,
        "safe_alternative_blockers": safe_alternative_blockers,
        "deferred_target_count": len(deferred_residuals),
        "operator_requests": operator_requests,
        "trusted_reanalysis_id": analysis_id,
        "analysis_evidence_digest": analysis_digest,
        "revised_manifest": revised_manifest,
        "new_operator_approval_required": True,
        "live_authority_changed": False,
    }


def _logical_items(manifest: Mapping[str, Any]) -> list[Dict[str, Any]]:
    items: list[Dict[str, Any]] = []
    for plan in manifest.get("plans") or []:
        if not isinstance(plan, Mapping):
            continue
        for raw in plan.get("operations") or []:
            if not isinstance(raw, Mapping):
                continue
            metadata = (
                dict(raw.get("metadata") or {})
                if isinstance(raw.get("metadata"), Mapping)
                else {}
            )
            operation_id = str(raw.get("operation_id") or "")
            merged_ids = [
                str(value)
                for value in metadata.get("merged_logical_correction_ids") or []
                if str(value).strip()
            ]
            logical_ids = list(dict.fromkeys(merged_ids or [operation_id]))
            if operation_id not in logical_ids:
                logical_ids.insert(0, operation_id)
            for logical_id in logical_ids:
                items.append(
                    {
                    "operation_id": logical_id,
                    "outcome_operation_id": operation_id,
                    "stage": str(metadata.get("stage") or plan.get("stage") or "destructive_high_risk"),
                    "required_for": str(metadata.get("required_for") or plan.get("required_for") or "show_mix"),
                    "musically_required": metadata.get("musically_required") is not False,
                    "audible_state_requirements": list(
                        metadata.get("audible_state_requirements") or []
                    ),
                    "physical_planned": logical_id == operation_id,
                    "default_disposition": "pending",
                    "reason": "awaiting_terminal_child_result",
                    }
                )
    for raw in manifest.get("unsupported_actions") or []:
        if not isinstance(raw, Mapping):
            continue
        items.append(
            {
                "operation_id": str(raw.get("logical_correction_id") or ""),
                "stage": str(raw.get("stage") or "destructive_high_risk"),
                "required_for": str(raw.get("required_for") or "show_mix"),
                "musically_required": raw.get("musically_required") is not False,
                "audible_state_requirements": [],
                "physical_planned": False,
                "default_disposition": str(raw.get("disposition") or "operator_review"),
                "reason": str(raw.get("reason") or "unsupported_soundcheck_action"),
            }
        )
    for raw in manifest.get("readiness_residuals") or []:
        if not isinstance(raw, Mapping):
            continue
        items.append(
            {
                "operation_id": str(raw.get("logical_correction_id") or ""),
                "stage": str(raw.get("stage") or "foundation_level"),
                "required_for": str(raw.get("required_for") or "safe_core"),
                "musically_required": raw.get("musically_required") is not False,
                "audible_state_requirements": [],
                "physical_planned": False,
                "default_disposition": str(
                    raw.get("disposition") or "operator_review"
                ),
                "reason": str(raw.get("reason") or "soundcheck_readiness_residual"),
            }
        )
    unique: Dict[str, Dict[str, Any]] = {}
    for item in items:
        unique.setdefault(str(item.get("operation_id") or ""), item)
    return list(unique.values())


def _fresh_evidence(
    payload: Mapping[str, Any],
    *,
    captured_key: str,
    max_age_key: str,
) -> bool:
    captured_at = payload.get(captured_key)
    max_age = payload.get(max_age_key)
    return bool(
        isinstance(captured_at, (int, float))
        and not isinstance(captured_at, bool)
        and isinstance(max_age, (int, float))
        and not isinstance(max_age, bool)
        and float(max_age) >= 0.0
        and 0.0 <= time.time() - float(captured_at) <= float(max_age)
    )


def _terminal_resolution(
    outcome: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    require_final_reconciliation: bool,
) -> str | None:
    if item.get("audible_state_requirements") and outcome.get(
        "audible_state_verified"
    ) is not True:
        return None
    reconciliation_fresh = bool(
        outcome.get("final_reconciliation_verified") is True
        and _fresh_evidence(
            outcome,
            captured_key="reconciliation_captured_at_epoch",
            max_age_key="reconciliation_max_age_sec",
        )
    )
    if require_final_reconciliation and not reconciliation_fresh:
        return None
    disposition = str(outcome.get("disposition") or "")
    if disposition == DISPOSITION_ALREADY_SATISFIED:
        evidence_fresh = outcome.get("evidence_fresh") is True
        captured_at = outcome.get("evidence_captured_at_epoch")
        max_age = outcome.get("evidence_max_age_sec")
        if isinstance(captured_at, (int, float)) and isinstance(max_age, (int, float)):
            evidence_fresh = 0.0 <= time.time() - float(captured_at) <= float(max_age)
        return (
            disposition
            if outcome.get("physical_verified") is True and evidence_fresh
            else None
        )
    if disposition == "replaced":
        if (
            outcome.get("replacement_verified") is True
            and outcome.get("physical_verified") is True
            and outcome.get("audio_recheck_passed") is True
            and str(outcome.get("wal_terminal") or "") == "committed"
        ):
            return "replaced"
        return None
    if (
        str(outcome.get("status") or "") == "applied"
        and outcome.get("physical_verified") is True
        and outcome.get("audio_recheck_passed") is True
        and str(outcome.get("wal_terminal") or "") == "committed"
    ):
        return "applied"
    return None


def aggregate_soundcheck_readiness(
    manifest: Mapping[str, Any],
    *,
    outcomes: Mapping[str, Mapping[str, Any]] | None = None,
    proposal_only: bool = True,
    global_blockers: Iterable[str] = (),
    baseline_verified: bool = False,
    source_coverage: Mapping[str, Any] | None = None,
    main_47_48_evidence: Mapping[str, Any] | None = None,
    require_final_reconciliation: bool = True,
) -> Dict[str, Any]:
    """Compute run readiness from logical corrections and terminal evidence."""
    outcomes = dict(outcomes or {})
    blockers = sorted(
        {
            str(item)
            for item in (
                *tuple(global_blockers),
                *tuple(soundcheck_manifest_logical_id_errors(manifest)),
            )
            if str(item).strip()
        }
    )
    source_evidence = dict(source_coverage or {})
    main_evidence = dict(main_47_48_evidence or {})
    logical_items = _logical_items(manifest)
    logical_counts = Counter()
    unresolved_required: list[str] = []
    unresolved_reasons: Dict[str, str] = {}
    stages: Dict[str, Dict[str, Any]] = {
        stage: {
            "required": 0,
            "resolved": 0,
            "applied": 0,
            "already_satisfied": 0,
            "replaced": 0,
            "operator_review": 0,
            "forbidden": 0,
            "pending": 0,
            "ready": False,
        }
        for stage in STAGE_ORDER
    }
    rollback_unknown = False
    physical_counts = Counter({"planned": 0, "committed": 0, "rolled_back": 0})

    for item in logical_items:
        operation_id = item["operation_id"]
        outcome = outcomes.get(
            str(item.get("outcome_operation_id") or operation_id),
            {},
        )
        stage = item["stage"]
        stages.setdefault(
            stage,
            {
                "required": 0,
                "resolved": 0,
                "applied": 0,
                "already_satisfied": 0,
                "replaced": 0,
                "operator_review": 0,
                "forbidden": 0,
                "pending": 0,
                "ready": False,
            },
        )
        if item["musically_required"]:
            stages[stage]["required"] += 1
        if item["physical_planned"]:
            physical_counts["planned"] += max(
                1,
                int(outcome.get("physical_step_count") or 1),
            )
        status = str(outcome.get("status") or "")
        if status in {"rolled_back", "proof_passed"}:
            physical_counts["rolled_back"] += max(
                1,
                int(outcome.get("physical_step_count") or 1),
            )
        if status in {"manual_review_required", "rollback_failed"} or outcome.get(
            "write_state_unknown"
        ) is True:
            rollback_unknown = True

        resolution = _terminal_resolution(
            outcome,
            item,
            require_final_reconciliation=require_final_reconciliation,
        )
        if resolution:
            logical_counts[resolution] += 1
            stages[stage][resolution] += 1
            stages[stage]["resolved"] += 1
            if resolution in {"applied", "replaced"}:
                physical_counts["committed"] += max(
                    1,
                    int(outcome.get("physical_step_count") or 1),
                )
            continue

        disposition = str(
            outcome.get("disposition") or item.get("default_disposition") or "pending"
        )
        if disposition not in {
            DISPOSITION_OPERATOR_REVIEW,
            DISPOSITION_FORBIDDEN,
        }:
            disposition = "pending"
        logical_counts[disposition] += 1
        stages[stage][disposition] += 1
        if item["musically_required"]:
            unresolved_required.append(operation_id)
            unresolved_reasons[operation_id] = str(
                outcome.get("reason") or item.get("reason") or disposition
            )

    for stage_payload in stages.values():
        stage_payload["ready"] = (
            stage_payload["required"] == stage_payload["resolved"]
        )

    source_fresh = _fresh_evidence(
        source_evidence,
        captured_key="captured_at_epoch",
        max_age_key="max_age_sec",
    )
    selected_source_ids = [
        str(value)
        for value in source_evidence.get("selected_channel_ids") or []
        if str(value).strip()
    ]
    fader_intents = (
        dict(source_evidence.get("per_source_fader_intent") or {})
        if isinstance(source_evidence.get("per_source_fader_intent"), Mapping)
        else {}
    )
    # Selected-source audible-path evidence is a readiness requirement even
    # when every fader intent is hold_current and the executable plan therefore
    # contains no fader operation. Otherwise an all-hold plan could be promoted
    # after a mute, Main-send, routing or DCA change that happened after analysis.
    selected_source_balance_path_required = bool(
        selected_source_ids and fader_intents
    )
    if selected_source_balance_path_required:
        source_captured_at = source_evidence.get("captured_at_epoch")
        source_fresh = bool(
            source_fresh
            and isinstance(source_captured_at, (int, float))
            and not isinstance(source_captured_at, bool)
            and 0.0
            <= time.time() - float(source_captured_at)
            <= SAFE_FADER_TRANSLATION_SOURCE_MAX_AGE_SEC
        )
    explicit_stage_coverage = (
        dict(source_evidence.get("stage_coverage") or {})
        if isinstance(source_evidence.get("stage_coverage"), Mapping)
        else {}
    )
    per_source_stage_coverage = (
        dict(source_evidence.get("per_source_stage_coverage") or {})
        if isinstance(source_evidence.get("per_source_stage_coverage"), Mapping)
        else {}
    )
    selected_path_evidence = (
        dict(source_evidence.get("selected_source_path_evidence") or {})
        if isinstance(
            source_evidence.get("selected_source_path_evidence"),
            Mapping,
        )
        else {}
    )
    selected_source_level_path_ready = True
    for source_id in selected_source_ids:
        intent = fader_intents.get(source_id)
        baseline = (
            intent.get("analysis_baseline_db")
            if isinstance(intent, Mapping)
            else None
        )
        target = (
            intent.get("absolute_target_db")
            if isinstance(intent, Mapping)
            else None
        )
        intent_ready = bool(
            isinstance(intent, Mapping)
            and intent.get("mode") in {"absolute_target", "hold_current"}
            and intent.get("channel_main_path_locally_enabled") is True
            and intent.get("downstream_dca_path_verified") is True
            and intent.get("analyzed_source_path_enabled") is True
            and intent.get("main_send_pre") is False
            and isinstance(baseline, (int, float))
            and not isinstance(baseline, bool)
            and -90.0 < float(baseline) <= 0.0
            and (
                intent.get("mode") != "absolute_target"
                or (
                    isinstance(target, (int, float))
                    and not isinstance(target, bool)
                    and -90.0 < float(target) <= 0.0
                )
            )
        )
        if selected_source_balance_path_required:
            path = selected_path_evidence.get(source_id)
            path_ready = bool(
                isinstance(path, Mapping)
                and path.get("physical_verified") is True
                and path.get("two_pass_consistent") is True
                and path.get("matches_analysis_baseline") is True
                and path.get("matches_analysis_main_send") is True
                and path.get("channel_main_path_locally_enabled") is True
                and path.get("downstream_dca_path_verified") is True
                and _fresh_evidence(
                    path,
                    captured_key="captured_at_epoch",
                    max_age_key="max_age_sec",
                )
            )
        else:
            path_ready = True
        selected_source_level_path_ready = bool(
            selected_source_level_path_ready
            and intent_ready
            and path_ready
        )
    source_ready = bool(
        source_evidence.get("verified") is True
        and source_fresh
        and selected_source_level_path_ready
    )
    main_fresh = _fresh_evidence(
        main_evidence,
        captured_key="captured_at_epoch",
        max_age_key="max_age_sec",
    )
    main_evidence["fresh"] = bool(
        main_evidence.get("fresh") is True and main_fresh
    )
    main_ready = all(
        main_evidence.get(key) is True
        for key in ("route_verified", "fresh", "signal_present", "headroom_safe")
    )
    safe_core_unresolved = any(
        item["operation_id"] in unresolved_required
        and item["required_for"] == "safe_core"
        for item in logical_items
    )
    show_mix_unresolved = bool(unresolved_required)
    stages_with_items = {item["stage"] for item in logical_items}

    def stage_has_coverage(stage: str) -> bool:
        # A current per-source coverage contract is authoritative.  The mere
        # presence of one operation for a stage cannot prove that every
        # selected source was analyzed for that stage.
        if selected_source_ids:
            if not per_source_stage_coverage:
                return False
            per_source_complete = all(
                isinstance(per_source_stage_coverage.get(source_id), Mapping)
                and per_source_stage_coverage[source_id].get(stage)
                in {"analyzed", "explicitly_not_required"}
                for source_id in selected_source_ids
            )
            return bool(
                per_source_complete
                and explicit_stage_coverage.get(stage) is not False
            )
        if stage in explicit_stage_coverage:
            return explicit_stage_coverage.get(stage) is True
        # Legacy manifests did not publish per-source stage evidence.  Keep
        # their weaker fallback isolated so it can never override an explicit
        # false value from the current contract.
        return bool(
            (baseline_verified and stage in SAFE_CORE_STAGES)
            or stage in stages_with_items
        )

    safe_core_stage_coverage = all(
        stage_has_coverage(stage) for stage in SAFE_CORE_STAGES
    )
    show_mix_stage_coverage = all(
        stage_has_coverage(stage) for stage in SHOW_MIX_STAGES
    )
    for stage, stage_payload in stages.items():
        stage_payload["coverage_verified"] = stage_has_coverage(stage)
    evidence_ready = source_ready and main_ready

    if source_evidence.get("verified") is not True:
        unresolved_reasons.setdefault(
            "soundcheck:evidence:source_coverage",
            "source_coverage_unverified",
        )
    elif not source_fresh:
        unresolved_reasons.setdefault(
            "soundcheck:evidence:source_coverage",
            "source_coverage_stale",
        )
    elif not selected_source_level_path_ready:
        unresolved_reasons.setdefault(
            "soundcheck:evidence:selected_source_level_path",
            "selected_source_level_or_path_unverified",
        )
    for key, reason in (
        ("route_verified", "main_47_48_route_unverified"),
        ("fresh", "main_47_48_evidence_stale"),
        ("signal_present", "main_47_48_signal_missing"),
        ("headroom_safe", "main_47_48_headroom_unsafe"),
    ):
        if main_evidence.get(key) is not True:
            unresolved_reasons.setdefault(f"soundcheck:evidence:{key}", reason)
    if not safe_core_stage_coverage:
        unresolved_reasons.setdefault(
            "soundcheck:coverage:safe_core",
            "safe_core_stage_coverage_incomplete",
        )
    if not show_mix_stage_coverage:
        unresolved_reasons.setdefault(
            "soundcheck:coverage:show_mix",
            "show_mix_stage_coverage_incomplete",
        )
    for source_id in selected_source_ids:
        source_stages = per_source_stage_coverage.get(source_id)
        if not isinstance(source_stages, Mapping):
            unresolved_reasons.setdefault(
                f"soundcheck:coverage:source:{source_id}",
                "per_source_stage_coverage_missing",
            )
            continue
        for stage in (*SAFE_CORE_STAGES, *SHOW_MIX_STAGES):
            status = str(source_stages.get(stage) or "not_analyzed")
            if status not in {"analyzed", "explicitly_not_required"}:
                unresolved_reasons.setdefault(
                    f"soundcheck:coverage:source:{source_id}:{stage}",
                    status,
                )

    mix_readiness = "soundcheck_incomplete"
    if (
        not proposal_only
        and not blockers
        and not rollback_unknown
        and evidence_ready
        and safe_core_stage_coverage
        and not safe_core_unresolved
    ):
        mix_readiness = "safe_core_ready"
        if not show_mix_unresolved and show_mix_stage_coverage:
            mix_readiness = "show_mix_ready"

    if blockers:
        safety_state = "blocked_global"
        execution_status = "blocked_global"
    elif rollback_unknown:
        safety_state = "manual_review_required"
        execution_status = "manual_review_required"
    elif logical_counts[DISPOSITION_OPERATOR_REVIEW] or logical_counts[DISPOSITION_FORBIDDEN]:
        safety_state = "manual_review_required"
        execution_status = "awaiting_operator_review"
    else:
        safety_state = "clear"
        if proposal_only:
            execution_status = "awaiting_operator_review"
        elif unresolved_required and physical_counts["committed"]:
            execution_status = "partially_applied"
        elif unresolved_required:
            execution_status = "pending"
        elif mix_readiness == "show_mix_ready":
            execution_status = "complete"
        else:
            execution_status = "child_execution_complete_mix_incomplete"

    logical_counts["total"] = len(logical_items)
    return {
        "schema_version": READINESS_SCHEMA_VERSION,
        "run_id": str(manifest.get("run_id") or ""),
        "plan_fingerprint": str(manifest.get("plan_fingerprint") or ""),
        "plan_revision": int(manifest.get("plan_revision") or 1),
        "status": mix_readiness,
        "mix_readiness": mix_readiness,
        "safety_state": safety_state,
        "execution_status": execution_status,
        "proposal_only": bool(proposal_only),
        "baseline_verified": bool(baseline_verified),
        "global_blockers": blockers,
        "logical_counts": {
            key: int(logical_counts[key])
            for key in (
                "total",
                "applied",
                "already_satisfied",
                "replaced",
                "operator_review",
                "forbidden",
                "pending",
            )
        },
        "physical_step_counts": {
            key: int(physical_counts[key])
            for key in ("planned", "committed", "rolled_back")
        },
        "stages": stages,
        "unresolved_required_operation_ids": unresolved_required,
        "unresolved_reasons": unresolved_reasons,
        "source_coverage": source_evidence,
        "main_47_48_evidence": main_evidence,
        "readiness_evidence_complete": evidence_ready,
        "source_coverage_fresh": source_fresh,
        "main_47_48_evidence_fresh": main_fresh,
        "final_reconciliation_required": bool(require_final_reconciliation),
        "safe_core_stage_coverage_complete": safe_core_stage_coverage,
        "show_mix_stage_coverage_complete": show_mix_stage_coverage,
        "constructive_lane": {
            "fader_increase": "disabled_pending_fresh_prewrite_proof",
            "send_increase": "disabled_separate_high_risk_lane",
            "preamp_gain_increase": "operator_only_open_loop",
        },
        "live_authority_changed": False,
    }


def build_proposal_only_readiness(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Fail-closed readiness for an analysis-only soundcheck manifest."""
    return aggregate_soundcheck_readiness(manifest, proposal_only=True)
