"""Dry-run executor contract for source-identity memory trust promotion.

The dry run converts accepted promotion execution candidates into deterministic
apply specifications for a future executor. It does not promote trust, change
downstream proposal priority, mutate models or runtime memory, update queues,
or touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from source_identity_calibration_memory_trust_promotion_execution_plan import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_STATUSES,
        build_source_identity_calibration_memory_trust_promotion_execution_plan,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_calibration_memory_trust_promotion_execution_plan import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_STATUSES,
        build_source_identity_calibration_memory_trust_promotion_execution_plan,
    )


SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID = (
    "source_identity_calibration_memory_trust_promotion_executor_dry_run"
)
SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_STATUSES = {
    "memory_trust_promotion_executor_dry_run_not_ready",
    "memory_trust_promotion_executor_dry_run_replacement_required",
    "memory_trust_promotion_executor_dry_run_ready",
    "memory_trust_promotion_executor_apply_blocked",
}
REQUIRED_EXECUTION_PLAN_TYPE = (
    "source_identity_calibration_memory_trust_promotion_execution_plan"
)
TARGET_STORE = "source_identity_downstream_gate"
REQUIRED_EXECUTION_CANDIDATE_FIELDS = (
    "promotion_execution_candidate_id",
    "promotion_acceptance_id",
    "promotion_contract_record_id",
    "promotion_candidate_id",
    "source_candidate_id",
    "source_type",
    "recommended_trust_state",
    "previous_trust_state",
    "rollback_strategy",
    "reviewed_by",
    "reviewed_at",
)
REQUIRED_PLAN_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "execution_allowed_by_plan",
    "memory_trust_promotion_allowed_by_plan",
    "queue_mutation_allowed_by_plan",
    "model_mutation_allowed_by_plan",
    "runtime_memory_mutation_allowed_by_plan",
    "downstream_priority_change_allowed_by_plan",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_PLAN_FALSE_SUMMARY_FLAGS = (
    "execution_allowed_by_plan",
    "memory_trust_promotion_allowed_by_plan",
    "queue_mutation_allowed_by_plan",
    "model_mutation_allowed_by_plan",
    "runtime_memory_mutation_allowed_by_plan",
    "downstream_priority_change_allowed_by_plan",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_CANDIDATE_FALSE_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "execution_allowed_by_plan",
    "memory_trust_promotion_allowed_by_plan",
    "downstream_priority_change_allowed_by_plan",
    "queue_mutation_allowed_by_plan",
    "model_mutation_allowed_by_plan",
    "runtime_memory_mutation_allowed_by_plan",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _record_list(value: Any) -> Tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string(value: Any) -> str:
    return value if isinstance(value, str) and value else ""


def _string_list(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _summary_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _is_utc_timestamp(value: str) -> bool:
    if not value:
        return False
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    if parsed.tzinfo is None:
        return False
    return parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _candidate_id(candidate: Mapping[str, Any], index: int) -> str:
    explicit = _string(candidate.get("promotion_execution_candidate_id"))
    if explicit:
        return explicit
    acceptance_id = _string(candidate.get("promotion_acceptance_id"))
    return f"{acceptance_id if acceptance_id else 'unknown_execution_candidate'}:{index}"


def _plan_envelope_violations(plan: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if plan.get("type") != REQUIRED_EXECUTION_PLAN_TYPE:
        violations.append("memory_trust_promotion_execution_plan_type_mismatch")
    generated_at = plan.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("memory_trust_promotion_execution_plan_generated_at_invalid")
    if (
        plan.get("method_id")
        != SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_METHOD_ID
    ):
        violations.append("memory_trust_promotion_execution_plan_method_id_mismatch")
    if (
        plan.get("status")
        not in SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_STATUSES
    ):
        violations.append("memory_trust_promotion_execution_plan_status_unknown")
    safety = _mapping(plan.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("memory_trust_promotion_execution_plan_read_only_not_true")
    if safety.get("operator_required") is not True:
        violations.append("memory_trust_promotion_execution_plan_operator_required_not_true")
    for flag in REQUIRED_PLAN_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"memory_trust_promotion_execution_plan_{flag}_not_false")
    return tuple(violations)


def _plan_status_violations(
    plan: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = plan.get("status")
    if status == "memory_trust_promotion_execution_plan_replacement_required":
        return ("memory_trust_promotion_execution_plan_status_replacement_required",)
    if status == "memory_trust_promotion_execution_plan_not_ready" and candidates:
        return ("memory_trust_promotion_execution_plan_status_not_ready_with_candidates",)
    if (
        status == "manual_memory_trust_promotion_execution_candidate_available"
        and not candidates
    ):
        return ("memory_trust_promotion_execution_plan_status_available_without_candidates",)
    return ()


def _candidate_violations(candidate: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    for field in REQUIRED_EXECUTION_CANDIDATE_FIELDS:
        if not _string(candidate.get(field)):
            violations.append(f"memory_trust_promotion_executor_candidate_{field}_missing")
    reviewed_at = _string(candidate.get("reviewed_at"))
    if reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append("memory_trust_promotion_executor_candidate_reviewed_at_invalid")
    if candidate.get("read_only") is not True:
        violations.append("memory_trust_promotion_executor_candidate_read_only_not_true")
    if candidate.get("operator_review_required") is not True:
        violations.append(
            "memory_trust_promotion_executor_candidate_operator_review_not_true"
        )
    if candidate.get("separate_executor_required") is not True:
        violations.append(
            "memory_trust_promotion_executor_candidate_separate_executor_not_true"
        )
    if candidate.get("execution_plan_ready_for_operator_review") is not True:
        violations.append(
            "memory_trust_promotion_executor_candidate_ready_for_review_not_true"
        )
    for flag in REQUIRED_CANDIDATE_FALSE_FLAGS:
        if candidate.get(flag) is not False:
            violations.append(f"memory_trust_promotion_executor_candidate_{flag}_not_false")
    return tuple(violations)


def _candidate_replacement_ids(
    candidates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    return tuple(
        _candidate_id(candidate, index)
        for index, candidate in enumerate(candidates)
        if _candidate_violations(candidate)
    )


def _ambiguous_source_candidate_ids(
    candidates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    candidate_ids_by_source: Dict[str, list[str]] = {}
    for index, candidate in enumerate(candidates):
        if _candidate_violations(candidate):
            continue
        source_id = _string(candidate.get("source_candidate_id"))
        if not source_id:
            continue
        candidate_ids_by_source.setdefault(source_id, []).append(
            _candidate_id(candidate, index)
        )
    return tuple(
        candidate_id
        for candidate_ids in candidate_ids_by_source.values()
        if len(candidate_ids) > 1
        for candidate_id in candidate_ids
    )


def _plan_summary_count_violations(
    summary: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
    invalid_record_ids: Tuple[str, ...],
    unreviewed_contract_record_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    expected_counts = {
        "execution_candidate_count": len(candidates),
        "invalid_record_count": len(invalid_record_ids),
        "unreviewed_contract_record_count": len(unreviewed_contract_record_ids),
    }
    return tuple(
        f"memory_trust_promotion_execution_plan_summary_{key}_mismatch"
        for key, expected in expected_counts.items()
        if _summary_int(summary.get(key)) != expected
    )


def _plan_summary_identity_violations(
    summary: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    expected_candidate_ids = tuple(
        candidate["promotion_execution_candidate_id"]
        for candidate in candidates
        if _string(candidate.get("promotion_execution_candidate_id"))
    )
    expected_source_types = tuple(
        candidate["source_type"]
        for candidate in candidates
        if _string(candidate.get("source_type"))
    )
    violations = []
    if _string_list(summary.get("execution_candidate_ids")) != expected_candidate_ids:
        violations.append("memory_trust_promotion_execution_plan_summary_candidate_ids_mismatch")
    if _string_list(summary.get("accepted_source_types")) != expected_source_types:
        violations.append("memory_trust_promotion_execution_plan_summary_source_types_mismatch")
    return tuple(violations)


def _plan_summary_safety_violations(summary: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if summary.get("operator_review_required") is not True:
        violations.append(
            "memory_trust_promotion_execution_plan_summary_operator_review_not_true"
        )
    for flag in REQUIRED_PLAN_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"memory_trust_promotion_execution_plan_summary_{flag}_not_false"
            )
    return tuple(violations)


def _target_path(candidate: Mapping[str, Any]) -> str:
    return f"{TARGET_STORE}/{candidate.get('source_candidate_id')}/trust_state"


def _idempotency_key(candidate: Mapping[str, Any]) -> str:
    return ":".join(
        (
            "source_identity_memory_trust_promotion",
            str(candidate.get("source_candidate_id") or "unknown_source"),
            str(candidate.get("recommended_trust_state") or "unknown_trust_state"),
            str(candidate.get("promotion_acceptance_id") or "unknown_acceptance"),
        )
    )


def _dry_run_record(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    previous = _string(candidate.get("previous_trust_state"))
    target = _string(candidate.get("recommended_trust_state"))
    return {
        "promotion_executor_dry_run_id": (
            "source_identity_memory_trust_promotion_executor_dry_run:"
            f"{candidate.get('promotion_execution_candidate_id')}"
        ),
        "promotion_execution_candidate_id": candidate.get(
            "promotion_execution_candidate_id"
        ),
        "promotion_acceptance_id": candidate.get("promotion_acceptance_id"),
        "promotion_contract_record_id": candidate.get("promotion_contract_record_id"),
        "promotion_candidate_id": candidate.get("promotion_candidate_id"),
        "source_candidate_id": candidate.get("source_candidate_id"),
        "source_type": candidate.get("source_type"),
        "target_store": TARGET_STORE,
        "target_path": _target_path(candidate),
        "operation": "set_source_identity_trust_state",
        "previous_trust_state": previous,
        "target_trust_state": target,
        "recommended_trust_state": target,
        "rollback_strategy": candidate.get("rollback_strategy"),
        "idempotency_key": _idempotency_key(candidate),
        "would_change_trust_state": previous != target,
        "dry_run": True,
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "requires_operator_ack_before_apply": True,
        "requires_executor_contract_before_apply": True,
        "execution_allowed_by_dry_run": False,
        "memory_trust_promotion_applied": False,
        "memory_trust_promotion_allowed_by_dry_run": False,
        "downstream_priority_change_allowed_by_dry_run": False,
        "queue_mutation_allowed_by_dry_run": False,
        "model_mutation_allowed_by_dry_run": False,
        "runtime_memory_mutation_allowed_by_dry_run": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }


def build_source_identity_calibration_memory_trust_promotion_executor_dry_run(
    *,
    promotion_execution_plan: Mapping[str, Any] | None = None,
    apply: bool = False,
    operator_ack: bool = False,
) -> Dict[str, Any]:
    """Build deterministic dry-run records for a future promotion executor."""

    plan = (
        _mapping(promotion_execution_plan)
        or build_source_identity_calibration_memory_trust_promotion_execution_plan()
    )
    summary = _mapping(plan.get("summary"))
    candidates = _record_list(plan.get("execution_candidates"))
    plan_invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    unreviewed_contract_record_ids = _string_list(
        summary.get("unreviewed_contract_record_ids")
    )
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *plan_invalid_record_ids,
                *_plan_envelope_violations(plan),
                *_plan_status_violations(plan, candidates=candidates),
                *_candidate_replacement_ids(candidates),
                *_ambiguous_source_candidate_ids(candidates),
                *_plan_summary_count_violations(
                    summary,
                    candidates=candidates,
                    invalid_record_ids=plan_invalid_record_ids,
                    unreviewed_contract_record_ids=unreviewed_contract_record_ids,
                ),
                *_plan_summary_identity_violations(summary, candidates=candidates),
                *_plan_summary_safety_violations(summary),
            )
        )
    )
    source_candidates = () if invalid_record_ids else candidates
    dry_run_records = tuple(_dry_run_record(candidate) for candidate in source_candidates)

    if invalid_record_ids:
        status = "memory_trust_promotion_executor_dry_run_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_execution_plan_before_executor_dry_run",
        )
    elif unreviewed_contract_record_ids:
        status = "memory_trust_promotion_executor_dry_run_not_ready"
        blocked_by = (
            "operator_must_review_all_promotion_contract_records_before_executor_dry_run",
        )
    elif apply:
        status = "memory_trust_promotion_executor_apply_blocked"
        blocked_by = (
            "dry_run_does_not_apply_memory_trust_promotion",
            "separate_executor_contract_required_before_apply",
        )
    elif dry_run_records:
        status = "memory_trust_promotion_executor_dry_run_ready"
        blocked_by = (
            "separate_executor_contract_required_before_apply",
        )
    else:
        status = "memory_trust_promotion_executor_dry_run_not_ready"
        blocked_by = (
            "execution_candidate_required_before_executor_dry_run",
        )

    return {
        "type": "source_identity_calibration_memory_trust_promotion_executor_dry_run",
        "generated_at": _utc_now(),
        "method_id": (
            SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID
        ),
        "source_method_id": (
            SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "dry_run": True,
            "apply_requested": bool(apply),
            "operator_ack": bool(operator_ack),
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "execution_allowed_by_dry_run": False,
            "memory_trust_promotion_applied": False,
            "memory_trust_promotion_allowed_by_dry_run": False,
            "queue_mutation_allowed_by_dry_run": False,
            "model_mutation_allowed_by_dry_run": False,
            "runtime_memory_mutation_allowed_by_dry_run": False,
            "downstream_priority_change_allowed_by_dry_run": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "promotion_execution_plan_status": plan.get("status") or "missing",
            "execution_candidate_count": len(candidates),
            "dry_run_record_count": len(dry_run_records),
            "would_change_trust_state_count": sum(
                1 for record in dry_run_records if record["would_change_trust_state"]
            ),
            "invalid_record_count": len(invalid_record_ids),
            "unreviewed_contract_record_count": len(unreviewed_contract_record_ids),
            "target_store": TARGET_STORE,
            "target_paths": [record["target_path"] for record in dry_run_records],
            "idempotency_keys": [
                record["idempotency_key"] for record in dry_run_records
            ],
            "accepted_source_types": [
                str(record["source_type"])
                for record in dry_run_records
                if isinstance(record.get("source_type"), str)
            ],
            "operator_review_required": True,
            "dry_run": True,
            "apply_requested": bool(apply),
            "operator_ack": bool(operator_ack),
            "execution_allowed_by_dry_run": False,
            "memory_trust_promotion_applied": False,
            "memory_trust_promotion_allowed_by_dry_run": False,
            "queue_mutation_allowed_by_dry_run": False,
            "model_mutation_allowed_by_dry_run": False,
            "runtime_memory_mutation_allowed_by_dry_run": False,
            "downstream_priority_change_allowed_by_dry_run": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "invalid_record_ids": list(invalid_record_ids),
            "unreviewed_contract_record_ids": list(unreviewed_contract_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid execution plan records before dry-run."
                if invalid_record_ids
                else (
                    "Operator reviews all promotion contract records before dry-run."
                    if unreviewed_contract_record_ids
                    else (
                        "Dry-run apply specs are ready; a separate executor contract "
                        "is required before any trust promotion is applied."
                    )
                )
            ),
        },
        "dry_run_records": list(dry_run_records),
    }
