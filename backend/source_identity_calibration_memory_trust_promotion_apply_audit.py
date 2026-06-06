"""Rollback audit for applied source-identity memory trust promotion.

The audit consumes the operator-gated promotion executor output and validates
applied results before any downstream policy is allowed to trust them. It does
not apply trust, change proposal priority, mutate queues or models, or touch
live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from source_identity_calibration_memory_trust_promotion_executor import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_STATUSES,
        build_source_identity_calibration_memory_trust_promotion_executor,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_calibration_memory_trust_promotion_executor import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_STATUSES,
        build_source_identity_calibration_memory_trust_promotion_executor,
    )


SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID = (
    "source_identity_calibration_memory_trust_promotion_apply_audit"
)
SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_STATUSES = {
    "memory_trust_promotion_apply_audit_not_ready",
    "memory_trust_promotion_apply_audit_replacement_required",
    "memory_trust_promotion_apply_audit_records_available",
}
REQUIRED_EXECUTOR_TYPE = "source_identity_calibration_memory_trust_promotion_executor"
TARGET_STORE = "source_identity_downstream_gate"
REQUIRED_APPLY_RESULT_FIELDS = (
    "promotion_executor_apply_id",
    "promotion_executor_dry_run_id",
    "source_candidate_id",
    "source_type",
    "target_store",
    "target_path",
    "operation",
    "previous_trust_state",
    "target_trust_state",
    "rollback_strategy",
    "idempotency_key",
)
REQUIRED_EXECUTOR_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "direct_downstream_priority_mutation",
    "downstream_priority_change_allowed_by_executor",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_EXECUTOR_FALSE_SUMMARY_FLAGS = (
    "queue_mutation",
    "model_mutation",
    "direct_downstream_priority_mutation",
    "downstream_priority_change_allowed_by_executor",
    "live_write",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_APPLY_RESULT_FALSE_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queue_mutation",
    "model_mutation",
    "direct_downstream_priority_mutation",
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


def _record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("promotion_executor_apply_id"))
    if explicit:
        return explicit
    dry_run_id = _string(record.get("promotion_executor_dry_run_id"))
    return f"{dry_run_id if dry_run_id else 'unknown_promotion_apply'}:{index}"


def _target_path(source_candidate_id: str) -> str:
    return f"{TARGET_STORE}/{source_candidate_id}/trust_state"


def _executor_envelope_violations(executor: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if executor.get("type") != REQUIRED_EXECUTOR_TYPE:
        violations.append("memory_trust_promotion_apply_audit_executor_type_mismatch")
    generated_at = executor.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("memory_trust_promotion_apply_audit_executor_generated_at_invalid")
    if (
        executor.get("method_id")
        != SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID
    ):
        violations.append("memory_trust_promotion_apply_audit_executor_method_id_mismatch")
    if (
        executor.get("status")
        not in SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_STATUSES
    ):
        violations.append("memory_trust_promotion_apply_audit_executor_status_unknown")
    safety = _mapping(executor.get("safety"))
    for flag in REQUIRED_EXECUTOR_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"memory_trust_promotion_apply_audit_executor_{flag}_not_false"
            )
    return tuple(violations)


def _executor_status_violations(
    executor: Mapping[str, Any],
    *,
    apply_results: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = executor.get("status")
    if status == "memory_trust_promotion_executor_replacement_required":
        return ("memory_trust_promotion_apply_audit_executor_replacement_required",)
    if status == "memory_trust_promotion_executor_applied" and not apply_results:
        return ("memory_trust_promotion_apply_audit_applied_without_results",)
    if status != "memory_trust_promotion_executor_applied" and apply_results:
        return ("memory_trust_promotion_apply_audit_results_without_applied_status",)
    return ()


def _executor_apply_safety_violations(
    executor: Mapping[str, Any],
    *,
    applied: bool,
) -> Tuple[str, ...]:
    violations = []
    safety = _mapping(executor.get("safety"))
    summary = _mapping(executor.get("summary"))
    expected_true_flags = (
        "runtime_memory_mutation",
        "source_identity_trust_store_mutation",
        "memory_trust_promotion_applied",
    )
    for field in expected_true_flags:
        if safety.get(field) is not applied:
            violations.append(f"memory_trust_promotion_apply_audit_executor_{field}_mismatch")
        if summary.get(field) is not applied:
            violations.append(f"memory_trust_promotion_apply_audit_summary_{field}_mismatch")
    if safety.get("read_only") is not (not applied):
        violations.append("memory_trust_promotion_apply_audit_executor_read_only_mismatch")
    for flag in REQUIRED_EXECUTOR_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"memory_trust_promotion_apply_audit_summary_{flag}_not_false"
            )
    if summary.get("target_store") != TARGET_STORE:
        violations.append("memory_trust_promotion_apply_audit_summary_target_store_invalid")
    return tuple(violations)


def _apply_result_violations(record: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    source_candidate_id = _string(record.get("source_candidate_id"))
    for field in REQUIRED_APPLY_RESULT_FIELDS:
        if not _string(record.get(field)):
            violations.append(f"memory_trust_promotion_apply_audit_record_{field}_missing")
    if record.get("target_store") != TARGET_STORE:
        violations.append("memory_trust_promotion_apply_audit_record_target_store_invalid")
    if record.get("target_path") != _target_path(source_candidate_id):
        violations.append("memory_trust_promotion_apply_audit_record_target_path_invalid")
    if record.get("operation") != "set_source_identity_trust_state":
        violations.append("memory_trust_promotion_apply_audit_record_operation_invalid")
    if record.get("applied") is not True:
        violations.append("memory_trust_promotion_apply_audit_record_applied_not_true")
    if record.get("source_identity_trust_store_mutation") is not True:
        violations.append(
            "memory_trust_promotion_apply_audit_record_trust_store_mutation_not_true"
        )
    for flag in REQUIRED_APPLY_RESULT_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(
                f"memory_trust_promotion_apply_audit_record_{flag}_not_false"
            )
    return tuple(violations)


def _record_replacement_ids(records: Tuple[Mapping[str, Any], ...]) -> Tuple[str, ...]:
    return tuple(
        _record_id(record, index)
        for index, record in enumerate(records)
        if _apply_result_violations(record)
    )


def _duplicate_ids(records: Tuple[Mapping[str, Any], ...], field: str) -> Tuple[str, ...]:
    values: Dict[str, list[str]] = {}
    for index, record in enumerate(records):
        if _apply_result_violations(record):
            continue
        value = _string(record.get(field))
        if not value:
            continue
        values.setdefault(value, []).append(_record_id(record, index))
    return tuple(
        record_id
        for record_ids in values.values()
        if len(record_ids) > 1
        for record_id in record_ids
    )


def _summary_count_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    expected_counts = {
        "apply_result_count": len(records),
        "invalid_record_count": 0,
    }
    return tuple(
        f"memory_trust_promotion_apply_audit_summary_{key}_mismatch"
        for key, expected in expected_counts.items()
        if _summary_int(summary.get(key)) != expected
    )


def _summary_identity_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    if not records:
        return ()
    expected_paths = tuple(
        record["target_path"] for record in records if _string(record.get("target_path"))
    )
    expected_keys = tuple(
        record["idempotency_key"]
        for record in records
        if _string(record.get("idempotency_key"))
    )
    expected_source_ids = tuple(
        record["source_candidate_id"]
        for record in records
        if _string(record.get("source_candidate_id"))
    )
    violations = []
    if _string_list(summary.get("target_paths")) != expected_paths:
        violations.append("memory_trust_promotion_apply_audit_summary_paths_mismatch")
    if _string_list(summary.get("idempotency_keys")) != expected_keys:
        violations.append("memory_trust_promotion_apply_audit_summary_keys_mismatch")
    if _string_list(summary.get("updated_source_candidate_ids")) != expected_source_ids:
        violations.append("memory_trust_promotion_apply_audit_summary_source_ids_mismatch")
    return tuple(violations)


def _audit_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "promotion_apply_audit_id": (
            "source_identity_memory_trust_promotion_apply_audit:"
            f"{record.get('promotion_executor_apply_id')}"
        ),
        "promotion_executor_apply_id": record.get("promotion_executor_apply_id"),
        "promotion_executor_dry_run_id": record.get("promotion_executor_dry_run_id"),
        "source_candidate_id": record.get("source_candidate_id"),
        "source_type": record.get("source_type"),
        "target_store": TARGET_STORE,
        "target_path": record.get("target_path"),
        "operation": record.get("operation"),
        "previous_trust_state": record.get("previous_trust_state"),
        "target_trust_state": record.get("target_trust_state"),
        "rollback_strategy": record.get("rollback_strategy"),
        "rollback_target_trust_state": record.get("previous_trust_state"),
        "rollback_evidence_required": True,
        "rollback_verified": False,
        "idempotency_key": record.get("idempotency_key"),
        "applied": True,
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "queue_mutation": False,
        "model_mutation": False,
        "runtime_memory_mutation": False,
        "direct_downstream_priority_mutation": False,
        "source_identity_trust_store_mutation_audited": True,
        "downstream_priority_change_allowed_by_audit": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }


def build_source_identity_calibration_memory_trust_promotion_apply_audit(
    *,
    promotion_executor: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a read-only rollback audit for applied memory-trust promotion."""

    executor = (
        _mapping(promotion_executor)
        or build_source_identity_calibration_memory_trust_promotion_executor()
    )
    summary = _mapping(executor.get("summary"))
    apply_results = _record_list(executor.get("apply_results"))
    applied = bool(apply_results)
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_executor_envelope_violations(executor),
                *_executor_status_violations(executor, apply_results=apply_results),
                *_executor_apply_safety_violations(executor, applied=applied),
                *_record_replacement_ids(apply_results),
                *_duplicate_ids(apply_results, "promotion_executor_apply_id"),
                *_duplicate_ids(apply_results, "source_candidate_id"),
                *_duplicate_ids(apply_results, "idempotency_key"),
                *_summary_count_violations(summary, records=apply_results),
                *_summary_identity_violations(summary, records=apply_results),
            )
        )
    )
    audit_records = (
        ()
        if invalid_record_ids
        else tuple(_audit_record(record) for record in apply_results)
    )

    if invalid_record_ids:
        status = "memory_trust_promotion_apply_audit_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_promotion_executor_before_apply_audit",
        )
    elif audit_records:
        status = "memory_trust_promotion_apply_audit_records_available"
        blocked_by = (
            "rollback_audit_is_not_downstream_priority_or_live_authority",
        )
    else:
        status = "memory_trust_promotion_apply_audit_not_ready"
        blocked_by = (
            "applied_memory_trust_promotion_result_required_before_apply_audit",
        )

    return {
        "type": "source_identity_calibration_memory_trust_promotion_apply_audit",
        "generated_at": _utc_now(),
        "method_id": (
            SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID
        ),
        "source_method_id": (
            SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "source_identity_trust_store_mutation": False,
            "audits_applied_source_identity_trust_store_mutation": bool(audit_records),
            "direct_downstream_priority_mutation": False,
            "downstream_priority_change_allowed_by_audit": False,
            "rollback_evidence_required": bool(audit_records),
            "rollback_verified": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "promotion_executor_status": executor.get("status") or "missing",
            "executor_apply_result_count": len(apply_results),
            "apply_audit_record_count": len(audit_records),
            "invalid_record_count": len(invalid_record_ids),
            "target_store": TARGET_STORE,
            "rollback_evidence_required": bool(audit_records),
            "rollback_verified": False,
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "source_identity_trust_store_mutation": False,
            "audits_applied_source_identity_trust_store_mutation": bool(audit_records),
            "direct_downstream_priority_mutation": False,
            "downstream_priority_change_allowed_by_audit": False,
            "live_write": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "audited_source_candidate_ids": [
                record["source_candidate_id"] for record in audit_records
            ],
            "rollback_target_trust_states": [
                record["rollback_target_trust_state"] for record in audit_records
            ],
            "target_paths": [record.get("target_path") for record in audit_records],
            "idempotency_keys": [
                record.get("idempotency_key") for record in audit_records
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid promotion executor output before audit."
                if invalid_record_ids
                else (
                    "Use rollback audit as review evidence only; a separate "
                    "downstream priority gate is required before proposal policy changes."
                    if audit_records
                    else "Applied promotion executor results are required before rollback audit."
                )
            ),
        },
        "audit_records": list(audit_records),
    }
