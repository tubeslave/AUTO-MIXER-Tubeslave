"""Operator-gated executor for source-identity memory trust promotion.

The executor consumes dry-run apply specs and can update an explicit caller
provided trust store only when `apply=True` and `operator_ack=True`. Product
Layer preview calls this with `apply=False`, so normal previews remain
non-mutating. The executor never touches live mixer transport, queues, or
classifier models.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from source_identity_calibration_memory_trust_promotion_executor_dry_run import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_STATUSES,
        build_source_identity_calibration_memory_trust_promotion_executor_dry_run,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_calibration_memory_trust_promotion_executor_dry_run import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_STATUSES,
        build_source_identity_calibration_memory_trust_promotion_executor_dry_run,
    )


SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID = (
    "source_identity_calibration_memory_trust_promotion_executor"
)
SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_STATUSES = {
    "memory_trust_promotion_executor_not_ready",
    "memory_trust_promotion_executor_replacement_required",
    "memory_trust_promotion_executor_operator_ack_required",
    "memory_trust_promotion_executor_target_store_required",
    "memory_trust_promotion_executor_ready",
    "memory_trust_promotion_executor_applied",
}
REQUIRED_DRY_RUN_TYPE = (
    "source_identity_calibration_memory_trust_promotion_executor_dry_run"
)
TARGET_STORE = "source_identity_downstream_gate"
REQUIRED_DRY_RUN_RECORD_FIELDS = (
    "promotion_executor_dry_run_id",
    "promotion_execution_candidate_id",
    "promotion_acceptance_id",
    "promotion_contract_record_id",
    "promotion_candidate_id",
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
REQUIRED_DRY_RUN_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "execution_allowed_by_dry_run",
    "memory_trust_promotion_applied",
    "memory_trust_promotion_allowed_by_dry_run",
    "queue_mutation_allowed_by_dry_run",
    "model_mutation_allowed_by_dry_run",
    "runtime_memory_mutation_allowed_by_dry_run",
    "downstream_priority_change_allowed_by_dry_run",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_DRY_RUN_FALSE_SUMMARY_FLAGS = (
    "execution_allowed_by_dry_run",
    "memory_trust_promotion_applied",
    "memory_trust_promotion_allowed_by_dry_run",
    "queue_mutation_allowed_by_dry_run",
    "model_mutation_allowed_by_dry_run",
    "runtime_memory_mutation_allowed_by_dry_run",
    "downstream_priority_change_allowed_by_dry_run",
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
    explicit = _string(record.get("promotion_executor_dry_run_id"))
    if explicit:
        return explicit
    candidate_id = _string(record.get("promotion_execution_candidate_id"))
    return f"{candidate_id if candidate_id else 'unknown_executor_dry_run'}:{index}"


def _target_path(source_candidate_id: str) -> str:
    return f"{TARGET_STORE}/{source_candidate_id}/trust_state"


def _idempotency_key(record: Mapping[str, Any]) -> str:
    return ":".join(
        (
            "source_identity_memory_trust_promotion",
            str(record.get("source_candidate_id") or "unknown_source"),
            str(record.get("target_trust_state") or "unknown_trust_state"),
            str(record.get("promotion_acceptance_id") or "unknown_acceptance"),
        )
    )


def _dry_run_envelope_violations(dry_run: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if dry_run.get("type") != REQUIRED_DRY_RUN_TYPE:
        violations.append("memory_trust_promotion_executor_dry_run_type_mismatch")
    generated_at = dry_run.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("memory_trust_promotion_executor_dry_run_generated_at_invalid")
    if (
        dry_run.get("method_id")
        != SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID
    ):
        violations.append("memory_trust_promotion_executor_dry_run_method_id_mismatch")
    if (
        dry_run.get("status")
        not in SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_STATUSES
    ):
        violations.append("memory_trust_promotion_executor_dry_run_status_unknown")
    safety = _mapping(dry_run.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("memory_trust_promotion_executor_dry_run_read_only_not_true")
    if safety.get("dry_run") is not True:
        violations.append("memory_trust_promotion_executor_dry_run_dry_run_not_true")
    if safety.get("operator_required") is not True:
        violations.append(
            "memory_trust_promotion_executor_dry_run_operator_required_not_true"
        )
    for flag in REQUIRED_DRY_RUN_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"memory_trust_promotion_executor_dry_run_{flag}_not_false"
            )
    return tuple(violations)


def _dry_run_status_violations(
    dry_run: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = dry_run.get("status")
    if status == "memory_trust_promotion_executor_dry_run_replacement_required":
        return ("memory_trust_promotion_executor_dry_run_status_replacement_required",)
    if status == "memory_trust_promotion_executor_apply_blocked":
        return ("memory_trust_promotion_executor_dry_run_status_apply_blocked",)
    if status == "memory_trust_promotion_executor_dry_run_not_ready" and records:
        return ("memory_trust_promotion_executor_dry_run_status_not_ready_with_records",)
    if status == "memory_trust_promotion_executor_dry_run_ready" and not records:
        return ("memory_trust_promotion_executor_dry_run_status_ready_without_records",)
    return ()


def _dry_run_record_violations(record: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    source_candidate_id = _string(record.get("source_candidate_id"))
    for field in REQUIRED_DRY_RUN_RECORD_FIELDS:
        if not _string(record.get(field)):
            violations.append(f"memory_trust_promotion_executor_record_{field}_missing")
    if record.get("target_store") != TARGET_STORE:
        violations.append("memory_trust_promotion_executor_record_target_store_invalid")
    if record.get("target_path") != _target_path(source_candidate_id):
        violations.append("memory_trust_promotion_executor_record_target_path_invalid")
    if record.get("operation") != "set_source_identity_trust_state":
        violations.append("memory_trust_promotion_executor_record_operation_invalid")
    if record.get("idempotency_key") != _idempotency_key(record):
        violations.append("memory_trust_promotion_executor_record_idempotency_key_invalid")
    if record.get("dry_run") is not True:
        violations.append("memory_trust_promotion_executor_record_dry_run_not_true")
    if record.get("read_only") is not True:
        violations.append("memory_trust_promotion_executor_record_read_only_not_true")
    if record.get("requires_operator_ack_before_apply") is not True:
        violations.append("memory_trust_promotion_executor_record_ack_required_not_true")
    if record.get("requires_executor_contract_before_apply") is not True:
        violations.append(
            "memory_trust_promotion_executor_record_contract_required_not_true"
        )
    for flag in REQUIRED_DRY_RUN_FALSE_SAFETY_FLAGS:
        if record.get(flag) is not False:
            violations.append(f"memory_trust_promotion_executor_record_{flag}_not_false")
    return tuple(violations)


def _record_replacement_ids(records: Tuple[Mapping[str, Any], ...]) -> Tuple[str, ...]:
    return tuple(
        _record_id(record, index)
        for index, record in enumerate(records)
        if _dry_run_record_violations(record)
    )


def _duplicate_ids(records: Tuple[Mapping[str, Any], ...], field: str) -> Tuple[str, ...]:
    values: Dict[str, list[str]] = {}
    for index, record in enumerate(records):
        if _dry_run_record_violations(record):
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


def _dry_run_summary_count_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
    invalid_record_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    expected_counts = {
        "dry_run_record_count": len(records),
        "invalid_record_count": len(invalid_record_ids),
        "would_change_trust_state_count": sum(
            1 for record in records if record.get("would_change_trust_state") is True
        ),
    }
    return tuple(
        f"memory_trust_promotion_executor_dry_run_summary_{key}_mismatch"
        for key, expected in expected_counts.items()
        if _summary_int(summary.get(key)) != expected
    )


def _dry_run_summary_identity_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    expected_paths = tuple(
        record["target_path"] for record in records if _string(record.get("target_path"))
    )
    expected_keys = tuple(
        record["idempotency_key"]
        for record in records
        if _string(record.get("idempotency_key"))
    )
    expected_source_types = tuple(
        record["source_type"] for record in records if _string(record.get("source_type"))
    )
    violations = []
    if _string_list(summary.get("target_paths")) != expected_paths:
        violations.append("memory_trust_promotion_executor_dry_run_summary_paths_mismatch")
    if _string_list(summary.get("idempotency_keys")) != expected_keys:
        violations.append("memory_trust_promotion_executor_dry_run_summary_keys_mismatch")
    if _string_list(summary.get("accepted_source_types")) != expected_source_types:
        violations.append(
            "memory_trust_promotion_executor_dry_run_summary_source_types_mismatch"
        )
    return tuple(violations)


def _dry_run_summary_safety_violations(summary: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if summary.get("operator_review_required") is not True:
        violations.append(
            "memory_trust_promotion_executor_dry_run_summary_operator_review_not_true"
        )
    if summary.get("dry_run") is not True:
        violations.append("memory_trust_promotion_executor_dry_run_summary_dry_run_not_true")
    for flag in REQUIRED_DRY_RUN_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"memory_trust_promotion_executor_dry_run_summary_{flag}_not_false"
            )
    return tuple(violations)


def _apply_record(
    record: Mapping[str, Any],
    target_store: MutableMapping[str, str],
) -> Dict[str, Any]:
    source_id = _string(record.get("source_candidate_id"))
    previous = target_store.get(source_id, _string(record.get("previous_trust_state")))
    target = _string(record.get("target_trust_state"))
    target_store[source_id] = target
    return {
        "promotion_executor_apply_id": (
            "source_identity_memory_trust_promotion_executor_apply:"
            f"{record.get('promotion_executor_dry_run_id')}"
        ),
        "promotion_executor_dry_run_id": record.get("promotion_executor_dry_run_id"),
        "source_candidate_id": source_id,
        "source_type": record.get("source_type"),
        "target_store": TARGET_STORE,
        "target_path": record.get("target_path"),
        "operation": "set_source_identity_trust_state",
        "previous_trust_state": previous,
        "target_trust_state": target,
        "rollback_strategy": record.get("rollback_strategy"),
        "idempotency_key": record.get("idempotency_key"),
        "applied": True,
        "live_write": False,
        "mutates_console_state": False,
        "queue_mutation": False,
        "model_mutation": False,
        "direct_downstream_priority_mutation": False,
        "source_identity_trust_store_mutation": True,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }


def build_source_identity_calibration_memory_trust_promotion_executor(
    *,
    executor_dry_run: Mapping[str, Any] | None = None,
    target_store: MutableMapping[str, str] | None = None,
    apply: bool = False,
    operator_ack: bool = False,
) -> Dict[str, Any]:
    """Validate and optionally apply memory-trust promotion to a supplied store."""

    dry_run = (
        _mapping(executor_dry_run)
        or build_source_identity_calibration_memory_trust_promotion_executor_dry_run()
    )
    summary = _mapping(dry_run.get("summary"))
    records = _record_list(dry_run.get("dry_run_records"))
    dry_run_invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *dry_run_invalid_record_ids,
                *_dry_run_envelope_violations(dry_run),
                *_dry_run_status_violations(dry_run, records=records),
                *_record_replacement_ids(records),
                *_duplicate_ids(records, "source_candidate_id"),
                *_duplicate_ids(records, "idempotency_key"),
                *_dry_run_summary_count_violations(
                    summary,
                    records=records,
                    invalid_record_ids=dry_run_invalid_record_ids,
                ),
                *_dry_run_summary_identity_violations(summary, records=records),
                *_dry_run_summary_safety_violations(summary),
            )
        )
    )
    target_available = isinstance(target_store, MutableMapping)
    apply_results: Tuple[Dict[str, Any], ...] = ()

    if invalid_record_ids:
        status = "memory_trust_promotion_executor_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_executor_dry_run_before_apply",
        )
    elif not records:
        status = "memory_trust_promotion_executor_not_ready"
        blocked_by = ("dry_run_record_required_before_memory_trust_promotion_apply",)
    elif not apply:
        status = "memory_trust_promotion_executor_ready"
        blocked_by = (
            "operator_ack_and_apply_required_before_memory_trust_promotion",
        )
    elif not operator_ack:
        status = "memory_trust_promotion_executor_operator_ack_required"
        blocked_by = ("operator_ack_required_before_memory_trust_promotion",)
    elif not target_available:
        status = "memory_trust_promotion_executor_target_store_required"
        blocked_by = ("target_trust_store_required_before_memory_trust_promotion",)
    else:
        apply_results = tuple(_apply_record(record, target_store) for record in records)
        status = "memory_trust_promotion_executor_applied"
        blocked_by = (
            "promotion_applied_to_explicit_trust_store_only_no_live_console_write",
        )

    applied = bool(apply_results)
    updated_store = dict(target_store) if target_available else {}
    return {
        "type": "source_identity_calibration_memory_trust_promotion_executor",
        "generated_at": _utc_now(),
        "method_id": SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID,
        "source_method_id": (
            SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": not applied,
            "apply_requested": bool(apply),
            "operator_ack": bool(operator_ack),
            "target_store_provided": target_available,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": applied,
            "source_identity_trust_store_mutation": applied,
            "direct_downstream_priority_mutation": False,
            "downstream_priority_change_allowed_by_executor": False,
            "memory_trust_promotion_applied": applied,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "executor_dry_run_status": dry_run.get("status") or "missing",
            "dry_run_record_count": len(records),
            "apply_result_count": len(apply_results),
            "invalid_record_count": len(invalid_record_ids),
            "target_store": TARGET_STORE,
            "apply_requested": bool(apply),
            "operator_ack": bool(operator_ack),
            "target_store_provided": target_available,
            "memory_trust_promotion_applied": applied,
            "runtime_memory_mutation": applied,
            "source_identity_trust_store_mutation": applied,
            "queue_mutation": False,
            "model_mutation": False,
            "direct_downstream_priority_mutation": False,
            "downstream_priority_change_allowed_by_executor": False,
            "live_write": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "target_paths": [record.get("target_path") for record in records],
            "idempotency_keys": [record.get("idempotency_key") for record in records],
            "updated_source_candidate_ids": [
                result["source_candidate_id"] for result in apply_results
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid executor dry-run records before apply."
                if invalid_record_ids
                else (
                    "Provide explicit operator acknowledgement and apply=true."
                    if records and not apply
                    else (
                        "Provide a target trust store before applying promotion."
                        if apply and operator_ack and not target_available
                        else (
                            "Promotion applied only to the explicit source identity trust store."
                            if applied
                            else "Dry-run records are required before memory trust promotion."
                        )
                    )
                )
            ),
        },
        "apply_results": list(apply_results),
        "updated_trust_store": updated_store,
    }
