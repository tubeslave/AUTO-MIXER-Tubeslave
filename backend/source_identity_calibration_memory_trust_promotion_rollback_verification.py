"""Rollback verification gate for source-identity memory trust promotion.

The verifier consumes apply-audit records and operator supplied rollback
verification records. It only produces read-only verification evidence for a
future downstream-priority gate; it does not mutate trust, queues, models,
proposal priority, or live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from source_identity_calibration_memory_trust_promotion_apply_audit import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_STATUSES,
        build_source_identity_calibration_memory_trust_promotion_apply_audit,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_calibration_memory_trust_promotion_apply_audit import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_STATUSES,
        build_source_identity_calibration_memory_trust_promotion_apply_audit,
    )


SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_METHOD_ID = (
    "source_identity_calibration_memory_trust_promotion_rollback_verification"
)
SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_STATUSES = {
    "memory_trust_promotion_rollback_verification_not_ready",
    "memory_trust_promotion_rollback_verification_replacement_required",
    "operator_memory_trust_promotion_rollback_verified_records_available",
}
REQUIRED_APPLY_AUDIT_TYPE = (
    "source_identity_calibration_memory_trust_promotion_apply_audit"
)
TARGET_STORE = "source_identity_downstream_gate"
REQUIRED_APPLY_AUDIT_RECORD_FIELDS = (
    "promotion_apply_audit_id",
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
    "rollback_target_trust_state",
    "idempotency_key",
)
REQUIRED_VERIFICATION_RECORD_FIELDS = (
    "rollback_verification_id",
    "promotion_apply_audit_id",
    "promotion_executor_apply_id",
    "source_candidate_id",
    "source_type",
    "target_store",
    "target_path",
    "target_trust_state",
    "rollback_target_trust_state",
    "rollback_strategy",
    "verification_evidence_status",
    "decision",
    "verified_by",
    "verified_at",
)
REQUIRED_AUDIT_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "source_identity_trust_store_mutation",
    "direct_downstream_priority_mutation",
    "downstream_priority_change_allowed_by_audit",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_AUDIT_FALSE_SUMMARY_FLAGS = (
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "source_identity_trust_store_mutation",
    "direct_downstream_priority_mutation",
    "downstream_priority_change_allowed_by_audit",
    "live_write",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_AUDIT_RECORD_FALSE_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "direct_downstream_priority_mutation",
    "downstream_priority_change_allowed_by_audit",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_VERIFICATION_FALSE_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "source_identity_trust_store_mutation",
    "direct_downstream_priority_mutation",
    "downstream_priority_change_allowed_by_verification",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
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


def _freshness_status(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status") or "missing").strip().lower() or "missing"


def _freshness_violations(payload: Mapping[str, Any]) -> Tuple[str, ...]:
    if not payload:
        return (
            "memory_trust_promotion_rollback_verification_record_freshness_missing",
        )
    violations = []
    if _freshness_status(payload) != "fresh":
        violations.append(
            "memory_trust_promotion_rollback_verification_record_freshness_not_fresh"
        )
    if payload.get("read_only") is not True:
        violations.append(
            "memory_trust_promotion_rollback_verification_record_freshness_read_only_not_true"
        )
    if payload.get("live_write") is not False:
        violations.append(
            "memory_trust_promotion_rollback_verification_record_freshness_live_write_not_false"
        )
    if payload.get("queries_console") is not False:
        violations.append(
            "memory_trust_promotion_rollback_verification_record_freshness_queries_console_not_false"
        )
    return tuple(violations)


def _record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("promotion_apply_audit_id"))
    if explicit:
        return explicit
    source_id = _string(record.get("source_candidate_id"))
    return f"{source_id if source_id else 'unknown_apply_audit'}:{index}"


def _verification_record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("rollback_verification_id"))
    if explicit:
        return explicit
    apply_id = _string(record.get("promotion_apply_audit_id"))
    return f"{apply_id if apply_id else 'unknown_rollback_verification'}:{index}"


def _target_path(source_candidate_id: str) -> str:
    return f"{TARGET_STORE}/{source_candidate_id}/trust_state"


def _apply_audit_envelope_violations(apply_audit: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if apply_audit.get("type") != REQUIRED_APPLY_AUDIT_TYPE:
        violations.append("memory_trust_promotion_rollback_verification_audit_type_mismatch")
    generated_at = apply_audit.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append(
            "memory_trust_promotion_rollback_verification_audit_generated_at_invalid"
        )
    if (
        apply_audit.get("method_id")
        != SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID
    ):
        violations.append("memory_trust_promotion_rollback_verification_audit_method_id_mismatch")
    if (
        apply_audit.get("status")
        not in SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_STATUSES
    ):
        violations.append("memory_trust_promotion_rollback_verification_audit_status_unknown")
    safety = _mapping(apply_audit.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("memory_trust_promotion_rollback_verification_audit_read_only_not_true")
    if safety.get("rollback_verified") is not False:
        violations.append(
            "memory_trust_promotion_rollback_verification_audit_rollback_verified_not_false"
        )
    for flag in REQUIRED_AUDIT_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"memory_trust_promotion_rollback_verification_audit_{flag}_not_false"
            )
    return tuple(violations)


def _apply_audit_status_violations(
    apply_audit: Mapping[str, Any],
    *,
    audit_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = apply_audit.get("status")
    if status == "memory_trust_promotion_apply_audit_replacement_required":
        return ("memory_trust_promotion_rollback_verification_audit_replacement_required",)
    if status == "memory_trust_promotion_apply_audit_records_available" and not audit_records:
        return ("memory_trust_promotion_rollback_verification_audit_available_without_records",)
    if status != "memory_trust_promotion_apply_audit_records_available" and audit_records:
        return ("memory_trust_promotion_rollback_verification_audit_records_without_available_status",)
    return ()


def _apply_audit_summary_violations(
    summary: Mapping[str, Any],
    *,
    audit_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    expected_counts = {
        "apply_audit_record_count": len(audit_records),
        "invalid_record_count": 0,
    }
    violations = [
        f"memory_trust_promotion_rollback_verification_audit_summary_{key}_mismatch"
        for key, expected in expected_counts.items()
        if _summary_int(summary.get(key)) != expected
    ]
    if summary.get("rollback_verified") is not False:
        violations.append(
            "memory_trust_promotion_rollback_verification_audit_summary_rollback_verified_not_false"
        )
    if summary.get("target_store") != TARGET_STORE:
        violations.append(
            "memory_trust_promotion_rollback_verification_audit_summary_target_store_invalid"
        )
    for flag in REQUIRED_AUDIT_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"memory_trust_promotion_rollback_verification_audit_summary_{flag}_not_false"
            )
    if audit_records:
        expected_source_ids = tuple(
            record["source_candidate_id"]
            for record in audit_records
            if _string(record.get("source_candidate_id"))
        )
        expected_rollback_states = tuple(
            record["rollback_target_trust_state"]
            for record in audit_records
            if _string(record.get("rollback_target_trust_state"))
        )
        expected_paths = tuple(
            record["target_path"]
            for record in audit_records
            if _string(record.get("target_path"))
        )
        if _string_list(summary.get("audited_source_candidate_ids")) != expected_source_ids:
            violations.append(
                "memory_trust_promotion_rollback_verification_audit_summary_source_ids_mismatch"
            )
        if (
            _string_list(summary.get("rollback_target_trust_states"))
            != expected_rollback_states
        ):
            violations.append(
                "memory_trust_promotion_rollback_verification_audit_summary_rollback_states_mismatch"
            )
        if _string_list(summary.get("target_paths")) != expected_paths:
            violations.append(
                "memory_trust_promotion_rollback_verification_audit_summary_paths_mismatch"
            )
    return tuple(violations)


def _apply_audit_record_violations(record: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    source_candidate_id = _string(record.get("source_candidate_id"))
    for field in REQUIRED_APPLY_AUDIT_RECORD_FIELDS:
        if not _string(record.get(field)):
            violations.append(f"memory_trust_promotion_rollback_verification_audit_record_{field}_missing")
    if record.get("target_store") != TARGET_STORE:
        violations.append(
            "memory_trust_promotion_rollback_verification_audit_record_target_store_invalid"
        )
    if record.get("target_path") != _target_path(source_candidate_id):
        violations.append(
            "memory_trust_promotion_rollback_verification_audit_record_target_path_invalid"
        )
    if record.get("operation") != "set_source_identity_trust_state":
        violations.append(
            "memory_trust_promotion_rollback_verification_audit_record_operation_invalid"
        )
    if record.get("applied") is not True:
        violations.append("memory_trust_promotion_rollback_verification_audit_record_applied_not_true")
    if record.get("rollback_evidence_required") is not True:
        violations.append(
            "memory_trust_promotion_rollback_verification_audit_record_rollback_required_not_true"
        )
    if record.get("rollback_verified") is not False:
        violations.append(
            "memory_trust_promotion_rollback_verification_audit_record_rollback_verified_not_false"
        )
    if record.get("source_identity_trust_store_mutation_audited") is not True:
        violations.append(
            "memory_trust_promotion_rollback_verification_audit_record_trust_store_audited_not_true"
        )
    if record.get("read_only") is not True:
        violations.append(
            "memory_trust_promotion_rollback_verification_audit_record_read_only_not_true"
        )
    for flag in REQUIRED_AUDIT_RECORD_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(
                f"memory_trust_promotion_rollback_verification_audit_record_{flag}_not_false"
            )
    return tuple(violations)


def _audit_record_replacement_ids(
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    return tuple(
        _record_id(record, index)
        for index, record in enumerate(records)
        if _apply_audit_record_violations(record)
    )


def _duplicate_ids(
    records: Tuple[Mapping[str, Any], ...],
    field: str,
    *,
    id_func,
) -> Tuple[str, ...]:
    values: Dict[str, list[str]] = {}
    for index, record in enumerate(records):
        value = _string(record.get(field))
        if not value:
            continue
        values.setdefault(value, []).append(id_func(record, index))
    return tuple(
        record_id
        for record_ids in values.values()
        if len(record_ids) > 1
        for record_id in record_ids
    )


def _verification_record_violations(
    record: Mapping[str, Any],
    audit_records_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[str, ...]:
    violations = []
    apply_audit_id = _string(record.get("promotion_apply_audit_id"))
    audit_record = audit_records_by_id.get(apply_audit_id, {})
    for field in REQUIRED_VERIFICATION_RECORD_FIELDS:
        if not _string(record.get(field)):
            violations.append(
                f"memory_trust_promotion_rollback_verification_record_{field}_missing"
            )
    verified_at = _string(record.get("verified_at"))
    if verified_at and not _is_utc_timestamp(verified_at):
        violations.append("memory_trust_promotion_rollback_verification_record_verified_at_invalid")
    if not audit_record:
        violations.append("memory_trust_promotion_rollback_verification_record_audit_missing")
    else:
        fields_to_match = (
            "promotion_executor_apply_id",
            "source_candidate_id",
            "source_type",
            "target_store",
            "target_path",
            "target_trust_state",
            "rollback_target_trust_state",
            "rollback_strategy",
        )
        for field in fields_to_match:
            if record.get(field) != audit_record.get(field):
                violations.append(
                    f"memory_trust_promotion_rollback_verification_record_{field}_mismatch"
                )
    if record.get("decision") != "verified":
        violations.append("memory_trust_promotion_rollback_verification_record_decision_not_verified")
    if record.get("verification_evidence_status") != "reviewed":
        violations.append(
            "memory_trust_promotion_rollback_verification_record_evidence_status_not_reviewed"
        )
    if record.get("rollback_verified") is not True:
        violations.append("memory_trust_promotion_rollback_verification_record_rollback_verified_not_true")
    if record.get("read_only") is not True:
        violations.append("memory_trust_promotion_rollback_verification_record_read_only_not_true")
    if record.get("operator_review_required") is not True:
        violations.append(
            "memory_trust_promotion_rollback_verification_record_operator_review_not_true"
        )
    if record.get("separate_downstream_priority_gate_required") is not True:
        violations.append(
            "memory_trust_promotion_rollback_verification_record_separate_priority_gate_not_true"
        )
    violations.extend(_freshness_violations(_mapping(record.get("current_state_cache_freshness"))))
    for flag in REQUIRED_VERIFICATION_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(
                f"memory_trust_promotion_rollback_verification_record_{flag}_not_false"
            )
    return tuple(violations)


def _verification_record_replacement_ids(
    records: Tuple[Mapping[str, Any], ...],
    audit_records_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[str, ...]:
    return tuple(
        _verification_record_id(record, index)
        for index, record in enumerate(records)
        if _verification_record_violations(record, audit_records_by_id)
    )


def _unverified_audit_record_ids(
    audit_records: Tuple[Mapping[str, Any], ...],
    verification_records: Tuple[Mapping[str, Any], ...],
    invalid_verification_record_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    invalid_ids = set(invalid_verification_record_ids)
    verified_audit_ids = {
        record["promotion_apply_audit_id"]
        for index, record in enumerate(verification_records)
        if _verification_record_id(record, index) not in invalid_ids
        and _string(record.get("promotion_apply_audit_id"))
    }
    return tuple(
        record["promotion_apply_audit_id"]
        for record in audit_records
        if _string(record.get("promotion_apply_audit_id"))
        and record["promotion_apply_audit_id"] not in verified_audit_ids
    )


def _verified_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "rollback_verification_record_id": (
            "source_identity_memory_trust_promotion_rollback_verification:"
            f"{record.get('promotion_apply_audit_id')}"
        ),
        "rollback_verification_id": record.get("rollback_verification_id"),
        "promotion_apply_audit_id": record.get("promotion_apply_audit_id"),
        "promotion_executor_apply_id": record.get("promotion_executor_apply_id"),
        "source_candidate_id": record.get("source_candidate_id"),
        "source_type": record.get("source_type"),
        "target_store": TARGET_STORE,
        "target_path": record.get("target_path"),
        "target_trust_state": record.get("target_trust_state"),
        "rollback_target_trust_state": record.get("rollback_target_trust_state"),
        "rollback_strategy": record.get("rollback_strategy"),
        "verification_evidence_status": record.get("verification_evidence_status"),
        "current_state_cache_freshness": record.get("current_state_cache_freshness"),
        "decision": "verified",
        "verified_by": record.get("verified_by"),
        "verified_at": record.get("verified_at"),
        "rollback_verified": True,
        "read_only": True,
        "operator_review_required": True,
        "separate_downstream_priority_gate_required": True,
        "usable_as_downstream_priority_gate_evidence": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "queue_mutation": False,
        "model_mutation": False,
        "runtime_memory_mutation": False,
        "source_identity_trust_store_mutation": False,
        "direct_downstream_priority_mutation": False,
        "downstream_priority_change_allowed_by_verification": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "can_import_to_queue": False,
        "can_apply_live": False,
    }


def build_source_identity_calibration_memory_trust_promotion_rollback_verification(
    *,
    apply_audit: Mapping[str, Any] | None = None,
    verification_records: Tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build read-only rollback verification evidence for applied trust promotion."""

    audit = (
        _mapping(apply_audit)
        or build_source_identity_calibration_memory_trust_promotion_apply_audit()
    )
    summary = _mapping(audit.get("summary"))
    audit_records = _record_list(audit.get("audit_records"))
    raw_verification_records = _record_list(verification_records)
    audit_invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    audit_records_by_id = {
        record["promotion_apply_audit_id"]: record
        for record in audit_records
        if _string(record.get("promotion_apply_audit_id"))
        and not _apply_audit_record_violations(record)
    }
    invalid_verification_record_ids = _verification_record_replacement_ids(
        raw_verification_records,
        audit_records_by_id,
    )
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *audit_invalid_record_ids,
                *_apply_audit_envelope_violations(audit),
                *_apply_audit_status_violations(audit, audit_records=audit_records),
                *_apply_audit_summary_violations(summary, audit_records=audit_records),
                *_audit_record_replacement_ids(audit_records),
                *_duplicate_ids(audit_records, "promotion_apply_audit_id", id_func=_record_id),
                *_duplicate_ids(audit_records, "source_candidate_id", id_func=_record_id),
                *_duplicate_ids(
                    raw_verification_records,
                    "rollback_verification_id",
                    id_func=_verification_record_id,
                ),
                *_duplicate_ids(
                    raw_verification_records,
                    "promotion_apply_audit_id",
                    id_func=_verification_record_id,
                ),
                *invalid_verification_record_ids,
            )
        )
    )
    unverified_audit_record_ids = (
        ()
        if invalid_record_ids
        else _unverified_audit_record_ids(
            audit_records,
            raw_verification_records,
            invalid_verification_record_ids,
        )
    )
    verified_records = (
        ()
        if invalid_record_ids or unverified_audit_record_ids
        else tuple(_verified_record(record) for record in raw_verification_records)
    )

    if invalid_record_ids:
        status = "memory_trust_promotion_rollback_verification_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_rollback_verification_evidence",
        )
    elif unverified_audit_record_ids:
        status = "memory_trust_promotion_rollback_verification_not_ready"
        blocked_by = (
            "operator_must_verify_all_apply_audit_records_before_priority_gate",
        )
    elif verified_records:
        status = "operator_memory_trust_promotion_rollback_verified_records_available"
        blocked_by = (
            "rollback_verification_is_not_downstream_priority_or_live_authority",
        )
    else:
        status = "memory_trust_promotion_rollback_verification_not_ready"
        blocked_by = (
            "apply_audit_records_required_before_rollback_verification",
        )

    return {
        "type": "source_identity_calibration_memory_trust_promotion_rollback_verification",
        "generated_at": _utc_now(),
        "method_id": (
            SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_METHOD_ID
        ),
        "source_method_id": (
            SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID
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
            "direct_downstream_priority_mutation": False,
            "downstream_priority_change_allowed_by_verification": False,
            "rollback_verified": bool(verified_records),
            "usable_as_downstream_priority_gate_evidence": bool(verified_records),
            "separate_downstream_priority_gate_required": True,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "apply_audit_status": audit.get("status") or "missing",
            "apply_audit_record_count": len(audit_records),
            "verification_record_count": len(raw_verification_records),
            "rollback_verified_record_count": len(verified_records),
            "invalid_record_count": len(invalid_record_ids),
            "unverified_apply_audit_record_count": len(unverified_audit_record_ids),
            "target_store": TARGET_STORE,
            "rollback_verified": bool(verified_records),
            "usable_as_downstream_priority_gate_evidence": bool(verified_records),
            "separate_downstream_priority_gate_required": True,
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "source_identity_trust_store_mutation": False,
            "direct_downstream_priority_mutation": False,
            "downstream_priority_change_allowed_by_verification": False,
            "live_write": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "verified_source_candidate_ids": [
                record["source_candidate_id"] for record in verified_records
            ],
            "verified_rollback_target_trust_states": [
                record["rollback_target_trust_state"] for record in verified_records
            ],
            "unverified_apply_audit_record_ids": list(unverified_audit_record_ids),
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid rollback verification evidence."
                if invalid_record_ids
                else (
                    "Operator verifies every apply-audit record before priority-gate review."
                    if unverified_audit_record_ids
                    else (
                        "Use verified rollback evidence only as input to a separate downstream-priority gate."
                        if verified_records
                        else "Apply-audit records are required before rollback verification."
                    )
                )
            ),
        },
        "verified_records": list(verified_records),
    }
