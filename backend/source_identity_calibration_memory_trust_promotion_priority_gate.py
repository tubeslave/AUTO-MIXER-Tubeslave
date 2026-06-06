"""Downstream-priority policy gate for verified source-identity trust promotion.

The gate consumes operator-verified rollback evidence and emits read-only
eligibility records for a future proposal-priority consumer. It never mutates
proposal order, queues, models, runtime memory, source-identity trust stores,
or live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from source_identity_calibration_memory_trust_promotion_rollback_verification import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_STATUSES,
        build_source_identity_calibration_memory_trust_promotion_rollback_verification,
    )
    from source_identity_downstream_gate import evaluate_source_identity_downstream_trust
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_calibration_memory_trust_promotion_rollback_verification import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_STATUSES,
        build_source_identity_calibration_memory_trust_promotion_rollback_verification,
    )
    from backend.source_identity_downstream_gate import (
        evaluate_source_identity_downstream_trust,
    )


SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PRIORITY_GATE_METHOD_ID = (
    "source_identity_calibration_memory_trust_promotion_priority_gate"
)
SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PRIORITY_GATE_STATUSES = {
    "memory_trust_promotion_priority_gate_not_ready",
    "memory_trust_promotion_priority_gate_replacement_required",
    "verified_memory_trust_promotion_priority_records_available",
}
REQUIRED_ROLLBACK_VERIFICATION_TYPE = (
    "source_identity_calibration_memory_trust_promotion_rollback_verification"
)
TARGET_STORE = "source_identity_downstream_gate"
REQUIRED_VERIFIED_RECORD_FIELDS = (
    "rollback_verification_record_id",
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
REQUIRED_ROLLBACK_FALSE_SAFETY_FLAGS = (
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
)
REQUIRED_ROLLBACK_FALSE_SUMMARY_FLAGS = (
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "source_identity_trust_store_mutation",
    "direct_downstream_priority_mutation",
    "downstream_priority_change_allowed_by_verification",
    "live_write",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_VERIFIED_RECORD_FALSE_FLAGS = (
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


def _target_path(source_candidate_id: str) -> str:
    return f"{TARGET_STORE}/{source_candidate_id}/trust_state"


def _record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("rollback_verification_record_id"))
    if explicit:
        return explicit
    apply_id = _string(record.get("promotion_apply_audit_id"))
    return f"{apply_id if apply_id else 'unknown_priority_gate_source'}:{index}"


def _duplicate_ids(
    records: Tuple[Mapping[str, Any], ...],
    field: str,
) -> Tuple[str, ...]:
    values: Dict[str, list[str]] = {}
    for index, record in enumerate(records):
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


def _rollback_envelope_violations(
    rollback_verification: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    if rollback_verification.get("type") != REQUIRED_ROLLBACK_VERIFICATION_TYPE:
        violations.append("memory_trust_promotion_priority_gate_rollback_type_mismatch")
    generated_at = rollback_verification.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("memory_trust_promotion_priority_gate_rollback_generated_at_invalid")
    if (
        rollback_verification.get("method_id")
        != SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_METHOD_ID
    ):
        violations.append("memory_trust_promotion_priority_gate_rollback_method_id_mismatch")
    if (
        rollback_verification.get("status")
        not in SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_STATUSES
    ):
        violations.append("memory_trust_promotion_priority_gate_rollback_status_unknown")
    safety = _mapping(rollback_verification.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("memory_trust_promotion_priority_gate_rollback_read_only_not_true")
    if safety.get("separate_downstream_priority_gate_required") is not True:
        violations.append(
            "memory_trust_promotion_priority_gate_rollback_separate_gate_not_true"
        )
    for flag in REQUIRED_ROLLBACK_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"memory_trust_promotion_priority_gate_rollback_{flag}_not_false")
    return tuple(violations)


def _rollback_status_violations(
    rollback_verification: Mapping[str, Any],
    *,
    verified_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = rollback_verification.get("status")
    if status == "memory_trust_promotion_rollback_verification_replacement_required":
        return ("memory_trust_promotion_priority_gate_rollback_replacement_required",)
    if (
        status == "operator_memory_trust_promotion_rollback_verified_records_available"
        and not verified_records
    ):
        return ("memory_trust_promotion_priority_gate_rollback_available_without_records",)
    if (
        status != "operator_memory_trust_promotion_rollback_verified_records_available"
        and verified_records
    ):
        return ("memory_trust_promotion_priority_gate_rollback_records_without_available_status",)
    return ()


def _rollback_summary_violations(
    summary: Mapping[str, Any],
    *,
    verified_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    expected_counts = {
        "rollback_verified_record_count": len(verified_records),
        "invalid_record_count": 0,
        "unverified_apply_audit_record_count": 0,
    }
    violations = [
        f"memory_trust_promotion_priority_gate_rollback_summary_{key}_mismatch"
        for key, expected in expected_counts.items()
        if _summary_int(summary.get(key)) != expected
    ]
    if summary.get("target_store") != TARGET_STORE:
        violations.append("memory_trust_promotion_priority_gate_rollback_summary_target_store_invalid")
    if summary.get("rollback_verified") is not bool(verified_records):
        violations.append(
            "memory_trust_promotion_priority_gate_rollback_summary_rollback_verified_mismatch"
        )
    if summary.get("usable_as_downstream_priority_gate_evidence") is not bool(verified_records):
        violations.append(
            "memory_trust_promotion_priority_gate_rollback_summary_priority_evidence_mismatch"
        )
    if summary.get("separate_downstream_priority_gate_required") is not True:
        violations.append(
            "memory_trust_promotion_priority_gate_rollback_summary_separate_gate_not_true"
        )
    for flag in REQUIRED_ROLLBACK_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"memory_trust_promotion_priority_gate_rollback_summary_{flag}_not_false"
            )
    if verified_records:
        expected_source_ids = tuple(
            record["source_candidate_id"]
            for record in verified_records
            if _string(record.get("source_candidate_id"))
        )
        if _string_list(summary.get("verified_source_candidate_ids")) != expected_source_ids:
            violations.append(
                "memory_trust_promotion_priority_gate_rollback_summary_source_ids_mismatch"
            )
    return tuple(violations)


def _verified_record_violations(record: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    source_candidate_id = _string(record.get("source_candidate_id"))
    for field in REQUIRED_VERIFIED_RECORD_FIELDS:
        if not _string(record.get(field)):
            violations.append(f"memory_trust_promotion_priority_gate_record_{field}_missing")
    verified_at = _string(record.get("verified_at"))
    if verified_at and not _is_utc_timestamp(verified_at):
        violations.append("memory_trust_promotion_priority_gate_record_verified_at_invalid")
    if record.get("target_store") != TARGET_STORE:
        violations.append("memory_trust_promotion_priority_gate_record_target_store_invalid")
    if record.get("target_path") != _target_path(source_candidate_id):
        violations.append("memory_trust_promotion_priority_gate_record_target_path_invalid")
    if record.get("decision") != "verified":
        violations.append("memory_trust_promotion_priority_gate_record_decision_not_verified")
    if record.get("verification_evidence_status") != "reviewed":
        violations.append("memory_trust_promotion_priority_gate_record_evidence_status_not_reviewed")
    if record.get("rollback_verified") is not True:
        violations.append("memory_trust_promotion_priority_gate_record_rollback_verified_not_true")
    if record.get("read_only") is not True:
        violations.append("memory_trust_promotion_priority_gate_record_read_only_not_true")
    if record.get("operator_review_required") is not True:
        violations.append("memory_trust_promotion_priority_gate_record_operator_review_not_true")
    if record.get("separate_downstream_priority_gate_required") is not True:
        violations.append("memory_trust_promotion_priority_gate_record_separate_gate_not_true")
    if record.get("usable_as_downstream_priority_gate_evidence") is not True:
        violations.append("memory_trust_promotion_priority_gate_record_priority_evidence_not_true")
    for flag in REQUIRED_VERIFIED_RECORD_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(f"memory_trust_promotion_priority_gate_record_{flag}_not_false")
    return tuple(violations)


def _invalid_verified_record_ids(
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    return tuple(
        _record_id(record, index)
        for index, record in enumerate(records)
        if _verified_record_violations(record)
    )


def _priority_gate_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    decision = evaluate_source_identity_downstream_trust(
        source_id=_string(record.get("source_candidate_id")),
        source_type=_string(record.get("source_type")),
        trust_state=record.get("target_trust_state"),
        trust_confidence=1.0,
        replay_outcome_count=1,
    )
    eligible = decision.usable_for_downstream and decision.status == "trusted"
    return {
        "priority_gate_record_id": (
            "source_identity_memory_trust_promotion_priority_gate:"
            f"{record.get('promotion_apply_audit_id')}"
        ),
        "rollback_verification_record_id": record.get("rollback_verification_record_id"),
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
        "rollback_verified": True,
        "source_identity_downstream_gate_status": decision.status,
        "source_identity_downstream_gate_usable": decision.usable_for_downstream,
        "source_identity_downstream_gate_blockers": list(decision.blockers),
        "source_identity_downstream_gate_evidence": decision.evidence,
        "downstream_priority_eligible": eligible,
        "eligible_for_downstream_priority_policy": eligible,
        "policy_gate_only": True,
        "operator_review_required": True,
        "proposal_priority_consumer_required": True,
        "downstream_priority_policy_consumer_required": True,
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "queue_mutation": False,
        "model_mutation": False,
        "runtime_memory_mutation": False,
        "source_identity_trust_store_mutation": False,
        "direct_downstream_priority_mutation": False,
        "downstream_priority_change_allowed_by_gate": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "can_import_to_queue": False,
        "can_apply_live": False,
    }


def build_source_identity_calibration_memory_trust_promotion_priority_gate(
    *,
    rollback_verification: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build read-only downstream-priority eligibility records."""

    verification = (
        _mapping(rollback_verification)
        or build_source_identity_calibration_memory_trust_promotion_rollback_verification()
    )
    summary = _mapping(verification.get("summary"))
    verified_records = _record_list(verification.get("verified_records"))
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_string_list(summary.get("invalid_record_ids")),
                *_rollback_envelope_violations(verification),
                *_rollback_status_violations(
                    verification,
                    verified_records=verified_records,
                ),
                *_rollback_summary_violations(
                    summary,
                    verified_records=verified_records,
                ),
                *_invalid_verified_record_ids(verified_records),
                *_duplicate_ids(verified_records, "rollback_verification_record_id"),
                *_duplicate_ids(verified_records, "promotion_apply_audit_id"),
                *_duplicate_ids(verified_records, "source_candidate_id"),
            )
        )
    )
    priority_records = (
        ()
        if invalid_record_ids
        else tuple(_priority_gate_record(record) for record in verified_records)
    )
    eligible_records = tuple(
        record for record in priority_records if record["downstream_priority_eligible"]
    )
    ineligible_records = tuple(
        record
        for record in priority_records
        if not record["downstream_priority_eligible"]
    )

    if invalid_record_ids:
        status = "memory_trust_promotion_priority_gate_replacement_required"
        blocked_by = ("operator_must_replace_invalid_priority_gate_evidence",)
    elif ineligible_records:
        status = "memory_trust_promotion_priority_gate_not_ready"
        blocked_by = ("source_identity_downstream_gate_blocked_records",)
    elif eligible_records:
        status = "verified_memory_trust_promotion_priority_records_available"
        blocked_by = (
            "priority_gate_records_are_not_queue_or_live_authority",
        )
    else:
        status = "memory_trust_promotion_priority_gate_not_ready"
        blocked_by = (
            "verified_rollback_records_required_before_priority_gate",
        )

    return {
        "type": "source_identity_calibration_memory_trust_promotion_priority_gate",
        "generated_at": _utc_now(),
        "method_id": (
            SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PRIORITY_GATE_METHOD_ID
        ),
        "source_method_id": (
            SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_METHOD_ID
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
            "downstream_priority_change_allowed_by_gate": False,
            "priority_gate_only": True,
            "downstream_priority_eligible": bool(eligible_records),
            "proposal_priority_consumer_required": bool(eligible_records),
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "rollback_verification_status": verification.get("status") or "missing",
            "rollback_verified_record_count": len(verified_records),
            "priority_gate_record_count": len(priority_records),
            "eligible_record_count": len(eligible_records),
            "ineligible_record_count": len(ineligible_records),
            "invalid_record_count": len(invalid_record_ids),
            "target_store": TARGET_STORE,
            "downstream_priority_eligible": bool(eligible_records),
            "policy_gate_only": True,
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "source_identity_trust_store_mutation": False,
            "direct_downstream_priority_mutation": False,
            "downstream_priority_change_allowed_by_gate": False,
            "live_write": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
            "eligible_source_candidate_ids": [
                record["source_candidate_id"] for record in eligible_records
            ],
            "ineligible_source_candidate_ids": [
                record["source_candidate_id"] for record in ineligible_records
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid priority-gate evidence."
                if invalid_record_ids
                else (
                    "Resolve blocked source identity downstream-gate decisions before policy use."
                    if ineligible_records
                    else (
                        "Route eligible records through a separate proposal-priority policy consumer; do not mutate queues or live state."
                        if eligible_records
                        else "Verified rollback records are required before priority-gate eligibility."
                    )
                )
            ),
        },
        "priority_records": list(priority_records),
    }
