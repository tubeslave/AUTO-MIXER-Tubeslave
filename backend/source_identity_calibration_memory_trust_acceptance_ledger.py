"""Read-only ledger for operator acceptance of memory-trust review candidates.

The ledger records accepted/rejected operator decisions against source identity
memory trust-review candidates. It does not promote source identity trust,
change downstream proposal priority, mutate models or memory, update queues, or
touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from source_identity_calibration_memory_trust_review import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_REVIEW_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_REVIEW_STATUSES,
        build_source_identity_calibration_memory_trust_review,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_calibration_memory_trust_review import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_REVIEW_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_REVIEW_STATUSES,
        build_source_identity_calibration_memory_trust_review,
    )


SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_ACCEPTANCE_LEDGER_METHOD_ID = (
    "source_identity_calibration_memory_trust_acceptance_ledger"
)
SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_ACCEPTANCE_LEDGER_STATUSES = {
    "operator_memory_trust_acceptance_required",
    "operator_memory_trust_acceptance_records_available",
    "operator_memory_trust_acceptance_replacement_required",
}
MEMORY_TRUST_ACCEPTANCE_DECISIONS = {"accepted", "rejected"}
REQUIRED_MEMORY_TRUST_REVIEW_TYPE = "source_identity_calibration_memory_trust_review"
REQUIRED_ACCEPTANCE_RECORD_FIELDS = (
    "acceptance_id",
    "review_candidate_id",
    "memory_record_id",
    "source_candidate_id",
    "source_type",
    "recommended_trust_state",
    "decision",
    "reviewed_by",
    "reviewed_at",
)
REQUIRED_REVIEW_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "downstream_priority_mutation",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_REVIEW_FALSE_SUMMARY_FLAGS = (
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "downstream_priority_mutation",
    "live_authority_granted",
    "full_autonomous_ready",
)
UNSAFE_ACCEPTANCE_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "downstream_priority_mutation",
    "requested_change_allowed",
    "memory_trust_promotion_allowed_by_acceptance",
    "downstream_priority_change_allowed_by_acceptance",
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


def _record_list(value: Iterable[Mapping[str, Any]] | None) -> Tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_list(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _summary_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _string(value: Any) -> str:
    return value if isinstance(value, str) and value else ""


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


def _review_candidate_id(candidate: Mapping[str, Any], index: int) -> str:
    explicit = candidate.get("review_candidate_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    memory_record_id = candidate.get("memory_record_id")
    return f"{memory_record_id if isinstance(memory_record_id, str) and memory_record_id else 'unknown_memory_trust_candidate'}:{index}"


def _candidate_basic_violations(candidate: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    required_fields = (
        "review_candidate_id",
        "memory_record_id",
        "source_candidate_id",
        "source_review_id",
        "source_type",
        "recommended_trust_state",
        "reviewed_by",
        "reviewed_at",
    )
    for field in required_fields:
        if not isinstance(candidate.get(field), str) or not candidate.get(field):
            violations.append(f"memory_trust_acceptance_candidate_{field}_missing")
    reviewed_at = candidate.get("reviewed_at")
    if isinstance(reviewed_at, str) and reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append("memory_trust_acceptance_candidate_reviewed_at_invalid")
    if candidate.get("operator_review_required") is not True:
        violations.append("memory_trust_acceptance_candidate_operator_review_not_true")
    if candidate.get("memory_trust_review_state") != "operator_review_required":
        violations.append("memory_trust_acceptance_candidate_review_state_invalid")
    false_flags = (
        "queue_mutation_allowed_by_review",
        "model_mutation_allowed_by_review",
        "runtime_memory_mutation_allowed_by_review",
        "downstream_priority_change_allowed_by_review",
        "live_authority_granted",
        "full_autonomous_ready",
    )
    for flag in false_flags:
        if candidate.get(flag) is not False:
            violations.append(f"memory_trust_acceptance_candidate_{flag}_not_false")
    return tuple(violations)


def _review_envelope_violations(review: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if review.get("type") != REQUIRED_MEMORY_TRUST_REVIEW_TYPE:
        violations.append("memory_trust_review_envelope_type_mismatch")
    generated_at = review.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("memory_trust_review_envelope_generated_at_invalid")
    if review.get("method_id") != SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_REVIEW_METHOD_ID:
        violations.append("memory_trust_review_envelope_method_id_mismatch")
    if review.get("status") not in SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_REVIEW_STATUSES:
        violations.append("memory_trust_review_envelope_status_unknown")
    safety = _mapping(review.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("memory_trust_review_envelope_read_only_not_true")
    for flag in REQUIRED_REVIEW_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"memory_trust_review_envelope_{flag}_not_false")
    return tuple(violations)


def _review_summary_violations(
    summary: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
    review_invalid_record_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if _summary_int(summary.get("trust_review_candidate_count")) != len(candidates):
        violations.append("memory_trust_review_summary_candidate_count_mismatch")
    if _summary_int(summary.get("invalid_record_count")) != len(review_invalid_record_ids):
        violations.append("memory_trust_review_summary_invalid_record_count_mismatch")
    if summary.get("operator_review_required") is not True:
        violations.append("memory_trust_review_summary_operator_review_required_not_true")
    for flag in REQUIRED_REVIEW_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(f"memory_trust_review_summary_{flag}_not_false")
    return tuple(violations)


def _candidates_by_id(memory_trust_review: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    candidates: Dict[str, Mapping[str, Any]] = {}
    for index, candidate in enumerate(_record_list(memory_trust_review.get("trust_review_candidates"))):
        candidate_id = _review_candidate_id(candidate, index)
        candidates[candidate_id] = candidate
    return candidates


def _acceptance_record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("acceptance_id"))
    if explicit:
        return explicit
    candidate_id = _string(record.get("review_candidate_id")) or "unknown_memory_trust_candidate"
    return f"{candidate_id}:{index}"


def _unsafe_flags(record: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(flag for flag in UNSAFE_ACCEPTANCE_FLAGS if record.get(flag) is True)


def _validated_record(
    record: Mapping[str, Any],
    *,
    index: int,
    candidates_by_candidate_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    record_id = _acceptance_record_id(record, index)
    review_candidate_id = _string(record.get("review_candidate_id"))
    memory_record_id = _string(record.get("memory_record_id"))
    source_candidate_id = _string(record.get("source_candidate_id"))
    source_type = _string(record.get("source_type"))
    recommended_trust_state = _string(record.get("recommended_trust_state"))
    decision = _string(record.get("decision"))
    reviewed_by = _string(record.get("reviewed_by"))
    reviewed_at = _string(record.get("reviewed_at"))
    unsafe_flags = _unsafe_flags(record)
    candidate = _mapping(candidates_by_candidate_id.get(review_candidate_id))
    violations = []

    for field in REQUIRED_ACCEPTANCE_RECORD_FIELDS:
        if not _string(record.get(field)):
            violations.append(f"memory_trust_acceptance_record_{field}_missing")
    if review_candidate_id not in candidates_by_candidate_id:
        violations.append("memory_trust_acceptance_record_candidate_missing")
    if decision not in MEMORY_TRUST_ACCEPTANCE_DECISIONS:
        violations.append("memory_trust_acceptance_record_decision_unknown")
    if reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append("memory_trust_acceptance_record_reviewed_at_invalid")
    if candidate:
        expected_pairs = {
            "memory_record_id": memory_record_id,
            "source_candidate_id": source_candidate_id,
            "source_type": source_type,
            "recommended_trust_state": recommended_trust_state,
        }
        for field, supplied in expected_pairs.items():
            if supplied != candidate.get(field):
                violations.append(f"memory_trust_acceptance_record_{field}_mismatch")
    if unsafe_flags:
        violations.append("memory_trust_acceptance_record_unsafe_authority_claim")

    valid = not violations
    return {
        "acceptance_id": record_id,
        "review_candidate_id": review_candidate_id,
        "memory_record_id": memory_record_id,
        "source_candidate_id": source_candidate_id,
        "source_type": source_type,
        "recommended_trust_state": recommended_trust_state,
        "decision": decision if decision in MEMORY_TRUST_ACCEPTANCE_DECISIONS else "invalid",
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at,
        "candidate_present": bool(candidate),
        "operator_accepted_memory_trust_candidate": valid and decision == "accepted",
        "operator_rejected_memory_trust_candidate": valid and decision == "rejected",
        "memory_trust_promotion_allowed_by_acceptance": False,
        "downstream_priority_change_allowed_by_acceptance": False,
        "queue_mutation_allowed_by_acceptance": False,
        "model_mutation_allowed_by_acceptance": False,
        "runtime_memory_mutation_allowed_by_acceptance": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "unsafe_authority_flags": list(unsafe_flags),
        "valid": valid,
        "violations": violations,
    }


def build_source_identity_calibration_memory_trust_acceptance_ledger(
    *,
    memory_trust_review: Mapping[str, Any] | None = None,
    acceptance_records: Iterable[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build a read-only ledger for memory-trust review decisions."""

    review = _mapping(memory_trust_review) or build_source_identity_calibration_memory_trust_review()
    summary = _mapping(review.get("summary"))
    review_candidates = _record_list(review.get("trust_review_candidates"))
    review_invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    review_violations = tuple(
        dict.fromkeys(
            (
                *review_invalid_record_ids,
                *_review_envelope_violations(review),
                *_review_summary_violations(
                    summary,
                    candidates=review_candidates,
                    review_invalid_record_ids=review_invalid_record_ids,
                ),
                *(
                    item
                    for index, candidate in enumerate(review_candidates)
                    for item in (
                        (_review_candidate_id(candidate, index), *_candidate_basic_violations(candidate))
                        if _candidate_basic_violations(candidate)
                        else ()
                    )
                ),
            )
        )
    )
    candidates_by_candidate_id = _candidates_by_id(review)
    records = tuple(
        _validated_record(
            record,
            index=index,
            candidates_by_candidate_id=candidates_by_candidate_id,
        )
        for index, record in enumerate(_record_list(acceptance_records))
    )
    valid_records = tuple(record for record in records if record["valid"])
    accepted_records = tuple(
        record for record in valid_records if record["operator_accepted_memory_trust_candidate"]
    )
    rejected_records = tuple(
        record for record in valid_records if record["operator_rejected_memory_trust_candidate"]
    )
    invalid_records = tuple(record for record in records if not record["valid"])
    reviewed_candidate_ids = {
        str(record["review_candidate_id"])
        for record in valid_records
        if isinstance(record.get("review_candidate_id"), str)
    }
    unreviewed_candidate_ids = tuple(
        candidate_id
        for candidate_id in candidates_by_candidate_id
        if candidate_id not in reviewed_candidate_ids
    )

    if review_violations or invalid_records:
        status = "operator_memory_trust_acceptance_replacement_required"
        blocked_by = ("operator_must_replace_invalid_memory_trust_acceptance_records",)
    elif unreviewed_candidate_ids:
        status = "operator_memory_trust_acceptance_required"
        blocked_by = ("operator_must_review_memory_trust_candidates_before_promotion",)
    elif accepted_records or rejected_records:
        status = "operator_memory_trust_acceptance_records_available"
        blocked_by = ("memory_trust_promotion_still_requires_separate_contract",)
    else:
        status = "operator_memory_trust_acceptance_required"
        blocked_by = ("valid_memory_trust_review_candidates_required_before_acceptance",)

    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *review_violations,
                *(record["acceptance_id"] for record in invalid_records),
            )
        )
    )
    accepted_source_types = [record["source_type"] for record in accepted_records]
    rejected_source_types = [record["source_type"] for record in rejected_records]
    return {
        "type": "source_identity_calibration_memory_trust_acceptance_ledger",
        "generated_at": _utc_now(),
        "method_id": SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_ACCEPTANCE_LEDGER_METHOD_ID,
        "source_method_id": SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_REVIEW_METHOD_ID,
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
            "downstream_priority_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "memory_trust_review_status": review.get("status") or "missing",
            "trust_review_candidate_count": len(review_candidates),
            "acceptance_record_count": len(records),
            "accepted_record_count": len(accepted_records),
            "rejected_record_count": len(rejected_records),
            "invalid_record_count": len(invalid_record_ids),
            "unreviewed_candidate_count": len(unreviewed_candidate_ids),
            "accepted_source_types": accepted_source_types,
            "rejected_source_types": rejected_source_types,
            "operator_review_required": True,
            "memory_trust_promotion_allowed_by_acceptance": False,
            "downstream_priority_change_allowed_by_acceptance": False,
            "queue_mutation_allowed_by_acceptance": False,
            "model_mutation_allowed_by_acceptance": False,
            "runtime_memory_mutation_allowed_by_acceptance": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "invalid_record_ids": list(invalid_record_ids),
            "unreviewed_candidate_ids": list(unreviewed_candidate_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid memory-trust acceptance records."
                if invalid_record_ids
                else (
                    "Operator reviews all memory-trust candidates before promotion."
                    if unreviewed_candidate_ids
                    else (
                        "Memory-trust acceptance is evidence only; separate "
                        "promotion is required before downstream priority changes."
                    )
                )
            ),
        },
        "records": list(records),
    }
