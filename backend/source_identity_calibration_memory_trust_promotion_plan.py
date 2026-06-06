"""Read-only plan for manual source-identity memory trust promotion.

The plan turns accepted memory-trust ledger records into manual promotion
candidates. It does not update source identity trust, downstream proposal
priority, classifier models, runtime memory, queues, or live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Tuple

try:
    from source_identity_calibration_memory_trust_acceptance_ledger import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_ACCEPTANCE_LEDGER_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_ACCEPTANCE_LEDGER_STATUSES,
        build_source_identity_calibration_memory_trust_acceptance_ledger,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_calibration_memory_trust_acceptance_ledger import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_ACCEPTANCE_LEDGER_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_ACCEPTANCE_LEDGER_STATUSES,
        build_source_identity_calibration_memory_trust_acceptance_ledger,
    )


SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID = (
    "source_identity_calibration_memory_trust_promotion_plan"
)
SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PLAN_STATUSES = {
    "memory_trust_promotion_not_ready",
    "memory_trust_promotion_replacement_required",
    "manual_memory_trust_promotion_candidate_available",
}
REQUIRED_ACCEPTANCE_LEDGER_TYPE = (
    "source_identity_calibration_memory_trust_acceptance_ledger"
)
REQUIRED_ACCEPTANCE_RECORD_FIELDS = (
    "acceptance_id",
    "review_candidate_id",
    "memory_record_id",
    "source_candidate_id",
    "source_type",
    "recommended_trust_state",
    "reviewed_by",
    "reviewed_at",
)
REQUIRED_LEDGER_FALSE_SAFETY_FLAGS = (
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
REQUIRED_LEDGER_TRUE_SUMMARY_FLAGS = ("operator_review_required",)
REQUIRED_LEDGER_FALSE_SUMMARY_FLAGS = (
    "memory_trust_promotion_allowed_by_acceptance",
    "downstream_priority_change_allowed_by_acceptance",
    "queue_mutation_allowed_by_acceptance",
    "model_mutation_allowed_by_acceptance",
    "runtime_memory_mutation_allowed_by_acceptance",
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
    explicit = record.get("acceptance_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    candidate_id = record.get("review_candidate_id")
    return f"{candidate_id if isinstance(candidate_id, str) and candidate_id else 'unknown_memory_trust_acceptance'}:{index}"


def _ledger_envelope_violations(ledger: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if ledger.get("type") != REQUIRED_ACCEPTANCE_LEDGER_TYPE:
        violations.append("memory_trust_acceptance_ledger_type_mismatch")
    generated_at = ledger.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("memory_trust_acceptance_ledger_generated_at_invalid")
    if (
        ledger.get("method_id")
        != SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_ACCEPTANCE_LEDGER_METHOD_ID
    ):
        violations.append("memory_trust_acceptance_ledger_method_id_mismatch")
    if (
        ledger.get("status")
        not in SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_ACCEPTANCE_LEDGER_STATUSES
    ):
        violations.append("memory_trust_acceptance_ledger_status_unknown")
    safety = _mapping(ledger.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("memory_trust_acceptance_ledger_read_only_not_true")
    for flag in REQUIRED_LEDGER_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"memory_trust_acceptance_ledger_{flag}_not_false")
    return tuple(violations)


def _ledger_status_violations(
    ledger: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = ledger.get("status")
    if status == "operator_memory_trust_acceptance_replacement_required":
        return ("memory_trust_acceptance_status_replacement_required",)
    if status == "operator_memory_trust_acceptance_required" and records:
        return ("memory_trust_acceptance_status_required_with_records",)
    if status == "operator_memory_trust_acceptance_records_available" and not records:
        return ("memory_trust_acceptance_status_available_without_records",)
    return ()


def _accepted_record_violations(record: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    for field in REQUIRED_ACCEPTANCE_RECORD_FIELDS:
        if not isinstance(record.get(field), str) or not record.get(field):
            violations.append(f"memory_trust_promotion_record_{field}_missing")
    reviewed_at = record.get("reviewed_at")
    if isinstance(reviewed_at, str) and reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append("memory_trust_promotion_record_reviewed_at_invalid")
    if record.get("valid") is not True:
        violations.append("memory_trust_promotion_record_not_valid")
    if record.get("candidate_present") is not True:
        violations.append("memory_trust_promotion_record_candidate_missing")
    if record.get("decision") != "accepted":
        violations.append("memory_trust_promotion_record_decision_not_accepted")
    if record.get("operator_accepted_memory_trust_candidate") is not True:
        violations.append("memory_trust_promotion_record_accepted_flag_not_true")
    if record.get("operator_rejected_memory_trust_candidate") is not False:
        violations.append("memory_trust_promotion_record_rejected_flag_not_false")
    false_flags = (
        "memory_trust_promotion_allowed_by_acceptance",
        "downstream_priority_change_allowed_by_acceptance",
        "queue_mutation_allowed_by_acceptance",
        "model_mutation_allowed_by_acceptance",
        "runtime_memory_mutation_allowed_by_acceptance",
        "live_authority_granted",
        "full_autonomous_ready",
    )
    for flag in false_flags:
        if record.get(flag) is not False:
            violations.append(f"memory_trust_promotion_record_{flag}_not_false")
    return tuple(violations)


def _accepted_record_replacement_ids(
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    return tuple(
        _record_id(record, index)
        for index, record in enumerate(records)
        if record.get("operator_accepted_memory_trust_candidate") is True
        and _accepted_record_violations(record)
    )


def _ledger_summary_count_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
    invalid_record_ids: Tuple[str, ...],
    unreviewed_candidate_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    expected_counts = {
        "acceptance_record_count": len(records),
        "accepted_record_count": sum(
            1
            for record in records
            if record.get("valid") is True
            and record.get("operator_accepted_memory_trust_candidate") is True
        ),
        "rejected_record_count": sum(
            1
            for record in records
            if record.get("valid") is True
            and record.get("operator_rejected_memory_trust_candidate") is True
        ),
        "invalid_record_count": len(invalid_record_ids),
        "unreviewed_candidate_count": len(unreviewed_candidate_ids),
    }
    return tuple(
        f"memory_trust_acceptance_summary_{key}_mismatch"
        for key, expected in expected_counts.items()
        if _summary_int(summary.get(key)) != expected
    )


def _source_types_for_decision(
    records: Tuple[Mapping[str, Any]],
    *,
    accepted: bool,
) -> Tuple[str, ...]:
    flag = (
        "operator_accepted_memory_trust_candidate"
        if accepted
        else "operator_rejected_memory_trust_candidate"
    )
    return tuple(
        record["source_type"]
        for record in records
        if record.get("valid") is True
        and record.get(flag) is True
        and isinstance(record.get("source_type"), str)
        and record.get("source_type")
    )


def _ledger_summary_identity_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    expected = {
        "accepted_source_types": _source_types_for_decision(records, accepted=True),
        "rejected_source_types": _source_types_for_decision(records, accepted=False),
    }
    return tuple(
        f"memory_trust_acceptance_summary_{key}_mismatch"
        for key, values in expected.items()
        if _string_list(summary.get(key)) != values
    )


def _ledger_summary_safety_violations(summary: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    for flag in REQUIRED_LEDGER_TRUE_SUMMARY_FLAGS:
        if summary.get(flag) is not True:
            violations.append(f"memory_trust_acceptance_summary_{flag}_not_true")
    for flag in REQUIRED_LEDGER_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(f"memory_trust_acceptance_summary_{flag}_not_false")
    return tuple(violations)


def _ambiguous_candidate_review_record_ids(
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    record_ids_by_candidate: Dict[str, List[str]] = {}
    for index, record in enumerate(records):
        if _accepted_record_violations(record):
            continue
        candidate_id = record.get("review_candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            continue
        record_ids_by_candidate.setdefault(candidate_id, []).append(_record_id(record, index))
    return tuple(
        record_id
        for record_ids in record_ids_by_candidate.values()
        if len(record_ids) > 1
        for record_id in record_ids
    )


def _accepted_records(
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[Mapping[str, Any], ...]:
    return tuple(
        record
        for record in records
        if record.get("operator_accepted_memory_trust_candidate") is True
        and not _accepted_record_violations(record)
    )


def _promotion_candidate(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "promotion_candidate_id": f"source_identity_memory_trust_promotion:{record.get('acceptance_id')}",
        "acceptance_id": record.get("acceptance_id"),
        "review_candidate_id": record.get("review_candidate_id"),
        "memory_record_id": record.get("memory_record_id"),
        "source_candidate_id": record.get("source_candidate_id"),
        "source_type": record.get("source_type"),
        "recommended_trust_state": record.get("recommended_trust_state"),
        "reviewed_by": record.get("reviewed_by"),
        "reviewed_at": record.get("reviewed_at"),
        "manual_memory_trust_promotion_required": True,
        "memory_trust_promotion_allowed_by_plan": False,
        "downstream_priority_change_allowed_by_plan": False,
        "queue_mutation_allowed_by_plan": False,
        "model_mutation_allowed_by_plan": False,
        "runtime_memory_mutation_allowed_by_plan": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }


def build_source_identity_calibration_memory_trust_promotion_plan(
    *,
    acceptance_ledger: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a read-only manual memory-trust promotion plan."""

    ledger = (
        _mapping(acceptance_ledger)
        or build_source_identity_calibration_memory_trust_acceptance_ledger()
    )
    summary = _mapping(ledger.get("summary"))
    records = _record_list(ledger.get("records"))
    ledger_invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    unreviewed_candidate_ids = _string_list(summary.get("unreviewed_candidate_ids"))
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *ledger_invalid_record_ids,
                *_ledger_envelope_violations(ledger),
                *_ledger_status_violations(ledger, records=records),
                *_accepted_record_replacement_ids(records),
                *_ambiguous_candidate_review_record_ids(records),
                *_ledger_summary_count_violations(
                    summary,
                    records=records,
                    invalid_record_ids=ledger_invalid_record_ids,
                    unreviewed_candidate_ids=unreviewed_candidate_ids,
                ),
                *_ledger_summary_identity_violations(summary, records=records),
                *_ledger_summary_safety_violations(summary),
            )
        )
    )
    accepted_records = _accepted_records(records)
    promotion_source_records = () if invalid_record_ids else accepted_records
    promotion_candidates = tuple(
        _promotion_candidate(record) for record in promotion_source_records
    )
    rejected_source_types = _string_list(summary.get("rejected_source_types"))

    if invalid_record_ids:
        status = "memory_trust_promotion_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_memory_trust_acceptance_before_promotion",
        )
    elif unreviewed_candidate_ids:
        status = "memory_trust_promotion_not_ready"
        blocked_by = (
            "operator_must_review_all_memory_trust_candidates_before_promotion",
        )
    elif promotion_candidates:
        status = "manual_memory_trust_promotion_candidate_available"
        blocked_by = (
            "manual_memory_trust_promotion_contract_required_before_downstream_priority",
        )
    else:
        status = "memory_trust_promotion_not_ready"
        blocked_by = ("accepted_memory_trust_record_required_before_promotion",)

    return {
        "type": "source_identity_calibration_memory_trust_promotion_plan",
        "generated_at": _utc_now(),
        "method_id": SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID,
        "source_method_id": (
            SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_ACCEPTANCE_LEDGER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "execution_allowed_by_plan": False,
            "memory_trust_promotion_allowed_by_plan": False,
            "queue_mutation_allowed_by_plan": False,
            "model_mutation_allowed_by_plan": False,
            "runtime_memory_mutation_allowed_by_plan": False,
            "downstream_priority_change_allowed_by_plan": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "acceptance_ledger_status": ledger.get("status") or "missing",
            "acceptance_record_count": len(records),
            "accepted_record_count": len(accepted_records),
            "rejected_record_count": int(summary.get("rejected_record_count") or 0),
            "invalid_record_count": len(invalid_record_ids),
            "unreviewed_candidate_count": len(unreviewed_candidate_ids),
            "promotion_candidate_count": len(promotion_candidates),
            "promotion_candidate_ids": [
                str(candidate["promotion_candidate_id"])
                for candidate in promotion_candidates
                if isinstance(candidate.get("promotion_candidate_id"), str)
            ],
            "accepted_source_types": [
                str(candidate["source_type"])
                for candidate in promotion_candidates
                if isinstance(candidate.get("source_type"), str)
            ],
            "rejected_source_types": list(rejected_source_types),
            "operator_review_required": True,
            "execution_allowed_by_plan": False,
            "memory_trust_promotion_allowed_by_plan": False,
            "queue_mutation_allowed_by_plan": False,
            "model_mutation_allowed_by_plan": False,
            "runtime_memory_mutation_allowed_by_plan": False,
            "downstream_priority_change_allowed_by_plan": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "invalid_record_ids": list(invalid_record_ids),
            "unreviewed_candidate_ids": list(unreviewed_candidate_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid memory-trust acceptance records before promotion."
                if invalid_record_ids
                else (
                    "Operator reviews all memory-trust candidates before promotion."
                    if unreviewed_candidate_ids
                    else (
                        "Separate promotion contract is required before indexed "
                        "memory can change downstream source identity trust."
                    )
                )
            ),
        },
        "promotion_candidates": list(promotion_candidates),
    }
