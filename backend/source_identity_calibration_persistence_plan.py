"""Read-only plan for manual source-identity calibration persistence.

The plan turns accepted calibration-review records into manual persistence
candidates. It does not write classifier models, long-term memory, downstream
proposal priority, queues, or live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Tuple

try:
    from source_identity_calibration_acceptance_ledger import (
        SOURCE_IDENTITY_CALIBRATION_ACCEPTANCE_LEDGER_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_ACCEPTANCE_LEDGER_STATUSES,
        build_source_identity_calibration_acceptance_ledger,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_calibration_acceptance_ledger import (
        SOURCE_IDENTITY_CALIBRATION_ACCEPTANCE_LEDGER_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_ACCEPTANCE_LEDGER_STATUSES,
        build_source_identity_calibration_acceptance_ledger,
    )


SOURCE_IDENTITY_CALIBRATION_PERSISTENCE_PLAN_METHOD_ID = (
    "source_identity_calibration_persistence_plan"
)
SOURCE_IDENTITY_CALIBRATION_PERSISTENCE_PLAN_STATUSES = {
    "calibration_persistence_not_ready",
    "calibration_persistence_replacement_required",
    "manual_calibration_persistence_candidate_available",
}
REQUIRED_ACCEPTANCE_LEDGER_TYPE = "source_identity_calibration_acceptance_ledger"
REQUIRED_PERSISTENCE_RECORD_FIELDS = (
    "review_id",
    "candidate_id",
    "calibration_method_id",
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
    "memory_mutation",
    "downstream_priority_mutation",
    "calibration_persistence_allowed_by_acceptance",
    "downstream_priority_change_allowed_by_acceptance",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_LEDGER_TRUE_SUMMARY_FLAGS = ("operator_review_required",)
REQUIRED_LEDGER_FALSE_SUMMARY_FLAGS = (
    "calibration_persistence_allowed_by_acceptance",
    "downstream_priority_change_allowed_by_acceptance",
    "queue_mutation_allowed_by_acceptance",
    "model_mutation_allowed_by_acceptance",
    "memory_mutation_allowed_by_acceptance",
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
    explicit = record.get("review_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    candidate_id = record.get("candidate_id")
    return f"{candidate_id if isinstance(candidate_id, str) and candidate_id else 'unknown_candidate'}:{index}"


def _persistence_record_basic_violations(record: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    for field in REQUIRED_PERSISTENCE_RECORD_FIELDS:
        if not isinstance(record.get(field), str) or not record.get(field):
            violations.append(f"calibration_persistence_record_{field}_missing")
    reviewed_at = record.get("reviewed_at")
    if isinstance(reviewed_at, str) and reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append("calibration_persistence_record_reviewed_at_invalid")
    if record.get("valid") is not True:
        violations.append("calibration_persistence_record_not_valid")
    if record.get("candidate_present") is not True:
        violations.append("calibration_persistence_record_candidate_missing")
    if record.get("calibration_persistence_allowed_by_acceptance") is not False:
        violations.append("calibration_persistence_record_acceptance_persistence_not_false")
    if record.get("downstream_priority_change_allowed_by_acceptance") is not False:
        violations.append("calibration_persistence_record_downstream_priority_not_false")
    if record.get("queue_mutation_allowed_by_acceptance") is not False:
        violations.append("calibration_persistence_record_queue_mutation_not_false")
    if record.get("model_mutation_allowed_by_acceptance") is not False:
        violations.append("calibration_persistence_record_model_mutation_not_false")
    if record.get("memory_mutation_allowed_by_acceptance") is not False:
        violations.append("calibration_persistence_record_memory_mutation_not_false")
    if record.get("live_authority_granted") is not False:
        violations.append("calibration_persistence_record_live_authority_not_false")
    if record.get("full_autonomous_ready") is not False:
        violations.append("calibration_persistence_record_full_autonomous_not_false")
    return tuple(violations)


def _accepted_record_violations(record: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = list(_persistence_record_basic_violations(record))
    if record.get("decision") != "accepted":
        violations.append("calibration_persistence_record_acceptance_decision_mismatch")
    if record.get("operator_rejected_calibration_candidate") is not False:
        violations.append("calibration_persistence_record_conflicting_review_flags")
    return tuple(violations)


def _accepted_record_replacement_ids(
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    return tuple(
        _record_id(record, index)
        for index, record in enumerate(records)
        if record.get("operator_accepted_calibration_candidate") is True
        and _accepted_record_violations(record)
    )


def _ledger_envelope_violations(ledger: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if ledger.get("type") != REQUIRED_ACCEPTANCE_LEDGER_TYPE:
        violations.append("ledger_envelope_type_mismatch")
    generated_at = ledger.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("ledger_envelope_generated_at_invalid")
    if ledger.get("method_id") != SOURCE_IDENTITY_CALIBRATION_ACCEPTANCE_LEDGER_METHOD_ID:
        violations.append("ledger_envelope_method_id_mismatch")
    if ledger.get("status") not in SOURCE_IDENTITY_CALIBRATION_ACCEPTANCE_LEDGER_STATUSES:
        violations.append("ledger_envelope_status_unknown")
    safety = _mapping(ledger.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("ledger_envelope_read_only_not_true")
    for flag in REQUIRED_LEDGER_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"ledger_envelope_{flag}_not_false")
    return tuple(violations)


def _ledger_summary_count_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
    invalid_record_ids: Tuple[str, ...],
    unreviewed_candidate_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    reviewed_candidate_ids = tuple(
        record["candidate_id"]
        for record in records
        if record.get("valid") is True
        and (
            record.get("operator_accepted_calibration_candidate") is True
            or record.get("operator_rejected_calibration_candidate") is True
        )
        and isinstance(record.get("candidate_id"), str)
        and record.get("candidate_id")
    )
    candidate_ids = tuple(dict.fromkeys((*reviewed_candidate_ids, *unreviewed_candidate_ids)))
    expected_counts = {
        "calibration_candidate_count": len(candidate_ids),
        "acceptance_record_count": len(records),
        "accepted_record_count": sum(
            1
            for record in records
            if record.get("valid") is True
            and record.get("operator_accepted_calibration_candidate") is True
        ),
        "rejected_record_count": sum(
            1
            for record in records
            if record.get("valid") is True
            and record.get("operator_rejected_calibration_candidate") is True
        ),
        "invalid_record_count": len(invalid_record_ids),
        "unreviewed_candidate_count": len(unreviewed_candidate_ids),
    }
    return tuple(
        f"ledger_summary_{key}_mismatch"
        for key, expected in expected_counts.items()
        if _summary_int(summary.get(key)) != expected
    )


def _candidate_ids_for_decision(
    records: Tuple[Mapping[str, Any], ...],
    *,
    accepted: bool,
) -> Tuple[str, ...]:
    flag = (
        "operator_accepted_calibration_candidate"
        if accepted
        else "operator_rejected_calibration_candidate"
    )
    return tuple(
        record["candidate_id"]
        for record in records
        if record.get("valid") is True
        and record.get(flag) is True
        and isinstance(record.get("candidate_id"), str)
        and record.get("candidate_id")
    )


def _source_types_for_decision(
    records: Tuple[Mapping[str, Any], ...],
    *,
    accepted: bool,
) -> Tuple[str, ...]:
    flag = (
        "operator_accepted_calibration_candidate"
        if accepted
        else "operator_rejected_calibration_candidate"
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
        "accepted_candidate_ids": _candidate_ids_for_decision(records, accepted=True),
        "rejected_candidate_ids": _candidate_ids_for_decision(records, accepted=False),
        "accepted_source_types": _source_types_for_decision(records, accepted=True),
        "rejected_source_types": _source_types_for_decision(records, accepted=False),
    }
    return tuple(
        f"ledger_summary_{key}_mismatch"
        for key, values in expected.items()
        if _string_list(summary.get(key)) != values
    )


def _ledger_summary_safety_violations(summary: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    for flag in REQUIRED_LEDGER_TRUE_SUMMARY_FLAGS:
        if summary.get(flag) is not True:
            violations.append(f"ledger_summary_{flag}_not_true")
    for flag in REQUIRED_LEDGER_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(f"ledger_summary_{flag}_not_false")
    return tuple(violations)


def _ambiguous_candidate_review_record_ids(
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    record_ids_by_candidate: Dict[str, List[str]] = {}
    for index, record in enumerate(records):
        if _persistence_record_basic_violations(record):
            continue
        decision = record.get("decision")
        if decision not in {"accepted", "rejected"}:
            continue
        candidate_id = record.get("candidate_id")
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
        if record.get("operator_accepted_calibration_candidate") is True
        and not _accepted_record_violations(record)
    )


def _persistence_candidate(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "review_id": record.get("review_id"),
        "candidate_id": record.get("candidate_id"),
        "calibration_method_id": record.get("calibration_method_id"),
        "source_type": record.get("source_type"),
        "recommended_trust_state": record.get("recommended_trust_state"),
        "reviewed_by": record.get("reviewed_by"),
        "reviewed_at": record.get("reviewed_at"),
        "manual_calibration_persistence_required": True,
        "calibration_persistence_write_allowed_by_plan": False,
        "queue_mutation_allowed_by_plan": False,
        "model_mutation_allowed_by_plan": False,
        "memory_mutation_allowed_by_plan": False,
        "downstream_priority_change_allowed_by_plan": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }


def build_source_identity_calibration_persistence_plan(
    *,
    acceptance_ledger: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a read-only manual persistence plan from accepted ledger records."""

    ledger = _mapping(acceptance_ledger) or build_source_identity_calibration_acceptance_ledger()
    summary = _mapping(ledger.get("summary"))
    records = _record_list(ledger.get("records"))
    ledger_invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    unreviewed_candidate_ids = _string_list(summary.get("unreviewed_candidate_ids"))
    ledger_envelope_violations = _ledger_envelope_violations(ledger)
    accepted_record_replacement_ids = _accepted_record_replacement_ids(records)
    ambiguous_candidate_review_record_ids = _ambiguous_candidate_review_record_ids(records)
    ledger_summary_count_violations = _ledger_summary_count_violations(
        summary,
        records=records,
        invalid_record_ids=ledger_invalid_record_ids,
        unreviewed_candidate_ids=unreviewed_candidate_ids,
    )
    ledger_summary_identity_violations = _ledger_summary_identity_violations(
        summary,
        records=records,
    )
    ledger_summary_safety_violations = _ledger_summary_safety_violations(summary)
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *ledger_invalid_record_ids,
                *ledger_envelope_violations,
                *accepted_record_replacement_ids,
                *ambiguous_candidate_review_record_ids,
                *ledger_summary_count_violations,
                *ledger_summary_identity_violations,
                *ledger_summary_safety_violations,
            )
        )
    )
    accepted_records = _accepted_records(records)
    persistence_candidate_records = () if invalid_record_ids else accepted_records
    persistence_candidates = tuple(
        _persistence_candidate(record) for record in persistence_candidate_records
    )
    rejected_candidate_ids = _string_list(summary.get("rejected_candidate_ids"))

    if invalid_record_ids:
        status = "calibration_persistence_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_source_identity_acceptance_records_before_persistence",
        )
    elif unreviewed_candidate_ids:
        status = "calibration_persistence_not_ready"
        blocked_by = (
            "operator_must_review_all_source_identity_candidates_before_persistence",
        )
    elif persistence_candidates:
        status = "manual_calibration_persistence_candidate_available"
        blocked_by = ("manual_calibration_persistence_contract_required_before_memory_write",)
    else:
        status = "calibration_persistence_not_ready"
        blocked_by = ("accepted_source_identity_calibration_required_before_persistence",)

    return {
        "type": "source_identity_calibration_persistence_plan",
        "generated_at": _utc_now(),
        "method_id": SOURCE_IDENTITY_CALIBRATION_PERSISTENCE_PLAN_METHOD_ID,
        "source_method_id": SOURCE_IDENTITY_CALIBRATION_ACCEPTANCE_LEDGER_METHOD_ID,
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "execution_allowed_by_plan": False,
            "calibration_persistence_write_allowed_by_plan": False,
            "queue_mutation_allowed_by_plan": False,
            "model_mutation_allowed_by_plan": False,
            "memory_mutation_allowed_by_plan": False,
            "downstream_priority_change_allowed_by_plan": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "ledger_status": ledger.get("status") or "missing",
            "calibration_candidate_count": int(
                summary.get("calibration_candidate_count") or 0
            ),
            "acceptance_record_count": len(records),
            "accepted_record_count": len(accepted_records),
            "rejected_record_count": int(summary.get("rejected_record_count") or 0),
            "invalid_record_count": len(invalid_record_ids),
            "unreviewed_candidate_count": len(unreviewed_candidate_ids),
            "persistence_candidate_count": len(persistence_candidates),
            "persistence_candidate_ids": [
                str(candidate["candidate_id"])
                for candidate in persistence_candidates
                if isinstance(candidate.get("candidate_id"), str)
            ],
            "accepted_source_types": [
                str(candidate["source_type"])
                for candidate in persistence_candidates
                if isinstance(candidate.get("source_type"), str)
            ],
            "rejected_candidate_ids": list(rejected_candidate_ids),
            "invalid_record_ids": list(invalid_record_ids),
            "unreviewed_candidate_ids": list(unreviewed_candidate_ids),
            "manual_calibration_persistence_candidate_available": (
                status == "manual_calibration_persistence_candidate_available"
            ),
            "operator_review_required": True,
            "calibration_persistence_write_allowed_by_plan": False,
            "queue_mutation_allowed_by_plan": False,
            "model_mutation_allowed_by_plan": False,
            "memory_mutation_allowed_by_plan": False,
            "downstream_priority_change_allowed_by_plan": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid source identity acceptance records."
                if invalid_record_ids
                else (
                    "Operator reviews all calibration candidates before persistence planning."
                    if unreviewed_candidate_ids
                    else (
                        "Manual calibration persistence contract is required before memory writes."
                        if persistence_candidates
                        else "Accepted source identity calibration is required before persistence."
                    )
                )
            ),
        },
        "persistence_candidates": list(persistence_candidates),
    }
