"""Read-only ledger for operator acceptance of source-identity calibration.

The ledger validates operator decisions against replay-calibration candidates.
It does not update classifier models, write long-term memory, change downstream
proposal priority, mutate queues, or touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Tuple


SOURCE_IDENTITY_CALIBRATION_ACCEPTANCE_LEDGER_METHOD_ID = (
    "source_identity_calibration_acceptance_ledger"
)
SOURCE_IDENTITY_CALIBRATION_ACCEPTANCE_LEDGER_STATUSES = {
    "operator_acceptance_required",
    "operator_acceptance_records_available",
    "operator_acceptance_replacement_required",
}
CALIBRATION_ACCEPTANCE_DECISIONS = {"accepted", "rejected"}
SOURCE_IDENTITY_REPLAY_CALIBRATION_METHOD_ID = "source_identity_replay_calibration"
UNSAFE_ACCEPTANCE_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "memory_mutation",
    "downstream_priority_mutation",
    "requested_change_allowed",
    "calibration_persistence_allowed_by_acceptance",
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


def _string(value: Any) -> str:
    return value if isinstance(value, str) and value else ""


def _record_list(value: Iterable[Mapping[str, Any]] | None) -> Tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


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


def _candidate_source_type(candidate: Mapping[str, Any]) -> str:
    proposed_action = _mapping(candidate.get("proposed_action"))
    return (
        _string(proposed_action.get("source_type"))
        or _string(candidate.get("target"))
        or _string(candidate.get("source_type"))
    )


def _candidate_trust_state(candidate: Mapping[str, Any]) -> str:
    proposed_action = _mapping(candidate.get("proposed_action"))
    return _string(proposed_action.get("recommended_trust_state"))


def _candidate_id(candidate: Mapping[str, Any], index: int) -> str:
    explicit = _string(candidate.get("id"))
    if explicit:
        return explicit
    source_type = _candidate_source_type(candidate) or "unknown_source_type"
    trust_state = _candidate_trust_state(candidate) or "unknown_trust_state"
    return f"{SOURCE_IDENTITY_REPLAY_CALIBRATION_METHOD_ID}:{source_type}:{trust_state}:{index}"


def _candidate_is_source_identity_calibration(candidate: Mapping[str, Any]) -> bool:
    if candidate.get("method_id") != SOURCE_IDENTITY_REPLAY_CALIBRATION_METHOD_ID:
        return False
    if candidate.get("can_import_to_queue") is True or candidate.get("can_apply_live") is True:
        return False
    proposed_action = _mapping(candidate.get("proposed_action"))
    if proposed_action.get("action_type") != "source_identity_replay_calibration_candidate":
        return False
    return bool(_candidate_source_type(candidate) and _candidate_trust_state(candidate))


def _candidates_by_id(
    calibration_candidates: Iterable[Mapping[str, Any]] | None,
) -> Dict[str, Mapping[str, Any]]:
    candidates: Dict[str, Mapping[str, Any]] = {}
    for index, candidate in enumerate(_record_list(calibration_candidates)):
        if not _candidate_is_source_identity_calibration(candidate):
            continue
        candidate_id = _candidate_id(candidate, index)
        candidates[candidate_id] = candidate
    return candidates


def _record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("review_id"))
    if explicit:
        return explicit
    candidate_id = _string(record.get("candidate_id")) or "unknown_candidate"
    return f"{candidate_id}:{index}"


def _unsafe_flags(record: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(key for key in UNSAFE_ACCEPTANCE_FLAGS if record.get(key) is True)


def _validated_record(
    record: Mapping[str, Any],
    *,
    index: int,
    candidates_by_candidate_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    record_id = _record_id(record, index)
    candidate_id = _string(record.get("candidate_id"))
    source_type = _string(record.get("source_type"))
    recommended_trust_state = _string(record.get("recommended_trust_state"))
    calibration_method_id = _string(record.get("calibration_method_id"))
    decision = _string(record.get("decision"))
    reviewed_by = _string(record.get("reviewed_by"))
    reviewed_at = _string(record.get("reviewed_at"))
    unsafe_flags = _unsafe_flags(record)
    candidate = _mapping(candidates_by_candidate_id.get(candidate_id))
    candidate_source_type = _candidate_source_type(candidate)
    candidate_trust_state = _candidate_trust_state(candidate)
    violations = []

    if candidate_id not in candidates_by_candidate_id:
        violations.append("acceptance_record_candidate_missing")
    if calibration_method_id != SOURCE_IDENTITY_REPLAY_CALIBRATION_METHOD_ID:
        violations.append("acceptance_record_calibration_method_mismatch")
    if decision not in CALIBRATION_ACCEPTANCE_DECISIONS:
        violations.append("acceptance_record_decision_unknown")
    if not reviewed_by:
        violations.append("acceptance_record_reviewer_missing")
    if not reviewed_at:
        violations.append("acceptance_record_reviewed_at_missing")
    elif not _is_utc_timestamp(reviewed_at):
        violations.append("acceptance_record_reviewed_at_invalid")
    if not source_type:
        violations.append("acceptance_record_source_type_missing")
    elif candidate and source_type != candidate_source_type:
        violations.append("acceptance_record_source_type_mismatch")
    if not recommended_trust_state:
        violations.append("acceptance_record_recommended_trust_state_missing")
    elif candidate and recommended_trust_state != candidate_trust_state:
        violations.append("acceptance_record_recommended_trust_state_mismatch")
    if unsafe_flags:
        violations.append("acceptance_record_unsafe_authority_claim")

    valid = not violations
    return {
        "review_id": record_id,
        "candidate_id": candidate_id,
        "calibration_method_id": calibration_method_id,
        "source_type": source_type,
        "recommended_trust_state": recommended_trust_state,
        "decision": decision if decision in CALIBRATION_ACCEPTANCE_DECISIONS else "invalid",
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at,
        "candidate_present": bool(candidate),
        "operator_accepted_calibration_candidate": valid and decision == "accepted",
        "operator_rejected_calibration_candidate": valid and decision == "rejected",
        "calibration_persistence_allowed_by_acceptance": False,
        "downstream_priority_change_allowed_by_acceptance": False,
        "queue_mutation_allowed_by_acceptance": False,
        "model_mutation_allowed_by_acceptance": False,
        "memory_mutation_allowed_by_acceptance": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "unsafe_authority_flags": list(unsafe_flags),
        "valid": valid,
        "violations": violations,
    }


def build_source_identity_calibration_acceptance_ledger(
    *,
    calibration_candidates: Iterable[Mapping[str, Any]] | None = None,
    acceptance_records: Iterable[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build a read-only ledger for operator-reviewed calibration decisions."""

    candidates_by_candidate_id = _candidates_by_id(calibration_candidates)
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
        record for record in valid_records if record["operator_accepted_calibration_candidate"]
    )
    rejected_records = tuple(
        record for record in valid_records if record["operator_rejected_calibration_candidate"]
    )
    invalid_records = tuple(record for record in records if not record["valid"])
    reviewed_candidate_ids = {
        str(record["candidate_id"])
        for record in valid_records
        if isinstance(record.get("candidate_id"), str)
    }
    unreviewed_candidate_ids = tuple(
        candidate_id
        for candidate_id in candidates_by_candidate_id
        if candidate_id not in reviewed_candidate_ids
    )

    if invalid_records:
        status = "operator_acceptance_replacement_required"
        blocked_by = ("operator_must_replace_invalid_source_identity_acceptance_records",)
    elif unreviewed_candidate_ids:
        status = "operator_acceptance_required"
        blocked_by = (
            "operator_must_review_source_identity_calibration_candidates_before_persistence",
        )
    elif accepted_records or rejected_records:
        status = "operator_acceptance_records_available"
        blocked_by = ("calibration_persistence_still_requires_separate_write_contract",)
    else:
        status = "operator_acceptance_required"
        blocked_by = ("operator_must_surface_valid_source_identity_calibration_candidates_first",)

    accepted_source_types = [record["source_type"] for record in accepted_records]
    rejected_source_types = [record["source_type"] for record in rejected_records]
    return {
        "type": "source_identity_calibration_acceptance_ledger",
        "generated_at": _utc_now(),
        "method_id": SOURCE_IDENTITY_CALIBRATION_ACCEPTANCE_LEDGER_METHOD_ID,
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "queue_mutation": False,
            "model_mutation": False,
            "memory_mutation": False,
            "downstream_priority_mutation": False,
            "operator_required": True,
            "calibration_persistence_allowed_by_acceptance": False,
            "downstream_priority_change_allowed_by_acceptance": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "calibration_candidate_count": len(candidates_by_candidate_id),
            "acceptance_record_count": len(records),
            "accepted_record_count": len(accepted_records),
            "rejected_record_count": len(rejected_records),
            "invalid_record_count": len(invalid_records),
            "unreviewed_candidate_count": len(unreviewed_candidate_ids),
            "accepted_candidate_ids": [record["candidate_id"] for record in accepted_records],
            "rejected_candidate_ids": [record["candidate_id"] for record in rejected_records],
            "accepted_source_types": accepted_source_types,
            "rejected_source_types": rejected_source_types,
            "invalid_record_ids": [record["review_id"] for record in invalid_records],
            "unreviewed_candidate_ids": list(unreviewed_candidate_ids),
            "operator_acceptance_required": status == "operator_acceptance_required",
            "operator_review_required": True,
            "calibration_persistence_allowed_by_acceptance": False,
            "downstream_priority_change_allowed_by_acceptance": False,
            "queue_mutation_allowed_by_acceptance": False,
            "model_mutation_allowed_by_acceptance": False,
            "memory_mutation_allowed_by_acceptance": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid source identity acceptance records."
                if invalid_records
                else (
                    "Operator reviews calibration candidates and records decisions."
                    if unreviewed_candidate_ids
                    else (
                        "Accepted calibration records still require a separate persistence contract."
                        if accepted_records or rejected_records
                        else "Surface valid source identity calibration candidates first."
                    )
                )
            ),
        },
        "records": list(records),
    }
