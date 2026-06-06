"""Read-only ledger for operator acceptance of sequence memory trust reviews.

The ledger records accepted/rejected operator decisions against supervised
sequence memory trust-review candidates. It does not promote sequence policy,
execute proposals, mutate queues, policy stores, models, runtime memory, or
touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_review import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_review,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_review import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_review,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_ACCEPTANCE_LEDGER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_ACCEPTANCE_LEDGER_STATUSES = {
    "operator_sequence_memory_trust_acceptance_required",
    "operator_sequence_memory_trust_acceptance_records_available",
    "operator_sequence_memory_trust_acceptance_replacement_required",
}
SEQUENCE_MEMORY_TRUST_ACCEPTANCE_DECISIONS = {"accepted", "rejected"}
REQUIRED_SEQUENCE_MEMORY_TRUST_REVIEW_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_review"
)
REQUIRED_ACCEPTANCE_RECORD_FIELDS = (
    "acceptance_id",
    "review_candidate_id",
    "memory_record_id",
    "proposal_id",
    "learning_scope",
    "value_type",
    "idempotency_key",
    "decision",
    "reviewed_by",
    "reviewed_at",
)
REQUIRED_REVIEW_FALSE_SAFETY_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "execution_allowed_by_review",
    "queue_mutation",
    "policy_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_REVIEW_FALSE_SUMMARY_FLAGS = (
    "execution_allowed_by_review",
    "queue_mutation",
    "policy_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "live_authority_granted",
    "full_autonomous_ready",
)
UNSAFE_ACCEPTANCE_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "policy_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "requested_change_allowed",
    "execution_allowed_by_acceptance",
    "sequence_policy_promotion_allowed_by_acceptance",
    "policy_mutation_allowed_by_acceptance",
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
    return f"{memory_record_id if isinstance(memory_record_id, str) and memory_record_id else 'unknown_sequence_memory_trust_candidate'}:{index}"


def _candidate_basic_violations(candidate: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    required_fields = (
        "review_candidate_id",
        "memory_record_id",
        "draft_id",
        "proposal_id",
        "learning_scope",
        "value_type",
        "idempotency_key",
        "readback_status",
        "proof_status",
    )
    for field in required_fields:
        if not isinstance(candidate.get(field), str) or not candidate.get(field):
            violations.append(f"sequence_memory_acceptance_candidate_{field}_missing")
    if candidate.get("operator_review_required") is not True:
        violations.append("sequence_memory_acceptance_candidate_operator_review_not_true")
    if candidate.get("sequence_policy_review_state") != "operator_review_required":
        violations.append("sequence_memory_acceptance_candidate_review_state_invalid")
    if candidate.get("value_type") == "fader":
        effective_value = candidate.get("effective_value")
        if isinstance(effective_value, bool) or not isinstance(effective_value, (int, float)):
            violations.append("sequence_memory_acceptance_candidate_effective_value_invalid")
        elif float(effective_value) > 0.0:
            violations.append("sequence_memory_acceptance_candidate_fader_ceiling_exceeded")
    false_flags = (
        "execution_allowed_by_review",
        "queue_mutation_allowed_by_review",
        "policy_mutation_allowed_by_review",
        "model_mutation_allowed_by_review",
        "runtime_memory_mutation_allowed_by_review",
        "live_authority_granted",
        "full_autonomous_ready",
    )
    for flag in false_flags:
        if candidate.get(flag) is not False:
            violations.append(f"sequence_memory_acceptance_candidate_{flag}_not_false")
    return tuple(violations)


def _review_envelope_violations(review: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if review.get("type") != REQUIRED_SEQUENCE_MEMORY_TRUST_REVIEW_TYPE:
        violations.append("sequence_memory_trust_review_envelope_type_mismatch")
    generated_at = review.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("sequence_memory_trust_review_envelope_generated_at_invalid")
    if (
        review.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_METHOD_ID
    ):
        violations.append("sequence_memory_trust_review_envelope_method_id_mismatch")
    if (
        review.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_STATUSES
    ):
        violations.append("sequence_memory_trust_review_envelope_status_unknown")
    safety = _mapping(review.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_memory_trust_review_envelope_read_only_not_true")
    for flag in REQUIRED_REVIEW_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"sequence_memory_trust_review_envelope_{flag}_not_false")
    return tuple(violations)


def _review_summary_violations(
    summary: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
    review_invalid_record_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if _summary_int(summary.get("trust_review_candidate_count")) != len(candidates):
        violations.append("sequence_memory_trust_review_summary_candidate_count_mismatch")
    if _summary_int(summary.get("invalid_record_count")) != len(review_invalid_record_ids):
        violations.append("sequence_memory_trust_review_summary_invalid_record_count_mismatch")
    if summary.get("operator_review_required") is not True:
        violations.append("sequence_memory_trust_review_summary_operator_review_required_not_true")
    for flag in REQUIRED_REVIEW_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(f"sequence_memory_trust_review_summary_{flag}_not_false")
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
    candidate_id = _string(record.get("review_candidate_id")) or "unknown_sequence_memory_trust_candidate"
    return f"{candidate_id}:{index}"


def _unsafe_flags(record: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(flag for flag in UNSAFE_ACCEPTANCE_FLAGS if record.get(flag) is True)


def _duplicate_review_candidate_ids(records: Tuple[Mapping[str, Any], ...]) -> Tuple[str, ...]:
    counts: Dict[str, int] = {}
    for record in records:
        if not record.get("valid"):
            continue
        review_candidate_id = record.get("review_candidate_id")
        if isinstance(review_candidate_id, str) and review_candidate_id:
            counts[review_candidate_id] = counts.get(review_candidate_id, 0) + 1
    return tuple(candidate_id for candidate_id, count in counts.items() if count > 1)


def _records_with_duplicate_review_violations(
    records: Tuple[Dict[str, Any], ...],
    duplicate_review_candidate_ids: Tuple[str, ...],
) -> Tuple[Dict[str, Any], ...]:
    if not duplicate_review_candidate_ids:
        return records
    duplicates = set(duplicate_review_candidate_ids)
    updated_records: list[Dict[str, Any]] = []
    for record in records:
        if record.get("valid") and record.get("review_candidate_id") in duplicates:
            updated_record = dict(record)
            updated_record["valid"] = False
            updated_record["operator_accepted_sequence_memory_trust_candidate"] = False
            updated_record["operator_rejected_sequence_memory_trust_candidate"] = False
            updated_record["violations"] = [
                *record["violations"],
                "sequence_memory_acceptance_record_review_candidate_duplicate",
            ]
            updated_records.append(updated_record)
        else:
            updated_records.append(record)
    return tuple(updated_records)


def _validated_record(
    record: Mapping[str, Any],
    *,
    index: int,
    candidates_by_candidate_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    record_id = _acceptance_record_id(record, index)
    review_candidate_id = _string(record.get("review_candidate_id"))
    memory_record_id = _string(record.get("memory_record_id"))
    proposal_id = _string(record.get("proposal_id"))
    learning_scope = _string(record.get("learning_scope"))
    value_type = _string(record.get("value_type"))
    idempotency_key = _string(record.get("idempotency_key"))
    decision = _string(record.get("decision"))
    reviewed_by = _string(record.get("reviewed_by"))
    reviewed_at = _string(record.get("reviewed_at"))
    unsafe_flags = _unsafe_flags(record)
    candidate = _mapping(candidates_by_candidate_id.get(review_candidate_id))
    violations = []

    for field in REQUIRED_ACCEPTANCE_RECORD_FIELDS:
        if not _string(record.get(field)):
            violations.append(f"sequence_memory_acceptance_record_{field}_missing")
    if review_candidate_id not in candidates_by_candidate_id:
        violations.append("sequence_memory_acceptance_record_candidate_missing")
    if decision not in SEQUENCE_MEMORY_TRUST_ACCEPTANCE_DECISIONS:
        violations.append("sequence_memory_acceptance_record_decision_unknown")
    if reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append("sequence_memory_acceptance_record_reviewed_at_invalid")
    if candidate:
        expected_pairs = {
            "memory_record_id": memory_record_id,
            "proposal_id": proposal_id,
            "learning_scope": learning_scope,
            "value_type": value_type,
            "idempotency_key": idempotency_key,
        }
        for field, supplied in expected_pairs.items():
            if supplied != candidate.get(field):
                violations.append(f"sequence_memory_acceptance_record_{field}_mismatch")
    if unsafe_flags:
        violations.append("sequence_memory_acceptance_record_unsafe_authority_claim")

    valid = not violations
    return {
        "acceptance_id": record_id,
        "review_candidate_id": review_candidate_id,
        "memory_record_id": memory_record_id,
        "proposal_id": proposal_id,
        "learning_scope": learning_scope,
        "value_type": value_type,
        "idempotency_key": idempotency_key,
        "decision": decision if decision in SEQUENCE_MEMORY_TRUST_ACCEPTANCE_DECISIONS else "invalid",
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at,
        "candidate_present": bool(candidate),
        "operator_accepted_sequence_memory_trust_candidate": valid and decision == "accepted",
        "operator_rejected_sequence_memory_trust_candidate": valid and decision == "rejected",
        "sequence_policy_promotion_allowed_by_acceptance": False,
        "execution_allowed_by_acceptance": False,
        "queue_mutation_allowed_by_acceptance": False,
        "policy_mutation_allowed_by_acceptance": False,
        "model_mutation_allowed_by_acceptance": False,
        "runtime_memory_mutation_allowed_by_acceptance": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "unsafe_authority_flags": list(unsafe_flags),
        "valid": valid,
        "violations": violations,
    }


def build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger(
    *,
    memory_trust_review: Mapping[str, Any] | None = None,
    acceptance_records: Iterable[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build a read-only ledger for sequence memory trust-review decisions."""

    review = _mapping(memory_trust_review) or (
        build_operator_proposal_apply_sequence_policy_memory_trust_review()
    )
    summary = _mapping(review.get("summary"))
    review_candidates = _record_list(review.get("trust_review_candidates"))
    review_invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    candidate_violations = []
    for index, candidate in enumerate(review_candidates):
        violations = _candidate_basic_violations(candidate)
        if violations:
            candidate_violations.append(_review_candidate_id(candidate, index))
            candidate_violations.extend(violations)
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
                *candidate_violations,
            )
        )
    )
    candidates_by_candidate_id = _candidates_by_id(review)
    initial_records = tuple(
        _validated_record(
            record,
            index=index,
            candidates_by_candidate_id=candidates_by_candidate_id,
        )
        for index, record in enumerate(_record_list(acceptance_records))
    )
    duplicate_review_candidate_ids = _duplicate_review_candidate_ids(initial_records)
    records = _records_with_duplicate_review_violations(
        initial_records,
        duplicate_review_candidate_ids,
    )
    valid_records = tuple(record for record in records if record["valid"])
    accepted_records = tuple(
        record
        for record in valid_records
        if record["operator_accepted_sequence_memory_trust_candidate"]
    )
    rejected_records = tuple(
        record
        for record in valid_records
        if record["operator_rejected_sequence_memory_trust_candidate"]
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
        status = "operator_sequence_memory_trust_acceptance_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_memory_trust_acceptance_records",
        )
    elif unreviewed_candidate_ids:
        status = "operator_sequence_memory_trust_acceptance_required"
        blocked_by = ("operator_must_review_sequence_memory_trust_candidates_before_promotion",)
    elif accepted_records or rejected_records:
        status = "operator_sequence_memory_trust_acceptance_records_available"
        blocked_by = ("sequence_policy_promotion_still_requires_separate_contract",)
    else:
        status = "operator_sequence_memory_trust_acceptance_required"
        blocked_by = ("valid_sequence_memory_trust_review_candidates_required_before_acceptance",)

    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *review_violations,
                *(record["acceptance_id"] for record in invalid_records),
            )
        )
    )
    accepted_proposal_ids = [record["proposal_id"] for record in accepted_records]
    rejected_proposal_ids = [record["proposal_id"] for record in rejected_records]
    return {
        "type": "operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger",
        "generated_at": _utc_now(),
        "method_id": OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_ACCEPTANCE_LEDGER_METHOD_ID,
        "source_method_id": OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_METHOD_ID,
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "apply_attempted": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "execution_allowed_by_acceptance": False,
            "queue_mutation": False,
            "policy_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "sequence_memory_trust_review_status": review.get("status") or "missing",
            "trust_review_candidate_count": len(review_candidates),
            "acceptance_record_count": len(records),
            "accepted_record_count": len(accepted_records),
            "rejected_record_count": len(rejected_records),
            "invalid_record_count": len(invalid_record_ids),
            "unreviewed_candidate_count": len(unreviewed_candidate_ids),
            "accepted_proposal_ids": accepted_proposal_ids,
            "rejected_proposal_ids": rejected_proposal_ids,
            "operator_review_required": True,
            "sequence_policy_promotion_allowed_by_acceptance": False,
            "execution_allowed_by_acceptance": False,
            "queue_mutation_allowed_by_acceptance": False,
            "policy_mutation_allowed_by_acceptance": False,
            "model_mutation_allowed_by_acceptance": False,
            "runtime_memory_mutation_allowed_by_acceptance": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "invalid_record_ids": list(invalid_record_ids),
            "unreviewed_candidate_ids": list(unreviewed_candidate_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid sequence-memory trust acceptance records."
                if invalid_record_ids
                else (
                    "Operator reviews all sequence-memory trust candidates before promotion."
                    if unreviewed_candidate_ids
                    else (
                        "Sequence-memory trust acceptance is evidence only; separate "
                        "promotion is required before policy or execution influence."
                    )
                )
            ),
        },
        "records": list(records),
    }
