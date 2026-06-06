"""Read-only ledger for sequence-policy memory trust promotion acceptance.

The ledger records operator accepted/rejected decisions against sequence-memory
promotion contract records. It does not promote sequence policy, execute
proposals, mutate queues, policy stores, models, runtime memory, append-only
memory, or touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_contract import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_contract import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_STATUSES = {
    "operator_sequence_memory_trust_promotion_acceptance_required",
    "operator_sequence_memory_trust_promotion_acceptance_records_available",
    "operator_sequence_memory_trust_promotion_acceptance_replacement_required",
}
SEQUENCE_MEMORY_TRUST_PROMOTION_ACCEPTANCE_DECISIONS = {"accepted", "rejected"}
REQUIRED_PROMOTION_CONTRACT_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_contract"
)
REQUIRED_PROMOTION_ACCEPTANCE_RECORD_FIELDS = (
    "promotion_acceptance_id",
    "promotion_contract_record_id",
    "promotion_candidate_id",
    "proposal_id",
    "learning_scope",
    "value_type",
    "idempotency_key",
    "previous_policy_state",
    "rollback_strategy",
    "decision",
    "reviewed_by",
    "reviewed_at",
)
REQUIRED_CONTRACT_RECORD_FIELDS = (
    "promotion_contract_record_id",
    "promotion_candidate_id",
    "proposal_id",
    "learning_scope",
    "value_type",
    "idempotency_key",
    "previous_policy_state",
    "rollback_strategy",
    "reviewed_by",
    "reviewed_at",
)
REQUIRED_CONTRACT_FALSE_SAFETY_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "execution_allowed_by_contract",
    "sequence_policy_promotion_allowed_by_contract",
    "queue_mutation_allowed_by_contract",
    "policy_mutation_allowed_by_contract",
    "model_mutation_allowed_by_contract",
    "runtime_memory_mutation_allowed_by_contract",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_CONTRACT_FALSE_SUMMARY_FLAGS = (
    "execution_allowed_by_contract",
    "sequence_policy_promotion_allowed_by_contract",
    "queue_mutation_allowed_by_contract",
    "policy_mutation_allowed_by_contract",
    "model_mutation_allowed_by_contract",
    "runtime_memory_mutation_allowed_by_contract",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_CONTRACT_RECORD_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "execution_allowed_by_contract",
    "sequence_policy_promotion_allowed_by_contract",
    "queue_mutation_allowed_by_contract",
    "policy_mutation_allowed_by_contract",
    "model_mutation_allowed_by_contract",
    "runtime_memory_mutation_allowed_by_contract",
    "live_widening_allowed",
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
    "execution_allowed_by_acceptance",
    "sequence_policy_promotion_allowed_by_acceptance",
    "queue_mutation_allowed_by_acceptance",
    "policy_mutation_allowed_by_acceptance",
    "model_mutation_allowed_by_acceptance",
    "runtime_memory_mutation_allowed_by_acceptance",
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


def _contract_record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("promotion_contract_record_id"))
    if explicit:
        return explicit
    candidate_id = _string(record.get("promotion_candidate_id"))
    return f"{candidate_id if candidate_id else 'unknown_sequence_promotion_contract'}:{index}"


def _acceptance_record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("promotion_acceptance_id"))
    if explicit:
        return explicit
    contract_id = _string(record.get("promotion_contract_record_id"))
    return f"{contract_id if contract_id else 'unknown_sequence_promotion_contract'}:{index}"


def _contract_envelope_violations(contract: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if contract.get("type") != REQUIRED_PROMOTION_CONTRACT_TYPE:
        violations.append("sequence_memory_trust_promotion_contract_type_mismatch")
    generated_at = contract.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("sequence_memory_trust_promotion_contract_generated_at_invalid")
    if (
        contract.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID
    ):
        violations.append("sequence_memory_trust_promotion_contract_method_id_mismatch")
    if (
        contract.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_STATUSES
    ):
        violations.append("sequence_memory_trust_promotion_contract_status_unknown")
    safety = _mapping(contract.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_memory_trust_promotion_contract_read_only_not_true")
    if safety.get("operator_required") is not True:
        violations.append("sequence_memory_trust_promotion_contract_operator_required_not_true")
    if safety.get("rollback_evidence_required") is not True:
        violations.append(
            "sequence_memory_trust_promotion_contract_rollback_evidence_required_not_true"
        )
    for flag in REQUIRED_CONTRACT_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"sequence_memory_trust_promotion_contract_{flag}_not_false")
    return tuple(violations)


def _contract_status_violations(
    contract: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = contract.get("status")
    if status == "sequence_memory_trust_promotion_contract_replacement_required":
        return ("sequence_memory_trust_promotion_contract_status_replacement_required",)
    if status == "sequence_memory_trust_promotion_contract_not_ready" and records:
        return ("sequence_memory_trust_promotion_contract_status_not_ready_with_records",)
    if status == "manual_sequence_memory_trust_promotion_contract_ready" and not records:
        return ("sequence_memory_trust_promotion_contract_status_ready_without_records",)
    return ()


def _contract_record_violations(record: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    for field in REQUIRED_CONTRACT_RECORD_FIELDS:
        if not _string(record.get(field)):
            violations.append(
                f"sequence_memory_trust_promotion_contract_record_{field}_missing"
            )
    reviewed_at = _string(record.get("reviewed_at"))
    if reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append("sequence_memory_trust_promotion_contract_record_reviewed_at_invalid")
    if record.get("read_only") is not True:
        violations.append("sequence_memory_trust_promotion_contract_record_read_only_not_true")
    if record.get("operator_review_required") is not True:
        violations.append(
            "sequence_memory_trust_promotion_contract_record_operator_review_not_true"
        )
    if record.get("rollback_evidence_required") is not True:
        violations.append(
            "sequence_memory_trust_promotion_contract_record_rollback_required_not_true"
        )
    if record.get("sequence_policy_promotion_ready_for_operator_review") is not True:
        violations.append(
            "sequence_memory_trust_promotion_contract_record_ready_for_review_not_true"
        )
    if record.get("promotion_executor_required") is not True:
        violations.append(
            "sequence_memory_trust_promotion_contract_record_executor_required_not_true"
        )
    for flag in REQUIRED_CONTRACT_RECORD_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(
                f"sequence_memory_trust_promotion_contract_record_{flag}_not_false"
            )
    return tuple(violations)


def _contract_record_replacement_ids(
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    return tuple(
        _contract_record_id(record, index)
        for index, record in enumerate(records)
        if _contract_record_violations(record)
    )


def _contract_summary_count_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
    contract_invalid_record_ids: Tuple[str, ...],
    unreviewed_candidate_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    expected_counts = {
        "contract_record_count": len(records),
        "invalid_record_count": len(contract_invalid_record_ids),
        "unreviewed_candidate_count": len(unreviewed_candidate_ids),
    }
    return tuple(
        f"sequence_memory_trust_promotion_contract_summary_{key}_mismatch"
        for key, expected in expected_counts.items()
        if _summary_int(summary.get(key)) != expected
    )


def _contract_summary_identity_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    expected_record_ids = tuple(
        record["promotion_contract_record_id"]
        for record in records
        if _string(record.get("promotion_contract_record_id"))
    )
    expected_proposal_ids = tuple(
        record["proposal_id"] for record in records if _string(record.get("proposal_id"))
    )
    violations = []
    if _string_list(summary.get("contract_record_ids")) != expected_record_ids:
        violations.append(
            "sequence_memory_trust_promotion_contract_summary_record_ids_mismatch"
        )
    if _string_list(summary.get("proposal_ids")) != expected_proposal_ids:
        violations.append(
            "sequence_memory_trust_promotion_contract_summary_proposal_ids_mismatch"
        )
    return tuple(violations)


def _contract_summary_safety_violations(summary: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if summary.get("operator_review_required") is not True:
        violations.append(
            "sequence_memory_trust_promotion_contract_summary_operator_review_not_true"
        )
    if summary.get("rollback_evidence_required") is not True:
        violations.append(
            "sequence_memory_trust_promotion_contract_summary_rollback_required_not_true"
        )
    for flag in REQUIRED_CONTRACT_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_memory_trust_promotion_contract_summary_{flag}_not_false"
            )
    return tuple(violations)


def _contract_records_by_id(
    promotion_contract: Mapping[str, Any],
) -> Dict[str, Mapping[str, Any]]:
    records: Dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(_record_list(promotion_contract.get("contract_records"))):
        record_id = _contract_record_id(record, index)
        records[record_id] = record
    return records


def _unsafe_flags(record: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(flag for flag in UNSAFE_ACCEPTANCE_FLAGS if record.get(flag) is True)


def _validated_record(
    record: Mapping[str, Any],
    *,
    index: int,
    contract_records_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    record_id = _acceptance_record_id(record, index)
    contract_record_id = _string(record.get("promotion_contract_record_id"))
    promotion_candidate_id = _string(record.get("promotion_candidate_id"))
    proposal_id = _string(record.get("proposal_id"))
    learning_scope = _string(record.get("learning_scope"))
    value_type = _string(record.get("value_type"))
    idempotency_key = _string(record.get("idempotency_key"))
    previous_policy_state = _string(record.get("previous_policy_state"))
    rollback_strategy = _string(record.get("rollback_strategy"))
    decision = _string(record.get("decision"))
    reviewed_by = _string(record.get("reviewed_by"))
    reviewed_at = _string(record.get("reviewed_at"))
    unsafe_flags = _unsafe_flags(record)
    contract_record = _mapping(contract_records_by_id.get(contract_record_id))
    violations = []

    for field in REQUIRED_PROMOTION_ACCEPTANCE_RECORD_FIELDS:
        if not _string(record.get(field)):
            violations.append(
                f"sequence_memory_trust_promotion_acceptance_record_{field}_missing"
            )
    if contract_record_id not in contract_records_by_id:
        violations.append("sequence_memory_trust_promotion_acceptance_record_contract_missing")
    if decision not in SEQUENCE_MEMORY_TRUST_PROMOTION_ACCEPTANCE_DECISIONS:
        violations.append("sequence_memory_trust_promotion_acceptance_record_decision_unknown")
    if reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append("sequence_memory_trust_promotion_acceptance_record_reviewed_at_invalid")
    if contract_record:
        expected_pairs = {
            "promotion_candidate_id": promotion_candidate_id,
            "proposal_id": proposal_id,
            "learning_scope": learning_scope,
            "value_type": value_type,
            "idempotency_key": idempotency_key,
            "previous_policy_state": previous_policy_state,
            "rollback_strategy": rollback_strategy,
        }
        for field, supplied in expected_pairs.items():
            if supplied != contract_record.get(field):
                violations.append(
                    f"sequence_memory_trust_promotion_acceptance_record_{field}_mismatch"
                )
    if unsafe_flags:
        violations.append("sequence_memory_trust_promotion_acceptance_record_unsafe_authority_claim")

    valid = not violations
    return {
        "promotion_acceptance_id": record_id,
        "promotion_contract_record_id": contract_record_id,
        "promotion_candidate_id": promotion_candidate_id,
        "proposal_id": proposal_id,
        "learning_scope": learning_scope,
        "value_type": value_type,
        "idempotency_key": idempotency_key,
        "previous_policy_state": previous_policy_state,
        "rollback_strategy": rollback_strategy,
        "decision": (
            decision
            if decision in SEQUENCE_MEMORY_TRUST_PROMOTION_ACCEPTANCE_DECISIONS
            else "invalid"
        ),
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at,
        "candidate_present": bool(contract_record),
        "operator_accepted_sequence_memory_trust_promotion_contract": (
            valid and decision == "accepted"
        ),
        "operator_rejected_sequence_memory_trust_promotion_contract": (
            valid and decision == "rejected"
        ),
        "read_only": True,
        "live_write": False,
        "apply_attempted": False,
        "mutates_console_state": False,
        "queries_console": False,
        "operator_review_required": True,
        "queue_mutation": False,
        "policy_mutation": False,
        "model_mutation": False,
        "runtime_memory_mutation": False,
        "execution_allowed_by_acceptance": False,
        "sequence_policy_promotion_allowed_by_acceptance": False,
        "queue_mutation_allowed_by_acceptance": False,
        "policy_mutation_allowed_by_acceptance": False,
        "model_mutation_allowed_by_acceptance": False,
        "runtime_memory_mutation_allowed_by_acceptance": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "can_import_to_queue": False,
        "can_apply_live": False,
        "unsafe_authority_flags": list(unsafe_flags),
        "valid": valid,
        "violations": violations,
    }


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger(
    *,
    promotion_contract: Mapping[str, Any] | None = None,
    acceptance_records: Iterable[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build a read-only ledger for sequence-promotion contract decisions."""

    contract = (
        _mapping(promotion_contract)
        or build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract()
    )
    summary = _mapping(contract.get("summary"))
    contract_records = _record_list(contract.get("contract_records"))
    contract_invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    unreviewed_candidate_ids = _string_list(summary.get("unreviewed_candidate_ids"))
    contract_violations = tuple(
        dict.fromkeys(
            (
                *contract_invalid_record_ids,
                *_contract_envelope_violations(contract),
                *_contract_status_violations(contract, records=contract_records),
                *_contract_record_replacement_ids(contract_records),
                *_contract_summary_count_violations(
                    summary,
                    records=contract_records,
                    contract_invalid_record_ids=contract_invalid_record_ids,
                    unreviewed_candidate_ids=unreviewed_candidate_ids,
                ),
                *_contract_summary_identity_violations(
                    summary,
                    records=contract_records,
                ),
                *_contract_summary_safety_violations(summary),
            )
        )
    )
    records_by_id = _contract_records_by_id(contract)
    records = tuple(
        _validated_record(
            record,
            index=index,
            contract_records_by_id=records_by_id,
        )
        for index, record in enumerate(_record_list(acceptance_records))
    )
    valid_records = tuple(record for record in records if record["valid"])
    accepted_records = tuple(
        record
        for record in valid_records
        if record["operator_accepted_sequence_memory_trust_promotion_contract"]
    )
    rejected_records = tuple(
        record
        for record in valid_records
        if record["operator_rejected_sequence_memory_trust_promotion_contract"]
    )
    invalid_records = tuple(record for record in records if not record["valid"])
    reviewed_contract_record_ids = {
        str(record["promotion_contract_record_id"])
        for record in valid_records
        if isinstance(record.get("promotion_contract_record_id"), str)
    }
    unreviewed_contract_record_ids = tuple(
        record_id for record_id in records_by_id if record_id not in reviewed_contract_record_ids
    )

    if contract_violations or invalid_records:
        status = "operator_sequence_memory_trust_promotion_acceptance_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_memory_trust_promotion_acceptance_records",
        )
    elif unreviewed_contract_record_ids:
        status = "operator_sequence_memory_trust_promotion_acceptance_required"
        blocked_by = (
            "operator_must_review_sequence_memory_trust_promotion_contract_records",
        )
    elif accepted_records or rejected_records:
        status = "operator_sequence_memory_trust_promotion_acceptance_records_available"
        blocked_by = (
            "sequence_policy_promotion_executor_still_requires_separate_contract",
        )
    else:
        status = "operator_sequence_memory_trust_promotion_acceptance_required"
        blocked_by = (
            "valid_sequence_memory_trust_promotion_contract_records_required_before_acceptance",
        )

    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *contract_violations,
                *(record["promotion_acceptance_id"] for record in invalid_records),
            )
        )
    )
    accepted_proposal_ids = [record["proposal_id"] for record in accepted_records]
    rejected_proposal_ids = [record["proposal_id"] for record in rejected_records]
    return {
        "type": "operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger",
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "apply_attempted": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "execution_allowed_by_acceptance": False,
            "sequence_policy_promotion_allowed_by_acceptance": False,
            "queue_mutation_allowed_by_acceptance": False,
            "policy_mutation_allowed_by_acceptance": False,
            "model_mutation_allowed_by_acceptance": False,
            "runtime_memory_mutation_allowed_by_acceptance": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "promotion_contract_status": contract.get("status") or "missing",
            "promotion_contract_record_count": len(contract_records),
            "acceptance_record_count": len(records),
            "accepted_record_count": len(accepted_records),
            "rejected_record_count": len(rejected_records),
            "invalid_record_count": len(invalid_record_ids),
            "unreviewed_contract_record_count": len(unreviewed_contract_record_ids),
            "accepted_proposal_ids": accepted_proposal_ids,
            "rejected_proposal_ids": rejected_proposal_ids,
            "operator_review_required": True,
            "execution_allowed_by_acceptance": False,
            "sequence_policy_promotion_allowed_by_acceptance": False,
            "queue_mutation_allowed_by_acceptance": False,
            "policy_mutation_allowed_by_acceptance": False,
            "model_mutation_allowed_by_acceptance": False,
            "runtime_memory_mutation_allowed_by_acceptance": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "invalid_record_ids": list(invalid_record_ids),
            "unreviewed_contract_record_ids": list(unreviewed_contract_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid sequence promotion acceptance records."
                if invalid_record_ids
                else (
                    "Operator reviews all sequence promotion contract records before executor planning."
                    if unreviewed_contract_record_ids
                    else (
                        "Sequence promotion acceptance is evidence only; a separate "
                        "executor is required before policy or execution influence."
                    )
                )
            ),
        },
        "records": list(records),
    }
