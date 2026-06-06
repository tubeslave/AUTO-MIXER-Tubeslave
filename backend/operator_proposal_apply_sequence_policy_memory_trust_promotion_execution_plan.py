"""Read-only execution plan for sequence-policy memory trust promotion.

The plan turns accepted sequence promotion-contract decisions into non-executing
execution candidates. It does not promote sequence policy, execute proposals,
mutate queues, policy stores, models, runtime memory, append-only memory, or
touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_STATUSES = {
    "sequence_memory_trust_promotion_execution_plan_not_ready",
    "sequence_memory_trust_promotion_execution_plan_replacement_required",
    "manual_sequence_memory_trust_promotion_execution_candidate_available",
}
REQUIRED_PROMOTION_ACCEPTANCE_LEDGER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger"
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
REQUIRED_ACCEPTANCE_FALSE_SAFETY_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "execution_allowed_by_acceptance",
    "sequence_policy_promotion_allowed_by_acceptance",
    "queue_mutation_allowed_by_acceptance",
    "policy_mutation_allowed_by_acceptance",
    "model_mutation_allowed_by_acceptance",
    "runtime_memory_mutation_allowed_by_acceptance",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_ACCEPTANCE_FALSE_SUMMARY_FLAGS = (
    "execution_allowed_by_acceptance",
    "sequence_policy_promotion_allowed_by_acceptance",
    "queue_mutation_allowed_by_acceptance",
    "policy_mutation_allowed_by_acceptance",
    "model_mutation_allowed_by_acceptance",
    "runtime_memory_mutation_allowed_by_acceptance",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_ACCEPTANCE_RECORD_FALSE_FLAGS = (
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
    explicit = _string(record.get("promotion_acceptance_id"))
    if explicit:
        return explicit
    contract_id = _string(record.get("promotion_contract_record_id"))
    return f"{contract_id if contract_id else 'unknown_sequence_promotion_acceptance'}:{index}"


def _ledger_envelope_violations(ledger: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if ledger.get("type") != REQUIRED_PROMOTION_ACCEPTANCE_LEDGER_TYPE:
        violations.append("sequence_memory_trust_promotion_acceptance_ledger_type_mismatch")
    generated_at = ledger.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("sequence_memory_trust_promotion_acceptance_ledger_generated_at_invalid")
    if (
        ledger.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID
    ):
        violations.append("sequence_memory_trust_promotion_acceptance_ledger_method_id_mismatch")
    if (
        ledger.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_STATUSES
    ):
        violations.append("sequence_memory_trust_promotion_acceptance_ledger_status_unknown")
    safety = _mapping(ledger.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_memory_trust_promotion_acceptance_ledger_read_only_not_true")
    if safety.get("operator_required") is not True:
        violations.append(
            "sequence_memory_trust_promotion_acceptance_ledger_operator_required_not_true"
        )
    for flag in REQUIRED_ACCEPTANCE_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_memory_trust_promotion_acceptance_ledger_{flag}_not_false"
            )
    return tuple(violations)


def _ledger_status_violations(
    ledger: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = ledger.get("status")
    if status == "operator_sequence_memory_trust_promotion_acceptance_replacement_required":
        return ("sequence_memory_trust_promotion_acceptance_status_replacement_required",)
    if (
        status == "operator_sequence_memory_trust_promotion_acceptance_records_available"
        and not records
    ):
        return ("sequence_memory_trust_promotion_acceptance_status_available_without_records",)
    return ()


def _accepted_record_violations(record: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    for field in REQUIRED_PROMOTION_ACCEPTANCE_RECORD_FIELDS:
        if not _string(record.get(field)):
            violations.append(
                f"sequence_memory_trust_promotion_execution_record_{field}_missing"
            )
    reviewed_at = _string(record.get("reviewed_at"))
    if reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append("sequence_memory_trust_promotion_execution_record_reviewed_at_invalid")
    if record.get("valid") is not True:
        violations.append("sequence_memory_trust_promotion_execution_record_not_valid")
    if record.get("candidate_present") is not True:
        violations.append("sequence_memory_trust_promotion_execution_record_contract_missing")
    if record.get("decision") != "accepted":
        violations.append("sequence_memory_trust_promotion_execution_record_decision_not_accepted")
    if record.get("operator_accepted_sequence_memory_trust_promotion_contract") is not True:
        violations.append("sequence_memory_trust_promotion_execution_record_accepted_not_true")
    if record.get("operator_rejected_sequence_memory_trust_promotion_contract") is not False:
        violations.append("sequence_memory_trust_promotion_execution_record_rejected_not_false")
    if record.get("read_only") is not True:
        violations.append("sequence_memory_trust_promotion_execution_record_read_only_not_true")
    if record.get("operator_review_required") is not True:
        violations.append(
            "sequence_memory_trust_promotion_execution_record_operator_review_not_true"
        )
    if _string_list(record.get("unsafe_authority_flags")):
        violations.append(
            "sequence_memory_trust_promotion_execution_record_unsafe_authority_claim"
        )
    for flag in REQUIRED_ACCEPTANCE_RECORD_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(
                f"sequence_memory_trust_promotion_execution_record_{flag}_not_false"
            )
    return tuple(violations)


def _accepted_record_replacement_ids(
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    return tuple(
        _record_id(record, index)
        for index, record in enumerate(records)
        if record.get("operator_accepted_sequence_memory_trust_promotion_contract")
        is True
        and _accepted_record_violations(record)
    )


def _ambiguous_contract_review_record_ids(
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    record_ids_by_contract: Dict[str, list[str]] = {}
    for index, record in enumerate(records):
        if _accepted_record_violations(record):
            continue
        contract_id = _string(record.get("promotion_contract_record_id"))
        if not contract_id:
            continue
        record_ids_by_contract.setdefault(contract_id, []).append(_record_id(record, index))
    return tuple(
        record_id
        for record_ids in record_ids_by_contract.values()
        if len(record_ids) > 1
        for record_id in record_ids
    )


def _ledger_summary_count_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
    invalid_record_ids: Tuple[str, ...],
    unreviewed_contract_record_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    expected_counts = {
        "acceptance_record_count": len(records),
        "accepted_record_count": sum(
            1
            for record in records
            if record.get("valid") is True
            and record.get("operator_accepted_sequence_memory_trust_promotion_contract")
            is True
        ),
        "rejected_record_count": sum(
            1
            for record in records
            if record.get("valid") is True
            and record.get("operator_rejected_sequence_memory_trust_promotion_contract")
            is True
        ),
        "invalid_record_count": len(invalid_record_ids),
        "unreviewed_contract_record_count": len(unreviewed_contract_record_ids),
    }
    return tuple(
        f"sequence_memory_trust_promotion_acceptance_summary_{key}_mismatch"
        for key, expected in expected_counts.items()
        if _summary_int(summary.get(key)) != expected
    )


def _proposal_ids_for_decision(
    records: Tuple[Mapping[str, Any], ...],
    *,
    accepted: bool,
) -> Tuple[str, ...]:
    flag = (
        "operator_accepted_sequence_memory_trust_promotion_contract"
        if accepted
        else "operator_rejected_sequence_memory_trust_promotion_contract"
    )
    return tuple(
        record["proposal_id"]
        for record in records
        if record.get("valid") is True
        and record.get(flag) is True
        and _string(record.get("proposal_id"))
    )


def _ledger_summary_identity_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    expected = {
        "accepted_proposal_ids": _proposal_ids_for_decision(records, accepted=True),
        "rejected_proposal_ids": _proposal_ids_for_decision(records, accepted=False),
    }
    return tuple(
        f"sequence_memory_trust_promotion_acceptance_summary_{key}_mismatch"
        for key, values in expected.items()
        if _string_list(summary.get(key)) != values
    )


def _ledger_summary_safety_violations(summary: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if summary.get("operator_review_required") is not True:
        violations.append(
            "sequence_memory_trust_promotion_acceptance_summary_operator_review_not_true"
        )
    for flag in REQUIRED_ACCEPTANCE_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_memory_trust_promotion_acceptance_summary_{flag}_not_false"
            )
    return tuple(violations)


def _accepted_records(
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[Mapping[str, Any], ...]:
    return tuple(
        record
        for record in records
        if record.get("operator_accepted_sequence_memory_trust_promotion_contract")
        is True
        and not _accepted_record_violations(record)
    )


def _execution_candidate(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "promotion_execution_candidate_id": (
            "operator_sequence_memory_trust_promotion_execution:"
            f"{record.get('promotion_acceptance_id')}"
        ),
        "promotion_acceptance_id": record.get("promotion_acceptance_id"),
        "promotion_contract_record_id": record.get("promotion_contract_record_id"),
        "promotion_candidate_id": record.get("promotion_candidate_id"),
        "proposal_id": record.get("proposal_id"),
        "learning_scope": record.get("learning_scope"),
        "value_type": record.get("value_type"),
        "idempotency_key": record.get("idempotency_key"),
        "previous_policy_state": record.get("previous_policy_state"),
        "rollback_strategy": record.get("rollback_strategy"),
        "reviewed_by": record.get("reviewed_by"),
        "reviewed_at": record.get("reviewed_at"),
        "read_only": True,
        "live_write": False,
        "apply_attempted": False,
        "mutates_console_state": False,
        "queries_console": False,
        "operator_review_required": True,
        "separate_executor_required": True,
        "manual_sequence_policy_promotion_execution_required": True,
        "execution_plan_ready_for_operator_review": True,
        "execution_allowed_by_plan": False,
        "sequence_policy_promotion_allowed_by_plan": False,
        "queue_mutation_allowed_by_plan": False,
        "policy_mutation_allowed_by_plan": False,
        "model_mutation_allowed_by_plan": False,
        "runtime_memory_mutation_allowed_by_plan": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan(
    *,
    promotion_acceptance_ledger: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a read-only plan for future sequence-policy promotion execution."""

    ledger = (
        _mapping(promotion_acceptance_ledger)
        or build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger()
    )
    summary = _mapping(ledger.get("summary"))
    records = _record_list(ledger.get("records"))
    ledger_invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    unreviewed_contract_record_ids = _string_list(
        summary.get("unreviewed_contract_record_ids")
    )
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *ledger_invalid_record_ids,
                *_ledger_envelope_violations(ledger),
                *_ledger_status_violations(ledger, records=records),
                *_accepted_record_replacement_ids(records),
                *_ambiguous_contract_review_record_ids(records),
                *_ledger_summary_count_violations(
                    summary,
                    records=records,
                    invalid_record_ids=ledger_invalid_record_ids,
                    unreviewed_contract_record_ids=unreviewed_contract_record_ids,
                ),
                *_ledger_summary_identity_violations(summary, records=records),
                *_ledger_summary_safety_violations(summary),
            )
        )
    )
    accepted_records = _accepted_records(records)
    execution_source_records = () if invalid_record_ids else accepted_records
    execution_candidates = tuple(
        _execution_candidate(record) for record in execution_source_records
    )
    rejected_proposal_ids = _string_list(summary.get("rejected_proposal_ids"))

    if invalid_record_ids:
        status = "sequence_memory_trust_promotion_execution_plan_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_promotion_acceptance_before_execution_plan",
        )
    elif unreviewed_contract_record_ids:
        status = "sequence_memory_trust_promotion_execution_plan_not_ready"
        blocked_by = (
            "operator_must_review_all_sequence_promotion_contract_records_before_execution_plan",
        )
    elif execution_candidates:
        status = "manual_sequence_memory_trust_promotion_execution_candidate_available"
        blocked_by = (
            "execution_plan_is_not_executor_authority_for_sequence_policy_or_live_apply",
        )
    else:
        status = "sequence_memory_trust_promotion_execution_plan_not_ready"
        blocked_by = (
            "accepted_sequence_memory_trust_promotion_record_required_before_execution_plan",
        )

    return {
        "type": "operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan",
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "apply_attempted": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "execution_allowed_by_plan": False,
            "sequence_policy_promotion_allowed_by_plan": False,
            "queue_mutation_allowed_by_plan": False,
            "policy_mutation_allowed_by_plan": False,
            "model_mutation_allowed_by_plan": False,
            "runtime_memory_mutation_allowed_by_plan": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "promotion_acceptance_ledger_status": ledger.get("status") or "missing",
            "acceptance_record_count": len(records),
            "accepted_record_count": len(accepted_records),
            "rejected_record_count": int(summary.get("rejected_record_count") or 0),
            "invalid_record_count": len(invalid_record_ids),
            "unreviewed_contract_record_count": len(unreviewed_contract_record_ids),
            "execution_candidate_count": len(execution_candidates),
            "execution_candidate_ids": [
                str(candidate["promotion_execution_candidate_id"])
                for candidate in execution_candidates
                if isinstance(candidate.get("promotion_execution_candidate_id"), str)
            ],
            "accepted_proposal_ids": [
                str(candidate["proposal_id"])
                for candidate in execution_candidates
                if isinstance(candidate.get("proposal_id"), str)
            ],
            "rejected_proposal_ids": list(rejected_proposal_ids),
            "operator_review_required": True,
            "execution_allowed_by_plan": False,
            "sequence_policy_promotion_allowed_by_plan": False,
            "queue_mutation_allowed_by_plan": False,
            "policy_mutation_allowed_by_plan": False,
            "model_mutation_allowed_by_plan": False,
            "runtime_memory_mutation_allowed_by_plan": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "invalid_record_ids": list(invalid_record_ids),
            "unreviewed_contract_record_ids": list(unreviewed_contract_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid sequence promotion acceptance records before execution planning."
                if invalid_record_ids
                else (
                    "Operator reviews all sequence promotion contract records before execution planning."
                    if unreviewed_contract_record_ids
                    else (
                        "A separate executor implementation and safety contract are "
                        "still required before sequence policy or live execution influence."
                    )
                )
            ),
        },
        "execution_candidates": list(execution_candidates),
    }
