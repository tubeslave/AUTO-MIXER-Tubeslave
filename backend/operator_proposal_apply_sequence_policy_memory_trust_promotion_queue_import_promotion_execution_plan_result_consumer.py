"""Read-only consumer for sequence-policy promotion execution plans.

This verifier connects a queue-import promotion acceptance-ledger result to the
existing promotion executor dry-run builder. It verifies execution-plan
candidates against accepted promotion records and emits only a dry-run request
template. It does not run dry-run itself, execute proposals, promote policy,
mutate queues, policy stores, runtime memory, models, append-only memory,
proposal state, idempotency, or mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_STATUSES,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ACCEPTANCE_LEDGER_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_STATUSES,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ACCEPTANCE_LEDGER_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_PLAN_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_PLAN_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_executor_dry_run_request_available",
}
REQUIRED_PROMOTION_ACCEPTANCE_LEDGER_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer"
)
REQUIRED_PROMOTION_ACCEPTANCE_LEDGER_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_execution_plan_request_available"
)
REQUIRED_PROMOTION_EXECUTION_PLAN_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan"
)
REQUIRED_PROMOTION_EXECUTION_PLAN_AVAILABLE_STATUS = (
    "manual_sequence_memory_trust_promotion_execution_candidate_available"
)
REQUIRED_PROMOTION_ACCEPTANCE_LEDGER_STATUS = (
    "operator_sequence_memory_trust_promotion_acceptance_records_available"
)
REQUIRED_CONSUMER_FALSE_FLAGS = (
    "promotion_execution_plan_handler_invoked",
    "promotion_execution_plan_created",
    "memory_write_allowed_by_consumer",
    "execution_allowed_by_consumer",
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "policy_mutation",
    "memory_mutation",
    "memory_file_mutation",
    "runtime_memory_mutation",
    "model_mutation",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_apply_live",
)
REQUIRED_REQUEST_FALSE_FLAGS = (
    "memory_write_allowed_by_consumer",
    "execution_allowed_by_consumer",
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "policy_mutation",
    "memory_mutation",
    "memory_file_mutation",
    "runtime_memory_mutation",
    "model_mutation",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_apply_live",
)
REQUIRED_PLAN_FALSE_SAFETY_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "execution_allowed_by_plan",
    "sequence_policy_promotion_allowed_by_plan",
    "queue_mutation_allowed_by_plan",
    "policy_mutation_allowed_by_plan",
    "model_mutation_allowed_by_plan",
    "runtime_memory_mutation_allowed_by_plan",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_PLAN_FALSE_SUMMARY_FLAGS = (
    "execution_allowed_by_plan",
    "sequence_policy_promotion_allowed_by_plan",
    "queue_mutation_allowed_by_plan",
    "policy_mutation_allowed_by_plan",
    "model_mutation_allowed_by_plan",
    "runtime_memory_mutation_allowed_by_plan",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_CANDIDATE_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "execution_allowed_by_plan",
    "sequence_policy_promotion_allowed_by_plan",
    "queue_mutation_allowed_by_plan",
    "policy_mutation_allowed_by_plan",
    "model_mutation_allowed_by_plan",
    "runtime_memory_mutation_allowed_by_plan",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_CANDIDATE_FIELDS = (
    "promotion_execution_candidate_id",
    "promotion_acceptance_id",
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


def _candidate_id(candidate: Mapping[str, Any], index: int) -> str:
    explicit = _string(candidate.get("promotion_execution_candidate_id"))
    if explicit:
        return explicit
    acceptance_id = _string(candidate.get("promotion_acceptance_id"))
    return f"{acceptance_id if acceptance_id else 'unknown_sequence_promotion_execution'}:{index}"


def _accepted_records_from_request(
    request: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    ledger = _mapping(request.get("promotion_acceptance_ledger"))
    return tuple(
        record
        for record in _record_list(ledger.get("records"))
        if record.get("valid") is True
        and record.get("operator_accepted_sequence_memory_trust_promotion_contract")
        is True
    )


def _rejected_proposal_ids_from_request(request: Mapping[str, Any]) -> Tuple[str, ...]:
    ledger = _mapping(request.get("promotion_acceptance_ledger"))
    return tuple(
        str(record["proposal_id"])
        for record in _record_list(ledger.get("records"))
        if record.get("valid") is True
        and record.get("operator_rejected_sequence_memory_trust_promotion_contract")
        is True
        and isinstance(record.get("proposal_id"), str)
        and record.get("proposal_id")
    )


def _candidates_by_acceptance_id(
    candidates: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(candidate["promotion_acceptance_id"]): candidate
        for candidate in candidates
        if isinstance(candidate.get("promotion_acceptance_id"), str)
        and candidate.get("promotion_acceptance_id")
    }


def _consumer_envelope_violations(consumer: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_PROMOTION_ACCEPTANCE_LEDGER_RESULT_CONSUMER_TYPE:
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_acceptance_consumer_type_mismatch"
        )
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_acceptance_consumer_method_id_mismatch"
        )
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ACCEPTANCE_LEDGER_RESULT_CONSUMER_STATUSES
    ):
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_acceptance_consumer_status_unknown"
        )
    if consumer.get("status") != REQUIRED_PROMOTION_ACCEPTANCE_LEDGER_RESULT_CONSUMER_AVAILABLE_STATUS:
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_acceptance_consumer_status_not_available"
        )
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_acceptance_consumer_read_only_not_true"
        )
    if safety.get("promotion_acceptance_ledger_result_consumer_only") is not True:
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_acceptance_consumer_only_not_true"
        )
    if safety.get("promotion_execution_plan_request_template_only") is not True:
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_acceptance_consumer_template_only_not_true"
        )
    if safety.get("accepted_promotion_record_required") is not True:
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_acceptance_consumer_accepted_record_required_not_true"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_execution_plan_result_consumer_acceptance_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    accepted_records: Tuple[Mapping[str, Any], ...],
    rejected_proposal_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    expected_counts = {
        "accepted_record_count": len(accepted_records),
        "rejected_record_count": len(rejected_proposal_ids),
        "invalid_record_count": 0,
        "unreviewed_contract_record_count": 0,
    }
    violations = []
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_promotion_execution_plan_result_consumer_acceptance_consumer_summary_{key}_mismatch"
            )
    if summary.get("promotion_execution_plan_request_available") is not True:
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_acceptance_consumer_request_not_available"
        )
    if _string_list(summary.get("accepted_proposal_ids")) != tuple(
        str(record["proposal_id"])
        for record in accepted_records
        if isinstance(record.get("proposal_id"), str) and record.get("proposal_id")
    ):
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_acceptance_consumer_accepted_proposal_ids_mismatch"
        )
    if _string_list(summary.get("rejected_proposal_ids")) != rejected_proposal_ids:
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_acceptance_consumer_rejected_proposal_ids_mismatch"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_execution_plan_result_consumer_acceptance_consumer_summary_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_request_violations(request: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if (
        request.get("handler")
        != "plan_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution"
    ):
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_request_handler_mismatch"
        )
    if request.get("read_only_template") is not True:
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_request_read_only_template_not_true"
        )
    if request.get("promotion_execution_plan_request_template_only") is not True:
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_request_template_only_not_true"
        )
    if request.get("promotion_execution_plan_handler_invoked") is not False:
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_request_handler_invoked_not_false"
        )
    if request.get("promotion_execution_plan_created") is not False:
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_request_plan_created_not_false"
        )
    if request.get("accepted_promotion_record_required") is not True:
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_request_accepted_record_required_not_true"
        )
    ledger = _mapping(request.get("promotion_acceptance_ledger"))
    if ledger.get("status") != REQUIRED_PROMOTION_ACCEPTANCE_LEDGER_STATUS:
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_request_acceptance_ledger_not_available"
        )
    for flag in REQUIRED_REQUEST_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_execution_plan_result_consumer_request_{flag}_not_false"
            )
    return tuple(violations)


def _plan_envelope_violations(plan: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if plan.get("type") != REQUIRED_PROMOTION_EXECUTION_PLAN_TYPE:
        violations.append("sequence_policy_promotion_execution_plan_result_consumer_plan_type_mismatch")
    generated_at = plan.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("sequence_policy_promotion_execution_plan_result_consumer_plan_generated_at_invalid")
    if plan.get("method_id") != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_METHOD_ID:
        violations.append("sequence_policy_promotion_execution_plan_result_consumer_plan_method_id_mismatch")
    if plan.get("source_method_id") != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID:
        violations.append("sequence_policy_promotion_execution_plan_result_consumer_plan_source_method_mismatch")
    if plan.get("status") not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_STATUSES:
        violations.append("sequence_policy_promotion_execution_plan_result_consumer_plan_status_unknown")
    safety = _mapping(plan.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_policy_promotion_execution_plan_result_consumer_plan_read_only_not_true")
    if safety.get("operator_required") is not True:
        violations.append("sequence_policy_promotion_execution_plan_result_consumer_plan_operator_required_not_true")
    for flag in REQUIRED_PLAN_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_execution_plan_result_consumer_plan_{flag}_not_false"
            )
    return tuple(violations)


def _plan_summary_violations(
    summary: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
    accepted_records: Tuple[Mapping[str, Any], ...],
    rejected_proposal_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    expected_counts = {
        "execution_candidate_count": len(candidates),
        "accepted_record_count": len(accepted_records),
        "rejected_record_count": len(rejected_proposal_ids),
        "invalid_record_count": 0,
        "unreviewed_contract_record_count": 0,
    }
    violations = []
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_promotion_execution_plan_result_consumer_plan_summary_{key}_mismatch"
            )
    if (
        summary.get("promotion_acceptance_ledger_status")
        != REQUIRED_PROMOTION_ACCEPTANCE_LEDGER_STATUS
    ):
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_plan_summary_acceptance_status_mismatch"
        )
    if summary.get("operator_review_required") is not True:
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_plan_summary_operator_review_not_true"
        )
    expected_candidate_ids = tuple(
        str(candidate["promotion_execution_candidate_id"])
        for candidate in candidates
        if isinstance(candidate.get("promotion_execution_candidate_id"), str)
        and candidate.get("promotion_execution_candidate_id")
    )
    if _string_list(summary.get("execution_candidate_ids")) != expected_candidate_ids:
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_plan_summary_candidate_ids_mismatch"
        )
    expected_accepted_proposal_ids = tuple(
        str(record["proposal_id"])
        for record in accepted_records
        if isinstance(record.get("proposal_id"), str) and record.get("proposal_id")
    )
    if _string_list(summary.get("accepted_proposal_ids")) != expected_accepted_proposal_ids:
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_plan_summary_accepted_proposal_ids_mismatch"
        )
    if _string_list(summary.get("rejected_proposal_ids")) != rejected_proposal_ids:
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_plan_summary_rejected_proposal_ids_mismatch"
        )
    if _string_list(summary.get("invalid_record_ids")):
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_plan_summary_invalid_record_ids_not_empty"
        )
    if _string_list(summary.get("unreviewed_contract_record_ids")):
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_plan_summary_unreviewed_contract_ids_not_empty"
        )
    for flag in REQUIRED_PLAN_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_execution_plan_result_consumer_plan_summary_{flag}_not_false"
            )
    return tuple(violations)


def _candidate_violations(
    candidate: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    index: int,
) -> Tuple[str, ...]:
    candidate_id = _candidate_id(candidate, index)
    violations = []
    for field in REQUIRED_CANDIDATE_FIELDS:
        if not _string(candidate.get(field)):
            violations.append(f"{candidate_id}:{field}_missing")
    reviewed_at = _string(candidate.get("reviewed_at"))
    if reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append(f"{candidate_id}:reviewed_at_invalid")
    expected_pairs = {
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
    }
    for field, expected in expected_pairs.items():
        if candidate.get(field) != expected:
            violations.append(f"{candidate_id}:{field}_mismatch")
    if candidate.get("read_only") is not True:
        violations.append(f"{candidate_id}:read_only_not_true")
    if candidate.get("operator_review_required") is not True:
        violations.append(f"{candidate_id}:operator_review_required_not_true")
    if candidate.get("separate_executor_required") is not True:
        violations.append(f"{candidate_id}:separate_executor_required_not_true")
    if candidate.get("manual_sequence_policy_promotion_execution_required") is not True:
        violations.append(f"{candidate_id}:manual_execution_required_not_true")
    if candidate.get("execution_plan_ready_for_operator_review") is not True:
        violations.append(f"{candidate_id}:ready_for_operator_review_not_true")
    for flag in REQUIRED_CANDIDATE_FALSE_FLAGS:
        if candidate.get(flag) is not False:
            violations.append(f"{candidate_id}:{flag}_not_false")
    return tuple(violations)


def _candidate_record_identity_violations(
    candidates: Tuple[Mapping[str, Any], ...],
    accepted_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    candidates_by_acceptance_id = _candidates_by_acceptance_id(candidates)
    violations = []
    if set(candidates_by_acceptance_id) != {
        str(record["promotion_acceptance_id"])
        for record in accepted_records
        if isinstance(record.get("promotion_acceptance_id"), str)
        and record.get("promotion_acceptance_id")
    }:
        violations.append(
            "sequence_policy_promotion_execution_plan_result_consumer_candidate_acceptance_ids_mismatch"
        )
    for index, record in enumerate(accepted_records):
        acceptance_id = _string(record.get("promotion_acceptance_id"))
        candidate = _mapping(candidates_by_acceptance_id.get(acceptance_id))
        if not candidate:
            violations.append(f"{acceptance_id}:execution_candidate_missing")
            continue
        violations.extend(_candidate_violations(candidate, record=record, index=index))
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer(
    *,
    promotion_acceptance_ledger_result_consumer: Mapping[str, Any] | None = None,
    promotion_execution_plan_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify promotion execution-plan evidence and emit dry-run request."""

    consumer = _mapping(promotion_acceptance_ledger_result_consumer)
    plan = _mapping(promotion_execution_plan_result)
    request = _mapping(consumer.get("promotion_execution_plan_request"))
    accepted_records = _accepted_records_from_request(request)
    rejected_proposal_ids = _rejected_proposal_ids_from_request(request)
    candidates = _record_list(plan.get("execution_candidates"))
    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_consumer_envelope_violations(consumer),
                *_consumer_summary_violations(
                    _mapping(consumer.get("summary")),
                    accepted_records=accepted_records,
                    rejected_proposal_ids=rejected_proposal_ids,
                ),
                *_consumer_request_violations(request),
                *_plan_envelope_violations(plan),
                *_plan_summary_violations(
                    _mapping(plan.get("summary")),
                    candidates=candidates,
                    accepted_records=accepted_records,
                    rejected_proposal_ids=rejected_proposal_ids,
                ),
                *(
                    ("sequence_policy_promotion_execution_plan_result_consumer_plan_status_not_available",)
                    if plan.get("status")
                    != REQUIRED_PROMOTION_EXECUTION_PLAN_AVAILABLE_STATUS
                    else ()
                ),
                *(
                    ("sequence_policy_promotion_execution_plan_result_consumer_accepted_records_required",)
                    if not accepted_records
                    else ()
                ),
                *(
                    ("sequence_policy_promotion_execution_plan_result_consumer_candidate_count_mismatch",)
                    if len(candidates) != len(accepted_records)
                    else ()
                ),
                *_candidate_record_identity_violations(candidates, accepted_records),
            )
        )
    )
    invalid_record_ids = () if not consumer or not plan else raw_invalid_record_ids

    if not consumer or not plan:
        status = "sequence_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer_not_ready"
        blocked_by = (
            "promotion_acceptance_ledger_consumer_and_execution_plan_result_required",
        )
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_memory_trust_promotion_execution_plan_evidence",
        )
    elif plan.get("status") != REQUIRED_PROMOTION_EXECUTION_PLAN_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer_blocked"
        blocked_by = (
            "sequence_memory_trust_promotion_execution_candidates_must_be_available_before_dry_run",
        )
    elif not candidates:
        status = "sequence_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer_not_ready"
        blocked_by = ("sequence_memory_trust_promotion_execution_candidates_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_executor_dry_run_request_available"
        blocked_by = (
            "operator_may_request_read_only_sequence_memory_trust_promotion_executor_dry_run",
        )

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_executor_dry_run_request_available"
    )
    executor_dry_run_request: Dict[str, Any] = {
        "handler": "dry_run_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor",
        "read_only_template": True,
        "promotion_execution_plan_result_consumer_only": True,
        "promotion_executor_dry_run_request_template_only": True,
        "promotion_executor_dry_run_handler_invoked": False,
        "promotion_executor_dry_run_created": False,
        "apply": False,
        "operator_ack": False,
        "operator_must_keep_apply_false_for_dry_run": True,
        "memory_write_allowed_by_consumer": False,
        "execution_allowed_by_consumer": False,
        "live_write": False,
        "apply_attempted": False,
        "queue_mutation": False,
        "proposal_status_mutation": False,
        "idempotency_reserved": False,
        "manual_console_write_called": False,
        "policy_mutation": False,
        "memory_mutation": False,
        "memory_file_mutation": False,
        "runtime_memory_mutation": False,
        "model_mutation": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "can_apply_live": False,
    }
    if request_available:
        executor_dry_run_request["promotion_execution_plan"] = plan

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_PLAN_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_execution_plan_result_consumer_only": True,
            "promotion_executor_dry_run_request_template_only": True,
            "requires_promotion_acceptance_ledger_result_consumer": True,
            "requires_promotion_execution_plan_result": True,
            "promotion_executor_dry_run_handler_invoked": False,
            "promotion_executor_dry_run_created": False,
            "operator_acceptance_required": True,
            "memory_write_allowed_by_consumer": False,
            "execution_allowed_by_consumer": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "policy_mutation": False,
            "memory_mutation": False,
            "memory_file_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
        },
        "summary": {
            "expected_execution_candidate_count": len(accepted_records),
            "execution_candidate_count": len(candidates),
            "accepted_record_count": len(accepted_records),
            "rejected_record_count": len(rejected_proposal_ids),
            "invalid_record_count": len(invalid_record_ids),
            "promotion_execution_plan_result_consumer_only": True,
            "promotion_executor_dry_run_request_template_only": True,
            "promotion_executor_dry_run_request_available": request_available,
            "promotion_executor_dry_run_handler_invoked": False,
            "promotion_executor_dry_run_created": False,
            "operator_acceptance_required": True,
            "memory_write_allowed_by_consumer": False,
            "execution_allowed_by_consumer": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "policy_mutation": False,
            "memory_mutation": False,
            "memory_file_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
            "accepted_proposal_ids": [
                str(record["proposal_id"])
                for record in accepted_records
                if isinstance(record.get("proposal_id"), str) and record.get("proposal_id")
            ],
            "rejected_proposal_ids": list(rejected_proposal_ids),
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion-acceptance-ledger result consumer and execution-plan result."
                if not consumer or not plan
                else (
                    "Replace invalid sequence-memory trust promotion execution-plan evidence before executor dry-run."
                    if invalid_record_ids
                    else (
                        "Run read-only promotion execution planning before executor dry-run."
                        if plan.get("status")
                        != REQUIRED_PROMOTION_EXECUTION_PLAN_AVAILABLE_STATUS
                        else "Run the read-only executor dry-run request; this consumer does not create dry-runs or apply policy changes."
                    )
                )
            ),
        },
        "promotion_executor_dry_run_request": executor_dry_run_request,
    }
