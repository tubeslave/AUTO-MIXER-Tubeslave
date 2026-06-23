"""Read-only consumer for sequence-policy queue-import execution results.

This module verifies that a generic apply-sequence executor result still
matches sequence-policy queue-import provenance before a caller may pass it to
the generic execution-audit handler. It does not invoke audit or apply handlers,
reserve idempotency keys, mutate queue or proposal state, or touch live mixer
transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_mode_policy import CURRENT_PROVEN_LIVE_WRITE_KINDS
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_mode_policy import CURRENT_PROVEN_LIVE_WRITE_KINDS

ALLOWED_VALUE_TYPES = set(CURRENT_PROVEN_LIVE_WRITE_KINDS)

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_GATE_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_GATE_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_GATE_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_GATE_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_execution_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_execution_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_execution_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_audit_request_available",
}
REQUIRED_EXECUTION_GATE_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer"
)
REQUIRED_EXECUTION_GATE_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_executor_request_available"
)
REQUIRED_EXECUTION_RESULT_TYPE = "operator_proposal_apply_sequence_execution_result"
VERIFIED_READBACK_STATUSES = {
    "verified_from_cache",
    "verified_from_physical_readback",
}
REQUIRED_CONSUMER_FALSE_FLAGS = (
    "sequence_executor_handler_invoked",
    "sequence_executor_created",
    "execution_allowed_by_consumer",
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_apply_live",
)
REQUIRED_EXECUTION_SUMMARY_TRUE_FLAGS = (
    "requires_gate_ready",
    "uses_existing_apply_operator_proposal",
    "sequence_executor_called_apply_operator_proposal",
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "idempotency_reserved",
    "idempotency_reservation_delegated_to_apply_operator_proposal",
    "manual_console_write_called_by_apply_operator_proposal",
    "mutates_console_state",
    "queue_freshness_checked",
)
REQUIRED_EXECUTION_SUMMARY_FALSE_FLAGS = (
    "direct_live_write",
    "idempotency_reserved_by_executor_directly",
    "manual_console_write_called_by_sequence_executor",
    "direct_manual_console_write_called",
    "queries_console",
    "live_widening_allowed",
)
REQUIRED_EXECUTION_SAFETY_TRUE_FLAGS = (
    "sequence_executor",
    "execution_gate_payload_required",
    "ready_for_executor_required",
    "requires_execute_true",
    "requires_operator_ack",
    "requires_approved_true",
    "requires_gate_ready",
    "uses_existing_apply_operator_proposal",
    "live_write",
    "apply_attempted",
    "idempotency_reservation_delegated_to_apply_operator_proposal",
    "does_not_call_manual_console_write_directly",
    "manual_console_write_called_by_apply_operator_proposal",
    "mutates_console_state",
)
REQUIRED_EXECUTION_SAFETY_FALSE_FLAGS = (
    "direct_live_write",
    "queue_mutation_by_executor_directly",
    "proposal_status_mutation_by_executor_directly",
    "idempotency_reserved_by_executor_directly",
    "manual_console_write_called_by_sequence_executor",
    "direct_manual_console_write_called",
    "queries_console",
    "live_widening_allowed",
)
REQUIRED_EXECUTOR_REQUEST_SPEC_FALSE_FLAGS = (
    "sequence_executor_invoked",
    "execution_allowed_by_consumer",
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_apply_live",
)
REQUIRED_EXECUTOR_REQUEST_SPEC_IDENTITY_FIELDS = (
    "source_execution_gate_request_spec_id",
    "source_dry_run_request_spec_id",
    "source_sequence_plan_request_spec_id",
    "source_preflight_request_spec_id",
    "queue_ordering_plan_record_id",
    "consumer_record_id",
    "sequence_policy_gate_record_id",
    "policy_key",
    "source_idempotency_key",
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


def _context_status(payload: Mapping[str, Any]) -> str:
    return _string(payload.get("status"))


def _context_violations(proposal_id: str, payload: Mapping[str, Any]) -> Tuple[str, ...]:
    if not payload:
        return (f"{proposal_id}:current_sequence_review_freshness_missing",)
    violations = []
    if _context_status(payload) != "fresh":
        violations.append(f"{proposal_id}:current_sequence_review_freshness_not_fresh")
    if payload.get("read_only") is not True:
        violations.append(
            f"{proposal_id}:current_sequence_review_freshness_read_only_not_true"
        )
    for flag in (
        "live_write",
        "apply_attempted",
        "queries_console",
        "queue_mutation",
        "policy_mutation",
    ):
        if payload.get(flag) is not False:
            violations.append(
                f"{proposal_id}:current_sequence_review_freshness_{flag}_not_false"
            )
    return tuple(violations)


def _duplicate_values(values: Tuple[str, ...]) -> Tuple[str, ...]:
    counts: Dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return tuple(value for value, count in counts.items() if count > 1)


def _execution_row_id(row: Mapping[str, Any], index: int) -> str:
    value = _string(row.get("proposal_id"))
    return value if value else f"execution_result:{index}"


def _consumer_envelope_violations(consumer: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_EXECUTION_GATE_RESULT_CONSUMER_TYPE:
        violations.append("sequence_policy_execution_result_consumer_gate_consumer_type_mismatch")
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_GATE_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append(
            "sequence_policy_execution_result_consumer_gate_consumer_method_id_mismatch"
        )
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_GATE_RESULT_CONSUMER_STATUSES
    ):
        violations.append(
            "sequence_policy_execution_result_consumer_gate_consumer_status_unknown"
        )
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_policy_execution_result_consumer_gate_consumer_read_only_not_true")
    if safety.get("execution_gate_result_consumer_only") is not True:
        violations.append("sequence_policy_execution_result_consumer_gate_consumer_only_not_true")
    if safety.get("executor_request_template_only") is not True:
        violations.append("sequence_policy_execution_result_consumer_gate_consumer_template_only_not_true")
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_execution_result_consumer_gate_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if summary.get("executor_request_available") is not bool(expected_ids):
        violations.append(
            "sequence_policy_execution_result_consumer_gate_consumer_request_available_mismatch"
        )
    expected_counts = {
        "expected_gate_candidate_count": len(expected_ids),
        "ready_for_executor_count": len(expected_ids),
        "blocked_gate_result_count": 0,
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_execution_result_consumer_gate_consumer_{key}_mismatch"
            )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_execution_result_consumer_gate_consumer_summary_{flag}_not_false"
            )
    if _string_list(summary.get("ready_for_executor_proposal_ids")) != expected_ids:
        violations.append(
            "sequence_policy_execution_result_consumer_gate_consumer_ready_ids_mismatch"
        )
    return tuple(violations)


def _execution_envelope_violations(execution: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if execution.get("type") != REQUIRED_EXECUTION_RESULT_TYPE:
        violations.append("sequence_policy_execution_result_consumer_execution_type_mismatch")
    if execution.get("status") not in {"applied", "partial", "blocked"}:
        violations.append("sequence_policy_execution_result_consumer_execution_status_unknown")
    if execution.get("status") == "applied" and execution.get("success") is not True:
        violations.append("sequence_policy_execution_result_consumer_execution_success_not_true")
    if execution.get("status") != "applied" and execution.get("success") is not False:
        violations.append("sequence_policy_execution_result_consumer_execution_blocked_success_not_false")

    summary = _mapping(execution.get("summary"))
    if summary.get("live_apply_entrypoint") != "apply_operator_proposal":
        violations.append("sequence_policy_execution_result_consumer_execution_summary_entrypoint_mismatch")
    for flag in REQUIRED_EXECUTION_SUMMARY_TRUE_FLAGS:
        if summary.get(flag) is not True:
            violations.append(f"sequence_policy_execution_result_consumer_execution_summary_{flag}_not_true")
    for flag in REQUIRED_EXECUTION_SUMMARY_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(f"sequence_policy_execution_result_consumer_execution_summary_{flag}_not_false")

    safety = _mapping(execution.get("safety"))
    if safety.get("live_apply_entrypoint") != "apply_operator_proposal":
        violations.append("sequence_policy_execution_result_consumer_execution_safety_entrypoint_mismatch")
    allowed_kinds = _string_list(safety.get("allowed_live_write_kinds"))
    if tuple(sorted(allowed_kinds)) != tuple(sorted(CURRENT_PROVEN_LIVE_WRITE_KINDS)):
        violations.append("sequence_policy_execution_result_consumer_execution_safety_allowed_kinds_mismatch")
    for flag in REQUIRED_EXECUTION_SAFETY_TRUE_FLAGS:
        if safety.get(flag) is not True:
            violations.append(f"sequence_policy_execution_result_consumer_execution_safety_{flag}_not_true")
    for flag in REQUIRED_EXECUTION_SAFETY_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"sequence_policy_execution_result_consumer_execution_safety_{flag}_not_false")
    return tuple(violations)


def _execution_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
    rows: Tuple[Mapping[str, Any], ...],
    blocked_rows: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "input_candidate_count": len(expected_ids),
        "attempted_count": len(rows),
        "applied_count": len(rows) - len(blocked_rows),
        "blocked_count": len(blocked_rows),
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"sequence_policy_execution_result_consumer_execution_summary_{key}_mismatch")
    if summary.get("execute_requested") is not True:
        violations.append("sequence_policy_execution_result_consumer_execution_summary_execute_not_true")
    if summary.get("operator_ack_received") is not True:
        violations.append("sequence_policy_execution_result_consumer_execution_summary_operator_ack_not_true")
    if summary.get("approved") is not True:
        violations.append("sequence_policy_execution_result_consumer_execution_summary_approved_not_true")
    applied_ids = tuple(
        _execution_row_id(row, index)
        for index, row in enumerate(rows)
        if row.get("status") == "applied"
    )
    if _string_list(summary.get("applied_proposal_ids")) != applied_ids:
        violations.append("sequence_policy_execution_result_consumer_execution_summary_applied_ids_mismatch")
    blocked_ids = tuple(
        _execution_row_id(row, index)
        for index, row in enumerate(blocked_rows)
    )
    if _string_list(summary.get("blocked_proposal_ids")) != blocked_ids:
        violations.append("sequence_policy_execution_result_consumer_execution_summary_blocked_ids_mismatch")
    return tuple(violations)


def _execution_row_violations(
    row: Mapping[str, Any],
    *,
    index: int,
    prefix: str,
) -> Tuple[str, ...]:
    proposal_id = _execution_row_id(row, index)
    violations = []
    if row.get("planned_apply_index") != index:
        violations.append(f"{proposal_id}:planned_apply_index_mismatch")
    if row.get("candidate_index") != index:
        violations.append(f"{proposal_id}:candidate_index_mismatch")
    if row.get("status") != "applied":
        violations.append(f"{proposal_id}:execution_status_not_applied")
    if row.get("apply_attempted") is not True:
        violations.append(f"{proposal_id}:apply_attempted_not_true")
    apply_request = _mapping(row.get("apply_request"))
    if apply_request.get("type") != "apply_operator_proposal":
        violations.append(f"{proposal_id}:apply_request_type_mismatch")
    if apply_request.get("proposal_id") != proposal_id:
        violations.append(f"{proposal_id}:apply_request_proposal_id_mismatch")
    if apply_request.get("approved") is not True:
        violations.append(f"{proposal_id}:apply_request_approved_not_true")
    idempotency_key = apply_request.get("idempotency_key")
    if idempotency_key != f"{prefix}:{proposal_id}":
        violations.append(f"{proposal_id}:apply_request_idempotency_key_mismatch")

    proposal = _mapping(row.get("proposal"))
    if proposal and proposal.get("id") != proposal_id:
        violations.append(f"{proposal_id}:proposal_snapshot_id_mismatch")
    if proposal and proposal.get("status") != "applied":
        violations.append(f"{proposal_id}:proposal_snapshot_not_applied")

    apply_result = _mapping(row.get("apply_result"))
    if apply_result.get("status") != "applied":
        violations.append(f"{proposal_id}:apply_result_not_applied")
    value_type = _string(apply_result.get("value_type"))
    if value_type not in ALLOWED_VALUE_TYPES:
        violations.append(f"{proposal_id}:apply_result_value_type_not_audit_safe")
    effective_value = apply_result.get("effective_value")
    if not isinstance(effective_value, (int, float)) or isinstance(effective_value, bool):
        violations.append(f"{proposal_id}:apply_result_effective_value_invalid")
    elif value_type == "fader" and float(effective_value) > 0.0:
        violations.append(f"{proposal_id}:apply_result_fader_ceiling_exceeded")
    if apply_result.get("readback_status") not in VERIFIED_READBACK_STATUSES:
        violations.append(f"{proposal_id}:readback_not_verified")
    if apply_result.get("proof_status", "verified") != "verified":
        violations.append(f"{proposal_id}:proof_status_not_verified")
    if apply_result.get("rollback_available") is not True:
        violations.append(f"{proposal_id}:rollback_unavailable")
    return tuple(violations)


def _executor_request_specs_violations(
    specs: Tuple[Mapping[str, Any], ...],
    *,
    expected_ids: Tuple[str, ...],
    prefix: str,
) -> Tuple[str, ...]:
    violations = []
    spec_ids = tuple(_string(spec.get("proposal_id")) for spec in specs)
    if spec_ids != expected_ids:
        violations.append("sequence_policy_execution_result_consumer_request_specs_candidate_ids_mismatch")
    for index, spec in enumerate(specs):
        proposal_id = _string(spec.get("proposal_id")) or f"executor_request_spec:{index}"
        record_id = _string(spec.get("executor_request_spec_id")) or proposal_id
        if spec.get("planned_executor_index") != index:
            violations.append(f"{record_id}:planned_executor_index_mismatch")
        if spec.get("source_gate_candidate_index") != index:
            violations.append(f"{record_id}:source_gate_candidate_index_mismatch")
        if spec.get("idempotency_key") != f"{prefix}:{proposal_id}":
            violations.append(f"{record_id}:idempotency_key_mismatch")
        if spec.get("sequence_executor_handler") != "execute_operator_proposal_apply_sequence":
            violations.append(f"{record_id}:sequence_executor_handler_mismatch")
        if spec.get("read_only") is not True:
            violations.append(f"{record_id}:read_only_not_true")
        if spec.get("execution_gate_result_consumer_only") is not True:
            violations.append(f"{record_id}:execution_gate_result_consumer_only_not_true")
        if spec.get("execute_true_required_before_executor") is not True:
            violations.append(f"{record_id}:execute_true_required_before_executor_not_true")
        if spec.get("operator_ack_required_before_executor") is not True:
            violations.append(f"{record_id}:operator_ack_required_before_executor_not_true")
        if spec.get("approved_true_required_before_executor") is not True:
            violations.append(f"{record_id}:approved_true_required_before_executor_not_true")
        if _summary_int(spec.get("planned_queue_index")) is None:
            violations.append(f"{proposal_id}:planned_queue_index_invalid")
        if spec.get("rank_source") != "sequence_policy_consumer":
            violations.append(f"{proposal_id}:rank_source_mismatch")
        for field in REQUIRED_EXECUTOR_REQUEST_SPEC_IDENTITY_FIELDS:
            if not _string(spec.get(field)):
                violations.append(f"{proposal_id}:{field}_missing")
        violations.extend(
            _context_violations(
                proposal_id,
                _mapping(spec.get("current_sequence_review_freshness")),
            )
        )
        source_idempotency_key = _string(spec.get("source_idempotency_key"))
        policy_key = _string(spec.get("policy_key"))
        if (
            source_idempotency_key
            and policy_key
            and policy_key != f"{proposal_id}/{source_idempotency_key}"
        ):
            violations.append(f"{proposal_id}:policy_key_identity_mismatch")
        for flag in REQUIRED_EXECUTOR_REQUEST_SPEC_FALSE_FLAGS:
            if spec.get(flag) is not False:
                violations.append(f"{record_id}:{flag}_not_false")
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer(
    *,
    execution_gate_result_consumer: Mapping[str, Any] | None = None,
    execution_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify sequence execution evidence and emit an audit request template."""

    consumer = _mapping(execution_gate_result_consumer)
    execution = _mapping(execution_result)
    consumer_summary = _mapping(consumer.get("summary"))
    expected_ids = _string_list(consumer_summary.get("ready_for_executor_proposal_ids"))
    prefix = _string(consumer_summary.get("idempotency_key_prefix"))
    executor_request_specs = _record_list(consumer.get("executor_request_specs"))
    executor_request_specs_by_proposal_id = {
        _string(spec.get("proposal_id")): spec
        for spec in executor_request_specs
        if _string(spec.get("proposal_id"))
    }
    rows = _record_list(execution.get("execution_results"))
    blocked_rows = tuple(row for row in rows if row.get("status") == "blocked")
    row_ids = tuple(_execution_row_id(row, index) for index, row in enumerate(rows))
    row_idempotency_keys = tuple(
        _string(_mapping(row.get("apply_request")).get("idempotency_key"))
        for row in rows
    )
    row_violations = tuple(
        violation
        for index, row in enumerate(rows)
        if row.get("status") == "applied"
        for violation in _execution_row_violations(row, index=index, prefix=prefix)
    )
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_consumer_envelope_violations(consumer),
                *_consumer_summary_violations(consumer_summary, expected_ids=expected_ids),
                *_executor_request_specs_violations(
                    executor_request_specs,
                    expected_ids=expected_ids,
                    prefix=prefix,
                ),
                *(
                    ("sequence_policy_execution_result_consumer_gate_consumer_status_not_available",)
                    if consumer.get("status") != REQUIRED_EXECUTION_GATE_RESULT_CONSUMER_AVAILABLE_STATUS
                    else ()
                ),
                *_execution_envelope_violations(execution),
                *_execution_summary_violations(
                    _mapping(execution.get("summary")),
                    expected_ids=expected_ids,
                    rows=rows,
                    blocked_rows=blocked_rows,
                ),
                *(
                    f"duplicate_execution_result_id:{value}"
                    for value in _duplicate_values(row_ids)
                ),
                *(
                    f"duplicate_execution_result_idempotency_key:{value}"
                    for value in _duplicate_values(row_idempotency_keys)
                ),
                *(
                    ("sequence_policy_execution_result_consumer_row_id_outside_expected_set",)
                    if any(proposal_id not in set(expected_ids) for proposal_id in row_ids)
                    else ()
                ),
                *(
                    ("sequence_policy_execution_result_consumer_row_ids_do_not_cover_expected_set",)
                    if not blocked_rows and set(row_ids) != set(expected_ids)
                    else ()
                ),
                *(
                    ("sequence_policy_execution_result_consumer_row_order_mismatch",)
                    if not blocked_rows and row_ids != expected_ids
                    else ()
                ),
                *row_violations,
            )
        )
    )

    if not consumer or not execution:
        status = "sequence_memory_trust_promotion_queue_import_execution_result_consumer_not_ready"
        blocked_by = ("execution_gate_result_consumer_and_execution_result_required",)
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_execution_result_consumer_replacement_required"
        blocked_by = ("operator_must_replace_invalid_sequence_policy_execution_result_evidence",)
    elif execution.get("status") != "applied" or blocked_rows:
        status = "sequence_memory_trust_promotion_queue_import_execution_result_consumer_blocked"
        blocked_by = ("all_sequence_policy_imported_execution_results_must_be_applied",)
    elif not rows:
        status = "sequence_memory_trust_promotion_queue_import_execution_result_consumer_not_ready"
        blocked_by = ("applied_execution_results_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_audit_request_available"
        blocked_by = ("operator_may_run_generic_execution_audit",)

    request_available = (
        status == "sequence_memory_trust_promotion_queue_import_audit_request_available"
    )
    selected_rows = rows if request_available else ()
    selected_executor_request_specs = tuple(
        executor_request_specs_by_proposal_id.get(proposal_id, {})
        for proposal_id in row_ids
        if request_available
    )
    audit_specs = tuple(
        {
            "audit_request_spec_id": (
                f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_RESULT_CONSUMER_METHOD_ID}:{proposal_id}"
            ),
            "proposal_id": proposal_id,
            "planned_audit_index": index,
            "source_execution_result_index": index,
            "source_executor_request_spec_id": _string(
                executor_request_specs_by_proposal_id.get(proposal_id, {}).get(
                    "executor_request_spec_id"
                )
            ),
            "source_execution_gate_request_spec_id": _string(
                executor_request_specs_by_proposal_id.get(proposal_id, {}).get(
                    "source_execution_gate_request_spec_id"
                )
            ),
            "source_dry_run_request_spec_id": _string(
                executor_request_specs_by_proposal_id.get(proposal_id, {}).get(
                    "source_dry_run_request_spec_id"
                )
            ),
            "source_sequence_plan_request_spec_id": _string(
                executor_request_specs_by_proposal_id.get(proposal_id, {}).get(
                    "source_sequence_plan_request_spec_id"
                )
            ),
            "source_preflight_request_spec_id": _string(
                executor_request_specs_by_proposal_id.get(proposal_id, {}).get(
                    "source_preflight_request_spec_id"
                )
            ),
            "queue_ordering_plan_record_id": executor_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("queue_ordering_plan_record_id"),
            "consumer_record_id": executor_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("consumer_record_id"),
            "sequence_policy_gate_record_id": executor_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("sequence_policy_gate_record_id"),
            "policy_key": executor_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("policy_key"),
            "source_idempotency_key": executor_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("source_idempotency_key"),
            "current_sequence_review_freshness": executor_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("current_sequence_review_freshness"),
            "planned_queue_index": executor_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("planned_queue_index"),
            "rank_source": executor_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("rank_source"),
            "idempotency_key": f"{prefix}:{proposal_id}",
            "read_only": True,
            "execution_result_consumer_only": True,
            "audit_handler": "audit_operator_proposal_apply_sequence_execution",
            "audit_handler_invoked": False,
            "audit_created": False,
            "execution_allowed_by_consumer": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
        }
        for index, proposal_id in enumerate(row_ids if request_available else ())
    )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_GATE_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "execution_result_consumer_only": True,
            "audit_request_template_only": True,
            "requires_execution_gate_result_consumer": True,
            "requires_execution_result": True,
            "requires_all_execution_results_applied": True,
            "requires_apply_operator_proposal_delegation": True,
            "audit_handler_invoked": False,
            "audit_created": False,
            "execution_allowed_by_consumer": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
        },
        "summary": {
            "expected_execution_result_count": len(expected_ids),
            "executor_request_spec_count": len(executor_request_specs),
            "execution_result_count": len(rows),
            "applied_execution_result_count": len(rows) - len(blocked_rows),
            "blocked_execution_result_count": len(blocked_rows),
            "invalid_record_count": len(invalid_record_ids),
            "idempotency_key_prefix": prefix,
            "execution_result_consumer_only": True,
            "audit_request_available": request_available,
            "audit_handler_invoked": False,
            "audit_created": False,
            "execution_allowed_by_consumer": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
            "expected_execution_proposal_ids": list(expected_ids),
            "executor_request_spec_proposal_ids": [
                spec.get("proposal_id") for spec in executor_request_specs
            ],
            "execution_result_proposal_ids": list(row_ids),
            "applied_execution_proposal_ids": [
                _execution_row_id(row, index)
                for index, row in enumerate(rows)
                if row.get("status") == "applied"
            ],
            "blocked_execution_proposal_ids": [
                _execution_row_id(row, index)
                for index, row in enumerate(blocked_rows)
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide execution-gate result consumer plus sequence execution result."
                if not consumer or not execution
                else (
                    "Replace invalid execution result evidence before audit."
                    if invalid_record_ids
                    else (
                        "Re-run sequence executor until every sequence-policy imported result is applied."
                        if execution.get("status") != "applied" or blocked_rows
                        else "Operator may run the generic execution audit; this consumer does not run it."
                    )
                )
            ),
        },
        "audit_request": {
            "handler": "audit_operator_proposal_apply_sequence_execution",
            "execution_result_payload": {
                **dict(execution),
                "execution_results": list(selected_rows),
                "executor_request_specs": list(selected_executor_request_specs),
            }
            if request_available
            else {},
            "read_only_template": True,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "audit_created": False,
            "execution_allowed_by_consumer": False,
        },
        "audit_request_specs": list(audit_specs),
    }
