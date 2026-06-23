"""Read-only consumer for sequence-policy queue-import execution-gate results.

This module verifies that a generic apply-sequence execution-gate payload
matches the sequence-policy queue-import dry-run consumer before a caller may
pass ready rows to the generic sequence executor. It does not invoke executor
handlers, reserve idempotency keys, mutate queue or proposal state, or touch
live mixer transport.
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
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_DRY_RUN_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_DRY_RUN_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_DRY_RUN_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_DRY_RUN_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_GATE_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_GATE_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_execution_gate_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_execution_gate_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_execution_gate_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_executor_request_available",
}
REQUIRED_DRY_RUN_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer"
)
REQUIRED_DRY_RUN_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_execution_gate_request_available"
)
REQUIRED_EXECUTION_GATE_TYPE = "operator_proposal_apply_sequence_execution_gate"
REQUIRED_CONSUMER_FALSE_FLAGS = (
    "execution_gate_handler_invoked",
    "execution_gate_created",
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
REQUIRED_GATE_FALSE_FLAGS = (
    "execution_allowed_by_gate",
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "mutates_console_state",
    "queries_console",
    "live_widening_allowed",
)
REQUIRED_GATE_REQUEST_SPEC_FALSE_FLAGS = (
    "execution_gate_invoked",
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
REQUIRED_GATE_REQUEST_SPEC_IDENTITY_FIELDS = (
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


def _candidate_id(candidate: Mapping[str, Any], index: int) -> str:
    value = _string(candidate.get("proposal_id"))
    return value if value else f"ready_for_executor:{index}"


def _blocked_id(blocked: Mapping[str, Any], index: int) -> str:
    value = _string(blocked.get("proposal_id"))
    return value if value else f"blocked_gate_result:{index}"


def _consumer_envelope_violations(consumer: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_DRY_RUN_RESULT_CONSUMER_TYPE:
        violations.append("sequence_policy_execution_gate_result_consumer_dry_run_consumer_type_mismatch")
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_DRY_RUN_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append(
            "sequence_policy_execution_gate_result_consumer_dry_run_consumer_method_id_mismatch"
        )
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_DRY_RUN_RESULT_CONSUMER_STATUSES
    ):
        violations.append(
            "sequence_policy_execution_gate_result_consumer_dry_run_consumer_status_unknown"
        )
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append(
            "sequence_policy_execution_gate_result_consumer_dry_run_consumer_read_only_not_true"
        )
    if safety.get("dry_run_result_consumer_only") is not True:
        violations.append(
            "sequence_policy_execution_gate_result_consumer_dry_run_consumer_only_not_true"
        )
    if safety.get("execution_gate_request_template_only") is not True:
        violations.append(
            "sequence_policy_execution_gate_result_consumer_dry_run_consumer_template_only_not_true"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_execution_gate_result_consumer_dry_run_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if summary.get("execution_gate_request_available") is not bool(expected_ids):
        violations.append(
            "sequence_policy_execution_gate_result_consumer_dry_run_consumer_request_available_mismatch"
        )
    expected_counts = {
        "expected_dry_run_result_count": len(expected_ids),
        "dry_run_result_count": len(expected_ids),
        "would_apply_result_count": len(expected_ids),
        "blocked_dry_run_result_count": 0,
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_execution_gate_result_consumer_dry_run_consumer_{key}_mismatch"
            )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_execution_gate_result_consumer_dry_run_consumer_summary_{flag}_not_false"
            )
    if _string_list(summary.get("dry_run_result_proposal_ids")) != expected_ids:
        violations.append(
            "sequence_policy_execution_gate_result_consumer_dry_run_consumer_result_ids_mismatch"
        )
    if _string_list(summary.get("would_apply_proposal_ids")) != expected_ids:
        violations.append(
            "sequence_policy_execution_gate_result_consumer_dry_run_consumer_would_apply_ids_mismatch"
        )
    return tuple(violations)


def _gate_envelope_violations(gate: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if gate.get("type") != REQUIRED_EXECUTION_GATE_TYPE:
        violations.append("sequence_policy_execution_gate_result_consumer_gate_type_mismatch")
    if gate.get("status") not in {"ready", "blocked"}:
        violations.append("sequence_policy_execution_gate_result_consumer_gate_status_unknown")
    if gate.get("status") == "ready" and gate.get("success") is not True:
        violations.append("sequence_policy_execution_gate_result_consumer_gate_success_not_true")
    if gate.get("status") != "ready" and gate.get("success") is not False:
        violations.append("sequence_policy_execution_gate_result_consumer_gate_blocked_success_not_false")

    safety = _mapping(gate.get("safety"))
    required_true = (
        "execution_gate_only",
        "dry_run_results_required",
        "operator_ack_required",
        "current_queue_proposal_required",
        "does_not_call_apply_operator_proposal",
        "requires_separate_executor",
    )
    for flag in required_true:
        if safety.get(flag) is not True:
            violations.append(f"sequence_policy_execution_gate_result_consumer_gate_{flag}_not_true")
    if safety.get("live_apply_entrypoint") != "apply_operator_proposal":
        violations.append("sequence_policy_execution_gate_result_consumer_gate_entrypoint_mismatch")
    allowed_kinds = _string_list(safety.get("allowed_live_write_kinds"))
    if tuple(sorted(allowed_kinds)) != tuple(sorted(CURRENT_PROVEN_LIVE_WRITE_KINDS)):
        violations.append("sequence_policy_execution_gate_result_consumer_gate_allowed_kinds_mismatch")
    for flag in REQUIRED_GATE_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"sequence_policy_execution_gate_result_consumer_gate_{flag}_not_false")
    return tuple(violations)


def _gate_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
    candidates: Tuple[Mapping[str, Any], ...],
    blocked: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "input_dry_run_result_count": len(expected_ids),
        "ready_for_executor_count": len(candidates),
        "blocked_count": len(blocked),
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"sequence_policy_execution_gate_result_consumer_gate_summary_{key}_mismatch")
    if summary.get("operator_ack_received") is not True:
        violations.append("sequence_policy_execution_gate_result_consumer_gate_summary_operator_ack_not_true")
    if summary.get("requires_separate_executor") is not True:
        violations.append("sequence_policy_execution_gate_result_consumer_gate_summary_separate_executor_not_true")
    if summary.get("queue_freshness_checked") is not True:
        violations.append("sequence_policy_execution_gate_result_consumer_gate_summary_queue_freshness_not_true")
    for flag in REQUIRED_GATE_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(f"sequence_policy_execution_gate_result_consumer_gate_summary_{flag}_not_false")
    candidate_ids = tuple(_candidate_id(candidate, index) for index, candidate in enumerate(candidates))
    if _string_list(summary.get("ready_for_executor_proposal_ids")) != candidate_ids:
        violations.append("sequence_policy_execution_gate_result_consumer_gate_summary_ready_ids_mismatch")
    blocked_ids = tuple(_blocked_id(row, index) for index, row in enumerate(blocked))
    if _string_list(summary.get("blocked_proposal_ids")) != blocked_ids:
        violations.append("sequence_policy_execution_gate_result_consumer_gate_summary_blocked_ids_mismatch")
    return tuple(violations)


def _gate_candidate_violations(
    candidate: Mapping[str, Any],
    *,
    index: int,
    prefix: str,
) -> Tuple[str, ...]:
    proposal_id = _candidate_id(candidate, index)
    violations = []
    if candidate.get("planned_apply_index") != index:
        violations.append(f"{proposal_id}:planned_apply_index_mismatch")
    if candidate.get("dry_run_result_index") != index:
        violations.append(f"{proposal_id}:dry_run_result_index_mismatch")
    if candidate.get("status") != "ready_for_executor":
        violations.append(f"{proposal_id}:gate_status_not_ready_for_executor")
    if candidate.get("execution_allowed_by_gate") is not False:
        violations.append(f"{proposal_id}:execution_authority_forged")
    if candidate.get("requires_separate_executor") is not True:
        violations.append(f"{proposal_id}:separate_executor_requirement_missing")
    apply_request = _mapping(candidate.get("apply_request"))
    if apply_request.get("type") != "apply_operator_proposal":
        violations.append(f"{proposal_id}:apply_request_type_mismatch")
    if apply_request.get("proposal_id") != proposal_id:
        violations.append(f"{proposal_id}:apply_request_proposal_id_mismatch")
    if apply_request.get("approved") is not True:
        violations.append(f"{proposal_id}:apply_request_approved_not_true")
    idempotency_key = apply_request.get("idempotency_key")
    if idempotency_key != f"{prefix}:{proposal_id}":
        violations.append(f"{proposal_id}:apply_request_idempotency_key_mismatch")
    current = _mapping(candidate.get("current_proposal"))
    if current.get("id") != proposal_id:
        violations.append(f"{proposal_id}:current_proposal_id_mismatch")
    if current.get("status") != "pending":
        violations.append(f"{proposal_id}:current_proposal_not_pending")
    if current.get("can_apply") is not True:
        violations.append(f"{proposal_id}:current_proposal_not_applyable")
    changes = _record_list(candidate.get("normalized_changes"))
    if not changes:
        violations.append(f"{proposal_id}:normalized_changes_missing")
    for change_index, change in enumerate(changes):
        if change.get("change_index") != change_index:
            violations.append(f"{proposal_id}:change_index_mismatch")
        value_type = _string(change.get("value_type"))
        if value_type not in ALLOWED_VALUE_TYPES:
            violations.append(f"{proposal_id}:change_value_type_not_executor_safe")
        effective_value = change.get("effective_value")
        if not isinstance(effective_value, (int, float)) or isinstance(effective_value, bool):
            violations.append(f"{proposal_id}:change_effective_value_invalid")
        elif value_type == "fader" and float(effective_value) > 0.0:
            violations.append(f"{proposal_id}:change_fader_ceiling_exceeded")
        if change.get("rollback_expected") is not True:
            violations.append(f"{proposal_id}:rollback_expected_not_true")
    return tuple(violations)


def _gate_request_specs_violations(
    specs: Tuple[Mapping[str, Any], ...],
    *,
    expected_ids: Tuple[str, ...],
    prefix: str,
) -> Tuple[str, ...]:
    violations = []
    spec_ids = tuple(_string(spec.get("proposal_id")) for spec in specs)
    if spec_ids != expected_ids:
        violations.append("sequence_policy_execution_gate_result_consumer_request_specs_candidate_ids_mismatch")
    for index, spec in enumerate(specs):
        proposal_id = _string(spec.get("proposal_id")) or f"execution_gate_request_spec:{index}"
        record_id = _string(spec.get("execution_gate_request_spec_id")) or proposal_id
        if spec.get("planned_gate_index") != index:
            violations.append(f"{record_id}:planned_gate_index_mismatch")
        if spec.get("source_dry_run_index") != index:
            violations.append(f"{record_id}:source_dry_run_index_mismatch")
        if spec.get("idempotency_key") != f"{prefix}:{proposal_id}":
            violations.append(f"{record_id}:idempotency_key_mismatch")
        if spec.get("execution_gate_handler") != "gate_operator_proposal_apply_sequence_execution":
            violations.append(f"{record_id}:execution_gate_handler_mismatch")
        if spec.get("read_only") is not True:
            violations.append(f"{record_id}:read_only_not_true")
        if spec.get("dry_run_result_consumer_only") is not True:
            violations.append(f"{record_id}:dry_run_result_consumer_only_not_true")
        if spec.get("operator_ack_required_before_gate_ready") is not True:
            violations.append(f"{record_id}:operator_ack_required_before_gate_ready_not_true")
        if _summary_int(spec.get("planned_queue_index")) is None:
            violations.append(f"{proposal_id}:planned_queue_index_invalid")
        if spec.get("rank_source") != "sequence_policy_consumer":
            violations.append(f"{proposal_id}:rank_source_mismatch")
        for field in REQUIRED_GATE_REQUEST_SPEC_IDENTITY_FIELDS:
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
        for flag in REQUIRED_GATE_REQUEST_SPEC_FALSE_FLAGS:
            if spec.get(flag) is not False:
                violations.append(f"{record_id}:{flag}_not_false")
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer(
    *,
    dry_run_result_consumer: Mapping[str, Any] | None = None,
    execution_gate_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify execution-gate evidence and emit a sequence-executor request template."""

    consumer = _mapping(dry_run_result_consumer)
    gate = _mapping(execution_gate_result)
    consumer_summary = _mapping(consumer.get("summary"))
    expected_ids = _string_list(consumer_summary.get("dry_run_result_proposal_ids"))
    prefix = _string(consumer_summary.get("idempotency_key_prefix"))
    gate_request_specs = _record_list(consumer.get("execution_gate_request_specs"))
    gate_request_specs_by_proposal_id = {
        _string(spec.get("proposal_id")): spec
        for spec in gate_request_specs
        if _string(spec.get("proposal_id"))
    }
    candidates = _record_list(gate.get("ready_for_executor"))
    blocked_gate_rows = _record_list(gate.get("blocked_results"))
    candidate_ids = tuple(_candidate_id(candidate, index) for index, candidate in enumerate(candidates))
    candidate_idempotency_keys = tuple(
        _string(_mapping(candidate.get("apply_request")).get("idempotency_key"))
        for candidate in candidates
    )
    candidate_violations = tuple(
        violation
        for index, candidate in enumerate(candidates)
        for violation in _gate_candidate_violations(candidate, index=index, prefix=prefix)
    )
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_consumer_envelope_violations(consumer),
                *_consumer_summary_violations(consumer_summary, expected_ids=expected_ids),
                *_gate_request_specs_violations(
                    gate_request_specs,
                    expected_ids=expected_ids,
                    prefix=prefix,
                ),
                *(
                    ("sequence_policy_execution_gate_result_consumer_dry_run_consumer_status_not_available",)
                    if consumer.get("status") != REQUIRED_DRY_RUN_RESULT_CONSUMER_AVAILABLE_STATUS
                    else ()
                ),
                *_gate_envelope_violations(gate),
                *_gate_summary_violations(
                    _mapping(gate.get("summary")),
                    expected_ids=expected_ids,
                    candidates=candidates,
                    blocked=blocked_gate_rows,
                ),
                *(
                    f"duplicate_ready_for_executor_id:{value}"
                    for value in _duplicate_values(candidate_ids)
                ),
                *(
                    f"duplicate_ready_for_executor_idempotency_key:{value}"
                    for value in _duplicate_values(candidate_idempotency_keys)
                ),
                *(
                    ("sequence_policy_execution_gate_result_consumer_candidate_id_outside_expected_set",)
                    if any(proposal_id not in set(expected_ids) for proposal_id in candidate_ids)
                    else ()
                ),
                *(
                    ("sequence_policy_execution_gate_result_consumer_candidate_ids_do_not_cover_expected_set",)
                    if not blocked_gate_rows and set(candidate_ids) != set(expected_ids)
                    else ()
                ),
                *(
                    ("sequence_policy_execution_gate_result_consumer_candidate_order_mismatch",)
                    if not blocked_gate_rows and candidate_ids != expected_ids
                    else ()
                ),
                *candidate_violations,
            )
        )
    )

    if not consumer or not gate:
        status = "sequence_memory_trust_promotion_queue_import_execution_gate_result_consumer_not_ready"
        blocked_by = ("dry_run_result_consumer_and_execution_gate_result_required",)
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_execution_gate_result_consumer_replacement_required"
        blocked_by = ("operator_must_replace_invalid_sequence_policy_execution_gate_evidence",)
    elif gate.get("status") != "ready" or blocked_gate_rows:
        status = "sequence_memory_trust_promotion_queue_import_execution_gate_result_consumer_blocked"
        blocked_by = ("all_sequence_policy_imported_execution_gate_rows_must_be_ready_for_executor",)
    elif not candidates:
        status = "sequence_memory_trust_promotion_queue_import_execution_gate_result_consumer_not_ready"
        blocked_by = ("ready_for_executor_rows_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_executor_request_available"
        blocked_by = ("operator_may_run_generic_sequence_executor_with_execute_ack_and_approval",)

    request_available = (
        status == "sequence_memory_trust_promotion_queue_import_executor_request_available"
    )
    selected_candidates = candidates if request_available else ()
    selected_gate_request_specs = tuple(
        gate_request_specs_by_proposal_id.get(proposal_id, {})
        for proposal_id in candidate_ids
        if request_available
    )
    executor_specs = tuple(
        {
            "executor_request_spec_id": (
                f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_GATE_RESULT_CONSUMER_METHOD_ID}:{proposal_id}"
            ),
            "proposal_id": proposal_id,
            "planned_executor_index": index,
            "source_gate_candidate_index": index,
            "source_execution_gate_request_spec_id": _string(
                gate_request_specs_by_proposal_id.get(proposal_id, {}).get(
                    "execution_gate_request_spec_id"
                )
            ),
            "source_dry_run_request_spec_id": _string(
                gate_request_specs_by_proposal_id.get(proposal_id, {}).get(
                    "source_dry_run_request_spec_id"
                )
            ),
            "source_sequence_plan_request_spec_id": _string(
                gate_request_specs_by_proposal_id.get(proposal_id, {}).get(
                    "source_sequence_plan_request_spec_id"
                )
            ),
            "source_preflight_request_spec_id": _string(
                gate_request_specs_by_proposal_id.get(proposal_id, {}).get(
                    "source_preflight_request_spec_id"
                )
            ),
            "queue_ordering_plan_record_id": gate_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("queue_ordering_plan_record_id"),
            "consumer_record_id": gate_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("consumer_record_id"),
            "sequence_policy_gate_record_id": gate_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("sequence_policy_gate_record_id"),
            "policy_key": gate_request_specs_by_proposal_id.get(proposal_id, {}).get(
                "policy_key"
            ),
            "source_idempotency_key": gate_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("source_idempotency_key"),
            "current_sequence_review_freshness": gate_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("current_sequence_review_freshness"),
            "planned_queue_index": gate_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("planned_queue_index"),
            "rank_source": gate_request_specs_by_proposal_id.get(proposal_id, {}).get(
                "rank_source"
            ),
            "idempotency_key": f"{prefix}:{proposal_id}",
            "read_only": True,
            "execution_gate_result_consumer_only": True,
            "sequence_executor_handler": "execute_operator_proposal_apply_sequence",
            "execute_true_required_before_executor": True,
            "operator_ack_required_before_executor": True,
            "approved_true_required_before_executor": True,
            "sequence_executor_invoked": False,
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
        for index, proposal_id in enumerate(candidate_ids if request_available else ())
    )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_GATE_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_DRY_RUN_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "execution_gate_result_consumer_only": True,
            "executor_request_template_only": True,
            "requires_dry_run_result_consumer": True,
            "requires_execution_gate_result": True,
            "requires_all_gate_rows_ready_for_executor": True,
            "execute_true_required_before_executor": True,
            "operator_ack_required_before_executor": True,
            "approved_true_required_before_executor": True,
            "sequence_executor_handler_invoked": False,
            "sequence_executor_created": False,
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
            "expected_gate_candidate_count": len(expected_ids),
            "execution_gate_request_spec_count": len(gate_request_specs),
            "ready_for_executor_count": len(candidates),
            "blocked_gate_result_count": len(blocked_gate_rows),
            "invalid_record_count": len(invalid_record_ids),
            "idempotency_key_prefix": prefix,
            "execution_gate_result_consumer_only": True,
            "executor_request_available": request_available,
            "sequence_executor_handler_invoked": False,
            "sequence_executor_created": False,
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
            "expected_gate_proposal_ids": list(expected_ids),
            "execution_gate_request_spec_proposal_ids": [
                spec.get("proposal_id") for spec in gate_request_specs
            ],
            "ready_for_executor_proposal_ids": list(candidate_ids),
            "blocked_gate_proposal_ids": [
                _blocked_id(row, index) for index, row in enumerate(blocked_gate_rows)
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide dry-run result consumer plus execution-gate result."
                if not consumer or not gate
                else (
                    "Replace invalid execution-gate evidence before sequence executor."
                    if invalid_record_ids
                    else (
                        "Re-run execution gate until every sequence-policy imported row is ready for executor."
                        if gate.get("status") != "ready" or blocked_gate_rows
                        else "Operator may run the generic sequence executor with execute=true, operator_ack=true, and approved=true; this consumer does not run it."
                    )
                )
            ),
        },
        "executor_request": {
            "handler": "execute_operator_proposal_apply_sequence",
            "execution_gate_payload": {
                **dict(gate),
                "ready_for_executor": list(selected_candidates),
                "execution_gate_request_specs": list(selected_gate_request_specs),
            }
            if request_available
            else {},
            "execute": False,
            "operator_ack": False,
            "approved": False,
            "operator_must_set_execute_true": True,
            "operator_must_set_operator_ack_true": True,
            "operator_must_set_approved_true": True,
            "read_only_template": True,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "sequence_executor_created": False,
            "execution_allowed_by_consumer": False,
        },
        "executor_request_specs": list(executor_specs),
    }
