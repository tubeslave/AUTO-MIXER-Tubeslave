"""Read-only consumer for sequence-policy queue-import dry-run results.

This module verifies that a generic apply-sequence dry-run payload matches the
sequence-policy queue-import sequence-plan consumer before a caller may pass the
dry-run results to the generic execution gate. It does not invoke gate or
executor handlers, reserve idempotency keys, mutate queue or proposal state, or
touch live mixer transport.
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
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_SEQUENCE_PLAN_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_SEQUENCE_PLAN_RESULT_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_SEQUENCE_PLAN_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_SEQUENCE_PLAN_RESULT_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_DRY_RUN_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_DRY_RUN_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_dry_run_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_dry_run_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_dry_run_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_execution_gate_request_available",
}
REQUIRED_SEQUENCE_PLAN_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer"
)
REQUIRED_SEQUENCE_PLAN_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_dry_run_request_available"
)
REQUIRED_DRY_RUN_TYPE = "operator_proposal_apply_sequence_dry_run"
REQUIRED_CONSUMER_FALSE_FLAGS = (
    "dry_run_handler_invoked",
    "dry_run_created",
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
REQUIRED_DRY_RUN_FALSE_FLAGS = (
    "execution_allowed_by_dry_run",
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


def _duplicate_values(values: Tuple[str, ...]) -> Tuple[str, ...]:
    counts: Dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return tuple(value for value, count in counts.items() if count > 1)


def _result_id(result: Mapping[str, Any], index: int) -> str:
    value = _string(result.get("proposal_id"))
    return value if value else f"dry_run_result:{index}"


def _consumer_envelope_violations(consumer: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_SEQUENCE_PLAN_RESULT_CONSUMER_TYPE:
        violations.append("sequence_policy_dry_run_result_consumer_sequence_plan_consumer_type_mismatch")
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_SEQUENCE_PLAN_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append(
            "sequence_policy_dry_run_result_consumer_sequence_plan_consumer_method_id_mismatch"
        )
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_SEQUENCE_PLAN_RESULT_CONSUMER_STATUSES
    ):
        violations.append(
            "sequence_policy_dry_run_result_consumer_sequence_plan_consumer_status_unknown"
        )
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append(
            "sequence_policy_dry_run_result_consumer_sequence_plan_consumer_read_only_not_true"
        )
    if safety.get("sequence_plan_result_consumer_only") is not True:
        violations.append(
            "sequence_policy_dry_run_result_consumer_sequence_plan_consumer_only_not_true"
        )
    if safety.get("dry_run_request_template_only") is not True:
        violations.append(
            "sequence_policy_dry_run_result_consumer_sequence_plan_consumer_template_only_not_true"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_dry_run_result_consumer_sequence_plan_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if summary.get("dry_run_request_available") is not bool(expected_ids):
        violations.append(
            "sequence_policy_dry_run_result_consumer_sequence_plan_consumer_request_available_mismatch"
        )
    if _summary_int(summary.get("sequence_plan_step_count")) != len(expected_ids):
        violations.append(
            "sequence_policy_dry_run_result_consumer_sequence_plan_consumer_step_count_mismatch"
        )
    if _summary_int(summary.get("blocked_sequence_plan_step_count")) != 0:
        violations.append(
            "sequence_policy_dry_run_result_consumer_sequence_plan_consumer_blocked_count_mismatch"
        )
    if _summary_int(summary.get("invalid_record_count")) != 0:
        violations.append(
            "sequence_policy_dry_run_result_consumer_sequence_plan_consumer_invalid_count_mismatch"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_dry_run_result_consumer_sequence_plan_consumer_summary_{flag}_not_false"
            )
    if _string_list(summary.get("sequence_plan_step_proposal_ids")) != expected_ids:
        violations.append(
            "sequence_policy_dry_run_result_consumer_sequence_plan_consumer_step_ids_mismatch"
        )
    return tuple(violations)


def _dry_run_envelope_violations(dry_run: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if dry_run.get("type") != REQUIRED_DRY_RUN_TYPE:
        violations.append("sequence_policy_dry_run_result_consumer_dry_run_type_mismatch")
    if dry_run.get("status") != "ok":
        violations.append("sequence_policy_dry_run_result_consumer_dry_run_status_not_ok")
    if dry_run.get("success") is not True:
        violations.append("sequence_policy_dry_run_result_consumer_dry_run_success_not_true")
    safety = _mapping(dry_run.get("safety"))
    if safety.get("dry_run_only") is not True:
        violations.append("sequence_policy_dry_run_result_consumer_dry_run_only_not_true")
    if safety.get("sequence_plan_required") is not True:
        violations.append("sequence_policy_dry_run_result_consumer_dry_run_sequence_plan_required_not_true")
    if safety.get("current_queue_proposal_required") is not True:
        violations.append("sequence_policy_dry_run_result_consumer_dry_run_current_queue_not_true")
    if safety.get("does_not_call_apply_operator_proposal") is not True:
        violations.append("sequence_policy_dry_run_result_consumer_dry_run_does_not_call_apply_not_true")
    if safety.get("requires_operator_apply") is not True:
        violations.append("sequence_policy_dry_run_result_consumer_dry_run_operator_apply_not_true")
    if safety.get("live_apply_entrypoint") != "apply_operator_proposal":
        violations.append("sequence_policy_dry_run_result_consumer_dry_run_entrypoint_mismatch")
    allowed_kinds = _string_list(safety.get("allowed_live_write_kinds"))
    if tuple(sorted(allowed_kinds)) != tuple(sorted(CURRENT_PROVEN_LIVE_WRITE_KINDS)):
        violations.append("sequence_policy_dry_run_result_consumer_dry_run_allowed_kinds_mismatch")
    for flag in REQUIRED_DRY_RUN_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"sequence_policy_dry_run_result_consumer_dry_run_{flag}_not_false")
    return tuple(violations)


def _dry_run_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
    results: Tuple[Mapping[str, Any], ...],
    blocked_results: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "input_step_count": len(expected_ids),
        "result_count": len(results),
        "would_apply_count": len(results) - len(blocked_results),
        "blocked_count": len(blocked_results),
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"sequence_policy_dry_run_result_consumer_dry_run_summary_{key}_mismatch")
    for flag in REQUIRED_DRY_RUN_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(f"sequence_policy_dry_run_result_consumer_dry_run_summary_{flag}_not_false")
    if summary.get("queue_freshness_checked") is not True:
        violations.append("sequence_policy_dry_run_result_consumer_dry_run_summary_queue_freshness_not_true")
    would_apply_ids = tuple(
        _result_id(result, index)
        for index, result in enumerate(results)
        if result.get("status") == "would_apply"
    )
    if _string_list(summary.get("would_apply_proposal_ids")) != would_apply_ids:
        violations.append("sequence_policy_dry_run_result_consumer_dry_run_summary_would_apply_ids_mismatch")
    if _string_list(summary.get("blocked_proposal_ids")) != tuple(
        _result_id(result, index) for index, result in enumerate(blocked_results)
    ):
        violations.append("sequence_policy_dry_run_result_consumer_dry_run_summary_blocked_ids_mismatch")
    return tuple(violations)


def _dry_run_result_violations(
    result: Mapping[str, Any],
    *,
    index: int,
    prefix: str,
) -> Tuple[str, ...]:
    proposal_id = _result_id(result, index)
    violations = []
    if result.get("planned_apply_index") != index:
        violations.append(f"{proposal_id}:planned_apply_index_mismatch")
    if result.get("status") != "would_apply":
        violations.append(f"{proposal_id}:dry_run_status_not_would_apply")
    apply_request = _mapping(result.get("apply_request"))
    if apply_request.get("type") != "apply_operator_proposal":
        violations.append(f"{proposal_id}:apply_request_type_mismatch")
    if apply_request.get("proposal_id") != proposal_id:
        violations.append(f"{proposal_id}:apply_request_proposal_id_mismatch")
    if apply_request.get("approved") is not True:
        violations.append(f"{proposal_id}:apply_request_approved_not_true")
    if apply_request.get("idempotency_key") != f"{prefix}:{proposal_id}":
        violations.append(f"{proposal_id}:apply_request_idempotency_key_mismatch")
    current = _mapping(result.get("current_proposal"))
    if current.get("id") != proposal_id:
        violations.append(f"{proposal_id}:current_proposal_id_mismatch")
    if current.get("status") != "pending":
        violations.append(f"{proposal_id}:current_proposal_not_pending")
    if current.get("can_apply") is not True:
        violations.append(f"{proposal_id}:current_proposal_not_applyable")
    changes = _record_list(result.get("normalized_changes"))
    if not changes:
        violations.append(f"{proposal_id}:normalized_changes_missing")
    for change_index, change in enumerate(changes):
        if change.get("change_index") != change_index:
            violations.append(f"{proposal_id}:change_index_mismatch")
        value_type = _string(change.get("value_type"))
        if value_type not in ALLOWED_VALUE_TYPES:
            violations.append(f"{proposal_id}:change_value_type_not_gate_safe")
        effective_value = change.get("effective_value")
        if not isinstance(effective_value, (int, float)) or isinstance(effective_value, bool):
            violations.append(f"{proposal_id}:change_effective_value_invalid")
        elif value_type == "fader" and float(effective_value) > 0.0:
            violations.append(f"{proposal_id}:change_fader_ceiling_exceeded")
        if change.get("rollback_expected") is not True:
            violations.append(f"{proposal_id}:rollback_expected_not_true")
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer(
    *,
    sequence_plan_result_consumer: Mapping[str, Any] | None = None,
    dry_run_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify dry-run evidence and emit an execution-gate request template."""

    consumer = _mapping(sequence_plan_result_consumer)
    dry_run = _mapping(dry_run_result)
    consumer_summary = _mapping(consumer.get("summary"))
    expected_ids = _string_list(consumer_summary.get("sequence_plan_step_proposal_ids"))
    prefix = _string(consumer_summary.get("idempotency_key_prefix"))
    results = _record_list(dry_run.get("dry_run_results"))
    blocked_results = tuple(result for result in results if result.get("status") == "blocked")
    result_ids = tuple(_result_id(result, index) for index, result in enumerate(results))
    result_violations = tuple(
        violation
        for index, result in enumerate(results)
        if result.get("status") == "would_apply"
        for violation in _dry_run_result_violations(result, index=index, prefix=prefix)
    )
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_consumer_envelope_violations(consumer),
                *_consumer_summary_violations(consumer_summary, expected_ids=expected_ids),
                *(
                    ("sequence_policy_dry_run_result_consumer_sequence_plan_consumer_status_not_available",)
                    if consumer.get("status")
                    != "sequence_memory_trust_promotion_queue_import_dry_run_request_available"
                    else ()
                ),
                *_dry_run_envelope_violations(dry_run),
                *_dry_run_summary_violations(
                    _mapping(dry_run.get("summary")),
                    expected_ids=expected_ids,
                    results=results,
                    blocked_results=blocked_results,
                ),
                *(
                    f"duplicate_dry_run_result_id:{value}"
                    for value in _duplicate_values(result_ids)
                ),
                *(
                    ("sequence_policy_dry_run_result_consumer_result_id_outside_expected_set",)
                    if any(proposal_id not in set(expected_ids) for proposal_id in result_ids)
                    else ()
                ),
                *(
                    ("sequence_policy_dry_run_result_consumer_result_ids_do_not_cover_expected_set",)
                    if set(result_ids) != set(expected_ids)
                    else ()
                ),
                *(
                    ("sequence_policy_dry_run_result_consumer_result_order_mismatch",)
                    if not blocked_results and result_ids != expected_ids
                    else ()
                ),
                *result_violations,
            )
        )
    )

    if not consumer or not dry_run:
        status = "sequence_memory_trust_promotion_queue_import_dry_run_result_consumer_not_ready"
        blocked_by = ("sequence_plan_result_consumer_and_dry_run_result_required",)
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_dry_run_result_consumer_replacement_required"
        blocked_by = ("operator_must_replace_invalid_sequence_policy_dry_run_evidence",)
    elif blocked_results:
        status = "sequence_memory_trust_promotion_queue_import_dry_run_result_consumer_blocked"
        blocked_by = ("all_sequence_policy_imported_dry_run_results_must_be_would_apply",)
    elif not results:
        status = "sequence_memory_trust_promotion_queue_import_dry_run_result_consumer_not_ready"
        blocked_by = ("dry_run_would_apply_results_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_execution_gate_request_available"
        blocked_by = ("operator_may_run_generic_execution_gate_with_ack",)

    request_available = (
        status == "sequence_memory_trust_promotion_queue_import_execution_gate_request_available"
    )
    selected_results = results if request_available else ()
    gate_specs = tuple(
        {
            "execution_gate_request_spec_id": (
                f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_DRY_RUN_RESULT_CONSUMER_METHOD_ID}:{proposal_id}"
            ),
            "proposal_id": proposal_id,
            "planned_gate_index": index,
            "source_dry_run_index": index,
            "idempotency_key": f"{prefix}:{proposal_id}",
            "read_only": True,
            "dry_run_result_consumer_only": True,
            "execution_gate_handler": "gate_operator_proposal_apply_sequence_execution",
            "operator_ack_required_before_gate_ready": True,
            "execution_gate_invoked": False,
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
        for index, proposal_id in enumerate(result_ids if request_available else ())
    )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_DRY_RUN_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_SEQUENCE_PLAN_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "dry_run_result_consumer_only": True,
            "execution_gate_request_template_only": True,
            "requires_sequence_plan_result_consumer": True,
            "requires_dry_run_result": True,
            "requires_all_dry_run_results_would_apply": True,
            "operator_ack_required_before_gate_ready": True,
            "execution_gate_handler_invoked": False,
            "execution_gate_created": False,
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
            "expected_dry_run_result_count": len(expected_ids),
            "dry_run_result_count": len(results),
            "would_apply_result_count": len(results) - len(blocked_results),
            "blocked_dry_run_result_count": len(blocked_results),
            "invalid_record_count": len(invalid_record_ids),
            "idempotency_key_prefix": prefix,
            "dry_run_result_consumer_only": True,
            "execution_gate_request_available": request_available,
            "execution_gate_handler_invoked": False,
            "execution_gate_created": False,
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
            "expected_dry_run_proposal_ids": list(expected_ids),
            "dry_run_result_proposal_ids": list(result_ids),
            "would_apply_proposal_ids": [
                _result_id(result, index)
                for index, result in enumerate(results)
                if result.get("status") == "would_apply"
            ],
            "blocked_dry_run_proposal_ids": [
                _result_id(result, index)
                for index, result in enumerate(blocked_results)
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide sequence-plan result consumer plus dry-run result."
                if not consumer or not dry_run
                else (
                    "Replace invalid dry-run evidence before execution gate."
                    if invalid_record_ids
                    else (
                        "Re-run dry-run until every sequence-policy imported result is would-apply."
                        if blocked_results
                        else "Operator may run the generic execution gate with explicit acknowledgement; this consumer does not run it."
                    )
                )
            ),
        },
        "execution_gate_request": {
            "handler": "gate_operator_proposal_apply_sequence_execution",
            "dry_run_payload": {
                **dict(dry_run),
                "dry_run_results": list(selected_results),
            }
            if request_available
            else {},
            "operator_ack": False,
            "operator_must_set_operator_ack_true": True,
            "read_only_template": True,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "execution_gate_created": False,
            "execution_allowed_by_consumer": False,
        },
        "execution_gate_request_specs": list(gate_specs),
    }
