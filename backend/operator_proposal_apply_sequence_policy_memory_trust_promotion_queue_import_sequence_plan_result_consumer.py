"""Read-only consumer for sequence-policy queue-import sequence-plan results.

This module verifies that a generic apply-sequence plan result matches the
sequence-policy queue-import preflight-result consumer before a caller may pass
the plan to the generic dry-run handler. It does not invoke dry-run, gate, or
executor handlers, reserve idempotency keys, mutate queue or proposal state, or
touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_RESULT_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_RESULT_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer,
    )
try:
    from operator_mode_policy import CURRENT_PROVEN_LIVE_WRITE_KINDS
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_mode_policy import CURRENT_PROVEN_LIVE_WRITE_KINDS

ALLOWED_VALUE_TYPES = set(CURRENT_PROVEN_LIVE_WRITE_KINDS)


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_SEQUENCE_PLAN_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_SEQUENCE_PLAN_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_sequence_plan_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_sequence_plan_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_sequence_plan_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_dry_run_request_available",
}
REQUIRED_PREFLIGHT_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer"
)
REQUIRED_SEQUENCE_PLAN_TYPE = "operator_proposal_apply_sequence_plan"
REQUIRED_PREFLIGHT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_sequence_plan_request_available"
)
REQUIRED_CONSUMER_FALSE_FLAGS = (
    "sequence_plan_handler_invoked",
    "sequence_plan_created",
    "sequence_execution_allowed_by_consumer",
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
REQUIRED_SEQUENCE_PLAN_FALSE_FLAGS = (
    "execution_allowed_by_plan",
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
REQUIRED_SEQUENCE_REQUEST_SPEC_FALSE_FLAGS = (
    "sequence_plan_created",
    "sequence_execution_allowed_by_consumer",
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
REQUIRED_SEQUENCE_REQUEST_SPEC_IDENTITY_FIELDS = (
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


def _step_id(step: Mapping[str, Any], index: int) -> str:
    value = _string(step.get("proposal_id"))
    return value if value else f"sequence_plan_step:{index}"


def _consumer_envelope_violations(consumer: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_PREFLIGHT_RESULT_CONSUMER_TYPE:
        violations.append("sequence_policy_sequence_plan_result_consumer_preflight_consumer_type_mismatch")
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append(
            "sequence_policy_sequence_plan_result_consumer_preflight_consumer_method_id_mismatch"
        )
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_RESULT_CONSUMER_STATUSES
    ):
        violations.append(
            "sequence_policy_sequence_plan_result_consumer_preflight_consumer_status_unknown"
        )
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append(
            "sequence_policy_sequence_plan_result_consumer_preflight_consumer_read_only_not_true"
        )
    if safety.get("preflight_result_consumer_only") is not True:
        violations.append(
            "sequence_policy_sequence_plan_result_consumer_preflight_consumer_only_not_true"
        )
    if safety.get("sequence_plan_request_template_only") is not True:
        violations.append(
            "sequence_policy_sequence_plan_result_consumer_preflight_consumer_template_only_not_true"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_sequence_plan_result_consumer_preflight_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if summary.get("sequence_plan_request_available") is not bool(expected_ids):
        violations.append(
            "sequence_policy_sequence_plan_result_consumer_preflight_consumer_request_available_mismatch"
        )
    if _summary_int(summary.get("ready_preflight_result_count")) != len(expected_ids):
        violations.append(
            "sequence_policy_sequence_plan_result_consumer_preflight_consumer_ready_count_mismatch"
        )
    if _summary_int(summary.get("blocked_preflight_result_count")) != 0:
        violations.append(
            "sequence_policy_sequence_plan_result_consumer_preflight_consumer_blocked_count_mismatch"
        )
    if _summary_int(summary.get("invalid_record_count")) != 0:
        violations.append(
            "sequence_policy_sequence_plan_result_consumer_preflight_consumer_invalid_count_mismatch"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_sequence_plan_result_consumer_preflight_consumer_summary_{flag}_not_false"
            )
    if _string_list(summary.get("ready_preflight_result_proposal_ids")) != expected_ids:
        violations.append(
            "sequence_policy_sequence_plan_result_consumer_preflight_consumer_ready_ids_mismatch"
        )
    return tuple(violations)


def _sequence_plan_envelope_violations(plan: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if plan.get("type") != REQUIRED_SEQUENCE_PLAN_TYPE:
        violations.append("sequence_policy_sequence_plan_result_consumer_plan_type_mismatch")
    if plan.get("status") != "ok":
        violations.append("sequence_policy_sequence_plan_result_consumer_plan_status_not_ok")
    if plan.get("success") is not True:
        violations.append("sequence_policy_sequence_plan_result_consumer_plan_success_not_true")
    safety = _mapping(plan.get("safety"))
    if safety.get("sequence_plan_only") is not True:
        violations.append("sequence_policy_sequence_plan_result_consumer_plan_only_not_true")
    if safety.get("preflight_results_required") is not True:
        violations.append("sequence_policy_sequence_plan_result_consumer_plan_preflight_required_not_true")
    if safety.get("current_queue_proposal_required") is not True:
        violations.append("sequence_policy_sequence_plan_result_consumer_plan_queue_current_not_true")
    if safety.get("requires_operator_apply") is not True:
        violations.append("sequence_policy_sequence_plan_result_consumer_plan_operator_apply_not_true")
    if safety.get("live_apply_entrypoint") != "apply_operator_proposal":
        violations.append("sequence_policy_sequence_plan_result_consumer_plan_entrypoint_mismatch")
    allowed_kinds = _string_list(safety.get("allowed_live_write_kinds"))
    if tuple(sorted(allowed_kinds)) != tuple(sorted(CURRENT_PROVEN_LIVE_WRITE_KINDS)):
        violations.append("sequence_policy_sequence_plan_result_consumer_plan_allowed_kinds_mismatch")
    for flag in REQUIRED_SEQUENCE_PLAN_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"sequence_policy_sequence_plan_result_consumer_plan_{flag}_not_false")
    return tuple(violations)


def _sequence_plan_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
    steps: Tuple[Mapping[str, Any], ...],
    blocked_results: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "input_preflight_result_count": len(expected_ids),
        "ready_result_count": len(expected_ids),
        "blocked_result_count": 0,
        "planned_apply_count": len(steps),
        "plan_blocked_count": len(blocked_results),
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"sequence_policy_sequence_plan_result_consumer_plan_summary_{key}_mismatch")
    for flag in (
        "execution_allowed_by_plan",
        "live_write",
        "apply_attempted",
        "queue_mutation",
        "idempotency_reserved",
        "manual_console_write_called",
        "mutates_console_state",
        "queries_console",
        "live_widening_allowed",
    ):
        if summary.get(flag) is not False:
            violations.append(f"sequence_policy_sequence_plan_result_consumer_plan_summary_{flag}_not_false")
    if summary.get("queue_freshness_checked") is not True:
        violations.append("sequence_policy_sequence_plan_result_consumer_plan_summary_queue_freshness_not_true")
    if _string_list(summary.get("planned_apply_proposal_ids")) != tuple(
        _step_id(step, index) for index, step in enumerate(steps)
    ):
        violations.append("sequence_policy_sequence_plan_result_consumer_plan_summary_planned_ids_mismatch")
    if _string_list(summary.get("blocked_proposal_ids")) != tuple(
        item.get("proposal_id")
        for item in blocked_results
        if item.get("proposal_id")
    ):
        violations.append("sequence_policy_sequence_plan_result_consumer_plan_summary_blocked_ids_mismatch")
    return tuple(violations)


def _sequence_step_violations(
    step: Mapping[str, Any],
    *,
    index: int,
    prefix: str,
) -> Tuple[str, ...]:
    proposal_id = _step_id(step, index)
    violations = []
    if step.get("planned_apply_index") != index:
        violations.append(f"{proposal_id}:planned_apply_index_mismatch")
    if step.get("requires_operator_apply") is not True:
        violations.append(f"{proposal_id}:requires_operator_apply_not_true")
    if step.get("approval_required") is not True:
        violations.append(f"{proposal_id}:approval_required_not_true")
    if step.get("live_apply_entrypoint") != "apply_operator_proposal":
        violations.append(f"{proposal_id}:live_apply_entrypoint_mismatch")
    if step.get("execution_allowed_by_plan") is not False:
        violations.append(f"{proposal_id}:execution_allowed_by_plan_not_false")
    if step.get("idempotency_key") != f"{prefix}:{proposal_id}":
        violations.append(f"{proposal_id}:idempotency_key_mismatch")
    current = _mapping(step.get("current_proposal"))
    if current.get("id") != proposal_id:
        violations.append(f"{proposal_id}:current_proposal_id_mismatch")
    if current.get("status") != "pending":
        violations.append(f"{proposal_id}:current_proposal_not_pending")
    if current.get("can_apply") is not True:
        violations.append(f"{proposal_id}:current_proposal_not_applyable")
    apply_request = _mapping(step.get("apply_request"))
    if apply_request.get("type") != "apply_operator_proposal":
        violations.append(f"{proposal_id}:apply_request_type_mismatch")
    if apply_request.get("proposal_id") != proposal_id:
        violations.append(f"{proposal_id}:apply_request_proposal_id_mismatch")
    if apply_request.get("approved") is not True:
        violations.append(f"{proposal_id}:apply_request_approved_not_true")
    if apply_request.get("idempotency_key") != step.get("idempotency_key"):
        violations.append(f"{proposal_id}:apply_request_idempotency_key_mismatch")
    changes = _record_list(step.get("normalized_changes"))
    if not changes:
        violations.append(f"{proposal_id}:normalized_changes_missing")
    if _summary_int(step.get("change_count")) != len(changes):
        violations.append(f"{proposal_id}:change_count_mismatch")
    for change_index, change in enumerate(changes):
        if change.get("change_index") != change_index:
            violations.append(f"{proposal_id}:change_index_mismatch")
        value_type = _string(change.get("value_type"))
        if value_type not in ALLOWED_VALUE_TYPES:
            violations.append(f"{proposal_id}:change_value_type_not_dry_run_safe")
        effective_value = change.get("effective_value")
        if not isinstance(effective_value, (int, float)) or isinstance(effective_value, bool):
            violations.append(f"{proposal_id}:change_effective_value_invalid")
        elif value_type == "fader" and float(effective_value) > 0.0:
            violations.append(f"{proposal_id}:change_fader_ceiling_exceeded")
        if change.get("rollback_expected") is not True:
            violations.append(f"{proposal_id}:rollback_expected_not_true")
    return tuple(violations)


def _sequence_request_specs_violations(
    specs: Tuple[Mapping[str, Any], ...],
    *,
    expected_ids: Tuple[str, ...],
    prefix: str,
) -> Tuple[str, ...]:
    violations = []
    spec_ids = tuple(_string(spec.get("proposal_id")) for spec in specs)
    if spec_ids != expected_ids:
        violations.append(
            "sequence_policy_sequence_plan_result_consumer_request_specs_candidate_ids_mismatch"
        )
    for index, spec in enumerate(specs):
        proposal_id = _string(spec.get("proposal_id")) or f"sequence_request_spec:{index}"
        record_id = _string(spec.get("sequence_plan_request_spec_id")) or proposal_id
        if spec.get("planned_sequence_index") != index:
            violations.append(f"{record_id}:planned_sequence_index_mismatch")
        if spec.get("source_preflight_index") != index:
            violations.append(f"{record_id}:source_preflight_index_mismatch")
        if spec.get("idempotency_key") != f"{prefix}:{proposal_id}":
            violations.append(f"{record_id}:idempotency_key_mismatch")
        if spec.get("sequence_plan_handler") != "plan_operator_proposal_apply_sequence":
            violations.append(f"{record_id}:sequence_plan_handler_mismatch")
        if spec.get("read_only") is not True:
            violations.append(f"{record_id}:read_only_not_true")
        if spec.get("preflight_result_consumer_only") is not True:
            violations.append(f"{record_id}:preflight_result_consumer_only_not_true")
        if spec.get("operator_sequence_plan_step_required_before_dry_run") is not True:
            violations.append(
                f"{record_id}:operator_sequence_plan_step_required_before_dry_run_not_true"
            )
        if _summary_int(spec.get("planned_queue_index")) is None:
            violations.append(f"{proposal_id}:planned_queue_index_invalid")
        if spec.get("rank_source") != "sequence_policy_consumer":
            violations.append(f"{proposal_id}:rank_source_mismatch")
        for field in REQUIRED_SEQUENCE_REQUEST_SPEC_IDENTITY_FIELDS:
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
        for flag in REQUIRED_SEQUENCE_REQUEST_SPEC_FALSE_FLAGS:
            if spec.get(flag) is not False:
                violations.append(f"{record_id}:{flag}_not_false")
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer(
    *,
    preflight_result_consumer: Mapping[str, Any] | None = None,
    sequence_plan_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify a sequence-policy sequence-plan result and emit dry-run template."""

    consumer = _mapping(preflight_result_consumer)
    plan = _mapping(sequence_plan_result)
    consumer_summary = _mapping(consumer.get("summary"))
    expected_ids = _string_list(consumer_summary.get("ready_preflight_result_proposal_ids"))
    prefix = _string(consumer_summary.get("idempotency_key_prefix"))
    sequence_request_specs = _record_list(consumer.get("sequence_plan_request_specs"))
    sequence_request_specs_by_proposal_id = {
        _string(spec.get("proposal_id")): spec
        for spec in sequence_request_specs
        if _string(spec.get("proposal_id"))
    }
    steps = _record_list(plan.get("planned_apply_sequence"))
    blocked_results = _record_list(plan.get("blocked_results"))
    step_ids = tuple(_step_id(step, index) for index, step in enumerate(steps))
    blocked_ids = tuple(
        _string(item.get("proposal_id"))
        for item in blocked_results
        if _string(item.get("proposal_id"))
    )
    step_violations = tuple(
        violation
        for index, step in enumerate(steps)
        for violation in _sequence_step_violations(step, index=index, prefix=prefix)
    )
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_consumer_envelope_violations(consumer),
                *_consumer_summary_violations(consumer_summary, expected_ids=expected_ids),
                *_sequence_request_specs_violations(
                    sequence_request_specs,
                    expected_ids=expected_ids,
                    prefix=prefix,
                ),
                *(
                    ("sequence_policy_sequence_plan_result_consumer_preflight_consumer_status_not_available",)
                    if consumer.get("status")
                    != REQUIRED_PREFLIGHT_CONSUMER_AVAILABLE_STATUS
                    else ()
                ),
                *_sequence_plan_envelope_violations(plan),
                *_sequence_plan_summary_violations(
                    _mapping(plan.get("summary")),
                    expected_ids=expected_ids,
                    steps=steps,
                    blocked_results=blocked_results,
                ),
                *(
                    f"duplicate_sequence_plan_step_id:{value}"
                    for value in _duplicate_values(step_ids)
                ),
                *(
                    f"duplicate_sequence_plan_blocked_id:{value}"
                    for value in _duplicate_values(blocked_ids)
                ),
                *(
                    ("sequence_policy_sequence_plan_result_consumer_step_order_mismatch",)
                    if not blocked_results and step_ids != expected_ids
                    else ()
                ),
                *(
                    ("sequence_policy_sequence_plan_result_consumer_step_or_blocked_id_outside_expected_set",)
                    if blocked_results
                    and any(
                        proposal_id not in set(expected_ids)
                        for proposal_id in (*step_ids, *blocked_ids)
                    )
                    else ()
                ),
                *(
                    ("sequence_policy_sequence_plan_result_consumer_step_and_blocked_ids_do_not_cover_expected_set",)
                    if blocked_results and set((*step_ids, *blocked_ids)) != set(expected_ids)
                    else ()
                ),
                *step_violations,
            )
        )
    )

    if not consumer or not plan:
        status = (
            "sequence_memory_trust_promotion_queue_import_sequence_plan_result_consumer_not_ready"
        )
        blocked_by = ("preflight_result_consumer_and_sequence_plan_result_required",)
    elif invalid_record_ids:
        status = (
            "sequence_memory_trust_promotion_queue_import_sequence_plan_result_consumer_replacement_required"
        )
        blocked_by = ("operator_must_replace_invalid_sequence_policy_sequence_plan_evidence",)
    elif blocked_results:
        status = (
            "sequence_memory_trust_promotion_queue_import_sequence_plan_result_consumer_blocked"
        )
        blocked_by = ("all_sequence_policy_imported_sequence_plan_steps_must_be_planned",)
    elif not steps:
        status = (
            "sequence_memory_trust_promotion_queue_import_sequence_plan_result_consumer_not_ready"
        )
        blocked_by = ("planned_sequence_steps_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_dry_run_request_available"
        blocked_by = ("operator_may_run_generic_sequence_dry_run_handler",)

    request_available = (
        status == "sequence_memory_trust_promotion_queue_import_dry_run_request_available"
    )
    selected_plan = plan if request_available else {}
    selected_sequence_request_specs = tuple(
        sequence_request_specs_by_proposal_id.get(proposal_id, {})
        for proposal_id in step_ids
        if request_available
    )
    dry_run_specs = tuple(
        {
            "dry_run_request_spec_id": (
                f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_SEQUENCE_PLAN_RESULT_CONSUMER_METHOD_ID}:{proposal_id}"
            ),
            "proposal_id": proposal_id,
            "planned_dry_run_index": index,
            "source_sequence_plan_index": index,
            "source_sequence_plan_request_spec_id": _string(
                sequence_request_specs_by_proposal_id.get(proposal_id, {}).get(
                    "sequence_plan_request_spec_id"
                )
            ),
            "source_preflight_request_spec_id": _string(
                sequence_request_specs_by_proposal_id.get(proposal_id, {}).get(
                    "source_preflight_request_spec_id"
                )
            ),
            "queue_ordering_plan_record_id": sequence_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("queue_ordering_plan_record_id"),
            "consumer_record_id": sequence_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("consumer_record_id"),
            "sequence_policy_gate_record_id": sequence_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("sequence_policy_gate_record_id"),
            "policy_key": sequence_request_specs_by_proposal_id.get(proposal_id, {}).get(
                "policy_key"
            ),
            "source_idempotency_key": sequence_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("source_idempotency_key"),
            "current_sequence_review_freshness": sequence_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("current_sequence_review_freshness"),
            "planned_queue_index": sequence_request_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("planned_queue_index"),
            "rank_source": sequence_request_specs_by_proposal_id.get(proposal_id, {}).get(
                "rank_source"
            ),
            "idempotency_key": f"{prefix}:{proposal_id}",
            "read_only": True,
            "sequence_plan_result_consumer_only": True,
            "dry_run_handler": "dry_run_operator_proposal_apply_sequence",
            "operator_dry_run_step_required_before_gate": True,
            "dry_run_invoked": False,
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
        for index, proposal_id in enumerate(step_ids if request_available else ())
    )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_SEQUENCE_PLAN_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "sequence_plan_result_consumer_only": True,
            "dry_run_request_template_only": True,
            "requires_preflight_result_consumer": True,
            "requires_sequence_plan_result": True,
            "requires_all_sequence_plan_steps_planned": True,
            "dry_run_handler_invoked": False,
            "dry_run_created": False,
            "operator_dry_run_step_required_before_gate": True,
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
            "expected_sequence_plan_step_count": len(expected_ids),
            "sequence_plan_request_spec_count": len(sequence_request_specs),
            "sequence_plan_step_count": len(steps),
            "blocked_sequence_plan_step_count": len(blocked_results),
            "invalid_record_count": len(invalid_record_ids),
            "idempotency_key_prefix": prefix,
            "sequence_plan_result_consumer_only": True,
            "dry_run_request_available": request_available,
            "dry_run_handler_invoked": False,
            "dry_run_created": False,
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
            "expected_sequence_plan_proposal_ids": list(expected_ids),
            "sequence_plan_request_spec_proposal_ids": [
                spec.get("proposal_id") for spec in sequence_request_specs
            ],
            "sequence_plan_step_proposal_ids": list(step_ids),
            "blocked_sequence_plan_proposal_ids": [
                item.get("proposal_id")
                for item in blocked_results
                if item.get("proposal_id")
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide preflight-result consumer plus sequence-plan result."
                if not consumer or not plan
                else (
                    "Replace invalid sequence-plan evidence before dry-run."
                    if invalid_record_ids
                    else (
                        "Re-run sequence planning until every sequence-policy imported candidate is planned."
                        if blocked_results
                        else "Operator may run the generic dry-run handler; this consumer does not run it."
                    )
                )
            ),
        },
        "dry_run_request": {
            "handler": "dry_run_operator_proposal_apply_sequence",
            "sequence_plan": selected_plan,
            "sequence_plan_request_specs": list(selected_sequence_request_specs),
            "read_only_template": True,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "dry_run_created": False,
            "execution_allowed_by_consumer": False,
        },
        "dry_run_request_specs": list(dry_run_specs),
    }
