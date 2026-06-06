"""Read-only consumer for promotion dry-run results.

This verifier connects the promotion sequence-plan result consumer to the
generic execution-gate handler. It verifies promotion provenance, delegates
dry-run validation to the existing queue-import dry-run result consumer, then
emits only an execution-gate request template. It does not invoke handlers,
reserve idempotency, mutate queue/proposal state, create execution gates,
execute sequences, or touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_DRY_RUN_RESULT_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_PLAN_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_PLAN_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_DRY_RUN_RESULT_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_PLAN_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_PLAN_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_DRY_RUN_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_DRY_RUN_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_execution_gate_request_available",
}
REQUIRED_PRIOR_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer"
)
REQUIRED_PRIOR_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_dry_run_request_available"
)
REQUIRED_GENERIC_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_execution_gate_request_available"
)
REQUIRED_GENERIC_REPLACEMENT_STATUS = (
    "sequence_memory_trust_promotion_queue_import_dry_run_result_consumer_replacement_required"
)
REQUIRED_GENERIC_BLOCKED_STATUS = (
    "sequence_memory_trust_promotion_queue_import_dry_run_result_consumer_blocked"
)
REQUIRED_PRIOR_FALSE_FLAGS = (
    "dry_run_handler_invoked",
    "dry_run_created",
    "execution_allowed_by_consumer",
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "proposal_order_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "policy_mutation",
    "memory_mutation",
    "memory_file_mutation",
    "runtime_memory_mutation",
    "model_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_apply_live",
)
REQUIRED_REQUEST_FALSE_FLAGS = (
    "dry_run_handler_invoked",
    "dry_run_created",
    "execution_allowed_by_consumer",
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "proposal_order_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "policy_mutation",
    "memory_mutation",
    "memory_file_mutation",
    "runtime_memory_mutation",
    "model_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "live_authority_granted",
    "full_autonomous_ready",
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


def _prior_envelope_violations(prior: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if prior.get("type") != REQUIRED_PRIOR_TYPE:
        violations.append("promotion_dry_run_result_consumer_prior_type_mismatch")
    generated_at = prior.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("promotion_dry_run_result_consumer_prior_generated_at_invalid")
    if (
        prior.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_PLAN_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append("promotion_dry_run_result_consumer_prior_method_id_mismatch")
    if (
        prior.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_PLAN_RESULT_CONSUMER_STATUSES
    ):
        violations.append("promotion_dry_run_result_consumer_prior_status_unknown")
    safety = _mapping(prior.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("promotion_dry_run_result_consumer_prior_read_only_not_true")
    if safety.get("promotion_sequence_plan_result_consumer_only") is not True:
        violations.append("promotion_dry_run_result_consumer_prior_only_not_true")
    if safety.get("promotion_dry_run_request_template_only") is not True:
        violations.append("promotion_dry_run_result_consumer_prior_template_only_not_true")
    for flag in REQUIRED_PRIOR_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"promotion_dry_run_result_consumer_prior_{flag}_not_false")
    return tuple(violations)


def _prior_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
    expected_prefix: str,
) -> Tuple[str, ...]:
    violations = []
    if summary.get("promotion_dry_run_request_available") is not bool(expected_ids):
        violations.append("promotion_dry_run_result_consumer_prior_summary_request_available_mismatch")
    if _summary_int(summary.get("sequence_plan_step_count")) != len(expected_ids):
        violations.append("promotion_dry_run_result_consumer_prior_summary_step_count_mismatch")
    if _summary_int(summary.get("blocked_sequence_plan_step_count")) != 0:
        violations.append("promotion_dry_run_result_consumer_prior_summary_blocked_count_mismatch")
    if _summary_int(summary.get("invalid_record_count")) != 0:
        violations.append("promotion_dry_run_result_consumer_prior_summary_invalid_count_mismatch")
    if summary.get("idempotency_key_prefix") != expected_prefix:
        violations.append("promotion_dry_run_result_consumer_prior_summary_prefix_mismatch")
    if _string_list(summary.get("sequence_plan_step_proposal_ids")) != expected_ids:
        violations.append("promotion_dry_run_result_consumer_prior_summary_step_ids_mismatch")
    for flag in REQUIRED_PRIOR_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"promotion_dry_run_result_consumer_prior_summary_{flag}_not_false"
            )
    return tuple(violations)


def _prior_request_violations(
    request: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if request.get("handler") != "dry_run_operator_proposal_apply_sequence":
        violations.append("promotion_dry_run_result_consumer_prior_request_handler_mismatch")
    if request.get("read_only_template") is not True:
        violations.append("promotion_dry_run_result_consumer_prior_request_template_not_true")
    if request.get("promotion_sequence_plan_result_consumer_only") is not True:
        violations.append("promotion_dry_run_result_consumer_prior_request_only_not_true")
    if request.get("promotion_dry_run_request_template_only") is not True:
        violations.append("promotion_dry_run_result_consumer_prior_request_template_only_not_true")
    plan = _mapping(request.get("sequence_plan"))
    if len(_record_list(plan.get("planned_apply_sequence"))) != len(expected_ids):
        violations.append("promotion_dry_run_result_consumer_prior_request_step_count_mismatch")
    for flag in REQUIRED_REQUEST_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(
                f"promotion_dry_run_result_consumer_prior_request_{flag}_not_false"
            )
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer(
    *,
    promotion_sequence_plan_result_consumer: Mapping[str, Any] | None = None,
    promotion_dry_run_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify promotion dry-run evidence and emit an execution-gate template."""

    prior = _mapping(promotion_sequence_plan_result_consumer)
    dry_run = _mapping(promotion_dry_run_result)
    prior_summary = _mapping(prior.get("summary"))
    prior_request = _mapping(prior.get("promotion_dry_run_request"))
    expected_ids = _string_list(prior_summary.get("sequence_plan_step_proposal_ids"))
    expected_prefix = _string(prior_summary.get("idempotency_key_prefix"))
    prior_status = _string(prior.get("status"))
    generic_sequence_plan_consumer = _mapping(
        prior.get("generic_sequence_plan_result_consumer")
    )

    generic_consumer: Mapping[str, Any] = {}
    if prior and dry_run:
        generic_consumer = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer(
                sequence_plan_result_consumer=generic_sequence_plan_consumer,
                dry_run_result=dry_run,
            )
        )
    generic_status = _string(generic_consumer.get("status"))
    generic_summary = _mapping(generic_consumer.get("summary"))
    generic_request = _mapping(generic_consumer.get("execution_gate_request"))
    generic_specs = _record_list(generic_consumer.get("execution_gate_request_specs"))
    generic_invalid_record_ids = tuple(
        f"generic_dry_run_result_consumer:{item}"
        for item in _string_list(generic_summary.get("invalid_record_ids"))
    )

    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_string_list(prior_summary.get("invalid_record_ids")),
                *_prior_envelope_violations(prior),
                *_prior_summary_violations(
                    prior_summary,
                    expected_ids=expected_ids,
                    expected_prefix=expected_prefix,
                ),
                *_prior_request_violations(
                    prior_request,
                    expected_ids=expected_ids,
                ),
                *(
                    generic_invalid_record_ids
                    if generic_status == REQUIRED_GENERIC_REPLACEMENT_STATUS
                    else ()
                ),
                *(
                    ("promotion_dry_run_result_consumer_generic_status_unknown",)
                    if (
                        generic_status
                        and generic_status
                        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_DRY_RUN_RESULT_CONSUMER_STATUSES
                    )
                    else ()
                ),
            )
        )
    )
    invalid_record_ids = () if not prior or not dry_run else raw_invalid_record_ids

    if not prior or not dry_run:
        status = "sequence_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer_not_ready"
        blocked_by = (
            "promotion_sequence_plan_result_consumer_and_dry_run_result_required",
        )
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer_replacement_required"
        blocked_by = ("operator_must_replace_invalid_promotion_dry_run_evidence",)
    elif prior_status != REQUIRED_PRIOR_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer_blocked"
        blocked_by = (
            "promotion_sequence_plan_result_consumer_must_request_dry_run_first",
        )
    elif generic_status == REQUIRED_GENERIC_BLOCKED_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer_blocked"
        blocked_by = ("all_promotion_dry_run_results_must_be_would_apply",)
    elif generic_status != REQUIRED_GENERIC_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer_not_ready"
        blocked_by = ("ready_promotion_dry_run_result_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_execution_gate_request_available"
        blocked_by = ("operator_may_run_generic_execution_gate_with_ack",)

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_execution_gate_request_available"
    )
    dry_run_payload = _mapping(generic_request.get("dry_run_payload")) if request_available else {}
    promotion_execution_gate_request: Dict[str, Any] = {
        "handler": "gate_operator_proposal_apply_sequence_execution",
        "dry_run_payload": dict(dry_run_payload),
        "operator_ack": False,
        "operator_must_set_operator_ack_true": True,
        "read_only_template": True,
        "promotion_dry_run_result_consumer_only": True,
        "promotion_execution_gate_request_template_only": True,
        "execution_gate_handler_invoked": False,
        "execution_gate_created": False,
        "execution_allowed_by_consumer": False,
        "live_write": False,
        "apply_attempted": False,
        "queue_mutation": False,
        "proposal_status_mutation": False,
        "proposal_order_mutation": False,
        "idempotency_reserved": False,
        "manual_console_write_called": False,
        "policy_mutation": False,
        "memory_mutation": False,
        "memory_file_mutation": False,
        "runtime_memory_mutation": False,
        "model_mutation": False,
        "sequence_policy_trust_store_mutation": False,
        "proposal_state_mutation": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "can_apply_live": False,
    }
    if request_available:
        promotion_execution_gate_request["promotion_dry_run_result_consumer"] = {
            "method_id": (
                OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_DRY_RUN_RESULT_CONSUMER_METHOD_ID
            ),
            "source_prior_method_id": (
                OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_PLAN_RESULT_CONSUMER_METHOD_ID
            ),
            "would_apply_proposal_ids": list(expected_ids),
            "idempotency_key_prefix": expected_prefix,
            "read_only": True,
        }

    execution_gate_request_specs = tuple(
        {
            **dict(spec),
            "promotion_execution_gate_request_spec_id": (
                f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_DRY_RUN_RESULT_CONSUMER_METHOD_ID}:{spec.get('proposal_id')}"
            ),
            "promotion_dry_run_result_consumer_only": True,
            "promotion_execution_gate_request_template_only": True,
            "execution_gate_handler_invoked": False,
            "execution_gate_created": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
        }
        for spec in generic_specs
    ) if request_available else ()

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_DRY_RUN_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_PLAN_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_dry_run_result_consumer_only": True,
            "promotion_execution_gate_request_template_only": True,
            "requires_promotion_sequence_plan_result_consumer": True,
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
            "proposal_order_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "policy_mutation": False,
            "memory_mutation": False,
            "memory_file_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "sequence_policy_trust_store_mutation": False,
            "proposal_state_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
        },
        "summary": {
            "expected_dry_run_result_count": len(expected_ids),
            "dry_run_result_count": _summary_int(
                generic_summary.get("dry_run_result_count")
            ) or 0,
            "would_apply_result_count": _summary_int(
                generic_summary.get("would_apply_result_count")
            ) or 0,
            "blocked_dry_run_result_count": _summary_int(
                generic_summary.get("blocked_dry_run_result_count")
            ) or 0,
            "invalid_record_count": len(invalid_record_ids),
            "idempotency_key_prefix": expected_prefix,
            "promotion_dry_run_result_consumer_only": True,
            "promotion_execution_gate_request_template_only": True,
            "promotion_execution_gate_request_available": request_available,
            "operator_ack_required_before_gate_ready": True,
            "execution_gate_handler_invoked": False,
            "execution_gate_created": False,
            "execution_allowed_by_consumer": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "proposal_order_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "policy_mutation": False,
            "memory_mutation": False,
            "memory_file_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "sequence_policy_trust_store_mutation": False,
            "proposal_state_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
            "expected_dry_run_proposal_ids": list(expected_ids),
            "dry_run_result_proposal_ids": list(
                _string_list(generic_summary.get("dry_run_result_proposal_ids"))
            ),
            "would_apply_proposal_ids": list(
                _string_list(generic_summary.get("would_apply_proposal_ids"))
            ),
            "blocked_dry_run_proposal_ids": list(
                _string_list(generic_summary.get("blocked_dry_run_proposal_ids"))
            ),
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion sequence-plan result consumer and dry-run result."
                if not prior or not dry_run
                else (
                    "Replace invalid promotion dry-run evidence before execution gate."
                    if invalid_record_ids
                    else (
                        "Re-run dry-run until every promotion result is would-apply."
                        if generic_status == REQUIRED_GENERIC_BLOCKED_STATUS
                        else "Operator may run the generic execution gate with explicit acknowledgement; this consumer does not run it."
                    )
                )
            ),
        },
        "promotion_execution_gate_request": promotion_execution_gate_request,
        "promotion_execution_gate_request_specs": list(execution_gate_request_specs),
        "generic_dry_run_result_consumer": generic_consumer if prior and dry_run else {},
    }
