"""Read-only consumer for promotion batch-preflight results.

This verifier connects the promotion preflight-plan result consumer to the
generic sequence-plan handler. It verifies promotion provenance, delegates
batch-preflight validation to the existing queue-import preflight-result
consumer, then emits only a sequence-plan request template. It does not invoke
handlers, reserve idempotency, mutate queue/proposal state, create sequence
plans, execute sequences, or touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_RESULT_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PREFLIGHT_PLAN_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PREFLIGHT_PLAN_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_RESULT_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PREFLIGHT_PLAN_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PREFLIGHT_PLAN_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_BATCH_PREFLIGHT_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_BATCH_PREFLIGHT_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_sequence_plan_request_available",
}
REQUIRED_PRIOR_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer"
)
REQUIRED_PRIOR_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_request_available"
)
REQUIRED_GENERIC_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_sequence_plan_request_available"
)
REQUIRED_GENERIC_REPLACEMENT_STATUS = (
    "sequence_memory_trust_promotion_queue_import_preflight_result_consumer_replacement_required"
)
REQUIRED_GENERIC_BLOCKED_STATUS = (
    "sequence_memory_trust_promotion_queue_import_preflight_result_consumer_blocked"
)
REQUIRED_PRIOR_FALSE_FLAGS = (
    "promotion_batch_preflight_handler_invoked",
    "promotion_batch_preflight_created",
    "batch_preflight_invoked",
    "sequence_plan_created",
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
    "promotion_batch_preflight_handler_invoked",
    "promotion_batch_preflight_created",
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
        violations.append("promotion_batch_preflight_result_consumer_prior_type_mismatch")
    generated_at = prior.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("promotion_batch_preflight_result_consumer_prior_generated_at_invalid")
    if (
        prior.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PREFLIGHT_PLAN_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append("promotion_batch_preflight_result_consumer_prior_method_id_mismatch")
    if (
        prior.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PREFLIGHT_PLAN_RESULT_CONSUMER_STATUSES
    ):
        violations.append("promotion_batch_preflight_result_consumer_prior_status_unknown")
    safety = _mapping(prior.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("promotion_batch_preflight_result_consumer_prior_read_only_not_true")
    if safety.get("promotion_preflight_plan_result_consumer_only") is not True:
        violations.append("promotion_batch_preflight_result_consumer_prior_only_not_true")
    if safety.get("promotion_batch_preflight_request_template_only") is not True:
        violations.append(
            "promotion_batch_preflight_result_consumer_prior_template_only_not_true"
        )
    for flag in REQUIRED_PRIOR_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"promotion_batch_preflight_result_consumer_prior_{flag}_not_false")
    return tuple(violations)


def _prior_summary_violations(
    summary: Mapping[str, Any],
    *,
    candidate_ids: Tuple[str, ...],
    expected_prefix: str,
    expected_limit: int,
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "preflight_candidate_count": len(candidate_ids),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"promotion_batch_preflight_result_consumer_prior_summary_{key}_mismatch")
    if summary.get("promotion_batch_preflight_request_available") is not bool(candidate_ids):
        violations.append("promotion_batch_preflight_result_consumer_prior_summary_request_available_mismatch")
    if summary.get("operator_must_set_approved_true") is not True:
        violations.append("promotion_batch_preflight_result_consumer_prior_summary_operator_must_set_not_true")
    if summary.get("idempotency_key_prefix") != expected_prefix:
        violations.append("promotion_batch_preflight_result_consumer_prior_summary_prefix_mismatch")
    if _summary_int(summary.get("limit")) != expected_limit:
        violations.append("promotion_batch_preflight_result_consumer_prior_summary_limit_mismatch")
    if _string_list(summary.get("preflight_candidate_proposal_ids")) != candidate_ids:
        violations.append("promotion_batch_preflight_result_consumer_prior_summary_candidate_ids_mismatch")
    for flag in REQUIRED_PRIOR_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"promotion_batch_preflight_result_consumer_prior_summary_{flag}_not_false"
            )
    return tuple(violations)


def _prior_request_violations(
    request: Mapping[str, Any],
    *,
    candidate_ids: Tuple[str, ...],
    expected_prefix: str,
    expected_limit: int,
) -> Tuple[str, ...]:
    violations = []
    if request.get("handler") != "preflight_operator_proposal_apply_batch":
        violations.append("promotion_batch_preflight_result_consumer_prior_request_handler_mismatch")
    if _string_list(request.get("proposal_ids")) != candidate_ids:
        violations.append("promotion_batch_preflight_result_consumer_prior_request_proposal_ids_mismatch")
    if request.get("idempotency_key_prefix") != expected_prefix:
        violations.append("promotion_batch_preflight_result_consumer_prior_request_prefix_mismatch")
    if _summary_int(request.get("limit")) != expected_limit:
        violations.append("promotion_batch_preflight_result_consumer_prior_request_limit_mismatch")
    if request.get("approved") is not False:
        violations.append("promotion_batch_preflight_result_consumer_prior_request_approved_not_false")
    if request.get("operator_must_set_approved_true") is not True:
        violations.append("promotion_batch_preflight_result_consumer_prior_request_operator_must_set_not_true")
    if request.get("read_only_template") is not True:
        violations.append("promotion_batch_preflight_result_consumer_prior_request_template_not_true")
    if request.get("promotion_preflight_plan_result_consumer_only") is not True:
        violations.append("promotion_batch_preflight_result_consumer_prior_request_only_not_true")
    if request.get("promotion_batch_preflight_request_template_only") is not True:
        violations.append(
            "promotion_batch_preflight_result_consumer_prior_request_template_only_not_true"
        )
    if not _mapping(request.get("queue_import_preflight_plan")):
        violations.append("promotion_batch_preflight_result_consumer_prior_request_plan_missing")
    for flag in REQUIRED_REQUEST_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(
                f"promotion_batch_preflight_result_consumer_prior_request_{flag}_not_false"
            )
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer(
    *,
    promotion_preflight_plan_result_consumer: Mapping[str, Any] | None = None,
    promotion_batch_preflight_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify promotion batch-preflight evidence and emit a sequence-plan template."""

    prior = _mapping(promotion_preflight_plan_result_consumer)
    batch = _mapping(promotion_batch_preflight_result)
    prior_request = _mapping(prior.get("promotion_batch_preflight_request"))
    queue_import_preflight_plan = _mapping(prior_request.get("queue_import_preflight_plan"))
    candidate_ids = _string_list(prior_request.get("proposal_ids"))
    expected_prefix = _string(prior_request.get("idempotency_key_prefix")) or _string(
        _mapping(prior.get("summary")).get("idempotency_key_prefix")
    )
    expected_limit = _summary_int(prior_request.get("limit")) or _summary_int(
        _mapping(prior.get("summary")).get("limit")
    ) or 20
    prior_status = _string(prior.get("status"))

    generic_consumer: Mapping[str, Any] = {}
    if prior and batch:
        generic_consumer = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer(
                queue_import_preflight_plan=queue_import_preflight_plan,
                batch_preflight_result=batch,
            )
        )
    generic_status = _string(generic_consumer.get("status"))
    generic_summary = _mapping(generic_consumer.get("summary"))
    generic_request = _mapping(generic_consumer.get("sequence_plan_request"))
    generic_specs = _record_list(generic_consumer.get("sequence_plan_request_specs"))
    generic_invalid_record_ids = tuple(
        f"generic_preflight_result_consumer:{item}"
        for item in _string_list(generic_summary.get("invalid_record_ids"))
    )

    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_string_list(_mapping(prior.get("summary")).get("invalid_record_ids")),
                *_prior_envelope_violations(prior),
                *_prior_summary_violations(
                    _mapping(prior.get("summary")),
                    candidate_ids=candidate_ids,
                    expected_prefix=expected_prefix,
                    expected_limit=expected_limit,
                ),
                *_prior_request_violations(
                    prior_request,
                    candidate_ids=candidate_ids,
                    expected_prefix=expected_prefix,
                    expected_limit=expected_limit,
                ),
                *(
                    generic_invalid_record_ids
                    if generic_status == REQUIRED_GENERIC_REPLACEMENT_STATUS
                    else ()
                ),
                *(
                    ("promotion_batch_preflight_result_consumer_generic_status_unknown",)
                    if (
                        generic_status
                        and generic_status
                        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_RESULT_CONSUMER_STATUSES
                    )
                    else ()
                ),
            )
        )
    )
    invalid_record_ids = () if not prior or not batch else raw_invalid_record_ids

    if not prior or not batch:
        status = "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer_not_ready"
        blocked_by = (
            "promotion_preflight_plan_result_consumer_and_batch_preflight_result_required",
        )
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_promotion_batch_preflight_evidence",
        )
    elif prior_status != REQUIRED_PRIOR_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer_blocked"
        blocked_by = (
            "promotion_preflight_plan_result_consumer_must_request_batch_preflight_first",
        )
    elif generic_status == REQUIRED_GENERIC_BLOCKED_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer_blocked"
        blocked_by = ("all_promotion_batch_preflight_results_must_be_ready",)
    elif generic_status != REQUIRED_GENERIC_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer_not_ready"
        blocked_by = ("ready_promotion_batch_preflight_result_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_sequence_plan_request_available"
        blocked_by = ("operator_may_run_generic_sequence_plan_handler",)

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_sequence_plan_request_available"
    )
    selected_results = (
        _record_list(generic_request.get("preflight_results")) if request_available else ()
    )
    sequence_plan_request: Dict[str, Any] = {
        "handler": "plan_operator_proposal_apply_sequence",
        "preflight_results": list(selected_results),
        "read_only_template": True,
        "promotion_batch_preflight_result_consumer_only": True,
        "promotion_sequence_plan_request_template_only": True,
        "sequence_plan_handler_invoked": False,
        "sequence_plan_created": False,
        "sequence_execution_allowed_by_consumer": False,
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
        sequence_plan_request["promotion_batch_preflight_result_consumer"] = {
            "method_id": (
                OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_BATCH_PREFLIGHT_RESULT_CONSUMER_METHOD_ID
            ),
            "source_prior_method_id": (
                OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PREFLIGHT_PLAN_RESULT_CONSUMER_METHOD_ID
            ),
            "candidate_proposal_ids": list(candidate_ids),
            "idempotency_key_prefix": expected_prefix,
            "read_only": True,
        }

    sequence_plan_request_specs = tuple(
        {
            **dict(spec),
            "promotion_sequence_plan_request_spec_id": (
                f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_BATCH_PREFLIGHT_RESULT_CONSUMER_METHOD_ID}:{spec.get('proposal_id')}"
            ),
            "promotion_batch_preflight_result_consumer_only": True,
            "promotion_sequence_plan_request_template_only": True,
            "sequence_plan_handler_invoked": False,
            "sequence_plan_created": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
        }
        for spec in generic_specs
    ) if request_available else ()

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_BATCH_PREFLIGHT_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PREFLIGHT_PLAN_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_batch_preflight_result_consumer_only": True,
            "promotion_sequence_plan_request_template_only": True,
            "requires_promotion_preflight_plan_result_consumer": True,
            "requires_batch_preflight_result": True,
            "requires_all_preflight_results_ready": True,
            "sequence_plan_handler_invoked": False,
            "sequence_plan_created": False,
            "sequence_execution_allowed_by_consumer": False,
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
            "planned_preflight_candidate_count": len(candidate_ids),
            "batch_preflight_result_count": _summary_int(
                generic_summary.get("batch_preflight_result_count")
            ) or 0,
            "ready_preflight_result_count": _summary_int(
                generic_summary.get("ready_preflight_result_count")
            ) or 0,
            "blocked_preflight_result_count": _summary_int(
                generic_summary.get("blocked_preflight_result_count")
            ) or 0,
            "invalid_record_count": len(invalid_record_ids),
            "idempotency_key_prefix": expected_prefix,
            "promotion_batch_preflight_result_consumer_only": True,
            "promotion_sequence_plan_request_template_only": True,
            "promotion_sequence_plan_request_available": request_available,
            "sequence_plan_handler_invoked": False,
            "sequence_plan_created": False,
            "sequence_execution_allowed_by_consumer": False,
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
            "planned_preflight_candidate_proposal_ids": list(candidate_ids),
            "batch_preflight_result_proposal_ids": list(
                _string_list(
                    generic_summary.get("batch_preflight_result_proposal_ids")
                )
            ),
            "ready_preflight_result_proposal_ids": list(
                _string_list(generic_summary.get("ready_preflight_result_proposal_ids"))
            ),
            "blocked_preflight_result_proposal_ids": list(
                _string_list(
                    generic_summary.get("blocked_preflight_result_proposal_ids")
                )
            ),
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion preflight-plan result consumer and batch preflight result."
                if not prior or not batch
                else (
                    "Replace invalid promotion batch-preflight evidence before sequence planning."
                    if invalid_record_ids
                    else (
                        "Re-run batch preflight until every promotion imported candidate is ready."
                        if generic_status == REQUIRED_GENERIC_BLOCKED_STATUS
                        else "Operator may run the generic sequence-plan handler; this consumer does not run it."
                    )
                )
            ),
        },
        "promotion_sequence_plan_request": sequence_plan_request,
        "promotion_sequence_plan_request_specs": list(sequence_plan_request_specs),
        "generic_preflight_result_consumer": generic_consumer if prior and batch else {},
    }
