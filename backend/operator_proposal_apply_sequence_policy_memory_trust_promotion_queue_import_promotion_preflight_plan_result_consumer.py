"""Read-only consumer for promotion queue-import preflight-plan results.

This verifier connects the promotion queue-importer result consumer to the
existing batch preflight handler. It verifies that a queue-import preflight plan
matches the applied queue-importer provenance, then emits only a batch-preflight
request template. It does not run batch preflight, reserve idempotency, mutate
queue/proposal state, create sequence plans, or touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_IMPORTER_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_IMPORTER_RESULT_CONSUMER_STATUSES,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_IMPORTER_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_IMPORTER_RESULT_CONSUMER_STATUSES,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PREFLIGHT_PLAN_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PREFLIGHT_PLAN_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_request_available",
}
REQUIRED_QUEUE_IMPORTER_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer"
)
REQUIRED_QUEUE_IMPORTER_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_queue_import_preflight_plan_request_available"
)
REQUIRED_PREFLIGHT_PLAN_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan"
)
REQUIRED_PREFLIGHT_PLAN_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_preflight_plan_available"
)
REQUIRED_PREFLIGHT_PLAN_REPLACEMENT_STATUS = (
    "sequence_memory_trust_promotion_queue_import_preflight_plan_replacement_required"
)
REQUIRED_PRIOR_FALSE_FLAGS = (
    "promotion_queue_import_preflight_plan_handler_invoked",
    "promotion_queue_import_preflight_plan_created",
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
    "can_import_to_queue",
    "can_apply_live",
)
REQUIRED_PLAN_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "sequence_plan_created",
    "sequence_execution_allowed_by_preflight_plan",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_apply_live",
)
REQUIRED_REQUEST_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "idempotency_reserved",
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


def _preflight_plan_request_from_prior(
    consumer: Mapping[str, Any],
) -> Mapping[str, Any]:
    request = _mapping(consumer.get("promotion_queue_import_preflight_plan_request"))
    if request:
        return request
    return _mapping(consumer.get("promotion_preflight_plan_request"))


def _expected_imported_ids(
    consumer: Mapping[str, Any],
    request: Mapping[str, Any],
) -> Tuple[str, ...]:
    request_ids = _string_list(request.get("imported_proposal_ids"))
    if request_ids:
        return request_ids
    return _string_list(_mapping(consumer.get("summary")).get("imported_proposal_ids"))


def _candidate_ids_from_plan(plan: Mapping[str, Any]) -> Tuple[str, ...]:
    return _string_list(_mapping(plan.get("summary")).get("preflight_candidate_proposal_ids"))


def _prior_consumer_envelope_violations(
    consumer: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_QUEUE_IMPORTER_RESULT_CONSUMER_TYPE:
        violations.append("promotion_preflight_plan_result_consumer_prior_type_mismatch")
    generated_at = consumer.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("promotion_preflight_plan_result_consumer_prior_generated_at_invalid")
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_IMPORTER_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append("promotion_preflight_plan_result_consumer_prior_method_id_mismatch")
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_IMPORTER_RESULT_CONSUMER_STATUSES
    ):
        violations.append("promotion_preflight_plan_result_consumer_prior_status_unknown")
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("promotion_preflight_plan_result_consumer_prior_read_only_not_true")
    if safety.get("promotion_queue_importer_result_consumer_only") is not True:
        violations.append("promotion_preflight_plan_result_consumer_prior_only_not_true")
    if safety.get("promotion_queue_import_preflight_plan_request_template_only") is not True:
        violations.append(
            "promotion_preflight_plan_result_consumer_prior_template_only_not_true"
        )
    for flag in REQUIRED_PRIOR_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"promotion_preflight_plan_result_consumer_prior_{flag}_not_false")
    return tuple(violations)


def _prior_consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_imported_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if _summary_int(summary.get("imported_proposal_count")) != len(expected_imported_ids):
        violations.append("promotion_preflight_plan_result_consumer_prior_summary_imported_count_mismatch")
    if _summary_int(summary.get("invalid_record_count")) != 0:
        violations.append("promotion_preflight_plan_result_consumer_prior_summary_invalid_count_mismatch")
    if summary.get("promotion_queue_import_preflight_plan_request_available") is not bool(expected_imported_ids):
        violations.append("promotion_preflight_plan_result_consumer_prior_summary_request_available_mismatch")
    if summary.get("promotion_queue_import_preflight_plan_handler_invoked") is not False:
        violations.append("promotion_preflight_plan_result_consumer_prior_summary_handler_invoked_not_false")
    if summary.get("promotion_queue_import_preflight_plan_created") is not False:
        violations.append("promotion_preflight_plan_result_consumer_prior_summary_created_not_false")
    if summary.get("requires_applied_queue_import_result") is not True:
        violations.append("promotion_preflight_plan_result_consumer_prior_summary_requires_applied_not_true")
    for flag in REQUIRED_PRIOR_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"promotion_preflight_plan_result_consumer_prior_summary_{flag}_not_false"
            )
    if _string_list(summary.get("imported_proposal_ids")) != expected_imported_ids:
        violations.append("promotion_preflight_plan_result_consumer_prior_summary_imported_ids_mismatch")
    return tuple(violations)


def _prior_request_violations(
    request: Mapping[str, Any],
    *,
    expected_imported_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if request.get("handler") != "plan_operator_proposal_apply_sequence_policy_queue_import_preflight":
        violations.append("promotion_preflight_plan_result_consumer_prior_request_handler_mismatch")
    if request.get("read_only_template") is not True:
        violations.append("promotion_preflight_plan_result_consumer_prior_request_template_not_true")
    if request.get("promotion_queue_importer_result_consumer_only") is not True:
        violations.append("promotion_preflight_plan_result_consumer_prior_request_only_not_true")
    if request.get("promotion_queue_import_preflight_plan_request_template_only") is not True:
        violations.append("promotion_preflight_plan_result_consumer_prior_request_template_only_not_true")
    if request.get("promotion_queue_import_preflight_plan_handler_invoked") is not False:
        violations.append("promotion_preflight_plan_result_consumer_prior_request_handler_invoked_not_false")
    if request.get("promotion_queue_import_preflight_plan_created") is not False:
        violations.append("promotion_preflight_plan_result_consumer_prior_request_created_not_false")
    requires_applied = (
        request.get("requires_applied_queue_importer_result") is True
        or request.get("requires_applied_queue_import_result") is True
    )
    if not requires_applied:
        violations.append("promotion_preflight_plan_result_consumer_prior_request_requires_applied_not_true")
    if request.get("pending_actions_required_from_operator_queue") is not True:
        violations.append("promotion_preflight_plan_result_consumer_prior_request_pending_actions_not_true")
    if _string_list(request.get("imported_proposal_ids")) != expected_imported_ids:
        violations.append("promotion_preflight_plan_result_consumer_prior_request_imported_ids_mismatch")
    for flag in REQUIRED_PRIOR_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(
                f"promotion_preflight_plan_result_consumer_prior_request_{flag}_not_false"
            )
    return tuple(violations)


def _preflight_plan_envelope_violations(plan: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if plan.get("type") != REQUIRED_PREFLIGHT_PLAN_TYPE:
        violations.append("promotion_preflight_plan_result_consumer_plan_type_mismatch")
    generated_at = plan.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("promotion_preflight_plan_result_consumer_plan_generated_at_invalid")
    if (
        plan.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_METHOD_ID
    ):
        violations.append("promotion_preflight_plan_result_consumer_plan_method_id_mismatch")
    if (
        plan.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_STATUSES
    ):
        violations.append("promotion_preflight_plan_result_consumer_plan_status_unknown")
    safety = _mapping(plan.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("promotion_preflight_plan_result_consumer_plan_read_only_not_true")
    if safety.get("batch_preflight_only") is not True:
        violations.append("promotion_preflight_plan_result_consumer_plan_batch_only_not_true")
    if safety.get("batch_preflight_invoked") is not False:
        violations.append("promotion_preflight_plan_result_consumer_plan_batch_invoked_not_false")
    if safety.get("operator_approval_required_for_preflight_ready") is not True:
        violations.append("promotion_preflight_plan_result_consumer_plan_operator_approval_not_true")
    if safety.get("sequence_consumer_required_before_execution") is not True:
        violations.append("promotion_preflight_plan_result_consumer_plan_sequence_consumer_not_true")
    for flag in REQUIRED_PLAN_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"promotion_preflight_plan_result_consumer_plan_{flag}_not_false")
    return tuple(violations)


def _preflight_plan_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_imported_ids: Tuple[str, ...],
    candidate_ids: Tuple[str, ...],
    expected_prefix: str,
    expected_limit: int,
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "sequence_policy_imported_pending_count": len(expected_imported_ids),
        "preflight_candidate_count": len(candidate_ids),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"promotion_preflight_plan_result_consumer_plan_summary_{key}_mismatch")
    if _summary_int(summary.get("limit")) != expected_limit:
        violations.append("promotion_preflight_plan_result_consumer_plan_summary_limit_mismatch")
    if summary.get("idempotency_key_prefix") != expected_prefix:
        violations.append("promotion_preflight_plan_result_consumer_plan_summary_prefix_mismatch")
    if summary.get("operator_approval_required_for_preflight_ready") is not True:
        violations.append("promotion_preflight_plan_result_consumer_plan_summary_operator_approval_not_true")
    if summary.get("sequence_consumer_required_before_execution") is not True:
        violations.append("promotion_preflight_plan_result_consumer_plan_summary_sequence_consumer_not_true")
    if summary.get("batch_preflight_request_available") is not bool(candidate_ids):
        violations.append("promotion_preflight_plan_result_consumer_plan_summary_request_available_mismatch")
    if summary.get("batch_preflight_invoked") is not False:
        violations.append("promotion_preflight_plan_result_consumer_plan_summary_batch_invoked_not_false")
    for flag in REQUIRED_PLAN_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"promotion_preflight_plan_result_consumer_plan_summary_{flag}_not_false"
            )
    if _string_list(summary.get("imported_proposal_ids")) != expected_imported_ids:
        violations.append("promotion_preflight_plan_result_consumer_plan_summary_imported_ids_mismatch")
    if _string_list(summary.get("preflight_candidate_proposal_ids")) != candidate_ids:
        violations.append("promotion_preflight_plan_result_consumer_plan_summary_candidate_ids_mismatch")
    return tuple(violations)


def _preflight_specs_violations(
    specs: Tuple[Mapping[str, Any], ...],
    *,
    candidate_ids: Tuple[str, ...],
    expected_prefix: str,
) -> Tuple[str, ...]:
    violations = []
    spec_ids = tuple(_string(spec.get("proposal_id")) for spec in specs)
    if spec_ids != candidate_ids:
        violations.append("promotion_preflight_plan_result_consumer_specs_candidate_ids_mismatch")
    for index, spec in enumerate(specs):
        record_id = _string(spec.get("preflight_request_spec_id")) or f"preflight_spec:{index}"
        proposal_id = _string(spec.get("proposal_id"))
        if spec.get("planned_preflight_index") != index:
            violations.append(f"{record_id}:planned_preflight_index_mismatch")
        if spec.get("idempotency_key") != f"{expected_prefix}:{proposal_id}":
            violations.append(f"{record_id}:idempotency_key_mismatch")
        for flag in (
            "operator_approval_required_for_preflight_ready",
            "sequence_consumer_required_before_execution",
            "batch_preflight_only",
            "read_only",
        ):
            if spec.get(flag) is not True:
                violations.append(f"{record_id}:{flag}_not_true")
        for flag in REQUIRED_PLAN_FALSE_FLAGS:
            if spec.get(flag) is not False:
                violations.append(f"{record_id}:{flag}_not_false")
    return tuple(violations)


def _batch_request_violations(
    request: Mapping[str, Any],
    *,
    candidate_ids: Tuple[str, ...],
    expected_prefix: str,
    expected_limit: int,
) -> Tuple[str, ...]:
    violations = []
    if request.get("handler") != "preflight_operator_proposal_apply_batch":
        violations.append("promotion_preflight_plan_result_consumer_batch_request_handler_mismatch")
    if _string_list(request.get("proposal_ids")) != candidate_ids:
        violations.append("promotion_preflight_plan_result_consumer_batch_request_proposal_ids_mismatch")
    if request.get("idempotency_key_prefix") != expected_prefix:
        violations.append("promotion_preflight_plan_result_consumer_batch_request_prefix_mismatch")
    if _summary_int(request.get("limit")) != expected_limit:
        violations.append("promotion_preflight_plan_result_consumer_batch_request_limit_mismatch")
    if request.get("approved") is not False:
        violations.append("promotion_preflight_plan_result_consumer_batch_request_approved_not_false")
    if request.get("operator_must_set_approved_true") is not True:
        violations.append("promotion_preflight_plan_result_consumer_batch_request_operator_must_set_not_true")
    if request.get("read_only_template") is not True:
        violations.append("promotion_preflight_plan_result_consumer_batch_request_template_not_true")
    for flag in REQUIRED_REQUEST_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(f"promotion_preflight_plan_result_consumer_batch_request_{flag}_not_false")
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer(
    *,
    promotion_queue_importer_result_consumer: Mapping[str, Any] | None = None,
    promotion_queue_import_preflight_plan_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify preflight-plan evidence and expose only a batch-preflight template."""

    prior = _mapping(promotion_queue_importer_result_consumer)
    plan = _mapping(promotion_queue_import_preflight_plan_result)
    prior_request = _preflight_plan_request_from_prior(prior)
    expected_imported_ids = _expected_imported_ids(prior, prior_request)
    candidate_ids = _candidate_ids_from_plan(plan)
    expected_prefix = str(prior_request.get("idempotency_key_prefix") or "sequence-policy-queue-import")
    expected_limit = _summary_int(prior_request.get("limit")) or 20
    preflight_specs = _record_list(plan.get("preflight_request_specs"))
    batch_request = _mapping(plan.get("batch_preflight_request"))
    prior_status = _string(prior.get("status"))
    plan_status = _string(plan.get("status"))

    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_string_list(_mapping(prior.get("summary")).get("invalid_record_ids")),
                *_string_list(_mapping(plan.get("summary")).get("invalid_record_ids")),
                *_prior_consumer_envelope_violations(prior),
                *_prior_consumer_summary_violations(
                    _mapping(prior.get("summary")),
                    expected_imported_ids=expected_imported_ids,
                ),
                *_prior_request_violations(
                    prior_request,
                    expected_imported_ids=expected_imported_ids,
                ),
                *_preflight_plan_envelope_violations(plan),
                *(
                    ("promotion_preflight_plan_result_consumer_plan_replacement_required",)
                    if plan_status == REQUIRED_PREFLIGHT_PLAN_REPLACEMENT_STATUS
                    else ()
                ),
                *_preflight_plan_summary_violations(
                    _mapping(plan.get("summary")),
                    expected_imported_ids=expected_imported_ids,
                    candidate_ids=candidate_ids,
                    expected_prefix=expected_prefix,
                    expected_limit=expected_limit,
                ),
                *_preflight_specs_violations(
                    preflight_specs,
                    candidate_ids=candidate_ids,
                    expected_prefix=expected_prefix,
                ),
                *_batch_request_violations(
                    batch_request,
                    candidate_ids=candidate_ids,
                    expected_prefix=expected_prefix,
                    expected_limit=expected_limit,
                ),
            )
        )
    )
    invalid_record_ids = () if not prior or not plan else raw_invalid_record_ids

    if not prior or not plan:
        status = "sequence_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer_not_ready"
        blocked_by = (
            "promotion_queue_importer_result_consumer_and_preflight_plan_result_required",
        )
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_policy_preflight_plan_evidence",
        )
    elif prior_status != REQUIRED_QUEUE_IMPORTER_RESULT_CONSUMER_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer_blocked"
        blocked_by = (
            "promotion_queue_importer_result_consumer_must_request_preflight_plan_first",
        )
    elif plan_status != REQUIRED_PREFLIGHT_PLAN_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer_blocked"
        blocked_by = ("sequence_policy_queue_import_preflight_plan_required_before_batch_preflight",)
    elif not candidate_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer_blocked"
        blocked_by = ("sequence_policy_queue_import_preflight_candidates_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_request_available"
        blocked_by = ("operator_may_run_batch_preflight_with_explicit_approval",)

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_request_available"
    )
    promotion_batch_preflight_request: Dict[str, Any] = {
        "handler": "preflight_operator_proposal_apply_batch",
        "proposal_ids": list(candidate_ids) if request_available else [],
        "idempotency_key_prefix": expected_prefix,
        "limit": expected_limit,
        "approved": False,
        "operator_must_set_approved_true": True,
        "read_only_template": True,
        "promotion_preflight_plan_result_consumer_only": True,
        "promotion_batch_preflight_request_template_only": True,
        "promotion_batch_preflight_handler_invoked": False,
        "promotion_batch_preflight_created": False,
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
        promotion_batch_preflight_request["queue_import_preflight_plan"] = plan

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PREFLIGHT_PLAN_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_IMPORTER_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_preflight_plan_result_consumer_only": True,
            "promotion_batch_preflight_request_template_only": True,
            "requires_promotion_queue_importer_result_consumer": True,
            "requires_queue_import_preflight_plan_result": True,
            "promotion_batch_preflight_handler_invoked": False,
            "promotion_batch_preflight_created": False,
            "batch_preflight_invoked": False,
            "sequence_plan_created": False,
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
            "expected_imported_proposal_count": len(expected_imported_ids),
            "preflight_candidate_count": len(candidate_ids),
            "preflight_request_spec_count": len(preflight_specs),
            "invalid_record_count": len(invalid_record_ids),
            "promotion_preflight_plan_result_consumer_only": True,
            "promotion_batch_preflight_request_template_only": True,
            "promotion_batch_preflight_request_available": request_available,
            "promotion_batch_preflight_handler_invoked": False,
            "promotion_batch_preflight_created": False,
            "batch_preflight_invoked": False,
            "sequence_plan_created": False,
            "operator_must_set_approved_true": True,
            "idempotency_key_prefix": expected_prefix,
            "limit": expected_limit,
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
            "imported_proposal_ids": list(expected_imported_ids),
            "preflight_candidate_proposal_ids": list(candidate_ids),
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion queue-importer result consumer and preflight-plan result."
                if not prior or not plan
                else (
                    "Replace invalid sequence-policy queue-import preflight-plan evidence before batch preflight."
                    if invalid_record_ids
                    else (
                        "Run the read-only preflight-plan request before batch preflight."
                        if plan_status != REQUIRED_PREFLIGHT_PLAN_AVAILABLE_STATUS
                        else "Run batch preflight with explicit operator approval; this consumer does not run preflight or reserve idempotency."
                    )
                )
            ),
        },
        "promotion_batch_preflight_request": promotion_batch_preflight_request,
    }
