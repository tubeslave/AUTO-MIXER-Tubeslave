"""Read-only consumer for promotion sequence-execution audit results.

This verifier connects the promotion execution-result consumer to the generic
audit-policy gate. It verifies promotion provenance, delegates audit-result
validation to the existing queue-import execution-audit result consumer, then
emits only an audit-policy-gate request template. It does not invoke policy
gates, audit handlers, apply handlers, reserve idempotency, mutate policy or
memory stores, or touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_AUDIT_RESULT_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_AUDIT_RESULT_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_AUDIT_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_AUDIT_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_audit_policy_gate_request_available",
}
REQUIRED_PRIOR_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer"
)
REQUIRED_PRIOR_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_execution_audit_request_available"
)
REQUIRED_GENERIC_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_audit_policy_gate_request_available"
)
REQUIRED_GENERIC_REPLACEMENT_STATUS = (
    "sequence_memory_trust_promotion_queue_import_execution_audit_result_consumer_replacement_required"
)
REQUIRED_GENERIC_BLOCKED_STATUS = (
    "sequence_memory_trust_promotion_queue_import_execution_audit_result_consumer_blocked"
)
REQUIRED_PRIOR_FALSE_FLAGS = (
    "execution_audit_handler_invoked",
    "execution_audit_created",
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
    "execution_audit_handler_invoked",
    "execution_audit_created",
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
REQUIRED_ACCEPTANCE_TEMPLATE_FIELDS = (
    "promotion_acceptance_record_template_id",
    "promotion_contract_record_id",
    "promotion_candidate_id",
    "proposal_id",
    "learning_scope",
    "value_type",
    "idempotency_key",
    "previous_policy_state",
    "rollback_strategy",
)
REQUIRED_ACCEPTANCE_TEMPLATE_FALSE_FLAGS = (
    "execution_allowed_by_acceptance",
    "sequence_policy_promotion_allowed_by_acceptance",
    "queue_mutation_allowed_by_acceptance",
    "policy_mutation_allowed_by_acceptance",
    "model_mutation_allowed_by_acceptance",
    "runtime_memory_mutation_allowed_by_acceptance",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
)
REQUIRED_SEQUENCE_REVIEW_FRESHNESS_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "queries_console",
    "queue_mutation",
    "policy_mutation",
)
REQUIRED_ACCEPTANCE_TEMPLATE_IDENTITY_FIELDS = (
    "source_acceptance_record_template_id",
    "source_review_candidate_id",
    "source_memory_record_id",
    "source_memory_trust_review_request_spec_id",
    "source_memory_index_request_spec_id",
    "source_memory_writer_apply_request_spec_id",
    "source_memory_writer_request_spec_id",
    "source_memory_draft_request_spec_id",
    "source_audit_policy_gate_request_spec_id",
    "source_audit_request_spec_id",
    "source_executor_request_spec_id",
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


def _context_status(payload: Mapping[str, Any]) -> str:
    return _string(payload.get("status"))


def _context_violations(prefix: str, payload: Mapping[str, Any]) -> Tuple[str, ...]:
    if not payload:
        return (f"{prefix}_current_sequence_review_freshness_missing",)
    violations = []
    if _context_status(payload) != "fresh":
        violations.append(f"{prefix}_current_sequence_review_freshness_not_fresh")
    if payload.get("read_only") is not True:
        violations.append(f"{prefix}_current_sequence_review_freshness_read_only_not_true")
    for flag in REQUIRED_SEQUENCE_REVIEW_FRESHNESS_FALSE_FLAGS:
        if payload.get(flag) is not False:
            violations.append(
                f"{prefix}_current_sequence_review_freshness_{flag}_not_false"
            )
    return tuple(violations)


def _sequence_review_freshness_core(
    payload: Mapping[str, Any],
) -> Tuple[Tuple[str, Any], ...]:
    return (
        ("status", _context_status(payload)),
        ("read_only", payload.get("read_only")),
        *(
            (flag, payload.get(flag))
            for flag in REQUIRED_SEQUENCE_REVIEW_FRESHNESS_FALSE_FLAGS
        ),
    )


def _source_sequence_review_freshness_status(
    templates: Tuple[Mapping[str, Any], ...],
) -> str:
    if not templates:
        return "missing"
    invalid = False
    missing = False
    for index, template in enumerate(templates):
        template_id = (
            _string(template.get("promotion_acceptance_record_template_id"))
            or f"promotion_acceptance_template:{index}"
        )
        proposal_id = _string(template.get("proposal_id")) or template_id
        freshness = _mapping(template.get("current_sequence_review_freshness"))
        violations = _context_violations(proposal_id, freshness)
        if not freshness:
            missing = True
        elif violations:
            invalid = True
    if not missing and not invalid:
        return "fresh"
    return "invalid" if invalid else "missing"


def _acceptance_templates_from_prior_request(
    request: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    return _record_list(request.get("promotion_acceptance_record_templates"))


def _request_specs_from_prior_request(
    request: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    return _record_list(request.get("memory_trust_review_request_specs"))


def _request_specs_by_proposal_id(
    request_specs: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(spec["proposal_id"]): spec
        for spec in request_specs
        if isinstance(spec.get("proposal_id"), str) and spec.get("proposal_id")
    }


def _prior_envelope_violations(prior: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if prior.get("type") != REQUIRED_PRIOR_TYPE:
        violations.append("promotion_execution_audit_result_consumer_prior_type_mismatch")
    generated_at = prior.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("promotion_execution_audit_result_consumer_prior_generated_at_invalid")
    if (
        prior.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append("promotion_execution_audit_result_consumer_prior_method_id_mismatch")
    if (
        prior.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_RESULT_CONSUMER_STATUSES
    ):
        violations.append("promotion_execution_audit_result_consumer_prior_status_unknown")
    safety = _mapping(prior.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("promotion_execution_audit_result_consumer_prior_read_only_not_true")
    if safety.get("promotion_execution_result_consumer_only") is not True:
        violations.append("promotion_execution_audit_result_consumer_prior_only_not_true")
    if safety.get("promotion_execution_audit_request_template_only") is not True:
        violations.append("promotion_execution_audit_result_consumer_prior_template_only_not_true")
    for flag in REQUIRED_PRIOR_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"promotion_execution_audit_result_consumer_prior_{flag}_not_false"
            )
    return tuple(violations)


def _prior_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
    expected_prefix: str,
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
    memory_trust_review_request_specs: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    if summary.get("promotion_execution_audit_request_available") is not bool(expected_ids):
        violations.append("promotion_execution_audit_result_consumer_prior_summary_request_available_mismatch")
    if _summary_int(summary.get("promotion_acceptance_record_template_count")) != len(
        promotion_acceptance_templates
    ):
        violations.append(
            "promotion_execution_audit_result_consumer_prior_summary_template_count_mismatch"
        )
    if _summary_int(summary.get("memory_trust_review_request_spec_count")) != len(
        memory_trust_review_request_specs
    ):
        violations.append(
            "promotion_execution_audit_result_consumer_prior_summary_review_request_spec_count_mismatch"
        )
    expected_counts = {
        "expected_execution_result_count": len(expected_ids),
        "execution_result_count": len(expected_ids),
        "applied_execution_result_count": len(expected_ids),
        "blocked_execution_result_count": 0,
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"promotion_execution_audit_result_consumer_prior_summary_{key}_mismatch")
    if summary.get("idempotency_key_prefix") != expected_prefix:
        violations.append("promotion_execution_audit_result_consumer_prior_summary_prefix_mismatch")
    if _string_list(summary.get("expected_execution_proposal_ids")) != expected_ids:
        violations.append("promotion_execution_audit_result_consumer_prior_summary_expected_ids_mismatch")
    if _string_list(summary.get("execution_result_proposal_ids")) != expected_ids:
        violations.append("promotion_execution_audit_result_consumer_prior_summary_result_ids_mismatch")
    if _string_list(summary.get("applied_execution_proposal_ids")) != expected_ids:
        violations.append("promotion_execution_audit_result_consumer_prior_summary_applied_ids_mismatch")
    if (
        promotion_acceptance_templates
        and summary.get("fresh_sequence_review_context_status") != "fresh"
    ):
        violations.append(
            "promotion_execution_audit_result_consumer_prior_summary_freshness_not_fresh"
        )
    expected_template_proposal_ids = tuple(
        str(template["proposal_id"])
        for template in promotion_acceptance_templates
        if isinstance(template.get("proposal_id"), str) and template.get("proposal_id")
    )
    if (
        _string_list(summary.get("promotion_acceptance_record_template_proposal_ids"))
        != expected_template_proposal_ids
    ):
        violations.append(
            "promotion_execution_audit_result_consumer_prior_summary_template_proposal_ids_mismatch"
        )
    expected_request_spec_proposal_ids = tuple(
        str(spec["proposal_id"])
        for spec in memory_trust_review_request_specs
        if isinstance(spec.get("proposal_id"), str) and spec.get("proposal_id")
    )
    if (
        _string_list(summary.get("memory_trust_review_request_spec_proposal_ids"))
        != expected_request_spec_proposal_ids
    ):
        violations.append(
            "promotion_execution_audit_result_consumer_prior_summary_review_request_spec_proposal_ids_mismatch"
        )
    for flag in REQUIRED_PRIOR_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"promotion_execution_audit_result_consumer_prior_summary_{flag}_not_false"
            )
    return tuple(violations)


def _prior_request_violations(
    request: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
    memory_trust_review_request_specs: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    if request.get("handler") != "audit_operator_proposal_apply_sequence_execution":
        violations.append("promotion_execution_audit_result_consumer_prior_request_handler_mismatch")
    if request.get("read_only_template") is not True:
        violations.append("promotion_execution_audit_result_consumer_prior_request_template_not_true")
    if request.get("promotion_execution_result_consumer_only") is not True:
        violations.append("promotion_execution_audit_result_consumer_prior_request_only_not_true")
    if request.get("promotion_execution_audit_request_template_only") is not True:
        violations.append("promotion_execution_audit_result_consumer_prior_request_template_only_not_true")
    payload = _mapping(request.get("execution_result_payload"))
    if len(_record_list(payload.get("execution_results"))) != len(expected_ids):
        violations.append("promotion_execution_audit_result_consumer_prior_request_execution_count_mismatch")
    for flag in REQUIRED_REQUEST_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(
                f"promotion_execution_audit_result_consumer_prior_request_{flag}_not_false"
            )
    template_proposal_ids = {
        str(template["proposal_id"])
        for template in promotion_acceptance_templates
        if isinstance(template.get("proposal_id"), str) and template.get("proposal_id")
    }
    if len(template_proposal_ids) != len(promotion_acceptance_templates):
        violations.append(
            "promotion_execution_audit_result_consumer_prior_request_acceptance_template_proposal_ids_not_unique"
        )
    if expected_ids and not set(expected_ids).issubset(template_proposal_ids):
        violations.append(
            "promotion_execution_audit_result_consumer_prior_request_acceptance_templates_missing_execution_result"
        )
    request_spec_proposal_ids = {
        str(spec["proposal_id"])
        for spec in memory_trust_review_request_specs
        if isinstance(spec.get("proposal_id"), str) and spec.get("proposal_id")
    }
    if not template_proposal_ids.issubset(request_spec_proposal_ids):
        violations.append(
            "promotion_execution_audit_result_consumer_prior_request_review_request_spec_proposal_ids_mismatch"
        )
    for index, spec in enumerate(memory_trust_review_request_specs):
        spec_id = _string(spec.get("memory_trust_review_request_spec_id")) or (
            f"memory_trust_review_request_spec:{index}"
        )
        proposal_id = _string(spec.get("proposal_id")) or spec_id
        violations.extend(
            _context_violations(
                proposal_id,
                _mapping(spec.get("current_sequence_review_freshness")),
            )
        )
    return tuple(violations)


def _acceptance_template_violations(
    template: Mapping[str, Any],
    *,
    index: int,
) -> Tuple[str, ...]:
    template_id = (
        _string(template.get("promotion_acceptance_record_template_id"))
        or f"promotion_acceptance_template:{index}"
    )
    proposal_id = _string(template.get("proposal_id")) or template_id
    violations = []
    for field in REQUIRED_ACCEPTANCE_TEMPLATE_FIELDS:
        if not _string(template.get(field)):
            violations.append(f"{template_id}:{field}_missing")
    for field in REQUIRED_ACCEPTANCE_TEMPLATE_IDENTITY_FIELDS:
        if not _string(template.get(field)):
            violations.append(f"{proposal_id}:{field}_missing")
    if template.get("source_memory_index_available") is not True:
        violations.append(f"{proposal_id}:source_memory_index_available_not_true")
    if _summary_int(template.get("planned_queue_index")) is None:
        violations.append(f"{proposal_id}:planned_queue_index_invalid")
    if template.get("rank_source") != "sequence_policy_consumer":
        violations.append(f"{proposal_id}:rank_source_mismatch")
    violations.extend(
        _context_violations(
            proposal_id,
            _mapping(template.get("current_sequence_review_freshness")),
        )
    )
    source_idempotency_key = _string(template.get("source_idempotency_key"))
    policy_key = _string(template.get("policy_key"))
    if (
        source_idempotency_key
        and policy_key
        and policy_key != f"{proposal_id}/{source_idempotency_key}"
    ):
        violations.append(f"{proposal_id}:policy_key_identity_mismatch")
    if template.get("decision_options") != ["accepted", "rejected"]:
        violations.append(f"{template_id}:decision_options_invalid")
    if template.get("operator_must_choose_decision") is not True:
        violations.append(f"{template_id}:operator_must_choose_decision_not_true")
    if template.get("reviewed_by_required") is not True:
        violations.append(f"{template_id}:reviewed_by_required_not_true")
    if template.get("reviewed_at_utc_required") is not True:
        violations.append(f"{template_id}:reviewed_at_utc_required_not_true")
    for flag in REQUIRED_ACCEPTANCE_TEMPLATE_FALSE_FLAGS:
        if template.get(flag) is not False:
            violations.append(f"{template_id}:{flag}_not_false")
    return tuple(violations)


def _request_spec_template_identity_violations(
    *,
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
    memory_trust_review_request_specs: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    request_specs_by_proposal_id = _request_specs_by_proposal_id(
        memory_trust_review_request_specs
    )
    violations = []
    for template in promotion_acceptance_templates:
        proposal_id = _string(template.get("proposal_id"))
        request_spec = _mapping(request_specs_by_proposal_id.get(proposal_id))
        if not request_spec:
            violations.append(f"{proposal_id}:memory_trust_review_request_spec_missing")
            continue
        if template.get("source_memory_trust_review_request_spec_id") != request_spec.get(
            "memory_trust_review_request_spec_id"
        ):
            violations.append(
                f"{proposal_id}:source_memory_trust_review_request_spec_id_mismatch"
            )
        if _sequence_review_freshness_core(
            _mapping(template.get("current_sequence_review_freshness"))
        ) != _sequence_review_freshness_core(
            _mapping(request_spec.get("current_sequence_review_freshness"))
        ):
            violations.append(
                f"{proposal_id}:current_sequence_review_freshness_request_spec_mismatch"
            )
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer(
    *,
    promotion_execution_result_consumer: Mapping[str, Any] | None = None,
    promotion_execution_audit_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify promotion execution-audit evidence and emit policy-gate template."""

    prior = _mapping(promotion_execution_result_consumer)
    audit = _mapping(promotion_execution_audit_result)
    prior_summary = _mapping(prior.get("summary"))
    prior_request = _mapping(prior.get("promotion_execution_audit_request"))
    promotion_acceptance_templates = _acceptance_templates_from_prior_request(
        prior_request
    )
    memory_trust_review_request_specs = _request_specs_from_prior_request(prior_request)
    prior_summary_ids = _string_list(prior_summary.get("execution_result_proposal_ids"))
    prior_summary_prefix = _string(prior_summary.get("idempotency_key_prefix"))
    prior_status = _string(prior.get("status"))
    generic_execution_result_consumer = _mapping(
        prior.get("generic_execution_result_consumer")
    )

    generic_consumer: Mapping[str, Any] = {}
    if prior and audit:
        generic_consumer = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer(
                execution_result_consumer=generic_execution_result_consumer,
                execution_audit_result=audit,
            )
        )
    generic_status = _string(generic_consumer.get("status"))
    generic_summary = _mapping(generic_consumer.get("summary"))
    generic_request = _mapping(generic_consumer.get("audit_policy_gate_request"))
    generic_specs = _record_list(generic_consumer.get("audit_policy_gate_request_specs"))
    expected_ids = (
        _string_list(generic_summary.get("expected_execution_audit_proposal_ids"))
        or prior_summary_ids
    )
    expected_prefix = (
        _string(generic_summary.get("idempotency_key_prefix"))
        or prior_summary_prefix
    )
    generic_invalid_record_ids = tuple(
        f"generic_execution_audit_result_consumer:{item}"
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
                    promotion_acceptance_templates=promotion_acceptance_templates,
                    memory_trust_review_request_specs=memory_trust_review_request_specs,
                ),
                *_prior_request_violations(
                    prior_request,
                    expected_ids=expected_ids,
                    promotion_acceptance_templates=promotion_acceptance_templates,
                    memory_trust_review_request_specs=memory_trust_review_request_specs,
                ),
                *(
                    violation
                    for index, template in enumerate(promotion_acceptance_templates)
                    for violation in _acceptance_template_violations(
                        template,
                        index=index,
                    )
                ),
                *_request_spec_template_identity_violations(
                    promotion_acceptance_templates=promotion_acceptance_templates,
                    memory_trust_review_request_specs=memory_trust_review_request_specs,
                ),
                *(
                    generic_invalid_record_ids
                    if generic_status == REQUIRED_GENERIC_REPLACEMENT_STATUS
                    else ()
                ),
                *(
                    ("promotion_execution_audit_result_consumer_generic_status_unknown",)
                    if (
                        generic_status
                        and generic_status
                        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_AUDIT_RESULT_CONSUMER_STATUSES
                    )
                    else ()
                ),
            )
        )
    )
    invalid_record_ids = () if not prior or not audit else raw_invalid_record_ids

    if not prior or not audit:
        status = "sequence_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer_not_ready"
        blocked_by = (
            "promotion_execution_result_consumer_and_execution_audit_result_required",
        )
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_promotion_execution_audit_evidence",
        )
    elif prior_status != REQUIRED_PRIOR_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer_blocked"
        blocked_by = (
            "promotion_execution_result_consumer_must_request_execution_audit_first",
        )
    elif generic_status == REQUIRED_GENERIC_BLOCKED_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer_blocked"
        blocked_by = ("all_promotion_execution_audit_rows_must_pass",)
    elif generic_status != REQUIRED_GENERIC_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer_not_ready"
        blocked_by = ("passed_promotion_execution_audit_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_audit_policy_gate_request_available"
        blocked_by = ("operator_may_run_generic_audit_policy_gate",)

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_audit_policy_gate_request_available"
    )
    execution_audit_payload = (
        _mapping(generic_request.get("execution_audit_payload"))
        if request_available
        else {}
    )
    promotion_audit_policy_gate_request: Dict[str, Any] = {
        "handler": "gate_operator_proposal_apply_sequence_audit_policy",
        "execution_audit_payload": dict(execution_audit_payload),
        "read_only_template": True,
        "promotion_execution_audit_result_consumer_only": True,
        "promotion_audit_policy_gate_request_template_only": True,
        "audit_policy_gate_handler_invoked": False,
        "policy_gate_created": False,
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
        promotion_audit_policy_gate_request["promotion_execution_audit_result_consumer"] = {
            "method_id": (
                OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_AUDIT_RESULT_CONSUMER_METHOD_ID
            ),
            "source_prior_method_id": (
                OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_RESULT_CONSUMER_METHOD_ID
            ),
            "execution_audit_proposal_ids": list(expected_ids),
            "idempotency_key_prefix": expected_prefix,
            "read_only": True,
        }
        promotion_audit_policy_gate_request[
            "promotion_acceptance_record_templates"
        ] = list(promotion_acceptance_templates)
        promotion_audit_policy_gate_request["memory_trust_review_request_specs"] = list(
            memory_trust_review_request_specs
        )

    audit_policy_gate_request_specs = tuple(
        {
            **dict(spec),
            "promotion_audit_policy_gate_request_spec_id": (
                f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_AUDIT_RESULT_CONSUMER_METHOD_ID}:{spec.get('proposal_id')}"
            ),
            "promotion_execution_audit_result_consumer_only": True,
            "promotion_audit_policy_gate_request_template_only": True,
            "audit_policy_gate_handler_invoked": False,
            "policy_gate_created": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
        }
        for spec in generic_specs
    ) if request_available else ()

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_AUDIT_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_execution_audit_result_consumer_only": True,
            "promotion_audit_policy_gate_request_template_only": True,
            "requires_promotion_execution_result_consumer": True,
            "requires_execution_audit_result": True,
            "requires_all_execution_audit_rows_passed": True,
            "audit_policy_gate_handler_invoked": False,
            "policy_gate_created": False,
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
            "expected_execution_audit_row_count": len(expected_ids),
            "promotion_acceptance_record_template_count": len(
                promotion_acceptance_templates
            ),
            "memory_trust_review_request_spec_count": len(
                memory_trust_review_request_specs
            ),
            "fresh_sequence_review_context_status": _source_sequence_review_freshness_status(
                promotion_acceptance_templates
            ),
            "execution_audit_row_count": _summary_int(
                generic_summary.get("execution_audit_row_count")
            ) or 0,
            "passed_execution_audit_row_count": _summary_int(
                generic_summary.get("passed_execution_audit_row_count")
            ) or 0,
            "blocked_execution_audit_row_count": _summary_int(
                generic_summary.get("blocked_execution_audit_row_count")
            ) or 0,
            "invalid_record_count": len(invalid_record_ids),
            "idempotency_key_prefix": expected_prefix,
            "promotion_execution_audit_result_consumer_only": True,
            "promotion_audit_policy_gate_request_template_only": True,
            "promotion_audit_policy_gate_request_available": request_available,
            "audit_policy_gate_handler_invoked": False,
            "policy_gate_created": False,
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
            "expected_execution_audit_proposal_ids": list(expected_ids),
            "promotion_acceptance_record_template_proposal_ids": [
                str(template["proposal_id"])
                for template in promotion_acceptance_templates
                if isinstance(template.get("proposal_id"), str)
                and template.get("proposal_id")
            ],
            "memory_trust_review_request_spec_proposal_ids": [
                str(spec["proposal_id"])
                for spec in memory_trust_review_request_specs
                if isinstance(spec.get("proposal_id"), str) and spec.get("proposal_id")
            ],
            "execution_audit_row_proposal_ids": list(
                _string_list(generic_summary.get("execution_audit_row_proposal_ids"))
            ),
            "passed_execution_audit_proposal_ids": list(
                _string_list(generic_summary.get("passed_execution_audit_proposal_ids"))
            ),
            "blocked_execution_audit_proposal_ids": list(
                _string_list(generic_summary.get("blocked_execution_audit_proposal_ids"))
            ),
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion execution-result consumer and execution-audit result."
                if not prior or not audit
                else (
                    "Replace invalid promotion execution-audit evidence before policy gate."
                    if invalid_record_ids
                    else (
                        "Re-run execution audit until every promotion row passes."
                        if generic_status == REQUIRED_GENERIC_BLOCKED_STATUS
                        else "Operator may run the generic audit-policy gate; this consumer does not run it."
                    )
                )
            ),
        },
        "promotion_audit_policy_gate_request": promotion_audit_policy_gate_request,
        "promotion_audit_policy_gate_request_specs": list(audit_policy_gate_request_specs),
        "generic_execution_audit_result_consumer": generic_consumer if prior and audit else {},
    }
