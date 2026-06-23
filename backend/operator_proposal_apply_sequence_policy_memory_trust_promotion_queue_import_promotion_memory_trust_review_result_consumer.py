"""Read-only consumer for promotion memory trust-review results.

This verifier matches memory trust-review evidence back to promotion
memory-index provenance before exposing an acceptance-ledger request template.
It delegates generic trust-review validation to the queue-import trust-review
result consumer and does not accept candidates, invoke ledger handlers, mutate
memory, policy, models, queues, idempotency, or mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_INDEX_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_INDEX_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_request_available",
}
REQUIRED_PRIOR_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer"
)
REQUIRED_PRIOR_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_review_request_available"
)
REQUIRED_GENERIC_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_request_available"
)
REQUIRED_GENERIC_REPLACEMENT_STATUS = (
    "sequence_memory_trust_promotion_queue_import_memory_trust_review_result_consumer_replacement_required"
)
REQUIRED_GENERIC_BLOCKED_STATUS = (
    "sequence_memory_trust_promotion_queue_import_memory_trust_review_result_consumer_blocked"
)
PRIOR_FALSE_FLAGS = (
    "memory_trust_review_handler_invoked",
    "memory_trust_review_created",
    "memory_write_allowed_by_consumer",
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
CONSUMER_FALSE_FLAGS = (
    "acceptance_ledger_handler_invoked",
    "acceptance_ledger_created",
    "memory_write_allowed_by_consumer",
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


def _records_by_proposal_id(
    records: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(record["proposal_id"]): record
        for record in records
        if isinstance(record.get("proposal_id"), str) and record.get("proposal_id")
    }


def _prior_envelope_violations(prior: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if prior.get("type") != REQUIRED_PRIOR_TYPE:
        violations.append("promotion_memory_trust_review_result_consumer_prior_type_mismatch")
    generated_at = prior.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("promotion_memory_trust_review_result_consumer_prior_generated_at_invalid")
    if (
        prior.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append("promotion_memory_trust_review_result_consumer_prior_method_id_mismatch")
    if (
        prior.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_INDEX_RESULT_CONSUMER_STATUSES
    ):
        violations.append("promotion_memory_trust_review_result_consumer_prior_status_unknown")
    safety = _mapping(prior.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("promotion_memory_trust_review_result_consumer_prior_read_only_not_true")
    if safety.get("promotion_memory_index_result_consumer_only") is not True:
        violations.append("promotion_memory_trust_review_result_consumer_prior_only_not_true")
    if safety.get("promotion_memory_trust_review_request_template_only") is not True:
        violations.append("promotion_memory_trust_review_result_consumer_prior_template_only_not_true")
    if safety.get("source_memory_index_available") is not True:
        violations.append("promotion_memory_trust_review_result_consumer_prior_source_index_not_true")
    for flag in PRIOR_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"promotion_memory_trust_review_result_consumer_prior_{flag}_not_false")
    return tuple(violations)


def _prior_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
    memory_trust_review_request_specs: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    if _summary_int(summary.get("promotion_acceptance_record_template_count")) != len(
        promotion_acceptance_templates
    ):
        violations.append(
            "promotion_memory_trust_review_result_consumer_prior_summary_template_count_mismatch"
        )
    if _summary_int(summary.get("memory_trust_review_request_spec_count")) != len(
        memory_trust_review_request_specs
    ):
        violations.append(
            "promotion_memory_trust_review_result_consumer_prior_summary_review_request_spec_count_mismatch"
        )
    expected_counts = {
        "expected_memory_index_record_count": len(expected_ids),
        "memory_index_record_count": len(expected_ids),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"promotion_memory_trust_review_result_consumer_prior_summary_{key}_mismatch")
    if summary.get("promotion_memory_trust_review_request_available") is not bool(expected_ids):
        violations.append("promotion_memory_trust_review_result_consumer_prior_summary_request_available_mismatch")
    if summary.get("source_memory_index_available") is not True:
        violations.append("promotion_memory_trust_review_result_consumer_prior_summary_source_index_not_true")
    if summary.get("fresh_sequence_review_context_status") != "fresh":
        violations.append("promotion_memory_trust_review_result_consumer_prior_summary_freshness_not_fresh")
    if (
        promotion_acceptance_templates
        and summary.get("source_sequence_review_context_status") != "fresh"
    ):
        violations.append(
            "promotion_memory_trust_review_result_consumer_prior_summary_source_freshness_not_fresh"
        )
    if _string_list(summary.get("memory_index_record_proposal_ids")) != expected_ids:
        violations.append("promotion_memory_trust_review_result_consumer_prior_summary_proposal_ids_mismatch")
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
            "promotion_memory_trust_review_result_consumer_prior_summary_template_proposal_ids_mismatch"
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
            "promotion_memory_trust_review_result_consumer_prior_summary_review_request_spec_proposal_ids_mismatch"
        )
    for flag in PRIOR_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"promotion_memory_trust_review_result_consumer_prior_summary_{flag}_not_false"
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
    if request.get("handler") != "review_operator_proposal_apply_sequence_policy_memory_trust":
        violations.append("promotion_memory_trust_review_result_consumer_prior_request_handler_mismatch")
    if request.get("read_only_template") is not True:
        violations.append("promotion_memory_trust_review_result_consumer_prior_request_template_not_true")
    if request.get("promotion_memory_index_result_consumer_only") is not True:
        violations.append("promotion_memory_trust_review_result_consumer_prior_request_only_not_true")
    if request.get("promotion_memory_trust_review_request_template_only") is not True:
        violations.append("promotion_memory_trust_review_result_consumer_prior_request_template_only_not_true")
    if request.get("source_memory_index_available") is not True:
        violations.append("promotion_memory_trust_review_result_consumer_prior_request_source_index_not_true")
    if _mapping(request.get("memory_index")).get("status") != "append_only_sequence_memory_index_available":
        violations.append("promotion_memory_trust_review_result_consumer_prior_request_memory_index_not_available")
    if _mapping(request.get("current_sequence_review_freshness")).get("status") != "fresh":
        violations.append("promotion_memory_trust_review_result_consumer_prior_request_freshness_not_fresh")
    for flag in PRIOR_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(
                f"promotion_memory_trust_review_result_consumer_prior_request_{flag}_not_false"
            )
    template_proposal_ids = {
        str(template["proposal_id"])
        for template in promotion_acceptance_templates
        if isinstance(template.get("proposal_id"), str) and template.get("proposal_id")
    }
    if len(template_proposal_ids) != len(promotion_acceptance_templates):
        violations.append(
            "promotion_memory_trust_review_result_consumer_prior_request_acceptance_template_proposal_ids_not_unique"
        )
    if expected_ids and not set(expected_ids).issubset(template_proposal_ids):
        violations.append(
            "promotion_memory_trust_review_result_consumer_prior_request_acceptance_templates_missing_trust_review"
        )
    request_spec_proposal_ids = {
        str(spec["proposal_id"])
        for spec in memory_trust_review_request_specs
        if isinstance(spec.get("proposal_id"), str) and spec.get("proposal_id")
    }
    if not template_proposal_ids.issubset(request_spec_proposal_ids):
        violations.append(
            "promotion_memory_trust_review_result_consumer_prior_request_review_request_spec_proposal_ids_mismatch"
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
    request_specs_by_proposal_id = _records_by_proposal_id(
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


def _prior_spec_violations(
    specs: Tuple[Mapping[str, Any], ...],
    *,
    expected_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if len(specs) != len(expected_ids):
        violations.append("promotion_memory_trust_review_result_consumer_prior_spec_count_mismatch")
    spec_ids = tuple(_string(spec.get("proposal_id")) for spec in specs)
    if spec_ids != expected_ids:
        violations.append("promotion_memory_trust_review_result_consumer_prior_spec_proposal_ids_mismatch")
    for index, spec in enumerate(specs):
        prefix = f"promotion_memory_trust_review_request_spec:{index}"
        if spec.get("promotion_memory_index_result_consumer_only") is not True:
            violations.append(f"{prefix}:only_not_true")
        if spec.get("promotion_memory_trust_review_request_template_only") is not True:
            violations.append(f"{prefix}:template_only_not_true")
        if spec.get("source_memory_index_available") is not True:
            violations.append(f"{prefix}:source_memory_index_available_not_true")
        for flag in PRIOR_FALSE_FLAGS:
            if spec.get(flag) is not False:
                violations.append(f"{prefix}:{flag}_not_false")
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer(
    *,
    promotion_memory_index_result_consumer: Mapping[str, Any] | None = None,
    promotion_memory_trust_review_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify promotion trust-review evidence and emit acceptance templates."""

    prior = _mapping(promotion_memory_index_result_consumer)
    review = _mapping(promotion_memory_trust_review_result)
    prior_summary = _mapping(prior.get("summary"))
    prior_request = _mapping(prior.get("promotion_memory_trust_review_request"))
    promotion_acceptance_templates = _acceptance_templates_from_prior_request(
        prior_request
    )
    memory_trust_review_request_specs = _request_specs_from_prior_request(prior_request)
    prior_specs = _record_list(prior.get("promotion_memory_trust_review_request_specs"))
    prior_summary_ids = _string_list(prior_summary.get("memory_index_record_proposal_ids"))
    prior_status = _string(prior.get("status"))
    generic_index_consumer = _mapping(prior.get("generic_memory_index_result_consumer"))

    generic_consumer: Mapping[str, Any] = {}
    if prior and review:
        generic_consumer = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer(
                memory_index_result_consumer=generic_index_consumer,
                memory_trust_review_result=review,
            )
        )
    generic_status = _string(generic_consumer.get("status"))
    generic_summary = _mapping(generic_consumer.get("summary"))
    generic_request = _mapping(generic_consumer.get("acceptance_ledger_request"))
    generic_templates = _record_list(generic_consumer.get("acceptance_record_templates"))
    generic_invalid_record_ids = tuple(
        f"generic_memory_trust_review_result_consumer:{item}"
        for item in _string_list(generic_summary.get("invalid_record_ids"))
    )
    generic_candidate_ids = _string_list(
        generic_summary.get("trust_review_candidate_proposal_ids")
    )
    expected_ids = generic_candidate_ids or prior_summary_ids

    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_string_list(prior_summary.get("invalid_record_ids")),
                *_prior_envelope_violations(prior),
                *_prior_summary_violations(
                    prior_summary,
                    expected_ids=expected_ids,
                    promotion_acceptance_templates=promotion_acceptance_templates,
                    memory_trust_review_request_specs=memory_trust_review_request_specs,
                ),
                *_prior_request_violations(
                    prior_request,
                    expected_ids=expected_ids,
                    promotion_acceptance_templates=promotion_acceptance_templates,
                    memory_trust_review_request_specs=memory_trust_review_request_specs,
                ),
                *_prior_spec_violations(prior_specs, expected_ids=expected_ids),
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
                    ("promotion_memory_trust_review_result_consumer_generic_status_unknown",)
                    if (
                        generic_status
                        and generic_status
                        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_STATUSES
                    )
                    else ()
                ),
            )
        )
    )
    invalid_record_ids = () if not prior or not review else raw_invalid_record_ids

    if not prior or not review:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer_not_ready"
        blocked_by = (
            "promotion_memory_index_result_consumer_and_memory_trust_review_result_required",
        )
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_promotion_memory_trust_review_evidence",
        )
    elif prior_status != REQUIRED_PRIOR_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer_blocked"
        blocked_by = (
            "promotion_memory_index_result_consumer_must_request_trust_review_first",
        )
    elif generic_status == REQUIRED_GENERIC_BLOCKED_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer_blocked"
        blocked_by = ("promotion_memory_trust_review_must_be_available_before_acceptance",)
    elif generic_status != REQUIRED_GENERIC_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer_not_ready"
        blocked_by = ("ready_promotion_memory_trust_review_result_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_request_available"
        blocked_by = ("operator_may_record_promotion_memory_trust_acceptance_decisions",)

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_request_available"
    )
    acceptance_templates = tuple(
        {
            **dict(template),
            "promotion_acceptance_record_template_id": (
                f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID}:{template.get('review_candidate_id')}"
            ),
            "promotion_memory_trust_review_result_consumer_only": True,
            "promotion_acceptance_ledger_request_template_only": True,
            "sequence_policy_trust_store_mutation": False,
            "proposal_state_mutation": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
        }
        for template in generic_templates
    ) if request_available else ()
    acceptance_ledger_request: Dict[str, Any] = {
        **dict(generic_request if request_available else {}),
        "handler": "record_operator_proposal_apply_sequence_policy_memory_trust_acceptance",
        "read_only_template": True,
        "promotion_memory_trust_review_result_consumer_only": True,
        "promotion_acceptance_ledger_request_template_only": True,
        "acceptance_ledger_handler_invoked": False,
        "acceptance_ledger_created": False,
        "operator_must_supply_acceptance_records": True,
        "acceptance_records": [],
        "acceptance_record_templates": list(acceptance_templates),
        **{flag: False for flag in CONSUMER_FALSE_FLAGS},
    }
    if request_available:
        acceptance_ledger_request["memory_trust_review"] = review
        acceptance_ledger_request[
            "promotion_acceptance_record_templates"
        ] = list(promotion_acceptance_templates)
        acceptance_ledger_request["memory_trust_review_request_specs"] = list(
            memory_trust_review_request_specs
        )
        acceptance_ledger_request["promotion_memory_trust_review_result_consumer"] = {
            "method_id": (
                OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID
            ),
            "source_prior_method_id": (
                OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID
            ),
            "trust_review_candidate_proposal_ids": list(expected_ids),
            "read_only": True,
        }

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_memory_trust_review_result_consumer_only": True,
            "promotion_acceptance_ledger_request_template_only": True,
            "requires_promotion_memory_index_result_consumer": True,
            "requires_memory_trust_review_result": True,
            "operator_acceptance_required": True,
            **{flag: False for flag in CONSUMER_FALSE_FLAGS},
        },
        "summary": {
            "expected_trust_review_candidate_count": len(expected_ids),
            "source_promotion_acceptance_record_template_count": len(
                promotion_acceptance_templates
            ),
            "memory_trust_review_request_spec_count": len(
                memory_trust_review_request_specs
            ),
            "source_sequence_review_context_status": _source_sequence_review_freshness_status(
                promotion_acceptance_templates
            ),
            "trust_review_candidate_count": (
                _summary_int(generic_summary.get("trust_review_candidate_count")) or 0
            ),
            "promotion_acceptance_record_template_count": len(acceptance_templates),
            "invalid_record_count": len(invalid_record_ids),
            "promotion_memory_trust_review_result_consumer_only": True,
            "promotion_acceptance_ledger_request_template_only": True,
            "promotion_acceptance_ledger_request_available": request_available,
            "operator_acceptance_required": True,
            **{flag: False for flag in CONSUMER_FALSE_FLAGS},
            "trust_review_candidate_proposal_ids": list(
                generic_candidate_ids if request_available else ()
            ),
            "promotion_acceptance_record_template_proposal_ids": [
                str(template["proposal_id"])
                for template in acceptance_templates
                if isinstance(template.get("proposal_id"), str)
                and template.get("proposal_id")
            ],
            "source_promotion_acceptance_record_template_proposal_ids": [
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
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion memory-index consumer and memory trust-review result."
                if not prior or not review
                else (
                    "Replace invalid promotion memory trust-review evidence before acceptance."
                    if invalid_record_ids
                    else (
                        "Run read-only memory trust-review before promotion acceptance."
                        if generic_status == REQUIRED_GENERIC_BLOCKED_STATUS
                        else "Operator may record accept/reject decisions; this consumer does not decide."
                    )
                )
            ),
        },
        "promotion_acceptance_ledger_request": acceptance_ledger_request,
        "promotion_acceptance_record_templates": list(acceptance_templates),
        "generic_memory_trust_review_result_consumer": (
            generic_consumer if prior and review else {}
        ),
    }
