"""Read-only consumer for promotion queue-ordering-plan results.

This verifier connects the queue-import promotion ordering-preview result
consumer to the existing queue importer. It verifies queue-ordering-plan rows
against the previously verified ordering-preview provenance and emits only a
queue-importer request template. It does not import queues, reserve
idempotency, mutate proposals, policy/runtime/model or memory state, or touch
mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ORDERING_PREVIEW_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ORDERING_PREVIEW_RESULT_CONSUMER_STATUSES,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_ORDERING_PLAN_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_ORDERING_PLAN_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ORDERING_PREVIEW_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ORDERING_PREVIEW_RESULT_CONSUMER_STATUSES,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_ORDERING_PLAN_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_ORDERING_PLAN_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_request_available",
}
REQUIRED_ORDERING_PREVIEW_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer"
)
REQUIRED_ORDERING_PREVIEW_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_request_available"
)
REQUIRED_QUEUE_ORDERING_PLAN_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan"
)
REQUIRED_QUEUE_ORDERING_PLAN_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_ordering_plan_available"
)
REQUIRED_PRIOR_CONSUMER_FALSE_FLAGS = (
    "promotion_queue_ordering_plan_handler_invoked",
    "promotion_queue_ordering_plan_created",
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
REQUIRED_QUEUE_ORDERING_PLAN_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "proposal_order_mutation",
    "queue_mutation",
    "queue_import_allowed_by_plan",
    "queue_reorder_allowed_by_plan",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_plan",
    "sequence_policy_change_allowed_by_plan",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
)
REQUIRED_QUEUE_ORDERING_PLAN_SUMMARY_FALSE_FLAGS = tuple(
    flag for flag in REQUIRED_QUEUE_ORDERING_PLAN_FALSE_FLAGS if flag != "apply_attempted"
)
REQUIRED_QUEUE_ORDERING_PLAN_RECORD_FALSE_FLAGS = REQUIRED_QUEUE_ORDERING_PLAN_SUMMARY_FALSE_FLAGS
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
REQUIRED_REQUEST_FALSE_FLAGS = (
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
ALLOWED_RANK_SOURCES = {"sequence_policy_consumer", "original_order"}


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


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in {float("inf"), float("-inf")}:
        return None
    return numeric


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


def _ordering_preview_from_prior_consumer(
    consumer: Mapping[str, Any],
) -> Mapping[str, Any]:
    request = _mapping(consumer.get("promotion_queue_ordering_plan_request"))
    return _mapping(request.get("ordering_preview"))


def _acceptance_templates_from_result_consumer_request(
    consumer: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    request = _mapping(consumer.get("promotion_queue_ordering_plan_request"))
    return _record_list(request.get("promotion_acceptance_record_templates"))


def _request_specs_from_result_consumer_request(
    consumer: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    request = _mapping(consumer.get("promotion_queue_ordering_plan_request"))
    return _record_list(request.get("memory_trust_review_request_specs"))


def _ordering_rows_by_id(
    rows: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(row["ordering_preview_record_id"]): row
        for row in rows
        if isinstance(row.get("ordering_preview_record_id"), str)
        and row.get("ordering_preview_record_id")
    }


def _templates_by_proposal_id(
    templates: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(template["proposal_id"]): template
        for template in templates
        if isinstance(template.get("proposal_id"), str) and template.get("proposal_id")
    }


def _request_specs_by_proposal_id(
    request_specs: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(spec["proposal_id"]): spec
        for spec in request_specs
        if isinstance(spec.get("proposal_id"), str) and spec.get("proposal_id")
    }


def _plan_record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("queue_ordering_plan_record_id"))
    if explicit:
        return explicit
    proposal_id = _string(record.get("proposal_id"))
    return f"{proposal_id if proposal_id else 'unknown_queue_ordering_plan_record'}:{index}"


def _duplicate_ids(
    records: Tuple[Mapping[str, Any], ...],
    field: str,
    *,
    id_func,
) -> Tuple[str, ...]:
    values: Dict[str, list[str]] = {}
    for index, record in enumerate(records):
        value = str(record.get(field)) if record.get(field) is not None else ""
        if not value:
            continue
        values.setdefault(value, []).append(id_func(record, index))
    return tuple(
        record_id
        for record_ids in values.values()
        if len(record_ids) > 1
        for record_id in record_ids
    )


def _prior_consumer_envelope_violations(
    consumer: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_ORDERING_PREVIEW_RESULT_CONSUMER_TYPE:
        violations.append("queue_ordering_plan_result_consumer_prior_type_mismatch")
    generated_at = consumer.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("queue_ordering_plan_result_consumer_prior_generated_at_invalid")
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ORDERING_PREVIEW_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append("queue_ordering_plan_result_consumer_prior_method_id_mismatch")
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ORDERING_PREVIEW_RESULT_CONSUMER_STATUSES
    ):
        violations.append("queue_ordering_plan_result_consumer_prior_status_unknown")
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("queue_ordering_plan_result_consumer_prior_read_only_not_true")
    if safety.get("promotion_ordering_preview_result_consumer_only") is not True:
        violations.append("queue_ordering_plan_result_consumer_prior_only_not_true")
    if safety.get("promotion_queue_ordering_plan_request_template_only") is not True:
        violations.append("queue_ordering_plan_result_consumer_prior_template_only_not_true")
    for flag in REQUIRED_PRIOR_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"queue_ordering_plan_result_consumer_prior_{flag}_not_false")
    return tuple(violations)


def _prior_consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    ordering_rows: Tuple[Mapping[str, Any], ...],
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
    memory_trust_review_request_specs: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    ranked_rows = tuple(
        row for row in ordering_rows if row.get("rank_source") == "sequence_policy_consumer"
    )
    unranked_rows = tuple(
        row for row in ordering_rows if row.get("rank_source") == "original_order"
    )
    violations = []
    expected_counts = {
        "ordered_preview_record_count": len(ordering_rows),
        "policy_ranked_preview_record_count": len(ranked_rows),
        "unranked_preview_record_count": len(unranked_rows),
        "promotion_acceptance_record_template_count": len(
            promotion_acceptance_templates
        ),
        "memory_trust_review_request_spec_count": len(
            memory_trust_review_request_specs
        ),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"queue_ordering_plan_result_consumer_prior_summary_{key}_mismatch")
    if summary.get("promotion_queue_ordering_plan_request_available") is not True:
        violations.append("queue_ordering_plan_result_consumer_prior_summary_request_not_true")
    if summary.get("promotion_ordering_preview_result_consumer_only") is not True:
        violations.append("queue_ordering_plan_result_consumer_prior_summary_only_not_true")
    if summary.get("promotion_queue_ordering_plan_request_template_only") is not True:
        violations.append("queue_ordering_plan_result_consumer_prior_summary_template_not_true")
    for flag in REQUIRED_PRIOR_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"queue_ordering_plan_result_consumer_prior_summary_{flag}_not_false"
            )
    expected_ids = tuple(
        row["proposal_id"] for row in ordering_rows if _string(row.get("proposal_id"))
    )
    if _string_list(summary.get("ordered_preview_proposal_ids")) != expected_ids:
        violations.append("queue_ordering_plan_result_consumer_prior_summary_ordered_ids_mismatch")
    if (
        promotion_acceptance_templates
        and summary.get("fresh_sequence_review_context_status") != "fresh"
    ):
        violations.append(
            "queue_ordering_plan_result_consumer_prior_summary_freshness_not_fresh"
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
            "queue_ordering_plan_result_consumer_prior_summary_template_proposal_ids_mismatch"
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
            "queue_ordering_plan_result_consumer_prior_summary_review_request_spec_proposal_ids_mismatch"
        )
    return tuple(violations)


def _prior_consumer_request_violations(
    request: Mapping[str, Any],
    *,
    ordering_rows: Tuple[Mapping[str, Any], ...],
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
    memory_trust_review_request_specs: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    if (
        request.get("handler")
        != "plan_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering"
    ):
        violations.append("queue_ordering_plan_result_consumer_prior_request_handler_mismatch")
    if request.get("read_only_template") is not True:
        violations.append("queue_ordering_plan_result_consumer_prior_request_read_only_not_true")
    if request.get("promotion_ordering_preview_result_consumer_only") is not True:
        violations.append("queue_ordering_plan_result_consumer_prior_request_only_not_true")
    if request.get("promotion_queue_ordering_plan_request_template_only") is not True:
        violations.append("queue_ordering_plan_result_consumer_prior_request_template_not_true")
    if request.get("promotion_queue_ordering_plan_handler_invoked") is not False:
        violations.append("queue_ordering_plan_result_consumer_prior_request_handler_invoked")
    if request.get("promotion_queue_ordering_plan_created") is not False:
        violations.append("queue_ordering_plan_result_consumer_prior_request_created")
    if request.get("ordering_preview_required") is not True:
        violations.append("queue_ordering_plan_result_consumer_prior_request_preview_not_required")
    for flag in REQUIRED_REQUEST_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(f"queue_ordering_plan_result_consumer_prior_request_{flag}_not_false")
    if not ordering_rows:
        violations.append("queue_ordering_plan_result_consumer_prior_request_ordering_rows_missing")
    ranked_proposal_ids = {
        str(row["proposal_id"])
        for row in ordering_rows
        if row.get("rank_source") == "sequence_policy_consumer"
        and isinstance(row.get("proposal_id"), str)
        and row.get("proposal_id")
    }
    template_proposal_ids = {
        str(template["proposal_id"])
        for template in promotion_acceptance_templates
        if isinstance(template.get("proposal_id"), str) and template.get("proposal_id")
    }
    if len(template_proposal_ids) != len(promotion_acceptance_templates):
        violations.append(
            "queue_ordering_plan_result_consumer_prior_request_acceptance_template_proposal_ids_not_unique"
        )
    if ranked_proposal_ids and not ranked_proposal_ids.issubset(template_proposal_ids):
        violations.append(
            "queue_ordering_plan_result_consumer_prior_request_acceptance_templates_missing_ordering_row"
        )
    request_spec_proposal_ids = {
        str(spec["proposal_id"])
        for spec in memory_trust_review_request_specs
        if isinstance(spec.get("proposal_id"), str) and spec.get("proposal_id")
    }
    if not template_proposal_ids.issubset(request_spec_proposal_ids):
        violations.append(
            "queue_ordering_plan_result_consumer_prior_request_review_request_spec_proposal_ids_mismatch"
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


def _queue_ordering_plan_envelope_violations(
    plan: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    if plan.get("type") != REQUIRED_QUEUE_ORDERING_PLAN_TYPE:
        violations.append("queue_ordering_plan_result_consumer_plan_type_mismatch")
    generated_at = plan.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("queue_ordering_plan_result_consumer_plan_generated_at_invalid")
    if (
        plan.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_ORDERING_PLAN_METHOD_ID
    ):
        violations.append("queue_ordering_plan_result_consumer_plan_method_id_mismatch")
    if plan.get("status") not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_ORDERING_PLAN_STATUSES:
        violations.append("queue_ordering_plan_result_consumer_plan_status_unknown")
    if (
        plan.get("source_method_id")
        != "operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview"
    ):
        violations.append("queue_ordering_plan_result_consumer_plan_source_mismatch")
    safety = _mapping(plan.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("queue_ordering_plan_result_consumer_plan_read_only_not_true")
    if safety.get("queue_ordering_plan_only") is not True:
        violations.append("queue_ordering_plan_result_consumer_plan_only_not_true")
    if safety.get("operator_required") is not True:
        violations.append("queue_ordering_plan_result_consumer_plan_operator_required_not_true")
    if safety.get("sequence_consumer_required_before_execution") is not True:
        violations.append("queue_ordering_plan_result_consumer_plan_sequence_consumer_required_not_true")
    for flag in REQUIRED_QUEUE_ORDERING_PLAN_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"queue_ordering_plan_result_consumer_plan_{flag}_not_false")
    return tuple(violations)


def _queue_ordering_plan_summary_violations(
    summary: Mapping[str, Any],
    *,
    ordering_rows: Tuple[Mapping[str, Any], ...],
    plan_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    ranked_records = tuple(
        record
        for record in plan_records
        if record.get("rank_source") == "sequence_policy_consumer"
    )
    unranked_records = tuple(
        record for record in plan_records if record.get("rank_source") == "original_order"
    )
    violations = []
    expected_counts = {
        "input_ordering_preview_count": len(ordering_rows),
        "queue_ordering_plan_record_count": len(plan_records),
        "ranked_queue_candidate_count": len(ranked_records),
        "unranked_queue_candidate_count": len(unranked_records),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"queue_ordering_plan_result_consumer_plan_summary_{key}_mismatch")
    if summary.get("queue_ordering_plan_available") is not bool(plan_records):
        violations.append("queue_ordering_plan_result_consumer_plan_summary_available_mismatch")
    if summary.get("queue_ordering_plan_only") is not True:
        violations.append("queue_ordering_plan_result_consumer_plan_summary_only_not_true")
    if summary.get("sequence_consumer_required_before_execution") is not True:
        violations.append("queue_ordering_plan_result_consumer_plan_summary_sequence_consumer_required_not_true")
    for flag in REQUIRED_QUEUE_ORDERING_PLAN_SUMMARY_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"queue_ordering_plan_result_consumer_plan_summary_{flag}_not_false"
            )
    expected_ids = tuple(
        record["proposal_id"]
        for record in plan_records
        if _string(record.get("proposal_id"))
    )
    if _string_list(summary.get("planned_queue_proposal_ids")) != expected_ids:
        violations.append("queue_ordering_plan_result_consumer_plan_summary_ids_mismatch")
    return tuple(violations)


def _plan_record_violations(
    record: Mapping[str, Any],
    *,
    ordering_rows_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[str, ...]:
    violations = []
    required_fields = (
        "queue_ordering_plan_record_id",
        "proposal_id",
        "ordering_preview_record_id",
        "planned_queue_index",
        "rank_source",
        "recommended_sequence_policy_bucket",
    )
    for field in required_fields:
        if field == "planned_queue_index":
            if _summary_int(record.get(field)) is None:
                violations.append(f"queue_ordering_plan_result_consumer_record_{field}_invalid")
        elif not _string(record.get(field)):
            violations.append(f"queue_ordering_plan_result_consumer_record_{field}_missing")
    if record.get("rank_source") not in ALLOWED_RANK_SOURCES:
        violations.append("queue_ordering_plan_result_consumer_record_rank_source_unknown")
    for field in ("original_index", "recommended_index"):
        if _summary_int(record.get(field)) is None:
            violations.append(f"queue_ordering_plan_result_consumer_record_{field}_invalid")
    for field in ("base_policy_score", "adjusted_policy_score"):
        score = _safe_float(record.get(field))
        if score is None or score < 0.0 or score > 1.0:
            violations.append(f"queue_ordering_plan_result_consumer_record_{field}_invalid")
    required_true_fields = (
        "operator_review_required",
        "operator_ack_required_before_queue_import",
        "operator_ack_required_before_sequence_execution",
        "idempotency_key_required",
        "queue_ordering_plan_only",
        "sequence_consumer_required_before_execution",
        "read_only",
    )
    for field in required_true_fields:
        if record.get(field) is not True:
            violations.append(f"queue_ordering_plan_result_consumer_record_{field}_not_true")
    for flag in REQUIRED_QUEUE_ORDERING_PLAN_RECORD_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(f"queue_ordering_plan_result_consumer_record_{flag}_not_false")

    ordering_row_id = _string(record.get("ordering_preview_record_id"))
    ordering_row = ordering_rows_by_id.get(ordering_row_id)
    if not ordering_row:
        violations.append("queue_ordering_plan_result_consumer_record_ordering_row_missing")
        return tuple(violations)

    expected_pairs = {
        "proposal_id": ordering_row.get("proposal_id"),
        "proposal_method_id": ordering_row.get("proposal_method_id"),
        "proposal_kind": ordering_row.get("proposal_kind"),
        "proposal_target": ordering_row.get("proposal_target"),
        "consumer_record_id": ordering_row.get("consumer_record_id"),
        "sequence_policy_gate_record_id": ordering_row.get("sequence_policy_gate_record_id"),
        "policy_key": ordering_row.get("policy_key"),
        "source_idempotency_key": ordering_row.get("source_idempotency_key"),
        "original_index": ordering_row.get("original_index"),
        "recommended_index": ordering_row.get("recommended_index"),
        "rank_source": ordering_row.get("rank_source"),
        "recommended_sequence_policy_bucket": ordering_row.get(
            "recommended_sequence_policy_bucket"
        ),
        "base_policy_score": ordering_row.get("base_policy_score"),
        "adjusted_policy_score": ordering_row.get("adjusted_policy_score"),
        "current_sequence_review_freshness": ordering_row.get(
            "current_sequence_review_freshness"
        ),
    }
    for field, expected in expected_pairs.items():
        if record.get(field) != expected:
            violations.append(f"queue_ordering_plan_result_consumer_record_{field}_mismatch")
    violations.extend(
        _context_violations(
            "queue_ordering_plan_result_consumer_record",
            _mapping(record.get("current_sequence_review_freshness")),
        )
    )
    return tuple(violations)


def _invalid_plan_record_ids(
    records: Tuple[Mapping[str, Any], ...],
    *,
    ordering_rows_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[str, ...]:
    return tuple(
        _plan_record_id(record, index)
        for index, record in enumerate(records)
        if _plan_record_violations(record, ordering_rows_by_id=ordering_rows_by_id)
    )


def _plan_record_template_identity_violations(
    *,
    records: Tuple[Mapping[str, Any], ...],
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
    memory_trust_review_request_specs: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    templates_by_proposal_id = _templates_by_proposal_id(promotion_acceptance_templates)
    request_specs_by_proposal_id = _request_specs_by_proposal_id(
        memory_trust_review_request_specs
    )
    violations = []
    for record in records:
        if record.get("rank_source") != "sequence_policy_consumer":
            continue
        proposal_id = _string(record.get("proposal_id"))
        template = _mapping(templates_by_proposal_id.get(proposal_id))
        if not template:
            violations.append(f"{proposal_id}:acceptance_template_missing")
            continue
        expected_pairs = {
            "proposal_id": record.get("proposal_id"),
            "idempotency_key": record.get("source_idempotency_key"),
            "planned_queue_index": record.get("planned_queue_index"),
            "rank_source": record.get("rank_source"),
        }
        for field, expected in expected_pairs.items():
            if template.get(field) != expected:
                violations.append(
                    f"{proposal_id}:{field}_queue_ordering_plan_record_mismatch"
                )
        if _sequence_review_freshness_core(
            _mapping(template.get("current_sequence_review_freshness"))
        ) != _sequence_review_freshness_core(
            _mapping(record.get("current_sequence_review_freshness"))
        ):
            violations.append(
                f"{proposal_id}:current_sequence_review_freshness_queue_ordering_plan_record_mismatch"
            )
        request_spec = _mapping(request_specs_by_proposal_id.get(proposal_id))
        if request_spec:
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


def _sequence_violations(records: Tuple[Mapping[str, Any], ...]) -> Tuple[str, ...]:
    planned_indexes = tuple(
        record.get("planned_queue_index")
        for record in records
        if _summary_int(record.get("planned_queue_index")) is not None
    )
    if planned_indexes != tuple(range(len(records))):
        return ("queue_ordering_plan_result_consumer_planned_index_sequence_mismatch",)
    return ()


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer(
    *,
    promotion_ordering_preview_result_consumer: Mapping[str, Any] | None = None,
    promotion_queue_ordering_plan_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify queue-ordering-plan evidence and emit a queue-importer request."""

    prior_consumer = _mapping(promotion_ordering_preview_result_consumer)
    queue_ordering_plan = _mapping(promotion_queue_ordering_plan_result)
    prior_request = _mapping(prior_consumer.get("promotion_queue_ordering_plan_request"))
    ordering_preview = _ordering_preview_from_prior_consumer(prior_consumer)
    promotion_acceptance_templates = _acceptance_templates_from_result_consumer_request(
        prior_consumer
    )
    memory_trust_review_request_specs = _request_specs_from_result_consumer_request(
        prior_consumer
    )
    ordering_rows = _record_list(ordering_preview.get("ordered_proposals"))
    plan_records = _record_list(queue_ordering_plan.get("queue_ordering_plan_records"))
    ranked_records = tuple(
        record
        for record in plan_records
        if record.get("rank_source") == "sequence_policy_consumer"
    )
    unranked_records = tuple(
        record for record in plan_records if record.get("rank_source") == "original_order"
    )
    ordering_rows_by_id = _ordering_rows_by_id(ordering_rows)
    prior_status = prior_consumer.get("status")
    plan_status = queue_ordering_plan.get("status")

    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_prior_consumer_envelope_violations(prior_consumer),
                *(
                    (
                        *_prior_consumer_summary_violations(
                            _mapping(prior_consumer.get("summary")),
                            ordering_rows=ordering_rows,
                            promotion_acceptance_templates=promotion_acceptance_templates,
                            memory_trust_review_request_specs=memory_trust_review_request_specs,
                        ),
                        *_prior_consumer_request_violations(
                            prior_request,
                            ordering_rows=ordering_rows,
                            promotion_acceptance_templates=promotion_acceptance_templates,
                            memory_trust_review_request_specs=memory_trust_review_request_specs,
                        ),
                        *(
                            violation
                            for index, template in enumerate(
                                promotion_acceptance_templates
                            )
                            for violation in _acceptance_template_violations(
                                template,
                                index=index,
                            )
                        ),
                    )
                    if prior_status
                    == REQUIRED_ORDERING_PREVIEW_RESULT_CONSUMER_AVAILABLE_STATUS
                    else ()
                ),
                *_queue_ordering_plan_envelope_violations(queue_ordering_plan),
                *(
                    (
                        "queue_ordering_plan_result_consumer_prior_replacement_required",
                    )
                    if prior_status
                    == "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer_replacement_required"
                    else ()
                ),
                *(
                    (
                        "queue_ordering_plan_result_consumer_plan_replacement_required",
                    )
                    if plan_status
                    == "sequence_memory_trust_promotion_queue_ordering_plan_replacement_required"
                    else ()
                ),
                *(
                    (
                        *_queue_ordering_plan_summary_violations(
                            _mapping(queue_ordering_plan.get("summary")),
                            ordering_rows=ordering_rows,
                            plan_records=plan_records,
                        ),
                        *_invalid_plan_record_ids(
                            plan_records,
                            ordering_rows_by_id=ordering_rows_by_id,
                        ),
                        *_plan_record_template_identity_violations(
                            records=plan_records,
                            promotion_acceptance_templates=promotion_acceptance_templates,
                            memory_trust_review_request_specs=memory_trust_review_request_specs,
                        ),
                        *_sequence_violations(plan_records),
                        *_duplicate_ids(
                            plan_records,
                            "queue_ordering_plan_record_id",
                            id_func=_plan_record_id,
                        ),
                        *_duplicate_ids(
                            plan_records,
                            "proposal_id",
                            id_func=_plan_record_id,
                        ),
                        *_duplicate_ids(
                            plan_records,
                            "planned_queue_index",
                            id_func=_plan_record_id,
                        ),
                    )
                    if plan_status == REQUIRED_QUEUE_ORDERING_PLAN_AVAILABLE_STATUS
                    or plan_records
                    else ()
                ),
            )
        )
    )
    if not prior_consumer or not queue_ordering_plan:
        invalid_record_ids: Tuple[str, ...] = ()
    else:
        invalid_record_ids = raw_invalid_record_ids

    if not prior_consumer or not queue_ordering_plan:
        status = "sequence_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer_not_ready"
        blocked_by = (
            "promotion_ordering_preview_result_consumer_and_queue_ordering_plan_required",
        )
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_memory_trust_promotion_queue_ordering_plan_evidence",
        )
    elif prior_status != REQUIRED_ORDERING_PREVIEW_RESULT_CONSUMER_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer_blocked"
        blocked_by = (
            "promotion_ordering_preview_result_consumer_must_request_queue_ordering_plan_first",
        )
    elif plan_status != REQUIRED_QUEUE_ORDERING_PLAN_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer_blocked"
        blocked_by = (
            "sequence_memory_trust_promotion_queue_ordering_plan_required_before_queue_importer",
        )
    elif not plan_records:
        status = "sequence_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer_not_ready"
        blocked_by = ("sequence_memory_trust_promotion_queue_ordering_plan_records_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_request_available"
        blocked_by = (
            "operator_may_run_read_only_sequence_memory_trust_promotion_queue_importer_dry_run",
        )

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_request_available"
    )
    queue_importer_request: Dict[str, Any] = {
        "handler": "import_operator_proposal_apply_sequence_policy_queue_ordering_plan",
        "read_only_template": True,
        "promotion_queue_ordering_plan_result_consumer_only": True,
        "promotion_queue_importer_request_template_only": True,
        "promotion_queue_importer_handler_invoked": False,
        "promotion_queue_importer_created": False,
        "proposal_payloads_required_from_operator": True,
        "apply": False,
        "operator_ack": False,
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
        "can_import_to_queue": False,
        "can_apply_live": False,
    }
    if request_available:
        queue_importer_request["queue_ordering_plan"] = queue_ordering_plan
        queue_importer_request["promotion_acceptance_record_templates"] = list(
            promotion_acceptance_templates
        )
        queue_importer_request["memory_trust_review_request_specs"] = list(
            memory_trust_review_request_specs
        )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ORDERING_PREVIEW_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_queue_ordering_plan_result_consumer_only": True,
            "promotion_queue_importer_request_template_only": True,
            "requires_promotion_ordering_preview_result_consumer": True,
            "requires_promotion_queue_ordering_plan_result": True,
            "requires_queue_ordering_plan_records": True,
            "promotion_queue_importer_handler_invoked": False,
            "promotion_queue_importer_created": False,
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
            "can_import_to_queue": False,
            "can_apply_live": False,
        },
        "summary": {
            "expected_ordering_preview_record_count": len(ordering_rows),
            "queue_ordering_plan_record_count": len(plan_records),
            "ranked_queue_candidate_count": len(ranked_records),
            "unranked_queue_candidate_count": len(unranked_records),
            "promotion_acceptance_record_template_count": len(
                promotion_acceptance_templates
            ),
            "memory_trust_review_request_spec_count": len(
                memory_trust_review_request_specs
            ),
            "invalid_record_count": len(invalid_record_ids),
            "fresh_sequence_review_context_status": _source_sequence_review_freshness_status(
                promotion_acceptance_templates
            ),
            "promotion_queue_ordering_plan_result_consumer_only": True,
            "promotion_queue_importer_request_template_only": True,
            "promotion_queue_importer_request_available": request_available,
            "promotion_queue_importer_handler_invoked": False,
            "promotion_queue_importer_created": False,
            "requires_queue_ordering_plan_records": True,
            "proposal_payloads_required_from_operator": True,
            "apply": False,
            "operator_ack": False,
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
            "can_import_to_queue": False,
            "can_apply_live": False,
            "ordering_preview_proposal_ids": [
                str(row["proposal_id"])
                for row in ordering_rows
                if isinstance(row.get("proposal_id"), str) and row.get("proposal_id")
            ],
            "planned_queue_proposal_ids": [
                str(record["proposal_id"])
                for record in plan_records
                if isinstance(record.get("proposal_id"), str)
                and record.get("proposal_id")
            ],
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
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion ordering-preview result consumer and queue-ordering-plan result."
                if not prior_consumer or not queue_ordering_plan
                else (
                    "Replace invalid sequence-memory trust promotion queue-ordering-plan evidence before queue import."
                    if invalid_record_ids
                    else (
                        "Run the read-only queue-ordering plan before queue import."
                        if plan_status != REQUIRED_QUEUE_ORDERING_PLAN_AVAILABLE_STATUS
                        else "Run queue importer dry-run with matching proposal payloads; this consumer does not import queues, reserve idempotency, mutate policy, memory, model, or live state."
                    )
                )
            ),
        },
        "promotion_queue_importer_request": queue_importer_request,
    }
