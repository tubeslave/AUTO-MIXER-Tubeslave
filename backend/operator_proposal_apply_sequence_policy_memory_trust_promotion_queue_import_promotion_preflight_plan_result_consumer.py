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


def _preflight_plan_request_from_prior(
    consumer: Mapping[str, Any],
) -> Mapping[str, Any]:
    request = _mapping(consumer.get("promotion_queue_import_preflight_plan_request"))
    if request:
        return request
    return _mapping(consumer.get("promotion_preflight_plan_request"))


def _acceptance_templates_from_prior_request(
    request: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    return _record_list(request.get("promotion_acceptance_record_templates"))


def _request_specs_from_prior_request(
    request: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    return _record_list(request.get("memory_trust_review_request_specs"))


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
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
    memory_trust_review_request_specs: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    if _summary_int(summary.get("imported_proposal_count")) != len(expected_imported_ids):
        violations.append("promotion_preflight_plan_result_consumer_prior_summary_imported_count_mismatch")
    if _summary_int(summary.get("promotion_acceptance_record_template_count")) != len(
        promotion_acceptance_templates
    ):
        violations.append(
            "promotion_preflight_plan_result_consumer_prior_summary_template_count_mismatch"
        )
    if _summary_int(summary.get("memory_trust_review_request_spec_count")) != len(
        memory_trust_review_request_specs
    ):
        violations.append(
            "promotion_preflight_plan_result_consumer_prior_summary_review_request_spec_count_mismatch"
        )
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
    if (
        promotion_acceptance_templates
        and summary.get("fresh_sequence_review_context_status") != "fresh"
    ):
        violations.append(
            "promotion_preflight_plan_result_consumer_prior_summary_freshness_not_fresh"
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
            "promotion_preflight_plan_result_consumer_prior_summary_template_proposal_ids_mismatch"
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
            "promotion_preflight_plan_result_consumer_prior_summary_review_request_spec_proposal_ids_mismatch"
        )
    return tuple(violations)


def _prior_request_violations(
    request: Mapping[str, Any],
    *,
    expected_imported_ids: Tuple[str, ...],
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
    memory_trust_review_request_specs: Tuple[Mapping[str, Any], ...],
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
    template_proposal_ids = {
        str(template["proposal_id"])
        for template in promotion_acceptance_templates
        if isinstance(template.get("proposal_id"), str) and template.get("proposal_id")
    }
    if len(template_proposal_ids) != len(promotion_acceptance_templates):
        violations.append(
            "promotion_preflight_plan_result_consumer_prior_request_acceptance_template_proposal_ids_not_unique"
        )
    if expected_imported_ids and not set(expected_imported_ids).issubset(
        template_proposal_ids
    ):
        violations.append(
            "promotion_preflight_plan_result_consumer_prior_request_acceptance_templates_missing_imported_proposal"
        )
    request_spec_proposal_ids = {
        str(spec["proposal_id"])
        for spec in memory_trust_review_request_specs
        if isinstance(spec.get("proposal_id"), str) and spec.get("proposal_id")
    }
    if not template_proposal_ids.issubset(request_spec_proposal_ids):
        violations.append(
            "promotion_preflight_plan_result_consumer_prior_request_review_request_spec_proposal_ids_mismatch"
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


def _preflight_spec_template_identity_violations(
    *,
    specs: Tuple[Mapping[str, Any], ...],
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
    memory_trust_review_request_specs: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    templates_by_proposal_id = _templates_by_proposal_id(promotion_acceptance_templates)
    request_specs_by_proposal_id = _request_specs_by_proposal_id(
        memory_trust_review_request_specs
    )
    violations = []
    for spec in specs:
        if spec.get("rank_source") != "sequence_policy_consumer":
            continue
        proposal_id = _string(spec.get("proposal_id"))
        template = _mapping(templates_by_proposal_id.get(proposal_id))
        if not template:
            violations.append(f"{proposal_id}:acceptance_template_missing")
            continue
        expected_pairs = {
            "proposal_id": spec.get("proposal_id"),
            "idempotency_key": spec.get("source_idempotency_key"),
            "planned_queue_index": spec.get("planned_queue_index"),
            "rank_source": spec.get("rank_source"),
        }
        for field, expected in expected_pairs.items():
            if template.get(field) != expected:
                violations.append(f"{proposal_id}:{field}_preflight_spec_mismatch")
        if _sequence_review_freshness_core(
            _mapping(template.get("current_sequence_review_freshness"))
        ) != _sequence_review_freshness_core(
            _mapping(spec.get("current_sequence_review_freshness"))
        ):
            violations.append(
                f"{proposal_id}:current_sequence_review_freshness_preflight_spec_mismatch"
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
    promotion_acceptance_templates = _acceptance_templates_from_prior_request(
        prior_request
    )
    memory_trust_review_request_specs = _request_specs_from_prior_request(
        prior_request
    )
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
                    promotion_acceptance_templates=promotion_acceptance_templates,
                    memory_trust_review_request_specs=memory_trust_review_request_specs,
                ),
                *_prior_request_violations(
                    prior_request,
                    expected_imported_ids=expected_imported_ids,
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
                *_preflight_spec_template_identity_violations(
                    specs=preflight_specs,
                    promotion_acceptance_templates=promotion_acceptance_templates,
                    memory_trust_review_request_specs=memory_trust_review_request_specs,
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
        promotion_batch_preflight_request["promotion_acceptance_record_templates"] = (
            list(promotion_acceptance_templates)
        )
        promotion_batch_preflight_request["memory_trust_review_request_specs"] = list(
            memory_trust_review_request_specs
        )

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
            "promotion_acceptance_record_template_count": len(
                promotion_acceptance_templates
            ),
            "memory_trust_review_request_spec_count": len(
                memory_trust_review_request_specs
            ),
            "fresh_sequence_review_context_status": _source_sequence_review_freshness_status(
                promotion_acceptance_templates
            ),
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
