"""Read-only consumer for promotion queue-importer results.

This verifier connects the queue-import promotion queue-ordering-plan result
consumer to the existing sequence-policy queue-import preflight planner. It
verifies importer results against the prior queue-ordering-plan provenance and
emits only a preflight-plan request template. It does not import queues, run
preflight, reserve idempotency, mutate proposal/policy/runtime/model/memory
state, execute sequences, or touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_STATUSES,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_STATUSES,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_IMPORTER_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_IMPORTER_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_queue_import_preflight_plan_request_available",
}
REQUIRED_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer"
)
REQUIRED_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_request_available"
)
REQUIRED_QUEUE_IMPORTER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer"
)
REQUIRED_QUEUE_IMPORTER_APPLIED_STATUS = (
    "sequence_memory_trust_promotion_queue_importer_applied"
)
REQUIRED_QUEUE_IMPORTER_APPLY_BLOCKED_STATUS = (
    "sequence_memory_trust_promotion_queue_importer_apply_blocked"
)
REQUIRED_PRIOR_FALSE_FLAGS = (
    "promotion_queue_importer_handler_invoked",
    "promotion_queue_importer_created",
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
REQUIRED_IMPORTER_NEVER_TRUE_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "proposal_order_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_importer",
    "sequence_policy_change_allowed_by_importer",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
)
REQUIRED_RESULT_FALSE_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_importer",
    "sequence_policy_change_allowed_by_importer",
    "live_widening_allowed",
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


def _queue_ordering_plan_from_prior_consumer(
    consumer: Mapping[str, Any],
) -> Mapping[str, Any]:
    request = _mapping(consumer.get("promotion_queue_importer_request"))
    return _mapping(request.get("queue_ordering_plan"))


def _acceptance_templates_from_prior_consumer_request(
    consumer: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    request = _mapping(consumer.get("promotion_queue_importer_request"))
    return _record_list(request.get("promotion_acceptance_record_templates"))


def _request_specs_from_prior_consumer_request(
    consumer: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    request = _mapping(consumer.get("promotion_queue_importer_request"))
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


def _proposal_from_import_result(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(result.get("operator_proposal"))


def _sequence_import_metadata(proposal: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = _mapping(proposal.get("raw"))
    return _mapping(raw.get("sequence_policy_queue_import")) or _mapping(
        proposal.get("sequence_policy_queue_import")
    )


def _prior_consumer_envelope_violations(
    consumer: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_TYPE:
        violations.append("queue_importer_result_consumer_prior_type_mismatch")
    generated_at = consumer.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("queue_importer_result_consumer_prior_generated_at_invalid")
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append("queue_importer_result_consumer_prior_method_id_mismatch")
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_STATUSES
    ):
        violations.append("queue_importer_result_consumer_prior_status_unknown")
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("queue_importer_result_consumer_prior_read_only_not_true")
    if safety.get("promotion_queue_ordering_plan_result_consumer_only") is not True:
        violations.append("queue_importer_result_consumer_prior_only_not_true")
    if safety.get("promotion_queue_importer_request_template_only") is not True:
        violations.append("queue_importer_result_consumer_prior_template_only_not_true")
    for flag in REQUIRED_PRIOR_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"queue_importer_result_consumer_prior_{flag}_not_false")
    return tuple(violations)


def _prior_consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    plan_records: Tuple[Mapping[str, Any], ...],
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
    memory_trust_review_request_specs: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "queue_ordering_plan_record_count": len(plan_records),
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
            violations.append(f"queue_importer_result_consumer_prior_summary_{key}_mismatch")
    if summary.get("promotion_queue_importer_request_available") is not bool(plan_records):
        violations.append("queue_importer_result_consumer_prior_summary_request_mismatch")
    if summary.get("promotion_queue_ordering_plan_result_consumer_only") is not True:
        violations.append("queue_importer_result_consumer_prior_summary_only_not_true")
    if summary.get("promotion_queue_importer_request_template_only") is not True:
        violations.append("queue_importer_result_consumer_prior_summary_template_not_true")
    for flag in REQUIRED_PRIOR_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"queue_importer_result_consumer_prior_summary_{flag}_not_false"
            )
    expected_ids = tuple(
        str(record["proposal_id"])
        for record in plan_records
        if isinstance(record.get("proposal_id"), str) and record.get("proposal_id")
    )
    if _string_list(summary.get("planned_queue_proposal_ids")) != expected_ids:
        violations.append("queue_importer_result_consumer_prior_summary_plan_ids_mismatch")
    if (
        promotion_acceptance_templates
        and summary.get("fresh_sequence_review_context_status") != "fresh"
    ):
        violations.append(
            "queue_importer_result_consumer_prior_summary_freshness_not_fresh"
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
            "queue_importer_result_consumer_prior_summary_template_proposal_ids_mismatch"
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
            "queue_importer_result_consumer_prior_summary_review_request_spec_proposal_ids_mismatch"
        )
    return tuple(violations)


def _prior_consumer_request_violations(
    request: Mapping[str, Any],
    *,
    plan_records: Tuple[Mapping[str, Any], ...],
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
    memory_trust_review_request_specs: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    if request.get("handler") != "import_operator_proposal_apply_sequence_policy_queue_ordering_plan":
        violations.append("queue_importer_result_consumer_prior_request_handler_mismatch")
    if request.get("read_only_template") is not True:
        violations.append("queue_importer_result_consumer_prior_request_template_not_true")
    if request.get("proposal_payloads_required_from_operator") is not True:
        violations.append("queue_importer_result_consumer_prior_request_payloads_not_required")
    if request.get("apply") is not False:
        violations.append("queue_importer_result_consumer_prior_request_apply_not_false")
    if request.get("operator_ack") is not False:
        violations.append("queue_importer_result_consumer_prior_request_operator_ack_not_false")
    for flag in REQUIRED_PRIOR_FALSE_FLAGS:
        if flag in request and request.get(flag) is not False:
            violations.append(
                f"queue_importer_result_consumer_prior_request_{flag}_not_false"
            )
    if plan_records and not _mapping(request.get("queue_ordering_plan")):
        violations.append("queue_importer_result_consumer_prior_request_plan_missing")
    plan_proposal_ids = {
        str(record["proposal_id"])
        for record in plan_records
        if record.get("rank_source") == "sequence_policy_consumer"
        and isinstance(record.get("proposal_id"), str)
        and record.get("proposal_id")
    }
    template_proposal_ids = {
        str(template["proposal_id"])
        for template in promotion_acceptance_templates
        if isinstance(template.get("proposal_id"), str) and template.get("proposal_id")
    }
    if len(template_proposal_ids) != len(promotion_acceptance_templates):
        violations.append(
            "queue_importer_result_consumer_prior_request_acceptance_template_proposal_ids_not_unique"
        )
    if plan_proposal_ids and not plan_proposal_ids.issubset(template_proposal_ids):
        violations.append(
            "queue_importer_result_consumer_prior_request_acceptance_templates_missing_plan_record"
        )
    request_spec_proposal_ids = {
        str(spec["proposal_id"])
        for spec in memory_trust_review_request_specs
        if isinstance(spec.get("proposal_id"), str) and spec.get("proposal_id")
    }
    if not template_proposal_ids.issubset(request_spec_proposal_ids):
        violations.append(
            "queue_importer_result_consumer_prior_request_review_request_spec_proposal_ids_mismatch"
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


def _queue_importer_envelope_violations(
    importer: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    if importer.get("type") != REQUIRED_QUEUE_IMPORTER_TYPE:
        violations.append("queue_importer_result_consumer_importer_type_mismatch")
    generated_at = importer.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("queue_importer_result_consumer_importer_generated_at_invalid")
    if (
        importer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID
    ):
        violations.append("queue_importer_result_consumer_importer_method_id_mismatch")
    if (
        importer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_STATUSES
    ):
        violations.append("queue_importer_result_consumer_importer_status_unknown")
    return tuple(violations)


def _queue_importer_safety_violations(
    safety: Mapping[str, Any],
    *,
    importer_status: str,
) -> Tuple[str, ...]:
    applied = importer_status == REQUIRED_QUEUE_IMPORTER_APPLIED_STATUS
    violations = []
    if safety.get("read_only") is not (not applied):
        violations.append("queue_importer_result_consumer_importer_read_only_mismatch")
    if safety.get("operator_required") is not True:
        violations.append("queue_importer_result_consumer_importer_operator_required_not_true")
    if safety.get("sequence_consumer_required_before_execution") is not True:
        violations.append("queue_importer_result_consumer_importer_sequence_consumer_not_true")
    if safety.get("queue_import_dry_run") is not (not applied):
        violations.append("queue_importer_result_consumer_importer_dry_run_mismatch")
    if safety.get("queue_import_applied") is not applied:
        violations.append("queue_importer_result_consumer_importer_applied_mismatch")
    if not isinstance(safety.get("queue_mutation"), bool):
        violations.append("queue_importer_result_consumer_importer_queue_mutation_mismatch")
    if applied and safety.get("operator_ack") is not True:
        violations.append("queue_importer_result_consumer_importer_ack_not_true")
    if applied and safety.get("apply_attempted") is not True:
        violations.append("queue_importer_result_consumer_importer_apply_attempted_not_true")
    for flag in REQUIRED_IMPORTER_NEVER_TRUE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"queue_importer_result_consumer_importer_{flag}_not_false")
    return tuple(violations)


def _queue_importer_summary_violations(
    summary: Mapping[str, Any],
    *,
    importer_status: str,
    plan_records: Tuple[Mapping[str, Any], ...],
    specs: Tuple[Mapping[str, Any], ...],
    results: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    applied = importer_status == REQUIRED_QUEUE_IMPORTER_APPLIED_STATUS
    violations = []
    expected_counts = {
        "queue_ordering_plan_record_count": len(plan_records),
        "queue_import_spec_count": len(specs),
        "queue_import_result_count": len(results),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"queue_importer_result_consumer_importer_summary_{key}_mismatch")
    if summary.get("queue_import_dry_run") is not (not applied):
        violations.append("queue_importer_result_consumer_importer_summary_dry_run_mismatch")
    if summary.get("queue_import_applied") is not applied:
        violations.append("queue_importer_result_consumer_importer_summary_applied_mismatch")
    if not isinstance(summary.get("queue_mutation"), bool):
        violations.append("queue_importer_result_consumer_importer_summary_queue_mutation_mismatch")
    if applied and summary.get("operator_ack") is not True:
        violations.append("queue_importer_result_consumer_importer_summary_ack_not_true")
    if applied and summary.get("apply_requested") is not True:
        violations.append("queue_importer_result_consumer_importer_summary_apply_not_true")
    for flag in REQUIRED_IMPORTER_NEVER_TRUE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"queue_importer_result_consumer_importer_summary_{flag}_not_false"
            )
    expected_ids = tuple(
        str(record["proposal_id"])
        for record in plan_records
        if isinstance(record.get("proposal_id"), str) and record.get("proposal_id")
    )
    imported_ids = tuple(
        str(result["proposal_id"])
        for result in results
        if result.get("queue_import_applied") is True
        and isinstance(result.get("proposal_id"), str)
        and result.get("proposal_id")
    )
    queue_mutation_ids = tuple(
        str(result["proposal_id"])
        for result in results
        if result.get("queue_mutation") is True
        and isinstance(result.get("proposal_id"), str)
        and result.get("proposal_id")
    )
    idempotent_noop_ids = tuple(
        str(result["proposal_id"])
        for result in results
        if result.get("status") == "exists"
        and result.get("success") is True
        and result.get("queue_mutation") is False
        and isinstance(result.get("proposal_id"), str)
        and result.get("proposal_id")
    )
    blocked_import_ids = tuple(
        str(result["proposal_id"])
        for result in results
        if result.get("success") is not True
        and isinstance(result.get("proposal_id"), str)
        and result.get("proposal_id")
    )
    identity_violation_count = sum(
        len(_string_list(result.get("identity_violations")))
        for result in results
    )
    if _string_list(summary.get("planned_import_proposal_ids")) != expected_ids:
        violations.append("queue_importer_result_consumer_importer_summary_planned_ids_mismatch")
    if _summary_int(summary.get("imported_proposal_count")) != len(imported_ids):
        violations.append("queue_importer_result_consumer_importer_summary_imported_count_mismatch")
    if _string_list(summary.get("imported_proposal_ids")) != imported_ids:
        violations.append("queue_importer_result_consumer_importer_summary_imported_ids_mismatch")
    if summary.get("queue_mutation") is not bool(queue_mutation_ids):
        violations.append("queue_importer_result_consumer_importer_summary_queue_mutation_mismatch")
    if _summary_int(summary.get("queue_mutation_count")) != len(queue_mutation_ids):
        violations.append("queue_importer_result_consumer_importer_summary_queue_mutation_count_mismatch")
    if _string_list(summary.get("queue_mutation_proposal_ids")) != queue_mutation_ids:
        violations.append("queue_importer_result_consumer_importer_summary_queue_mutation_ids_mismatch")
    if _summary_int(summary.get("idempotent_import_noop_count")) != len(idempotent_noop_ids):
        violations.append("queue_importer_result_consumer_importer_summary_idempotent_noop_count_mismatch")
    if _string_list(summary.get("idempotent_import_noop_proposal_ids")) != idempotent_noop_ids:
        violations.append("queue_importer_result_consumer_importer_summary_idempotent_noop_ids_mismatch")
    if _summary_int(summary.get("blocked_import_count")) != len(blocked_import_ids):
        violations.append("queue_importer_result_consumer_importer_summary_blocked_import_count_mismatch")
    if _string_list(summary.get("blocked_import_proposal_ids")) != blocked_import_ids:
        violations.append("queue_importer_result_consumer_importer_summary_blocked_import_ids_mismatch")
    if _summary_int(summary.get("identity_violation_count")) != identity_violation_count:
        violations.append("queue_importer_result_consumer_importer_summary_identity_violation_count_mismatch")
    if applied and any(result.get("success") is not True for result in results):
        violations.append("queue_importer_result_consumer_importer_summary_applied_result_failure")
    if (
        importer_status == REQUIRED_QUEUE_IMPORTER_APPLY_BLOCKED_STATUS
        and results
        and not blocked_import_ids
    ):
        violations.append("queue_importer_result_consumer_importer_summary_blocked_result_missing")
    if (
        not applied
        and importer_status != REQUIRED_QUEUE_IMPORTER_APPLY_BLOCKED_STATUS
        and results
    ):
        violations.append("queue_importer_result_consumer_importer_summary_unexpected_results_for_status")
    if applied and _string_list(summary.get("imported_proposal_ids")) != expected_ids:
        violations.append("queue_importer_result_consumer_importer_summary_imported_ids_mismatch")
    return tuple(violations)


def _spec_violations(
    spec: Mapping[str, Any],
    *,
    plan_record: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    expected = {
        "queue_ordering_plan_record_id": plan_record.get("queue_ordering_plan_record_id"),
        "proposal_id": plan_record.get("proposal_id"),
        "planned_queue_index": plan_record.get("planned_queue_index"),
        "rank_source": plan_record.get("rank_source"),
        "source_idempotency_key": plan_record.get("source_idempotency_key"),
        "current_sequence_review_freshness": plan_record.get(
            "current_sequence_review_freshness"
        ),
    }
    for key, value in expected.items():
        if spec.get(key) != value:
            violations.append(f"queue_importer_result_consumer_spec_{key}_mismatch")
    violations.extend(
        _context_violations(
            "queue_importer_result_consumer_spec",
            _mapping(spec.get("current_sequence_review_freshness")),
        )
    )
    if spec.get("operator_ack_required_before_queue_import") is not True:
        violations.append("queue_importer_result_consumer_spec_queue_ack_not_true")
    if spec.get("operator_ack_required_before_sequence_execution") is not True:
        violations.append("queue_importer_result_consumer_spec_sequence_ack_not_true")
    if spec.get("sequence_consumer_required_before_execution") is not True:
        violations.append("queue_importer_result_consumer_spec_sequence_consumer_not_true")
    if spec.get("queue_import_dry_run") is not True:
        violations.append("queue_importer_result_consumer_spec_dry_run_not_true")
    if spec.get("queue_import_applied") is not False:
        violations.append("queue_importer_result_consumer_spec_applied_not_false")
    for flag in (
        "queue_mutation",
        "proposal_order_mutation",
        "mutates_console_state",
        "queries_console",
        "model_mutation",
        "runtime_memory_mutation",
        "sequence_policy_trust_store_mutation",
        "proposal_state_mutation",
        "sequence_execution_allowed_by_importer",
        "sequence_policy_change_allowed_by_importer",
        "live_widening_allowed",
        "live_write",
        "live_authority_granted",
        "full_autonomous_ready",
        "can_import_to_queue",
        "can_apply_live",
    ):
        if spec.get(flag) is not False:
            violations.append(f"queue_importer_result_consumer_spec_{flag}_not_false")
    payload = _mapping(spec.get("operator_proposal_payload"))
    metadata = _mapping(payload.get("sequence_policy_queue_import"))
    if metadata.get("planned_queue_index") != plan_record.get("planned_queue_index"):
        violations.append("queue_importer_result_consumer_spec_payload_index_mismatch")
    if metadata.get("source_idempotency_key") != plan_record.get("source_idempotency_key"):
        violations.append(
            "queue_importer_result_consumer_spec_payload_source_idempotency_key_mismatch"
        )
    if metadata.get("current_sequence_review_freshness") != spec.get(
        "current_sequence_review_freshness"
    ):
        violations.append(
            "queue_importer_result_consumer_spec_payload_current_sequence_review_freshness_mismatch"
        )
    violations.extend(
        _context_violations(
            "queue_importer_result_consumer_spec_payload",
            _mapping(metadata.get("current_sequence_review_freshness")),
        )
    )
    if metadata.get("live_write") is not False:
        violations.append("queue_importer_result_consumer_spec_payload_live_write_not_false")
    if metadata.get("can_apply_live") is not False:
        violations.append("queue_importer_result_consumer_spec_payload_can_apply_live_not_false")
    return tuple(violations)


def _spec_template_identity_violations(
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
                violations.append(f"{proposal_id}:{field}_queue_import_spec_mismatch")
        if _sequence_review_freshness_core(
            _mapping(template.get("current_sequence_review_freshness"))
        ) != _sequence_review_freshness_core(
            _mapping(spec.get("current_sequence_review_freshness"))
        ):
            violations.append(
                f"{proposal_id}:current_sequence_review_freshness_queue_import_spec_mismatch"
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


def _result_violations(
    result: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    importer_status: str,
) -> Tuple[str, ...]:
    applied = importer_status == REQUIRED_QUEUE_IMPORTER_APPLIED_STATUS
    violations = []
    expected = {
        "queue_import_spec_id": spec.get("queue_import_spec_id"),
        "proposal_id": spec.get("proposal_id"),
        "planned_queue_index": spec.get("planned_queue_index"),
    }
    for key, value in expected.items():
        if result.get(key) != value:
            violations.append(f"queue_importer_result_consumer_result_{key}_mismatch")
    success = result.get("success") is True
    if not success and applied:
        violations.append("queue_importer_result_consumer_result_success_not_true")
    if not success:
        if result.get("queue_import_applied") is not False:
            violations.append("queue_importer_result_consumer_result_applied_not_false")
        if result.get("queue_mutation") is not False:
            violations.append("queue_importer_result_consumer_result_queue_mutation_not_false")
        for flag in REQUIRED_RESULT_FALSE_FLAGS:
            if result.get(flag) is not False:
                violations.append(f"queue_importer_result_consumer_result_{flag}_not_false")
        return tuple(violations)
    if result.get("queue_import_applied") is not True:
        violations.append("queue_importer_result_consumer_result_applied_not_true")
    if result.get("status") == "exists":
        if result.get("queue_mutation") is not False:
            violations.append("queue_importer_result_consumer_result_queue_mutation_not_false")
    elif result.get("queue_mutation") is not True:
        violations.append("queue_importer_result_consumer_result_queue_mutation_not_true")
    if _string_list(result.get("identity_violations")):
        violations.append("queue_importer_result_consumer_result_identity_violations_present")
    if _string_list(result.get("blocked_by")):
        violations.append("queue_importer_result_consumer_result_blocked_by_present")
    for flag in REQUIRED_RESULT_FALSE_FLAGS:
        if result.get(flag) is not False:
            violations.append(f"queue_importer_result_consumer_result_{flag}_not_false")
    proposal = _proposal_from_import_result(result)
    if not proposal:
        violations.append("queue_importer_result_consumer_result_operator_proposal_missing")
        return tuple(violations)
    if proposal.get("id") != spec.get("proposal_id"):
        violations.append("queue_importer_result_consumer_result_proposal_id_mismatch")
    metadata = _sequence_import_metadata(proposal)
    if not metadata:
        violations.append("queue_importer_result_consumer_result_import_metadata_missing")
    elif metadata.get("planned_queue_index") != spec.get("planned_queue_index"):
        violations.append("queue_importer_result_consumer_result_import_index_mismatch")
    if metadata and metadata.get("source_idempotency_key") != spec.get("source_idempotency_key"):
        violations.append(
            "queue_importer_result_consumer_result_import_source_idempotency_key_mismatch"
        )
    if metadata and metadata.get("current_sequence_review_freshness") != spec.get(
        "current_sequence_review_freshness"
    ):
        violations.append(
            "queue_importer_result_consumer_result_import_current_sequence_review_freshness_mismatch"
        )
    if metadata:
        violations.extend(
            _context_violations(
                "queue_importer_result_consumer_result",
                _mapping(metadata.get("current_sequence_review_freshness")),
            )
        )
    if metadata and metadata.get("live_write") is not False:
        violations.append("metadata_live_write_not_false")
    if metadata and metadata.get("can_apply_live") is not False:
        violations.append("metadata_can_apply_live_not_false")
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer(
    *,
    promotion_queue_ordering_plan_result_consumer: Mapping[str, Any] | None = None,
    promotion_queue_importer_result: Mapping[str, Any] | None = None,
    idempotency_key_prefix: str = "sequence-policy-queue-import",
    limit: int = 20,
) -> Dict[str, Any]:
    """Verify queue-importer output and expose only a preflight-plan template."""

    prior = _mapping(promotion_queue_ordering_plan_result_consumer)
    importer = _mapping(promotion_queue_importer_result)
    prior_request = _mapping(prior.get("promotion_queue_importer_request"))
    queue_ordering_plan = _queue_ordering_plan_from_prior_consumer(prior)
    promotion_acceptance_templates = _acceptance_templates_from_prior_consumer_request(
        prior
    )
    memory_trust_review_request_specs = _request_specs_from_prior_consumer_request(
        prior
    )
    plan_records = _record_list(queue_ordering_plan.get("queue_ordering_plan_records"))
    specs = _record_list(importer.get("queue_import_specs"))
    results = _record_list(importer.get("queue_import_results"))
    plan_records_by_proposal_id = {
        str(record["proposal_id"]): record
        for record in plan_records
        if isinstance(record.get("proposal_id"), str) and record.get("proposal_id")
    }
    specs_by_proposal_id = {
        str(spec["proposal_id"]): spec
        for spec in specs
        if isinstance(spec.get("proposal_id"), str) and spec.get("proposal_id")
    }
    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_string_list(_mapping(prior.get("summary")).get("invalid_record_ids")),
                *_string_list(_mapping(importer.get("summary")).get("invalid_record_ids")),
                *_prior_consumer_envelope_violations(prior),
                *_prior_consumer_summary_violations(
                    _mapping(prior.get("summary")),
                    plan_records=plan_records,
                    promotion_acceptance_templates=promotion_acceptance_templates,
                    memory_trust_review_request_specs=memory_trust_review_request_specs,
                ),
                *_prior_consumer_request_violations(
                    prior_request,
                    plan_records=plan_records,
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
                *_queue_importer_envelope_violations(importer),
                *_queue_importer_safety_violations(
                    _mapping(importer.get("safety")),
                    importer_status=_string(importer.get("status")),
                ),
                *_queue_importer_summary_violations(
                    _mapping(importer.get("summary")),
                    importer_status=_string(importer.get("status")),
                    plan_records=plan_records,
                    specs=specs,
                    results=results,
                ),
                *(
                    f"{proposal_id}:queue_importer_result_consumer_spec_missing"
                    for proposal_id in plan_records_by_proposal_id
                    if proposal_id not in specs_by_proposal_id
                ),
                *(
                    f"{_string(spec.get('proposal_id'))}:"
                    f"{violation}"
                    for spec in specs
                    for violation in _spec_violations(
                        spec,
                        plan_record=plan_records_by_proposal_id.get(
                            _string(spec.get("proposal_id")),
                            {},
                        ),
                    )
                ),
                *_spec_template_identity_violations(
                    specs=specs,
                    promotion_acceptance_templates=promotion_acceptance_templates,
                    memory_trust_review_request_specs=memory_trust_review_request_specs,
                ),
                *(
                    f"{_string(result.get('proposal_id'))}:"
                    f"{violation}"
                    for result in results
                    for violation in _result_violations(
                        result,
                        spec=specs_by_proposal_id.get(
                            _string(result.get("proposal_id")),
                            {},
                        ),
                        importer_status=_string(importer.get("status")),
                    )
                ),
            )
        )
    )
    invalid_record_ids = () if not prior or not importer else raw_invalid_record_ids

    prior_status = _string(prior.get("status"))
    importer_status = _string(importer.get("status"))
    if not prior or not importer:
        status = "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_not_ready"
        blocked_by = ("promotion_queue_ordering_plan_result_consumer_and_queue_importer_result_required",)
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_memory_trust_promotion_queue_importer_evidence",
        )
    elif prior_status != REQUIRED_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_blocked"
        blocked_by = (
            "promotion_queue_ordering_plan_result_consumer_must_request_queue_importer_first",
        )
    elif importer_status != REQUIRED_QUEUE_IMPORTER_APPLIED_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_blocked"
        blocked_by = (
            "applied_sequence_policy_queue_importer_result_required_before_preflight_plan",
        )
    elif not results:
        status = "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_not_ready"
        blocked_by = ("applied_queue_import_results_required_before_preflight_plan",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_queue_import_preflight_plan_request_available"
        blocked_by = (
            "operator_may_run_read_only_sequence_policy_queue_import_preflight_plan",
        )

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_queue_import_preflight_plan_request_available"
    )
    preflight_plan_request: Dict[str, Any] = {
        "handler": "plan_operator_proposal_apply_sequence_policy_queue_import_preflight",
        "read_only_template": True,
        "promotion_queue_importer_result_consumer_only": True,
        "promotion_queue_import_preflight_plan_request_template_only": True,
        "promotion_preflight_plan_request_template_only": True,
        "pending_actions_required_from_operator_queue": True,
        "pending_actions_source": "operator_proposal_queue",
        "requires_applied_queue_import_result": True,
        "idempotency_key_prefix": str(
            idempotency_key_prefix or "sequence-policy-queue-import"
        ),
        "limit": max(1, min(int(limit or 20), 50)),
        "promotion_queue_import_preflight_plan_handler_invoked": False,
        "promotion_preflight_plan_handler_invoked": False,
        "promotion_queue_import_preflight_plan_created": False,
        "promotion_preflight_plan_created": False,
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
        "can_import_to_queue": False,
        "can_apply_live": False,
    }
    if request_available:
        preflight_plan_request["imported_proposal_ids"] = [
            str(result["proposal_id"])
            for result in results
            if result.get("queue_import_applied") is True
            and isinstance(result.get("proposal_id"), str)
            and result.get("proposal_id")
        ]
        preflight_plan_request["promotion_queue_importer_result"] = importer
        preflight_plan_request["promotion_acceptance_record_templates"] = list(
            promotion_acceptance_templates
        )
        preflight_plan_request["memory_trust_review_request_specs"] = list(
            memory_trust_review_request_specs
        )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_IMPORTER_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_queue_importer_result_consumer_only": True,
            "promotion_queue_import_preflight_plan_request_template_only": True,
            "promotion_preflight_plan_request_template_only": True,
            "requires_promotion_queue_ordering_plan_result_consumer": True,
            "requires_applied_queue_import_result": True,
            "promotion_queue_import_preflight_plan_handler_invoked": False,
            "promotion_preflight_plan_handler_invoked": False,
            "promotion_queue_import_preflight_plan_created": False,
            "promotion_preflight_plan_created": False,
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
            "can_import_to_queue": False,
            "can_apply_live": False,
        },
        "summary": {
            "queue_ordering_plan_record_count": len(plan_records),
            "expected_queue_ordering_plan_record_count": len(plan_records),
            "queue_import_spec_count": len(specs),
            "queue_import_result_count": len(results),
            "promotion_acceptance_record_template_count": len(
                promotion_acceptance_templates
            ),
            "memory_trust_review_request_spec_count": len(
                memory_trust_review_request_specs
            ),
            "fresh_sequence_review_context_status": _source_sequence_review_freshness_status(
                promotion_acceptance_templates
            ),
            "imported_proposal_count": len(
                tuple(
                    result
                    for result in results
                    if result.get("queue_import_applied") is True
                )
            ),
            "importer_dry_run_seen": importer_status
            == "sequence_memory_trust_promotion_queue_importer_dry_run_ready",
            "invalid_record_count": len(invalid_record_ids),
            "promotion_queue_importer_result_consumer_only": True,
            "promotion_queue_import_preflight_plan_request_template_only": True,
            "promotion_preflight_plan_request_template_only": True,
            "promotion_queue_import_preflight_plan_request_available": request_available,
            "promotion_preflight_plan_request_available": request_available,
            "promotion_queue_import_preflight_plan_handler_invoked": False,
            "promotion_preflight_plan_handler_invoked": False,
            "promotion_queue_import_preflight_plan_created": False,
            "promotion_preflight_plan_created": False,
            "batch_preflight_invoked": False,
            "sequence_plan_created": False,
            "requires_applied_queue_import_result": True,
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
            "planned_import_proposal_ids": [
                str(record["proposal_id"])
                for record in plan_records
                if isinstance(record.get("proposal_id"), str)
                and record.get("proposal_id")
            ],
            "imported_proposal_ids": [
                str(result["proposal_id"])
                for result in results
                if result.get("queue_import_applied") is True
                and isinstance(result.get("proposal_id"), str)
                and result.get("proposal_id")
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
                "Provide promotion queue-ordering-plan result consumer and queue-importer result."
                if not prior or not importer
                else (
                    "Replace invalid sequence-memory trust promotion queue-importer evidence before preflight planning."
                    if invalid_record_ids
                    else (
                        "Applied sequence-policy queue importer result with no blocked imports is required before preflight planning."
                        if importer_status != REQUIRED_QUEUE_IMPORTER_APPLIED_STATUS
                        else "Run the separate read-only preflight-plan handler against the operator queue; this consumer does not invoke preflight or reserve idempotency."
                    )
                )
            ),
        },
        "promotion_queue_import_preflight_plan_request": preflight_plan_request,
        "promotion_preflight_plan_request": preflight_plan_request,
    }
