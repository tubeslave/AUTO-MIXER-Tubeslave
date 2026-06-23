"""Read-only consumer for sequence-policy promotion executor dry runs.

This verifier connects a queue-import promotion execution-plan result consumer
to the existing operator-gated promotion executor. It verifies dry-run policy
specs against execution-plan candidates and emits only an executor request
template. It does not invoke the executor, apply sequence policy, mutate queues,
policy stores, runtime memory, models, append-only memory, proposal state,
idempotency, or mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_STATUSES,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_PLAN_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_PLAN_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_STATUSES,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_PLAN_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_PLAN_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_DRY_RUN_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_DRY_RUN_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_executor_request_available",
}
REQUIRED_EXECUTION_PLAN_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer"
)
REQUIRED_EXECUTION_PLAN_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_executor_dry_run_request_available"
)
REQUIRED_EXECUTOR_DRY_RUN_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run"
)
REQUIRED_EXECUTOR_DRY_RUN_READY_STATUS = (
    "sequence_memory_trust_promotion_executor_dry_run_ready"
)
REQUIRED_EXECUTION_PLAN_READY_STATUS = (
    "manual_sequence_memory_trust_promotion_execution_candidate_available"
)
TARGET_POLICY_STORE = "operator_sequence_policy_trust_gate"
REQUIRED_CONSUMER_FALSE_FLAGS = (
    "promotion_executor_dry_run_handler_invoked",
    "promotion_executor_dry_run_created",
    "memory_write_allowed_by_consumer",
    "execution_allowed_by_consumer",
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "policy_mutation",
    "memory_mutation",
    "memory_file_mutation",
    "runtime_memory_mutation",
    "model_mutation",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_apply_live",
)
REQUIRED_REQUEST_FALSE_FLAGS = (
    "memory_write_allowed_by_consumer",
    "execution_allowed_by_consumer",
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "policy_mutation",
    "memory_mutation",
    "memory_file_mutation",
    "runtime_memory_mutation",
    "model_mutation",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_apply_live",
)
REQUIRED_DRY_RUN_FALSE_SAFETY_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "execution_allowed_by_dry_run",
    "sequence_policy_promotion_applied",
    "sequence_policy_promotion_allowed_by_dry_run",
    "queue_mutation_allowed_by_dry_run",
    "policy_mutation_allowed_by_dry_run",
    "model_mutation_allowed_by_dry_run",
    "runtime_memory_mutation_allowed_by_dry_run",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_DRY_RUN_FALSE_SUMMARY_FLAGS = (
    "execution_allowed_by_dry_run",
    "sequence_policy_promotion_applied",
    "sequence_policy_promotion_allowed_by_dry_run",
    "queue_mutation_allowed_by_dry_run",
    "policy_mutation_allowed_by_dry_run",
    "model_mutation_allowed_by_dry_run",
    "runtime_memory_mutation_allowed_by_dry_run",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_DRY_RUN_RECORD_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "execution_allowed_by_dry_run",
    "sequence_policy_promotion_applied",
    "sequence_policy_promotion_allowed_by_dry_run",
    "queue_mutation_allowed_by_dry_run",
    "policy_mutation_allowed_by_dry_run",
    "model_mutation_allowed_by_dry_run",
    "runtime_memory_mutation_allowed_by_dry_run",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_DRY_RUN_RECORD_FIELDS = (
    "promotion_executor_dry_run_id",
    "promotion_execution_candidate_id",
    "promotion_acceptance_id",
    "promotion_contract_record_id",
    "promotion_candidate_id",
    "proposal_id",
    "learning_scope",
    "value_type",
    "source_idempotency_key",
    "target_policy_store",
    "target_policy_path",
    "operation",
    "previous_policy_state",
    "target_policy_state",
    "rollback_strategy",
    "idempotency_key",
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


def _current_sequence_review_freshness_violations(
    proposal_id: str,
    freshness: Mapping[str, Any],
) -> Tuple[str, ...]:
    if not freshness:
        return (f"{proposal_id}:current_sequence_review_freshness_missing",)

    violations = []
    if freshness.get("status") != "fresh":
        violations.append(f"{proposal_id}:current_sequence_review_freshness_not_fresh")
    if freshness.get("read_only") is not True:
        violations.append(f"{proposal_id}:current_sequence_review_freshness_read_only_not_true")
    for flag in REQUIRED_SEQUENCE_REVIEW_FRESHNESS_FALSE_FLAGS:
        if freshness.get(flag) is not False:
            violations.append(f"{proposal_id}:current_sequence_review_freshness_{flag}_not_false")
    return tuple(violations)


def _source_sequence_review_freshness_status(
    templates: Tuple[Mapping[str, Any], ...],
) -> str:
    if not templates:
        return "missing"
    invalid = False
    missing = False
    for index, template in enumerate(templates):
        template_id = _string(template.get("promotion_acceptance_record_template_id")) or (
            f"promotion_acceptance_template:{index}"
        )
        proposal_id = _string(template.get("proposal_id")) or template_id
        freshness = _mapping(template.get("current_sequence_review_freshness"))
        violations = _current_sequence_review_freshness_violations(proposal_id, freshness)
        if not freshness:
            missing = True
        elif violations:
            invalid = True
    if not missing and not invalid:
        return "fresh"
    return "invalid" if invalid else "missing"


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


def _record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("promotion_executor_dry_run_id"))
    if explicit:
        return explicit
    candidate_id = _string(record.get("promotion_execution_candidate_id"))
    return f"{candidate_id if candidate_id else 'unknown_sequence_executor_dry_run'}:{index}"


def _target_policy_path(record: Mapping[str, Any]) -> str:
    proposal_id = _string(record.get("proposal_id")) or "unknown_proposal"
    source_key = _string(record.get("source_idempotency_key")) or "unknown_idempotency"
    return f"{TARGET_POLICY_STORE}/{proposal_id}/{source_key}/trust_state"


def _idempotency_key(record: Mapping[str, Any]) -> str:
    return ":".join(
        (
            "operator_sequence_policy_memory_trust_promotion",
            str(record.get("proposal_id") or "unknown_proposal"),
            str(record.get("source_idempotency_key") or "unknown_idempotency"),
            str(record.get("promotion_acceptance_id") or "unknown_acceptance"),
        )
    )


def _records_by_candidate_id(
    records: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(record["promotion_execution_candidate_id"]): record
        for record in records
        if isinstance(record.get("promotion_execution_candidate_id"), str)
        and record.get("promotion_execution_candidate_id")
    }


def _candidates_from_request(request: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    plan = _mapping(request.get("promotion_execution_plan"))
    return _record_list(plan.get("execution_candidates"))


def _acceptance_templates_from_request(
    request: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    return _record_list(request.get("promotion_acceptance_record_templates"))


def _templates_by_contract_record_id(
    templates: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(template["promotion_contract_record_id"]): template
        for template in templates
        if isinstance(template.get("promotion_contract_record_id"), str)
        and template.get("promotion_contract_record_id")
    }


def _consumer_envelope_violations(consumer: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_EXECUTION_PLAN_RESULT_CONSUMER_TYPE:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_execution_plan_consumer_type_mismatch"
        )
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_PLAN_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_execution_plan_consumer_method_id_mismatch"
        )
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_PLAN_RESULT_CONSUMER_STATUSES
    ):
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_execution_plan_consumer_status_unknown"
        )
    if consumer.get("status") != REQUIRED_EXECUTION_PLAN_RESULT_CONSUMER_AVAILABLE_STATUS:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_execution_plan_consumer_status_not_available"
        )
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_execution_plan_consumer_read_only_not_true"
        )
    if safety.get("promotion_execution_plan_result_consumer_only") is not True:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_execution_plan_consumer_only_not_true"
        )
    if safety.get("promotion_executor_dry_run_request_template_only") is not True:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_execution_plan_consumer_template_only_not_true"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_executor_dry_run_result_consumer_execution_plan_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "expected_execution_candidate_count": len(candidates),
        "execution_candidate_count": len(candidates),
        "promotion_acceptance_record_template_count": len(
            promotion_acceptance_templates
        ),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_promotion_executor_dry_run_result_consumer_execution_plan_consumer_summary_{key}_mismatch"
            )
    if summary.get("promotion_executor_dry_run_request_available") is not bool(candidates):
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_execution_plan_consumer_request_available_mismatch"
        )
    if (
        promotion_acceptance_templates
        and summary.get("fresh_sequence_review_context_status") != "fresh"
    ):
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_execution_plan_consumer_summary_freshness_not_fresh"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_executor_dry_run_result_consumer_execution_plan_consumer_summary_{flag}_not_false"
            )
    expected_proposal_ids = tuple(
        str(candidate["proposal_id"])
        for candidate in candidates
        if isinstance(candidate.get("proposal_id"), str) and candidate.get("proposal_id")
    )
    if _string_list(summary.get("accepted_proposal_ids")) != expected_proposal_ids:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_execution_plan_consumer_summary_accepted_proposal_ids_mismatch"
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
            "sequence_policy_promotion_executor_dry_run_result_consumer_execution_plan_consumer_summary_template_proposal_ids_mismatch"
        )
    return tuple(violations)


def _consumer_request_violations(
    request: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    if (
        request.get("handler")
        != "dry_run_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor"
    ):
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_request_handler_mismatch"
        )
    if request.get("read_only_template") is not True:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_request_read_only_template_not_true"
        )
    if request.get("promotion_executor_dry_run_request_template_only") is not True:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_request_template_only_not_true"
        )
    if request.get("promotion_executor_dry_run_handler_invoked") is not False:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_request_handler_invoked_not_false"
        )
    if request.get("promotion_executor_dry_run_created") is not False:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_request_created_not_false"
        )
    if request.get("apply") is not False:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_request_apply_not_false"
        )
    if request.get("operator_ack") is not False:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_request_operator_ack_not_false"
        )
    plan = _mapping(request.get("promotion_execution_plan"))
    if plan.get("status") != REQUIRED_EXECUTION_PLAN_READY_STATUS:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_request_execution_plan_not_ready"
        )
    candidate_contract_ids = {
        str(candidate["promotion_contract_record_id"])
        for candidate in candidates
        if isinstance(candidate.get("promotion_contract_record_id"), str)
        and candidate.get("promotion_contract_record_id")
    }
    template_contract_ids = {
        str(template["promotion_contract_record_id"])
        for template in promotion_acceptance_templates
        if isinstance(template.get("promotion_contract_record_id"), str)
        and template.get("promotion_contract_record_id")
    }
    if len(template_contract_ids) != len(promotion_acceptance_templates):
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_request_acceptance_template_contract_ids_not_unique"
        )
    if candidate_contract_ids and not candidate_contract_ids.issubset(
        template_contract_ids
    ):
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_request_acceptance_templates_missing_execution_candidate"
        )
    request_specs = _record_list(request.get("memory_trust_review_request_specs"))
    request_spec_proposal_ids = {
        str(spec["proposal_id"])
        for spec in request_specs
        if isinstance(spec.get("proposal_id"), str) and spec.get("proposal_id")
    }
    template_proposal_ids = {
        str(template["proposal_id"])
        for template in promotion_acceptance_templates
        if isinstance(template.get("proposal_id"), str) and template.get("proposal_id")
    }
    if not template_proposal_ids.issubset(request_spec_proposal_ids):
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_request_review_request_spec_proposal_ids_mismatch"
        )
    for index, spec in enumerate(request_specs):
        spec_id = _string(spec.get("memory_trust_review_request_spec_id")) or (
            f"memory_trust_review_request_spec:{index}"
        )
        proposal_id = _string(spec.get("proposal_id")) or spec_id
        violations.extend(
            _current_sequence_review_freshness_violations(
                proposal_id,
                _mapping(spec.get("current_sequence_review_freshness")),
            )
        )
    for flag in REQUIRED_REQUEST_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_executor_dry_run_result_consumer_request_{flag}_not_false"
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
        _current_sequence_review_freshness_violations(
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


def _dry_run_envelope_violations(dry_run: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if dry_run.get("type") != REQUIRED_EXECUTOR_DRY_RUN_TYPE:
        violations.append("sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_type_mismatch")
    generated_at = dry_run.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_generated_at_invalid")
    if (
        dry_run.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID
    ):
        violations.append("sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_method_id_mismatch")
    if (
        dry_run.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_STATUSES
    ):
        violations.append("sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_status_unknown")
    safety = _mapping(dry_run.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_read_only_not_true")
    if safety.get("dry_run") is not True:
        violations.append("sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_dry_run_not_true")
    if safety.get("apply_requested") is not False:
        violations.append("sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_apply_requested_not_false")
    if safety.get("operator_ack") is not False:
        violations.append("sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_operator_ack_not_false")
    if safety.get("operator_required") is not True:
        violations.append("sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_operator_required_not_true")
    for flag in REQUIRED_DRY_RUN_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_{flag}_not_false"
            )
    return tuple(violations)


def _dry_run_summary_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
    candidates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "execution_candidate_count": len(candidates),
        "dry_run_record_count": len(records),
        "invalid_record_count": 0,
        "unreviewed_contract_record_count": 0,
        "would_change_policy_state_count": sum(
            1 for record in records if record.get("would_change_policy_state") is True
        ),
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_summary_{key}_mismatch"
            )
    if summary.get("promotion_execution_plan_status") != REQUIRED_EXECUTION_PLAN_READY_STATUS:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_summary_plan_status_mismatch"
        )
    if summary.get("target_policy_store") != TARGET_POLICY_STORE:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_summary_target_store_mismatch"
        )
    if summary.get("operator_review_required") is not True:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_summary_operator_review_not_true"
        )
    if summary.get("dry_run") is not True:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_summary_dry_run_not_true"
        )
    if summary.get("apply_requested") is not False:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_summary_apply_requested_not_false"
        )
    if summary.get("operator_ack") is not False:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_summary_operator_ack_not_false"
        )
    for flag in REQUIRED_DRY_RUN_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_summary_{flag}_not_false"
            )
    expected_paths = tuple(
        str(record["target_policy_path"])
        for record in records
        if isinstance(record.get("target_policy_path"), str)
        and record.get("target_policy_path")
    )
    expected_keys = tuple(
        str(record["idempotency_key"])
        for record in records
        if isinstance(record.get("idempotency_key"), str)
        and record.get("idempotency_key")
    )
    expected_proposals = tuple(
        str(record["proposal_id"])
        for record in records
        if isinstance(record.get("proposal_id"), str) and record.get("proposal_id")
    )
    if _string_list(summary.get("target_policy_paths")) != expected_paths:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_summary_target_paths_mismatch"
        )
    if _string_list(summary.get("idempotency_keys")) != expected_keys:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_summary_idempotency_keys_mismatch"
        )
    if _string_list(summary.get("accepted_proposal_ids")) != expected_proposals:
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_summary_accepted_proposal_ids_mismatch"
        )
    return tuple(violations)


def _dry_run_record_violations(
    record: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    index: int,
) -> Tuple[str, ...]:
    record_id = _record_id(record, index)
    violations = []
    for field in REQUIRED_DRY_RUN_RECORD_FIELDS:
        if not _string(record.get(field)):
            violations.append(f"{record_id}:{field}_missing")
    expected_pairs = {
        "promotion_execution_candidate_id": candidate.get(
            "promotion_execution_candidate_id"
        ),
        "promotion_acceptance_id": candidate.get("promotion_acceptance_id"),
        "promotion_contract_record_id": candidate.get("promotion_contract_record_id"),
        "promotion_candidate_id": candidate.get("promotion_candidate_id"),
        "proposal_id": candidate.get("proposal_id"),
        "learning_scope": candidate.get("learning_scope"),
        "value_type": candidate.get("value_type"),
        "source_idempotency_key": candidate.get("idempotency_key"),
        "previous_policy_state": candidate.get("previous_policy_state"),
        "rollback_strategy": candidate.get("rollback_strategy"),
    }
    for field, expected in expected_pairs.items():
        if record.get(field) != expected:
            violations.append(f"{record_id}:{field}_mismatch")
    if record.get("target_policy_store") != TARGET_POLICY_STORE:
        violations.append(f"{record_id}:target_policy_store_mismatch")
    if record.get("target_policy_path") != _target_policy_path(record):
        violations.append(f"{record_id}:target_policy_path_mismatch")
    if record.get("operation") != "set_operator_sequence_policy_trust_state":
        violations.append(f"{record_id}:operation_mismatch")
    if record.get("target_policy_state") != "trusted_for_sequence_policy":
        violations.append(f"{record_id}:target_policy_state_mismatch")
    if record.get("idempotency_key") != _idempotency_key(record):
        violations.append(f"{record_id}:idempotency_key_mismatch")
    if record.get("dry_run") is not True:
        violations.append(f"{record_id}:dry_run_not_true")
    if record.get("read_only") is not True:
        violations.append(f"{record_id}:read_only_not_true")
    if record.get("requires_operator_ack_before_apply") is not True:
        violations.append(f"{record_id}:operator_ack_requirement_missing")
    if record.get("requires_executor_contract_before_apply") is not True:
        violations.append(f"{record_id}:executor_contract_requirement_missing")
    for flag in REQUIRED_DRY_RUN_RECORD_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(f"{record_id}:{flag}_not_false")
    return tuple(violations)


def _record_candidate_identity_violations(
    records: Tuple[Mapping[str, Any], ...],
    candidates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    records_by_candidate_id = _records_by_candidate_id(records)
    candidate_ids = tuple(
        str(candidate["promotion_execution_candidate_id"])
        for candidate in candidates
        if isinstance(candidate.get("promotion_execution_candidate_id"), str)
        and candidate.get("promotion_execution_candidate_id")
    )
    violations = []
    if set(records_by_candidate_id) != set(candidate_ids):
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_candidate_ids_mismatch"
        )
    if len(records_by_candidate_id) != len(records):
        violations.append(
            "sequence_policy_promotion_executor_dry_run_result_consumer_duplicate_candidate_ids"
        )
    for index, candidate in enumerate(candidates):
        candidate_id = _string(candidate.get("promotion_execution_candidate_id"))
        record = _mapping(records_by_candidate_id.get(candidate_id))
        if not record:
            violations.append(f"{candidate_id}:dry_run_record_missing")
            continue
        violations.extend(
            _dry_run_record_violations(record, candidate=candidate, index=index)
        )
    return tuple(violations)


def _candidate_template_identity_violations(
    candidates: Tuple[Mapping[str, Any], ...],
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    templates_by_contract_record_id = _templates_by_contract_record_id(
        promotion_acceptance_templates
    )
    violations = []
    for candidate in candidates:
        contract_id = _string(candidate.get("promotion_contract_record_id"))
        proposal_id = _string(candidate.get("proposal_id")) or contract_id
        template = _mapping(templates_by_contract_record_id.get(contract_id))
        if not template:
            violations.append(f"{proposal_id}:acceptance_template_missing")
            continue
        expected_pairs = {
            "promotion_contract_record_id": candidate.get("promotion_contract_record_id"),
            "promotion_candidate_id": candidate.get("promotion_candidate_id"),
            "proposal_id": candidate.get("proposal_id"),
            "learning_scope": candidate.get("learning_scope"),
            "value_type": candidate.get("value_type"),
            "idempotency_key": candidate.get("idempotency_key"),
            "previous_policy_state": candidate.get("previous_policy_state"),
            "rollback_strategy": candidate.get("rollback_strategy"),
        }
        for field, expected in expected_pairs.items():
            if template.get(field) != expected:
                violations.append(f"{proposal_id}:{field}_candidate_mismatch")
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer(
    *,
    promotion_execution_plan_result_consumer: Mapping[str, Any] | None = None,
    promotion_executor_dry_run_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify executor dry-run evidence and emit an executor request template."""

    consumer = _mapping(promotion_execution_plan_result_consumer)
    dry_run = _mapping(promotion_executor_dry_run_result)
    request = _mapping(consumer.get("promotion_executor_dry_run_request"))
    candidates = _candidates_from_request(request)
    promotion_acceptance_templates = _acceptance_templates_from_request(request)
    memory_trust_review_request_specs = _record_list(
        request.get("memory_trust_review_request_specs")
    )
    records = _record_list(dry_run.get("dry_run_records"))
    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_consumer_envelope_violations(consumer),
                *_consumer_summary_violations(
                    _mapping(consumer.get("summary")),
                    candidates=candidates,
                    promotion_acceptance_templates=promotion_acceptance_templates,
                ),
                *_consumer_request_violations(
                    request,
                    candidates=candidates,
                    promotion_acceptance_templates=promotion_acceptance_templates,
                ),
                *(
                    violation
                    for index, template in enumerate(promotion_acceptance_templates)
                    for violation in _acceptance_template_violations(
                        template,
                        index=index,
                    )
                ),
                *_candidate_template_identity_violations(
                    candidates,
                    promotion_acceptance_templates,
                ),
                *_dry_run_envelope_violations(dry_run),
                *_dry_run_summary_violations(
                    _mapping(dry_run.get("summary")),
                    records=records,
                    candidates=candidates,
                ),
                *(
                    ("sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_status_not_ready",)
                    if dry_run.get("status") != REQUIRED_EXECUTOR_DRY_RUN_READY_STATUS
                    else ()
                ),
                *(
                    ("sequence_policy_promotion_executor_dry_run_result_consumer_execution_candidates_required",)
                    if not candidates
                    else ()
                ),
                *(
                    ("sequence_policy_promotion_executor_dry_run_result_consumer_dry_run_records_required",)
                    if not records
                    else ()
                ),
                *_record_candidate_identity_violations(records, candidates),
            )
        )
    )
    invalid_record_ids = () if not consumer or not dry_run else raw_invalid_record_ids

    if not consumer or not dry_run:
        status = "sequence_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer_not_ready"
        blocked_by = (
            "promotion_execution_plan_result_consumer_and_executor_dry_run_result_required",
        )
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_memory_trust_promotion_executor_dry_run_evidence",
        )
    elif dry_run.get("status") != REQUIRED_EXECUTOR_DRY_RUN_READY_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer_blocked"
        blocked_by = (
            "sequence_memory_trust_promotion_executor_dry_run_must_be_ready_before_executor_request",
        )
    elif not records:
        status = "sequence_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer_not_ready"
        blocked_by = ("sequence_memory_trust_promotion_executor_dry_run_records_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_executor_request_available"
        blocked_by = (
            "operator_may_run_read_only_sequence_memory_trust_promotion_executor_preview",
        )

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_executor_request_available"
    )
    executor_request: Dict[str, Any] = {
        "handler": "execute_operator_proposal_apply_sequence_policy_memory_trust_promotion",
        "read_only_template": True,
        "promotion_executor_dry_run_result_consumer_only": True,
        "promotion_executor_request_template_only": True,
        "promotion_executor_handler_invoked": False,
        "promotion_executor_created": False,
        "apply": False,
        "operator_ack": False,
        "operator_must_keep_apply_false_for_preview": True,
        "operator_must_set_apply_true_before_policy_mutation": True,
        "operator_must_set_operator_ack_true_before_policy_mutation": True,
        "target_policy_store_required_before_apply": True,
        "memory_write_allowed_by_consumer": False,
        "execution_allowed_by_consumer": False,
        "live_write": False,
        "apply_attempted": False,
        "queue_mutation": False,
        "proposal_status_mutation": False,
        "idempotency_reserved": False,
        "manual_console_write_called": False,
        "policy_mutation": False,
        "memory_mutation": False,
        "memory_file_mutation": False,
        "runtime_memory_mutation": False,
        "model_mutation": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "can_apply_live": False,
    }
    if request_available:
        executor_request["executor_dry_run"] = dry_run
        executor_request["promotion_acceptance_record_templates"] = list(
            promotion_acceptance_templates
        )
        executor_request["memory_trust_review_request_specs"] = list(
            memory_trust_review_request_specs
        )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_DRY_RUN_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTION_PLAN_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_executor_dry_run_result_consumer_only": True,
            "promotion_executor_request_template_only": True,
            "requires_promotion_execution_plan_result_consumer": True,
            "requires_promotion_executor_dry_run_result": True,
            "promotion_executor_handler_invoked": False,
            "promotion_executor_created": False,
            "operator_ack_required_before_policy_mutation": True,
            "apply_true_required_before_policy_mutation": True,
            "target_policy_store_required_before_apply": True,
            "memory_write_allowed_by_consumer": False,
            "execution_allowed_by_consumer": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "policy_mutation": False,
            "memory_mutation": False,
            "memory_file_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
        },
        "summary": {
            "expected_executor_dry_run_record_count": len(candidates),
            "executor_dry_run_record_count": len(records),
            "promotion_acceptance_record_template_count": len(
                promotion_acceptance_templates
            ),
            "memory_trust_review_request_spec_count": len(
                memory_trust_review_request_specs
            ),
            "would_change_policy_state_count": sum(
                1 for record in records if record.get("would_change_policy_state") is True
            ),
            "invalid_record_count": len(invalid_record_ids),
            "fresh_sequence_review_context_status": _source_sequence_review_freshness_status(
                promotion_acceptance_templates
            ),
            "target_policy_store": TARGET_POLICY_STORE,
            "promotion_executor_dry_run_result_consumer_only": True,
            "promotion_executor_request_template_only": True,
            "promotion_executor_request_available": request_available,
            "promotion_executor_handler_invoked": False,
            "promotion_executor_created": False,
            "operator_ack_required_before_policy_mutation": True,
            "apply_true_required_before_policy_mutation": True,
            "target_policy_store_required_before_apply": True,
            "memory_write_allowed_by_consumer": False,
            "execution_allowed_by_consumer": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "policy_mutation": False,
            "memory_mutation": False,
            "memory_file_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
            "accepted_proposal_ids": [
                str(record["proposal_id"])
                for record in records
                if isinstance(record.get("proposal_id"), str) and record.get("proposal_id")
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
            "target_policy_paths": [
                str(record["target_policy_path"])
                for record in records
                if isinstance(record.get("target_policy_path"), str)
                and record.get("target_policy_path")
            ],
            "idempotency_keys": [
                str(record["idempotency_key"])
                for record in records
                if isinstance(record.get("idempotency_key"), str)
                and record.get("idempotency_key")
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion execution-plan result consumer and executor dry-run result."
                if not consumer or not dry_run
                else (
                    "Replace invalid sequence-memory trust promotion executor dry-run evidence before executor preview."
                    if invalid_record_ids
                    else (
                        "Run read-only promotion executor dry-run before executor preview."
                        if dry_run.get("status") != REQUIRED_EXECUTOR_DRY_RUN_READY_STATUS
                        else (
                            "Run the read-only executor preview request; this consumer does not apply policy changes."
                        )
                    )
                )
            ),
        },
        "promotion_executor_request": executor_request,
    }
