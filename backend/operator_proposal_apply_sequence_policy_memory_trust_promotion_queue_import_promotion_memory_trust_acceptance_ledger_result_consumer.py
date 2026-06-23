"""Read-only consumer for promotion memory trust acceptance ledgers.

This verifier matches operator acceptance-ledger evidence back to promotion
memory trust-review provenance before exposing a promotion-plan request
template. It delegates generic ledger validation to the queue-import memory
trust acceptance-ledger result consumer and does not build plans, mutate policy,
runtime memory, models, queues, proposal state, trust stores, idempotency, or
mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_ACCEPTANCE_LEDGER_RESULT_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_ACCEPTANCE_LEDGER_RESULT_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_ACCEPTANCE_LEDGER_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_plan_request_available",
}
REQUIRED_PRIOR_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer"
)
REQUIRED_PRIOR_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_request_available"
)
REQUIRED_GENERIC_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_plan_request_available"
)
REQUIRED_GENERIC_REPLACEMENT_STATUS = (
    "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer_replacement_required"
)
REQUIRED_GENERIC_BLOCKED_STATUS = (
    "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer_blocked"
)
PRIOR_FALSE_FLAGS = (
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
CONSUMER_FALSE_FLAGS = (
    "promotion_plan_handler_invoked",
    "promotion_plan_created",
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
REQUIRED_TEMPLATE_FALSE_FLAGS = (
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_policy_promotion_allowed_by_acceptance",
    "execution_allowed_by_acceptance",
    "queue_mutation_allowed_by_acceptance",
    "policy_mutation_allowed_by_acceptance",
    "model_mutation_allowed_by_acceptance",
    "runtime_memory_mutation_allowed_by_acceptance",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
)
REQUIRED_SOURCE_TEMPLATE_FALSE_FLAGS = (
    "sequence_policy_promotion_allowed_by_acceptance",
    "execution_allowed_by_acceptance",
    "queue_mutation_allowed_by_acceptance",
    "policy_mutation_allowed_by_acceptance",
    "model_mutation_allowed_by_acceptance",
    "runtime_memory_mutation_allowed_by_acceptance",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
)
REQUIRED_SOURCE_TEMPLATE_FIELDS = (
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
REQUIRED_SOURCE_TEMPLATE_IDENTITY_FIELDS = (
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
REQUIRED_MEMORY_TRUST_REVIEW_SPEC_FALSE_FLAGS = (
    "memory_trust_review_handler_invoked",
    "memory_trust_review_created",
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


def _sequence_review_freshness_core(
    payload: Mapping[str, Any],
) -> Tuple[Tuple[str, Any], ...]:
    return (
        ("status", _context_status(payload)),
        ("read_only", payload.get("read_only")),
        ("live_write", payload.get("live_write")),
        ("apply_attempted", payload.get("apply_attempted")),
        ("queries_console", payload.get("queries_console")),
        ("queue_mutation", payload.get("queue_mutation")),
        ("policy_mutation", payload.get("policy_mutation")),
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
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_type_mismatch")
    generated_at = prior.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_generated_at_invalid")
    if (
        prior.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_method_id_mismatch")
    if (
        prior.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_STATUSES
    ):
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_status_unknown")
    safety = _mapping(prior.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_read_only_not_true")
    if safety.get("promotion_memory_trust_review_result_consumer_only") is not True:
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_only_not_true")
    if safety.get("promotion_acceptance_ledger_request_template_only") is not True:
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_template_only_not_true")
    if safety.get("operator_acceptance_required") is not True:
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_operator_acceptance_not_true")
    for flag in PRIOR_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"promotion_memory_trust_acceptance_ledger_result_consumer_prior_{flag}_not_false")
    return tuple(violations)


def _prior_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
    prior_templates: Tuple[Mapping[str, Any], ...],
    source_templates: Tuple[Mapping[str, Any], ...],
    memory_trust_review_request_specs: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "expected_trust_review_candidate_count": len(expected_ids),
        "trust_review_candidate_count": len(expected_ids),
        "promotion_acceptance_record_template_count": len(prior_templates),
        "source_promotion_acceptance_record_template_count": len(source_templates),
        "memory_trust_review_request_spec_count": len(
            memory_trust_review_request_specs
        ),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"promotion_memory_trust_acceptance_ledger_result_consumer_prior_summary_{key}_mismatch"
            )
    if summary.get("promotion_acceptance_ledger_request_available") is not bool(expected_ids):
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_summary_request_available_mismatch")
    if summary.get("operator_acceptance_required") is not True:
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_summary_operator_acceptance_not_true")
    if _string_list(summary.get("trust_review_candidate_proposal_ids")) != expected_ids:
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_summary_proposal_ids_mismatch")
    expected_template_proposal_ids = tuple(
        str(template["proposal_id"])
        for template in prior_templates
        if isinstance(template.get("proposal_id"), str) and template.get("proposal_id")
    )
    if (
        _string_list(summary.get("promotion_acceptance_record_template_proposal_ids"))
        != expected_template_proposal_ids
    ):
        violations.append(
            "promotion_memory_trust_acceptance_ledger_result_consumer_prior_summary_template_proposal_ids_mismatch"
        )
    expected_source_template_proposal_ids = tuple(
        str(template["proposal_id"])
        for template in source_templates
        if isinstance(template.get("proposal_id"), str) and template.get("proposal_id")
    )
    if (
        _string_list(
            summary.get("source_promotion_acceptance_record_template_proposal_ids")
        )
        != expected_source_template_proposal_ids
    ):
        violations.append(
            "promotion_memory_trust_acceptance_ledger_result_consumer_prior_summary_source_template_proposal_ids_mismatch"
        )
    if source_templates and summary.get("source_sequence_review_context_status") != "fresh":
        violations.append(
            "promotion_memory_trust_acceptance_ledger_result_consumer_prior_summary_source_freshness_not_fresh"
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
            "promotion_memory_trust_acceptance_ledger_result_consumer_prior_summary_review_request_spec_proposal_ids_mismatch"
        )
    for flag in PRIOR_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"promotion_memory_trust_acceptance_ledger_result_consumer_prior_summary_{flag}_not_false"
            )
    return tuple(violations)


def _prior_request_violations(
    request: Mapping[str, Any],
    *,
    expected_template_ids: Tuple[str, ...],
    prior_templates: Tuple[Mapping[str, Any], ...],
    expected_ids: Tuple[str, ...],
    source_templates: Tuple[Mapping[str, Any], ...],
    memory_trust_review_request_specs: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    if request.get("handler") != "record_operator_proposal_apply_sequence_policy_memory_trust_acceptance":
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_request_handler_mismatch")
    if request.get("read_only_template") is not True:
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_request_template_not_true")
    if request.get("promotion_memory_trust_review_result_consumer_only") is not True:
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_request_only_not_true")
    if request.get("promotion_acceptance_ledger_request_template_only") is not True:
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_request_template_only_not_true")
    if request.get("operator_must_supply_acceptance_records") is not True:
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_request_operator_records_not_true")
    if request.get("acceptance_records") != []:
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_request_acceptance_records_not_empty")
    request_template_ids = tuple(
        _string(template.get("review_candidate_id"))
        for template in _record_list(request.get("acceptance_record_templates"))
    )
    if request_template_ids != expected_template_ids:
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_request_template_ids_mismatch")
    request_templates_by_id = {
        _string(template.get("review_candidate_id")): template
        for template in _record_list(request.get("acceptance_record_templates"))
        if _string(template.get("review_candidate_id"))
    }
    for prior_template in prior_templates:
        review_candidate_id = _string(prior_template.get("review_candidate_id"))
        proposal_id = _string(prior_template.get("proposal_id")) or review_candidate_id
        request_template = _mapping(request_templates_by_id.get(review_candidate_id))
        if not request_template:
            violations.append(f"{proposal_id}:request_template_missing")
            continue
        expected_pairs = {
            "review_candidate_id": prior_template.get("review_candidate_id"),
            "memory_record_id": prior_template.get("memory_record_id"),
            "proposal_id": prior_template.get("proposal_id"),
            "learning_scope": prior_template.get("learning_scope"),
            "value_type": prior_template.get("value_type"),
            "idempotency_key": prior_template.get("idempotency_key"),
        }
        for field, expected in expected_pairs.items():
            if request_template.get(field) != expected:
                violations.append(f"{proposal_id}:{field}_request_template_mismatch")
    if _mapping(request.get("memory_trust_review")).get("status") != "sequence_memory_trust_review_candidate_available":
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_request_review_not_available")
    source_template_proposal_ids = {
        str(template["proposal_id"])
        for template in source_templates
        if isinstance(template.get("proposal_id"), str) and template.get("proposal_id")
    }
    if len(source_template_proposal_ids) != len(source_templates):
        violations.append(
            "promotion_memory_trust_acceptance_ledger_result_consumer_prior_request_source_template_proposal_ids_not_unique"
        )
    if expected_ids and not set(expected_ids).issubset(source_template_proposal_ids):
        violations.append(
            "promotion_memory_trust_acceptance_ledger_result_consumer_prior_request_source_templates_missing_trust_review"
        )
    request_spec_proposal_ids = {
        str(spec["proposal_id"])
        for spec in memory_trust_review_request_specs
        if isinstance(spec.get("proposal_id"), str) and spec.get("proposal_id")
    }
    if not source_template_proposal_ids.issubset(request_spec_proposal_ids):
        violations.append(
            "promotion_memory_trust_acceptance_ledger_result_consumer_prior_request_review_request_spec_proposal_ids_mismatch"
        )
    for flag in PRIOR_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(
                f"promotion_memory_trust_acceptance_ledger_result_consumer_prior_request_{flag}_not_false"
            )
    return tuple(violations)


def _prior_template_violations(
    templates: Tuple[Mapping[str, Any], ...],
    *,
    expected_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if len(templates) != len(expected_ids):
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_template_count_mismatch")
    proposal_ids = tuple(_string(template.get("proposal_id")) for template in templates)
    if proposal_ids != expected_ids:
        violations.append("promotion_memory_trust_acceptance_ledger_result_consumer_prior_template_proposal_ids_mismatch")
    for index, template in enumerate(templates):
        prefix = _string(template.get("promotion_acceptance_record_template_id")) or f"promotion_acceptance_template:{index}"
        if template.get("promotion_memory_trust_review_result_consumer_only") is not True:
            violations.append(f"{prefix}:only_not_true")
        if template.get("promotion_acceptance_ledger_request_template_only") is not True:
            violations.append(f"{prefix}:template_only_not_true")
        if template.get("decision_options") != ["accepted", "rejected"]:
            violations.append(f"{prefix}:decision_options_invalid")
        if template.get("operator_must_choose_decision") is not True:
            violations.append(f"{prefix}:operator_must_choose_decision_not_true")
        for flag in REQUIRED_TEMPLATE_FALSE_FLAGS:
            if template.get(flag) is not False:
                violations.append(f"{prefix}:{flag}_not_false")
    return tuple(violations)


def _source_template_violations(
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
    for field in REQUIRED_SOURCE_TEMPLATE_FIELDS:
        if not _string(template.get(field)):
            violations.append(f"{template_id}:{field}_missing")
    for field in REQUIRED_SOURCE_TEMPLATE_IDENTITY_FIELDS:
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
    for flag in REQUIRED_SOURCE_TEMPLATE_FALSE_FLAGS:
        if template.get(flag) is not False:
            violations.append(f"{template_id}:{flag}_not_false")
    return tuple(violations)


def _request_spec_violations(
    specs: Tuple[Mapping[str, Any], ...],
    *,
    expected_source_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    spec_ids = tuple(_string(spec.get("proposal_id")) for spec in specs)
    spec_id_set = {proposal_id for proposal_id in spec_ids if proposal_id}
    if len(spec_id_set) != len(specs):
        violations.append(
            "promotion_memory_trust_acceptance_ledger_result_consumer_prior_request_spec_proposal_ids_not_unique"
        )
    if expected_source_ids and not set(expected_source_ids).issubset(spec_id_set):
        violations.append(
            "promotion_memory_trust_acceptance_ledger_result_consumer_prior_request_specs_missing_source_templates"
        )
    for index, spec in enumerate(specs):
        prefix = f"memory_trust_review_request_spec:{index}"
        proposal_id = _string(spec.get("proposal_id")) or prefix
        if spec.get("read_only") is not True:
            violations.append(f"{prefix}:read_only_not_true")
        if spec.get("memory_index_result_consumer_only") is not True:
            violations.append(f"{prefix}:only_not_true")
        if spec.get("memory_trust_review_request_template_only") is not True:
            violations.append(f"{prefix}:template_only_not_true")
        if spec.get("source_memory_index_available") is not True:
            violations.append(f"{prefix}:source_memory_index_available_not_true")
        violations.extend(
            _context_violations(
                proposal_id,
                _mapping(spec.get("current_sequence_review_freshness")),
            )
        )
        for flag in REQUIRED_MEMORY_TRUST_REVIEW_SPEC_FALSE_FLAGS:
            if spec.get(flag) is not False:
                violations.append(f"{prefix}:{flag}_not_false")
    return tuple(violations)


def _source_template_request_spec_identity_violations(
    *,
    source_templates: Tuple[Mapping[str, Any], ...],
    memory_trust_review_request_specs: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    request_specs_by_proposal_id = _records_by_proposal_id(
        memory_trust_review_request_specs
    )
    violations = []
    for template in source_templates:
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


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer(
    *,
    promotion_memory_trust_review_result_consumer: Mapping[str, Any] | None = None,
    promotion_memory_trust_acceptance_ledger_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify promotion acceptance-ledger evidence and emit a plan template."""

    prior = _mapping(promotion_memory_trust_review_result_consumer)
    ledger = _mapping(promotion_memory_trust_acceptance_ledger_result)
    prior_summary = _mapping(prior.get("summary"))
    prior_request = _mapping(prior.get("promotion_acceptance_ledger_request"))
    prior_templates = _record_list(prior.get("promotion_acceptance_record_templates"))
    source_templates = _record_list(
        prior_request.get("promotion_acceptance_record_templates")
    )
    memory_trust_review_request_specs = _record_list(
        prior_request.get("memory_trust_review_request_specs")
    )
    prior_summary_ids = _string_list(prior_summary.get("trust_review_candidate_proposal_ids"))
    template_ids = tuple(_string(template.get("review_candidate_id")) for template in prior_templates)
    prior_status = _string(prior.get("status"))
    generic_trust_review_consumer = _mapping(
        prior.get("generic_memory_trust_review_result_consumer")
    )

    generic_consumer: Mapping[str, Any] = {}
    if prior and ledger:
        generic_consumer = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer(
                memory_trust_review_result_consumer=generic_trust_review_consumer,
                memory_trust_acceptance_ledger_result=ledger,
            )
        )
    generic_status = _string(generic_consumer.get("status"))
    generic_summary = _mapping(generic_consumer.get("summary"))
    generic_request = _mapping(generic_consumer.get("promotion_plan_request"))
    generic_invalid_record_ids = tuple(
        f"generic_memory_trust_acceptance_ledger_result_consumer:{item}"
        for item in _string_list(generic_summary.get("invalid_record_ids"))
    )
    generic_template_proposal_ids = _string_list(
        generic_summary.get("acceptance_record_template_proposal_ids")
    )
    expected_ids = generic_template_proposal_ids or prior_summary_ids

    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_string_list(prior_summary.get("invalid_record_ids")),
                *_prior_envelope_violations(prior),
                *_prior_summary_violations(
                    prior_summary,
                    expected_ids=expected_ids,
                    prior_templates=prior_templates,
                    source_templates=source_templates,
                    memory_trust_review_request_specs=memory_trust_review_request_specs,
                ),
                *_prior_request_violations(
                    prior_request,
                    expected_template_ids=template_ids,
                    prior_templates=prior_templates,
                    expected_ids=expected_ids,
                    source_templates=source_templates,
                    memory_trust_review_request_specs=memory_trust_review_request_specs,
                ),
                *_prior_template_violations(prior_templates, expected_ids=expected_ids),
                *(
                    violation
                    for index, template in enumerate(source_templates)
                    for violation in _source_template_violations(
                        template,
                        index=index,
                    )
                ),
                *_request_spec_violations(
                    memory_trust_review_request_specs,
                    expected_source_ids=tuple(
                        str(template["proposal_id"])
                        for template in source_templates
                        if isinstance(template.get("proposal_id"), str)
                        and template.get("proposal_id")
                    ),
                ),
                *_source_template_request_spec_identity_violations(
                    source_templates=source_templates,
                    memory_trust_review_request_specs=memory_trust_review_request_specs,
                ),
                *(
                    generic_invalid_record_ids
                    if generic_status == REQUIRED_GENERIC_REPLACEMENT_STATUS
                    else ()
                ),
                *(
                    ("promotion_memory_trust_acceptance_ledger_result_consumer_generic_accepted_proposal_ids_not_subset",)
                    if set(_string_list(generic_summary.get("accepted_proposal_ids")))
                    - set(expected_ids)
                    else ()
                ),
                *(
                    ("promotion_memory_trust_acceptance_ledger_result_consumer_generic_rejected_proposal_ids_not_subset",)
                    if set(_string_list(generic_summary.get("rejected_proposal_ids")))
                    - set(expected_ids)
                    else ()
                ),
                *(
                    ("promotion_memory_trust_acceptance_ledger_result_consumer_generic_status_unknown",)
                    if (
                        generic_status
                        and generic_status
                        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_ACCEPTANCE_LEDGER_RESULT_CONSUMER_STATUSES
                    )
                    else ()
                ),
            )
        )
    )
    invalid_record_ids = () if not prior or not ledger else raw_invalid_record_ids

    if not prior or not ledger:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer_not_ready"
        blocked_by = (
            "promotion_memory_trust_review_result_consumer_and_acceptance_ledger_result_required",
        )
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_promotion_memory_trust_acceptance_ledger_evidence",
        )
    elif prior_status != REQUIRED_PRIOR_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer_blocked"
        blocked_by = (
            "promotion_memory_trust_review_result_consumer_must_request_acceptance_ledger_first",
        )
    elif generic_status == REQUIRED_GENERIC_BLOCKED_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer_blocked"
        blocked_by = (
            "operator_must_supply_accept_or_reject_records_before_promotion_plan",
        )
    elif generic_status != REQUIRED_GENERIC_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer_not_ready"
        blocked_by = ("ready_promotion_memory_trust_acceptance_ledger_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_plan_request_available"
        blocked_by = (
            "operator_may_request_read_only_sequence_memory_trust_promotion_plan",
        )

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_plan_request_available"
    )
    promotion_plan_request: Dict[str, Any] = {
        **dict(generic_request if request_available else {}),
        "handler": "plan_operator_proposal_apply_sequence_policy_memory_trust_promotion",
        "read_only_template": True,
        "memory_trust_acceptance_ledger_result_consumer_only": True,
        "promotion_memory_trust_acceptance_ledger_result_consumer_only": True,
        "promotion_plan_request_template_only": True,
        "promotion_plan_handler_invoked": False,
        "promotion_plan_created": False,
        **{flag: False for flag in CONSUMER_FALSE_FLAGS},
    }
    if request_available:
        promotion_plan_request["promotion_acceptance_record_templates"] = list(
            source_templates
        )
        promotion_plan_request["promotion_memory_trust_review_request_specs"] = list(
            memory_trust_review_request_specs
        )
        promotion_plan_request["promotion_memory_trust_acceptance_ledger_result_consumer"] = {
            "method_id": (
                OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID
            ),
            "source_prior_method_id": (
                OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID
            ),
            "accepted_proposal_ids": list(
                _string_list(generic_summary.get("accepted_proposal_ids"))
            ),
            "rejected_proposal_ids": list(
                _string_list(generic_summary.get("rejected_proposal_ids"))
            ),
            "read_only": True,
        }

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "memory_trust_acceptance_ledger_result_consumer_only": True,
            "promotion_memory_trust_acceptance_ledger_result_consumer_only": True,
            "promotion_plan_request_template_only": True,
            "requires_promotion_memory_trust_review_result_consumer": True,
            "requires_memory_trust_acceptance_ledger_result": True,
            "operator_acceptance_required": True,
            **{flag: False for flag in CONSUMER_FALSE_FLAGS},
        },
        "summary": {
            "expected_trust_review_candidate_count": len(expected_ids),
            "source_promotion_acceptance_record_template_count": len(
                source_templates
            ),
            "memory_trust_review_request_spec_count": len(
                memory_trust_review_request_specs
            ),
            "source_sequence_review_context_status": _source_sequence_review_freshness_status(
                source_templates
            ),
            "acceptance_record_template_count": (
                _summary_int(generic_summary.get("acceptance_record_template_count")) or 0
            ),
            "acceptance_record_count": (
                _summary_int(generic_summary.get("acceptance_record_count")) or 0
            ),
            "accepted_record_count": (
                _summary_int(generic_summary.get("accepted_record_count")) or 0
            ),
            "rejected_record_count": (
                _summary_int(generic_summary.get("rejected_record_count")) or 0
            ),
            "invalid_record_count": len(invalid_record_ids),
            "unreviewed_candidate_count": (
                _summary_int(generic_summary.get("unreviewed_candidate_count")) or 0
            ),
            "memory_trust_acceptance_ledger_result_consumer_only": True,
            "promotion_memory_trust_acceptance_ledger_result_consumer_only": True,
            "promotion_plan_request_template_only": True,
            "promotion_plan_request_available": request_available,
            "operator_acceptance_required": True,
            **{flag: False for flag in CONSUMER_FALSE_FLAGS},
            "accepted_proposal_ids": list(
                _string_list(generic_summary.get("accepted_proposal_ids"))
                if request_available
                else ()
            ),
            "rejected_proposal_ids": list(
                _string_list(generic_summary.get("rejected_proposal_ids"))
                if request_available
                else ()
            ),
            "acceptance_record_template_proposal_ids": list(
                _string_list(generic_summary.get("acceptance_record_template_proposal_ids"))
                if request_available
                else ()
            ),
            "source_promotion_acceptance_record_template_proposal_ids": [
                str(template["proposal_id"])
                for template in source_templates
                if isinstance(template.get("proposal_id"), str)
                and template.get("proposal_id")
            ],
            "memory_trust_review_request_spec_proposal_ids": [
                str(spec["proposal_id"])
                for spec in memory_trust_review_request_specs
                if isinstance(spec.get("proposal_id"), str) and spec.get("proposal_id")
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "unreviewed_candidate_ids": list(
                _string_list(generic_summary.get("unreviewed_candidate_ids"))
            ),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion memory trust-review consumer and acceptance ledger result."
                if not prior or not ledger
                else (
                    "Replace invalid promotion memory trust acceptance-ledger evidence before promotion planning."
                    if invalid_record_ids
                    else (
                        "Operator must provide accept/reject records before promotion planning."
                        if generic_status == REQUIRED_GENERIC_BLOCKED_STATUS
                        else "Run the read-only promotion-plan request; this consumer does not promote policy."
                    )
                )
            ),
        },
        "promotion_plan_request": promotion_plan_request,
        "generic_memory_trust_acceptance_ledger_result_consumer": (
            generic_consumer if prior and ledger else {}
        ),
    }
