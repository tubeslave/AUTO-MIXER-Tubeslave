"""Read-only consumer for sequence-policy memory trust acceptance ledgers.

This module verifies that an operator-supplied memory trust acceptance ledger
matches queue-import memory trust-review provenance before a caller may prepare
a manual promotion-plan request. It does not accept decisions, build promotion
plans, mutate policy, runtime memory, models, queues, memory files, proposal
state, idempotency, or mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_ACCEPTANCE_LEDGER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_ACCEPTANCE_LEDGER_STATUSES,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_review import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_METHOD_ID,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_ACCEPTANCE_LEDGER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_ACCEPTANCE_LEDGER_STATUSES,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_review import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_METHOD_ID,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_ACCEPTANCE_LEDGER_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_plan_request_available",
}
REQUIRED_TRUST_REVIEW_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer"
)
REQUIRED_TRUST_REVIEW_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_request_available"
)
REQUIRED_ACCEPTANCE_LEDGER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger"
)
REQUIRED_ACCEPTANCE_LEDGER_AVAILABLE_STATUS = (
    "operator_sequence_memory_trust_acceptance_records_available"
)
REQUIRED_MEMORY_TRUST_REVIEW_AVAILABLE_STATUS = (
    "sequence_memory_trust_review_candidate_available"
)
REQUIRED_CONSUMER_FALSE_FLAGS = (
    "acceptance_ledger_handler_invoked",
    "acceptance_ledger_created",
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
REQUIRED_TEMPLATE_FALSE_FLAGS = (
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
REQUIRED_TEMPLATE_IDENTITY_FIELDS = (
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
REQUIRED_LEDGER_FALSE_SAFETY_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "execution_allowed_by_acceptance",
    "queue_mutation",
    "policy_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_LEDGER_FALSE_SUMMARY_FLAGS = (
    "sequence_policy_promotion_allowed_by_acceptance",
    "execution_allowed_by_acceptance",
    "queue_mutation_allowed_by_acceptance",
    "policy_mutation_allowed_by_acceptance",
    "model_mutation_allowed_by_acceptance",
    "runtime_memory_mutation_allowed_by_acceptance",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_RECORD_FALSE_FLAGS = (
    "sequence_policy_promotion_allowed_by_acceptance",
    "execution_allowed_by_acceptance",
    "queue_mutation_allowed_by_acceptance",
    "policy_mutation_allowed_by_acceptance",
    "model_mutation_allowed_by_acceptance",
    "runtime_memory_mutation_allowed_by_acceptance",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_RECORD_FIELDS = (
    "acceptance_id",
    "review_candidate_id",
    "memory_record_id",
    "proposal_id",
    "learning_scope",
    "value_type",
    "idempotency_key",
    "decision",
    "reviewed_by",
    "reviewed_at",
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


def _templates_by_candidate_id(
    templates: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(template["review_candidate_id"]): template
        for template in templates
        if isinstance(template.get("review_candidate_id"), str)
        and template.get("review_candidate_id")
    }


def _record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("acceptance_id"))
    if explicit:
        return explicit
    candidate_id = _string(record.get("review_candidate_id"))
    return f"{candidate_id if candidate_id else 'unknown_sequence_memory_trust_acceptance'}:{index}"


def _consumer_envelope_violations(consumer: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_TRUST_REVIEW_RESULT_CONSUMER_TYPE:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_trust_review_consumer_type_mismatch"
        )
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_trust_review_consumer_method_id_mismatch"
        )
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_STATUSES
    ):
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_trust_review_consumer_status_unknown"
        )
    if consumer.get("status") != REQUIRED_TRUST_REVIEW_RESULT_CONSUMER_AVAILABLE_STATUS:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_trust_review_consumer_status_not_available"
        )
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_trust_review_consumer_read_only_not_true"
        )
    if safety.get("memory_trust_review_result_consumer_only") is not True:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_trust_review_consumer_only_not_true"
        )
    if safety.get("acceptance_ledger_request_template_only") is not True:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_trust_review_consumer_template_only_not_true"
        )
    if safety.get("operator_acceptance_required") is not True:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_trust_review_consumer_operator_acceptance_not_true"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_trust_acceptance_ledger_result_consumer_trust_review_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    templates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "expected_trust_review_candidate_count": len(templates),
        "trust_review_candidate_count": len(templates),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_memory_trust_acceptance_ledger_result_consumer_trust_review_consumer_summary_{key}_mismatch"
            )
    if summary.get("acceptance_ledger_request_available") is not True:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_trust_review_consumer_request_not_available"
        )
    if summary.get("operator_acceptance_required") is not True:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_trust_review_consumer_summary_operator_acceptance_not_true"
        )
    expected_proposal_ids = tuple(
        str(template["proposal_id"])
        for template in templates
        if isinstance(template.get("proposal_id"), str) and template.get("proposal_id")
    )
    if _string_list(summary.get("trust_review_candidate_proposal_ids")) != expected_proposal_ids:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_trust_review_consumer_proposal_ids_mismatch"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_trust_acceptance_ledger_result_consumer_trust_review_consumer_summary_{flag}_not_false"
            )
    return tuple(violations)


def _template_violations(
    template: Mapping[str, Any],
    *,
    index: int,
) -> Tuple[str, ...]:
    template_id = _string(template.get("acceptance_record_template_id")) or f"acceptance_template:{index}"
    proposal_id = _string(template.get("proposal_id")) or template_id
    violations = []
    for field in (
        "review_candidate_id",
        "memory_record_id",
        "proposal_id",
        "learning_scope",
        "value_type",
        "idempotency_key",
    ):
        if not _string(template.get(field)):
            violations.append(f"{template_id}:{field}_missing")
    for field in REQUIRED_TEMPLATE_IDENTITY_FIELDS:
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
    for flag in REQUIRED_TEMPLATE_FALSE_FLAGS:
        if template.get(flag) is not False:
            violations.append(f"{template_id}:{flag}_not_false")
    return tuple(violations)


def _consumer_request_violations(
    request: Mapping[str, Any],
    *,
    templates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    if request.get("handler") != "record_operator_proposal_apply_sequence_policy_memory_trust_acceptance":
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_request_handler_mismatch"
        )
    if request.get("read_only_template") is not True:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_request_read_only_template_not_true"
        )
    if request.get("acceptance_ledger_request_template_only") is not True:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_request_template_only_not_true"
        )
    if request.get("operator_must_supply_acceptance_records") is not True:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_request_operator_records_not_true"
        )
    if request.get("acceptance_records") != []:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_request_acceptance_records_not_empty"
        )
    for flag in (
        "acceptance_ledger_handler_invoked",
        "acceptance_ledger_created",
        "memory_write_allowed_by_consumer",
        "execution_allowed_by_consumer",
        "live_write",
        "apply_attempted",
        "queue_mutation",
        "proposal_status_mutation",
        "idempotency_reserved",
        "manual_console_write_called",
        "memory_mutation",
        "memory_file_mutation",
        "policy_mutation",
        "runtime_memory_mutation",
        "model_mutation",
        "live_authority_granted",
        "full_autonomous_ready",
        "can_apply_live",
    ):
        if request.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_trust_acceptance_ledger_result_consumer_request_{flag}_not_false"
            )
    review = _mapping(request.get("memory_trust_review"))
    if review.get("status") != REQUIRED_MEMORY_TRUST_REVIEW_AVAILABLE_STATUS:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_request_review_not_available"
        )
    request_templates = _record_list(request.get("acceptance_record_templates"))
    request_template_ids = tuple(
        _string(template.get("review_candidate_id")) for template in request_templates
    )
    top_level_template_ids = tuple(
        _string(template.get("review_candidate_id")) for template in templates
    )
    if request_template_ids != top_level_template_ids:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_request_templates_mismatch"
        )
    top_level_template_proposal_ids = tuple(
        _string(template.get("proposal_id")) for template in templates
    )
    request_template_proposal_ids = tuple(
        _string(template.get("proposal_id")) for template in request_templates
    )
    if request_template_proposal_ids != top_level_template_proposal_ids:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_request_template_proposal_ids_mismatch"
        )
    for index, template in enumerate(request_templates):
        proposal_id = (
            _string(template.get("proposal_id"))
            or f"acceptance_request_template:{index}"
        )
        violations.extend(
            _context_violations(
                proposal_id,
                _mapping(template.get("current_sequence_review_freshness")),
            )
        )
    request_specs = _record_list(request.get("memory_trust_review_request_specs"))
    request_spec_proposal_ids = tuple(
        _string(spec.get("proposal_id")) for spec in request_specs
    )
    if request_spec_proposal_ids != top_level_template_proposal_ids:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_request_specs_proposal_ids_mismatch"
        )
    for index, spec in enumerate(request_specs):
        proposal_id = (
            _string(spec.get("proposal_id"))
            or f"memory_trust_review_request_spec:{index}"
        )
        violations.extend(
            _context_violations(
                proposal_id,
                _mapping(spec.get("current_sequence_review_freshness")),
            )
        )
    return tuple(violations)


def _ledger_envelope_violations(ledger: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if ledger.get("type") != REQUIRED_ACCEPTANCE_LEDGER_TYPE:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_ledger_type_mismatch"
        )
    generated_at = ledger.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_ledger_generated_at_invalid"
        )
    if ledger.get("method_id") != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_ACCEPTANCE_LEDGER_METHOD_ID:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_ledger_method_id_mismatch"
        )
    if ledger.get("source_method_id") != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_METHOD_ID:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_ledger_source_method_mismatch"
        )
    if (
        ledger.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_ACCEPTANCE_LEDGER_STATUSES
    ):
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_ledger_status_unknown"
        )
    if ledger.get("status") == "operator_sequence_memory_trust_acceptance_replacement_required":
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_ledger_status_replacement_required"
        )
    safety = _mapping(ledger.get("safety"))
    if safety.get("read_only") is not True:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_ledger_read_only_not_true"
        )
    if safety.get("operator_required") is not True:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_ledger_operator_required_not_true"
        )
    for flag in REQUIRED_LEDGER_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_trust_acceptance_ledger_result_consumer_ledger_{flag}_not_false"
            )
    return tuple(violations)


def _valid_reviewed_candidate_ids(
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    return tuple(
        str(record["review_candidate_id"])
        for record in records
        if record.get("valid") is True
        and isinstance(record.get("review_candidate_id"), str)
        and record.get("review_candidate_id")
    )


def _unreviewed_candidate_ids(
    templates: Tuple[Mapping[str, Any], ...],
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    reviewed = set(_valid_reviewed_candidate_ids(records))
    return tuple(
        str(template["review_candidate_id"])
        for template in templates
        if isinstance(template.get("review_candidate_id"), str)
        and template.get("review_candidate_id")
        and template.get("review_candidate_id") not in reviewed
    )


def _proposal_ids_for_records(
    records: Tuple[Mapping[str, Any], ...],
    *,
    accepted: bool,
) -> Tuple[str, ...]:
    flag = (
        "operator_accepted_sequence_memory_trust_candidate"
        if accepted
        else "operator_rejected_sequence_memory_trust_candidate"
    )
    return tuple(
        str(record["proposal_id"])
        for record in records
        if record.get("valid") is True
        and record.get(flag) is True
        and isinstance(record.get("proposal_id"), str)
        and record.get("proposal_id")
    )


def _ledger_summary_violations(
    summary: Mapping[str, Any],
    *,
    templates: Tuple[Mapping[str, Any], ...],
    records: Tuple[Mapping[str, Any], ...],
    invalid_record_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    expected_unreviewed = _unreviewed_candidate_ids(templates, records)
    expected_counts = {
        "trust_review_candidate_count": len(templates),
        "acceptance_record_count": len(records),
        "accepted_record_count": len(_proposal_ids_for_records(records, accepted=True)),
        "rejected_record_count": len(_proposal_ids_for_records(records, accepted=False)),
        "invalid_record_count": len(invalid_record_ids),
        "unreviewed_candidate_count": len(expected_unreviewed),
    }
    violations = []
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_memory_trust_acceptance_ledger_result_consumer_ledger_summary_{key}_mismatch"
            )
    if summary.get("sequence_memory_trust_review_status") != REQUIRED_MEMORY_TRUST_REVIEW_AVAILABLE_STATUS:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_ledger_summary_review_not_available"
        )
    if summary.get("operator_review_required") is not True:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_ledger_summary_operator_review_not_true"
        )
    if _string_list(summary.get("accepted_proposal_ids")) != _proposal_ids_for_records(records, accepted=True):
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_ledger_summary_accepted_proposal_ids_mismatch"
        )
    if _string_list(summary.get("rejected_proposal_ids")) != _proposal_ids_for_records(records, accepted=False):
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_ledger_summary_rejected_proposal_ids_mismatch"
        )
    if _string_list(summary.get("unreviewed_candidate_ids")) != expected_unreviewed:
        violations.append(
            "sequence_policy_memory_trust_acceptance_ledger_result_consumer_ledger_summary_unreviewed_candidate_ids_mismatch"
        )
    for flag in REQUIRED_LEDGER_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_trust_acceptance_ledger_result_consumer_ledger_summary_{flag}_not_false"
            )
    return tuple(violations)


def _record_violations(
    record: Mapping[str, Any],
    *,
    templates_by_candidate_id: Mapping[str, Mapping[str, Any]],
    index: int,
) -> Tuple[str, ...]:
    record_id = _record_id(record, index)
    violations = []
    for field in REQUIRED_RECORD_FIELDS:
        if not _string(record.get(field)):
            violations.append(f"{record_id}:{field}_missing")
    template = _mapping(templates_by_candidate_id.get(_string(record.get("review_candidate_id"))))
    if not template:
        violations.append(f"{record_id}:template_missing")
    if record.get("valid") is not True:
        violations.append(f"{record_id}:record_not_valid")
    if record.get("candidate_present") is not True:
        violations.append(f"{record_id}:candidate_present_not_true")
    if record.get("decision") not in {"accepted", "rejected"}:
        violations.append(f"{record_id}:decision_invalid")
    reviewed_at = _string(record.get("reviewed_at"))
    if reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append(f"{record_id}:reviewed_at_invalid")
    if template:
        expected_pairs = {
            "memory_record_id": template.get("memory_record_id"),
            "proposal_id": template.get("proposal_id"),
            "learning_scope": template.get("learning_scope"),
            "value_type": template.get("value_type"),
            "idempotency_key": template.get("idempotency_key"),
        }
        for field, expected in expected_pairs.items():
            if record.get(field) != expected:
                violations.append(f"{record_id}:{field}_mismatch")
    accepted = record.get("operator_accepted_sequence_memory_trust_candidate")
    rejected = record.get("operator_rejected_sequence_memory_trust_candidate")
    if record.get("decision") == "accepted" and accepted is not True:
        violations.append(f"{record_id}:accepted_flag_not_true")
    if record.get("decision") == "accepted" and rejected is not False:
        violations.append(f"{record_id}:rejected_flag_not_false")
    if record.get("decision") == "rejected" and rejected is not True:
        violations.append(f"{record_id}:rejected_flag_not_true")
    if record.get("decision") == "rejected" and accepted is not False:
        violations.append(f"{record_id}:accepted_flag_not_false")
    for flag in REQUIRED_RECORD_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(f"{record_id}:{flag}_not_false")
    if record.get("unsafe_authority_flags") not in ([], ()):
        violations.append(f"{record_id}:unsafe_authority_flags_not_empty")
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer(
    *,
    memory_trust_review_result_consumer: Mapping[str, Any] | None = None,
    memory_trust_acceptance_ledger_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify operator acceptance-ledger evidence and emit promotion-plan request."""

    consumer = _mapping(memory_trust_review_result_consumer)
    ledger = _mapping(memory_trust_acceptance_ledger_result)
    templates = _record_list(consumer.get("acceptance_record_templates"))
    request = _mapping(consumer.get("acceptance_ledger_request"))
    summary = _mapping(ledger.get("summary"))
    records = _record_list(ledger.get("records"))
    ledger_invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    template_violations = tuple(
        violation
        for index, template in enumerate(templates)
        for violation in _template_violations(template, index=index)
    )
    templates_by_candidate_id = _templates_by_candidate_id(templates)
    record_violations = tuple(
        violation
        for index, record in enumerate(records)
        for violation in _record_violations(
            record,
            templates_by_candidate_id=templates_by_candidate_id,
            index=index,
        )
    )
    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_consumer_envelope_violations(consumer),
                *_consumer_summary_violations(_mapping(consumer.get("summary")), templates=templates),
                *_consumer_request_violations(request, templates=templates),
                *template_violations,
                *_ledger_envelope_violations(ledger),
                *ledger_invalid_record_ids,
                *_ledger_summary_violations(
                    summary,
                    templates=templates,
                    records=records,
                    invalid_record_ids=ledger_invalid_record_ids,
                ),
                *record_violations,
            )
        )
    )
    invalid_record_ids = () if not consumer or not ledger else raw_invalid_record_ids
    unreviewed_candidate_ids = _unreviewed_candidate_ids(templates, records)

    if not consumer or not ledger:
        status = (
            "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer_not_ready"
        )
        blocked_by = (
            "memory_trust_review_result_consumer_and_acceptance_ledger_result_required",
        )
    elif invalid_record_ids:
        status = (
            "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer_replacement_required"
        )
        blocked_by = (
            "operator_must_replace_invalid_sequence_memory_trust_acceptance_ledger_evidence",
        )
    elif ledger.get("status") != REQUIRED_ACCEPTANCE_LEDGER_AVAILABLE_STATUS:
        status = (
            "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer_blocked"
        )
        blocked_by = (
            "operator_must_supply_accept_or_reject_records_before_promotion_plan",
        )
    elif not records:
        status = (
            "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer_not_ready"
        )
        blocked_by = ("sequence_memory_trust_acceptance_records_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_plan_request_available"
        blocked_by = ("operator_may_request_read_only_sequence_memory_trust_promotion_plan",)

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_plan_request_available"
    )
    promotion_plan_request: Dict[str, Any] = {
        "handler": "plan_operator_proposal_apply_sequence_policy_memory_trust_promotion",
        "read_only_template": True,
        "memory_trust_acceptance_ledger_result_consumer_only": True,
        "promotion_plan_request_template_only": True,
        "promotion_plan_handler_invoked": False,
        "promotion_plan_created": False,
        "acceptance_ledger_status_required": REQUIRED_ACCEPTANCE_LEDGER_AVAILABLE_STATUS,
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
        promotion_plan_request["acceptance_ledger"] = ledger
        promotion_plan_request["acceptance_record_templates"] = list(templates)
        promotion_plan_request["memory_trust_review_request_specs"] = list(
            _record_list(request.get("memory_trust_review_request_specs"))
        )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "memory_trust_acceptance_ledger_result_consumer_only": True,
            "promotion_plan_request_template_only": True,
            "requires_memory_trust_review_result_consumer": True,
            "requires_memory_trust_acceptance_ledger_result": True,
            "promotion_plan_handler_invoked": False,
            "promotion_plan_created": False,
            "operator_acceptance_required": True,
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
            "expected_trust_review_candidate_count": len(templates),
            "acceptance_record_template_count": len(templates),
            "acceptance_record_count": len(records),
            "accepted_record_count": len(_proposal_ids_for_records(records, accepted=True)),
            "rejected_record_count": len(_proposal_ids_for_records(records, accepted=False)),
            "invalid_record_count": len(invalid_record_ids),
            "unreviewed_candidate_count": len(unreviewed_candidate_ids),
            "memory_trust_acceptance_ledger_result_consumer_only": True,
            "promotion_plan_request_template_only": True,
            "promotion_plan_request_available": request_available,
            "promotion_plan_handler_invoked": False,
            "promotion_plan_created": False,
            "operator_acceptance_required": True,
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
            "accepted_proposal_ids": list(_proposal_ids_for_records(records, accepted=True)),
            "rejected_proposal_ids": list(_proposal_ids_for_records(records, accepted=False)),
            "fresh_sequence_review_context_status": (
                "fresh"
                if templates
                and all(
                    _context_status(
                        _mapping(template.get("current_sequence_review_freshness"))
                    )
                    == "fresh"
                    for template in templates
                )
                else "missing"
            ),
            "acceptance_record_template_proposal_ids": [
                template.get("proposal_id") for template in templates
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "unreviewed_candidate_ids": list(unreviewed_candidate_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide trust-review result consumer and memory trust acceptance ledger."
                if not consumer or not ledger
                else (
                    "Replace invalid memory trust acceptance-ledger evidence before promotion planning."
                    if invalid_record_ids
                    else (
                        "Operator must provide accept/reject records for every sequence memory trust candidate."
                        if ledger.get("status") != REQUIRED_ACCEPTANCE_LEDGER_AVAILABLE_STATUS
                        else "Run the read-only promotion-plan request; this consumer does not promote policy."
                    )
                )
            ),
        },
        "promotion_plan_request": promotion_plan_request,
    }
