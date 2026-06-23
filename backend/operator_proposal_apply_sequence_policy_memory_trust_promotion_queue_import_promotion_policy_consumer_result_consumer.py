"""Read-only consumer for promotion policy-consumer results.

This verifier connects the queue-import promotion sequence-policy gate result
consumer to the existing ordering-preview builder. It verifies policy-consumer
records against the previously verified sequence-policy gate provenance and
emits only an ordering-preview request template. It does not reorder proposals,
import queues, reserve idempotency, mutate policy/runtime/model/memory state, or
touch mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_STATUSES,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_POLICY_GATE_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_POLICY_GATE_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_STATUSES,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_POLICY_GATE_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_POLICY_GATE_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_POLICY_CONSUMER_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_POLICY_CONSUMER_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_request_available",
}
REQUIRED_GATE_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer"
)
REQUIRED_GATE_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_policy_consumer_request_available"
)
REQUIRED_POLICY_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer"
)
REQUIRED_POLICY_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_policy_consumer_records_available"
)
REQUIRED_GATE_RESULT_CONSUMER_FALSE_FLAGS = (
    "promotion_policy_consumer_handler_invoked",
    "promotion_policy_consumer_created",
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
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
)
REQUIRED_POLICY_CONSUMER_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "proposal_order_mutation",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_consumer",
    "sequence_policy_change_allowed_by_consumer",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
)
REQUIRED_POLICY_CONSUMER_SUMMARY_FALSE_FLAGS = tuple(
    flag for flag in REQUIRED_POLICY_CONSUMER_FALSE_FLAGS if flag != "apply_attempted"
)
REQUIRED_POLICY_RECORD_FIELDS = (
    "consumer_record_id",
    "proposal_id",
    "proposal_original_index",
    "sequence_policy_gate_record_id",
    "policy_key",
    "learning_scope",
    "value_type",
    "source_idempotency_key",
    "target_policy_state",
    "sequence_policy_gate_status",
    "recommended_sequence_policy_bucket",
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
REQUIRED_POLICY_RECORD_FALSE_FLAGS = REQUIRED_POLICY_CONSUMER_SUMMARY_FALSE_FLAGS
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
    return str(payload.get("status") or "missing").strip().lower() or "missing"


def _context_violations(record_id: str, payload: Mapping[str, Any]) -> Tuple[str, ...]:
    if not payload:
        return (f"{record_id}:current_sequence_review_freshness_missing",)
    violations = []
    if _context_status(payload) != "fresh":
        violations.append(f"{record_id}:current_sequence_review_freshness_not_fresh")
    if payload.get("read_only") is not True:
        violations.append(
            f"{record_id}:current_sequence_review_freshness_read_only_not_true"
        )
    for flag in REQUIRED_SEQUENCE_REVIEW_FRESHNESS_FALSE_FLAGS:
        if payload.get(flag) is not False:
            violations.append(
                f"{record_id}:current_sequence_review_freshness_{flag}_not_false"
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


def _gate_records_from_result_consumer(
    consumer: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    request = _mapping(consumer.get("promotion_policy_consumer_request"))
    gate = _mapping(request.get("sequence_policy_gate"))
    return _record_list(gate.get("policy_records"))


def _acceptance_templates_from_result_consumer_request(
    consumer: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    request = _mapping(consumer.get("promotion_policy_consumer_request"))
    return _record_list(request.get("promotion_acceptance_record_templates"))


def _gate_records_by_id(
    records: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(record["sequence_policy_gate_record_id"]): record
        for record in records
        if isinstance(record.get("sequence_policy_gate_record_id"), str)
        and record.get("sequence_policy_gate_record_id")
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


def _policy_record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("consumer_record_id"))
    if explicit:
        return explicit
    proposal_id = _string(record.get("proposal_id"))
    return f"{proposal_id if proposal_id else 'unknown_policy_consumer_record'}:{index}"


def _duplicate_ids(
    records: Tuple[Mapping[str, Any], ...],
    field: str,
    *,
    id_func,
) -> Tuple[str, ...]:
    values: Dict[str, list[str]] = {}
    for index, record in enumerate(records):
        value = _string(record.get(field))
        if not value:
            continue
        values.setdefault(value, []).append(id_func(record, index))
    return tuple(
        record_id
        for record_ids in values.values()
        if len(record_ids) > 1
        for record_id in record_ids
    )


def _gate_result_consumer_envelope_violations(
    consumer: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_GATE_RESULT_CONSUMER_TYPE:
        violations.append(
            "policy_consumer_result_consumer_gate_result_consumer_type_mismatch"
        )
    generated_at = consumer.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append(
            "policy_consumer_result_consumer_gate_result_consumer_generated_at_invalid"
        )
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_POLICY_GATE_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append(
            "policy_consumer_result_consumer_gate_result_consumer_method_id_mismatch"
        )
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_POLICY_GATE_RESULT_CONSUMER_STATUSES
    ):
        violations.append(
            "policy_consumer_result_consumer_gate_result_consumer_status_unknown"
        )
    if consumer.get("status") != REQUIRED_GATE_RESULT_CONSUMER_AVAILABLE_STATUS:
        violations.append(
            "policy_consumer_result_consumer_gate_result_consumer_status_not_available"
        )
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append(
            "policy_consumer_result_consumer_gate_result_consumer_read_only_not_true"
        )
    if safety.get("promotion_sequence_policy_gate_result_consumer_only") is not True:
        violations.append(
            "policy_consumer_result_consumer_gate_result_consumer_only_not_true"
        )
    if safety.get("promotion_policy_consumer_request_template_only") is not True:
        violations.append(
            "policy_consumer_result_consumer_gate_result_consumer_template_only_not_true"
        )
    for flag in REQUIRED_GATE_RESULT_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"policy_consumer_result_consumer_gate_result_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _gate_result_consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    gate_records: Tuple[Mapping[str, Any], ...],
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
    memory_trust_review_request_specs: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "expected_sequence_policy_gate_record_count": len(gate_records),
        "sequence_policy_gate_record_count": len(gate_records),
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
            violations.append(
                f"policy_consumer_result_consumer_gate_result_consumer_summary_{key}_mismatch"
            )
    if summary.get("promotion_policy_consumer_request_available") is not bool(gate_records):
        violations.append(
            "policy_consumer_result_consumer_gate_result_consumer_request_available_mismatch"
        )
    for flag in REQUIRED_GATE_RESULT_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"policy_consumer_result_consumer_gate_result_consumer_summary_{flag}_not_false"
            )
    expected_policy_keys = tuple(
        str(record["policy_key"])
        for record in gate_records
        if isinstance(record.get("policy_key"), str) and record.get("policy_key")
    )
    if _string_list(summary.get("eligible_policy_keys")) != expected_policy_keys:
        violations.append(
            "policy_consumer_result_consumer_gate_result_consumer_summary_policy_keys_mismatch"
        )
    if (
        promotion_acceptance_templates
        and summary.get("fresh_sequence_review_context_status") != "fresh"
    ):
        violations.append(
            "policy_consumer_result_consumer_gate_result_consumer_summary_freshness_not_fresh"
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
            "policy_consumer_result_consumer_gate_result_consumer_summary_template_proposal_ids_mismatch"
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
            "policy_consumer_result_consumer_gate_result_consumer_summary_review_request_spec_proposal_ids_mismatch"
        )
    return tuple(violations)


def _gate_result_consumer_request_violations(
    request: Mapping[str, Any],
    *,
    gate_records: Tuple[Mapping[str, Any], ...],
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    if (
        request.get("handler")
        != "consume_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy"
    ):
        violations.append("policy_consumer_result_consumer_request_handler_mismatch")
    if request.get("read_only_template") is not True:
        violations.append(
            "policy_consumer_result_consumer_request_read_only_template_not_true"
        )
    if request.get("promotion_policy_consumer_request_template_only") is not True:
        violations.append("policy_consumer_result_consumer_request_template_only_not_true")
    if request.get("promotion_policy_consumer_handler_invoked") is not False:
        violations.append("policy_consumer_result_consumer_request_handler_invoked_not_false")
    if request.get("promotion_policy_consumer_created") is not False:
        violations.append("policy_consumer_result_consumer_request_created_not_false")
    if request.get("proposals_required_from_operator") is not True:
        violations.append("policy_consumer_result_consumer_request_proposals_required_not_true")
    for flag in REQUIRED_GATE_RESULT_CONSUMER_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(f"policy_consumer_result_consumer_request_{flag}_not_false")
    if not gate_records:
        violations.append("policy_consumer_result_consumer_request_gate_records_missing")
    gate_proposal_ids = {
        str(record["proposal_id"])
        for record in gate_records
        if isinstance(record.get("proposal_id"), str) and record.get("proposal_id")
    }
    template_proposal_ids = {
        str(template["proposal_id"])
        for template in promotion_acceptance_templates
        if isinstance(template.get("proposal_id"), str) and template.get("proposal_id")
    }
    if len(template_proposal_ids) != len(promotion_acceptance_templates):
        violations.append(
            "policy_consumer_result_consumer_request_acceptance_template_proposal_ids_not_unique"
        )
    if gate_proposal_ids and not gate_proposal_ids.issubset(template_proposal_ids):
        violations.append(
            "policy_consumer_result_consumer_request_acceptance_templates_missing_gate_record"
        )
    request_specs = _record_list(request.get("memory_trust_review_request_specs"))
    request_spec_proposal_ids = {
        str(spec["proposal_id"])
        for spec in request_specs
        if isinstance(spec.get("proposal_id"), str) and spec.get("proposal_id")
    }
    if not template_proposal_ids.issubset(request_spec_proposal_ids):
        violations.append(
            "policy_consumer_result_consumer_request_review_request_spec_proposal_ids_mismatch"
        )
    for index, spec in enumerate(request_specs):
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


def _policy_consumer_envelope_violations(
    policy_consumer: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    if policy_consumer.get("type") != REQUIRED_POLICY_CONSUMER_TYPE:
        violations.append("policy_consumer_result_consumer_policy_consumer_type_mismatch")
    generated_at = policy_consumer.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("policy_consumer_result_consumer_policy_consumer_generated_at_invalid")
    if (
        policy_consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_METHOD_ID
    ):
        violations.append("policy_consumer_result_consumer_policy_consumer_method_id_mismatch")
    if (
        policy_consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_STATUSES
    ):
        violations.append("policy_consumer_result_consumer_policy_consumer_status_unknown")
    safety = _mapping(policy_consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("policy_consumer_result_consumer_policy_consumer_read_only_not_true")
    if safety.get("operator_required") is not True:
        violations.append(
            "policy_consumer_result_consumer_policy_consumer_operator_required_not_true"
        )
    if safety.get("policy_evidence_only") is not True:
        violations.append(
            "policy_consumer_result_consumer_policy_consumer_evidence_only_not_true"
        )
    for flag in REQUIRED_POLICY_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"policy_consumer_result_consumer_policy_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _policy_consumer_status_violations(
    policy_consumer: Mapping[str, Any],
    *,
    consumer_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = policy_consumer.get("status")
    if status == "sequence_memory_trust_promotion_policy_consumer_replacement_required":
        return ("policy_consumer_result_consumer_policy_consumer_replacement_required",)
    if status == REQUIRED_POLICY_CONSUMER_AVAILABLE_STATUS and not consumer_records:
        return ("policy_consumer_result_consumer_policy_consumer_available_without_records",)
    if status != REQUIRED_POLICY_CONSUMER_AVAILABLE_STATUS and consumer_records:
        return ("policy_consumer_result_consumer_policy_consumer_records_without_available_status",)
    if status != REQUIRED_POLICY_CONSUMER_AVAILABLE_STATUS:
        return ("policy_consumer_result_consumer_policy_consumer_records_available_required",)
    return ()


def _policy_consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    gate_records: Tuple[Mapping[str, Any], ...],
    consumer_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    eligible_gate_records = tuple(
        record for record in gate_records if record.get("sequence_policy_eligible") is True
    )
    violations = []
    expected_counts = {
        "sequence_policy_gate_record_count": len(gate_records),
        "eligible_sequence_policy_gate_record_count": len(eligible_gate_records),
        "consumer_record_count": len(consumer_records),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"policy_consumer_result_consumer_policy_consumer_summary_{key}_mismatch"
            )
    input_count = _summary_int(summary.get("input_proposal_count"))
    if input_count is None or input_count < len(consumer_records):
        violations.append(
            "policy_consumer_result_consumer_policy_consumer_summary_input_proposal_count_invalid"
        )
    if summary.get("policy_evidence_only") is not True:
        violations.append(
            "policy_consumer_result_consumer_policy_consumer_summary_evidence_only_not_true"
        )
    for flag in REQUIRED_POLICY_CONSUMER_SUMMARY_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"policy_consumer_result_consumer_policy_consumer_summary_{flag}_not_false"
            )
    expected_proposal_ids = tuple(
        str(record["proposal_id"])
        for record in consumer_records
        if isinstance(record.get("proposal_id"), str) and record.get("proposal_id")
    )
    if _string_list(summary.get("policy_consumer_proposal_ids")) != expected_proposal_ids:
        violations.append(
            "policy_consumer_result_consumer_policy_consumer_summary_proposal_ids_mismatch"
        )
    expected_policy_keys = tuple(
        str(record["policy_key"])
        for record in eligible_gate_records
        if isinstance(record.get("policy_key"), str) and record.get("policy_key")
    )
    if _string_list(summary.get("eligible_policy_keys")) != expected_policy_keys:
        violations.append(
            "policy_consumer_result_consumer_policy_consumer_summary_eligible_policy_keys_mismatch"
        )
    return tuple(violations)


def _policy_record_violations(
    record: Mapping[str, Any],
    *,
    gate_record: Mapping[str, Any],
    index: int,
) -> Tuple[str, ...]:
    record_id = _policy_record_id(record, index)
    violations = []
    for field in REQUIRED_POLICY_RECORD_FIELDS:
        if field == "proposal_original_index":
            if _summary_int(record.get(field)) is None:
                violations.append(f"{record_id}:{field}_missing")
            continue
        if not _string(record.get(field)):
            violations.append(f"{record_id}:{field}_missing")
    if not gate_record:
        violations.append(f"{record_id}:sequence_policy_gate_record_missing")
    else:
        for field in (
            "proposal_id",
            "sequence_policy_gate_record_id",
            "policy_key",
            "learning_scope",
            "value_type",
            "source_idempotency_key",
            "target_policy_state",
            "sequence_policy_gate_status",
        ):
            if record.get(field) != gate_record.get(field):
                violations.append(f"{record_id}:{field}_mismatch")
        if record.get("current_sequence_review_freshness") != gate_record.get(
            "current_sequence_review_freshness"
        ):
            violations.append(f"{record_id}:current_sequence_review_freshness_mismatch")
        violations.extend(
            _context_violations(
                record_id,
                _mapping(gate_record.get("current_sequence_review_freshness")),
            )
        )
    violations.extend(
        _context_violations(
            record_id,
            _mapping(record.get("current_sequence_review_freshness")),
        )
    )
    for field in ("base_policy_score", "adjusted_policy_score"):
        score = _safe_float(record.get(field))
        if score is None or score < 0.0 or score > 1.0:
            violations.append(f"{record_id}:{field}_invalid")
    if record.get("sequence_policy_eligible") is not True:
        violations.append(f"{record_id}:sequence_policy_eligible_not_true")
    if record.get("policy_consumer_required_before_ordering") is not True:
        violations.append(f"{record_id}:policy_consumer_required_before_ordering_not_true")
    if record.get("read_only") is not True:
        violations.append(f"{record_id}:read_only_not_true")
    if record.get("policy_evidence_only") is not True:
        violations.append(f"{record_id}:policy_evidence_only_not_true")
    for flag in REQUIRED_POLICY_RECORD_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(f"{record_id}:{flag}_not_false")
    return tuple(violations)


def _policy_record_identity_violations(
    *,
    gate_records: Tuple[Mapping[str, Any], ...],
    consumer_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    gate_by_id = _gate_records_by_id(gate_records)
    violations = []
    for index, record in enumerate(consumer_records):
        gate_record_id = _string(record.get("sequence_policy_gate_record_id"))
        gate_record = _mapping(gate_by_id.get(gate_record_id))
        violations.extend(
            _policy_record_violations(
                record,
                gate_record=gate_record,
                index=index,
            )
        )
    return tuple(violations)


def _policy_record_template_identity_violations(
    *,
    gate_records: Tuple[Mapping[str, Any], ...],
    consumer_records: Tuple[Mapping[str, Any], ...],
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
    memory_trust_review_request_specs: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    gate_by_id = _gate_records_by_id(gate_records)
    templates_by_proposal_id = _templates_by_proposal_id(promotion_acceptance_templates)
    request_specs_by_proposal_id = _request_specs_by_proposal_id(
        memory_trust_review_request_specs
    )
    violations = []
    for consumer_record in consumer_records:
        proposal_id = _string(consumer_record.get("proposal_id"))
        template = _mapping(templates_by_proposal_id.get(proposal_id))
        if not template:
            violations.append(f"{proposal_id}:acceptance_template_missing")
            continue
        gate_record_id = _string(consumer_record.get("sequence_policy_gate_record_id"))
        gate_record = _mapping(gate_by_id.get(gate_record_id))
        expected_pairs = {
            "proposal_id": consumer_record.get("proposal_id"),
            "learning_scope": consumer_record.get("learning_scope"),
            "value_type": consumer_record.get("value_type"),
            "idempotency_key": consumer_record.get("source_idempotency_key"),
            "previous_policy_state": gate_record.get("rollback_target_policy_state"),
            "rollback_strategy": gate_record.get("rollback_strategy"),
        }
        for field, expected in expected_pairs.items():
            if template.get(field) != expected:
                violations.append(f"{proposal_id}:{field}_consumer_record_mismatch")
        if _sequence_review_freshness_core(
            _mapping(template.get("current_sequence_review_freshness"))
        ) != _sequence_review_freshness_core(
            _mapping(consumer_record.get("current_sequence_review_freshness"))
        ):
            violations.append(
                f"{proposal_id}:current_sequence_review_freshness_consumer_record_mismatch"
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
            if _sequence_review_freshness_core(
                _mapping(consumer_record.get("current_sequence_review_freshness"))
            ) != _sequence_review_freshness_core(
                _mapping(request_spec.get("current_sequence_review_freshness"))
            ):
                violations.append(
                    f"{proposal_id}:consumer_record_current_sequence_review_freshness_request_spec_mismatch"
                )
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer(
    *,
    promotion_sequence_policy_gate_result_consumer: Mapping[str, Any] | None = None,
    promotion_policy_consumer_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify policy-consumer evidence and emit an ordering-preview request."""

    gate_result_consumer = _mapping(promotion_sequence_policy_gate_result_consumer)
    policy_consumer = _mapping(promotion_policy_consumer_result)
    gate_request = _mapping(gate_result_consumer.get("promotion_policy_consumer_request"))
    gate_records = _gate_records_from_result_consumer(gate_result_consumer)
    promotion_acceptance_templates = (
        _acceptance_templates_from_result_consumer_request(gate_result_consumer)
    )
    memory_trust_review_request_specs = _record_list(
        gate_request.get("memory_trust_review_request_specs")
    )
    consumer_records = _record_list(policy_consumer.get("consumer_records"))
    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_gate_result_consumer_envelope_violations(gate_result_consumer),
                *_gate_result_consumer_summary_violations(
                    _mapping(gate_result_consumer.get("summary")),
                    gate_records=gate_records,
                    promotion_acceptance_templates=promotion_acceptance_templates,
                    memory_trust_review_request_specs=memory_trust_review_request_specs,
                ),
                *_gate_result_consumer_request_violations(
                    gate_request,
                    gate_records=gate_records,
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
                *_policy_consumer_envelope_violations(policy_consumer),
                *_policy_consumer_status_violations(
                    policy_consumer,
                    consumer_records=consumer_records,
                ),
                *(
                    (
                        *_policy_consumer_summary_violations(
                            _mapping(policy_consumer.get("summary")),
                            gate_records=gate_records,
                            consumer_records=consumer_records,
                        ),
                        *_duplicate_ids(
                            consumer_records,
                            "consumer_record_id",
                            id_func=_policy_record_id,
                        ),
                        *_duplicate_ids(
                            consumer_records,
                            "proposal_id",
                            id_func=_policy_record_id,
                        ),
                        *_duplicate_ids(
                            consumer_records,
                            "sequence_policy_gate_record_id",
                            id_func=_policy_record_id,
                        ),
                        *_duplicate_ids(
                            consumer_records,
                            "policy_key",
                            id_func=_policy_record_id,
                        ),
                        *_policy_record_identity_violations(
                            gate_records=gate_records,
                            consumer_records=consumer_records,
                        ),
                        *_policy_record_template_identity_violations(
                            gate_records=gate_records,
                            consumer_records=consumer_records,
                            promotion_acceptance_templates=promotion_acceptance_templates,
                            memory_trust_review_request_specs=memory_trust_review_request_specs,
                        ),
                    )
                    if consumer_records
                    or policy_consumer.get("status") == REQUIRED_POLICY_CONSUMER_AVAILABLE_STATUS
                    else ()
                ),
            )
        )
    )
    if not gate_result_consumer or not policy_consumer:
        invalid_record_ids: Tuple[str, ...] = ()
    else:
        blocking_only = (
            "policy_consumer_result_consumer_policy_consumer_records_available_required",
        )
        invalid_record_ids = tuple(
            item for item in raw_invalid_record_ids if item not in blocking_only
        )

    if not gate_result_consumer or not policy_consumer:
        status = "sequence_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer_not_ready"
        blocked_by = ("promotion_sequence_policy_gate_result_consumer_and_policy_consumer_required",)
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_memory_trust_promotion_policy_consumer_evidence",
        )
    elif policy_consumer.get("status") != REQUIRED_POLICY_CONSUMER_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer_blocked"
        blocked_by = (
            "sequence_memory_trust_promotion_policy_consumer_records_required_before_ordering_preview",
        )
    elif not consumer_records:
        status = "sequence_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer_not_ready"
        blocked_by = ("sequence_memory_trust_promotion_policy_consumer_records_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_request_available"
        blocked_by = (
            "operator_may_run_read_only_sequence_memory_trust_promotion_ordering_preview",
        )

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_request_available"
    )
    ordering_preview_request: Dict[str, Any] = {
        "handler": (
            "preview_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering"
        ),
        "read_only_template": True,
        "promotion_policy_consumer_result_consumer_only": True,
        "promotion_ordering_preview_request_template_only": True,
        "promotion_ordering_preview_handler_invoked": False,
        "promotion_ordering_preview_created": False,
        "proposals_required_from_operator": True,
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
        ordering_preview_request["policy_consumer"] = policy_consumer
        ordering_preview_request["promotion_acceptance_record_templates"] = list(
            promotion_acceptance_templates
        )
        ordering_preview_request["memory_trust_review_request_specs"] = list(
            memory_trust_review_request_specs
        )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_POLICY_CONSUMER_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_POLICY_GATE_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_policy_consumer_result_consumer_only": True,
            "promotion_ordering_preview_request_template_only": True,
            "requires_promotion_sequence_policy_gate_result_consumer": True,
            "requires_promotion_policy_consumer_result": True,
            "requires_policy_consumer_records": True,
            "promotion_ordering_preview_handler_invoked": False,
            "promotion_ordering_preview_created": False,
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
            "expected_policy_consumer_record_count": len(
                tuple(
                    record
                    for record in gate_records
                    if record.get("sequence_policy_eligible") is True
                )
            ),
            "policy_consumer_record_count": len(consumer_records),
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
            "promotion_policy_consumer_result_consumer_only": True,
            "promotion_ordering_preview_request_template_only": True,
            "promotion_ordering_preview_request_available": request_available,
            "promotion_ordering_preview_handler_invoked": False,
            "promotion_ordering_preview_created": False,
            "requires_policy_consumer_records": True,
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
            "policy_consumer_proposal_ids": [
                str(record["proposal_id"])
                for record in consumer_records
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
            "policy_keys": [
                str(record["policy_key"])
                for record in consumer_records
                if isinstance(record.get("policy_key"), str) and record.get("policy_key")
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion sequence-policy gate result consumer and policy-consumer result."
                if not gate_result_consumer or not policy_consumer
                else (
                    "Replace invalid sequence-memory trust promotion policy-consumer evidence before ordering preview."
                    if invalid_record_ids
                    else (
                        "Run the read-only policy consumer before ordering preview."
                        if policy_consumer.get("status") != REQUIRED_POLICY_CONSUMER_AVAILABLE_STATUS
                        else "Run ordering preview with matching preview proposals; this consumer does not mutate order, queue, policy, memory, model, or live state."
                    )
                )
            ),
        },
        "promotion_ordering_preview_request": ordering_preview_request,
    }
