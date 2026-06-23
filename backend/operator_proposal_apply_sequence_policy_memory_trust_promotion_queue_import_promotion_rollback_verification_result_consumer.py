"""Read-only consumer for sequence-policy promotion rollback-verification results.

This verifier connects a queue-import promotion apply-audit result consumer to
the existing sequence-policy gate builder. It verifies rollback-verification
records against the apply-audit provenance and emits only a sequence-policy gate
request template. It does not invoke the gate, consume policy, reorder
proposals, import queues, reserve idempotency, mutate runtime or append-only
memory, mutate policy stores, or touch mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_APPLY_AUDIT_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_APPLY_AUDIT_RESULT_CONSUMER_STATUSES,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_APPLY_AUDIT_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_APPLY_AUDIT_RESULT_CONSUMER_STATUSES,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ROLLBACK_VERIFICATION_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ROLLBACK_VERIFICATION_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_request_available",
}
REQUIRED_APPLY_AUDIT_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer"
)
REQUIRED_APPLY_AUDIT_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_request_available"
)
REQUIRED_ROLLBACK_VERIFICATION_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification"
)
REQUIRED_ROLLBACK_VERIFICATION_AVAILABLE_STATUS = (
    "operator_sequence_memory_trust_promotion_rollback_verified_records_available"
)
TARGET_POLICY_STORE = "operator_sequence_policy_trust_gate"
REQUIRED_CONSUMER_FALSE_FLAGS = (
    "promotion_rollback_verification_handler_invoked",
    "promotion_rollback_verification_created",
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
REQUIRED_ROLLBACK_FALSE_SAFETY_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_verification",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_ROLLBACK_FALSE_SUMMARY_FLAGS = (
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_verification",
    "live_write",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_AUDIT_RECORD_FIELDS = (
    "promotion_apply_audit_id",
    "promotion_executor_apply_id",
    "proposal_id",
    "learning_scope",
    "value_type",
    "source_idempotency_key",
    "policy_key",
    "target_policy_store",
    "target_policy_path",
    "target_policy_state",
    "rollback_target_policy_state",
    "rollback_strategy",
)
REQUIRED_VERIFIED_RECORD_FIELDS = (
    "rollback_verification_record_id",
    "rollback_verification_id",
    "promotion_apply_audit_id",
    "promotion_executor_apply_id",
    "proposal_id",
    "learning_scope",
    "value_type",
    "source_idempotency_key",
    "policy_key",
    "target_policy_store",
    "target_policy_path",
    "target_policy_state",
    "rollback_target_policy_state",
    "rollback_strategy",
    "verification_evidence_status",
    "decision",
    "verified_by",
    "verified_at",
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
REQUIRED_VERIFIED_RECORD_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_verification",
    "live_widening_allowed",
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


def _context_core(payload: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return (
        ("status", payload.get("status") if isinstance(payload.get("status"), str) else ""),
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


def _audit_record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("promotion_apply_audit_id"))
    if explicit:
        return explicit
    apply_id = _string(record.get("promotion_executor_apply_id"))
    return f"{apply_id if apply_id else 'unknown_sequence_apply_audit'}:{index}"


def _verified_record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("rollback_verification_record_id"))
    if explicit:
        return explicit
    apply_id = _string(record.get("promotion_apply_audit_id"))
    return f"{apply_id if apply_id else 'unknown_sequence_rollback_verification'}:{index}"


def _target_policy_path(policy_key: str) -> str:
    return f"{TARGET_POLICY_STORE}/{policy_key}/trust_state"


def _audit_records_from_consumer_request(
    consumer: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    request = _mapping(consumer.get("promotion_rollback_verification_request"))
    apply_audit = _mapping(request.get("apply_audit"))
    return _record_list(apply_audit.get("audit_records"))


def _acceptance_templates_from_consumer_request(
    consumer: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    request = _mapping(consumer.get("promotion_rollback_verification_request"))
    return _record_list(request.get("promotion_acceptance_record_templates"))


def _request_specs_from_consumer_request(
    consumer: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    request = _mapping(consumer.get("promotion_rollback_verification_request"))
    return _record_list(request.get("memory_trust_review_request_specs"))


def _audit_records_by_id(
    records: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(record["promotion_apply_audit_id"]): record
        for record in records
        if isinstance(record.get("promotion_apply_audit_id"), str)
        and record.get("promotion_apply_audit_id")
    }


def _templates_by_contract_record_id(
    templates: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(template["promotion_contract_record_id"]): template
        for template in templates
        if isinstance(template.get("promotion_contract_record_id"), str)
        and template.get("promotion_contract_record_id")
    }


def _request_specs_by_proposal_id(
    request_specs: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(spec["proposal_id"]): spec
        for spec in request_specs
        if isinstance(spec.get("proposal_id"), str) and spec.get("proposal_id")
    }


def _consumer_envelope_violations(consumer: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_APPLY_AUDIT_RESULT_CONSUMER_TYPE:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_type_mismatch"
        )
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_APPLY_AUDIT_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_method_id_mismatch"
        )
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_APPLY_AUDIT_RESULT_CONSUMER_STATUSES
    ):
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_status_unknown"
        )
    if consumer.get("status") != REQUIRED_APPLY_AUDIT_RESULT_CONSUMER_AVAILABLE_STATUS:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_status_not_available"
        )
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_read_only_not_true"
        )
    if safety.get("promotion_apply_audit_result_consumer_only") is not True:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_only_not_true"
        )
    if safety.get("promotion_rollback_verification_request_template_only") is not True:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_template_only_not_true"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    audit_records: Tuple[Mapping[str, Any], ...],
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
    memory_trust_review_request_specs: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "expected_apply_audit_record_count": len(audit_records),
        "apply_audit_record_count": len(audit_records),
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
                f"sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_summary_{key}_mismatch"
            )
    if summary.get("promotion_rollback_verification_request_available") is not bool(audit_records):
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_request_available_mismatch"
        )
    if (
        promotion_acceptance_templates
        and summary.get("fresh_sequence_review_context_status") != "fresh"
    ):
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_summary_freshness_not_fresh"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_summary_{flag}_not_false"
            )
    expected_policy_keys = tuple(
        str(record["policy_key"])
        for record in audit_records
        if isinstance(record.get("policy_key"), str) and record.get("policy_key")
    )
    expected_paths = tuple(
        str(record["target_policy_path"])
        for record in audit_records
        if isinstance(record.get("target_policy_path"), str)
        and record.get("target_policy_path")
    )
    expected_keys = tuple(
        str(record["idempotency_key"])
        for record in audit_records
        if isinstance(record.get("idempotency_key"), str)
        and record.get("idempotency_key")
    )
    if _string_list(summary.get("audited_policy_keys")) != expected_policy_keys:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_summary_policy_keys_mismatch"
        )
    if _string_list(summary.get("target_policy_paths")) != expected_paths:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_summary_target_paths_mismatch"
        )
    if _string_list(summary.get("idempotency_keys")) != expected_keys:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_summary_idempotency_keys_mismatch"
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
            "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_summary_template_proposal_ids_mismatch"
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
            "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_consumer_summary_review_request_spec_proposal_ids_mismatch"
        )
    return tuple(violations)


def _consumer_request_violations(
    request: Mapping[str, Any],
    *,
    audit_records: Tuple[Mapping[str, Any], ...],
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    if (
        request.get("handler")
        != "verify_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback"
    ):
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_request_handler_mismatch"
        )
    if request.get("read_only_template") is not True:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_request_read_only_template_not_true"
        )
    if request.get("promotion_rollback_verification_request_template_only") is not True:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_request_template_only_not_true"
        )
    if request.get("promotion_rollback_verification_handler_invoked") is not False:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_request_handler_invoked_not_false"
        )
    if request.get("promotion_rollback_verification_created") is not False:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_request_created_not_false"
        )
    if request.get("verification_records_required_from_operator") is not True:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_request_verification_records_requirement_missing"
        )
    for flag in REQUIRED_REQUEST_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_rollback_verification_result_consumer_request_{flag}_not_false"
            )
    if not _record_list(_mapping(request.get("apply_audit")).get("audit_records")):
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_request_audit_records_missing"
        )
    audit_contract_ids = {
        str(record["promotion_contract_record_id"])
        for record in audit_records
        if isinstance(record.get("promotion_contract_record_id"), str)
        and record.get("promotion_contract_record_id")
    }
    template_contract_ids = {
        str(template["promotion_contract_record_id"])
        for template in promotion_acceptance_templates
        if isinstance(template.get("promotion_contract_record_id"), str)
        and template.get("promotion_contract_record_id")
    }
    if len(template_contract_ids) != len(promotion_acceptance_templates):
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_request_acceptance_template_contract_ids_not_unique"
        )
    if audit_contract_ids and not audit_contract_ids.issubset(template_contract_ids):
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_request_acceptance_templates_missing_audit_record"
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
            "sequence_policy_promotion_rollback_verification_result_consumer_request_review_request_spec_proposal_ids_mismatch"
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


def _rollback_envelope_violations(
    rollback_verification: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    if rollback_verification.get("type") != REQUIRED_ROLLBACK_VERIFICATION_TYPE:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_rollback_type_mismatch"
        )
    generated_at = rollback_verification.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_rollback_generated_at_invalid"
        )
    if (
        rollback_verification.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_METHOD_ID
    ):
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_rollback_method_id_mismatch"
        )
    if (
        rollback_verification.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ROLLBACK_VERIFICATION_STATUSES
    ):
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_rollback_status_unknown"
        )
    safety = _mapping(rollback_verification.get("safety"))
    if safety.get("read_only") is not True:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_rollback_read_only_not_true"
        )
    if safety.get("operator_required") is not True:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_rollback_operator_required_not_true"
        )
    if safety.get("separate_sequence_policy_gate_required") is not True:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_rollback_separate_gate_not_true"
        )
    for flag in REQUIRED_ROLLBACK_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_rollback_verification_result_consumer_rollback_{flag}_not_false"
            )
    return tuple(violations)


def _rollback_safety_state_violations(
    rollback_verification: Mapping[str, Any],
    *,
    verified_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    safety = _mapping(rollback_verification.get("safety"))
    expected = bool(verified_records)
    violations = []
    if safety.get("rollback_verified") is not expected:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_rollback_verified_mismatch"
        )
    if safety.get("usable_as_sequence_policy_gate_evidence") is not expected:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_rollback_policy_evidence_mismatch"
        )
    return tuple(violations)


def _rollback_status_violations(
    rollback_verification: Mapping[str, Any],
    *,
    verified_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = rollback_verification.get("status")
    if status == "sequence_memory_trust_promotion_rollback_verification_replacement_required":
        return (
            "sequence_policy_promotion_rollback_verification_result_consumer_rollback_replacement_required",
        )
    if status == REQUIRED_ROLLBACK_VERIFICATION_AVAILABLE_STATUS and not verified_records:
        return (
            "sequence_policy_promotion_rollback_verification_result_consumer_rollback_available_without_records",
        )
    if status != REQUIRED_ROLLBACK_VERIFICATION_AVAILABLE_STATUS and verified_records:
        return (
            "sequence_policy_promotion_rollback_verification_result_consumer_rollback_records_without_available_status",
        )
    if status != REQUIRED_ROLLBACK_VERIFICATION_AVAILABLE_STATUS:
        return (
            "sequence_policy_promotion_rollback_verification_result_consumer_verified_records_available_required",
        )
    return ()


def _rollback_summary_violations(
    summary: Mapping[str, Any],
    *,
    audit_records: Tuple[Mapping[str, Any], ...],
    verified_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "apply_audit_record_count": len(audit_records),
        "verification_record_count": len(verified_records),
        "rollback_verified_record_count": len(verified_records),
        "invalid_record_count": 0,
        "unverified_apply_audit_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_promotion_rollback_verification_result_consumer_rollback_summary_{key}_mismatch"
            )
    if summary.get("target_policy_store") != TARGET_POLICY_STORE:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_rollback_summary_target_store_mismatch"
        )
    if summary.get("rollback_verified") is not bool(verified_records):
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_rollback_summary_rollback_verified_mismatch"
        )
    if summary.get("usable_as_sequence_policy_gate_evidence") is not bool(verified_records):
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_rollback_summary_policy_evidence_mismatch"
        )
    if summary.get("separate_sequence_policy_gate_required") is not True:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_rollback_summary_separate_gate_not_true"
        )
    for flag in REQUIRED_ROLLBACK_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_rollback_verification_result_consumer_rollback_summary_{flag}_not_false"
            )
    expected_policy_keys = tuple(
        record["policy_key"]
        for record in verified_records
        if _string(record.get("policy_key"))
    )
    expected_rollback_states = tuple(
        record["rollback_target_policy_state"]
        for record in verified_records
        if _string(record.get("rollback_target_policy_state"))
    )
    if _string_list(summary.get("verified_policy_keys")) != expected_policy_keys:
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_rollback_summary_policy_keys_mismatch"
        )
    if (
        _string_list(summary.get("verified_rollback_target_policy_states"))
        != expected_rollback_states
    ):
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_rollback_summary_rollback_states_mismatch"
        )
    return tuple(violations)


def _audit_record_violations(record: Mapping[str, Any], index: int) -> Tuple[str, ...]:
    record_id = _audit_record_id(record, index)
    policy_key = _string(record.get("policy_key"))
    violations = []
    for field in REQUIRED_AUDIT_RECORD_FIELDS:
        if not _string(record.get(field)):
            violations.append(f"{record_id}:{field}_missing")
    if record.get("target_policy_store") != TARGET_POLICY_STORE:
        violations.append(f"{record_id}:target_policy_store_mismatch")
    if record.get("target_policy_path") != _target_policy_path(policy_key):
        violations.append(f"{record_id}:target_policy_path_mismatch")
    if record.get("rollback_verified") is not False:
        violations.append(f"{record_id}:audit_rollback_verified_not_false")
    if record.get("read_only") is not True:
        violations.append(f"{record_id}:audit_read_only_not_true")
    return tuple(violations)


def _verified_record_violations(
    record: Mapping[str, Any],
    *,
    audit_record: Mapping[str, Any],
    index: int,
) -> Tuple[str, ...]:
    record_id = _verified_record_id(record, index)
    policy_key = _string(record.get("policy_key"))
    violations = []
    for field in REQUIRED_VERIFIED_RECORD_FIELDS:
        if not _string(record.get(field)):
            violations.append(f"{record_id}:{field}_missing")
    verified_at = _string(record.get("verified_at"))
    if verified_at and not _is_utc_timestamp(verified_at):
        violations.append(f"{record_id}:verified_at_invalid")
    if not audit_record:
        violations.append(f"{record_id}:apply_audit_record_missing")
    else:
        for field in REQUIRED_AUDIT_RECORD_FIELDS:
            if field in {"rollback_strategy"} and record.get(field) == audit_record.get(field):
                continue
            if field in record and record.get(field) != audit_record.get(field):
                violations.append(f"{record_id}:{field}_mismatch")
    if record.get("target_policy_store") != TARGET_POLICY_STORE:
        violations.append(f"{record_id}:target_policy_store_mismatch")
    if record.get("target_policy_path") != _target_policy_path(policy_key):
        violations.append(f"{record_id}:target_policy_path_mismatch")
    if record.get("decision") != "verified":
        violations.append(f"{record_id}:decision_not_verified")
    if record.get("verification_evidence_status") != "reviewed":
        violations.append(f"{record_id}:evidence_status_not_reviewed")
    if record.get("rollback_verified") is not True:
        violations.append(f"{record_id}:rollback_verified_not_true")
    if record.get("read_only") is not True:
        violations.append(f"{record_id}:read_only_not_true")
    if record.get("operator_review_required") is not True:
        violations.append(f"{record_id}:operator_review_required_not_true")
    if record.get("separate_sequence_policy_gate_required") is not True:
        violations.append(f"{record_id}:separate_sequence_policy_gate_required_not_true")
    if record.get("usable_as_sequence_policy_gate_evidence") is not True:
        violations.append(f"{record_id}:usable_as_sequence_policy_gate_evidence_not_true")
    violations.extend(
        _context_violations(
            record_id,
            _mapping(record.get("current_sequence_review_freshness")),
        )
    )
    for flag in REQUIRED_VERIFIED_RECORD_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(f"{record_id}:{flag}_not_false")
    return tuple(violations)


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


def _rollback_identity_violations(
    *,
    audit_records: Tuple[Mapping[str, Any], ...],
    verified_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    audit_by_id = _audit_records_by_id(audit_records)
    verified_by_audit_id = {
        str(record["promotion_apply_audit_id"]): record
        for record in verified_records
        if isinstance(record.get("promotion_apply_audit_id"), str)
        and record.get("promotion_apply_audit_id")
    }
    violations = []
    if set(verified_by_audit_id) != set(audit_by_id):
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_apply_audit_ids_mismatch"
        )
    if len(verified_by_audit_id) != len(verified_records):
        violations.append(
            "sequence_policy_promotion_rollback_verification_result_consumer_duplicate_apply_audit_ids"
        )
    for index, audit_record in enumerate(audit_records):
        audit_id = _string(audit_record.get("promotion_apply_audit_id"))
        violations.extend(_audit_record_violations(audit_record, index))
        verified_record = _mapping(verified_by_audit_id.get(audit_id))
        if not verified_record:
            violations.append(f"{audit_id}:rollback_verified_record_missing")
            continue
        violations.extend(
            _verified_record_violations(
                verified_record,
                audit_record=audit_record,
                index=index,
            )
        )
    return tuple(violations)


def _verified_record_template_identity_violations(
    *,
    audit_records: Tuple[Mapping[str, Any], ...],
    verified_records: Tuple[Mapping[str, Any], ...],
    promotion_acceptance_templates: Tuple[Mapping[str, Any], ...],
    memory_trust_review_request_specs: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    audit_by_id = _audit_records_by_id(audit_records)
    templates_by_contract_record_id = _templates_by_contract_record_id(
        promotion_acceptance_templates
    )
    request_specs_by_proposal_id = _request_specs_by_proposal_id(
        memory_trust_review_request_specs
    )
    violations = []
    for verified_record in verified_records:
        audit_id = _string(verified_record.get("promotion_apply_audit_id"))
        audit_record = _mapping(audit_by_id.get(audit_id))
        contract_id = _string(audit_record.get("promotion_contract_record_id"))
        proposal_id = _string(verified_record.get("proposal_id")) or contract_id
        template = _mapping(templates_by_contract_record_id.get(contract_id))
        if not template:
            violations.append(f"{proposal_id}:acceptance_template_missing")
            continue
        expected_pairs = {
            "promotion_contract_record_id": contract_id,
            "promotion_candidate_id": audit_record.get("promotion_candidate_id"),
            "proposal_id": verified_record.get("proposal_id"),
            "learning_scope": verified_record.get("learning_scope"),
            "value_type": verified_record.get("value_type"),
            "idempotency_key": verified_record.get("source_idempotency_key"),
            "previous_policy_state": verified_record.get(
                "rollback_target_policy_state"
            ),
            "rollback_strategy": verified_record.get("rollback_strategy"),
        }
        for field, expected in expected_pairs.items():
            if template.get(field) != expected:
                violations.append(f"{proposal_id}:{field}_verified_record_mismatch")
        request_spec = _mapping(request_specs_by_proposal_id.get(proposal_id))
        if request_spec:
            if template.get("source_memory_trust_review_request_spec_id") != request_spec.get(
                "memory_trust_review_request_spec_id"
            ):
                violations.append(
                    f"{proposal_id}:source_memory_trust_review_request_spec_id_mismatch"
                )
            if _context_core(
                _mapping(template.get("current_sequence_review_freshness"))
            ) != _context_core(
                _mapping(request_spec.get("current_sequence_review_freshness"))
            ):
                violations.append(
                    f"{proposal_id}:current_sequence_review_freshness_request_spec_mismatch"
                )
            if _context_core(
                _mapping(verified_record.get("current_sequence_review_freshness"))
            ) != _context_core(
                _mapping(request_spec.get("current_sequence_review_freshness"))
            ):
                violations.append(
                    f"{proposal_id}:verified_current_sequence_review_freshness_request_spec_mismatch"
                )
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer(
    *,
    promotion_apply_audit_result_consumer: Mapping[str, Any] | None = None,
    promotion_rollback_verification_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify rollback evidence and emit a sequence-policy gate request."""

    consumer = _mapping(promotion_apply_audit_result_consumer)
    rollback = _mapping(promotion_rollback_verification_result)
    request = _mapping(consumer.get("promotion_rollback_verification_request"))
    audit_records = _audit_records_from_consumer_request(consumer)
    promotion_acceptance_templates = _acceptance_templates_from_consumer_request(
        consumer
    )
    memory_trust_review_request_specs = _request_specs_from_consumer_request(consumer)
    verified_records = _record_list(rollback.get("verified_records"))
    rollback_status_available = (
        rollback.get("status") == REQUIRED_ROLLBACK_VERIFICATION_AVAILABLE_STATUS
    )
    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_consumer_envelope_violations(consumer),
                *_consumer_summary_violations(
                    _mapping(consumer.get("summary")),
                    audit_records=audit_records,
                    promotion_acceptance_templates=promotion_acceptance_templates,
                    memory_trust_review_request_specs=memory_trust_review_request_specs,
                ),
                *_consumer_request_violations(
                    request,
                    audit_records=audit_records,
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
                *_rollback_envelope_violations(rollback),
                *_rollback_safety_state_violations(
                    rollback,
                    verified_records=verified_records,
                ),
                *_rollback_status_violations(
                    rollback,
                    verified_records=verified_records,
                ),
                *(
                    (
                        *_rollback_summary_violations(
                            _mapping(rollback.get("summary")),
                            audit_records=audit_records,
                            verified_records=verified_records,
                        ),
                        *_duplicate_ids(
                            verified_records,
                            "rollback_verification_record_id",
                            id_func=_verified_record_id,
                        ),
                        *_duplicate_ids(
                            verified_records,
                            "promotion_apply_audit_id",
                            id_func=_verified_record_id,
                        ),
                        *_duplicate_ids(
                            verified_records,
                            "policy_key",
                            id_func=_verified_record_id,
                        ),
                        *_rollback_identity_violations(
                            audit_records=audit_records,
                            verified_records=verified_records,
                        ),
                        *_verified_record_template_identity_violations(
                            audit_records=audit_records,
                            verified_records=verified_records,
                            promotion_acceptance_templates=promotion_acceptance_templates,
                            memory_trust_review_request_specs=memory_trust_review_request_specs,
                        ),
                    )
                    if rollback_status_available or verified_records
                    else ()
                ),
            )
        )
    )
    if not consumer or not rollback:
        invalid_record_ids: Tuple[str, ...] = ()
    else:
        blocking_only = (
            "sequence_policy_promotion_rollback_verification_result_consumer_verified_records_available_required",
        )
        invalid_record_ids = tuple(
            item for item in raw_invalid_record_ids if item not in blocking_only
        )

    if not consumer or not rollback:
        status = "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer_not_ready"
        blocked_by = ("promotion_apply_audit_consumer_and_rollback_verification_required",)
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_memory_trust_promotion_rollback_verification_evidence",
        )
    elif rollback.get("status") != REQUIRED_ROLLBACK_VERIFICATION_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer_blocked"
        blocked_by = (
            "verified_sequence_memory_trust_promotion_rollback_records_required_before_sequence_policy_gate",
        )
    elif not verified_records:
        status = "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer_not_ready"
        blocked_by = (
            "sequence_memory_trust_promotion_rollback_verified_records_required",
        )
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_request_available"
        blocked_by = (
            "operator_may_run_read_only_sequence_memory_trust_promotion_sequence_policy_gate",
        )

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_request_available"
    )
    sequence_policy_gate_request: Dict[str, Any] = {
        "handler": (
            "gate_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy"
        ),
        "read_only_template": True,
        "promotion_rollback_verification_result_consumer_only": True,
        "promotion_sequence_policy_gate_request_template_only": True,
        "promotion_sequence_policy_gate_handler_invoked": False,
        "promotion_sequence_policy_gate_created": False,
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
        "can_import_to_queue": False,
        "can_apply_live": False,
    }
    if request_available:
        sequence_policy_gate_request["rollback_verification"] = rollback
        sequence_policy_gate_request["promotion_acceptance_record_templates"] = list(
            promotion_acceptance_templates
        )
        sequence_policy_gate_request["memory_trust_review_request_specs"] = list(
            memory_trust_review_request_specs
        )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ROLLBACK_VERIFICATION_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_APPLY_AUDIT_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_rollback_verification_result_consumer_only": True,
            "promotion_sequence_policy_gate_request_template_only": True,
            "requires_promotion_apply_audit_result_consumer": True,
            "requires_promotion_rollback_verification_result": True,
            "requires_verified_rollback_records": True,
            "promotion_sequence_policy_gate_handler_invoked": False,
            "promotion_sequence_policy_gate_created": False,
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
            "can_import_to_queue": False,
            "can_apply_live": False,
        },
        "summary": {
            "expected_rollback_verified_record_count": len(audit_records),
            "rollback_verified_record_count": len(verified_records),
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
            "target_policy_store": TARGET_POLICY_STORE,
            "promotion_rollback_verification_result_consumer_only": True,
            "promotion_sequence_policy_gate_request_template_only": True,
            "promotion_sequence_policy_gate_request_available": request_available,
            "promotion_sequence_policy_gate_handler_invoked": False,
            "promotion_sequence_policy_gate_created": False,
            "requires_verified_rollback_records": True,
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
            "can_import_to_queue": False,
            "can_apply_live": False,
            "verified_policy_keys": [
                str(record["policy_key"])
                for record in verified_records
                if isinstance(record.get("policy_key"), str) and record.get("policy_key")
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
                for record in verified_records
                if isinstance(record.get("target_policy_path"), str)
                and record.get("target_policy_path")
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion apply-audit result consumer and rollback-verification result."
                if not consumer or not rollback
                else (
                    "Replace invalid sequence-memory trust promotion rollback-verification evidence before sequence-policy gate."
                    if invalid_record_ids
                    else (
                        "Run rollback verification with operator-reviewed verification records before sequence-policy gate."
                        if rollback.get("status") != REQUIRED_ROLLBACK_VERIFICATION_AVAILABLE_STATUS
                        else "Run the read-only sequence-policy gate request; this consumer does not create policy records or mutate state."
                    )
                )
            ),
        },
        "promotion_sequence_policy_gate_request": sequence_policy_gate_request,
    }
