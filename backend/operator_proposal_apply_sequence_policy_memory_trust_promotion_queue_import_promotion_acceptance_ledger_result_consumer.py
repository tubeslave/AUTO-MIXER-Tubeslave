"""Read-only consumer for sequence-policy promotion acceptance ledgers.

This verifier connects a queue-import promotion-contract result to the existing
promotion execution-plan builder. It verifies explicit operator accept/reject
records against contract provenance and emits only an execution-plan request
template. It does not build execution plans, execute proposals, promote policy,
mutate queues, policy stores, runtime memory, models, append-only memory,
proposal state, idempotency, or mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_STATUSES,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_contract import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_CONTRACT_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_CONTRACT_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_STATUSES,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_contract import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_CONTRACT_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_CONTRACT_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ACCEPTANCE_LEDGER_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_execution_plan_request_available",
}
REQUIRED_PROMOTION_CONTRACT_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer"
)
REQUIRED_PROMOTION_CONTRACT_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_request_available"
)
REQUIRED_PROMOTION_ACCEPTANCE_LEDGER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger"
)
REQUIRED_PROMOTION_ACCEPTANCE_LEDGER_AVAILABLE_STATUS = (
    "operator_sequence_memory_trust_promotion_acceptance_records_available"
)
REQUIRED_CONSUMER_FALSE_FLAGS = (
    "promotion_acceptance_ledger_handler_invoked",
    "promotion_acceptance_ledger_created",
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
REQUIRED_LEDGER_FALSE_SAFETY_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "execution_allowed_by_acceptance",
    "sequence_policy_promotion_allowed_by_acceptance",
    "queue_mutation_allowed_by_acceptance",
    "policy_mutation_allowed_by_acceptance",
    "model_mutation_allowed_by_acceptance",
    "runtime_memory_mutation_allowed_by_acceptance",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_LEDGER_FALSE_SUMMARY_FLAGS = (
    "execution_allowed_by_acceptance",
    "sequence_policy_promotion_allowed_by_acceptance",
    "queue_mutation_allowed_by_acceptance",
    "policy_mutation_allowed_by_acceptance",
    "model_mutation_allowed_by_acceptance",
    "runtime_memory_mutation_allowed_by_acceptance",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_RECORD_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "policy_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "execution_allowed_by_acceptance",
    "sequence_policy_promotion_allowed_by_acceptance",
    "queue_mutation_allowed_by_acceptance",
    "policy_mutation_allowed_by_acceptance",
    "model_mutation_allowed_by_acceptance",
    "runtime_memory_mutation_allowed_by_acceptance",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
)
REQUIRED_RECORD_FIELDS = (
    "promotion_acceptance_id",
    "promotion_contract_record_id",
    "promotion_candidate_id",
    "proposal_id",
    "learning_scope",
    "value_type",
    "idempotency_key",
    "previous_policy_state",
    "rollback_strategy",
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
    explicit = _string(record.get("promotion_acceptance_id"))
    if explicit:
        return explicit
    contract_id = _string(record.get("promotion_contract_record_id"))
    return f"{contract_id if contract_id else 'unknown_sequence_promotion_acceptance'}:{index}"


def _templates_by_contract_record_id(
    templates: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(template["promotion_contract_record_id"]): template
        for template in templates
        if isinstance(template.get("promotion_contract_record_id"), str)
        and template.get("promotion_contract_record_id")
    }


def _reviewed_contract_record_ids(
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    return tuple(
        str(record["promotion_contract_record_id"])
        for record in records
        if record.get("valid") is True
        and isinstance(record.get("promotion_contract_record_id"), str)
        and record.get("promotion_contract_record_id")
    )


def _unreviewed_contract_record_ids(
    templates: Tuple[Mapping[str, Any], ...],
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    reviewed = set(_reviewed_contract_record_ids(records))
    return tuple(
        str(template["promotion_contract_record_id"])
        for template in templates
        if isinstance(template.get("promotion_contract_record_id"), str)
        and template.get("promotion_contract_record_id")
        and template.get("promotion_contract_record_id") not in reviewed
    )


def _proposal_ids_for_records(
    records: Tuple[Mapping[str, Any], ...],
    *,
    accepted: bool,
) -> Tuple[str, ...]:
    flag = (
        "operator_accepted_sequence_memory_trust_promotion_contract"
        if accepted
        else "operator_rejected_sequence_memory_trust_promotion_contract"
    )
    return tuple(
        str(record["proposal_id"])
        for record in records
        if record.get("valid") is True
        and record.get(flag) is True
        and isinstance(record.get("proposal_id"), str)
        and record.get("proposal_id")
    )


def _consumer_envelope_violations(consumer: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_PROMOTION_CONTRACT_RESULT_CONSUMER_TYPE:
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_contract_consumer_type_mismatch"
        )
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_CONTRACT_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_contract_consumer_method_id_mismatch"
        )
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_CONTRACT_RESULT_CONSUMER_STATUSES
    ):
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_contract_consumer_status_unknown"
        )
    if (
        consumer.get("status")
        != REQUIRED_PROMOTION_CONTRACT_RESULT_CONSUMER_AVAILABLE_STATUS
    ):
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_contract_consumer_status_not_available"
        )
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_contract_consumer_read_only_not_true"
        )
    if safety.get("promotion_contract_result_consumer_only") is not True:
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_contract_consumer_only_not_true"
        )
    if safety.get("promotion_acceptance_ledger_request_template_only") is not True:
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_contract_consumer_template_only_not_true"
        )
    if safety.get("operator_acceptance_required") is not True:
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_contract_consumer_operator_acceptance_not_true"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_acceptance_ledger_result_consumer_contract_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    templates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    expected_counts = {
        "expected_contract_record_count": len(templates),
        "contract_record_count": len(templates),
        "invalid_record_count": 0,
    }
    violations = []
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_promotion_acceptance_ledger_result_consumer_contract_consumer_summary_{key}_mismatch"
            )
    if summary.get("promotion_acceptance_ledger_request_available") is not True:
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_contract_consumer_request_not_available"
        )
    if summary.get("operator_acceptance_required") is not True:
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_contract_consumer_summary_operator_acceptance_not_true"
        )
    expected_proposal_ids = tuple(
        str(template["proposal_id"])
        for template in templates
        if isinstance(template.get("proposal_id"), str) and template.get("proposal_id")
    )
    if _string_list(summary.get("proposal_ids")) != expected_proposal_ids:
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_contract_consumer_proposal_ids_mismatch"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_acceptance_ledger_result_consumer_contract_consumer_summary_{flag}_not_false"
            )
    return tuple(violations)


def _template_violations(
    template: Mapping[str, Any],
    *,
    index: int,
) -> Tuple[str, ...]:
    template_id = (
        _string(template.get("promotion_acceptance_record_template_id"))
        or f"promotion_acceptance_template:{index}"
    )
    violations = []
    for field in (
        "promotion_contract_record_id",
        "promotion_candidate_id",
        "proposal_id",
        "learning_scope",
        "value_type",
        "idempotency_key",
        "previous_policy_state",
        "rollback_strategy",
    ):
        if not _string(template.get(field)):
            violations.append(f"{template_id}:{field}_missing")
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
    if (
        request.get("handler")
        != "record_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance"
    ):
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_request_handler_mismatch"
        )
    if request.get("read_only_template") is not True:
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_request_read_only_template_not_true"
        )
    if request.get("promotion_acceptance_ledger_request_template_only") is not True:
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_request_template_only_not_true"
        )
    if request.get("operator_must_supply_promotion_acceptance_records") is not True:
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_request_operator_records_not_true"
        )
    if request.get("acceptance_records") != []:
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_request_acceptance_records_not_empty"
        )
    request_templates = _record_list(request.get("promotion_acceptance_record_templates"))
    request_template_ids = tuple(
        _string(template.get("promotion_contract_record_id"))
        for template in request_templates
    )
    top_level_template_ids = tuple(
        _string(template.get("promotion_contract_record_id")) for template in templates
    )
    if request_template_ids != top_level_template_ids:
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_request_templates_mismatch"
        )
    contract = _mapping(request.get("promotion_contract"))
    if contract.get("status") != "manual_sequence_memory_trust_promotion_contract_ready":
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_request_contract_not_ready"
        )
    for flag in REQUIRED_REQUEST_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_acceptance_ledger_result_consumer_request_{flag}_not_false"
            )
    return tuple(violations)


def _ledger_envelope_violations(ledger: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if ledger.get("type") != REQUIRED_PROMOTION_ACCEPTANCE_LEDGER_TYPE:
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_ledger_type_mismatch"
        )
    generated_at = ledger.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_ledger_generated_at_invalid"
        )
    if (
        ledger.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_METHOD_ID
    ):
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_ledger_method_id_mismatch"
        )
    if (
        ledger.get("source_method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID
    ):
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_ledger_source_method_mismatch"
        )
    if (
        ledger.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ACCEPTANCE_LEDGER_STATUSES
    ):
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_ledger_status_unknown"
        )
    if (
        ledger.get("status")
        == "operator_sequence_memory_trust_promotion_acceptance_replacement_required"
    ):
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_ledger_status_replacement_required"
        )
    safety = _mapping(ledger.get("safety"))
    if safety.get("read_only") is not True:
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_ledger_read_only_not_true"
        )
    if safety.get("operator_required") is not True:
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_ledger_operator_required_not_true"
        )
    for flag in REQUIRED_LEDGER_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_acceptance_ledger_result_consumer_ledger_{flag}_not_false"
            )
    return tuple(violations)


def _ledger_summary_violations(
    summary: Mapping[str, Any],
    *,
    templates: Tuple[Mapping[str, Any], ...],
    records: Tuple[Mapping[str, Any], ...],
    invalid_record_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    expected_unreviewed = _unreviewed_contract_record_ids(templates, records)
    expected_counts = {
        "promotion_contract_record_count": len(templates),
        "acceptance_record_count": len(records),
        "accepted_record_count": len(_proposal_ids_for_records(records, accepted=True)),
        "rejected_record_count": len(_proposal_ids_for_records(records, accepted=False)),
        "invalid_record_count": len(invalid_record_ids),
        "unreviewed_contract_record_count": len(expected_unreviewed),
    }
    violations = []
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_promotion_acceptance_ledger_result_consumer_ledger_summary_{key}_mismatch"
            )
    if summary.get("operator_review_required") is not True:
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_ledger_summary_operator_review_not_true"
        )
    if (
        _string_list(summary.get("accepted_proposal_ids"))
        != _proposal_ids_for_records(records, accepted=True)
    ):
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_ledger_summary_accepted_proposal_ids_mismatch"
        )
    if (
        _string_list(summary.get("rejected_proposal_ids"))
        != _proposal_ids_for_records(records, accepted=False)
    ):
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_ledger_summary_rejected_proposal_ids_mismatch"
        )
    if _string_list(summary.get("unreviewed_contract_record_ids")) != expected_unreviewed:
        violations.append(
            "sequence_policy_promotion_acceptance_ledger_result_consumer_ledger_summary_unreviewed_contract_record_ids_mismatch"
        )
    for flag in REQUIRED_LEDGER_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_acceptance_ledger_result_consumer_ledger_summary_{flag}_not_false"
            )
    return tuple(violations)


def _record_violations(
    record: Mapping[str, Any],
    *,
    templates_by_contract_record_id: Mapping[str, Mapping[str, Any]],
    index: int,
) -> Tuple[str, ...]:
    record_id = _record_id(record, index)
    violations = []
    for field in REQUIRED_RECORD_FIELDS:
        if not _string(record.get(field)):
            violations.append(f"{record_id}:{field}_missing")
    template = _mapping(
        templates_by_contract_record_id.get(
            _string(record.get("promotion_contract_record_id"))
        )
    )
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
            "promotion_candidate_id": template.get("promotion_candidate_id"),
            "proposal_id": template.get("proposal_id"),
            "learning_scope": template.get("learning_scope"),
            "value_type": template.get("value_type"),
            "idempotency_key": template.get("idempotency_key"),
            "previous_policy_state": template.get("previous_policy_state"),
            "rollback_strategy": template.get("rollback_strategy"),
        }
        for field, expected in expected_pairs.items():
            if record.get(field) != expected:
                violations.append(f"{record_id}:{field}_mismatch")
    accepted = record.get("operator_accepted_sequence_memory_trust_promotion_contract")
    rejected = record.get("operator_rejected_sequence_memory_trust_promotion_contract")
    if record.get("decision") == "accepted" and accepted is not True:
        violations.append(f"{record_id}:accepted_flag_not_true")
    if record.get("decision") == "accepted" and rejected is not False:
        violations.append(f"{record_id}:rejected_flag_not_false")
    if record.get("decision") == "rejected" and rejected is not True:
        violations.append(f"{record_id}:rejected_flag_not_true")
    if record.get("decision") == "rejected" and accepted is not False:
        violations.append(f"{record_id}:accepted_flag_not_false")
    if record.get("read_only") is not True:
        violations.append(f"{record_id}:read_only_not_true")
    if record.get("operator_review_required") is not True:
        violations.append(f"{record_id}:operator_review_required_not_true")
    if record.get("unsafe_authority_flags") not in ([], ()):
        violations.append(f"{record_id}:unsafe_authority_flags_not_empty")
    for flag in REQUIRED_RECORD_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(f"{record_id}:{flag}_not_false")
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer(
    *,
    promotion_contract_result_consumer: Mapping[str, Any] | None = None,
    promotion_acceptance_ledger_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify promotion acceptance-ledger evidence and emit execution-plan request."""

    consumer = _mapping(promotion_contract_result_consumer)
    ledger = _mapping(promotion_acceptance_ledger_result)
    templates = _record_list(consumer.get("promotion_acceptance_record_templates"))
    request = _mapping(consumer.get("promotion_acceptance_ledger_request"))
    summary = _mapping(ledger.get("summary"))
    records = _record_list(ledger.get("records"))
    ledger_invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    template_violations = tuple(
        violation
        for index, template in enumerate(templates)
        for violation in _template_violations(template, index=index)
    )
    templates_by_contract_record_id = _templates_by_contract_record_id(templates)
    record_violations = tuple(
        violation
        for index, record in enumerate(records)
        for violation in _record_violations(
            record,
            templates_by_contract_record_id=templates_by_contract_record_id,
            index=index,
        )
    )
    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_consumer_envelope_violations(consumer),
                *_consumer_summary_violations(
                    _mapping(consumer.get("summary")),
                    templates=templates,
                ),
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
    unreviewed_contract_record_ids = _unreviewed_contract_record_ids(
        templates,
        records,
    )
    accepted_proposal_ids = _proposal_ids_for_records(records, accepted=True)
    rejected_proposal_ids = _proposal_ids_for_records(records, accepted=False)

    if not consumer or not ledger:
        status = (
            "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer_not_ready"
        )
        blocked_by = (
            "promotion_contract_result_consumer_and_promotion_acceptance_ledger_result_required",
        )
    elif invalid_record_ids:
        status = (
            "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer_replacement_required"
        )
        blocked_by = (
            "operator_must_replace_invalid_sequence_memory_trust_promotion_acceptance_ledger_evidence",
        )
    elif ledger.get("status") != REQUIRED_PROMOTION_ACCEPTANCE_LEDGER_AVAILABLE_STATUS:
        status = (
            "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer_blocked"
        )
        blocked_by = (
            "operator_must_supply_accept_or_reject_promotion_records_before_execution_plan",
        )
    elif not accepted_proposal_ids:
        status = (
            "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer_blocked"
        )
        blocked_by = (
            "accepted_sequence_memory_trust_promotion_record_required_before_execution_plan",
        )
    elif not records:
        status = (
            "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer_not_ready"
        )
        blocked_by = ("sequence_memory_trust_promotion_acceptance_records_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_execution_plan_request_available"
        blocked_by = (
            "operator_may_request_read_only_sequence_memory_trust_promotion_execution_plan",
        )

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_execution_plan_request_available"
    )
    execution_plan_request: Dict[str, Any] = {
        "handler": "plan_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution",
        "read_only_template": True,
        "promotion_acceptance_ledger_result_consumer_only": True,
        "promotion_execution_plan_request_template_only": True,
        "promotion_execution_plan_handler_invoked": False,
        "promotion_execution_plan_created": False,
        "accepted_promotion_record_required": True,
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
        execution_plan_request["promotion_acceptance_ledger"] = ledger

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ACCEPTANCE_LEDGER_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_CONTRACT_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_acceptance_ledger_result_consumer_only": True,
            "promotion_execution_plan_request_template_only": True,
            "requires_promotion_contract_result_consumer": True,
            "requires_promotion_acceptance_ledger_result": True,
            "promotion_execution_plan_handler_invoked": False,
            "promotion_execution_plan_created": False,
            "operator_acceptance_required": True,
            "accepted_promotion_record_required": True,
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
            "expected_promotion_contract_record_count": len(templates),
            "promotion_contract_record_count": _summary_int(
                summary.get("promotion_contract_record_count")
            )
            or 0,
            "acceptance_record_count": len(records),
            "accepted_record_count": len(accepted_proposal_ids),
            "rejected_record_count": len(rejected_proposal_ids),
            "invalid_record_count": len(invalid_record_ids),
            "unreviewed_contract_record_count": len(unreviewed_contract_record_ids),
            "promotion_acceptance_ledger_result_consumer_only": True,
            "promotion_execution_plan_request_template_only": True,
            "promotion_execution_plan_request_available": request_available,
            "promotion_execution_plan_handler_invoked": False,
            "promotion_execution_plan_created": False,
            "operator_acceptance_required": True,
            "accepted_promotion_record_required": True,
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
            "accepted_proposal_ids": list(accepted_proposal_ids),
            "rejected_proposal_ids": list(rejected_proposal_ids),
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion-contract result consumer and promotion acceptance-ledger result."
                if not consumer or not ledger
                else (
                    "Replace invalid sequence-memory trust promotion acceptance-ledger evidence before execution planning."
                    if invalid_record_ids
                    else (
                        "Operator must accept at least one promotion record before execution planning."
                        if not accepted_proposal_ids
                        else "Run the read-only promotion execution-plan request; this consumer does not create plans or execute policy changes."
                    )
                )
            ),
        },
        "promotion_execution_plan_request": execution_plan_request,
    }
