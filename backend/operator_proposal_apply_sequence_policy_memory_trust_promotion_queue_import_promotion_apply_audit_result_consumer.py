"""Read-only consumer for sequence-policy promotion apply-audit results.

This verifier connects a queue-import promotion executor-result consumer to the
existing rollback-verification builder. It verifies apply-audit records against
the applied executor provenance and emits only a rollback-verification request
template. It does not invoke rollback verification, reserve idempotency, mutate
queues, policy stores, runtime memory, models, append-only memory, proposal
state, or mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_STATUSES,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_STATUSES,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_APPLY_AUDIT_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_APPLY_AUDIT_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_request_available",
}
REQUIRED_EXECUTOR_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer"
)
REQUIRED_EXECUTOR_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_request_available"
)
REQUIRED_APPLY_AUDIT_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit"
)
REQUIRED_APPLY_AUDIT_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_apply_audit_records_available"
)
TARGET_POLICY_STORE = "operator_sequence_policy_trust_gate"
REQUIRED_CONSUMER_FALSE_FLAGS = (
    "promotion_apply_audit_handler_invoked",
    "promotion_apply_audit_created",
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
REQUIRED_APPLY_AUDIT_FALSE_SAFETY_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_audit",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_APPLY_AUDIT_FALSE_SUMMARY_FLAGS = (
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_audit",
    "live_write",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_APPLY_AUDIT_RECORD_FIELDS = (
    "promotion_apply_audit_id",
    "promotion_executor_apply_id",
    "promotion_executor_dry_run_id",
    "promotion_execution_candidate_id",
    "promotion_acceptance_id",
    "promotion_contract_record_id",
    "promotion_candidate_id",
    "proposal_id",
    "learning_scope",
    "value_type",
    "source_idempotency_key",
    "policy_key",
    "target_policy_store",
    "target_policy_path",
    "operation",
    "previous_policy_state",
    "target_policy_state",
    "rollback_strategy",
    "rollback_target_policy_state",
    "idempotency_key",
)
REQUIRED_APPLY_RESULT_FIELDS = (
    "promotion_executor_apply_id",
    "promotion_executor_dry_run_id",
    "promotion_execution_candidate_id",
    "promotion_acceptance_id",
    "promotion_contract_record_id",
    "promotion_candidate_id",
    "proposal_id",
    "learning_scope",
    "value_type",
    "source_idempotency_key",
    "policy_key",
    "target_policy_store",
    "target_policy_path",
    "operation",
    "previous_policy_state",
    "target_policy_state",
    "rollback_strategy",
    "idempotency_key",
)
REQUIRED_APPLY_AUDIT_RECORD_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_audit",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
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


def _apply_result_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("promotion_executor_apply_id"))
    if explicit:
        return explicit
    dry_run_id = _string(record.get("promotion_executor_dry_run_id"))
    return f"{dry_run_id if dry_run_id else 'unknown_sequence_promotion_apply'}:{index}"


def _audit_record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("promotion_apply_audit_id"))
    if explicit:
        return explicit
    apply_id = _string(record.get("promotion_executor_apply_id"))
    return f"{apply_id if apply_id else 'unknown_sequence_apply_audit'}:{index}"


def _target_policy_path(policy_key: str) -> str:
    return f"{TARGET_POLICY_STORE}/{policy_key}/trust_state"


def _apply_results_from_consumer_request(
    consumer: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    request = _mapping(consumer.get("promotion_apply_audit_request"))
    executor = _mapping(request.get("promotion_executor"))
    return _record_list(executor.get("apply_results"))


def _apply_results_by_id(
    records: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(record["promotion_executor_apply_id"]): record
        for record in records
        if isinstance(record.get("promotion_executor_apply_id"), str)
        and record.get("promotion_executor_apply_id")
    }


def _consumer_envelope_violations(consumer: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_EXECUTOR_RESULT_CONSUMER_TYPE:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_type_mismatch"
        )
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_method_id_mismatch"
        )
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_RESULT_CONSUMER_STATUSES
    ):
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_status_unknown"
        )
    if consumer.get("status") != REQUIRED_EXECUTOR_RESULT_CONSUMER_AVAILABLE_STATUS:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_status_not_available"
        )
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_read_only_not_true"
        )
    if safety.get("promotion_executor_result_consumer_only") is not True:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_only_not_true"
        )
    if safety.get("promotion_apply_audit_request_template_only") is not True:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_template_only_not_true"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    apply_results: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "expected_executor_apply_result_count": len(apply_results),
        "executor_apply_result_count": len(apply_results),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_summary_{key}_mismatch"
            )
    if summary.get("promotion_apply_audit_request_available") is not bool(apply_results):
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_request_available_mismatch"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_summary_{flag}_not_false"
            )
    expected_paths = tuple(
        str(record["target_policy_path"])
        for record in apply_results
        if isinstance(record.get("target_policy_path"), str)
        and record.get("target_policy_path")
    )
    expected_keys = tuple(
        str(record["idempotency_key"])
        for record in apply_results
        if isinstance(record.get("idempotency_key"), str)
        and record.get("idempotency_key")
    )
    expected_policy_keys = tuple(
        str(record["policy_key"])
        for record in apply_results
        if isinstance(record.get("policy_key"), str) and record.get("policy_key")
    )
    if _string_list(summary.get("target_policy_paths")) != expected_paths:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_summary_target_paths_mismatch"
        )
    if _string_list(summary.get("idempotency_keys")) != expected_keys:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_summary_idempotency_keys_mismatch"
        )
    if _string_list(summary.get("updated_policy_keys")) != expected_policy_keys:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_executor_result_consumer_summary_policy_keys_mismatch"
        )
    return tuple(violations)


def _consumer_request_violations(request: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if (
        request.get("handler")
        != "audit_operator_proposal_apply_sequence_policy_memory_trust_promotion"
    ):
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_request_handler_mismatch"
        )
    if request.get("read_only_template") is not True:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_request_read_only_template_not_true"
        )
    if request.get("promotion_apply_audit_request_template_only") is not True:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_request_template_only_not_true"
        )
    if request.get("promotion_apply_audit_handler_invoked") is not False:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_request_handler_invoked_not_false"
        )
    if request.get("promotion_apply_audit_created") is not False:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_request_created_not_false"
        )
    for flag in REQUIRED_REQUEST_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_apply_audit_result_consumer_request_{flag}_not_false"
            )
    if not _record_list(_mapping(request.get("promotion_executor")).get("apply_results")):
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_request_apply_results_missing"
        )
    return tuple(violations)


def _apply_audit_envelope_violations(apply_audit: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if apply_audit.get("type") != REQUIRED_APPLY_AUDIT_TYPE:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_audit_type_mismatch"
        )
    generated_at = apply_audit.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_audit_generated_at_invalid"
        )
    if (
        apply_audit.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID
    ):
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_audit_method_id_mismatch"
        )
    if (
        apply_audit.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_STATUSES
    ):
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_audit_status_unknown"
        )
    safety = _mapping(apply_audit.get("safety"))
    if safety.get("read_only") is not True:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_audit_read_only_not_true"
        )
    if safety.get("operator_required") is not True:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_audit_operator_required_not_true"
        )
    if safety.get("rollback_verified") is not False:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_audit_rollback_verified_not_false"
        )
    for flag in REQUIRED_APPLY_AUDIT_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_apply_audit_result_consumer_audit_{flag}_not_false"
            )
    return tuple(violations)


def _apply_audit_status_violations(
    apply_audit: Mapping[str, Any],
    *,
    audit_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = apply_audit.get("status")
    if status == "sequence_memory_trust_promotion_apply_audit_replacement_required":
        return (
            "sequence_policy_promotion_apply_audit_result_consumer_audit_replacement_required",
        )
    if status == REQUIRED_APPLY_AUDIT_AVAILABLE_STATUS and not audit_records:
        return (
            "sequence_policy_promotion_apply_audit_result_consumer_audit_available_without_records",
        )
    if status != REQUIRED_APPLY_AUDIT_AVAILABLE_STATUS and audit_records:
        return (
            "sequence_policy_promotion_apply_audit_result_consumer_audit_records_without_available_status",
        )
    if status != REQUIRED_APPLY_AUDIT_AVAILABLE_STATUS:
        return (
            "sequence_policy_promotion_apply_audit_result_consumer_audit_records_available_required",
        )
    return ()


def _apply_audit_summary_violations(
    summary: Mapping[str, Any],
    *,
    apply_results: Tuple[Mapping[str, Any], ...],
    audit_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "executor_apply_result_count": len(apply_results),
        "apply_audit_record_count": len(audit_records),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_promotion_apply_audit_result_consumer_audit_summary_{key}_mismatch"
            )
    if summary.get("target_policy_store") != TARGET_POLICY_STORE:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_audit_summary_target_store_mismatch"
        )
    if summary.get("rollback_evidence_required") is not bool(audit_records):
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_audit_summary_rollback_required_mismatch"
        )
    if summary.get("rollback_verified") is not False:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_audit_summary_rollback_verified_not_false"
        )
    for flag in REQUIRED_APPLY_AUDIT_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_apply_audit_result_consumer_audit_summary_{flag}_not_false"
            )
    expected_policy_keys = tuple(
        record["policy_key"] for record in audit_records if _string(record.get("policy_key"))
    )
    expected_rollback_states = tuple(
        record["rollback_target_policy_state"]
        for record in audit_records
        if _string(record.get("rollback_target_policy_state"))
    )
    expected_paths = tuple(
        record["target_policy_path"]
        for record in audit_records
        if _string(record.get("target_policy_path"))
    )
    expected_keys = tuple(
        record["idempotency_key"]
        for record in audit_records
        if _string(record.get("idempotency_key"))
    )
    if _string_list(summary.get("audited_policy_keys")) != expected_policy_keys:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_audit_summary_policy_keys_mismatch"
        )
    if _string_list(summary.get("rollback_target_policy_states")) != expected_rollback_states:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_audit_summary_rollback_states_mismatch"
        )
    if _string_list(summary.get("target_policy_paths")) != expected_paths:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_audit_summary_target_paths_mismatch"
        )
    if _string_list(summary.get("idempotency_keys")) != expected_keys:
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_audit_summary_idempotency_keys_mismatch"
        )
    return tuple(violations)


def _apply_audit_record_violations(record: Mapping[str, Any], index: int) -> Tuple[str, ...]:
    record_id = _audit_record_id(record, index)
    policy_key = _string(record.get("policy_key"))
    violations = []
    for field in REQUIRED_APPLY_AUDIT_RECORD_FIELDS:
        if not _string(record.get(field)):
            violations.append(f"{record_id}:{field}_missing")
    if record.get("target_policy_store") != TARGET_POLICY_STORE:
        violations.append(f"{record_id}:target_policy_store_mismatch")
    if record.get("target_policy_path") != _target_policy_path(policy_key):
        violations.append(f"{record_id}:target_policy_path_mismatch")
    if record.get("operation") != "set_operator_sequence_policy_trust_state":
        violations.append(f"{record_id}:operation_mismatch")
    if record.get("applied") is not True:
        violations.append(f"{record_id}:applied_not_true")
    if record.get("rollback_evidence_required") is not True:
        violations.append(f"{record_id}:rollback_evidence_required_not_true")
    if record.get("rollback_verified") is not False:
        violations.append(f"{record_id}:rollback_verified_not_false")
    if record.get("audits_applied_sequence_policy_trust_store_mutation") is not True:
        violations.append(f"{record_id}:policy_store_audit_not_true")
    if record.get("read_only") is not True:
        violations.append(f"{record_id}:read_only_not_true")
    for flag in REQUIRED_APPLY_AUDIT_RECORD_FALSE_FLAGS:
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


def _apply_audit_identity_violations(
    *,
    apply_results: Tuple[Mapping[str, Any], ...],
    audit_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    apply_by_id = _apply_results_by_id(apply_results)
    audit_by_apply_id = {
        str(record["promotion_executor_apply_id"]): record
        for record in audit_records
        if isinstance(record.get("promotion_executor_apply_id"), str)
        and record.get("promotion_executor_apply_id")
    }
    violations = []
    if set(audit_by_apply_id) != set(apply_by_id):
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_apply_ids_mismatch"
        )
    if len(audit_by_apply_id) != len(audit_records):
        violations.append(
            "sequence_policy_promotion_apply_audit_result_consumer_duplicate_executor_apply_ids"
        )
    for index, apply_result in enumerate(apply_results):
        apply_id = _string(apply_result.get("promotion_executor_apply_id"))
        audit_record = _mapping(audit_by_apply_id.get(apply_id))
        if not audit_record:
            violations.append(f"{apply_id}:apply_audit_record_missing")
            continue
        for field in REQUIRED_APPLY_RESULT_FIELDS:
            if audit_record.get(field) != apply_result.get(field):
                violations.append(f"{apply_id}:{field}_mismatch")
        if (
            audit_record.get("rollback_target_policy_state")
            != apply_result.get("previous_policy_state")
        ):
            violations.append(f"{apply_id}:rollback_target_policy_state_mismatch")
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer(
    *,
    promotion_executor_result_consumer: Mapping[str, Any] | None = None,
    promotion_apply_audit_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify promotion apply-audit evidence and emit a rollback-verification request."""

    consumer = _mapping(promotion_executor_result_consumer)
    apply_audit = _mapping(promotion_apply_audit_result)
    request = _mapping(consumer.get("promotion_apply_audit_request"))
    apply_results = _apply_results_from_consumer_request(consumer)
    audit_records = _record_list(apply_audit.get("audit_records"))
    audit_status_available = apply_audit.get("status") == REQUIRED_APPLY_AUDIT_AVAILABLE_STATUS
    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_consumer_envelope_violations(consumer),
                *_consumer_summary_violations(
                    _mapping(consumer.get("summary")),
                    apply_results=apply_results,
                ),
                *_consumer_request_violations(request),
                *_apply_audit_envelope_violations(apply_audit),
                *_apply_audit_status_violations(
                    apply_audit,
                    audit_records=audit_records,
                ),
                *(
                    (
                        *_apply_audit_summary_violations(
                            _mapping(apply_audit.get("summary")),
                            apply_results=apply_results,
                            audit_records=audit_records,
                        ),
                        *(
                            violation
                            for index, record in enumerate(audit_records)
                            for violation in _apply_audit_record_violations(record, index)
                        ),
                        *_duplicate_ids(
                            audit_records,
                            "promotion_apply_audit_id",
                            id_func=_audit_record_id,
                        ),
                        *_duplicate_ids(
                            audit_records,
                            "promotion_executor_apply_id",
                            id_func=_audit_record_id,
                        ),
                        *_duplicate_ids(
                            audit_records,
                            "policy_key",
                            id_func=_audit_record_id,
                        ),
                        *_duplicate_ids(
                            audit_records,
                            "idempotency_key",
                            id_func=_audit_record_id,
                        ),
                        *_apply_audit_identity_violations(
                            apply_results=apply_results,
                            audit_records=audit_records,
                        ),
                    )
                    if audit_status_available or audit_records
                    else ()
                ),
            )
        )
    )
    if not consumer or not apply_audit:
        invalid_record_ids: Tuple[str, ...] = ()
    else:
        blocking_only = (
            "sequence_policy_promotion_apply_audit_result_consumer_audit_records_available_required",
        )
        invalid_record_ids = tuple(
            item for item in raw_invalid_record_ids if item not in blocking_only
        )

    if not consumer or not apply_audit:
        status = "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer_not_ready"
        blocked_by = ("promotion_executor_result_consumer_and_apply_audit_required",)
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_memory_trust_promotion_apply_audit_evidence",
        )
    elif apply_audit.get("status") != REQUIRED_APPLY_AUDIT_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer_blocked"
        blocked_by = (
            "sequence_memory_trust_promotion_apply_audit_records_required_before_rollback_verification",
        )
    elif not audit_records:
        status = "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer_not_ready"
        blocked_by = ("sequence_memory_trust_promotion_apply_audit_records_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_request_available"
        blocked_by = (
            "operator_may_run_read_only_sequence_memory_trust_promotion_rollback_verification",
        )

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_request_available"
    )
    rollback_verification_request: Dict[str, Any] = {
        "handler": (
            "verify_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback"
        ),
        "read_only_template": True,
        "promotion_apply_audit_result_consumer_only": True,
        "promotion_rollback_verification_request_template_only": True,
        "promotion_rollback_verification_handler_invoked": False,
        "promotion_rollback_verification_created": False,
        "verification_records_required_from_operator": True,
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
        rollback_verification_request["apply_audit"] = apply_audit

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_APPLY_AUDIT_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_apply_audit_result_consumer_only": True,
            "promotion_rollback_verification_request_template_only": True,
            "requires_promotion_executor_result_consumer": True,
            "requires_promotion_apply_audit_result": True,
            "requires_apply_audit_records": True,
            "promotion_rollback_verification_handler_invoked": False,
            "promotion_rollback_verification_created": False,
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
            "expected_apply_audit_record_count": len(apply_results),
            "apply_audit_record_count": len(audit_records),
            "invalid_record_count": len(invalid_record_ids),
            "target_policy_store": TARGET_POLICY_STORE,
            "promotion_apply_audit_result_consumer_only": True,
            "promotion_rollback_verification_request_template_only": True,
            "promotion_rollback_verification_request_available": request_available,
            "promotion_rollback_verification_handler_invoked": False,
            "promotion_rollback_verification_created": False,
            "requires_apply_audit_records": True,
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
            "audited_policy_keys": [
                str(record["policy_key"])
                for record in audit_records
                if isinstance(record.get("policy_key"), str) and record.get("policy_key")
            ],
            "target_policy_paths": [
                str(record["target_policy_path"])
                for record in audit_records
                if isinstance(record.get("target_policy_path"), str)
                and record.get("target_policy_path")
            ],
            "idempotency_keys": [
                str(record["idempotency_key"])
                for record in audit_records
                if isinstance(record.get("idempotency_key"), str)
                and record.get("idempotency_key")
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion executor-result consumer and promotion apply-audit result."
                if not consumer or not apply_audit
                else (
                    "Replace invalid sequence-memory trust promotion apply-audit evidence before rollback verification."
                    if invalid_record_ids
                    else (
                        "Run the read-only apply audit before rollback verification."
                        if apply_audit.get("status") != REQUIRED_APPLY_AUDIT_AVAILABLE_STATUS
                        else "Run rollback verification with operator-supplied verification records; this consumer does not verify rollback or mutate policy."
                    )
                )
            ),
        },
        "promotion_rollback_verification_request": rollback_verification_request,
    }
