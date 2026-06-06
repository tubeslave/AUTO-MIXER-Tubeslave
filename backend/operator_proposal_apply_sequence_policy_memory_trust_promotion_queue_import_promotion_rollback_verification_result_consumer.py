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


def _audit_records_by_id(
    records: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(record["promotion_apply_audit_id"]): record
        for record in records
        if isinstance(record.get("promotion_apply_audit_id"), str)
        and record.get("promotion_apply_audit_id")
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
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "expected_apply_audit_record_count": len(audit_records),
        "apply_audit_record_count": len(audit_records),
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
    return tuple(violations)


def _consumer_request_violations(request: Mapping[str, Any]) -> Tuple[str, ...]:
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
                ),
                *_consumer_request_violations(request),
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
            "invalid_record_count": len(invalid_record_ids),
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
