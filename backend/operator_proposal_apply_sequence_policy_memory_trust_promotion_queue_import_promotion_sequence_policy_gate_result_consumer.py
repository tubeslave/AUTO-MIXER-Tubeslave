"""Read-only consumer for sequence-policy gate results.

This verifier connects a queue-import promotion rollback-verification result
consumer to the existing sequence-policy consumer builder. It verifies
sequence-policy gate records against rollback-verification provenance and emits
only a policy-consumer request template. It does not build consumer records,
reorder proposals, import queues, reserve idempotency, mutate runtime or
append-only memory, mutate policy stores, or touch mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ROLLBACK_VERIFICATION_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ROLLBACK_VERIFICATION_RESULT_CONSUMER_STATUSES,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_SEQUENCE_POLICY_GATE_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_SEQUENCE_POLICY_GATE_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ROLLBACK_VERIFICATION_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ROLLBACK_VERIFICATION_RESULT_CONSUMER_STATUSES,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_SEQUENCE_POLICY_GATE_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_SEQUENCE_POLICY_GATE_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_POLICY_GATE_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_POLICY_GATE_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_policy_consumer_request_available",
}
REQUIRED_ROLLBACK_VERIFICATION_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer"
)
REQUIRED_ROLLBACK_VERIFICATION_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_request_available"
)
REQUIRED_SEQUENCE_POLICY_GATE_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate"
)
REQUIRED_SEQUENCE_POLICY_GATE_AVAILABLE_STATUS = (
    "verified_sequence_memory_trust_promotion_policy_records_available"
)
TARGET_POLICY_STORE = "operator_sequence_policy_trust_gate"
REQUIRED_CONSUMER_FALSE_FLAGS = (
    "promotion_sequence_policy_gate_handler_invoked",
    "promotion_sequence_policy_gate_created",
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
    "can_import_to_queue",
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
    "can_import_to_queue",
    "can_apply_live",
)
REQUIRED_GATE_FALSE_SAFETY_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_gate",
    "sequence_policy_change_allowed_by_gate",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_GATE_FALSE_SUMMARY_FLAGS = (
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_gate",
    "sequence_policy_change_allowed_by_gate",
    "live_write",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
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
)
REQUIRED_GATE_RECORD_FIELDS = (
    "sequence_policy_gate_record_id",
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
    "sequence_policy_gate_status",
)
REQUIRED_GATE_RECORD_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_gate",
    "sequence_policy_change_allowed_by_gate",
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


def _verified_record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("rollback_verification_record_id"))
    if explicit:
        return explicit
    apply_id = _string(record.get("promotion_apply_audit_id"))
    return f"{apply_id if apply_id else 'unknown_sequence_rollback_verification'}:{index}"


def _gate_record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("sequence_policy_gate_record_id"))
    if explicit:
        return explicit
    policy_key = _string(record.get("policy_key"))
    return f"{policy_key if policy_key else 'unknown_sequence_policy_gate_record'}:{index}"


def _target_policy_path(policy_key: str) -> str:
    return f"{TARGET_POLICY_STORE}/{policy_key}/trust_state"


def _verified_records_from_consumer_request(
    consumer: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    request = _mapping(consumer.get("promotion_sequence_policy_gate_request"))
    rollback = _mapping(request.get("rollback_verification"))
    return _record_list(rollback.get("verified_records"))


def _verified_records_by_id(
    records: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(record["rollback_verification_record_id"]): record
        for record in records
        if isinstance(record.get("rollback_verification_record_id"), str)
        and record.get("rollback_verification_record_id")
    }


def _consumer_envelope_violations(consumer: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_ROLLBACK_VERIFICATION_RESULT_CONSUMER_TYPE:
        violations.append(
            "sequence_policy_promotion_gate_result_consumer_rollback_result_consumer_type_mismatch"
        )
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ROLLBACK_VERIFICATION_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append(
            "sequence_policy_promotion_gate_result_consumer_rollback_result_consumer_method_id_mismatch"
        )
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ROLLBACK_VERIFICATION_RESULT_CONSUMER_STATUSES
    ):
        violations.append(
            "sequence_policy_promotion_gate_result_consumer_rollback_result_consumer_status_unknown"
        )
    if consumer.get("status") != REQUIRED_ROLLBACK_VERIFICATION_RESULT_CONSUMER_AVAILABLE_STATUS:
        violations.append(
            "sequence_policy_promotion_gate_result_consumer_rollback_result_consumer_status_not_available"
        )
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append(
            "sequence_policy_promotion_gate_result_consumer_rollback_result_consumer_read_only_not_true"
        )
    if safety.get("promotion_rollback_verification_result_consumer_only") is not True:
        violations.append(
            "sequence_policy_promotion_gate_result_consumer_rollback_result_consumer_only_not_true"
        )
    if safety.get("promotion_sequence_policy_gate_request_template_only") is not True:
        violations.append(
            "sequence_policy_promotion_gate_result_consumer_rollback_result_consumer_template_only_not_true"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_gate_result_consumer_rollback_result_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    verified_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "expected_rollback_verified_record_count": len(verified_records),
        "rollback_verified_record_count": len(verified_records),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_promotion_gate_result_consumer_rollback_result_consumer_summary_{key}_mismatch"
            )
    if summary.get("promotion_sequence_policy_gate_request_available") is not bool(verified_records):
        violations.append(
            "sequence_policy_promotion_gate_result_consumer_rollback_result_consumer_request_available_mismatch"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_gate_result_consumer_rollback_result_consumer_summary_{flag}_not_false"
            )
    expected_policy_keys = tuple(
        str(record["policy_key"])
        for record in verified_records
        if isinstance(record.get("policy_key"), str) and record.get("policy_key")
    )
    expected_paths = tuple(
        str(record["target_policy_path"])
        for record in verified_records
        if isinstance(record.get("target_policy_path"), str)
        and record.get("target_policy_path")
    )
    if _string_list(summary.get("verified_policy_keys")) != expected_policy_keys:
        violations.append(
            "sequence_policy_promotion_gate_result_consumer_rollback_result_consumer_summary_policy_keys_mismatch"
        )
    if _string_list(summary.get("target_policy_paths")) != expected_paths:
        violations.append(
            "sequence_policy_promotion_gate_result_consumer_rollback_result_consumer_summary_target_paths_mismatch"
        )
    return tuple(violations)


def _consumer_request_violations(request: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if (
        request.get("handler")
        != "gate_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy"
    ):
        violations.append("sequence_policy_promotion_gate_result_consumer_request_handler_mismatch")
    if request.get("read_only_template") is not True:
        violations.append("sequence_policy_promotion_gate_result_consumer_request_read_only_template_not_true")
    if request.get("promotion_sequence_policy_gate_request_template_only") is not True:
        violations.append("sequence_policy_promotion_gate_result_consumer_request_template_only_not_true")
    if request.get("promotion_sequence_policy_gate_handler_invoked") is not False:
        violations.append("sequence_policy_promotion_gate_result_consumer_request_handler_invoked_not_false")
    if request.get("promotion_sequence_policy_gate_created") is not False:
        violations.append("sequence_policy_promotion_gate_result_consumer_request_created_not_false")
    for flag in REQUIRED_REQUEST_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_gate_result_consumer_request_{flag}_not_false"
            )
    if not _record_list(_mapping(request.get("rollback_verification")).get("verified_records")):
        violations.append("sequence_policy_promotion_gate_result_consumer_request_verified_records_missing")
    return tuple(violations)


def _gate_envelope_violations(sequence_policy_gate: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if sequence_policy_gate.get("type") != REQUIRED_SEQUENCE_POLICY_GATE_TYPE:
        violations.append("sequence_policy_promotion_gate_result_consumer_gate_type_mismatch")
    generated_at = sequence_policy_gate.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("sequence_policy_promotion_gate_result_consumer_gate_generated_at_invalid")
    if (
        sequence_policy_gate.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_SEQUENCE_POLICY_GATE_METHOD_ID
    ):
        violations.append("sequence_policy_promotion_gate_result_consumer_gate_method_id_mismatch")
    if (
        sequence_policy_gate.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_SEQUENCE_POLICY_GATE_STATUSES
    ):
        violations.append("sequence_policy_promotion_gate_result_consumer_gate_status_unknown")
    safety = _mapping(sequence_policy_gate.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_policy_promotion_gate_result_consumer_gate_read_only_not_true")
    if safety.get("operator_required") is not True:
        violations.append("sequence_policy_promotion_gate_result_consumer_gate_operator_required_not_true")
    if safety.get("policy_gate_only") is not True:
        violations.append("sequence_policy_promotion_gate_result_consumer_gate_policy_only_not_true")
    for flag in REQUIRED_GATE_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_gate_result_consumer_gate_{flag}_not_false"
            )
    return tuple(violations)


def _gate_safety_state_violations(
    sequence_policy_gate: Mapping[str, Any],
    *,
    policy_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    safety = _mapping(sequence_policy_gate.get("safety"))
    eligible = any(record.get("sequence_policy_eligible") is True for record in policy_records)
    violations = []
    if safety.get("sequence_policy_eligible") is not eligible:
        violations.append("sequence_policy_promotion_gate_result_consumer_gate_eligible_mismatch")
    if safety.get("sequence_policy_consumer_required") is not eligible:
        violations.append("sequence_policy_promotion_gate_result_consumer_gate_consumer_required_mismatch")
    return tuple(violations)


def _gate_status_violations(
    sequence_policy_gate: Mapping[str, Any],
    *,
    policy_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = sequence_policy_gate.get("status")
    if status == "sequence_memory_trust_promotion_sequence_policy_gate_replacement_required":
        return ("sequence_policy_promotion_gate_result_consumer_gate_replacement_required",)
    if status == REQUIRED_SEQUENCE_POLICY_GATE_AVAILABLE_STATUS and not policy_records:
        return ("sequence_policy_promotion_gate_result_consumer_gate_available_without_records",)
    if status != REQUIRED_SEQUENCE_POLICY_GATE_AVAILABLE_STATUS and policy_records:
        return ("sequence_policy_promotion_gate_result_consumer_gate_records_without_available_status",)
    if status != REQUIRED_SEQUENCE_POLICY_GATE_AVAILABLE_STATUS:
        return ("sequence_policy_promotion_gate_result_consumer_policy_records_available_required",)
    return ()


def _gate_summary_violations(
    summary: Mapping[str, Any],
    *,
    verified_records: Tuple[Mapping[str, Any], ...],
    policy_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    eligible_records = tuple(
        record for record in policy_records if record.get("sequence_policy_eligible") is True
    )
    violations = []
    expected_counts = {
        "rollback_verified_record_count": len(verified_records),
        "sequence_policy_gate_record_count": len(policy_records),
        "eligible_record_count": len(eligible_records),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_promotion_gate_result_consumer_gate_summary_{key}_mismatch"
            )
    if summary.get("target_policy_store") != TARGET_POLICY_STORE:
        violations.append("sequence_policy_promotion_gate_result_consumer_gate_summary_target_store_mismatch")
    if summary.get("sequence_policy_eligible") is not bool(eligible_records):
        violations.append("sequence_policy_promotion_gate_result_consumer_gate_summary_eligibility_mismatch")
    if summary.get("policy_gate_only") is not True:
        violations.append("sequence_policy_promotion_gate_result_consumer_gate_summary_policy_only_not_true")
    for flag in REQUIRED_GATE_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_gate_result_consumer_gate_summary_{flag}_not_false"
            )
    expected_policy_keys = tuple(
        record["policy_key"]
        for record in eligible_records
        if _string(record.get("policy_key"))
    )
    if _string_list(summary.get("eligible_policy_keys")) != expected_policy_keys:
        violations.append(
            "sequence_policy_promotion_gate_result_consumer_gate_summary_policy_keys_mismatch"
        )
    return tuple(violations)


def _verified_record_violations(record: Mapping[str, Any], index: int) -> Tuple[str, ...]:
    record_id = _verified_record_id(record, index)
    policy_key = _string(record.get("policy_key"))
    violations = []
    for field in REQUIRED_VERIFIED_RECORD_FIELDS:
        if not _string(record.get(field)):
            violations.append(f"{record_id}:{field}_missing")
    if record.get("target_policy_store") != TARGET_POLICY_STORE:
        violations.append(f"{record_id}:target_policy_store_mismatch")
    if record.get("target_policy_path") != _target_policy_path(policy_key):
        violations.append(f"{record_id}:target_policy_path_mismatch")
    if record.get("rollback_verified") is not True:
        violations.append(f"{record_id}:rollback_verified_not_true")
    if record.get("read_only") is not True:
        violations.append(f"{record_id}:read_only_not_true")
    return tuple(violations)


def _gate_record_violations(
    record: Mapping[str, Any],
    *,
    verified_record: Mapping[str, Any],
    index: int,
) -> Tuple[str, ...]:
    record_id = _gate_record_id(record, index)
    policy_key = _string(record.get("policy_key"))
    violations = []
    for field in REQUIRED_GATE_RECORD_FIELDS:
        if not _string(record.get(field)):
            violations.append(f"{record_id}:{field}_missing")
    if not verified_record:
        violations.append(f"{record_id}:verified_record_missing")
    else:
        for field in REQUIRED_VERIFIED_RECORD_FIELDS:
            if field in record and record.get(field) != verified_record.get(field):
                violations.append(f"{record_id}:{field}_mismatch")
    if record.get("target_policy_store") != TARGET_POLICY_STORE:
        violations.append(f"{record_id}:target_policy_store_mismatch")
    if record.get("target_policy_path") != _target_policy_path(policy_key):
        violations.append(f"{record_id}:target_policy_path_mismatch")
    if record.get("rollback_verified") is not True:
        violations.append(f"{record_id}:rollback_verified_not_true")
    if record.get("sequence_policy_gate_status") != "trusted":
        violations.append(f"{record_id}:sequence_policy_gate_status_not_trusted")
    if record.get("sequence_policy_eligible") is not True:
        violations.append(f"{record_id}:sequence_policy_eligible_not_true")
    if record.get("eligible_for_sequence_policy_consumer") is not True:
        violations.append(f"{record_id}:eligible_for_sequence_policy_consumer_not_true")
    if record.get("policy_gate_only") is not True:
        violations.append(f"{record_id}:policy_gate_only_not_true")
    if record.get("sequence_policy_consumer_required") is not True:
        violations.append(f"{record_id}:sequence_policy_consumer_required_not_true")
    if record.get("read_only") is not True:
        violations.append(f"{record_id}:read_only_not_true")
    for flag in REQUIRED_GATE_RECORD_FALSE_FLAGS:
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


def _gate_identity_violations(
    *,
    verified_records: Tuple[Mapping[str, Any], ...],
    policy_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    verified_by_id = _verified_records_by_id(verified_records)
    gate_by_verified_id = {
        str(record["rollback_verification_record_id"]): record
        for record in policy_records
        if isinstance(record.get("rollback_verification_record_id"), str)
        and record.get("rollback_verification_record_id")
    }
    violations = []
    if set(gate_by_verified_id) != set(verified_by_id):
        violations.append("sequence_policy_promotion_gate_result_consumer_verified_record_ids_mismatch")
    if len(gate_by_verified_id) != len(policy_records):
        violations.append("sequence_policy_promotion_gate_result_consumer_duplicate_verified_record_ids")
    for index, verified_record in enumerate(verified_records):
        verified_id = _string(verified_record.get("rollback_verification_record_id"))
        violations.extend(_verified_record_violations(verified_record, index))
        gate_record = _mapping(gate_by_verified_id.get(verified_id))
        if not gate_record:
            violations.append(f"{verified_id}:sequence_policy_gate_record_missing")
            continue
        violations.extend(
            _gate_record_violations(
                gate_record,
                verified_record=verified_record,
                index=index,
            )
        )
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer(
    *,
    promotion_rollback_verification_result_consumer: Mapping[str, Any] | None = None,
    promotion_sequence_policy_gate_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify sequence-policy gate evidence and emit a policy-consumer request."""

    consumer = _mapping(promotion_rollback_verification_result_consumer)
    gate = _mapping(promotion_sequence_policy_gate_result)
    request = _mapping(consumer.get("promotion_sequence_policy_gate_request"))
    verified_records = _verified_records_from_consumer_request(consumer)
    policy_records = _record_list(gate.get("policy_records"))
    gate_status_available = gate.get("status") == REQUIRED_SEQUENCE_POLICY_GATE_AVAILABLE_STATUS
    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_consumer_envelope_violations(consumer),
                *_consumer_summary_violations(
                    _mapping(consumer.get("summary")),
                    verified_records=verified_records,
                ),
                *_consumer_request_violations(request),
                *_gate_envelope_violations(gate),
                *_gate_safety_state_violations(
                    gate,
                    policy_records=policy_records,
                ),
                *_gate_status_violations(
                    gate,
                    policy_records=policy_records,
                ),
                *(
                    (
                        *_gate_summary_violations(
                            _mapping(gate.get("summary")),
                            verified_records=verified_records,
                            policy_records=policy_records,
                        ),
                        *_duplicate_ids(
                            policy_records,
                            "sequence_policy_gate_record_id",
                            id_func=_gate_record_id,
                        ),
                        *_duplicate_ids(
                            policy_records,
                            "rollback_verification_record_id",
                            id_func=_gate_record_id,
                        ),
                        *_duplicate_ids(
                            policy_records,
                            "policy_key",
                            id_func=_gate_record_id,
                        ),
                        *_gate_identity_violations(
                            verified_records=verified_records,
                            policy_records=policy_records,
                        ),
                    )
                    if gate_status_available or policy_records
                    else ()
                ),
            )
        )
    )
    if not consumer or not gate:
        invalid_record_ids: Tuple[str, ...] = ()
    else:
        blocking_only = (
            "sequence_policy_promotion_gate_result_consumer_policy_records_available_required",
        )
        invalid_record_ids = tuple(
            item for item in raw_invalid_record_ids if item not in blocking_only
        )

    if not consumer or not gate:
        status = "sequence_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer_not_ready"
        blocked_by = ("promotion_rollback_consumer_and_sequence_policy_gate_required",)
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_memory_trust_promotion_sequence_policy_gate_evidence",
        )
    elif gate.get("status") != REQUIRED_SEQUENCE_POLICY_GATE_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer_blocked"
        blocked_by = (
            "verified_sequence_memory_trust_promotion_policy_records_required_before_policy_consumer",
        )
    elif not policy_records:
        status = "sequence_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer_not_ready"
        blocked_by = ("sequence_memory_trust_promotion_policy_records_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_policy_consumer_request_available"
        blocked_by = (
            "operator_may_run_read_only_sequence_memory_trust_promotion_policy_consumer",
        )

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_policy_consumer_request_available"
    )
    policy_consumer_request: Dict[str, Any] = {
        "handler": (
            "consume_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy"
        ),
        "read_only_template": True,
        "promotion_sequence_policy_gate_result_consumer_only": True,
        "promotion_policy_consumer_request_template_only": True,
        "promotion_policy_consumer_handler_invoked": False,
        "promotion_policy_consumer_created": False,
        "proposals_required_from_operator": True,
        "memory_write_allowed_by_consumer": False,
        "execution_allowed_by_consumer": False,
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
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "can_import_to_queue": False,
        "can_apply_live": False,
    }
    if request_available:
        policy_consumer_request["sequence_policy_gate"] = gate

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_SEQUENCE_POLICY_GATE_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ROLLBACK_VERIFICATION_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_sequence_policy_gate_result_consumer_only": True,
            "promotion_policy_consumer_request_template_only": True,
            "requires_promotion_rollback_verification_result_consumer": True,
            "requires_promotion_sequence_policy_gate_result": True,
            "requires_policy_records": True,
            "promotion_policy_consumer_handler_invoked": False,
            "promotion_policy_consumer_created": False,
            "memory_write_allowed_by_consumer": False,
            "execution_allowed_by_consumer": False,
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
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
        },
        "summary": {
            "expected_sequence_policy_gate_record_count": len(verified_records),
            "sequence_policy_gate_record_count": len(policy_records),
            "invalid_record_count": len(invalid_record_ids),
            "target_policy_store": TARGET_POLICY_STORE,
            "promotion_sequence_policy_gate_result_consumer_only": True,
            "promotion_policy_consumer_request_template_only": True,
            "promotion_policy_consumer_request_available": request_available,
            "promotion_policy_consumer_handler_invoked": False,
            "promotion_policy_consumer_created": False,
            "requires_policy_records": True,
            "memory_write_allowed_by_consumer": False,
            "execution_allowed_by_consumer": False,
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
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
            "eligible_policy_keys": [
                str(record["policy_key"])
                for record in policy_records
                if isinstance(record.get("policy_key"), str) and record.get("policy_key")
            ],
            "target_policy_paths": [
                str(record["target_policy_path"])
                for record in policy_records
                if isinstance(record.get("target_policy_path"), str)
                and record.get("target_policy_path")
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion rollback-verification result consumer and sequence-policy gate result."
                if not consumer or not gate
                else (
                    "Replace invalid sequence-memory trust promotion sequence-policy gate evidence before policy consumer."
                    if invalid_record_ids
                    else (
                        "Run the read-only sequence-policy gate before policy consumer."
                        if gate.get("status") != REQUIRED_SEQUENCE_POLICY_GATE_AVAILABLE_STATUS
                        else "Run policy consumer with matching preview proposals; this consumer does not create consumer records or mutate order, queue, policy, or live state."
                    )
                )
            ),
        },
        "promotion_policy_consumer_request": policy_consumer_request,
    }
