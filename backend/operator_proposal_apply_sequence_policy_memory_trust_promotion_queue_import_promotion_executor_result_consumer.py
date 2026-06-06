"""Read-only consumer for sequence-policy promotion executor results.

This verifier connects a queue-import promotion executor dry-run result consumer
to the existing promotion apply-audit builder. It verifies executor output
against dry-run provenance and emits only an apply-audit request template. It
does not invoke audit handlers, apply sequence policy, mutate queues, policy
stores, runtime memory, models, append-only memory, proposal state, idempotency,
or mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_executor import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_STATUSES,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_DRY_RUN_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_DRY_RUN_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_executor import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_STATUSES,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_DRY_RUN_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_DRY_RUN_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_executor_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_executor_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_executor_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_request_available",
}
REQUIRED_EXECUTOR_DRY_RUN_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer"
)
REQUIRED_EXECUTOR_DRY_RUN_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_executor_request_available"
)
REQUIRED_EXECUTOR_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor"
)
REQUIRED_EXECUTOR_APPLIED_STATUS = "sequence_memory_trust_promotion_executor_applied"
TARGET_POLICY_STORE = "operator_sequence_policy_trust_gate"
REQUIRED_CONSUMER_FALSE_FLAGS = (
    "promotion_executor_handler_invoked",
    "promotion_executor_created",
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
REQUIRED_EXECUTOR_FALSE_SAFETY_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_executor",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_EXECUTOR_FALSE_SUMMARY_FLAGS = (
    "queue_mutation",
    "model_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_executor",
    "live_write",
    "live_authority_granted",
    "full_autonomous_ready",
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
REQUIRED_APPLY_RESULT_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_executor",
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


def _policy_key(record: Mapping[str, Any]) -> str:
    proposal_id = _string(record.get("proposal_id")) or "unknown_proposal"
    source_key = _string(record.get("source_idempotency_key")) or "unknown_idempotency"
    return f"{proposal_id}/{source_key}"


def _target_policy_path(policy_key: str) -> str:
    return f"{TARGET_POLICY_STORE}/{policy_key}/trust_state"


def _dry_run_records_from_request(
    request: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    executor_dry_run = _mapping(request.get("executor_dry_run"))
    return _record_list(executor_dry_run.get("dry_run_records"))


def _dry_run_records_by_id(
    records: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(record["promotion_executor_dry_run_id"]): record
        for record in records
        if isinstance(record.get("promotion_executor_dry_run_id"), str)
        and record.get("promotion_executor_dry_run_id")
    }


def _consumer_envelope_violations(consumer: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_EXECUTOR_DRY_RUN_RESULT_CONSUMER_TYPE:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_dry_run_consumer_type_mismatch"
        )
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_DRY_RUN_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_dry_run_consumer_method_id_mismatch"
        )
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_DRY_RUN_RESULT_CONSUMER_STATUSES
    ):
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_dry_run_consumer_status_unknown"
        )
    if consumer.get("status") != REQUIRED_EXECUTOR_DRY_RUN_RESULT_CONSUMER_AVAILABLE_STATUS:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_dry_run_consumer_status_not_available"
        )
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_dry_run_consumer_read_only_not_true"
        )
    if safety.get("promotion_executor_dry_run_result_consumer_only") is not True:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_dry_run_consumer_only_not_true"
        )
    if safety.get("promotion_executor_request_template_only") is not True:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_dry_run_consumer_template_only_not_true"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_executor_result_consumer_dry_run_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    dry_run_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "expected_executor_dry_run_record_count": len(dry_run_records),
        "executor_dry_run_record_count": len(dry_run_records),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_promotion_executor_result_consumer_dry_run_consumer_summary_{key}_mismatch"
            )
    if summary.get("promotion_executor_request_available") is not bool(dry_run_records):
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_dry_run_consumer_request_available_mismatch"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_executor_result_consumer_dry_run_consumer_summary_{flag}_not_false"
            )
    expected_paths = tuple(
        str(record["target_policy_path"])
        for record in dry_run_records
        if isinstance(record.get("target_policy_path"), str)
        and record.get("target_policy_path")
    )
    expected_keys = tuple(
        str(record["idempotency_key"])
        for record in dry_run_records
        if isinstance(record.get("idempotency_key"), str)
        and record.get("idempotency_key")
    )
    if _string_list(summary.get("target_policy_paths")) != expected_paths:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_dry_run_consumer_summary_target_paths_mismatch"
        )
    if _string_list(summary.get("idempotency_keys")) != expected_keys:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_dry_run_consumer_summary_idempotency_keys_mismatch"
        )
    return tuple(violations)


def _consumer_request_violations(request: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if (
        request.get("handler")
        != "execute_operator_proposal_apply_sequence_policy_memory_trust_promotion"
    ):
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_request_handler_mismatch"
        )
    if request.get("read_only_template") is not True:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_request_read_only_template_not_true"
        )
    if request.get("promotion_executor_request_template_only") is not True:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_request_template_only_not_true"
        )
    if request.get("promotion_executor_handler_invoked") is not False:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_request_handler_invoked_not_false"
        )
    if request.get("promotion_executor_created") is not False:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_request_created_not_false"
        )
    if request.get("apply") is not False:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_request_apply_not_false"
        )
    if request.get("operator_ack") is not False:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_request_operator_ack_not_false"
        )
    if request.get("target_policy_store_required_before_apply") is not True:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_request_target_store_requirement_missing"
        )
    for flag in REQUIRED_REQUEST_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_executor_result_consumer_request_{flag}_not_false"
            )
    if not _record_list(_mapping(request.get("executor_dry_run")).get("dry_run_records")):
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_request_dry_run_records_missing"
        )
    return tuple(violations)


def _executor_envelope_violations(executor: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if executor.get("type") != REQUIRED_EXECUTOR_TYPE:
        violations.append("sequence_policy_promotion_executor_result_consumer_executor_type_mismatch")
    generated_at = executor.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("sequence_policy_promotion_executor_result_consumer_executor_generated_at_invalid")
    if (
        executor.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID
    ):
        violations.append("sequence_policy_promotion_executor_result_consumer_executor_method_id_mismatch")
    if (
        executor.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_STATUSES
    ):
        violations.append("sequence_policy_promotion_executor_result_consumer_executor_status_unknown")
    safety = _mapping(executor.get("safety"))
    for flag in REQUIRED_EXECUTOR_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_executor_result_consumer_executor_{flag}_not_false"
            )
    return tuple(violations)


def _executor_summary_violations(
    summary: Mapping[str, Any],
    *,
    dry_run_records: Tuple[Mapping[str, Any], ...],
    apply_results: Tuple[Mapping[str, Any], ...],
    applied: bool,
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "dry_run_record_count": len(dry_run_records),
        "apply_result_count": len(apply_results),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_promotion_executor_result_consumer_executor_summary_{key}_mismatch"
            )
    if summary.get("target_policy_store") != TARGET_POLICY_STORE:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_executor_summary_target_store_mismatch"
        )
    if summary.get("apply_requested") is not applied:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_executor_summary_apply_requested_mismatch"
        )
    if summary.get("operator_ack") is not applied:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_executor_summary_operator_ack_mismatch"
        )
    if summary.get("target_store_provided") is not applied:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_executor_summary_target_store_provided_mismatch"
        )
    for field in (
        "sequence_policy_promotion_applied",
        "runtime_memory_mutation",
        "sequence_policy_trust_store_mutation",
    ):
        if summary.get(field) is not applied:
            violations.append(
                f"sequence_policy_promotion_executor_result_consumer_executor_summary_{field}_mismatch"
            )
    for flag in REQUIRED_EXECUTOR_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_executor_result_consumer_executor_summary_{flag}_not_false"
            )
    expected_paths = tuple(
        str(record["target_policy_path"])
        for record in dry_run_records
        if isinstance(record.get("target_policy_path"), str)
        and record.get("target_policy_path")
    )
    expected_keys = tuple(
        str(record["idempotency_key"])
        for record in dry_run_records
        if isinstance(record.get("idempotency_key"), str)
        and record.get("idempotency_key")
    )
    expected_updated = tuple(
        str(record["policy_key"])
        for record in apply_results
        if isinstance(record.get("policy_key"), str) and record.get("policy_key")
    )
    if _string_list(summary.get("target_policy_paths")) != expected_paths:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_executor_summary_target_paths_mismatch"
        )
    if _string_list(summary.get("idempotency_keys")) != expected_keys:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_executor_summary_idempotency_keys_mismatch"
        )
    if _string_list(summary.get("updated_policy_keys")) != expected_updated:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_executor_summary_updated_policy_keys_mismatch"
        )
    return tuple(violations)


def _executor_safety_apply_violations(
    executor: Mapping[str, Any],
    *,
    applied: bool,
) -> Tuple[str, ...]:
    safety = _mapping(executor.get("safety"))
    violations = []
    if safety.get("read_only") is not (not applied):
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_executor_read_only_mismatch"
        )
    if safety.get("apply_requested") is not applied:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_executor_apply_requested_mismatch"
        )
    if safety.get("operator_ack") is not applied:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_executor_operator_ack_mismatch"
        )
    if safety.get("target_store_provided") is not applied:
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_executor_target_store_provided_mismatch"
        )
    for field in (
        "runtime_memory_mutation",
        "sequence_policy_trust_store_mutation",
        "sequence_policy_promotion_applied",
    ):
        if safety.get(field) is not applied:
            violations.append(
                f"sequence_policy_promotion_executor_result_consumer_executor_{field}_mismatch"
            )
    return tuple(violations)


def _apply_result_violations(
    record: Mapping[str, Any],
    *,
    dry_run_record: Mapping[str, Any],
    index: int,
) -> Tuple[str, ...]:
    record_id = _apply_result_id(record, index)
    violations = []
    for field in REQUIRED_APPLY_RESULT_FIELDS:
        if not _string(record.get(field)):
            violations.append(f"{record_id}:{field}_missing")
    expected_pairs = {
        "promotion_executor_dry_run_id": dry_run_record.get(
            "promotion_executor_dry_run_id"
        ),
        "promotion_execution_candidate_id": dry_run_record.get(
            "promotion_execution_candidate_id"
        ),
        "promotion_acceptance_id": dry_run_record.get("promotion_acceptance_id"),
        "promotion_contract_record_id": dry_run_record.get(
            "promotion_contract_record_id"
        ),
        "promotion_candidate_id": dry_run_record.get("promotion_candidate_id"),
        "proposal_id": dry_run_record.get("proposal_id"),
        "learning_scope": dry_run_record.get("learning_scope"),
        "value_type": dry_run_record.get("value_type"),
        "source_idempotency_key": dry_run_record.get("source_idempotency_key"),
        "target_policy_store": dry_run_record.get("target_policy_store"),
        "target_policy_path": dry_run_record.get("target_policy_path"),
        "operation": dry_run_record.get("operation"),
        "target_policy_state": dry_run_record.get("target_policy_state"),
        "rollback_strategy": dry_run_record.get("rollback_strategy"),
        "idempotency_key": dry_run_record.get("idempotency_key"),
    }
    for field, expected in expected_pairs.items():
        if record.get(field) != expected:
            violations.append(f"{record_id}:{field}_mismatch")
    policy_key = _policy_key(record)
    if record.get("policy_key") != policy_key:
        violations.append(f"{record_id}:policy_key_mismatch")
    if record.get("target_policy_store") != TARGET_POLICY_STORE:
        violations.append(f"{record_id}:target_policy_store_mismatch")
    if record.get("target_policy_path") != _target_policy_path(policy_key):
        violations.append(f"{record_id}:target_policy_path_mismatch")
    if record.get("applied") is not True:
        violations.append(f"{record_id}:applied_not_true")
    if record.get("runtime_memory_mutation") is not True:
        violations.append(f"{record_id}:runtime_memory_mutation_not_true")
    if record.get("sequence_policy_trust_store_mutation") is not True:
        violations.append(f"{record_id}:sequence_policy_trust_store_mutation_not_true")
    for flag in REQUIRED_APPLY_RESULT_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(f"{record_id}:{flag}_not_false")
    return tuple(violations)


def _apply_result_identity_violations(
    apply_results: Tuple[Mapping[str, Any], ...],
    dry_run_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    dry_run_by_id = _dry_run_records_by_id(dry_run_records)
    apply_by_dry_run_id = {
        str(record["promotion_executor_dry_run_id"]): record
        for record in apply_results
        if isinstance(record.get("promotion_executor_dry_run_id"), str)
        and record.get("promotion_executor_dry_run_id")
    }
    violations = []
    if set(apply_by_dry_run_id) != set(dry_run_by_id):
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_dry_run_ids_mismatch"
        )
    if len(apply_by_dry_run_id) != len(apply_results):
        violations.append(
            "sequence_policy_promotion_executor_result_consumer_duplicate_dry_run_ids"
        )
    for index, dry_run_record in enumerate(dry_run_records):
        dry_run_id = _string(dry_run_record.get("promotion_executor_dry_run_id"))
        apply_record = _mapping(apply_by_dry_run_id.get(dry_run_id))
        if not apply_record:
            violations.append(f"{dry_run_id}:apply_result_missing")
            continue
        violations.extend(
            _apply_result_violations(
                apply_record,
                dry_run_record=dry_run_record,
                index=index,
            )
        )
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer(
    *,
    promotion_executor_dry_run_result_consumer: Mapping[str, Any] | None = None,
    promotion_executor_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify promotion executor evidence and emit an apply-audit request."""

    consumer = _mapping(promotion_executor_dry_run_result_consumer)
    executor = _mapping(promotion_executor_result)
    request = _mapping(consumer.get("promotion_executor_request"))
    dry_run_records = _dry_run_records_from_request(request)
    apply_results = _record_list(executor.get("apply_results"))
    applied = executor.get("status") == REQUIRED_EXECUTOR_APPLIED_STATUS
    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_consumer_envelope_violations(consumer),
                *_consumer_summary_violations(
                    _mapping(consumer.get("summary")),
                    dry_run_records=dry_run_records,
                ),
                *_consumer_request_violations(request),
                *_executor_envelope_violations(executor),
                *_executor_safety_apply_violations(executor, applied=applied),
                *_executor_summary_violations(
                    _mapping(executor.get("summary")),
                    dry_run_records=dry_run_records,
                    apply_results=apply_results,
                    applied=applied,
                ),
                *(
                    ("sequence_policy_promotion_executor_result_consumer_applied_status_required",)
                    if executor and executor.get("status") != REQUIRED_EXECUTOR_APPLIED_STATUS
                    else ()
                ),
                *(
                    ("sequence_policy_promotion_executor_result_consumer_apply_results_required",)
                    if executor and applied and not apply_results
                    else ()
                ),
                *(
                    ("sequence_policy_promotion_executor_result_consumer_dry_run_records_required",)
                    if consumer and not dry_run_records
                    else ()
                ),
                *(
                    _apply_result_identity_violations(apply_results, dry_run_records)
                    if applied
                    else ()
                ),
            )
        )
    )
    if not consumer or not executor:
        invalid_record_ids: Tuple[str, ...] = ()
    else:
        blocking_only = (
            "sequence_policy_promotion_executor_result_consumer_applied_status_required",
        )
        invalid_record_ids = tuple(
            item for item in raw_invalid_record_ids if item not in blocking_only
        )

    if not consumer or not executor:
        status = "sequence_memory_trust_promotion_queue_import_promotion_executor_result_consumer_not_ready"
        blocked_by = ("promotion_executor_dry_run_consumer_and_executor_result_required",)
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_executor_result_consumer_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_memory_trust_promotion_executor_evidence",
        )
    elif executor.get("status") != REQUIRED_EXECUTOR_APPLIED_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_executor_result_consumer_blocked"
        blocked_by = (
            "applied_sequence_memory_trust_promotion_executor_result_required_before_apply_audit",
        )
    elif not apply_results:
        status = "sequence_memory_trust_promotion_queue_import_promotion_executor_result_consumer_not_ready"
        blocked_by = ("sequence_memory_trust_promotion_executor_apply_results_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_request_available"
        blocked_by = (
            "operator_may_run_read_only_sequence_memory_trust_promotion_apply_audit",
        )

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_request_available"
    )
    apply_audit_request: Dict[str, Any] = {
        "handler": "audit_operator_proposal_apply_sequence_policy_memory_trust_promotion",
        "read_only_template": True,
        "promotion_executor_result_consumer_only": True,
        "promotion_apply_audit_request_template_only": True,
        "promotion_apply_audit_handler_invoked": False,
        "promotion_apply_audit_created": False,
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
        apply_audit_request["promotion_executor"] = executor

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_EXECUTOR_DRY_RUN_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_executor_result_consumer_only": True,
            "promotion_apply_audit_request_template_only": True,
            "requires_promotion_executor_dry_run_result_consumer": True,
            "requires_promotion_executor_result": True,
            "requires_applied_executor_result": True,
            "promotion_apply_audit_handler_invoked": False,
            "promotion_apply_audit_created": False,
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
            "expected_executor_apply_result_count": len(dry_run_records),
            "executor_apply_result_count": len(apply_results),
            "invalid_record_count": len(invalid_record_ids),
            "target_policy_store": TARGET_POLICY_STORE,
            "promotion_executor_result_consumer_only": True,
            "promotion_apply_audit_request_template_only": True,
            "promotion_apply_audit_request_available": request_available,
            "promotion_apply_audit_handler_invoked": False,
            "promotion_apply_audit_created": False,
            "requires_applied_executor_result": True,
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
            "updated_policy_keys": [
                str(record["policy_key"])
                for record in apply_results
                if isinstance(record.get("policy_key"), str) and record.get("policy_key")
            ],
            "target_policy_paths": [
                str(record["target_policy_path"])
                for record in apply_results
                if isinstance(record.get("target_policy_path"), str)
                and record.get("target_policy_path")
            ],
            "idempotency_keys": [
                str(record["idempotency_key"])
                for record in apply_results
                if isinstance(record.get("idempotency_key"), str)
                and record.get("idempotency_key")
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion executor dry-run result consumer and executor result."
                if not consumer or not executor
                else (
                    "Replace invalid sequence-memory trust promotion executor evidence before apply audit."
                    if invalid_record_ids
                    else (
                        "Run the promotion executor with explicit apply, operator acknowledgement, and target policy store before apply audit."
                        if executor.get("status") != REQUIRED_EXECUTOR_APPLIED_STATUS
                        else "Run the read-only apply-audit request; this consumer does not audit or mutate policy."
                    )
                )
            ),
        },
        "promotion_apply_audit_request": apply_audit_request,
    }
