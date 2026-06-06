"""Rollback audit for applied sequence-policy memory trust promotion.

The audit consumes the operator-gated sequence-policy promotion executor output
and validates applied results before any policy consumer is allowed to trust
them. It does not promote sequence policy, execute proposals, mutate queues or
models, or touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_executor import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_executor import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_STATUSES = {
    "sequence_memory_trust_promotion_apply_audit_not_ready",
    "sequence_memory_trust_promotion_apply_audit_replacement_required",
    "sequence_memory_trust_promotion_apply_audit_records_available",
}
REQUIRED_EXECUTOR_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor"
)
TARGET_POLICY_STORE = "operator_sequence_policy_trust_gate"
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


def _record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("promotion_executor_apply_id"))
    if explicit:
        return explicit
    dry_run_id = _string(record.get("promotion_executor_dry_run_id"))
    return f"{dry_run_id if dry_run_id else 'unknown_sequence_promotion_apply'}:{index}"


def _target_policy_path(policy_key: str) -> str:
    return f"{TARGET_POLICY_STORE}/{policy_key}/trust_state"


def _executor_envelope_violations(executor: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if executor.get("type") != REQUIRED_EXECUTOR_TYPE:
        violations.append("sequence_memory_trust_promotion_apply_audit_executor_type_mismatch")
    generated_at = executor.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append(
            "sequence_memory_trust_promotion_apply_audit_executor_generated_at_invalid"
        )
    if (
        executor.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID
    ):
        violations.append(
            "sequence_memory_trust_promotion_apply_audit_executor_method_id_mismatch"
        )
    if (
        executor.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_STATUSES
    ):
        violations.append("sequence_memory_trust_promotion_apply_audit_executor_status_unknown")
    safety = _mapping(executor.get("safety"))
    for flag in REQUIRED_EXECUTOR_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_memory_trust_promotion_apply_audit_executor_{flag}_not_false"
            )
    return tuple(violations)


def _executor_status_violations(
    executor: Mapping[str, Any],
    *,
    apply_results: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = executor.get("status")
    if status == "sequence_memory_trust_promotion_executor_replacement_required":
        return (
            "sequence_memory_trust_promotion_apply_audit_executor_replacement_required",
        )
    if status == "sequence_memory_trust_promotion_executor_applied" and not apply_results:
        return ("sequence_memory_trust_promotion_apply_audit_applied_without_results",)
    if status != "sequence_memory_trust_promotion_executor_applied" and apply_results:
        return (
            "sequence_memory_trust_promotion_apply_audit_results_without_applied_status",
        )
    return ()


def _executor_apply_safety_violations(
    executor: Mapping[str, Any],
    *,
    applied: bool,
) -> Tuple[str, ...]:
    violations = []
    safety = _mapping(executor.get("safety"))
    summary = _mapping(executor.get("summary"))
    expected_true_flags = (
        "runtime_memory_mutation",
        "sequence_policy_trust_store_mutation",
        "sequence_policy_promotion_applied",
    )
    for field in expected_true_flags:
        if safety.get(field) is not applied:
            violations.append(
                f"sequence_memory_trust_promotion_apply_audit_executor_{field}_mismatch"
            )
        if summary.get(field) is not applied:
            violations.append(
                f"sequence_memory_trust_promotion_apply_audit_summary_{field}_mismatch"
            )
    if safety.get("read_only") is not (not applied):
        violations.append("sequence_memory_trust_promotion_apply_audit_executor_read_only_mismatch")
    for flag in REQUIRED_EXECUTOR_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_memory_trust_promotion_apply_audit_summary_{flag}_not_false"
            )
    if summary.get("target_policy_store") != TARGET_POLICY_STORE:
        violations.append(
            "sequence_memory_trust_promotion_apply_audit_summary_target_policy_store_invalid"
        )
    return tuple(violations)


def _apply_result_violations(record: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    policy_key = _string(record.get("policy_key"))
    for field in REQUIRED_APPLY_RESULT_FIELDS:
        if not _string(record.get(field)):
            violations.append(
                f"sequence_memory_trust_promotion_apply_audit_record_{field}_missing"
            )
    if record.get("target_policy_store") != TARGET_POLICY_STORE:
        violations.append(
            "sequence_memory_trust_promotion_apply_audit_record_target_policy_store_invalid"
        )
    if record.get("target_policy_path") != _target_policy_path(policy_key):
        violations.append(
            "sequence_memory_trust_promotion_apply_audit_record_target_policy_path_invalid"
        )
    if record.get("operation") != "set_operator_sequence_policy_trust_state":
        violations.append("sequence_memory_trust_promotion_apply_audit_record_operation_invalid")
    if record.get("applied") is not True:
        violations.append("sequence_memory_trust_promotion_apply_audit_record_applied_not_true")
    if record.get("sequence_policy_trust_store_mutation") is not True:
        violations.append(
            "sequence_memory_trust_promotion_apply_audit_record_policy_store_mutation_not_true"
        )
    if record.get("runtime_memory_mutation") is not True:
        violations.append(
            "sequence_memory_trust_promotion_apply_audit_record_runtime_memory_mutation_not_true"
        )
    for flag in REQUIRED_APPLY_RESULT_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(
                f"sequence_memory_trust_promotion_apply_audit_record_{flag}_not_false"
            )
    return tuple(violations)


def _record_replacement_ids(records: Tuple[Mapping[str, Any], ...]) -> Tuple[str, ...]:
    return tuple(
        _record_id(record, index)
        for index, record in enumerate(records)
        if _apply_result_violations(record)
    )


def _duplicate_ids(records: Tuple[Mapping[str, Any], ...], field: str) -> Tuple[str, ...]:
    values: Dict[str, list[str]] = {}
    for index, record in enumerate(records):
        if _apply_result_violations(record):
            continue
        value = _string(record.get(field))
        if not value:
            continue
        values.setdefault(value, []).append(_record_id(record, index))
    return tuple(
        record_id
        for record_ids in values.values()
        if len(record_ids) > 1
        for record_id in record_ids
    )


def _summary_count_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    expected_counts = {
        "apply_result_count": len(records),
        "invalid_record_count": 0,
    }
    return tuple(
        f"sequence_memory_trust_promotion_apply_audit_summary_{key}_mismatch"
        for key, expected in expected_counts.items()
        if _summary_int(summary.get(key)) != expected
    )


def _summary_identity_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    if not records:
        return ()
    expected_paths = tuple(
        record["target_policy_path"]
        for record in records
        if _string(record.get("target_policy_path"))
    )
    expected_keys = tuple(
        record["idempotency_key"]
        for record in records
        if _string(record.get("idempotency_key"))
    )
    expected_policy_keys = tuple(
        record["policy_key"] for record in records if _string(record.get("policy_key"))
    )
    violations = []
    if _string_list(summary.get("target_policy_paths")) != expected_paths:
        violations.append("sequence_memory_trust_promotion_apply_audit_summary_paths_mismatch")
    if _string_list(summary.get("idempotency_keys")) != expected_keys:
        violations.append("sequence_memory_trust_promotion_apply_audit_summary_keys_mismatch")
    if _string_list(summary.get("updated_policy_keys")) != expected_policy_keys:
        violations.append(
            "sequence_memory_trust_promotion_apply_audit_summary_policy_keys_mismatch"
        )
    return tuple(violations)


def _audit_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "promotion_apply_audit_id": (
            "operator_sequence_memory_trust_promotion_apply_audit:"
            f"{record.get('promotion_executor_apply_id')}"
        ),
        "promotion_executor_apply_id": record.get("promotion_executor_apply_id"),
        "promotion_executor_dry_run_id": record.get("promotion_executor_dry_run_id"),
        "promotion_execution_candidate_id": record.get(
            "promotion_execution_candidate_id"
        ),
        "promotion_acceptance_id": record.get("promotion_acceptance_id"),
        "promotion_contract_record_id": record.get("promotion_contract_record_id"),
        "promotion_candidate_id": record.get("promotion_candidate_id"),
        "proposal_id": record.get("proposal_id"),
        "learning_scope": record.get("learning_scope"),
        "value_type": record.get("value_type"),
        "source_idempotency_key": record.get("source_idempotency_key"),
        "policy_key": record.get("policy_key"),
        "target_policy_store": TARGET_POLICY_STORE,
        "target_policy_path": record.get("target_policy_path"),
        "operation": record.get("operation"),
        "previous_policy_state": record.get("previous_policy_state"),
        "target_policy_state": record.get("target_policy_state"),
        "rollback_strategy": record.get("rollback_strategy"),
        "rollback_target_policy_state": record.get("previous_policy_state"),
        "rollback_evidence_required": True,
        "rollback_verified": False,
        "idempotency_key": record.get("idempotency_key"),
        "applied": True,
        "read_only": True,
        "live_write": False,
        "apply_attempted": False,
        "mutates_console_state": False,
        "queries_console": False,
        "queue_mutation": False,
        "model_mutation": False,
        "runtime_memory_mutation": False,
        "sequence_policy_trust_store_mutation": False,
        "audits_applied_sequence_policy_trust_store_mutation": True,
        "proposal_state_mutation": False,
        "sequence_execution_allowed_by_audit": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit(
    *,
    promotion_executor: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a read-only rollback audit for applied sequence-policy promotion."""

    executor = (
        _mapping(promotion_executor)
        or build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor()
    )
    summary = _mapping(executor.get("summary"))
    apply_results = _record_list(executor.get("apply_results"))
    applied = bool(apply_results)
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_executor_envelope_violations(executor),
                *_executor_status_violations(executor, apply_results=apply_results),
                *_executor_apply_safety_violations(executor, applied=applied),
                *_record_replacement_ids(apply_results),
                *_duplicate_ids(apply_results, "promotion_executor_apply_id"),
                *_duplicate_ids(apply_results, "policy_key"),
                *_duplicate_ids(apply_results, "idempotency_key"),
                *_summary_count_violations(summary, records=apply_results),
                *_summary_identity_violations(summary, records=apply_results),
            )
        )
    )
    audit_records = (
        ()
        if invalid_record_ids
        else tuple(_audit_record(record) for record in apply_results)
    )

    if invalid_record_ids:
        status = "sequence_memory_trust_promotion_apply_audit_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_promotion_executor_before_apply_audit",
        )
    elif audit_records:
        status = "sequence_memory_trust_promotion_apply_audit_records_available"
        blocked_by = (
            "rollback_audit_is_not_sequence_policy_or_live_authority",
        )
    else:
        status = "sequence_memory_trust_promotion_apply_audit_not_ready"
        blocked_by = (
            "applied_sequence_policy_promotion_result_required_before_apply_audit",
        )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_APPLY_AUDIT_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "apply_attempted": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "sequence_policy_trust_store_mutation": False,
            "audits_applied_sequence_policy_trust_store_mutation": bool(audit_records),
            "proposal_state_mutation": False,
            "sequence_execution_allowed_by_audit": False,
            "rollback_evidence_required": bool(audit_records),
            "rollback_verified": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "promotion_executor_status": executor.get("status") or "missing",
            "executor_apply_result_count": len(apply_results),
            "apply_audit_record_count": len(audit_records),
            "invalid_record_count": len(invalid_record_ids),
            "target_policy_store": TARGET_POLICY_STORE,
            "rollback_evidence_required": bool(audit_records),
            "rollback_verified": False,
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "sequence_policy_trust_store_mutation": False,
            "audits_applied_sequence_policy_trust_store_mutation": bool(audit_records),
            "proposal_state_mutation": False,
            "sequence_execution_allowed_by_audit": False,
            "live_write": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "audited_policy_keys": [record["policy_key"] for record in audit_records],
            "rollback_target_policy_states": [
                record["rollback_target_policy_state"] for record in audit_records
            ],
            "target_policy_paths": [
                record.get("target_policy_path") for record in audit_records
            ],
            "idempotency_keys": [
                record.get("idempotency_key") for record in audit_records
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid sequence promotion executor output before audit."
                if invalid_record_ids
                else (
                    "Use rollback audit as review evidence only; a separate "
                    "rollback verification gate is required before sequence policy consumption."
                    if audit_records
                    else "Applied sequence policy promotion executor results are required before rollback audit."
                )
            ),
        },
        "audit_records": list(audit_records),
    }
