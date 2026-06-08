"""Read-only consumer for sequence-policy queue-import audit-policy gates.

This module verifies that a generic apply-sequence audit-policy gate still
matches sequence-policy queue-import provenance before a caller may pass it to
the generic policy-memory draft handler. It does not invoke draft or write
handlers, mutate memory or policy stores, reserve idempotency keys, or touch
live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_mode_policy import CURRENT_PROVEN_LIVE_WRITE_KINDS
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_mode_policy import CURRENT_PROVEN_LIVE_WRITE_KINDS

ALLOWED_VALUE_TYPES = set(CURRENT_PROVEN_LIVE_WRITE_KINDS)

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_AUDIT_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_AUDIT_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_AUDIT_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_AUDIT_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_AUDIT_POLICY_GATE_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_AUDIT_POLICY_GATE_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_memory_draft_request_available",
}
REQUIRED_EXECUTION_AUDIT_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer"
)
REQUIRED_EXECUTION_AUDIT_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_audit_policy_gate_request_available"
)
REQUIRED_AUDIT_POLICY_GATE_TYPE = "operator_proposal_apply_sequence_audit_policy_gate"
VERIFIED_READBACK_STATUSES = {
    "verified_from_cache",
    "verified_from_physical_readback",
}
REQUIRED_CONSUMER_FALSE_FLAGS = (
    "audit_policy_gate_handler_invoked",
    "policy_gate_created",
    "execution_allowed_by_consumer",
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "policy_mutation",
    "runtime_memory_mutation",
    "model_mutation",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_apply_live",
)
REQUIRED_POLICY_GATE_FALSE_FLAGS = (
    "execution_allowed_by_policy_gate",
    "execution_allowed_by_audit",
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "mutates_console_state",
    "queries_console",
    "policy_mutation",
    "runtime_memory_mutation",
    "model_mutation",
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


def _duplicate_values(values: Tuple[str, ...]) -> Tuple[str, ...]:
    counts: Dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return tuple(value for value, count in counts.items() if count > 1)


def _policy_row_id(row: Mapping[str, Any], index: int) -> str:
    value = _string(row.get("proposal_id"))
    return value if value else f"policy_evidence:{index}"


def _consumer_envelope_violations(consumer: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_EXECUTION_AUDIT_RESULT_CONSUMER_TYPE:
        violations.append("sequence_policy_audit_policy_gate_result_consumer_audit_consumer_type_mismatch")
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_AUDIT_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append(
            "sequence_policy_audit_policy_gate_result_consumer_audit_consumer_method_id_mismatch"
        )
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_AUDIT_RESULT_CONSUMER_STATUSES
    ):
        violations.append(
            "sequence_policy_audit_policy_gate_result_consumer_audit_consumer_status_unknown"
        )
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_policy_audit_policy_gate_result_consumer_audit_consumer_read_only_not_true")
    if safety.get("execution_audit_result_consumer_only") is not True:
        violations.append("sequence_policy_audit_policy_gate_result_consumer_audit_consumer_only_not_true")
    if safety.get("audit_policy_gate_request_template_only") is not True:
        violations.append("sequence_policy_audit_policy_gate_result_consumer_audit_consumer_template_only_not_true")
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_audit_policy_gate_result_consumer_audit_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if summary.get("audit_policy_gate_request_available") is not bool(expected_ids):
        violations.append(
            "sequence_policy_audit_policy_gate_result_consumer_audit_consumer_request_available_mismatch"
        )
    expected_counts = {
        "expected_execution_audit_row_count": len(expected_ids),
        "execution_audit_row_count": len(expected_ids),
        "passed_execution_audit_row_count": len(expected_ids),
        "blocked_execution_audit_row_count": 0,
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_audit_policy_gate_result_consumer_audit_consumer_{key}_mismatch"
            )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_audit_policy_gate_result_consumer_audit_consumer_summary_{flag}_not_false"
            )
    if _string_list(summary.get("execution_audit_row_proposal_ids")) != expected_ids:
        violations.append(
            "sequence_policy_audit_policy_gate_result_consumer_audit_consumer_row_ids_mismatch"
        )
    if _string_list(summary.get("passed_execution_audit_proposal_ids")) != expected_ids:
        violations.append(
            "sequence_policy_audit_policy_gate_result_consumer_audit_consumer_passed_ids_mismatch"
        )
    return tuple(violations)


def _policy_gate_envelope_violations(policy_gate: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if policy_gate.get("type") != REQUIRED_AUDIT_POLICY_GATE_TYPE:
        violations.append("sequence_policy_audit_policy_gate_result_consumer_policy_gate_type_mismatch")
    if policy_gate.get("status") not in {"ready", "blocked"}:
        violations.append("sequence_policy_audit_policy_gate_result_consumer_policy_gate_status_unknown")
    if policy_gate.get("status") == "ready" and policy_gate.get("success") is not True:
        violations.append("sequence_policy_audit_policy_gate_result_consumer_policy_gate_success_not_true")
    if policy_gate.get("status") != "ready" and policy_gate.get("success") is not False:
        violations.append("sequence_policy_audit_policy_gate_result_consumer_policy_gate_blocked_success_not_false")

    summary = _mapping(policy_gate.get("summary"))
    if summary.get("policy_gate_only") is not True:
        violations.append("sequence_policy_audit_policy_gate_result_consumer_policy_gate_summary_only_not_true")
    if summary.get("audit_payload_required") is not True:
        violations.append("sequence_policy_audit_policy_gate_result_consumer_policy_gate_summary_audit_required_not_true")
    for flag in REQUIRED_POLICY_GATE_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(f"sequence_policy_audit_policy_gate_result_consumer_policy_gate_summary_{flag}_not_false")

    safety = _mapping(policy_gate.get("safety"))
    if safety.get("policy_gate_only") is not True:
        violations.append("sequence_policy_audit_policy_gate_result_consumer_policy_gate_safety_only_not_true")
    if safety.get("audit_payload_required") is not True:
        violations.append("sequence_policy_audit_policy_gate_result_consumer_policy_gate_safety_audit_required_not_true")
    if safety.get("does_not_call_apply_operator_proposal") is not True:
        violations.append("sequence_policy_audit_policy_gate_result_consumer_policy_gate_safety_apply_denial_missing")
    if safety.get("does_not_call_manual_console_write") is not True:
        violations.append("sequence_policy_audit_policy_gate_result_consumer_policy_gate_safety_manual_write_denial_missing")
    allowed_kinds = _string_list(safety.get("allowed_live_write_kinds"))
    if tuple(sorted(allowed_kinds)) != tuple(sorted(CURRENT_PROVEN_LIVE_WRITE_KINDS)):
        violations.append("sequence_policy_audit_policy_gate_result_consumer_policy_gate_safety_allowed_kinds_mismatch")
    for flag in REQUIRED_POLICY_GATE_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"sequence_policy_audit_policy_gate_result_consumer_policy_gate_safety_{flag}_not_false")
    if _record_list(policy_gate.get("blocked_results")):
        violations.append("sequence_policy_audit_policy_gate_result_consumer_policy_gate_has_blocked_results")
    return tuple(violations)


def _policy_gate_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
    rows: Tuple[Mapping[str, Any], ...],
    blocked_rows: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "input_audit_row_count": len(expected_ids),
        "policy_evidence_ready_count": len(rows) - len(blocked_rows),
        "blocked_count": len(blocked_rows),
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"sequence_policy_audit_policy_gate_result_consumer_policy_gate_summary_{key}_mismatch")
    ready_ids = tuple(
        _policy_row_id(row, index)
        for index, row in enumerate(rows)
        if row.get("status") == "policy_evidence_ready"
    )
    if _string_list(summary.get("policy_evidence_ready_proposal_ids")) != ready_ids:
        violations.append("sequence_policy_audit_policy_gate_result_consumer_policy_gate_summary_ready_ids_mismatch")
    blocked_ids = tuple(_policy_row_id(row, index) for index, row in enumerate(blocked_rows))
    if _string_list(summary.get("blocked_proposal_ids")) != blocked_ids:
        violations.append("sequence_policy_audit_policy_gate_result_consumer_policy_gate_summary_blocked_ids_mismatch")
    return tuple(violations)


def _policy_evidence_violations(
    row: Mapping[str, Any],
    *,
    index: int,
    prefix: str,
) -> Tuple[str, ...]:
    proposal_id = _policy_row_id(row, index)
    violations = []
    if row.get("audit_row_index") != index:
        violations.append(f"{proposal_id}:audit_row_index_mismatch")
    if row.get("status") != "policy_evidence_ready":
        violations.append(f"{proposal_id}:policy_evidence_status_not_ready")
    if row.get("eligible_for_supervised_sequence_policy") is not True:
        violations.append(f"{proposal_id}:not_supervised_sequence_policy_eligible")
    for flag in (
        "execution_allowed_by_policy_gate",
        "policy_mutation_allowed",
        "live_authority_granted",
        "full_autonomous_ready",
    ):
        if row.get(flag) is not False:
            violations.append(f"{proposal_id}:{flag}_not_false")
    value_type = _string(row.get("value_type"))
    if value_type not in ALLOWED_VALUE_TYPES:
        violations.append(f"{proposal_id}:value_type_not_memory_safe")
    effective_value = row.get("effective_value")
    if (
        not isinstance(effective_value, (int, float))
        or isinstance(effective_value, bool)
        or not isfinite(float(effective_value))
    ):
        violations.append(f"{proposal_id}:effective_value_invalid")
    elif value_type == "fader" and float(effective_value) > 0.0:
        violations.append(f"{proposal_id}:fader_ceiling_exceeded")
    if row.get("idempotency_key") != f"{prefix}:{proposal_id}":
        violations.append(f"{proposal_id}:idempotency_key_mismatch")
    if row.get("readback_status") not in VERIFIED_READBACK_STATUSES:
        violations.append(f"{proposal_id}:readback_not_verified")
    if row.get("proof_status") != "verified":
        violations.append(f"{proposal_id}:proof_status_not_verified")
    if row.get("rollback_available") is not True:
        violations.append(f"{proposal_id}:rollback_unavailable")
    if row.get("live_apply_entrypoint") != "apply_operator_proposal":
        violations.append(f"{proposal_id}:live_apply_entrypoint_mismatch")
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer(
    *,
    execution_audit_result_consumer: Mapping[str, Any] | None = None,
    audit_policy_gate_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify policy-gate evidence and emit a policy-memory-draft request."""

    consumer = _mapping(execution_audit_result_consumer)
    policy_gate = _mapping(audit_policy_gate_result)
    consumer_summary = _mapping(consumer.get("summary"))
    expected_ids = _string_list(consumer_summary.get("execution_audit_row_proposal_ids"))
    prefix = _string(consumer_summary.get("idempotency_key_prefix"))
    rows = _record_list(policy_gate.get("policy_evidence"))
    blocked_rows = tuple(row for row in rows if row.get("status") == "blocked")
    row_ids = tuple(_policy_row_id(row, index) for index, row in enumerate(rows))
    row_idempotency_keys = tuple(_string(row.get("idempotency_key")) for row in rows)
    row_violations = tuple(
        violation
        for index, row in enumerate(rows)
        if row.get("status") == "policy_evidence_ready"
        for violation in _policy_evidence_violations(row, index=index, prefix=prefix)
    )
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_consumer_envelope_violations(consumer),
                *_consumer_summary_violations(consumer_summary, expected_ids=expected_ids),
                *(
                    ("sequence_policy_audit_policy_gate_result_consumer_audit_consumer_status_not_available",)
                    if consumer.get("status")
                    != REQUIRED_EXECUTION_AUDIT_RESULT_CONSUMER_AVAILABLE_STATUS
                    else ()
                ),
                *_policy_gate_envelope_violations(policy_gate),
                *_policy_gate_summary_violations(
                    _mapping(policy_gate.get("summary")),
                    expected_ids=expected_ids,
                    rows=rows,
                    blocked_rows=blocked_rows,
                ),
                *(
                    f"duplicate_policy_evidence_id:{value}"
                    for value in _duplicate_values(row_ids)
                ),
                *(
                    f"duplicate_policy_evidence_idempotency_key:{value}"
                    for value in _duplicate_values(row_idempotency_keys)
                ),
                *(
                    ("sequence_policy_audit_policy_gate_result_consumer_row_id_outside_expected_set",)
                    if any(proposal_id not in set(expected_ids) for proposal_id in row_ids)
                    else ()
                ),
                *(
                    ("sequence_policy_audit_policy_gate_result_consumer_row_ids_do_not_cover_expected_set",)
                    if not blocked_rows and set(row_ids) != set(expected_ids)
                    else ()
                ),
                *(
                    ("sequence_policy_audit_policy_gate_result_consumer_row_order_mismatch",)
                    if not blocked_rows and row_ids != expected_ids
                    else ()
                ),
                *row_violations,
            )
        )
    )

    if not consumer or not policy_gate:
        status = "sequence_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer_not_ready"
        blocked_by = ("execution_audit_result_consumer_and_audit_policy_gate_result_required",)
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer_replacement_required"
        blocked_by = ("operator_must_replace_invalid_sequence_policy_audit_policy_gate_evidence",)
    elif policy_gate.get("status") != "ready" or blocked_rows:
        status = "sequence_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer_blocked"
        blocked_by = ("all_sequence_policy_imported_policy_evidence_rows_must_be_ready",)
    elif not rows:
        status = "sequence_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer_not_ready"
        blocked_by = ("policy_evidence_ready_rows_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_memory_draft_request_available"
        blocked_by = ("operator_may_run_generic_policy_memory_draft",)

    request_available = (
        status == "sequence_memory_trust_promotion_queue_import_memory_draft_request_available"
    )
    selected_rows = rows if request_available else ()
    draft_specs = tuple(
        {
            "memory_draft_request_spec_id": (
                f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_AUDIT_POLICY_GATE_RESULT_CONSUMER_METHOD_ID}:{proposal_id}"
            ),
            "proposal_id": proposal_id,
            "planned_memory_draft_index": index,
            "source_policy_evidence_index": index,
            "idempotency_key": f"{prefix}:{proposal_id}",
            "read_only": True,
            "audit_policy_gate_result_consumer_only": True,
            "memory_draft_handler": "draft_operator_proposal_apply_sequence_policy_memory",
            "memory_draft_handler_invoked": False,
            "memory_draft_created": False,
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
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
        }
        for index, proposal_id in enumerate(row_ids if request_available else ())
    )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_AUDIT_POLICY_GATE_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_EXECUTION_AUDIT_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "audit_policy_gate_result_consumer_only": True,
            "memory_draft_request_template_only": True,
            "requires_execution_audit_result_consumer": True,
            "requires_audit_policy_gate_result": True,
            "requires_all_policy_evidence_ready": True,
            "memory_draft_handler_invoked": False,
            "memory_draft_created": False,
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
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
        },
        "summary": {
            "expected_policy_evidence_count": len(expected_ids),
            "policy_evidence_count": len(rows),
            "ready_policy_evidence_count": len(rows) - len(blocked_rows),
            "blocked_policy_evidence_count": len(blocked_rows),
            "invalid_record_count": len(invalid_record_ids),
            "idempotency_key_prefix": prefix,
            "audit_policy_gate_result_consumer_only": True,
            "memory_draft_request_available": request_available,
            "memory_draft_handler_invoked": False,
            "memory_draft_created": False,
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
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
            "expected_policy_evidence_proposal_ids": list(expected_ids),
            "policy_evidence_proposal_ids": list(row_ids),
            "ready_policy_evidence_proposal_ids": [
                _policy_row_id(row, index)
                for index, row in enumerate(rows)
                if row.get("status") == "policy_evidence_ready"
            ],
            "blocked_policy_evidence_proposal_ids": [
                _policy_row_id(row, index)
                for index, row in enumerate(blocked_rows)
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide execution-audit result consumer plus audit-policy-gate result."
                if not consumer or not policy_gate
                else (
                    "Replace invalid audit-policy-gate evidence before memory draft."
                    if invalid_record_ids
                    else (
                        "Re-run audit-policy gate until every sequence-policy imported row is policy-evidence ready."
                        if policy_gate.get("status") != "ready" or blocked_rows
                        else "Operator may run the generic policy-memory draft handler; this consumer does not run it."
                    )
                )
            ),
        },
        "memory_draft_request": {
            "handler": "draft_operator_proposal_apply_sequence_policy_memory",
            "policy_gate_payload": {
                **dict(policy_gate),
                "policy_evidence": list(selected_rows),
            }
            if request_available
            else {},
            "read_only_template": True,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "memory_draft_created": False,
            "execution_allowed_by_consumer": False,
            "policy_mutation": False,
            "memory_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
        },
        "memory_draft_request_specs": list(draft_specs),
    }
