"""Read-only consumer for sequence-policy queue-import memory drafts.

This module verifies that a generic policy-memory draft still matches
sequence-policy queue-import provenance before a caller may pass it to the
generic append-only memory writer. It does not invoke writer handlers, mutate
memory files or stores, reserve idempotency keys, or touch live mixer transport.
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
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_AUDIT_POLICY_GATE_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_AUDIT_POLICY_GATE_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_AUDIT_POLICY_GATE_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_AUDIT_POLICY_GATE_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_DRAFT_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_DRAFT_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_memory_draft_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_memory_draft_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_memory_draft_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_memory_writer_request_available",
}
REQUIRED_AUDIT_POLICY_GATE_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer"
)
REQUIRED_AUDIT_POLICY_GATE_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_memory_draft_request_available"
)
REQUIRED_MEMORY_DRAFT_TYPE = "operator_proposal_apply_sequence_policy_memory_draft"
REQUIRED_TARGET_MEMORY_PATH = ".ai/memory/project.md"
REQUIRED_LEARNING_SCOPE = "supervised_fader_gain_sequence_execution"
VERIFIED_READBACK_STATUSES = {
    "verified_from_cache",
    "verified_from_physical_readback",
}
REQUIRED_CONSUMER_FALSE_FLAGS = (
    "memory_draft_handler_invoked",
    "memory_draft_created",
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
    "runtime_memory_mutation",
    "model_mutation",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_apply_live",
)
REQUIRED_MEMORY_DRAFT_FALSE_FLAGS = (
    "append_only_memory_write_allowed_by_draft",
    "execution_allowed_by_draft",
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "mutates_console_state",
    "queries_console",
    "policy_mutation",
    "memory_mutation",
    "runtime_memory_mutation",
    "model_mutation",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_RECORD_FALSE_FLAGS = (
    "memory_write_allowed_by_draft",
    "append_only_memory_write_allowed_by_draft",
    "memory_mutation",
    "runtime_memory_mutation",
    "policy_mutation",
    "model_mutation",
    "queue_mutation",
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


def _draft_record_id(record: Mapping[str, Any], index: int) -> str:
    value = _string(record.get("proposal_id"))
    return value if value else f"memory_draft:{index}"


def _consumer_envelope_violations(consumer: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_AUDIT_POLICY_GATE_RESULT_CONSUMER_TYPE:
        violations.append("sequence_policy_memory_draft_result_consumer_policy_gate_consumer_type_mismatch")
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_AUDIT_POLICY_GATE_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append(
            "sequence_policy_memory_draft_result_consumer_policy_gate_consumer_method_id_mismatch"
        )
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_AUDIT_POLICY_GATE_RESULT_CONSUMER_STATUSES
    ):
        violations.append(
            "sequence_policy_memory_draft_result_consumer_policy_gate_consumer_status_unknown"
        )
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_policy_memory_draft_result_consumer_policy_gate_consumer_read_only_not_true")
    if safety.get("audit_policy_gate_result_consumer_only") is not True:
        violations.append("sequence_policy_memory_draft_result_consumer_policy_gate_consumer_only_not_true")
    if safety.get("memory_draft_request_template_only") is not True:
        violations.append("sequence_policy_memory_draft_result_consumer_policy_gate_consumer_template_only_not_true")
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_draft_result_consumer_policy_gate_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if summary.get("memory_draft_request_available") is not bool(expected_ids):
        violations.append(
            "sequence_policy_memory_draft_result_consumer_policy_gate_consumer_request_available_mismatch"
        )
    expected_counts = {
        "expected_policy_evidence_count": len(expected_ids),
        "policy_evidence_count": len(expected_ids),
        "ready_policy_evidence_count": len(expected_ids),
        "blocked_policy_evidence_count": 0,
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_memory_draft_result_consumer_policy_gate_consumer_{key}_mismatch"
            )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_draft_result_consumer_policy_gate_consumer_summary_{flag}_not_false"
            )
    if _string_list(summary.get("policy_evidence_proposal_ids")) != expected_ids:
        violations.append(
            "sequence_policy_memory_draft_result_consumer_policy_gate_consumer_policy_ids_mismatch"
        )
    if _string_list(summary.get("ready_policy_evidence_proposal_ids")) != expected_ids:
        violations.append(
            "sequence_policy_memory_draft_result_consumer_policy_gate_consumer_ready_policy_ids_mismatch"
        )
    return tuple(violations)


def _memory_draft_envelope_violations(memory_draft: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if memory_draft.get("type") != REQUIRED_MEMORY_DRAFT_TYPE:
        violations.append("sequence_policy_memory_draft_result_consumer_draft_type_mismatch")
    if memory_draft.get("status") not in {"ready", "blocked"}:
        violations.append("sequence_policy_memory_draft_result_consumer_draft_status_unknown")
    if memory_draft.get("status") == "ready" and memory_draft.get("success") is not True:
        violations.append("sequence_policy_memory_draft_result_consumer_draft_success_not_true")
    if memory_draft.get("status") != "ready" and memory_draft.get("success") is not False:
        violations.append("sequence_policy_memory_draft_result_consumer_draft_blocked_success_not_false")

    summary = _mapping(memory_draft.get("summary"))
    if summary.get("memory_draft_only") is not True:
        violations.append("sequence_policy_memory_draft_result_consumer_draft_summary_only_not_true")
    if summary.get("policy_gate_payload_required") is not True:
        violations.append("sequence_policy_memory_draft_result_consumer_draft_summary_policy_gate_required_not_true")
    if summary.get("requires_separate_memory_writer") is not True:
        violations.append("sequence_policy_memory_draft_result_consumer_draft_summary_separate_writer_not_true")
    for flag in REQUIRED_MEMORY_DRAFT_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(f"sequence_policy_memory_draft_result_consumer_draft_summary_{flag}_not_false")

    safety = _mapping(memory_draft.get("safety"))
    if safety.get("memory_draft_only") is not True:
        violations.append("sequence_policy_memory_draft_result_consumer_draft_safety_only_not_true")
    if safety.get("policy_gate_payload_required") is not True:
        violations.append("sequence_policy_memory_draft_result_consumer_draft_safety_policy_gate_required_not_true")
    if safety.get("requires_separate_memory_writer") is not True:
        violations.append("sequence_policy_memory_draft_result_consumer_draft_safety_separate_writer_not_true")
    if safety.get("does_not_call_apply_operator_proposal") is not True:
        violations.append("sequence_policy_memory_draft_result_consumer_draft_safety_apply_denial_missing")
    if safety.get("does_not_call_manual_console_write") is not True:
        violations.append("sequence_policy_memory_draft_result_consumer_draft_safety_manual_write_denial_missing")
    allowed_kinds = _string_list(safety.get("allowed_live_write_kinds"))
    if tuple(sorted(allowed_kinds)) != tuple(sorted(CURRENT_PROVEN_LIVE_WRITE_KINDS)):
        violations.append("sequence_policy_memory_draft_result_consumer_draft_safety_allowed_kinds_mismatch")
    for flag in REQUIRED_MEMORY_DRAFT_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"sequence_policy_memory_draft_result_consumer_draft_safety_{flag}_not_false")
    if _record_list(memory_draft.get("blocked_results")):
        violations.append("sequence_policy_memory_draft_result_consumer_draft_has_blocked_results")
    return tuple(violations)


def _memory_draft_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
    records: Tuple[Mapping[str, Any], ...],
    blocked_results: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "input_policy_evidence_count": len(expected_ids),
        "memory_draft_count": len(records),
        "blocked_count": len(blocked_results),
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"sequence_policy_memory_draft_result_consumer_draft_summary_{key}_mismatch")
    record_ids = tuple(_draft_record_id(record, index) for index, record in enumerate(records))
    if _string_list(summary.get("memory_draft_proposal_ids")) != record_ids:
        violations.append("sequence_policy_memory_draft_result_consumer_draft_summary_record_ids_mismatch")
    blocked_ids = tuple(
        _string(blocked.get("proposal_id"))
        for blocked in blocked_results
        if _string(blocked.get("proposal_id"))
    )
    if _string_list(summary.get("blocked_proposal_ids")) != blocked_ids:
        violations.append("sequence_policy_memory_draft_result_consumer_draft_summary_blocked_ids_mismatch")
    return tuple(violations)


def _memory_draft_record_violations(
    record: Mapping[str, Any],
    *,
    index: int,
    prefix: str,
) -> Tuple[str, ...]:
    proposal_id = _draft_record_id(record, index)
    violations = []
    expected_idempotency = f"{prefix}:{proposal_id}"
    expected_draft_id = f"operator-sequence-policy-memory:{proposal_id}:{expected_idempotency}"
    if record.get("draft_id") != expected_draft_id:
        violations.append(f"{proposal_id}:draft_id_mismatch")
    if record.get("policy_evidence_index") != index:
        violations.append(f"{proposal_id}:policy_evidence_index_mismatch")
    if record.get("status") != "memory_draft_ready":
        violations.append(f"{proposal_id}:memory_draft_status_not_ready")
    if record.get("learning_scope") != REQUIRED_LEARNING_SCOPE:
        violations.append(f"{proposal_id}:learning_scope_mismatch")
    if record.get("target_memory_path") != REQUIRED_TARGET_MEMORY_PATH:
        violations.append(f"{proposal_id}:target_memory_path_mismatch")
    if record.get("target_section") != "Project Memory Lessons":
        violations.append(f"{proposal_id}:target_section_mismatch")
    if record.get("append_only") is not True:
        violations.append(f"{proposal_id}:append_only_not_true")
    if record.get("requires_separate_memory_writer") is not True:
        violations.append(f"{proposal_id}:requires_separate_memory_writer_not_true")
    if record.get("operator_ack_required_for_write") is not True:
        violations.append(f"{proposal_id}:operator_ack_required_for_write_not_true")
    for flag in REQUIRED_RECORD_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(f"{proposal_id}:{flag}_not_false")
    draft_text = record.get("draft_text")
    if not isinstance(draft_text, str) or not draft_text.strip():
        violations.append(f"{proposal_id}:draft_text_missing")
    evidence = _mapping(record.get("evidence"))
    value_type = _string(evidence.get("value_type"))
    if value_type not in ALLOWED_VALUE_TYPES:
        violations.append(f"{proposal_id}:evidence_value_type_not_memory_safe")
    effective_value = evidence.get("effective_value")
    if (
        not isinstance(effective_value, (int, float))
        or isinstance(effective_value, bool)
        or not isfinite(float(effective_value))
    ):
        violations.append(f"{proposal_id}:evidence_effective_value_invalid")
    elif value_type == "fader" and float(effective_value) > 0.0:
        violations.append(f"{proposal_id}:evidence_fader_ceiling_exceeded")
    if evidence.get("idempotency_key") != expected_idempotency:
        violations.append(f"{proposal_id}:evidence_idempotency_key_mismatch")
    if evidence.get("readback_status") not in VERIFIED_READBACK_STATUSES:
        violations.append(f"{proposal_id}:evidence_readback_not_verified")
    if evidence.get("proof_status") != "verified":
        violations.append(f"{proposal_id}:evidence_proof_status_not_verified")
    if evidence.get("rollback_available") is not True:
        violations.append(f"{proposal_id}:evidence_rollback_unavailable")
    if evidence.get("live_apply_entrypoint") != "apply_operator_proposal":
        violations.append(f"{proposal_id}:evidence_live_apply_entrypoint_mismatch")
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer(
    *,
    audit_policy_gate_result_consumer: Mapping[str, Any] | None = None,
    memory_draft_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify memory draft evidence and emit a memory-writer request template."""

    consumer = _mapping(audit_policy_gate_result_consumer)
    memory_draft = _mapping(memory_draft_result)
    consumer_summary = _mapping(consumer.get("summary"))
    expected_ids = _string_list(consumer_summary.get("policy_evidence_proposal_ids"))
    prefix = _string(consumer_summary.get("idempotency_key_prefix"))
    records = _record_list(memory_draft.get("memory_draft_records"))
    blocked_results = _record_list(memory_draft.get("blocked_results"))
    record_ids = tuple(_draft_record_id(record, index) for index, record in enumerate(records))
    record_draft_ids = tuple(_string(record.get("draft_id")) for record in records)
    row_violations = tuple(
        violation
        for index, record in enumerate(records)
        for violation in _memory_draft_record_violations(
            record,
            index=index,
            prefix=prefix,
        )
    )
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_consumer_envelope_violations(consumer),
                *_consumer_summary_violations(consumer_summary, expected_ids=expected_ids),
                *(
                    ("sequence_policy_memory_draft_result_consumer_policy_gate_consumer_status_not_available",)
                    if consumer.get("status")
                    != REQUIRED_AUDIT_POLICY_GATE_RESULT_CONSUMER_AVAILABLE_STATUS
                    else ()
                ),
                *_memory_draft_envelope_violations(memory_draft),
                *_memory_draft_summary_violations(
                    _mapping(memory_draft.get("summary")),
                    expected_ids=expected_ids,
                    records=records,
                    blocked_results=blocked_results,
                ),
                *(
                    f"duplicate_memory_draft_record_id:{value}"
                    for value in _duplicate_values(record_ids)
                ),
                *(
                    f"duplicate_memory_draft_id:{value}"
                    for value in _duplicate_values(record_draft_ids)
                ),
                *(
                    ("sequence_policy_memory_draft_result_consumer_record_id_outside_expected_set",)
                    if any(proposal_id not in set(expected_ids) for proposal_id in record_ids)
                    else ()
                ),
                *(
                    ("sequence_policy_memory_draft_result_consumer_record_ids_do_not_cover_expected_set",)
                    if not blocked_results and set(record_ids) != set(expected_ids)
                    else ()
                ),
                *(
                    ("sequence_policy_memory_draft_result_consumer_record_order_mismatch",)
                    if not blocked_results and record_ids != expected_ids
                    else ()
                ),
                *row_violations,
            )
        )
    )

    if not consumer or not memory_draft:
        status = "sequence_memory_trust_promotion_queue_import_memory_draft_result_consumer_not_ready"
        blocked_by = ("audit_policy_gate_result_consumer_and_memory_draft_result_required",)
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_memory_draft_result_consumer_replacement_required"
        blocked_by = ("operator_must_replace_invalid_sequence_policy_memory_draft_evidence",)
    elif memory_draft.get("status") != "ready" or blocked_results:
        status = "sequence_memory_trust_promotion_queue_import_memory_draft_result_consumer_blocked"
        blocked_by = ("all_sequence_policy_imported_memory_draft_records_must_be_ready",)
    elif not records:
        status = "sequence_memory_trust_promotion_queue_import_memory_draft_result_consumer_not_ready"
        blocked_by = ("memory_draft_ready_records_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_memory_writer_request_available"
        blocked_by = ("operator_may_run_generic_policy_memory_writer_dry_run",)

    request_available = (
        status == "sequence_memory_trust_promotion_queue_import_memory_writer_request_available"
    )
    selected_records = records if request_available else ()
    writer_specs = tuple(
        {
            "memory_writer_request_spec_id": (
                f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_DRAFT_RESULT_CONSUMER_METHOD_ID}:{proposal_id}"
            ),
            "proposal_id": proposal_id,
            "draft_id": record.get("draft_id"),
            "planned_memory_write_index": index,
            "source_memory_draft_index": index,
            "idempotency_key": f"{prefix}:{proposal_id}",
            "target_memory_path": REQUIRED_TARGET_MEMORY_PATH,
            "read_only": True,
            "memory_draft_result_consumer_only": True,
            "memory_writer_handler": "write_operator_proposal_apply_sequence_policy_memory",
            "memory_writer_handler_invoked": False,
            "memory_file_write_applied": False,
            "operator_must_set_apply_true": True,
            "operator_must_set_operator_ack_true": True,
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
        for index, (proposal_id, record) in enumerate(
            zip(record_ids if request_available else (), selected_records)
        )
    )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_DRAFT_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_AUDIT_POLICY_GATE_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "memory_draft_result_consumer_only": True,
            "memory_writer_request_template_only": True,
            "requires_audit_policy_gate_result_consumer": True,
            "requires_memory_draft_result": True,
            "requires_all_memory_draft_records_ready": True,
            "memory_writer_handler_invoked": False,
            "memory_file_write_applied": False,
            "memory_write_allowed_by_consumer": False,
            "operator_ack_required_before_write": True,
            "apply_required_before_write": True,
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
            "expected_memory_draft_count": len(expected_ids),
            "memory_draft_record_count": len(records),
            "ready_memory_draft_record_count": len(records),
            "blocked_memory_draft_record_count": len(blocked_results),
            "invalid_record_count": len(invalid_record_ids),
            "idempotency_key_prefix": prefix,
            "memory_draft_result_consumer_only": True,
            "memory_writer_request_available": request_available,
            "memory_writer_handler_invoked": False,
            "memory_file_write_applied": False,
            "memory_write_allowed_by_consumer": False,
            "operator_ack_required_before_write": True,
            "apply_required_before_write": True,
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
            "expected_memory_draft_proposal_ids": list(expected_ids),
            "memory_draft_proposal_ids": list(record_ids),
            "ready_memory_draft_proposal_ids": list(record_ids),
            "blocked_memory_draft_proposal_ids": [
                _string(blocked.get("proposal_id"))
                for blocked in blocked_results
                if _string(blocked.get("proposal_id"))
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide audit-policy-gate result consumer plus memory-draft result."
                if not consumer or not memory_draft
                else (
                    "Replace invalid memory-draft evidence before writer dry-run."
                    if invalid_record_ids
                    else (
                        "Re-run memory draft until every sequence-policy imported row is draft-ready."
                        if memory_draft.get("status") != "ready" or blocked_results
                        else "Operator may run the generic memory writer dry-run; this consumer does not run it."
                    )
                )
            ),
        },
        "memory_writer_request": {
            "handler": "write_operator_proposal_apply_sequence_policy_memory",
            "memory_draft_payload": {
                **dict(memory_draft),
                "memory_draft_records": list(selected_records),
            }
            if request_available
            else {},
            "apply": False,
            "operator_ack": False,
            "read_only_template": True,
            "operator_must_set_apply_true": True,
            "operator_must_set_operator_ack_true": True,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "memory_write_allowed_by_consumer": False,
            "memory_mutation": False,
            "memory_file_mutation": False,
            "execution_allowed_by_consumer": False,
            "policy_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
        },
        "memory_writer_request_specs": list(writer_specs),
    }
