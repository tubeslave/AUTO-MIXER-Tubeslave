"""Read-only consumer for sequence-policy queue-import memory-writer dry runs.

This module verifies that a generic policy-memory writer dry run still matches
sequence-policy queue-import provenance before a caller may prepare an explicit
append-only memory write request. It does not invoke writer handlers, write
memory files, reserve idempotency keys, or touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_DRAFT_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_DRAFT_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_DRAFT_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_DRAFT_RESULT_CONSUMER_STATUSES,
    )
try:
    from operator_mode_policy import CURRENT_PROVEN_LIVE_WRITE_KINDS
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_mode_policy import CURRENT_PROVEN_LIVE_WRITE_KINDS

ALLOWED_VALUE_TYPES = set(CURRENT_PROVEN_LIVE_WRITE_KINDS)


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_DRY_RUN_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_DRY_RUN_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_memory_writer_apply_request_available",
}
REQUIRED_MEMORY_DRAFT_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer"
)
REQUIRED_MEMORY_DRAFT_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_memory_writer_request_available"
)
REQUIRED_MEMORY_WRITER_TYPE = "operator_proposal_apply_sequence_policy_memory_writer"
REQUIRED_MEMORY_WRITER_DRY_RUN_STATUS = "append_only_memory_write_dry_run_ready"
REQUIRED_TARGET_MEMORY_PATH = ".ai/memory/project.md"
REQUIRED_DRY_RUN_BLOCKER = "operator_ack_and_apply_required_before_append_only_memory_write"
REQUIRED_CONSUMER_FALSE_FLAGS = (
    "memory_writer_handler_invoked",
    "memory_file_write_applied",
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
REQUIRED_WRITER_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "mutates_console_state",
    "queries_console",
    "policy_mutation",
    "memory_file_mutation",
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


def _draft_record_id(record: Mapping[str, Any], index: int) -> str:
    value = _string(record.get("proposal_id"))
    return value if value else f"memory_draft:{index}"


def _blocked_reasons(blocked_results: Tuple[Mapping[str, Any], ...]) -> Tuple[str, ...]:
    return tuple(_string(item.get("reason")) for item in blocked_results if _string(item.get("reason")))


def _consumer_envelope_violations(consumer: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_MEMORY_DRAFT_RESULT_CONSUMER_TYPE:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_draft_consumer_type_mismatch")
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_DRAFT_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append(
            "sequence_policy_memory_writer_dry_run_consumer_draft_consumer_method_id_mismatch"
        )
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_DRAFT_RESULT_CONSUMER_STATUSES
    ):
        violations.append(
            "sequence_policy_memory_writer_dry_run_consumer_draft_consumer_status_unknown"
        )
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_draft_consumer_read_only_not_true")
    if safety.get("memory_draft_result_consumer_only") is not True:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_draft_consumer_only_not_true")
    if safety.get("memory_writer_request_template_only") is not True:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_draft_consumer_template_only_not_true")
    if safety.get("operator_ack_required_before_write") is not True:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_draft_consumer_ack_required_not_true")
    if safety.get("apply_required_before_write") is not True:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_draft_consumer_apply_required_not_true")
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_writer_dry_run_consumer_draft_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if summary.get("memory_writer_request_available") is not bool(expected_ids):
        violations.append(
            "sequence_policy_memory_writer_dry_run_consumer_draft_consumer_request_available_mismatch"
        )
    expected_counts = {
        "expected_memory_draft_count": len(expected_ids),
        "memory_draft_record_count": len(expected_ids),
        "ready_memory_draft_record_count": len(expected_ids),
        "blocked_memory_draft_record_count": 0,
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_memory_writer_dry_run_consumer_draft_consumer_{key}_mismatch"
            )
    if summary.get("operator_ack_required_before_write") is not True:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_draft_consumer_summary_ack_required_not_true")
    if summary.get("apply_required_before_write") is not True:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_draft_consumer_summary_apply_required_not_true")
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_writer_dry_run_consumer_draft_consumer_summary_{flag}_not_false"
            )
    if _string_list(summary.get("memory_draft_proposal_ids")) != expected_ids:
        violations.append(
            "sequence_policy_memory_writer_dry_run_consumer_draft_consumer_record_ids_mismatch"
        )
    if _string_list(summary.get("ready_memory_draft_proposal_ids")) != expected_ids:
        violations.append(
            "sequence_policy_memory_writer_dry_run_consumer_draft_consumer_ready_record_ids_mismatch"
        )
    return tuple(violations)


def _consumer_request_violations(
    request: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if request.get("handler") != "write_operator_proposal_apply_sequence_policy_memory":
        violations.append("sequence_policy_memory_writer_dry_run_consumer_request_handler_mismatch")
    if request.get("apply") is not False:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_request_apply_not_false")
    if request.get("operator_ack") is not False:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_request_operator_ack_not_false")
    if request.get("operator_must_set_apply_true") is not True:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_request_apply_requirement_missing")
    if request.get("operator_must_set_operator_ack_true") is not True:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_request_ack_requirement_missing")
    for flag in (
        "live_write",
        "apply_attempted",
        "queue_mutation",
        "proposal_status_mutation",
        "idempotency_reserved",
        "memory_write_allowed_by_consumer",
        "memory_mutation",
        "memory_file_mutation",
        "execution_allowed_by_consumer",
        "policy_mutation",
        "runtime_memory_mutation",
        "model_mutation",
    ):
        if request.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_writer_dry_run_consumer_request_{flag}_not_false"
            )
    payload = _mapping(request.get("memory_draft_payload"))
    records = _record_list(payload.get("memory_draft_records"))
    record_ids = tuple(_draft_record_id(record, index) for index, record in enumerate(records))
    if payload.get("type") != "operator_proposal_apply_sequence_policy_memory_draft":
        violations.append("sequence_policy_memory_writer_dry_run_consumer_request_payload_type_mismatch")
    if payload.get("status") != "ready":
        violations.append("sequence_policy_memory_writer_dry_run_consumer_request_payload_not_ready")
    if record_ids != expected_ids:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_request_record_ids_mismatch")
    return tuple(violations)


def _writer_envelope_violations(writer: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if writer.get("type") != REQUIRED_MEMORY_WRITER_TYPE:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_type_mismatch")
    if writer.get("status") != REQUIRED_MEMORY_WRITER_DRY_RUN_STATUS:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_status_not_dry_run_ready")
    if writer.get("success") is not False:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_success_not_false")
    if writer.get("source_memory_draft_status") != "ready":
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_source_draft_not_ready")

    summary = _mapping(writer.get("summary"))
    if summary.get("append_only_memory_write_applied") is not False:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_summary_write_applied_not_false")
    if summary.get("operator_ack_required") is not True:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_summary_ack_required_not_true")
    if summary.get("operator_ack") is not False:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_summary_operator_ack_not_false")
    if summary.get("apply_requested") is not False:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_summary_apply_requested_not_false")
    if summary.get("target_memory_path") != REQUIRED_TARGET_MEMORY_PATH:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_summary_target_path_mismatch")
    for flag in REQUIRED_WRITER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_writer_dry_run_consumer_writer_summary_{flag}_not_false"
            )

    safety = _mapping(writer.get("safety"))
    if safety.get("operator_required") is not True:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_safety_operator_required_not_true")
    if safety.get("operator_ack_required") is not True:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_safety_ack_required_not_true")
    if safety.get("apply_requested") is not False:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_safety_apply_requested_not_false")
    if safety.get("operator_ack") is not False:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_safety_operator_ack_not_false")
    if safety.get("append_only_memory_file_write_applied") is not False:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_safety_write_applied_not_false")
    if safety.get("target_memory_path") != REQUIRED_TARGET_MEMORY_PATH:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_safety_target_path_mismatch")
    if safety.get("does_not_call_apply_operator_proposal") is not True:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_safety_apply_denial_missing")
    if safety.get("does_not_call_manual_console_write") is not True:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_safety_manual_write_denial_missing")
    allowed_kinds = _string_list(safety.get("allowed_live_write_kinds"))
    if tuple(sorted(allowed_kinds)) != tuple(sorted(CURRENT_PROVEN_LIVE_WRITE_KINDS)):
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_safety_allowed_kinds_mismatch")
    for flag in REQUIRED_WRITER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_writer_dry_run_consumer_writer_safety_{flag}_not_false"
            )
    if _record_list(writer.get("write_results")):
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_has_write_results")
    return tuple(violations)


def _writer_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
    blocked_reasons: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "memory_draft_record_count": len(expected_ids),
        "invalid_record_count": 1,
        "write_result_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_memory_writer_dry_run_consumer_writer_summary_{key}_mismatch"
            )
    if tuple(summary.get("blocked_by") or ()) != blocked_reasons:
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_summary_blocked_by_mismatch")
    if blocked_reasons != (REQUIRED_DRY_RUN_BLOCKER,):
        violations.append("sequence_policy_memory_writer_dry_run_consumer_writer_dry_run_blocker_mismatch")
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer(
    *,
    memory_draft_result_consumer: Mapping[str, Any] | None = None,
    memory_writer_dry_run_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify writer dry-run evidence and emit an explicit writer apply template."""

    consumer = _mapping(memory_draft_result_consumer)
    writer = _mapping(memory_writer_dry_run_result)
    consumer_summary = _mapping(consumer.get("summary"))
    expected_ids = _string_list(consumer_summary.get("memory_draft_proposal_ids"))
    request = _mapping(consumer.get("memory_writer_request"))
    request_payload = _mapping(request.get("memory_draft_payload"))
    request_records = _record_list(request_payload.get("memory_draft_records"))
    blocked_results = _record_list(writer.get("blocked_results"))
    blocked_reasons = _blocked_reasons(blocked_results)
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_consumer_envelope_violations(consumer),
                *_consumer_summary_violations(consumer_summary, expected_ids=expected_ids),
                *(
                    ("sequence_policy_memory_writer_dry_run_consumer_draft_consumer_status_not_available",)
                    if consumer.get("status")
                    != REQUIRED_MEMORY_DRAFT_RESULT_CONSUMER_AVAILABLE_STATUS
                    else ()
                ),
                *_consumer_request_violations(request, expected_ids=expected_ids),
                *_writer_envelope_violations(writer),
                *_writer_summary_violations(
                    _mapping(writer.get("summary")),
                    expected_ids=expected_ids,
                    blocked_reasons=blocked_reasons,
                ),
            )
        )
    )

    if not consumer or not writer:
        status = "sequence_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer_not_ready"
        blocked_by = ("memory_draft_result_consumer_and_memory_writer_dry_run_result_required",)
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer_replacement_required"
        blocked_by = ("operator_must_replace_invalid_sequence_policy_memory_writer_dry_run_evidence",)
    elif writer.get("status") != REQUIRED_MEMORY_WRITER_DRY_RUN_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer_blocked"
        blocked_by = ("generic_memory_writer_dry_run_must_be_ready_before_apply_template",)
    elif not expected_ids:
        status = "sequence_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer_not_ready"
        blocked_by = ("memory_writer_dry_run_ready_records_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_memory_writer_apply_request_available"
        blocked_by = ("operator_may_prepare_generic_policy_memory_writer_apply_request",)

    request_available = (
        status == "sequence_memory_trust_promotion_queue_import_memory_writer_apply_request_available"
    )
    apply_specs = tuple(
        {
            "memory_writer_apply_request_spec_id": (
                f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_DRY_RUN_RESULT_CONSUMER_METHOD_ID}:{proposal_id}"
            ),
            "proposal_id": proposal_id,
            "draft_id": record.get("draft_id"),
            "planned_memory_apply_index": index,
            "target_memory_path": REQUIRED_TARGET_MEMORY_PATH,
            "read_only": True,
            "memory_writer_dry_run_result_consumer_only": True,
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
            "memory_file_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
        }
        for index, (proposal_id, record) in enumerate(
            zip(expected_ids if request_available else (), request_records if request_available else ())
        )
    )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_DRY_RUN_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_DRAFT_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "memory_writer_dry_run_result_consumer_only": True,
            "memory_writer_apply_request_template_only": True,
            "requires_memory_draft_result_consumer": True,
            "requires_memory_writer_dry_run_result": True,
            "requires_operator_ack_before_write": True,
            "requires_apply_true_before_write": True,
            "memory_writer_handler_invoked": False,
            "memory_file_write_applied": False,
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
            "expected_memory_writer_dry_run_count": len(expected_ids),
            "memory_writer_dry_run_record_count": len(expected_ids),
            "invalid_record_count": len(invalid_record_ids),
            "dry_run_blocker_count": len(blocked_results),
            "target_memory_path": REQUIRED_TARGET_MEMORY_PATH,
            "memory_writer_dry_run_result_consumer_only": True,
            "memory_writer_apply_request_available": request_available,
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
            "memory_writer_dry_run_proposal_ids": list(expected_ids),
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide memory-draft result consumer plus memory-writer dry-run result."
                if not consumer or not writer
                else (
                    "Replace invalid memory-writer dry-run evidence before apply request."
                    if invalid_record_ids
                    else (
                        "Re-run generic memory writer as dry-run before apply request."
                        if writer.get("status") != REQUIRED_MEMORY_WRITER_DRY_RUN_STATUS
                        else "Operator may prepare the generic memory writer apply request; this consumer does not run it."
                    )
                )
            ),
        },
        "memory_writer_apply_request": {
            "handler": "write_operator_proposal_apply_sequence_policy_memory",
            "memory_draft_payload": {
                **dict(request_payload),
                "memory_draft_records": list(request_records),
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
        "memory_writer_apply_request_specs": list(apply_specs),
    }
