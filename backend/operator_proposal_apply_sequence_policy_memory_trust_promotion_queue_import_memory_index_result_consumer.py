"""Read-only consumer for sequence-policy queue-import memory-index results.

This module verifies that a read-only sequence policy memory index matches the
queue-import writer-apply provenance before a caller may build trust-review
candidates. It does not invoke review handlers, mutate policy, runtime memory,
models, queues, memory files, or mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_index import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_INDEX_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_INDEX_STATUSES,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_APPLY_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_APPLY_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_index import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_INDEX_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_INDEX_STATUSES,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_APPLY_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_APPLY_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_INDEX_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_memory_index_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_memory_index_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_memory_index_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_memory_trust_review_request_available",
}
REQUIRED_APPLY_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer"
)
REQUIRED_APPLY_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_memory_index_request_available"
)
REQUIRED_MEMORY_INDEX_TYPE = "operator_proposal_apply_sequence_policy_memory_index"
REQUIRED_MEMORY_INDEX_AVAILABLE_STATUS = "append_only_sequence_memory_index_available"
REQUIRED_TARGET_MEMORY_PATH = ".ai/memory/project.md"
REQUIRED_LEARNING_SCOPE = "supervised_fader_gain_sequence_execution"
ALLOWED_VALUE_TYPES = {"fader", "gain"}
VERIFIED_READBACK_STATUSES = {"verified_from_cache", "verified_from_physical_readback"}
REQUIRED_CONSUMER_FALSE_FLAGS = (
    "memory_index_handler_invoked",
    "memory_index_created",
    "memory_file_write_applied_by_consumer",
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
REQUIRED_INDEX_FALSE_SAFETY_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "policy_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_INDEX_FALSE_SUMMARY_FLAGS = (
    "execution_allowed_by_index",
    "queue_mutation",
    "policy_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_FRESHNESS_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "queries_console",
    "queue_mutation",
    "policy_mutation",
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


def _consumer_envelope_violations(consumer: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_APPLY_RESULT_CONSUMER_TYPE:
        violations.append("sequence_policy_memory_index_result_consumer_apply_consumer_type_mismatch")
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_APPLY_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append(
            "sequence_policy_memory_index_result_consumer_apply_consumer_method_id_mismatch"
        )
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_APPLY_RESULT_CONSUMER_STATUSES
    ):
        violations.append(
            "sequence_policy_memory_index_result_consumer_apply_consumer_status_unknown"
        )
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_policy_memory_index_result_consumer_apply_consumer_read_only_not_true")
    if safety.get("memory_writer_apply_result_consumer_only") is not True:
        violations.append("sequence_policy_memory_index_result_consumer_apply_consumer_only_not_true")
    if safety.get("memory_index_request_template_only") is not True:
        violations.append("sequence_policy_memory_index_result_consumer_apply_consumer_template_only_not_true")
    if safety.get("source_memory_file_write_applied") is not True:
        violations.append("sequence_policy_memory_index_result_consumer_apply_consumer_source_write_not_true")
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_index_result_consumer_apply_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "expected_memory_write_result_count": len(expected_ids),
        "memory_write_result_count": len(expected_ids),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_memory_index_result_consumer_apply_consumer_{key}_mismatch"
            )
    if summary.get("memory_index_request_available") is not bool(expected_ids):
        violations.append("sequence_policy_memory_index_result_consumer_apply_consumer_request_available_mismatch")
    if summary.get("source_memory_file_write_applied") is not True:
        violations.append("sequence_policy_memory_index_result_consumer_apply_consumer_summary_source_write_not_true")
    if _string_list(summary.get("memory_write_result_proposal_ids")) != expected_ids:
        violations.append("sequence_policy_memory_index_result_consumer_apply_consumer_proposal_ids_mismatch")
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_index_result_consumer_apply_consumer_summary_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_request_violations(request: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if request.get("handler") != "index_operator_proposal_apply_sequence_policy_memory":
        violations.append("sequence_policy_memory_index_result_consumer_request_handler_mismatch")
    if request.get("target_memory_path") != REQUIRED_TARGET_MEMORY_PATH:
        violations.append("sequence_policy_memory_index_result_consumer_request_target_path_mismatch")
    if request.get("read_only_template") is not True:
        violations.append("sequence_policy_memory_index_result_consumer_request_read_only_template_not_true")
    if request.get("source_memory_file_write_applied") is not True:
        violations.append("sequence_policy_memory_index_result_consumer_request_source_write_not_true")
    for flag in (
        "live_write",
        "apply_attempted",
        "queue_mutation",
        "proposal_status_mutation",
        "idempotency_reserved",
        "manual_console_write_called",
        "memory_mutation",
        "memory_file_mutation",
        "execution_allowed_by_consumer",
        "policy_mutation",
        "runtime_memory_mutation",
        "model_mutation",
        "live_authority_granted",
        "full_autonomous_ready",
        "can_apply_live",
    ):
        if request.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_index_result_consumer_request_{flag}_not_false"
            )
    return tuple(violations)


def _index_envelope_violations(index_result: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if index_result.get("type") != REQUIRED_MEMORY_INDEX_TYPE:
        violations.append("sequence_policy_memory_index_result_consumer_index_type_mismatch")
    if index_result.get("method_id") != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_INDEX_METHOD_ID:
        violations.append("sequence_policy_memory_index_result_consumer_index_method_id_mismatch")
    if index_result.get("source_method_id") != "write_operator_proposal_apply_sequence_policy_memory":
        violations.append("sequence_policy_memory_index_result_consumer_index_source_method_mismatch")
    if index_result.get("status") not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_INDEX_STATUSES:
        violations.append("sequence_policy_memory_index_result_consumer_index_status_unknown")
    safety = _mapping(index_result.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_policy_memory_index_result_consumer_index_read_only_not_true")
    if safety.get("memory_file_read") is not True:
        violations.append("sequence_policy_memory_index_result_consumer_index_memory_file_read_not_true")
    for flag in REQUIRED_INDEX_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_index_result_consumer_index_safety_{flag}_not_false"
            )
    return tuple(violations)


def _index_summary_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
    expected_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if summary.get("target_memory_path") != REQUIRED_TARGET_MEMORY_PATH:
        violations.append("sequence_policy_memory_index_result_consumer_index_target_path_mismatch")
    expected_counts = {
        "memory_record_count": len(records),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_memory_index_result_consumer_index_summary_{key}_mismatch"
            )
    if _string_list(summary.get("indexed_proposal_ids")) != expected_ids:
        violations.append("sequence_policy_memory_index_result_consumer_index_proposal_ids_mismatch")
    if _string_list(summary.get("indexed_learning_scopes")) != tuple(
        REQUIRED_LEARNING_SCOPE for _ in records
    ):
        violations.append("sequence_policy_memory_index_result_consumer_index_learning_scopes_mismatch")
    for flag in REQUIRED_INDEX_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_index_result_consumer_index_summary_{flag}_not_false"
            )
    return tuple(violations)


def _record_violations(
    record: Mapping[str, Any],
    *,
    expected_proposal_id: str,
    expected_draft_id: str,
    index: int,
) -> Tuple[str, ...]:
    record_id = _string(record.get("memory_record_id")) or f"memory_index_record:{index}"
    violations = []
    if record.get("proposal_id") != expected_proposal_id:
        violations.append(f"{record_id}:proposal_id_mismatch")
    if record.get("draft_id") != expected_draft_id:
        violations.append(f"{record_id}:draft_id_mismatch")
    if record.get("target_memory_path") != REQUIRED_TARGET_MEMORY_PATH:
        violations.append(f"{record_id}:target_memory_path_mismatch")
    if record.get("learning_scope") != REQUIRED_LEARNING_SCOPE:
        violations.append(f"{record_id}:learning_scope_mismatch")
    value_type = str(record.get("value_type") or "").strip().lower()
    if value_type not in ALLOWED_VALUE_TYPES:
        violations.append(f"{record_id}:value_type_not_allowed")
    channel = record.get("channel")
    if isinstance(channel, bool) or not isinstance(channel, int) or channel < 1:
        violations.append(f"{record_id}:channel_invalid")
    effective_value = record.get("effective_value")
    if isinstance(effective_value, bool) or not isinstance(effective_value, (int, float)):
        violations.append(f"{record_id}:effective_value_invalid")
    elif value_type == "fader" and float(effective_value) > 0.0:
        violations.append(f"{record_id}:fader_ceiling_exceeded")
    if record.get("readback_status") not in VERIFIED_READBACK_STATUSES:
        violations.append(f"{record_id}:readback_not_verified")
    if record.get("proof_status") != "verified":
        violations.append(f"{record_id}:proof_not_verified")
    if record.get("runtime_authority") != "none":
        violations.append(f"{record_id}:runtime_authority_not_none")
    if record.get("usable_for_sequence_policy_review") is not True:
        violations.append(f"{record_id}:usable_for_sequence_policy_review_not_true")
    for flag in (
        "execution_allowed_by_index",
        "queue_mutation_allowed_by_index",
        "policy_mutation_allowed_by_index",
        "model_mutation_allowed_by_index",
        "runtime_memory_mutation_allowed_by_index",
        "live_authority_granted",
        "full_autonomous_ready",
    ):
        if record.get(flag) is not False:
            violations.append(f"{record_id}:{flag}_not_false")
    return tuple(violations)


def _freshness_violations(freshness: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if freshness.get("status") != "fresh":
        violations.append("sequence_policy_memory_index_result_consumer_freshness_not_fresh")
    if freshness.get("read_only") is not True:
        violations.append("sequence_policy_memory_index_result_consumer_freshness_read_only_not_true")
    for flag in REQUIRED_FRESHNESS_FALSE_FLAGS:
        if freshness.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_index_result_consumer_freshness_{flag}_not_false"
            )
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer(
    *,
    memory_writer_apply_result_consumer: Mapping[str, Any] | None = None,
    memory_index_result: Mapping[str, Any] | None = None,
    current_sequence_review_freshness: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify memory-index evidence and emit a read-only trust-review request."""

    consumer = _mapping(memory_writer_apply_result_consumer)
    index_result = _mapping(memory_index_result)
    freshness = _mapping(current_sequence_review_freshness)
    consumer_summary = _mapping(consumer.get("summary"))
    expected_ids = _string_list(consumer_summary.get("memory_write_result_proposal_ids"))
    specs = _record_list(consumer.get("memory_index_request_specs"))
    expected_draft_ids = tuple(_string(spec.get("draft_id")) for spec in specs)
    records = _record_list(index_result.get("memory_records"))
    record_violations = tuple(
        violation
        for index, (record, proposal_id, draft_id) in enumerate(
            zip(records, expected_ids, expected_draft_ids)
        )
        for violation in _record_violations(
            record,
            expected_proposal_id=proposal_id,
            expected_draft_id=draft_id,
            index=index,
        )
    )
    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_consumer_envelope_violations(consumer),
                *_consumer_summary_violations(consumer_summary, expected_ids=expected_ids),
                *(
                    ("sequence_policy_memory_index_result_consumer_apply_consumer_status_not_available",)
                    if consumer.get("status") != REQUIRED_APPLY_RESULT_CONSUMER_AVAILABLE_STATUS
                    else ()
                ),
                *_consumer_request_violations(_mapping(consumer.get("memory_index_request"))),
                *_index_envelope_violations(index_result),
                *_index_summary_violations(
                    _mapping(index_result.get("summary")),
                    records=records,
                    expected_ids=expected_ids,
                ),
                *(
                    ("sequence_policy_memory_index_result_consumer_index_status_not_available",)
                    if index_result.get("status") != REQUIRED_MEMORY_INDEX_AVAILABLE_STATUS
                    else ()
                ),
                *(
                    ("sequence_policy_memory_index_result_consumer_record_count_mismatch",)
                    if len(records) != len(expected_ids)
                    else ()
                ),
                *(
                    ("sequence_policy_memory_index_result_consumer_draft_id_count_mismatch",)
                    if len(expected_draft_ids) != len(expected_ids)
                    else ()
                ),
                *record_violations,
                *_freshness_violations(freshness),
            )
        )
    )
    invalid_record_ids = (
        ()
        if not consumer or not index_result
        else raw_invalid_record_ids
    )

    if not consumer or not index_result:
        status = "sequence_memory_trust_promotion_queue_import_memory_index_result_consumer_not_ready"
        blocked_by = ("memory_writer_apply_result_consumer_and_memory_index_result_required",)
    elif not freshness:
        status = "sequence_memory_trust_promotion_queue_import_memory_index_result_consumer_not_ready"
        blocked_by = ("fresh_sequence_review_context_required_before_trust_review_request",)
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_memory_index_result_consumer_replacement_required"
        blocked_by = ("operator_must_replace_invalid_sequence_policy_memory_index_evidence",)
    elif index_result.get("status") != REQUIRED_MEMORY_INDEX_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_memory_index_result_consumer_blocked"
        blocked_by = ("append_only_sequence_memory_index_must_be_available_before_trust_review",)
    elif not records:
        status = "sequence_memory_trust_promotion_queue_import_memory_index_result_consumer_not_ready"
        blocked_by = ("indexed_sequence_policy_memory_records_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_memory_trust_review_request_available"
        blocked_by = ("operator_may_run_read_only_sequence_memory_trust_review",)

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_memory_trust_review_request_available"
    )
    review_specs = tuple(
        {
            "memory_trust_review_request_spec_id": (
                f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID}:{proposal_id}"
            ),
            "proposal_id": proposal_id,
            "draft_id": draft_id,
            "memory_record_id": record.get("memory_record_id"),
            "planned_trust_review_index": index,
            "target_memory_path": REQUIRED_TARGET_MEMORY_PATH,
            "read_only": True,
            "memory_index_result_consumer_only": True,
            "memory_trust_review_request_template_only": True,
            "memory_trust_review_handler": "review_operator_proposal_apply_sequence_policy_memory_trust",
            "memory_trust_review_handler_invoked": False,
            "memory_trust_review_created": False,
            "source_memory_index_available": True,
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
        for index, (record, proposal_id, draft_id) in enumerate(
            zip(
                records if request_available else (),
                expected_ids if request_available else (),
                expected_draft_ids if request_available else (),
            )
        )
    )

    trust_review_request: Dict[str, Any] = {
        "handler": "review_operator_proposal_apply_sequence_policy_memory_trust",
        "read_only_template": True,
        "memory_index_result_consumer_only": True,
        "memory_trust_review_request_template_only": True,
        "memory_trust_review_handler_invoked": False,
        "memory_trust_review_created": False,
        "source_memory_index_available": request_available,
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
        trust_review_request["memory_index"] = index_result
        trust_review_request["current_sequence_review_freshness"] = freshness

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_APPLY_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "memory_index_result_consumer_only": True,
            "memory_trust_review_request_template_only": True,
            "requires_memory_writer_apply_result_consumer": True,
            "requires_memory_index_result": True,
            "requires_fresh_sequence_review_context": True,
            "memory_trust_review_handler_invoked": False,
            "memory_trust_review_created": False,
            "source_memory_index_available": request_available,
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
            "expected_memory_index_record_count": len(expected_ids),
            "memory_index_record_count": len(records),
            "invalid_record_count": len(invalid_record_ids),
            "target_memory_path": REQUIRED_TARGET_MEMORY_PATH,
            "memory_index_result_consumer_only": True,
            "memory_trust_review_request_template_only": True,
            "memory_trust_review_request_available": request_available,
            "memory_trust_review_handler_invoked": False,
            "memory_trust_review_created": False,
            "source_memory_index_available": request_available,
            "fresh_sequence_review_context_status": (
                freshness.get("status") if freshness else "missing"
            ),
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
            "memory_index_record_proposal_ids": list(expected_ids),
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide writer-apply consumer, memory index result, and fresh review context."
                if not consumer or not index_result or not freshness
                else (
                    "Replace invalid memory-index evidence before trust review."
                    if invalid_record_ids
                    else (
                        "Build append-only sequence memory index before trust review."
                        if index_result.get("status") != REQUIRED_MEMORY_INDEX_AVAILABLE_STATUS
                        else "Operator may run the read-only sequence memory trust review; this consumer does not run it."
                    )
                )
            ),
        },
        "memory_trust_review_request": trust_review_request,
        "memory_trust_review_request_specs": list(review_specs),
    }
