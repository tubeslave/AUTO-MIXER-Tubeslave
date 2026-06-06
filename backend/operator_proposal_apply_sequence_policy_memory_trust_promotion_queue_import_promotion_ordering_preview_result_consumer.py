"""Read-only consumer for promotion ordering-preview results.

This verifier connects the queue-import promotion policy-consumer result
consumer to the existing queue-ordering plan builder. It verifies ordering
preview rows against the previously verified policy-consumer provenance and
emits only a queue-ordering-plan request template. It does not reorder
proposals, import queues, reserve idempotency, mutate policy/runtime/model or
memory state, or touch mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_STATUSES,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_POLICY_CONSUMER_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_POLICY_CONSUMER_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_STATUSES,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_POLICY_CONSUMER_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_POLICY_CONSUMER_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ORDERING_PREVIEW_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ORDERING_PREVIEW_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_request_available",
}
REQUIRED_POLICY_CONSUMER_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer"
)
REQUIRED_POLICY_CONSUMER_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_request_available"
)
REQUIRED_ORDERING_PREVIEW_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview"
)
REQUIRED_ORDERING_PREVIEW_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_ordering_preview_available"
)
REQUIRED_POLICY_CONSUMER_RESULT_CONSUMER_FALSE_FLAGS = (
    "promotion_ordering_preview_handler_invoked",
    "promotion_ordering_preview_created",
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "proposal_order_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "policy_mutation",
    "memory_mutation",
    "memory_file_mutation",
    "runtime_memory_mutation",
    "model_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
)
REQUIRED_ORDERING_PREVIEW_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "proposal_order_mutation",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_ordering_preview",
    "sequence_policy_change_allowed_by_ordering_preview",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
)
REQUIRED_ORDERING_PREVIEW_SUMMARY_FALSE_FLAGS = tuple(
    flag for flag in REQUIRED_ORDERING_PREVIEW_FALSE_FLAGS if flag != "apply_attempted"
)
REQUIRED_ORDERING_PREVIEW_ROW_FALSE_FLAGS = REQUIRED_ORDERING_PREVIEW_SUMMARY_FALSE_FLAGS
REQUIRED_REQUEST_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "proposal_order_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "policy_mutation",
    "memory_mutation",
    "memory_file_mutation",
    "runtime_memory_mutation",
    "model_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
)
ALLOWED_RANK_SOURCES = {"sequence_policy_consumer", "original_order"}


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


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in {float("inf"), float("-inf")}:
        return None
    return numeric


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


def _policy_consumer_from_result_consumer(
    consumer: Mapping[str, Any],
) -> Mapping[str, Any]:
    request = _mapping(consumer.get("promotion_ordering_preview_request"))
    return _mapping(request.get("policy_consumer"))


def _policy_records_by_consumer_id(
    records: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(record["consumer_record_id"]): record
        for record in records
        if isinstance(record.get("consumer_record_id"), str)
        and record.get("consumer_record_id")
    }


def _ordering_row_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("ordering_preview_record_id"))
    if explicit:
        return explicit
    proposal_id = _string(record.get("proposal_id"))
    return f"{proposal_id if proposal_id else 'unknown_ordering_preview_record'}:{index}"


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


def _policy_result_consumer_envelope_violations(
    consumer: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_POLICY_CONSUMER_RESULT_CONSUMER_TYPE:
        violations.append("ordering_preview_result_consumer_prior_type_mismatch")
    generated_at = consumer.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("ordering_preview_result_consumer_prior_generated_at_invalid")
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_POLICY_CONSUMER_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append("ordering_preview_result_consumer_prior_method_id_mismatch")
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_POLICY_CONSUMER_RESULT_CONSUMER_STATUSES
    ):
        violations.append("ordering_preview_result_consumer_prior_status_unknown")
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("ordering_preview_result_consumer_prior_read_only_not_true")
    if safety.get("promotion_policy_consumer_result_consumer_only") is not True:
        violations.append("ordering_preview_result_consumer_prior_only_not_true")
    if safety.get("promotion_ordering_preview_request_template_only") is not True:
        violations.append("ordering_preview_result_consumer_prior_template_only_not_true")
    for flag in REQUIRED_POLICY_CONSUMER_RESULT_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"ordering_preview_result_consumer_prior_{flag}_not_false")
    return tuple(violations)


def _policy_result_consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    policy_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "expected_policy_consumer_record_count": len(policy_records),
        "policy_consumer_record_count": len(policy_records),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"ordering_preview_result_consumer_prior_summary_{key}_mismatch")
    if summary.get("promotion_ordering_preview_request_available") is not True:
        violations.append(
            "ordering_preview_result_consumer_prior_summary_request_available_not_true"
        )
    if summary.get("promotion_policy_consumer_result_consumer_only") is not True:
        violations.append("ordering_preview_result_consumer_prior_summary_only_not_true")
    if summary.get("promotion_ordering_preview_request_template_only") is not True:
        violations.append(
            "ordering_preview_result_consumer_prior_summary_template_only_not_true"
        )
    for flag in REQUIRED_POLICY_CONSUMER_RESULT_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"ordering_preview_result_consumer_prior_summary_{flag}_not_false"
            )
    expected_proposal_ids = tuple(
        record["proposal_id"]
        for record in policy_records
        if _string(record.get("proposal_id"))
    )
    expected_policy_keys = tuple(
        record["policy_key"]
        for record in policy_records
        if _string(record.get("policy_key"))
    )
    if _string_list(summary.get("policy_consumer_proposal_ids")) != expected_proposal_ids:
        violations.append(
            "ordering_preview_result_consumer_prior_summary_proposal_ids_mismatch"
        )
    if _string_list(summary.get("policy_keys")) != expected_policy_keys:
        violations.append("ordering_preview_result_consumer_prior_summary_policy_keys_mismatch")
    return tuple(violations)


def _policy_result_consumer_request_violations(
    request: Mapping[str, Any],
    *,
    policy_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    if (
        request.get("handler")
        != "preview_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering"
    ):
        violations.append("ordering_preview_result_consumer_prior_request_handler_mismatch")
    if request.get("read_only_template") is not True:
        violations.append("ordering_preview_result_consumer_prior_request_read_only_not_true")
    if request.get("promotion_policy_consumer_result_consumer_only") is not True:
        violations.append("ordering_preview_result_consumer_prior_request_only_not_true")
    if request.get("promotion_ordering_preview_request_template_only") is not True:
        violations.append("ordering_preview_result_consumer_prior_request_template_not_true")
    if request.get("promotion_ordering_preview_handler_invoked") is not False:
        violations.append("ordering_preview_result_consumer_prior_request_handler_invoked")
    if request.get("promotion_ordering_preview_created") is not False:
        violations.append("ordering_preview_result_consumer_prior_request_created")
    if request.get("proposals_required_from_operator") is not True:
        violations.append("ordering_preview_result_consumer_prior_request_proposals_not_required")
    for flag in REQUIRED_REQUEST_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(f"ordering_preview_result_consumer_prior_request_{flag}_not_false")
    if not policy_records:
        violations.append("ordering_preview_result_consumer_prior_request_policy_records_missing")
    return tuple(violations)


def _ordering_preview_envelope_violations(
    ordering_preview: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    if ordering_preview.get("type") != REQUIRED_ORDERING_PREVIEW_TYPE:
        violations.append("ordering_preview_result_consumer_ordering_preview_type_mismatch")
    generated_at = ordering_preview.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("ordering_preview_result_consumer_ordering_preview_generated_at_invalid")
    if (
        ordering_preview.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_METHOD_ID
    ):
        violations.append("ordering_preview_result_consumer_ordering_preview_method_id_mismatch")
    if (
        ordering_preview.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_STATUSES
    ):
        violations.append("ordering_preview_result_consumer_ordering_preview_status_unknown")
    if (
        ordering_preview.get("source_method_id")
        != "operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer"
    ):
        violations.append("ordering_preview_result_consumer_ordering_preview_source_mismatch")
    safety = _mapping(ordering_preview.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("ordering_preview_result_consumer_ordering_preview_read_only_not_true")
    if safety.get("ordering_preview_only") is not True:
        violations.append("ordering_preview_result_consumer_ordering_preview_only_not_true")
    if safety.get("operator_required") is not True:
        violations.append("ordering_preview_result_consumer_ordering_preview_operator_required_not_true")
    for flag in REQUIRED_ORDERING_PREVIEW_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"ordering_preview_result_consumer_ordering_preview_{flag}_not_false"
            )
    return tuple(violations)


def _ordering_preview_summary_violations(
    summary: Mapping[str, Any],
    *,
    policy_records: Tuple[Mapping[str, Any], ...],
    ordered_rows: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    ranked_rows = tuple(
        row for row in ordered_rows if row.get("rank_source") == "sequence_policy_consumer"
    )
    unranked_rows = tuple(
        row for row in ordered_rows if row.get("rank_source") == "original_order"
    )
    violations = []
    expected_counts = {
        "input_proposal_count": len(ordered_rows),
        "consumer_record_count": len(policy_records),
        "ordered_preview_count": len(ordered_rows),
        "policy_ranked_proposal_count": len(ranked_rows),
        "unranked_proposal_count": len(unranked_rows),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"ordering_preview_result_consumer_ordering_preview_summary_{key}_mismatch"
            )
    if summary.get("recommended_order_available") is not bool(ordered_rows):
        violations.append(
            "ordering_preview_result_consumer_ordering_preview_summary_recommended_order_mismatch"
        )
    if summary.get("ordering_preview_only") is not True:
        violations.append(
            "ordering_preview_result_consumer_ordering_preview_summary_only_not_true"
        )
    for flag in REQUIRED_ORDERING_PREVIEW_SUMMARY_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"ordering_preview_result_consumer_ordering_preview_summary_{flag}_not_false"
            )
    expected_order = tuple(
        row["proposal_id"] for row in ordered_rows if _string(row.get("proposal_id"))
    )
    expected_ranked = tuple(
        row["proposal_id"] for row in ranked_rows if _string(row.get("proposal_id"))
    )
    expected_unranked = tuple(
        row["proposal_id"] for row in unranked_rows if _string(row.get("proposal_id"))
    )
    if _string_list(summary.get("recommended_ordered_proposal_ids")) != expected_order:
        violations.append(
            "ordering_preview_result_consumer_ordering_preview_summary_ordered_ids_mismatch"
        )
    if _string_list(summary.get("policy_ranked_proposal_ids")) != expected_ranked:
        violations.append(
            "ordering_preview_result_consumer_ordering_preview_summary_ranked_ids_mismatch"
        )
    if _string_list(summary.get("unranked_proposal_ids")) != expected_unranked:
        violations.append(
            "ordering_preview_result_consumer_ordering_preview_summary_unranked_ids_mismatch"
        )
    return tuple(violations)


def _ordering_row_violations(
    row: Mapping[str, Any],
    *,
    policy_records_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[str, ...]:
    violations = []
    required_fields = (
        "ordering_preview_record_id",
        "proposal_id",
        "recommended_sequence_policy_bucket",
        "rank_source",
    )
    for field in required_fields:
        if not _string(row.get(field)):
            violations.append(f"ordering_preview_result_consumer_row_{field}_missing")
    rank_source = row.get("rank_source")
    if rank_source not in ALLOWED_RANK_SOURCES:
        violations.append("ordering_preview_result_consumer_row_rank_source_unknown")
    for field in ("original_index", "recommended_index"):
        if _summary_int(row.get(field)) is None:
            violations.append(f"ordering_preview_result_consumer_row_{field}_invalid")
    for field in ("base_policy_score", "adjusted_policy_score"):
        score = _safe_float(row.get(field))
        if score is None or score < 0.0 or score > 1.0:
            violations.append(f"ordering_preview_result_consumer_row_{field}_invalid")
    if row.get("read_only") is not True:
        violations.append("ordering_preview_result_consumer_row_read_only_not_true")
    if row.get("ordering_preview_only") is not True:
        violations.append("ordering_preview_result_consumer_row_preview_only_not_true")
    if row.get("policy_consumer_required_before_queue_mutation") is not True:
        violations.append("ordering_preview_result_consumer_row_policy_consumer_required_not_true")
    if row.get("sequence_consumer_required_before_execution") is not True:
        violations.append("ordering_preview_result_consumer_row_sequence_consumer_required_not_true")
    for flag in REQUIRED_ORDERING_PREVIEW_ROW_FALSE_FLAGS:
        if row.get(flag) is not False:
            violations.append(f"ordering_preview_result_consumer_row_{flag}_not_false")

    if rank_source == "sequence_policy_consumer":
        consumer_record_id = _string(row.get("consumer_record_id"))
        policy_record = policy_records_by_id.get(consumer_record_id)
        if not policy_record:
            violations.append("ordering_preview_result_consumer_row_policy_record_missing")
            return tuple(violations)
        expected_pairs = {
            "proposal_id": policy_record.get("proposal_id"),
            "original_index": policy_record.get("proposal_original_index"),
            "consumer_record_id": policy_record.get("consumer_record_id"),
            "sequence_policy_gate_record_id": policy_record.get(
                "sequence_policy_gate_record_id"
            ),
            "policy_key": policy_record.get("policy_key"),
            "recommended_sequence_policy_bucket": policy_record.get(
                "recommended_sequence_policy_bucket"
            ),
            "base_policy_score": policy_record.get("base_policy_score"),
            "adjusted_policy_score": policy_record.get("adjusted_policy_score"),
        }
        for field, expected in expected_pairs.items():
            if row.get(field) != expected:
                violations.append(f"ordering_preview_result_consumer_row_{field}_mismatch")
    elif rank_source == "original_order":
        if row.get("consumer_record_id") is not None:
            violations.append("ordering_preview_result_consumer_row_unranked_consumer_id_present")
        if row.get("sequence_policy_gate_record_id") is not None:
            violations.append("ordering_preview_result_consumer_row_unranked_gate_id_present")
        if row.get("policy_key") is not None:
            violations.append("ordering_preview_result_consumer_row_unranked_policy_key_present")
        if row.get("recommended_sequence_policy_bucket") != "unranked_original_order":
            violations.append("ordering_preview_result_consumer_row_unranked_bucket_mismatch")
        if row.get("base_policy_score") != 0.0 or row.get("adjusted_policy_score") != 0.0:
            violations.append("ordering_preview_result_consumer_row_unranked_score_mismatch")
    return tuple(violations)


def _invalid_ordering_row_ids(
    rows: Tuple[Mapping[str, Any], ...],
    *,
    policy_records_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[str, ...]:
    return tuple(
        _ordering_row_id(row, index)
        for index, row in enumerate(rows)
        if _ordering_row_violations(row, policy_records_by_id=policy_records_by_id)
    )


def _sequence_violations(rows: Tuple[Mapping[str, Any], ...]) -> Tuple[str, ...]:
    recommended_indexes = tuple(
        row.get("recommended_index")
        for row in rows
        if _summary_int(row.get("recommended_index")) is not None
    )
    if recommended_indexes != tuple(range(len(rows))):
        return ("ordering_preview_result_consumer_recommended_index_sequence_mismatch",)
    return ()


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer(
    *,
    promotion_policy_consumer_result_consumer: Mapping[str, Any] | None = None,
    promotion_ordering_preview_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify ordering-preview evidence and emit a queue-ordering-plan request."""

    policy_result_consumer = _mapping(promotion_policy_consumer_result_consumer)
    ordering_preview = _mapping(promotion_ordering_preview_result)
    prior_request = _mapping(
        policy_result_consumer.get("promotion_ordering_preview_request")
    )
    policy_consumer = _policy_consumer_from_result_consumer(policy_result_consumer)
    policy_records = _record_list(policy_consumer.get("consumer_records"))
    ordered_rows = _record_list(ordering_preview.get("ordered_proposals"))
    ranked_rows = tuple(
        row for row in ordered_rows if row.get("rank_source") == "sequence_policy_consumer"
    )
    unranked_rows = tuple(
        row for row in ordered_rows if row.get("rank_source") == "original_order"
    )
    policy_records_by_id = _policy_records_by_consumer_id(policy_records)
    prior_status = policy_result_consumer.get("status")
    ordering_status = ordering_preview.get("status")

    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_policy_result_consumer_envelope_violations(policy_result_consumer),
                *(
                    (
                        *_policy_result_consumer_summary_violations(
                            _mapping(policy_result_consumer.get("summary")),
                            policy_records=policy_records,
                        ),
                        *_policy_result_consumer_request_violations(
                            prior_request,
                            policy_records=policy_records,
                        ),
                    )
                    if prior_status
                    == REQUIRED_POLICY_CONSUMER_RESULT_CONSUMER_AVAILABLE_STATUS
                    else ()
                ),
                *_ordering_preview_envelope_violations(ordering_preview),
                *(
                    (
                        "ordering_preview_result_consumer_prior_replacement_required",
                    )
                    if prior_status
                    == "sequence_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer_replacement_required"
                    else ()
                ),
                *(
                    (
                        "ordering_preview_result_consumer_ordering_preview_replacement_required",
                    )
                    if ordering_status
                    == "sequence_memory_trust_promotion_ordering_preview_replacement_required"
                    else ()
                ),
                *(
                    (
                        *_ordering_preview_summary_violations(
                            _mapping(ordering_preview.get("summary")),
                            policy_records=policy_records,
                            ordered_rows=ordered_rows,
                        ),
                        *_invalid_ordering_row_ids(
                            ordered_rows,
                            policy_records_by_id=policy_records_by_id,
                        ),
                        *_sequence_violations(ordered_rows),
                        *_duplicate_ids(
                            ordered_rows,
                            "ordering_preview_record_id",
                            id_func=_ordering_row_id,
                        ),
                        *_duplicate_ids(
                            ordered_rows,
                            "proposal_id",
                            id_func=_ordering_row_id,
                        ),
                        *_duplicate_ids(
                            ordered_rows,
                            "recommended_index",
                            id_func=_ordering_row_id,
                        ),
                    )
                    if ordering_status == REQUIRED_ORDERING_PREVIEW_AVAILABLE_STATUS
                    or ordered_rows
                    else ()
                ),
            )
        )
    )
    if not policy_result_consumer or not ordering_preview:
        invalid_record_ids: Tuple[str, ...] = ()
    else:
        invalid_record_ids = raw_invalid_record_ids

    if not policy_result_consumer or not ordering_preview:
        status = "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer_not_ready"
        blocked_by = (
            "promotion_policy_consumer_result_consumer_and_ordering_preview_required",
        )
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_memory_trust_promotion_ordering_preview_evidence",
        )
    elif prior_status != REQUIRED_POLICY_CONSUMER_RESULT_CONSUMER_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer_blocked"
        blocked_by = (
            "promotion_policy_consumer_result_consumer_must_request_ordering_preview_first",
        )
    elif ordering_status != REQUIRED_ORDERING_PREVIEW_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer_blocked"
        blocked_by = (
            "sequence_memory_trust_promotion_ordering_preview_required_before_queue_ordering_plan",
        )
    elif not ordered_rows:
        status = "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer_not_ready"
        blocked_by = ("sequence_memory_trust_promotion_ordering_preview_records_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_request_available"
        blocked_by = (
            "operator_may_run_read_only_sequence_memory_trust_promotion_queue_ordering_plan",
        )

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_request_available"
    )
    queue_ordering_plan_request: Dict[str, Any] = {
        "handler": (
            "plan_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering"
        ),
        "read_only_template": True,
        "promotion_ordering_preview_result_consumer_only": True,
        "promotion_queue_ordering_plan_request_template_only": True,
        "promotion_queue_ordering_plan_handler_invoked": False,
        "promotion_queue_ordering_plan_created": False,
        "ordering_preview_required": True,
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
        "sequence_policy_trust_store_mutation": False,
        "proposal_state_mutation": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "can_import_to_queue": False,
        "can_apply_live": False,
    }
    if request_available:
        queue_ordering_plan_request["ordering_preview"] = ordering_preview

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_ORDERING_PREVIEW_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_POLICY_CONSUMER_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_ordering_preview_result_consumer_only": True,
            "promotion_queue_ordering_plan_request_template_only": True,
            "requires_promotion_policy_consumer_result_consumer": True,
            "requires_promotion_ordering_preview_result": True,
            "requires_ordered_preview_records": True,
            "promotion_queue_ordering_plan_handler_invoked": False,
            "promotion_queue_ordering_plan_created": False,
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
            "sequence_policy_trust_store_mutation": False,
            "proposal_state_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
        },
        "summary": {
            "expected_policy_ranked_record_count": len(policy_records),
            "ordered_preview_record_count": len(ordered_rows),
            "policy_ranked_preview_record_count": len(ranked_rows),
            "unranked_preview_record_count": len(unranked_rows),
            "invalid_record_count": len(invalid_record_ids),
            "promotion_ordering_preview_result_consumer_only": True,
            "promotion_queue_ordering_plan_request_template_only": True,
            "promotion_queue_ordering_plan_request_available": request_available,
            "promotion_queue_ordering_plan_handler_invoked": False,
            "promotion_queue_ordering_plan_created": False,
            "requires_ordered_preview_records": True,
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
            "sequence_policy_trust_store_mutation": False,
            "proposal_state_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
            "policy_consumer_proposal_ids": [
                str(record["proposal_id"])
                for record in policy_records
                if isinstance(record.get("proposal_id"), str) and record.get("proposal_id")
            ],
            "ordered_preview_proposal_ids": [
                str(row["proposal_id"])
                for row in ordered_rows
                if isinstance(row.get("proposal_id"), str) and row.get("proposal_id")
            ],
            "policy_keys": [
                str(row["policy_key"])
                for row in ranked_rows
                if isinstance(row.get("policy_key"), str) and row.get("policy_key")
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion policy-consumer result consumer and ordering-preview result."
                if not policy_result_consumer or not ordering_preview
                else (
                    "Replace invalid sequence-memory trust promotion ordering-preview evidence before queue-ordering planning."
                    if invalid_record_ids
                    else (
                        "Run the read-only ordering preview before queue-ordering planning."
                        if ordering_status != REQUIRED_ORDERING_PREVIEW_AVAILABLE_STATUS
                        else "Run queue-ordering plan from verified ordering-preview evidence; this consumer does not mutate order, queue, policy, memory, model, or live state."
                    )
                )
            ),
        },
        "promotion_queue_ordering_plan_request": queue_ordering_plan_request,
    }
