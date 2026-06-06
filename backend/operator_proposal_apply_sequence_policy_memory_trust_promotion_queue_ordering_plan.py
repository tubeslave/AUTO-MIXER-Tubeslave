"""Read-only queue-ordering plan from sequence-policy ordering preview evidence.

This module consumes the sequence-policy ordering preview and emits a separate
plan for a future queue or sequence consumer. It does not mutate the operator
queue, proposal payload order, runtime memory, sequence-policy stores, models,
proposal state, or live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_ORDERING_PLAN_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_ORDERING_PLAN_STATUSES = {
    "sequence_memory_trust_promotion_queue_ordering_plan_not_ready",
    "sequence_memory_trust_promotion_queue_ordering_plan_replacement_required",
    "sequence_memory_trust_promotion_queue_ordering_plan_available",
}
REQUIRED_ORDERING_PREVIEW_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview"
)
REQUIRED_ORDERING_PREVIEW_FALSE_FLAGS = (
    "live_write",
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
REQUIRED_ORDERING_ROW_FALSE_FLAGS = REQUIRED_ORDERING_PREVIEW_FALSE_FLAGS
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


def _duplicate_values(values: Tuple[str, ...]) -> Tuple[str, ...]:
    counts: Dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return tuple(value for value, count in counts.items() if count > 1)


def _ordering_row_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("ordering_preview_record_id"))
    if explicit:
        return explicit
    proposal_id = _string(record.get("proposal_id"))
    return f"{proposal_id if proposal_id else 'unknown_sequence_ordering_preview_record'}:{index}"


def _ordering_preview_envelope_violations(
    ordering_preview: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    if ordering_preview.get("type") != REQUIRED_ORDERING_PREVIEW_TYPE:
        violations.append("sequence_policy_queue_ordering_plan_preview_type_mismatch")
    generated_at = ordering_preview.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("sequence_policy_queue_ordering_plan_preview_generated_at_invalid")
    if (
        ordering_preview.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_METHOD_ID
    ):
        violations.append("sequence_policy_queue_ordering_plan_preview_method_id_mismatch")
    if (
        ordering_preview.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_STATUSES
    ):
        violations.append("sequence_policy_queue_ordering_plan_preview_status_unknown")
    safety = _mapping(ordering_preview.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_policy_queue_ordering_plan_preview_read_only_not_true")
    if safety.get("ordering_preview_only") is not True:
        violations.append("sequence_policy_queue_ordering_plan_preview_only_not_true")
    if safety.get("apply_attempted") is not False:
        violations.append("sequence_policy_queue_ordering_plan_preview_apply_attempted_not_false")
    for flag in REQUIRED_ORDERING_PREVIEW_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"sequence_policy_queue_ordering_plan_preview_{flag}_not_false")
    return tuple(violations)


def _ordering_preview_status_violations(
    ordering_preview: Mapping[str, Any],
    *,
    ordered_rows: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = ordering_preview.get("status")
    if status == "sequence_memory_trust_promotion_ordering_preview_replacement_required":
        return ("sequence_policy_queue_ordering_plan_preview_replacement_required",)
    if (
        status == "sequence_memory_trust_promotion_ordering_preview_available"
        and not ordered_rows
    ):
        return ("sequence_policy_queue_ordering_plan_preview_available_without_rows",)
    if (
        status != "sequence_memory_trust_promotion_ordering_preview_available"
        and ordered_rows
    ):
        return ("sequence_policy_queue_ordering_plan_preview_rows_without_available_status",)
    return ()


def _ordering_preview_summary_violations(
    summary: Mapping[str, Any],
    *,
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
        "ordered_preview_count": len(ordered_rows),
        "policy_ranked_proposal_count": len(ranked_rows),
        "unranked_proposal_count": len(unranked_rows),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_queue_ordering_plan_preview_summary_{key}_mismatch"
            )
    if summary.get("recommended_order_available") is not bool(ordered_rows):
        violations.append(
            "sequence_policy_queue_ordering_plan_preview_summary_recommended_order_mismatch"
        )
    if summary.get("ordering_preview_only") is not True:
        violations.append("sequence_policy_queue_ordering_plan_preview_summary_only_not_true")
    for flag in REQUIRED_ORDERING_PREVIEW_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_queue_ordering_plan_preview_summary_{flag}_not_false"
            )
    expected_order = tuple(
        row["proposal_id"] for row in ordered_rows if _string(row.get("proposal_id"))
    )
    if _string_list(summary.get("recommended_ordered_proposal_ids")) != expected_order:
        violations.append(
            "sequence_policy_queue_ordering_plan_preview_summary_ordered_ids_mismatch"
        )
    return tuple(violations)


def _ordering_row_violations(record: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    required_fields = (
        "ordering_preview_record_id",
        "proposal_id",
        "recommended_sequence_policy_bucket",
        "rank_source",
    )
    for field in required_fields:
        if not _string(record.get(field)):
            violations.append(f"sequence_policy_queue_ordering_plan_row_{field}_missing")
    rank_source = record.get("rank_source")
    if rank_source not in ALLOWED_RANK_SOURCES:
        violations.append("sequence_policy_queue_ordering_plan_row_rank_source_unknown")
    for field in ("original_index", "recommended_index"):
        if _summary_int(record.get(field)) is None:
            violations.append(f"sequence_policy_queue_ordering_plan_row_{field}_invalid")
    for field in ("base_policy_score", "adjusted_policy_score"):
        score = _safe_float(record.get(field))
        if score is None or score < 0.0 or score > 1.0:
            violations.append(f"sequence_policy_queue_ordering_plan_row_{field}_invalid")
    if record.get("read_only") is not True:
        violations.append("sequence_policy_queue_ordering_plan_row_read_only_not_true")
    if record.get("ordering_preview_only") is not True:
        violations.append("sequence_policy_queue_ordering_plan_row_preview_only_not_true")
    if record.get("policy_consumer_required_before_queue_mutation") is not True:
        violations.append(
            "sequence_policy_queue_ordering_plan_row_queue_consumer_required_not_true"
        )
    if record.get("sequence_consumer_required_before_execution") is not True:
        violations.append(
            "sequence_policy_queue_ordering_plan_row_sequence_consumer_required_not_true"
        )
    for flag in REQUIRED_ORDERING_ROW_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(f"sequence_policy_queue_ordering_plan_row_{flag}_not_false")
    return tuple(violations)


def _invalid_ordering_row_ids(
    rows: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    return tuple(
        _ordering_row_id(row, index)
        for index, row in enumerate(rows)
        if _ordering_row_violations(row)
    )


def _sequence_violations(rows: Tuple[Mapping[str, Any], ...]) -> Tuple[str, ...]:
    recommended_indexes = tuple(
        row.get("recommended_index")
        for row in rows
        if _summary_int(row.get("recommended_index")) is not None
    )
    if recommended_indexes != tuple(range(len(rows))):
        return ("sequence_policy_queue_ordering_plan_recommended_index_sequence_mismatch",)
    return ()


def _queue_ordering_plan_records(
    rows: Tuple[Mapping[str, Any], ...],
) -> Tuple[Dict[str, Any], ...]:
    plan_records = []
    for planned_index, row in enumerate(
        sorted(rows, key=lambda item: int(item.get("recommended_index") or 0))
    ):
        proposal_id = str(row["proposal_id"])
        plan_records.append(
            {
                "queue_ordering_plan_record_id": (
                    f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_ORDERING_PLAN_METHOD_ID}:{proposal_id}"
                ),
                "proposal_id": proposal_id,
                "proposal_method_id": row.get("proposal_method_id"),
                "proposal_kind": row.get("proposal_kind"),
                "proposal_target": row.get("proposal_target"),
                "ordering_preview_record_id": row.get("ordering_preview_record_id"),
                "consumer_record_id": row.get("consumer_record_id"),
                "sequence_policy_gate_record_id": row.get("sequence_policy_gate_record_id"),
                "policy_key": row.get("policy_key"),
                "original_index": row.get("original_index"),
                "recommended_index": row.get("recommended_index"),
                "planned_queue_index": planned_index,
                "rank_source": row.get("rank_source"),
                "recommended_sequence_policy_bucket": row.get(
                    "recommended_sequence_policy_bucket"
                ),
                "base_policy_score": row.get("base_policy_score"),
                "adjusted_policy_score": row.get("adjusted_policy_score"),
                "operator_review_required": True,
                "operator_ack_required_before_queue_import": True,
                "operator_ack_required_before_sequence_execution": True,
                "idempotency_key_required": True,
                "queue_ordering_plan_only": True,
                "sequence_consumer_required_before_execution": True,
                "read_only": True,
                "proposal_order_mutation": False,
                "queue_mutation": False,
                "queue_import_allowed_by_plan": False,
                "queue_reorder_allowed_by_plan": False,
                "mutates_console_state": False,
                "queries_console": False,
                "model_mutation": False,
                "runtime_memory_mutation": False,
                "sequence_policy_trust_store_mutation": False,
                "proposal_state_mutation": False,
                "sequence_execution_allowed_by_plan": False,
                "sequence_policy_change_allowed_by_plan": False,
                "live_widening_allowed": False,
                "live_write": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "can_import_to_queue": False,
                "can_apply_live": False,
            }
        )
    return tuple(plan_records)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan(
    *,
    ordering_preview: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a read-only queue-ordering plan from sequence ordering-preview rows."""

    preview = (
        _mapping(ordering_preview)
        or build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview()
    )
    summary = _mapping(preview.get("summary"))
    ordered_rows = _record_list(preview.get("ordered_proposals"))
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_string_list(summary.get("invalid_record_ids")),
                *_ordering_preview_envelope_violations(preview),
                *_ordering_preview_status_violations(
                    preview,
                    ordered_rows=ordered_rows,
                ),
                *_ordering_preview_summary_violations(
                    summary,
                    ordered_rows=ordered_rows,
                ),
                *_invalid_ordering_row_ids(ordered_rows),
                *_sequence_violations(ordered_rows),
                *_duplicate_values(
                    tuple(
                        _string(row.get("ordering_preview_record_id"))
                        for row in ordered_rows
                    )
                ),
                *_duplicate_values(
                    tuple(_string(row.get("proposal_id")) for row in ordered_rows)
                ),
                *_duplicate_values(
                    tuple(str(row.get("recommended_index")) for row in ordered_rows)
                ),
            )
        )
    )
    plan_records = (
        () if invalid_record_ids or not ordered_rows else _queue_ordering_plan_records(ordered_rows)
    )
    ranked_records = tuple(
        record for record in plan_records if record.get("rank_source") == "sequence_policy_consumer"
    )
    unranked_records = tuple(
        record for record in plan_records if record.get("rank_source") == "original_order"
    )
    planned_ids = tuple(record["proposal_id"] for record in plan_records)

    if invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_ordering_plan_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_policy_queue_ordering_plan_evidence",
        )
    elif not ordered_rows:
        status = "sequence_memory_trust_promotion_queue_ordering_plan_not_ready"
        blocked_by = ("valid_sequence_policy_ordering_preview_records_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_ordering_plan_available"
        blocked_by = (
            "sequence_policy_queue_ordering_plan_is_not_queue_execution_or_live_authority",
        )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_ORDERING_PLAN_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "apply_attempted": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "queue_ordering_plan_only": True,
            "sequence_consumer_required_before_execution": True,
            "proposal_order_mutation": False,
            "queue_mutation": False,
            "queue_import_allowed_by_plan": False,
            "queue_reorder_allowed_by_plan": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "sequence_policy_trust_store_mutation": False,
            "proposal_state_mutation": False,
            "sequence_execution_allowed_by_plan": False,
            "sequence_policy_change_allowed_by_plan": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
        },
        "summary": {
            "ordering_preview_status": preview.get("status") or "missing",
            "input_ordering_preview_count": len(ordered_rows),
            "queue_ordering_plan_record_count": len(plan_records),
            "ranked_queue_candidate_count": len(ranked_records),
            "unranked_queue_candidate_count": len(unranked_records),
            "invalid_record_count": len(invalid_record_ids),
            "queue_ordering_plan_available": bool(plan_records),
            "queue_ordering_plan_only": True,
            "sequence_consumer_required_before_execution": True,
            "proposal_order_mutation": False,
            "queue_mutation": False,
            "queue_import_allowed_by_plan": False,
            "queue_reorder_allowed_by_plan": False,
            "mutates_console_state": False,
            "queries_console": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "sequence_policy_trust_store_mutation": False,
            "proposal_state_mutation": False,
            "sequence_execution_allowed_by_plan": False,
            "sequence_policy_change_allowed_by_plan": False,
            "live_widening_allowed": False,
            "live_write": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
            "planned_queue_proposal_ids": list(planned_ids),
            "ranked_queue_candidate_proposal_ids": [
                record["proposal_id"] for record in ranked_records
            ],
            "unranked_queue_candidate_proposal_ids": [
                record["proposal_id"] for record in unranked_records
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid sequence-policy queue-ordering plan evidence."
                if invalid_record_ids
                else (
                    "Valid sequence-policy ordering-preview records are required before queue-ordering planning."
                    if not ordered_rows
                    else "Operator may inspect this plan; a separate acknowledged queue or sequence consumer is required before mutation."
                )
            ),
        },
        "queue_ordering_plan_records": list(plan_records),
    }
