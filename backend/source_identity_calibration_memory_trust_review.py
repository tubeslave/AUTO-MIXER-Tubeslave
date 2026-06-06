"""Read-only trust-review candidates from source-identity calibration memory.

The review layer consumes the append-only memory index and decides whether
indexed records are ready for operator trust review. It never mutates source
identity models, runtime trust, downstream proposal priority, queues, live mixer
transport, or console state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Tuple

try:
    from source_identity_calibration_memory_index import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_INDEX_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_INDEX_STATUSES,
        build_source_identity_calibration_memory_index,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_calibration_memory_index import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_INDEX_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_INDEX_STATUSES,
        build_source_identity_calibration_memory_index,
    )


SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_REVIEW_METHOD_ID = (
    "source_identity_calibration_memory_trust_review"
)
SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_REVIEW_STATUSES = {
    "memory_trust_review_not_ready",
    "memory_trust_review_replacement_required",
    "memory_trust_review_candidate_available",
}
REQUIRED_MEMORY_INDEX_TYPE = "source_identity_calibration_memory_index"
REQUIRED_INDEX_RECORD_FIELDS = (
    "memory_record_id",
    "draft_id",
    "source_candidate_id",
    "source_review_id",
    "source_type",
    "recommended_trust_state",
    "reviewed_by",
    "reviewed_at",
)
REQUIRED_INDEX_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "downstream_priority_mutation",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_INDEX_FALSE_SUMMARY_FLAGS = (
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "downstream_priority_mutation",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_FRESHNESS_FALSE_FLAGS = (
    "live_write",
    "queries_console",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _record_list(value: Any) -> Tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


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


def _freshness_status(payload: Mapping[str, Any] | None) -> str:
    if not isinstance(payload, Mapping):
        return "not_provided"
    return str(payload.get("status") or "unknown").strip().lower() or "unknown"


def _index_record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = record.get("memory_record_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    draft_id = record.get("draft_id")
    return f"{draft_id if isinstance(draft_id, str) and draft_id else 'unknown_memory_record'}:{index}"


def _index_envelope_violations(memory_index: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if memory_index.get("type") != REQUIRED_MEMORY_INDEX_TYPE:
        violations.append("memory_index_envelope_type_mismatch")
    generated_at = memory_index.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("memory_index_envelope_generated_at_invalid")
    if memory_index.get("method_id") != SOURCE_IDENTITY_CALIBRATION_MEMORY_INDEX_METHOD_ID:
        violations.append("memory_index_envelope_method_id_mismatch")
    if memory_index.get("status") not in SOURCE_IDENTITY_CALIBRATION_MEMORY_INDEX_STATUSES:
        violations.append("memory_index_envelope_status_unknown")
    safety = _mapping(memory_index.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("memory_index_envelope_read_only_not_true")
    for flag in REQUIRED_INDEX_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"memory_index_envelope_{flag}_not_false")
    return tuple(violations)


def _index_status_violations(
    memory_index: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = memory_index.get("status")
    if status == "memory_index_replacement_required":
        return ("memory_index_status_replacement_required",)
    if status == "memory_index_not_ready" and records:
        return ("memory_index_status_not_ready_with_records",)
    if status == "append_only_memory_index_available" and not records:
        return ("memory_index_status_available_without_records",)
    return ()


def _index_summary_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
    invalid_record_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if _summary_int(summary.get("memory_record_count")) != len(records):
        violations.append("memory_index_summary_memory_record_count_mismatch")
    if _summary_int(summary.get("invalid_record_count")) != len(invalid_record_ids):
        violations.append("memory_index_summary_invalid_record_count_mismatch")
    expected_candidate_ids = tuple(
        record["source_candidate_id"]
        for record in records
        if isinstance(record.get("source_candidate_id"), str)
        and record.get("source_candidate_id")
    )
    expected_source_types = tuple(
        record["source_type"]
        for record in records
        if isinstance(record.get("source_type"), str) and record.get("source_type")
    )
    if _string_list(summary.get("indexed_candidate_ids")) != expected_candidate_ids:
        violations.append("memory_index_summary_indexed_candidate_ids_mismatch")
    if _string_list(summary.get("indexed_source_types")) != expected_source_types:
        violations.append("memory_index_summary_indexed_source_types_mismatch")
    for flag in REQUIRED_INDEX_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(f"memory_index_summary_{flag}_not_false")
    return tuple(violations)


def _freshness_context_violations(
    payload: Mapping[str, Any] | None,
    *,
    freshness_status: str,
) -> Tuple[str, ...]:
    if freshness_status != "fresh":
        return ()
    freshness = _mapping(payload)
    violations = []
    if freshness.get("read_only") is not True:
        violations.append("memory_trust_review_freshness_read_only_not_true")
    for flag in REQUIRED_FRESHNESS_FALSE_FLAGS:
        if freshness.get(flag) is not False:
            violations.append(f"memory_trust_review_freshness_{flag}_not_false")
    return tuple(violations)


def _index_record_violations(record: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    for field in REQUIRED_INDEX_RECORD_FIELDS:
        if not isinstance(record.get(field), str) or not record.get(field):
            violations.append(f"memory_trust_review_index_record_{field}_missing")
    reviewed_at = record.get("reviewed_at")
    if isinstance(reviewed_at, str) and reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append("memory_trust_review_index_record_reviewed_at_invalid")
    if record.get("runtime_authority") != "none":
        violations.append("memory_trust_review_index_record_runtime_authority_not_none")
    if record.get("usable_for_downstream_priority") is not False:
        violations.append("memory_trust_review_index_record_downstream_usable_not_false")
    false_flags = (
        "queue_mutation_allowed_by_index",
        "model_mutation_allowed_by_index",
        "runtime_memory_mutation_allowed_by_index",
        "downstream_priority_change_allowed_by_index",
        "live_authority_granted",
        "full_autonomous_ready",
    )
    for flag in false_flags:
        if record.get(flag) is not False:
            violations.append(f"memory_trust_review_index_record_{flag}_not_false")
    return tuple(violations)


def _review_candidate(record: Mapping[str, Any], freshness_status: str) -> Dict[str, Any]:
    return {
        "review_candidate_id": f"source_identity_memory_trust_review:{record.get('memory_record_id')}",
        "memory_record_id": record.get("memory_record_id"),
        "source_candidate_id": record.get("source_candidate_id"),
        "source_review_id": record.get("source_review_id"),
        "source_type": record.get("source_type"),
        "recommended_trust_state": record.get("recommended_trust_state"),
        "reviewed_by": record.get("reviewed_by"),
        "reviewed_at": record.get("reviewed_at"),
        "current_state_cache_freshness_status": freshness_status,
        "operator_review_required": True,
        "memory_trust_review_state": "operator_review_required",
        "queue_mutation_allowed_by_review": False,
        "model_mutation_allowed_by_review": False,
        "runtime_memory_mutation_allowed_by_review": False,
        "downstream_priority_change_allowed_by_review": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }


def build_source_identity_calibration_memory_trust_review(
    *,
    memory_index: Mapping[str, Any] | None = None,
    current_state_cache_freshness: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build operator-review candidates from indexed source identity memory."""

    index_payload = _mapping(memory_index) or build_source_identity_calibration_memory_index()
    summary = _mapping(index_payload.get("summary"))
    records = _record_list(index_payload.get("memory_records"))
    index_invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    freshness_status = _freshness_status(current_state_cache_freshness)
    record_violations: List[str] = []
    for index, record in enumerate(records):
        violations = _index_record_violations(record)
        if violations:
            record_violations.append(_index_record_id(record, index))
            record_violations.extend(violations)
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *index_invalid_record_ids,
                *_index_envelope_violations(index_payload),
                *_index_status_violations(index_payload, records=records),
                *_index_summary_violations(
                    summary,
                    records=records,
                    invalid_record_ids=index_invalid_record_ids,
                ),
                *record_violations,
                *_freshness_context_violations(
                    current_state_cache_freshness,
                    freshness_status=freshness_status,
                ),
            )
        )
    )
    stale_context = freshness_status not in {"fresh"}

    if invalid_record_ids:
        status = "memory_trust_review_replacement_required"
        blocked_by = ("operator_must_replace_invalid_memory_index_before_trust_review",)
        candidates: Tuple[Dict[str, Any], ...] = ()
    elif not records:
        status = "memory_trust_review_not_ready"
        blocked_by = ("indexed_source_identity_memory_required_before_trust_review",)
        candidates = ()
    elif stale_context:
        status = "memory_trust_review_not_ready"
        blocked_by = ("fresh_source_identity_replay_context_required_before_trust_review",)
        candidates = ()
    else:
        status = "memory_trust_review_candidate_available"
        blocked_by = (
            "operator_review_required_before_memory_can_influence_downstream_trust",
        )
        candidates = tuple(_review_candidate(record, freshness_status) for record in records)

    return {
        "type": "source_identity_calibration_memory_trust_review",
        "generated_at": _utc_now(),
        "method_id": SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_REVIEW_METHOD_ID,
        "source_method_id": SOURCE_IDENTITY_CALIBRATION_MEMORY_INDEX_METHOD_ID,
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "downstream_priority_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "memory_index_status": index_payload.get("status") or "missing",
            "current_state_cache_freshness_status": freshness_status,
            "memory_record_count": len(records),
            "trust_review_candidate_count": len(candidates),
            "invalid_record_count": len(invalid_record_ids),
            "operator_review_required": True,
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "downstream_priority_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid memory index records before trust review."
                if invalid_record_ids
                else (
                    "Indexed source identity memory is required before trust review."
                    if not records
                    else (
                        "Fresh source identity replay context is required before trust review."
                        if stale_context
                        else (
                            "Operator reviews indexed memory before any separate "
                            "downstream trust promotion."
                        )
                    )
                )
            ),
        },
        "trust_review_candidates": list(candidates),
    }
