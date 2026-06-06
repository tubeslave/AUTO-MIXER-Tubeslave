"""Read-only memory draft records for source-identity calibration.

The draft converts manual persistence candidates into append-only memory record
proposals. It does not write `.ai` files, mutate classifier models, change
downstream proposal priority, update queues, or touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Tuple

try:
    from source_identity_calibration_persistence_plan import (
        SOURCE_IDENTITY_CALIBRATION_PERSISTENCE_PLAN_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_PERSISTENCE_PLAN_STATUSES,
        build_source_identity_calibration_persistence_plan,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_calibration_persistence_plan import (
        SOURCE_IDENTITY_CALIBRATION_PERSISTENCE_PLAN_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_PERSISTENCE_PLAN_STATUSES,
        build_source_identity_calibration_persistence_plan,
    )


SOURCE_IDENTITY_CALIBRATION_MEMORY_DRAFT_METHOD_ID = (
    "source_identity_calibration_memory_draft"
)
SOURCE_IDENTITY_CALIBRATION_MEMORY_DRAFT_STATUSES = {
    "memory_draft_not_ready",
    "memory_draft_replacement_required",
    "append_only_memory_draft_available",
}
REQUIRED_PERSISTENCE_PLAN_TYPE = "source_identity_calibration_persistence_plan"
REQUIRED_DRAFT_CANDIDATE_FIELDS = (
    "review_id",
    "candidate_id",
    "calibration_method_id",
    "source_type",
    "recommended_trust_state",
    "reviewed_by",
    "reviewed_at",
)
REQUIRED_PLAN_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "execution_allowed_by_plan",
    "calibration_persistence_write_allowed_by_plan",
    "queue_mutation_allowed_by_plan",
    "model_mutation_allowed_by_plan",
    "memory_mutation_allowed_by_plan",
    "downstream_priority_change_allowed_by_plan",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_PLAN_TRUE_SUMMARY_FLAGS = ("operator_review_required",)
REQUIRED_PLAN_FALSE_SUMMARY_FLAGS = (
    "calibration_persistence_write_allowed_by_plan",
    "queue_mutation_allowed_by_plan",
    "model_mutation_allowed_by_plan",
    "memory_mutation_allowed_by_plan",
    "downstream_priority_change_allowed_by_plan",
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


def _candidate_id(candidate: Mapping[str, Any], index: int) -> str:
    explicit = candidate.get("candidate_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    return f"unknown_memory_candidate:{index}"


def _candidate_basic_violations(candidate: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    for field in REQUIRED_DRAFT_CANDIDATE_FIELDS:
        if not isinstance(candidate.get(field), str) or not candidate.get(field):
            violations.append(f"memory_draft_candidate_{field}_missing")
    reviewed_at = candidate.get("reviewed_at")
    if isinstance(reviewed_at, str) and reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append("memory_draft_candidate_reviewed_at_invalid")
    false_flags = (
        "calibration_persistence_write_allowed_by_plan",
        "queue_mutation_allowed_by_plan",
        "model_mutation_allowed_by_plan",
        "memory_mutation_allowed_by_plan",
        "downstream_priority_change_allowed_by_plan",
        "live_authority_granted",
        "full_autonomous_ready",
    )
    for flag in false_flags:
        if candidate.get(flag) is not False:
            violations.append(f"memory_draft_candidate_{flag}_not_false")
    if candidate.get("manual_calibration_persistence_required") is not True:
        violations.append("memory_draft_candidate_manual_persistence_not_true")
    return tuple(violations)


def _plan_envelope_violations(plan: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if plan.get("type") != REQUIRED_PERSISTENCE_PLAN_TYPE:
        violations.append("persistence_plan_envelope_type_mismatch")
    generated_at = plan.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("persistence_plan_envelope_generated_at_invalid")
    if plan.get("method_id") != SOURCE_IDENTITY_CALIBRATION_PERSISTENCE_PLAN_METHOD_ID:
        violations.append("persistence_plan_envelope_method_id_mismatch")
    if plan.get("status") not in SOURCE_IDENTITY_CALIBRATION_PERSISTENCE_PLAN_STATUSES:
        violations.append("persistence_plan_envelope_status_unknown")
    safety = _mapping(plan.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("persistence_plan_envelope_read_only_not_true")
    for flag in REQUIRED_PLAN_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"persistence_plan_envelope_{flag}_not_false")
    return tuple(violations)


def _plan_summary_count_violations(
    summary: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
    invalid_record_ids: Tuple[str, ...],
    unreviewed_candidate_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    candidate_ids = tuple(
        candidate["candidate_id"]
        for candidate in candidates
        if isinstance(candidate.get("candidate_id"), str)
        and candidate.get("candidate_id")
    )
    expected_counts = {
        "persistence_candidate_count": len(candidate_ids),
        "invalid_record_count": len(invalid_record_ids),
        "unreviewed_candidate_count": len(unreviewed_candidate_ids),
    }
    return tuple(
        f"persistence_plan_summary_{key}_mismatch"
        for key, expected in expected_counts.items()
        if _summary_int(summary.get(key)) != expected
    )


def _plan_summary_identity_violations(
    summary: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    expected_candidate_ids = tuple(
        candidate["candidate_id"]
        for candidate in candidates
        if isinstance(candidate.get("candidate_id"), str)
        and candidate.get("candidate_id")
    )
    expected_source_types = tuple(
        candidate["source_type"]
        for candidate in candidates
        if isinstance(candidate.get("source_type"), str)
        and candidate.get("source_type")
    )
    violations = []
    if _string_list(summary.get("persistence_candidate_ids")) != expected_candidate_ids:
        violations.append("persistence_plan_summary_persistence_candidate_ids_mismatch")
    if _string_list(summary.get("accepted_source_types")) != expected_source_types:
        violations.append("persistence_plan_summary_accepted_source_types_mismatch")
    return tuple(violations)


def _plan_summary_safety_violations(summary: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    for flag in REQUIRED_PLAN_TRUE_SUMMARY_FLAGS:
        if summary.get(flag) is not True:
            violations.append(f"persistence_plan_summary_{flag}_not_true")
    for flag in REQUIRED_PLAN_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(f"persistence_plan_summary_{flag}_not_false")
    return tuple(violations)


def _draft_entry(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    source_type = str(candidate.get("source_type"))
    trust_state = str(candidate.get("recommended_trust_state"))
    return {
        "draft_id": f"source_identity_memory:{candidate.get('candidate_id')}",
        "target_memory_path": ".ai/memory/project.md",
        "target_section": "Project Memory Lessons",
        "append_only": True,
        "memory_write_allowed_by_draft": False,
        "model_mutation_allowed_by_draft": False,
        "downstream_priority_change_allowed_by_draft": False,
        "queue_mutation_allowed_by_draft": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "source_candidate_id": candidate.get("candidate_id"),
        "source_review_id": candidate.get("review_id"),
        "source_type": source_type,
        "recommended_trust_state": trust_state,
        "reviewed_by": candidate.get("reviewed_by"),
        "reviewed_at": candidate.get("reviewed_at"),
        "draft_text": (
            f"Source identity calibration for `{source_type}` was operator-reviewed "
            f"as `{trust_state}` from replay evidence; future memory or downstream "
            "priority writes still require a separate audited append-only write."
        ),
    }


def build_source_identity_calibration_memory_draft(
    *,
    persistence_plan: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build read-only append-only memory draft records from persistence candidates."""

    plan = _mapping(persistence_plan) or build_source_identity_calibration_persistence_plan()
    summary = _mapping(plan.get("summary"))
    candidates = _record_list(plan.get("persistence_candidates"))
    plan_invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    unreviewed_candidate_ids = _string_list(summary.get("unreviewed_candidate_ids"))
    plan_envelope_violations = _plan_envelope_violations(plan)
    candidate_violations = tuple(
        _candidate_id(candidate, index)
        for index, candidate in enumerate(candidates)
        if _candidate_basic_violations(candidate)
    )
    summary_count_violations = _plan_summary_count_violations(
        summary,
        candidates=candidates,
        invalid_record_ids=plan_invalid_record_ids,
        unreviewed_candidate_ids=unreviewed_candidate_ids,
    )
    summary_identity_violations = _plan_summary_identity_violations(
        summary,
        candidates=candidates,
    )
    summary_safety_violations = _plan_summary_safety_violations(summary)
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *plan_invalid_record_ids,
                *plan_envelope_violations,
                *candidate_violations,
                *summary_count_violations,
                *summary_identity_violations,
                *summary_safety_violations,
            )
        )
    )
    draft_records = () if invalid_record_ids else tuple(_draft_entry(candidate) for candidate in candidates)

    if invalid_record_ids:
        status = "memory_draft_replacement_required"
        blocked_by = ("operator_must_replace_invalid_persistence_plan_before_memory_draft",)
    elif unreviewed_candidate_ids:
        status = "memory_draft_not_ready"
        blocked_by = ("operator_must_review_all_candidates_before_memory_draft",)
    elif draft_records:
        status = "append_only_memory_draft_available"
        blocked_by = ("separate_append_only_memory_write_required",)
    else:
        status = "memory_draft_not_ready"
        blocked_by = ("manual_persistence_candidate_required_before_memory_draft",)

    return {
        "type": "source_identity_calibration_memory_draft",
        "generated_at": _utc_now(),
        "method_id": SOURCE_IDENTITY_CALIBRATION_MEMORY_DRAFT_METHOD_ID,
        "source_method_id": SOURCE_IDENTITY_CALIBRATION_PERSISTENCE_PLAN_METHOD_ID,
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "append_only_memory_write_allowed_by_draft": False,
            "queue_mutation_allowed_by_draft": False,
            "model_mutation_allowed_by_draft": False,
            "memory_mutation_allowed_by_draft": False,
            "downstream_priority_change_allowed_by_draft": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "persistence_plan_status": plan.get("status") or "missing",
            "persistence_candidate_count": len(candidates),
            "draft_record_count": len(draft_records),
            "invalid_record_count": len(invalid_record_ids),
            "unreviewed_candidate_count": len(unreviewed_candidate_ids),
            "draft_candidate_ids": [
                str(record["source_candidate_id"])
                for record in draft_records
                if isinstance(record.get("source_candidate_id"), str)
            ],
            "draft_source_types": [
                str(record["source_type"])
                for record in draft_records
                if isinstance(record.get("source_type"), str)
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "unreviewed_candidate_ids": list(unreviewed_candidate_ids),
            "append_only_memory_draft_available": status == "append_only_memory_draft_available",
            "operator_review_required": True,
            "append_only_memory_write_allowed_by_draft": False,
            "queue_mutation_allowed_by_draft": False,
            "model_mutation_allowed_by_draft": False,
            "memory_mutation_allowed_by_draft": False,
            "downstream_priority_change_allowed_by_draft": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid persistence-plan records before drafting memory."
                if invalid_record_ids
                else (
                    "Operator reviews all source identity candidates before memory draft."
                    if unreviewed_candidate_ids
                    else (
                        "Separate append-only memory write contract is required."
                        if draft_records
                        else "Manual persistence candidates are required before memory draft."
                    )
                )
            ),
        },
        "draft_records": list(draft_records),
    }
