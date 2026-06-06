"""Append-only writer contract for source-identity calibration memory drafts.

The writer validates draft records and can append them to the project memory
file only when explicitly invoked with operator acknowledgement. Product Layer
previews call it as dry-run evidence only. It never mutates classifier models,
downstream proposal priority, queues, live mixer transport, or console state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

try:
    from source_identity_calibration_memory_draft import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_DRAFT_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_DRAFT_STATUSES,
        build_source_identity_calibration_memory_draft,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_calibration_memory_draft import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_DRAFT_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_DRAFT_STATUSES,
        build_source_identity_calibration_memory_draft,
    )


SOURCE_IDENTITY_CALIBRATION_MEMORY_WRITER_METHOD_ID = (
    "source_identity_calibration_memory_writer"
)
SOURCE_IDENTITY_CALIBRATION_MEMORY_WRITER_STATUSES = {
    "memory_write_not_ready",
    "memory_write_replacement_required",
    "memory_write_operator_ack_required",
    "append_only_memory_write_dry_run_ready",
    "append_only_memory_write_applied",
}
REQUIRED_MEMORY_DRAFT_TYPE = "source_identity_calibration_memory_draft"
ALLOWED_TARGET_MEMORY_PATH = ".ai/memory/project.md"
REQUIRED_DRAFT_RECORD_FIELDS = (
    "draft_id",
    "target_memory_path",
    "target_section",
    "source_candidate_id",
    "source_review_id",
    "source_type",
    "recommended_trust_state",
    "reviewed_by",
    "reviewed_at",
    "draft_text",
)
REQUIRED_DRAFT_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "append_only_memory_write_allowed_by_draft",
    "queue_mutation_allowed_by_draft",
    "model_mutation_allowed_by_draft",
    "memory_mutation_allowed_by_draft",
    "downstream_priority_change_allowed_by_draft",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_DRAFT_FALSE_SUMMARY_FLAGS = (
    "append_only_memory_write_allowed_by_draft",
    "queue_mutation_allowed_by_draft",
    "model_mutation_allowed_by_draft",
    "memory_mutation_allowed_by_draft",
    "downstream_priority_change_allowed_by_draft",
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


def _draft_record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = record.get("draft_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    candidate_id = record.get("source_candidate_id")
    return f"{candidate_id if isinstance(candidate_id, str) and candidate_id else 'unknown_draft'}:{index}"


def _target_is_allowed(path_value: Any) -> bool:
    if path_value != ALLOWED_TARGET_MEMORY_PATH:
        return False
    path = Path(str(path_value))
    return not path.is_absolute() and ".." not in path.parts


def _draft_record_violations(record: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    for field in REQUIRED_DRAFT_RECORD_FIELDS:
        if not isinstance(record.get(field), str) or not record.get(field):
            violations.append(f"memory_writer_draft_record_{field}_missing")
    reviewed_at = record.get("reviewed_at")
    if isinstance(reviewed_at, str) and reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append("memory_writer_draft_record_reviewed_at_invalid")
    if record.get("append_only") is not True:
        violations.append("memory_writer_draft_record_append_only_not_true")
    if not _target_is_allowed(record.get("target_memory_path")):
        violations.append("memory_writer_draft_record_target_path_not_allowed")
    false_flags = (
        "memory_write_allowed_by_draft",
        "queue_mutation_allowed_by_draft",
        "model_mutation_allowed_by_draft",
        "downstream_priority_change_allowed_by_draft",
        "live_authority_granted",
        "full_autonomous_ready",
    )
    for flag in false_flags:
        if record.get(flag) is not False:
            violations.append(f"memory_writer_draft_record_{flag}_not_false")
    return tuple(violations)


def _draft_envelope_violations(draft: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if draft.get("type") != REQUIRED_MEMORY_DRAFT_TYPE:
        violations.append("memory_draft_envelope_type_mismatch")
    generated_at = draft.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("memory_draft_envelope_generated_at_invalid")
    if draft.get("method_id") != SOURCE_IDENTITY_CALIBRATION_MEMORY_DRAFT_METHOD_ID:
        violations.append("memory_draft_envelope_method_id_mismatch")
    if draft.get("status") not in SOURCE_IDENTITY_CALIBRATION_MEMORY_DRAFT_STATUSES:
        violations.append("memory_draft_envelope_status_unknown")
    safety = _mapping(draft.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("memory_draft_envelope_read_only_not_true")
    for flag in REQUIRED_DRAFT_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"memory_draft_envelope_{flag}_not_false")
    return tuple(violations)


def _draft_summary_count_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
    invalid_record_ids: Tuple[str, ...],
    unreviewed_candidate_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    expected_counts = {
        "draft_record_count": len(records),
        "invalid_record_count": len(invalid_record_ids),
        "unreviewed_candidate_count": len(unreviewed_candidate_ids),
    }
    return tuple(
        f"memory_draft_summary_{key}_mismatch"
        for key, expected in expected_counts.items()
        if _summary_int(summary.get(key)) != expected
    )


def _draft_summary_identity_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    expected_candidate_ids = tuple(
        record["source_candidate_id"]
        for record in records
        if isinstance(record.get("source_candidate_id"), str)
        and record.get("source_candidate_id")
    )
    expected_source_types = tuple(
        record["source_type"]
        for record in records
        if isinstance(record.get("source_type"), str)
        and record.get("source_type")
    )
    violations = []
    if _string_list(summary.get("draft_candidate_ids")) != expected_candidate_ids:
        violations.append("memory_draft_summary_draft_candidate_ids_mismatch")
    if _string_list(summary.get("draft_source_types")) != expected_source_types:
        violations.append("memory_draft_summary_draft_source_types_mismatch")
    return tuple(violations)


def _draft_summary_safety_violations(summary: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if summary.get("operator_review_required") is not True:
        violations.append("memory_draft_summary_operator_review_required_not_true")
    for flag in REQUIRED_DRAFT_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(f"memory_draft_summary_{flag}_not_false")
    return tuple(violations)


def _render_append_block(record: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "---",
            "",
            f"### Source identity calibration memory: {record.get('source_type')}",
            "",
            f"- Draft ID: `{record.get('draft_id')}`",
            f"- Source candidate: `{record.get('source_candidate_id')}`",
            f"- Source review: `{record.get('source_review_id')}`",
            f"- Recommended trust state: `{record.get('recommended_trust_state')}`",
            f"- Reviewed by: `{record.get('reviewed_by')}`",
            f"- Reviewed at: `{record.get('reviewed_at')}`",
            "- Runtime authority: no model, memory-priority, queue, or live-console mutation",
            "",
            str(record.get("draft_text") or "").strip(),
            "",
        ]
    )


def _append_records(root: Path, records: Tuple[Mapping[str, Any], ...]) -> Tuple[Dict[str, Any], ...]:
    results: List[Dict[str, Any]] = []
    for record in records:
        target = root / str(record["target_memory_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        current = target.read_text(encoding="utf-8") if target.exists() else "# Project Memory\n"
        block = _render_append_block(record)
        next_content = current.rstrip() + "\n\n" + block
        target.write_text(next_content, encoding="utf-8")
        results.append(
            {
                "draft_id": record.get("draft_id"),
                "target_memory_path": str(record.get("target_memory_path")),
                "applied": True,
                "bytes_appended": len(block.encode("utf-8")),
            }
        )
    return tuple(results)


def build_source_identity_calibration_memory_writer(
    *,
    memory_draft: Mapping[str, Any] | None = None,
    root: Path | str | None = None,
    apply: bool = False,
    operator_ack: bool = False,
) -> Dict[str, Any]:
    """Validate and optionally apply append-only source identity memory drafts."""

    draft = _mapping(memory_draft) or build_source_identity_calibration_memory_draft()
    summary = _mapping(draft.get("summary"))
    records = _record_list(draft.get("draft_records"))
    draft_invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    unreviewed_candidate_ids = _string_list(summary.get("unreviewed_candidate_ids"))
    draft_envelope_violations = _draft_envelope_violations(draft)
    draft_record_violations = tuple(
        _draft_record_id(record, index)
        for index, record in enumerate(records)
        if _draft_record_violations(record)
    )
    summary_count_violations = _draft_summary_count_violations(
        summary,
        records=records,
        invalid_record_ids=draft_invalid_record_ids,
        unreviewed_candidate_ids=unreviewed_candidate_ids,
    )
    summary_identity_violations = _draft_summary_identity_violations(
        summary,
        records=records,
    )
    summary_safety_violations = _draft_summary_safety_violations(summary)
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *draft_invalid_record_ids,
                *draft_envelope_violations,
                *draft_record_violations,
                *summary_count_violations,
                *summary_identity_violations,
                *summary_safety_violations,
            )
        )
    )

    write_results: Tuple[Dict[str, Any], ...] = ()
    if invalid_record_ids:
        status = "memory_write_replacement_required"
        blocked_by = ("operator_must_replace_invalid_memory_draft_before_write",)
    elif unreviewed_candidate_ids:
        status = "memory_write_not_ready"
        blocked_by = ("operator_must_review_all_candidates_before_memory_write",)
    elif not records:
        status = "memory_write_not_ready"
        blocked_by = ("append_only_memory_draft_required_before_write",)
    elif apply and not operator_ack:
        status = "memory_write_operator_ack_required"
        blocked_by = ("operator_ack_required_before_append_only_memory_write",)
    elif apply and operator_ack:
        status = "append_only_memory_write_applied"
        blocked_by = ()
        write_results = _append_records(Path(root or ".").resolve(), records)
    else:
        status = "append_only_memory_write_dry_run_ready"
        blocked_by = ("operator_ack_and_apply_required_before_append_only_memory_write",)

    applied = status == "append_only_memory_write_applied"
    return {
        "type": "source_identity_calibration_memory_writer",
        "generated_at": _utc_now(),
        "method_id": SOURCE_IDENTITY_CALIBRATION_MEMORY_WRITER_METHOD_ID,
        "source_method_id": SOURCE_IDENTITY_CALIBRATION_MEMORY_DRAFT_METHOD_ID,
        "status": status,
        "safety": {
            "read_only": not applied,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "operator_ack_required": True,
            "apply_requested": bool(apply),
            "operator_ack": bool(operator_ack),
            "append_only_memory_file_write_applied": applied,
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "downstream_priority_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "memory_draft_status": draft.get("status") or "missing",
            "draft_record_count": len(records),
            "invalid_record_count": len(invalid_record_ids),
            "unreviewed_candidate_count": len(unreviewed_candidate_ids),
            "write_result_count": len(write_results),
            "append_only_memory_write_applied": applied,
            "operator_ack_required": True,
            "operator_ack": bool(operator_ack),
            "apply_requested": bool(apply),
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "downstream_priority_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "invalid_record_ids": list(invalid_record_ids),
            "unreviewed_candidate_ids": list(unreviewed_candidate_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid memory draft records before writing."
                if invalid_record_ids
                else (
                    "Operator reviews all source identity candidates before memory write."
                    if unreviewed_candidate_ids
                    else (
                        "Operator acknowledgement is required before append-only write."
                        if status == "memory_write_operator_ack_required"
                        else (
                            "Append-only memory write has been applied."
                            if applied
                            else "Dry-run ready; pass apply with operator acknowledgement to write."
                        )
                    )
                )
            ),
        },
        "write_results": list(write_results),
    }
