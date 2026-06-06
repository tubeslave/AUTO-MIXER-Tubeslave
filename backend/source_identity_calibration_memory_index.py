"""Read-only index for append-only source-identity calibration memory.

The index parses the markdown blocks emitted by the source identity calibration
memory writer. It exposes durable memory evidence for review, but it does not
mutate models, runtime trust, downstream proposal priority, queues, live mixer
transport, or console state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Tuple


SOURCE_IDENTITY_CALIBRATION_MEMORY_INDEX_METHOD_ID = (
    "source_identity_calibration_memory_index"
)
SOURCE_IDENTITY_CALIBRATION_MEMORY_INDEX_STATUSES = {
    "memory_index_not_ready",
    "memory_index_replacement_required",
    "append_only_memory_index_available",
}
ALLOWED_TARGET_MEMORY_PATH = ".ai/memory/project.md"
RUNTIME_AUTHORITY_LINE = (
    "- Runtime authority: no model, memory-priority, queue, or live-console mutation"
)

_HEADING_RE = re.compile(r"^### Source identity calibration memory:\s*(?P<source_type>.+?)\s*$")
_FIELD_RE = re.compile(
    r"^- (?P<label>Draft ID|Source candidate|Source review|Recommended trust state|"
    r"Reviewed by|Reviewed at): `(?P<value>[^`]*)`\s*$"
)
_FIELD_KEYS = {
    "Draft ID": "draft_id",
    "Source candidate": "source_candidate_id",
    "Source review": "source_review_id",
    "Recommended trust state": "recommended_trust_state",
    "Reviewed by": "reviewed_by",
    "Reviewed at": "reviewed_at",
}
_REQUIRED_FIELDS = (
    "draft_id",
    "source_candidate_id",
    "source_review_id",
    "recommended_trust_state",
    "reviewed_by",
    "reviewed_at",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def _target_is_allowed(path_value: Any) -> bool:
    if path_value != ALLOWED_TARGET_MEMORY_PATH:
        return False
    path = Path(str(path_value))
    return not path.is_absolute() and ".." not in path.parts


def _extract_blocks(memory_text: str) -> Tuple[Tuple[str, ...], ...]:
    lines = memory_text.splitlines()
    starts = [index for index, line in enumerate(lines) if _HEADING_RE.match(line.strip())]
    blocks: List[Tuple[str, ...]] = []
    for offset, start in enumerate(starts):
        end = starts[offset + 1] if offset + 1 < len(starts) else len(lines)
        blocks.append(tuple(line.strip() for line in lines[start:end]))
    return tuple(blocks)


def _block_fallback_id(index: int) -> str:
    return f"source_identity_memory_index_block:{index}"


def _parse_block(block: Tuple[str, ...], index: int) -> Tuple[Dict[str, Any], Tuple[str, ...]]:
    violations: List[str] = []
    fields: Dict[str, str] = {}
    duplicate_fields = set()
    heading = _HEADING_RE.match(block[0] if block else "")
    source_type = heading.group("source_type").strip() if heading else ""
    runtime_authority_seen = False
    unsafe_runtime_authority_seen = False

    if not source_type:
        violations.append("memory_index_source_type_missing")
    elif "`" in source_type:
        violations.append("memory_index_source_type_invalid")

    for line in block[1:]:
        field_match = _FIELD_RE.match(line)
        if field_match:
            key = _FIELD_KEYS[field_match.group("label")]
            if key in fields:
                duplicate_fields.add(key)
            fields[key] = field_match.group("value").strip()
            continue
        if line == RUNTIME_AUTHORITY_LINE:
            runtime_authority_seen = True
        elif line.startswith("- Runtime authority:"):
            unsafe_runtime_authority_seen = True

    for key in sorted(duplicate_fields):
        violations.append(f"memory_index_record_{key}_duplicate")
    for key in _REQUIRED_FIELDS:
        if not fields.get(key):
            violations.append(f"memory_index_record_{key}_missing")
    if fields.get("draft_id") and not fields["draft_id"].startswith("source_identity_memory:"):
        violations.append("memory_index_record_draft_id_not_source_identity_memory")
    reviewed_at = fields.get("reviewed_at", "")
    if reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append("memory_index_record_reviewed_at_invalid")
    if not runtime_authority_seen:
        violations.append("memory_index_record_runtime_authority_missing")
    if unsafe_runtime_authority_seen:
        violations.append("memory_index_record_runtime_authority_unsafe")

    draft_id = fields.get("draft_id") or _block_fallback_id(index)
    record = {
        "memory_record_id": f"source_identity_memory_index:{draft_id}",
        "draft_id": draft_id,
        "target_memory_path": ALLOWED_TARGET_MEMORY_PATH,
        "source_candidate_id": fields.get("source_candidate_id", ""),
        "source_review_id": fields.get("source_review_id", ""),
        "source_type": source_type,
        "recommended_trust_state": fields.get("recommended_trust_state", ""),
        "reviewed_by": fields.get("reviewed_by", ""),
        "reviewed_at": reviewed_at,
        "runtime_authority": "none",
        "usable_for_downstream_priority": False,
        "queue_mutation_allowed_by_index": False,
        "model_mutation_allowed_by_index": False,
        "runtime_memory_mutation_allowed_by_index": False,
        "downstream_priority_change_allowed_by_index": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }
    return record, tuple(violations)


def _read_memory_text(
    *,
    root: Path | str | None,
    target_memory_path: str,
) -> Tuple[str | None, bool, Tuple[str, ...]]:
    if not _target_is_allowed(target_memory_path):
        return None, False, ("memory_index_target_path_not_allowed",)
    if root is None:
        return None, False, ()

    target = Path(root).resolve() / target_memory_path
    try:
        if not target.exists():
            return None, True, ()
        return target.read_text(encoding="utf-8"), True, ()
    except OSError:
        return None, True, ("memory_index_file_read_failed",)


def build_source_identity_calibration_memory_index(
    *,
    memory_text: str | None = None,
    root: Path | str | None = None,
    target_memory_path: str = ALLOWED_TARGET_MEMORY_PATH,
) -> Dict[str, Any]:
    """Build a read-only index of append-only source identity memory records."""

    explicit_text = memory_text if isinstance(memory_text, str) else None
    read_text, memory_file_read, read_violations = _read_memory_text(
        root=root,
        target_memory_path=target_memory_path,
    )
    text = explicit_text if explicit_text is not None else read_text
    invalid_record_ids: List[str] = list(read_violations)
    records: List[Dict[str, Any]] = []

    if text:
        for index, block in enumerate(_extract_blocks(text)):
            record, violations = _parse_block(block, index)
            if violations:
                invalid_record_ids.append(record["draft_id"])
                invalid_record_ids.extend(violations)
            else:
                records.append(record)

    if invalid_record_ids:
        status = "memory_index_replacement_required"
        blocked_by = ("operator_must_replace_invalid_memory_index_source",)
        indexed_records: Tuple[Dict[str, Any], ...] = ()
    elif not text:
        status = "memory_index_not_ready"
        blocked_by = ("append_only_memory_file_required_before_index",)
        indexed_records = ()
    elif not records:
        status = "memory_index_not_ready"
        blocked_by = ("append_only_source_identity_memory_records_required",)
        indexed_records = ()
    else:
        status = "append_only_memory_index_available"
        blocked_by = (
            "memory_index_is_review_evidence_only_not_downstream_priority_authority",
        )
        indexed_records = tuple(records)

    return {
        "type": "source_identity_calibration_memory_index",
        "generated_at": _utc_now(),
        "method_id": SOURCE_IDENTITY_CALIBRATION_MEMORY_INDEX_METHOD_ID,
        "source_method_id": "source_identity_calibration_memory_writer",
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "memory_file_read": memory_file_read,
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "downstream_priority_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "target_memory_path": target_memory_path,
            "memory_file_read": memory_file_read,
            "memory_record_count": len(indexed_records),
            "invalid_record_count": len(tuple(dict.fromkeys(invalid_record_ids))),
            "indexed_candidate_ids": [
                record["source_candidate_id"] for record in indexed_records
            ],
            "indexed_source_types": [record["source_type"] for record in indexed_records],
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "downstream_priority_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "invalid_record_ids": list(dict.fromkeys(invalid_record_ids)),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid append-only source identity memory records."
                if invalid_record_ids
                else (
                    "Append-only source identity memory records must exist before indexing."
                    if not indexed_records
                    else (
                        "Use index as review evidence only; separate replay/trust "
                        "promotion is required before downstream priority changes."
                    )
                )
            ),
        },
        "memory_records": list(indexed_records),
    }
