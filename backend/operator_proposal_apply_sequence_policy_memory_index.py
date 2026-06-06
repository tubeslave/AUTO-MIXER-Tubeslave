"""Read-only index for supervised operator proposal apply sequence memory.

The index parses the markdown blocks emitted by the sequence policy memory
writer. It exposes durable sequence evidence for review only; it does not
mutate queues, models, runtime memory, policy stores, mixer transport, or
console state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_INDEX_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_index"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_INDEX_STATUSES = {
    "sequence_memory_index_not_ready",
    "sequence_memory_index_replacement_required",
    "append_only_sequence_memory_index_available",
}
ALLOWED_TARGET_MEMORY_PATH = ".ai/memory/project.md"
RUNTIME_AUTHORITY_LINE = "- Runtime authority: no live, queue, model, or policy mutation"
ALLOWED_VALUE_TYPES = {"fader", "gain"}
VERIFIED_READBACK_STATUSES = {"verified_from_cache", "verified_from_physical_readback"}

_HEADING_RE = re.compile(r"^### Supervised sequence policy memory\s*$")
_FIELD_RE = re.compile(
    r"^- (?P<label>Draft ID|Proposal ID|Learning scope|Value type|Channel|"
    r"Effective value|Idempotency key|Readback status|Proof status): "
    r"`(?P<value>[^`]*)`\s*$"
)
_FIELD_KEYS = {
    "Draft ID": "draft_id",
    "Proposal ID": "proposal_id",
    "Learning scope": "learning_scope",
    "Value type": "value_type",
    "Channel": "channel",
    "Effective value": "effective_value",
    "Idempotency key": "idempotency_key",
    "Readback status": "readback_status",
    "Proof status": "proof_status",
}
_REQUIRED_FIELDS = (
    "draft_id",
    "proposal_id",
    "learning_scope",
    "value_type",
    "channel",
    "effective_value",
    "idempotency_key",
    "readback_status",
    "proof_status",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _target_is_allowed(path_value: Any) -> bool:
    if path_value != ALLOWED_TARGET_MEMORY_PATH:
        return False
    path = Path(str(path_value))
    return not path.is_absolute() and ".." not in path.parts


def _extract_blocks(memory_text: str) -> Tuple[Tuple[str, ...], ...]:
    lines = memory_text.splitlines()
    starts = [index for index, line in enumerate(lines) if _HEADING_RE.match(line.strip())]
    blocks: List[Tuple[str, ...]] = []
    for start in starts:
        end = len(lines)
        for cursor in range(start + 1, len(lines)):
            if lines[cursor].strip().startswith("### "):
                end = cursor
                break
        blocks.append(tuple(line.strip() for line in lines[start:end]))
    return tuple(blocks)


def _block_fallback_id(index: int) -> str:
    return f"operator_sequence_policy_memory_index_block:{index}"


def _coerce_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_block(block: Tuple[str, ...], index: int) -> Tuple[Dict[str, Any], Tuple[str, ...]]:
    violations: List[str] = []
    fields: Dict[str, str] = {}
    duplicate_fields = set()
    runtime_authority_seen = False
    unsafe_runtime_authority_seen = False

    if not block or not _HEADING_RE.match(block[0]):
        violations.append("sequence_memory_index_heading_missing")

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
        violations.append(f"sequence_memory_index_record_{key}_duplicate")
    for key in _REQUIRED_FIELDS:
        if not fields.get(key):
            violations.append(f"sequence_memory_index_record_{key}_missing")
    if fields.get("draft_id") and not fields["draft_id"].startswith(
        "operator-sequence-policy-memory:"
    ):
        violations.append("sequence_memory_index_record_draft_id_not_operator_sequence")

    value_type = fields.get("value_type", "").strip().lower()
    if value_type not in ALLOWED_VALUE_TYPES:
        violations.append("sequence_memory_index_record_value_type_not_allowed")
    channel = _coerce_int(fields.get("channel", ""))
    if channel is None or channel < 1:
        violations.append("sequence_memory_index_record_channel_invalid")
    effective_value = _coerce_float(fields.get("effective_value", ""))
    if effective_value is None:
        violations.append("sequence_memory_index_record_effective_value_invalid")
    elif value_type == "fader" and effective_value > 0.0:
        violations.append("sequence_memory_index_record_fader_ceiling_exceeded")
    if fields.get("readback_status") not in VERIFIED_READBACK_STATUSES:
        violations.append("sequence_memory_index_record_readback_not_verified")
    if fields.get("proof_status") != "verified":
        violations.append("sequence_memory_index_record_proof_not_verified")
    if fields.get("learning_scope") != "supervised_fader_gain_sequence_execution":
        violations.append("sequence_memory_index_record_learning_scope_invalid")
    if not runtime_authority_seen:
        violations.append("sequence_memory_index_record_runtime_authority_missing")
    if unsafe_runtime_authority_seen:
        violations.append("sequence_memory_index_record_runtime_authority_unsafe")

    draft_id = fields.get("draft_id") or _block_fallback_id(index)
    record = {
        "memory_record_id": f"operator_sequence_policy_memory_index:{draft_id}",
        "draft_id": draft_id,
        "target_memory_path": ALLOWED_TARGET_MEMORY_PATH,
        "proposal_id": fields.get("proposal_id", ""),
        "learning_scope": fields.get("learning_scope", ""),
        "value_type": value_type,
        "channel": channel,
        "effective_value": effective_value,
        "idempotency_key": fields.get("idempotency_key", ""),
        "readback_status": fields.get("readback_status", ""),
        "proof_status": fields.get("proof_status", ""),
        "runtime_authority": "none",
        "usable_for_sequence_policy_review": True,
        "execution_allowed_by_index": False,
        "queue_mutation_allowed_by_index": False,
        "policy_mutation_allowed_by_index": False,
        "model_mutation_allowed_by_index": False,
        "runtime_memory_mutation_allowed_by_index": False,
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
        return None, False, ("sequence_memory_index_target_path_not_allowed",)
    if root is None:
        return None, False, ()

    target = Path(root).resolve() / target_memory_path
    try:
        if not target.exists():
            return None, True, ()
        return target.read_text(encoding="utf-8"), True, ()
    except OSError:
        return None, True, ("sequence_memory_index_file_read_failed",)


def build_operator_proposal_apply_sequence_policy_memory_index(
    *,
    memory_text: str | None = None,
    root: Path | str | None = None,
    target_memory_path: str = ALLOWED_TARGET_MEMORY_PATH,
) -> Dict[str, Any]:
    """Build a read-only index of append-only supervised sequence memory."""

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
        status = "sequence_memory_index_replacement_required"
        blocked_by = ("operator_must_replace_invalid_sequence_memory_index_source",)
        indexed_records: Tuple[Dict[str, Any], ...] = ()
    elif not text:
        status = "sequence_memory_index_not_ready"
        blocked_by = ("append_only_memory_file_required_before_sequence_index",)
        indexed_records = ()
    elif not records:
        status = "sequence_memory_index_not_ready"
        blocked_by = ("append_only_sequence_policy_memory_records_required",)
        indexed_records = ()
    else:
        status = "append_only_sequence_memory_index_available"
        blocked_by = (
            "sequence_memory_index_is_review_evidence_only_not_execution_authority",
        )
        indexed_records = tuple(records)

    return {
        "type": "operator_proposal_apply_sequence_policy_memory_index",
        "generated_at": _utc_now(),
        "method_id": OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_INDEX_METHOD_ID,
        "source_method_id": "write_operator_proposal_apply_sequence_policy_memory",
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "apply_attempted": False,
            "mutates_console_state": False,
            "queries_console": False,
            "memory_file_read": memory_file_read,
            "queue_mutation": False,
            "policy_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "target_memory_path": target_memory_path,
            "memory_file_read": memory_file_read,
            "memory_record_count": len(indexed_records),
            "invalid_record_count": len(tuple(dict.fromkeys(invalid_record_ids))),
            "indexed_proposal_ids": [
                record["proposal_id"] for record in indexed_records
            ],
            "indexed_value_types": [record["value_type"] for record in indexed_records],
            "indexed_learning_scopes": [
                record["learning_scope"] for record in indexed_records
            ],
            "execution_allowed_by_index": False,
            "queue_mutation": False,
            "policy_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "invalid_record_ids": list(dict.fromkeys(invalid_record_ids)),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid append-only sequence memory records."
                if invalid_record_ids
                else (
                    "Append-only sequence policy memory records must exist before indexing."
                    if not indexed_records
                    else (
                        "Use sequence index as review evidence only; separate "
                        "policy/trust promotion is required before execution authority."
                    )
                )
            ),
        },
        "memory_records": list(indexed_records),
    }
