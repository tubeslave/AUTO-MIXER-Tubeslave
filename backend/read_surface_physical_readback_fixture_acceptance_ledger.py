"""Read-only ledger for operator review of physical readback candidates.

The ledger validates operator review records against the aggregate artifact
review queue. It does not accept fixtures into the automated test baseline,
close physical fixture gaps, query hardware, execute tests, or grant live
authority.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from read_surface_physical_readback_artifact_review_queue import (
        build_read_surface_physical_readback_artifact_review_queue,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.read_surface_physical_readback_artifact_review_queue import (
        build_read_surface_physical_readback_artifact_review_queue,
    )


READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_METHOD_ID = (
    "read_surface_physical_readback_fixture_acceptance_ledger"
)
READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_STATUSES = {
    "operator_acceptance_required",
    "operator_acceptance_records_available",
    "operator_acceptance_replacement_required",
}
ACCEPTANCE_DECISIONS = {"accepted", "rejected"}
UNSAFE_ACCEPTANCE_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "executes_tests",
    "execution_allowed_by_acceptance",
    "fixture_closure_allowed_by_acceptance",
    "fixture_closure_allowed",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
    "scene_load_allowed",
    "scene_save_allowed",
    "preset_recall_allowed",
    "preset_store_allowed",
    "midi_send_allowed",
    "tcp_send_allowed",
    "dlive_write_allowed",
    "dlive_apply_allowed",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return value if isinstance(value, str) and value else ""


def _string_list(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)


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


def _record_list(value: Iterable[Mapping[str, Any]] | None) -> Tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _queue_items_by_surface(queue: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    items = queue.get("queue_items")
    if not isinstance(items, list):
        return {}
    return {
        str(item["surface_id"]): item
        for item in items
        if isinstance(item, Mapping)
        and isinstance(item.get("surface_id"), str)
        and item.get("accepted_as_physical_fixture_candidate") is True
    }


def _record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("review_id"))
    if explicit:
        return explicit
    surface_id = _string(record.get("surface_id")) or "unknown_surface"
    artifact_source_id = _string(record.get("artifact_source_id")) or "unknown_artifact"
    return f"{surface_id}:{artifact_source_id}:{index}"


def _unsafe_flags(record: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(key for key in UNSAFE_ACCEPTANCE_FLAGS if record.get(key) is True)


def _validated_record(
    record: Mapping[str, Any],
    *,
    index: int,
    candidates_by_surface: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    record_id = _record_id(record, index)
    surface_id = _string(record.get("surface_id"))
    artifact_source_id = _string(record.get("artifact_source_id"))
    decision = _string(record.get("decision"))
    reviewed_by = _string(record.get("reviewed_by"))
    reviewed_at = _string(record.get("reviewed_at"))
    required_contract_method_id = _string(record.get("required_contract_method_id"))
    candidate = _mapping(candidates_by_surface.get(surface_id))
    candidate_artifact_sources = _string_list(candidate.get("artifact_source_ids"))
    unsafe_flags = _unsafe_flags(record)
    violations = []

    if surface_id not in candidates_by_surface:
        violations.append("acceptance_record_candidate_missing")
    if decision not in ACCEPTANCE_DECISIONS:
        violations.append("acceptance_record_decision_unknown")
    if not reviewed_by:
        violations.append("acceptance_record_reviewer_missing")
    if not reviewed_at:
        violations.append("acceptance_record_reviewed_at_missing")
    elif not _is_utc_timestamp(reviewed_at):
        violations.append("acceptance_record_reviewed_at_invalid")
    if not artifact_source_id:
        violations.append("acceptance_record_artifact_source_missing")
    if required_contract_method_id != candidate.get("required_contract_method_id"):
        violations.append("acceptance_record_contract_mismatch")
    if candidate and not candidate_artifact_sources:
        violations.append("acceptance_record_candidate_artifact_sources_missing")
    elif candidate_artifact_sources and artifact_source_id not in candidate_artifact_sources:
        violations.append("acceptance_record_artifact_source_mismatch")
    if unsafe_flags:
        violations.append("acceptance_record_unsafe_authority_claim")

    valid = not violations
    return {
        "review_id": record_id,
        "surface_id": surface_id,
        "artifact_source_id": artifact_source_id,
        "required_contract_method_id": required_contract_method_id,
        "decision": decision if decision in ACCEPTANCE_DECISIONS else "invalid",
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at,
        "candidate_present": bool(candidate),
        "candidate_status": candidate.get("status") or "missing",
        "operator_accepted_review_candidate": valid and decision == "accepted",
        "operator_rejected_review_candidate": valid and decision == "rejected",
        "fixture_closure_allowed_by_acceptance": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "unsafe_authority_flags": list(unsafe_flags),
        "valid": valid,
        "violations": violations,
    }


def build_read_surface_physical_readback_fixture_acceptance_ledger(
    *,
    review_queue: Mapping[str, Any] | None = None,
    acceptance_records: Iterable[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build a read-only ledger for operator review records."""

    queue = _mapping(review_queue) or build_read_surface_physical_readback_artifact_review_queue()
    candidates_by_surface = _queue_items_by_surface(queue)
    records = tuple(
        _validated_record(
            record,
            index=index,
            candidates_by_surface=candidates_by_surface,
        )
        for index, record in enumerate(_record_list(acceptance_records))
    )
    valid_records = tuple(record for record in records if record["valid"])
    accepted_records = tuple(
        record for record in valid_records if record["operator_accepted_review_candidate"]
    )
    rejected_records = tuple(
        record for record in valid_records if record["operator_rejected_review_candidate"]
    )
    invalid_records = tuple(record for record in records if not record["valid"])
    reviewed_surface_ids = {
        str(record["surface_id"])
        for record in valid_records
        if isinstance(record.get("surface_id"), str)
    }
    unreviewed_candidate_surface_ids = tuple(
        surface_id
        for surface_id in candidates_by_surface
        if surface_id not in reviewed_surface_ids
    )

    if invalid_records:
        status = "operator_acceptance_replacement_required"
        blocked_by = ("operator_must_replace_invalid_acceptance_records",)
    elif unreviewed_candidate_surface_ids:
        status = "operator_acceptance_required"
        blocked_by = ("operator_must_review_candidate_artifacts_before_fixture_closure",)
    elif accepted_records or rejected_records:
        status = "operator_acceptance_records_available"
        blocked_by = ("fixture_gap_still_requires_separate_manual_closure_decision",)
    else:
        status = "operator_acceptance_required"
        blocked_by = ("operator_must_capture_or_surface_valid_candidates_before_acceptance",)

    accepted_surface_ids = [record["surface_id"] for record in accepted_records]
    rejected_surface_ids = [record["surface_id"] for record in rejected_records]
    return {
        "type": "read_surface_physical_readback_fixture_acceptance_ledger",
        "generated_at": _utc_now(),
        "method_id": READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_METHOD_ID,
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_acceptance": False,
            "fixture_closure_allowed_by_acceptance": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "queue_status": queue.get("status") or "missing",
            "review_candidate_count": len(candidates_by_surface),
            "acceptance_record_count": len(records),
            "accepted_record_count": len(accepted_records),
            "rejected_record_count": len(rejected_records),
            "invalid_record_count": len(invalid_records),
            "unreviewed_candidate_count": len(unreviewed_candidate_surface_ids),
            "accepted_surface_ids": accepted_surface_ids,
            "rejected_surface_ids": rejected_surface_ids,
            "invalid_record_ids": [record["review_id"] for record in invalid_records],
            "unreviewed_candidate_surface_ids": list(unreviewed_candidate_surface_ids),
            "operator_acceptance_required": status == "operator_acceptance_required",
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_acceptance": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid acceptance records."
                if invalid_records
                else (
                    "Operator reviews candidate artifacts and records decisions."
                    if unreviewed_candidate_surface_ids
                    else (
                        "Manual fixture closure remains separate from acceptance records."
                        if accepted_records or rejected_records
                        else "Operator captures valid physical readback candidates first."
                    )
                )
            ),
        },
        "records": list(records),
    }
