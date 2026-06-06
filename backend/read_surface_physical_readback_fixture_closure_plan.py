"""Read-only plan for manual physical readback fixture closure.

The plan turns accepted operator review records into manual fixture-closure
candidates. It does not update fixture files, run tests, query hardware, mutate
console state, close physical fixture gaps, or grant live authority.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Tuple

try:
    from read_surface_physical_readback_fixture_acceptance_ledger import (
        READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_METHOD_ID,
        READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_STATUSES,
        build_read_surface_physical_readback_fixture_acceptance_ledger,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.read_surface_physical_readback_fixture_acceptance_ledger import (
        READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_METHOD_ID,
        READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_STATUSES,
        build_read_surface_physical_readback_fixture_acceptance_ledger,
    )


READ_SURFACE_PHYSICAL_READBACK_FIXTURE_CLOSURE_PLAN_METHOD_ID = (
    "read_surface_physical_readback_fixture_closure_plan"
)
READ_SURFACE_PHYSICAL_READBACK_FIXTURE_CLOSURE_PLAN_STATUSES = {
    "fixture_closure_not_ready",
    "fixture_closure_replacement_required",
    "manual_fixture_closure_candidate_available",
}
REQUIRED_ACCEPTANCE_LEDGER_TYPE = "read_surface_physical_readback_fixture_acceptance_ledger"
REQUIRED_CLOSURE_RECORD_FIELDS = (
    "review_id",
    "surface_id",
    "artifact_source_id",
    "required_contract_method_id",
    "reviewed_by",
    "reviewed_at",
)
REQUIRED_LEDGER_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "executes_tests",
    "execution_allowed_by_acceptance",
    "fixture_closure_allowed_by_acceptance",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_LEDGER_TRUE_SUMMARY_FLAGS = (
    "operator_review_required",
    "physical_fixture_gap_remains",
)
REQUIRED_LEDGER_FALSE_SUMMARY_FLAGS = (
    "fixture_closure_allowed_by_acceptance",
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


def _record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = record.get("review_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    surface_id = record.get("surface_id")
    artifact_source_id = record.get("artifact_source_id")
    return (
        f"{surface_id if isinstance(surface_id, str) and surface_id else 'unknown_surface'}:"
        f"{artifact_source_id if isinstance(artifact_source_id, str) and artifact_source_id else 'unknown_artifact'}:"
        f"{index}"
    )


def _closure_record_basic_violations(record: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    for field in REQUIRED_CLOSURE_RECORD_FIELDS:
        if not isinstance(record.get(field), str) or not record.get(field):
            violations.append(f"fixture_closure_record_{field}_missing")
    reviewed_at = record.get("reviewed_at")
    if isinstance(reviewed_at, str) and reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append("fixture_closure_record_reviewed_at_invalid")
    if record.get("valid") is not True:
        violations.append("fixture_closure_record_not_valid")
    if record.get("candidate_present") is not True:
        violations.append("fixture_closure_record_candidate_missing")
    if record.get("fixture_closure_allowed_by_acceptance") is not False:
        violations.append("fixture_closure_record_acceptance_closure_not_false")
    if record.get("live_authority_granted") is not False:
        violations.append("fixture_closure_record_live_authority_not_false")
    if record.get("full_autonomous_ready") is not False:
        violations.append("fixture_closure_record_full_autonomous_not_false")
    return tuple(violations)


def _accepted_record_violations(record: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = list(_closure_record_basic_violations(record))
    if record.get("decision") != "accepted":
        violations.append("fixture_closure_record_acceptance_decision_mismatch")
    if record.get("operator_rejected_review_candidate") is not False:
        violations.append("fixture_closure_record_conflicting_review_flags")
    return tuple(violations)


def _accepted_record_replacement_ids(
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    return tuple(
        _record_id(record, index)
        for index, record in enumerate(records)
        if record.get("operator_accepted_review_candidate") is True
        and _accepted_record_violations(record)
    )


def _ledger_envelope_violations(ledger: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if ledger.get("type") != REQUIRED_ACCEPTANCE_LEDGER_TYPE:
        violations.append("ledger_envelope_type_mismatch")
    generated_at = ledger.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("ledger_envelope_generated_at_invalid")
    if ledger.get("method_id") != READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_METHOD_ID:
        violations.append("ledger_envelope_method_id_mismatch")
    if ledger.get("status") not in READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_STATUSES:
        violations.append("ledger_envelope_status_unknown")
    safety = _mapping(ledger.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("ledger_envelope_read_only_not_true")
    for flag in REQUIRED_LEDGER_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"ledger_envelope_{flag}_not_false")
    return tuple(violations)


def _ledger_summary_count_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
    invalid_record_ids: Tuple[str, ...],
    unreviewed_surface_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    reviewed_surface_ids = tuple(
        record["surface_id"]
        for record in records
        if record.get("valid") is True
        and (
            record.get("operator_accepted_review_candidate") is True
            or record.get("operator_rejected_review_candidate") is True
        )
        and isinstance(record.get("surface_id"), str)
        and record.get("surface_id")
    )
    review_candidate_surface_ids = tuple(
        dict.fromkeys((*reviewed_surface_ids, *unreviewed_surface_ids))
    )
    expected_counts = {
        "review_candidate_count": len(review_candidate_surface_ids),
        "acceptance_record_count": len(records),
        "accepted_record_count": sum(
            1
            for record in records
            if record.get("valid") is True
            and record.get("operator_accepted_review_candidate") is True
        ),
        "rejected_record_count": sum(
            1
            for record in records
            if record.get("valid") is True
            and record.get("operator_rejected_review_candidate") is True
        ),
        "invalid_record_count": len(invalid_record_ids),
        "unreviewed_candidate_count": len(unreviewed_surface_ids),
    }
    return tuple(
        f"ledger_summary_{key}_mismatch"
        for key, expected in expected_counts.items()
        if _summary_int(summary.get(key)) != expected
    )


def _surface_ids_for_decision(
    records: Tuple[Mapping[str, Any], ...],
    *,
    accepted: bool,
) -> Tuple[str, ...]:
    flag = (
        "operator_accepted_review_candidate"
        if accepted
        else "operator_rejected_review_candidate"
    )
    return tuple(
        record["surface_id"]
        for record in records
        if record.get("valid") is True
        and record.get(flag) is True
        and isinstance(record.get("surface_id"), str)
        and record.get("surface_id")
    )


def _ledger_summary_surface_id_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    expected = {
        "accepted_surface_ids": _surface_ids_for_decision(records, accepted=True),
        "rejected_surface_ids": _surface_ids_for_decision(records, accepted=False),
    }
    return tuple(
        f"ledger_summary_{key}_mismatch"
        for key, surface_ids in expected.items()
        if _string_list(summary.get(key)) != surface_ids
    )


def _ledger_summary_safety_violations(summary: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    for flag in REQUIRED_LEDGER_TRUE_SUMMARY_FLAGS:
        if summary.get(flag) is not True:
            violations.append(f"ledger_summary_{flag}_not_true")
    for flag in REQUIRED_LEDGER_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(f"ledger_summary_{flag}_not_false")
    return tuple(violations)


def _ambiguous_surface_review_record_ids(
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    record_ids_by_surface: Dict[str, List[str]] = {}
    for index, record in enumerate(records):
        if _closure_record_basic_violations(record):
            continue
        decision = record.get("decision")
        if decision not in {"accepted", "rejected"}:
            continue
        surface_id = record.get("surface_id")
        if not isinstance(surface_id, str) or not surface_id:
            continue
        record_ids_by_surface.setdefault(surface_id, []).append(_record_id(record, index))
    return tuple(
        record_id
        for record_ids in record_ids_by_surface.values()
        if len(record_ids) > 1
        for record_id in record_ids
    )


def _accepted_records(
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[Mapping[str, Any], ...]:
    return tuple(
        record
        for record in records
        if record.get("operator_accepted_review_candidate") is True
        and not _accepted_record_violations(record)
    )


def _closure_candidate(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "review_id": record.get("review_id"),
        "surface_id": record.get("surface_id"),
        "artifact_source_id": record.get("artifact_source_id"),
        "required_contract_method_id": record.get("required_contract_method_id"),
        "reviewed_by": record.get("reviewed_by"),
        "reviewed_at": record.get("reviewed_at"),
        "manual_fixture_closure_required": True,
        "fixture_update_allowed_by_plan": False,
        "fixture_closure_allowed_by_plan": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }


def build_read_surface_physical_readback_fixture_closure_plan(
    *,
    acceptance_ledger: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a read-only manual closure plan from accepted ledger records."""

    ledger = (
        _mapping(acceptance_ledger)
        or build_read_surface_physical_readback_fixture_acceptance_ledger()
    )
    summary = _mapping(ledger.get("summary"))
    records = _record_list(ledger.get("records"))
    ledger_invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    unreviewed_surface_ids = _string_list(summary.get("unreviewed_candidate_surface_ids"))
    ledger_envelope_violations = _ledger_envelope_violations(ledger)
    accepted_record_replacement_ids = _accepted_record_replacement_ids(records)
    ambiguous_surface_review_record_ids = _ambiguous_surface_review_record_ids(records)
    ledger_summary_count_violations = _ledger_summary_count_violations(
        summary,
        records=records,
        invalid_record_ids=ledger_invalid_record_ids,
        unreviewed_surface_ids=unreviewed_surface_ids,
    )
    ledger_summary_surface_id_violations = _ledger_summary_surface_id_violations(
        summary,
        records=records,
    )
    ledger_summary_safety_violations = _ledger_summary_safety_violations(summary)
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *ledger_invalid_record_ids,
                *ledger_envelope_violations,
                *accepted_record_replacement_ids,
                *ambiguous_surface_review_record_ids,
                *ledger_summary_count_violations,
                *ledger_summary_surface_id_violations,
                *ledger_summary_safety_violations,
            )
        )
    )
    accepted_records = _accepted_records(records)
    closure_candidate_records = () if invalid_record_ids else accepted_records
    closure_candidates = tuple(
        _closure_candidate(record) for record in closure_candidate_records
    )
    rejected_surface_ids = _string_list(summary.get("rejected_surface_ids"))

    if invalid_record_ids:
        status = "fixture_closure_replacement_required"
        blocked_by = ("operator_must_replace_invalid_acceptance_records_before_fixture_closure",)
    elif unreviewed_surface_ids:
        status = "fixture_closure_not_ready"
        blocked_by = ("operator_must_review_all_candidates_before_fixture_closure",)
    elif closure_candidates:
        status = "manual_fixture_closure_candidate_available"
        blocked_by = ("manual_fixture_baseline_update_required_before_gap_closure",)
    else:
        status = "fixture_closure_not_ready"
        blocked_by = ("accepted_physical_readback_candidate_required_before_fixture_closure",)

    return {
        "type": "read_surface_physical_readback_fixture_closure_plan",
        "generated_at": _utc_now(),
        "method_id": READ_SURFACE_PHYSICAL_READBACK_FIXTURE_CLOSURE_PLAN_METHOD_ID,
        "source_method_id": READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_METHOD_ID,
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_plan": False,
            "fixture_update_allowed_by_plan": False,
            "fixture_closure_allowed_by_plan": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "ledger_status": ledger.get("status") or "missing",
            "review_candidate_count": int(summary.get("review_candidate_count") or 0),
            "acceptance_record_count": len(records),
            "accepted_record_count": len(accepted_records),
            "rejected_record_count": int(summary.get("rejected_record_count") or 0),
            "invalid_record_count": len(invalid_record_ids),
            "unreviewed_candidate_count": len(unreviewed_surface_ids),
            "closure_candidate_count": len(closure_candidates),
            "closure_candidate_surface_ids": [
                str(candidate["surface_id"])
                for candidate in closure_candidates
                if isinstance(candidate.get("surface_id"), str)
            ],
            "rejected_surface_ids": list(rejected_surface_ids),
            "invalid_record_ids": list(invalid_record_ids),
            "unreviewed_candidate_surface_ids": list(unreviewed_surface_ids),
            "manual_fixture_closure_candidate_available": (
                status == "manual_fixture_closure_candidate_available"
            ),
            "manual_fixture_closure_required": True,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_update_allowed_by_plan": False,
            "fixture_closure_allowed_by_plan": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid acceptance records."
                if invalid_record_ids
                else (
                    "Operator finishes review of every candidate artifact."
                    if unreviewed_surface_ids
                    else (
                        "Operator performs separate manual fixture baseline update."
                        if closure_candidates
                        else "Operator needs accepted physical readback candidates first."
                    )
                )
            ),
        },
        "closure_candidates": list(closure_candidates),
    }
