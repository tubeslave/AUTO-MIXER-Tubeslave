"""Read-only plan for manual physical readback fixture closure.

The plan turns accepted operator review records into manual fixture-closure
candidates. It does not update fixture files, run tests, query hardware, mutate
console state, close physical fixture gaps, or grant live authority.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from read_surface_physical_readback_fixture_acceptance_ledger import (
        READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_METHOD_ID,
        build_read_surface_physical_readback_fixture_acceptance_ledger,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.read_surface_physical_readback_fixture_acceptance_ledger import (
        READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_METHOD_ID,
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


def _accepted_records(ledger: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    return tuple(
        record
        for record in _record_list(ledger.get("records"))
        if record.get("valid") is True
        and record.get("operator_accepted_review_candidate") is True
        and record.get("fixture_closure_allowed_by_acceptance") is False
        and record.get("live_authority_granted") is False
        and record.get("full_autonomous_ready") is False
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
    accepted_records = _accepted_records(ledger)
    closure_candidates = tuple(_closure_candidate(record) for record in accepted_records)
    invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    unreviewed_surface_ids = _string_list(summary.get("unreviewed_candidate_surface_ids"))
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
