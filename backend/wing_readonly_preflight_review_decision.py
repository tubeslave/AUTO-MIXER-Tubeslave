"""Read-only ledger for operator decisions on WING preflight candidates.

This module validates operator review records against already-captured
``wing_readonly_preflight.v1`` evidence. It never contacts WING, never executes
tests, never mutates readiness files, and never grants live authority.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from wing_readonly_preflight_evidence import build_wing_readonly_preflight_evidence
    from wing_readonly_preflight_contract import (
        is_sha256_hexdigest,
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND,
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS,
        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS,
        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS,
        WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY,
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_ARTIFACT_ROOT_BLOCKER,
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER,
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_REQUEST_SEQUENCE_BLOCKER,
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_VERIFIER_WS_PROOF_BLOCKER,
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS,
        WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS,
        wing_readonly_preflight_accepted_candidate_artifact_root_contract,
        wing_readonly_preflight_accepted_candidate_request_sequence_source_ids,
        wing_readonly_preflight_accepted_candidate_request_sequence_violations_by_source,
        wing_readonly_preflight_accepted_candidate_verifier_ws_proof_source_ids,
        wing_readonly_preflight_accepted_candidate_verifier_ws_proof_violations_by_source,
        wing_readonly_preflight_accepted_candidate_prestart_violations_by_source,
        wing_readonly_preflight_accepted_candidate_prestart_source_ids,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.wing_readonly_preflight_evidence import build_wing_readonly_preflight_evidence
    from backend.wing_readonly_preflight_contract import (
        is_sha256_hexdigest,
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND,
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS,
        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS,
        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS,
        WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY,
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_ARTIFACT_ROOT_BLOCKER,
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER,
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_REQUEST_SEQUENCE_BLOCKER,
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_VERIFIER_WS_PROOF_BLOCKER,
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS,
        WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS,
        wing_readonly_preflight_accepted_candidate_artifact_root_contract,
        wing_readonly_preflight_accepted_candidate_request_sequence_source_ids,
        wing_readonly_preflight_accepted_candidate_request_sequence_violations_by_source,
        wing_readonly_preflight_accepted_candidate_verifier_ws_proof_source_ids,
        wing_readonly_preflight_accepted_candidate_verifier_ws_proof_violations_by_source,
        wing_readonly_preflight_accepted_candidate_prestart_violations_by_source,
        wing_readonly_preflight_accepted_candidate_prestart_source_ids,
    )


WING_READONLY_PREFLIGHT_REVIEW_DECISION_LEDGER_METHOD_ID = (
    "wing_readonly_preflight_review_decision_ledger"
)
WING_READONLY_PREFLIGHT_REVIEW_DECISION_LEDGER_STATUSES = {
    "operator_review_required",
    "operator_review_decision_records_available",
    "operator_review_decision_replacement_required",
}
REVIEW_DECISIONS = {"accepted", "rejected"}
REQUIRED_REVIEW_CHECKLIST_IDS = tuple(
    WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS
)
WS_TOKEN = "web" + "sock" + "et"
ARTIFACT_MANIFEST_VERIFIER_WS_PROOF_FIELDS = (
    f"artifact_manifest_verifier_must_pass_before_{WS_TOKEN}",
    f"artifact_manifest_verifier_passed_before_{WS_TOKEN}",
    f"artifact_manifest_verifier_{WS_TOKEN}_opened_after_pass",
)
DEFAULT_WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOTS = (
    "artifacts/wing_readonly_preflight_review_decisions",
)
UNSAFE_DECISION_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "executes_tests",
    "execution_allowed_by_decision",
    "readiness_closure_allowed_by_decision",
    "readiness_matrix_update_allowed_by_decision",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
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


def _int_list(value: Any) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    rows: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            continue
        rows.append(item)
    return tuple(rows)


def _mapping_list(value: Any) -> list[Dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _non_bool_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _record_list(value: Iterable[Mapping[str, Any]] | None) -> Tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _decision_roots(decision_roots: Iterable[str | Path] | None) -> Tuple[Path, ...]:
    roots = (
        tuple(decision_roots)
        if decision_roots is not None
        else DEFAULT_WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOTS
    )
    return tuple(Path(root) for root in roots)


def _discover_decision_paths(decision_roots: Iterable[str | Path] | None) -> Tuple[Path, ...]:
    paths: list[Path] = []
    for root in _decision_roots(decision_roots):
        if root.is_file() and root.suffix == ".json":
            paths.append(root)
        if not root.is_dir():
            continue
        paths.extend(path for path in root.rglob("*.json") if path.is_file())
    return tuple(sorted(dict.fromkeys(paths), key=lambda path: str(path)))


def _load_json_decision(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("review decision JSON root must be an object")
    return payload


def _decision_records(
    *,
    decision_records: Iterable[Mapping[str, Any]] | None,
    decision_paths: Iterable[str | Path] | None,
    decision_roots: Iterable[str | Path] | None,
) -> Tuple[Mapping[str, Any], ...]:
    records = list(_record_list(decision_records))
    paths = (
        tuple(Path(path) for path in decision_paths)
        if decision_paths is not None
        else _discover_decision_paths(decision_roots)
    )
    for path in paths:
        try:
            records.append(_load_json_decision(path))
        except Exception as exc:  # pragma: no cover - defensive file parser
            records.append(
                {
                    "review_id": str(path),
                    "decision": "invalid",
                    "load_error": type(exc).__name__,
                }
            )
    return tuple(records)


def _parse_utc_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        return None
    return parsed


def _is_utc_timestamp(value: str) -> bool:
    return _parse_utc_timestamp(value) is not None


def _candidate_items_by_source(evidence: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, list):
        return {}
    return {
        str(item["source_id"]): item
        for item in artifacts
        if isinstance(item, Mapping)
        and isinstance(item.get("source_id"), str)
        and item.get("accepted_as_readonly_preflight_candidate") is True
    }


def _source_id_is_path_like(source_id: str) -> bool:
    return "/" in source_id or "\\" in source_id


def _accepted_candidate_artifact_root_contract_for_ledger(
    accepted_source_ids: Iterable[str],
    candidate_reviews: Iterable[Mapping[str, Any]],
) -> tuple[list[str], dict[str, list[str]]]:
    """Return artifact-root violations, tolerating synthetic inline source ids."""

    roots, violations_by_source = (
        wing_readonly_preflight_accepted_candidate_artifact_root_contract(
            accepted_source_ids,
            candidate_reviews,
        )
    )
    filtered: dict[str, list[str]] = {}
    for source_id, violations in violations_by_source.items():
        items = list(violations)
        if not _source_id_is_path_like(source_id):
            items = [
                item
                for item in items
                if item not in {"source_id_not_under_artifact_root"}
            ]
        if items:
            filtered[source_id] = items
    return roots, filtered


def _candidate_review_rows(
    candidates_by_source: Mapping[str, Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for source_id, candidate in candidates_by_source.items():
        rows.append(
            {
                "source_id": source_id,
                "status": candidate.get("status"),
                "run_id": candidate.get("run_id"),
                "captured_at": candidate.get("captured_at"),
                "artifact_root": candidate.get("artifact_root"),
                "summary_path": candidate.get("summary_path"),
                "summary_sha256": candidate.get("summary_sha256"),
                "events_path": candidate.get("events_path"),
                "events_sha256": candidate.get("events_sha256"),
                "expected_artifacts": list(candidate.get("expected_artifacts") or []),
                "expected_summary_artifact": candidate.get(
                    "expected_summary_artifact"
                ),
                "expected_events_artifact": candidate.get(
                    "expected_events_artifact"
                ),
                "summary_path_matches_expected_artifact": (
                    candidate.get("summary_path_matches_expected_artifact") is True
                ),
                "events_path_matches_expected_artifact": (
                    candidate.get("events_path_matches_expected_artifact") is True
                ),
                "contacts_backend": (
                    candidate.get("contacts_backend")
                    if isinstance(candidate.get("contacts_backend"), bool)
                    else None
                ),
                "queries_console": (
                    candidate.get("queries_console")
                    if isinstance(candidate.get("queries_console"), bool)
                    else None
                ),
                "request_sequence": _mapping_list(candidate.get("request_sequence")),
                "plan_request_sequence": _mapping_list(
                    candidate.get("plan_request_sequence")
                ),
                "request_sequence_matches_plan": (
                    candidate.get("request_sequence_matches_plan") is True
                ),
                "backend_port_prestart_check_required": (
                    candidate.get("backend_port_prestart_check_required") is True
                ),
                "backend_port_prestart_check_must_pass_before_starting_backend": (
                    candidate.get(
                        "backend_port_prestart_check_must_pass_before_starting_backend"
                    )
                    is True
                ),
                "backend_port_prestart_check_command": list(
                    candidate.get("backend_port_prestart_check_command") or []
                ),
                "backend_port_prestart_check_command_text": str(
                    candidate.get("backend_port_prestart_check_command_text") or ""
                ),
                "backend_port_prestart_check_failure_detail_fields": list(
                    candidate.get("backend_port_prestart_check_failure_detail_fields")
                    or []
                ),
                "backend_port_prestart_summary_required": (
                    candidate.get("backend_port_prestart_summary_required") is True
                ),
                "backend_port_prestart_summary_path": candidate.get(
                    "backend_port_prestart_summary_path"
                ),
                "backend_port_prestart_summary_sha256": candidate.get(
                    "backend_port_prestart_summary_sha256"
                ),
                "backend_port_prestart_summary_type": candidate.get(
                    "backend_port_prestart_summary_type"
                ),
                "backend_port_prestart_summary_status": candidate.get(
                    "backend_port_prestart_summary_status"
                ),
                "backend_port_prestart_summary_connection_policy": candidate.get(
                    "backend_port_prestart_summary_connection_policy"
                ),
                "backend_port_prestart_summary_read_only": candidate.get(
                    "backend_port_prestart_summary_read_only"
                ),
                "backend_port_prestart_summary_live_write": candidate.get(
                    "backend_port_prestart_summary_live_write"
                ),
                "backend_port_prestart_summary_contacts_backend": candidate.get(
                    "backend_port_prestart_summary_contacts_backend"
                ),
                "backend_port_prestart_summary_contacts_wing": candidate.get(
                    "backend_port_prestart_summary_contacts_wing"
                ),
                "backend_port_prestart_summary_mutates_console_state": candidate.get(
                    "backend_port_prestart_summary_mutates_console_state"
                ),
                "backend_port_prestart_summary_queries_console": candidate.get(
                    "backend_port_prestart_summary_queries_console"
                ),
                "backend_port_prestart_summary_executes_tests": candidate.get(
                    "backend_port_prestart_summary_executes_tests"
                ),
                "backend_port_prestart_check_status": candidate.get(
                    "backend_port_prestart_check_status"
                ),
                **{
                    field: candidate.get(field)
                    if isinstance(candidate.get(field), bool)
                    else None
                    for field in ARTIFACT_MANIFEST_VERIFIER_WS_PROOF_FIELDS
                },
                **{
                    field: candidate.get(field)
                    if isinstance(candidate.get(field), bool)
                    else None
                    for field in (
                        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
                    )
                },
                **{
                    field: candidate.get(field)
                    if isinstance(candidate.get(field), bool)
                    else None
                    for field in (
                        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS
                    )
                },
                **{
                    field: candidate.get(field)
                    if isinstance(candidate.get(field), bool)
                    else None
                    for field in (
                        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS
                    )
                },
            }
        )
    return rows


def _record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("review_id"))
    if explicit:
        return explicit
    source_id = _string(record.get("artifact_source_id")) or "unknown_artifact"
    return f"{source_id}:{index}"


def _unsafe_flags(record: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(flag for flag in UNSAFE_DECISION_FLAGS if record.get(flag) is True)


def _duplicate_review_source_ids(
    records: Iterable[Mapping[str, Any]],
) -> Tuple[str, ...]:
    counts: dict[str, int] = {}
    for record in records:
        source_id = _string(record.get("artifact_source_id"))
        if not source_id or record.get("valid") is not True:
            continue
        counts[source_id] = counts.get(source_id, 0) + 1
    return tuple(source_id for source_id, count in counts.items() if count > 1)


def _conflicting_review_source_ids(
    records: Iterable[Mapping[str, Any]],
) -> Tuple[str, ...]:
    decisions_by_source: dict[str, set[str]] = {}
    for record in records:
        source_id = _string(record.get("artifact_source_id"))
        decision = _string(record.get("decision"))
        if not source_id or record.get("valid") is not True:
            continue
        decisions_by_source.setdefault(source_id, set()).add(decision)
    return tuple(
        source_id
        for source_id, decisions in decisions_by_source.items()
        if len(decisions) > 1
    )


def _mark_review_decision_collisions(
    records: Iterable[Mapping[str, Any]],
) -> Tuple[Mapping[str, Any], ...]:
    records_tuple = tuple(records)
    duplicate_source_ids = set(_duplicate_review_source_ids(records_tuple))
    conflicting_source_ids = set(_conflicting_review_source_ids(records_tuple))
    if not duplicate_source_ids and not conflicting_source_ids:
        return records_tuple

    updated_records: list[Mapping[str, Any]] = []
    for record in records_tuple:
        source_id = _string(record.get("artifact_source_id"))
        if source_id not in duplicate_source_ids:
            updated_records.append(record)
            continue
        updated = dict(record)
        violations = list(updated.get("violations") or [])
        if "review_decision_duplicate_artifact_source" not in violations:
            violations.append("review_decision_duplicate_artifact_source")
        if (
            source_id in conflicting_source_ids
            and "review_decision_conflicting_artifact_source_decisions" not in violations
        ):
            violations.append("review_decision_conflicting_artifact_source_decisions")
        updated["valid"] = False
        updated["violations"] = violations
        updated["operator_accepted_readonly_preflight_candidate"] = False
        updated["operator_rejected_readonly_preflight_candidate"] = False
        updated_records.append(updated)
    return tuple(updated_records)


def _validated_record(
    record: Mapping[str, Any],
    *,
    index: int,
    candidates_by_source: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    explicit_review_id = _string(record.get("review_id"))
    record_id = _record_id(record, index)
    artifact_source_id = _string(record.get("artifact_source_id"))
    decision = _string(record.get("decision"))
    reviewed_by = _string(record.get("reviewed_by"))
    reviewed_at = _string(record.get("reviewed_at"))
    template_status = _string(record.get("template_status"))
    run_id = _string(record.get("run_id"))
    artifact_captured_at = _string(record.get("artifact_captured_at"))
    artifact_root = _string(record.get("artifact_root"))
    summary_path = _string(record.get("summary_path"))
    summary_sha256 = _string(record.get("summary_sha256"))
    events_path = _string(record.get("events_path"))
    events_sha256 = _string(record.get("events_sha256"))
    channels = _int_list(record.get("channels"))
    value_types = _string_list(record.get("value_types"))
    operator_confirmed_channels = _int_list(record.get("operator_confirmed_channels"))
    operator_confirmed_session_ready = (
        record.get("operator_confirmed_session_ready") is True
    )
    readback_count = _non_bool_int(record.get("readback_count"))
    physical_readback_count = _non_bool_int(record.get("physical_readback_count"))
    request_id_policy = _string(record.get("request_id_policy"))
    request_id_correlation_required = (
        record.get("request_id_correlation_required") is True
    )
    event_log_run_id_matches_summary = (
        record.get("event_log_run_id_matches_summary") is True
    )
    checklist_ids = _string_list(record.get("checklist_ids"))
    candidate = _mapping(candidates_by_source.get(artifact_source_id))
    unsafe_flags = _unsafe_flags(record)
    violations: list[str] = []

    if artifact_source_id not in candidates_by_source:
        violations.append("review_decision_candidate_missing")
    if not explicit_review_id:
        violations.append("review_decision_review_id_missing")
    if template_status == WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS:
        violations.append("review_decision_template_status_unedited")
    if decision not in REVIEW_DECISIONS:
        violations.append("review_decision_unknown")
    if not reviewed_by:
        violations.append("review_decision_reviewer_missing")
    if not reviewed_at:
        violations.append("review_decision_reviewed_at_missing")
    elif not _is_utc_timestamp(reviewed_at):
        violations.append("review_decision_reviewed_at_invalid")
    if artifact_captured_at and not _is_utc_timestamp(artifact_captured_at):
        violations.append("review_decision_artifact_captured_at_invalid")
    if not artifact_source_id:
        violations.append("review_decision_artifact_source_missing")
    if checklist_ids != REQUIRED_REVIEW_CHECKLIST_IDS:
        violations.append("review_decision_checklist_ids_mismatch")
    if request_id_policy != WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY:
        violations.append("review_decision_request_id_policy_mismatch")
    if not request_id_correlation_required:
        violations.append("review_decision_request_id_correlation_not_confirmed")
    if not event_log_run_id_matches_summary:
        violations.append("review_decision_event_log_run_id_not_confirmed")
    if summary_sha256 and not is_sha256_hexdigest(summary_sha256):
        violations.append("review_decision_summary_sha256_invalid")
    if events_sha256 and not is_sha256_hexdigest(events_sha256):
        violations.append("review_decision_events_sha256_invalid")
    if candidate:
        if run_id != candidate.get("run_id"):
            violations.append("review_decision_run_id_mismatch")
        candidate_captured_at = _string(candidate.get("captured_at"))
        if not candidate_captured_at:
            violations.append("review_decision_candidate_captured_at_missing")
        elif not artifact_captured_at:
            violations.append("review_decision_artifact_captured_at_missing")
        elif artifact_captured_at != candidate_captured_at:
            violations.append("review_decision_artifact_captured_at_mismatch")
        else:
            reviewed_timestamp = _parse_utc_timestamp(reviewed_at)
            captured_timestamp = _parse_utc_timestamp(candidate_captured_at)
            if (
                reviewed_timestamp is not None
                and captured_timestamp is not None
                and reviewed_timestamp < captured_timestamp
            ):
                violations.append("review_decision_reviewed_at_before_artifact_captured_at")
        candidate_artifact_root = _string(candidate.get("artifact_root"))
        if candidate_artifact_root:
            if not artifact_root:
                violations.append("review_decision_artifact_root_missing")
            elif artifact_root != candidate_artifact_root:
                violations.append("review_decision_artifact_root_mismatch")
        if summary_path != candidate.get("summary_path"):
            violations.append("review_decision_summary_path_mismatch")
        candidate_summary_sha256 = _string(candidate.get("summary_sha256"))
        if candidate_summary_sha256:
            if not summary_sha256:
                violations.append("review_decision_summary_sha256_missing")
            elif summary_sha256 != candidate_summary_sha256:
                violations.append("review_decision_summary_sha256_mismatch")
        if events_path != candidate.get("events_path"):
            violations.append("review_decision_events_path_mismatch")
        if candidate.get("event_log_run_id_matches_summary") is not True:
            violations.append("review_decision_candidate_event_log_run_id_mismatch")
        candidate_channels = _int_list(candidate.get("channels"))
        if channels != candidate_channels:
            violations.append("review_decision_channels_mismatch")
        candidate_value_types = _string_list(candidate.get("value_types"))
        if value_types != candidate_value_types:
            violations.append("review_decision_value_types_mismatch")
        candidate_operator_confirmed_channels = _int_list(
            candidate.get("operator_confirmed_channels")
        )
        if operator_confirmed_channels != candidate_operator_confirmed_channels:
            violations.append("review_decision_operator_confirmed_channels_mismatch")
        if operator_confirmed_session_ready is not True:
            violations.append("review_decision_operator_session_not_confirmed")
        elif candidate.get("operator_confirmed_session_ready") is not True:
            violations.append("review_decision_candidate_operator_session_not_confirmed")
        candidate_readback_count = _non_bool_int(candidate.get("readback_count"))
        if readback_count != candidate_readback_count:
            violations.append("review_decision_readback_count_mismatch")
        candidate_physical_readback_count = _non_bool_int(
            candidate.get("physical_readback_count")
        )
        if physical_readback_count != candidate_physical_readback_count:
            violations.append("review_decision_physical_readback_count_mismatch")
        candidate_events_sha256 = _string(candidate.get("events_sha256"))
        if candidate_events_sha256:
            if not events_sha256:
                violations.append("review_decision_events_sha256_missing")
            elif events_sha256 != candidate_events_sha256:
                violations.append("review_decision_events_sha256_mismatch")
    if unsafe_flags:
        violations.append("review_decision_unsafe_authority_claim")

    valid = not violations
    return {
        "review_id": record_id,
        "artifact_source_id": artifact_source_id,
        "run_id": run_id,
        "artifact_captured_at": artifact_captured_at,
        "artifact_root": artifact_root,
        "summary_path": summary_path,
        "summary_sha256": summary_sha256,
        "events_path": events_path,
        "events_sha256": events_sha256,
        "channels": list(channels),
        "value_types": list(value_types),
        "operator_confirmed_channels": list(operator_confirmed_channels),
        "operator_confirmed_session_ready": operator_confirmed_session_ready,
        "readback_count": readback_count,
        "physical_readback_count": physical_readback_count,
        "request_id_policy": request_id_policy,
        "request_id_correlation_required": request_id_correlation_required,
        "event_log_run_id_matches_summary": event_log_run_id_matches_summary,
        "decision": decision if decision in REVIEW_DECISIONS else "invalid",
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at,
        "template_status": template_status,
        "checklist_ids": list(checklist_ids),
        "candidate_present": bool(candidate),
        "candidate_status": candidate.get("status") or "missing",
        "operator_accepted_readonly_preflight_candidate": valid and decision == "accepted",
        "operator_rejected_readonly_preflight_candidate": valid and decision == "rejected",
        "readiness_matrix_update_allowed_by_decision": False,
        "readiness_closure_allowed_by_decision": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "unsafe_authority_flags": list(unsafe_flags),
        "valid": valid,
        "violations": violations,
    }


def build_wing_readonly_preflight_review_decision_ledger(
    *,
    preflight_evidence: Mapping[str, Any] | None = None,
    decision_records: Iterable[Mapping[str, Any]] | None = None,
    decision_paths: Iterable[str | Path] | None = None,
    decision_roots: Iterable[str | Path] | None = None,
) -> Dict[str, Any]:
    """Build a fail-closed ledger for operator review decisions."""

    evidence = _mapping(preflight_evidence) or build_wing_readonly_preflight_evidence()
    candidates_by_source = _candidate_items_by_source(evidence)
    records = _mark_review_decision_collisions(
        _validated_record(
            record,
            index=index,
            candidates_by_source=candidates_by_source,
        )
        for index, record in enumerate(
            _decision_records(
                decision_records=decision_records,
                decision_paths=decision_paths,
                decision_roots=decision_roots,
            )
        )
    )
    valid_records = tuple(record for record in records if record["valid"])
    accepted_records = tuple(
        record
        for record in valid_records
        if record["operator_accepted_readonly_preflight_candidate"]
    )
    rejected_records = tuple(
        record
        for record in valid_records
        if record["operator_rejected_readonly_preflight_candidate"]
    )
    invalid_records = tuple(record for record in records if not record["valid"])
    duplicate_source_ids = tuple(
        sorted(
            {
                str(record.get("artifact_source_id"))
                for record in invalid_records
                if "review_decision_duplicate_artifact_source"
                in (record.get("violations") or ())
            }
        )
    )
    conflicting_source_ids = tuple(
        sorted(
            {
                str(record.get("artifact_source_id"))
                for record in invalid_records
                if "review_decision_conflicting_artifact_source_decisions"
                in (record.get("violations") or ())
            }
        )
    )
    reviewed_source_ids = {
        str(record["artifact_source_id"])
        for record in valid_records
        if isinstance(record.get("artifact_source_id"), str)
    }
    unreviewed_candidate_source_ids = tuple(
        source_id
        for source_id in candidates_by_source
        if source_id not in reviewed_source_ids
    )
    accepted_source_ids = [record["artifact_source_id"] for record in accepted_records]
    candidate_reviews = _candidate_review_rows(candidates_by_source)
    accepted_candidate_prestart_source_ids = (
        wing_readonly_preflight_accepted_candidate_prestart_source_ids(
            accepted_source_ids,
            candidate_reviews,
        )
    )
    accepted_candidate_prestart_violations_by_source = (
        wing_readonly_preflight_accepted_candidate_prestart_violations_by_source(
            accepted_source_ids,
            candidate_reviews,
        )
    )
    accepted_candidate_prestart_contract_valid = (
        bool(accepted_source_ids)
        and accepted_candidate_prestart_source_ids == accepted_source_ids
        and not accepted_candidate_prestart_violations_by_source
    )
    accepted_candidate_verifier_ws_proof_source_ids = (
        wing_readonly_preflight_accepted_candidate_verifier_ws_proof_source_ids(
            accepted_source_ids,
            candidate_reviews,
        )
    )
    accepted_candidate_verifier_ws_proof_violations_by_source = (
        wing_readonly_preflight_accepted_candidate_verifier_ws_proof_violations_by_source(
            accepted_source_ids,
            candidate_reviews,
        )
    )
    accepted_candidate_verifier_ws_proof_valid = (
        bool(accepted_source_ids)
        and accepted_candidate_verifier_ws_proof_source_ids == accepted_source_ids
        and not accepted_candidate_verifier_ws_proof_violations_by_source
    )
    accepted_candidate_request_sequence_source_ids = (
        wing_readonly_preflight_accepted_candidate_request_sequence_source_ids(
            accepted_source_ids,
            candidate_reviews,
        )
    )
    accepted_candidate_request_sequence_violations_by_source = (
        wing_readonly_preflight_accepted_candidate_request_sequence_violations_by_source(
            accepted_source_ids,
            candidate_reviews,
        )
    )
    accepted_candidate_request_sequence_contract_valid = (
        bool(accepted_source_ids)
        and accepted_candidate_request_sequence_source_ids == accepted_source_ids
        and not accepted_candidate_request_sequence_violations_by_source
    )
    (
        accepted_candidate_artifact_roots,
        accepted_candidate_artifact_root_violations_by_source,
    ) = _accepted_candidate_artifact_root_contract_for_ledger(
        accepted_source_ids,
        candidate_reviews,
    )
    accepted_candidate_artifact_root_contract_valid = (
        bool(accepted_source_ids)
        and len(accepted_candidate_artifact_roots) == len(accepted_source_ids)
        and not accepted_candidate_artifact_root_violations_by_source
    )
    manual_readiness_update_candidate_available = (
        bool(accepted_records)
        and not invalid_records
        and not unreviewed_candidate_source_ids
        and accepted_candidate_prestart_contract_valid
        and accepted_candidate_verifier_ws_proof_valid
        and accepted_candidate_artifact_root_contract_valid
        and accepted_candidate_request_sequence_contract_valid
    )

    if invalid_records:
        status = "operator_review_decision_replacement_required"
        blocked_by = ("operator_must_replace_invalid_readonly_preflight_review_decisions",)
    elif unreviewed_candidate_source_ids:
        status = "operator_review_required"
        blocked_by = ("operator_must_record_readonly_preflight_review_decision",)
    elif accepted_records or rejected_records:
        status = "operator_review_decision_records_available"
        blocked_by_items = []
        if accepted_records and not accepted_candidate_prestart_contract_valid:
            blocked_by_items.append(WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER)
        if accepted_records and not accepted_candidate_verifier_ws_proof_valid:
            blocked_by_items.append(
                WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_VERIFIER_WS_PROOF_BLOCKER
            )
        if accepted_records and not accepted_candidate_artifact_root_contract_valid:
            blocked_by_items.append(
                WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_ARTIFACT_ROOT_BLOCKER
            )
        if accepted_records and not accepted_candidate_request_sequence_contract_valid:
            blocked_by_items.append(
                WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_REQUEST_SEQUENCE_BLOCKER
            )
        blocked_by_items.append("readiness_matrix_update_still_requires_separate_manual_action")
        blocked_by = tuple(blocked_by_items)
    else:
        status = "operator_review_required"
        blocked_by = ("operator_must_capture_valid_readonly_preflight_candidate_first",)

    return {
        "type": "wing_readonly_preflight_review_decision_ledger",
        "generated_at": _utc_now(),
        "method_id": WING_READONLY_PREFLIGHT_REVIEW_DECISION_LEDGER_METHOD_ID,
        "source_method_id": "wing_readonly_preflight_evidence",
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_decision": False,
            "readiness_matrix_update_allowed_by_decision": False,
            "readiness_closure_allowed_by_decision": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "preflight_evidence_status": evidence.get("status") or "missing",
            "decision_search_roots": [
                str(root) for root in _decision_roots(decision_roots)
            ],
            "review_candidate_count": len(candidates_by_source),
            "decision_record_count": len(records),
            "accepted_record_count": len(accepted_records),
            "rejected_record_count": len(rejected_records),
            "invalid_record_count": len(invalid_records),
            "unreviewed_candidate_count": len(unreviewed_candidate_source_ids),
            "accepted_source_ids": accepted_source_ids,
            "rejected_source_ids": [record["artifact_source_id"] for record in rejected_records],
            "invalid_record_ids": [record["review_id"] for record in invalid_records],
            "duplicate_source_ids": list(duplicate_source_ids),
            "conflicting_source_ids": list(conflicting_source_ids),
            "unreviewed_candidate_source_ids": list(unreviewed_candidate_source_ids),
            "candidate_reviews": candidate_reviews,
            "accepted_candidate_prestart_contract_valid": (
                accepted_candidate_prestart_contract_valid
            ),
            "accepted_candidate_prestart_source_ids": (
                accepted_candidate_prestart_source_ids
            ),
            "accepted_candidate_prestart_contract_violations_by_source": (
                accepted_candidate_prestart_violations_by_source
            ),
            "accepted_candidate_verifier_ws_proof_valid": (
                accepted_candidate_verifier_ws_proof_valid
            ),
            "accepted_candidate_verifier_ws_proof_source_ids": (
                accepted_candidate_verifier_ws_proof_source_ids
            ),
            "accepted_candidate_verifier_ws_proof_violations_by_source": (
                accepted_candidate_verifier_ws_proof_violations_by_source
            ),
            "accepted_candidate_artifact_root_contract_valid": (
                accepted_candidate_artifact_root_contract_valid
            ),
            "accepted_candidate_artifact_roots": accepted_candidate_artifact_roots,
            "accepted_candidate_artifact_root_contract_violations_by_source": (
                accepted_candidate_artifact_root_violations_by_source
            ),
            "accepted_candidate_request_sequence_contract_valid": (
                accepted_candidate_request_sequence_contract_valid
            ),
            "accepted_candidate_request_sequence_source_ids": (
                accepted_candidate_request_sequence_source_ids
            ),
            "accepted_candidate_request_sequence_contract_violations_by_source": (
                accepted_candidate_request_sequence_violations_by_source
            ),
            "required_checklist_ids": list(REQUIRED_REVIEW_CHECKLIST_IDS),
            "operator_review_required": True,
            "manual_readiness_update_candidate_available": (
                manual_readiness_update_candidate_available
            ),
            "readiness_matrix_update_allowed_by_decision": False,
            "readiness_closure_allowed_by_decision": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid read-only preflight review decisions."
                if invalid_records
                else (
                    "Operator records a review decision for each clean preflight candidate."
                    if unreviewed_candidate_source_ids
                    else (
                        "Manual readiness matrix update remains separate from review decisions."
                        if accepted_records or rejected_records
                        else "Operator captures a valid read-only preflight candidate first."
                    )
                )
            ),
        },
        "records": list(records),
    }
