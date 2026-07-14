#!/usr/bin/env python3
"""Verify the M006 owner decision before any readiness-matrix update.

This verifier is offline only. It reads decision/review artifacts from disk,
never opens backend/WING transport, and never grants live authority by itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from verify_m006_owner_readiness_packet import (  # noqa: E402
    EXPECTED_HISTORICAL_EVENTS_LOGICAL_PATH,
    EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH,
    M005_ROUTE_MIGRATION,
    M005_ROUTE_NATIVE,
    verify_packet_paths,
)
DEFAULT_DECISION_SUMMARY = (
    "artifacts/wing_owner_readiness_review/"
    "20260704T191101Z_m006_owner_readiness_review_packet/owner_decision_template.json"
)
DEFAULT_PACKET_SUMMARY = (
    "artifacts/wing_owner_readiness_review/"
    "20260704T191101Z_m006_owner_readiness_review_packet/summary.json"
)
DEFAULT_ARTIFACT_ROOT = "artifacts/wing_owner_readiness_decision_verifier"
EXPECTED_DECISION_SCHEMA = "automixer.m006_owner_readiness_decision.v1"
EXPECTED_PACKET_SCHEMA = "automixer.m006_owner_readiness_review_packet.v1"
EXPECTED_PACKET_VERIFIER_SCHEMA = "automixer.m006_owner_readiness_packet_verifier.v1"
EXPECTED_ALLOWED_ROW_IDS = ["fader_supervised", "gain_supervised"]
EXPECTED_ALLOWED_KINDS = ["fader", "gain"]
EXPECTED_M005_SUMMARY_GLOB = "artifacts/wing_timeout_disconnect_proof/*/summary.json"
EXPECTED_M005_READBACK_TARGET_BINDING = [
    "baseline_readback",
    "stale_readback",
    "reconnect_readback",
    "events_stale_readback",
    "events_reconnect_readback",
]
EXPECTED_PACKET_STATUS = "ready_for_owner_review_after_m005"
EXPECTED_PACKET_VERIFIER_STATUS = "ready_for_owner_review"
EXPECTED_PACKET_VERIFIER_REASON = "m006_owner_review_inputs_verified"
EXPECTED_PACKET_VERIFIER_MIGRATION_REASON = (
    "m006_owner_review_inputs_verified_via_m005_migration"
)
EXPECTED_PACKET_VERIFIER_M005_STALE_EVENT_CHECKS = {
    "m005_events_stale_readback_payload_present",
    "m005_events_stale_readback_matches_summary",
    "m005_events_stale_readback_target_matches",
    "m005_events_stale_readback_unavailable",
}
EXPECTED_PACKET_VERIFIER_M005_RECONNECT_EVENT_CHECKS = {
    "m005_events_reconnect_readback_payload_present",
    "m005_events_reconnect_readback_matches_summary",
    "m005_events_reconnect_readback_physical",
    "m005_events_reconnect_readback_target_matches",
}
EXPECTED_M005_ACCEPTANCE = {
    "physical_cable_loss": True,
    "stale_readback_unavailable": True,
    "blocked_write_while_disconnected": True,
    "reconnect_recovery_verified": True,
    "rollback_request_sent": False,
    "request_audit_contains_rollback": False,
    "authority_denied_during_m005": True,
}
AUTHORITY_DENIAL_FLAGS = {
    "live_authority_granted": False,
    "readiness_closure_allowed": False,
    "readiness_matrix_update_allowed": False,
}
TRANSPORT_DENIAL_FLAGS = {
    "contacts_wing": False,
    "contacts_backend": False,
    "live_write": False,
}
SCOPE_GUARDRAILS = {
    "gain_up_allowed": False,
    "fader_boosts_allowed": False,
    "direct_write_allowed": False,
    "full_auto_apply_allowed": False,
    "non_fader_gain_surfaces_allowed": False,
}
FORBIDDEN_ROW_IDS = {
    "eq_autoeq",
    "dynamics_compressor",
    "gate_expander",
    "pan_phase_delay",
    "routing_snapshots_scenes",
    "fx_sends_inserts",
    "bus_group_matrix",
    "mute_supervised",
}
FORBIDDEN_KINDS = {
    "compressor",
    "dynamics",
    "eq",
    "fader_boost",
    "full_auto_apply",
    "fx",
    "gain_up",
    "gate",
    "master",
    "mute",
    "pan",
    "routing",
    "scene",
}


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path | str | None) -> str | None:
    if path is None:
        return None
    resolved = _resolve(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _get_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    return value if isinstance(value, Mapping) else {}


def _record(checks: dict[str, bool], name: str, passed: bool) -> None:
    checks[name] = bool(passed)


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _set_equals(value: Any, expected: list[str] | set[str]) -> bool:
    if not isinstance(value, list):
        return False
    strings = _as_string_list(value)
    return len(strings) == len(value) and set(strings) == set(expected)


def _is_repo_relative_path(path_text: Any, *, required_parts: tuple[str, ...], filename: str) -> bool:
    if not isinstance(path_text, str) or not path_text:
        return False
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts:
        return False
    return (
        len(path.parts) >= len(required_parts) + 2
        and tuple(path.parts[: len(required_parts)]) == required_parts
        and path.name == filename
    )


def _is_repo_relative_packet_summary(path_text: Any) -> bool:
    return _is_repo_relative_path(
        path_text,
        required_parts=("artifacts", "wing_owner_readiness_review"),
        filename="summary.json",
    )


def _is_repo_relative_decision_summary(path_text: Any) -> bool:
    if not isinstance(path_text, str) or not path_text:
        return False
    path = Path(path_text)
    filename_allowed = (
        path.name in {"owner_decision_template.json", "owner_decision.json"}
        or (path.name.startswith("owner_decision_") and path.suffix == ".json")
    )
    if not filename_allowed:
        return False
    return _is_repo_relative_path(
        path_text,
        required_parts=("artifacts", "wing_owner_readiness_review"),
        filename=path.name,
    )


def _is_repo_relative_packet_verifier_summary(path_text: Any) -> bool:
    return _is_repo_relative_path(
        path_text,
        required_parts=("artifacts", "wing_owner_readiness_review_verifier"),
        filename="summary.json",
    )


def _is_repo_relative_m005_migration_summary(path_text: Any) -> bool:
    if not isinstance(path_text, str):
        return False
    path = Path(path_text)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and len(path.parts) >= 4
        and path.parts[0] == "artifacts"
        and path.parts[1] == "wing_m005_current_release_migration"
        and path.name == "summary.json"
    )


def _is_repo_relative_m005_summary(path_text: Any) -> bool:
    return _is_repo_relative_path(
        path_text,
        required_parts=("artifacts", "wing_timeout_disconnect_proof"),
        filename="summary.json",
    )


def _is_repo_relative_m005_events(path_text: Any) -> bool:
    return _is_repo_relative_path(
        path_text,
        required_parts=("artifacts", "wing_timeout_disconnect_proof"),
        filename="events.jsonl",
    )


def _m005_events_path_for_summary(path_text: Any) -> str | None:
    if not _is_repo_relative_m005_summary(path_text):
        return None
    return str(Path(path_text).with_name("events.jsonl"))


def _is_utc_z_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return True


def _packet_row_ids(packet: Mapping[str, Any]) -> list[str]:
    rows = packet.get("proofs")
    if not isinstance(rows, list):
        return []
    return [
        row.get("safety_row_id")
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("safety_row_id"), str)
    ]


def _packet_kinds(packet: Mapping[str, Any]) -> list[str]:
    rows = packet.get("proofs")
    if not isinstance(rows, list):
        return []
    return [
        row.get("kind")
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("kind"), str)
    ]


def _packet_proof_artifacts(packet: Mapping[str, Any]) -> list[str]:
    return _as_string_list(packet.get("input_proof_artifacts"))


def _record_packet_checks(
    checks: dict[str, bool],
    *,
    packet: Mapping[str, Any],
    packet_summary_path: str | None,
) -> None:
    _record(checks, "packet_summary_path_supplied", isinstance(packet_summary_path, str) and bool(packet_summary_path))
    _record(checks, "packet_summary_path_repo_relative", _is_repo_relative_packet_summary(packet_summary_path))
    _record(checks, "packet_schema_matches", packet.get("schema") == EXPECTED_PACKET_SCHEMA)
    _record(checks, "packet_status_ready_after_m005", packet.get("status") == EXPECTED_PACKET_STATUS)
    _record(
        checks,
        "packet_scope_report_only",
        packet.get("scope") == "owner_review_packet_only_no_readiness_update",
    )
    for field_name, expected in {**AUTHORITY_DENIAL_FLAGS, **TRANSPORT_DENIAL_FLAGS}.items():
        _record(checks, f"packet_{field_name}_denied", packet.get(field_name) is expected)

    contract = _get_mapping(packet, "m005_completion_contract")
    _record(
        checks,
        "packet_requires_m005_summary_glob",
        contract.get("required_summary_glob") == EXPECTED_M005_SUMMARY_GLOB,
    )
    _record(
        checks,
        "packet_requires_m005_complete",
        contract.get("required_m005_complete") is True,
    )
    _record(
        checks,
        "packet_requires_no_missing_m005_evidence",
        contract.get("required_missing_m005_evidence") == [],
    )
    _record(
        checks,
        "packet_requires_no_m005_rollback_request",
        contract.get("required_rollback_request_sent") is False,
    )
    _record(
        checks,
        "packet_requires_m005_readback_target_binding",
        contract.get("required_readback_target_binding")
        == EXPECTED_M005_READBACK_TARGET_BINDING,
    )

    _record(checks, "packet_rows_match_allowed_rows", _set_equals(_packet_row_ids(packet), EXPECTED_ALLOWED_ROW_IDS))
    _record(checks, "packet_kinds_match_allowed_kinds", _set_equals(_packet_kinds(packet), EXPECTED_ALLOWED_KINDS))
    proofs = packet.get("proofs")
    packet_artifacts = _packet_proof_artifacts(packet)
    proof_row_artifacts = (
        [
            row.get("artifact")
            for row in proofs
            if isinstance(row, Mapping) and isinstance(row.get("artifact"), str)
        ]
        if isinstance(proofs, list)
        else []
    )
    _record(
        checks,
        "packet_proof_artifacts_match_rows",
        set(packet_artifacts) == set(proof_row_artifacts)
        and len(packet_artifacts) == len(proof_row_artifacts),
    )


def _record_packet_verifier_checks(
    checks: dict[str, bool],
    *,
    packet_verifier: Mapping[str, Any] | None,
    packet_verifier_summary_path: str | None,
    packet_summary_path: str | None,
    m005_summary_path: str | None,
    m005_migration_summary_path: str | None,
) -> None:
    _record(
        checks,
        "packet_verifier_summary_supplied",
        isinstance(packet_verifier_summary_path, str) and bool(packet_verifier_summary_path),
    )
    _record(
        checks,
        "packet_verifier_summary_path_repo_relative",
        _is_repo_relative_packet_verifier_summary(packet_verifier_summary_path),
    )
    _record(checks, "packet_verifier_loaded", isinstance(packet_verifier, Mapping))
    if not isinstance(packet_verifier, Mapping):
        return

    _record(
        checks,
        "packet_verifier_schema_matches",
        packet_verifier.get("schema") == EXPECTED_PACKET_VERIFIER_SCHEMA,
    )
    _record(
        checks,
        "packet_verifier_status_ready",
        packet_verifier.get("status") == EXPECTED_PACKET_VERIFIER_STATUS,
    )
    route = packet_verifier.get("m005_acceptance_route")
    if route is None and (
        packet_verifier.get("m005_provenance_ready") is True
        and packet_verifier.get("m005_current_release_accepted") is True
    ):
        route = M005_ROUTE_NATIVE
    _record(
        checks,
        "packet_verifier_m005_acceptance_route_valid",
        route in {M005_ROUTE_NATIVE, M005_ROUTE_MIGRATION},
    )
    _record(
        checks,
        "packet_verifier_reason_verified",
        packet_verifier.get("reason")
        == (
            EXPECTED_PACKET_VERIFIER_MIGRATION_REASON
            if route == M005_ROUTE_MIGRATION
            else EXPECTED_PACKET_VERIFIER_REASON
        ),
    )
    _record(
        checks,
        "packet_verifier_failed_checks_empty",
        packet_verifier.get("failed_checks") == [],
    )
    _record(
        checks,
        "packet_verifier_owner_review_input_ready",
        packet_verifier.get("owner_review_input_ready") is True,
    )
    _record(
        checks,
        "packet_verifier_m005_m006_input_accepted",
        packet_verifier.get("m005_m006_input_accepted") is True,
    )
    packet_verifier_checks = _get_mapping(packet_verifier, "checks")
    _record(
        checks,
        "packet_verifier_checks_present",
        isinstance(packet_verifier.get("checks"), Mapping),
    )
    if route == M005_ROUTE_NATIVE:
        for field_name in (
            "m005_behavioral_proof_complete",
            "m005_owner_review_shape_ready",
            "m005_provenance_ready",
            "m005_current_release_accepted",
        ):
            _record(
                checks,
                f"packet_verifier_{field_name}",
                packet_verifier.get(field_name) is True,
            )
        provenance_check_names = [
            name
            for name in packet_verifier_checks
            if name.startswith("m005_provenance_")
        ]
        _record(
            checks,
            "packet_verifier_m005_provenance_checks_present_and_true",
            bool(provenance_check_names)
            and all(
                packet_verifier_checks.get(name) is True
                for name in provenance_check_names
            ),
        )
        for check_name in sorted(
            EXPECTED_PACKET_VERIFIER_M005_STALE_EVENT_CHECKS
            | EXPECTED_PACKET_VERIFIER_M005_RECONNECT_EVENT_CHECKS
        ):
            _record(
                checks,
                f"packet_verifier_{check_name}",
                packet_verifier_checks.get(check_name) is True,
            )
        _record(
            checks,
            "packet_verifier_native_migration_path_absent",
            packet_verifier.get("m005_migration_summary") is None
            and m005_migration_summary_path is None,
        )
    elif route == M005_ROUTE_MIGRATION:
        _record(
            checks,
            "packet_verifier_migration_ready",
            packet_verifier.get("m005_migration_ready") is True,
        )
        _record(
            checks,
            "packet_verifier_migration_native_claims_false",
            packet_verifier.get("m005_provenance_ready") is False
            and packet_verifier.get("m005_current_release_accepted") is False,
        )
        _record(
            checks,
            "packet_verifier_migration_failures_empty",
            packet_verifier.get("m005_migration_failed_checks") == [],
        )
        migration_checks = _get_mapping(packet_verifier, "m005_migration_checks")
        _record(
            checks,
            "packet_verifier_migration_checks_present_and_true",
            bool(migration_checks)
            and all(value is True for value in migration_checks.values()),
        )
        _record(
            checks,
            "packet_verifier_migration_summary_matches",
            packet_verifier.get("m005_migration_summary")
            == m005_migration_summary_path
            and _is_repo_relative_m005_migration_summary(
                m005_migration_summary_path
            ),
        )
        _record(
            checks,
            "packet_verifier_migration_legacy_paths_exact",
            m005_summary_path == EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH
            and packet_verifier.get("m005_summary")
            == EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH
            and packet_verifier.get("m005_events")
            == EXPECTED_HISTORICAL_EVENTS_LOGICAL_PATH,
        )
    for field_name, expected in {**AUTHORITY_DENIAL_FLAGS, **TRANSPORT_DENIAL_FLAGS}.items():
        _record(checks, f"packet_verifier_{field_name}_denied", packet_verifier.get(field_name) is expected)
    _record(
        checks,
        "packet_verifier_packet_summary_matches",
        packet_verifier.get("packet_summary") == packet_summary_path,
    )
    _record(
        checks,
        "packet_verifier_m005_summary_matches",
        packet_verifier.get("m005_summary") == m005_summary_path,
    )
    _record(
        checks,
        "packet_verifier_m005_events_path_repo_relative",
        _is_repo_relative_m005_events(packet_verifier.get("m005_events")),
    )
    _record(
        checks,
        "packet_verifier_m005_events_matches_summary_sibling",
        packet_verifier.get("m005_events") == _m005_events_path_for_summary(m005_summary_path),
    )


def _record_decision_checks(
    checks: dict[str, bool],
    *,
    decision: Mapping[str, Any],
    packet: Mapping[str, Any],
    decision_summary_path: str | None,
    packet_summary_path: str | None,
    packet_verifier_summary_path: str | None,
    m005_summary_path: str | None,
    m005_migration_summary_path: str | None,
    m005_acceptance_route: str | None,
) -> None:
    decision_value = decision.get("decision")
    _record(checks, "decision_summary_path_supplied", isinstance(decision_summary_path, str) and bool(decision_summary_path))
    _record(checks, "decision_summary_path_repo_relative", _is_repo_relative_decision_summary(decision_summary_path))
    _record(checks, "decision_schema_matches", decision.get("schema") == EXPECTED_DECISION_SCHEMA)
    _record(checks, "decision_is_accepted", decision_value == "accepted")
    _record(
        checks,
        "decision_status_matches_acceptance",
        decision.get("status") == "accepted_for_readiness_matrix_update",
    )
    _record(
        checks,
        "decision_reviewed_by_present",
        isinstance(decision.get("reviewed_by"), str) and bool(str(decision.get("reviewed_by")).strip()),
    )
    _record(checks, "decision_reviewed_at_utc_valid", _is_utc_z_timestamp(decision.get("reviewed_at_utc")))
    _record(checks, "decision_packet_summary_matches", decision.get("packet_summary") == packet_summary_path)
    _record(
        checks,
        "decision_packet_verifier_summary_matches",
        decision.get("packet_verifier_summary") == packet_verifier_summary_path,
    )
    _record(checks, "decision_m005_summary_matches", decision.get("m005_summary") == m005_summary_path)
    _record(checks, "decision_m005_summary_repo_relative", _is_repo_relative_m005_summary(decision.get("m005_summary")))
    if m005_acceptance_route == M005_ROUTE_MIGRATION:
        _record(
            checks,
            "decision_m005_migration_legacy_summary_exact",
            m005_summary_path == EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH
            and decision.get("m005_summary")
            == EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH,
        )
        _record(
            checks,
            "decision_m005_acceptance_route_matches_migration",
            decision.get("m005_acceptance_route") == M005_ROUTE_MIGRATION,
        )
        _record(
            checks,
            "decision_m005_migration_summary_matches",
            decision.get("m005_migration_summary")
            == m005_migration_summary_path
            and _is_repo_relative_m005_migration_summary(
                m005_migration_summary_path
            ),
        )
    else:
        _record(
            checks,
            "decision_m005_acceptance_route_native_or_legacy",
            decision.get("m005_acceptance_route")
            in {None, M005_ROUTE_NATIVE},
        )
        _record(
            checks,
            "decision_m005_migration_summary_absent_for_native",
            decision.get("m005_migration_summary") is None
            and m005_migration_summary_path is None,
        )

    accepted_rows = decision.get("accepted_safety_row_ids")
    accepted_kinds = decision.get("accepted_live_write_kinds")
    accepted_artifacts = decision.get("accepted_proof_artifacts")
    _record(checks, "decision_accepted_rows_match_allowed", _set_equals(accepted_rows, EXPECTED_ALLOWED_ROW_IDS))
    _record(checks, "decision_accepted_kinds_match_allowed", _set_equals(accepted_kinds, EXPECTED_ALLOWED_KINDS))
    _record(
        checks,
        "decision_accepted_artifacts_match_packet",
        _set_equals(accepted_artifacts, _packet_proof_artifacts(packet)),
    )
    _record(
        checks,
        "decision_no_forbidden_rows_accepted",
        not (set(_as_string_list(accepted_rows)) & FORBIDDEN_ROW_IDS),
    )
    _record(
        checks,
        "decision_no_forbidden_kinds_accepted",
        not (set(_as_string_list(accepted_kinds)) & FORBIDDEN_KINDS),
    )

    rejected_rows = decision.get("rejected_safety_row_ids")
    _record(
        checks,
        "decision_rejects_all_forbidden_rows",
        set(_as_string_list(rejected_rows)).issuperset(FORBIDDEN_ROW_IDS),
    )

    m005_acceptance = _get_mapping(decision, "m005_acceptance")
    for field_name, expected in EXPECTED_M005_ACCEPTANCE.items():
        _record(
            checks,
            f"decision_m005_acceptance_{field_name}",
            m005_acceptance.get(field_name) is expected,
        )

    scope_guardrails = _get_mapping(decision, "scope_guardrails")
    for field_name, expected in SCOPE_GUARDRAILS.items():
        _record(
            checks,
            f"decision_scope_guardrail_{field_name}_denied",
            scope_guardrails.get(field_name) is expected,
        )

    for field_name, expected in {**AUTHORITY_DENIAL_FLAGS, **TRANSPORT_DENIAL_FLAGS}.items():
        _record(checks, f"decision_{field_name}_denied", decision.get(field_name) is expected)

    _record(
        checks,
        "decision_candidate_flag_true",
        decision.get("readiness_matrix_update_candidate") is True,
    )
    _record(
        checks,
        "decision_does_not_claim_live_authority",
        decision.get("live_authority_granted") is False
        and decision.get("live_write") is False,
    )


def verify_decision(
    decision: Mapping[str, Any],
    *,
    packet: Mapping[str, Any],
    packet_verifier: Mapping[str, Any] | None = None,
    decision_summary_path: str | None = None,
    packet_summary_path: str | None = None,
    packet_verifier_summary_path: str | None = None,
    m005_summary_path: str | None = None,
    m005_migration_summary_path: str | None = None,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    _record_packet_checks(
        checks,
        packet=packet,
        packet_summary_path=packet_summary_path,
    )
    packet_route = (
        packet_verifier.get("m005_acceptance_route")
        if isinstance(packet_verifier, Mapping)
        else None
    )
    if packet_route is None and isinstance(packet_verifier, Mapping) and (
        packet_verifier.get("m005_provenance_ready") is True
        and packet_verifier.get("m005_current_release_accepted") is True
    ):
        packet_route = M005_ROUTE_NATIVE
    _record_packet_verifier_checks(
        checks,
        packet_verifier=packet_verifier,
        packet_verifier_summary_path=packet_verifier_summary_path,
        packet_summary_path=packet_summary_path,
        m005_summary_path=m005_summary_path,
        m005_migration_summary_path=m005_migration_summary_path,
    )
    _record_decision_checks(
        checks,
        decision=decision,
        packet=packet,
        decision_summary_path=decision_summary_path,
        packet_summary_path=packet_summary_path,
        packet_verifier_summary_path=packet_verifier_summary_path,
        m005_summary_path=m005_summary_path,
        m005_migration_summary_path=m005_migration_summary_path,
        m005_acceptance_route=packet_route,
    )

    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    ready = not failed_checks
    decision_value = decision.get("decision")
    if ready:
        reason = "m006_owner_decision_accepts_narrow_readiness_rows"
    elif decision_value == "pending_owner_review":
        reason = "m006_owner_decision_pending"
    elif decision_value == "rejected":
        reason = "m006_owner_decision_rejected"
    else:
        reason = "m006_owner_decision_failed_closed"

    return {
        "schema": "automixer.m006_owner_readiness_decision_verifier.v1",
        "generated_at_utc": _iso_now(),
        "status": "ready_for_readiness_matrix_update" if ready else "blocked",
        "reason": reason,
        "decision_summary": decision_summary_path,
        "packet_summary": packet_summary_path,
        "packet_verifier_summary": packet_verifier_summary_path,
        "m005_summary": m005_summary_path,
        "m005_migration_summary": m005_migration_summary_path,
        "m005_acceptance_route": packet_route if ready else None,
        "m005_m006_input_accepted": ready,
        "owner_decision_accepted": ready,
        "readiness_matrix_update_candidate_ready": ready,
        "accepted_safety_row_ids": EXPECTED_ALLOWED_ROW_IDS if ready else [],
        "accepted_live_write_kinds": EXPECTED_ALLOWED_KINDS if ready else [],
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "readiness_matrix_update_allowed": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "live_write": False,
        "checks": checks,
        "failed_checks": failed_checks,
    }


def verify_decision_paths(
    decision_summary: Path,
    packet_summary: Path,
    packet_verifier_summary: Path | None = None,
    m005_summary: Path | None = None,
    m005_migration_summary: Path | None = None,
) -> dict[str, Any]:
    decision_path = _resolve(decision_summary)
    packet_path = _resolve(packet_summary)
    packet_verifier_path = _resolve(packet_verifier_summary) if packet_verifier_summary is not None else None
    decision = _load_json(decision_path)
    packet = _load_json(packet_path)
    saved_packet_verifier = (
        _load_json(packet_verifier_path) if packet_verifier_path is not None else None
    )
    effective_m005_path = (
        _resolve(m005_summary)
        if m005_summary is not None
        else _resolve(decision.get("m005_summary"))
        if isinstance(decision.get("m005_summary"), str)
        else None
    )
    effective_migration_path = (
        _resolve(m005_migration_summary)
        if m005_migration_summary is not None
        else _resolve(decision.get("m005_migration_summary"))
        if isinstance(decision.get("m005_migration_summary"), str)
        else None
    )
    recomputed_packet_verifier = (
        verify_packet_paths(
            Path(_display_path(packet_path)),
            Path(_display_path(effective_m005_path)),
            (
                Path(_display_path(effective_migration_path))
                if effective_migration_path is not None
                else None
            ),
        )
        if packet_verifier_path is not None and effective_m005_path is not None
        else None
    )
    result = verify_decision(
        decision,
        packet=packet,
        packet_verifier=recomputed_packet_verifier,
        decision_summary_path=_display_path(decision_path),
        packet_summary_path=_display_path(packet_path),
        packet_verifier_summary_path=_display_path(packet_verifier_path)
        if packet_verifier_path is not None
        else None,
        m005_summary_path=(
            _display_path(effective_m005_path)
            if effective_m005_path is not None
            else decision.get("m005_summary")
        ),
        m005_migration_summary_path=(
            _display_path(effective_migration_path)
            if effective_migration_path is not None
            else None
        ),
    )
    result["packet_verifier_recomputed_from_current_artifacts"] = bool(
        recomputed_packet_verifier is not None
    )
    result["saved_packet_verifier_status"] = (
        saved_packet_verifier.get("status")
        if isinstance(saved_packet_verifier, Mapping)
        else None
    )
    return result


def write_verifier_artifact(
    result: Mapping[str, Any],
    *,
    artifact_root: Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve(artifact_root)
    effective_run_id = run_id or f"{_utc_stamp()}_m006_owner_decision_verifier"
    run_dir = resolved_root / effective_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    if artifact_root.is_absolute():
        display_summary_path = str(summary_path)
        display_artifact_root = str(resolved_root)
    else:
        display_summary_path = str(artifact_root / effective_run_id / "summary.json")
        display_artifact_root = str(artifact_root)

    output = dict(result)
    output.update(
        {
            "artifact_root": display_artifact_root,
            "run_id": effective_run_id,
            "summary_path": display_summary_path,
            "artifact_written": True,
        }
    )
    summary_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline fail-closed verifier for the M006 owner readiness decision."
    )
    parser.add_argument("--decision-summary", default=DEFAULT_DECISION_SUMMARY)
    parser.add_argument("--packet-summary", default=DEFAULT_PACKET_SUMMARY)
    parser.add_argument(
        "--packet-verifier-summary",
        help="Required accepted M006 packet verifier summary from artifacts/wing_owner_readiness_review_verifier/.",
    )
    parser.add_argument(
        "--m005-summary",
        help=(
            "Accepted M005 disconnect/timeout/stale-state summary. If omitted, "
            "the decision's m005_summary field is used for path binding checks."
        ),
    )
    parser.add_argument(
        "--m005-migration-summary",
        help=(
            "Explicit verified M005 migration summary. Required when the "
            "owner decision selects the verified_migration route."
        ),
    )
    parser.add_argument(
        "--write-artifact",
        action="store_true",
        help="Write decision-verifier summary.json under --artifact-root/<run-id>/.",
    )
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-id", help="Optional deterministic run id for artifact output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = verify_decision_paths(
        Path(args.decision_summary),
        Path(args.packet_summary),
        Path(args.packet_verifier_summary) if args.packet_verifier_summary else None,
        Path(args.m005_summary) if args.m005_summary else None,
        (
            Path(args.m005_migration_summary)
            if args.m005_migration_summary
            else None
        ),
    )
    if args.write_artifact:
        result = write_verifier_artifact(
            result,
            artifact_root=Path(args.artifact_root),
            run_id=args.run_id,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ready_for_readiness_matrix_update" else 2


if __name__ == "__main__":
    raise SystemExit(main())
