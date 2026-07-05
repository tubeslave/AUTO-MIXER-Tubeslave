#!/usr/bin/env python3
"""Verify the M006 owner decision before any readiness-matrix update.

This verifier is offline only. It reads decision/review artifacts from disk,
never opens backend/WING transport, and never grants live authority by itself.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
EXPECTED_PACKET_STATUS = "ready_for_owner_review_after_m005"
EXPECTED_PACKET_VERIFIER_STATUS = "ready_for_owner_review"
EXPECTED_PACKET_VERIFIER_REASON = "m006_owner_review_inputs_verified"
EXPECTED_PACKET_VERIFIER_M005_STALE_EVENT_CHECKS = {
    "m005_events_stale_readback_payload_present",
    "m005_events_stale_readback_matches_summary",
    "m005_events_stale_readback_unavailable",
}
EXPECTED_PACKET_VERIFIER_M005_RECONNECT_EVENT_CHECKS = {
    "m005_events_reconnect_readback_payload_present",
    "m005_events_reconnect_readback_matches_summary",
    "m005_events_reconnect_readback_physical",
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


def _is_repo_relative_m005_summary(path_text: Any) -> bool:
    return _is_repo_relative_path(
        path_text,
        required_parts=("artifacts", "wing_timeout_disconnect_proof"),
        filename="summary.json",
    )


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
    _record(
        checks,
        "packet_verifier_reason_verified",
        packet_verifier.get("reason") == EXPECTED_PACKET_VERIFIER_REASON,
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
    packet_verifier_checks = _get_mapping(packet_verifier, "checks")
    _record(
        checks,
        "packet_verifier_checks_present",
        isinstance(packet_verifier.get("checks"), Mapping),
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


def _record_decision_checks(
    checks: dict[str, bool],
    *,
    decision: Mapping[str, Any],
    packet: Mapping[str, Any],
    decision_summary_path: str | None,
    packet_summary_path: str | None,
    packet_verifier_summary_path: str | None,
    m005_summary_path: str | None,
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
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    _record_packet_checks(
        checks,
        packet=packet,
        packet_summary_path=packet_summary_path,
    )
    _record_packet_verifier_checks(
        checks,
        packet_verifier=packet_verifier,
        packet_verifier_summary_path=packet_verifier_summary_path,
        packet_summary_path=packet_summary_path,
        m005_summary_path=m005_summary_path,
    )
    _record_decision_checks(
        checks,
        decision=decision,
        packet=packet,
        decision_summary_path=decision_summary_path,
        packet_summary_path=packet_summary_path,
        packet_verifier_summary_path=packet_verifier_summary_path,
        m005_summary_path=m005_summary_path,
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
) -> dict[str, Any]:
    decision_path = _resolve(decision_summary)
    packet_path = _resolve(packet_summary)
    packet_verifier_path = _resolve(packet_verifier_summary) if packet_verifier_summary is not None else None
    decision = _load_json(decision_path)
    packet = _load_json(packet_path)
    packet_verifier = _load_json(packet_verifier_path) if packet_verifier_path is not None else None
    return verify_decision(
        decision,
        packet=packet,
        packet_verifier=packet_verifier,
        decision_summary_path=_display_path(decision_path),
        packet_summary_path=_display_path(packet_path),
        packet_verifier_summary_path=_display_path(packet_verifier_path)
        if packet_verifier_path is not None
        else None,
        m005_summary_path=_display_path(m005_summary) if m005_summary is not None else decision.get("m005_summary"),
    )


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
