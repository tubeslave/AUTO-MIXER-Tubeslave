#!/usr/bin/env python3
"""Verify the report-only M006 owner readiness packet before owner review.

This verifier is offline only. It reads packet artifacts from disk, never opens
backend/WING transport, and cannot grant readiness or live authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET_SUMMARY = (
    "artifacts/wing_owner_readiness_review/"
    "20260704T191101Z_m006_owner_readiness_review_packet/summary.json"
)
EXPECTED_PACKET_SCHEMA = "automixer.m006_owner_readiness_review_packet.v1"
EXPECTED_M005_SCHEMA = "wing_timeout_disconnect_proof.v1"
EXPECTED_M005_SUMMARY_GLOB = "artifacts/wing_timeout_disconnect_proof/*/summary.json"
EXPECTED_M005_EVIDENCE = [
    "physical_baseline_readback",
    "operator_confirmed_physical_cable_loss",
    "stale_readback_unavailable",
    "blocked_write_while_disconnected",
    "reconnect_recovery_verified",
]
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


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    return value if isinstance(value, Mapping) else {}


def _record(checks: dict[str, bool], name: str, passed: bool) -> None:
    checks[name] = bool(passed)


def _packet_proofs_by_safety_row(packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    proofs = packet.get("proofs")
    if not isinstance(proofs, list):
        return {}
    return {
        str(row.get("safety_row_id")): row
        for row in proofs
        if isinstance(row, Mapping) and isinstance(row.get("safety_row_id"), str)
    }


def _record_packet_checks(checks: dict[str, bool], packet: Mapping[str, Any]) -> None:
    _record(checks, "packet_schema_matches", packet.get("schema") == EXPECTED_PACKET_SCHEMA)
    _record(checks, "packet_status_is_report_only_ready", packet.get("status") == "ready_for_owner_review_after_m005")
    _record(
        checks,
        "packet_scope_denies_readiness_update",
        packet.get("scope") == "owner_review_packet_only_no_readiness_update",
    )
    for field_name, expected in {**AUTHORITY_DENIAL_FLAGS, **TRANSPORT_DENIAL_FLAGS}.items():
        _record(checks, f"packet_{field_name}_denied", packet.get(field_name) is expected)

    contract = _get_mapping(packet, "m005_completion_contract")
    _record(
        checks,
        "m005_contract_required_summary_glob_matches",
        contract.get("required_summary_glob") == EXPECTED_M005_SUMMARY_GLOB,
    )
    _record(
        checks,
        "m005_contract_required_schema_matches",
        contract.get("required_schema") == EXPECTED_M005_SCHEMA,
    )
    _record(checks, "m005_contract_required_status_passed", contract.get("required_status") == "passed")
    _record(
        checks,
        "m005_contract_requires_complete_true",
        contract.get("required_m005_complete") is True,
    )
    _record(
        checks,
        "m005_contract_requires_no_missing_evidence",
        contract.get("required_missing_m005_evidence") == [],
    )
    _record(
        checks,
        "m005_contract_required_evidence_matches",
        contract.get("required_completed_m005_evidence") == EXPECTED_M005_EVIDENCE,
    )
    _record(
        checks,
        "m005_contract_keeps_authority_denied",
        contract.get("must_keep_authority_denied") == AUTHORITY_DENIAL_FLAGS,
    )

    proofs = _packet_proofs_by_safety_row(packet)
    for row_id, expected in {
        "fader_supervised": {
            "kind": "fader",
            "wing_surface": "channel_fader",
            "proof_direction": "negative_db_move_only",
        },
        "gain_supervised": {
            "kind": "gain",
            "wing_surface": "input_trim",
            "proof_direction": "negative_input_trim_move_only",
        },
    }.items():
        row = proofs.get(row_id, {})
        _record(checks, f"{row_id}_proof_present", bool(row))
        for field_name, expected_value in expected.items():
            _record(checks, f"{row_id}_{field_name}_matches", row.get(field_name) == expected_value)
        _record(checks, f"{row_id}_artifact_passed", row.get("artifact_status") == "passed")
        _record(checks, f"{row_id}_uses_apply_operator_proposal", row.get("via_apply_operator_proposal") is True)
        _record(checks, f"{row_id}_direct_live_write_denied", row.get("direct_live_write_request_sent") is False)
        _record(checks, f"{row_id}_operator_approval_required", row.get("operator_approval_required") is True)
        _record(checks, f"{row_id}_rollback_applied", row.get("rollback_status") == "applied")
        _record(checks, f"{row_id}_emergency_stop_recorded", row.get("emergency_stop_status") == "stopped")
        _record(
            checks,
            f"{row_id}_post_emergency_apply_blocked",
            row.get("post_emergency_apply_status") == "blocked"
            and row.get("post_emergency_apply_reason") == "emergency_stop_active",
        )
        for readback_field in (
            "physical_pre_read",
            "physical_apply_readback",
            "physical_rollback_readback",
        ):
            _record(
                checks,
                f"{row_id}_{readback_field}_physical",
                row.get(readback_field) == "verified_from_physical_readback",
            )
        if row_id == "gain_supervised":
            _record(checks, "gain_supervised_gain_up_denied", row.get("gain_up_allowed") is False)


def _record_m005_checks(
    checks: dict[str, bool],
    *,
    m005_summary: Mapping[str, Any] | None,
) -> None:
    _record(checks, "m005_summary_supplied", m005_summary is not None)
    if m005_summary is None:
        return

    _record(
        checks,
        "m005_summary_schema_matches",
        m005_summary.get("proof_artifact_schema") == EXPECTED_M005_SCHEMA,
    )
    _record(checks, "m005_summary_status_passed", m005_summary.get("status") == "passed")
    _record(checks, "m005_summary_complete_true", m005_summary.get("m005_complete") is True)
    _record(
        checks,
        "m005_summary_required_evidence_matches",
        m005_summary.get("required_m005_evidence") == EXPECTED_M005_EVIDENCE,
    )
    _record(
        checks,
        "m005_summary_completed_evidence_matches",
        m005_summary.get("completed_m005_evidence") == EXPECTED_M005_EVIDENCE,
    )
    _record(
        checks,
        "m005_summary_missing_evidence_empty",
        m005_summary.get("missing_m005_evidence") == [],
    )
    _record(
        checks,
        "m005_summary_remaining_evidence_empty",
        m005_summary.get("remaining_m005_evidence") == [],
    )
    _record(
        checks,
        "m005_summary_physical_cable_loss_mode",
        m005_summary.get("disconnect_mode") == "physical-cable-loss",
    )
    for field_name, expected in AUTHORITY_DENIAL_FLAGS.items():
        _record(checks, f"m005_summary_{field_name}_denied", m005_summary.get(field_name) is expected)
    _record(
        checks,
        "m005_summary_direct_live_write_denied",
        m005_summary.get("direct_live_write_request_sent") is False,
    )
    _record(
        checks,
        "m005_summary_forbidden_requests_empty",
        m005_summary.get("forbidden_request_types_sent") == [],
    )

    phases = _get_mapping(m005_summary, "phases")
    disconnect = _get_mapping(phases, "disconnect")
    stale = _get_mapping(phases, "stale_readback")
    blocked_write = _get_mapping(phases, "blocked_write_while_disconnected")
    reconnect = _get_mapping(phases, "reconnect_recovery")
    _record(
        checks,
        "m005_disconnect_phase_operator_confirmed_physical_loss",
        disconnect.get("status") == "operator_confirmed_physical_cable_loss",
    )
    _record(checks, "m005_stale_readback_phase_passed", stale.get("status") == "passed")
    _record(
        checks,
        "m005_blocked_write_phase_passed",
        blocked_write.get("status") == "passed",
    )
    _record(
        checks,
        "m005_reconnect_recovery_phase_passed",
        reconnect.get("status") == "passed"
        and reconnect.get("unchanged_from_baseline") is True,
    )


def verify_packet(
    packet: Mapping[str, Any],
    *,
    m005_summary: Mapping[str, Any] | None = None,
    packet_summary_path: str | None = None,
    m005_summary_path: str | None = None,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    _record_packet_checks(checks, packet)
    _record_m005_checks(checks, m005_summary=m005_summary)

    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    ready = not failed_checks
    if ready:
        reason = "m006_owner_review_inputs_verified"
    elif m005_summary is None:
        reason = "m005_summary_required_for_owner_review"
    else:
        reason = "m006_owner_review_inputs_failed"

    return {
        "schema": "automixer.m006_owner_readiness_packet_verifier.v1",
        "status": "ready_for_owner_review" if ready else "blocked",
        "reason": reason,
        "packet_summary": packet_summary_path,
        "m005_summary": m005_summary_path,
        "owner_review_input_ready": ready,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "readiness_matrix_update_allowed": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "live_write": False,
        "checks": checks,
        "failed_checks": failed_checks,
    }


def verify_packet_paths(packet_summary: Path, m005_summary: Path | None = None) -> dict[str, Any]:
    packet_path = _resolve(packet_summary)
    m005_path = _resolve(m005_summary) if m005_summary is not None else None
    packet = _load_json(packet_path)
    m005 = _load_json(m005_path) if m005_path is not None else None
    return verify_packet(
        packet,
        m005_summary=m005,
        packet_summary_path=str(packet_summary),
        m005_summary_path=str(m005_summary) if m005_summary is not None else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline fail-closed verifier for the M006 owner readiness packet."
    )
    parser.add_argument("--packet-summary", default=DEFAULT_PACKET_SUMMARY)
    parser.add_argument(
        "--m005-summary",
        help=(
            "Required for owner-review readiness. Must be a complete M005 "
            "disconnect/timeout/stale-state summary artifact."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = verify_packet_paths(
        Path(args.packet_summary),
        Path(args.m005_summary) if args.m005_summary else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ready_for_owner_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
