#!/usr/bin/env python3
"""Verify the report-only M006 owner readiness packet before owner review.

This verifier is offline only. It reads packet artifacts from disk, never opens
backend/WING transport, and cannot grant readiness or live authority.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET_SUMMARY = (
    "artifacts/wing_owner_readiness_review/"
    "20260704T191101Z_m006_owner_readiness_review_packet/summary.json"
)
DEFAULT_ARTIFACT_ROOT = "artifacts/wing_owner_readiness_review_verifier"
EXPECTED_PACKET_SCHEMA = "automixer.m006_owner_readiness_review_packet.v1"
EXPECTED_M005_SCHEMA = "wing_timeout_disconnect_proof.v1"
EXPECTED_M005_SUMMARY_GLOB = "artifacts/wing_timeout_disconnect_proof/*/summary.json"
EXPECTED_M005_PHYSICAL_CABLE_CONFIRMATION = "WING_CABLE_PULLED_BY_OPERATOR"
EXPECTED_PROOF_SCHEMA = "wing_operator_proposal_validation.v2"
EXPECTED_PROOF_REASON = "proposal_apply_readback_rollback_complete"
EXPECTED_M005_EVIDENCE = [
    "physical_baseline_readback",
    "operator_confirmed_physical_cable_loss",
    "stale_readback_unavailable",
    "blocked_write_while_disconnected",
    "reconnect_recovery_verified",
]
EXPECTED_M005_SAFE_BLOCK_REASONS = {
    "manual_write_exception",
    "mixer_not_connected",
    "mixer_write_failed",
    "pre_read_cache_not_fresh",
    "pre_read_unavailable",
}
EXPECTED_M005_FORBIDDEN_DIRECT_REQUEST_TYPES = {
    "load_" + "snap",
    "route_" + "output",
    "save_" + "snap",
    "set_" + "fader",
    "set_" + "gain",
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
EXPECTED_PROOF_ROWS = {
    "fader_supervised": {
        "kind": "fader",
        "wing_surface": "channel_fader",
        "wing_address": "/ch/3/fdr",
        "proof_direction": "negative_db_move_only",
        "method": "set_channel_fader",
    },
    "gain_supervised": {
        "kind": "gain",
        "wing_surface": "input_trim",
        "wing_address": "/ch/3/in/set/trim",
        "proof_direction": "negative_input_trim_move_only",
        "method": "set_channel_gain",
    },
}
PHYSICAL_READBACK_STATUS = "verified_from_physical_readback"
PHYSICAL_READBACK_SOURCE = "physical_readback"


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _get_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    return value if isinstance(value, Mapping) else {}


def _record(checks: dict[str, bool], name: str, passed: bool) -> None:
    checks[name] = bool(passed)


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 1e-6
    return left == right


def _m005_readback_is_physical(readback: Mapping[str, Any]) -> bool:
    return (
        readback.get("type") == "mixer_parameter_value"
        and readback.get("status") == PHYSICAL_READBACK_STATUS
        and readback.get("source") == PHYSICAL_READBACK_SOURCE
        and readback.get("value") is not None
    )


def _m005_readback_is_unavailable(readback: Mapping[str, Any]) -> bool:
    if readback.get("type") == "mixer_parameter_read_blocked":
        return True
    return (
        readback.get("type") == "mixer_parameter_value"
        and readback.get("status") == "readback_unavailable"
        and readback.get("value") is None
        and readback.get("source") in {"physical_readback", "unavailable"}
    )


def _is_repo_relative_proof_path(path_text: Any) -> bool:
    if not isinstance(path_text, str):
        return False
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts:
        return False
    return (
        len(path.parts) >= 4
        and path.parts[0] == "artifacts"
        and path.parts[1] == "wing_operator_proposal_validation"
        and path.name == "summary.json"
    )


def _packet_input_proof_artifacts(packet: Mapping[str, Any]) -> list[str]:
    artifacts = packet.get("input_proof_artifacts")
    if not isinstance(artifacts, list):
        return []
    return [item for item in artifacts if isinstance(item, str)]


def _packet_proofs_by_safety_row(packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    proofs = packet.get("proofs")
    if not isinstance(proofs, list):
        return {}
    return {
        str(row.get("safety_row_id")): row
        for row in proofs
        if isinstance(row, Mapping) and isinstance(row.get("safety_row_id"), str)
    }


def _proof_artifact_paths_by_safety_row(packet: Mapping[str, Any]) -> dict[str, str]:
    rows = _packet_proofs_by_safety_row(packet)
    return {
        row_id: str(row.get("artifact"))
        for row_id, row in rows.items()
        if isinstance(row.get("artifact"), str)
    }


def load_packet_proof_summaries(packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Load proof summaries explicitly referenced by the packet.

    This intentionally does not glob for alternates. Missing, absolute,
    parent-traversing, or out-of-root paths are left unloaded so verification
    fails closed through the corresponding checks.
    """

    loaded: dict[str, Mapping[str, Any]] = {}
    for artifact in _proof_artifact_paths_by_safety_row(packet).values():
        if not _is_repo_relative_proof_path(artifact):
            continue
        path = _resolve(artifact)
        if not path.is_file():
            continue
        loaded[artifact] = _load_json(path)
    return loaded


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
    row_artifacts = _proof_artifact_paths_by_safety_row(packet)
    _record(
        checks,
        "packet_input_proof_artifacts_match_proof_rows",
        sorted(_packet_input_proof_artifacts(packet)) == sorted(row_artifacts.values())
        and len(_packet_input_proof_artifacts(packet)) == len(row_artifacts),
    )
    for row_id, expected in EXPECTED_PROOF_ROWS.items():
        row = proofs.get(row_id, {})
        _record(checks, f"{row_id}_proof_present", bool(row))
        for field_name, expected_value in expected.items():
            if field_name == "method":
                continue
            _record(checks, f"{row_id}_{field_name}_matches", row.get(field_name) == expected_value)
        _record(checks, f"{row_id}_artifact_path_present", isinstance(row.get("artifact"), str))
        _record(checks, f"{row_id}_artifact_path_repo_relative", _is_repo_relative_proof_path(row.get("artifact")))
        _record(
            checks,
            f"{row_id}_artifact_in_input_proof_artifacts",
            row.get("artifact") in _packet_input_proof_artifacts(packet),
        )
        _record(checks, f"{row_id}_artifact_passed", row.get("artifact_status") == "passed")
        _record(checks, f"{row_id}_artifact_reason_complete", row.get("artifact_reason") == EXPECTED_PROOF_REASON)
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


def _record_proof_artifact_checks(
    checks: dict[str, bool],
    packet: Mapping[str, Any],
    *,
    proof_summaries_by_artifact: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    proofs = _packet_proofs_by_safety_row(packet)
    loaded = proof_summaries_by_artifact or {}
    for row_id, expected in EXPECTED_PROOF_ROWS.items():
        row = proofs.get(row_id, {})
        artifact = row.get("artifact")
        summary = loaded.get(artifact) if isinstance(artifact, str) else None
        _record(checks, f"{row_id}_artifact_summary_loaded", isinstance(summary, Mapping))
        if not isinstance(summary, Mapping):
            continue

        apply_result = _get_mapping(_get_mapping(summary, "apply_result"), "apply_result")
        rollback_result = _get_mapping(summary, "rollback_result")
        emergency_result = _get_mapping(summary, "emergency_stop_result")
        post_emergency = _get_mapping(
            _get_mapping(summary, "post_emergency_apply_result"),
            "apply_result",
        )
        blocked_before = _get_mapping(_get_mapping(summary, "blocked_apply_result"), "apply_result")
        blocked_before_outer = _get_mapping(summary, "blocked_apply_result")
        surface_contract = _get_mapping(summary, "proof_surface_contract")
        sent_request_types = summary.get("sent_request_types")

        _record(checks, f"{row_id}_artifact_schema_matches", summary.get("proof_artifact_schema") == EXPECTED_PROOF_SCHEMA)
        _record(checks, f"{row_id}_artifact_status_passed", summary.get("status") == "passed")
        _record(checks, f"{row_id}_artifact_reason_complete", summary.get("reason") == EXPECTED_PROOF_REASON)
        _record(checks, f"{row_id}_artifact_summary_path_matches_packet", summary.get("summary_path") == artifact)
        _record(checks, f"{row_id}_artifact_kind_matches_packet", summary.get("value_type") == row.get("kind") == expected["kind"])
        _record(
            checks,
            f"{row_id}_artifact_channel_matches_packet",
            summary.get("operator_confirmed_channel") == row.get("channel")
            and apply_result.get("channel") == row.get("channel")
            and rollback_result.get("channel") == row.get("channel")
            and post_emergency.get("channel") == row.get("channel"),
        )
        _record(
            checks,
            f"{row_id}_artifact_surface_matches_packet",
            summary.get("wing_surface") == row.get("wing_surface") == expected["wing_surface"]
            and surface_contract.get("wing_surface") == row.get("wing_surface"),
        )
        _record(
            checks,
            f"{row_id}_artifact_address_matches_packet",
            summary.get("wing_address") == row.get("wing_address") == expected["wing_address"]
            and summary.get("address") == row.get("wing_address")
            and surface_contract.get("wing_address") == row.get("wing_address"),
        )
        _record(
            checks,
            f"{row_id}_artifact_proof_direction_matches_packet",
            summary.get("proof_direction") == row.get("proof_direction") == expected["proof_direction"]
            and surface_contract.get("proof_direction") == row.get("proof_direction"),
        )
        _record(
            checks,
            f"{row_id}_artifact_method_matches_packet",
            apply_result.get("method") == row.get("method") == expected["method"]
            and rollback_result.get("method") == row.get("method"),
        )
        _record(checks, f"{row_id}_artifact_write_policy_apply_operator_proposal_only", summary.get("write_policy") == "apply_operator_proposal_only")
        _record(checks, f"{row_id}_artifact_forbidden_requests_empty", summary.get("forbidden_request_types_sent") == [])
        _record(
            checks,
            f"{row_id}_artifact_uses_apply_operator_proposal",
            isinstance(sent_request_types, list)
            and "apply_operator_proposal" in sent_request_types
            and isinstance(apply_result.get("source"), str)
            and str(apply_result.get("source")).startswith("operator_proposal:"),
        )
        _record(
            checks,
            f"{row_id}_artifact_operator_confirmation_matches_packet",
            summary.get("operator_confirmation_required") is True
            and summary.get("operator_confirmation_match") is True
            and summary.get("operator_confirmed_channel") == row.get("channel")
            and summary.get("operator_confirmed_value_type") == row.get("kind")
            and _same_value(summary.get("operator_confirmed_delta_db"), row.get("requested_value")),
        )
        _record(
            checks,
            f"{row_id}_artifact_old_value_matches_packet",
            _same_value(summary.get("old_value"), row.get("old_value"))
            and _same_value(apply_result.get("old_value"), row.get("old_value")),
        )
        _record(
            checks,
            f"{row_id}_artifact_requested_value_matches_packet",
            _same_value(summary.get("target_value"), row.get("requested_value"))
            and _same_value(apply_result.get("requested_value"), row.get("requested_value")),
        )
        _record(
            checks,
            f"{row_id}_artifact_effective_value_matches_packet",
            _same_value(apply_result.get("effective_value"), row.get("effective_value")),
        )
        _record(
            checks,
            f"{row_id}_artifact_apply_readback_matches_packet",
            _same_value(apply_result.get("readback_value"), row.get("apply_readback_value")),
        )
        _record(
            checks,
            f"{row_id}_artifact_physical_pre_readback",
            summary.get("old_value_status") == PHYSICAL_READBACK_STATUS
            and summary.get("old_value_source") == PHYSICAL_READBACK_SOURCE
            and apply_result.get("pre_read_status") == PHYSICAL_READBACK_STATUS
            and apply_result.get("pre_read_source") == PHYSICAL_READBACK_SOURCE,
        )
        _record(
            checks,
            f"{row_id}_artifact_physical_apply_readback",
            apply_result.get("readback_status") == PHYSICAL_READBACK_STATUS
            and apply_result.get("readback_source") == PHYSICAL_READBACK_SOURCE,
        )
        _record(
            checks,
            f"{row_id}_artifact_rollback_applied",
            rollback_result.get("status") == "applied"
            and rollback_result.get("success") is True
            and rollback_result.get("rollback_status") == "applied"
            and rollback_result.get("rollback_proof_status") == "verified"
            and row.get("rollback_status") == "applied",
        )
        _record(
            checks,
            f"{row_id}_artifact_physical_rollback_readback",
            rollback_result.get("readback_status") == PHYSICAL_READBACK_STATUS
            and rollback_result.get("readback_source") == PHYSICAL_READBACK_SOURCE
            and rollback_result.get("rollback_readback_status") == PHYSICAL_READBACK_STATUS
            and rollback_result.get("rollback_readback_source") == PHYSICAL_READBACK_SOURCE,
        )
        _record(
            checks,
            f"{row_id}_artifact_emergency_stop_recorded",
            emergency_result.get("status") == row.get("emergency_stop_status") == "stopped"
            and emergency_result.get("success") is True,
        )
        _record(
            checks,
            f"{row_id}_artifact_post_emergency_apply_blocked",
            post_emergency.get("status") == row.get("post_emergency_apply_status") == "blocked"
            and post_emergency.get("success") is False
            and post_emergency.get("reason") == row.get("post_emergency_apply_reason") == "emergency_stop_active",
        )
        _record(
            checks,
            f"{row_id}_artifact_blocked_before_supervised",
            blocked_before_outer.get("status") == row.get("blocked_apply_before_supervised") == "apply_blocked"
            and blocked_before.get("status") == "blocked"
            and blocked_before.get("reason") == "operator_mode_blocks_live_write",
        )


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
    sent_request_types = m005_summary.get("sent_request_types")
    if isinstance(sent_request_types, list):
        m005_sent_request_types = [str(item) for item in sent_request_types]
    else:
        m005_sent_request_types = []
    forbidden_from_sent = [
        request_type
        for request_type in m005_sent_request_types
        if request_type in EXPECTED_M005_FORBIDDEN_DIRECT_REQUEST_TYPES
    ]
    _record(
        checks,
        "m005_summary_sent_request_types_present",
        isinstance(sent_request_types, list) and bool(m005_sent_request_types),
    )
    _record(
        checks,
        "m005_summary_apply_probe_sent",
        "apply_operator_proposal" in m005_sent_request_types,
    )
    _record(
        checks,
        "m005_summary_no_forbidden_direct_request_types",
        forbidden_from_sent == [],
    )
    _record(
        checks,
        "m005_summary_forbidden_requests_match_sent_audit",
        m005_summary.get("forbidden_request_types_sent") == forbidden_from_sent,
    )
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
    stale_readback = _get_mapping(stale, "readback")
    reconnect_status = _get_mapping(reconnect, "connection_status")
    reconnect_readback = _get_mapping(reconnect, "readback")
    _record(
        checks,
        "m005_disconnect_phase_operator_confirmed_physical_loss",
        disconnect.get("status") == "operator_confirmed_physical_cable_loss"
        and disconnect.get("confirmation") == EXPECTED_M005_PHYSICAL_CABLE_CONFIRMATION,
    )
    _record(checks, "m005_stale_readback_phase_passed", stale.get("status") == "passed")
    _record(
        checks,
        "m005_stale_readback_unavailable",
        stale.get("status") == "passed" and _m005_readback_is_unavailable(stale_readback),
    )
    _record(
        checks,
        "m005_blocked_write_phase_passed",
        blocked_write.get("status") == "passed",
    )
    _record(
        checks,
        "m005_blocked_write_reason_safe",
        blocked_write.get("status") == "passed"
        and blocked_write.get("block_reason") in EXPECTED_M005_SAFE_BLOCK_REASONS,
    )
    _record(
        checks,
        "m005_reconnect_recovery_phase_passed",
        reconnect.get("status") == "passed"
        and reconnect_status.get("connected") is True
        and reconnect.get("unchanged_from_baseline") is True,
    )
    _record(
        checks,
        "m005_reconnect_recovery_physical_readback",
        reconnect.get("status") == "passed" and _m005_readback_is_physical(reconnect_readback),
    )


def verify_packet(
    packet: Mapping[str, Any],
    *,
    m005_summary: Mapping[str, Any] | None = None,
    proof_summaries_by_artifact: Mapping[str, Mapping[str, Any]] | None = None,
    packet_summary_path: str | None = None,
    m005_summary_path: str | None = None,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    _record_packet_checks(checks, packet)
    _record_proof_artifact_checks(
        checks,
        packet,
        proof_summaries_by_artifact=proof_summaries_by_artifact,
    )
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
        "generated_at_utc": _iso_now(),
        "status": "ready_for_owner_review" if ready else "blocked",
        "reason": reason,
        "packet_summary": packet_summary_path,
        "m005_summary": m005_summary_path,
        "proof_artifact_summaries": sorted((proof_summaries_by_artifact or {}).keys()),
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
    proof_summaries = load_packet_proof_summaries(packet)
    return verify_packet(
        packet,
        m005_summary=m005,
        proof_summaries_by_artifact=proof_summaries,
        packet_summary_path=str(packet_summary),
        m005_summary_path=str(m005_summary) if m005_summary is not None else None,
    )


def write_verifier_artifact(
    result: Mapping[str, Any],
    *,
    artifact_root: Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve(artifact_root)
    effective_run_id = run_id or f"{_utc_stamp()}_m006_owner_readiness_packet_verifier"
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
    parser.add_argument(
        "--write-artifact",
        action="store_true",
        help="Write verifier summary.json under --artifact-root/<run-id>/.",
    )
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-id", help="Optional deterministic run id for artifact output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = verify_packet_paths(
        Path(args.packet_summary),
        Path(args.m005_summary) if args.m005_summary else None,
    )
    if args.write_artifact:
        result = write_verifier_artifact(
            result,
            artifact_root=Path(args.artifact_root),
            run_id=args.run_id,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ready_for_owner_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
