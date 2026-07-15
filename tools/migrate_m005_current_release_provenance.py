#!/usr/bin/env python3
"""Build and verify a truthful current-release migration for closed M005 facts.

The historical hardware artifact remains immutable and never becomes native
runtime-identity-v2 provenance.  This tool combines its exact bytes with a
clean-source, proof-owned, no-network compatibility replay and a separate
owner console identity attestation.  It is report-only and cannot grant live
authority or contact WING.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from runtime_identity import (  # noqa: E402
    canonical_json_bytes,
    capture_source_runtime_components,
    runtime_identity_manifest_sha256,
)


MIGRATION_SCHEMA = "automixer.m005_current_release_migration.v1"
REPLAY_SCHEMA = "automixer.m005_current_release_compatibility_replay.v1"
REPLAY_INVOCATION_SCHEMA = (
    "automixer.m005_current_release_compatibility_replay_invocation.v1"
)
CONSOLE_ATTESTATION_SCHEMA = "automixer.m005_migration_console_attestation.v1"
CONSOLE_ATTESTATION_CHALLENGE_SCHEMA = (
    "automixer.m005_migration_console_attestation_challenge.v1"
)
M005_PROVENANCE_SCHEMA = "automixer.m005_current_release_provenance.v1"
EXPECTED_RUNTIME_IDENTITY_SCHEMA = "automixer.backend_runtime_identity.v2"
EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH = (
    "artifacts/wing_timeout_disconnect_proof/"
    "20260711T131505Z_wing_m005_disconnect_timeout_stale_proof/summary.json"
)
EXPECTED_HISTORICAL_EVENTS_LOGICAL_PATH = (
    "artifacts/wing_timeout_disconnect_proof/"
    "20260711T131505Z_wing_m005_disconnect_timeout_stale_proof/events.jsonl"
)
EXPECTED_HISTORICAL_RUN_ID = (
    "20260711T131505Z_wing_m005_disconnect_timeout_stale_proof"
)
EXPECTED_HISTORICAL_SUMMARY_SHA256 = (
    "a5830f94cf35dad9af96127837efe04fc345ff65c7c319dc3aa216b57a047e64"
)
EXPECTED_HISTORICAL_EVENTS_SHA256 = (
    "8d40010f529f553537eee821f04e1fab585086ff4d28d8763bd4256bb2ffeb45"
)
EXPECTED_OWNER_DECISION_LOGICAL_PATH = (
    "artifacts/wing_owner_readiness_review/"
    "20260704T191101Z_m006_owner_readiness_review_packet/"
    "owner_decision_20260712T095437Z.json"
)
EXPECTED_OWNER_DECISION_SCHEMA = "automixer.m006_owner_readiness_decision.v1"
EXPECTED_OWNER_DECISION_SHA256 = (
    "2d17ad2181fd7ea4ed7d730d8bdae30b155382559cf9b62c45440953a63bc018"
)
EXPECTED_PACKET_SUMMARY_LOGICAL_PATH = (
    "artifacts/wing_owner_readiness_review/"
    "20260704T191101Z_m006_owner_readiness_review_packet/summary.json"
)
EXPECTED_PACKET_VERIFIER_LOGICAL_PATH = (
    "artifacts/wing_owner_readiness_review_verifier/"
    "20260711T131654Z_m006_owner_readiness_packet_verifier/summary.json"
)
EXPECTED_OWNER_REVIEWER = (
    "project_owner_via_codex_task_019f52a4-2a15-78d1-97ad-9d151e151c72"
)
EXPECTED_OWNER_REVIEWED_AT_UTC = "2026-07-12T09:54:37Z"
EXPECTED_PROOF_ARTIFACTS = [
    "artifacts/wing_operator_proposal_validation/"
    "20260704T045741Z_wing_operator_proposal_ch3_fader/summary.json",
    "artifacts/wing_operator_proposal_validation/"
    "20260704T045820Z_wing_operator_proposal_ch3_gain/summary.json",
]
EXPECTED_ACCEPTED_READONLY_SOURCE_ID = (
    "artifacts/wing_readonly_preflight/"
    "20260702T095758Z_wing_readonly_preflight/summary.json"
)
EXPECTED_ROWS = ["fader_supervised", "gain_supervised"]
EXPECTED_KINDS = ["fader", "gain"]
EXPECTED_EVIDENCE = [
    "physical_baseline_readback",
    "operator_confirmed_physical_cable_loss",
    "stale_readback_unavailable",
    "blocked_write_while_disconnected",
    "reconnect_recovery_verified",
]
EXPECTED_LEGACY_REQUEST_SEQUENCE = [
    "get_operator_mode",
    "get_state",
    "query_mixer_parameter",
    "set_operator_mode",
    "create_operator_proposal",
    "query_mixer_parameter",
    "apply_operator_proposal",
    "connect_wing",
    "query_mixer_parameter",
]
FORBIDDEN_REQUEST_TYPES = {
    "rollback_last_manual_write",
    "set_fader",
    "set_gain",
    "set_mute",
    "set_pan",
    "set_eq",
    "load_snapshot",
    "load_scene",
}
AUTHORITY_DENIAL_FLAGS = {
    "live_authority_granted": False,
    "readiness_closure_allowed": False,
    "readiness_matrix_update_allowed": False,
    "live_write": False,
}
SOURCE_IDENTITY_PATHS = (
    "tools/migrate_m005_current_release_provenance.py",
    "tools/verify_m006_owner_readiness_packet.py",
    "tools/verify_m006_owner_readiness_decision.py",
    "backend/server.py",
    "backend/audio_capture.py",
    "backend/handlers/product_state_handlers.py",
    "backend/operator_proposal_queue.py",
    "backend/operator_proposal_result_store.py",
    "backend/runtime_identity.py",
    "tests/conftest.py",
    "tests/test_m005_current_release_migration_replay.py",
)
REPLAY_MODULE_PATHS = (
    "tools/migrate_m005_current_release_provenance.py",
    "backend/server.py",
    "backend/audio_capture.py",
    "backend/handlers/product_state_handlers.py",
    "backend/operator_proposal_queue.py",
    "backend/operator_proposal_result_store.py",
    "backend/runtime_identity.py",
    "tests/conftest.py",
    "tests/test_m005_current_release_migration_replay.py",
)
DEFAULT_HISTORICAL_SUMMARY = EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH
DEFAULT_OWNER_DECISION = EXPECTED_OWNER_DECISION_LOGICAL_PATH
DEFAULT_ARTIFACT_ROOT = "artifacts/wing_m005_current_release_migration"
DEFAULT_CONSOLE_ATTESTATION = ""
DEFAULT_CONSOLE_ATTESTATION_CHALLENGE = ""
ALLOWED_GENERATED_UNTRACKED_PREFIXES = ("artifacts/", "reports/")
ATTESTATION_CHALLENGE_TTL = timedelta(hours=24)
RUN_ID_PATTERN = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]{0,127}")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _format_utc_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(canonical_json_bytes(value))


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _display(path: str | Path) -> str:
    resolved = _resolve(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _parse_utc_z(value: Any) -> datetime | None:
    text = str(value or "")
    if not text.endswith("Z"):
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        return None
    return parsed


def _validate_run_id(run_id: str) -> str:
    if (
        not RUN_ID_PATTERN.fullmatch(run_id)
        or run_id in {".", ".."}
        or "/" in run_id
        or "\\" in run_id
    ):
        raise ValueError(
            "run_id must be one safe 1-128 character artifact-directory name"
        )
    return run_id


def _get_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    candidate = value.get(key)
    return candidate if isinstance(candidate, Mapping) else {}


def _record(checks: dict[str, bool], name: str, passed: Any) -> None:
    checks[name] = bool(passed)


def _file_identity(path: str | Path) -> dict[str, Any]:
    resolved = _resolve(path).resolve()
    try:
        display = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        display = str(resolved)
    if not resolved.is_file() or resolved.is_symlink():
        return {"path": display, "exists": False, "size": None, "sha256": None}
    raw = resolved.read_bytes()
    return {
        "path": display,
        "exists": True,
        "size": len(raw),
        "sha256": _sha256(raw),
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _is_exact_canonical_input(path: str | Path, expected: str | Path) -> bool:
    candidate = _resolve(path)
    canonical = _resolve(expected)
    return (
        candidate.is_file()
        and not candidate.is_symlink()
        and candidate.resolve() == canonical.resolve()
    )


def _git_bytes(*args: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", *args],
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=20,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "git command failed: " + " ".join(args) + ": "
            + completed.stderr.decode("utf-8", errors="replace")
        )
    return completed.stdout


def _worktree_status_audit() -> dict[str, Any]:
    raw = _git_bytes("status", "--porcelain=v1", "-z", "--untracked-files=all")
    entries: list[dict[str, str]] = []
    allowed_untracked_paths: list[str] = []
    disallowed_entries: list[dict[str, str]] = []
    for raw_entry in raw.split(b"\0"):
        if not raw_entry:
            continue
        text = raw_entry.decode("utf-8", errors="surrogateescape")
        if len(text) >= 4 and text[2] == " ":
            status = text[:2]
            path = text[3:]
        else:
            status = "!!"
            path = text
        entry = {"status": status, "path": path}
        entries.append(entry)
        allowed = status == "??" and path.startswith(
            ALLOWED_GENERATED_UNTRACKED_PREFIXES
        )
        if allowed:
            allowed_untracked_paths.append(path)
        else:
            disallowed_entries.append(entry)
    return {
        "schema": "automixer.m005_migration_worktree_status_audit.v1",
        "status_sha256": _sha256(raw),
        "entries": entries,
        "allowed_generated_untracked_prefixes": list(
            ALLOWED_GENERATED_UNTRACKED_PREFIXES
        ),
        "allowed_untracked_paths": allowed_untracked_paths,
        "disallowed_entries": disallowed_entries,
    }


def capture_current_source() -> dict[str, Any]:
    worktree_audit = _worktree_status_audit()
    tracked_diff = _git_bytes("diff", "HEAD", "--binary")
    identities = {path: _file_identity(path) for path in SOURCE_IDENTITY_PATHS}
    runtime_source = capture_source_runtime_components(PROJECT_ROOT)
    runtime_worktree_audit = _worktree_status_audit()
    return {
        "head": _git_bytes("rev-parse", "HEAD").decode().strip(),
        "tree": _git_bytes("rev-parse", "HEAD^{tree}").decode().strip(),
        "worktree_dirty": bool(worktree_audit["entries"]),
        "status_sha256": worktree_audit["status_sha256"],
        "tracked_diff_sha256": _sha256(tracked_diff),
        "worktree_status_audit": worktree_audit,
        "runtime_source": runtime_source,
        "runtime_worktree_status_audit": runtime_worktree_audit,
        "file_identities": identities,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        if not isinstance(value, dict):
            raise ValueError("events row must be an object")
        rows.append(value)
    return rows


def _event_payload(
    rows: Sequence[Mapping[str, Any]],
    *,
    seq: int,
    event_type: str,
    payload_type: str,
) -> Mapping[str, Any]:
    for row in rows:
        payload = row.get("payload")
        if (
            row.get("seq") == seq
            and row.get("event_type") == event_type
            and isinstance(payload, Mapping)
            and payload.get("type") == payload_type
        ):
            return payload
    return {}


def _readback_matches_target(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("channel") == 3
        and payload.get("value_type") == "fader"
        and payload.get("address") == "/ch/3/fdr"
    )


def verify_historical_evidence(
    summary_path: str | Path,
    events_path: str | Path | None = None,
    *,
    expected_summary_sha256: str = EXPECTED_HISTORICAL_SUMMARY_SHA256,
    expected_events_sha256: str = EXPECTED_HISTORICAL_EVENTS_SHA256,
) -> dict[str, Any]:
    summary_file = _resolve(summary_path).resolve()
    events_file = (
        _resolve(events_path).resolve()
        if events_path is not None
        else summary_file.with_name("events.jsonl")
    )
    checks: dict[str, bool] = {}
    try:
        summary_raw = summary_file.read_bytes()
        events_raw = events_file.read_bytes()
        summary = json.loads(summary_raw.decode("utf-8"))
        rows = _read_jsonl(events_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        summary_raw = b""
        events_raw = b""
        summary = {}
        rows = []
    _record(checks, "historical_summary_hash_matches", _sha256(summary_raw) == expected_summary_sha256)
    _record(checks, "historical_events_hash_matches", _sha256(events_raw) == expected_events_sha256)
    _record(checks, "historical_summary_object", isinstance(summary, dict) and bool(summary))
    _record(checks, "historical_event_count_matches", len(rows) == 62)
    _record(
        checks,
        "historical_event_sequence_contiguous",
        [row.get("seq") for row in rows] == list(range(1, 63)),
    )
    _record(checks, "historical_run_id_matches", summary.get("run_id") == EXPECTED_HISTORICAL_RUN_ID)
    _record(checks, "historical_schema_matches", summary.get("proof_artifact_schema") == "wing_timeout_disconnect_proof.v1")
    _record(checks, "historical_status_passed", summary.get("status") == "passed")
    _record(checks, "historical_behavior_complete", summary.get("m005_complete") is True)
    _record(checks, "historical_required_evidence_matches", summary.get("required_m005_evidence") == EXPECTED_EVIDENCE)
    _record(checks, "historical_completed_evidence_matches", summary.get("completed_m005_evidence") == EXPECTED_EVIDENCE)
    _record(checks, "historical_missing_evidence_empty", summary.get("missing_m005_evidence") == [])
    _record(checks, "historical_target_matches", summary.get("channel") == 3 and summary.get("value_type") == "fader" and summary.get("address") == "/ch/3/fdr")
    preflight_gate = _get_mapping(summary, "preflight_decision_gate")
    _record(
        checks,
        "historical_readonly_source_gate_matches",
        preflight_gate.get("accepted") is True
        and preflight_gate.get("status") == "accepted_decision_ready"
        and preflight_gate.get("expected_accepted_readonly_source_id")
        == EXPECTED_ACCEPTED_READONLY_SOURCE_ID,
    )
    _record(checks, "historical_physical_cable_loss", summary.get("disconnect_mode") == "physical-cable-loss" and summary.get("physical_cable_confirmation") == "WING_CABLE_PULLED_BY_OPERATOR")
    sent = summary.get("sent_request_types")
    _record(checks, "historical_request_sequence_matches", sent == EXPECTED_LEGACY_REQUEST_SEQUENCE)
    event_requests = [
        row.get("payload", {}).get("type")
        for row in rows
        if row.get("event_type") == "send" and isinstance(row.get("payload"), Mapping)
    ]
    _record(checks, "historical_event_request_sequence_matches", event_requests == EXPECTED_LEGACY_REQUEST_SEQUENCE)
    _record(checks, "historical_forbidden_requests_absent", not (set(event_requests) & FORBIDDEN_REQUEST_TYPES) and summary.get("forbidden_request_types_sent") == [])
    _record(checks, "historical_rollback_absent", summary.get("rollback_request_sent") is False and "rollback_last_manual_write" not in event_requests)
    for flag, expected in AUTHORITY_DENIAL_FLAGS.items():
        if flag in summary or flag != "live_write":
            _record(checks, f"historical_{flag}_denied", summary.get(flag) is expected)

    phases = _get_mapping(summary, "phases")
    baseline = _get_mapping(_get_mapping(phases, "baseline"), "readback")
    stale = _get_mapping(_get_mapping(phases, "stale_readback"), "readback")
    blocked_phase = _get_mapping(phases, "blocked_write_while_disconnected")
    blocked = _get_mapping(blocked_phase, "apply_result")
    reconnect_phase = _get_mapping(phases, "reconnect_recovery")
    reconnect = _get_mapping(reconnect_phase, "readback")
    baseline_event = _event_payload(rows, seq=7, event_type="recv", payload_type="mixer_parameter_value")
    stale_event = _event_payload(rows, seq=15, event_type="recv", payload_type="mixer_parameter_value")
    blocked_event = _event_payload(rows, seq=17, event_type="recv", payload_type="operator_proposal_apply_blocked")
    reconnect_event = _event_payload(rows, seq=62, event_type="recv", payload_type="mixer_parameter_value")
    _record(checks, "historical_baseline_physical", _readback_matches_target(baseline) and baseline.get("status") == "verified_from_physical_readback" and baseline.get("source") == "physical_readback")
    _record(checks, "historical_baseline_event_matches", dict(baseline_event) == dict(baseline))
    _record(checks, "historical_stale_unavailable", _readback_matches_target(stale) and stale.get("status") == "readback_unavailable" and stale.get("source") == "physical_readback" and stale.get("value") is None)
    _record(checks, "historical_stale_event_matches", dict(stale_event) == dict(stale))
    _record(checks, "historical_blocked_apply_safe", blocked_phase.get("status") == "passed" and blocked_phase.get("block_reason") == "pre_read_cache_not_fresh" and blocked.get("type") == "operator_proposal_apply_blocked" and blocked.get("status") == "apply_blocked")
    _record(
        checks,
        "historical_blocked_event_matches",
        _get_mapping(blocked_event, "apply_result").get("reason")
        == "pre_read_cache_not_fresh"
        and blocked_event.get("status") == "apply_blocked",
    )
    _record(checks, "historical_reconnect_physical", reconnect_phase.get("status") == "passed" and reconnect_phase.get("unchanged_from_baseline") is True and _readback_matches_target(reconnect) and reconnect.get("status") == "verified_from_physical_readback" and reconnect.get("source") == "physical_readback")
    _record(checks, "historical_reconnect_event_matches", dict(reconnect_event) == dict(reconnect))
    _record(checks, "historical_value_unchanged", baseline.get("value") == reconnect.get("value") == summary.get("old_value"))
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "ready": not failed,
        "logical_summary_path": EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH,
        "logical_events_path": EXPECTED_HISTORICAL_EVENTS_LOGICAL_PATH,
        "loaded_summary_path": str(summary_file),
        "loaded_events_path": str(events_file),
        "summary_sha256": _sha256(summary_raw),
        "events_sha256": _sha256(events_raw),
        "run_id": summary.get("run_id"),
        "accepted_readonly_source_id": preflight_gate.get(
            "expected_accepted_readonly_source_id"
        ),
        "target": {"channel": 3, "value_type": "fader", "address": "/ch/3/fdr"},
        "facts": {
            "physical_baseline_readback": dict(baseline),
            "operator_confirmed_physical_cable_loss": dict(_get_mapping(phases, "disconnect")),
            "stale_readback_unavailable": dict(stale),
            "blocked_write_while_disconnected": {
                "status": blocked_phase.get("status"),
                "reason": blocked_phase.get("block_reason"),
            },
            "reconnect_recovery_verified": dict(reconnect),
        },
        "checks": checks,
        "failed_checks": failed,
    }


def verify_owner_decision(path: str | Path) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    try:
        decision = _load_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        decision = {}
    identity = _file_identity(path)
    _record(
        checks,
        "owner_decision_hash_matches",
        identity.get("sha256") == EXPECTED_OWNER_DECISION_SHA256,
    )
    _record(checks, "owner_decision_schema_matches", decision.get("schema") == EXPECTED_OWNER_DECISION_SCHEMA)
    _record(checks, "owner_decision_accepted", decision.get("decision") == "accepted" and decision.get("status") == "accepted_for_readiness_matrix_update")
    _record(checks, "owner_decision_rows_exact", decision.get("accepted_safety_row_ids") == EXPECTED_ROWS)
    _record(checks, "owner_decision_kinds_exact", decision.get("accepted_live_write_kinds") == EXPECTED_KINDS)
    _record(
        checks,
        "owner_decision_proof_artifacts_exact",
        decision.get("accepted_proof_artifacts") == EXPECTED_PROOF_ARTIFACTS,
    )
    _record(
        checks,
        "owner_decision_packet_binding_exact",
        decision.get("packet_summary") == EXPECTED_PACKET_SUMMARY_LOGICAL_PATH
        and decision.get("packet_verifier_summary")
        == EXPECTED_PACKET_VERIFIER_LOGICAL_PATH,
    )
    _record(
        checks,
        "owner_decision_reviewer_binding_exact",
        decision.get("reviewed_by") == EXPECTED_OWNER_REVIEWER
        and decision.get("reviewed_at_utc") == EXPECTED_OWNER_REVIEWED_AT_UTC,
    )
    _record(
        checks,
        "owner_decision_update_candidate_true",
        decision.get("readiness_matrix_update_candidate") is True,
    )
    _record(checks, "owner_decision_historical_path_matches", decision.get("m005_summary") == EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH)
    acceptance = _get_mapping(decision, "m005_acceptance")
    _record(checks, "owner_decision_behavior_acceptance_exact", acceptance == {
        "physical_cable_loss": True,
        "stale_readback_unavailable": True,
        "blocked_write_while_disconnected": True,
        "reconnect_recovery_verified": True,
        "rollback_request_sent": False,
        "request_audit_contains_rollback": False,
        "authority_denied_during_m005": True,
    })
    guardrails = _get_mapping(decision, "scope_guardrails")
    _record(checks, "owner_decision_scope_stays_narrow", guardrails == {
        "gain_up_allowed": False,
        "fader_boosts_allowed": False,
        "direct_write_allowed": False,
        "full_auto_apply_allowed": False,
        "non_fader_gain_surfaces_allowed": False,
    })
    for flag in (
        "live_authority_granted",
        "readiness_closure_allowed",
        "readiness_matrix_update_allowed",
        "contacts_wing",
        "contacts_backend",
        "live_write",
    ):
        _record(checks, f"owner_decision_{flag}_denied", decision.get(flag) is False)
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "ready": not failed,
        "path": _display(path),
        "identity": identity,
        "decision": decision,
        "checks": checks,
        "failed_checks": failed,
    }


def build_console_attestation_challenge_payload(
    *,
    source: Mapping[str, Any],
    migration_run_id: str,
    nonce: str | None = None,
    captured_at: datetime | None = None,
) -> dict[str, Any]:
    run_id = _validate_run_id(migration_run_id)
    challenge_nonce = nonce or secrets.token_hex(32)
    if not re.fullmatch(r"[0-9a-f]{64}", challenge_nonce):
        raise ValueError("challenge nonce must be 64 lowercase hexadecimal characters")
    captured = (captured_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expires = captured + ATTESTATION_CHALLENGE_TTL
    return {
        "schema": CONSOLE_ATTESTATION_CHALLENGE_SCHEMA,
        "status": "ready_for_owner_attestation",
        "migration_run_id": run_id,
        "challenge_nonce": challenge_nonce,
        "captured_at_utc": _format_utc_z(captured),
        "expires_at_utc": _format_utc_z(expires),
        "source_head": source.get("head"),
        "source_tree": source.get("tree"),
        "source_binding_sha256": _source_release_binding_sha256(source),
        "historical_summary_sha256": EXPECTED_HISTORICAL_SUMMARY_SHA256,
        "historical_events_sha256": EXPECTED_HISTORICAL_EVENTS_SHA256,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "readiness_matrix_update_allowed": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "live_write": False,
    }


def write_console_attestation_challenge(
    output_path: str | Path,
    *,
    migration_run_id: str | None = None,
    source_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = (
        dict(source_override)
        if isinstance(source_override, Mapping)
        else capture_current_source()
    )
    source_checks = _source_checks(source)
    if not all(source_checks.values()):
        raise ValueError("a clean current source is required for an attestation challenge")
    effective_run_id = _validate_run_id(
        migration_run_id
        or f"{_utc_stamp()}_{uuid.uuid4()}_m005_current_release_migration"
    )
    destination = _resolve(output_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    challenge = build_console_attestation_challenge_payload(
        source=source,
        migration_run_id=effective_run_id,
    )
    destination.write_bytes(canonical_json_bytes(challenge) + b"\n")
    return {
        "schema": CONSOLE_ATTESTATION_CHALLENGE_SCHEMA,
        "status": "challenge_ready_for_owner_attestation",
        "migration_run_id": effective_run_id,
        "challenge_path": _display(destination),
        "challenge_identity": _file_identity(destination),
        "source_binding_sha256": challenge["source_binding_sha256"],
        "live_authority_granted": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "live_write": False,
    }


def verify_console_attestation_challenge(
    path: str | Path | None,
    *,
    expected_source: Mapping[str, Any] | None,
    expected_run_id: str | None,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    if path is None or not str(path).strip():
        challenge: dict[str, Any] = {}
        identity = None
        display_path = None
    else:
        try:
            challenge = _load_json(path)
            identity = _file_identity(path)
            display_path = _display(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            challenge = {}
            identity = _file_identity(path)
            display_path = _display(path)
    _record(checks, "console_attestation_challenge_loaded", bool(challenge))
    _record(
        checks,
        "console_attestation_challenge_schema_matches",
        challenge.get("schema") == CONSOLE_ATTESTATION_CHALLENGE_SCHEMA,
    )
    _record(
        checks,
        "console_attestation_challenge_status_ready",
        challenge.get("status") == "ready_for_owner_attestation",
    )
    _record(
        checks,
        "console_attestation_challenge_run_matches",
        isinstance(expected_run_id, str)
        and challenge.get("migration_run_id") == expected_run_id,
    )
    _record(
        checks,
        "console_attestation_challenge_nonce_valid",
        isinstance(challenge.get("challenge_nonce"), str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", challenge["challenge_nonce"])),
    )
    captured = _parse_utc_z(challenge.get("captured_at_utc"))
    expires = _parse_utc_z(challenge.get("expires_at_utc"))
    now = datetime.now(timezone.utc)
    observation = (observed_at or now).astimezone(timezone.utc)
    _record(
        checks,
        "console_attestation_challenge_window_valid",
        captured is not None
        and expires is not None
        and expires - captured == ATTESTATION_CHALLENGE_TTL
        and captured <= observation <= expires
        and observation <= now,
    )
    source = expected_source if isinstance(expected_source, Mapping) else {}
    _record(
        checks,
        "console_attestation_challenge_source_matches",
        bool(source)
        and challenge.get("source_head") == source.get("head")
        and challenge.get("source_tree") == source.get("tree")
        and challenge.get("source_binding_sha256")
        == _source_release_binding_sha256(source),
    )
    _record(
        checks,
        "console_attestation_challenge_historical_hashes_match",
        challenge.get("historical_summary_sha256")
        == EXPECTED_HISTORICAL_SUMMARY_SHA256
        and challenge.get("historical_events_sha256")
        == EXPECTED_HISTORICAL_EVENTS_SHA256,
    )
    for flag in (
        "live_authority_granted",
        "readiness_closure_allowed",
        "readiness_matrix_update_allowed",
        "contacts_wing",
        "contacts_backend",
        "live_write",
    ):
        _record(
            checks,
            f"console_attestation_challenge_{flag}_denied",
            challenge.get(flag) is False,
        )
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "ready": not failed,
        "path": display_path,
        "identity": identity,
        "challenge": challenge,
        "checks": checks,
        "failed_checks": failed,
    }


def verify_console_attestation(
    path: str | Path | None,
    *,
    challenge_path: str | Path | None = None,
    expected_source: Mapping[str, Any] | None = None,
    expected_run_id: str | None = None,
    observed_at: datetime | None = None,
    expected_summary_sha256: str = EXPECTED_HISTORICAL_SUMMARY_SHA256,
    expected_events_sha256: str = EXPECTED_HISTORICAL_EVENTS_SHA256,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    if path is None or not str(path).strip():
        attestation: dict[str, Any] = {}
        identity = None
        display_path = None
    else:
        try:
            attestation = _load_json(path)
            identity = _file_identity(path)
            display_path = _display(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            attestation = {}
            identity = _file_identity(path)
            display_path = _display(path)
    challenge_verification = verify_console_attestation_challenge(
        challenge_path,
        expected_source=expected_source,
        expected_run_id=expected_run_id,
        observed_at=observed_at,
    )
    challenge = _get_mapping(challenge_verification, "challenge")
    challenge_identity = _get_mapping(challenge_verification, "identity")
    _record(
        checks,
        "console_attestation_challenge_ready",
        challenge_verification.get("ready") is True,
    )
    _record(checks, "console_attestation_loaded", bool(attestation))
    _record(checks, "console_attestation_schema_matches", attestation.get("schema") == CONSOLE_ATTESTATION_SCHEMA)
    _record(checks, "console_attestation_status_accepted", attestation.get("status") == "accepted")
    _record(checks, "console_attestation_identity_source_honest", attestation.get("identity_source") == "operator_attested_same_console" and attestation.get("hardware_identity_query_performed") is False)
    _record(checks, "console_attestation_model_matches", attestation.get("console_model") == "WING Rack")
    _record(checks, "console_attestation_firmware_matches", attestation.get("firmware") == "3.0.5")
    serial = attestation.get("serial")
    _record(
        checks,
        "console_attestation_serial_present",
        isinstance(serial, str)
        and bool(serial.strip())
        and serial.strip() == serial
        and len(serial) <= 128
        and serial.isprintable(),
    )
    _record(checks, "console_attestation_same_console_confirmed", attestation.get("same_console_as_historical_m005") is True)
    _record(checks, "console_attestation_no_historical_runtime_claim", attestation.get("historical_runtime_identity_claimed") is False and attestation.get("historical_prestart_provenance_claimed") is False)
    _record(checks, "console_attestation_historical_summary_hash_bound", attestation.get("historical_summary_sha256") == expected_summary_sha256)
    _record(checks, "console_attestation_historical_events_hash_bound", attestation.get("historical_events_sha256") == expected_events_sha256)
    reviewer = attestation.get("reviewed_by")
    _record(
        checks,
        "console_attestation_reviewer_present",
        isinstance(reviewer, str)
        and bool(reviewer.strip())
        and reviewer.strip() == reviewer
        and len(reviewer) <= 256
        and reviewer.isprintable(),
    )
    reviewed_at = _parse_utc_z(attestation.get("reviewed_at_utc"))
    challenge_captured = _parse_utc_z(challenge.get("captured_at_utc"))
    challenge_expires = _parse_utc_z(challenge.get("expires_at_utc"))
    reviewed_at_valid = (
        reviewed_at is not None
        and challenge_captured is not None
        and challenge_expires is not None
        and challenge_captured <= reviewed_at <= challenge_expires
        and reviewed_at
        <= (observed_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    )
    _record(checks, "console_attestation_review_time_valid", reviewed_at_valid)
    _record(
        checks,
        "console_attestation_challenge_binding_exact",
        bool(challenge)
        and bool(challenge_identity)
        and attestation.get("migration_run_id") == expected_run_id
        == challenge.get("migration_run_id")
        and attestation.get("challenge_nonce") == challenge.get("challenge_nonce")
        and attestation.get("challenge_sha256") == challenge_identity.get("sha256")
        and attestation.get("source_head") == challenge.get("source_head")
        and attestation.get("source_tree") == challenge.get("source_tree")
        and attestation.get("source_binding_sha256")
        == challenge.get("source_binding_sha256"),
    )
    for flag in (
        "live_authority_granted",
        "readiness_closure_allowed",
        "readiness_matrix_update_allowed",
        "contacts_wing",
        "contacts_backend",
        "live_write",
    ):
        _record(checks, f"console_attestation_{flag}_denied", attestation.get(flag) is False)
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "ready": not failed,
        "path": display_path,
        "identity": identity,
        "attestation": attestation,
        "challenge_verification": challenge_verification,
        "checks": checks,
        "failed_checks": failed,
    }


def _status_audit_matches_source(
    audit: Mapping[str, Any],
    source: Mapping[str, Any],
) -> bool:
    entries = audit.get("entries")
    if not isinstance(entries, list):
        return False
    reconstructed = b""
    allowed_paths: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            return False
        status = entry.get("status")
        path = entry.get("path")
        if status != "??" or not isinstance(path, str):
            return False
        reconstructed += b"?? " + path.encode(
            "utf-8",
            errors="surrogateescape",
        ) + b"\0"
        if not path.startswith(ALLOWED_GENERATED_UNTRACKED_PREFIXES):
            return False
        allowed_paths.append(path)
    return (
        audit.get("schema")
        == "automixer.m005_migration_worktree_status_audit.v1"
        and audit.get("allowed_generated_untracked_prefixes")
        == list(ALLOWED_GENERATED_UNTRACKED_PREFIXES)
        and audit.get("status_sha256")
        == source.get("status_sha256")
        == _sha256(reconstructed)
        and source.get("worktree_dirty") is bool(entries)
        and audit.get("allowed_untracked_paths") == allowed_paths
        and audit.get("disallowed_entries") == []
    )


def _source_checks(source: Mapping[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    _record(checks, "current_source_head_valid", isinstance(source.get("head"), str) and bool(re.fullmatch(r"[0-9a-f]{40,64}", source["head"])))
    _record(checks, "current_source_tree_valid", isinstance(source.get("tree"), str) and bool(re.fullmatch(r"[0-9a-f]{40,64}", source["tree"])))
    worktree_audit = _get_mapping(source, "worktree_status_audit")
    _record(
        checks,
        "current_source_clean",
        source.get("tracked_diff_sha256") == _sha256(b"")
        and _status_audit_matches_source(worktree_audit, source),
    )
    runtime = _get_mapping(source, "runtime_source")
    runtime_source = _get_mapping(runtime, "source")
    runtime_code = _get_mapping(runtime, "code")
    _record(checks, "current_runtime_source_head_matches", runtime_source.get("head") == source.get("head") and runtime_source.get("tree") == source.get("tree"))
    runtime_worktree_audit = _get_mapping(
        source,
        "runtime_worktree_status_audit",
    )
    _record(
        checks,
        "current_runtime_source_clean",
        runtime_source.get("tracked_diff_sha256") == _sha256(b"")
        and _status_audit_matches_source(
            runtime_worktree_audit,
            runtime_source,
        ),
    )
    files = runtime_code.get("files")
    _record(checks, "current_runtime_source_self_consistent", isinstance(files, list) and runtime_code.get("files_sha256") == _canonical_sha256(files) and runtime_source.get("source_content_sha256") == runtime_code.get("files_sha256"))
    identities = _get_mapping(source, "file_identities")
    _record(checks, "current_source_identity_paths_exact", set(identities) == set(SOURCE_IDENTITY_PATHS))
    for relative_path in SOURCE_IDENTITY_PATHS:
        identity = _get_mapping(identities, relative_path)
        _record(checks, f"current_source_identity_{relative_path.replace('/', '_').replace('.', '_')}", identity.get("path") == relative_path and identity.get("exists") is True and isinstance(identity.get("size"), int) and identity.get("size", 0) > 0 and isinstance(identity.get("sha256"), str) and bool(re.fullmatch(r"[0-9a-f]{64}", identity["sha256"])))
    return checks


def _source_release_binding_payload(source: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _get_mapping(source, "runtime_source")
    runtime_source = _get_mapping(runtime, "source")
    return {
        "head": source.get("head"),
        "tree": source.get("tree"),
        "tracked_diff_sha256": source.get("tracked_diff_sha256"),
        "file_identities": source.get("file_identities"),
        "runtime_source": {
            "source": {
                field_name: runtime_source.get(field_name)
                for field_name in (
                    "schema",
                    "observation",
                    "head",
                    "tree",
                    "tracked_diff_sha256",
                    "source_content_sha256",
                    "content_scope",
                )
            },
            "code": runtime.get("code"),
        },
    }


def _source_release_binding_sha256(source: Mapping[str, Any]) -> str:
    return _canonical_sha256(_source_release_binding_payload(source))


def _source_release_binding_matches(
    declared: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    return _source_release_binding_payload(declared) == _source_release_binding_payload(
        current
    )


def _replay_prestart(
    source: Mapping[str, Any],
    *,
    launch_nonce: str,
    isolation_roots: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": M005_PROVENANCE_SCHEMA,
        "evidence_class": "current_source_no_network_compatibility_replay",
        "captured_at_utc": _iso_now(),
        "runtime_launch_nonce": launch_nonce,
        "source": {
            key: source.get(key)
            for key in (
                "head",
                "tree",
                "worktree_dirty",
                "status_sha256",
                "tracked_diff_sha256",
                "worktree_status_audit",
                "runtime_worktree_status_audit",
            )
        },
        "runtime_source": source.get("runtime_source"),
        "file_identities": source.get("file_identities"),
        "isolation_roots": dict(isolation_roots),
        "contacts_backend_transport": False,
        "contacts_wing": False,
        "network_listener_opened": False,
        "audio_device_opened": False,
        "live_authority_granted": False,
        "live_write": False,
    }


def _allowed_child_environment() -> dict[str, str]:
    allowed = {
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "TMPDIR",
        "VIRTUAL_ENV",
        "PYTHONHOME",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    }
    return {
        name: value
        for name, value in os.environ.items()
        if name in allowed and isinstance(value, str) and value
    }


def run_compatibility_replay(
    *,
    run_dir: Path,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    launch_nonce = uuid.uuid4().hex
    challenge = secrets.token_hex(32)
    isolation_roots = {
        "console_runtime": str((run_dir / "isolated" / "console-runtime").resolve()),
        "operator_results": str((run_dir / "isolated" / "proposal-results").resolve()),
        "manual_write_artifacts": str(
            (run_dir / "isolated" / "manual-write-artifacts").resolve()
        ),
    }
    prestart = _replay_prestart(
        source,
        launch_nonce=launch_nonce,
        isolation_roots=isolation_roots,
    )
    prestart_raw = canonical_json_bytes(prestart)
    prestart_sha256 = _sha256(prestart_raw)
    report_path = run_dir / "compatibility-replay.json"
    stdout_path = run_dir / "compatibility-replay.stdout.log"
    stderr_path = run_dir / "compatibility-replay.stderr.log"
    invocation_path = run_dir / "compatibility-replay.invocation.json"
    environment = _allowed_child_environment()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [str(BACKEND_ROOT), str(TOOLS_ROOT), str(PROJECT_ROOT)]
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "AUTOMIXER_M005_LAUNCH_NONCE": launch_nonce,
            "AUTOMIXER_M005_PRESTART_PROVENANCE_SCHEMA": M005_PROVENANCE_SCHEMA,
            "AUTOMIXER_M005_PRESTART_PROVENANCE_SHA256": prestart_sha256,
            "AUTOMIXER_M005_PRESTART_PROVENANCE_SIZE": str(len(prestart_raw)),
            "AUTOMIXER_M005_MIGRATION_REPLAY_CHALLENGE": challenge,
            "AUTOMIXER_M005_MIGRATION_REPLAY_OUTPUT": str(report_path),
            "AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR": isolation_roots[
                "console_runtime"
            ],
            "AUTOMIXER_OPERATOR_PROPOSAL_RESULT_DIR": isolation_roots[
                "operator_results"
            ],
            "AUTOMIXER_MANUAL_WRITE_ARTIFACTS_DIR": isolation_roots[
                "manual_write_artifacts"
            ],
            "AUTOMIXER_OPERATOR_MODE": "auto",
            "AUTOMIXER_AUTO_MODE_UNLOCKED": "true",
            "AUTOMIXER_WING_DEPLOYMENT_WRITE_BOUNDARY_ENABLED": "false",
            "AUTOMIXER_M005_MIGRATION_REPLAY_ISOLATED": "1",
        }
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-p",
        "pytest_asyncio.plugin",
        "tests/test_m005_current_release_migration_replay.py::test_m005_current_release_migration_replay_emits_bound_report",
        "-q",
    ]
    started_at_utc = _iso_now()
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
        )
        child_pid = process.pid
        try:
            exit_code = process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait(timeout=10)
            exit_code = 124
    ended_at_utc = _iso_now()
    try:
        report = _load_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        report = {}
    verification = verify_replay_report(
        report,
        source=source,
        prestart=prestart,
        challenge=challenge,
        exit_code=exit_code,
        expected_child_pid=child_pid,
    )
    invocation = {
        "schema": REPLAY_INVOCATION_SCHEMA,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "parent_pid": os.getpid(),
        "child_pid": child_pid,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "challenge": challenge,
        "launch_nonce": launch_nonce,
        "prestart": prestart,
        "prestart_sha256": prestart_sha256,
        "prestart_size": len(prestart_raw),
        "command": command,
        "cwd": str(PROJECT_ROOT.resolve()),
        "interpreter": _file_identity(sys.executable),
        "isolation_roots": isolation_roots,
        "report_identity": _file_identity(report_path),
        "stdout_identity": _file_identity(stdout_path),
        "stderr_identity": _file_identity(stderr_path),
    }
    invocation_path.write_bytes(canonical_json_bytes(invocation) + b"\n")
    return {
        "ready": verification["ready"],
        "command": command,
        "exit_code": exit_code,
        "prestart": prestart,
        "prestart_sha256": prestart_sha256,
        "prestart_size": len(prestart_raw),
        "report_path": str(report_path),
        "report_identity": _file_identity(report_path),
        "stdout_identity": _file_identity(stdout_path),
        "stderr_identity": _file_identity(stderr_path),
        "invocation_path": str(invocation_path),
        "invocation_identity": _file_identity(invocation_path),
        "invocation": invocation,
        "report": report,
        "checks": verification["checks"],
        "failed_checks": verification["failed_checks"],
    }


def verify_replay_report(
    report: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    prestart: Mapping[str, Any],
    challenge: str,
    exit_code: int,
    expected_child_pid: int | None = None,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    _record(checks, "replay_process_exit_zero", exit_code == 0)
    _record(checks, "replay_schema_matches", report.get("schema") == REPLAY_SCHEMA)
    _record(checks, "replay_challenge_matches", report.get("challenge") == challenge)
    _record(checks, "replay_proof_owned_process", report.get("proof_owned_process") is True)
    runtime = _get_mapping(report, "runtime_identity")
    process = _get_mapping(runtime, "process")
    launch = _get_mapping(runtime, "launch_binding")
    websocket = _get_mapping(runtime, "websocket")
    _record(checks, "replay_runtime_schema_matches", runtime.get("schema") == EXPECTED_RUNTIME_IDENTITY_SCHEMA and runtime.get("runtime_kind") == "source")
    _record(
        checks,
        "replay_child_pid_matches",
        report.get("child_pid") == process.get("pid")
        and isinstance(process.get("pid"), int)
        and process.get("pid", 0) > 1
        and (
            expected_child_pid is None
            or process.get("pid") == expected_child_pid
        ),
    )
    _record(checks, "replay_runtime_challenge_matches", runtime.get("request_challenge_nonce") == challenge[:32])
    _record(checks, "replay_runtime_manifest_matches", runtime.get("manifest_sha256") == runtime_identity_manifest_sha256(dict(runtime)))
    _record(checks, "replay_launch_binding_matches", launch.get("mode") == "proof_runner" and launch.get("launch_nonce") == prestart.get("runtime_launch_nonce") and launch.get("prestart_provenance_schema") == M005_PROVENANCE_SCHEMA and launch.get("prestart_provenance_sha256") == _canonical_sha256(prestart) and launch.get("prestart_provenance_size") == len(canonical_json_bytes(prestart)) and launch.get("bound_before_process_start") is True and launch.get("proof_eligible") is True)
    expected_runtime = _get_mapping(source, "runtime_source")
    runtime_source = _get_mapping(runtime, "source")
    expected_runtime_source = _get_mapping(expected_runtime, "source")
    stable_runtime_source_fields = (
        "schema",
        "observation",
        "head",
        "tree",
        "tracked_diff_sha256",
        "source_content_sha256",
        "content_scope",
    )
    _record(
        checks,
        "replay_runtime_source_matches",
        all(
            runtime_source.get(field_name)
            == expected_runtime_source.get(field_name)
            for field_name in stable_runtime_source_fields
        )
        and _get_mapping(runtime, "code") == _get_mapping(expected_runtime, "code"),
    )
    _record(
        checks,
        "replay_runtime_worktree_status_allowed",
        _status_audit_matches_source(
            _get_mapping(report, "runtime_worktree_status_audit"),
            runtime_source,
        ),
    )
    instance_nonce = runtime.get("instance_nonce")
    _record(
        checks,
        "replay_runtime_instance_nonce_matches_process",
        isinstance(instance_nonce, str)
        and bool(re.fullmatch(r"[0-9a-f]{32}", instance_nonce))
        and process.get("instance_nonce") == instance_nonce,
    )
    _record(
        checks,
        "replay_runtime_identity_read_only",
        runtime.get("read_only_identity") is True
        and runtime.get("live_authority_granted") is False,
    )
    _record(
        checks,
        "replay_runtime_backend_matches_source",
        _get_mapping(runtime, "backend")
        == _get_mapping(_get_mapping(source, "file_identities"), "backend/server.py"),
    )
    _record(
        checks,
        "replay_runtime_websocket_loopback_identity",
        websocket.get("host") == "127.0.0.1"
        and websocket.get("loopback_only") is True
        and websocket.get("port") == 8765,
    )
    try:
        prestart_time = datetime.fromisoformat(
            str(prestart.get("captured_at_utc") or "").replace("Z", "+00:00")
        )
        runtime_time = datetime.fromisoformat(
            str(runtime.get("started_at_utc") or "").replace("Z", "+00:00")
        )
        runtime_delay = (runtime_time - prestart_time).total_seconds()
        runtime_time_valid = (
            process.get("started_at_utc") == runtime.get("started_at_utc")
            and 0 <= runtime_delay <= 600
        )
    except ValueError:
        runtime_time_valid = False
    _record(
        checks,
        "replay_runtime_started_after_prestart",
        runtime_time_valid,
    )
    reported_modules = _get_mapping(report, "module_identities")
    source_identities = _get_mapping(source, "file_identities")
    _record(checks, "replay_module_paths_exact", set(reported_modules) == set(REPLAY_MODULE_PATHS))
    for relative_path in REPLAY_MODULE_PATHS:
        _record(checks, f"replay_module_{relative_path.replace('/', '_').replace('.', '_')}_matches", _get_mapping(reported_modules, relative_path) == _get_mapping(source_identities, relative_path))
    _record(checks, "replay_canonical_entrypoints_exact", report.get("canonical_entrypoints") == ["create_operator_proposal", "apply_operator_proposal"])
    scenarios = report.get("scenarios")
    expected_scenarios = [
        ("stale_cache", "fader", "pre_read_cache_not_fresh"),
        ("stale_cache", "gain", "pre_read_cache_not_fresh"),
        ("disconnected", "fader", "operator_proposal_global_hazard_active"),
        ("disconnected", "gain", "operator_proposal_global_hazard_active"),
    ]
    _record(
        checks,
        "replay_scenarios_exact",
        isinstance(scenarios, list)
        and [
            (row.get("scenario"), row.get("value_type"))
            for row in scenarios
            if isinstance(row, Mapping)
        ]
        == [(scenario, value_type) for scenario, value_type, _ in expected_scenarios]
        and len(scenarios) == len(expected_scenarios),
    )
    if isinstance(scenarios, list):
        for (expected_scenario, expected_kind, expected_reason), row in zip(
            expected_scenarios,
            scenarios,
        ):
            row = row if isinstance(row, Mapping) else {}
            terminal = _get_mapping(row, "terminal")
            persisted = _get_mapping(row, "persisted_apply_result")
            request = _get_mapping(row, "request")
            prefix = f"replay_{expected_scenario}_{expected_kind}"
            expected_address = (
                "/ch/3/fdr"
                if expected_kind == "fader"
                else "/ch/3/in/set/trim"
            )
            expected_old = -12.1 if expected_kind == "fader" else 0.0
            expected_new = -12.35 if expected_kind == "fader" else -0.25
            _record(
                checks,
                f"{prefix}_request_bounded_negative",
                row.get("scenario") == expected_scenario
                and row.get("value_type") == expected_kind
                and request.get("channel") == 3
                and request.get("address") == expected_address
                and request.get("old_value") == expected_old
                and request.get("requested_value") == expected_new
                and request.get("delta") == -0.25
                and request.get("direction") == "negative_only",
            )
            terminal_common = (
                terminal.get("type") == "operator_proposal_apply_blocked"
                and terminal.get("status") == "apply_blocked"
                and terminal.get("reason") == expected_reason
                and terminal.get("proposal_id") == row.get("proposal_id")
            )
            if expected_reason == "pre_read_cache_not_fresh":
                terminal_binding = (
                    isinstance(terminal.get("result_digest"), str)
                    and bool(
                        re.fullmatch(
                            r"sha256:[0-9a-f]{64}",
                            terminal["result_digest"],
                        )
                    )
                )
                result_binding = (
                    row.get("persisted_result_sha256")
                    == terminal.get("result_digest")
                )
                persisted_safe = (
                    persisted.get("no_console_send_proven") is True
                    and persisted.get("reason") == expected_reason
                    and _get_mapping(persisted, "cache_freshness").get("status")
                    == "stale"
                )
            else:
                terminal_binding = terminal.get("result_digest") is None
                result_binding = (
                    row.get("persisted_result_sha256") is None
                    and row.get("persisted_apply_result") is None
                )
                persisted_safe = True
            _record(
                checks,
                f"{prefix}_terminal_blocked",
                terminal_common and terminal_binding,
            )
            _record(
                checks,
                f"{prefix}_result_binding_matches",
                result_binding,
            )
            _record(
                checks,
                f"{prefix}_no_console_send",
                row.get("no_console_send_proven") is True
                and persisted_safe,
            )
    counters = _get_mapping(report, "writer_call_counts")
    expected_counters = {"_apply_manual_console_write", "set_channel_fader", "set_channel_gain", "send", "send_message"}
    _record(checks, "replay_writer_counters_exact_zero", set(counters) == expected_counters and all(counters.get(name) == 0 for name in expected_counters))
    io_counters = _get_mapping(report, "io_guard_call_counts")
    expected_io_counters = {
        "socket_inet_construct",
        "socket_bind",
        "socket_connect",
        "socket_connect_ex",
        "socket_listen",
        "socket_send",
        "socket_sendall",
        "socket_sendmsg",
        "socket_sendto",
        "pyaudio_instance",
        "sounddevice_input_stream",
        "audio_capture_start",
    }
    _record(
        checks,
        "replay_io_guard_counters_exact_zero",
        set(io_counters) == expected_io_counters
        and all(io_counters.get(name) == 0 for name in expected_io_counters),
    )
    for flag in (
        "contacts_backend_transport",
        "contacts_wing",
        "network_listener_opened",
        "audio_device_opened",
        "live_authority_granted",
        "live_write",
        "readiness_closure_allowed",
        "readiness_matrix_update_allowed",
    ):
        _record(checks, f"replay_{flag}_denied", report.get(flag) is False)
    _record(checks, "replay_no_console_send_proven", report.get("no_console_send_proven") is True)
    _record(
        checks,
        "replay_isolation_roots_match_prestart",
        report.get("isolation_roots") == prestart.get("isolation_roots"),
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"ready": not failed, "checks": checks, "failed_checks": failed}


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copyfile(source, destination)


def _merge_checks(
    destination: dict[str, bool],
    prefix: str,
    source: Mapping[str, Any],
) -> None:
    raw_checks = source.get("checks")
    if not isinstance(raw_checks, Mapping):
        destination[f"{prefix}_checks_present"] = False
        return
    for name, passed in raw_checks.items():
        destination[f"{prefix}_{name}"] = passed is True


def _directory_chain_is_real(path: Path) -> bool:
    """Require one existing directory path with no symlink component."""

    candidate = path.absolute()
    for component in (*reversed(candidate.parents), candidate):
        try:
            if component.is_symlink() or not component.is_dir():
                return False
        except OSError:
            return False
    return True


def build_migration_artifact(
    *,
    historical_summary: str | Path,
    owner_decision: str | Path,
    console_attestation: str | Path | None,
    console_attestation_challenge: str | Path | None = None,
    artifact_root: str | Path,
    run_id: str | None = None,
    execute_replay: bool,
    source_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not _is_exact_canonical_input(
        historical_summary,
        EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH,
    ):
        raise ValueError("historical_summary must be the canonical immutable M005 artifact")
    if not _is_exact_canonical_input(
        owner_decision,
        EXPECTED_OWNER_DECISION_LOGICAL_PATH,
    ):
        raise ValueError("owner_decision must be the canonical immutable M006 decision")
    canonical_events = _resolve(EXPECTED_HISTORICAL_EVENTS_LOGICAL_PATH)
    if not canonical_events.is_file() or canonical_events.is_symlink():
        raise ValueError("canonical historical events artifact is missing or is a symlink")
    source = (
        dict(source_override)
        if isinstance(source_override, Mapping)
        else capture_current_source()
    )
    source_checks = _source_checks(source)
    challenge_input: dict[str, Any] = {}
    if (
        console_attestation_challenge is not None
        and str(console_attestation_challenge).strip()
    ):
        challenge_input = _load_json(console_attestation_challenge)
    challenge_run_id = challenge_input.get("migration_run_id")
    if run_id is not None and challenge_run_id is not None and run_id != challenge_run_id:
        raise ValueError("run_id must match the console attestation challenge")
    effective_run_id = _validate_run_id(
        run_id
        or (challenge_run_id if isinstance(challenge_run_id, str) else "")
        or f"{_utc_stamp()}_{uuid.uuid4()}_m005_current_release_migration"
    )
    root_path = _resolve(artifact_root).absolute()
    if root_path.exists():
        if not _directory_chain_is_real(root_path):
            raise ValueError("artifact_root must be a real non-symlink directory")
    else:
        if not _directory_chain_is_real(root_path.parent):
            raise ValueError("artifact_root parent must be a real directory")
        root_path.mkdir(exist_ok=False)
    if not _directory_chain_is_real(root_path):
        raise ValueError("artifact_root must remain a real non-symlink directory")
    root = root_path.resolve()
    run_path = root / effective_run_id
    if run_path.exists() or run_path.is_symlink():
        raise FileExistsError(f"migration run path already exists: {run_path}")
    run_path.mkdir(parents=False, exist_ok=False)
    if not _directory_chain_is_real(run_path):
        raise ValueError("migration run directory must be a real directory")
    run_dir = run_path.resolve()
    if run_dir.parent != root:
        raise ValueError("migration run directory must remain inside artifact_root")
    generated_at_utc = _iso_now()
    generated_at = _parse_utc_z(generated_at_utc)
    if generated_at is None:
        raise RuntimeError("failed to capture canonical migration generation time")

    historical_source = _resolve(historical_summary).resolve()
    events_source = historical_source.with_name("events.jsonl")
    historical_snapshot = run_dir / "historical" / "summary.json"
    events_snapshot = run_dir / "historical" / "events.jsonl"
    decision_snapshot = run_dir / "owner" / "m006_owner_decision.json"
    challenge_snapshot: Path | None = None
    _copy_exact(historical_source, historical_snapshot)
    _copy_exact(events_source, events_snapshot)
    _copy_exact(_resolve(owner_decision).resolve(), decision_snapshot)
    if (
        console_attestation_challenge is not None
        and str(console_attestation_challenge).strip()
    ):
        challenge_snapshot = run_dir / "owner" / "console_attestation_challenge.json"
        _copy_exact(
            _resolve(console_attestation_challenge).resolve(),
            challenge_snapshot,
        )
    if console_attestation is not None and str(console_attestation).strip():
        attestation_snapshot: Path | None = run_dir / "owner" / "console_attestation.json"
        _copy_exact(_resolve(console_attestation).resolve(), attestation_snapshot)
    else:
        attestation_snapshot = None

    historical = verify_historical_evidence(historical_snapshot, events_snapshot)
    decision = verify_owner_decision(decision_snapshot)
    attestation_challenge = verify_console_attestation_challenge(
        challenge_snapshot,
        expected_source=source,
        expected_run_id=effective_run_id,
        observed_at=generated_at,
    )
    attestation = verify_console_attestation(
        attestation_snapshot,
        challenge_path=challenge_snapshot,
        expected_source=source,
        expected_run_id=effective_run_id,
        observed_at=generated_at,
    )
    if execute_replay and all(source_checks.values()):
        replay = run_compatibility_replay(run_dir=run_dir, source=source)
    else:
        replay = {
            "ready": False,
            "command": None,
            "exit_code": None,
            "prestart": None,
            "prestart_sha256": None,
            "prestart_size": None,
            "report_path": None,
            "report_identity": None,
            "stdout_identity": None,
            "stderr_identity": None,
            "report": None,
            "checks": {
                "replay_explicitly_executed": execute_replay,
                "replay_source_was_clean": all(source_checks.values()),
            },
            "failed_checks": [
                name
                for name, passed in {
                    "replay_explicitly_executed": execute_replay,
                    "replay_source_was_clean": all(source_checks.values()),
                }.items()
                if not passed
            ],
        }

    checks: dict[str, bool] = {}
    for name, passed in source_checks.items():
        checks[f"source_{name}"] = passed
    _merge_checks(checks, "historical", historical)
    _merge_checks(checks, "owner_decision", decision)
    _merge_checks(checks, "console_attestation_challenge", attestation_challenge)
    _merge_checks(checks, "console_attestation", attestation)
    _merge_checks(checks, "compatibility", replay)
    _record(checks, "historical_evidence_ready", historical.get("ready") is True)
    _record(checks, "owner_decision_ready", decision.get("ready") is True)
    _record(checks, "current_source_ready", all(source_checks.values()))
    _record(checks, "compatibility_replay_ready", replay.get("ready") is True)
    _record(
        checks,
        "console_attestation_challenge_ready",
        attestation_challenge.get("ready") is True,
    )
    _record(checks, "console_attestation_ready", attestation.get("ready") is True)
    failed = sorted(name for name, passed in checks.items() if not passed)
    ready = not failed
    if ready:
        reason = "m005_current_release_migration_verified"
    elif not historical.get("ready"):
        reason = "historical_m005_evidence_failed"
    elif not all(source_checks.values()):
        reason = "clean_current_source_required"
    elif not replay.get("ready"):
        reason = "current_source_compatibility_replay_required"
    elif not attestation_challenge.get("ready"):
        reason = "fresh_owner_console_attestation_challenge_required"
    elif not attestation.get("ready"):
        reason = "owner_console_identity_attestation_required"
    elif not decision.get("ready"):
        reason = "m006_owner_decision_not_accepted"
    else:
        reason = "m005_current_release_migration_failed"

    output = {
        "schema": MIGRATION_SCHEMA,
        "generated_at_utc": generated_at_utc,
        "run_id": effective_run_id,
        "status": "ready_for_m006" if ready else "blocked",
        "reason": reason,
        "summary_path": _display(run_dir / "summary.json"),
        "historical_m005_summary": EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH,
        "historical_m005_events": EXPECTED_HISTORICAL_EVENTS_LOGICAL_PATH,
        "accepted_readonly_source_id": historical.get(
            "accepted_readonly_source_id"
        ),
        "historical_evidence_ready": historical.get("ready") is True,
        "current_source_ready": all(source_checks.values()),
        "compatibility_replay_ready": replay.get("ready") is True,
        "console_attestation_challenge_ready": attestation_challenge.get("ready")
        is True,
        "console_attestation_ready": attestation.get("ready") is True,
        "owner_decision_ready": decision.get("ready") is True,
        "migration_ready": ready,
        "m005_native_provenance_claimed": False,
        "m005_native_current_release_accepted": False,
        "m005_m006_input_accepted": ready,
        "historical_evidence": {
            **historical,
            "snapshot_summary": _file_identity(historical_snapshot),
            "snapshot_events": _file_identity(events_snapshot),
        },
        "current_source": source,
        "compatibility_replay": replay,
        "owner_decision": {
            **decision,
            "snapshot": _file_identity(decision_snapshot),
        },
        "console_attestation_challenge": {
            **attestation_challenge,
            "snapshot": (
                _file_identity(challenge_snapshot)
                if challenge_snapshot is not None
                else None
            ),
        },
        "console_attestation": {
            **attestation,
            "snapshot": (
                _file_identity(attestation_snapshot)
                if attestation_snapshot is not None
                else None
            ),
        },
        "contacts_backend": False,
        "backend_process_loaded_for_replay": replay.get("report") is not None,
        "contacts_wing": False,
        "network_listener_opened": False,
        "audio_device_opened": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "readiness_matrix_update_allowed": False,
        "live_write": False,
        "checks": checks,
        "failed_checks": failed,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_bytes(canonical_json_bytes(output) + b"\n")
    return output


def _is_regular_confined_file(
    path: Path,
    *,
    run_dir: Path,
    relative_path: str,
) -> bool:
    expected = (run_dir / relative_path).resolve()
    try:
        expected.relative_to(run_dir.resolve())
    except ValueError:
        return False
    return (
        path.exists()
        and path.is_file()
        and not path.is_symlink()
        and path.resolve() == expected
    )


def _valid_replay_window(started_at: Any, ended_at: Any) -> bool:
    try:
        started_text = str(started_at)
        ended_text = str(ended_at)
        if not started_text.endswith("Z") or not ended_text.endswith("Z"):
            return False
        started = datetime.fromisoformat(started_text.replace("Z", "+00:00"))
        ended = datetime.fromisoformat(ended_text.replace("Z", "+00:00"))
    except ValueError:
        return False
    duration = (ended - started).total_seconds()
    return started.tzinfo is not None and ended.tzinfo is not None and 0 <= duration <= 100


def verify_migration_paths(
    summary_path: str | Path,
    *,
    current_source_override: Mapping[str, Any] | None = None,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    summary_file = _resolve(summary_path).absolute()
    resolved_summary = summary_file.resolve()
    run_dir = summary_file.parent
    artifact_root_path = _resolve(artifact_root).absolute()
    resolved_artifact_root = artifact_root_path.resolve()
    resolved_run_dir = run_dir.resolve()
    artifact_root_real = _directory_chain_is_real(artifact_root_path)
    run_dir_real = _directory_chain_is_real(run_dir)
    run_dir_confined = (
        artifact_root_real
        and run_dir_real
        and resolved_run_dir.parent == resolved_artifact_root
    )
    _record(
        checks,
        "migration_artifact_root_real_directory",
        artifact_root_real,
    )
    _record(
        checks,
        "migration_run_directory_real_direct_child",
        run_dir_real and resolved_run_dir.parent == resolved_artifact_root,
    )
    _record(
        checks,
        "migration_summary_regular_confined_file",
        summary_file.is_file()
        and not summary_file.is_symlink()
        and resolved_summary.name == "summary.json"
        and run_dir_confined,
    )
    try:
        summary = _load_json(summary_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        summary = {}
    _record(checks, "migration_schema_matches", summary.get("schema") == MIGRATION_SCHEMA)
    generated_at = _parse_utc_z(summary.get("generated_at_utc"))
    _record(
        checks,
        "migration_generated_at_valid",
        generated_at is not None and generated_at <= datetime.now(timezone.utc),
    )
    _record(
        checks,
        "migration_run_id_matches_directory",
        summary.get("run_id") == run_dir.name
        and bool(RUN_ID_PATTERN.fullmatch(run_dir.name))
        and run_dir.name not in {".", ".."},
    )
    _record(checks, "migration_summary_path_matches", summary.get("summary_path") == _display(summary_file))
    _record(
        checks,
        "migration_historical_logical_paths_match",
        summary.get("historical_m005_summary")
        == EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH
        and summary.get("historical_m005_events")
        == EXPECTED_HISTORICAL_EVENTS_LOGICAL_PATH,
    )

    historical_summary_path = run_dir / "historical" / "summary.json"
    historical_events_path = run_dir / "historical" / "events.jsonl"
    historical_declared = _get_mapping(summary, "historical_evidence")
    summary_snapshot = _get_mapping(historical_declared, "snapshot_summary")
    events_snapshot = _get_mapping(historical_declared, "snapshot_events")
    _record(
        checks,
        "migration_historical_summary_snapshot_confined",
        _is_regular_confined_file(
            historical_summary_path,
            run_dir=run_dir,
            relative_path="historical/summary.json",
        )
        and summary_snapshot == _file_identity(historical_summary_path),
    )
    _record(
        checks,
        "migration_historical_events_snapshot_confined",
        _is_regular_confined_file(
            historical_events_path,
            run_dir=run_dir,
            relative_path="historical/events.jsonl",
        )
        and events_snapshot == _file_identity(historical_events_path),
    )
    historical = verify_historical_evidence(
        historical_summary_path,
        historical_events_path,
    )
    historical_core = {
        key: value
        for key, value in historical_declared.items()
        if key not in {"snapshot_summary", "snapshot_events"}
    }
    _record(
        checks,
        "migration_historical_declared_matches_recomputed",
        historical_core == historical,
    )
    _record(
        checks,
        "migration_accepted_readonly_source_matches_historical",
        summary.get("accepted_readonly_source_id")
        == historical.get("accepted_readonly_source_id")
        == EXPECTED_ACCEPTED_READONLY_SOURCE_ID,
    )
    _merge_checks(checks, "historical", historical)

    current_source = (
        dict(current_source_override)
        if isinstance(current_source_override, Mapping)
        else capture_current_source()
    )
    declared_source = _get_mapping(summary, "current_source")
    source_checks = _source_checks(current_source)
    _record(
        checks,
        "migration_current_source_matches",
        _source_release_binding_matches(declared_source, current_source)
        and all(_source_checks(declared_source).values()),
    )
    for name, passed in source_checks.items():
        checks[f"source_{name}"] = passed

    decision_path = run_dir / "owner" / "m006_owner_decision.json"
    decision_declared = _get_mapping(summary, "owner_decision")
    decision_snapshot = _get_mapping(decision_declared, "snapshot")
    _record(
        checks,
        "migration_owner_decision_snapshot_confined",
        _is_regular_confined_file(
            decision_path,
            run_dir=run_dir,
            relative_path="owner/m006_owner_decision.json",
        )
        and decision_snapshot == _file_identity(decision_path),
    )
    decision = verify_owner_decision(decision_path)
    decision_core = {
        key: value for key, value in decision_declared.items() if key != "snapshot"
    }
    _record(
        checks,
        "migration_owner_decision_declared_matches",
        decision_core == decision,
    )
    _merge_checks(checks, "owner_decision", decision)

    challenge_path = run_dir / "owner" / "console_attestation_challenge.json"
    challenge_declared = _get_mapping(summary, "console_attestation_challenge")
    challenge_snapshot = _get_mapping(challenge_declared, "snapshot")
    _record(
        checks,
        "migration_console_attestation_challenge_snapshot_confined",
        _is_regular_confined_file(
            challenge_path,
            run_dir=run_dir,
            relative_path="owner/console_attestation_challenge.json",
        )
        and challenge_snapshot == _file_identity(challenge_path),
    )
    attestation_challenge = verify_console_attestation_challenge(
        challenge_path,
        expected_source=current_source,
        expected_run_id=run_dir.name,
        observed_at=generated_at,
    )
    challenge_core = {
        key: value
        for key, value in challenge_declared.items()
        if key != "snapshot"
    }
    _record(
        checks,
        "migration_console_attestation_challenge_declared_matches",
        challenge_core == attestation_challenge,
    )
    _merge_checks(
        checks,
        "console_attestation_challenge",
        attestation_challenge,
    )

    attestation_path = run_dir / "owner" / "console_attestation.json"
    attestation_declared = _get_mapping(summary, "console_attestation")
    attestation_snapshot = _get_mapping(attestation_declared, "snapshot")
    _record(
        checks,
        "migration_console_attestation_snapshot_confined",
        _is_regular_confined_file(
            attestation_path,
            run_dir=run_dir,
            relative_path="owner/console_attestation.json",
        )
        and attestation_snapshot == _file_identity(attestation_path),
    )
    attestation = verify_console_attestation(
        attestation_path,
        challenge_path=challenge_path,
        expected_source=current_source,
        expected_run_id=run_dir.name,
        observed_at=generated_at,
    )
    attestation_core = {
        key: value for key, value in attestation_declared.items() if key != "snapshot"
    }
    _record(
        checks,
        "migration_console_attestation_declared_matches",
        attestation_core == attestation,
    )
    _merge_checks(checks, "console_attestation", attestation)

    replay_declared = _get_mapping(summary, "compatibility_replay")
    report_path = run_dir / "compatibility-replay.json"
    stdout_path = run_dir / "compatibility-replay.stdout.log"
    stderr_path = run_dir / "compatibility-replay.stderr.log"
    invocation_path = run_dir / "compatibility-replay.invocation.json"
    report_identity = _get_mapping(replay_declared, "report_identity")
    stdout_identity = _get_mapping(replay_declared, "stdout_identity")
    stderr_identity = _get_mapping(replay_declared, "stderr_identity")
    invocation_identity = _get_mapping(replay_declared, "invocation_identity")
    _record(
        checks,
        "migration_replay_files_confined",
        _is_regular_confined_file(report_path, run_dir=run_dir, relative_path="compatibility-replay.json")
        and _is_regular_confined_file(stdout_path, run_dir=run_dir, relative_path="compatibility-replay.stdout.log")
        and _is_regular_confined_file(stderr_path, run_dir=run_dir, relative_path="compatibility-replay.stderr.log")
        and _is_regular_confined_file(invocation_path, run_dir=run_dir, relative_path="compatibility-replay.invocation.json"),
    )
    _record(checks, "migration_replay_report_identity_matches", report_identity == _file_identity(report_path))
    _record(checks, "migration_replay_stdout_identity_matches", stdout_identity == _file_identity(stdout_path))
    _record(checks, "migration_replay_stderr_identity_matches", stderr_identity == _file_identity(stderr_path))
    _record(checks, "migration_replay_invocation_identity_matches", invocation_identity == _file_identity(invocation_path))
    try:
        report = _load_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        report = {}
    try:
        invocation = _load_json(invocation_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        invocation = {}
    invocation_declared = _get_mapping(replay_declared, "invocation")
    _record(checks, "migration_replay_invocation_declared_matches", invocation_declared == invocation)
    _record(checks, "migration_replay_invocation_schema_matches", invocation.get("schema") == REPLAY_INVOCATION_SCHEMA)
    _record(checks, "migration_replay_invocation_window_valid", _valid_replay_window(invocation.get("started_at_utc"), invocation.get("ended_at_utc")))
    _record(checks, "migration_replay_invocation_exit_zero", invocation.get("exit_code") == 0 and invocation.get("timed_out") is False)
    child_pid = invocation.get("child_pid")
    _record(checks, "migration_replay_parent_child_pids_valid", isinstance(invocation.get("parent_pid"), int) and invocation.get("parent_pid", 0) > 1 and isinstance(child_pid, int) and child_pid > 1)
    challenge = str(invocation.get("challenge") or "")
    launch_nonce = str(invocation.get("launch_nonce") or "")
    _record(checks, "migration_replay_invocation_nonces_valid", bool(re.fullmatch(r"[0-9a-f]{64}", challenge)) and bool(re.fullmatch(r"[0-9a-f]{32}", launch_nonce)))
    prestart = _get_mapping(invocation, "prestart")
    _record(checks, "migration_replay_prestart_declared_matches", _get_mapping(replay_declared, "prestart") == prestart)
    _record(checks, "migration_replay_prestart_nonce_matches", prestart.get("runtime_launch_nonce") == launch_nonce)
    _record(checks, "migration_replay_prestart_hash_matches", invocation.get("prestart_sha256") == replay_declared.get("prestart_sha256") == _canonical_sha256(prestart) and invocation.get("prestart_size") == replay_declared.get("prestart_size") == len(canonical_json_bytes(prestart)))
    expected_isolation = {
        "console_runtime": str((run_dir / "isolated" / "console-runtime").resolve()),
        "operator_results": str((run_dir / "isolated" / "proposal-results").resolve()),
        "manual_write_artifacts": str((run_dir / "isolated" / "manual-write-artifacts").resolve()),
    }
    _record(checks, "migration_replay_isolation_roots_confined", invocation.get("isolation_roots") == prestart.get("isolation_roots") == expected_isolation)
    interpreter = _get_mapping(invocation, "interpreter")
    interpreter_path = str(interpreter.get("path") or "")
    command = invocation.get("command")
    expected_command_tail = [
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-p",
        "pytest_asyncio.plugin",
        "tests/test_m005_current_release_migration_replay.py::test_m005_current_release_migration_replay_emits_bound_report",
        "-q",
    ]
    _record(checks, "migration_replay_interpreter_identity_matches", bool(interpreter_path) and _file_identity(interpreter_path) == interpreter)
    _record(checks, "migration_replay_command_exact", isinstance(command, list) and len(command) == len(expected_command_tail) + 1 and command[1:] == expected_command_tail and Path(str(command[0])).resolve() == Path(interpreter_path).resolve())
    _record(checks, "migration_replay_cwd_exact", invocation.get("cwd") == str(PROJECT_ROOT.resolve()))
    _record(checks, "migration_replay_invocation_artifact_identities_match", invocation.get("report_identity") == report_identity and invocation.get("stdout_identity") == stdout_identity and invocation.get("stderr_identity") == stderr_identity)
    _record(checks, "migration_replay_embedded_report_matches", replay_declared.get("report") == report)
    replay = verify_replay_report(
        report,
        source=current_source,
        prestart=prestart,
        challenge=challenge,
        exit_code=(invocation.get("exit_code") if isinstance(invocation.get("exit_code"), int) else -1),
        expected_child_pid=child_pid if isinstance(child_pid, int) else -1,
    )
    _record(checks, "migration_replay_declared_matches", replay_declared.get("ready") is replay.get("ready") and replay_declared.get("checks") == replay.get("checks") and replay_declared.get("exit_code") == invocation.get("exit_code"))
    _merge_checks(checks, "compatibility", replay)

    for flag in (
        "contacts_backend",
        "contacts_wing",
        "network_listener_opened",
        "audio_device_opened",
        "live_authority_granted",
        "readiness_closure_allowed",
        "readiness_matrix_update_allowed",
        "live_write",
    ):
        _record(checks, f"migration_{flag}_denied", summary.get(flag) is False)
    _record(checks, "migration_backend_replay_claim_matches", summary.get("backend_process_loaded_for_replay") is (bool(report)))
    _record(checks, "migration_native_provenance_not_claimed", summary.get("m005_native_provenance_claimed") is False and summary.get("m005_native_current_release_accepted") is False)
    roots_ready = {
        "historical_evidence_ready": historical.get("ready") is True,
        "current_source_ready": all(source_checks.values()),
        "compatibility_replay_ready": replay.get("ready") is True,
        "console_attestation_challenge_ready": attestation_challenge.get("ready")
        is True,
        "console_attestation_ready": attestation.get("ready") is True,
        "owner_decision_ready": decision.get("ready") is True,
    }
    for field_name, expected in roots_ready.items():
        _record(checks, f"migration_{field_name}_claim_matches", summary.get(field_name) is expected)
    recomputed_ready = all(roots_ready.values())
    _record(checks, "migration_ready_claim_matches", summary.get("migration_ready") is recomputed_ready and summary.get("m005_m006_input_accepted") is recomputed_ready)
    _record(checks, "migration_status_matches", summary.get("status") == ("ready_for_m006" if recomputed_ready else "blocked"))
    expected_reason = "m005_current_release_migration_verified" if recomputed_ready else summary.get("reason")
    _record(checks, "migration_ready_reason_matches", not recomputed_ready or summary.get("reason") == expected_reason)
    failed = sorted(name for name, passed in checks.items() if not passed)
    ready = not failed and recomputed_ready
    return {
        "schema": MIGRATION_SCHEMA,
        "status": "ready_for_m006" if ready else "blocked",
        "reason": "m005_current_release_migration_verified" if ready else "m005_current_release_migration_verification_failed",
        "summary_path": _display(summary_file),
        "historical_m005_summary": EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH,
        "historical_m005_events": EXPECTED_HISTORICAL_EVENTS_LOGICAL_PATH,
        "accepted_readonly_source_id": historical.get("accepted_readonly_source_id"),
        **roots_ready,
        "migration_ready": ready,
        "m005_native_provenance_claimed": False,
        "m005_native_current_release_accepted": False,
        "m005_m006_input_accepted": ready,
        "contacts_backend": False,
        "contacts_wing": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "readiness_matrix_update_allowed": False,
        "live_write": False,
        "checks": checks,
        "failed_checks": failed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a report-only migration for the closed historical M005 proof."
    )
    parser.add_argument("--historical-summary", default=DEFAULT_HISTORICAL_SUMMARY)
    parser.add_argument("--owner-decision", default=DEFAULT_OWNER_DECISION)
    parser.add_argument("--console-attestation", default=DEFAULT_CONSOLE_ATTESTATION)
    parser.add_argument(
        "--console-attestation-challenge",
        default=DEFAULT_CONSOLE_ATTESTATION_CHALLENGE,
    )
    parser.add_argument(
        "--write-attestation-challenge",
        help="Write a fresh source/run-bound owner-attestation challenge and stop.",
    )
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--execute-replay", action="store_true")
    parser.add_argument("--verify-summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_attestation_challenge:
        result = write_console_attestation_challenge(
            args.write_attestation_challenge,
            migration_run_id=args.run_id,
        )
    elif args.verify_summary:
        result = verify_migration_paths(
            args.verify_summary,
            artifact_root=args.artifact_root,
        )
    else:
        result = build_migration_artifact(
            historical_summary=args.historical_summary,
            owner_decision=args.owner_decision,
            console_attestation=(args.console_attestation or None),
            console_attestation_challenge=(
                args.console_attestation_challenge or None
            ),
            artifact_root=args.artifact_root,
            run_id=args.run_id,
            execute_replay=bool(args.execute_replay),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return (
        0
        if result.get("status")
        in {"ready_for_m006", "challenge_ready_for_owner_attestation"}
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
