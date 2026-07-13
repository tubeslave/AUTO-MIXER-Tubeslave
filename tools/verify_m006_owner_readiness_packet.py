#!/usr/bin/env python3
"""Verify the report-only M006 owner readiness packet before owner review.

This verifier is offline only. It reads packet artifacts from disk, never opens
backend/WING transport, and cannot grant readiness or live authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from runtime_identity import (  # noqa: E402
    RUNTIME_IDENTITY_SCHEMA,
    canonical_json_bytes,
    capture_source_runtime_components,
    runtime_identity_manifest_sha256,
    valid_lower_hex,
)
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
EXPECTED_M005_READBACK_TARGET_BINDING = [
    "baseline_readback",
    "stale_readback",
    "reconnect_readback",
    "events_stale_readback",
    "events_reconnect_readback",
]
EXPECTED_M005_SAFE_BLOCK_REASONS = {
    "manual_write_exception",
    "mixer_not_connected",
    "mixer_write_failed",
    "pre_read_cache_not_fresh",
    "pre_read_unavailable",
}
EXPECTED_M005_STALE_TIMEOUT_REASON = "stale_readback_timeout_after_disconnect"
EXPECTED_M005_SAFE_STALE_READBACK_BLOCK_REASONS = {
    EXPECTED_M005_STALE_TIMEOUT_REASON,
    "mixer_client_state_source_unavailable",
    "mixer_not_connected",
    "pre_read_cache_not_fresh",
    "pre_read_unavailable",
    "readback_unavailable",
}
EXPECTED_M005_FORBIDDEN_DIRECT_REQUEST_TYPES = {
    "load_" + "snap",
    "route_" + "output",
    "save_" + "snap",
    "set_" + "fader",
    "set_" + "gain",
}
EXPECTED_M005_FORBIDDEN_REMEDIATION_REQUEST_TYPES = {
    "rollback_last_manual_write",
}
EXPECTED_M005_STALE_EVENT_TYPES = {"recv", "synthetic_recv_timeout"}
EXPECTED_M005_STALE_EVENT_PAYLOAD_TYPES = {
    "mixer_parameter_read_blocked",
    "mixer_parameter_value",
}
EXPECTED_M005_RECONNECT_EVENT_TYPES = {"recv"}
EXPECTED_M005_RECONNECT_EVENT_PAYLOAD_TYPES = {"mixer_parameter_value"}
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
EXPECTED_M005_PROVENANCE_SCHEMA = "automixer.m005_current_release_provenance.v1"
EXPECTED_M005_RUNNER_PATH = "tools/wing_timeout_disconnect_proof_runner.py"
EXPECTED_SOURCE_BACKEND_PATH = "backend/server.py"
EXPECTED_M005_CANONICAL_URL = "ws://127.0.0.1:8765"
EXPECTED_M005_RUNTIME_IDENTITY_SCHEMA = RUNTIME_IDENTITY_SCHEMA
EXPECTED_M005_CONSOLE_IDENTITY_SCHEMA = "automixer.wing_console_identity_owner_challenge.v1"
EXPECTED_M005_CONSOLE_IDENTITY_ROOT = "artifacts/wing_console_identity_owner_challenge"
EXPECTED_M005_CONSOLE_MODEL = "WING Rack"
EXPECTED_M005_CONSOLE_FIRMWARE = "3.0.5"
MAX_M005_BOUND_BACKEND_START_DELAY_SECONDS = 600


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display_path = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        display_path = str(resolved)
    if not resolved.is_file():
        return {"path": display_path, "exists": False, "size": None, "sha256": None}
    value = resolved.read_bytes()
    return {
        "path": display_path,
        "exists": True,
        "size": len(value),
        "sha256": _sha256_bytes(value),
    }


def _git_bytes(*args: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", *args],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({completed.returncode}): "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return completed.stdout


def _status_excluding_m005_run(m005_summary_path: str | Path | None) -> bytes:
    raw = _git_bytes("status", "--porcelain=v1", "-z", "-uall")
    if m005_summary_path is None:
        return raw
    summary_path = _resolve(m005_summary_path).resolve()
    try:
        run_dir = summary_path.parent.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return raw
    retained: list[bytes] = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        text = entry.decode("utf-8", errors="surrogateescape")
        candidate = text[3:] if len(text) >= 4 else ""
        if candidate == run_dir or candidate.startswith(f"{run_dir}/"):
            continue
        retained.append(entry)
    return b"\0".join(retained) + (b"\0" if retained else b"")


def current_release_provenance_expectations(
    m005_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    status = _status_excluding_m005_run(m005_summary_path)
    tracked_diff = _git_bytes("diff", "HEAD", "--binary")
    return {
        "source": {
            "head": _git_bytes("rev-parse", "HEAD").decode().strip(),
            "tree": _git_bytes("rev-parse", "HEAD^{tree}").decode().strip(),
            "worktree_dirty": bool(status.strip()),
            "status_sha256": _sha256_bytes(status),
            "tracked_diff_sha256": _sha256_bytes(tracked_diff),
        },
        "runner": _file_identity(PROJECT_ROOT / EXPECTED_M005_RUNNER_PATH),
        "source_backend": _file_identity(PROJECT_ROOT / EXPECTED_SOURCE_BACKEND_PATH),
        "runtime_source": capture_source_runtime_components(PROJECT_ROOT),
    }


def _display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _resolve_m005_events_path(
    *,
    m005_summary_path: str | Path | None,
    m005_summary: Mapping[str, Any],
) -> Path | None:
    if m005_summary_path is None:
        return None

    summary_path = _resolve(m005_summary_path).resolve()
    events_text = m005_summary.get("events_path")
    if isinstance(events_text, str) and events_text and events_text.strip() == events_text:
        event_candidate = Path(events_text)
        if ".." in event_candidate.parts:
            return None
        events_path = event_candidate if event_candidate.is_absolute() else PROJECT_ROOT / event_candidate
    else:
        events_path = summary_path.parent / "events.jsonl"

    resolved_events = events_path.resolve()
    if resolved_events.name != "events.jsonl":
        return None
    if resolved_events.parent != summary_path.parent:
        return None
    return resolved_events


def load_m005_console_identity_challenge(
    m005_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(m005_summary, Mapping):
        return None
    provenance = _get_mapping(m005_summary, "provenance")
    declared = _get_mapping(provenance, "console_identity_challenge")
    path_text = declared.get("path")
    if not isinstance(path_text, str) or not path_text or path_text.strip() != path_text:
        return None
    candidate = Path(path_text)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    expected_root = Path(EXPECTED_M005_CONSOLE_IDENTITY_ROOT)
    if candidate.name != "summary.json" or not candidate.is_relative_to(expected_root):
        return None
    resolved = (PROJECT_ROOT / candidate).resolve()
    if not resolved.is_file():
        return None
    identity = _file_identity(resolved)
    if dict(declared) != identity:
        return None
    try:
        content = _load_json(resolved)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return {"identity": identity, "content": content}


def load_m005_event_request_types(
    *,
    m005_summary_path: str | Path | None,
    m005_summary: Mapping[str, Any] | None,
) -> tuple[list[str] | None, str | None]:
    audit, display_events_path = load_m005_event_audit(
        m005_summary_path=m005_summary_path,
        m005_summary=m005_summary,
    )
    if audit is None:
        return None, display_events_path
    request_types = audit.get("request_types")
    if not isinstance(request_types, list):
        return None, display_events_path
    return [str(item) for item in request_types], display_events_path


def load_m005_event_audit(
    *,
    m005_summary_path: str | Path | None,
    m005_summary: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if m005_summary is None:
        return None, None
    events_path = _resolve_m005_events_path(
        m005_summary_path=m005_summary_path,
        m005_summary=m005_summary,
    )
    display_events_path = _display_path(events_path)
    if events_path is None or not events_path.is_file():
        return None, display_events_path

    request_types: list[str] = []
    rows: list[Mapping[str, Any]] = []
    try:
        for raw_line in events_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, Mapping):
                return None, display_events_path
            rows.append(row)
            if row.get("event_type") != "send":
                continue
            payload = row.get("payload")
            if not isinstance(payload, Mapping):
                return None, display_events_path
            request_type = payload.get("type")
            if not isinstance(request_type, str) or not request_type:
                return None, display_events_path
            request_types.append(request_type)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, display_events_path
    return {
        "request_types": request_types,
        "initial_connection": _m005_event_initial_connection_payload(rows),
        "stale_readback": _m005_event_stale_readback_payload(rows),
        "reconnect_readback": _m005_event_reconnect_readback_payload(rows),
        "first_event": dict(rows[0]) if rows else None,
        "runtime_identity": next(
            (
                dict(row.get("payload"))
                for row in rows
                if row.get("event_type") == "recv"
                and isinstance(row.get("payload"), Mapping)
                and row.get("payload", {}).get("type") == "runtime_identity"
            ),
            None,
        ),
        "sha256": _sha256_bytes(events_path.read_bytes()),
    }, display_events_path


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


def _m005_payload_target_field(payload: Mapping[str, Any], field_name: str) -> Any:
    if field_name in payload:
        return payload.get(field_name)
    request = payload.get("request")
    if isinstance(request, Mapping):
        return request.get(field_name)
    return None


def _m005_summary_target(m005_summary: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        channel = int(m005_summary.get("channel"))
    except (TypeError, ValueError):
        return None
    value_type = m005_summary.get("value_type")
    address = m005_summary.get("address")
    if not isinstance(value_type, str) or not value_type:
        return None
    if not isinstance(address, str) or not address:
        return None
    return {"channel": channel, "value_type": value_type, "address": address}


def _m005_readback_matches_target(
    readback: Mapping[str, Any],
    m005_summary: Mapping[str, Any],
) -> bool:
    target = _m005_summary_target(m005_summary)
    if target is None:
        return False
    try:
        channel_matches = int(_m005_payload_target_field(readback, "channel")) == target["channel"]
    except (TypeError, ValueError):
        channel_matches = False
    return (
        channel_matches
        and _m005_payload_target_field(readback, "value_type") == target["value_type"]
        and _m005_payload_target_field(readback, "address") == target["address"]
    )


def _m005_readback_is_timeout_evidence(readback: Mapping[str, Any]) -> bool:
    return (
        readback.get("status") == "readback_timeout"
        or readback.get("source") == "timeout"
        or readback.get("reason") == EXPECTED_M005_STALE_TIMEOUT_REASON
    )


def _m005_timeout_readback_is_unavailable(readback: Mapping[str, Any]) -> bool:
    transport_failure = _get_mapping(readback, "transport_failure")
    return (
        readback.get("type") == "mixer_parameter_read_blocked"
        and readback.get("status") == "readback_timeout"
        and readback.get("source") == "timeout"
        and readback.get("reason") == EXPECTED_M005_STALE_TIMEOUT_REASON
        and readback.get("value") is None
        and transport_failure.get("type") == "TimeoutError"
    )


def _m005_blocked_readback_is_unavailable(readback: Mapping[str, Any]) -> bool:
    if _m005_readback_is_timeout_evidence(readback):
        return _m005_timeout_readback_is_unavailable(readback)
    return (
        readback.get("type") == "mixer_parameter_read_blocked"
        and readback.get("status") == "blocked"
        and readback.get("reason") in EXPECTED_M005_SAFE_STALE_READBACK_BLOCK_REASONS
        and readback.get("value") is None
    )


def _m005_readback_is_unavailable(readback: Mapping[str, Any]) -> bool:
    if readback.get("type") == "mixer_parameter_read_blocked":
        return _m005_blocked_readback_is_unavailable(readback)
    return (
        readback.get("type") == "mixer_parameter_value"
        and readback.get("status") == "readback_unavailable"
        and readback.get("value") is None
        and readback.get("source") in {"physical_readback", "unavailable"}
    )


def _m005_apply_block_reason(payload: Mapping[str, Any]) -> str | None:
    result = payload.get("apply_result")
    if isinstance(result, Mapping) and result.get("reason"):
        return str(result.get("reason"))
    if payload.get("reason"):
        return str(payload.get("reason"))
    return None


def _m005_apply_probe_blocked(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("type") == "operator_proposal_apply_blocked"
        and _m005_apply_block_reason(payload) in EXPECTED_M005_SAFE_BLOCK_REASONS
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


def _is_repo_relative_m005_summary_path(path_text: Any) -> bool:
    if not isinstance(path_text, str):
        return False
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts:
        return False
    return (
        len(path.parts) >= 4
        and path.parts[0] == "artifacts"
        and path.parts[1] == "wing_timeout_disconnect_proof"
        and path.name == "summary.json"
    )


def _m005_events_path_matches_summary_path(
    *,
    m005_summary_path: str | None,
    m005_events_path: str | None,
) -> bool:
    if not isinstance(m005_summary_path, str) or not isinstance(m005_events_path, str):
        return False
    summary_path = Path(m005_summary_path)
    events_path = Path(m005_events_path)
    if summary_path.is_absolute() or events_path.is_absolute():
        return False
    if ".." in summary_path.parts or ".." in events_path.parts:
        return False
    return events_path == summary_path.with_name("events.jsonl")


def _m005_disconnect_flow_indexes(request_types: Any) -> dict[str, int] | None:
    if not isinstance(request_types, list):
        return None
    sequence = [item for item in request_types if isinstance(item, str)]
    if len(sequence) != len(request_types):
        return None

    def find_after(request_type: str, start_index: int) -> int | None:
        try:
            return sequence.index(request_type, start_index)
        except ValueError:
            return None

    runtime_identity = find_after("get_runtime_identity", 0)
    if runtime_identity != 0:
        return None
    initial_connect = find_after("connect_wing", runtime_identity + 1)
    if initial_connect != 1:
        return None
    baseline_query = find_after("query_mixer_parameter", initial_connect + 1)
    if baseline_query is None:
        return None
    supervised_mode = find_after("set_operator_mode", baseline_query + 1)
    if supervised_mode is None:
        return None
    proposal = find_after("create_operator_proposal", supervised_mode + 1)
    if proposal is None:
        return None
    stale_query = find_after("query_mixer_parameter", proposal + 1)
    if stale_query is None:
        return None
    blocked_apply = find_after("apply_operator_proposal", stale_query + 1)
    if blocked_apply is None:
        return None
    reconnect = find_after("connect_wing", blocked_apply + 1)
    if reconnect is None:
        return None
    reconnect_query = find_after("query_mixer_parameter", reconnect + 1)
    if reconnect_query is None:
        return None

    if find_after("apply_operator_proposal", 0) != blocked_apply:
        return None
    if find_after("query_mixer_parameter", 0) != baseline_query:
        return None
    if find_after("connect_wing", 0) != initial_connect:
        return None
    if find_after("connect_wing", initial_connect + 1) != reconnect:
        return None
    if find_after("connect_wing", reconnect + 1) is not None:
        return None

    return {
        "runtime_identity": runtime_identity,
        "initial_connect": initial_connect,
        "baseline_query": baseline_query,
        "supervised_mode": supervised_mode,
        "proposal": proposal,
        "stale_query": stale_query,
        "blocked_apply": blocked_apply,
        "reconnect": reconnect,
        "reconnect_query": reconnect_query,
    }


def _m005_request_sequence_proves_disconnect_flow(request_types: Any) -> bool:
    return _m005_disconnect_flow_indexes(request_types) is not None


def _m005_event_payload_between_requests(
    rows: list[Mapping[str, Any]],
    *,
    after_request: str,
    before_request: str | None,
    expected_event_types: set[str],
    expected_payload_types: set[str],
) -> dict[str, Any] | None:
    request_types: list[str] = []
    send_row_indexes: list[int] = []
    for row_index, row in enumerate(rows):
        if row.get("event_type") != "send":
            continue
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            return None
        request_type = payload.get("type")
        if not isinstance(request_type, str) or not request_type:
            return None
        request_types.append(request_type)
        send_row_indexes.append(row_index)

    flow_indexes = _m005_disconnect_flow_indexes(request_types)
    if flow_indexes is None:
        return None

    start_row = send_row_indexes[flow_indexes[after_request]]
    if before_request is None:
        candidate_rows = rows[start_row + 1 :]
    else:
        end_row = send_row_indexes[flow_indexes[before_request]]
        candidate_rows = rows[start_row + 1 : end_row]

    for row in candidate_rows:
        if row.get("event_type") not in expected_event_types:
            continue
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            return None
        if payload.get("type") not in expected_payload_types:
            continue
        return dict(payload)
    return None


def _m005_event_stale_readback_payload(rows: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    return _m005_event_payload_between_requests(
        rows,
        after_request="stale_query",
        before_request="blocked_apply",
        expected_event_types=EXPECTED_M005_STALE_EVENT_TYPES,
        expected_payload_types=EXPECTED_M005_STALE_EVENT_PAYLOAD_TYPES,
    )


def _m005_event_initial_connection_payload(
    rows: list[Mapping[str, Any]],
) -> dict[str, Any] | None:
    return _m005_event_payload_between_requests(
        rows,
        after_request="initial_connect",
        before_request="baseline_query",
        expected_event_types={"recv"},
        expected_payload_types={"connection_status"},
    )


def _m005_event_reconnect_readback_payload(rows: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    return _m005_event_payload_between_requests(
        rows,
        after_request="reconnect_query",
        before_request=None,
        expected_event_types=EXPECTED_M005_RECONNECT_EVENT_TYPES,
        expected_payload_types=EXPECTED_M005_RECONNECT_EVENT_PAYLOAD_TYPES,
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
        "m005_contract_requires_blocked_apply_result",
        contract.get("required_blocked_write_apply_result_type")
        == "operator_proposal_apply_blocked",
    )
    _record(
        checks,
        "m005_contract_requires_no_rollback_request",
        contract.get("required_rollback_request_sent") is False,
    )
    _record(
        checks,
        "m005_contract_requires_readback_target_binding",
        contract.get("required_readback_target_binding")
        == EXPECTED_M005_READBACK_TARGET_BINDING,
    )
    _record(
        checks,
        "m005_contract_forbids_remediation_request_types",
        contract.get("forbidden_remediation_request_types")
        == sorted(EXPECTED_M005_FORBIDDEN_REMEDIATION_REQUEST_TYPES),
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
    m005_event_request_types: list[str] | None = None,
    m005_event_initial_connection: Mapping[str, Any] | None = None,
    m005_event_stale_readback: Mapping[str, Any] | None = None,
    m005_event_reconnect_readback: Mapping[str, Any] | None = None,
    m005_event_stale_readback_required: bool = False,
    m005_summary_path: str | None = None,
    m005_events_path: str | None = None,
    m005_events_sha256: str | None = None,
    m005_event_audit: Mapping[str, Any] | None = None,
    m005_console_identity_challenge: Mapping[str, Any] | None = None,
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
    _record(
        checks,
        "m005_summary_path_supplied",
        isinstance(m005_summary_path, str) and bool(m005_summary_path),
    )
    _record(
        checks,
        "m005_summary_path_matches_required_glob",
        _is_repo_relative_m005_summary_path(m005_summary_path),
    )
    _record(
        checks,
        "m005_summary_declared_path_matches_input",
        isinstance(m005_summary_path, str) and m005_summary.get("summary_path") == m005_summary_path,
    )
    _record(
        checks,
        "m005_summary_declared_events_path_matches_loaded",
        isinstance(m005_events_path, str) and m005_summary.get("events_path") == m005_events_path,
    )
    _record(
        checks,
        "m005_events_path_matches_summary_sibling",
        _m005_events_path_matches_summary_path(
            m005_summary_path=m005_summary_path,
            m005_events_path=m005_events_path,
        ),
    )
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
    _record(
        checks,
        "m005_summary_url_is_canonical_loopback",
        m005_summary.get("url") == EXPECTED_M005_CANONICAL_URL,
    )
    for field_name, expected in AUTHORITY_DENIAL_FLAGS.items():
        _record(checks, f"m005_summary_{field_name}_denied", m005_summary.get(field_name) is expected)
    sent_request_types = m005_summary.get("sent_request_types")
    rollback_request_sent = m005_summary.get("rollback_request_sent")
    if isinstance(sent_request_types, list):
        m005_sent_request_types = [str(item) for item in sent_request_types]
    else:
        m005_sent_request_types = []
    forbidden_from_sent = [
        request_type
        for request_type in m005_sent_request_types
        if request_type in EXPECTED_M005_FORBIDDEN_DIRECT_REQUEST_TYPES
    ]
    remediation_from_sent = [
        request_type
        for request_type in m005_sent_request_types
        if request_type in EXPECTED_M005_FORBIDDEN_REMEDIATION_REQUEST_TYPES
    ]
    if isinstance(m005_event_request_types, list):
        event_request_types = [str(item) for item in m005_event_request_types]
    else:
        event_request_types = []
    forbidden_from_events = [
        request_type
        for request_type in event_request_types
        if request_type in EXPECTED_M005_FORBIDDEN_DIRECT_REQUEST_TYPES
    ]
    remediation_from_events = [
        request_type
        for request_type in event_request_types
        if request_type in EXPECTED_M005_FORBIDDEN_REMEDIATION_REQUEST_TYPES
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
        "m005_summary_request_sequence_proves_disconnect_flow",
        _m005_request_sequence_proves_disconnect_flow(m005_sent_request_types),
    )
    _record(
        checks,
        "m005_summary_no_forbidden_direct_request_types",
        forbidden_from_sent == [],
    )
    _record(
        checks,
        "m005_summary_no_forbidden_remediation_request_types",
        remediation_from_sent == [],
    )
    _record(
        checks,
        "m005_summary_forbidden_requests_match_sent_audit",
        m005_summary.get("forbidden_request_types_sent") == forbidden_from_sent,
    )
    _record(
        checks,
        "m005_summary_rollback_request_not_sent",
        rollback_request_sent is False,
    )
    _record(
        checks,
        "m005_summary_rollback_request_matches_sent_audit",
        isinstance(rollback_request_sent, bool)
        and rollback_request_sent == bool(remediation_from_sent),
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
    _record(
        checks,
        "m005_events_request_audit_loaded",
        isinstance(m005_event_request_types, list) and bool(event_request_types),
    )
    _record(
        checks,
        "m005_events_request_audit_matches_summary",
        isinstance(m005_event_request_types, list)
        and event_request_types == m005_sent_request_types,
    )
    _record(
        checks,
        "m005_events_apply_probe_sent",
        "apply_operator_proposal" in event_request_types,
    )
    _record(
        checks,
        "m005_events_request_sequence_proves_disconnect_flow",
        _m005_request_sequence_proves_disconnect_flow(event_request_types),
    )
    _record(
        checks,
        "m005_events_no_forbidden_direct_request_types",
        forbidden_from_events == [],
    )
    _record(
        checks,
        "m005_events_no_forbidden_remediation_request_types",
        remediation_from_events == [],
    )

    phases = _get_mapping(m005_summary, "phases")
    initial_connection = _get_mapping(phases, "initial_connection")
    baseline = _get_mapping(phases, "baseline")
    disconnect = _get_mapping(phases, "disconnect")
    stale = _get_mapping(phases, "stale_readback")
    blocked_write = _get_mapping(phases, "blocked_write_while_disconnected")
    reconnect = _get_mapping(phases, "reconnect_recovery")
    initial_connection_status = _get_mapping(initial_connection, "connection_status")
    baseline_readback = _get_mapping(baseline, "readback")
    stale_readback = _get_mapping(stale, "readback")
    blocked_apply_result = _get_mapping(blocked_write, "apply_result")
    reconnect_status = _get_mapping(reconnect, "connection_status")
    reconnect_readback = _get_mapping(reconnect, "readback")
    initial_connection_event_present = isinstance(
        m005_event_initial_connection, Mapping
    ) and bool(m005_event_initial_connection)
    stale_event_present = isinstance(m005_event_stale_readback, Mapping) and bool(
        m005_event_stale_readback
    )
    reconnect_event_present = isinstance(m005_event_reconnect_readback, Mapping) and bool(
        m005_event_reconnect_readback
    )
    if m005_event_stale_readback_required:
        _record(
            checks,
            "m005_events_initial_connection_payload_present",
            initial_connection_event_present,
        )
        _record(
            checks,
            "m005_events_initial_connection_matches_summary",
            initial_connection_event_present
            and dict(m005_event_initial_connection) == dict(initial_connection_status),
        )
        _record(
            checks,
            "m005_events_initial_connection_write_capable",
            initial_connection_event_present
            and m005_event_initial_connection.get("connected") is True
            and m005_event_initial_connection.get("mode") == "wing"
            and m005_event_initial_connection.get("ip") == m005_summary.get("wing_ip")
            and m005_event_initial_connection.get("observation_only") is False
            and m005_event_initial_connection.get("record_telemetry") is False,
        )
        _record(
            checks,
            "m005_events_stale_readback_payload_present",
            stale_event_present,
        )
        _record(
            checks,
            "m005_events_stale_readback_matches_summary",
            stale_event_present and dict(m005_event_stale_readback) == dict(stale_readback),
        )
        _record(
            checks,
            "m005_events_stale_readback_unavailable",
            stale_event_present and _m005_readback_is_unavailable(m005_event_stale_readback),
        )
        _record(
            checks,
            "m005_events_reconnect_readback_payload_present",
            reconnect_event_present,
        )
        _record(
            checks,
            "m005_events_reconnect_readback_matches_summary",
            reconnect_event_present
            and dict(m005_event_reconnect_readback) == dict(reconnect_readback),
        )
        _record(
            checks,
            "m005_events_reconnect_readback_physical",
            reconnect_event_present and _m005_readback_is_physical(m005_event_reconnect_readback),
    )
    _record(
        checks,
        "m005_summary_target_fields_present",
        _m005_summary_target(m005_summary) is not None,
    )
    _record(
        checks,
        "m005_initial_connection_phase_passed",
        m005_summary.get("initial_connection_verified") is True
        and initial_connection.get("status") == "passed"
        and initial_connection_status.get("connected") is True
        and initial_connection_status.get("mode") == "wing"
        and initial_connection_status.get("ip") == m005_summary.get("wing_ip")
        and initial_connection_status.get("observation_only") is False
        and initial_connection_status.get("record_telemetry") is False,
    )
    _record(checks, "m005_baseline_phase_passed", baseline.get("status") == "passed")
    _record(
        checks,
        "m005_baseline_readback_target_matches",
        _m005_readback_matches_target(baseline_readback, m005_summary),
    )
    _record(
        checks,
        "m005_baseline_physical_readback",
        baseline.get("status") == "passed"
        and _m005_readback_is_physical(baseline_readback),
    )
    _record(
        checks,
        "m005_disconnect_phase_operator_confirmed_physical_loss",
        disconnect.get("status") == "operator_confirmed_physical_cable_loss"
        and disconnect.get("confirmation") == EXPECTED_M005_PHYSICAL_CABLE_CONFIRMATION,
    )
    _record(
        checks,
        "m005_disconnect_phase_operator_gate_required",
        disconnect.get("operator_gate") is True,
    )
    _record(checks, "m005_stale_readback_phase_passed", stale.get("status") == "passed")
    _record(
        checks,
        "m005_stale_readback_unavailable",
        stale.get("status") == "passed" and _m005_readback_is_unavailable(stale_readback),
    )
    _record(
        checks,
        "m005_stale_readback_target_matches",
        stale.get("status") == "passed"
        and _m005_readback_matches_target(stale_readback, m005_summary),
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
        "m005_blocked_write_apply_result_blocked",
        blocked_write.get("status") == "passed"
        and _m005_apply_probe_blocked(blocked_apply_result),
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
    _record(
        checks,
        "m005_reconnect_readback_target_matches",
        reconnect.get("status") == "passed"
        and _m005_readback_matches_target(reconnect_readback, m005_summary),
    )
    if m005_event_stale_readback_required:
        _record(
            checks,
            "m005_events_stale_readback_target_matches",
            stale_event_present
            and _m005_readback_matches_target(m005_event_stale_readback, m005_summary),
        )
        _record(
            checks,
            "m005_events_reconnect_readback_target_matches",
            reconnect_event_present
            and _m005_readback_matches_target(m005_event_reconnect_readback, m005_summary),
        )

    _record_m005_provenance_checks(
        checks,
        m005_summary=m005_summary,
        m005_events_path=m005_events_path,
        m005_events_sha256=m005_events_sha256,
        m005_event_audit=m005_event_audit,
        m005_console_identity_challenge=m005_console_identity_challenge,
        m005_summary_path=m005_summary_path,
    )


def _record_m005_provenance_checks(
    checks: dict[str, bool],
    *,
    m005_summary: Mapping[str, Any],
    m005_events_path: str | None,
    m005_events_sha256: str | None,
    m005_event_audit: Mapping[str, Any] | None,
    m005_console_identity_challenge: Mapping[str, Any] | None,
    m005_summary_path: str | None,
) -> None:
    provenance = _get_mapping(m005_summary, "provenance")
    source = _get_mapping(provenance, "source")
    declared_runtime_source = _get_mapping(provenance, "runtime_source")
    runner = _get_mapping(provenance, "runner")
    backend = _get_mapping(provenance, "backend")
    console = _get_mapping(provenance, "console")
    events = _get_mapping(provenance, "events")
    expected = current_release_provenance_expectations(m005_summary_path)

    _record(
        checks,
        "m005_provenance_schema_matches",
        provenance.get("schema") == EXPECTED_M005_PROVENANCE_SCHEMA,
    )
    for field_name, expected_value in expected["source"].items():
        _record(
            checks,
            f"m005_provenance_source_{field_name}_matches_current",
            source.get(field_name) == expected_value,
        )
    _record(
        checks,
        "m005_provenance_source_clean_before_run",
        source.get("worktree_dirty") is False
        and source.get("status_sha256") == _sha256_bytes(b"")
        and source.get("tracked_diff_sha256") == _sha256_bytes(b""),
    )
    _record(
        checks,
        "m005_provenance_current_checkout_clean_excluding_run_artifact",
        expected["source"]["worktree_dirty"] is False
        and expected["source"]["status_sha256"] == _sha256_bytes(b"")
        and expected["source"]["tracked_diff_sha256"] == _sha256_bytes(b""),
    )
    expected_runtime_source = expected["runtime_source"]
    runtime_source_code = _get_mapping(declared_runtime_source, "code")
    runtime_source_source = _get_mapping(declared_runtime_source, "source")
    runtime_source_files = runtime_source_code.get("files")
    _record(
        checks,
        "m005_provenance_runtime_source_matches_current",
        dict(declared_runtime_source) == expected_runtime_source,
    )
    _record(
        checks,
        "m005_provenance_runtime_source_hash_is_self_consistent",
        isinstance(runtime_source_files, list)
        and runtime_source_code.get("files_sha256")
        == _canonical_json_sha256(runtime_source_files)
        and runtime_source_source.get("source_content_sha256")
        == runtime_source_code.get("files_sha256"),
    )
    _record(
        checks,
        "m005_provenance_runtime_launch_fields_present",
        valid_lower_hex(provenance.get("runtime_launch_nonce"), 32)
        and isinstance(provenance.get("captured_at_utc"), str)
        and bool(str(provenance.get("captured_at_utc")).strip()),
    )
    _record(
        checks,
        "m005_provenance_runner_matches_current",
        dict(runner) == expected["runner"],
    )
    runtime_kind = backend.get("runtime_kind")
    _record(
        checks,
        "m005_provenance_backend_runtime_kind_supported",
        runtime_kind == "source",
    )
    expected_backend = {"runtime_kind": "source", **expected["source_backend"]}
    backend_matches = dict(backend) == expected_backend
    package_matches = provenance.get("package") is None
    _record(checks, "m005_provenance_backend_matches_current", backend_matches)
    _record(checks, "m005_provenance_package_contract_matches_runtime_kind", package_matches)
    _record(
        checks,
        "m005_provenance_console_identity_source_is_operator_challenge",
        console.get("identity_source") == "operator_attested_owner_challenge"
        and console.get("console_identity_query_available") is False,
    )
    _record(
        checks,
        "m005_provenance_console_model_matches",
        console.get("model") == EXPECTED_M005_CONSOLE_MODEL,
    )
    _record(checks, "m005_provenance_console_firmware_matches", console.get("firmware") == EXPECTED_M005_CONSOLE_FIRMWARE)

    challenge = (
        _get_mapping(m005_console_identity_challenge, "content")
        if isinstance(m005_console_identity_challenge, Mapping)
        else {}
    )
    challenge_identity = (
        _get_mapping(m005_console_identity_challenge, "identity")
        if isinstance(m005_console_identity_challenge, Mapping)
        else {}
    )
    declared_challenge = _get_mapping(provenance, "console_identity_challenge")
    preflight_gate = _get_mapping(m005_summary, "preflight_decision_gate")
    _record(checks, "m005_provenance_console_challenge_loaded", bool(challenge))
    _record(
        checks,
        "m005_provenance_console_challenge_identity_matches",
        bool(challenge_identity) and dict(declared_challenge) == dict(challenge_identity),
    )
    _record(
        checks,
        "m005_provenance_console_challenge_contract_matches",
        challenge.get("schema") == EXPECTED_M005_CONSOLE_IDENTITY_SCHEMA
        and challenge.get("operator_confirmed") is True
        and challenge.get("console_model") == EXPECTED_M005_CONSOLE_MODEL
        and challenge.get("firmware") == EXPECTED_M005_CONSOLE_FIRMWARE
        and isinstance(challenge.get("serial"), str)
        and bool(str(challenge.get("serial")).strip())
        and challenge.get("accepted_readonly_source_id")
        == preflight_gate.get("expected_accepted_readonly_source_id"),
    )
    _record(
        checks,
        "m005_provenance_console_matches_owner_challenge",
        bool(challenge)
        and console.get("serial") == challenge.get("serial")
        and console.get("firmware") == challenge.get("firmware")
        and console.get("model") == challenge.get("console_model"),
    )
    events_hash = events.get("sha256")
    _record(
        checks,
        "m005_provenance_events_path_matches_loaded",
        isinstance(m005_events_path, str)
        and bool(m005_events_path)
        and events.get("path") == m005_events_path,
    )
    _record(
        checks,
        "m005_provenance_events_hash_present",
        events.get("exists") is True
        and isinstance(events_hash, str)
        and len(events_hash) == 64
        and all(character in "0123456789abcdef" for character in events_hash),
    )
    _record(
        checks,
        "m005_provenance_events_hash_matches_loaded",
        isinstance(m005_events_sha256, str) and events_hash == m005_events_sha256,
    )

    event_audit = m005_event_audit if isinstance(m005_event_audit, Mapping) else {}
    first_event = _get_mapping(event_audit, "first_event")
    first_payload = _get_mapping(first_event, "payload")
    prestart_provenance = dict(provenance)
    prestart_provenance["events"] = None
    prestart_sha256 = _canonical_json_sha256(prestart_provenance)
    _record(
        checks,
        "m005_provenance_first_event_is_prestart_capture",
        first_event.get("seq") == 1
        and first_event.get("event_type") == "provenance_capture"
        and first_payload.get("captured_before_backend_contact") is True,
    )
    _record(
        checks,
        "m005_provenance_first_event_matches_summary",
        dict(_get_mapping(first_payload, "provenance")) == prestart_provenance
        and first_payload.get("provenance_sha256") == prestart_sha256
        and m005_summary.get("provenance_prestart_sha256") == prestart_sha256
        and first_payload.get("url") == EXPECTED_M005_CANONICAL_URL,
    )

    runtime_identity = _get_mapping(m005_summary, "runtime_identity")
    event_runtime_identity = _get_mapping(event_audit, "runtime_identity")
    runtime_backend = _get_mapping(runtime_identity, "backend")
    runtime_websocket = _get_mapping(runtime_identity, "websocket")
    runtime_process = _get_mapping(runtime_identity, "process")
    runtime_launch_binding = _get_mapping(runtime_identity, "launch_binding")
    runtime_source = {
        "source": dict(_get_mapping(runtime_identity, "source")),
        "code": dict(_get_mapping(runtime_identity, "code")),
    }
    expected_runtime_source = dict(_get_mapping(provenance, "runtime_source"))
    expected_runtime_backend = {
        key: backend.get(key) for key in ("path", "exists", "size", "sha256")
    }
    nonce = str(runtime_identity.get("instance_nonce") or "")
    challenge_nonce = str(m005_summary.get("runtime_identity_challenge_nonce") or "")
    launch_nonce = str(provenance.get("runtime_launch_nonce") or "")
    try:
        captured_at = datetime.fromisoformat(
            str(provenance.get("captured_at_utc") or "").replace("Z", "+00:00")
        )
        started_at = datetime.fromisoformat(
            str(runtime_identity.get("started_at_utc") or "").replace("Z", "+00:00")
        )
        start_delay = (started_at - captured_at).total_seconds()
    except (TypeError, ValueError):
        start_delay = -1
    _record(
        checks,
        "m005_provenance_backend_runtime_identity_matches",
        runtime_identity.get("type") == "runtime_identity"
        and runtime_identity.get("schema") == EXPECTED_M005_RUNTIME_IDENTITY_SCHEMA
        and runtime_identity.get("runtime_kind") == "source"
        and valid_lower_hex(nonce, 32)
        and runtime_process.get("instance_nonce") == nonce
        and runtime_process.get("started_at_utc")
        == runtime_identity.get("started_at_utc")
        and runtime_process.get("pid") == m005_summary.get("launched_backend_pid")
        and 0 <= start_delay <= MAX_M005_BOUND_BACKEND_START_DELAY_SECONDS
        and runtime_identity.get("read_only_identity") is True
        and runtime_identity.get("live_authority_granted") is False
        and runtime_identity.get("request_challenge_nonce") == challenge_nonce
        and valid_lower_hex(challenge_nonce, 32)
        and runtime_launch_binding.get("mode") == "proof_runner"
        and runtime_launch_binding.get("launch_nonce") == launch_nonce
        and valid_lower_hex(launch_nonce, 32)
        and runtime_launch_binding.get("prestart_provenance_schema")
        == EXPECTED_M005_PROVENANCE_SCHEMA
        and runtime_launch_binding.get("prestart_provenance_sha256") == prestart_sha256
        and runtime_launch_binding.get("prestart_provenance_size")
        == len(canonical_json_bytes(prestart_provenance))
        and runtime_launch_binding.get("bound_before_process_start") is True
        and runtime_launch_binding.get("proof_eligible") is True
        and runtime_source == expected_runtime_source
        and runtime_websocket.get("loopback_only") is True
        and runtime_websocket.get("host") == "127.0.0.1"
        and runtime_websocket.get("port") == 8765
        and dict(runtime_backend) == expected_runtime_backend
        and runtime_identity.get("manifest_sha256")
        == runtime_identity_manifest_sha256(dict(runtime_identity)),
    )
    _record(
        checks,
        "m005_provenance_backend_runtime_identity_event_matches_summary",
        bool(event_runtime_identity) and dict(event_runtime_identity) == dict(runtime_identity),
    )


def verify_packet(
    packet: Mapping[str, Any],
    *,
    m005_summary: Mapping[str, Any] | None = None,
    m005_event_request_types: list[str] | None = None,
    m005_event_initial_connection: Mapping[str, Any] | None = None,
    m005_event_stale_readback: Mapping[str, Any] | None = None,
    m005_event_reconnect_readback: Mapping[str, Any] | None = None,
    m005_event_stale_readback_required: bool = False,
    proof_summaries_by_artifact: Mapping[str, Mapping[str, Any]] | None = None,
    packet_summary_path: str | None = None,
    m005_summary_path: str | None = None,
    m005_events_path: str | None = None,
    m005_events_sha256: str | None = None,
    m005_event_audit: Mapping[str, Any] | None = None,
    m005_console_identity_challenge: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    _record_packet_checks(checks, packet)
    _record_proof_artifact_checks(
        checks,
        packet,
        proof_summaries_by_artifact=proof_summaries_by_artifact,
    )
    _record_m005_checks(
        checks,
        m005_summary=m005_summary,
        m005_event_request_types=m005_event_request_types,
        m005_event_initial_connection=m005_event_initial_connection,
        m005_event_stale_readback=m005_event_stale_readback,
        m005_event_reconnect_readback=m005_event_reconnect_readback,
        m005_event_stale_readback_required=m005_event_stale_readback_required,
        m005_summary_path=m005_summary_path,
        m005_events_path=m005_events_path,
        m005_events_sha256=m005_events_sha256,
        m005_event_audit=m005_event_audit,
        m005_console_identity_challenge=m005_console_identity_challenge,
    )
    m005_preflight_gate = (
        _get_mapping(m005_summary, "preflight_decision_gate")
        if isinstance(m005_summary, Mapping)
        else {}
    )
    _record(
        checks,
        "packet_accepted_readonly_source_matches_m005_gate",
        bool(m005_preflight_gate)
        and isinstance(packet.get("accepted_readonly_source_id"), str)
        and packet.get("accepted_readonly_source_id")
        == m005_preflight_gate.get("expected_accepted_readonly_source_id"),
    )

    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    provenance_checks = sorted(
        name for name in checks if name.startswith("m005_provenance_")
    )
    provenance_ready = bool(provenance_checks) and all(checks[name] for name in provenance_checks)
    owner_review_shape_ready = all(
        passed for name, passed in checks.items() if name not in provenance_checks
    )
    behavioral_complete = bool(
        m005_summary
        and m005_summary.get("status") == "passed"
        and m005_summary.get("m005_complete") is True
        and m005_summary.get("missing_m005_evidence") == []
    )
    ready = not failed_checks
    if ready:
        reason = "m006_owner_review_inputs_verified"
    elif m005_summary is None:
        reason = "m005_summary_required_for_owner_review"
    elif behavioral_complete and owner_review_shape_ready and not provenance_ready:
        reason = "m005_current_release_provenance_missing_or_mismatch"
    else:
        reason = "m006_owner_review_inputs_failed"

    return {
        "schema": "automixer.m006_owner_readiness_packet_verifier.v1",
        "generated_at_utc": _iso_now(),
        "status": "ready_for_owner_review" if ready else "blocked",
        "reason": reason,
        "packet_summary": packet_summary_path,
        "m005_summary": m005_summary_path,
        "m005_events": m005_events_path,
        "proof_artifact_summaries": sorted((proof_summaries_by_artifact or {}).keys()),
        "owner_review_input_ready": ready,
        "m005_behavioral_proof_complete": behavioral_complete,
        "m005_owner_review_shape_ready": owner_review_shape_ready,
        "m005_provenance_ready": provenance_ready,
        "m005_current_release_accepted": ready,
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
    m005_console_identity_challenge = load_m005_console_identity_challenge(m005)
    m005_event_audit, m005_events_path = load_m005_event_audit(
        m005_summary_path=m005_path,
        m005_summary=m005,
    )
    m005_event_request_types = None
    m005_event_initial_connection = None
    m005_event_stale_readback = None
    m005_event_reconnect_readback = None
    m005_events_sha256 = None
    if isinstance(m005_event_audit, Mapping):
        request_types = m005_event_audit.get("request_types")
        if isinstance(request_types, list):
            m005_event_request_types = [str(item) for item in request_types]
        initial_connection = m005_event_audit.get("initial_connection")
        if isinstance(initial_connection, Mapping):
            m005_event_initial_connection = initial_connection
        stale_readback = m005_event_audit.get("stale_readback")
        if isinstance(stale_readback, Mapping):
            m005_event_stale_readback = stale_readback
        reconnect_readback = m005_event_audit.get("reconnect_readback")
        if isinstance(reconnect_readback, Mapping):
            m005_event_reconnect_readback = reconnect_readback
        event_hash = m005_event_audit.get("sha256")
        if isinstance(event_hash, str):
            m005_events_sha256 = event_hash
    proof_summaries = load_packet_proof_summaries(packet)
    return verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=m005_event_request_types,
        m005_event_initial_connection=m005_event_initial_connection,
        m005_event_stale_readback=m005_event_stale_readback,
        m005_event_reconnect_readback=m005_event_reconnect_readback,
        m005_event_stale_readback_required=True,
        proof_summaries_by_artifact=proof_summaries,
        packet_summary_path=str(packet_summary),
        m005_summary_path=_display_path(m005_path) if m005_path is not None else None,
        m005_events_path=m005_events_path,
        m005_events_sha256=m005_events_sha256,
        m005_event_audit=m005_event_audit,
        m005_console_identity_challenge=m005_console_identity_challenge,
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
