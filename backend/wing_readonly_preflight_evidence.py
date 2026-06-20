"""Read-only evidence intake for WING preflight runner artifacts.

This module reviews already-captured ``wing_readonly_preflight.v1`` summaries.
It never opens WebSocket/OSC connections, never queries hardware, and never
turns a preflight artifact into supervised apply permission.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple
try:
    from wing_readonly_preflight_contract import (
        is_sha256_hexdigest,
        WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES,
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND,
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT,
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
        WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
        WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME,
        WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_SOURCES,
        WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_STATUSES,
        WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS,
        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS,
        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS,
        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS,
        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_NAME,
        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_STATUS,
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY,
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS,
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE,
        WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA,
        WING_READONLY_PREFLIGHT_PROOF_RUNNER,
        WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE,
        WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY,
        WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON,
        WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS,
        WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES,
        WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER,
        WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS,
        WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.wing_readonly_preflight_contract import (
        is_sha256_hexdigest,
        WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES,
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND,
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT,
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
        WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
        WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME,
        WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_SOURCES,
        WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_STATUSES,
        WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS,
        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS,
        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS,
        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS,
        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_NAME,
        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_STATUS,
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY,
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS,
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE,
        WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA,
        WING_READONLY_PREFLIGHT_PROOF_RUNNER,
        WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE,
        WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY,
        WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON,
        WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS,
        WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES,
        WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER,
        WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS,
        WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME,
    )


WING_READONLY_PREFLIGHT_EVIDENCE_METHOD_ID = "wing_readonly_preflight_evidence"
WING_READONLY_PREFLIGHT_SCHEMA_ID = WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA
WING_READONLY_PREFLIGHT_RUNNER = WING_READONLY_PREFLIGHT_PROOF_RUNNER
DEFAULT_WING_READONLY_PREFLIGHT_ARTIFACT_ROOTS = (
    WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
)
RESERVED_NON_WING_EVIDENCE_ROOTS = (
    "artifacts/no_wing_release_check",
    "ops/autopilot/evidence",
)
EVIDENCE_STATUSES = {
    "operator_preflight_required",
    "readonly_preflight_candidate_available",
    "readonly_preflight_artifact_replacement_required",
}
ALLOWED_SENT_REQUEST_TYPES = set(WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES)
ALLOWED_VALUE_TYPES = set(WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES)
_EXPECTED_RESPONSE_TYPES_CONTRACT = (
    WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE
)
EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE = {
    request_type: set(response_types)
    for request_type, response_types in _EXPECTED_RESPONSE_TYPES_CONTRACT.items()
}
EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE_SUMMARY = {
    request_type: list(response_types)
    for request_type, response_types in _EXPECTED_RESPONSE_TYPES_CONTRACT.items()
}
EXPECTED_MODE_EVENT_TYPES = EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE[
    "get_operator_mode"
]
EXPECTED_STATE_EVENT_TYPES = EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE["get_state"]
EXPECTED_READBACK_EVENT_TYPES = EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE[
    "query_mixer_parameter"
]
LIVE_WRITE_REQUEST_TYPES = {
    "set_fader",
    "set_gain",
    "set_trim",
    "set_input_trim",
    "set_channel_trim",
    "set_preamp_trim",
    "set_preamp_gain",
    "set_eq",
    "set_compressor",
    "route_output",
    "load_snap",
    "save_snap",
    "reset",
    "mute",
    "unmute",
    "set_insert",
    "set_fx_model",
    "set_fx_on",
    "set_fx_mix",
    "set_fx_parameter",
}
PROPOSAL_REQUEST_TYPES = {"create_operator_proposal", "apply_operator_proposal"}
FORBIDDEN_REQUEST_TYPES = {
    *LIVE_WRITE_REQUEST_TYPES,
    "set_operator_mode",
    *PROPOSAL_REQUEST_TYPES,
    "rollback_last_manual_write",
    "emergency_stop",
    "connect_mixer",
    "connect_wing",
    "disconnect_wing",
    "disconnect_mixer",
}
PHYSICAL_READBACK_STATUSES = set(WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_STATUSES)
PHYSICAL_READBACK_SOURCES = set(WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_SOURCES)
EXPECTED_RUN_SUMMARY_STATUS = WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS
EXPECTED_RUN_SUMMARY_REASON = WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON
MUTATION_EVIDENCE_FIELDS = {
    "apply_result",
    "rollback_result",
    "manual_write_events",
    "requested_change",
    "requested_changes",
    "operator_proposal",
    "write_policy",
}
EVENT_PAYLOAD_FORBIDDEN_MUTATION_FIELDS = MUTATION_EVIDENCE_FIELDS | {
    "write_events",
    "live_write_events",
    "console_write_events",
    "proposal_apply_result",
}
EVENT_PAYLOAD_UNSAFE_TRUE_FIELDS = {
    "write_request_sent",
    "proposal_request_sent",
    "live_write",
    "mutates_console_state",
    "execution_allowed_by_event",
    "execution_allowed_by_evidence",
    "execution_allowed_by_review",
    "readiness_matrix_update_allowed_by_event",
    "readiness_matrix_update_allowed_by_evidence",
    "readiness_closure_allowed_by_event",
    "readiness_closure_allowed_by_evidence",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
}
REQUEST_ID_POLICY = WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY
EXPECTED_SELECTED_MVP_BLOCKER = WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER
EXPECTED_SELECTED_MVP_BLOCKER_STATUS = (
    WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS
)
EXPECTED_PLAN_ALLOWED_REQUEST_TYPES = tuple(
    WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES
)
EXPECTED_PLAN_FORBIDDEN_REQUEST_TYPES = tuple(sorted(FORBIDDEN_REQUEST_TYPES))
EXPECTED_PLAN_FAIL_CLOSED_CONDITIONS = tuple(
    WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS
)
EXPECTED_BACKEND_PORT_PRESTART_CHECK_COMMAND = tuple(
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
)
EXPECTED_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT = (
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT
)
EXPECTED_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS = tuple(
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
)
EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS = tuple(
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
)
EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS = tuple(
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS
)
EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS = tuple(
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS
)
WS_TOKEN = "web" + "sock" + "et"
ARTIFACT_MANIFEST_VERIFIER_WEBSOCKET_PROOF_FIELDS = (
    f"artifact_manifest_verifier_must_pass_before_{WS_TOKEN}",
    f"artifact_manifest_verifier_passed_before_{WS_TOKEN}",
    f"artifact_manifest_verifier_{WS_TOKEN}_opened_after_pass",
)
ARTIFACT_MANIFEST_VERIFIER_MUST_PASS_BEFORE_WS_FIELD = (
    ARTIFACT_MANIFEST_VERIFIER_WEBSOCKET_PROOF_FIELDS[0]
)
ARTIFACT_MANIFEST_VERIFIER_PASSED_BEFORE_WS_FIELD = (
    ARTIFACT_MANIFEST_VERIFIER_WEBSOCKET_PROOF_FIELDS[1]
)
ARTIFACT_MANIFEST_VERIFIER_WS_OPENED_AFTER_PASS_FIELD = (
    ARTIFACT_MANIFEST_VERIFIER_WEBSOCKET_PROOF_FIELDS[2]
)
EXPECTED_PLAN_SUMMARY_FIELDS = (
    "proof_artifact_schema",
    "proof_runner",
    "artifact_root",
    "channels",
    "operator_channel_confirmation_required",
    "operator_confirmed_channels",
    "operator_session_confirmation_required",
    "operator_confirmed_session_ready",
    "value_types",
    "require_physical_readback",
    "read_only",
    "live_write",
    "expected_plan_required_for_real_run",
    "physical_preflight_requires_existing_backend_connection",
    "backend_port_prestart_summary_required",
    "backend_port_prestart_summary_path",
    "backend_port_prestart_summary_sha256",
    "backend_port_prestart_summary_type",
    "backend_port_prestart_summary_status",
    "backend_port_prestart_summary_connection_policy",
    "backend_port_prestart_summary_read_only",
    "backend_port_prestart_summary_live_write",
    "backend_port_prestart_summary_contacts_backend",
    "backend_port_prestart_summary_contacts_wing",
    "backend_port_prestart_summary_mutates_console_state",
    "backend_port_prestart_summary_queries_console",
    "backend_port_prestart_summary_executes_tests",
    "backend_port_prestart_check_status",
    *EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS,
    "request_id_policy",
    "expected_response_types_by_request_type",
)
PRESTART_SUMMARY_TYPE = WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE
PRESTART_SUMMARY_CONNECTION_POLICY = (
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY
)
PRESTART_SUMMARY_STATUS = WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS
PRESTART_SUMMARY_BACKEND_CHECK_NAME = (
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_NAME
)
PRESTART_SUMMARY_BACKEND_CHECK_STATUS = (
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_STATUS
)


@dataclass(frozen=True)
class WingReadonlyPreflightArtifactRecord:
    source_id: str
    status: str
    accepted_as_readonly_preflight_candidate: bool
    run_id: str | None
    captured_at: str | None
    artifact_root: str | None
    summary_path: str | None
    summary_sha256: str | None
    events_path: str | None
    events_sha256: str | None
    event_log_run_id_matches_summary: bool | None
    expected_artifacts: Tuple[str, ...]
    expected_summary_artifact: str | None
    expected_events_artifact: str | None
    summary_path_matches_expected_artifact: bool | None
    events_path_matches_expected_artifact: bool | None
    connection_mode: str | None
    contacts_backend: bool | None
    queries_console: bool | None
    readback_count: int
    physical_readback_count: int
    sent_request_types: Tuple[str, ...]
    write_request_sent: bool | None
    proposal_request_sent: bool | None
    forbidden_request_types_sent: Tuple[str, ...]
    request_sequence: Tuple[Dict[str, Any], ...]
    plan_request_sequence: Tuple[Dict[str, Any], ...]
    request_sequence_matches_plan: bool | None
    channels: Tuple[int, ...]
    operator_confirmed_channels: Tuple[int, ...]
    operator_session_confirmation_required: bool
    operator_confirmed_session_ready: bool
    backend_port_prestart_check_required: bool
    backend_port_prestart_check_must_pass_before_starting_backend: bool
    backend_port_prestart_check_command: Tuple[str, ...]
    backend_port_prestart_check_command_text: str
    backend_port_prestart_check_failure_detail_fields: Tuple[str, ...]
    backend_port_prestart_summary_required: bool
    backend_port_prestart_summary_path: str | None
    backend_port_prestart_summary_sha256: str | None
    backend_port_prestart_summary_type: str | None
    backend_port_prestart_summary_status: str | None
    backend_port_prestart_summary_connection_policy: str | None
    backend_port_prestart_summary_read_only: bool | None
    backend_port_prestart_summary_live_write: bool | None
    backend_port_prestart_summary_contacts_backend: bool | None
    backend_port_prestart_summary_contacts_wing: bool | None
    backend_port_prestart_summary_mutates_console_state: bool | None
    backend_port_prestart_summary_queries_console: bool | None
    backend_port_prestart_summary_executes_tests: bool | None
    backend_port_prestart_check_status: str | None
    artifact_manifest_verifier_must_pass_before_ws: bool | None
    artifact_manifest_verifier_passed_before_ws: bool | None
    artifact_manifest_verifier_ws_opened_after_pass: bool | None
    safe_to_stop_listener_by_gate: bool | None
    live_authority_granted: bool | None
    backend_port_prestart_gate_execution_allowed: bool | None
    backend_port_prestart_gate_readiness_closure_allowed: bool | None
    backend_port_prestart_gate_live_authority_granted: bool | None
    plan_safe_to_stop_listener_by_gate: bool | None
    plan_live_authority_granted: bool | None
    plan_backend_port_prestart_gate_execution_allowed: bool | None
    plan_backend_port_prestart_gate_readiness_closure_allowed: bool | None
    plan_backend_port_prestart_gate_live_authority_granted: bool | None
    safe_to_stop_listener_by_gate_matches_plan: bool | None
    live_authority_granted_matches_plan: bool | None
    backend_port_prestart_gate_execution_allowed_matches_plan: bool | None
    backend_port_prestart_gate_readiness_closure_allowed_matches_plan: bool | None
    backend_port_prestart_gate_live_authority_granted_matches_plan: bool | None
    operator_next_step: str | None
    value_types: Tuple[str, ...]
    violations: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in {
            "valid_readonly_preflight_artifact",
            "invalid_readonly_preflight_artifact",
            "artifact_load_error",
        }:
            raise ValueError(f"Unknown WING preflight artifact record status: {self.status}")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["sent_request_types"] = list(self.sent_request_types)
        payload["forbidden_request_types_sent"] = list(self.forbidden_request_types_sent)
        payload["request_sequence"] = [dict(row) for row in self.request_sequence]
        payload["plan_request_sequence"] = [
            dict(row) for row in self.plan_request_sequence
        ]
        payload["request_sequence_matches_plan"] = (
            self.request_sequence_matches_plan
        )
        payload["channels"] = list(self.channels)
        payload["operator_confirmed_channels"] = list(self.operator_confirmed_channels)
        payload["expected_artifacts"] = list(self.expected_artifacts)
        payload["event_log_run_id_matches_summary"] = (
            self.event_log_run_id_matches_summary
        )
        payload["contacts_backend"] = self.contacts_backend
        payload["queries_console"] = self.queries_console
        payload["backend_port_prestart_check_command"] = list(
            self.backend_port_prestart_check_command
        )
        payload["backend_port_prestart_check_command_text"] = (
            self.backend_port_prestart_check_command_text
        )
        payload["backend_port_prestart_check_failure_detail_fields"] = list(
            self.backend_port_prestart_check_failure_detail_fields
        )
        payload["backend_port_prestart_summary_required"] = (
            self.backend_port_prestart_summary_required
        )
        payload["backend_port_prestart_summary_path"] = (
            self.backend_port_prestart_summary_path
        )
        payload["backend_port_prestart_summary_sha256"] = (
            self.backend_port_prestart_summary_sha256
        )
        payload["backend_port_prestart_summary_type"] = (
            self.backend_port_prestart_summary_type
        )
        payload["backend_port_prestart_summary_status"] = (
            self.backend_port_prestart_summary_status
        )
        payload["backend_port_prestart_summary_connection_policy"] = (
            self.backend_port_prestart_summary_connection_policy
        )
        payload["backend_port_prestart_summary_read_only"] = (
            self.backend_port_prestart_summary_read_only
        )
        payload["backend_port_prestart_summary_live_write"] = (
            self.backend_port_prestart_summary_live_write
        )
        payload["backend_port_prestart_summary_contacts_backend"] = (
            self.backend_port_prestart_summary_contacts_backend
        )
        payload["backend_port_prestart_summary_contacts_wing"] = (
            self.backend_port_prestart_summary_contacts_wing
        )
        payload["backend_port_prestart_summary_mutates_console_state"] = (
            self.backend_port_prestart_summary_mutates_console_state
        )
        payload["backend_port_prestart_summary_queries_console"] = (
            self.backend_port_prestart_summary_queries_console
        )
        payload["backend_port_prestart_summary_executes_tests"] = (
            self.backend_port_prestart_summary_executes_tests
        )
        payload["backend_port_prestart_check_status"] = (
            self.backend_port_prestart_check_status
        )
        payload[ARTIFACT_MANIFEST_VERIFIER_MUST_PASS_BEFORE_WS_FIELD] = (
            self.artifact_manifest_verifier_must_pass_before_ws
        )
        payload[ARTIFACT_MANIFEST_VERIFIER_PASSED_BEFORE_WS_FIELD] = (
            self.artifact_manifest_verifier_passed_before_ws
        )
        payload[ARTIFACT_MANIFEST_VERIFIER_WS_OPENED_AFTER_PASS_FIELD] = (
            self.artifact_manifest_verifier_ws_opened_after_pass
        )
        payload.pop("artifact_manifest_verifier_must_pass_before_ws", None)
        payload.pop("artifact_manifest_verifier_passed_before_ws", None)
        payload.pop("artifact_manifest_verifier_ws_opened_after_pass", None)
        payload["safe_to_stop_listener_by_gate"] = self.safe_to_stop_listener_by_gate
        payload["live_authority_granted"] = self.live_authority_granted
        payload["backend_port_prestart_gate_execution_allowed"] = (
            self.backend_port_prestart_gate_execution_allowed
        )
        payload["backend_port_prestart_gate_readiness_closure_allowed"] = (
            self.backend_port_prestart_gate_readiness_closure_allowed
        )
        payload["backend_port_prestart_gate_live_authority_granted"] = (
            self.backend_port_prestart_gate_live_authority_granted
        )
        for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS:
            payload[f"plan_{field}"] = getattr(self, f"plan_{field}")
            payload[f"{field}_matches_plan"] = getattr(self, f"{field}_matches_plan")
        payload["operator_next_step"] = self.operator_next_step
        payload["value_types"] = list(self.value_types)
        payload["violations"] = list(self.violations)
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    return value if isinstance(value, int) and value >= 0 else 0


def _optional_bool(mapping: Mapping[str, Any], key: str) -> bool | None:
    value = mapping.get(key)
    return value if isinstance(value, bool) else None


def _string_tuple(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)


def _int_tuple(value: Any) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    items: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            continue
        items.append(item)
    return tuple(items)


def _request_sequence_tuple(value: Any) -> Tuple[Dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict(row) for row in value if isinstance(row, Mapping))


def _artifact_roots(artifact_roots: Iterable[str | Path] | None) -> Tuple[Path, ...]:
    roots = (
        tuple(artifact_roots)
        if artifact_roots is not None
        else DEFAULT_WING_READONLY_PREFLIGHT_ARTIFACT_ROOTS
    )
    return tuple(Path(root) for root in roots)


def _discover_artifact_paths(artifact_roots: Iterable[str | Path] | None) -> Tuple[Path, ...]:
    paths: list[Path] = []
    for root in _artifact_roots(artifact_roots):
        if root.is_file() and root.name == WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME:
            paths.append(root)
        if not root.is_dir():
            continue
        paths.extend(
            path
            for path in root.rglob(WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME)
            if path.is_file()
        )
    return tuple(sorted(dict.fromkeys(paths), key=lambda path: str(path)))


def _load_json_artifact(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("artifact JSON root must be an object")
    return payload


def _load_jsonl_events(path: Path) -> Tuple[Mapping[str, Any], ...]:
    events: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"event log line {line_number} must be an object")
            events.append(payload)
    return tuple(events)


def _event_payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(event.get("payload"))


def _iter_nested_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _iter_nested_mappings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_nested_mappings(child)


def _request_id_for(channel: int, value_type: str) -> str:
    return f"wing-readonly-preflight:ch{channel}:{value_type}"


def _address_for(channel: int, value_type: str) -> str:
    if value_type == "fader":
        return f"/ch/{channel}/fdr"
    if value_type == "gain":
        return f"/ch/{channel}/in/set/trim"
    if value_type == "pan":
        return f"/ch/{channel}/pan"
    return ""


def _summary_event_path(payload: Mapping[str, Any], summary_path: Path) -> Path | None:
    raw_path = payload.get("events_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    event_path = Path(raw_path)
    if event_path.is_absolute():
        return event_path
    cwd_candidate = Path.cwd() / event_path
    if cwd_candidate.exists():
        return cwd_candidate
    return summary_path.parent / event_path


def _event_log_sha256(payload: Mapping[str, Any], summary_path: Path) -> str | None:
    event_path = _summary_event_path(payload, summary_path)
    if event_path is None or not event_path.exists():
        return None
    return _sha256_file(event_path)


def _summary_expected_plan_path(
    payload: Mapping[str, Any],
    summary_path: Path,
) -> Path | None:
    raw_path = payload.get("expected_plan_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    plan_path = Path(raw_path)
    if plan_path.is_absolute():
        return plan_path
    cwd_candidate = Path.cwd() / plan_path
    if cwd_candidate.exists():
        return cwd_candidate
    return summary_path.parent / plan_path


def _summary_prestart_summary_path(
    payload: Mapping[str, Any],
    summary_path: Path,
) -> Path | None:
    raw_path = payload.get("backend_port_prestart_summary_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    prestart_path = Path(raw_path)
    if prestart_path.is_absolute():
        return prestart_path
    cwd_candidate = Path.cwd() / prestart_path
    if cwd_candidate.exists():
        return cwd_candidate
    return summary_path.parent / prestart_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_artifact_globs(artifact_root: str) -> Tuple[str, str]:
    return (
        f"{artifact_root}/<run_id>/{WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME}",
        f"{artifact_root}/<run_id>/{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME}",
    )


def _artifact_path_string(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    return Path(value).as_posix()


def _path_is_under_reserved_non_wing_evidence_root(value: Any) -> bool:
    if not isinstance(value, (str, Path)):
        return False
    normalized = Path(value).as_posix().strip()
    if not normalized:
        return False
    normalized = normalized.removeprefix("./").rstrip("/")
    for root in RESERVED_NON_WING_EVIDENCE_ROOTS:
        if (
            normalized == root
            or normalized.startswith(f"{root}/")
            or normalized.endswith(f"/{root}")
            or f"/{root}/" in normalized
        ):
            return True
    return False


def _path_matches_wing_readonly_capture_root(value: Any) -> bool:
    if not isinstance(value, (str, Path)):
        return False
    normalized = Path(value).as_posix().strip().removeprefix("./").rstrip("/")
    expected_root = WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT
    return normalized == expected_root or normalized.endswith(f"/{expected_root}")


def _run_id_is_single_safe_path_segment(value: str) -> bool:
    return value not in {".", ".."} and "/" not in value and "\\" not in value


def _validate_candidate_artifact_intake_paths(
    payload: Mapping[str, Any],
    *,
    summary_path: Path | None,
    violations: list[str],
) -> None:
    artifact_root = payload.get("artifact_root")
    run_id = payload.get("run_id")
    declared_paths = {
        "summary_path": WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME,
        "events_path": WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME,
    }
    if _path_is_under_reserved_non_wing_evidence_root(artifact_root):
        violations.append("artifact_root_reserved_for_non_wing_evidence")
    elif isinstance(artifact_root, str) and artifact_root:
        if not _path_matches_wing_readonly_capture_root(artifact_root):
            violations.append("artifact_root_not_wing_readonly_capture_root")
    for field in ("summary_path", "events_path"):
        if _path_is_under_reserved_non_wing_evidence_root(payload.get(field)):
            violations.append(f"{field}_reserved_for_non_wing_evidence")
    if summary_path is not None and _path_is_under_reserved_non_wing_evidence_root(
        summary_path
    ):
        violations.append("source_path_reserved_for_non_wing_evidence")
    if summary_path is not None:
        declared_summary_path = payload.get("summary_path")
        if isinstance(declared_summary_path, str) and declared_summary_path:
            if Path(declared_summary_path).as_posix() != summary_path.as_posix():
                violations.append("source_path_declared_summary_path_mismatch")
    if not isinstance(run_id, str) or not run_id:
        return
    if not _run_id_is_single_safe_path_segment(run_id):
        violations.append("run_id_not_single_safe_path_segment")
        return
    if not isinstance(artifact_root, str) or not artifact_root:
        return
    for field, artifact_name in declared_paths.items():
        actual_path = payload.get(field)
        if not isinstance(actual_path, str) or not actual_path:
            violations.append(f"{field}_missing")
            continue
        expected_path = (Path(artifact_root) / run_id / artifact_name).as_posix()
        if _artifact_path_string(actual_path) != expected_path:
            violations.append(f"{field}_declared_artifact_path_mismatch")


def _expected_artifact_for_run(expected_artifact: str, run_id: str) -> str:
    return _artifact_path_string(expected_artifact.replace("<run_id>", run_id))


def _expected_plan_artifacts(plan: Mapping[str, Any] | None) -> Tuple[str, ...]:
    if not isinstance(plan, Mapping):
        return ()
    return _string_tuple(plan.get("expected_artifacts"))


def _artifact_path_matches_expected(
    actual_path: Any,
    expected_artifact: str | None,
    run_id: str,
) -> bool | None:
    if not expected_artifact:
        return None
    expected_path = _expected_artifact_for_run(expected_artifact, run_id)
    if not expected_path:
        return None
    return _artifact_path_string(actual_path) == expected_path


def _expected_summary_artifact(
    expected_artifacts: Tuple[str, ...],
    run_id: str,
) -> str | None:
    if not expected_artifacts:
        return None
    return _expected_artifact_for_run(expected_artifacts[0], run_id)


def _expected_events_artifact(
    expected_artifacts: Tuple[str, ...],
    run_id: str,
) -> str | None:
    if len(expected_artifacts) < 2:
        return None
    return _expected_artifact_for_run(expected_artifacts[1], run_id)


def _validate_artifact_paths_match_expected_plan(
    summary: Mapping[str, Any],
    plan: Mapping[str, Any] | None,
    violations: list[str],
) -> None:
    run_id = str(summary.get("run_id") or "")
    expected_artifacts = _expected_plan_artifacts(plan)
    if not expected_artifacts:
        return
    summary_matches = _artifact_path_matches_expected(
        summary.get("summary_path"),
        expected_artifacts[0] if expected_artifacts else None,
        run_id,
    )
    events_matches = _artifact_path_matches_expected(
        summary.get("events_path"),
        expected_artifacts[1] if len(expected_artifacts) >= 2 else None,
        run_id,
    )
    if summary_matches is not True:
        violations.append("summary_path_expected_artifact_mismatch")
    if events_matches is not True:
        violations.append("events_path_expected_artifact_mismatch")


def _expected_plan_request_sequence(payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
    readbacks_raw = payload.get("readbacks")
    readbacks = (
        tuple(row for row in readbacks_raw if isinstance(row, Mapping))
        if isinstance(readbacks_raw, list)
        else ()
    )
    return [
        {"type": "get_operator_mode"},
        {"type": "get_state"},
        *[
            {
                "type": "query_mixer_parameter",
                "request_id": str(row.get("request_id") or ""),
                "channel": _safe_int(row.get("channel")),
                "value_type": str(row.get("value_type") or ""),
                "address": str(row.get("address") or ""),
            }
            for row in readbacks
        ],
    ]


def _validate_expected_plan_payload(
    summary: Mapping[str, Any],
    plan: Mapping[str, Any],
    violations: list[str],
) -> None:
    if plan.get("type") != "wing_readonly_preflight_offline_plan":
        violations.append("expected_plan_type_mismatch")
    if plan.get("status") != "ready_for_operator_review":
        violations.append("expected_plan_status_mismatch")
    if plan.get("selected_mvp_blocker") != EXPECTED_SELECTED_MVP_BLOCKER:
        violations.append("expected_plan_selected_mvp_blocker_mismatch")
    if plan.get("selected_mvp_blocker_status") != EXPECTED_SELECTED_MVP_BLOCKER_STATUS:
        violations.append("expected_plan_selected_mvp_blocker_status_mismatch")
    if plan.get("operator_required") is not True:
        violations.append("expected_plan_operator_required_not_true")
    if plan.get("plan_output") != summary.get("expected_plan_path"):
        violations.append("expected_plan_plan_output_mismatch")
    artifact_root = plan.get("artifact_root")
    if not isinstance(artifact_root, str) or not artifact_root:
        violations.append("expected_plan_artifact_root_missing")
        artifact_root = WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT
    if plan.get("expected_artifacts") != list(_expected_artifact_globs(artifact_root)):
        violations.append("expected_plan_expected_artifacts_mismatch")
    for field in EXPECTED_PLAN_SUMMARY_FIELDS:
        if plan.get(field) != summary.get(field):
            violations.append(f"expected_plan_{field}_mismatch")
    if plan.get(ARTIFACT_MANIFEST_VERIFIER_MUST_PASS_BEFORE_WS_FIELD) is not True:
        violations.append(
            f"expected_plan_{ARTIFACT_MANIFEST_VERIFIER_MUST_PASS_BEFORE_WS_FIELD}_not_true"
        )
    if plan.get(ARTIFACT_MANIFEST_VERIFIER_PASSED_BEFORE_WS_FIELD) != summary.get(
        ARTIFACT_MANIFEST_VERIFIER_PASSED_BEFORE_WS_FIELD
    ):
        violations.append(
            f"expected_plan_{ARTIFACT_MANIFEST_VERIFIER_PASSED_BEFORE_WS_FIELD}_mismatch"
        )
    if plan.get(ARTIFACT_MANIFEST_VERIFIER_WS_OPENED_AFTER_PASS_FIELD) is not False:
        violations.append(
            f"expected_plan_{ARTIFACT_MANIFEST_VERIFIER_WS_OPENED_AFTER_PASS_FIELD}_not_false"
        )
    if plan.get("request_sequence") != _expected_plan_request_sequence(summary):
        violations.append("expected_plan_request_sequence_mismatch")
    for field in (
        "contacts_backend",
        "queries_console",
        "execution_allowed_by_plan",
        "live_authority_granted",
    ):
        if plan.get(field) is not False:
            violations.append(f"expected_plan_{field}_not_false")
    if plan.get("manual_review_required_before_readiness_update") is not True:
        violations.append("expected_plan_manual_review_required_not_true")
    if plan.get("backend_port_prestart_check_required") is not True:
        violations.append("expected_plan_backend_port_prestart_check_required_not_true")
    if plan.get("backend_port_prestart_check_must_pass_before_starting_backend") is not True:
        violations.append(
            "expected_plan_backend_port_prestart_check_must_pass_before_starting_backend_not_true"
        )
    if plan.get("backend_port_prestart_check_command") != list(
        EXPECTED_BACKEND_PORT_PRESTART_CHECK_COMMAND
    ):
        violations.append("expected_plan_backend_port_prestart_check_command_mismatch")
    if (
        plan.get("backend_port_prestart_check_command_text")
        != EXPECTED_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT
    ):
        violations.append(
            "expected_plan_backend_port_prestart_check_command_text_mismatch"
        )
    if plan.get("backend_port_prestart_check_failure_detail_fields") != list(
        EXPECTED_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    ):
        violations.append(
            "expected_plan_backend_port_prestart_check_failure_detail_fields_mismatch"
        )
    if plan.get("allowed_request_types") != list(EXPECTED_PLAN_ALLOWED_REQUEST_TYPES):
        violations.append("expected_plan_allowed_request_types_mismatch")
    if (
        plan.get("expected_response_types_by_request_type")
        != EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE_SUMMARY
    ):
        violations.append("expected_plan_response_type_mapping_mismatch")

    forbidden_request_types = plan.get("forbidden_request_types")
    if forbidden_request_types != list(EXPECTED_PLAN_FORBIDDEN_REQUEST_TYPES):
        violations.append("expected_plan_forbidden_request_types_mismatch")
    if not isinstance(forbidden_request_types, list):
        forbidden_request_types = []
    for request_type in EXPECTED_PLAN_FORBIDDEN_REQUEST_TYPES:
        if request_type not in forbidden_request_types:
            violations.append(f"expected_plan_forbidden_request_type_missing:{request_type}")

    fail_closed_conditions = plan.get("fail_closed_conditions")
    if fail_closed_conditions != list(EXPECTED_PLAN_FAIL_CLOSED_CONDITIONS):
        violations.append("expected_plan_fail_closed_conditions_mismatch")
    if not isinstance(fail_closed_conditions, list):
        fail_closed_conditions = []
    for condition in EXPECTED_PLAN_FAIL_CLOSED_CONDITIONS:
        if condition not in fail_closed_conditions:
            violations.append(f"expected_plan_fail_closed_condition_missing:{condition}")


def _validate_expected_plan_file(
    payload: Mapping[str, Any],
    summary_path: Path,
    violations: list[str],
) -> Mapping[str, Any] | None:
    plan_path = _summary_expected_plan_path(payload, summary_path)
    if plan_path is None:
        violations.append("expected_plan_file_path_missing")
        return None
    if not plan_path.exists():
        violations.append("expected_plan_file_missing")
        return None
    expected_sha = str(payload.get("expected_plan_sha256") or "")
    try:
        actual_sha = _sha256_file(plan_path)
    except Exception as exc:
        violations.append(f"expected_plan_file_hash_error:{type(exc).__name__}")
        return None
    if actual_sha != expected_sha:
        violations.append("expected_plan_sha256_mismatch")
    try:
        plan_payload = _load_json_artifact(plan_path)
    except Exception as exc:
        violations.append(f"expected_plan_file_load_error:{type(exc).__name__}")
        return None
    _validate_expected_plan_payload(payload, plan_payload, violations)
    return plan_payload


def _backend_port_check_from_prestart_summary(
    payload: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    results = payload.get("results")
    if not isinstance(results, list):
        return None
    for item in results:
        if (
            isinstance(item, Mapping)
            and item.get("name") == PRESTART_SUMMARY_BACKEND_CHECK_NAME
        ):
            return item
    return None


def _validate_prestart_summary_payload(
    summary: Mapping[str, Any],
    prestart: Mapping[str, Any],
    violations: list[str],
) -> None:
    if prestart.get("type") != PRESTART_SUMMARY_TYPE:
        violations.append("prestart_summary_type_mismatch")
    if prestart.get("status") != PRESTART_SUMMARY_STATUS:
        violations.append("prestart_summary_status_not_passed")
    if prestart.get("connection_policy") != PRESTART_SUMMARY_CONNECTION_POLICY:
        violations.append("prestart_summary_connection_policy_mismatch")
    if prestart.get("read_only") is not True:
        violations.append("prestart_summary_read_only_not_true")
    if prestart.get("live_write") is not False:
        violations.append("prestart_summary_live_write_not_false")
    if prestart.get("contacts_backend") is not False:
        violations.append("prestart_summary_contacts_backend_not_false")
    if prestart.get("contacts_wing") is not False:
        violations.append("prestart_summary_contacts_wing_not_false")
    if prestart.get("mutates_console_state") is not False:
        violations.append("prestart_summary_mutates_console_state_not_false")
    if prestart.get("queries_console") is not False:
        violations.append("prestart_summary_queries_console_not_false")
    if prestart.get("executes_tests") is not False:
        violations.append("prestart_summary_executes_tests_not_false")

    backend_check = _backend_port_check_from_prestart_summary(prestart)
    if backend_check is None:
        violations.append("prestart_summary_backend_port_check_missing")
        return
    if backend_check.get("status") != PRESTART_SUMMARY_BACKEND_CHECK_STATUS:
        violations.append("prestart_summary_backend_port_check_not_passed")
    check_detail = backend_check.get("detail")
    if isinstance(check_detail, Mapping):
        failure_detail_fields = check_detail.get(
            "backend_port_prestart_check_failure_detail_fields"
        )
        if failure_detail_fields != list(
            EXPECTED_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
        ):
            violations.append(
                "prestart_summary_backend_port_check_failure_detail_fields_mismatch"
            )
        if summary.get("backend_port_prestart_check_failure_detail_fields") != (
            failure_detail_fields
        ):
            violations.append(
                "prestart_summary_recorded_backend_port_check_failure_detail_fields_mismatch"
            )
        for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS:
            if check_detail.get(field) is not False:
                violations.append(f"prestart_summary_{field}_not_false")
            if summary.get(field) != check_detail.get(field):
                violations.append(f"prestart_summary_recorded_{field}_mismatch")
            plan_field = f"plan_{field}"
            if check_detail.get(plan_field) is not False:
                violations.append(f"prestart_summary_{plan_field}_not_false")
            if summary.get(plan_field) != check_detail.get(plan_field):
                violations.append(f"prestart_summary_recorded_{plan_field}_mismatch")
            match_field = f"{field}_matches_plan"
            if check_detail.get(match_field) is not True:
                violations.append(f"prestart_summary_{match_field}_not_true")
            if summary.get(match_field) != check_detail.get(match_field):
                violations.append(f"prestart_summary_recorded_{match_field}_mismatch")
        if check_detail.get("live_authority_granted") is True:
            violations.append("prestart_summary_live_authority_granted")
        if check_detail.get("safe_to_stop_listener_by_gate") is True:
            violations.append("prestart_summary_safe_to_stop_listener_true")
    else:
        violations.append("prestart_summary_backend_port_check_detail_missing")

    if summary.get("backend_port_prestart_summary_type") != prestart.get("type"):
        violations.append("prestart_summary_recorded_type_mismatch")
    if summary.get("backend_port_prestart_summary_status") != prestart.get("status"):
        violations.append("prestart_summary_recorded_status_mismatch")
    if (
        summary.get("backend_port_prestart_summary_connection_policy")
        != prestart.get("connection_policy")
    ):
        violations.append("prestart_summary_recorded_connection_policy_mismatch")
    if summary.get("backend_port_prestart_summary_read_only") != prestart.get(
        "read_only"
    ):
        violations.append("prestart_summary_recorded_read_only_mismatch")
    if summary.get("backend_port_prestart_summary_live_write") != prestart.get(
        "live_write"
    ):
        violations.append("prestart_summary_recorded_live_write_mismatch")
    if summary.get("backend_port_prestart_summary_contacts_backend") != prestart.get(
        "contacts_backend"
    ):
        violations.append("prestart_summary_recorded_contacts_backend_mismatch")
    if summary.get("backend_port_prestart_summary_contacts_wing") != prestart.get(
        "contacts_wing"
    ):
        violations.append("prestart_summary_recorded_contacts_wing_mismatch")
    if summary.get(
        "backend_port_prestart_summary_mutates_console_state"
    ) != prestart.get("mutates_console_state"):
        violations.append(
            "prestart_summary_recorded_mutates_console_state_mismatch"
        )
    if summary.get("backend_port_prestart_summary_queries_console") != prestart.get(
        "queries_console"
    ):
        violations.append("prestart_summary_recorded_queries_console_mismatch")
    if summary.get("backend_port_prestart_summary_executes_tests") != prestart.get(
        "executes_tests"
    ):
        violations.append("prestart_summary_recorded_executes_tests_mismatch")
    if summary.get("backend_port_prestart_check_status") != backend_check.get("status"):
        violations.append("prestart_summary_recorded_backend_port_status_mismatch")


def _validate_prestart_summary_file(
    payload: Mapping[str, Any],
    summary_path: Path,
    violations: list[str],
) -> None:
    prestart_path = _summary_prestart_summary_path(payload, summary_path)
    if prestart_path is None:
        violations.append("prestart_summary_file_path_missing")
        return
    if not prestart_path.exists():
        violations.append("prestart_summary_file_missing")
        return
    expected_sha = str(payload.get("backend_port_prestart_summary_sha256") or "")
    if not expected_sha:
        violations.append("prestart_summary_sha256_missing")
        return
    try:
        actual_sha = _sha256_file(prestart_path)
    except Exception as exc:
        violations.append(f"prestart_summary_file_hash_error:{type(exc).__name__}")
        return
    if actual_sha != expected_sha:
        violations.append("prestart_summary_sha256_mismatch")
    try:
        prestart_payload = _load_json_artifact(prestart_path)
    except Exception as exc:
        violations.append(f"prestart_summary_file_load_error:{type(exc).__name__}")
        return
    _validate_prestart_summary_payload(payload, prestart_payload, violations)


def _validate_event_log(
    payload: Mapping[str, Any],
    summary_path: Path,
    violations: list[str],
) -> tuple[bool | None, Tuple[str, ...]]:
    event_path = _summary_event_path(payload, summary_path)
    if event_path is None:
        violations.append("event_log_path_missing")
        return None, ()
    if not event_path.exists():
        violations.append("event_log_missing")
        return None, ()
    try:
        events = _load_jsonl_events(event_path)
    except Exception as exc:
        violations.append(f"event_log_load_error:{type(exc).__name__}")
        return None, ()
    if not events:
        violations.append("event_log_empty")
        return None, ()
    summary_run_id = str(payload.get("run_id") or "")
    event_log_run_id_matches_summary = True
    for index, event in enumerate(events):
        if event.get("run_id") != summary_run_id:
            violations.append(f"event_log_run_id_mismatch:{index}")
            event_log_run_id_matches_summary = False
    event_sequences = tuple(_safe_int(event.get("seq")) for event in events)
    if event_sequences != tuple(range(1, len(events) + 1)):
        violations.append("event_log_seq_sequence_invalid")
    captured_timestamp = _parse_utc_timestamp(str(payload.get("captured_at") or ""))
    parsed_event_timestamps: list[datetime] = []
    for index, event in enumerate(events):
        raw_timestamp = event.get("timestamp")
        if not isinstance(raw_timestamp, str) or not raw_timestamp:
            violations.append(f"event_log_timestamp_missing:{index}")
            continue
        parsed_timestamp = _parse_utc_timestamp(raw_timestamp)
        if parsed_timestamp is None:
            violations.append(f"event_log_timestamp_invalid:{index}")
            continue
        parsed_event_timestamps.append(parsed_timestamp)
        if captured_timestamp is not None and parsed_timestamp > captured_timestamp:
            violations.append(f"event_log_timestamp_after_captured_at:{index}")
    if any(
        previous > current
        for previous, current in zip(parsed_event_timestamps, parsed_event_timestamps[1:])
    ):
        violations.append("event_log_timestamp_sequence_invalid")
    for event_index, event in enumerate(events):
        for mapping in _iter_nested_mappings(_event_payload(event)):
            for field in sorted(EVENT_PAYLOAD_FORBIDDEN_MUTATION_FIELDS):
                if field in mapping:
                    violations.append(
                        f"event_log_mutation_evidence_field_present:{event_index}:{field}"
                    )
            for field in sorted(EVENT_PAYLOAD_UNSAFE_TRUE_FIELDS):
                if mapping.get(field) is True:
                    violations.append(
                        f"event_log_unsafe_authority_field_true:{event_index}:{field}"
                    )
    event_types = tuple(str(event.get("event_type") or "") for event in events)
    if "send_error" in event_types:
        violations.append("event_log_send_error_present")
    if len(event_types) % 2 != 0 or any(
        event_types[index : index + 2] != ("send", "recv")
        for index in range(0, len(event_types), 2)
    ):
        violations.append("event_log_send_recv_sequence_invalid")

    send_payloads = tuple(
        _event_payload(event)
        for event in events
        if event.get("event_type") == "send"
    )
    event_request_types = tuple(
        str(row.get("type") or "unknown")
        for row in send_payloads
    )
    summary_request_types = _string_tuple(payload.get("sent_request_types"))
    if event_request_types != summary_request_types:
        violations.append("event_log_sent_request_types_mismatch")
    unexpected_event_request_types = tuple(
        sorted(set(event_request_types) - ALLOWED_SENT_REQUEST_TYPES)
    )
    forbidden_event_request_types = tuple(
        sorted(set(event_request_types).intersection(FORBIDDEN_REQUEST_TYPES))
    )
    if unexpected_event_request_types:
        violations.append("event_log_unexpected_sent_request_types")
    if forbidden_event_request_types:
        violations.append("event_log_forbidden_request_type_sent")
        violations.append("forbidden_request_types_sent_not_empty")
    if (
        len(event_request_types) < 3
        or event_request_types[0] != "get_operator_mode"
        or event_request_types[1] != "get_state"
        or any(request_type != "query_mixer_parameter" for request_type in event_request_types[2:])
    ):
        violations.append("event_log_request_sequence_invalid")

    recv_payloads = tuple(
        _event_payload(event)
        for event in events
        if event.get("event_type") == "recv"
    )
    recv_types = {str(row.get("type") or "") for row in recv_payloads}
    if "operator_mode_status" not in recv_types:
        violations.append("event_log_operator_mode_status_missing")
    else:
        operator_mode_payloads = tuple(
            row
            for row in recv_payloads
            if str(row.get("type") or "") == "operator_mode_status"
        )
        if any(
            str(row.get("connection_mode") or "") != "wing"
            for row in operator_mode_payloads
        ):
            violations.append("event_log_operator_mode_connection_mode_not_wing")
    if "state_update" not in recv_types:
        violations.append("event_log_state_update_missing")
    else:
        state_update_payloads = tuple(
            row
            for row in recv_payloads
            if str(row.get("type") or "") == "state_update"
        )
        if any(str(row.get("mode") or "") != "wing" for row in state_update_payloads):
            violations.append("event_log_state_update_mode_not_wing")

    event_readbacks = tuple(
        row
        for row in recv_payloads
        if str(row.get("type") or "") in EXPECTED_READBACK_EVENT_TYPES
    )
    readbacks_raw = payload.get("readbacks")
    summary_readbacks = (
        tuple(row for row in readbacks_raw if isinstance(row, Mapping))
        if isinstance(readbacks_raw, list)
        else ()
    )
    if event_readbacks != summary_readbacks:
        violations.append("event_log_readbacks_mismatch")
    query_payloads = tuple(
        row
        for row in send_payloads
        if row.get("type") == "query_mixer_parameter"
    )
    if len(query_payloads) != len(event_readbacks):
        violations.append("event_log_query_readback_count_mismatch")
    for index, (query, readback) in enumerate(zip(query_payloads, event_readbacks)):
        query_request_id = str(query.get("request_id") or "")
        readback_request_id = str(readback.get("request_id") or "")
        if not query_request_id:
            violations.append(f"event_log_query_request_id_missing:{index}")
        if not readback_request_id:
            violations.append(f"event_log_readback_request_id_missing:{index}")
        if query_request_id != readback_request_id:
            violations.append(f"event_log_query_readback_request_id_mismatch:{index}")
        query_identity = (
            query_request_id,
            _safe_int(query.get("channel")),
            str(query.get("value_type") or ""),
            str(query.get("address") or ""),
        )
        readback_identity = (
            readback_request_id,
            _safe_int(readback.get("channel")),
            str(readback.get("value_type") or ""),
            str(readback.get("address") or ""),
        )
        if query_identity != readback_identity:
            violations.append(f"event_log_query_readback_identity_mismatch:{index}")

    for index in range(0, len(events), 2):
        if index + 1 >= len(events):
            break
        request = _event_payload(events[index])
        response = _event_payload(events[index + 1])
        request_type = str(request.get("type") or "")
        response_type = str(response.get("type") or "")
        expected_response_types = EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE.get(
            request_type,
            set(),
        )
        if response_type not in expected_response_types:
            violations.append(f"event_log_response_type_mismatch:{index // 2}")
    return event_log_run_id_matches_summary, forbidden_event_request_types


def _readback_is_physical(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("status") in PHYSICAL_READBACK_STATUSES
        and str(payload.get("source") or "") in PHYSICAL_READBACK_SOURCES
    )


def _validate_readback_row(index: int, row: Mapping[str, Any], violations: list[str]) -> None:
    if not _readback_is_physical(row):
        violations.append(f"readback_{index}_not_physical")
    if not str(row.get("request_id") or ""):
        violations.append(f"readback_{index}_request_id_missing")
    value_type = str(row.get("value_type") or "")
    if value_type not in ALLOWED_VALUE_TYPES:
        violations.append(f"readback_{index}_unsupported_value_type")
    channel = _safe_int(row.get("channel"))
    if channel < 1 or channel > 48:
        violations.append(f"readback_{index}_invalid_channel")
    address = str(row.get("address") or "")
    if not address:
        violations.append(f"readback_{index}_address_missing")
    elif value_type in ALLOWED_VALUE_TYPES and 1 <= channel <= 48:
        if address != _address_for(channel, value_type):
            violations.append(f"readback_{index}_address_mismatch")
    value = row.get("value")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        violations.append(f"readback_{index}_value_missing_or_non_numeric")


def _target_pairs(channels: Tuple[int, ...], value_types: Tuple[str, ...]) -> Tuple[Tuple[int, str], ...]:
    return tuple((channel, value_type) for channel in channels for value_type in value_types)


def _readback_pairs(readbacks: Tuple[Mapping[str, Any], ...]) -> Tuple[Tuple[int, str], ...]:
    return tuple(
        (_safe_int(row.get("channel")), str(row.get("value_type") or ""))
        for row in readbacks
    )


def _target_request_ids(channels: Tuple[int, ...], value_types: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(_request_id_for(channel, value_type) for channel in channels for value_type in value_types)


def _readback_request_ids(readbacks: Tuple[Mapping[str, Any], ...]) -> Tuple[str, ...]:
    return tuple(str(row.get("request_id") or "") for row in readbacks)


def validate_wing_readonly_preflight_artifact(
    payload: Mapping[str, Any],
    *,
    summary_path: Path | None = None,
) -> Dict[str, Any]:
    """Validate one preflight summary without granting live authority."""

    violations: list[str] = []
    if payload.get("proof_artifact_schema") != WING_READONLY_PREFLIGHT_SCHEMA_ID:
        violations.append("schema_not_wing_readonly_preflight_v1")
    if payload.get("proof_runner") != WING_READONLY_PREFLIGHT_RUNNER:
        violations.append("proof_runner_not_wing_readonly_preflight_runner")
    artifact_root = payload.get("artifact_root")
    if not isinstance(artifact_root, str) or not artifact_root:
        violations.append("artifact_root_missing")
    _validate_candidate_artifact_intake_paths(
        payload,
        summary_path=summary_path,
        violations=violations,
    )
    if payload.get("status") != EXPECTED_RUN_SUMMARY_STATUS:
        violations.append("status_not_passed")
    if payload.get("reason") != EXPECTED_RUN_SUMMARY_REASON:
        violations.append("reason_not_passed")
    captured_at = payload.get("captured_at")
    if not isinstance(captured_at, str) or not captured_at:
        violations.append("captured_at_missing")
    elif not _is_utc_timestamp(captured_at):
        violations.append("captured_at_invalid")
    if payload.get("connection_mode") != "wing":
        violations.append("connection_mode_not_wing")
    if payload.get("request_id_policy") != REQUEST_ID_POLICY:
        violations.append("request_id_policy_missing_or_invalid")
    if (
        payload.get("expected_response_types_by_request_type")
        != EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE_SUMMARY
    ):
        violations.append("expected_response_type_mapping_missing_or_invalid")
    if not str(payload.get("expected_plan_path") or ""):
        violations.append("expected_plan_path_missing")
    expected_plan_sha256 = str(payload.get("expected_plan_sha256") or "")
    if not expected_plan_sha256:
        violations.append("expected_plan_sha256_missing")
    elif not is_sha256_hexdigest(expected_plan_sha256):
        violations.append("expected_plan_sha256_invalid")
    if payload.get("expected_plan_match") is not True:
        violations.append("expected_plan_match_not_true")
    if payload.get("expected_plan_violations") not in ([], ()):
        violations.append("expected_plan_violations_not_empty")
    if payload.get("expected_plan_required_for_real_run") is not True:
        violations.append("expected_plan_required_for_real_run_not_true")
    if (
        payload.get("physical_preflight_requires_existing_backend_connection")
        is not True
    ):
        violations.append("physical_preflight_requires_existing_backend_connection_not_true")
    if payload.get("backend_port_prestart_check_required") is not True:
        violations.append("backend_port_prestart_check_required_not_true")
    if payload.get("backend_port_prestart_check_must_pass_before_starting_backend") is not True:
        violations.append(
            "backend_port_prestart_check_must_pass_before_starting_backend_not_true"
        )
    if payload.get("backend_port_prestart_check_command") != list(
        EXPECTED_BACKEND_PORT_PRESTART_CHECK_COMMAND
    ):
        violations.append("backend_port_prestart_check_command_mismatch")
    if (
        payload.get("backend_port_prestart_check_command_text")
        != EXPECTED_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT
    ):
        violations.append("backend_port_prestart_check_command_text_mismatch")
    if payload.get("backend_port_prestart_check_failure_detail_fields") != list(
        EXPECTED_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    ):
        violations.append(
            "backend_port_prestart_check_failure_detail_fields_mismatch"
        )
    if payload.get("backend_port_prestart_summary_required") is not True:
        violations.append("backend_port_prestart_summary_required_not_true")
    if not str(payload.get("backend_port_prestart_summary_path") or ""):
        violations.append("backend_port_prestart_summary_path_missing")
    prestart_summary_sha256 = str(
        payload.get("backend_port_prestart_summary_sha256") or ""
    )
    if not prestart_summary_sha256:
        violations.append("backend_port_prestart_summary_sha256_missing")
    elif not is_sha256_hexdigest(prestart_summary_sha256):
        violations.append("backend_port_prestart_summary_sha256_invalid")
    if payload.get("backend_port_prestart_summary_type") != PRESTART_SUMMARY_TYPE:
        violations.append("backend_port_prestart_summary_type_mismatch")
    if payload.get("backend_port_prestart_summary_status") != PRESTART_SUMMARY_STATUS:
        violations.append("backend_port_prestart_summary_status_not_passed")
    if (
        payload.get("backend_port_prestart_summary_connection_policy")
        != PRESTART_SUMMARY_CONNECTION_POLICY
    ):
        violations.append("backend_port_prestart_summary_connection_policy_mismatch")
    if payload.get("backend_port_prestart_summary_read_only") is not True:
        violations.append("backend_port_prestart_summary_read_only_not_true")
    if payload.get("backend_port_prestart_summary_live_write") is not False:
        violations.append("backend_port_prestart_summary_live_write_not_false")
    if payload.get("backend_port_prestart_summary_contacts_backend") is not False:
        violations.append("backend_port_prestart_summary_contacts_backend_not_false")
    if payload.get("backend_port_prestart_summary_contacts_wing") is not False:
        violations.append("backend_port_prestart_summary_contacts_wing_not_false")
    if payload.get("backend_port_prestart_summary_mutates_console_state") is not False:
        violations.append(
            "backend_port_prestart_summary_mutates_console_state_not_false"
        )
    if payload.get("backend_port_prestart_summary_queries_console") is not False:
        violations.append("backend_port_prestart_summary_queries_console_not_false")
    if payload.get("backend_port_prestart_summary_executes_tests") is not False:
        violations.append("backend_port_prestart_summary_executes_tests_not_false")
    if (
        payload.get("backend_port_prestart_check_status")
        != PRESTART_SUMMARY_BACKEND_CHECK_STATUS
    ):
        violations.append("backend_port_prestart_check_status_not_passed")
    if payload.get(ARTIFACT_MANIFEST_VERIFIER_MUST_PASS_BEFORE_WS_FIELD) is not True:
        violations.append(
            f"{ARTIFACT_MANIFEST_VERIFIER_MUST_PASS_BEFORE_WS_FIELD}_not_true"
        )
    if payload.get(ARTIFACT_MANIFEST_VERIFIER_PASSED_BEFORE_WS_FIELD) is not True:
        violations.append(
            f"{ARTIFACT_MANIFEST_VERIFIER_PASSED_BEFORE_WS_FIELD}_not_true"
        )
    if payload.get(ARTIFACT_MANIFEST_VERIFIER_WS_OPENED_AFTER_PASS_FIELD) is not True:
        violations.append(
            f"{ARTIFACT_MANIFEST_VERIFIER_WS_OPENED_AFTER_PASS_FIELD}_not_true"
        )
    if payload.get("require_physical_readback") is not True:
        violations.append("require_physical_readback_not_true")
    channels = _int_tuple(payload.get("channels"))
    confirmed_channels = _int_tuple(payload.get("operator_confirmed_channels"))
    value_types = _string_tuple(payload.get("value_types"))
    if payload.get("operator_channel_confirmation_required") is not True:
        violations.append("operator_channel_confirmation_required_not_true")
    if not confirmed_channels:
        violations.append("operator_confirmed_channels_missing")
    if confirmed_channels != channels:
        violations.append("operator_confirmed_channels_mismatch")
    if payload.get("operator_session_confirmation_required") is not True:
        violations.append("operator_session_confirmation_required_not_true")
    if payload.get("operator_confirmed_session_ready") is not True:
        violations.append("operator_confirmed_session_ready_not_true")
    if payload.get("read_only") is not True:
        violations.append("read_only_not_true")
    if payload.get("live_write") is not False:
        violations.append("live_write_not_false")
    if payload.get("write_request_sent") is not False:
        violations.append("write_request_sent_not_false")
    if payload.get("proposal_request_sent") is not False:
        violations.append("proposal_request_sent_not_false")
    if payload.get("forbidden_request_types_sent") not in ([], ()):
        violations.append("forbidden_request_types_sent_not_empty")

    sent_request_types = _string_tuple(payload.get("sent_request_types"))
    if not sent_request_types:
        violations.append("sent_request_types_missing")
    unexpected_request_types = tuple(sorted(set(sent_request_types) - ALLOWED_SENT_REQUEST_TYPES))
    if unexpected_request_types:
        violations.append("unexpected_sent_request_types")
    for required_type in ("get_operator_mode", "get_state", "query_mixer_parameter"):
        if required_type not in sent_request_types:
            violations.append(f"sent_request_type_missing:{required_type}")

    readbacks_raw = payload.get("readbacks")
    readbacks = tuple(row for row in readbacks_raw if isinstance(row, Mapping)) if isinstance(readbacks_raw, list) else ()
    readback_count = _safe_int(payload.get("readback_count"))
    physical_readback_count = _safe_int(payload.get("physical_readback_count"))
    if readback_count <= 0:
        violations.append("readback_count_missing")
    if readback_count != len(readbacks):
        violations.append("readback_count_mismatch")
    physical_rows = tuple(row for row in readbacks if _readback_is_physical(row))
    if physical_readback_count != len(physical_rows):
        violations.append("physical_readback_count_mismatch")
    if readbacks and physical_readback_count != readback_count:
        violations.append("not_all_readbacks_physical")
    for index, row in enumerate(readbacks):
        _validate_readback_row(index, row, violations)
    if channels and value_types and readbacks:
        expected_pairs = _target_pairs(channels, value_types)
        actual_pairs = _readback_pairs(readbacks)
        if actual_pairs != expected_pairs:
            violations.append("readback_targets_do_not_match_declared_channels_and_value_types")
        if _readback_request_ids(readbacks) != _target_request_ids(channels, value_types):
            violations.append("readback_request_ids_do_not_match_declared_targets")

    for field in sorted(MUTATION_EVIDENCE_FIELDS):
        if field in payload:
            violations.append(f"mutation_evidence_field_present:{field}")

    expected_plan_payload: Mapping[str, Any] | None = None
    event_log_run_id_matches_summary: bool | None = None
    event_log_forbidden_request_types_sent: Tuple[str, ...] = ()
    if summary_path is not None:
        expected_plan_payload = _validate_expected_plan_file(
            payload,
            summary_path,
            violations,
        )
        _validate_artifact_paths_match_expected_plan(
            payload,
            expected_plan_payload,
            violations,
        )
        _validate_prestart_summary_file(payload, summary_path, violations)
        (
            event_log_run_id_matches_summary,
            event_log_forbidden_request_types_sent,
        ) = _validate_event_log(
            payload,
            summary_path,
            violations,
        )
    if any(
        violation.startswith("event_log_")
        for violation in violations
    ):
        violations.append("event_log_missing_or_invalid")

    accepted = not violations
    expected_artifacts = _expected_plan_artifacts(expected_plan_payload)
    run_id = str(payload.get("run_id") or "")
    request_sequence = _request_sequence_tuple(_expected_plan_request_sequence(payload))
    plan_request_sequence = _request_sequence_tuple(
        expected_plan_payload.get("request_sequence")
        if expected_plan_payload is not None
        else None
    )
    request_sequence_matches_plan = (
        request_sequence == plan_request_sequence
        if expected_plan_payload is not None
        else None
    )
    expected_summary_artifact = _expected_summary_artifact(expected_artifacts, run_id)
    expected_events_artifact = _expected_events_artifact(expected_artifacts, run_id)
    return {
        "status": (
            "valid_readonly_preflight_artifact"
            if accepted
            else "invalid_readonly_preflight_artifact"
        ),
        "accepted_as_readonly_preflight_candidate": accepted,
        "violations": sorted(dict.fromkeys(violations)),
        "evidence": {
            "run_id": payload.get("run_id"),
            "captured_at": captured_at if isinstance(captured_at, str) else None,
            "artifact_root": artifact_root if isinstance(artifact_root, str) else None,
            "summary_path": payload.get("summary_path"),
            "summary_sha256": (
                payload.get("summary_sha256")
                if isinstance(payload.get("summary_sha256"), str)
                else None
            ),
            "events_path": payload.get("events_path"),
            "events_sha256": (
                payload.get("events_sha256")
                if isinstance(payload.get("events_sha256"), str)
                else None
            ),
            "event_log_run_id_matches_summary": event_log_run_id_matches_summary,
            "expected_artifacts": list(expected_artifacts),
            "expected_summary_artifact": expected_summary_artifact,
            "expected_events_artifact": expected_events_artifact,
            "summary_path_matches_expected_artifact": (
                _artifact_path_matches_expected(
                    payload.get("summary_path"),
                    expected_artifacts[0] if expected_artifacts else None,
                    run_id,
                )
            ),
            "events_path_matches_expected_artifact": (
                _artifact_path_matches_expected(
                    payload.get("events_path"),
                    expected_artifacts[1] if len(expected_artifacts) >= 2 else None,
                    run_id,
                )
            ),
            "connection_mode": payload.get("connection_mode"),
            "contacts_backend": (
                payload.get("contacts_backend")
                if isinstance(payload.get("contacts_backend"), bool)
                else None
            ),
            "queries_console": (
                payload.get("queries_console")
                if isinstance(payload.get("queries_console"), bool)
                else None
            ),
            "readback_count": readback_count,
            "physical_readback_count": physical_readback_count,
            "sent_request_types": list(sent_request_types),
            "write_request_sent": payload.get("write_request_sent") is True,
            "proposal_request_sent": payload.get("proposal_request_sent") is True,
            "forbidden_request_types_sent": sorted(
                set(_string_tuple(payload.get("forbidden_request_types_sent")))
                | set(event_log_forbidden_request_types_sent)
            ),
            "request_sequence": [dict(row) for row in request_sequence],
            "plan_request_sequence": [dict(row) for row in plan_request_sequence],
            "request_sequence_matches_plan": request_sequence_matches_plan,
            "channels": list(channels),
            "operator_confirmed_channels": list(confirmed_channels),
            "operator_session_confirmation_required": (
                payload.get("operator_session_confirmation_required") is True
            ),
            "operator_confirmed_session_ready": (
                payload.get("operator_confirmed_session_ready") is True
            ),
            "backend_port_prestart_check_required": (
                payload.get("backend_port_prestart_check_required") is True
            ),
            "backend_port_prestart_check_must_pass_before_starting_backend": (
                payload.get("backend_port_prestart_check_must_pass_before_starting_backend")
                is True
            ),
            "backend_port_prestart_check_command": (
                list(payload.get("backend_port_prestart_check_command"))
                if isinstance(payload.get("backend_port_prestart_check_command"), list)
                else []
            ),
            "backend_port_prestart_check_command_text": str(
                payload.get("backend_port_prestart_check_command_text") or ""
            ),
            "backend_port_prestart_check_failure_detail_fields": (
                list(payload.get("backend_port_prestart_check_failure_detail_fields"))
                if isinstance(
                    payload.get("backend_port_prestart_check_failure_detail_fields"),
                    list,
                )
                else []
            ),
            "backend_port_prestart_summary_required": (
                payload.get("backend_port_prestart_summary_required") is True
            ),
            "backend_port_prestart_summary_path": (
                str(payload.get("backend_port_prestart_summary_path"))
                if payload.get("backend_port_prestart_summary_path")
                else None
            ),
            "backend_port_prestart_summary_sha256": (
                str(payload.get("backend_port_prestart_summary_sha256"))
                if payload.get("backend_port_prestart_summary_sha256")
                else None
            ),
            "backend_port_prestart_summary_type": (
                str(payload.get("backend_port_prestart_summary_type"))
                if payload.get("backend_port_prestart_summary_type")
                else None
            ),
            "backend_port_prestart_summary_status": (
                str(payload.get("backend_port_prestart_summary_status"))
                if payload.get("backend_port_prestart_summary_status")
                else None
            ),
            "backend_port_prestart_summary_connection_policy": (
                str(payload.get("backend_port_prestart_summary_connection_policy"))
                if payload.get("backend_port_prestart_summary_connection_policy")
                else None
            ),
            "backend_port_prestart_summary_read_only": (
                payload.get("backend_port_prestart_summary_read_only")
                if isinstance(
                    payload.get("backend_port_prestart_summary_read_only"), bool
                )
                else None
            ),
            "backend_port_prestart_summary_live_write": (
                payload.get("backend_port_prestart_summary_live_write")
                if isinstance(payload.get("backend_port_prestart_summary_live_write"), bool)
                else None
            ),
            "backend_port_prestart_summary_contacts_backend": (
                payload.get("backend_port_prestart_summary_contacts_backend")
                if isinstance(
                    payload.get("backend_port_prestart_summary_contacts_backend"), bool
                )
                else None
            ),
            "backend_port_prestart_summary_contacts_wing": (
                payload.get("backend_port_prestart_summary_contacts_wing")
                if isinstance(
                    payload.get("backend_port_prestart_summary_contacts_wing"), bool
                )
                else None
            ),
            "backend_port_prestart_summary_mutates_console_state": (
                payload.get("backend_port_prestart_summary_mutates_console_state")
                if isinstance(
                    payload.get("backend_port_prestart_summary_mutates_console_state"),
                    bool,
                )
                else None
            ),
            "backend_port_prestart_summary_queries_console": (
                payload.get("backend_port_prestart_summary_queries_console")
                if isinstance(
                    payload.get("backend_port_prestart_summary_queries_console"), bool
                )
                else None
            ),
            "backend_port_prestart_summary_executes_tests": (
                payload.get("backend_port_prestart_summary_executes_tests")
                if isinstance(
                    payload.get("backend_port_prestart_summary_executes_tests"), bool
                )
                else None
            ),
            "backend_port_prestart_check_status": (
                str(payload.get("backend_port_prestart_check_status"))
                if payload.get("backend_port_prestart_check_status")
                else None
            ),
            **{
                field: payload.get(field)
                if isinstance(payload.get(field), bool)
                else None
                for field in ARTIFACT_MANIFEST_VERIFIER_WEBSOCKET_PROOF_FIELDS
            },
            **{
                field: payload.get(field)
                if isinstance(payload.get(field), bool)
                else None
                for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
            },
            **{
                field: payload.get(field)
                if isinstance(payload.get(field), bool)
                else None
                for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS
            },
            **{
                field: payload.get(field)
                if isinstance(payload.get(field), bool)
                else None
                for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS
            },
            "operator_next_step": (
                str(payload.get("operator_next_step"))
                if payload.get("operator_next_step")
                else None
            ),
            "value_types": list(value_types),
        },
    }


def _record_from_validation(
    source_id: str,
    validation: Mapping[str, Any],
    *,
    summary_sha256: str | None = None,
    events_sha256: str | None = None,
) -> WingReadonlyPreflightArtifactRecord:
    evidence = _mapping(validation.get("evidence"))
    return WingReadonlyPreflightArtifactRecord(
        source_id=source_id,
        status=str(validation.get("status") or "invalid_readonly_preflight_artifact"),
        accepted_as_readonly_preflight_candidate=(
            validation.get("accepted_as_readonly_preflight_candidate") is True
        ),
        run_id=str(evidence.get("run_id")) if evidence.get("run_id") else None,
        captured_at=(
            str(evidence.get("captured_at")) if evidence.get("captured_at") else None
        ),
        artifact_root=(
            str(evidence.get("artifact_root")) if evidence.get("artifact_root") else None
        ),
        summary_path=str(evidence.get("summary_path")) if evidence.get("summary_path") else None,
        summary_sha256=summary_sha256
        or (str(evidence.get("summary_sha256")) if evidence.get("summary_sha256") else None),
        events_path=str(evidence.get("events_path")) if evidence.get("events_path") else None,
        events_sha256=events_sha256
        or (str(evidence.get("events_sha256")) if evidence.get("events_sha256") else None),
        event_log_run_id_matches_summary=(
            evidence.get("event_log_run_id_matches_summary")
            if isinstance(evidence.get("event_log_run_id_matches_summary"), bool)
            else None
        ),
        expected_artifacts=_string_tuple(evidence.get("expected_artifacts")),
        expected_summary_artifact=(
            str(evidence.get("expected_summary_artifact"))
            if evidence.get("expected_summary_artifact")
            else None
        ),
        expected_events_artifact=(
            str(evidence.get("expected_events_artifact"))
            if evidence.get("expected_events_artifact")
            else None
        ),
        summary_path_matches_expected_artifact=(
            evidence.get("summary_path_matches_expected_artifact")
            if isinstance(evidence.get("summary_path_matches_expected_artifact"), bool)
            else None
        ),
        events_path_matches_expected_artifact=(
            evidence.get("events_path_matches_expected_artifact")
            if isinstance(evidence.get("events_path_matches_expected_artifact"), bool)
            else None
        ),
        connection_mode=str(evidence.get("connection_mode")) if evidence.get("connection_mode") else None,
        contacts_backend=_optional_bool(evidence, "contacts_backend"),
        queries_console=_optional_bool(evidence, "queries_console"),
        readback_count=_safe_int(evidence.get("readback_count")),
        physical_readback_count=_safe_int(evidence.get("physical_readback_count")),
        sent_request_types=_string_tuple(evidence.get("sent_request_types")),
        write_request_sent=_optional_bool(evidence, "write_request_sent"),
        proposal_request_sent=_optional_bool(evidence, "proposal_request_sent"),
        forbidden_request_types_sent=_string_tuple(
            evidence.get("forbidden_request_types_sent")
        ),
        request_sequence=_request_sequence_tuple(evidence.get("request_sequence")),
        plan_request_sequence=_request_sequence_tuple(
            evidence.get("plan_request_sequence")
        ),
        request_sequence_matches_plan=(
            evidence.get("request_sequence_matches_plan")
            if isinstance(evidence.get("request_sequence_matches_plan"), bool)
            else None
        ),
        channels=_int_tuple(evidence.get("channels")),
        operator_confirmed_channels=_int_tuple(evidence.get("operator_confirmed_channels")),
        operator_session_confirmation_required=(
            evidence.get("operator_session_confirmation_required") is True
        ),
        operator_confirmed_session_ready=(
            evidence.get("operator_confirmed_session_ready") is True
        ),
        backend_port_prestart_check_required=(
            evidence.get("backend_port_prestart_check_required") is True
        ),
        backend_port_prestart_check_must_pass_before_starting_backend=(
            evidence.get("backend_port_prestart_check_must_pass_before_starting_backend")
            is True
        ),
        backend_port_prestart_check_command=_string_tuple(
            evidence.get("backend_port_prestart_check_command")
        ),
        backend_port_prestart_check_command_text=str(
            evidence.get("backend_port_prestart_check_command_text") or ""
        ),
        backend_port_prestart_check_failure_detail_fields=_string_tuple(
            evidence.get("backend_port_prestart_check_failure_detail_fields")
        ),
        backend_port_prestart_summary_required=(
            evidence.get("backend_port_prestart_summary_required") is True
        ),
        backend_port_prestart_summary_path=(
            str(evidence.get("backend_port_prestart_summary_path"))
            if evidence.get("backend_port_prestart_summary_path")
            else None
        ),
        backend_port_prestart_summary_sha256=(
            str(evidence.get("backend_port_prestart_summary_sha256"))
            if evidence.get("backend_port_prestart_summary_sha256")
            else None
        ),
        backend_port_prestart_summary_type=(
            str(evidence.get("backend_port_prestart_summary_type"))
            if evidence.get("backend_port_prestart_summary_type")
            else None
        ),
        backend_port_prestart_summary_status=(
            str(evidence.get("backend_port_prestart_summary_status"))
            if evidence.get("backend_port_prestart_summary_status")
            else None
        ),
        backend_port_prestart_summary_connection_policy=(
            str(evidence.get("backend_port_prestart_summary_connection_policy"))
            if evidence.get("backend_port_prestart_summary_connection_policy")
            else None
        ),
        backend_port_prestart_summary_read_only=(
            evidence.get("backend_port_prestart_summary_read_only")
            if isinstance(evidence.get("backend_port_prestart_summary_read_only"), bool)
            else None
        ),
        backend_port_prestart_summary_live_write=(
            evidence.get("backend_port_prestart_summary_live_write")
            if isinstance(evidence.get("backend_port_prestart_summary_live_write"), bool)
            else None
        ),
        backend_port_prestart_summary_contacts_backend=(
            evidence.get("backend_port_prestart_summary_contacts_backend")
            if isinstance(
                evidence.get("backend_port_prestart_summary_contacts_backend"), bool
            )
            else None
        ),
        backend_port_prestart_summary_contacts_wing=(
            evidence.get("backend_port_prestart_summary_contacts_wing")
            if isinstance(
                evidence.get("backend_port_prestart_summary_contacts_wing"), bool
            )
            else None
        ),
        backend_port_prestart_summary_mutates_console_state=(
            evidence.get("backend_port_prestart_summary_mutates_console_state")
            if isinstance(
                evidence.get("backend_port_prestart_summary_mutates_console_state"),
                bool,
            )
            else None
        ),
        backend_port_prestart_summary_queries_console=(
            evidence.get("backend_port_prestart_summary_queries_console")
            if isinstance(
                evidence.get("backend_port_prestart_summary_queries_console"), bool
            )
            else None
        ),
        backend_port_prestart_summary_executes_tests=(
            evidence.get("backend_port_prestart_summary_executes_tests")
            if isinstance(
                evidence.get("backend_port_prestart_summary_executes_tests"), bool
            )
            else None
        ),
        backend_port_prestart_check_status=(
            str(evidence.get("backend_port_prestart_check_status"))
            if evidence.get("backend_port_prestart_check_status")
            else None
        ),
        artifact_manifest_verifier_must_pass_before_ws=_optional_bool(
            evidence, ARTIFACT_MANIFEST_VERIFIER_MUST_PASS_BEFORE_WS_FIELD
        ),
        artifact_manifest_verifier_passed_before_ws=_optional_bool(
            evidence, ARTIFACT_MANIFEST_VERIFIER_PASSED_BEFORE_WS_FIELD
        ),
        artifact_manifest_verifier_ws_opened_after_pass=_optional_bool(
            evidence, ARTIFACT_MANIFEST_VERIFIER_WS_OPENED_AFTER_PASS_FIELD
        ),
        safe_to_stop_listener_by_gate=(
            evidence.get("safe_to_stop_listener_by_gate")
            if isinstance(evidence.get("safe_to_stop_listener_by_gate"), bool)
            else None
        ),
        live_authority_granted=(
            evidence.get("live_authority_granted")
            if isinstance(evidence.get("live_authority_granted"), bool)
            else None
        ),
        backend_port_prestart_gate_execution_allowed=(
            evidence.get("backend_port_prestart_gate_execution_allowed")
            if isinstance(
                evidence.get("backend_port_prestart_gate_execution_allowed"), bool
            )
            else None
        ),
        backend_port_prestart_gate_readiness_closure_allowed=(
            evidence.get("backend_port_prestart_gate_readiness_closure_allowed")
            if isinstance(
                evidence.get("backend_port_prestart_gate_readiness_closure_allowed"),
                bool,
            )
            else None
        ),
        backend_port_prestart_gate_live_authority_granted=(
            evidence.get("backend_port_prestart_gate_live_authority_granted")
            if isinstance(
                evidence.get("backend_port_prestart_gate_live_authority_granted"), bool
            )
            else None
        ),
        **{
            f"plan_{field}": _optional_bool(evidence, f"plan_{field}")
            for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
        },
        **{
            f"{field}_matches_plan": _optional_bool(
                evidence, f"{field}_matches_plan"
            )
            for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
        },
        operator_next_step=(
            str(evidence.get("operator_next_step"))
            if evidence.get("operator_next_step")
            else None
        ),
        value_types=_string_tuple(evidence.get("value_types")),
        violations=_string_tuple(validation.get("violations")),
    )


def _record_from_load_error(source_id: str, exc: Exception) -> WingReadonlyPreflightArtifactRecord:
    return WingReadonlyPreflightArtifactRecord(
        source_id=source_id,
        status="artifact_load_error",
        accepted_as_readonly_preflight_candidate=False,
        run_id=None,
        captured_at=None,
        artifact_root=None,
        summary_path=None,
        summary_sha256=None,
        events_path=None,
        events_sha256=None,
        event_log_run_id_matches_summary=None,
        expected_artifacts=(),
        expected_summary_artifact=None,
        expected_events_artifact=None,
        summary_path_matches_expected_artifact=None,
        events_path_matches_expected_artifact=None,
        connection_mode=None,
        contacts_backend=None,
        queries_console=None,
        readback_count=0,
        physical_readback_count=0,
        sent_request_types=(),
        write_request_sent=None,
        proposal_request_sent=None,
        forbidden_request_types_sent=(),
        request_sequence=(),
        plan_request_sequence=(),
        request_sequence_matches_plan=None,
        channels=(),
        operator_confirmed_channels=(),
        operator_session_confirmation_required=False,
        operator_confirmed_session_ready=False,
        backend_port_prestart_check_required=False,
        backend_port_prestart_check_must_pass_before_starting_backend=False,
        backend_port_prestart_check_command=(),
        backend_port_prestart_check_command_text="",
        backend_port_prestart_check_failure_detail_fields=(),
        backend_port_prestart_summary_required=False,
        backend_port_prestart_summary_path=None,
        backend_port_prestart_summary_sha256=None,
        backend_port_prestart_summary_type=None,
        backend_port_prestart_summary_status=None,
        backend_port_prestart_summary_connection_policy=None,
        backend_port_prestart_summary_read_only=None,
        backend_port_prestart_summary_live_write=None,
        backend_port_prestart_summary_contacts_backend=None,
        backend_port_prestart_summary_contacts_wing=None,
        backend_port_prestart_summary_mutates_console_state=None,
        backend_port_prestart_summary_queries_console=None,
        backend_port_prestart_summary_executes_tests=None,
        backend_port_prestart_check_status=None,
        artifact_manifest_verifier_must_pass_before_ws=None,
        artifact_manifest_verifier_passed_before_ws=None,
        artifact_manifest_verifier_ws_opened_after_pass=None,
        safe_to_stop_listener_by_gate=None,
        live_authority_granted=None,
        backend_port_prestart_gate_execution_allowed=None,
        backend_port_prestart_gate_readiness_closure_allowed=None,
        backend_port_prestart_gate_live_authority_granted=None,
        **{
            f"plan_{field}": None
            for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
        },
        **{
            f"{field}_matches_plan": None
            for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
        },
        operator_next_step=None,
        value_types=(),
        violations=(f"artifact_load_error:{type(exc).__name__}",),
    )


def _artifact_records(
    *,
    artifact_paths: Iterable[str | Path] | None,
    artifact_payloads: Iterable[Mapping[str, Any]] | None,
    artifact_roots: Iterable[str | Path] | None,
) -> Tuple[WingReadonlyPreflightArtifactRecord, ...]:
    records: list[WingReadonlyPreflightArtifactRecord] = []
    paths = (
        tuple(Path(path) for path in artifact_paths)
        if artifact_paths is not None
        else _discover_artifact_paths(artifact_roots)
    )
    for path in paths:
        try:
            payload = _load_json_artifact(path)
            validation = validate_wing_readonly_preflight_artifact(
                payload,
                summary_path=path,
            )
            records.append(
                _record_from_validation(
                    str(path),
                    validation,
                    summary_sha256=_sha256_file(path),
                    events_sha256=_event_log_sha256(payload, path),
                )
            )
        except Exception as exc:  # pragma: no cover - defensive artifact parser
            records.append(_record_from_load_error(str(path), exc))

    if artifact_payloads is not None:
        for index, payload in enumerate(artifact_payloads):
            validation = validate_wing_readonly_preflight_artifact(payload)
            records.append(_record_from_validation(f"inline_payload_{index}", validation))

    return tuple(records)


def _matrix_rows(candidate_available: bool, blocked_status: str) -> Tuple[Dict[str, Any], ...]:
    blocker_ids = (
        "wing_readonly_connection_proof",
        "physical_console_readback_proof",
        "operator_visible_evidence_report",
        "readiness_matrix_update_based_on_real_proof",
    )
    return tuple(
        {
            "id": blocker_id,
            "status": (
                "candidate_available"
                if candidate_available
                else blocked_status
            ),
            "closed": False,
            "manual_review_required": True,
            "automatic_live_authority_allowed": False,
        }
        for blocker_id in blocker_ids
    )


def build_wing_readonly_preflight_evidence(
    *,
    artifact_paths: Iterable[str | Path] | None = None,
    artifact_payloads: Iterable[Mapping[str, Any]] | None = None,
    artifact_roots: Iterable[str | Path] | None = None,
) -> Dict[str, Any]:
    """Build a fail-closed readiness matrix for read-only WING preflight artifacts."""

    records = _artifact_records(
        artifact_paths=artifact_paths,
        artifact_payloads=artifact_payloads,
        artifact_roots=artifact_roots,
    )
    valid_records = tuple(row for row in records if row.accepted_as_readonly_preflight_candidate)
    invalid_records = tuple(row for row in records if not row.accepted_as_readonly_preflight_candidate)
    candidate_available = bool(valid_records) and not invalid_records
    if invalid_records:
        status = "readonly_preflight_artifact_replacement_required"
        blocked_by = ("operator_must_replace_invalid_readonly_preflight_artifact",)
        matrix_blocked_status = "artifact_replacement_required"
        next_global_step = (
            "Operator replaces invalid read-only preflight artifacts before any "
            "readiness matrix review."
        )
    elif valid_records:
        status = "readonly_preflight_candidate_available"
        blocked_by = ("operator_must_review_readonly_preflight_candidate_before_readiness_update",)
        matrix_blocked_status = "operator_preflight_required"
        next_global_step = (
            "Operator reviews the valid read-only preflight candidate before "
            "readiness matrix rows may be updated."
        )
    else:
        status = "operator_preflight_required"
        blocked_by = ("operator_must_run_wing_readonly_preflight_runner",)
        matrix_blocked_status = "operator_preflight_required"
        next_global_step = "Operator runs the read-only preflight runner against the intended WING backend."

    all_violations = tuple(
        dict.fromkeys(violation for row in records for violation in row.violations)
    )
    latest_valid = valid_records[-1] if valid_records else None
    return {
        "type": "wing_readonly_preflight_evidence",
        "generated_at": _utc_now(),
        "method_id": WING_READONLY_PREFLIGHT_EVIDENCE_METHOD_ID,
        "schema_id": WING_READONLY_PREFLIGHT_SCHEMA_ID,
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_evidence": False,
            "readiness_closure_allowed_by_evidence": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "artifact_search_roots": [str(root) for root in _artifact_roots(artifact_roots)],
            "artifact_count": len(records),
            "valid_candidate_count": len(valid_records),
            "invalid_candidate_count": len(invalid_records),
            "accepted_as_readonly_preflight_candidate": candidate_available,
            "operator_review_required": True,
            "mvp_blocker_candidate_count": 4 if candidate_available else 0,
            "mvp_blockers_closed_by_evidence": [],
            "readiness_matrix_update_allowed": candidate_available,
            "readiness_closure_allowed_by_evidence": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "latest_valid_candidate_source_id": latest_valid.source_id if latest_valid else None,
            "latest_valid_candidate_run_id": latest_valid.run_id if latest_valid else None,
            "latest_valid_candidate_captured_at": (
                latest_valid.captured_at if latest_valid else None
            ),
            "latest_valid_candidate_request_sequence": (
                [dict(row) for row in latest_valid.request_sequence]
                if latest_valid
                else []
            ),
            "latest_valid_candidate_plan_request_sequence": (
                [dict(row) for row in latest_valid.plan_request_sequence]
                if latest_valid
                else []
            ),
            "latest_valid_candidate_request_sequence_matches_plan": (
                latest_valid.request_sequence_matches_plan if latest_valid else None
            ),
            "next_global_step": next_global_step,
            "blocked_by": list(blocked_by),
            "violations": list(all_violations),
        },
        "mvp_blocker_matrix": list(_matrix_rows(candidate_available, matrix_blocked_status)),
        "artifacts": [record.to_dict() for record in records],
    }
