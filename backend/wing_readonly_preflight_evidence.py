"""Read-only evidence intake for WING preflight runner artifacts.

This module reviews already-captured ``wing_readonly_preflight.v1`` summaries.
It never opens WebSocket/OSC connections, never queries hardware, and never
turns a preflight artifact into supervised apply permission.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple


WING_READONLY_PREFLIGHT_EVIDENCE_METHOD_ID = "wing_readonly_preflight_evidence"
WING_READONLY_PREFLIGHT_SCHEMA_ID = "wing_readonly_preflight.v1"
WING_READONLY_PREFLIGHT_RUNNER = "wing_readonly_preflight_runner"
DEFAULT_WING_READONLY_PREFLIGHT_ARTIFACT_ROOTS = ("artifacts/wing_readonly_preflight",)
EVIDENCE_STATUSES = {
    "operator_preflight_required",
    "readonly_preflight_candidate_available",
    "readonly_preflight_artifact_replacement_required",
}
ALLOWED_SENT_REQUEST_TYPES = {"get_operator_mode", "get_state", "query_mixer_parameter"}
EXPECTED_READBACK_EVENT_TYPES = {"mixer_parameter_value", "mixer_parameter_read_blocked"}
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
    "connect_wing",
    "disconnect_wing",
    "disconnect_mixer",
}
PHYSICAL_READBACK_STATUSES = {"verified_from_physical_readback"}
PHYSICAL_READBACK_SOURCES = {
    "physical_readback",
    "wing_physical_readback",
    "physical_wing",
    "wing",
}
MUTATION_EVIDENCE_FIELDS = {
    "apply_result",
    "rollback_result",
    "manual_write_events",
    "requested_change",
    "requested_changes",
    "operator_proposal",
    "write_policy",
}
REQUEST_ID_POLICY = "required_for_every_query_mixer_parameter_response"
EXPECTED_PLAN_SUMMARY_FIELDS = (
    "proof_artifact_schema",
    "proof_runner",
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
    "request_id_policy",
)


@dataclass(frozen=True)
class WingReadonlyPreflightArtifactRecord:
    source_id: str
    status: str
    accepted_as_readonly_preflight_candidate: bool
    run_id: str | None
    captured_at: str | None
    summary_path: str | None
    summary_sha256: str | None
    events_path: str | None
    events_sha256: str | None
    connection_mode: str | None
    readback_count: int
    physical_readback_count: int
    sent_request_types: Tuple[str, ...]
    channels: Tuple[int, ...]
    operator_confirmed_channels: Tuple[int, ...]
    operator_session_confirmation_required: bool
    operator_confirmed_session_ready: bool
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
        payload["channels"] = list(self.channels)
        payload["operator_confirmed_channels"] = list(self.operator_confirmed_channels)
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
    return value if isinstance(value, int) and value >= 0 else 0


def _string_tuple(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)


def _int_tuple(value: Any) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    items: list[int] = []
    for item in value:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        items.append(number)
    return tuple(items)


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
        if root.is_file() and root.name == "summary.json":
            paths.append(root)
        if not root.is_dir():
            continue
        paths.extend(path for path in root.rglob("summary.json") if path.is_file())
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


def _request_id_for(channel: int, value_type: str) -> str:
    return f"wing-readonly-preflight:ch{channel}:{value_type}"


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    for field in EXPECTED_PLAN_SUMMARY_FIELDS:
        if plan.get(field) != summary.get(field):
            violations.append(f"expected_plan_{field}_mismatch")
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


def _validate_expected_plan_file(
    payload: Mapping[str, Any],
    summary_path: Path,
    violations: list[str],
) -> None:
    plan_path = _summary_expected_plan_path(payload, summary_path)
    if plan_path is None:
        violations.append("expected_plan_file_path_missing")
        return
    if not plan_path.exists():
        violations.append("expected_plan_file_missing")
        return
    expected_sha = str(payload.get("expected_plan_sha256") or "")
    try:
        actual_sha = _sha256_file(plan_path)
    except Exception as exc:
        violations.append(f"expected_plan_file_hash_error:{type(exc).__name__}")
        return
    if actual_sha != expected_sha:
        violations.append("expected_plan_sha256_mismatch")
    try:
        plan_payload = _load_json_artifact(plan_path)
    except Exception as exc:
        violations.append(f"expected_plan_file_load_error:{type(exc).__name__}")
        return
    _validate_expected_plan_payload(payload, plan_payload, violations)


def _validate_event_log(
    payload: Mapping[str, Any],
    summary_path: Path,
    violations: list[str],
) -> None:
    event_path = _summary_event_path(payload, summary_path)
    if event_path is None:
        violations.append("event_log_path_missing")
        return
    if not event_path.exists():
        violations.append("event_log_missing")
        return
    try:
        events = _load_jsonl_events(event_path)
    except Exception as exc:
        violations.append(f"event_log_load_error:{type(exc).__name__}")
        return
    if not events:
        violations.append("event_log_empty")
        return
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
    event_types = tuple(str(event.get("event_type") or "") for event in events)
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
    if unexpected_event_request_types:
        violations.append("event_log_unexpected_sent_request_types")
    if set(event_request_types).intersection(FORBIDDEN_REQUEST_TYPES):
        violations.append("event_log_forbidden_request_type_sent")
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
    if "state_update" not in recv_types:
        violations.append("event_log_state_update_missing")

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

    expected_recv_by_request_type = {
        "get_operator_mode": {"operator_mode_status"},
        "get_state": {"state_update"},
        "query_mixer_parameter": EXPECTED_READBACK_EVENT_TYPES,
    }
    for index in range(0, len(events), 2):
        if index + 1 >= len(events):
            break
        request = _event_payload(events[index])
        response = _event_payload(events[index + 1])
        request_type = str(request.get("type") or "")
        response_type = str(response.get("type") or "")
        expected_response_types = expected_recv_by_request_type.get(request_type, set())
        if response_type not in expected_response_types:
            violations.append(f"event_log_response_type_mismatch:{index // 2}")


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
    if value_type not in {"fader", "gain"}:
        violations.append(f"readback_{index}_unsupported_value_type")
    channel = _safe_int(row.get("channel"))
    if channel < 1 or channel > 48:
        violations.append(f"readback_{index}_invalid_channel")


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
    if payload.get("status") != "passed":
        violations.append("status_not_passed")
    if payload.get("reason") != "wing_readonly_preflight_passed":
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
    if not str(payload.get("expected_plan_path") or ""):
        violations.append("expected_plan_path_missing")
    if not str(payload.get("expected_plan_sha256") or ""):
        violations.append("expected_plan_sha256_missing")
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

    if summary_path is not None:
        _validate_expected_plan_file(payload, summary_path, violations)
        _validate_event_log(payload, summary_path, violations)

    accepted = not violations
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
            "connection_mode": payload.get("connection_mode"),
            "readback_count": readback_count,
            "physical_readback_count": physical_readback_count,
            "sent_request_types": list(sent_request_types),
            "channels": list(channels),
            "operator_confirmed_channels": list(confirmed_channels),
            "operator_session_confirmation_required": (
                payload.get("operator_session_confirmation_required") is True
            ),
            "operator_confirmed_session_ready": (
                payload.get("operator_confirmed_session_ready") is True
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
        summary_path=str(evidence.get("summary_path")) if evidence.get("summary_path") else None,
        summary_sha256=summary_sha256
        or (str(evidence.get("summary_sha256")) if evidence.get("summary_sha256") else None),
        events_path=str(evidence.get("events_path")) if evidence.get("events_path") else None,
        events_sha256=events_sha256
        or (str(evidence.get("events_sha256")) if evidence.get("events_sha256") else None),
        connection_mode=str(evidence.get("connection_mode")) if evidence.get("connection_mode") else None,
        readback_count=_safe_int(evidence.get("readback_count")),
        physical_readback_count=_safe_int(evidence.get("physical_readback_count")),
        sent_request_types=_string_tuple(evidence.get("sent_request_types")),
        channels=_int_tuple(evidence.get("channels")),
        operator_confirmed_channels=_int_tuple(evidence.get("operator_confirmed_channels")),
        operator_session_confirmation_required=(
            evidence.get("operator_session_confirmation_required") is True
        ),
        operator_confirmed_session_ready=(
            evidence.get("operator_confirmed_session_ready") is True
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
        summary_path=None,
        summary_sha256=None,
        events_path=None,
        events_sha256=None,
        connection_mode=None,
        readback_count=0,
        physical_readback_count=0,
        sent_request_types=(),
        channels=(),
        operator_confirmed_channels=(),
        operator_session_confirmation_required=False,
        operator_confirmed_session_ready=False,
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
            "next_global_step": next_global_step,
            "blocked_by": list(blocked_by),
            "violations": list(all_violations),
        },
        "mvp_blocker_matrix": list(_matrix_rows(candidate_available, matrix_blocked_status)),
        "artifacts": [record.to_dict() for record in records],
    }
