#!/usr/bin/env python3
"""Read-only WING Rack preflight runner through the backend WebSocket.

The runner never connects to the mixer directly and never sends write,
proposal, operator-mode mutation, rollback, or emergency-stop messages. It
only asks the already-running backend for operator mode, state, and explicit
read-only parameter readback.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import websockets

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.wing_readonly_preflight_contract import (  # noqa: E402
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
    WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME,
    WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS,
    WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT,
    WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME,
)

EXPECTED_MODE_EVENTS = {"operator_mode_status"}
EXPECTED_STATE_EVENTS = {"state_update"}
EXPECTED_PARAMETER_EVENTS = {"mixer_parameter_value", "mixer_parameter_read_blocked"}
PRESTART_SUMMARY_TYPE = "no_wing_release_check"
PRESTART_SUMMARY_CONNECTION_POLICY = "no_hardware_no_network_no_live_writes"
PRESTART_SUMMARY_BACKEND_CHECK_NAME = "backend_port_free"
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
PHYSICAL_READBACK_STATUSES = {"verified_from_physical_readback"}
PHYSICAL_READBACK_SOURCES = {"physical_readback", "wing_physical_readback", "physical_wing", "wing"}
BACKEND_PORT_PRESTART_CHECK_COMMAND = list(
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
)
BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS = list(
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
)
EXPECTED_PLAN_MATCH_KEYS = (
    "type",
    "status",
    "proof_artifact_schema",
    "proof_runner",
    "selected_mvp_blocker",
    "selected_mvp_blocker_status",
    "url",
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
    "operator_required",
    "expected_plan_required_for_real_run",
    "physical_preflight_requires_existing_backend_connection",
    "backend_port_prestart_check_required",
    "backend_port_prestart_check_must_pass_before_starting_backend",
    "backend_port_prestart_check_command",
    "backend_port_prestart_check_failure_detail_fields",
    "backend_port_prestart_summary_required",
    "backend_port_prestart_summary_path",
    "backend_port_prestart_summary_sha256",
    "backend_port_prestart_summary_type",
    "backend_port_prestart_summary_status",
    "backend_port_prestart_summary_connection_policy",
    "backend_port_prestart_summary_live_write",
    "backend_port_prestart_summary_contacts_backend",
    "backend_port_prestart_summary_queries_console",
    "backend_port_prestart_summary_executes_tests",
    "backend_port_prestart_check_status",
    "manual_review_required_before_readiness_update",
    "contacts_backend",
    "queries_console",
    "execution_allowed_by_plan",
    "live_authority_granted",
    "allowed_request_types",
    "forbidden_request_types",
    "request_sequence",
    "request_id_policy",
    "expected_artifacts",
    "post_run_review_commands",
    "fail_closed_conditions",
    "next_step",
)


class UnexpectedResponseError(RuntimeError):
    def __init__(
        self,
        *,
        request: Dict[str, Any],
        expected_types: Iterable[str],
        payload: Dict[str, Any],
    ):
        self.request = request
        self.expected_types = tuple(sorted(set(expected_types)))
        self.payload = payload
        super().__init__(
            "Unexpected response while waiting for "
            f"{self.expected_types} after {request.get('type')}: {payload.get('type')}"
        )


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_default(value: Any) -> Any:
    try:
        return str(value)
    except Exception:
        return "<unserializable>"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def address_for(channel: int, value_type: str) -> str:
    if value_type == "fader":
        return f"/ch/{channel}/fdr"
    if value_type == "gain":
        return f"/ch/{channel}/in/set/trim"
    if value_type == "pan":
        return f"/ch/{channel}/pan"
    raise ValueError(f"Unsupported value_type: {value_type}")


def request_id_for(channel: int, value_type: str) -> str:
    return f"wing-readonly-preflight:ch{channel}:{value_type}"


def parse_csv_ints(value: str) -> tuple[int, ...]:
    items: list[int] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        items.append(int(raw))
    return tuple(items)


def parse_csv_ints_for_summary(value: str) -> list[int]:
    try:
        return list(parse_csv_ints(value))
    except ValueError:
        return []


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def readback_is_physical(payload: Dict[str, Any]) -> bool:
    return (
        payload.get("status") in PHYSICAL_READBACK_STATUSES
        and str(payload.get("source") or "") in PHYSICAL_READBACK_SOURCES
    )


def readback_identity_matches(
    request: Dict[str, Any],
    payload: Dict[str, Any],
) -> bool:
    return (
        payload.get("request_id") == request.get("request_id")
        and payload.get("channel") == request.get("channel")
        and payload.get("value_type") == request.get("value_type")
        and payload.get("address") == request.get("address")
    )


def response_request_id_matches(
    request: Dict[str, Any],
    payload: Dict[str, Any],
) -> bool:
    return payload.get("request_id") == request.get("request_id")


def transport_failure_classification(exc: Exception) -> tuple[str, list[str], Dict[str, Any]]:
    exc_type = type(exc).__name__
    message = str(exc)
    if (
        exc_type in {"InvalidStatus", "InvalidStatusCode"}
        or "server rejected WebSocket connection" in message
        or "HTTP 403" in message
    ):
        return (
            "read_only_preflight_ws_handshake_rejected",
            ["ws_handshake_rejected"],
            {
                "handshake_rejected_before_request": True,
                "preflight_requests_sent": False,
                "operator_next_step": (
                    "Verify --url points to the intended Automixer backend "
                    "WebSocket, not another local service, then rerun the "
                    "expected-plan preflight."
                ),
            },
        )
    return (
        "read_only_preflight_transport_error",
        ["transport_timeout_or_connection_loss"],
        {
            "handshake_rejected_before_request": False,
            "operator_next_step": (
                "Verify backend WebSocket reachability and rerun the "
                "expected-plan preflight."
            ),
        },
    )


class EventRecorder:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.events_path = run_dir / WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME
        self.summary_path = run_dir / WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME
        self.seq = 0
        self.sent_request_types: list[str] = []

    def record(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.seq += 1
        if event_type == "send":
            self.sent_request_types.append(str(payload.get("type") or "unknown"))
        event = {
            "seq": self.seq,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "event_type": event_type,
            "payload": payload,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=json_default, sort_keys=True) + "\n")

    def write_summary(self, summary: Dict[str, Any]) -> None:
        summary = dict(summary)
        summary.setdefault(
            "captured_at",
            datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        )
        summary.setdefault("events_path", str(self.events_path))
        summary.setdefault("summary_path", str(self.summary_path))
        summary.setdefault("sent_request_types", list(self.sent_request_types))
        summary.setdefault(
            "forbidden_request_types_sent",
            sorted(set(self.sent_request_types).intersection(FORBIDDEN_REQUEST_TYPES)),
        )
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True, default=json_default)
            handle.write("\n")


async def send_and_wait(
    ws: Any,
    recorder: EventRecorder,
    request: Dict[str, Any],
    expected_types: Iterable[str],
    *,
    timeout: float,
) -> Dict[str, Any]:
    expected = set(expected_types)
    request_json = json.dumps(request)
    try:
        await ws.send(request_json)
    except Exception as exc:
        recorder.record(
            "send_error",
            {
                "request_type": request.get("type"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    recorder.record("send", request)
    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"type": "invalid_json", "raw": raw}
    recorder.record("recv", payload)
    if payload.get("type") not in expected:
        raise UnexpectedResponseError(
            request=request,
            expected_types=expected,
            payload=payload,
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a read-only WING Rack preflight through backend WebSocket.")
    parser.add_argument("--url", default="ws://localhost:8765", help="Backend WebSocket URL")
    parser.add_argument("--channels", default="1", help="Comma-separated WING input channels, each 1..48")
    parser.add_argument("--value-types", default="fader,gain,pan", help="Comma-separated read-only value types: fader,gain,pan")
    parser.add_argument(
        "--require-physical-readback",
        action="store_true",
        help="Require every queried parameter to return verified physical readback.",
    )
    parser.add_argument(
        "--operator-confirm-channels",
        default="",
        help=(
            "Comma-separated channels explicitly confirmed by the operator as "
            "safe to inspect; required to match --channels when physical "
            "readback is required."
        ),
    )
    parser.add_argument(
        "--operator-confirm-session-ready",
        action="store_true",
        help=(
            "Confirm the intended backend/WING session is ready for read-only "
            "preflight; required with --require-physical-readback."
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate inputs and print the offline read-only request plan without opening WebSocket.",
    )
    parser.add_argument(
        "--plan-output",
        default=None,
        help="Optional JSON output path for --plan-only.",
    )
    parser.add_argument(
        "--expected-plan",
        default="",
        help=(
            "Required JSON plan for real preflight runs, produced by "
            "--plan-only. Non-plan-only runs fail closed before WebSocket when "
            "this file is omitted or mismatched."
        ),
    )
    parser.add_argument(
        "--prestart-summary",
        default="",
        help=(
            "Required no-WING prestart summary.json produced by "
            "tools/no_wing_release_check.py --quick before the intended "
            "backend/WING session."
        ),
    )
    parser.add_argument("--timeout", type=float, default=5.0, help="Timeout per WebSocket step")
    parser.add_argument("--artifact-root", default="artifacts/wing_readonly_preflight", help="Artifact root directory")
    return parser


def validate_args(args: argparse.Namespace) -> Optional[str]:
    try:
        channels = parse_csv_ints(args.channels)
    except ValueError:
        return "--channels must be comma-separated integers"
    if not channels:
        return "--channels must include at least one channel"
    if len(set(channels)) != len(channels):
        return "--channels must not contain duplicates"
    if any(channel < 1 or channel > 48 for channel in channels):
        return "--channels values must be in 1..48"
    value_types = parse_csv_strings(args.value_types)
    if not value_types:
        return "--value-types must include at least one value type"
    if len(set(value_types)) != len(value_types):
        return "--value-types must not contain duplicates"
    unsupported = sorted(set(value_types) - {"fader", "gain", "pan"})
    if unsupported:
        return f"--value-types unsupported: {','.join(unsupported)}"
    if args.require_physical_readback:
        try:
            confirmed_channels = parse_csv_ints(args.operator_confirm_channels)
        except ValueError:
            return "--operator-confirm-channels must be comma-separated integers"
        if not confirmed_channels:
            return "--operator-confirm-channels must match --channels when --require-physical-readback is set"
        if confirmed_channels != channels:
            return "--operator-confirm-channels must exactly match --channels"
        if not args.operator_confirm_session_ready:
            return "--operator-confirm-session-ready is required when --require-physical-readback is set"
    if args.timeout <= 0:
        return "--timeout must be positive"
    return None


def build_read_requests(channels: tuple[int, ...], value_types: tuple[str, ...]) -> list[Dict[str, Any]]:
    requests: list[Dict[str, Any]] = []
    for channel in channels:
        for value_type in value_types:
            requests.append(
                {
                    "type": "query_mixer_parameter",
                    "request_id": request_id_for(channel, value_type),
                    "channel": channel,
                    "value_type": value_type,
                    "address": address_for(channel, value_type),
                }
            )
    return requests


def load_expected_plan(path: str) -> tuple[Dict[str, Any], str]:
    plan_path = Path(path)
    raw = plan_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected plan JSON root must be an object")
    return payload, sha256_bytes(raw)


def prestart_summary_operator_next_step(reason: str) -> str:
    if reason == "prestart_summary_missing":
        return (
            "Run PYTHONPATH=backend python3 tools/no_wing_release_check.py "
            "--quick before starting the intended backend/WING session, then "
            "rerun the read-only plan and real preflight with --prestart-summary "
            "pointing to that passed summary.json."
        )
    return (
        "Use a passed no-WING prestart summary.json from the current intended "
        "backend/WING session. Do not open the backend/WING WebSocket until "
        "the prestart summary validates."
    )


def load_backend_port_prestart_summary(path: str) -> tuple[Dict[str, Any], list[str]]:
    detail: Dict[str, Any] = {"backend_port_prestart_summary_required": True}
    violations: list[str] = []
    if not path:
        violations.append("prestart_summary_missing")
        return detail, violations

    summary_path = Path(path)
    detail["backend_port_prestart_summary_path"] = str(summary_path)
    try:
        raw = summary_path.read_bytes()
    except FileNotFoundError:
        violations.append("prestart_summary_file_missing")
        return detail, violations
    except OSError as exc:
        violations.append(f"prestart_summary_file_error:{type(exc).__name__}")
        return detail, violations

    detail["backend_port_prestart_summary_sha256"] = sha256_bytes(raw)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        violations.append(f"prestart_summary_load_error:{type(exc).__name__}")
        return detail, violations
    if not isinstance(payload, dict):
        violations.append("prestart_summary_root_not_object")
        return detail, violations

    detail["backend_port_prestart_summary_type"] = payload.get("type")
    detail["backend_port_prestart_summary_status"] = payload.get("status")
    detail["backend_port_prestart_summary_connection_policy"] = payload.get(
        "connection_policy"
    )
    detail["backend_port_prestart_summary_live_write"] = payload.get("live_write")
    detail["backend_port_prestart_summary_contacts_backend"] = payload.get(
        "contacts_backend"
    )
    detail["backend_port_prestart_summary_queries_console"] = payload.get(
        "queries_console"
    )
    detail["backend_port_prestart_summary_executes_tests"] = payload.get(
        "executes_tests"
    )

    if payload.get("type") != PRESTART_SUMMARY_TYPE:
        violations.append("prestart_summary_type_mismatch")
    if payload.get("status") != "passed":
        violations.append("prestart_summary_status_not_passed")
    if payload.get("connection_policy") != PRESTART_SUMMARY_CONNECTION_POLICY:
        violations.append("prestart_summary_connection_policy_mismatch")
    if payload.get("live_write") is not False:
        violations.append("prestart_summary_live_write_not_false")
    if payload.get("contacts_backend") is not False:
        violations.append("prestart_summary_contacts_backend_not_false")
    if payload.get("queries_console") is not False:
        violations.append("prestart_summary_queries_console_not_false")
    if payload.get("executes_tests") is not False:
        violations.append("prestart_summary_executes_tests_not_false")

    results = payload.get("results")
    backend_check = None
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict) and item.get("name") == PRESTART_SUMMARY_BACKEND_CHECK_NAME:
                backend_check = item
                break
    if backend_check is None:
        violations.append("prestart_summary_backend_port_check_missing")
    else:
        detail["backend_port_prestart_check_status"] = backend_check.get("status")
        if backend_check.get("status") != "passed":
            violations.append("prestart_summary_backend_port_check_not_passed")
        check_detail = backend_check.get("detail")
        if isinstance(check_detail, dict):
            if check_detail.get("live_authority_granted") is True:
                violations.append("prestart_summary_live_authority_granted")
            if check_detail.get("safe_to_stop_listener_by_gate") is True:
                violations.append("prestart_summary_safe_to_stop_listener_true")

    return detail, violations


def build_offline_preflight_plan(args: argparse.Namespace) -> Dict[str, Any]:
    channels = parse_csv_ints(args.channels)
    value_types = parse_csv_strings(args.value_types)
    read_requests = build_read_requests(channels, value_types)
    artifact_root = str(args.artifact_root)
    prestart_detail, _ = load_backend_port_prestart_summary(args.prestart_summary)
    return {
        "type": "wing_readonly_preflight_offline_plan",
        "status": "ready_for_operator_review",
        "proof_artifact_schema": WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA,
        "proof_runner": "wing_readonly_preflight_runner",
        "selected_mvp_blocker": "WING read-only connection proof",
        "selected_mvp_blocker_status": "tooling_ready_waits_real_wing",
        "url": args.url,
        "artifact_root": artifact_root,
        "channels": list(channels),
        "operator_channel_confirmation_required": bool(args.require_physical_readback),
        "operator_confirmed_channels": parse_csv_ints_for_summary(args.operator_confirm_channels),
        "operator_session_confirmation_required": bool(args.require_physical_readback),
        "operator_confirmed_session_ready": bool(args.operator_confirm_session_ready),
        "value_types": list(value_types),
        "require_physical_readback": bool(args.require_physical_readback),
        "read_only": True,
        "live_write": False,
        "operator_required": True,
        "expected_plan_required_for_real_run": True,
        "physical_preflight_requires_existing_backend_connection": True,
        "backend_port_prestart_check_required": True,
        "backend_port_prestart_check_must_pass_before_starting_backend": True,
        "backend_port_prestart_check_command": list(BACKEND_PORT_PRESTART_CHECK_COMMAND),
        "backend_port_prestart_check_failure_detail_fields": list(
            BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
        ),
        **prestart_detail,
        "manual_review_required_before_readiness_update": True,
        "contacts_backend": False,
        "queries_console": False,
        "execution_allowed_by_plan": False,
        "live_authority_granted": False,
        "allowed_request_types": [
            "get_operator_mode",
            "get_state",
            "query_mixer_parameter",
        ],
        "forbidden_request_types": sorted(FORBIDDEN_REQUEST_TYPES),
        "request_sequence": [
            {"type": "get_operator_mode"},
            {"type": "get_state"},
            *read_requests,
        ],
        "request_id_policy": "required_for_every_query_mixer_parameter_response",
        "expected_artifacts": [
            f"{artifact_root}/<run_id>/{WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME}",
            f"{artifact_root}/<run_id>/{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME}",
        ],
        "post_run_review_commands": [
            (
                "python3 tools/review_wing_readonly_preflight_artifacts.py "
                f"--artifact-root {artifact_root} "
                "--output Docs/reports/WING_READONLY_PREFLIGHT_REVIEW_LATEST.md "
                "--decision-template-output "
                f"{WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT}"
            ),
            (
                "python3 tools/review_wing_readonly_preflight_decisions.py "
                f"--artifact-root {artifact_root} "
                f"--decision-root {WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT} "
                "--output Docs/reports/WING_READONLY_PREFLIGHT_DECISION_REVIEW_LATEST.md"
            ),
        ],
        "fail_closed_conditions": list(
            WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS
        ),
        "next_step": "Run without --plan-only only when the intended backend/WING session is ready.",
    }


def validate_expected_plan(args: argparse.Namespace, plan: Dict[str, Any]) -> list[str]:
    expected = build_offline_preflight_plan(args)
    violations: list[str] = []
    for key in EXPECTED_PLAN_MATCH_KEYS:
        if plan.get(key) != expected.get(key):
            violations.append(f"expected_plan_{key}_mismatch")
    if plan.get("plan_output") != args.expected_plan:
        violations.append("expected_plan_output_path_mismatch")
    return violations


def expected_plan_operator_next_step(reason: str) -> str:
    if reason == "expected_plan_missing":
        return (
            "Run the read-only preflight runner with --plan-only to generate "
            "the offline expected plan, then rerun the real preflight with "
            "--expected-plan pointing to that file."
        )
    if reason == "expected_plan_missing_or_invalid":
        return (
            "Regenerate the offline expected plan from the same channel, value "
            "type, artifact-root, and operator-confirmation arguments, then "
            "rerun the real preflight with --expected-plan."
        )
    if reason == "expected_plan_mismatch":
        return (
            "Regenerate or select the expected plan that exactly matches the "
            "current runtime arguments and prestart contract before opening "
            "the backend/WING WebSocket."
        )
    return "Resolve the expected-plan gate before opening the backend/WING WebSocket."


async def run(args: argparse.Namespace) -> int:
    validation_error = validate_args(args)
    prestart_detail, prestart_violations = load_backend_port_prestart_summary(
        args.prestart_summary
    )
    if args.plan_only:
        if validation_error:
            plan = {
                "type": "wing_readonly_preflight_offline_plan",
                "status": "blocked",
                "reason": validation_error,
                "read_only": True,
                "live_write": False,
                "operator_required": True,
                "selected_mvp_blocker": "WING read-only connection proof",
                "contacts_backend": False,
                "queries_console": False,
                "execution_allowed_by_plan": False,
                "live_authority_granted": False,
                "manual_review_required_before_readiness_update": True,
            }
            print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        if prestart_violations:
            plan = {
                "type": "wing_readonly_preflight_offline_plan",
                "status": "blocked",
                "reason": "prestart_summary_missing_or_invalid",
                "read_only": True,
                "live_write": False,
                "operator_required": True,
                "selected_mvp_blocker": "WING read-only connection proof",
                "contacts_backend": False,
                "queries_console": False,
                "execution_allowed_by_plan": False,
                "live_authority_granted": False,
                "manual_review_required_before_readiness_update": True,
                "prestart_summary_violations": prestart_violations,
                "operator_next_step": prestart_summary_operator_next_step(
                    prestart_violations[0]
                ),
            }
            plan.update(prestart_detail)
            print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        plan = build_offline_preflight_plan(args)
        if args.plan_output:
            output_path = Path(args.plan_output)
            plan["plan_output"] = str(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    channels = parse_csv_ints(args.channels) if validation_error is None else ()
    value_types = parse_csv_strings(args.value_types) if validation_error is None else ()
    run_id = f"{utc_stamp()}_wing_readonly_preflight"
    run_dir = Path(args.artifact_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    recorder = EventRecorder(run_dir)

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "status": "started",
        "proof_artifact_schema": WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA,
        "proof_runner": "wing_readonly_preflight_runner",
        "url": args.url,
        "channels": list(channels),
        "operator_channel_confirmation_required": bool(args.require_physical_readback),
        "operator_confirmed_channels": parse_csv_ints_for_summary(
            args.operator_confirm_channels
        ),
        "operator_session_confirmation_required": bool(args.require_physical_readback),
        "operator_confirmed_session_ready": bool(args.operator_confirm_session_ready),
        "value_types": list(value_types),
        "require_physical_readback": bool(args.require_physical_readback),
        "expected_plan_required_for_real_run": True,
        "read_only": True,
        "live_write": False,
        "write_request_sent": False,
        "proposal_request_sent": False,
        "physical_preflight_requires_existing_backend_connection": True,
        "backend_port_prestart_check_required": True,
        "backend_port_prestart_check_must_pass_before_starting_backend": True,
        "backend_port_prestart_check_command": list(BACKEND_PORT_PRESTART_CHECK_COMMAND),
        "backend_port_prestart_check_failure_detail_fields": list(
            BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
        ),
        **prestart_detail,
        "prestart_summary_violations": prestart_violations,
        "request_id_policy": "required_for_every_query_mixer_parameter_response",
        "forbidden_request_types": sorted(FORBIDDEN_REQUEST_TYPES),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }

    if validation_error:
        summary.update({"status": "blocked", "reason": validation_error})
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 2

    if prestart_violations:
        summary.update(
            {
                "status": "blocked",
                "reason": "prestart_summary_missing_or_invalid",
                "operator_next_step": prestart_summary_operator_next_step(
                    prestart_violations[0]
                ),
            }
        )
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 2

    if not args.expected_plan:
        summary.update(
            {
                "status": "blocked",
                "reason": "expected_plan_missing",
                "expected_plan_match": False,
                "expected_plan_violations": ["expected_plan_missing"],
                "operator_next_step": expected_plan_operator_next_step(
                    "expected_plan_missing"
                ),
            }
        )
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 2

    if args.expected_plan:
        expected_plan_path = str(args.expected_plan)
        summary["expected_plan_path"] = expected_plan_path
        try:
            expected_plan, expected_plan_sha256 = load_expected_plan(expected_plan_path)
        except Exception as exc:
            summary.update(
                {
                    "status": "blocked",
                    "reason": "expected_plan_missing_or_invalid",
                    "expected_plan_match": False,
                    "expected_plan_error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "operator_next_step": expected_plan_operator_next_step(
                        "expected_plan_missing_or_invalid"
                    ),
                }
            )
            recorder.write_summary(summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 2
        expected_plan_violations = validate_expected_plan(args, expected_plan)
        summary.update(
            {
                "expected_plan_sha256": expected_plan_sha256,
                "expected_plan_match": not expected_plan_violations,
                "expected_plan_violations": expected_plan_violations,
            }
        )
        if expected_plan_violations:
            summary.update(
                {
                    "status": "blocked",
                    "reason": "expected_plan_mismatch",
                    "operator_next_step": expected_plan_operator_next_step(
                        "expected_plan_mismatch"
                    ),
                }
            )
            recorder.write_summary(summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 2

    try:
        async with websockets.connect(args.url) as ws:
            mode = await send_and_wait(
                ws,
                recorder,
                {"type": "get_operator_mode"},
                EXPECTED_MODE_EVENTS,
                timeout=args.timeout,
            )
            state_payload = await send_and_wait(
                ws,
                recorder,
                {"type": "get_state"},
                EXPECTED_STATE_EVENTS,
                timeout=args.timeout,
            )
            state = state_payload.get("state") if isinstance(state_payload.get("state"), dict) else {}
            connection_mode = str(state_payload.get("mode") or mode.get("connection_mode") or "")
            summary.update(
                {
                    "operator_mode": mode.get("mode"),
                    "operator_mode_status": mode,
                    "connection_mode": connection_mode,
                    "state_key_count": len(state),
                    "state_sample_addresses": sorted(list(state.keys()))[:12],
                }
            )

            if connection_mode != "wing":
                summary.update(
                    {
                        "status": "blocked",
                        "reason": "backend_not_in_wing_mode",
                        "fail_closed": True,
                        "blockers": ["event_log_wing_mode_not_proven"],
                    }
                )
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 3

            readbacks: list[Dict[str, Any]] = []
            blockers: list[str] = []
            for request in build_read_requests(channels, value_types):
                payload = await send_and_wait(
                    ws,
                    recorder,
                    request,
                    EXPECTED_PARAMETER_EVENTS,
                    timeout=args.timeout,
                )
                readbacks.append(payload)
                if not response_request_id_matches(request, payload):
                    blockers.append(
                        "readback_request_id_mismatch:"
                        f"ch{request['channel']}:{request['value_type']}"
                    )
                if payload.get("type") == "mixer_parameter_read_blocked":
                    blockers.append(f"{payload.get('reason')}:ch{request['channel']}:{request['value_type']}")
                    continue
                if not readback_identity_matches(request, payload):
                    blockers.append(
                        "readback_identity_mismatch:"
                        f"ch{request['channel']}:{request['value_type']}"
                    )
                if args.require_physical_readback and not readback_is_physical(payload):
                    blockers.append(f"physical_readback_missing:ch{request['channel']}:{request['value_type']}")

            physical_count = sum(1 for item in readbacks if readback_is_physical(item))
            forbidden_sent = sorted(set(recorder.sent_request_types).intersection(FORBIDDEN_REQUEST_TYPES))
            if forbidden_sent:
                blockers.append("forbidden_request_type_sent")
            status = "passed" if not blockers else "blocked"
            reason = (
                "wing_readonly_preflight_passed"
                if not blockers
                else "wing_readonly_preflight_blocked"
            )
            summary.update(
                {
                    "status": status,
                    "reason": reason,
                    "readback_count": len(readbacks),
                    "physical_readback_count": physical_count,
                    "readbacks": readbacks,
                    "blockers": blockers,
                    "write_request_sent": bool(set(recorder.sent_request_types).intersection(LIVE_WRITE_REQUEST_TYPES)),
                    "proposal_request_sent": bool(set(recorder.sent_request_types).intersection(PROPOSAL_REQUEST_TYPES)),
                }
            )
            recorder.write_summary(summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0 if status == "passed" else 4
    except UnexpectedResponseError as exc:
        summary.update(
            {
                "status": "blocked",
                "reason": "unexpected_response_type",
                "fail_closed": True,
                "blockers": ["unexpected_response_type"],
                "unexpected_response": {
                    "request_type": exc.request.get("type"),
                    "expected_types": list(exc.expected_types),
                    "actual_type": exc.payload.get("type"),
                    "payload": exc.payload,
                },
            }
        )
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 4
    except asyncio.TimeoutError as exc:
        summary.update(
            {
                "status": "blocked",
                "reason": "read_only_preflight_timeout",
                "fail_closed": True,
                "blockers": ["transport_timeout_or_connection_loss"],
                "timeout_sec": args.timeout,
                "transport_failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1
    except Exception as exc:
        reason, blockers, transport_context = transport_failure_classification(exc)
        summary.update(
            {
                "status": "blocked",
                "reason": reason,
                "fail_closed": True,
                "blockers": blockers,
                "transport_context": transport_context,
                "transport_failure": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
