"""Offline tests for the read-only WING preflight runner."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from backend.wing_readonly_preflight_contract import (
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
    WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
    WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME,
    WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_SOURCES,
    WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_STATUSES,
    WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS,
    WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA,
    WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON,
    WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS,
    WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
    WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES,
    WING_READONLY_PREFLIGHT_RUNNER_URL,
    WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT,
    WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME,
)


RUNNER_PATH = Path(__file__).resolve().parents[1] / "tools" / "wing_readonly_preflight_runner.py"
SPEC = importlib.util.spec_from_file_location("wing_readonly_preflight_runner", RUNNER_PATH)
assert SPEC is not None
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


class FakeWebSocket:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.sent: list[dict] = []
        self.url: str | None = None

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    async def recv(self) -> str:
        if not self.responses:
            raise AssertionError("No fake WebSocket response queued")
        return json.dumps(self.responses.pop(0))


class FakeWebSocketConnect:
    def __init__(self, websocket: FakeWebSocket):
        self.websocket = websocket

    async def __aenter__(self) -> FakeWebSocket:
        return self.websocket

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class TimeoutWebSocket(FakeWebSocket):
    async def recv(self) -> str:
        raise asyncio.TimeoutError()


class SendFailWebSocket(FakeWebSocket):
    async def send(self, payload: str) -> None:
        raise RuntimeError("send failed")


class InvalidStatus(Exception):
    pass


def patch_websocket_connect(monkeypatch: pytest.MonkeyPatch, websocket: FakeWebSocket) -> None:
    def fake_connect(url: str) -> FakeWebSocketConnect:
        websocket.url = url
        return FakeWebSocketConnect(websocket)

    monkeypatch.setattr(runner.websockets, "connect", fake_connect)


def test_runner_uses_shared_run_summary_contract() -> None:
    assert runner.RUN_SUMMARY_STATUS == WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS
    assert runner.RUN_SUMMARY_REASON == WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON


def prestart_summary_path(tmp_path: Path) -> Path:
    return tmp_path / "no-wing-prestart-summary.json"


def ensure_prestart_summary(tmp_path: Path) -> Path:
    path = prestart_summary_path(tmp_path)
    if path.exists():
        return path
    payload = {
        "type": "no_wing_release_check",
        "status": "passed",
        "connection_policy": "no_hardware_no_network_no_live_writes",
        "live_write": False,
        "contacts_backend": False,
        "queries_console": False,
        "executes_tests": False,
        "results": [
            {
                "name": "backend_port_free",
                "status": "passed",
                "detail": {
                    "host": "127.0.0.1",
                    "port": 8765,
                    "safe_to_stop_listener_by_gate": False,
                    "live_authority_granted": False,
                },
            }
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(tmp_path: Path, *argv: str):
    extra = list(argv)
    if "--prestart-summary" not in extra:
        extra.extend(["--prestart-summary", str(ensure_prestart_summary(tmp_path))])
    return runner.build_parser().parse_args(
        [
            "--url",
            "ws://fake-backend",
            "--artifact-root",
            str(tmp_path),
            "--timeout",
            "0.2",
            *extra,
        ]
    )


def read_summary(artifact_root: Path) -> dict:
    summaries = list(artifact_root.glob(f"*/{WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME}"))
    assert len(summaries) == 1
    return json.loads(summaries[0].read_text(encoding="utf-8"))


def read_events(summary: dict) -> list[dict]:
    events_path = Path(summary["events_path"])
    return [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]


def test_parser_defaults_use_shared_target_contract() -> None:
    args = runner.build_parser().parse_args([])

    assert args.url == WING_READONLY_PREFLIGHT_RUNNER_URL
    assert args.channels == WING_READONLY_PREFLIGHT_RUNNER_CHANNELS
    assert args.value_types == WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES
    assert args.artifact_root == WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT
    assert runner.SUPPORTED_VALUE_TYPES == set(
        WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES
    )
    assert runner.PHYSICAL_READBACK_STATUSES == set(
        WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_STATUSES
    )
    assert runner.PHYSICAL_READBACK_SOURCES == set(
        WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_SOURCES
    )


def write_expected_plan(tmp_path: Path, *argv: str) -> Path:
    plan_path = tmp_path / "expected-plan.json"
    args = parse_args(
        tmp_path,
        *argv,
        "--plan-only",
        "--plan-output",
        str(plan_path),
    )
    plan = runner.build_offline_preflight_plan(args)
    plan["plan_output"] = str(plan_path)
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return plan_path


def assert_summary_prestart_contract(summary: dict) -> None:
    assert summary["backend_port_prestart_check_required"] is True
    assert summary["backend_port_prestart_check_must_pass_before_starting_backend"] is True
    assert summary["backend_port_prestart_check_command"] == list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
    )
    assert summary["backend_port_prestart_check_failure_detail_fields"] == list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    )
    prestart_path = prestart_summary_path(Path(summary["backend_port_prestart_summary_path"]).parent)
    assert summary["backend_port_prestart_summary_required"] is True
    assert Path(summary["backend_port_prestart_summary_path"]).exists()
    assert summary["backend_port_prestart_summary_sha256"] == sha256_file(
        Path(summary["backend_port_prestart_summary_path"])
    )
    assert summary["backend_port_prestart_summary_path"] == str(prestart_path)
    assert summary["backend_port_prestart_summary_type"] == "no_wing_release_check"
    assert summary["backend_port_prestart_summary_status"] == "passed"
    assert (
        summary["backend_port_prestart_summary_connection_policy"]
        == "no_hardware_no_network_no_live_writes"
    )
    assert summary["backend_port_prestart_summary_live_write"] is False
    assert summary["backend_port_prestart_summary_contacts_backend"] is False
    assert summary["backend_port_prestart_summary_queries_console"] is False
    assert summary["backend_port_prestart_summary_executes_tests"] is False
    assert summary["backend_port_prestart_check_status"] == "passed"


def test_forbidden_request_types_cover_hard_safety_boundary() -> None:
    expected = {
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
        "set_operator_mode",
        "create_operator_proposal",
        "apply_operator_proposal",
        "rollback_last_manual_write",
        "emergency_stop",
        "connect_mixer",
        "connect_wing",
        "disconnect_wing",
        "disconnect_mixer",
    }

    assert expected <= runner.FORBIDDEN_REQUEST_TYPES
    assert expected & {"get_operator_mode", "get_state", "query_mixer_parameter"} == set()


@pytest.mark.parametrize(
    ("argv", "expected_error"),
    [
        (["--channels", "0"], "--channels values must be in 1..48"),
        (["--channels", "49"], "--channels values must be in 1..48"),
        (["--channels", ""], "--channels must include at least one channel"),
        (["--channels", "3,3"], "--channels must not contain duplicates"),
        (["--value-types", "fader,fader"], "--value-types must not contain duplicates"),
        (["--value-types", "fader,eq"], "--value-types unsupported: eq"),
        (
            ["--channels", "3", "--require-physical-readback"],
            "--operator-confirm-channels must match --channels when --require-physical-readback is set",
        ),
        (
            [
                "--channels",
                "3",
                "--operator-confirm-channels",
                "4",
                "--require-physical-readback",
            ],
            "--operator-confirm-channels must exactly match --channels",
        ),
        (
            [
                "--channels",
                "3",
                "--operator-confirm-channels",
                "3",
                "--require-physical-readback",
            ],
            "--operator-confirm-session-ready is required when --require-physical-readback is set",
        ),
        (
            [
                "--operator-confirm-channels",
                "bad",
                "--require-physical-readback",
            ],
            "--operator-confirm-channels must be comma-separated integers",
        ),
        (["--timeout", "0"], "--timeout must be positive"),
    ],
)
def test_validate_args_blocks_invalid_preflight_requests(
    tmp_path: Path,
    argv: list[str],
    expected_error: str,
) -> None:
    args = parse_args(tmp_path, *argv)

    assert runner.validate_args(args) == expected_error


@pytest.mark.asyncio
async def test_readonly_preflight_passes_with_physical_readbacks_without_write_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [
            {"type": "operator_mode_status", "mode": "assist", "connection_mode": "wing"},
            {"type": "state_update", "mode": "wing", "state": {"/ch/3/fdr": -12.0}},
            {
                "type": "mixer_parameter_value",
                "request_id": "wing-readonly-preflight:ch3:fader",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "channel": 3,
                "value_type": "fader",
                "address": "/ch/3/fdr",
                "value": -12.0,
            },
            {
                "type": "mixer_parameter_value",
                "request_id": "wing-readonly-preflight:ch3:gain",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "channel": 3,
                "value_type": "gain",
                "address": "/ch/3/in/set/trim",
                "value": -0.25,
            },
            {
                "type": "mixer_parameter_value",
                "request_id": "wing-readonly-preflight:ch3:pan",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "channel": 3,
                "value_type": "pan",
                "address": "/ch/3/pan",
                "value": 0.0,
            },
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)
    expected_plan = write_expected_plan(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader,gain,pan",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
    )
    args = parse_args(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader,gain,pan",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
        "--expected-plan",
        str(expected_plan),
    )

    exit_code = await runner.run(args)

    assert exit_code == 0
    assert websocket.url == "ws://fake-backend"
    assert [request["type"] for request in websocket.sent] == [
        "get_operator_mode",
        "get_state",
        "query_mixer_parameter",
        "query_mixer_parameter",
        "query_mixer_parameter",
    ]
    summary = read_summary(tmp_path)
    assert summary["status"] == WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS
    assert summary["reason"] == WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON
    assert summary["captured_at"].endswith("Z")
    assert summary["connection_mode"] == "wing"
    assert summary["operator_channel_confirmation_required"] is True
    assert summary["operator_confirmed_channels"] == [3]
    assert summary["operator_session_confirmation_required"] is True
    assert summary["operator_confirmed_session_ready"] is True
    assert summary["expected_plan_path"] == str(expected_plan)
    assert summary["expected_plan_match"] is True
    assert summary["expected_plan_sha256"]
    assert summary["expected_plan_violations"] == []
    assert_summary_prestart_contract(summary)
    assert summary["request_id_policy"] == "required_for_every_query_mixer_parameter_response"
    assert summary["read_only"] is True
    assert summary["live_write"] is False
    assert summary["write_request_sent"] is False
    assert summary["proposal_request_sent"] is False
    assert summary["physical_readback_count"] == 3
    assert summary["forbidden_request_types_sent"] == []
    assert [event["event_type"] for event in read_events(summary)] == [
        "send",
        "recv",
        "send",
        "recv",
        "send",
        "recv",
        "send",
        "recv",
        "send",
        "recv",
    ]


@pytest.mark.asyncio
async def test_readonly_preflight_passes_with_state_unavailable_but_readbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [
            {"type": "operator_mode_status", "mode": "assist", "connection_mode": "wing"},
            {
                "type": "state_update",
                "mode": "wing",
                "state": {},
                "status": "state_unavailable",
                "reason": "mixer_client_state_source_unavailable",
            },
            {
                "type": "mixer_parameter_value",
                "request_id": "wing-readonly-preflight:ch3:fader",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "channel": 3,
                "value_type": "fader",
                "address": "/ch/3/fdr",
                "value": -12.0,
            },
            {
                "type": "mixer_parameter_value",
                "request_id": "wing-readonly-preflight:ch3:gain",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "channel": 3,
                "value_type": "gain",
                "address": "/ch/3/in/set/trim",
                "value": -0.25,
            },
            {
                "type": "mixer_parameter_value",
                "request_id": "wing-readonly-preflight:ch3:pan",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "channel": 3,
                "value_type": "pan",
                "address": "/ch/3/pan",
                "value": 0.0,
            },
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)
    expected_plan = write_expected_plan(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader,gain,pan",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
    )
    args = parse_args(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader,gain,pan",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
        "--expected-plan",
        str(expected_plan),
    )

    exit_code = await runner.run(args)

    assert exit_code == 0
    assert websocket.url == "ws://fake-backend"
    assert [request["type"] for request in websocket.sent] == [
        "get_operator_mode",
        "get_state",
        "query_mixer_parameter",
        "query_mixer_parameter",
        "query_mixer_parameter",
    ]
    summary = read_summary(tmp_path)
    assert summary["status"] == WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS
    assert summary["reason"] == WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON
    assert summary["captured_at"].endswith("Z")
    assert summary["connection_mode"] == "wing"
    assert summary["operator_channel_confirmation_required"] is True
    assert summary["operator_confirmed_channels"] == [3]
    assert summary["operator_session_confirmation_required"] is True
    assert summary["operator_confirmed_session_ready"] is True
    assert summary["expected_plan_path"] == str(expected_plan)
    assert summary["expected_plan_match"] is True
    assert summary["expected_plan_sha256"]
    assert summary["expected_plan_violations"] == []
    assert_summary_prestart_contract(summary)
    assert summary["request_id_policy"] == "required_for_every_query_mixer_parameter_response"
    assert summary["read_only"] is True
    assert summary["live_write"] is False
    assert summary["write_request_sent"] is False
    assert summary["proposal_request_sent"] is False
    assert summary["physical_readback_count"] == 3
    assert summary["forbidden_request_types_sent"] == []
    assert [event["event_type"] for event in read_events(summary)] == [
        "send",
        "recv",
        "send",
        "recv",
        "send",
        "recv",
        "send",
        "recv",
        "send",
        "recv",
    ]


@pytest.mark.asyncio
async def test_readonly_preflight_blocks_when_backend_is_not_wing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [
            {"type": "operator_mode_status", "mode": "assist", "connection_mode": "offline"},
            {"type": "state_update", "mode": "offline", "state": {}},
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)
    expected_plan = write_expected_plan(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "gain",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
    )
    args = parse_args(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "gain",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
        "--expected-plan",
        str(expected_plan),
    )

    exit_code = await runner.run(args)

    assert exit_code == 3
    assert [request["type"] for request in websocket.sent] == ["get_operator_mode", "get_state"]
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "backend_not_in_wing_mode"
    assert summary["fail_closed"] is True
    assert summary["blockers"] == ["event_log_wing_mode_not_proven"]
    assert summary["connection_mode"] == "offline"
    assert summary["write_request_sent"] is False
    assert summary["proposal_request_sent"] is False


@pytest.mark.asyncio
async def test_readonly_preflight_blocks_unexpected_response_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [
            {"type": "operator_mode_status", "mode": "assist", "connection_mode": "wing"},
            {"type": "heartbeat", "status": "ok"},
            {"type": "state_update", "mode": "wing", "state": {"/ch/3/fdr": -12.0}},
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)
    expected_plan = write_expected_plan(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
    )
    args = parse_args(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
        "--expected-plan",
        str(expected_plan),
    )

    exit_code = await runner.run(args)

    assert exit_code == 4
    assert [request["type"] for request in websocket.sent] == [
        "get_operator_mode",
        "get_state",
    ]
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "unexpected_response_type"
    assert summary["fail_closed"] is True
    assert summary["blockers"] == ["unexpected_response_type"]
    assert summary["unexpected_response"] == {
        "request_type": "get_state",
        "expected_types": ["state_update"],
        "actual_type": "heartbeat",
        "payload": {"type": "heartbeat", "status": "ok"},
    }
    assert summary["sent_request_types"] == ["get_operator_mode", "get_state"]
    assert [event["event_type"] for event in read_events(summary)] == [
        "send",
        "recv",
        "send",
        "recv",
    ]


@pytest.mark.asyncio
async def test_readonly_preflight_writes_fail_closed_timeout_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = TimeoutWebSocket([])
    patch_websocket_connect(monkeypatch, websocket)
    expected_plan = write_expected_plan(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
    )
    args = parse_args(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
        "--expected-plan",
        str(expected_plan),
    )

    exit_code = await runner.run(args)

    assert exit_code == 1
    assert [request["type"] for request in websocket.sent] == ["get_operator_mode"]
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "read_only_preflight_timeout"
    assert summary["fail_closed"] is True
    assert summary["blockers"] == ["transport_timeout_or_connection_loss"]
    assert summary["timeout_sec"] == 0.2
    assert summary["transport_failure"]["type"] == "TimeoutError"
    assert summary["write_request_sent"] is False
    assert summary["proposal_request_sent"] is False
    assert summary["sent_request_types"] == ["get_operator_mode"]
    assert [event["event_type"] for event in read_events(summary)] == ["send"]


@pytest.mark.asyncio
async def test_readonly_preflight_writes_fail_closed_send_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = SendFailWebSocket([])
    patch_websocket_connect(monkeypatch, websocket)
    expected_plan = write_expected_plan(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
    )
    args = parse_args(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
        "--expected-plan",
        str(expected_plan),
    )

    exit_code = await runner.run(args)

    assert exit_code == 1
    assert [request["type"] for request in websocket.sent] == []
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "read_only_preflight_transport_error"
    assert summary["fail_closed"] is True
    assert summary["blockers"] == ["transport_timeout_or_connection_loss"]
    assert summary["transport_failure"]["type"] == "RuntimeError"
    assert summary["transport_failure"]["message"] == "send failed"
    assert summary["write_request_sent"] is False
    assert summary["proposal_request_sent"] is False
    assert summary["sent_request_types"] == []
    assert [event["event_type"] for event in read_events(summary)] == ["send_error"]


@pytest.mark.asyncio
async def test_readonly_preflight_writes_fail_closed_handshake_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_connect(url: str):
        raise InvalidStatus("server rejected WebSocket connection: HTTP 403")

    monkeypatch.setattr(runner.websockets, "connect", reject_connect)
    expected_plan = write_expected_plan(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
    )
    args = parse_args(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
        "--expected-plan",
        str(expected_plan),
    )

    exit_code = await runner.run(args)

    assert exit_code == 1
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "read_only_preflight_ws_handshake_rejected"
    assert summary["fail_closed"] is True
    assert summary["blockers"] == ["ws_handshake_rejected"]
    assert summary["transport_context"]["handshake_rejected_before_request"] is True
    assert summary["transport_context"]["preflight_requests_sent"] is False
    assert "intended Automixer backend WebSocket" in summary["transport_context"][
        "operator_next_step"
    ]
    assert summary["transport_failure"]["type"] == "InvalidStatus"
    assert summary["transport_failure"]["message"] == (
        "server rejected WebSocket connection: HTTP 403"
    )
    assert summary["write_request_sent"] is False
    assert summary["proposal_request_sent"] is False
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_blocks_missing_physical_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [
            {"type": "operator_mode_status", "mode": "assist", "connection_mode": "wing"},
            {"type": "state_update", "mode": "wing", "state": {"/ch/3/fdr": -12.0}},
            {
                "type": "mixer_parameter_value",
                "request_id": "wing-readonly-preflight:ch3:fader",
                "status": "verified_from_cache",
                "source": "mixer_state_cache",
                "channel": 3,
                "value_type": "fader",
                "address": "/ch/3/fdr",
                "value": -12.0,
            },
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)
    expected_plan = write_expected_plan(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
    )
    args = parse_args(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
        "--expected-plan",
        str(expected_plan),
    )

    exit_code = await runner.run(args)

    assert exit_code == 4
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "wing_readonly_preflight_blocked"
    assert summary["physical_readback_count"] == 0
    assert summary["blockers"] == ["physical_readback_missing:ch3:fader"]
    assert summary["write_request_sent"] is False
    assert summary["proposal_request_sent"] is False


@pytest.mark.asyncio
async def test_readonly_preflight_blocks_readback_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [
            {"type": "operator_mode_status", "mode": "assist", "connection_mode": "wing"},
            {"type": "state_update", "mode": "wing", "state": {"/ch/3/fdr": -12.0}},
            {
                "type": "mixer_parameter_value",
                "request_id": "wing-readonly-preflight:ch3:fader",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "channel": 4,
                "value_type": "fader",
                "address": "/ch/4/fdr",
                "value": -12.0,
            },
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)
    expected_plan = write_expected_plan(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
    )
    args = parse_args(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
        "--expected-plan",
        str(expected_plan),
    )

    exit_code = await runner.run(args)

    assert exit_code == 4
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "wing_readonly_preflight_blocked"
    assert summary["physical_readback_count"] == 1
    assert summary["blockers"] == ["readback_identity_mismatch:ch3:fader"]
    assert summary["write_request_sent"] is False
    assert summary["proposal_request_sent"] is False


@pytest.mark.asyncio
async def test_readonly_preflight_blocks_readback_request_id_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [
            {"type": "operator_mode_status", "mode": "assist", "connection_mode": "wing"},
            {"type": "state_update", "mode": "wing", "state": {"/ch/3/fdr": -12.0}},
            {
                "type": "mixer_parameter_value",
                "request_id": "stale:ch3:fader",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "channel": 3,
                "value_type": "fader",
                "address": "/ch/3/fdr",
                "value": -12.0,
            },
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)
    expected_plan = write_expected_plan(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
    )
    args = parse_args(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
        "--expected-plan",
        str(expected_plan),
    )

    exit_code = await runner.run(args)

    assert exit_code == 4
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "wing_readonly_preflight_blocked"
    assert summary["blockers"] == [
        "readback_request_id_mismatch:ch3:fader",
        "readback_identity_mismatch:ch3:fader",
    ]


@pytest.mark.asyncio
async def test_readonly_preflight_requires_expected_plan_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(f"missing expected-plan must not open WebSocket: {url}")

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    args = parse_args(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "expected_plan_missing"
    assert summary["expected_plan_required_for_real_run"] is True
    assert_summary_prestart_contract(summary)
    assert summary["expected_plan_match"] is False
    assert summary["expected_plan_violations"] == ["expected_plan_missing"]
    assert "--plan-only" in summary["operator_next_step"]
    assert "--expected-plan" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_reports_invalid_expected_plan_next_step_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(f"invalid expected-plan must not open WebSocket: {url}")

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    expected_plan = tmp_path / "invalid-expected-plan.json"
    expected_plan.write_text("{not json", encoding="utf-8")
    args = parse_args(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
        "--expected-plan",
        str(expected_plan),
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "expected_plan_missing_or_invalid"
    assert summary["expected_plan_match"] is False
    assert summary["expected_plan_error"]["type"] == "JSONDecodeError"
    assert "Regenerate the offline expected plan" in summary["operator_next_step"]
    assert "--expected-plan" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_blocks_expected_plan_mismatch_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(f"expected-plan mismatch must not open WebSocket: {url}")

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    expected_plan = write_expected_plan(
        tmp_path,
        "--channels",
        "4",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "4",
        "--operator-confirm-session-ready",
    )
    args = parse_args(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
        "--expected-plan",
        str(expected_plan),
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "expected_plan_mismatch"
    assert summary["expected_plan_match"] is False
    assert "expected_plan_channels_mismatch" in summary["expected_plan_violations"]
    assert "expected_plan_request_sequence_mismatch" in summary["expected_plan_violations"]
    assert "current runtime arguments" in summary["operator_next_step"]
    assert "backend/WING WebSocket" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_blocks_stale_prestart_detail_plan_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(f"stale expected-plan must not open WebSocket: {url}")

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    expected_plan = write_expected_plan(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
    )
    plan_payload = json.loads(expected_plan.read_text(encoding="utf-8"))
    plan_payload["backend_port_prestart_check_failure_detail_fields"] = [
        field
        for field in plan_payload["backend_port_prestart_check_failure_detail_fields"]
        if field != "listener_confirmation_checklist"
    ]
    expected_plan.write_text(
        json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args = parse_args(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
        "--expected-plan",
        str(expected_plan),
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "expected_plan_mismatch"
    assert_summary_prestart_contract(summary)
    assert summary["expected_plan_match"] is False
    assert "expected_plan_backend_port_prestart_check_failure_detail_fields_mismatch" in (
        summary["expected_plan_violations"]
    )
    assert "prestart contract" in summary["operator_next_step"]
    assert "backend/WING WebSocket" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_plan_only_outputs_readonly_request_plan_without_websocket_or_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(f"plan-only must not open WebSocket: {url}")

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    plan_output = tmp_path / "readonly-preflight-plan.json"
    args = parse_args(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader,gain,pan",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
        "--plan-only",
        "--plan-output",
        str(plan_output),
    )

    exit_code = await runner.run(args)

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(plan_output.read_text(encoding="utf-8"))
    assert printed["type"] == "wing_readonly_preflight_offline_plan"
    assert printed["status"] == "ready_for_operator_review"
    assert printed["proof_artifact_schema"] == WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA
    assert printed["plan_output"] == str(plan_output)
    assert written["plan_output"] == str(plan_output)
    assert written["selected_mvp_blocker"] == "WING read-only connection proof"
    assert written["selected_mvp_blocker_status"] == "tooling_ready_waits_real_wing"
    assert written["contacts_backend"] is False
    assert written["queries_console"] is False
    assert written["execution_allowed_by_plan"] is False
    assert written["live_authority_granted"] is False
    assert written["operator_required"] is True
    assert written["expected_plan_required_for_real_run"] is True
    assert written["physical_preflight_requires_existing_backend_connection"] is True
    assert written["backend_port_prestart_check_required"] is True
    assert written["backend_port_prestart_check_must_pass_before_starting_backend"] is True
    assert written["backend_port_prestart_check_command"] == list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
    )
    assert written["backend_port_prestart_check_failure_detail_fields"] == list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    )
    assert written["backend_port_prestart_summary_required"] is True
    assert written["backend_port_prestart_summary_path"] == str(
        prestart_summary_path(tmp_path)
    )
    assert written["backend_port_prestart_summary_sha256"] == sha256_file(
        prestart_summary_path(tmp_path)
    )
    assert written["backend_port_prestart_summary_type"] == "no_wing_release_check"
    assert written["backend_port_prestart_summary_status"] == "passed"
    assert (
        written["backend_port_prestart_summary_connection_policy"]
        == "no_hardware_no_network_no_live_writes"
    )
    assert written["backend_port_prestart_summary_live_write"] is False
    assert written["backend_port_prestart_summary_contacts_backend"] is False
    assert written["backend_port_prestart_summary_queries_console"] is False
    assert written["backend_port_prestart_summary_executes_tests"] is False
    assert written["backend_port_prestart_check_status"] == "passed"
    assert written["manual_review_required_before_readiness_update"] is True
    assert written["request_id_policy"] == "required_for_every_query_mixer_parameter_response"
    assert written["proof_artifact_schema"] == WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA
    assert written["expected_artifacts"] == [
        f"{tmp_path}/<run_id>/{WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME}",
        f"{tmp_path}/<run_id>/{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME}",
    ]
    assert written["read_only"] is True
    assert written["live_write"] is False
    assert written["channels"] == [3]
    assert written["operator_confirmed_channels"] == [3]
    assert written["operator_session_confirmation_required"] is True
    assert written["operator_confirmed_session_ready"] is True
    assert [request["type"] for request in written["request_sequence"]] == [
        "get_operator_mode",
        "get_state",
        "query_mixer_parameter",
        "query_mixer_parameter",
        "query_mixer_parameter",
    ]
    assert written["request_sequence"][2]["request_id"] == "wing-readonly-preflight:ch3:fader"
    assert written["request_sequence"][2]["address"] == "/ch/3/fdr"
    assert written["request_sequence"][3]["request_id"] == "wing-readonly-preflight:ch3:gain"
    assert written["request_sequence"][3]["address"] == "/ch/3/in/set/trim"
    assert written["request_sequence"][4]["request_id"] == "wing-readonly-preflight:ch3:pan"
    assert written["request_sequence"][4]["address"] == "/ch/3/pan"
    assert "set_gain" in written["forbidden_request_types"]
    assert written["post_run_review_commands"] == [
        (
            "python3 tools/review_wing_readonly_preflight_artifacts.py "
            f"--artifact-root {tmp_path} "
            "--output Docs/reports/WING_READONLY_PREFLIGHT_REVIEW_LATEST.md "
            "--decision-template-output "
            f"{WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT}"
        ),
        (
            "python3 tools/review_wing_readonly_preflight_decisions.py "
            f"--artifact-root {tmp_path} "
            f"--decision-root {WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT} "
            "--output Docs/reports/WING_READONLY_PREFLIGHT_DECISION_REVIEW_LATEST.md"
        ),
    ]
    assert "physical_readback_missing" in written["fail_closed_conditions"]
    assert written["fail_closed_conditions"] == list(
        WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS
    )
    assert "event_log_wing_mode_not_proven" in written["fail_closed_conditions"]
    assert "transport_timeout_or_connection_loss" in written["fail_closed_conditions"]
    assert "ws_handshake_rejected" in written["fail_closed_conditions"]
    assert "operator_session_confirmation_missing" in written["fail_closed_conditions"]
    assert "readback_request_id_mismatch" in written["fail_closed_conditions"]
    assert "expected_plan_missing_or_mismatch" in written["fail_closed_conditions"]
    assert "prestart_summary_missing_or_invalid" in written["fail_closed_conditions"]
    assert "prestart_summary_sha256_mismatch" in written["fail_closed_conditions"]
    assert (
        "stale_backend_port_prestart_detail_contract"
        in written["fail_closed_conditions"]
    )
    assert (
        "existing_backend_connection_boundary_missing"
        in written["fail_closed_conditions"]
    )
    assert "unexpected_response_type" in written["fail_closed_conditions"]
    assert "manual_review_decision_missing_or_invalid" in written["fail_closed_conditions"]
    assert not list(tmp_path.glob(f"*/{WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME}"))


@pytest.mark.asyncio
async def test_readonly_preflight_blocks_missing_prestart_summary_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(f"missing prestart summary must not open WebSocket: {url}")

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    args = runner.build_parser().parse_args(
        [
            "--url",
            "ws://fake-backend",
            "--artifact-root",
            str(tmp_path),
            "--timeout",
            "0.2",
            "--channels",
            "3",
            "--value-types",
            "fader",
            "--require-physical-readback",
            "--operator-confirm-channels",
            "3",
            "--operator-confirm-session-ready",
            "--expected-plan",
            str(tmp_path / "expected-plan.json"),
        ]
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "prestart_summary_missing_or_invalid"
    assert summary["backend_port_prestart_summary_required"] is True
    assert summary["prestart_summary_violations"] == ["prestart_summary_missing"]
    assert "tools/no_wing_release_check.py" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_plan_only_blocks_invalid_channel_confirmation_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(f"plan-only must not open WebSocket: {url}")

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    plan_output = tmp_path / "blocked-plan.json"
    args = parse_args(
        tmp_path,
        "--channels",
        "3",
        "--value-types",
        "fader,gain,pan",
        "--require-physical-readback",
        "--operator-confirm-channels",
        "4",
        "--operator-confirm-session-ready",
        "--plan-only",
        "--plan-output",
        str(plan_output),
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed["type"] == "wing_readonly_preflight_offline_plan"
    assert printed["status"] == "blocked"
    assert printed["reason"] == "--operator-confirm-channels must exactly match --channels"
    assert printed["selected_mvp_blocker"] == "WING read-only connection proof"
    assert printed["operator_required"] is True
    assert printed["contacts_backend"] is False
    assert printed["queries_console"] is False
    assert printed["manual_review_required_before_readiness_update"] is True
    assert printed["live_authority_granted"] is False
    assert not plan_output.exists()
