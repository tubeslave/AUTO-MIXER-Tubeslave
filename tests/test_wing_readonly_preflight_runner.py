"""Offline tests for the read-only WING preflight runner."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from backend.wing_readonly_preflight_contract import (
    WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS,
    WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
    WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME,
    WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_SOURCES,
    WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_STATUSES,
    WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS,
    WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA,
    WING_READONLY_PREFLIGHT_PROOF_RUNNER,
    WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE,
    WING_READONLY_PREFLIGHT_OPERATOR_MODE_RESPONSE_TYPES,
    WING_READONLY_PREFLIGHT_READBACK_RESPONSE_TYPES,
    WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY,
    WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON,
    WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS,
    WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
    WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES,
    WING_READONLY_PREFLIGHT_RUNNER_URL,
    WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
    WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER,
    WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT,
    WING_READONLY_PREFLIGHT_STATE_RESPONSE_TYPES,
    WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME,
)
from tools.no_wing_release_check import (
    CheckResult,
    NO_WING_RELEASE_CHECK_ARTIFACT_MANIFEST_NAME,
    write_no_wing_release_check_artifact_manifest,
    write_summary as write_no_wing_release_summary,
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


def test_runner_uses_shared_request_id_policy_contract() -> None:
    assert runner.REQUEST_ID_POLICY == WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY


def test_runner_uses_shared_allowed_request_types_contract() -> None:
    assert runner.ALLOWED_REQUEST_TYPES == WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES


def test_runner_uses_shared_response_type_contract() -> None:
    contract_mapping = WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE
    assert runner.EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE == {
        request_type: set(response_types)
        for request_type, response_types in contract_mapping.items()
    }
    assert runner.EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE_SUMMARY == {
        request_type: list(response_types)
        for request_type, response_types in contract_mapping.items()
    }
    assert runner.EXPECTED_MODE_EVENTS == set(
        WING_READONLY_PREFLIGHT_OPERATOR_MODE_RESPONSE_TYPES
    )
    assert runner.EXPECTED_STATE_EVENTS == set(
        WING_READONLY_PREFLIGHT_STATE_RESPONSE_TYPES
    )
    assert runner.EXPECTED_PARAMETER_EVENTS == set(
        WING_READONLY_PREFLIGHT_READBACK_RESPONSE_TYPES
    )


def test_runner_uses_shared_run_summary_contract() -> None:
    assert runner.PROOF_RUNNER == WING_READONLY_PREFLIGHT_PROOF_RUNNER
    assert runner.SELECTED_MVP_BLOCKER == WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER
    assert (
        runner.SELECTED_MVP_BLOCKER_STATUS
        == WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS
    )
    assert runner.RUN_SUMMARY_STATUS == WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS
    assert runner.RUN_SUMMARY_REASON == WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON


def test_runner_requires_manifest_verifier_surface_value_checks_before_websocket() -> None:
    required_surface_value_checks = {
        f"{surface}_artifact_manifest_verifier_{field}_matches_expected"
        for surface in ("summary_json", "summary_md", "manifest")
        for field in (
            "read_only",
            "live_write",
            "contacts_backend",
            "executes_tests",
            "contacts_wing",
            "mutates_console_state",
            "queries_console",
            "readiness_closure_allowed",
            "live_authority_granted",
        )
    }

    assert required_surface_value_checks <= set(
        runner.ARTIFACT_MANIFEST_VERIFIER_REQUIRED_CHECKS
    )


def prestart_summary_path(tmp_path: Path) -> Path:
    return tmp_path / WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME


def ensure_prestart_summary(tmp_path: Path) -> Path:
    path = prestart_summary_path(tmp_path)
    if path.exists():
        return path
    generated_at = "2026-06-15T00:00:00Z"
    write_no_wing_release_summary(
        tmp_path,
        [
            CheckResult(
                name="backend_port_free",
                status="passed",
                started_at="2026-06-15T00:00:00Z",
                finished_at="2026-06-15T00:00:01Z",
                duration_sec=1.0,
                command="PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick",
                cwd=str(Path.cwd()),
                returncode=0,
                detail={
                    "host": "127.0.0.1",
                    "port": 8765,
                    "backend_port_prestart_check_failure_detail_fields": list(
                        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
                    ),
                    "safe_to_stop_listener_by_gate": False,
                    "live_authority_granted": False,
                    "backend_port_prestart_gate_execution_allowed": False,
                    "backend_port_prestart_gate_readiness_closure_allowed": False,
                    "backend_port_prestart_gate_live_authority_granted": False,
                },
            )
        ],
        generated_at=generated_at,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "read_only": True,
            "contacts_wing": False,
            "contacts_backend": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "readiness_closure_allowed_by_packet": False,
            "live_authority_granted": False,
        }
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_no_wing_release_check_artifact_manifest(tmp_path, generated_at=generated_at)
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
    assert args.validate_run_inputs_only is False
    assert runner.SUPPORTED_VALUE_TYPES == set(
        WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES
    )
    assert runner.PHYSICAL_READBACK_STATUSES == set(
        WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_STATUSES
    )
    assert runner.PHYSICAL_READBACK_SOURCES == set(
        WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_SOURCES
    )


def test_validate_args_rejects_plan_only_with_validate_run_inputs_only(
    tmp_path: Path,
) -> None:
    args = parse_args(
        tmp_path,
        "--plan-only",
        "--validate-run-inputs-only",
    )

    assert runner.validate_args(args) == (
        "--plan-only and --validate-run-inputs-only cannot be combined"
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
    assert (
        summary["backend_port_prestart_check_command_text"]
        == WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT
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
    assert summary["backend_port_prestart_summary_read_only"] is True
    assert summary["backend_port_prestart_summary_live_write"] is False
    assert summary["backend_port_prestart_summary_contacts_wing"] is False
    assert summary["backend_port_prestart_summary_contacts_backend"] is False
    assert summary["backend_port_prestart_summary_mutates_console_state"] is False
    assert summary["backend_port_prestart_summary_queries_console"] is False
    assert summary["backend_port_prestart_summary_executes_tests"] is False
    assert summary["artifact_manifest_verifier_required"] is True
    assert (
        summary["artifact_manifest_verifier_run_dir"]
        == str(Path(summary["backend_port_prestart_summary_path"]).parent)
    )
    assert summary["artifact_manifest_verifier_status"] == "passed"
    assert summary["artifact_manifest_verifier_failed_checks"] == []
    assert summary["artifact_manifest_verifier_failed_checks_missing_check_entries"] == []
    assert summary["artifact_manifest_verifier_required_fields_missing"] == []
    assert summary["artifact_manifest_verifier_failed_checks_have_check_entries"] is True
    assert summary["artifact_manifest_verifier_report_required_fields_present"] is True
    assert summary["artifact_manifest_verifier_report_required_fields_match_contract"] is True
    assert summary["artifact_manifest_verifier_readiness_closure_allowed"] is False
    assert summary["artifact_manifest_verifier_live_authority_granted"] is False
    assert summary["artifact_manifest_verifier_report_value_mismatches"] == []
    assert summary["artifact_manifest_verifier_check_mismatches"] == []
    for check_name in runner.ARTIFACT_MANIFEST_VERIFIER_REQUIRED_CHECKS:
        assert summary[f"artifact_manifest_verifier_check_{check_name}"] is True
    assert (
        summary[
            "artifact_manifest_verifier_report_artifact_manifest_self_consistency_match"
        ]
        is True
    )
    assert (
        summary[
            "artifact_manifest_verifier_report_artifact_manifest_contract_cross_surface_match"
        ]
        is True
    )
    assert (
        summary[
            "artifact_manifest_verifier_report_artifact_manifest_self_consistency_summary_surface_match"
        ]
        is True
    )
    assert summary["artifact_manifest_verifier_report_read_only"] is True
    assert summary["artifact_manifest_verifier_report_live_write"] is False
    assert summary["artifact_manifest_verifier_report_contacts_wing"] is False
    assert summary["artifact_manifest_verifier_report_contacts_backend"] is False
    assert summary["artifact_manifest_verifier_report_mutates_console_state"] is False
    assert summary["artifact_manifest_verifier_report_queries_console"] is False
    assert summary["artifact_manifest_verifier_report_executes_tests"] is False
    assert summary["artifact_manifest_verifier_must_pass_before_websocket"] is True
    assert summary["artifact_manifest_verifier_passed_before_websocket"] is True
    assert summary["backend_port_prestart_check_status"] == "passed"
    for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS:
        assert summary[field] is False


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
    assert expected & set(WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES) == set()


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
    assert summary["artifact_root"] == str(tmp_path)
    assert summary["expected_plan_path"] == str(expected_plan)
    assert summary["expected_plan_match"] is True
    assert summary["expected_plan_sha256"]
    assert summary["expected_plan_violations"] == []
    assert_summary_prestart_contract(summary)
    assert summary["request_id_policy"] == WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY
    assert summary["read_only"] is True
    assert summary["live_write"] is False
    assert summary["write_request_sent"] is False
    assert summary["proposal_request_sent"] is False
    assert summary["physical_readback_count"] == 3
    assert summary["forbidden_request_types_sent"] == []
    events = read_events(summary)
    assert [event["run_id"] for event in events] == [summary["run_id"]] * len(events)
    assert [event["event_type"] for event in events] == [
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
async def test_readonly_preflight_verifies_manifest_before_opening_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
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
        ]
    )
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
    verifier_report = runner.verify_no_wing_artifact_manifest(tmp_path)

    def verify_before_websocket(run_dir: Path) -> dict:
        order.append("verifier")
        assert run_dir == tmp_path
        return verifier_report

    def connect_after_verifier(url: str) -> FakeWebSocketConnect:
        order.append("connect")
        websocket.url = url
        return FakeWebSocketConnect(websocket)

    monkeypatch.setattr(
        runner,
        "verify_no_wing_artifact_manifest",
        verify_before_websocket,
    )
    monkeypatch.setattr(runner.websockets, "connect", connect_after_verifier)
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

    assert exit_code == 0
    assert order.count("verifier") >= 1
    assert order[-1] == "connect"
    assert "connect" not in order[:-1]
    assert websocket.url == "ws://fake-backend"
    summary = read_summary(tmp_path)
    assert summary["websocket_opened"] is True
    assert summary["artifact_manifest_verifier_websocket_opened_after_pass"] is True
    assert summary["artifact_manifest_verifier_status"] == "passed"
    assert summary["artifact_manifest_verifier_failed_checks"] == []


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
    assert summary["artifact_root"] == str(tmp_path)
    assert summary["expected_plan_path"] == str(expected_plan)
    assert summary["expected_plan_match"] is True
    assert summary["expected_plan_sha256"]
    assert summary["expected_plan_violations"] == []
    assert_summary_prestart_contract(summary)
    assert summary["request_id_policy"] == WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY
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
    assert summary["contacts_backend"] is True
    assert summary["queries_console"] is False
    assert "WING mode" in summary["operator_next_step"]
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
    assert summary["contacts_backend"] is True
    assert summary["queries_console"] is False
    assert "response contract" in summary["operator_next_step"]
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
    assert summary["contacts_backend"] is True
    assert summary["queries_console"] is False
    assert "backend WebSocket responsiveness" in summary["operator_next_step"]
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
    assert summary["contacts_backend"] is False
    assert summary["queries_console"] is False
    assert summary["transport_context"]["preflight_requests_sent"] is False
    assert "backend WebSocket reachability" in summary["operator_next_step"]
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
    assert summary["contacts_backend"] is False
    assert summary["queries_console"] is False
    assert summary["operator_next_step"] == summary["transport_context"][
        "operator_next_step"
    ]
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
    assert summary["contacts_backend"] is True
    assert summary["queries_console"] is True
    assert "read-only preflight blockers" in summary["operator_next_step"]
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
async def test_validate_run_inputs_only_requires_expected_plan_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(
            f"validate-run-inputs-only missing expected-plan must not open WebSocket: {url}"
        )

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
        "--validate-run-inputs-only",
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "expected_plan_missing"
    assert summary["validation_only"] is True
    assert summary["websocket_opened"] is False
    assert summary["selected_mvp_blocker"] == WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER
    assert (
        summary["selected_mvp_blocker_status"]
        == WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS
    )
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_validate_run_inputs_only_passes_ready_command_without_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(
            f"validate-run-inputs-only must not open WebSocket: {url}"
        )

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
        "--validate-run-inputs-only",
    )

    exit_code = await runner.run(args)

    assert exit_code == 0
    summary = read_summary(tmp_path)
    assert summary["status"] == "ready_for_websocket"
    assert summary["reason"] == "run_inputs_validated_no_websocket"
    assert summary["selected_mvp_blocker"] == WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER
    assert (
        summary["selected_mvp_blocker_status"]
        == WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS
    )
    assert summary["validation_only"] is True
    assert summary["websocket_opened"] is False
    assert summary["expected_plan_match"] is True
    assert summary["expected_plan_violations"] == []
    assert summary["contacts_backend"] is False
    assert summary["queries_console"] is False
    assert summary["execution_allowed_by_validation"] is False
    assert summary["live_authority_granted"] is False
    assert "--validate-run-inputs-only" in summary["operator_next_step"]
    assert "real read-only preflight artifact" in summary["operator_next_step"]
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
    plan_payload["artifact_manifest_verifier_passed_before_websocket"] = False
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
    assert (
        "expected_plan_artifact_manifest_verifier_passed_before_websocket_mismatch"
        in summary["expected_plan_violations"]
    )
    assert "prestart contract" in summary["operator_next_step"]
    assert "backend/WING WebSocket" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_blocks_stale_verifier_check_plan_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(f"stale verifier-check plan must not open WebSocket: {url}")

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
    plan_payload[
        "artifact_manifest_verifier_check_summary_json_sha256_matches_manifest"
    ] = False
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
    assert (
        "expected_plan_artifact_manifest_verifier_check_summary_json_sha256_matches_manifest_mismatch"
        in summary["expected_plan_violations"]
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
    assert printed["proof_runner"] == WING_READONLY_PREFLIGHT_PROOF_RUNNER
    assert printed["plan_output"] == str(plan_output)
    assert written["plan_output"] == str(plan_output)
    assert written["selected_mvp_blocker"] == WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER
    assert (
        written["selected_mvp_blocker_status"]
        == WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS
    )
    assert written["contacts_backend"] is False
    assert written["queries_console"] is False
    assert written["execution_allowed_by_plan"] is False
    assert written["live_authority_granted"] is False
    assert written["operator_required"] is True
    assert written["expected_plan_required_for_real_run"] is True
    assert written["physical_preflight_requires_existing_backend_connection"] is True
    assert written["backend_port_prestart_check_required"] is True
    assert written["backend_port_prestart_check_must_pass_before_starting_backend"] is True
    assert written["artifact_manifest_verifier_must_pass_before_websocket"] is True
    assert written["artifact_manifest_verifier_passed_before_websocket"] is True
    assert written["artifact_manifest_verifier_websocket_opened_after_pass"] is False
    assert written["backend_port_prestart_check_command"] == list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
    )
    assert (
        written["backend_port_prestart_check_command_text"]
        == WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT
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
    assert written["backend_port_prestart_summary_read_only"] is True
    assert written["backend_port_prestart_summary_live_write"] is False
    assert written["backend_port_prestart_summary_contacts_wing"] is False
    assert written["backend_port_prestart_summary_contacts_backend"] is False
    assert written["backend_port_prestart_summary_mutates_console_state"] is False
    assert written["backend_port_prestart_summary_queries_console"] is False
    assert written["backend_port_prestart_summary_executes_tests"] is False
    assert written["artifact_manifest_verifier_report_required_fields_match_contract"] is True
    assert written["artifact_manifest_verifier_report_value_mismatches"] == []
    assert written["artifact_manifest_verifier_check_mismatches"] == []
    for check_name in runner.ARTIFACT_MANIFEST_VERIFIER_REQUIRED_CHECKS:
        assert written[f"artifact_manifest_verifier_check_{check_name}"] is True
    assert (
        written[
            "artifact_manifest_verifier_report_artifact_manifest_self_consistency_match"
        ]
        is True
    )
    assert (
        written[
            "artifact_manifest_verifier_report_artifact_manifest_contract_cross_surface_match"
        ]
        is True
    )
    assert (
        written[
            "artifact_manifest_verifier_report_artifact_manifest_self_consistency_summary_surface_match"
        ]
        is True
    )
    assert written["artifact_manifest_verifier_report_read_only"] is True
    assert written["artifact_manifest_verifier_report_live_write"] is False
    assert written["artifact_manifest_verifier_report_contacts_wing"] is False
    assert written["artifact_manifest_verifier_report_contacts_backend"] is False
    assert written["artifact_manifest_verifier_report_mutates_console_state"] is False
    assert written["artifact_manifest_verifier_report_queries_console"] is False
    assert written["artifact_manifest_verifier_report_executes_tests"] is False
    assert written["backend_port_prestart_check_status"] == "passed"
    for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS:
        assert written[field] is False
    assert written["manual_review_required_before_readiness_update"] is True
    assert written["request_id_policy"] == WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY
    assert written["expected_response_types_by_request_type"] == {
        request_type: list(response_types)
        for request_type, response_types in (
            WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE.items()
        )
    }
    assert written["proof_artifact_schema"] == WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA
    assert written["proof_runner"] == WING_READONLY_PREFLIGHT_PROOF_RUNNER
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
    assert "event_log_run_id_mismatch" in written["fail_closed_conditions"]
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
async def test_readonly_preflight_blocks_environment_blocked_prestart_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(
            f"environment-blocked prestart summary must not open WebSocket: {url}"
        )

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    prestart_path = ensure_prestart_summary(tmp_path)
    payload = json.loads(prestart_path.read_text(encoding="utf-8"))
    payload["status"] = "failed"
    payload["failed_checks"] = ["frontend_e2e"]
    payload["environment_blocked_checks"] = ["frontend_e2e"]
    payload["environment_blocked_check_details"] = [
        {
            "name": "frontend_e2e",
            "reason": "local_dev_server_bind_denied",
            "address": "127.0.0.1:3000",
            "operator_next_step": (
                "Rerun the no-WING quick check in an environment that can bind "
                "the frontend dev server; this does not grant readiness."
            ),
            "stdout_log": str(tmp_path / "frontend-stdout.log"),
            "stderr_log": str(tmp_path / "frontend-stderr.log"),
        }
    ]
    prestart_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
            "--prestart-summary",
            str(prestart_path),
            "--expected-plan",
            str(tmp_path / "expected-plan.json"),
        ]
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "prestart_summary_missing_or_invalid"
    assert summary["backend_port_prestart_summary_status"] == "failed"
    assert summary["environment_blocked_checks"] == ["frontend_e2e"]
    assert summary["environment_blocked_check_details"][0]["reason"] == (
        "local_dev_server_bind_denied"
    )
    assert summary["environment_blocked_check_details"][0]["address"] == (
        "127.0.0.1:3000"
    )
    assert summary["prestart_summary_violations"][:2] == [
        "prestart_summary_environment_blocked",
        "prestart_summary_status_not_passed",
    ]
    assert "127.0.0.1:3000" in summary["operator_next_step"]
    assert "does not grant readiness" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_blocks_stale_prestart_manifest_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(f"stale prestart manifest must not open WebSocket: {url}")

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    prestart_path = ensure_prestart_summary(tmp_path)
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
    manifest_path = tmp_path / NO_WING_RELEASE_CHECK_ARTIFACT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_manifest_verifier_command"] = "stale verifier command"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
            "--prestart-summary",
            str(prestart_path),
            "--expected-plan",
            str(expected_plan),
        ]
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "prestart_summary_missing_or_invalid"
    assert "prestart_summary_artifact_manifest_verifier_failed" in summary[
        "prestart_summary_violations"
    ]
    assert summary["artifact_manifest_verifier_status"] == "failed"
    assert (
        "manifest_artifact_manifest_verifier_command_matches_expected"
        in summary["artifact_manifest_verifier_failed_checks"]
    )
    assert summary["artifact_manifest_verifier_failed_checks_missing_check_entries"] == []
    assert summary["artifact_manifest_verifier_required_fields_missing"] == []
    assert summary["artifact_manifest_verifier_failed_checks_have_check_entries"] is True
    assert summary["artifact_manifest_verifier_report_required_fields_present"] is True
    assert summary["artifact_manifest_verifier_readiness_closure_allowed"] is False
    assert summary["artifact_manifest_verifier_live_authority_granted"] is False
    assert "tools/verify_no_wing_artifact_manifest.py" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_blocks_bad_verifier_report_values_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(f"bad verifier report must not open WebSocket: {url}")

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    prestart_path = ensure_prestart_summary(tmp_path)
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
    verifier_report = runner.verify_no_wing_artifact_manifest(prestart_path.parent)
    verifier_report.update(
        {
            "status": "passed",
            "failed_checks": [],
            "failed_checks_missing_check_entries": [],
            "artifact_manifest_contract_cross_surface_match": False,
        }
    )
    monkeypatch.setattr(
        runner,
        "verify_no_wing_artifact_manifest",
        lambda _run_dir: verifier_report,
    )
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
            "--prestart-summary",
            str(prestart_path),
            "--expected-plan",
            str(expected_plan),
        ]
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "prestart_summary_missing_or_invalid"
    assert "prestart_summary_artifact_manifest_verifier_failed" in summary[
        "prestart_summary_violations"
    ]
    assert (
        "prestart_summary_artifact_manifest_verifier_report_value_mismatch"
        in summary["prestart_summary_violations"]
    )
    assert summary["artifact_manifest_verifier_status"] == "passed"
    assert summary["artifact_manifest_verifier_failed_checks"] == []
    assert summary["artifact_manifest_verifier_report_value_mismatches"] == [
        "artifact_manifest_contract_cross_surface_match"
    ]
    assert (
        summary[
            "artifact_manifest_verifier_report_artifact_manifest_contract_cross_surface_match"
        ]
        is False
    )
    assert "tools/verify_no_wing_artifact_manifest.py" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_reports_verifier_authority_claim_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(
            f"verifier authority claim must not open WebSocket: {url}"
        )

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    prestart_path = ensure_prestart_summary(tmp_path)
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
    verifier_report = runner.verify_no_wing_artifact_manifest(prestart_path.parent)
    verifier_report.update(
        {
            "status": "passed",
            "failed_checks": [],
            "failed_checks_missing_check_entries": [],
            "readiness_closure_allowed": True,
            "live_authority_granted": True,
        }
    )
    monkeypatch.setattr(
        runner,
        "verify_no_wing_artifact_manifest",
        lambda _run_dir: verifier_report,
    )
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
            "--prestart-summary",
            str(prestart_path),
            "--expected-plan",
            str(expected_plan),
        ]
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "prestart_summary_missing_or_invalid"
    assert "prestart_summary_artifact_manifest_verifier_failed" in summary[
        "prestart_summary_violations"
    ]
    assert (
        "prestart_summary_artifact_manifest_verifier_readiness_closure_allowed_not_false"
        in summary["prestart_summary_violations"]
    )
    assert (
        "prestart_summary_artifact_manifest_verifier_live_authority_granted_not_false"
        in summary["prestart_summary_violations"]
    )
    assert summary["artifact_manifest_verifier_readiness_closure_allowed"] is True
    assert summary["artifact_manifest_verifier_live_authority_granted"] is True
    assert summary["artifact_manifest_verifier_report_value_mismatches"] == [
        "readiness_closure_allowed",
        "live_authority_granted",
    ]
    assert "tools/verify_no_wing_artifact_manifest.py" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_reports_verifier_safety_flag_mismatches_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(
            f"verifier safety flag mismatch must not open WebSocket: {url}"
        )

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    prestart_path = ensure_prestart_summary(tmp_path)
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
    verifier_report = runner.verify_no_wing_artifact_manifest(prestart_path.parent)
    verifier_report.update(
        {
            "status": "passed",
            "failed_checks": [],
            "failed_checks_missing_check_entries": [],
            "read_only": False,
            "live_write": True,
            "contacts_wing": True,
            "contacts_backend": True,
            "mutates_console_state": True,
            "queries_console": True,
            "executes_tests": True,
        }
    )
    monkeypatch.setattr(
        runner,
        "verify_no_wing_artifact_manifest",
        lambda _run_dir: verifier_report,
    )
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
            "--prestart-summary",
            str(prestart_path),
            "--expected-plan",
            str(expected_plan),
        ]
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "prestart_summary_missing_or_invalid"
    assert "prestart_summary_artifact_manifest_verifier_failed" in summary[
        "prestart_summary_violations"
    ]
    for safety_field in (
        "read_only",
        "live_write",
        "contacts_wing",
        "contacts_backend",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
    ):
        assert (
            f"prestart_summary_artifact_manifest_verifier_{safety_field}_mismatch"
            in summary["prestart_summary_violations"]
        )
    assert summary["artifact_manifest_verifier_report_value_mismatches"] == [
        "read_only",
        "live_write",
        "contacts_wing",
        "contacts_backend",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
    ]
    assert summary["artifact_manifest_verifier_report_read_only"] is False
    assert summary["artifact_manifest_verifier_report_live_write"] is True
    assert summary["artifact_manifest_verifier_report_contacts_wing"] is True
    assert summary["artifact_manifest_verifier_report_contacts_backend"] is True
    assert summary["artifact_manifest_verifier_report_mutates_console_state"] is True
    assert summary["artifact_manifest_verifier_report_queries_console"] is True
    assert summary["artifact_manifest_verifier_report_executes_tests"] is True
    assert "tools/verify_no_wing_artifact_manifest.py" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_reports_verifier_summary_surface_mismatches_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(
            f"verifier summary-surface mismatch must not open WebSocket: {url}"
        )

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    prestart_path = ensure_prestart_summary(tmp_path)
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
    verifier_report = runner.verify_no_wing_artifact_manifest(prestart_path.parent)
    verifier_report.update(
        {
            "status": "passed",
            "failed_checks": [],
            "failed_checks_missing_check_entries": [],
            "wing_readonly_handoff_plan_verifier_summary_surface_mismatches": [
                "stale_summary_surface"
            ],
            "wing_readonly_handoff_plan_verifier_summary_surface_match": False,
            "wing_readonly_handoff_plan_verifier_summary_surface_readiness_closure_allowed": True,
            "wing_readonly_handoff_plan_verifier_summary_surface_live_authority_granted": True,
        }
    )
    monkeypatch.setattr(
        runner,
        "verify_no_wing_artifact_manifest",
        lambda _run_dir: verifier_report,
    )
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
            "--prestart-summary",
            str(prestart_path),
            "--expected-plan",
            str(expected_plan),
        ]
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "prestart_summary_missing_or_invalid"
    assert "prestart_summary_artifact_manifest_verifier_failed" in summary[
        "prestart_summary_violations"
    ]
    for summary_surface_field in (
        "wing_readonly_handoff_plan_verifier_summary_surface_mismatches",
        "wing_readonly_handoff_plan_verifier_summary_surface_match",
        "wing_readonly_handoff_plan_verifier_summary_surface_readiness_closure_allowed",
        "wing_readonly_handoff_plan_verifier_summary_surface_live_authority_granted",
    ):
        assert (
            "prestart_summary_artifact_manifest_verifier_"
            f"{summary_surface_field}_mismatch"
        ) in summary["prestart_summary_violations"]
    assert summary["artifact_manifest_verifier_report_value_mismatches"] == [
        "wing_readonly_handoff_plan_verifier_summary_surface_mismatches",
        "wing_readonly_handoff_plan_verifier_summary_surface_match",
        "wing_readonly_handoff_plan_verifier_summary_surface_readiness_closure_allowed",
        "wing_readonly_handoff_plan_verifier_summary_surface_live_authority_granted",
    ]
    assert summary[
        "artifact_manifest_verifier_report_wing_readonly_handoff_plan_verifier_summary_surface_mismatches"
    ] == ["stale_summary_surface"]
    assert (
        summary[
            "artifact_manifest_verifier_report_wing_readonly_handoff_plan_verifier_summary_surface_match"
        ]
        is False
    )
    assert (
        summary[
            "artifact_manifest_verifier_report_wing_readonly_handoff_plan_verifier_summary_surface_readiness_closure_allowed"
        ]
        is True
    )
    assert (
        summary[
            "artifact_manifest_verifier_report_wing_readonly_handoff_plan_verifier_summary_surface_live_authority_granted"
        ]
        is True
    )
    assert "tools/verify_no_wing_artifact_manifest.py" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_reports_malformed_verifier_list_fields_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(
            f"malformed verifier list fields must not open WebSocket: {url}"
        )

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    prestart_path = ensure_prestart_summary(tmp_path)
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
    verifier_report = runner.verify_no_wing_artifact_manifest(prestart_path.parent)
    verifier_report.update(
        {
            "status": "passed",
            "failed_checks": "not-a-list",
            "failed_checks_missing_check_entries": "not-a-list",
            "verifier_report_required_fields_missing": "not-a-list",
        }
    )
    monkeypatch.setattr(
        runner,
        "verify_no_wing_artifact_manifest",
        lambda _run_dir: verifier_report,
    )
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
            "--prestart-summary",
            str(prestart_path),
            "--expected-plan",
            str(expected_plan),
        ]
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "prestart_summary_missing_or_invalid"
    assert "prestart_summary_artifact_manifest_verifier_failed" in summary[
        "prestart_summary_violations"
    ]
    assert (
        "prestart_summary_artifact_manifest_verifier_failed_checks_not_list"
        in summary["prestart_summary_violations"]
    )
    assert (
        "prestart_summary_artifact_manifest_verifier_failed_checks_missing_check_entries_not_list"
        in summary["prestart_summary_violations"]
    )
    assert (
        "prestart_summary_artifact_manifest_verifier_required_fields_missing_not_list"
        in summary["prestart_summary_violations"]
    )
    assert summary["artifact_manifest_verifier_failed_checks"] == []
    assert summary["artifact_manifest_verifier_failed_checks_missing_check_entries"] == []
    assert summary["artifact_manifest_verifier_required_fields_missing"] == []
    assert "tools/verify_no_wing_artifact_manifest.py" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_reports_malformed_verifier_checks_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(
            f"malformed verifier checks must not open WebSocket: {url}"
        )

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    prestart_path = ensure_prestart_summary(tmp_path)
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
    verifier_report = runner.verify_no_wing_artifact_manifest(prestart_path.parent)
    verifier_report.update(
        {
            "status": "passed",
            "failed_checks": [],
            "failed_checks_missing_check_entries": [],
            "checks": "not-an-object",
        }
    )
    monkeypatch.setattr(
        runner,
        "verify_no_wing_artifact_manifest",
        lambda _run_dir: verifier_report,
    )
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
            "--prestart-summary",
            str(prestart_path),
            "--expected-plan",
            str(expected_plan),
        ]
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "prestart_summary_missing_or_invalid"
    assert "prestart_summary_artifact_manifest_verifier_failed" in summary[
        "prestart_summary_violations"
    ]
    assert (
        "prestart_summary_artifact_manifest_verifier_checks_not_object"
        in summary["prestart_summary_violations"]
    )
    assert summary["artifact_manifest_verifier_failed_checks"] == []
    assert summary["artifact_manifest_verifier_failed_checks_have_check_entries"] is False
    assert summary["artifact_manifest_verifier_report_required_fields_present"] is False
    assert summary["artifact_manifest_verifier_check_mismatches"] == list(
        runner.ARTIFACT_MANIFEST_VERIFIER_REQUIRED_CHECKS
    )
    assert "tools/verify_no_wing_artifact_manifest.py" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_reports_missing_verifier_required_fields_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(
            f"missing verifier report field must not open WebSocket: {url}"
        )

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    prestart_path = ensure_prestart_summary(tmp_path)
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
    verifier_report = dict(
        runner.verify_no_wing_artifact_manifest(prestart_path.parent)
    )
    verifier_report.update(
        {
            "status": "passed",
            "failed_checks": [],
            "failed_checks_missing_check_entries": [],
            "verifier_report_required_fields_missing": [
                "future_required_report_field"
            ],
        }
    )
    monkeypatch.setattr(
        runner,
        "verify_no_wing_artifact_manifest",
        lambda _run_dir: verifier_report,
    )
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
            "--prestart-summary",
            str(prestart_path),
            "--expected-plan",
            str(expected_plan),
        ]
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "prestart_summary_missing_or_invalid"
    assert "prestart_summary_artifact_manifest_verifier_failed" in summary[
        "prestart_summary_violations"
    ]
    assert (
        "prestart_summary_artifact_manifest_verifier_required_field_missing"
        in summary["prestart_summary_violations"]
    )
    assert summary["artifact_manifest_verifier_status"] == "passed"
    assert summary["artifact_manifest_verifier_failed_checks"] == []
    assert summary["artifact_manifest_verifier_required_fields_missing"] == [
        "future_required_report_field"
    ]
    assert summary["artifact_manifest_verifier_check_mismatches"] == []
    assert "tools/verify_no_wing_artifact_manifest.py" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_reports_false_verifier_required_fields_present_flag_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(
            f"false verifier required-fields flag must not open WebSocket: {url}"
        )

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    prestart_path = ensure_prestart_summary(tmp_path)
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
    verifier_report = dict(
        runner.verify_no_wing_artifact_manifest(prestart_path.parent)
    )
    verifier_checks = dict(verifier_report["checks"])
    verifier_checks["verifier_report_required_fields_present"] = False
    verifier_report.update(
        {
            "status": "passed",
            "failed_checks": [],
            "failed_checks_missing_check_entries": [],
            "verifier_report_required_fields_missing": [],
            "checks": verifier_checks,
        }
    )
    monkeypatch.setattr(
        runner,
        "verify_no_wing_artifact_manifest",
        lambda _run_dir: verifier_report,
    )
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
            "--prestart-summary",
            str(prestart_path),
            "--expected-plan",
            str(expected_plan),
        ]
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "prestart_summary_missing_or_invalid"
    assert "prestart_summary_artifact_manifest_verifier_failed" in summary[
        "prestart_summary_violations"
    ]
    assert (
        "prestart_summary_artifact_manifest_verifier_required_fields_present_flag_not_true"
        in summary["prestart_summary_violations"]
    )
    assert summary["artifact_manifest_verifier_status"] == "passed"
    assert summary["artifact_manifest_verifier_failed_checks"] == []
    assert summary["artifact_manifest_verifier_required_fields_missing"] == []
    assert summary["artifact_manifest_verifier_report_required_fields_present"] is False
    assert "tools/verify_no_wing_artifact_manifest.py" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_reports_false_verifier_required_fields_contract_flag_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(
            f"false verifier required-fields contract flag must not open WebSocket: {url}"
        )

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    prestart_path = ensure_prestart_summary(tmp_path)
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
    verifier_report = dict(
        runner.verify_no_wing_artifact_manifest(prestart_path.parent)
    )
    verifier_report.update(
        {
            "status": "passed",
            "failed_checks": [],
            "failed_checks_missing_check_entries": [],
            "verifier_report_required_fields_missing": [],
            "verifier_report_required_fields_match_contract": False,
        }
    )
    monkeypatch.setattr(
        runner,
        "verify_no_wing_artifact_manifest",
        lambda _run_dir: verifier_report,
    )
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
            "--prestart-summary",
            str(prestart_path),
            "--expected-plan",
            str(expected_plan),
        ]
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "prestart_summary_missing_or_invalid"
    assert "prestart_summary_artifact_manifest_verifier_failed" in summary[
        "prestart_summary_violations"
    ]
    assert (
        "prestart_summary_artifact_manifest_verifier_required_fields_contract_flag_not_true"
        in summary["prestart_summary_violations"]
    )
    assert summary["artifact_manifest_verifier_status"] == "passed"
    assert summary["artifact_manifest_verifier_failed_checks"] == []
    assert summary["artifact_manifest_verifier_required_fields_missing"] == []
    assert (
        summary["artifact_manifest_verifier_report_required_fields_match_contract"]
        is False
    )
    assert "tools/verify_no_wing_artifact_manifest.py" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_reports_missing_verifier_failed_check_entries_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(
            f"missing verifier failed-check entry must not open WebSocket: {url}"
        )

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    prestart_path = ensure_prestart_summary(tmp_path)
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
    verifier_report = dict(
        runner.verify_no_wing_artifact_manifest(prestart_path.parent)
    )
    verifier_report.update(
        {
            "status": "failed",
            "failed_checks": ["future_manifest_check"],
            "failed_checks_missing_check_entries": ["future_manifest_check"],
        }
    )
    monkeypatch.setattr(
        runner,
        "verify_no_wing_artifact_manifest",
        lambda _run_dir: verifier_report,
    )
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
            "--prestart-summary",
            str(prestart_path),
            "--expected-plan",
            str(expected_plan),
        ]
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "prestart_summary_missing_or_invalid"
    assert "prestart_summary_artifact_manifest_verifier_failed" in summary[
        "prestart_summary_violations"
    ]
    assert (
        "prestart_summary_artifact_manifest_verifier_failed_check_entry_missing"
        in summary["prestart_summary_violations"]
    )
    assert summary["artifact_manifest_verifier_status"] == "failed"
    assert summary["artifact_manifest_verifier_failed_checks"] == [
        "future_manifest_check"
    ]
    assert summary["artifact_manifest_verifier_failed_checks_missing_check_entries"] == [
        "future_manifest_check"
    ]
    assert "tools/verify_no_wing_artifact_manifest.py" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_reports_false_verifier_failed_check_entry_flag_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(
            f"false verifier failed-check entry flag must not open WebSocket: {url}"
        )

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    prestart_path = ensure_prestart_summary(tmp_path)
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
    verifier_report = dict(
        runner.verify_no_wing_artifact_manifest(prestart_path.parent)
    )
    verifier_checks = dict(verifier_report["checks"])
    verifier_checks["failed_checks_have_check_entries"] = False
    verifier_report.update(
        {
            "status": "passed",
            "failed_checks": [],
            "failed_checks_missing_check_entries": [],
            "checks": verifier_checks,
        }
    )
    monkeypatch.setattr(
        runner,
        "verify_no_wing_artifact_manifest",
        lambda _run_dir: verifier_report,
    )
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
            "--prestart-summary",
            str(prestart_path),
            "--expected-plan",
            str(expected_plan),
        ]
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "prestart_summary_missing_or_invalid"
    assert "prestart_summary_artifact_manifest_verifier_failed" in summary[
        "prestart_summary_violations"
    ]
    assert (
        "prestart_summary_artifact_manifest_verifier_failed_check_entry_flag_not_true"
        in summary["prestart_summary_violations"]
    )
    assert summary["artifact_manifest_verifier_status"] == "passed"
    assert summary["artifact_manifest_verifier_failed_checks"] == []
    assert summary["artifact_manifest_verifier_failed_checks_missing_check_entries"] == []
    assert summary["artifact_manifest_verifier_failed_checks_have_check_entries"] is False
    assert "tools/verify_no_wing_artifact_manifest.py" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_blocks_bad_verifier_check_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(f"bad verifier check must not open WebSocket: {url}")

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    prestart_path = ensure_prestart_summary(tmp_path)
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
    verifier_report = dict(
        runner.verify_no_wing_artifact_manifest(prestart_path.parent)
    )
    verifier_checks = dict(verifier_report["checks"])
    verifier_checks["summary_json_sha256_matches_manifest"] = False
    verifier_report.update(
        {
            "status": "passed",
            "failed_checks": [],
            "failed_checks_missing_check_entries": [],
            "checks": verifier_checks,
        }
    )
    monkeypatch.setattr(
        runner,
        "verify_no_wing_artifact_manifest",
        lambda _run_dir: verifier_report,
    )
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
            "--prestart-summary",
            str(prestart_path),
            "--expected-plan",
            str(expected_plan),
        ]
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "prestart_summary_missing_or_invalid"
    assert "prestart_summary_artifact_manifest_verifier_failed" in summary[
        "prestart_summary_violations"
    ]
    assert (
        "prestart_summary_artifact_manifest_verifier_check_mismatch"
        in summary["prestart_summary_violations"]
    )
    assert summary["artifact_manifest_verifier_status"] == "passed"
    assert summary["artifact_manifest_verifier_failed_checks"] == []
    assert summary["artifact_manifest_verifier_check_mismatches"] == [
        "summary_json_sha256_matches_manifest"
    ]
    assert (
        summary[
            "artifact_manifest_verifier_check_summary_json_sha256_matches_manifest"
        ]
        is False
    )
    assert summary["artifact_manifest_verifier_report_value_mismatches"] == []
    assert "tools/verify_no_wing_artifact_manifest.py" in summary["operator_next_step"]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_blocks_unsafe_prestart_summary_safety_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(f"unsafe prestart summary must not open WebSocket: {url}")

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    prestart_path = ensure_prestart_summary(tmp_path)
    payload = json.loads(prestart_path.read_text(encoding="utf-8"))
    payload["read_only"] = False
    payload["contacts_wing"] = True
    payload["mutates_console_state"] = True
    prestart_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
            "--prestart-summary",
            str(prestart_path),
            "--expected-plan",
            str(tmp_path / "expected-plan.json"),
        ]
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "prestart_summary_missing_or_invalid"
    assert summary["backend_port_prestart_summary_read_only"] is False
    assert summary["backend_port_prestart_summary_contacts_wing"] is True
    assert summary["backend_port_prestart_summary_mutates_console_state"] is True
    assert "prestart_summary_read_only_not_true" in summary[
        "prestart_summary_violations"
    ]
    assert "prestart_summary_contacts_wing_not_false" in summary[
        "prestart_summary_violations"
    ]
    assert "prestart_summary_mutates_console_state_not_false" in summary[
        "prestart_summary_violations"
    ]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_blocks_prestart_missing_backend_detail_safety_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(f"unsafe prestart summary must not open WebSocket: {url}")

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    prestart_path = ensure_prestart_summary(tmp_path)
    payload = json.loads(prestart_path.read_text(encoding="utf-8"))
    detail = payload["results"][0]["detail"]
    for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS:
        detail.pop(field, None)
    prestart_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
            "--prestart-summary",
            str(prestart_path),
            "--expected-plan",
            str(tmp_path / "expected-plan.json"),
        ]
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "prestart_summary_missing_or_invalid"
    for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS:
        assert f"prestart_summary_{field}_not_false" in summary[
            "prestart_summary_violations"
        ]
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_readonly_preflight_blocks_stale_prestart_failure_detail_fields_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(url: str):  # pragma: no cover - exercised only on regression
        raise AssertionError(f"stale prestart summary must not open WebSocket: {url}")

    monkeypatch.setattr(runner.websockets, "connect", fail_connect)
    prestart_path = ensure_prestart_summary(tmp_path)
    payload = json.loads(prestart_path.read_text(encoding="utf-8"))
    detail = payload["results"][0]["detail"]
    detail["backend_port_prestart_check_failure_detail_fields"] = [
        field
        for field in detail["backend_port_prestart_check_failure_detail_fields"]
        if field != "listener_confirmation_checklist"
    ]
    prestart_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
            "--prestart-summary",
            str(prestart_path),
            "--expected-plan",
            str(tmp_path / "expected-plan.json"),
        ]
    )

    exit_code = await runner.run(args)

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "prestart_summary_missing_or_invalid"
    assert (
        "prestart_summary_backend_port_check_failure_detail_fields_mismatch"
        in summary["prestart_summary_violations"]
    )
    assert "failure-detail contract" in summary["operator_next_step"]
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
    assert printed["selected_mvp_blocker"] == WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER
    assert printed["operator_required"] is True
    assert printed["contacts_backend"] is False
    assert printed["queries_console"] is False
    assert printed["manual_review_required_before_readiness_update"] is True
    assert printed["live_authority_granted"] is False
    assert not plan_output.exists()
