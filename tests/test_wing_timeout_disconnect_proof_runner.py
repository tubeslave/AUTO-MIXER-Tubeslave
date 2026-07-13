"""Offline tests for the M005 WING disconnect/timeout proof runner."""

from __future__ import annotations

import importlib.util
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "wing_timeout_disconnect_proof_runner.py"
)
SPEC = importlib.util.spec_from_file_location("wing_timeout_disconnect_proof_runner", RUNNER_PATH)
assert SPEC is not None
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)
ORIGINAL_EXECUTE_ARTIFACT_ROOT_ERROR = runner._execute_artifact_root_error
ORIGINAL_LAUNCH_BOUND_BACKEND = runner._launch_bound_backend

ACCEPTED_READONLY_SOURCE_ID = (
    "artifacts/wing_readonly_preflight/"
    "20260702T061111Z_wing_readonly_preflight/summary.json"
)
FakeResponse = dict[str, Any] | BaseException


class FakeWebSocket:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.sent: list[dict[str, Any]] = []
        self.auth_requests: list[dict[str, Any]] = []
        self._authentication_ack_pending = False
        self.url: str | None = None

    async def send(self, payload: str) -> None:
        request = json.loads(payload)
        if request.get("type") == "authenticate":
            self.auth_requests.append(request)
            self._authentication_ack_pending = True
            return
        self.sent.append(request)

    async def recv(self) -> str:
        if self._authentication_ack_pending:
            self._authentication_ack_pending = False
            return json.dumps(
                {"type": "authentication_status", "authenticated": True}
            )
        if not self.responses:
            raise AssertionError("No fake WebSocket response queued")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if response.get("type") == "runtime_identity":
            response = json.loads(json.dumps(response))
            request = self.sent[-1]
            preserve_started_at = response.pop("_test_preserve_started_at", False)
            if not preserve_started_at:
                started_at = runner._iso_now()
                response["started_at_utc"] = started_at
                response["process"]["started_at_utc"] = started_at
            response["request_challenge_nonce"] = request.get("challenge_nonce")
            response["launch_binding"] = {
                "mode": "proof_runner",
                "launch_nonce": request.get("expected_launch_nonce"),
                "prestart_provenance_schema": runner.M005_PROVENANCE_SCHEMA,
                "prestart_provenance_size": request.get(
                    "expected_prestart_provenance_size"
                ),
                "prestart_provenance_sha256": request.get(
                    "expected_prestart_provenance_sha256"
                ),
                "bound_before_process_start": True,
                "proof_eligible": True,
            }
            response["manifest_sha256"] = runner.runtime_identity_manifest_sha256(
                response
            )
        return json.dumps(response)


class FakeWebSocketConnect:
    def __init__(self, websocket: FakeWebSocket):
        self.websocket = websocket

    async def __aenter__(self) -> FakeWebSocket:
        return self.websocket

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def patch_websocket_connect(monkeypatch: pytest.MonkeyPatch, websocket: FakeWebSocket) -> None:
    def fake_connect(url: str) -> FakeWebSocketConnect:
        websocket.url = url
        return FakeWebSocketConnect(websocket)

    monkeypatch.setattr(runner.websockets, "connect", fake_connect)


def patch_websocket_connect_failure(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    def fake_connect(url: str):
        raise error

    monkeypatch.setattr(runner.websockets, "connect", fake_connect)


@pytest.mark.asyncio
async def test_bound_backend_receives_auth_token_only_through_one_shot_pipe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    class FakeProcess:
        pid = 4242

    def fake_popen(*_args, **kwargs):
        captured["environment"] = dict(kwargs["env"])
        captured["pass_fds"] = tuple(kwargs["pass_fds"])
        captured["read_fd"] = os.dup(kwargs["pass_fds"][0])
        return FakeProcess()

    async def no_wait(_process, _timeout):
        return None

    async def no_stop(_process):
        return None

    monkeypatch.setattr(runner, "_launch_bound_backend", ORIGINAL_LAUNCH_BOUND_BACKEND)
    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner, "_loopback_port_accepting", lambda _port: False)
    monkeypatch.setattr(runner, "_wait_for_bound_backend", no_wait)
    monkeypatch.setattr(runner, "_stop_bound_backend", no_stop)
    monkeypatch.setenv("AUTOMIXER_WS_AUTH_TOKEN", "stale-parent-environment-token")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-proof-backend")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-reach-proof-backend")
    monkeypatch.setenv("PYTHONHOME", "/tmp/untrusted-python-home")
    monkeypatch.setenv("PYTHONPATH", "/tmp/untrusted-python-path")
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/tmp/untrusted.dylib")
    token = "a" * 43
    args = SimpleNamespace(backend_startup_timeout=0.2)

    async with runner._launch_bound_backend(
        args,
        tmp_path,
        launch_nonce="b" * 32,
        prestart_provenance_sha256="c" * 64,
        prestart_provenance_size=123,
        auth_token=token,
    ):
        environment = captured["environment"]
        assert "AUTOMIXER_WS_AUTH_TOKEN" not in environment
        assert environment["AUTOMIXER_WS_AUTH_TOKEN_FD"] == str(
            captured["pass_fds"][0]
        )
        assert environment["AUTOMIXER_WS_CLIENT_ID"] == "automixer-ui"
        assert environment["AUTOMIXER_WS_TOKEN_TTL_HOURS"] == "1"
        assert environment["AUTOMIXER_WS_ALLOWED_ORIGINS"] == ""
        assert environment["AUTOMIXER_M005_LAUNCH_NONCE"] == "b" * 32
        assert environment["PYTHONPATH"] == os.pathsep.join(
            [str(runner.BACKEND_ROOT), str(runner.REPO_ROOT)]
        )
        for forbidden_name in (
            "OPENAI_API_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "PYTHONHOME",
            "DYLD_INSERT_LIBRARIES",
        ):
            assert forbidden_name not in environment
        assert os.read(captured["read_fd"], 128) == token.encode("ascii")

    os.close(captured["read_fd"])


def patch_accepted_preflight_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_gate(args):
        return (
            {
                "status": "accepted_decision_ready",
                "accepted": True,
                "expected_accepted_readonly_source_id": args.accepted_readonly_source_id,
                "live_authority_granted": False,
                "execution_allowed_by_decision": False,
            },
            None,
        )

    monkeypatch.setattr(runner, "build_preflight_decision_gate", fake_gate)


def parse_args(tmp_path: Path, *argv: str):
    return runner.build_parser().parse_args(
        [
            "--url",
            runner.CANONICAL_BACKEND_URL,
            "--artifact-root",
            str(tmp_path),
            "--timeout",
            "0.2",
            *argv,
        ]
    )


def execute_args(tmp_path: Path, *argv: str):
    return parse_args(
        tmp_path,
        "--channel",
        "3",
        "--value-type",
        "fader",
        "--delta-db",
        "-0.25",
        "--operator-confirm-channel",
        "3",
        "--operator-confirm-value-type",
        "fader",
        "--operator-confirm-delta-db",
        "-0.25",
        "--accepted-readonly-source-id",
        ACCEPTED_READONLY_SOURCE_ID,
        "--console-serial",
        "TEST-WING-RACK-SERIAL",
        "--console-firmware",
        runner.EXPECTED_WING_FIRMWARE,
        "--backend-artifact",
        str(RUNNER_PATH.parents[1] / "backend" / "server.py"),
        "--wing-ip",
        "192.0.2.10",
        "--operator-gated-physical-loss",
        "--execute",
        *argv,
    )


def read_summary(artifact_root: Path) -> dict[str, Any]:
    summaries = list(artifact_root.glob("*/summary.json"))
    assert len(summaries) == 1
    return json.loads(summaries[0].read_text(encoding="utf-8"))


def physical_readback(value: float = 0.0) -> dict[str, Any]:
    return {
        "type": "mixer_parameter_value",
        "status": "verified_from_physical_readback",
        "source": "physical_readback",
        "address": "/ch/3/fdr",
        "channel": 3,
        "value_type": "fader",
        "value": value,
    }


def runtime_identity() -> dict[str, Any]:
    backend = runner._file_identity(RUNNER_PATH.parents[1] / "backend" / "server.py")
    runtime_source = clean_runtime_source()
    started_at = runner._iso_now()
    identity = {
        "type": "runtime_identity",
        "schema": runner.M005_RUNTIME_IDENTITY_SCHEMA,
        "runtime_kind": "source",
        "instance_nonce": "a" * 32,
        "started_at_utc": started_at,
        "process": {
            "pid": 4242,
            "instance_nonce": "a" * 32,
            "started_at_utc": started_at,
        },
        "source": runtime_source["source"],
        "code": runtime_source["code"],
        "backend": backend,
        "launch_binding": {},
        "websocket": {"host": "127.0.0.1", "port": 8765, "loopback_only": True},
        "read_only_identity": True,
        "live_authority_granted": False,
    }
    identity["manifest_sha256"] = runner.runtime_identity_manifest_sha256(identity)
    return identity


def clean_runtime_source() -> dict[str, Any]:
    runtime_source = runner.capture_source_runtime_components(RUNNER_PATH.parents[1])
    runtime_source["source"].update(
        {
            "worktree_dirty": False,
            "status_sha256": runner._sha256_bytes(b""),
            "tracked_diff_sha256": runner._sha256_bytes(b""),
        }
    )
    return runtime_source


@pytest.fixture(autouse=True)
def clean_bound_provenance(monkeypatch: pytest.MonkeyPatch):
    original_capture = runner.capture_current_release_provenance

    def capture(args):
        provenance = original_capture(args)
        provenance["source"].update({
            "worktree_dirty": False,
            "status_sha256": runner._sha256_bytes(b""),
            "tracked_diff_sha256": runner._sha256_bytes(b""),
        })
        provenance["runtime_source"] = clean_runtime_source()
        provenance["console"] = {
            "model": runner.EXPECTED_WING_MODEL,
            "serial": "TEST-WING-RACK-SERIAL",
            "firmware": runner.EXPECTED_WING_FIRMWARE,
            "identity_source": "operator_attested_owner_challenge",
            "console_identity_query_available": False,
        }
        provenance["console_identity_challenge"] = {
            "path": (
                "artifacts/wing_console_identity_owner_challenge/"
                "unit-owner-challenge/summary.json"
            ),
            "exists": True,
            "size": 512,
            "sha256": "b" * 64,
        }
        return provenance

    async def no_wait(_prompt: str) -> None:
        return None

    @asynccontextmanager
    async def fake_bound_backend(*_args, **_kwargs):
        yield SimpleNamespace(pid=4242)

    monkeypatch.setattr(runner, "capture_current_release_provenance", capture)
    monkeypatch.setattr(runner, "_console_identity_challenge_error", lambda _args: None)
    monkeypatch.setattr(runner, "_wait_for_operator", no_wait)
    monkeypatch.setattr(runner, "_launch_bound_backend", fake_bound_backend)
    monkeypatch.setattr(runner, "_execute_artifact_root_error", lambda _args: None)


def unavailable_readback() -> dict[str, Any]:
    return {
        "type": "mixer_parameter_value",
        "status": "readback_unavailable",
        "source": "unavailable",
        "address": "/ch/3/fdr",
        "channel": 3,
        "value_type": "fader",
        "value": None,
    }


def wrong_target_readback(payload: dict[str, Any]) -> dict[str, Any]:
    wrong = dict(payload)
    wrong["channel"] = 4
    wrong["address"] = "/ch/4/fdr"
    return wrong


def test_m005_completion_requires_physical_reconnect_readback() -> None:
    summary = {
        "disconnect_mode": "physical-cable-loss",
        "channel": 3,
        "value_type": "fader",
        "address": "/ch/3/fdr",
        "phases": {
            "baseline": {
                "status": "passed",
                "readback": physical_readback(0.0),
            },
            "disconnect": {
                "status": "operator_confirmed_physical_cable_loss",
                "confirmation": runner.PHYSICAL_CABLE_CONFIRMATION,
            },
            "stale_readback": {
                "status": "passed",
                "readback": unavailable_readback(),
            },
            "blocked_write_while_disconnected": {
                "status": "passed",
                "block_reason": "mixer_not_connected",
                "apply_result": {
                    "type": "operator_proposal_apply_blocked",
                    "status": "apply_blocked",
                    "apply_result": {"status": "blocked", "reason": "mixer_not_connected"},
                },
            },
            "reconnect_recovery": {
                "status": "passed",
                "connection_status": {"connected": True},
                "readback": unavailable_readback(),
                "unchanged_from_baseline": True,
            },
        },
    }

    annotated = runner._annotate_m005_evidence(summary)

    assert annotated["m005_complete"] is False
    assert "reconnect_recovery_verified" not in annotated["completed_m005_evidence"]
    assert annotated["missing_m005_evidence"] == ["reconnect_recovery_verified"]


@pytest.mark.asyncio
async def test_m005_authenticates_without_recording_bearer_in_event_stream() -> None:
    websocket = FakeWebSocket([])
    token = "a" * 43

    await runner._authenticate_proof_websocket(
        websocket,
        token,
        timeout=0.2,
    )

    assert websocket.sent == []
    assert websocket.auth_requests == [
        {
            "type": "authenticate",
            "token": token,
            "client_id": "automixer-ui",
        }
    ]


def test_execute_artifact_root_is_canonical_and_git_ignored(tmp_path: Path) -> None:
    assert ORIGINAL_EXECUTE_ARTIFACT_ROOT_ERROR(
        SimpleNamespace(artifact_root=str(tmp_path))
    ) == f"--artifact-root must equal {runner.DEFAULT_ARTIFACT_ROOT} with --execute"
    assert runner._canonical_m005_artifact_root_is_ignored() is True
    assert ORIGINAL_EXECUTE_ARTIFACT_ROOT_ERROR(
        SimpleNamespace(artifact_root=runner.DEFAULT_ARTIFACT_ROOT)
    ) is None


def test_m005_completion_requires_blocked_apply_result_payload() -> None:
    summary = {
        "disconnect_mode": "physical-cable-loss",
        "channel": 3,
        "value_type": "fader",
        "address": "/ch/3/fdr",
        "phases": {
            "baseline": {
                "status": "passed",
                "readback": physical_readback(0.0),
            },
            "disconnect": {
                "status": "operator_confirmed_physical_cable_loss",
                "confirmation": runner.PHYSICAL_CABLE_CONFIRMATION,
            },
            "stale_readback": {
                "status": "passed",
                "readback": unavailable_readback(),
            },
            "blocked_write_while_disconnected": {
                "status": "passed",
                "block_reason": "mixer_not_connected",
                "apply_result": {
                    "type": "operator_proposal_apply_result",
                    "status": "applied",
                    "apply_result": {"status": "applied", "reason": "mixer_not_connected"},
                },
            },
            "reconnect_recovery": {
                "status": "passed",
                "connection_status": {"connected": True},
                "readback": physical_readback(0.0),
                "unchanged_from_baseline": True,
            },
        },
    }

    annotated = runner._annotate_m005_evidence(summary)

    assert annotated["m005_complete"] is False
    assert "blocked_write_while_disconnected" not in annotated["completed_m005_evidence"]
    assert annotated["missing_m005_evidence"] == ["blocked_write_while_disconnected"]


def standard_prefix_responses() -> list[dict[str, Any]]:
    return [
        runtime_identity(),
        {
            "type": "connection_status",
            "connected": True,
            "mode": "wing",
            "ip": "192.0.2.10",
            "observation_only": False,
            "record_telemetry": False,
        },
        {"type": "operator_mode_status", "mode": "assist"},
        {"type": "state_update", "state": {"/ch/3/fdr": 0.0}},
        physical_readback(0.0),
        {"type": "operator_mode_status", "mode": "supervised"},
        {"type": "operator_proposal_created", "status": "created", "proposal": {"id": "probe"}},
        {"type": "connection_status", "connected": False},
    ]


@pytest.mark.asyncio
async def test_dry_run_marks_all_m005_evidence_missing(tmp_path: Path) -> None:
    exit_code = await runner.run(parse_args(tmp_path))

    assert exit_code == 0
    summary = read_summary(tmp_path)
    assert summary["status"] == "planned"
    assert summary["m005_complete"] is False
    assert summary["completed_m005_evidence"] == []
    assert summary["missing_m005_evidence"] == list(runner.M005_REQUIRED_EVIDENCE)
    assert summary["remaining_m005_evidence"] == [
        runner.M005_EVIDENCE_DESCRIPTIONS[item] for item in runner.M005_REQUIRED_EVIDENCE
    ]
    assert summary["live_authority_granted"] is False
    assert summary["readiness_closure_allowed"] is False
    assert summary["live_apply_request_sent"] is False


@pytest.mark.asyncio
async def test_software_disconnect_probe_uses_canonical_disconnect_and_blocks_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_accepted_preflight_gate(monkeypatch)
    websocket = FakeWebSocket(
        [
            *standard_prefix_responses(),
            unavailable_readback(),
            {
                "type": "operator_proposal_apply_blocked",
                "status": "apply_blocked",
                "apply_result": {"status": "blocked", "reason": "mixer_not_connected"},
            },
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)

    exit_code = await runner.run(execute_args(tmp_path))

    assert exit_code == 0
    sent_types = [request["type"] for request in websocket.sent]
    assert sent_types == [
        "get_runtime_identity",
        "connect_wing",
        "get_operator_mode",
        "get_state",
        "query_mixer_parameter",
        "set_operator_mode",
        "create_operator_proposal",
        "disconnect_mixer",
        "query_mixer_parameter",
        "apply_operator_proposal",
    ]
    assert "disconnect" not in sent_types
    assert "set_fader" not in sent_types
    assert "set_gain" not in sent_types

    summary = read_summary(tmp_path)
    assert summary["status"] == "passed"
    assert summary["reason"] == "blocked_write_while_disconnected_verified"
    assert summary["fail_closed"] is True
    assert summary["live_authority_granted"] is False
    assert summary["readiness_closure_allowed"] is False
    assert summary["safety_pre_write_disconnect_verified"] is True
    assert summary["safety_pre_write_readback_unavailable"] is True
    assert summary["direct_live_write_request_sent"] is False
    assert summary["forbidden_request_types_sent"] == []
    assert summary["phases"]["blocked_write_while_disconnected"]["block_reason"] == (
        "mixer_not_connected"
    )
    assert summary["m005_complete"] is False
    assert summary["completed_m005_evidence"] == [
        "physical_baseline_readback",
        "stale_readback_unavailable",
        "blocked_write_while_disconnected",
    ]
    assert summary["missing_m005_evidence"] == [
        "operator_confirmed_physical_cable_loss",
        "reconnect_recovery_verified",
    ]
    assert summary["remaining_m005_evidence"] == [
        runner.M005_EVIDENCE_DESCRIPTIONS["operator_confirmed_physical_cable_loss"],
        runner.M005_EVIDENCE_DESCRIPTIONS["reconnect_recovery_verified"],
    ]


@pytest.mark.asyncio
async def test_physical_cable_loss_with_reconnect_marks_m005_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_accepted_preflight_gate(monkeypatch)
    websocket = FakeWebSocket(
        [
            *standard_prefix_responses()[:-1],
            unavailable_readback(),
            {
                "type": "operator_proposal_apply_blocked",
                "status": "apply_blocked",
                "apply_result": {"status": "blocked", "reason": "mixer_not_connected"},
            },
            {"type": "connection_status", "connected": False},
            {"type": "connection_status", "connected": True},
            physical_readback(0.0),
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)

    exit_code = await runner.run(
        execute_args(
            tmp_path,
            "--disconnect-mode",
            "physical-cable-loss",
            "--operator-confirm-physical-cable-loss",
            runner.PHYSICAL_CABLE_CONFIRMATION,
            "--physical-loss-wait-sec",
            "0",
            "--reconnect-wing-ip",
            "192.0.2.10",
        )
    )

    assert exit_code == 0
    sent_types = [request["type"] for request in websocket.sent]
    assert sent_types == [
        "get_runtime_identity",
        "connect_wing",
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
    assert "disconnect_mixer" not in sent_types
    connect_requests = [
        request for request in websocket.sent if request["type"] == "connect_wing"
    ]
    assert [request["observation_only"] for request in connect_requests] == [False, True]

    summary = read_summary(tmp_path)
    assert summary["status"] == "passed"
    assert summary["m005_complete"] is True
    assert summary["completed_m005_evidence"] == list(runner.M005_REQUIRED_EVIDENCE)
    assert summary["missing_m005_evidence"] == []
    assert summary["remaining_m005_evidence"] == []
    assert summary["live_authority_granted"] is False
    assert summary["readiness_closure_allowed"] is False
    assert summary["readiness_matrix_update_allowed"] is False
    assert summary["m005_behavioral_proof_complete"] is True
    assert summary["m005_current_release_provenance_captured"] is True
    assert summary["provenance"]["schema"] == runner.M005_PROVENANCE_SCHEMA
    assert summary["provenance"]["console"] == {
        "model": runner.EXPECTED_WING_MODEL,
        "serial": "TEST-WING-RACK-SERIAL",
        "firmware": runner.EXPECTED_WING_FIRMWARE,
        "identity_source": "operator_attested_owner_challenge",
        "console_identity_query_available": False,
    }
    assert summary["runtime_identity"]["backend"] == {
        key: summary["provenance"]["backend"][key]
        for key in ("path", "exists", "size", "sha256")
    }
    assert summary["provenance"]["events"]["exists"] is True
    assert len(summary["provenance"]["events"]["sha256"]) == 64


@pytest.mark.asyncio
async def test_physical_stale_readback_timeout_counts_as_unavailable_before_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_accepted_preflight_gate(monkeypatch)
    websocket = FakeWebSocket(
        [
            *standard_prefix_responses()[:-1],
            TimeoutError("Timed out waiting for stale readback"),
            {
                "type": "operator_proposal_apply_blocked",
                "status": "apply_blocked",
                "apply_result": {"status": "blocked", "reason": "mixer_not_connected"},
            },
            {"type": "connection_status", "connected": True},
            physical_readback(0.0),
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)

    exit_code = await runner.run(
        execute_args(
            tmp_path,
            "--disconnect-mode",
            "physical-cable-loss",
            "--operator-confirm-physical-cable-loss",
            runner.PHYSICAL_CABLE_CONFIRMATION,
            "--physical-loss-wait-sec",
            "0",
            "--reconnect-wing-ip",
            "192.0.2.10",
        )
    )

    assert exit_code == 0
    sent_types = [request["type"] for request in websocket.sent]
    assert sent_types == [
        "get_runtime_identity",
        "connect_wing",
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

    summary = read_summary(tmp_path)
    stale_readback = summary["phases"]["stale_readback"]["readback"]
    assert summary["status"] == "passed"
    assert summary["m005_complete"] is True
    assert stale_readback["type"] == "mixer_parameter_read_blocked"
    assert stale_readback["status"] == "readback_timeout"
    assert stale_readback["source"] == "timeout"
    assert stale_readback["reason"] == "stale_readback_timeout_after_disconnect"
    assert stale_readback["transport_failure"]["type"] == "TimeoutError"
    assert summary["completed_m005_evidence"] == list(runner.M005_REQUIRED_EVIDENCE)
    assert summary["live_authority_granted"] is False
    assert summary["readiness_closure_allowed"] is False


@pytest.mark.asyncio
async def test_live_or_stale_readback_after_disconnect_stops_before_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_accepted_preflight_gate(monkeypatch)
    websocket = FakeWebSocket([*standard_prefix_responses(), physical_readback(0.0)])
    patch_websocket_connect(monkeypatch, websocket)

    exit_code = await runner.run(execute_args(tmp_path))

    assert exit_code == 7
    sent_types = [request["type"] for request in websocket.sent]
    assert sent_types == [
        "get_runtime_identity",
        "connect_wing",
        "get_operator_mode",
        "get_state",
        "query_mixer_parameter",
        "set_operator_mode",
        "create_operator_proposal",
        "disconnect_mixer",
        "query_mixer_parameter",
    ]
    assert "apply_operator_proposal" not in sent_types

    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "stale_or_live_readback_after_disconnect"
    assert summary["fail_closed"] is True
    assert summary["safety_pre_write_disconnect_verified"] is True
    assert summary["safety_pre_write_readback_unavailable"] is False
    assert summary["live_apply_request_sent"] is False
    assert summary["direct_live_write_request_sent"] is False


@pytest.mark.asyncio
async def test_wrong_target_stale_readback_stops_before_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_accepted_preflight_gate(monkeypatch)
    websocket = FakeWebSocket(
        [*standard_prefix_responses(), wrong_target_readback(unavailable_readback())]
    )
    patch_websocket_connect(monkeypatch, websocket)

    exit_code = await runner.run(execute_args(tmp_path))

    assert exit_code == 7
    sent_types = [request["type"] for request in websocket.sent]
    assert sent_types == [
        "get_runtime_identity",
        "connect_wing",
        "get_operator_mode",
        "get_state",
        "query_mixer_parameter",
        "set_operator_mode",
        "create_operator_proposal",
        "disconnect_mixer",
        "query_mixer_parameter",
    ]
    assert "apply_operator_proposal" not in sent_types

    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "stale_or_live_readback_after_disconnect"
    assert summary["phases"]["stale_readback"]["status"] == "failed"
    assert summary["safety_pre_write_readback_unavailable"] is False
    assert summary["m005_complete"] is False
    assert "stale_readback_unavailable" in summary["missing_m005_evidence"]
    assert summary["live_apply_request_sent"] is False
    assert summary["direct_live_write_request_sent"] is False


@pytest.mark.asyncio
async def test_wrong_target_reconnect_readback_cannot_complete_m005(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_accepted_preflight_gate(monkeypatch)
    websocket = FakeWebSocket(
        [
            *standard_prefix_responses()[:-1],
            unavailable_readback(),
            {
                "type": "operator_proposal_apply_blocked",
                "status": "apply_blocked",
                "apply_result": {"status": "blocked", "reason": "mixer_not_connected"},
            },
            {"type": "connection_status", "connected": True},
            wrong_target_readback(physical_readback(0.0)),
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)

    exit_code = await runner.run(
        execute_args(
            tmp_path,
            "--disconnect-mode",
            "physical-cable-loss",
            "--operator-confirm-physical-cable-loss",
            runner.PHYSICAL_CABLE_CONFIRMATION,
            "--physical-loss-wait-sec",
            "0",
            "--reconnect-wing-ip",
            "192.0.2.10",
        )
    )

    assert exit_code == 10
    summary = read_summary(tmp_path)
    assert summary["status"] == "failed"
    assert summary["reason"] == "reconnect_recovery_not_verified"
    assert summary["phases"]["reconnect_recovery"]["status"] == "failed"
    assert summary["phases"]["reconnect_recovery"]["unchanged_from_baseline"] is False
    assert summary["m005_complete"] is False
    assert "reconnect_recovery_verified" in summary["missing_m005_evidence"]
    assert summary["direct_live_write_request_sent"] is False


def test_physical_cable_loss_mode_requires_explicit_operator_confirmation(
    tmp_path: Path,
) -> None:
    args = execute_args(tmp_path, "--disconnect-mode", "physical-cable-loss")

    assert (
        runner.validate_args(args)
        == "--operator-confirm-physical-cable-loss must equal WING_CABLE_PULLED_BY_OPERATOR"
    )

    args = execute_args(
        tmp_path,
        "--disconnect-mode",
        "physical-cable-loss",
        "--operator-confirm-physical-cable-loss",
        "wrong-token",
    )

    assert (
        runner.validate_args(args)
        == "--operator-confirm-physical-cable-loss must equal WING_CABLE_PULLED_BY_OPERATOR"
    )


def test_noncanonical_backend_url_and_packaged_backend_are_rejected(tmp_path: Path) -> None:
    noncanonical = execute_args(tmp_path)
    noncanonical.url = "ws://127.0.0.1:9999"
    assert "canonical loopback" in str(runner.validate_args(noncanonical))

    packaged = execute_args(tmp_path)
    packaged.backend_runtime_kind = "packaged"
    packaged.backend_artifact = "/etc/hosts"
    packaged.package_artifact = "/etc/hosts"
    assert runner.validate_args(packaged) == (
        "packaged M005 proof is unsupported until an authenticated package path exists"
    )


@pytest.mark.asyncio
async def test_execute_blocks_dirty_checkout_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = execute_args(tmp_path)
    clean = runner.capture_current_release_provenance(args)
    dirty = json.loads(json.dumps(clean))
    dirty["source"]["worktree_dirty"] = True
    monkeypatch.setattr(runner, "capture_current_release_provenance", lambda _args: dirty)
    opened = False

    def should_not_connect(_url: str):
        nonlocal opened
        opened = True
        raise AssertionError("WebSocket must not open for a dirty checkout")

    monkeypatch.setattr(runner.websockets, "connect", should_not_connect)

    exit_code = await runner.run(args)

    assert exit_code == 15
    assert opened is False
    assert read_summary(tmp_path)["reason"] == (
        "m005_execute_requires_clean_checkout_before_artifact_creation"
    )


@pytest.mark.asyncio
async def test_runtime_identity_mismatch_blocks_before_mixer_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_identity = runtime_identity()
    bad_identity["backend"]["sha256"] = "f" * 64
    websocket = FakeWebSocket([bad_identity])
    patch_websocket_connect(monkeypatch, websocket)
    patch_accepted_preflight_gate(monkeypatch)

    exit_code = await runner.run(execute_args(tmp_path))

    assert exit_code == 16
    assert [request["type"] for request in websocket.sent] == ["get_runtime_identity"]
    assert read_summary(tmp_path)["reason"] == "backend_runtime_identity_mismatch"


@pytest.mark.asyncio
async def test_initial_connection_must_be_write_capable_before_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [
            runtime_identity(),
            {
                "type": "connection_status",
                "connected": True,
                "mode": "wing",
                "ip": "192.0.2.10",
                "observation_only": True,
                "record_telemetry": False,
            },
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)
    patch_accepted_preflight_gate(monkeypatch)

    exit_code = await runner.run(execute_args(tmp_path))

    assert exit_code == 18
    assert [request["type"] for request in websocket.sent] == [
        "get_runtime_identity",
        "connect_wing",
    ]
    summary = read_summary(tmp_path)
    assert summary["reason"] == "initial_wing_connection_not_verified_for_write_probe"
    assert summary["fail_closed"] is True
    assert summary["initial_connection_verified"] is False
    assert summary["live_apply_request_sent"] is False


@pytest.mark.asyncio
async def test_initial_connection_timeout_blocks_before_baseline_or_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [runtime_identity(), TimeoutError("initial WING connect timed out")]
    )
    patch_websocket_connect(monkeypatch, websocket)
    patch_accepted_preflight_gate(monkeypatch)

    exit_code = await runner.run(execute_args(tmp_path))

    assert exit_code == 1
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "connection_or_timeout_fail_closed"
    assert summary["fail_closed"] is True
    assert summary["live_apply_request_sent"] is False
    assert [request["type"] for request in websocket.sent] == [
        "get_runtime_identity",
        "connect_wing",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["v1", "stale_process", "changed_handler"])
async def test_runtime_identity_rejects_v1_stale_process_and_changed_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    identity = runtime_identity()
    if mutation == "v1":
        identity["schema"] = "automixer.backend_runtime_identity.v1"
    elif mutation == "stale_process":
        identity["started_at_utc"] = "1999-01-01T00:00:00Z"
        identity["process"]["started_at_utc"] = "1999-01-01T00:00:00Z"
        identity["_test_preserve_started_at"] = True
    else:
        identity["code"]["files"][0]["sha256"] = "f" * 64
    identity["manifest_sha256"] = runner.runtime_identity_manifest_sha256(identity)
    websocket = FakeWebSocket([identity])
    patch_websocket_connect(monkeypatch, websocket)
    patch_accepted_preflight_gate(monkeypatch)

    exit_code = await runner.run(execute_args(tmp_path))

    assert exit_code == 16
    assert [request["type"] for request in websocket.sent] == ["get_runtime_identity"]
    assert read_summary(tmp_path)["reason"] == "backend_runtime_identity_mismatch"


@pytest.mark.asyncio
async def test_same_second_runs_use_exclusive_uuid_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "utc_stamp", lambda: "20260712T000000Z")

    assert await runner.run(parse_args(tmp_path)) == 0
    assert await runner.run(parse_args(tmp_path)) == 0

    run_dirs = sorted(path for path in tmp_path.iterdir() if path.is_dir())
    assert len(run_dirs) == 2
    assert run_dirs[0].name != run_dirs[1].name
    for run_dir in run_dirs:
        first_event = json.loads((run_dir / "events.jsonl").read_text().splitlines()[0])
        assert first_event["seq"] == 1
        assert first_event["event_type"] == "provenance_capture"
        assert first_event["payload"]["captured_before_backend_contact"] is True


@pytest.mark.asyncio
async def test_operator_gated_physical_loss_waits_before_stale_probe_and_reconnect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_accepted_preflight_gate(monkeypatch)
    websocket = FakeWebSocket(
        [
            *standard_prefix_responses()[:-1],
            unavailable_readback(),
            {
                "type": "operator_proposal_apply_blocked",
                "status": "apply_blocked",
                "apply_result": {"status": "blocked", "reason": "mixer_not_connected"},
            },
            {"type": "connection_status", "connected": True},
            physical_readback(0.0),
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)
    prompts: list[str] = []

    async def fake_wait_for_operator(prompt: str) -> None:
        prompts.append(prompt)

    monkeypatch.setattr(runner, "_wait_for_operator", fake_wait_for_operator)

    exit_code = await runner.run(
        execute_args(
            tmp_path,
            "--disconnect-mode",
            "physical-cable-loss",
            "--operator-confirm-physical-cable-loss",
            runner.PHYSICAL_CABLE_CONFIRMATION,
            "--operator-gated-physical-loss",
            "--reconnect-wing-ip",
            "192.0.2.10",
        )
    )

    assert exit_code == 0
    assert len(prompts) == 2
    assert "disconnect" in prompts[0].lower()
    assert "reconnect" in prompts[1].lower()
    summary = read_summary(tmp_path)
    assert summary["m005_complete"] is True
    assert summary["phases"]["disconnect"]["operator_gate"] is True


@pytest.mark.asyncio
async def test_physical_cable_loss_execute_requires_reconnect_target_before_websocket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_websocket_connect_failure(monkeypatch, AssertionError("backend must not be contacted"))

    exit_code = await runner.run(
        execute_args(
            tmp_path,
            "--disconnect-mode",
            "physical-cable-loss",
            "--operator-confirm-physical-cable-loss",
            runner.PHYSICAL_CABLE_CONFIRMATION,
        )
    )

    assert exit_code == 2
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == (
        "--reconnect-wing-ip is required for executed physical-cable-loss proof"
    )
    assert summary["m005_complete"] is False
    assert "reconnect_recovery_verified" in summary["missing_m005_evidence"]
    assert summary["sent_request_types"] == []
    assert summary["live_apply_request_sent"] is False
    assert summary["direct_live_write_request_sent"] is False


@pytest.mark.asyncio
async def test_execute_blocks_before_websocket_without_accepted_preflight_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked_gate(args):
        return (
            {
                "status": "blocked",
                "accepted": False,
                "expected_accepted_readonly_source_id": args.accepted_readonly_source_id,
            },
            "preflight_decision_review_not_accepted",
        )

    monkeypatch.setattr(runner, "build_preflight_decision_gate", blocked_gate)
    patch_websocket_connect_failure(monkeypatch, AssertionError("backend must not be contacted"))

    exit_code = await runner.run(execute_args(tmp_path))

    assert exit_code == 13
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "preflight_decision_review_not_accepted"
    assert summary["fail_closed"] is True
    assert summary["live_apply_request_sent"] is False
    assert summary["rollback_request_sent"] is False
    assert summary["sent_request_types"] == []


@pytest.mark.asyncio
async def test_unexpected_apply_triggers_rollback_and_manual_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_accepted_preflight_gate(monkeypatch)
    websocket = FakeWebSocket(
        [
            *standard_prefix_responses(),
            unavailable_readback(),
            {
                "type": "operator_proposal_apply_result",
                "status": "applied",
                "apply_result": {"status": "applied", "readback_value": -0.25},
            },
            {
                "type": "manual_write_rollback_applied",
                "status": "applied",
                "readback_value": 0.0,
                "readback_status": "verified_from_physical_readback",
            },
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)

    exit_code = await runner.run(execute_args(tmp_path))

    assert exit_code == 8
    sent_types = [request["type"] for request in websocket.sent]
    assert sent_types[-2:] == ["apply_operator_proposal", "rollback_last_manual_write"]

    summary = read_summary(tmp_path)
    assert summary["status"] == "manual_review_required"
    assert summary["reason"] == "blocked_write_probe_applied_unexpectedly"
    assert summary["fail_closed"] is False
    assert summary["live_apply_request_sent"] is True
    assert summary["rollback_request_sent"] is True
    assert summary["direct_live_write_request_sent"] is False
    assert summary["unexpected_apply_rollback_result"]["type"] == "manual_write_rollback_applied"


@pytest.mark.asyncio
async def test_timeout_after_apply_request_requires_manual_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_accepted_preflight_gate(monkeypatch)
    websocket = FakeWebSocket(
        [
            *standard_prefix_responses(),
            unavailable_readback(),
            TimeoutError("Timed out waiting for blocked apply result"),
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)

    exit_code = await runner.run(execute_args(tmp_path))

    assert exit_code == 1
    assert [request["type"] for request in websocket.sent][-2:] == [
        "query_mixer_parameter",
        "apply_operator_proposal",
    ]
    summary = read_summary(tmp_path)
    assert summary["status"] == "manual_review_required"
    assert summary["reason"] == "connection_or_timeout_after_apply_request"
    assert summary["fail_closed"] is False
    assert summary["live_apply_request_sent"] is True
    assert summary["rollback_request_sent"] is False
    assert summary["direct_live_write_request_sent"] is False
    assert summary["transport_failure"]["type"] == "TimeoutError"
    assert summary["m005_complete"] is False
    assert "blocked_write_while_disconnected" in summary["missing_m005_evidence"]
    assert "do not use this artifact as promotion evidence" in summary[
        "operator_action_required"
    ]


@pytest.mark.asyncio
async def test_generic_exception_after_apply_request_requires_manual_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_accepted_preflight_gate(monkeypatch)
    websocket = FakeWebSocket(
        [
            *standard_prefix_responses(),
            unavailable_readback(),
            RuntimeError("unexpected receive failure after apply send"),
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)

    exit_code = await runner.run(execute_args(tmp_path))

    assert exit_code == 1
    summary = read_summary(tmp_path)
    assert summary["status"] == "manual_review_required"
    assert summary["reason"] == "connection_or_timeout_after_apply_request"
    assert summary["fail_closed"] is False
    assert summary["live_apply_request_sent"] is True
    assert summary["rollback_request_sent"] is False
    assert summary["transport_failure"]["type"] == "RuntimeError"
    assert "do not use this artifact as promotion evidence" in summary[
        "operator_action_required"
    ]


@pytest.mark.asyncio
async def test_timeout_after_rollback_request_never_claims_rollback_applied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_accepted_preflight_gate(monkeypatch)
    websocket = FakeWebSocket(
        [
            *standard_prefix_responses(),
            unavailable_readback(),
            {
                "type": "operator_proposal_apply_result",
                "status": "applied",
                "apply_result": {
                    "status": "applied",
                    "transaction_id": "tx-unknown-rollback-result",
                },
            },
            TimeoutError("Timed out waiting for rollback result"),
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)

    exit_code = await runner.run(execute_args(tmp_path))

    assert exit_code == 1
    summary = read_summary(tmp_path)
    assert summary["status"] == "manual_review_required"
    assert summary["reason"] == "connection_or_timeout_after_apply_request"
    assert summary["fail_closed"] is False
    assert summary["live_apply_request_sent"] is True
    assert summary["rollback_request_sent"] is True
    assert summary["transport_failure"]["type"] == "TimeoutError"
    assert summary.get("unexpected_apply_rollback_result") is None


@pytest.mark.asyncio
async def test_generic_exception_after_rollback_never_claims_rollback_applied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_accepted_preflight_gate(monkeypatch)
    websocket = FakeWebSocket(
        [
            *standard_prefix_responses(),
            unavailable_readback(),
            {
                "type": "operator_proposal_apply_result",
                "status": "applied",
                "apply_result": {
                    "status": "applied",
                    "transaction_id": "tx-runtime-error-after-rollback",
                },
            },
            RuntimeError("unexpected receive failure after rollback send"),
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)

    exit_code = await runner.run(execute_args(tmp_path))

    assert exit_code == 1
    summary = read_summary(tmp_path)
    assert summary["status"] == "manual_review_required"
    assert summary["fail_closed"] is False
    assert summary["live_apply_request_sent"] is True
    assert summary["rollback_request_sent"] is True
    assert summary["transport_failure"]["type"] == "RuntimeError"
    assert summary.get("unexpected_apply_rollback_result") is None
