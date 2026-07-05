"""Offline tests for the M005 WING disconnect/timeout proof runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
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

ACCEPTED_READONLY_SOURCE_ID = (
    "artifacts/wing_readonly_preflight/"
    "20260702T061111Z_wing_readonly_preflight/summary.json"
)
FakeResponse = dict[str, Any] | BaseException


class FakeWebSocket:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.sent: list[dict[str, Any]] = []
        self.url: str | None = None

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    async def recv(self) -> str:
        if not self.responses:
            raise AssertionError("No fake WebSocket response queued")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
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
            "ws://fake-backend",
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
    connect_request = next(
        request for request in websocket.sent if request["type"] == "connect_wing"
    )
    assert connect_request["observation_only"] is True

    summary = read_summary(tmp_path)
    assert summary["status"] == "passed"
    assert summary["m005_complete"] is True
    assert summary["completed_m005_evidence"] == list(runner.M005_REQUIRED_EVIDENCE)
    assert summary["missing_m005_evidence"] == []
    assert summary["remaining_m005_evidence"] == []
    assert summary["live_authority_granted"] is False
    assert summary["readiness_closure_allowed"] is False
    assert summary["readiness_matrix_update_allowed"] is False


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
