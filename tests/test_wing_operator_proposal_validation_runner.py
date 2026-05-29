"""Offline tests for the proposal-based Wing validation runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


RUNNER_PATH = Path(__file__).resolve().parents[1] / "tools" / "wing_operator_proposal_validation_runner.py"
SPEC = importlib.util.spec_from_file_location("wing_operator_proposal_validation_runner", RUNNER_PATH)
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


def patch_websocket_connect(monkeypatch: pytest.MonkeyPatch, websocket: FakeWebSocket) -> None:
    def fake_connect(url: str) -> FakeWebSocketConnect:
        websocket.url = url
        return FakeWebSocketConnect(websocket)

    monkeypatch.setattr(runner.websockets, "connect", fake_connect)


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


def read_summary(artifact_root: Path) -> dict:
    summaries = list(artifact_root.glob("*/summary.json"))
    assert len(summaries) == 1
    return json.loads(summaries[0].read_text(encoding="utf-8"))


def read_events(summary: dict) -> list[dict]:
    events_path = Path(summary["events_path"])
    return [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize(
    ("argv", "expected_error"),
    [
        (["--channel", "0", "--value-type", "fader", "--delta-db", "-0.1"], "--channel must be in 1..48"),
        (["--channel", "49", "--value-type", "fader", "--delta-db", "-0.1"], "--channel must be in 1..48"),
        (["--channel", "1", "--value-type", "fader", "--delta-db", "0.1"], "--delta-db must be negative"),
        (["--channel", "1", "--value-type", "gain", "--delta-db", "-0.75"], "--delta-db must be negative"),
        (
            ["--channel", "1", "--value-type", "fader", "--delta-db", "-0.1", "--prove-emergency-stop"],
            "--prove-emergency-stop requires --execute",
        ),
        (
            ["--channel", "1", "--value-type", "fader", "--delta-db", "-0.1", "--prove-blocked-apply"],
            "--prove-blocked-apply requires --execute",
        ),
        (
            ["--channel", "1", "--value-type", "fader", "--delta-db", "-0.1", "--rollback-delay-sec", "6"],
            "--rollback-delay-sec must be between 0 and 5 seconds",
        ),
    ],
)
def test_validate_args_blocks_unsafe_invocations(tmp_path: Path, argv: list[str], expected_error: str) -> None:
    args = parse_args(tmp_path, *argv)

    assert expected_error in runner.validate_args(args)


@pytest.mark.asyncio
async def test_dry_run_plans_without_creating_or_applying_proposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [
            {"type": "operator_mode_status", "mode": "assist"},
            {"type": "state_update", "state": {"/ch/7/fdr": -10.0}},
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)
    args = parse_args(tmp_path, "--channel", "7", "--value-type", "fader", "--delta-db", "-0.25")

    exit_code = await runner.run(args)

    assert exit_code == 0
    assert websocket.url == "ws://fake-backend"
    assert [request["type"] for request in websocket.sent] == ["get_operator_mode", "get_state"]
    summary = read_summary(tmp_path)
    assert summary["status"] == "planned"
    assert summary["reason"] == "dry_run_only_add_execute_to_write"
    assert summary["address"] == "/ch/7/fdr"
    assert summary["old_value"] == -10.0
    assert summary["target_value"] == -10.25
    assert [event["event_type"] for event in read_events(summary)] == ["send", "recv", "send", "recv"]


@pytest.mark.asyncio
async def test_dry_run_uses_readonly_physical_read_when_backend_state_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [
            {"type": "operator_mode_status", "mode": "assist"},
            {"type": "state_update", "state": {}},
            {
                "type": "mixer_parameter_value",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "address": "/ch/3/in/set/trim",
                "channel": 3,
                "value_type": "gain",
                "value": 1.0,
            },
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)
    args = parse_args(tmp_path, "--channel", "3", "--value-type", "gain", "--delta-db", "-0.25")

    exit_code = await runner.run(args)

    assert exit_code == 0
    assert [request["type"] for request in websocket.sent] == [
        "get_operator_mode",
        "get_state",
        "query_mixer_parameter",
    ]
    query = websocket.sent[2]
    assert query == {
        "type": "query_mixer_parameter",
        "channel": 3,
        "value_type": "gain",
        "address": "/ch/3/in/set/trim",
    }
    summary = read_summary(tmp_path)
    assert summary["status"] == "planned"
    assert summary["address"] == "/ch/3/in/set/trim"
    assert summary["old_value"] == 1.0
    assert summary["old_value_source"] == "physical_readback"
    assert summary["old_value_status"] == "verified_from_physical_readback"
    assert summary["target_value"] == 0.75


@pytest.mark.asyncio
async def test_execute_path_uses_apply_operator_proposal_only_for_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal_id_prefix = "wing-proposal-validation"
    websocket = FakeWebSocket(
        [
            {"type": "operator_mode_status", "mode": "assist"},
            {"type": "state_update", "state": {"/ch/4/fdr": -12.0}},
            {"type": "operator_mode_status", "mode": "assist"},
            {"type": "operator_proposal_created", "status": "created", "proposal": {"id": "blocked"}},
            {
                "type": "operator_proposal_apply_blocked",
                "status": "apply_blocked",
                "apply_result": {"reason": "operator_mode_blocks_live_write"},
            },
            {"type": "operator_proposal_dismissed", "status": "dismissed", "proposal": {"id": "blocked"}},
            {"type": "operator_mode_status", "mode": "supervised"},
            {"type": "operator_proposal_created", "status": "created", "proposal": {"id": "apply"}},
            {
                "type": "operator_proposal_apply_result",
                "status": "applied",
                "apply_result": {
                    "status": "applied",
                    "channel": 4,
                    "value_type": "fader",
                    "effective_value": -12.25,
                    "old_value": -12.0,
                    "readback_value": -12.25,
                    "readback_status": "verified_from_cache",
                    "rollback_available": True,
                    "rollback_value": -12.0,
                },
            },
            {
                "type": "manual_write_rollback_applied",
                "status": "applied",
                "readback_value": -12.0,
                "readback_status": "verified_from_cache",
            },
            {"type": "emergency_stop_applied", "status": "applied"},
            {"type": "operator_proposal_created", "status": "created", "proposal": {"id": "emergency"}},
            {
                "type": "operator_proposal_apply_blocked",
                "status": "apply_blocked",
                "apply_result": {"reason": "emergency_stop_active"},
            },
            {"type": "operator_proposal_dismissed", "status": "dismissed", "proposal": {"id": "emergency"}},
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)
    args = parse_args(
        tmp_path,
        "--channel",
        "4",
        "--value-type",
        "fader",
        "--delta-db",
        "-0.25",
        "--execute",
        "--prove-blocked-apply",
        "--prove-emergency-stop",
    )

    exit_code = await runner.run(args)

    assert exit_code == 0
    sent_types = [request["type"] for request in websocket.sent]
    assert sent_types == [
        "get_operator_mode",
        "get_state",
        "set_operator_mode",
        "create_operator_proposal",
        "apply_operator_proposal",
        "dismiss_operator_proposal",
        "set_operator_mode",
        "create_operator_proposal",
        "apply_operator_proposal",
        "rollback_last_manual_write",
        "emergency_stop",
        "create_operator_proposal",
        "apply_operator_proposal",
        "dismiss_operator_proposal",
    ]
    assert "set_fader" not in sent_types
    assert "set_gain" not in sent_types
    apply_requests = [request for request in websocket.sent if request["type"] == "apply_operator_proposal"]
    assert all(request["approved"] is True for request in apply_requests)
    assert all(request["approval_id"].startswith(proposal_id_prefix) for request in apply_requests)
    rollback_request = websocket.sent[9]
    assert rollback_request["type"] == "rollback_last_manual_write"
    assert rollback_request["approved"] is True

    summary = read_summary(tmp_path)
    assert summary["status"] == "passed"
    assert summary["reason"] == "proposal_apply_readback_rollback_complete"
    assert summary["write_policy"] == "apply_operator_proposal_only"
    assert summary["apply_result"]["type"] == "operator_proposal_apply_result"
    assert summary["rollback_result"]["type"] == "manual_write_rollback_applied"
    assert summary["post_emergency_apply_result"]["apply_result"]["reason"] == "emergency_stop_active"
