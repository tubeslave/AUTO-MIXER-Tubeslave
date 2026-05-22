"""Offline tests for the Wing supervised smoke-test runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


RUNNER_PATH = Path(__file__).resolve().parents[1] / "tools" / "wing_supervised_smoke_runner.py"
SPEC = importlib.util.spec_from_file_location("wing_supervised_smoke_runner", RUNNER_PATH)
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
        (["--channel", "0", "--value-type", "fader", "--delta-db", "0.1"], "--channel must be in 1..48"),
        (["--channel", "49", "--value-type", "fader", "--delta-db", "0.1"], "--channel must be in 1..48"),
        (["--channel", "1", "--value-type", "fader", "--delta-db", "0"], "--delta-db must be non-zero"),
        (["--channel", "1", "--value-type", "gain", "--delta-db", "0.75"], "--delta-db must be non-zero"),
        (
            ["--channel", "1", "--value-type", "fader", "--delta-db", "0.1", "--prove-emergency-stop"],
            "--prove-emergency-stop requires --execute",
        ),
    ],
)
def test_validate_args_blocks_unsafe_invocations(tmp_path: Path, argv: list[str], expected_error: str) -> None:
    args = parse_args(tmp_path, *argv)

    assert expected_error in runner.validate_args(args)


@pytest.mark.asyncio
async def test_dry_run_plans_without_sending_console_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    websocket = FakeWebSocket(
        [
            {"type": "operator_mode_status", "mode": "assist"},
            {"type": "state_update", "state": {"/ch/7/fdr": -10.0}},
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)
    args = parse_args(tmp_path, "--channel", "7", "--value-type", "fader", "--delta-db", "0.5")

    exit_code = await runner.run(args)

    assert exit_code == 0
    assert websocket.url == "ws://fake-backend"
    assert [request["type"] for request in websocket.sent] == ["get_operator_mode", "get_state"]
    summary = read_summary(tmp_path)
    assert summary["status"] == "planned"
    assert summary["reason"] == "dry_run_only_add_execute_to_write"
    assert summary["operator_mode"] == "assist"
    assert summary["address"] == "/ch/7/fdr"
    assert summary["old_value"] == -10.0
    assert summary["target_value"] == -9.5
    assert [event["event_type"] for event in read_events(summary)] == ["send", "recv", "send", "recv"]


@pytest.mark.asyncio
async def test_execute_path_writes_rolls_back_and_proves_emergency_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [
            {"type": "operator_mode_status", "mode": "assist"},
            {"type": "state_update", "state": {"/ch/4/fdr": -12.0}},
            {"type": "operator_mode_status", "mode": "supervised"},
            {"type": "manual_write_applied", "status": "applied", "address": "/ch/4/fdr", "new_value": -11.75},
            {"type": "manual_write_rollback_applied", "status": "applied", "address": "/ch/4/fdr", "restored_value": -12.0},
            {"type": "emergency_stop_applied", "status": "applied"},
            {"type": "manual_write_blocked", "status": "blocked", "reason": "emergency_stop_active"},
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
        "0.25",
        "--execute",
        "--prove-emergency-stop",
    )

    exit_code = await runner.run(args)

    assert exit_code == 0
    assert [request["type"] for request in websocket.sent] == [
        "get_operator_mode",
        "get_state",
        "set_operator_mode",
        "set_fader",
        "rollback_last_manual_write",
        "emergency_stop",
        "set_fader",
    ]
    assert websocket.sent[3]["approved"] is True
    assert websocket.sent[3]["value"] == -11.75
    assert websocket.sent[6]["value"] == -12.0
    summary = read_summary(tmp_path)
    assert summary["status"] == "passed"
    assert summary["reason"] == "write_readback_rollback_complete"
    assert summary["write_result"]["type"] == "manual_write_applied"
    assert summary["rollback_result"]["type"] == "manual_write_rollback_applied"
    assert summary["post_emergency_write_result"]["reason"] == "emergency_stop_active"
