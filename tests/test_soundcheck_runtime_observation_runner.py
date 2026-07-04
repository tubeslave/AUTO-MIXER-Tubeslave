"""Offline tests for read-only soundcheck runtime observation runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "soundcheck_runtime_observation_runner.py"
)
SPEC = importlib.util.spec_from_file_location("soundcheck_runtime_observation_runner", RUNNER_PATH)
assert SPEC is not None
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


class FakeWebSocket:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.sent: list[dict[str, Any]] = []
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
            "--duration-sec",
            "0",
            "--recv-timeout-sec",
            "0.1",
            *argv,
        ]
    )


def read_summary(artifact_root: Path) -> dict[str, Any]:
    summaries = list(artifact_root.glob("*/summary.json"))
    assert len(summaries) == 1
    return json.loads(summaries[0].read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_runtime_observation_collects_readonly_dashboard_and_meter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [
            {"type": "connection_status", "connected": True},
            {"type": "meter_stream_status", "subscribed": True, "read_only": True, "live_write": False},
            {"type": "operator_mode_status", "mode": "assist"},
            {
                "type": "dashboard_snapshot",
                "connection": {"connected": True},
                "master_bus": {"available": True, "read_only": True},
                "analyzer": {"available": True, "spectrum_bins": [10, 20, 30], "read_only": True},
            },
            {"type": "channel_inventory", "channels": [{"id": 1, "name": "Kick"}]},
            {"type": "decision_queue", "items": []},
            {"type": "safety_monitor", "live_authority_granted": False},
            {
                "type": "meter_frame",
                "available": True,
                "read_only": True,
                "live_write": False,
                "channels": [{"channel": 1, "rms_db": -32.0, "peak_db": -12.0}],
            },
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)
    args = parse_args(tmp_path, "--max-events", "8")

    exit_code = await runner.run(args)

    assert exit_code == 0
    sent_types = [request["type"] for request in websocket.sent]
    assert sent_types == [
        "subscribe_meters",
        "get_operator_mode",
        "get_dashboard_snapshot",
        "get_channel_inventory",
        "get_decision_queue",
        "get_safety_monitor",
        "unsubscribe_meters",
    ]
    assert not set(sent_types) & runner.WRITE_REQUEST_TYPES

    summary = read_summary(tmp_path)
    assert summary["status"] == "passed"
    assert summary["reason"] == "read_only_soundcheck_runtime_observation_passed"
    assert summary["write_request_sent"] is False
    assert summary["live_apply_request_sent"] is False
    assert summary["dashboard_contract"]["has_connection"] is True
    assert summary["dashboard_contract"]["has_master_bus"] is True
    assert summary["dashboard_contract"]["has_analyzer"] is True
    assert summary["dashboard_contract"]["analyzer_available_count"] == 1
    assert summary["meter_contract"]["subscribed"] is True
    assert summary["meter_contract"]["read_only_frames"] == 1
    assert summary["meter_contract"]["live_write_frames"] == 0
    assert summary["meter_contract"]["available_frames"] == 1
    assert summary["forbidden_request_types_sent"] == []


@pytest.mark.asyncio
async def test_runtime_observation_marks_unexpected_live_apply_for_manual_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [
            {"type": "meter_stream_status", "subscribed": True, "read_only": True, "live_write": False},
            {
                "type": "dashboard_snapshot",
                "connection": {"connected": True},
                "master_bus": {"available": True},
                "analyzer": {"available": True},
            },
            {
                "type": "meter_frame",
                "available": True,
                "read_only": True,
                "live_write": False,
                "channels": [],
            },
            {
                "type": "operator_proposal_apply_result",
                "status": "applied",
                "apply_result": {"status": "applied"},
            },
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)
    args = parse_args(tmp_path, "--max-events", "4")

    exit_code = await runner.run(args)

    assert exit_code == 4
    sent_types = [request["type"] for request in websocket.sent]
    assert "apply_operator_proposal" not in sent_types
    assert not set(sent_types) & runner.WRITE_REQUEST_TYPES

    summary = read_summary(tmp_path)
    assert summary["status"] == "manual_review_required"
    assert summary["reason"] == "unexpected_live_event_observed_during_runtime_monitor"
    assert summary["write_request_sent"] is False
    assert summary["live_apply_request_sent"] is False
    assert summary["counts"]["unexpected_live_event"] == 1
    assert summary["unexpected_live_events"][0]["type"] == "operator_proposal_apply_result"
