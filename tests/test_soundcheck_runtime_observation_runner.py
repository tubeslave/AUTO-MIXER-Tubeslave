"""Offline tests for read-only soundcheck runtime observation runner."""

from __future__ import annotations

import importlib.util
import hashlib
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
    def __init__(self, responses: list[Any]):
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


def safe_operator_mode() -> dict[str, Any]:
    return {
        "mode": "analyze",
        "connection_mode": "wing",
        "live_write_policy": "analysis_only",
        "apply_path": "none",
        "capabilities": {
            "can_create_proposals": False,
            "can_apply_to_console": False,
            "can_autonomous_apply": False,
            "allowed_live_write_kinds": [],
            "allowed_live_write_surfaces": [],
        },
    }


def safe_manual_supervision() -> dict[str, Any]:
    return {
        "wing_observation_only": True,
        "operator_mode": "analyze",
        "connection_mode": "wing",
        "approved_live_write_kinds": [],
    }


def safe_dashboard(*, master_signal_present: bool = True) -> dict[str, Any]:
    return {
        "type": "dashboard_snapshot",
        "connection": {"connected": True, "connection_mode": "wing"},
        "operator_mode": safe_operator_mode(),
        "safety": {"manual_write_supervision": safe_manual_supervision()},
        "master_bus": {
            "available": True,
            "signal_present": master_signal_present,
            "source_channels": [47, 48],
            "read_only": True,
        },
        "analyzer": {
            "available": True,
            "source_channels": [47, 48],
            "spectrum_bins": [10, 20, 30],
            "read_only": True,
        },
    }


def safe_safety_monitor() -> dict[str, Any]:
    return {
        "type": "safety_monitor",
        "operator_mode": safe_operator_mode(),
        "manual_write_supervision": safe_manual_supervision(),
        "live_authority_granted": False,
    }


@pytest.mark.asyncio
async def test_runtime_observation_collects_readonly_dashboard_and_meter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [
            {"type": "connection_status", "connected": True, "mode": "wing"},
            {"type": "meter_stream_status", "subscribed": True, "read_only": True, "live_write": False},
            {"type": "operator_mode_status", **safe_operator_mode()},
            safe_dashboard(),
            {"type": "channel_inventory", "channels": [{"id": 1, "name": "Kick"}]},
            {"type": "decision_queue", "items": []},
            safe_safety_monitor(),
            {
                "type": "meter_frame",
                "available": True,
                "read_only": True,
                "live_write": False,
                "signal_present": True,
                "active_channel_count": 1,
                "capture": {"real_input": True},
                "channels": [{"channel": 1, "rms_db": -32.0, "peak_db": -12.0}],
                "master_bus": {
                    "available": True,
                    "signal_present": True,
                    "source_channels": [47, 48],
                },
                "analyzer": {"available": True, "source_channels": [47, 48]},
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
    assert summary["operator_runtime_contract"]["complete"] is True
    assert summary["operator_runtime_contract"]["missing_surfaces"] == []
    assert all(summary["operator_runtime_contract"]["checks"].values())
    assert summary["meter_contract"]["subscribed"] is True
    assert summary["meter_contract"]["read_only_frames"] == 1
    assert summary["meter_contract"]["live_write_frames"] == 0
    assert summary["meter_contract"]["available_frames"] == 1
    assert summary["meter_contract"]["signal_frames"] == 1
    assert summary["meter_contract"]["real_input_frames"] == 1
    assert summary["meter_contract"]["master_source_channels"] == [47, 48]
    assert summary["forbidden_request_types_sent"] == []
    manifest_path = next(tmp_path.glob("*/artifact_manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_path = next(tmp_path.glob("*/summary.json"))
    events_path = next(tmp_path.glob("*/events.jsonl"))
    assert manifest["summary"]["sha256"] == hashlib.sha256(summary_path.read_bytes()).hexdigest()
    assert manifest["events"]["sha256"] == hashlib.sha256(events_path.read_bytes()).hexdigest()
    assert manifest["live_authority_granted"] is False


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
                "master_bus": {"available": True, "source_channels": [47, 48]},
                "analyzer": {"available": True, "source_channels": [47, 48]},
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


@pytest.mark.asyncio
async def test_runtime_observation_blocks_when_operator_runtime_surfaces_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [
            {"type": "meter_stream_status", "subscribed": True, "read_only": True, "live_write": False},
            {
                "type": "dashboard_snapshot",
                "connection": {"connected": True},
                "master_bus": {"available": True, "read_only": True},
                "analyzer": {"available": True, "spectrum_bins": [10, 20, 30], "read_only": True},
            },
            {
                "type": "meter_frame",
                "available": True,
                "read_only": True,
                "live_write": False,
                "signal_present": True,
                "active_channel_count": 1,
                "capture": {"real_input": True},
                "channels": [{"channel": 1, "rms_db": -32.0, "peak_db": -12.0}],
                "master_bus": {
                    "available": True,
                    "signal_present": True,
                    "source_channels": [47, 48],
                },
                "analyzer": {"available": True, "source_channels": [47, 48]},
            },
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)
    args = parse_args(tmp_path, "--max-events", "3")

    exit_code = await runner.run(args)

    assert exit_code == 5
    sent_types = [request["type"] for request in websocket.sent]
    assert not set(sent_types) & runner.WRITE_REQUEST_TYPES

    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "operator_runtime_contract_incomplete"
    contract = summary["operator_runtime_contract"]
    assert contract["complete"] is False
    assert contract["checks"]["connection_state"] is False
    assert contract["checks"]["meter_frame"] is True
    assert contract["checks"]["analyzer_status"] is True
    assert contract["checks"]["master_bus_status"] is True
    assert "operator_mode_status" in contract["missing_surfaces"]
    assert "connection_state" in contract["missing_surfaces"]
    assert "channel_inventory" in contract["missing_surfaces"]
    assert "decision_queue" in contract["missing_surfaces"]
    assert "safety_monitor" in contract["missing_surfaces"]
    assert summary["write_request_sent"] is False
    assert summary["live_apply_request_sent"] is False


@pytest.mark.asyncio
async def test_runtime_observation_bootstraps_and_releases_owned_real_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [
            {"type": "connection_status", "connected": True, "mode": "wing"},
            {"type": "audio_capture_status", "running": False},
            {
                "type": "audio_capture_status",
                "running": True,
                "real_input": True,
                "source_type": "sounddevice",
                "device_name": 1,
                "num_channels": 48,
                "sample_rate": 48000,
            },
            {"type": "meter_stream_status", "subscribed": True, "read_only": True, "live_write": False},
            {"type": "operator_mode_status", **safe_operator_mode()},
            safe_dashboard(),
            {"type": "channel_inventory", "channels": []},
            {"type": "decision_queue", "items": []},
            safe_safety_monitor(),
            {
                "type": "meter_frame",
                "available": True,
                "read_only": True,
                "live_write": False,
                "signal_present": True,
                "active_channel_count": 43,
                "capture": {"real_input": True},
                "channels": [{"channel": 3, "rms_db": -20.0, "peak_db": -8.0}],
                "master_bus": {
                    "available": True,
                    "signal_present": True,
                    "source_channels": [47, 48],
                },
                "analyzer": {"available": True, "source_channels": [47, 48]},
            },
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)
    args = parse_args(
        tmp_path,
        "--audio-device-id",
        "1",
        "--max-events",
        "10",
    )

    exit_code = await runner.run(args)

    assert exit_code == 0
    sent_types = [request["type"] for request in websocket.sent]
    assert sent_types == [
        "get_audio_capture_status",
        "start_audio_capture_preview",
        "subscribe_meters",
        "get_operator_mode",
        "get_dashboard_snapshot",
        "get_channel_inventory",
        "get_decision_queue",
        "get_safety_monitor",
        "unsubscribe_meters",
        "stop_audio_capture_preview",
    ]
    assert not set(sent_types) & runner.WRITE_REQUEST_TYPES
    summary = read_summary(tmp_path)
    assert summary["audio_capture_bootstrap"]["started_by_runner"] is True
    assert summary["audio_capture_bootstrap"]["verified_real_input"] is True
    assert summary["meter_contract"]["max_active_channel_count"] == 43


@pytest.mark.asyncio
async def test_runtime_observation_rejects_available_but_silent_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [
            {"type": "connection_status", "connected": True, "mode": "wing"},
            {"type": "meter_stream_status", "subscribed": True, "read_only": True, "live_write": False},
            {"type": "operator_mode_status", **safe_operator_mode()},
            safe_dashboard(master_signal_present=False),
            {"type": "channel_inventory", "channels": []},
            {"type": "decision_queue", "items": []},
            safe_safety_monitor(),
            {
                "type": "meter_frame",
                "available": True,
                "read_only": True,
                "live_write": False,
                "signal_present": False,
                "active_channel_count": 0,
                "capture": {"real_input": True},
                "channels": [{"channel": 1, "rms_db": -200.0, "peak_db": -200.0}],
                "master_bus": {
                    "available": True,
                    "signal_present": False,
                    "source_channels": [47, 48],
                },
                "analyzer": {"available": True, "source_channels": [47, 48]},
            },
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)
    args = parse_args(tmp_path, "--max-events", "8")

    exit_code = await runner.run(args)

    assert exit_code == 5
    summary = read_summary(tmp_path)
    assert summary["meter_contract"]["available_frames"] == 1
    assert summary["meter_contract"]["signal_frames"] == 0
    assert summary["operator_runtime_contract"]["checks"]["meter_signal_present"] is False


@pytest.mark.asyncio
async def test_runtime_observation_rejects_disconnected_wing_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard = safe_dashboard()
    dashboard["connection"]["connected"] = False
    websocket = FakeWebSocket(
        [
            {"type": "connection_status", "connected": False, "mode": "wing"},
            {"type": "meter_stream_status", "subscribed": True, "read_only": True, "live_write": False},
            {"type": "operator_mode_status", **safe_operator_mode()},
            dashboard,
            {"type": "channel_inventory", "channels": []},
            {"type": "decision_queue", "items": []},
            safe_safety_monitor(),
            {
                "type": "meter_frame",
                "available": True,
                "read_only": True,
                "live_write": False,
                "signal_present": True,
                "active_channel_count": 1,
                "capture": {"real_input": True},
                "channels": [{"channel": 3, "peak_db": -8.0}],
                "master_bus": {
                    "signal_present": True,
                    "source_channels": [47, 48],
                },
                "analyzer": {"source_channels": [47, 48]},
            },
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)

    exit_code = await runner.run(parse_args(tmp_path, "--max-events", "8"))

    assert exit_code == 5
    summary = read_summary(tmp_path)
    assert summary["operator_runtime_contract"]["checks"]["connection_state"] is False
    assert summary["runtime_truth"]["connection_status"]["connected"] is False
    assert summary["runtime_truth"]["dashboard_connection"]["connected"] is False


@pytest.mark.asyncio
async def test_runtime_observation_rejects_wrong_existing_audio_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket(
        [
            {
                "type": "audio_capture_status",
                "running": True,
                "real_input": True,
                "source_type": "sounddevice",
                "device_name": 999,
                "num_channels": 48,
                "sample_rate": 48000,
            }
        ]
    )
    patch_websocket_connect(monkeypatch, websocket)
    args = parse_args(tmp_path, "--audio-device-id", "1")

    exit_code = await runner.run(args)

    assert exit_code == 6
    assert websocket.sent == [{"type": "get_audio_capture_status"}]
    summary = read_summary(tmp_path)
    assert summary["reason"] == "existing_audio_capture_contract_mismatch"
    assert summary["audio_capture_bootstrap"]["device_identity_matches"] is False
    assert summary["audio_capture_bootstrap"]["requested_device_identity"] == ["index", 1]
    assert summary["audio_capture_bootstrap"]["observed_device_identity"] == ["index", 999]


@pytest.mark.asyncio
async def test_runtime_observation_records_websocket_close_as_blocked_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Closed(Exception):
        pass

    monkeypatch.setattr(runner.websockets.exceptions, "ConnectionClosed", Closed)
    websocket = FakeWebSocket([Closed("backend disconnected")])
    patch_websocket_connect(monkeypatch, websocket)
    args = parse_args(tmp_path)

    exit_code = await runner.run(args)

    assert exit_code == 1
    summary = read_summary(tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "backend_connection_or_runtime_observation_failed"
    assert summary["transport_failure"]["type"] == "Closed"
    assert next(tmp_path.glob("*/artifact_manifest.json")).is_file()
