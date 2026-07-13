#!/usr/bin/env python3
"""Collect read-only soundcheck runtime telemetry over the backend WebSocket.

The runner is intentionally telemetry-only. It never creates proposals, never
applies proposals, never changes operator mode, and never sends direct mixer
write requests. It writes an artifact that can prove the dashboard/meter loop
was alive while live-authority surfaces stayed quiet.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import websockets


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = "artifacts/soundcheck_runtime_observation"

READ_REQUESTS = (
    {"type": "get_operator_mode"},
    {"type": "get_dashboard_snapshot"},
    {"type": "get_channel_inventory", "total_channels": 48},
    {"type": "get_decision_queue", "limit": 50},
    {"type": "get_safety_monitor"},
)
WRITE_REQUEST_TYPES = {
    "set_operator_mode",
    "create_operator_proposal",
    "accept_operator_proposal",
    "dismiss_operator_proposal",
    "apply_operator_proposal",
    "execute_operator_proposal_apply_sequence",
    "execute_operator_proposal_apply_sequence_policy_memory_trust_promotion",
    "connect_wing",
    "disconnect_wing",
    "disconnect_mixer",
    "set_fader",
    "set_gain",
    "set_mute",
    "mute_channel",
    "unmute_channel",
    "set_pan",
    "set_eq",
    "set_compressor",
    "set_gate",
    "set_fx",
    "route_output",
    "load_snap",
    "save_snap",
    "rollback_last_manual_write",
    "emergency_stop",
}
LIVE_EVENT_TYPES = {
    "operator_proposal_apply_result",
    "manual_write_rollback_applied",
    "manual_write_rollback_blocked",
    "manual_write_rollback_error",
    "emergency_stop_applied",
    "voice_command_executed",
}
RUNTIME_CONTRACT_SURFACES = (
    "operator_mode_status",
    "connection_state",
    "dashboard_snapshot",
    "channel_inventory",
    "decision_queue",
    "safety_monitor",
    "meter_subscription",
    "meter_frame",
    "meter_channel_levels",
    "meter_signal_present",
    "real_audio_capture",
    "analyzer_status",
    "master_bus_status",
    "master_signal_present",
    "master_channels_47_48_observed",
)
SIGNAL_FLOOR_DB = -90.0
EXPECTED_MASTER_CHANNELS = [47, 48]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def json_default(value: Any) -> Any:
    try:
        return str(value)
    except Exception:
        return "<unserializable>"


class EventRecorder:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.events_path = run_dir / "events.jsonl"
        self.summary_path = run_dir / "summary.json"
        self.manifest_path = run_dir / "artifact_manifest.json"
        self.seq = 0
        self.sent_request_types: list[str] = []
        self.recv_event_types: list[str] = []

    def record(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.seq += 1
        payload_type = str(payload.get("type") or "unknown")
        if event_type == "send":
            self.sent_request_types.append(payload_type)
        if event_type == "recv":
            self.recv_event_types.append(payload_type)
        event = {
            "seq": self.seq,
            "timestamp": datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "event_type": event_type,
            "payload": payload,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True, default=json_default))
            handle.write("\n")

    def write_summary(self, summary: Dict[str, Any]) -> None:
        summary = dict(summary)
        sent_request_types = list(self.sent_request_types)
        recv_event_types = list(self.recv_event_types)
        forbidden_sent = sorted({item for item in sent_request_types if item in WRITE_REQUEST_TYPES})
        summary.setdefault("events_path", str(self.events_path))
        summary.setdefault("summary_path", str(self.summary_path))
        summary.setdefault("sent_request_types", sent_request_types)
        summary.setdefault("received_event_types", recv_event_types)
        summary.setdefault("forbidden_request_types_sent", forbidden_sent)
        summary.setdefault("write_request_sent", bool(forbidden_sent))
        summary.setdefault("live_apply_request_sent", "apply_operator_proposal" in sent_request_types)
        events_bytes = self.events_path.read_bytes() if self.events_path.exists() else b""
        summary.setdefault("events_size", len(events_bytes))
        summary.setdefault("events_sha256", hashlib.sha256(events_bytes).hexdigest())
        summary.setdefault("artifact_manifest_path", str(self.manifest_path))
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True, default=json_default)
            handle.write("\n")
        summary_bytes = self.summary_path.read_bytes()
        manifest = {
            "schema": "soundcheck_runtime_observation_artifact_manifest.v1",
            "run_id": summary.get("run_id"),
            "summary": {
                "path": str(self.summary_path),
                "size": len(summary_bytes),
                "sha256": hashlib.sha256(summary_bytes).hexdigest(),
            },
            "events": {
                "path": str(self.events_path),
                "size": len(events_bytes),
                "sha256": hashlib.sha256(events_bytes).hexdigest(),
            },
            "read_only": True,
            "mutates_console_state": False,
            "live_write": False,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
        }
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect read-only soundcheck dashboard/meter telemetry."
    )
    parser.add_argument("--url", default="ws://localhost:8765")
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--sample-interval-sec", type=float, default=5.0)
    parser.add_argument("--recv-timeout-sec", type=float, default=0.5)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--reason", default="soundcheck_runtime_observation")
    parser.add_argument(
        "--audio-device-id",
        default=None,
        help="Optional real input device to start read-only capture before metering.",
    )
    parser.add_argument("--audio-num-channels", type=int, default=48)
    parser.add_argument("--audio-sample-rate", type=int, default=48000)
    parser.add_argument("--audio-start-timeout-sec", type=float, default=5.0)
    return parser


def validate_args(args: argparse.Namespace) -> Optional[str]:
    if args.duration_sec < 0 or args.duration_sec > 3600:
        return "--duration-sec must be between 0 and 3600"
    if args.sample_interval_sec <= 0 or args.sample_interval_sec > 120:
        return "--sample-interval-sec must be between 0 and 120"
    if args.recv_timeout_sec <= 0 or args.recv_timeout_sec > 10:
        return "--recv-timeout-sec must be between 0 and 10"
    if args.max_events < 0:
        return "--max-events must be non-negative"
    if args.audio_num_channels < 1 or args.audio_num_channels > 64:
        return "--audio-num-channels must be between 1 and 64"
    if args.audio_sample_rate < 8000 or args.audio_sample_rate > 192000:
        return "--audio-sample-rate must be between 8000 and 192000"
    if args.audio_start_timeout_sec <= 0 or args.audio_start_timeout_sec > 30:
        return "--audio-start-timeout-sec must be between 0 and 30"
    return None


def _is_unexpected_live_event(payload: Dict[str, Any]) -> bool:
    payload_type = payload.get("type")
    if payload_type in LIVE_EVENT_TYPES:
        if payload_type == "operator_proposal_apply_result":
            return payload.get("status") == "applied" or (
                isinstance(payload.get("apply_result"), dict)
                and payload["apply_result"].get("status") == "applied"
            )
        return True
    if payload.get("live_write") is True:
        return True
    return False


def _dashboard_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload.get("type") != "dashboard_snapshot":
        return {}
    analyzer = payload.get("analyzer")
    master = payload.get("master_bus")
    connection = payload.get("connection")
    return {
        "has_connection": isinstance(connection, dict),
        "connection_connected": bool(
            isinstance(connection, dict) and connection.get("connected") is True
        ),
        "connection_mode": (
            str(connection.get("connection_mode") or connection.get("mode") or "")
            if isinstance(connection, dict)
            else ""
        ),
        "has_master_bus": isinstance(master, dict),
        "has_analyzer": isinstance(analyzer, dict),
        "analyzer_available": bool(isinstance(analyzer, dict) and analyzer.get("available")),
        "master_bus_available": bool(isinstance(master, dict) and master.get("available")),
        "analyzer_source_channels": (
            list(analyzer.get("source_channels") or []) if isinstance(analyzer, dict) else []
        ),
        "master_source_channels": (
            list(master.get("source_channels") or []) if isinstance(master, dict) else []
        ),
        "master_signal_present": bool(
            isinstance(master, dict) and master.get("signal_present") is True
        ),
    }


def _normalized_device_identity(value: Any) -> tuple[str, Any]:
    """Normalize sounddevice indices/names without conflating different devices."""
    if value is None:
        return ("none", None)
    if isinstance(value, bool):
        return ("name", str(value).casefold())
    if isinstance(value, int):
        return ("index", value)
    text = str(value).strip()
    if text.isdigit():
        return ("index", int(text))
    return ("name", text.casefold())


def _operator_mode_truth(payload: Any) -> Dict[str, bool]:
    value = payload if isinstance(payload, dict) else {}
    capabilities = value.get("capabilities")
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    return {
        "mode_analyze": value.get("mode") == "analyze",
        "connection_mode_wing": value.get("connection_mode") == "wing",
        "live_write_policy_analysis_only": value.get("live_write_policy") == "analysis_only",
        "apply_path_none": value.get("apply_path") == "none",
        "can_create_proposals_false": capabilities.get("can_create_proposals") is False,
        "can_apply_to_console_false": capabilities.get("can_apply_to_console") is False,
        "can_autonomous_apply_false": capabilities.get("can_autonomous_apply") is False,
        "allowed_live_write_kinds_empty": (
            isinstance(capabilities.get("allowed_live_write_kinds"), list)
            and capabilities.get("allowed_live_write_kinds") == []
        ),
        "allowed_live_write_surfaces_empty": (
            isinstance(capabilities.get("allowed_live_write_surfaces"), list)
            and capabilities.get("allowed_live_write_surfaces") == []
        ),
    }


def _manual_supervision_truth(payload: Any) -> Dict[str, bool]:
    value = payload if isinstance(payload, dict) else {}
    return {
        "wing_observation_only": value.get("wing_observation_only") is True,
        "operator_mode_analyze": value.get("operator_mode") == "analyze",
        "connection_mode_wing": value.get("connection_mode") == "wing",
        "approved_live_write_kinds_empty": (
            isinstance(value.get("approved_live_write_kinds"), list)
            and value.get("approved_live_write_kinds") == []
        ),
    }


def _record_runtime_truth(summary: Dict[str, Any], payload: Dict[str, Any]) -> None:
    payload_type = payload.get("type")
    truth = summary["runtime_truth"]
    if payload_type == "connection_status":
        truth["connection_status"] = {
            "connected": payload.get("connected") is True,
            "connection_mode_wing": (
                payload.get("connection_mode") == "wing" or payload.get("mode") == "wing"
            ),
        }
    elif payload_type == "operator_mode_status":
        truth["operator_mode_status"] = _operator_mode_truth(payload)
    elif payload_type == "dashboard_snapshot":
        connection = payload.get("connection")
        connection = connection if isinstance(connection, dict) else {}
        truth["dashboard_connection"] = {
            "connected": connection.get("connected") is True,
            "connection_mode_wing": (
                connection.get("connection_mode") == "wing" or connection.get("mode") == "wing"
            ),
        }
        truth["dashboard_operator_mode"] = _operator_mode_truth(payload.get("operator_mode"))
        safety = payload.get("safety")
        safety = safety if isinstance(safety, dict) else {}
        truth["dashboard_manual_write_supervision"] = _manual_supervision_truth(
            safety.get("manual_write_supervision")
        )
    elif payload_type == "safety_monitor":
        truth["safety_operator_mode"] = _operator_mode_truth(payload.get("operator_mode"))
        truth["safety_manual_write_supervision"] = _manual_supervision_truth(
            payload.get("manual_write_supervision")
        )


def _all_truthy(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(item is True for item in value.values())


def _operator_runtime_contract(summary: Dict[str, Any]) -> Dict[str, Any]:
    counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
    dashboard = (
        summary.get("dashboard_contract")
        if isinstance(summary.get("dashboard_contract"), dict)
        else {}
    )
    meter = summary.get("meter_contract") if isinstance(summary.get("meter_contract"), dict) else {}
    runtime_truth = (
        summary.get("runtime_truth") if isinstance(summary.get("runtime_truth"), dict) else {}
    )
    checks = {
        "operator_mode_status": (
            int(counts.get("operator_mode_status") or 0) > 0
            and _all_truthy(runtime_truth.get("operator_mode_status"))
            and _all_truthy(runtime_truth.get("dashboard_operator_mode"))
        ),
        "connection_state": (
            _all_truthy(runtime_truth.get("connection_status"))
            and _all_truthy(runtime_truth.get("dashboard_connection"))
        ),
        "dashboard_snapshot": int(counts.get("dashboard_snapshot") or 0) > 0,
        "channel_inventory": int(counts.get("channel_inventory") or 0) > 0,
        "decision_queue": int(counts.get("decision_queue") or 0) > 0,
        "safety_monitor": (
            int(counts.get("safety_monitor") or 0) > 0
            and _all_truthy(runtime_truth.get("safety_operator_mode"))
            and _all_truthy(runtime_truth.get("safety_manual_write_supervision"))
            and _all_truthy(runtime_truth.get("dashboard_manual_write_supervision"))
        ),
        "meter_subscription": meter.get("subscribed") is True,
        "meter_frame": int(counts.get("meter_frame") or 0) > 0
        and int(meter.get("read_only_frames") or 0) > 0,
        "meter_channel_levels": int(meter.get("max_channel_count") or 0) > 0,
        "meter_signal_present": int(meter.get("signal_frames") or 0) > 0
        and int(meter.get("max_active_channel_count") or 0) > 0,
        "real_audio_capture": int(meter.get("real_input_frames") or 0) > 0,
        "analyzer_status": bool(dashboard.get("has_analyzer")),
        "master_bus_status": bool(dashboard.get("has_master_bus")),
        "master_signal_present": int(meter.get("master_signal_frames") or 0) > 0
        and int(dashboard.get("master_signal_present_count") or 0) > 0,
        "master_channels_47_48_observed": (
            meter.get("master_source_channels") == EXPECTED_MASTER_CHANNELS
            and meter.get("analyzer_source_channels") == EXPECTED_MASTER_CHANNELS
            and int(meter.get("master_signal_frames") or 0) > 0
        ),
    }
    missing = [surface for surface in RUNTIME_CONTRACT_SURFACES if not checks.get(surface)]
    return {
        "required_surfaces": list(RUNTIME_CONTRACT_SURFACES),
        "checks": checks,
        "missing_surfaces": missing,
        "complete": not missing,
        "read_only": True,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
    }


async def _send(ws: Any, recorder: EventRecorder, payload: Dict[str, Any]) -> None:
    recorder.record("send", payload)
    await ws.send(json.dumps(payload))


async def _send_sample_requests(ws: Any, recorder: EventRecorder) -> None:
    for request in READ_REQUESTS:
        await _send(ws, recorder, dict(request))


async def _recv_payload(
    ws: Any,
    recorder: EventRecorder,
    *,
    timeout_sec: float,
) -> Dict[str, Any]:
    raw = await asyncio.wait_for(ws.recv(), timeout=timeout_sec)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"type": "invalid_json", "raw": raw}
    if not isinstance(payload, dict):
        payload = {"type": "non_object_payload", "payload": payload}
    recorder.record("recv", payload)
    return payload


async def _wait_for_audio_capture_status(
    ws: Any,
    recorder: EventRecorder,
    summary: Dict[str, Any],
    *,
    timeout_sec: float,
) -> Optional[Dict[str, Any]]:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        payload = await _recv_payload(
            ws,
            recorder,
            timeout_sec=max(0.01, deadline - time.monotonic()),
        )
        if _is_unexpected_live_event(payload):
            summary["counts"]["unexpected_live_event"] += 1
            summary["unexpected_live_events"].append(payload)
        payload_type = str(payload.get("type") or "unknown")
        if payload_type in summary["counts"]:
            summary["counts"][payload_type] += 1
        _record_runtime_truth(summary, payload)
        if payload.get("type") == "audio_capture_status":
            return payload
    return None


async def _bootstrap_real_audio_capture(
    ws: Any,
    recorder: EventRecorder,
    summary: Dict[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, bool, Optional[str]]:
    """Start a real local input only when none is already running.

    Returns ``(ready, started_by_runner, failure_reason)``. This controls only
    the Mac input stream; it never sends a mixer or WING write request.
    """
    if args.audio_device_id is None:
        return True, False, None

    await _send(ws, recorder, {"type": "get_audio_capture_status"})
    status = await _wait_for_audio_capture_status(
        ws,
        recorder,
        summary,
        timeout_sec=args.audio_start_timeout_sec,
    )
    if status is None:
        return False, False, "audio_capture_status_timeout"

    started_by_runner = False
    if not status.get("running"):
        await _send(
            ws,
            recorder,
            {
                "type": "start_audio_capture_preview",
                "device_id": args.audio_device_id,
                "num_channels": args.audio_num_channels,
                "sample_rate": args.audio_sample_rate,
            },
        )
        started_by_runner = True

    deadline = time.monotonic() + args.audio_start_timeout_sec
    while True:
        if started_by_runner or not status.get("running"):
            status = await _wait_for_audio_capture_status(
                ws,
                recorder,
                summary,
                timeout_sec=max(0.01, deadline - time.monotonic()),
            )
            started_by_runner = True
        if status is None:
            return False, started_by_runner, "audio_capture_start_timeout"
        summary["audio_capture_bootstrap"]["last_status"] = status
        real_input = status.get("real_input") is True
        channel_count = int(status.get("num_channels") or 0)
        sample_rate = int(status.get("sample_rate") or 0)
        requested_device = _normalized_device_identity(args.audio_device_id)
        observed_device = _normalized_device_identity(status.get("device_name"))
        device_matches = requested_device == observed_device
        summary["audio_capture_bootstrap"].update(
            {
                "requested_device_identity": list(requested_device),
                "observed_device_identity": list(observed_device),
                "device_identity_matches": device_matches,
            }
        )
        if (
            status.get("running") is True
            and real_input
            and channel_count >= args.audio_num_channels
            and sample_rate == args.audio_sample_rate
            and device_matches
        ):
            summary["audio_capture_bootstrap"].update(
                {
                    "started_by_runner": started_by_runner,
                    "verified_real_input": True,
                    "status": status,
                }
            )
            return True, started_by_runner, None
        if status.get("error"):
            return False, started_by_runner, "real_audio_capture_start_failed"
        if not started_by_runner:
            return False, False, "existing_audio_capture_contract_mismatch"
        if time.monotonic() >= deadline:
            return False, True, "real_audio_capture_not_fresh"
        await asyncio.sleep(0.1)
        await _send(ws, recorder, {"type": "get_audio_capture_status"})


async def run(args: argparse.Namespace) -> int:
    run_id = f"{utc_stamp()}_soundcheck_runtime_observation"
    run_dir = Path(args.artifact_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    recorder = EventRecorder(run_dir)
    started_monotonic = time.monotonic()
    deadline = started_monotonic + float(args.duration_sec)
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "schema": "soundcheck_runtime_observation.v1",
        "runner": "soundcheck_runtime_observation_runner",
        "status": "started",
        "started_at": iso_now(),
        "url": args.url,
        "reason": args.reason,
        "duration_target_sec": args.duration_sec,
        "sample_interval_sec": args.sample_interval_sec,
        "read_only": True,
        "live_write": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "readiness_matrix_update_allowed": False,
        "m005_closure_allowed": False,
        "m006_closure_allowed": False,
        "proof_scope": "runtime_telemetry_only_no_wing_readiness",
        "mutates_console_state": False,
        "local_audio_capture_control_only": bool(args.audio_device_id is not None),
        "audio_capture_bootstrap": {
            "requested": args.audio_device_id is not None,
            "device_id": args.audio_device_id,
            "num_channels": args.audio_num_channels,
            "sample_rate": args.audio_sample_rate,
            "started_by_runner": False,
            "verified_real_input": False,
        },
        "runtime_truth": {
            "connection_status": {},
            "dashboard_connection": {},
            "operator_mode_status": {},
            "dashboard_operator_mode": {},
            "safety_operator_mode": {},
            "safety_manual_write_supervision": {},
            "dashboard_manual_write_supervision": {},
        },
        "counts": {
            "dashboard_snapshot": 0,
            "meter_frame": 0,
            "channel_inventory": 0,
            "decision_queue": 0,
            "safety_monitor": 0,
            "operator_mode_status": 0,
            "connection_status": 0,
            "unexpected_live_event": 0,
        },
        "dashboard_contract": {
            "has_connection": False,
            "connection_connected": False,
            "connection_mode": "",
            "has_master_bus": False,
            "has_analyzer": False,
            "analyzer_available_count": 0,
            "master_bus_available_count": 0,
            "analyzer_source_channels": [],
            "master_source_channels": [],
            "master_signal_present_count": 0,
        },
        "meter_contract": {
            "subscribed": False,
            "read_only_frames": 0,
            "live_write_frames": 0,
            "available_frames": 0,
            "max_channel_count": 0,
            "signal_frames": 0,
            "real_input_frames": 0,
            "max_active_channel_count": 0,
            "max_peak_db": None,
            "master_signal_frames": 0,
            "master_source_channels": [],
            "analyzer_source_channels": [],
        },
        "unexpected_live_events": [],
    }

    validation_error = validate_args(args)
    if validation_error:
        summary.update({"status": "blocked", "reason": validation_error})
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return 2

    try:
        async with websockets.connect(args.url) as ws:
            capture_started_by_runner = False
            subscribed = False
            ready, capture_started_by_runner, capture_failure = (
                await _bootstrap_real_audio_capture(ws, recorder, summary, args)
            )
            if not ready:
                if capture_started_by_runner:
                    await _send(ws, recorder, {"type": "stop_audio_capture_preview"})
                summary.update(
                    {
                        "status": "blocked",
                        "reason": capture_failure or "real_audio_capture_unavailable",
                    }
                )
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
                return 6

            try:
                await _send(ws, recorder, {"type": "subscribe_meters"})
                subscribed = True
                await _send_sample_requests(ws, recorder)
                next_sample = time.monotonic() + float(args.sample_interval_sec)
                while True:
                    now = time.monotonic()
                    if args.max_events and len(recorder.recv_event_types) >= args.max_events:
                        break
                    if now >= deadline and args.duration_sec > 0:
                        break
                    if now >= next_sample:
                        await _send_sample_requests(ws, recorder)
                        next_sample = now + float(args.sample_interval_sec)
                    timeout = min(float(args.recv_timeout_sec), max(0.01, deadline - now))
                    try:
                        payload = await _recv_payload(
                            ws,
                            recorder,
                            timeout_sec=timeout,
                        )
                    except asyncio.TimeoutError:
                        if args.duration_sec == 0:
                            break
                        continue
                    payload_type = str(payload.get("type") or "unknown")
                    if payload_type in summary["counts"]:
                        summary["counts"][payload_type] += 1
                    if _is_unexpected_live_event(payload):
                        summary["counts"]["unexpected_live_event"] += 1
                        summary["unexpected_live_events"].append(payload)

                    _record_runtime_truth(summary, payload)

                    if payload_type == "meter_stream_status":
                        summary["meter_contract"]["subscribed"] = bool(payload.get("subscribed"))
                        if payload.get("live_write") is True:
                            summary["meter_contract"]["live_write_frames"] += 1
                    elif payload_type == "meter_frame":
                        channels = (
                            payload.get("channels")
                            if isinstance(payload.get("channels"), list)
                            else []
                        )
                        meter = summary["meter_contract"]
                        meter["max_channel_count"] = max(
                            int(meter["max_channel_count"]),
                            len(channels),
                        )
                        if payload.get("read_only") is True:
                            meter["read_only_frames"] += 1
                        if payload.get("live_write") is True:
                            meter["live_write_frames"] += 1
                        if payload.get("available") is True:
                            meter["available_frames"] += 1

                        peak_values = []
                        for channel in channels:
                            try:
                                peak = float(channel.get("peak_db"))
                            except (AttributeError, TypeError, ValueError):
                                continue
                            peak_values.append(peak)
                        active_count = int(payload.get("active_channel_count") or 0)
                        if not active_count:
                            active_count = sum(peak > SIGNAL_FLOOR_DB for peak in peak_values)
                        meter["max_active_channel_count"] = max(
                            int(meter["max_active_channel_count"]),
                            active_count,
                        )
                        if payload.get("signal_present") is True or active_count > 0:
                            meter["signal_frames"] += 1
                        if peak_values:
                            frame_peak = max(peak_values)
                            previous_peak = meter.get("max_peak_db")
                            meter["max_peak_db"] = (
                                frame_peak
                                if previous_peak is None
                                else max(float(previous_peak), frame_peak)
                            )

                        capture = payload.get("capture") if isinstance(payload.get("capture"), dict) else {}
                        if capture.get("real_input") is True:
                            meter["real_input_frames"] += 1
                        master = payload.get("master_bus") if isinstance(payload.get("master_bus"), dict) else {}
                        analyzer = payload.get("analyzer") if isinstance(payload.get("analyzer"), dict) else {}
                        master_sources = list(master.get("source_channels") or [])
                        analyzer_sources = list(analyzer.get("source_channels") or [])
                        if master_sources:
                            meter["master_source_channels"] = master_sources
                        if analyzer_sources:
                            meter["analyzer_source_channels"] = analyzer_sources
                        if master.get("signal_present") is True:
                            meter["master_signal_frames"] += 1
                    elif payload_type == "dashboard_snapshot":
                        fields = _dashboard_fields(payload)
                        dashboard = summary["dashboard_contract"]
                        for key in (
                            "has_connection",
                            "connection_connected",
                            "has_master_bus",
                            "has_analyzer",
                        ):
                            dashboard[key] = bool(dashboard[key] or fields.get(key))
                        if fields.get("connection_mode"):
                            dashboard["connection_mode"] = fields["connection_mode"]
                        if fields.get("analyzer_available"):
                            dashboard["analyzer_available_count"] += 1
                        if fields.get("master_bus_available"):
                            dashboard["master_bus_available_count"] += 1
                        if fields.get("master_signal_present"):
                            dashboard["master_signal_present_count"] += 1
                        if fields.get("master_source_channels"):
                            dashboard["master_source_channels"] = fields["master_source_channels"]
                        if fields.get("analyzer_source_channels"):
                            dashboard["analyzer_source_channels"] = fields["analyzer_source_channels"]
            finally:
                if subscribed:
                    await _send(ws, recorder, {"type": "unsubscribe_meters"})
                if capture_started_by_runner:
                    await _send(ws, recorder, {"type": "stop_audio_capture_preview"})
    except (
        OSError,
        TimeoutError,
        ConnectionError,
        websockets.exceptions.ConnectionClosed,
    ) as exc:
        summary.update(
            {
                "status": "blocked",
                "reason": "backend_connection_or_runtime_observation_failed",
                "transport_failure": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return 1
    finally:
        summary["duration_actual_sec"] = round(time.monotonic() - started_monotonic, 3)
        summary["finished_at"] = iso_now()

    forbidden_sent = sorted(
        {item for item in recorder.sent_request_types if item in WRITE_REQUEST_TYPES}
    )
    if forbidden_sent:
        summary.update(
            {
                "status": "failed",
                "reason": "observation_runner_sent_forbidden_request",
                "forbidden_request_types_sent": forbidden_sent,
            }
        )
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return 3
    if summary["counts"]["unexpected_live_event"]:
        summary.update(
            {
                "status": "manual_review_required",
                "reason": "unexpected_live_event_observed_during_runtime_monitor",
            }
        )
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return 4

    runtime_contract = _operator_runtime_contract(summary)
    summary["operator_runtime_contract"] = runtime_contract
    if not runtime_contract["complete"]:
        summary.update(
            {
                "status": "blocked",
                "reason": "operator_runtime_contract_incomplete",
            }
        )
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return 5

    summary.update(
        {
            "status": "passed",
            "reason": "read_only_soundcheck_runtime_observation_passed",
            "forbidden_request_types_sent": [],
            "write_request_sent": False,
            "live_apply_request_sent": False,
        }
    )
    recorder.write_summary(summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
