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
    "analyzer_status",
    "master_bus_status",
)


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
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True, default=json_default)
            handle.write("\n")


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
        "has_master_bus": isinstance(master, dict),
        "has_analyzer": isinstance(analyzer, dict),
        "analyzer_available": bool(isinstance(analyzer, dict) and analyzer.get("available")),
        "master_bus_available": bool(isinstance(master, dict) and master.get("available")),
    }


def _operator_runtime_contract(summary: Dict[str, Any]) -> Dict[str, Any]:
    counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
    dashboard = (
        summary.get("dashboard_contract")
        if isinstance(summary.get("dashboard_contract"), dict)
        else {}
    )
    meter = summary.get("meter_contract") if isinstance(summary.get("meter_contract"), dict) else {}
    checks = {
        "operator_mode_status": int(counts.get("operator_mode_status") or 0) > 0,
        "connection_state": bool(dashboard.get("has_connection"))
        or int(counts.get("connection_status") or 0) > 0,
        "dashboard_snapshot": int(counts.get("dashboard_snapshot") or 0) > 0,
        "channel_inventory": int(counts.get("channel_inventory") or 0) > 0,
        "decision_queue": int(counts.get("decision_queue") or 0) > 0,
        "safety_monitor": int(counts.get("safety_monitor") or 0) > 0,
        "meter_subscription": meter.get("subscribed") is True,
        "meter_frame": int(counts.get("meter_frame") or 0) > 0
        and int(meter.get("read_only_frames") or 0) > 0,
        "meter_channel_levels": int(meter.get("max_channel_count") or 0) > 0,
        "analyzer_status": bool(dashboard.get("has_analyzer")),
        "master_bus_status": bool(dashboard.get("has_master_bus")),
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
            "has_master_bus": False,
            "has_analyzer": False,
            "analyzer_available_count": 0,
            "master_bus_available_count": 0,
        },
        "meter_contract": {
            "subscribed": False,
            "read_only_frames": 0,
            "live_write_frames": 0,
            "available_frames": 0,
            "max_channel_count": 0,
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
            await _send(ws, recorder, {"type": "subscribe_meters"})
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
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    if args.duration_sec == 0:
                        break
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {"type": "invalid_json", "raw": raw}
                if not isinstance(payload, dict):
                    payload = {"type": "non_object_payload", "payload": payload}
                recorder.record("recv", payload)
                payload_type = str(payload.get("type") or "unknown")
                if payload_type in summary["counts"]:
                    summary["counts"][payload_type] += 1
                if _is_unexpected_live_event(payload):
                    summary["counts"]["unexpected_live_event"] += 1
                    summary["unexpected_live_events"].append(payload)

                if payload_type == "meter_stream_status":
                    summary["meter_contract"]["subscribed"] = bool(payload.get("subscribed"))
                    if payload.get("live_write") is True:
                        summary["meter_contract"]["live_write_frames"] += 1
                elif payload_type == "meter_frame":
                    channels = payload.get("channels") if isinstance(payload.get("channels"), list) else []
                    summary["meter_contract"]["max_channel_count"] = max(
                        int(summary["meter_contract"]["max_channel_count"]),
                        len(channels),
                    )
                    if payload.get("read_only") is True:
                        summary["meter_contract"]["read_only_frames"] += 1
                    if payload.get("live_write") is True:
                        summary["meter_contract"]["live_write_frames"] += 1
                    if payload.get("available") is True:
                        summary["meter_contract"]["available_frames"] += 1
                elif payload_type == "dashboard_snapshot":
                    fields = _dashboard_fields(payload)
                    for key in ("has_connection", "has_master_bus", "has_analyzer"):
                        summary["dashboard_contract"][key] = bool(
                            summary["dashboard_contract"][key] or fields.get(key)
                        )
                    if fields.get("analyzer_available"):
                        summary["dashboard_contract"]["analyzer_available_count"] += 1
                    if fields.get("master_bus_available"):
                        summary["dashboard_contract"]["master_bus_available_count"] += 1

            await _send(ws, recorder, {"type": "unsubscribe_meters"})
    except (OSError, TimeoutError, ConnectionError) as exc:
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
