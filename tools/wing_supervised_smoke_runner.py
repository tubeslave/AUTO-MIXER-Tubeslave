#!/usr/bin/env python3
"""Narrow Wing Rack supervised smoke-test runner.

This runner talks to the existing backend WebSocket. It does not discover or
connect the mixer by itself; the backend must already be connected to Wing Rack.

Default mode is dry-run planning. Real console writes require --execute.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import websockets


EXPECTED_WRITE_EVENTS = {"manual_write_applied", "manual_write_blocked", "manual_write_error"}
EXPECTED_ROLLBACK_EVENTS = {
    "manual_write_rollback_applied",
    "manual_write_rollback_blocked",
    "manual_write_rollback_error",
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_default(value: Any) -> Any:
    try:
        return str(value)
    except Exception:
        return "<unserializable>"


def address_for(channel: int, value_type: str) -> str:
    if value_type == "fader":
        return f"/ch/{channel}/fdr"
    if value_type == "gain":
        return f"/ch/{channel}/in/set/trim"
    raise ValueError(f"Unsupported value_type: {value_type}")


def clamp(value: float, value_type: str) -> float:
    if value_type == "fader":
        return max(-144.0, min(10.0, value))
    if value_type == "gain":
        return max(-18.0, min(18.0, value))
    raise ValueError(f"Unsupported value_type: {value_type}")


def as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class EventRecorder:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.events_path = run_dir / "events.jsonl"
        self.summary_path = run_dir / "summary.json"
        self.seq = 0

    def record(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.seq += 1
        event = {
            "seq": self.seq,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "event_type": event_type,
            "payload": payload,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=json_default, sort_keys=True) + "\n")

    def write_summary(self, summary: Dict[str, Any]) -> None:
        summary = dict(summary)
        summary.setdefault("events_path", str(self.events_path))
        summary.setdefault("summary_path", str(self.summary_path))
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True, default=json_default)
            handle.write("\n")


async def send_and_wait(
    ws: Any,
    recorder: EventRecorder,
    request: Dict[str, Any],
    expected_types: Iterable[str],
    *,
    timeout: float,
) -> Dict[str, Any]:
    expected = set(expected_types)
    recorder.record("send", request)
    await ws.send(json.dumps(request))
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for {sorted(expected)} after {request.get('type')}")
        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"type": "invalid_json", "raw": raw}
        recorder.record("recv", payload)
        if payload.get("type") in expected:
            return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a narrow supervised Wing Rack smoke-test through backend WebSocket.")
    parser.add_argument("--url", default="ws://localhost:8765", help="Backend WebSocket URL")
    parser.add_argument("--channel", type=int, required=True, help="Wing input channel, 1..48")
    parser.add_argument("--value-type", choices=["fader", "gain"], required=True, help="Parameter to test")
    parser.add_argument("--delta-db", type=float, required=True, help="Tiny dB delta to apply before rollback")
    parser.add_argument("--execute", action="store_true", help="Actually send approved write and rollback")
    parser.add_argument("--prove-emergency-stop", action="store_true", help="After rollback, trigger emergency_stop and prove a no-op write is blocked")
    parser.add_argument("--timeout", type=float, default=5.0, help="Timeout per WebSocket step")
    parser.add_argument("--artifact-root", default="artifacts/wing_telemetry", help="Artifact root directory")
    parser.add_argument("--approval-prefix", default="wing-supervised-smoke", help="Approval id prefix")
    parser.add_argument("--reason", default="wing_supervised_smoke_test", help="Reason text sent to backend")
    return parser


def validate_args(args: argparse.Namespace) -> Optional[str]:
    if args.channel < 1 or args.channel > 48:
        return "--channel must be in 1..48"
    if abs(args.delta_db) <= 0.0 or abs(args.delta_db) > 0.5:
        return "--delta-db must be non-zero and no larger than 0.5 dB"
    if args.prove_emergency_stop and not args.execute:
        return "--prove-emergency-stop requires --execute"
    return None


async def run(args: argparse.Namespace) -> int:
    validation_error = validate_args(args)
    run_id = f"{utc_stamp()}_wing_supervised_smoke_ch{args.channel}_{args.value_type}"
    run_dir = Path(args.artifact_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    recorder = EventRecorder(run_dir)

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "status": "started",
        "execute": bool(args.execute),
        "url": args.url,
        "channel": args.channel,
        "value_type": args.value_type,
        "delta_db": args.delta_db,
        "prove_emergency_stop": bool(args.prove_emergency_stop),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }

    if validation_error:
        summary.update({"status": "blocked", "reason": validation_error})
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 2

    try:
        async with websockets.connect(args.url) as ws:
            mode = await send_and_wait(
                ws,
                recorder,
                {"type": "get_operator_mode"},
                {"operator_mode_status"},
                timeout=args.timeout,
            )
            state_payload = await send_and_wait(
                ws,
                recorder,
                {"type": "get_state"},
                {"state_update"},
                timeout=args.timeout,
            )
            state = state_payload.get("state") or {}
            address = address_for(args.channel, args.value_type)
            old_value = as_float(state.get(address))
            if old_value is None:
                summary.update({
                    "status": "blocked",
                    "reason": "current_value_missing_from_backend_state",
                    "operator_mode": mode.get("mode"),
                    "address": address,
                })
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 3

            target_value = round(clamp(old_value + args.delta_db, args.value_type), 2)
            summary.update({
                "status": "planned",
                "operator_mode": mode.get("mode"),
                "address": address,
                "old_value": old_value,
                "target_value": target_value,
            })

            if not args.execute:
                summary["reason"] = "dry_run_only_add_execute_to_write"
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 0

            mode = await send_and_wait(
                ws,
                recorder,
                {"type": "set_operator_mode", "mode": "supervised", "reason": args.reason},
                {"operator_mode_status"},
                timeout=args.timeout,
            )
            if mode.get("mode") != "supervised":
                summary.update({"status": "blocked", "reason": "operator_mode_not_supervised", "operator_mode_result": mode})
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 4

            command_type = "set_fader" if args.value_type == "fader" else "set_gain"
            write_result = await send_and_wait(
                ws,
                recorder,
                {
                    "type": command_type,
                    "channel": args.channel,
                    "value": target_value,
                    "approved": True,
                    "approval_id": f"{args.approval_prefix}:write:{run_id}",
                    "reason": args.reason,
                },
                EXPECTED_WRITE_EVENTS,
                timeout=args.timeout,
            )
            summary["write_result"] = write_result
            if write_result.get("type") != "manual_write_applied" or write_result.get("status") != "applied":
                summary.update({"status": "failed", "reason": "write_not_applied"})
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 5

            rollback_result = await send_and_wait(
                ws,
                recorder,
                {
                    "type": "rollback_last_manual_write",
                    "approved": True,
                    "approval_id": f"{args.approval_prefix}:rollback:{run_id}",
                    "reason": f"rollback:{args.reason}",
                },
                EXPECTED_ROLLBACK_EVENTS,
                timeout=args.timeout,
            )
            summary["rollback_result"] = rollback_result
            if rollback_result.get("type") != "manual_write_rollback_applied" or rollback_result.get("status") != "applied":
                summary.update({"status": "failed", "reason": "rollback_not_applied"})
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 6

            if args.prove_emergency_stop:
                emergency = await send_and_wait(
                    ws,
                    recorder,
                    {"type": "emergency_stop"},
                    {"emergency_stop_applied"},
                    timeout=args.timeout,
                )
                blocked_after_stop = await send_and_wait(
                    ws,
                    recorder,
                    {
                        "type": command_type,
                        "channel": args.channel,
                        "value": old_value,
                        "approved": True,
                        "approval_id": f"{args.approval_prefix}:emergency-proof:{run_id}",
                        "reason": f"emergency_stop_proof:{args.reason}",
                    },
                    EXPECTED_WRITE_EVENTS,
                    timeout=args.timeout,
                )
                summary["emergency_stop_result"] = emergency
                summary["post_emergency_write_result"] = blocked_after_stop
                if blocked_after_stop.get("reason") != "emergency_stop_active":
                    summary.update({"status": "failed", "reason": "emergency_stop_did_not_block_write"})
                    recorder.write_summary(summary)
                    print(json.dumps(summary, indent=2, sort_keys=True))
                    return 7

            summary.update({"status": "passed", "reason": "write_readback_rollback_complete"})
            recorder.write_summary(summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
    except Exception as exc:
        summary.update({"status": "error", "reason": type(exc).__name__, "error": str(exc)})
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
