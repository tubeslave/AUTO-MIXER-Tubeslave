#!/usr/bin/env python3
"""Dry-run readiness proof for a supervised gain proposal.

This runner deliberately uses an in-process fake mixer. It exercises the same
Product Layer proposal path that a physical proof will use later, but it does
not open a WebSocket, OSC socket, or WING connection.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = REPO_ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

VERIFIED_READBACK_STATUSES = {"verified_from_cache", "verified_from_physical_readback"}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


class FakeProofMixer:
    def __init__(self, channel: int, initial_gain_db: float):
        self.is_connected = True
        self.physical = {self.gain_address(channel): float(initial_gain_db)}
        self.state = dict(self.physical)
        self.write_calls: list[Dict[str, Any]] = []

    @staticmethod
    def gain_address(channel: int) -> str:
        return f"/ch/{channel}/in/set/trim"

    def query_parameter(self, address: str) -> Dict[str, Any]:
        return {"status": "received", "address": address, "value": self.physical.get(address)}

    def set_channel_gain(self, channel: int, value: float) -> bool:
        address = self.gain_address(channel)
        value = max(-18.0, min(18.0, float(value)))
        self.physical[address] = value
        self.state[address] = value
        self.write_calls.append({"method": "set_channel_gain", "channel": channel, "value": value})
        return True

    def set_channel_fader(self, channel: int, value: float) -> bool:
        self.write_calls.append({"method": "set_channel_fader", "channel": channel, "value": value})
        return True

    def get_write_gate_status(self) -> Dict[str, Any]:
        return {"state": "fake_dry_run"}

    def disconnect(self) -> None:
        self.is_connected = False


async def dispatch(server: Any, recorder: EventRecorder, message_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    captured: list[Dict[str, Any]] = []

    async def capture_send(_websocket: Any, response: Dict[str, Any]) -> None:
        captured.append(response)
        recorder.record("recv", response)

    server.send_to_client = capture_send
    request = {"type": message_type, **payload}
    recorder.record("send", request)
    await server._dispatch[message_type](object(), payload)
    if not captured:
        raise RuntimeError(f"No response captured for {message_type}")
    return captured[-1]


def build_gain_proposal(proposal_id: str, channel: int, old_value: float, target_value: float, reason: str) -> Dict[str, Any]:
    return {
        "id": proposal_id,
        "title": f"Dry-run gain proof: CH{channel} {old_value:.2f} -> {target_value:.2f} dB",
        "kind": "gain",
        "target": channel,
        "channel": channel,
        "value_type": "gain",
        "value": target_value,
        "severity": "low",
        "confidence": 1.0,
        "source": "gain_proof_dry_run_runner",
        "reason": reason,
        "requested_change": {
            "value_type": "gain",
            "channel": channel,
            "value": target_value,
            "current_value": old_value,
            "delta_db": round(target_value - old_value, 3),
        },
    }


def extract_apply_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = payload.get("apply_result")
    if isinstance(result, dict):
        return result
    return payload


def verified(result: Dict[str, Any]) -> bool:
    return (
        result.get("status") == "applied"
        and result.get("readback_status") in VERIFIED_READBACK_STATUSES
        and result.get("proof_status", "verified") == "verified"
        and bool(result.get("rollback_available"))
    )


def validate_args(args: argparse.Namespace) -> Optional[str]:
    if args.channel < 1 or args.channel > 48:
        return "--channel must be in 1..48"
    if abs(args.delta_db) <= 0 or abs(args.delta_db) > 0.5:
        return "--delta-db absolute value must be in 0..0.5 dB"
    if args.cooldown_sec < 0 or args.cooldown_sec > 30:
        return "--cooldown-sec must be between 0 and 30"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an offline dry-run proof for one supervised gain proposal.")
    parser.add_argument("--channel", type=int, default=1, help="Fake channel to test, 1..48")
    parser.add_argument("--initial-gain-db", type=float, default=1.0, help="Initial fake gain value")
    parser.add_argument("--delta-db", type=float, default=-0.25, help="Small gain delta for apply before rollback")
    parser.add_argument("--cooldown-sec", type=float, default=2.0, help="Manual gate cooldown to prove")
    parser.add_argument(
        "--artifact-root",
        default="artifacts/wing_telemetry/gain_proof_dry_run",
        help="Artifact root directory",
    )
    parser.add_argument("--reason", default="gain_proof_dry_run", help="Reason recorded in proposal metadata")
    return parser


async def run(args: argparse.Namespace) -> int:
    validation_error = validate_args(args)
    run_id = f"{utc_stamp()}_gain_proof_dry_run_ch{args.channel}"
    run_dir = Path(args.artifact_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    recorder = EventRecorder(run_dir)

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "status": "started",
        "channel": args.channel,
        "initial_gain_db": args.initial_gain_db,
        "delta_db": args.delta_db,
        "cooldown_sec": args.cooldown_sec,
        "write_policy": "fake_mixer_apply_operator_proposal_only",
        "physical_wing": False,
        "forbidden_messages": ["set_gain", "set_fader", "set_eq", "set_compressor", "route_output", "load_snap", "save_snap"],
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    if validation_error:
        summary.update({"status": "blocked", "reason": validation_error})
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 2

    try:
        from server import AutoMixerServer

        with patch("server.BleedService"), patch.object(AutoMixerServer, "_load_config", return_value={}):
            server = AutoMixerServer(ws_host="localhost", ws_port=8765)
        fake_mixer = FakeProofMixer(args.channel, args.initial_gain_db)
        server.connection_mode = "wing"
        server.mixer_client = fake_mixer
        server._manual_write_cooldown_sec = args.cooldown_sec
        server._manual_write_artifacts_dir = run_dir / "manual_write_gate"

        await dispatch(server, recorder, "set_operator_mode", {"mode": "supervised", "reason": args.reason})

        address = fake_mixer.gain_address(args.channel)
        old_value = float(fake_mixer.physical[address])
        target_value = round(max(-18.0, min(18.0, old_value + args.delta_db)), 2)
        summary.update({"address": address, "old_value": old_value, "target_value": target_value})

        proposal_id = f"gain-proof-dry-run:apply:{run_id}"
        proposal = build_gain_proposal(proposal_id, args.channel, old_value, target_value, args.reason)
        proposal_result = await dispatch(server, recorder, "create_operator_proposal", {"proposal": proposal})
        apply_payload = await dispatch(
            server,
            recorder,
            "apply_operator_proposal",
            {
                "proposal_id": proposal_id,
                "approved": True,
                "approval_id": f"gain-proof-dry-run:apply-approved:{run_id}",
                "idempotency_key": f"gain-proof-dry-run:apply-approved:{run_id}",
                "reason": args.reason,
            },
        )
        apply_result = extract_apply_result(apply_payload)
        summary["proposal_result"] = proposal_result
        summary["apply_result"] = apply_payload
        if not verified(apply_result):
            summary.update({"status": "failed", "reason": "apply_not_verified"})
            recorder.write_summary(summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 3

        cooldown_id = f"gain-proof-dry-run:cooldown:{run_id}"
        cooldown_proposal = build_gain_proposal(cooldown_id, args.channel, target_value, target_value + args.delta_db, args.reason)
        cooldown_result = await dispatch(server, recorder, "create_operator_proposal", {"proposal": cooldown_proposal})
        cooldown_apply = await dispatch(
            server,
            recorder,
            "apply_operator_proposal",
            {
                "proposal_id": cooldown_id,
                "approved": True,
                "approval_id": f"gain-proof-dry-run:cooldown-approved:{run_id}",
                "idempotency_key": f"gain-proof-dry-run:cooldown-approved:{run_id}",
                "reason": f"cooldown:{args.reason}",
            },
        )
        summary["cooldown_proposal_result"] = cooldown_result
        summary["cooldown_apply_result"] = cooldown_apply
        if extract_apply_result(cooldown_apply).get("reason") != "cooldown":
            summary.update({"status": "failed", "reason": "cooldown_not_blocked"})
            recorder.write_summary(summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 4

        rollback_payload = await dispatch(
            server,
            recorder,
            "rollback_last_manual_write",
            {
                "approved": True,
                "approval_id": f"gain-proof-dry-run:rollback:{run_id}",
                "reason": f"rollback:{args.reason}",
            },
        )
        rollback_status = rollback_payload.get("rollback_readback_status") or rollback_payload.get("readback_status")
        rollback_proof_status = rollback_payload.get("rollback_proof_status") or rollback_payload.get("proof_status")
        summary["rollback_result"] = rollback_payload
        if (
            rollback_payload.get("type") != "manual_write_rollback_applied"
            or rollback_payload.get("rollback_status") != "applied"
            or rollback_status not in VERIFIED_READBACK_STATUSES
            or rollback_proof_status != "verified"
        ):
            summary.update({"status": "failed", "reason": "rollback_not_verified"})
            recorder.write_summary(summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 5

        emergency = await dispatch(server, recorder, "emergency_stop", {})
        emergency_id = f"gain-proof-dry-run:emergency:{run_id}"
        emergency_proposal = build_gain_proposal(emergency_id, args.channel, old_value, target_value, f"emergency:{args.reason}")
        emergency_proposal_result = await dispatch(server, recorder, "create_operator_proposal", {"proposal": emergency_proposal})
        emergency_apply = await dispatch(
            server,
            recorder,
            "apply_operator_proposal",
            {
                "proposal_id": emergency_id,
                "approved": True,
                "approval_id": f"gain-proof-dry-run:emergency-approved:{run_id}",
                "idempotency_key": f"gain-proof-dry-run:emergency-approved:{run_id}",
                "reason": f"emergency:{args.reason}",
            },
        )
        summary["emergency_stop_result"] = emergency
        summary["emergency_proposal_result"] = emergency_proposal_result
        summary["post_emergency_apply_result"] = emergency_apply
        if extract_apply_result(emergency_apply).get("reason") != "emergency_stop_active":
            summary.update({"status": "failed", "reason": "emergency_stop_not_blocked"})
            recorder.write_summary(summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 6

        summary.update({
            "status": "passed",
            "reason": "fake_gain_proof_readback_rollback_cooldown_emergency_complete",
            "fake_mixer_write_calls": fake_mixer.write_calls,
        })
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
