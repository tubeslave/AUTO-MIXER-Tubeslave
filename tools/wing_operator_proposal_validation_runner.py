#!/usr/bin/env python3
"""Validate one Wing Rack supervised apply through operator proposals.

The runner talks to the existing backend WebSocket and deliberately avoids
direct write messages such as set_fader/set_gain. Real console mutation only
happens after a Product Layer proposal is created and applied through
apply_operator_proposal. Rollback uses the existing supervised rollback gate.

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

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
TOOLS_ROOT = REPO_ROOT / "tools"
for import_path in (str(TOOLS_ROOT), str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

try:
    from review_wing_readonly_preflight_decisions import (  # type: ignore
        ACCEPTED_DECISION_READY_STATUS,
        build_decision_review_report,
    )
except Exception:  # pragma: no cover - fail closed when import is unavailable
    ACCEPTED_DECISION_READY_STATUS = "accepted_decision_ready"
    build_decision_review_report = None  # type: ignore[assignment]


EXPECTED_PROPOSAL_EVENTS = {"operator_proposal_created"}
EXPECTED_APPLY_EVENTS = {"operator_proposal_apply_result", "operator_proposal_apply_blocked"}
EXPECTED_DISMISS_EVENTS = {"operator_proposal_dismissed"}
EXPECTED_ROLLBACK_EVENTS = {
    "manual_write_rollback_applied",
    "manual_write_rollback_blocked",
    "manual_write_rollback_error",
}
EXPECTED_PARAMETER_EVENTS = {"mixer_parameter_value", "mixer_parameter_read_blocked"}
PHYSICAL_READBACK_STATUSES = {"verified_from_physical_readback"}
FORBIDDEN_DIRECT_REQUEST_TYPES = {
    "set_fader",
    "set_gain",
    "route_output",
    "load_snap",
    "save_snap",
}
DEFAULT_PREFLIGHT_ARTIFACT_ROOT = "artifacts/wing_readonly_preflight"
DEFAULT_PREFLIGHT_DECISION_ROOT = "artifacts/wing_readonly_preflight_review_decisions"
PROOF_SURFACE_CONTRACT_BY_VALUE_TYPE = {
    "fader": {
        "wing_surface": "channel_fader",
        "address_template": "/ch/{channel}/fdr",
        "proof_direction": "negative_db_move_only",
    },
    "gain": {
        "wing_surface": "input_trim",
        "address_template": "/ch/{channel}/in/set/trim",
        "proof_direction": "negative_input_trim_move_only",
    },
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_default(value: Any) -> Any:
    try:
        return str(value)
    except Exception:
        return "<unserializable>"


def proof_surface_contract(channel: int, value_type: str) -> Dict[str, str]:
    contract = PROOF_SURFACE_CONTRACT_BY_VALUE_TYPE.get(value_type)
    if not contract:
        raise ValueError(f"Unsupported value_type: {value_type}")
    return {
        "wing_surface": contract["wing_surface"],
        "wing_address": contract["address_template"].format(channel=channel),
        "proof_direction": contract["proof_direction"],
        "address_template": contract["address_template"],
    }


def address_for(channel: int, value_type: str) -> str:
    return proof_surface_contract(channel, value_type)["wing_address"]


def clamp(value: float, value_type: str) -> float:
    if value_type == "fader":
        return max(-144.0, min(0.0, value))
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


async def read_current_parameter(
    ws: Any,
    recorder: EventRecorder,
    *,
    channel: int,
    value_type: str,
    address: str,
    state: Dict[str, Any],
    timeout: float,
    require_physical_readback: bool = False,
) -> tuple[Optional[float], Dict[str, Any]]:
    cached_value = as_float(state.get(address))
    if cached_value is not None and not require_physical_readback:
        return cached_value, {
            "status": "verified_from_cache",
            "source": "mixer_state_cache",
            "address": address,
            "value": cached_value,
        }

    try:
        payload = await send_and_wait(
            ws,
            recorder,
            {
                "type": "query_mixer_parameter",
                "channel": channel,
                "value_type": value_type,
                "address": address,
            },
            EXPECTED_PARAMETER_EVENTS,
            timeout=timeout,
        )
    except Exception as exc:
        return None, {
            "type": "mixer_parameter_read_error",
            "status": "readback_unavailable",
            "source": "physical_readback",
            "address": address,
            "error": str(exc),
        }

    if payload.get("type") == "mixer_parameter_read_blocked":
        return None, payload

    return as_float(payload.get("value")), payload


class EventRecorder:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.events_path = run_dir / "events.jsonl"
        self.summary_path = run_dir / "summary.json"
        self.seq = 0
        self.sent_request_types: list[str] = []

    def record(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.seq += 1
        if event_type == "send":
            self.sent_request_types.append(str(payload.get("type") or "unknown"))
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
        sent_request_types = list(self.sent_request_types)
        forbidden_sent = [
            request_type
            for request_type in sent_request_types
            if request_type in FORBIDDEN_DIRECT_REQUEST_TYPES
        ]
        summary.setdefault("sent_request_types", sent_request_types)
        summary.setdefault(
            "proposal_create_request_sent",
            "create_operator_proposal" in sent_request_types,
        )
        summary.setdefault(
            "live_apply_request_sent",
            "apply_operator_proposal" in sent_request_types,
        )
        summary.setdefault(
            "rollback_request_sent",
            "rollback_last_manual_write" in sent_request_types,
        )
        summary.setdefault("forbidden_request_types_sent", forbidden_sent)
        summary.setdefault("direct_live_write_request_sent", bool(forbidden_sent))
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
    parser = argparse.ArgumentParser(
        description="Run a narrow Wing Rack validation through Product Layer operator proposals."
    )
    parser.add_argument("--url", default="ws://localhost:8765", help="Backend WebSocket URL")
    parser.add_argument("--channel", type=int, required=True, help="Wing input channel, 1..48")
    parser.add_argument("--value-type", choices=["fader", "gain"], required=True, help="Parameter to test")
    parser.add_argument("--delta-db", type=float, required=True, help="Small negative dB delta to apply before rollback")
    parser.add_argument(
        "--operator-confirm-channel",
        type=int,
        help="Operator-confirmed channel; required for --execute and must match --channel",
    )
    parser.add_argument(
        "--operator-confirm-value-type",
        choices=["fader", "gain"],
        help="Operator-confirmed value type; required for --execute and must match --value-type",
    )
    parser.add_argument(
        "--operator-confirm-delta-db",
        type=float,
        help="Operator-confirmed negative delta; required for --execute and must match --delta-db",
    )
    parser.add_argument("--execute", action="store_true", help="Actually apply proposal and rollback")
    parser.add_argument(
        "--prove-blocked-apply",
        action="store_true",
        help="Before the approved apply, prove apply_operator_proposal is blocked outside Supervised mode",
    )
    parser.add_argument(
        "--prove-emergency-stop",
        action="store_true",
        help="After rollback, trigger emergency_stop and prove a later proposal apply is blocked",
    )
    parser.add_argument("--timeout", type=float, default=5.0, help="Timeout per WebSocket step")
    parser.add_argument("--rollback-delay-sec", type=float, default=0.0, help="Delay after apply before rollback")
    parser.add_argument("--artifact-root", default="artifacts/wing_telemetry", help="Artifact root directory")
    parser.add_argument("--approval-prefix", default="wing-proposal-validation", help="Approval id prefix")
    parser.add_argument("--reason", default="wing_operator_proposal_validation", help="Reason text sent to backend")
    parser.add_argument(
        "--preflight-artifact-root",
        default=DEFAULT_PREFLIGHT_ARTIFACT_ROOT,
        help="Read-only WING preflight artifact root used to gate --execute",
    )
    parser.add_argument(
        "--preflight-decision-root",
        default=DEFAULT_PREFLIGHT_DECISION_ROOT,
        help="Operator decision root used to gate --execute",
    )
    parser.add_argument(
        "--accepted-readonly-source-id",
        help="Accepted read-only preflight summary.json source id required before --execute",
    )
    return parser


def validate_args(args: argparse.Namespace) -> Optional[str]:
    if args.channel < 1 or args.channel > 48:
        return "--channel must be in 1..48"
    if args.delta_db >= 0.0 or abs(args.delta_db) > 0.5:
        return "--delta-db must be negative and no larger than 0.5 dB"
    if args.execute:
        if args.operator_confirm_channel is None:
            return "--operator-confirm-channel is required with --execute"
        if args.operator_confirm_value_type is None:
            return "--operator-confirm-value-type is required with --execute"
        if args.operator_confirm_delta_db is None:
            return "--operator-confirm-delta-db is required with --execute"
        if args.operator_confirm_channel != args.channel:
            return "--operator-confirm-channel must match --channel"
        if args.operator_confirm_value_type != args.value_type:
            return "--operator-confirm-value-type must match --value-type"
        if round(float(args.operator_confirm_delta_db), 3) != round(float(args.delta_db), 3):
            return "--operator-confirm-delta-db must match --delta-db"
        if not args.accepted_readonly_source_id:
            return "--accepted-readonly-source-id is required with --execute"
    if args.prove_emergency_stop and not args.execute:
        return "--prove-emergency-stop requires --execute"
    if args.prove_blocked_apply and not args.execute:
        return "--prove-blocked-apply requires --execute"
    if args.rollback_delay_sec < 0.0 or args.rollback_delay_sec > 5.0:
        return "--rollback-delay-sec must be between 0 and 5 seconds"
    return None


def build_preflight_decision_gate(args: argparse.Namespace) -> tuple[Dict[str, Any], Optional[str]]:
    gate: Dict[str, Any] = {
        "required_for_execute": True,
        "artifact_root": args.preflight_artifact_root,
        "decision_root": args.preflight_decision_root,
        "expected_accepted_readonly_source_id": args.accepted_readonly_source_id,
    }
    if not args.execute:
        gate.update({"status": "not_required_for_dry_run", "accepted": False})
        return gate, None
    if build_decision_review_report is None:
        gate.update({"status": "unavailable", "accepted": False})
        return gate, "preflight_decision_review_import_unavailable"

    try:
        report = build_decision_review_report(
            artifact_roots=[args.preflight_artifact_root],
            decision_roots=[args.preflight_decision_root],
        )
    except Exception as exc:
        gate.update(
            {
                "status": "error",
                "accepted": False,
                "error": type(exc).__name__,
                "message": str(exc),
            }
        )
        return gate, "preflight_decision_review_unavailable"

    accepted_source_ids = [
        str(source_id)
        for source_id in report.get("accepted_source_ids", [])
        if source_id
    ]
    expected_source_id = str(args.accepted_readonly_source_id or "")
    gate.update(
        {
            "status": report.get("status"),
            "ledger_status": report.get("ledger_status"),
            "accepted": report.get("status") == ACCEPTED_DECISION_READY_STATUS,
            "accepted_source_ids": accepted_source_ids,
            "accepted_record_count": report.get("accepted_record_count"),
            "decision_record_count": report.get("decision_record_count"),
            "invalid_record_count": report.get("invalid_record_count"),
            "unreviewed_candidate_count": report.get("unreviewed_candidate_count"),
            "manual_readiness_update_candidate_available": report.get(
                "manual_readiness_update_candidate_available"
            ),
            "source_authority_denied": report.get("source_authority_denied"),
            "live_authority_granted": report.get("live_authority_granted"),
            "execution_allowed_by_decision": report.get("execution_allowed_by_decision"),
            "next_required_action": report.get("next_required_action"),
            "blocked_by": report.get("blocked_by", []),
        }
    )
    if report.get("status") != ACCEPTED_DECISION_READY_STATUS:
        return gate, "preflight_decision_review_not_accepted"
    if expected_source_id not in accepted_source_ids:
        return gate, "accepted_readonly_source_id_not_accepted"
    if report.get("live_authority_granted") is True:
        return gate, "preflight_decision_must_not_grant_live_authority"
    if report.get("execution_allowed_by_decision") is True:
        return gate, "preflight_decision_must_not_allow_execution_by_itself"
    if report.get("source_authority_denied") is not True:
        return gate, "preflight_decision_source_authority_not_denied"
    return gate, None


def build_proposal(
    *,
    proposal_id: str,
    channel: int,
    value_type: str,
    old_value: float,
    target_value: float,
    reason: str,
) -> Dict[str, Any]:
    surface_contract = proof_surface_contract(channel, value_type)
    return {
        "id": proposal_id,
        "title": f"WING validation: CH{channel} {value_type} {old_value:.2f} -> {target_value:.2f} dB",
        "kind": value_type,
        "target": channel,
        "channel": channel,
        "value_type": value_type,
        "wing_surface": surface_contract["wing_surface"],
        "wing_address": surface_contract["wing_address"],
        "proof_direction": surface_contract["proof_direction"],
        "value": target_value,
        "severity": "low",
        "confidence": 1.0,
        "source": "wing_operator_proposal_validation_runner",
        "reason": reason,
        "requested_change": {
            "value_type": value_type,
            "channel": channel,
            "wing_surface": surface_contract["wing_surface"],
            "wing_address": surface_contract["wing_address"],
            "proof_direction": surface_contract["proof_direction"],
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


def apply_succeeded(payload: Dict[str, Any]) -> bool:
    result = extract_apply_result(payload)
    return (
        payload.get("type") == "operator_proposal_apply_result"
        and payload.get("status") == "applied"
        and result.get("status") == "applied"
        and result.get("readback_status") in PHYSICAL_READBACK_STATUSES
        and result.get("proof_status", "verified") == "verified"
        and bool(result.get("rollback_available"))
    )


def summarize_transport_failure(
    summary: Dict[str, Any],
    recorder: EventRecorder,
    exc: BaseException,
) -> Dict[str, Any]:
    sent_request_types = list(recorder.sent_request_types)
    apply_sent = "apply_operator_proposal" in sent_request_types
    rollback_sent = "rollback_last_manual_write" in sent_request_types
    summary.update(
        {
            "transport_failure": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "sent_request_types": sent_request_types,
            "proposal_create_request_sent": "create_operator_proposal" in sent_request_types,
            "live_apply_request_sent": apply_sent,
            "rollback_request_sent": rollback_sent,
        }
    )
    if apply_sent:
        summary.update(
            {
                "status": "manual_review_required",
                "reason": "connection_or_timeout_after_apply_request",
                "fail_closed": False,
                "operator_action_required": (
                    "Physically verify the WING value, perform or confirm rollback, "
                    "and do not use this artifact as promotion evidence."
                ),
            }
        )
    else:
        summary.update(
            {
                "status": "blocked",
                "reason": "connection_or_timeout_fail_closed",
                "fail_closed": True,
                "operator_action_required": (
                    "Fix backend/WING connectivity and rerun; no live apply request was sent."
                ),
            }
        )
    return summary


async def create_proposal(
    ws: Any,
    recorder: EventRecorder,
    *,
    proposal_id: str,
    channel: int,
    value_type: str,
    old_value: float,
    target_value: float,
    reason: str,
    timeout: float,
) -> Dict[str, Any]:
    return await send_and_wait(
        ws,
        recorder,
        {
            "type": "create_operator_proposal",
            "proposal": build_proposal(
                proposal_id=proposal_id,
                channel=channel,
                value_type=value_type,
                old_value=old_value,
                target_value=target_value,
                reason=reason,
            ),
        },
        EXPECTED_PROPOSAL_EVENTS,
        timeout=timeout,
    )


async def dismiss_proposal(
    ws: Any,
    recorder: EventRecorder,
    *,
    proposal_id: str,
    reason: str,
    timeout: float,
) -> Dict[str, Any]:
    return await send_and_wait(
        ws,
        recorder,
        {
            "type": "dismiss_operator_proposal",
            "proposal_id": proposal_id,
            "reason": reason,
        },
        EXPECTED_DISMISS_EVENTS,
        timeout=timeout,
    )


async def run(args: argparse.Namespace) -> int:
    validation_error = validate_args(args)
    run_id = f"{utc_stamp()}_wing_operator_proposal_ch{args.channel}_{args.value_type}"
    run_dir = Path(args.artifact_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    recorder = EventRecorder(run_dir)

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "status": "started",
        "execute": bool(args.execute),
        "proof_artifact_schema": "wing_operator_proposal_validation.v2",
        "proof_runner": "wing_operator_proposal_validation_runner",
        "proof_mode": "physical" if args.execute else "dry_run",
        "physical_wing": bool(args.execute),
        "physical_readback_required_for_execute": True,
        "physical_readback_required": bool(args.execute),
        "url": args.url,
        "channel": args.channel,
        "value_type": args.value_type,
        "delta_db": args.delta_db,
        "operator_confirmation_required": bool(args.execute),
        "operator_confirmed_channel": args.operator_confirm_channel,
        "operator_confirmed_value_type": args.operator_confirm_value_type,
        "operator_confirmed_delta_db": args.operator_confirm_delta_db,
        "operator_confirmation_match": (
            bool(args.execute)
            and args.operator_confirm_channel == args.channel
            and args.operator_confirm_value_type == args.value_type
            and args.operator_confirm_delta_db is not None
            and round(float(args.operator_confirm_delta_db), 3) == round(float(args.delta_db), 3)
        ),
        "rollback_delay_sec": args.rollback_delay_sec,
        "prove_blocked_apply": bool(args.prove_blocked_apply),
        "prove_emergency_stop": bool(args.prove_emergency_stop),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "write_policy": "apply_operator_proposal_only",
        "physical_proof_requires_execute": True,
        "forbidden_messages": sorted(FORBIDDEN_DIRECT_REQUEST_TYPES),
        "preflight_decision_gate_required_for_execute": True,
        "accepted_readonly_source_id": args.accepted_readonly_source_id,
    }

    if validation_error:
        summary.update({"status": "blocked", "reason": validation_error})
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 2

    surface_contract = proof_surface_contract(args.channel, args.value_type)
    summary.update(
        {
            "wing_surface": surface_contract["wing_surface"],
            "wing_address": surface_contract["wing_address"],
            "proof_direction": surface_contract["proof_direction"],
            "proof_surface_contract": surface_contract,
        }
    )

    preflight_gate, preflight_gate_error = build_preflight_decision_gate(args)
    summary["preflight_decision_gate"] = preflight_gate
    if preflight_gate_error:
        summary.update(
            {
                "status": "blocked",
                "reason": preflight_gate_error,
                "fail_closed": True,
                "live_apply_request_sent": False,
                "rollback_request_sent": False,
                "sent_request_types": [],
            }
        )
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 13

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
            old_value, old_value_read = await read_current_parameter(
                ws,
                recorder,
                channel=args.channel,
                value_type=args.value_type,
                address=address,
                state=state,
                timeout=args.timeout,
                require_physical_readback=bool(args.execute),
            )
            if old_value is None:
                summary.update({
                    "status": "blocked",
                    "reason": "current_value_unavailable_from_backend_state_and_physical_readback",
                    "operator_mode": mode.get("mode"),
                    "address": address,
                    "old_value_read": old_value_read,
                })
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 3
            if args.execute and old_value_read.get("status") not in PHYSICAL_READBACK_STATUSES:
                summary.update({
                    "status": "blocked",
                    "reason": "current_value_not_verified_from_physical_readback",
                    "operator_mode": mode.get("mode"),
                    "address": address,
                    "old_value": old_value,
                    "old_value_read": old_value_read,
                })
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 3

            target_value = round(clamp(old_value + args.delta_db, args.value_type), 2)
            if target_value == old_value:
                summary.update({
                    "status": "blocked",
                    "reason": "target_value_would_not_change",
                    "operator_mode": mode.get("mode"),
                    "address": address,
                    "old_value": old_value,
                    "target_value": target_value,
                })
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 4

            summary.update({
                "status": "planned",
                "operator_mode": mode.get("mode"),
                "address": address,
                "old_value": old_value,
                "old_value_read": old_value_read,
                "old_value_source": old_value_read.get("source"),
                "old_value_status": old_value_read.get("status"),
                "target_value": target_value,
            })

            if not args.execute:
                summary["reason"] = "dry_run_only_add_execute_to_write"
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 0

            if args.prove_blocked_apply:
                assist_mode = await send_and_wait(
                    ws,
                    recorder,
                    {"type": "set_operator_mode", "mode": "assist", "reason": f"blocked-proof:{args.reason}"},
                    {"operator_mode_status"},
                    timeout=args.timeout,
                )
                summary["blocked_apply_mode"] = assist_mode
                if assist_mode.get("mode") != "assist":
                    summary.update({"status": "blocked", "reason": "operator_mode_not_assist_for_blocked_apply_proof"})
                    recorder.write_summary(summary)
                    print(json.dumps(summary, indent=2, sort_keys=True))
                    return 5
                blocked_proposal_id = f"{args.approval_prefix}:blocked:{run_id}"
                blocked_proposal = await create_proposal(
                    ws,
                    recorder,
                    proposal_id=blocked_proposal_id,
                    channel=args.channel,
                    value_type=args.value_type,
                    old_value=old_value,
                    target_value=target_value,
                    reason=f"blocked-proof:{args.reason}",
                    timeout=args.timeout,
                )
                blocked_apply = await send_and_wait(
                    ws,
                    recorder,
                    {
                        "type": "apply_operator_proposal",
                        "proposal_id": blocked_proposal_id,
                        "approved": True,
                        "approval_id": f"{args.approval_prefix}:blocked-apply:{run_id}",
                        "idempotency_key": f"{args.approval_prefix}:blocked-apply:{run_id}",
                        "reason": f"blocked-proof:{args.reason}",
                    },
                    EXPECTED_APPLY_EVENTS,
                    timeout=args.timeout,
                )
                summary["blocked_apply_proposal"] = blocked_proposal
                summary["blocked_apply_result"] = blocked_apply
                blocked_reason = extract_apply_result(blocked_apply).get("reason")
                if blocked_reason != "operator_mode_blocks_live_write":
                    summary.update({"status": "failed", "reason": "apply_was_not_blocked_outside_supervised"})
                    recorder.write_summary(summary)
                    print(json.dumps(summary, indent=2, sort_keys=True))
                    return 6
                summary["blocked_apply_cleanup"] = await dismiss_proposal(
                    ws,
                    recorder,
                    proposal_id=blocked_proposal_id,
                    reason=f"cleanup:block-proof:{args.reason}",
                    timeout=args.timeout,
                )

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
                return 7

            proposal_id = f"{args.approval_prefix}:apply:{run_id}"
            proposal_result = await create_proposal(
                ws,
                recorder,
                proposal_id=proposal_id,
                channel=args.channel,
                value_type=args.value_type,
                old_value=old_value,
                target_value=target_value,
                reason=args.reason,
                timeout=args.timeout,
            )
            if proposal_result.get("status") not in {"created", "exists"}:
                summary.update({"status": "failed", "reason": "proposal_not_created", "proposal_result": proposal_result})
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 8

            apply_result = await send_and_wait(
                ws,
                recorder,
                {
                    "type": "apply_operator_proposal",
                    "proposal_id": proposal_id,
                    "approved": True,
                    "approval_id": f"{args.approval_prefix}:apply-approved:{run_id}",
                    "idempotency_key": f"{args.approval_prefix}:apply-approved:{run_id}",
                    "reason": args.reason,
                },
                EXPECTED_APPLY_EVENTS,
                timeout=args.timeout,
            )
            summary["proposal_result"] = proposal_result
            summary["apply_result"] = apply_result
            if not apply_succeeded(apply_result):
                summary.update({"status": "failed", "reason": "proposal_apply_not_verified"})
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 9

            if args.rollback_delay_sec > 0:
                await asyncio.sleep(args.rollback_delay_sec)

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
                return 10
            rollback_readback_status = rollback_result.get("rollback_readback_status") or rollback_result.get("readback_status")
            rollback_proof_status = rollback_result.get("rollback_proof_status") or rollback_result.get("proof_status", "verified")
            if (
                rollback_readback_status not in PHYSICAL_READBACK_STATUSES
                or rollback_proof_status != "verified"
            ):
                summary.update({"status": "failed", "reason": "rollback_readback_not_verified"})
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 11

            if args.prove_emergency_stop:
                emergency = await send_and_wait(
                    ws,
                    recorder,
                    {"type": "emergency_stop"},
                    {"emergency_stop_applied"},
                    timeout=args.timeout,
                )
                emergency_proposal_id = f"{args.approval_prefix}:emergency:{run_id}"
                emergency_proposal = await create_proposal(
                    ws,
                    recorder,
                    proposal_id=emergency_proposal_id,
                    channel=args.channel,
                    value_type=args.value_type,
                    old_value=old_value,
                    target_value=target_value,
                    reason=f"emergency-proof:{args.reason}",
                    timeout=args.timeout,
                )
                emergency_apply = await send_and_wait(
                    ws,
                    recorder,
                    {
                        "type": "apply_operator_proposal",
                        "proposal_id": emergency_proposal_id,
                        "approved": True,
                        "approval_id": f"{args.approval_prefix}:emergency-proof:{run_id}",
                        "idempotency_key": f"{args.approval_prefix}:emergency-proof:{run_id}",
                        "reason": f"emergency-proof:{args.reason}",
                    },
                    EXPECTED_APPLY_EVENTS,
                    timeout=args.timeout,
                )
                summary["emergency_stop_result"] = emergency
                summary["emergency_proposal"] = emergency_proposal
                summary["post_emergency_apply_result"] = emergency_apply
                if extract_apply_result(emergency_apply).get("reason") != "emergency_stop_active":
                    summary.update({"status": "failed", "reason": "emergency_stop_did_not_block_proposal_apply"})
                    recorder.write_summary(summary)
                    print(json.dumps(summary, indent=2, sort_keys=True))
                    return 12
                summary["emergency_proposal_cleanup"] = await dismiss_proposal(
                    ws,
                    recorder,
                    proposal_id=emergency_proposal_id,
                    reason=f"cleanup:emergency-proof:{args.reason}",
                    timeout=args.timeout,
                )

            summary.update({"status": "passed", "reason": "proposal_apply_readback_rollback_complete"})
            recorder.write_summary(summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
    except (TimeoutError, OSError, ConnectionError) as exc:
        summarize_transport_failure(summary, recorder, exc)
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1
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
