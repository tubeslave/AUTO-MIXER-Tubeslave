#!/usr/bin/env python3
"""Run the M005 WING disconnect/timeout/stale-state proof safely.

The runner talks only to the backend WebSocket. It never sends direct write
messages such as set_fader/set_gain. The blocked-write probe is attempted only
after the runner has evidence that fresh physical readback is unavailable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import websockets

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
BACKEND_ROOT = REPO_ROOT / "backend"
for import_path in (str(TOOLS_ROOT), str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from wing_operator_proposal_validation_runner import (  # noqa: E402
    EventRecorder,
    address_for,
    build_preflight_decision_gate,
    build_proposal,
    json_default,
    send_and_wait,
    utc_stamp,
)


EXPECTED_MODE_EVENTS = {"operator_mode_status"}
EXPECTED_STATE_EVENTS = {"state_update"}
EXPECTED_CONNECTION_EVENTS = {"connection_status"}
EXPECTED_PROPOSAL_EVENTS = {"operator_proposal_created"}
EXPECTED_APPLY_EVENTS = {"operator_proposal_apply_result", "operator_proposal_apply_blocked"}
EXPECTED_PARAMETER_EVENTS = {"mixer_parameter_value", "mixer_parameter_read_blocked"}
EXPECTED_ROLLBACK_EVENTS = {
    "manual_write_rollback_applied",
    "manual_write_rollback_blocked",
    "manual_write_rollback_error",
}
PHYSICAL_READBACK_STATUSES = {"verified_from_physical_readback"}
SAFE_DISCONNECT_REQUEST_TYPES = {"disconnect_mixer", "disconnect_wing"}
SAFE_BLOCK_REASONS = {
    "mixer_not_connected",
    "pre_read_unavailable",
    "pre_read_cache_not_fresh",
    "mixer_write_failed",
    "manual_write_exception",
}
DEFAULT_ARTIFACT_ROOT = "artifacts/wing_timeout_disconnect_proof"
PHYSICAL_CABLE_CONFIRMATION = "WING_CABLE_PULLED_BY_OPERATOR"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run supervised M005 disconnect/timeout/stale-state proof."
    )
    parser.add_argument("--url", default="ws://localhost:8765")
    parser.add_argument("--channel", type=int, default=3)
    parser.add_argument("--value-type", choices=["fader", "gain"], default="fader")
    parser.add_argument("--delta-db", type=float, default=-0.25)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--reason", default="m005_disconnect_timeout_stale_state_proof")
    parser.add_argument("--approval-prefix", default="wing-m005-disconnect-proof")
    parser.add_argument(
        "--disconnect-mode",
        choices=["software", "physical-cable-loss"],
        default="software",
    )
    parser.add_argument(
        "--disconnect-request-type",
        choices=sorted(SAFE_DISCONNECT_REQUEST_TYPES),
        default="disconnect_mixer",
    )
    parser.add_argument(
        "--operator-confirm-physical-cable-loss",
        help=f"Required for physical-cable-loss mode; must equal {PHYSICAL_CABLE_CONFIRMATION}",
    )
    parser.add_argument(
        "--physical-loss-wait-sec",
        type=float,
        default=2.0,
        help="Delay before stale-state query in physical-cable-loss mode.",
    )
    parser.add_argument("--operator-confirm-channel", type=int)
    parser.add_argument("--operator-confirm-value-type", choices=["fader", "gain"])
    parser.add_argument("--operator-confirm-delta-db", type=float)
    parser.add_argument("--accepted-readonly-source-id")
    parser.add_argument(
        "--preflight-artifact-root",
        default="artifacts/wing_readonly_preflight",
    )
    parser.add_argument(
        "--preflight-decision-root",
        default="artifacts/wing_readonly_preflight_review_decisions",
    )
    parser.add_argument("--reconnect-wing-ip")
    parser.add_argument("--reconnect-send-port", type=int, default=2222)
    parser.add_argument("--reconnect-receive-port", type=int, default=2223)
    parser.add_argument(
        "--rollback-on-unexpected-apply",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def validate_args(args: argparse.Namespace) -> Optional[str]:
    if args.channel < 1 or args.channel > 48:
        return "--channel must be in 1..48"
    if args.delta_db >= 0.0 or abs(args.delta_db) > 0.5:
        return "--delta-db must be negative and no larger than 0.5 dB"
    if args.timeout <= 0:
        return "--timeout must be positive"
    if args.physical_loss_wait_sec < 0 or args.physical_loss_wait_sec > 30:
        return "--physical-loss-wait-sec must be between 0 and 30 seconds"
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
    if args.disconnect_mode == "physical-cable-loss":
        if args.operator_confirm_physical_cable_loss != PHYSICAL_CABLE_CONFIRMATION:
            return (
                "--operator-confirm-physical-cable-loss must equal "
                f"{PHYSICAL_CABLE_CONFIRMATION}"
            )
    return None


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _readback_is_physical(payload: Dict[str, Any]) -> bool:
    return (
        payload.get("type") == "mixer_parameter_value"
        and payload.get("status") in PHYSICAL_READBACK_STATUSES
        and payload.get("source") == "physical_readback"
        and payload.get("value") is not None
    )


def _readback_is_unavailable(payload: Dict[str, Any]) -> bool:
    if payload.get("type") == "mixer_parameter_read_blocked":
        return True
    return (
        payload.get("type") == "mixer_parameter_value"
        and payload.get("status") == "readback_unavailable"
        and payload.get("value") is None
        and payload.get("source") in {"unavailable", "physical_readback"}
    )


def _apply_block_reason(payload: Dict[str, Any]) -> Optional[str]:
    result = payload.get("apply_result")
    if isinstance(result, dict) and result.get("reason"):
        return str(result.get("reason"))
    if payload.get("reason"):
        return str(payload.get("reason"))
    return None


def _apply_probe_blocked(payload: Dict[str, Any]) -> bool:
    return payload.get("type") == "operator_proposal_apply_blocked" and (
        _apply_block_reason(payload) in SAFE_BLOCK_REASONS
    )


def _unexpected_apply(payload: Dict[str, Any]) -> bool:
    result = payload.get("apply_result")
    if not isinstance(result, dict):
        result = payload
    return (
        payload.get("type") == "operator_proposal_apply_result"
        and payload.get("status") == "applied"
        and result.get("status") == "applied"
    )


async def _query_parameter(
    ws: Any,
    recorder: EventRecorder,
    *,
    channel: int,
    value_type: str,
    address: str,
    timeout: float,
    request_id: str,
) -> Dict[str, Any]:
    return await send_and_wait(
        ws,
        recorder,
        {
            "type": "query_mixer_parameter",
            "request_id": request_id,
            "channel": channel,
            "value_type": value_type,
            "address": address,
        },
        EXPECTED_PARAMETER_EVENTS,
        timeout=timeout,
    )


async def _create_probe_proposal(
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


async def _apply_probe(
    ws: Any,
    recorder: EventRecorder,
    *,
    proposal_id: str,
    run_id: str,
    approval_prefix: str,
    reason: str,
    timeout: float,
) -> Dict[str, Any]:
    return await send_and_wait(
        ws,
        recorder,
        {
            "type": "apply_operator_proposal",
            "proposal_id": proposal_id,
            "approved": True,
            "approval_id": f"{approval_prefix}:blocked-write:{run_id}",
            "idempotency_key": f"{approval_prefix}:blocked-write:{run_id}",
            "reason": reason,
        },
        EXPECTED_APPLY_EVENTS,
        timeout=timeout,
    )


async def _rollback_unexpected_apply(
    ws: Any,
    recorder: EventRecorder,
    *,
    run_id: str,
    approval_prefix: str,
    timeout: float,
) -> Dict[str, Any]:
    return await send_and_wait(
        ws,
        recorder,
        {
            "type": "rollback_last_manual_write",
            "approved": True,
            "approval_id": f"{approval_prefix}:rollback-unexpected:{run_id}",
            "reason": f"rollback_unexpected_m005_apply:{run_id}",
        },
        EXPECTED_ROLLBACK_EVENTS,
        timeout=timeout,
    )


async def run(args: argparse.Namespace) -> int:
    run_id = f"{utc_stamp()}_wing_m005_disconnect_timeout_stale_proof"
    run_dir = Path(args.artifact_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    recorder = EventRecorder(run_dir)
    address = address_for(args.channel, args.value_type)
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "proof": "wing_m005_disconnect_timeout_stale_protection",
        "proof_artifact_schema": "wing_timeout_disconnect_proof.v1",
        "proof_runner": "wing_timeout_disconnect_proof_runner",
        "status": "started",
        "execute": bool(args.execute),
        "started_at": _iso_now(),
        "url": args.url,
        "channel": args.channel,
        "value_type": args.value_type,
        "address": address,
        "delta_db": args.delta_db,
        "disconnect_mode": args.disconnect_mode,
        "disconnect_request_type": args.disconnect_request_type,
        "write_policy": "blocked_write_attempts_only",
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "readiness_matrix_update_allowed": False,
        "physical_cable_confirmation": args.operator_confirm_physical_cable_loss,
        "phases": {},
    }

    validation_error = validate_args(args)
    if validation_error:
        summary.update({"status": "blocked", "reason": validation_error})
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return 2

    if not args.execute:
        summary.update(
            {
                "status": "planned",
                "reason": "dry_run_only_add_execute_for_operator_attended_proof",
                "live_apply_request_sent": False,
                "rollback_request_sent": False,
            }
        )
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return 0

    preflight_gate, preflight_error = build_preflight_decision_gate(args)
    summary["preflight_decision_gate"] = preflight_gate
    if preflight_error:
        summary.update(
            {
                "status": "blocked",
                "reason": preflight_error,
                "fail_closed": True,
                "live_apply_request_sent": False,
                "rollback_request_sent": False,
            }
        )
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return 13

    proposal_id = f"{args.approval_prefix}:probe:{run_id}"
    try:
        async with websockets.connect(args.url) as ws:
            mode = await send_and_wait(
                ws,
                recorder,
                {"type": "get_operator_mode"},
                EXPECTED_MODE_EVENTS,
                timeout=args.timeout,
            )
            summary["initial_operator_mode"] = mode
            await send_and_wait(
                ws,
                recorder,
                {"type": "get_state"},
                EXPECTED_STATE_EVENTS,
                timeout=args.timeout,
            )
            baseline = await _query_parameter(
                ws,
                recorder,
                channel=args.channel,
                value_type=args.value_type,
                address=address,
                timeout=args.timeout,
                request_id=f"{run_id}:baseline",
            )
            summary["phases"]["baseline"] = {"status": "passed", "readback": baseline}
            if not _readback_is_physical(baseline):
                summary.update(
                    {
                        "status": "blocked",
                        "reason": "baseline_not_verified_from_physical_readback",
                        "fail_closed": True,
                    }
                )
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
                return 3

            old_value = float(baseline["value"])
            target_value = round(old_value + float(args.delta_db), 2)
            summary["old_value"] = old_value
            summary["target_value"] = target_value

            supervised_mode = await send_and_wait(
                ws,
                recorder,
                {"type": "set_operator_mode", "mode": "supervised", "reason": args.reason},
                EXPECTED_MODE_EVENTS,
                timeout=args.timeout,
            )
            summary["supervised_mode"] = supervised_mode
            if supervised_mode.get("mode") != "supervised":
                summary.update({"status": "blocked", "reason": "operator_mode_not_supervised"})
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
                return 4

            proposal = await _create_probe_proposal(
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
            summary["probe_proposal"] = proposal
            if proposal.get("status") not in {"created", "exists"}:
                summary.update({"status": "failed", "reason": "probe_proposal_not_created"})
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
                return 5

            if args.disconnect_mode == "software":
                disconnect = await send_and_wait(
                    ws,
                    recorder,
                    {"type": args.disconnect_request_type, "reason": args.reason},
                    EXPECTED_CONNECTION_EVENTS,
                    timeout=args.timeout,
                )
                summary["phases"]["disconnect"] = {
                    "status": "passed" if disconnect.get("connected") is False else "failed",
                    "request_type": args.disconnect_request_type,
                    "connection_status": disconnect,
                }
                if disconnect.get("connected") is not False:
                    summary.update(
                        {
                            "status": "blocked",
                            "reason": "disconnect_not_confirmed_before_write_probe",
                            "fail_closed": True,
                        }
                    )
                    recorder.write_summary(summary)
                    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
                    return 6
            else:
                await asyncio.sleep(float(args.physical_loss_wait_sec))
                summary["phases"]["disconnect"] = {
                    "status": "operator_confirmed_physical_cable_loss",
                    "confirmation": args.operator_confirm_physical_cable_loss,
                    "wait_sec": args.physical_loss_wait_sec,
                }

            stale_probe = await _query_parameter(
                ws,
                recorder,
                channel=args.channel,
                value_type=args.value_type,
                address=address,
                timeout=args.timeout,
                request_id=f"{run_id}:stale-probe",
            )
            summary["phases"]["stale_readback"] = {
                "status": "passed" if _readback_is_unavailable(stale_probe) else "failed",
                "readback": stale_probe,
            }
            if not _readback_is_unavailable(stale_probe):
                summary.update(
                    {
                        "status": "blocked",
                        "reason": "stale_or_live_readback_after_disconnect",
                        "fail_closed": True,
                        "safety_pre_write_disconnect_verified": True,
                        "safety_pre_write_readback_unavailable": False,
                    }
                )
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
                return 7

            summary["safety_pre_write_disconnect_verified"] = True
            summary["safety_pre_write_readback_unavailable"] = True
            apply_probe = await _apply_probe(
                ws,
                recorder,
                proposal_id=proposal_id,
                run_id=run_id,
                approval_prefix=args.approval_prefix,
                reason=args.reason,
                timeout=args.timeout,
            )
            summary["phases"]["blocked_write_while_disconnected"] = {
                "status": "passed" if _apply_probe_blocked(apply_probe) else "failed",
                "apply_result": apply_probe,
                "block_reason": _apply_block_reason(apply_probe),
            }
            if _unexpected_apply(apply_probe):
                rollback = None
                if args.rollback_on_unexpected_apply:
                    rollback = await _rollback_unexpected_apply(
                        ws,
                        recorder,
                        run_id=run_id,
                        approval_prefix=args.approval_prefix,
                        timeout=args.timeout,
                    )
                summary["unexpected_apply_rollback_result"] = rollback
                summary.update(
                    {
                        "status": "manual_review_required",
                        "reason": "blocked_write_probe_applied_unexpectedly",
                        "fail_closed": False,
                        "operator_action_required": (
                            "Verify physical WING state and rollback before reusing this session."
                        ),
                    }
                )
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
                return 8
            if not _apply_probe_blocked(apply_probe):
                summary.update(
                    {
                        "status": "failed",
                        "reason": "blocked_write_probe_not_blocked_with_expected_reason",
                        "fail_closed": False,
                    }
                )
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
                return 9

            if args.reconnect_wing_ip:
                reconnect = await send_and_wait(
                    ws,
                    recorder,
                    {
                        "type": "connect_wing",
                        "ip": args.reconnect_wing_ip,
                        "send_port": args.reconnect_send_port,
                        "receive_port": args.reconnect_receive_port,
                        "observation_only": True,
                        "record_telemetry": False,
                    },
                    EXPECTED_CONNECTION_EVENTS,
                    timeout=args.timeout,
                )
                recovery_readback = await _query_parameter(
                    ws,
                    recorder,
                    channel=args.channel,
                    value_type=args.value_type,
                    address=address,
                    timeout=args.timeout,
                    request_id=f"{run_id}:reconnect",
                )
                recovered_value = recovery_readback.get("value")
                unchanged = (
                    _readback_is_physical(recovery_readback)
                    and abs(float(recovered_value) - old_value) <= 0.06
                )
                summary["phases"]["reconnect_recovery"] = {
                    "status": "passed" if reconnect.get("connected") and unchanged else "failed",
                    "connection_status": reconnect,
                    "readback": recovery_readback,
                    "unchanged_from_baseline": unchanged,
                }
                if not (reconnect.get("connected") and unchanged):
                    summary.update({"status": "failed", "reason": "reconnect_recovery_not_verified"})
                    recorder.write_summary(summary)
                    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
                    return 10
            else:
                summary["phases"]["reconnect_recovery"] = {
                    "status": "not_requested",
                    "reason": "reconnect-wing-ip not provided",
                }

            summary.update(
                {
                    "status": "passed",
                    "reason": "blocked_write_while_disconnected_verified",
                    "fail_closed": True,
                    "remaining_m005_evidence": (
                        []
                        if args.disconnect_mode == "physical-cable-loss"
                        else ["physical network link-loss phase still requires operator cable pull"]
                    ),
                }
            )
            recorder.write_summary(summary)
            print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
            return 0
    except (TimeoutError, OSError, ConnectionError) as exc:
        summary.update(
            {
                "status": "blocked",
                "reason": "connection_or_timeout_before_completed_m005_proof",
                "fail_closed": True,
                "transport_failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return 1
    except Exception as exc:
        summary.update({"status": "error", "reason": type(exc).__name__, "error": str(exc)})
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
