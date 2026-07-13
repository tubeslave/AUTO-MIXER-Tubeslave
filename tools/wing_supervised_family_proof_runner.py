#!/usr/bin/env python3
"""Local, restore-after-verify hardware proof for one WING write family."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping

import websockets

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from supervised_console_parameter_registry import (  # noqa: E402
    FAMILY_CONTRACTS,
    LIFECYCLE_PROOF_CANDIDATE,
)
from supervised_console_transaction import (  # noqa: E402
    AudioRecheckResult,
    ConsoleTransactionContext,
    ConsoleTransactionOperation,
    ConsoleTransactionPlan,
    SupervisedConsoleTransactionCoordinator,
)
from wing_client import WingClient  # noqa: E402
from wing_supervised_console_adapters import (  # noqa: E402
    build_wing_supervised_family_adapters,
)


PROOF_FAMILIES = (
    "fader",
    "gain_trim",
    "mute",
    "pan_phase_delay",
    "eq_hpf",
    "gate_expander",
    "dynamics_compressor",
    "fx_sends_inserts",
    "bus_group_dca_matrix",
    "master_output",
)

LIVE_EXECUTION_CONFIRMATION = "WING_SUPERVISED_FAMILY_PROOF_EXECUTE"
RESTORE_CONFIRMATION = "WING_RESTORE_AFTER_VERIFY_APPROVED"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00",
        "Z",
    )


def target_scope_fingerprint(args: argparse.Namespace) -> str:
    """Return the exact console/surface scope the operator must confirm."""
    return (
        f"family={args.family};ip={args.ip};channel={args.channel};"
        f"bus={args.bus};dca={args.dca};main={args.main};send_bus={args.send_bus}"
    )


def execution_authorization_failures(args: argparse.Namespace) -> list[str]:
    """Fail before transport construction unless every live barrier matches."""
    failures = []
    if not bool(getattr(args, "execute", False)):
        failures.append("execute_not_requested")
    if not str(getattr(args, "approval_id", "") or "").strip():
        failures.append("operator_approval_id_required")
    if getattr(args, "confirm_live_write", None) != LIVE_EXECUTION_CONFIRMATION:
        failures.append("live_execution_confirmation_mismatch")
    if getattr(args, "confirm_restore", None) != RESTORE_CONFIRMATION:
        failures.append("restore_confirmation_mismatch")
    if str(getattr(args, "confirm_family", "") or "") != str(args.family):
        failures.append("family_confirmation_mismatch")
    if str(getattr(args, "confirm_ip", "") or "") != str(args.ip):
        failures.append("wing_ip_confirmation_mismatch")
    if str(getattr(args, "confirm_target_scope", "") or "") != target_scope_fingerprint(args):
        failures.append("target_scope_confirmation_mismatch")
    if not 1 <= int(args.channel) <= 48:
        failures.append("channel_out_of_range")
    if not 1 <= int(args.bus) <= 16:
        failures.append("bus_out_of_range")
    if not 1 <= int(args.dca) <= 16:
        failures.append("dca_out_of_range")
    if not 1 <= int(args.main) <= 4:
        failures.append("main_out_of_range")
    if not 1 <= int(args.send_bus) <= 16:
        failures.append("send_bus_out_of_range")
    contract = next(
        (item for item in FAMILY_CONTRACTS if item.family_id == args.family),
        None,
    )
    if contract is None or contract.lifecycle == "proposal_only":
        failures.append("canonical_family_lifecycle_blocks_hardware_proof")
    # Direct-to-WING proof execution cannot observe the backend's emergency
    # generation or share its single-writer lock. Keep this tool plan-only
    # until proof execution is routed through that canonical control plane.
    failures.append("backend_integrated_exclusive_writer_required")
    return failures


def planned_summary(
    args: argparse.Namespace,
    failures: Iterable[str],
) -> Dict[str, Any]:
    failures = list(failures)
    execute_requested = bool(getattr(args, "execute", False))
    return {
        "schema": "wing-supervised-family-proof/v1",
        "timestamp_utc": utc_now(),
        "family_id": args.family,
        "wing_ip": args.ip,
        "status": "blocked" if execute_requested else "planned",
        "reason": (
            "execution_authorization_failed"
            if execute_requested
            else "plan_only_execute_flag_required"
        ),
        "authorization_failures": failures,
        "required_live_execution_confirmation": LIVE_EXECUTION_CONFIRMATION,
        "required_restore_confirmation": RESTORE_CONFIRMATION,
        "required_target_scope_confirmation": target_scope_fingerprint(args),
        "contacts_wing": False,
        "contacts_backend": False,
        "live_write": False,
        "proof_passed": False,
        "accepted": False,
        "owner_decision_required": True,
        "readiness_matrix_update_allowed": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
    }


def number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def physical_value(client: WingClient, address: str) -> Any:
    result = client.query_parameter(address, timeout=1.0)
    if result.get("status") != "received":
        raise RuntimeError(f"physical_read_failed:{address}:{result.get('status')}")
    if int(result.get("confirmed_round_trips") or 0) < 2:
        raise RuntimeError(f"physical_read_unconfirmed:{address}")
    return result.get("value")


def operation(
    family_id: str,
    write_kind: str,
    target_value: Any,
    *,
    suffix: str,
    channel: int | None = None,
    parameter: str = "",
) -> ConsoleTransactionOperation:
    return ConsoleTransactionOperation(
        operation_id=f"proof:{family_id}:{suffix}",
        family_id=family_id,
        write_kind=write_kind,
        target_value=target_value,
        channel=channel,
        parameter=parameter,
        metadata={
            "source": "wing_supervised_family_proof_runner",
            "proof_only": True,
            "restore_after_verify": True,
        },
    )


def safe_proof_operations(
    client: WingClient,
    family_id: str,
    *,
    channel: int,
    bus: int,
    dca: int,
    main: int,
    send_bus: int,
) -> tuple[ConsoleTransactionOperation, ...]:
    ch = f"/ch/{channel}"
    if family_id == "fader":
        current = number(physical_value(client, f"{ch}/fdr"), -12.0)
        return (operation(family_id, "fader", max(-144.0, current - 0.5), suffix="fader_down", channel=channel),)
    if family_id == "gain_trim":
        current = number(physical_value(client, f"{ch}/in/set/trim"), 0.0)
        return (operation(family_id, "gain", max(-18.0, current - 0.5), suffix="trim_down", channel=channel),)
    if family_id == "mute":
        current = bool(number(physical_value(client, f"{ch}/mute"), 0.0))
        if current:
            raise RuntimeError("mute_already_engaged_no_safe_probe")
        return (operation(family_id, "mute", True, suffix="mute_only", channel=channel),)
    if family_id == "pan_phase_delay":
        pan = number(physical_value(client, f"{ch}/pan"), 0.0)
        pan_target = pan - 1.0 if pan > 0 else pan + 1.0
        inverted = bool(number(physical_value(client, f"{ch}/in/set/inv"), 0.0))
        delay = number(physical_value(client, f"{ch}/in/set/dly"), 0.0)
        delay_enabled = bool(number(physical_value(client, f"{ch}/in/set/dlyon"), 0.0))
        return (
            operation(family_id, "pan", max(-100.0, min(100.0, pan_target)), suffix="pan_toward_center", channel=channel),
            operation(family_id, "polarity", not inverted, suffix="polarity_toggle", channel=channel),
            operation(
                family_id,
                "delay",
                {
                    "delay_ms": min(500.0, max(0.1, delay + 0.1)),
                    "enabled": delay_enabled,
                },
                suffix="delay_0_1ms",
                channel=channel,
            ),
        )
    if family_id == "eq_hpf":
        hpf_freq = number(physical_value(client, f"{ch}/flt/lcf"), 80.0)
        hpf_enabled = bool(number(physical_value(client, f"{ch}/flt/lc"), 0.0))
        band = 4
        eq_freq = number(physical_value(client, f"{ch}/eq/{band}f"), 5000.0)
        eq_gain = number(physical_value(client, f"{ch}/eq/{band}g"), 0.0)
        eq_q = number(physical_value(client, f"{ch}/eq/{band}q"), 1.0)
        return (
            operation(
                family_id,
                "hpf",
                {"freq_hz": min(2000.0, max(20.0, hpf_freq + 5.0)), "enabled": hpf_enabled},
                suffix="hpf_frequency",
                channel=channel,
            ),
            operation(
                family_id,
                "eq",
                {
                    "band": band,
                    "freq_hz": eq_freq,
                    "gain_db": max(-15.0, min(0.0, eq_gain - 0.5)),
                    "q": eq_q,
                },
                suffix="eq_cut",
                channel=channel,
            ),
        )
    if family_id == "gate_expander":
        values = {
            "threshold_db": max(-80.0, number(physical_value(client, f"{ch}/gate/thr"), -50.0) - 1.0),
            "range_db": max(3.0, min(60.0, number(physical_value(client, f"{ch}/gate/range"), 10.0))),
            "attack_ms": max(0.0, min(120.0, number(physical_value(client, f"{ch}/gate/att"), 5.0))),
            "hold_ms": max(0.0, min(2000.0, number(physical_value(client, f"{ch}/gate/hld"), 50.0))),
            "release_ms": max(4.0, min(4000.0, number(physical_value(client, f"{ch}/gate/rel"), 120.0))),
            "enabled": bool(number(physical_value(client, f"{ch}/gate/on"), 0.0)),
        }
        return (operation(family_id, "gate", values, suffix="gate_threshold_down", channel=channel),)
    if family_id == "dynamics_compressor":
        values = {
            "threshold_db": max(-60.0, number(physical_value(client, f"{ch}/dyn/thr"), -18.0) - 0.5),
            "ratio": number(physical_value(client, f"{ch}/dyn/ratio"), 2.0),
            "attack_ms": max(0.0, min(120.0, number(physical_value(client, f"{ch}/dyn/att"), 20.0))),
            "release_ms": max(4.0, min(4000.0, number(physical_value(client, f"{ch}/dyn/rel"), 100.0))),
            "makeup_db": max(-6.0, min(12.0, number(physical_value(client, f"{ch}/dyn/gain"), 0.0))),
            "enabled": bool(number(physical_value(client, f"{ch}/dyn/on"), 0.0)),
        }
        return (operation(family_id, "compressor", values, suffix="compressor_threshold_down", channel=channel),)
    if family_id == "fx_sends_inserts":
        current = number(physical_value(client, f"{ch}/send/{send_bus}/lvl"), -30.0)
        if current <= -143.0:
            raise RuntimeError("send_inactive_no_safe_downward_probe")
        target = max(-144.0, current - 1.0)
        return (
            operation(
                family_id,
                "send",
                {"send_bus": send_bus, "level_db": target},
                suffix="send_down",
                channel=channel,
            ),
        )
    if family_id == "bus_group_dca_matrix":
        bus_fader = number(physical_value(client, f"/bus/{bus}/fdr"), -12.0)
        dca_fader = number(physical_value(client, f"/dca/{dca}/fdr"), 0.0)
        if dca_fader <= -143.0:
            raise RuntimeError("dca_inactive_no_safe_downward_probe")
        dca_target = max(-144.0, dca_fader - 0.5)
        band = 4
        eq_freq = number(physical_value(client, f"/bus/{bus}/eq/{band}f"), 5000.0)
        eq_gain = number(physical_value(client, f"/bus/{bus}/eq/{band}g"), 0.0)
        eq_q = number(physical_value(client, f"/bus/{bus}/eq/{band}q"), 1.0)
        compressor = {
            "bus_id": bus,
            "threshold_db": max(-60.0, number(physical_value(client, f"/bus/{bus}/dyn/thr"), -18.0) - 0.5),
            "ratio": number(physical_value(client, f"/bus/{bus}/dyn/ratio"), 2.0),
            "attack_ms": max(0.0, min(120.0, number(physical_value(client, f"/bus/{bus}/dyn/att"), 20.0))),
            "release_ms": max(4.0, min(4000.0, number(physical_value(client, f"/bus/{bus}/dyn/rel"), 100.0))),
            "makeup_db": max(-6.0, min(12.0, number(physical_value(client, f"/bus/{bus}/dyn/gain"), 0.0))),
            "enabled": bool(number(physical_value(client, f"/bus/{bus}/dyn/on"), 0.0)),
        }
        return (
            operation(family_id, "bus", {"bus_id": bus, "target_db": max(-144.0, bus_fader - 0.5)}, suffix="bus_fader_down", parameter=f"bus_{bus}_fader_db"),
            operation(family_id, "dca", {"dca_id": dca, "target_db": dca_target}, suffix="dca_fader_down", parameter=f"dca_{dca}_fader_db"),
            operation(
                family_id,
                "bus",
                {"bus_id": bus, "band": band, "freq_hz": eq_freq, "gain_db": max(-15.0, min(0.0, eq_gain - 0.5)), "q": eq_q},
                suffix="bus_eq_cut",
                parameter=f"bus_{bus}_eq_band_{band}",
            ),
            operation(family_id, "bus", compressor, suffix="bus_compressor_threshold_down", parameter=f"bus_{bus}_compressor"),
        )
    if family_id == "master_output":
        current = number(physical_value(client, f"/main/{main}/fdr"), -12.0)
        return (
            operation(
                family_id,
                "master",
                {"main_id": main, "target_db": max(-144.0, current - 0.5)},
                suffix="master_fader_down",
                parameter=f"main_{main}_fader_db",
            ),
        )
    raise ValueError(f"unsupported proof family: {family_id}")


class MasterAudioRecheck:
    def __init__(self, websocket_url: str) -> None:
        self.websocket_url = websocket_url
        self.evidence: list[Dict[str, Any]] = []

    async def __call__(self, operation, snapshot, apply_evidence) -> AudioRecheckResult:
        async with websockets.connect(self.websocket_url, max_size=8_000_000) as ws:
            await ws.send(json.dumps({"type": "get_channel_inventory", "total_channels": 48}))
            while True:
                payload = json.loads(await asyncio.wait_for(ws.recv(), timeout=15.0))
                if payload.get("type") == "channel_inventory":
                    break
        rows = payload.get("channels") or []
        peaks = []
        for channel in (47, 48):
            if len(rows) < channel:
                continue
            peak = rows[channel - 1].get("peak_db")
            try:
                peaks.append(float(peak))
            except (TypeError, ValueError):
                pass
        evidence = {
            "operation_id": operation.operation_id,
            "master_channels": [47, 48],
            "peak_dbfs": peaks,
            "timestamp_utc": utc_now(),
        }
        self.evidence.append(evidence)
        if not peaks or max(peaks) <= -90.0:
            return AudioRecheckResult("failed", "master_audio_missing", evidence)
        if max(peaks) > -1.0:
            return AudioRecheckResult("failed", "master_headroom_insufficient", evidence)
        return AudioRecheckResult("passed", "master_audio_present_and_safe", evidence)


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    authorization_failures = execution_authorization_failures(args)
    if authorization_failures:
        raise PermissionError(
            "execution_authorization_failed:" + ",".join(authorization_failures)
        )
    client = WingClient(args.ip, args.port)
    if not client.connect():
        raise RuntimeError("wing_connect_failed")
    try:
        operations = safe_proof_operations(
            client,
            args.family,
            channel=args.channel,
            bus=args.bus,
            dca=args.dca,
            main=args.main,
            send_bus=args.send_bus,
        )
        contract_map = {item.family_id: item for item in FAMILY_CONTRACTS}
        selected = contract_map[args.family]
        selected = replace(selected, live_allowed_write_kinds=())
        audio_recheck = MasterAudioRecheck(args.websocket_url)
        adapters = build_wing_supervised_family_adapters(
            client,
            audio_recheck=audio_recheck,
            query_timeout=1.0,
        )
        emergency_reasons = []

        async def emergency_stop(reason: str) -> None:
            emergency_reasons.append(reason)

        coordinator = SupervisedConsoleTransactionCoordinator(
            family_contracts={args.family: selected},
            adapters={args.family: adapters[args.family]},
            emergency_generation_getter=lambda: 0,
            emergency_stop=emergency_stop,
        )
        plan = ConsoleTransactionPlan(
            transaction_id=f"hardware-proof:{args.family}:{utc_now()}",
            operations=operations,
            reason=f"Local restore-after-verify hardware proof for {args.family}",
        )
        result = await coordinator.execute(
            plan,
            ConsoleTransactionContext(
                approved=True,
                approval_id=str(args.approval_id).strip(),
                idempotency_key=plan.transaction_id,
                emergency_generation=0,
                proof_run=True,
                restore_after_verify=True,
            ),
        )
        proof_passed = result.status == "proof_passed"
        return {
            "schema": "wing-supervised-family-proof/v1",
            "timestamp_utc": utc_now(),
            "family_id": args.family,
            "wing_ip": args.ip,
            "firmware": client.state.get("/-stat/ver") or "3.0.5-observed",
            "operation_ids": [item.operation_id for item in operations],
            "result": result.to_dict(),
            "audio_evidence": audio_recheck.evidence,
            "emergency_reasons": emergency_reasons,
            "status": result.status,
            "proof_passed": proof_passed,
            "accepted": False,
            "owner_decision_required": True,
            "operator_approval_id": str(args.approval_id).strip(),
            "target_scope_confirmation": target_scope_fingerprint(args),
            "contacts_wing": True,
            "contacts_backend": True,
            "live_write": True,
            "readiness_matrix_update_allowed": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
        }
    finally:
        client.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=PROOF_FAMILIES, required=True)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", type=int, default=2223)
    parser.add_argument("--websocket-url", default="ws://127.0.0.1:8765")
    parser.add_argument("--channel", type=int, default=3)
    parser.add_argument("--bus", type=int, default=1)
    parser.add_argument("--dca", type=int, default=1)
    parser.add_argument("--main", type=int, default=1)
    parser.add_argument("--send-bus", type=int, default=13)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval-id")
    parser.add_argument("--confirm-live-write")
    parser.add_argument("--confirm-restore")
    parser.add_argument("--confirm-family")
    parser.add_argument("--confirm-ip")
    parser.add_argument("--confirm-target-scope")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "wing_supervised_family_proof",
        help="Artifact root; tests must provide a temporary directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    authorization_failures = execution_authorization_failures(args)
    exit_code = 0
    if authorization_failures:
        summary = planned_summary(args, authorization_failures)
        exit_code = 2 if args.execute else 0
    else:
        try:
            summary = asyncio.run(run(args))
            exit_code = 0 if summary.get("proof_passed") is True else 1
        except Exception as exc:
            summary = {
                "schema": "wing-supervised-family-proof/v1",
                "timestamp_utc": utc_now(),
                "family_id": args.family,
                "wing_ip": args.ip,
                "status": "failed",
                "proof_passed": False,
                "accepted": False,
                "owner_decision_required": True,
                "contacts_wing": True,
                "contacts_backend": None,
                "live_write": None,
                "physical_state_restored": None,
                "manual_review_required": True,
                "readiness_matrix_update_allowed": False,
                "readiness_closure_allowed": False,
                "live_authority_granted": False,
                "error": f"{type(exc).__name__}:{exc}",
            }
            exit_code = 1
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.artifact_root) / f"{run_id}_{args.family}"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"artifact": str(path), **summary}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
