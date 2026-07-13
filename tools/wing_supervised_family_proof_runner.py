#!/usr/bin/env python3
"""Plan-only compatibility surface for the retired direct WING proof tool.

Real restore-only proof execution belongs exclusively to
``wing_attended_soundcheck_runner.py``.  That runner owns one backend process,
one authenticated WebSocket, the single-writer scope and the process-private
capability from route proof through verified rollback.  This compatibility
entry point never imports or constructs a mixer or network transport.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
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
CANONICAL_RUNNER = "tools/wing_attended_soundcheck_runner.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00",
        "Z",
    )


def target_scope_fingerprint(args: argparse.Namespace) -> str:
    """Retain the legacy plan fingerprint for old operator notes only."""

    return (
        f"family={args.family};ip={args.ip};channel={args.channel};"
        f"bus={args.bus};dca={args.dca};main={args.main};send_bus={args.send_bus}"
    )


def execution_authorization_failures(args: argparse.Namespace) -> list[str]:
    """Explain why this retired entry point cannot execute live proof."""

    failures: list[str] = []
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
    if str(getattr(args, "confirm_target_scope", "") or "") != (
        target_scope_fingerprint(args)
    ):
        failures.append("target_scope_confirmation_mismatch")
    for field_name, lower, upper in (
        ("channel", 1, 48),
        ("bus", 1, 16),
        ("dca", 1, 16),
        ("main", 1, 4),
        ("send_bus", 1, 16),
    ):
        value = getattr(args, field_name, None)
        if isinstance(value, bool) or not isinstance(value, int) or not (
            lower <= value <= upper
        ):
            failures.append(f"{field_name}_out_of_range")
    failures.append("canonical_attended_backend_session_required")
    return failures


def planned_summary(
    args: argparse.Namespace,
    failures: Iterable[str],
) -> Dict[str, Any]:
    failures = list(failures)
    execute_requested = bool(getattr(args, "execute", False))
    return {
        "schema": "wing-supervised-family-proof/v2",
        "timestamp_utc": utc_now(),
        "family_id": args.family,
        "wing_ip": args.ip,
        "status": "blocked" if execute_requested else "planned",
        "reason": (
            "canonical_attended_backend_session_required"
            if execute_requested
            else "plan_only_execute_flag_required"
        ),
        "authorization_failures": failures,
        "canonical_runner": CANONICAL_RUNNER,
        "required_live_execution_confirmation": LIVE_EXECUTION_CONFIRMATION,
        "required_restore_confirmation": RESTORE_CONFIRMATION,
        "required_target_scope_confirmation": target_scope_fingerprint(args),
        "contacts_wing": False,
        "contacts_backend": False,
        "opens_udp_or_osc": False,
        "live_write": False,
        "proof_passed": False,
        "accepted": False,
        "owner_decision_required": True,
        "readiness_matrix_update_allowed": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
    }


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Never execute from this compatibility module."""

    failures = execution_authorization_failures(args)
    raise PermissionError(
        "execution_authorization_failed:" + ",".join(failures)
    )


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
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures = execution_authorization_failures(args)
    summary = planned_summary(args, failures)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.artifact_root) / f"{run_id}_{args.family}"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"artifact": str(path), **summary}, ensure_ascii=False))
    return 2 if args.execute else 0


if __name__ == "__main__":
    raise SystemExit(main())
