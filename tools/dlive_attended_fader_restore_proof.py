#!/usr/bin/env python3
"""Attended one-input dLive fader cut/readback/exact-rollback proof.

This command is intentionally restore-only. It requires a literal confirmation,
an already muted input, physical PA/output containment, a bounded downward move,
the expected wired interface, and exact channel identity. It never grants the
normal AUTO-MIXER server standing dLive write authority.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from dlive_supervised_fader_proof import (  # noqa: E402
    DLiveFaderProofApproval,
    build_dlive_fader_proof_confirmation,
    run_dlive_fader_restore_proof,
)


def resolve_route_interface(host: str) -> str | None:
    """Resolve the macOS route without sending console traffic."""

    try:
        completed = subprocess.run(
            ["route", "-n", "get", host],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    match = re.search(
        r"^\s*interface:\s*(\S+)\s*$",
        completed.stdout,
        re.MULTILINE,
    )
    return match.group(1) if match else None


def _write_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        os.write(descriptor, encoded.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-attended", action="store_true", required=True)
    parser.add_argument("--execute-restore-only-proof", action="store_true", required=True)
    parser.add_argument("--pa-outputs-contained", action="store_true", required=True)
    parser.add_argument("--input-confirmed-muted", action="store_true", required=True)
    parser.add_argument("--receive-midi-confirmed-enabled", action="store_true", required=True)
    parser.add_argument("--readonly-artifact", type=Path, required=True)
    parser.add_argument("--readonly-artifact-sha256", required=True)
    parser.add_argument("--wire-evidence", type=Path, required=True)
    parser.add_argument("--wire-evidence-sha256", required=True)
    parser.add_argument("--security-state-attested", action="store_true", required=True)
    parser.add_argument(
        "--security-mode",
        choices=("no_security", "tls_authenticated"),
        required=True,
    )
    parser.add_argument("--security-changed-for-setup", action="store_true")
    parser.add_argument(
        "--reboot-after-security-change-confirmed",
        action="store_true",
    )
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--approval-id", default=None)
    parser.add_argument("--approval-lifetime-sec", type=float, default=120.0)
    parser.add_argument("--host", required=True)
    parser.add_argument("--endpoint", choices=("surface", "mixrack"), required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--tls", action="store_true")
    parser.add_argument("--tls-ca-file")
    parser.add_argument("--tls-server-hostname")
    parser.add_argument("--midi-base-channel", type=int, default=11)
    parser.add_argument("--user-profile", type=int, default=0)
    parser.add_argument("--expected-interface", required=True)
    parser.add_argument("--input-channel", type=int, required=True)
    parser.add_argument("--expected-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--firmware", required=True)
    parser.add_argument("--show", required=True)
    parser.add_argument("--target-db", type=float, required=True)
    parser.add_argument("--max-cut-db", type=float, default=3.0)
    parser.add_argument("--timeout-sec", type=float, default=0.75)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    now = time.time()
    approval = DLiveFaderProofApproval(
        approval_id=args.approval_id or str(uuid.uuid4()),
        issued_at_epoch=now,
        expires_at_epoch=now + args.approval_lifetime_sec,
        host=args.host,
        endpoint=args.endpoint,
        port=args.port,
        tls=args.tls,
        midi_base_channel=args.midi_base_channel,
        user_profile=args.user_profile,
        expected_interface=args.expected_interface,
        input_channel=args.input_channel,
        expected_name=args.expected_name,
        operator_attested_model=args.model,
        operator_attested_firmware=args.firmware,
        operator_attested_show=args.show,
        target_db=args.target_db,
        max_cut_db=args.max_cut_db,
        pa_outputs_contained=args.pa_outputs_contained,
        require_channel_muted=args.input_confirmed_muted,
        receive_midi_confirmed_enabled=args.receive_midi_confirmed_enabled,
        readonly_artifact_path=str(args.readonly_artifact),
        readonly_artifact_sha256=args.readonly_artifact_sha256,
        confirmation=args.confirmation,
        security_state_attested=args.security_state_attested,
        security_mode_attested=args.security_mode,
        security_changed_for_setup=args.security_changed_for_setup,
        reboot_after_security_change_confirmed=(
            args.reboot_after_security_change_confirmed
        ),
        wire_evidence_path=str(args.wire_evidence),
        wire_evidence_sha256=args.wire_evidence_sha256,
    )
    required_confirmation = build_dlive_fader_proof_confirmation(approval)
    if args.confirmation != required_confirmation:
        print(json.dumps({
            "status": "blocked",
            "reason": "artifact_target_bound_operator_confirmation_required",
            "required_confirmation": required_confirmation,
            "required_approval_id": approval.approval_id,
            "hardware_contacted": False,
            "live_write": False,
        }, sort_keys=True))
        return 2
    password = getpass.getpass("dLive user password: ") if args.tls else ""
    handled_signals = tuple(
        item for item in (signal.SIGTERM, getattr(signal, "SIGHUP", None))
        if item is not None
    )
    prior_handlers = {item: signal.getsignal(item) for item in handled_signals}

    def request_emergency_recovery(signum, _frame):
        raise KeyboardInterrupt(f"attended_dlive_proof_signal_{signum}")

    for item in handled_signals:
        signal.signal(item, request_emergency_recovery)
    try:
        receipt = run_dlive_fader_restore_proof(
            approval,
            password=password,
            tls_ca_file=args.tls_ca_file,
            tls_server_hostname=args.tls_server_hostname,
            timeout_sec=args.timeout_sec,
            route_interface_resolver=resolve_route_interface,
        )
    finally:
        for item, handler in prior_handlers.items():
            signal.signal(item, handler)
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = (
            Path.home()
            / ".automixer"
            / "runtime"
            / "dlive_fader_restore_proofs"
            / f"{stamp}_{receipt['transaction_id']}.json"
        )
    _write_receipt(output, receipt)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps({
        "status": receipt["status"],
        "reason": receipt["reason"],
        "artifact": str(output.resolve()),
        "sha256": digest,
        "mutation_write_attempt_count": receipt["mutation_write_attempt_count"],
        "live_write": receipt["live_write"],
        "standing_live_authority": False,
        "client_connected_after_teardown": receipt["client_connected_after_teardown"],
    }, sort_keys=True))
    return 0 if receipt["status"] == "proof_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
