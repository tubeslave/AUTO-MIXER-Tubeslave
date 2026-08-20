#!/usr/bin/env python3
"""Attended recovery for one incomplete dLive fader restore-proof WAL row."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from dlive_supervised_fader_proof import recover_dlive_fader_restore_proof  # noqa: E402


def resolve_route_interface(host: str) -> str | None:
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
    match = re.search(
        r"^\s*interface:\s*(\S+)\s*$",
        completed.stdout if completed.returncode == 0 else "",
        re.MULTILINE,
    )
    return match.group(1) if match else None


def write_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    raw = (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-attended", action="store_true", required=True)
    parser.add_argument("--recover-exact-preimage-only", action="store_true", required=True)
    parser.add_argument("--pa-outputs-contained", action="store_true", required=True)
    parser.add_argument("--input-confirmed-muted", action="store_true", required=True)
    parser.add_argument("--transaction-id", required=True)
    parser.add_argument("--tls", action="store_true")
    parser.add_argument("--tls-ca-file")
    parser.add_argument("--tls-server-hostname")
    parser.add_argument("--timeout-sec", type=float, default=0.75)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    password = getpass.getpass("dLive user password: ") if args.tls else ""
    handled_signals = tuple(
        item for item in (signal.SIGTERM, getattr(signal, "SIGHUP", None))
        if item is not None
    )
    prior_handlers = {item: signal.getsignal(item) for item in handled_signals}

    def request_fail_closed_exit(signum, _frame):
        raise KeyboardInterrupt(f"dlive_recovery_signal_{signum}")

    for item in handled_signals:
        signal.signal(item, request_fail_closed_exit)
    try:
        receipt = recover_dlive_fader_restore_proof(
            args.transaction_id,
            pa_outputs_contained=args.pa_outputs_contained,
            input_confirmed_muted=args.input_confirmed_muted,
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
            / "dlive_fader_restore_recovery"
            / f"{stamp}_{args.transaction_id}.json"
        )
    write_receipt(output, receipt)
    print(json.dumps({
        "status": receipt["status"],
        "reason": receipt["reason"],
        "artifact": str(output.resolve()),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "mutation_write_attempt_count": receipt["mutation_write_attempt_count"],
        "live_write": receipt["live_write"],
        "standing_live_authority": False,
    }, sort_keys=True))
    return 0 if receipt["status"] == "recovery_restored" else 2


if __name__ == "__main__":
    raise SystemExit(main())
