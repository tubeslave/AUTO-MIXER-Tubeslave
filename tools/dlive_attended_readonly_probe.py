#!/usr/bin/env python3
"""Operator-attended dLive read-only identity/state probe.

This runner sends only documented GET requests for one input's name, fader and
mute. It cannot grant or accept mutation authority.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from dlive_client import DLiveClient  # noqa: E402


def resolve_route_interface(host: str) -> str | None:
    """Return the macOS route interface without sending network payload."""

    try:
        result = subprocess.run(
            ["route", "-n", "get", host],
            text=True,
            capture_output=True,
            check=False,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"^\s*interface:\s*(\S+)\s*$", result.stdout, re.MULTILINE)
    return match.group(1) if match else None


def run_probe(
    *,
    host: str,
    endpoint: str,
    port: int | None,
    tls: bool,
    tls_ca_file: str | None,
    tls_server_hostname: str | None,
    midi_base_channel: int,
    user_profile: int,
    password: str,
    input_channel: int,
    operator_attested_model: str,
    operator_attested_firmware: str,
    operator_attested_show: str,
    expected_name: str | None,
    timeout_sec: float,
    expected_interface: str,
    client_factory: Callable[..., Any] = DLiveClient,
    route_interface_resolver: Callable[[str], str | None] = resolve_route_interface,
) -> Dict[str, Any]:
    probe_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    artifact: Dict[str, Any] = {
        "schema": "automixer.dlive_attended_readonly_probe.v1",
        "probe_id": probe_id,
        "started_at_utc": started_at,
        "status": "blocked",
        "endpoint": endpoint,
        "host": host,
        "port": port,
        "tls": tls,
        "midi_base_channel": midi_base_channel,
        "user_profile": user_profile,
        "input_channel": input_channel,
        "expected_route_interface": expected_interface,
        "operator_attestation": {
            "model": operator_attested_model,
            "firmware": operator_attested_firmware,
            "show": operator_attested_show,
            "machine_verified": False,
        },
        "readbacks": {},
        "midi_get_request_count": 0,
        "mutation_write_attempt_count": 0,
        "live_write": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "console_contacted": False,
    }

    def finish_without_connection() -> Dict[str, Any]:
        artifact["teardown"] = {
            "disconnect_requested": False,
            "client_connected_after_teardown": False,
        }
        artifact["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        return artifact

    if not re.fullmatch(r"[A-Za-z0-9._-]+", expected_interface or ""):
        artifact["reason"] = "expected_route_interface_invalid"
        return finish_without_connection()
    observed_interface = route_interface_resolver(host)
    artifact["observed_route_interface"] = observed_interface
    if observed_interface != expected_interface:
        artifact["reason"] = "dlive_route_interface_mismatch"
        return finish_without_connection()

    client = client_factory(
        ip=host,
        endpoint=endpoint,
        port=port,
        tls=tls,
        tls_ca_file=tls_ca_file,
        tls_server_hostname=tls_server_hostname,
        midi_base_channel=midi_base_channel,
        user_profile=user_profile,
        password=password,
    )
    artifact["port"] = client.port

    connected = False
    try:
        connected = bool(client.connect(timeout=timeout_sec))
        artifact["console_contacted"] = connected
        if not connected:
            artifact["reason"] = "dlive_connect_failed"
            return artifact

        for address in (
            f"/ch/{input_channel}/name",
            f"/ch/{input_channel}/fdr",
            f"/ch/{input_channel}/mute",
        ):
            result = client.query_parameter(address, timeout_sec=timeout_sec)
            artifact["readbacks"][address] = result
            artifact["midi_get_request_count"] += 2

        statuses = {
            item.get("status")
            for item in artifact["readbacks"].values()
            if isinstance(item, dict)
        }
        if statuses != {"received"}:
            artifact["reason"] = "fresh_stable_readback_incomplete"
            return artifact

        actual_name = artifact["readbacks"][f"/ch/{input_channel}/name"].get("value")
        if expected_name is not None and actual_name != expected_name:
            artifact["reason"] = "operator_expected_channel_name_mismatch"
            artifact["expected_name"] = expected_name
            return artifact

        artifact["status"] = "readback_observed"
        artifact["reason"] = "three_fields_received_twice_no_mutation_authority"
        return artifact
    finally:
        if connected:
            client.disconnect()
        artifact["teardown"] = {
            "disconnect_requested": connected,
            "client_connected_after_teardown": bool(
                getattr(client, "is_connected", False)
            ),
        }
        artifact["finished_at_utc"] = datetime.now(timezone.utc).isoformat()


def _write_artifact(output: Path, artifact: Dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-attended", action="store_true", required=True)
    parser.add_argument("--ack-read-only", action="store_true", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--endpoint", choices=("surface", "mixrack"), required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--tls", action="store_true")
    parser.add_argument("--tls-ca-file")
    parser.add_argument("--tls-server-hostname")
    parser.add_argument("--midi-base-channel", type=int, default=0)
    parser.add_argument("--user-profile", type=int, default=0)
    parser.add_argument("--input-channel", type=int, required=True)
    parser.add_argument("--expected-interface", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--firmware", required=True)
    parser.add_argument("--show", required=True)
    parser.add_argument("--expected-name")
    parser.add_argument("--timeout-sec", type=float, default=0.75)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    password = getpass.getpass("dLive user password: ") if args.tls else ""
    artifact = run_probe(
        host=args.host,
        endpoint=args.endpoint,
        port=args.port,
        tls=args.tls,
        tls_ca_file=args.tls_ca_file,
        tls_server_hostname=args.tls_server_hostname,
        midi_base_channel=args.midi_base_channel,
        user_profile=args.user_profile,
        password=password,
        input_channel=args.input_channel,
        operator_attested_model=args.model,
        operator_attested_firmware=args.firmware,
        operator_attested_show=args.show,
        expected_name=args.expected_name,
        timeout_sec=args.timeout_sec,
        expected_interface=args.expected_interface,
    )
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = (
            Path.home()
            / ".automixer"
            / "runtime"
            / "dlive_readonly_probes"
            / f"{stamp}_{artifact['probe_id']}.json"
        )
    _write_artifact(output, artifact)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps({
        "status": artifact["status"],
        "reason": artifact["reason"],
        "artifact": str(output.resolve()),
        "sha256": digest,
        "live_write": False,
        "live_authority_granted": False,
    }, sort_keys=True))
    return 0 if artifact["status"] == "readback_observed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
