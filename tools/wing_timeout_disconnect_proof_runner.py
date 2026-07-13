#!/usr/bin/env python3
"""Run the M005 WING disconnect/timeout/stale-state proof safely.

The runner talks only to the backend WebSocket. It never sends direct write
messages such as set_fader/set_gain. The blocked-write probe is attempted only
after the runner has evidence that fresh physical readback is unavailable.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
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
    summarize_transport_failure,
    utc_stamp,
)
from runtime_identity import (  # noqa: E402
    RUNTIME_IDENTITY_SCHEMA,
    canonical_json_bytes,
    capture_source_runtime_components,
    runtime_identity_manifest_sha256,
    valid_lower_hex,
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
    "manual_write_rollback_unverified",
}
EXPECTED_RUNTIME_IDENTITY_EVENTS = {"runtime_identity"}
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
M005_CHILD_ENV_ALLOWLIST = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "PATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TMP",
        "TMPDIR",
        "TZ",
        "USER",
    }
)
M005_REQUIRED_EVIDENCE = (
    "physical_baseline_readback",
    "operator_confirmed_physical_cable_loss",
    "stale_readback_unavailable",
    "blocked_write_while_disconnected",
    "reconnect_recovery_verified",
)
M005_EVIDENCE_DESCRIPTIONS = {
    "physical_baseline_readback": "physical baseline readback before link loss",
    "operator_confirmed_physical_cable_loss": (
        "operator-confirmed physical cable-loss phase still requires supervised WING cable pull"
    ),
    "stale_readback_unavailable": "stale-state query must report readback unavailable",
    "blocked_write_while_disconnected": "blocked write probe must be blocked while disconnected",
    "reconnect_recovery_verified": "reconnect recovery must verify channel unchanged from baseline",
}
M005_PROVENANCE_SCHEMA = "automixer.m005_current_release_provenance.v1"
M005_CONSOLE_IDENTITY_SCHEMA = "automixer.wing_console_identity_owner_challenge.v1"
M005_CONSOLE_IDENTITY_ROOT = REPO_ROOT / "artifacts" / "wing_console_identity_owner_challenge"
M005_RUNTIME_IDENTITY_SCHEMA = RUNTIME_IDENTITY_SCHEMA
CANONICAL_BACKEND_URL = "ws://127.0.0.1:8765"
CANONICAL_SOURCE_BACKEND = (REPO_ROOT / "backend" / "server.py").resolve()
EXPECTED_WING_MODEL = "WING Rack"
EXPECTED_WING_FIRMWARE = "3.0.5"
MAX_BOUND_BACKEND_START_AGE_SECONDS = 600


class BoundBackendLaunchError(RuntimeError):
    """The proof-owned backend could not be started without ambiguity."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_identity(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    try:
        display_path = str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        display_path = str(resolved)
    if not resolved.is_file():
        return {"path": display_path, "exists": False, "size": None, "sha256": None}
    value = resolved.read_bytes()
    return {
        "path": display_path,
        "exists": True,
        "size": len(value),
        "sha256": _sha256_bytes(value),
    }


def _git_bytes(*args: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", *args],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({completed.returncode}): "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return completed.stdout


def _canonical_json_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _console_identity_challenge_path(args: argparse.Namespace) -> Path | None:
    if not args.console_identity_artifact:
        return None
    path = Path(args.console_identity_artifact)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def load_console_identity_challenge(args: argparse.Namespace) -> Dict[str, Any] | None:
    path = _console_identity_challenge_path(args)
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _console_identity_challenge_error(args: argparse.Namespace) -> Optional[str]:
    path = _console_identity_challenge_path(args)
    if path is None or not path.is_file():
        return "--console-identity-artifact is required and must exist for physical M005 proof"
    try:
        path.relative_to(M005_CONSOLE_IDENTITY_ROOT)
    except ValueError:
        return "--console-identity-artifact must be under the canonical owner-challenge root"
    if path.name != "summary.json":
        return "--console-identity-artifact must name a canonical summary.json"
    challenge = load_console_identity_challenge(args)
    if challenge is None:
        return "--console-identity-artifact must contain valid JSON"
    if challenge.get("schema") != M005_CONSOLE_IDENTITY_SCHEMA:
        return "console identity owner challenge schema mismatch"
    if challenge.get("operator_confirmed") is not True:
        return "console identity owner challenge must be operator-confirmed"
    if challenge.get("console_model") != EXPECTED_WING_MODEL:
        return f"console identity owner challenge must identify {EXPECTED_WING_MODEL}"
    if challenge.get("firmware") != EXPECTED_WING_FIRMWARE:
        return f"console identity owner challenge firmware must be {EXPECTED_WING_FIRMWARE}"
    if not str(challenge.get("serial") or "").strip():
        return "console identity owner challenge serial is required"
    if challenge.get("accepted_readonly_source_id") != args.accepted_readonly_source_id:
        return "console identity owner challenge must bind the accepted read-only preflight"
    if str(args.console_serial or "").strip() != str(challenge.get("serial")):
        return "--console-serial must match the owner challenge"
    if str(args.console_firmware or "").strip() != str(challenge.get("firmware")):
        return "--console-firmware must match the owner challenge"
    return None


def capture_current_release_provenance(args: argparse.Namespace) -> Dict[str, Any]:
    status = _git_bytes("status", "--porcelain=v1", "-uall")
    tracked_diff = _git_bytes("diff", "HEAD", "--binary")
    backend_artifact = Path(args.backend_artifact)
    if not backend_artifact.is_absolute():
        backend_artifact = REPO_ROOT / backend_artifact
    challenge_path = _console_identity_challenge_path(args)
    challenge = load_console_identity_challenge(args)
    return {
        "schema": M005_PROVENANCE_SCHEMA,
        "captured_at_utc": _iso_now(),
        "source": {
            "head": _git_bytes("rev-parse", "HEAD").decode().strip(),
            "tree": _git_bytes("rev-parse", "HEAD^{tree}").decode().strip(),
            "worktree_dirty": bool(status.strip()),
            "status_sha256": _sha256_bytes(status),
            "tracked_diff_sha256": _sha256_bytes(tracked_diff),
        },
        "runtime_source": capture_source_runtime_components(REPO_ROOT),
        "runner": _file_identity(Path(__file__)),
        "backend": {
            "runtime_kind": args.backend_runtime_kind,
            **_file_identity(backend_artifact),
        },
        "package": None,
        "console": {
            "model": challenge.get("console_model") if challenge else None,
            "serial": challenge.get("serial") if challenge else None,
            "firmware": challenge.get("firmware") if challenge else None,
            "identity_source": "operator_attested_owner_challenge",
            "console_identity_query_available": False,
        },
        "console_identity_challenge": (
            _file_identity(challenge_path) if challenge_path is not None else None
        ),
        "events": None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run supervised M005 disconnect/timeout/stale-state proof."
    )
    parser.add_argument("--url", default=CANONICAL_BACKEND_URL)
    parser.add_argument("--channel", type=int, default=3)
    parser.add_argument("--value-type", choices=["fader", "gain"], default="fader")
    parser.add_argument("--delta-db", type=float, default=-0.25)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--backend-startup-timeout",
        type=float,
        default=20.0,
        help="Maximum seconds for the proof-owned bound source backend to listen.",
    )
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
    parser.add_argument(
        "--operator-gated-physical-loss",
        action="store_true",
        help=(
            "Pause for stdin confirmation after the operator removes the cable and "
            "again after the operator reconnects it."
        ),
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
    parser.add_argument(
        "--wing-ip",
        help=(
            "Exact WING address for the initial proof-owned connection. When omitted, "
            "--reconnect-wing-ip is reused."
        ),
    )
    parser.add_argument("--reconnect-send-port", type=int, default=2222)
    parser.add_argument("--reconnect-receive-port", type=int, default=2223)
    parser.add_argument("--console-serial")
    parser.add_argument("--console-firmware")
    parser.add_argument(
        "--console-identity-artifact",
        help=(
            "Pre-existing operator-confirmed WING identity owner challenge under "
            "artifacts/wing_console_identity_owner_challenge/."
        ),
    )
    parser.add_argument(
        "--backend-runtime-kind",
        choices=["source", "packaged"],
        default="source",
    )
    parser.add_argument(
        "--backend-artifact",
        default="backend/server.py",
        help="Exact backend source or executable used by the attended proof.",
    )
    parser.add_argument(
        "--package-artifact",
        help="Required when --backend-runtime-kind=packaged; exact app/archive used.",
    )
    parser.add_argument(
        "--rollback-on-unexpected-apply",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def validate_args(args: argparse.Namespace) -> Optional[str]:
    if args.url != CANONICAL_BACKEND_URL:
        return f"--url must equal the canonical loopback backend URL {CANONICAL_BACKEND_URL}"
    if args.channel < 1 or args.channel > 48:
        return "--channel must be in 1..48"
    if args.delta_db >= 0.0 or abs(args.delta_db) > 0.5:
        return "--delta-db must be negative and no larger than 0.5 dB"
    if args.timeout <= 0:
        return "--timeout must be positive"
    if args.backend_startup_timeout < 1 or args.backend_startup_timeout > 60:
        return "--backend-startup-timeout must be between 1 and 60 seconds"
    if args.physical_loss_wait_sec < 0 or args.physical_loss_wait_sec > 30:
        return "--physical-loss-wait-sec must be between 0 and 30 seconds"
    if args.execute:
        artifact_root_error = _execute_artifact_root_error(args)
        if artifact_root_error:
            return artifact_root_error
        if not (args.wing_ip or args.reconnect_wing_ip):
            return "--wing-ip or --reconnect-wing-ip is required with --execute"
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
        if not str(args.console_serial or "").strip():
            return "--console-serial is required with --execute"
        if not str(args.console_firmware or "").strip():
            return "--console-firmware is required with --execute"
        backend_artifact = Path(args.backend_artifact)
        if not backend_artifact.is_absolute():
            backend_artifact = REPO_ROOT / backend_artifact
        if args.backend_runtime_kind != "source":
            return "packaged M005 proof is unsupported until an authenticated package path exists"
        if backend_artifact.resolve() != CANONICAL_SOURCE_BACKEND or not backend_artifact.is_file():
            return "--backend-artifact must equal the canonical backend/server.py source"
        if args.package_artifact:
            return "--package-artifact is forbidden for source M005 proof"
    if args.disconnect_mode == "physical-cable-loss":
        if args.operator_confirm_physical_cable_loss != PHYSICAL_CABLE_CONFIRMATION:
            return (
                "--operator-confirm-physical-cable-loss must equal "
                f"{PHYSICAL_CABLE_CONFIRMATION}"
            )
        if args.execute and not args.reconnect_wing_ip:
            return "--reconnect-wing-ip is required for executed physical-cable-loss proof"
        if args.execute and not args.operator_gated_physical_loss:
            return "--operator-gated-physical-loss is required for executed physical M005 proof"
        if args.execute:
            challenge_error = _console_identity_challenge_error(args)
            if challenge_error:
                return challenge_error
    return None


def _canonical_m005_artifact_root_is_ignored() -> bool:
    probe = f"{DEFAULT_ARTIFACT_ROOT}/.m005-runtime-identity-probe/events.jsonl"
    completed = subprocess.run(
        ["/usr/bin/git", "check-ignore", "--no-index", "--quiet", probe],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=5,
    )
    return completed.returncode == 0


def _execute_artifact_root_error(args: argparse.Namespace) -> Optional[str]:
    if str(args.artifact_root) != DEFAULT_ARTIFACT_ROOT:
        return f"--artifact-root must equal {DEFAULT_ARTIFACT_ROOT} with --execute"
    if not _canonical_m005_artifact_root_is_ignored():
        return "canonical M005 artifact root must be ignored before --execute"
    return None


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


async def _wait_for_operator(prompt: str) -> None:
    """Wait without blocking the event loop while an attended hardware step completes."""
    await asyncio.to_thread(input, prompt)


def _loopback_port_accepting(port: int = 8765) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", int(port))) == 0


async def _wait_for_bound_backend(process: subprocess.Popen[Any], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise BoundBackendLaunchError(
                f"proof-owned backend exited before listening (code={process.returncode})"
            )
        if await asyncio.to_thread(_loopback_port_accepting, 8765):
            return
        await asyncio.sleep(0.1)
    raise BoundBackendLaunchError(
        f"proof-owned backend did not listen within {timeout:.1f} seconds"
    )


async def _stop_bound_backend(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        await asyncio.to_thread(process.wait, 5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        await asyncio.to_thread(process.wait, 5)


@asynccontextmanager
async def _launch_bound_backend(
    args: argparse.Namespace,
    run_dir: Path,
    *,
    launch_nonce: str,
    prestart_provenance_sha256: str,
    prestart_provenance_size: int,
    auth_token: str,
):
    if await asyncio.to_thread(_loopback_port_accepting, 8765):
        raise BoundBackendLaunchError(
            "canonical backend port 8765 is already in use; M005 requires its own bound process"
        )
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in M005_CHILD_ENV_ALLOWLIST and isinstance(value, str) and value
    }
    environment.update(
        {
            "AUTOMIXER_WS_HOST": "127.0.0.1",
            "AUTOMIXER_WS_PORT": "8765",
            "AUTOMIXER_WS_CLIENT_ID": "automixer-ui",
            "AUTOMIXER_WS_TOKEN_TTL_HOURS": "1",
            "AUTOMIXER_WS_ALLOWED_ORIGINS": "",
            "AUTOMIXER_FORCE_CLEANUP_WS_PORT": "false",
            "AUTOMIXER_REPO_ROOT": str(REPO_ROOT),
            "AUTOMIXER_M005_LAUNCH_NONCE": launch_nonce,
            "AUTOMIXER_M005_PRESTART_PROVENANCE_SCHEMA": M005_PROVENANCE_SCHEMA,
            "AUTOMIXER_M005_PRESTART_PROVENANCE_SHA256": prestart_provenance_sha256,
            "AUTOMIXER_M005_PRESTART_PROVENANCE_SIZE": str(prestart_provenance_size),
            "PYTHONPATH": os.pathsep.join([str(BACKEND_ROOT), str(REPO_ROOT)]),
        }
    )
    # Never place the live-control bearer in the child process environment.
    # The source backend consumes it once from an inherited pipe and closes the
    # descriptor before it starts accepting WebSocket clients.
    read_fd, write_fd = os.pipe()
    environment["AUTOMIXER_WS_AUTH_TOKEN_FD"] = str(read_fd)
    stdout_path = run_dir / "backend.stdout.log"
    stderr_path = run_dir / "backend.stderr.log"
    process: Optional[subprocess.Popen[Any]] = None
    try:
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(
                [sys.executable, str(CANONICAL_SOURCE_BACKEND)],
                cwd=REPO_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
                pass_fds=(read_fd,),
            )
        os.close(read_fd)
        read_fd = -1
        token_bytes = bytearray(auth_token.encode("ascii"))
        try:
            offset = 0
            while offset < len(token_bytes):
                offset += os.write(write_fd, token_bytes[offset:])
        finally:
            token_bytes[:] = b"\x00" * len(token_bytes)
            os.close(write_fd)
            write_fd = -1
            auth_token = ""
        await _wait_for_bound_backend(process, args.backend_startup_timeout)
        yield process
    finally:
        for fd in (read_fd, write_fd):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        auth_token = ""
        if process is not None:
            await _stop_bound_backend(process)


def _readback_is_physical(payload: Dict[str, Any]) -> bool:
    return (
        payload.get("type") == "mixer_parameter_value"
        and payload.get("status") in PHYSICAL_READBACK_STATUSES
        and payload.get("source") == "physical_readback"
        and payload.get("value") is not None
    )


def _runtime_identity_matches_provenance(
    payload: Dict[str, Any],
    provenance: Dict[str, Any],
    *,
    request_challenge_nonce: str,
    expected_backend_pid: int,
) -> bool:
    backend = provenance.get("backend") if isinstance(provenance.get("backend"), dict) else {}
    runtime_backend = payload.get("backend") if isinstance(payload.get("backend"), dict) else {}
    websocket = payload.get("websocket") if isinstance(payload.get("websocket"), dict) else {}
    process = payload.get("process") if isinstance(payload.get("process"), dict) else {}
    launch_binding = (
        payload.get("launch_binding")
        if isinstance(payload.get("launch_binding"), dict)
        else {}
    )
    runtime_source = {
        "source": payload.get("source") if isinstance(payload.get("source"), dict) else {},
        "code": payload.get("code") if isinstance(payload.get("code"), dict) else {},
    }
    expected_runtime_source = (
        provenance.get("runtime_source")
        if isinstance(provenance.get("runtime_source"), dict)
        else {}
    )
    nonce = str(payload.get("instance_nonce") or "")
    try:
        captured_at = datetime.fromisoformat(
            str(provenance.get("captured_at_utc") or "").replace("Z", "+00:00")
        )
        started_at = datetime.fromisoformat(
            str(payload.get("started_at_utc") or "").replace("Z", "+00:00")
        )
        start_delay = (started_at - captured_at).total_seconds()
    except (TypeError, ValueError):
        start_delay = -1
    prestart_provenance = dict(provenance)
    prestart_provenance["events"] = None
    expected_prestart_sha256 = _canonical_json_sha256(prestart_provenance)
    return (
        payload.get("type") == "runtime_identity"
        and payload.get("schema") == M005_RUNTIME_IDENTITY_SCHEMA
        and payload.get("runtime_kind") == "source"
        and valid_lower_hex(nonce, 32)
        and process.get("instance_nonce") == nonce
        and process.get("started_at_utc") == payload.get("started_at_utc")
        and process.get("pid") == expected_backend_pid
        and 0 <= start_delay <= MAX_BOUND_BACKEND_START_AGE_SECONDS
        and payload.get("read_only_identity") is True
        and payload.get("live_authority_granted") is False
        and payload.get("request_challenge_nonce") == request_challenge_nonce
        and valid_lower_hex(request_challenge_nonce, 32)
        and launch_binding.get("mode") == "proof_runner"
        and launch_binding.get("launch_nonce") == provenance.get("runtime_launch_nonce")
        and launch_binding.get("prestart_provenance_schema") == M005_PROVENANCE_SCHEMA
        and launch_binding.get("prestart_provenance_sha256") == expected_prestart_sha256
        and launch_binding.get("prestart_provenance_size")
        == len(canonical_json_bytes(prestart_provenance))
        and launch_binding.get("bound_before_process_start") is True
        and launch_binding.get("proof_eligible") is True
        and runtime_source == expected_runtime_source
        and websocket.get("loopback_only") is True
        and websocket.get("host") == "127.0.0.1"
        and websocket.get("port") == 8765
        and runtime_backend == {
            key: backend.get(key) for key in ("path", "exists", "size", "sha256")
        }
        and payload.get("manifest_sha256") == runtime_identity_manifest_sha256(payload)
    )


def _payload_target_field(payload: Dict[str, Any], field_name: str) -> Any:
    if field_name in payload:
        return payload.get(field_name)
    request = payload.get("request")
    if isinstance(request, dict):
        return request.get(field_name)
    return None


def _readback_matches_target(
    payload: Dict[str, Any],
    *,
    channel: int,
    value_type: str,
    address: str,
) -> bool:
    payload_channel = _payload_target_field(payload, "channel")
    try:
        channel_matches = int(payload_channel) == int(channel)
    except (TypeError, ValueError):
        channel_matches = False
    return (
        channel_matches
        and _payload_target_field(payload, "value_type") == value_type
        and _payload_target_field(payload, "address") == address
    )


def _readback_is_physical_for_target(
    payload: Dict[str, Any],
    *,
    channel: int,
    value_type: str,
    address: str,
) -> bool:
    return _readback_is_physical(payload) and _readback_matches_target(
        payload,
        channel=channel,
        value_type=value_type,
        address=address,
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


def _readback_is_unavailable_for_target(
    payload: Dict[str, Any],
    *,
    channel: int,
    value_type: str,
    address: str,
) -> bool:
    return _readback_is_unavailable(payload) and _readback_matches_target(
        payload,
        channel=channel,
        value_type=value_type,
        address=address,
    )


def _timeout_readback_unavailable(
    *,
    channel: int,
    value_type: str,
    address: str,
    request_id: str,
    timeout: float,
    error: BaseException,
) -> Dict[str, Any]:
    return {
        "type": "mixer_parameter_read_blocked",
        "status": "readback_timeout",
        "source": "timeout",
        "reason": "stale_readback_timeout_after_disconnect",
        "address": address,
        "channel": channel,
        "value_type": value_type,
        "value": None,
        "request_id": request_id,
        "timeout_sec": timeout,
        "transport_failure": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


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


def _blocked_write_phase_is_safe(blocked_write: Dict[str, Any]) -> bool:
    apply_result = (
        blocked_write.get("apply_result")
        if isinstance(blocked_write.get("apply_result"), dict)
        else {}
    )
    return (
        blocked_write.get("status") == "passed"
        and blocked_write.get("block_reason") in SAFE_BLOCK_REASONS
        and _apply_probe_blocked(apply_result)
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


def _completed_m005_evidence(summary: Dict[str, Any]) -> list[str]:
    phases = summary.get("phases") if isinstance(summary.get("phases"), dict) else {}
    completed: list[str] = []
    try:
        channel = int(summary.get("channel"))
    except (TypeError, ValueError):
        channel = 0
    value_type = str(summary.get("value_type") or "")
    address = str(summary.get("address") or "")

    baseline = phases.get("baseline") if isinstance(phases.get("baseline"), dict) else {}
    baseline_readback = baseline.get("readback") if isinstance(baseline.get("readback"), dict) else {}
    if baseline.get("status") == "passed" and _readback_is_physical_for_target(
        baseline_readback,
        channel=channel,
        value_type=value_type,
        address=address,
    ):
        completed.append("physical_baseline_readback")

    disconnect = phases.get("disconnect") if isinstance(phases.get("disconnect"), dict) else {}
    if (
        summary.get("disconnect_mode") == "physical-cable-loss"
        and disconnect.get("status") == "operator_confirmed_physical_cable_loss"
        and disconnect.get("confirmation") == PHYSICAL_CABLE_CONFIRMATION
    ):
        completed.append("operator_confirmed_physical_cable_loss")

    stale_readback = (
        phases.get("stale_readback") if isinstance(phases.get("stale_readback"), dict) else {}
    )
    if stale_readback.get("status") == "passed" and _readback_is_unavailable(
        stale_readback.get("readback") if isinstance(stale_readback.get("readback"), dict) else {}
    ) and _readback_matches_target(
        stale_readback.get("readback") if isinstance(stale_readback.get("readback"), dict) else {},
        channel=channel,
        value_type=value_type,
        address=address,
    ):
        completed.append("stale_readback_unavailable")

    blocked_write = (
        phases.get("blocked_write_while_disconnected")
        if isinstance(phases.get("blocked_write_while_disconnected"), dict)
        else {}
    )
    if _blocked_write_phase_is_safe(blocked_write):
        completed.append("blocked_write_while_disconnected")

    reconnect = (
        phases.get("reconnect_recovery")
        if isinstance(phases.get("reconnect_recovery"), dict)
        else {}
    )
    reconnect_readback = (
        reconnect.get("readback") if isinstance(reconnect.get("readback"), dict) else {}
    )
    reconnect_status = (
        reconnect.get("connection_status")
        if isinstance(reconnect.get("connection_status"), dict)
        else {}
    )
    if (
        reconnect.get("status") == "passed"
        and reconnect_status.get("connected") is True
        and reconnect.get("unchanged_from_baseline") is True
        and _readback_is_physical_for_target(
            reconnect_readback,
            channel=channel,
            value_type=value_type,
            address=address,
        )
    ):
        completed.append("reconnect_recovery_verified")

    return completed


def _annotate_m005_evidence(summary: Dict[str, Any]) -> Dict[str, Any]:
    completed = _completed_m005_evidence(summary)
    missing = [item for item in M005_REQUIRED_EVIDENCE if item not in completed]
    summary["required_m005_evidence"] = list(M005_REQUIRED_EVIDENCE)
    summary["completed_m005_evidence"] = completed
    summary["missing_m005_evidence"] = missing
    summary["remaining_m005_evidence"] = [M005_EVIDENCE_DESCRIPTIONS[item] for item in missing]
    phases = summary.get("phases") if isinstance(summary.get("phases"), dict) else {}
    initial_connection = (
        phases.get("initial_connection")
        if isinstance(phases.get("initial_connection"), dict)
        else {}
    )
    initial_status = (
        initial_connection.get("connection_status")
        if isinstance(initial_connection.get("connection_status"), dict)
        else {}
    )
    initial_connection_verified = bool(
        initial_connection.get("status") == "passed"
        and initial_status.get("connected") is True
        and initial_status.get("observation_only") is False
        and initial_status.get("record_telemetry") is False
    )
    summary["initial_connection_verified"] = initial_connection_verified
    summary["m005_complete"] = not missing and initial_connection_verified
    return summary


class M005EventRecorder(EventRecorder):
    def write_summary(self, summary: Dict[str, Any]) -> None:
        annotated = _annotate_m005_evidence(summary)
        provenance = annotated.get("provenance")
        if isinstance(provenance, dict):
            provenance = dict(provenance)
            provenance["events"] = _file_identity(self.events_path)
            annotated["provenance"] = provenance
            annotated["m005_current_release_provenance_captured"] = bool(
                provenance.get("schema") == M005_PROVENANCE_SCHEMA
                and provenance["events"].get("exists") is True
                and provenance["events"].get("sha256")
                and provenance.get("runner", {}).get("exists") is True
                and provenance.get("backend", {}).get("exists") is True
                and provenance.get("source", {}).get("worktree_dirty") is False
                and valid_lower_hex(provenance.get("runtime_launch_nonce"), 32)
                and provenance.get("runtime_source", {}).get("code", {}).get(
                    "files_sha256"
                )
                == provenance.get("runtime_source", {}).get("source", {}).get(
                    "source_content_sha256"
                )
                and provenance.get("console", {}).get("serial")
                and provenance.get("console", {}).get("firmware")
                and (provenance.get("console_identity_challenge") or {}).get("exists") is True
                and annotated.get("url") == CANONICAL_BACKEND_URL
            )
        else:
            annotated["m005_current_release_provenance_captured"] = False
        annotated["m005_behavioral_proof_complete"] = annotated["m005_complete"]
        annotated["m005_backend_runtime_identity_captured"] = bool(
            isinstance(annotated.get("runtime_identity"), dict)
            and _runtime_identity_matches_provenance(
                annotated["runtime_identity"],
                provenance if isinstance(provenance, dict) else {},
                request_challenge_nonce=str(
                    annotated.get("runtime_identity_challenge_nonce") or ""
                ),
                expected_backend_pid=int(annotated.get("launched_backend_pid") or 0),
            )
        )
        super().write_summary(annotated)


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


async def _connect_wing_and_wait_connected(
    ws: Any,
    recorder: EventRecorder,
    *,
    ip: str,
    send_port: int,
    receive_port: int,
    observation_only: bool,
    timeout: float,
) -> Dict[str, Any]:
    """Ignore disconnect transitions and wait for the requested WING mode."""
    request = {
        "type": "connect_wing",
        "ip": ip,
        "send_port": send_port,
        "receive_port": receive_port,
        "observation_only": bool(observation_only),
        "record_telemetry": False,
    }
    recorder.record("send", request)
    await ws.send(json.dumps(request))
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for connected WING connection")
        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"type": "invalid_json", "raw": raw}
        recorder.record("recv", payload)
        if payload.get("type") == "connection_status" and payload.get("connected") is True:
            return payload


async def _authenticate_proof_websocket(
    ws: Any,
    token: str,
    *,
    timeout: float,
) -> None:
    """Authenticate without ever copying the bearer into proof artifacts."""

    await ws.send(
        json.dumps(
            {
                "type": "authenticate",
                "token": token,
                "client_id": "automixer-ui",
            }
        )
    )
    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError("M005 backend authentication returned invalid JSON") from exc
    if payload != {"type": "authentication_status", "authenticated": True}:
        raise RuntimeError("M005 backend authentication failed")


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
    transaction_id: str,
) -> Dict[str, Any]:
    return await send_and_wait(
        ws,
        recorder,
        {
            "type": "rollback_last_manual_write",
            "approved": True,
            "approval_id": f"{approval_prefix}:rollback-unexpected:{run_id}",
            "reason": f"rollback_unexpected_m005_apply:{run_id}",
            "transaction_id": transaction_id,
        },
        EXPECTED_ROLLBACK_EVENTS,
        timeout=timeout,
    )


async def run(args: argparse.Namespace) -> int:
    validation_error = validate_args(args)
    provenance: Dict[str, Any] | None = None
    provenance_error: str | None = None
    runtime_launch_nonce = uuid.uuid4().hex
    runtime_identity_challenge_nonce = uuid.uuid4().hex
    runtime_auth_token = secrets.token_urlsafe(32)
    if validation_error is None:
        try:
            provenance = capture_current_release_provenance(args)
            provenance["runtime_launch_nonce"] = runtime_launch_nonce
        except (OSError, RuntimeError) as exc:
            provenance_error = str(exc)

    run_id = (
        f"{utc_stamp()}_{uuid.uuid4()}_wing_m005_disconnect_timeout_stale_proof"
    )
    run_dir = Path(args.artifact_root) / run_id
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    recorder = M005EventRecorder(run_dir)
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
        "wing_ip": args.wing_ip or args.reconnect_wing_ip,
        "write_policy": "blocked_write_attempts_only",
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "readiness_matrix_update_allowed": False,
        "physical_cable_confirmation": args.operator_confirm_physical_cable_loss,
        "runtime_identity_challenge_nonce": runtime_identity_challenge_nonce,
        "phases": {},
    }

    if provenance is not None:
        summary["provenance"] = provenance
        summary["provenance_prestart_sha256"] = _canonical_json_sha256(provenance)
        recorder.record(
            "provenance_capture",
            {
                "schema": M005_PROVENANCE_SCHEMA,
                "provenance": provenance,
                "provenance_sha256": summary["provenance_prestart_sha256"],
                "url": args.url,
                "captured_before_backend_contact": True,
            },
        )

    if validation_error:
        summary.update({"status": "blocked", "reason": validation_error})
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return 2

    if provenance_error is not None or provenance is None:
        summary.update(
            {
                "status": "blocked",
                "reason": "m005_current_release_provenance_capture_failed",
                "fail_closed": True,
                "provenance_error": provenance_error or "missing_provenance",
            }
        )
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return 14

    if args.execute and provenance.get("source", {}).get("worktree_dirty") is not False:
        summary.update(
            {
                "status": "blocked",
                "reason": "m005_execute_requires_clean_checkout_before_artifact_creation",
                "fail_closed": True,
            }
        )
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return 15

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
    backend_context = None
    backend_context_entered = False
    try:
        prestart_sha256 = _canonical_json_sha256(provenance)
        backend_context = _launch_bound_backend(
            args,
            run_dir,
            launch_nonce=runtime_launch_nonce,
            prestart_provenance_sha256=prestart_sha256,
            prestart_provenance_size=len(canonical_json_bytes(provenance)),
            auth_token=runtime_auth_token,
        )
        backend_process = await backend_context.__aenter__()
        backend_context_entered = True
        summary["backend_launch_mode"] = "proof_runner_bound"
        summary["launched_backend_pid"] = int(backend_process.pid)
        recorder.record(
            "backend_process_started",
            {
                "pid": int(backend_process.pid),
                "launch_nonce": runtime_launch_nonce,
                "prestart_provenance_sha256": prestart_sha256,
            },
        )
        async with websockets.connect(args.url) as ws:
            await _authenticate_proof_websocket(
                ws,
                runtime_auth_token,
                timeout=args.timeout,
            )
            runtime_auth_token = ""
            runtime_identity = await send_and_wait(
                ws,
                recorder,
                {
                    "type": "get_runtime_identity",
                    "challenge_nonce": runtime_identity_challenge_nonce,
                    "expected_launch_nonce": runtime_launch_nonce,
                    "expected_prestart_provenance_sha256": prestart_sha256,
                    "expected_prestart_provenance_size": len(
                        canonical_json_bytes(provenance)
                    ),
                },
                EXPECTED_RUNTIME_IDENTITY_EVENTS,
                timeout=args.timeout,
            )
            summary["runtime_identity"] = runtime_identity
            if not _runtime_identity_matches_provenance(
                runtime_identity,
                provenance,
                request_challenge_nonce=runtime_identity_challenge_nonce,
                expected_backend_pid=int(backend_process.pid),
            ):
                summary.update(
                    {
                        "status": "blocked",
                        "reason": "backend_runtime_identity_mismatch",
                        "fail_closed": True,
                    }
                )
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
                return 16
            initial_connection = await _connect_wing_and_wait_connected(
                ws,
                recorder,
                ip=str(args.wing_ip or args.reconnect_wing_ip),
                send_port=args.reconnect_send_port,
                receive_port=args.reconnect_receive_port,
                observation_only=False,
                timeout=args.timeout,
            )
            initial_connection_passed = bool(
                initial_connection.get("connected") is True
                and initial_connection.get("mode") == "wing"
                and initial_connection.get("ip") == (args.wing_ip or args.reconnect_wing_ip)
                and initial_connection.get("observation_only") is False
                and initial_connection.get("record_telemetry") is False
            )
            summary["phases"]["initial_connection"] = {
                "status": "passed" if initial_connection_passed else "failed",
                "connection_status": initial_connection,
            }
            if not initial_connection_passed:
                summary.update(
                    {
                        "status": "blocked",
                        "reason": "initial_wing_connection_not_verified_for_write_probe",
                        "fail_closed": True,
                    }
                )
                recorder.write_summary(summary)
                print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
                return 18
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
            if not _readback_is_physical_for_target(
                baseline,
                channel=args.channel,
                value_type=args.value_type,
                address=address,
            ):
                summary.update(
                    {
                        "status": "blocked",
                        "reason": "baseline_not_verified_from_target_physical_readback",
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
                if args.operator_gated_physical_loss:
                    await _wait_for_operator(
                        "Physically disconnect the WING Ethernet cable, then press Enter: "
                    )
                else:
                    await asyncio.sleep(float(args.physical_loss_wait_sec))
                summary["phases"]["disconnect"] = {
                    "status": "operator_confirmed_physical_cable_loss",
                    "confirmation": args.operator_confirm_physical_cable_loss,
                    "wait_sec": args.physical_loss_wait_sec,
                    "operator_gate": bool(args.operator_gated_physical_loss),
                }

            stale_request_id = f"{run_id}:stale-probe"
            try:
                stale_probe = await _query_parameter(
                    ws,
                    recorder,
                    channel=args.channel,
                    value_type=args.value_type,
                    address=address,
                    timeout=args.timeout,
                    request_id=stale_request_id,
                )
            except TimeoutError as exc:
                stale_probe = _timeout_readback_unavailable(
                    channel=args.channel,
                    value_type=args.value_type,
                    address=address,
                    request_id=stale_request_id,
                    timeout=args.timeout,
                    error=exc,
                )
                recorder.record("synthetic_recv_timeout", stale_probe)
            summary["phases"]["stale_readback"] = {
                "status": (
                    "passed"
                    if _readback_is_unavailable_for_target(
                        stale_probe,
                        channel=args.channel,
                        value_type=args.value_type,
                        address=address,
                    )
                    else "failed"
                ),
                "readback": stale_probe,
            }
            if not _readback_is_unavailable_for_target(
                stale_probe,
                channel=args.channel,
                value_type=args.value_type,
                address=address,
            ):
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
                    nested_apply = apply_probe.get("apply_result")
                    if not isinstance(nested_apply, dict):
                        nested_apply = apply_probe
                    rollback = await _rollback_unexpected_apply(
                        ws,
                        recorder,
                        run_id=run_id,
                        approval_prefix=args.approval_prefix,
                        timeout=args.timeout,
                        transaction_id=str(nested_apply.get("transaction_id") or ""),
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
                if args.operator_gated_physical_loss:
                    await _wait_for_operator(
                        "Reconnect the WING Ethernet cable, then press Enter: "
                    )
                reconnect = await _connect_wing_and_wait_connected(
                    ws,
                    recorder,
                    ip=args.reconnect_wing_ip,
                    send_port=args.reconnect_send_port,
                    receive_port=args.reconnect_receive_port,
                    observation_only=True,
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
                    _readback_is_physical_for_target(
                        recovery_readback,
                        channel=args.channel,
                        value_type=args.value_type,
                        address=address,
                    )
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
                }
            )
            recorder.write_summary(summary)
            print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
            return 0
    except BoundBackendLaunchError as exc:
        summary.update(
            {
                "status": "blocked",
                "reason": "proof_bound_backend_launch_failed",
                "fail_closed": True,
                "backend_launch_error": str(exc),
                "live_apply_request_sent": False,
                "rollback_request_sent": False,
            }
        )
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return 17
    except (TimeoutError, OSError, ConnectionError) as exc:
        summarize_transport_failure(summary, recorder, exc)
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return 1
    except Exception as exc:
        summarize_transport_failure(summary, recorder, exc)
        recorder.write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return 1
    finally:
        if backend_context is not None and backend_context_entered:
            await backend_context.__aexit__(None, None, None)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
