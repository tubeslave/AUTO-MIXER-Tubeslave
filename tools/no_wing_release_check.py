#!/usr/bin/env python3
"""Run the no-WING Automixer release readiness checks.

The runner is intentionally local/offline. It executes tests, builds, and the
offline operator validation harness, then writes a small artifact bundle with
logs and a machine-readable summary. It must never connect to WING or send live
write commands.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "no_wing_release_check"
DEFAULT_BACKEND_PORT = 8765
DEFAULT_COMMAND_TIMEOUT_SEC = 900

FORBIDDEN_COMMAND_TOKENS = (
    "wing_operator_proposal_validation_runner.py",
    "wing_supervised_smoke_runner.py",
    "apply_operator_proposal",
    "set_fader",
    "set_channel_fader",
    "set_main_fader",
    "set_bus_fader",
    "set_dca_fader",
    "set_dca_level",
    "set_group_fader",
    "set_matrix_fader",
    "set_gain",
    "set_channel_gain",
    "set_trim",
    "set_input_trim",
    "set_channel_trim",
    "set_preamp_trim",
    "set_preamp_gain",
    "set_eq",
    "set_eq_band",
    "set_bus_eq_band",
    "set_bus_eq_on",
    "set_main_eq_band",
    "set_main_eq_on",
    "set_matrix_eq_band",
    "set_matrix_eq_on",
    "set_compressor",
    "set_compressor_gain",
    "set_bus_compressor",
    "set_bus_compressor_gain",
    "set_gate",
    "set_gate_on",
    "set_hpf",
    "set_bus_pan",
    "set_main_pan",
    "set_delay",
    "set_polarity",
    "set_mute",
    "set_channel_mute",
    "set_bus_mute",
    "set_dca_mute",
    "set_group_mute",
    "set_main_mute",
    "set_matrix_mute",
    "route_output",
    "route_multiple_outputs",
    "load_snap",
    "save_snap",
    "recall_snap",
    "store_snap",
    "restore_snapshot",
    "load_scene",
    "save_scene",
    "recall_scene",
    "restore_scene",
    "reset",
    "reset_channel_eq",
    "reset_channel_hpf",
    "reset_channel_dynamics",
    "reset_channel_processing",
    "reset_phase_delay",
    "mute_channel",
    "unmute_channel",
    "unmute",
    "mute",
    "set_channel_input",
    "set_channel_alt_input",
    "set_insert",
    "set_fx_model",
    "set_fx_on",
    "set_fx_mix",
    "set_fx_parameter",
    "set_fx_return_level",
    "set_send_level",
    "set_channel_send",
    "set_channel_main_send",
    "set_reverb_send",
    "connect_wing",
    "mixer_discovery",
    "WingClient",
)

FORBIDDEN_COMMAND_TOKEN_BOUNDARY = re.compile(r"[A-Za-z0-9]")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def compact_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class CommandCheck:
    name: str
    command: tuple[str, ...]
    cwd: str
    env: dict[str, str] | None = None
    timeout_sec: float | None = None

    def shell_display(self) -> str:
        return " ".join(self.command)


@dataclass
class CheckResult:
    name: str
    status: str
    started_at: str
    finished_at: str
    duration_sec: float
    command: str | None = None
    cwd: str | None = None
    returncode: int | None = None
    stdout_log: str | None = None
    stderr_log: str | None = None
    detail: dict[str, object] | None = None


def build_checks(
    *,
    repo_root: Path = REPO_ROOT,
    run_dir: Path | None = None,
    include_frontend: bool = True,
    include_package: bool = False,
    quick: bool = False,
    command_timeout_sec: float | None = DEFAULT_COMMAND_TIMEOUT_SEC,
) -> list[CommandCheck]:
    offline_artifact_root = (run_dir or DEFAULT_ARTIFACT_ROOT / "pending") / "offline_validation"
    proof_report_path = (run_dir or DEFAULT_ARTIFACT_ROOT / "pending") / "supervised_proof_artifact_report.json"
    checks = [
        CommandCheck(
            name="write_surface_static_scan",
            command=(
                sys.executable,
                "-m",
                "pytest",
                "tests/test_write_surface_inventory_static_scan.py",
                "-q",
                "--tb=short",
            ),
            cwd=str(repo_root),
            env={"PYTHONPATH": "backend"},
        ),
        CommandCheck(
            name="backend_pytest",
            command=(
                sys.executable,
                "-m",
                "pytest",
                "tests/test_offline_validation_harness.py" if quick else "tests/",
                "-q" if quick else "-x",
                "--tb=short",
                "-q",
            ),
            cwd=str(repo_root),
            env={"PYTHONPATH": "backend"},
        ),
        CommandCheck(
            name="offline_operator_validation_harness",
            command=(
                sys.executable,
                "tools/offline_operator_validation_harness.py",
                "--artifact-root",
                str(offline_artifact_root),
            ),
            cwd=str(repo_root),
        ),
        CommandCheck(
            name="supervised_proof_artifact_report",
            command=(
                sys.executable,
                "tools/evaluate_supervised_proof_artifacts.py",
                "--artifact-root",
                str(offline_artifact_root),
                "--output",
                str(proof_report_path),
                "--quiet",
            ),
            cwd=str(repo_root),
            env={"PYTHONPATH": "backend"},
        ),
        CommandCheck(
            name="memory_lint",
            command=(sys.executable, "tools/memory/memory_lint.py"),
            cwd=str(repo_root),
        ),
    ]

    if include_frontend:
        checks.extend(
            [
                CommandCheck(
                    name="frontend_typescript",
                    command=("npx", "tsc", "--noEmit"),
                    cwd=str(repo_root / "frontend"),
                ),
                CommandCheck(
                    name="frontend_unit",
                    command=("npm", "test", "--", "--watch=false"),
                    cwd=str(repo_root / "frontend"),
                ),
                CommandCheck(
                    name="frontend_e2e",
                    command=("npm", "run", "test:e2e"),
                    cwd=str(repo_root / "frontend"),
                ),
                CommandCheck(
                    name="frontend_build",
                    command=("npm", "run", "build"),
                    cwd=str(repo_root / "frontend"),
                ),
            ]
        )
        if include_package:
            checks.append(
                CommandCheck(
                    name="frontend_dist_package",
                    command=("npm", "run", "dist"),
                    cwd=str(repo_root / "frontend"),
                )
            )

    checks.append(
        CommandCheck(
            name="diff_check",
            command=("git", "diff", "--check"),
            cwd=str(repo_root),
        )
    )
    if command_timeout_sec is not None:
        checks = [replace(check, timeout_sec=command_timeout_sec) for check in checks]
    return checks


def validate_no_live_commands(checks: Iterable[CommandCheck]) -> None:
    violations: list[str] = []
    for check in checks:
        command_text = " ".join((check.name, check.shell_display())).lower()
        for token in FORBIDDEN_COMMAND_TOKENS:
            if _contains_forbidden_command_token(command_text, token):
                violations.append(f"{check.name}:{token}")
    if violations:
        joined = ", ".join(sorted(violations))
        raise ValueError(f"no-WING release checks include forbidden live command tokens: {joined}")


def _contains_forbidden_command_token(command_text: str, token: str) -> bool:
    normalized_token = token.lower()
    start = 0
    while True:
        index = command_text.find(normalized_token, start)
        if index == -1:
            return False

        before = command_text[index - 1] if index > 0 else ""
        after_index = index + len(normalized_token)
        after = command_text[after_index] if after_index < len(command_text) else ""
        before_is_word = bool(before and FORBIDDEN_COMMAND_TOKEN_BOUNDARY.match(before))
        after_is_word = bool(after and FORBIDDEN_COMMAND_TOKEN_BOUNDARY.match(after))

        if not before_is_word and not after_is_word:
            return True

        start = index + 1


def check_port_free(port: int = DEFAULT_BACKEND_PORT, host: str = "127.0.0.1") -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {"host": host, "port": port, "policy": "port_must_not_have_unexpected_listener"}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        connected = sock.connect_ex((host, port)) == 0
    finished = datetime.now(timezone.utc)
    detail["listener_detected"] = connected
    return CheckResult(
        name="backend_port_free",
        status="failed" if connected else "passed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _coerce_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def run_command_check(check: CommandCheck, log_dir: Path) -> CheckResult:
    started = datetime.now(timezone.utc)
    env = os.environ.copy()
    if check.env:
        env.update(check.env)
    detail: dict[str, object] | None = None
    try:
        completed = subprocess.run(
            check.command,
            cwd=check.cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=check.timeout_sec,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        status = "passed" if completed.returncode == 0 else "failed"
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = _coerce_timeout_output(exc.stdout)
        stderr = _coerce_timeout_output(exc.stderr)
        timeout_note = f"Command timed out after {check.timeout_sec} seconds."
        stderr = f"{stderr.rstrip()}\n{timeout_note}\n" if stderr else f"{timeout_note}\n"
        status = "timeout"
        returncode = None
        detail = {
            "timeout_sec": check.timeout_sec,
            "reason": "command_timeout",
        }
    finished = datetime.now(timezone.utc)

    stdout_log = log_dir / f"{check.name}.stdout.log"
    stderr_log = log_dir / f"{check.name}.stderr.log"
    _write_text(stdout_log, stdout)
    _write_text(stderr_log, stderr)

    return CheckResult(
        name=check.name,
        status=status,
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        command=check.shell_display(),
        cwd=check.cwd,
        returncode=returncode,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        detail=detail,
    )


def write_summary(run_dir: Path, results: Sequence[CheckResult], *, generated_at: str) -> dict[str, object]:
    failed = [result.name for result in results if result.status != "passed"]
    summary = {
        "type": "no_wing_release_check",
        "generated_at": generated_at,
        "status": "passed" if not failed else "failed",
        "connection_policy": "no_hardware_no_network_no_live_writes",
        "live_write": False,
        "failed_checks": failed,
        "results": [asdict(result) for result in results],
    }
    summary_path = run_dir / "summary.json"
    _write_text(summary_path, json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    lines = [
        "# No-WING Release Check",
        "",
        f"- generated_at: `{generated_at}`",
        f"- status: `{summary['status']}`",
        "- connection_policy: `no_hardware_no_network_no_live_writes`",
        "- live_write: `false`",
        "",
        "## Checks",
        "",
    ]
    for result in results:
        lines.append(f"- `{result.name}`: `{result.status}`")
    if failed:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{name}`" for name in failed)
    _write_text(run_dir / "summary.md", "\n".join(lines) + "\n")
    return summary


def run_release_checks(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    include_frontend: bool = True,
    include_package: bool = False,
    quick: bool = False,
    continue_on_failure: bool = False,
    backend_port: int = DEFAULT_BACKEND_PORT,
    command_timeout_sec: float | None = DEFAULT_COMMAND_TIMEOUT_SEC,
) -> dict[str, object]:
    generated_at = utc_now()
    run_dir = artifact_root / f"{compact_stamp()}_no_wing_release_check"
    log_dir = run_dir / "logs"

    checks = build_checks(
        run_dir=run_dir,
        include_frontend=include_frontend,
        include_package=include_package,
        quick=quick,
        command_timeout_sec=command_timeout_sec,
    )
    validate_no_live_commands(checks)

    results: list[CheckResult] = [check_port_free(backend_port)]
    if results[-1].status != "passed" and not continue_on_failure:
        summary = write_summary(run_dir, results, generated_at=generated_at)
        summary["run_dir"] = str(run_dir)
        return summary

    log_dir.mkdir(parents=True, exist_ok=True)
    for check in checks:
        result = run_command_check(check, log_dir)
        results.append(result)
        if result.status != "passed" and not continue_on_failure:
            break

    summary = write_summary(run_dir, results, generated_at=generated_at)
    summary["run_dir"] = str(run_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run no-WING Automixer readiness checks without connecting to mixer hardware."
    )
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help="Directory where release-check artifacts are written.",
    )
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="Run backend/offline checks only.",
    )
    parser.add_argument(
        "--include-package",
        action="store_true",
        help="Also run the Electron/macOS package build via npm run dist.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use focused backend tests instead of the full backend suite.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue running later checks after a failed check.",
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=DEFAULT_BACKEND_PORT,
        help="Local backend port that should be free before checks start.",
    )
    parser.add_argument(
        "--command-timeout-sec",
        type=float,
        default=DEFAULT_COMMAND_TIMEOUT_SEC,
        help="Timeout in seconds for each subprocess check. Use <= 0 to disable.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_release_checks(
        artifact_root=Path(args.artifact_root),
        include_frontend=not args.skip_frontend,
        include_package=args.include_package and not args.skip_frontend,
        quick=args.quick,
        continue_on_failure=args.continue_on_failure,
        backend_port=args.backend_port,
        command_timeout_sec=args.command_timeout_sec if args.command_timeout_sec > 0 else None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if summary.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
