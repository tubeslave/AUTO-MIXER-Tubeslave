#!/usr/bin/env python3
"""Run the no-WING Automixer release readiness checks.

The runner is intentionally local/offline. It executes tests, builds, and the
offline operator validation harness, then writes a small artifact bundle with
logs and a machine-readable summary. It must never connect to WING or send live
write commands.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.operator_mode_policy import MODE_SUPERVISED, build_operator_mode_status  # noqa: E402
from backend.read_surface_fixture_evidence import (  # noqa: E402
    FIXTURE_EVIDENCE_STATUSES,
    build_read_surface_fixture_evidence_snapshot,
)
from backend.read_surface_remediation_plan import build_read_surface_remediation_plan  # noqa: E402

DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "no_wing_release_check"
DEFAULT_BACKEND_PORT = 8765
DEFAULT_COMMAND_TIMEOUT_SEC = 900
AUTONOMY_GAP_ROLLUP_SOURCE = "autonomy_mode_transition_gate.rows[mode=auto]"
DEFAULT_HOMEBREW_GIT = Path("/opt/homebrew/bin/git")
GAIN_DRY_RUN_REASON = "no_wing_release_gain_dry_run"
REPO_LOCAL_COMMAND_OUTPUT_FLAGS = {"--artifact-root", "--output"}

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


def git_executable() -> str:
    configured = os.environ.get("AUTOMIXER_GIT_BIN")
    if configured:
        return configured
    if DEFAULT_HOMEBREW_GIT.exists():
        return str(DEFAULT_HOMEBREW_GIT)
    return "git"


def current_supervised_live_write_kinds() -> tuple[str, ...]:
    status = build_operator_mode_status(
        mode=MODE_SUPERVISED,
        connection_mode="wing",
        wing_boundary_active=True,
        auto_mode_unlocked=False,
    )
    capabilities = status.get("capabilities") if isinstance(status.get("capabilities"), Mapping) else {}
    return tuple(str(kind) for kind in capabilities.get("allowed_live_write_kinds", ()) if kind)


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
    gain_proof_root = gain_proof_artifact_root_for_run(run_dir)
    proof_report_path = proof_report_path_for_run(run_dir)
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
            name="read_surface_fixture_evidence",
            command=(
                sys.executable,
                "-m",
                "pytest",
                "tests/test_read_surface_fixture_evidence.py",
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
            name="gain_dry_run_proof",
            command=(
                sys.executable,
                "tools/gain_proof_dry_run_runner.py",
                "--artifact-root",
                str(gain_proof_root),
                "--channel",
                "1",
                "--initial-gain-db",
                "1.0",
                "--delta-db",
                "-0.25",
                "--cooldown-sec",
                "2.0",
                "--reason",
                GAIN_DRY_RUN_REASON,
            ),
            cwd=str(repo_root),
            env={"PYTHONPATH": "backend"},
        ),
        CommandCheck(
            name="supervised_proof_artifact_report",
            command=(
                sys.executable,
                "tools/evaluate_supervised_proof_artifacts.py",
                "--artifact-root",
                str(offline_artifact_root),
                "--artifact-root",
                str(gain_proof_root),
                "--require-physical",
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
            command=(git_executable(), "diff", "--check"),
            cwd=str(repo_root),
        )
    )
    if command_timeout_sec is not None:
        checks = [replace(check, timeout_sec=command_timeout_sec) for check in checks]
    return checks


def proof_report_path_for_run(run_dir: Path | None) -> Path:
    return (run_dir or DEFAULT_ARTIFACT_ROOT / "pending") / "supervised_proof_artifact_report.json"


def gain_proof_artifact_root_for_run(run_dir: Path | None) -> Path:
    return (run_dir or DEFAULT_ARTIFACT_ROOT / "pending") / "gain_proof_dry_run"


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


def validate_repo_local_commands(
    checks: Iterable[CommandCheck],
    *,
    repo_root: Path = REPO_ROOT,
) -> None:
    """Fail the no-WING plan if local command paths leave this checkout."""

    try:
        resolved_repo_root = repo_root.resolve()
    except OSError as exc:
        raise ValueError(f"no-WING release check repo root is not resolvable: {repo_root}") from exc

    violations: list[str] = []
    for check in checks:
        cwd_path = _resolve_repo_scoped_path(check.cwd, resolved_repo_root)
        if not _is_path_inside(cwd_path, resolved_repo_root):
            violations.append(f"{check.name}:cwd:{check.cwd}")

        skip_next_path_token = False
        for token in check.command[1:]:
            if skip_next_path_token:
                skip_next_path_token = False
                continue
            if token in REPO_LOCAL_COMMAND_OUTPUT_FLAGS:
                skip_next_path_token = True
                continue
            if not _looks_like_local_path_token(token):
                continue
            token_path = _resolve_repo_scoped_path(token, resolved_repo_root)
            if not _is_path_inside(token_path, resolved_repo_root):
                violations.append(f"{check.name}:command_path:{token}")

        for key, value in (check.env or {}).items():
            if not _looks_like_local_path_token(value):
                continue
            value_path = _resolve_repo_scoped_path(value, resolved_repo_root)
            if not _is_path_inside(value_path, resolved_repo_root):
                violations.append(f"{check.name}:env_path:{key}")

    if violations:
        joined = ", ".join(sorted(violations))
        raise ValueError(f"no-WING release checks leave the Automixer repo boundary: {joined}")


def _resolve_repo_scoped_path(value: str | os.PathLike[str], repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _is_path_inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _looks_like_local_path_token(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    if text in {".", ".."} or text.startswith(("~", "./", "../", "/")):
        return True
    return "/" in text or "\\" in text


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


def _json_object_from_path(path: Path) -> tuple[Mapping[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing_report"
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(payload, Mapping):
        return None, "json_payload_not_object"
    return payload, None


def _declared_pytest_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def validate_supervised_proof_report_gap_rollup(report_path: Path) -> CheckResult:
    started = datetime.now(timezone.utc)
    payload, load_error = _json_object_from_path(report_path)
    detail: dict[str, object] = {
        "report_path": str(report_path),
        "policy": "supervised_proof_report_must_expose_read_only_autonomy_gap_rollup",
    }
    violations: list[str] = []

    if load_error is not None:
        violations.append(load_error)
    else:
        assert payload is not None
        summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
        rollup = summary.get("autonomy_gap_rollup") if isinstance(summary.get("autonomy_gap_rollup"), Mapping) else {}
        mode_gate = (
            payload.get("autonomy_mode_transition_gate")
            if isinstance(payload.get("autonomy_mode_transition_gate"), Mapping)
            else {}
        )
        gate_summary = mode_gate.get("summary") if isinstance(mode_gate.get("summary"), Mapping) else {}

        expected_false = ("live_write", "live_widening_allowed", "full_autonomous_ready")
        for key in expected_false:
            if rollup.get(key) is not False:
                violations.append(f"autonomy_gap_rollup_{key}_not_false")
        if rollup.get("read_only") is not True:
            violations.append("autonomy_gap_rollup_read_only_not_true")
        if rollup.get("full_autonomous_blocked") is not True:
            violations.append("autonomy_gap_rollup_full_autonomous_blocked_not_true")
        if rollup.get("source") != AUTONOMY_GAP_ROLLUP_SOURCE:
            violations.append("autonomy_gap_rollup_source_not_auto_mode_gate_row")
        claim = rollup.get("claim")
        if not isinstance(claim, str) or "Full Autonomous remains blocked" not in claim:
            violations.append("autonomy_gap_rollup_claim_does_not_preserve_auto_block")
        if not isinstance(claim, str) or "does not widen" not in claim:
            violations.append("autonomy_gap_rollup_claim_does_not_preserve_live_boundary")
        next_required_evidence = rollup.get("next_required_evidence")
        if not isinstance(next_required_evidence, list) or not next_required_evidence:
            violations.append("autonomy_gap_rollup_next_required_evidence_missing")
        rollup_counts = rollup.get("full_auto_gap_bucket_counts")
        if not isinstance(rollup_counts, Mapping):
            violations.append("autonomy_gap_rollup_bucket_counts_missing")
            rollup_counts = {}
        if gate_summary.get("autonomous_apply_allowed_modes") != []:
            violations.append("mode_gate_autonomous_apply_modes_not_empty")
        blocked_modes = gate_summary.get("blocked_modes")
        if not isinstance(blocked_modes, list):
            blocked_modes = []
        if "auto" not in blocked_modes:
            violations.append("mode_gate_auto_not_blocked")
        supervised_write_kinds = rollup.get("supervised_current_live_write_kinds")
        if not isinstance(supervised_write_kinds, list):
            supervised_write_kinds = []
        unexpected_write_kinds = [
            str(kind)
            for kind in supervised_write_kinds
            if str(kind) not in current_supervised_live_write_kinds()
        ]
        if unexpected_write_kinds:
            violations.append("autonomy_gap_rollup_unexpected_supervised_live_write_kinds")

        detail["autonomy_gap_rollup_present"] = bool(rollup)
        detail["full_auto_gap_bucket_counts"] = dict(rollup_counts)
        detail["supervised_current_live_write_kinds"] = list(supervised_write_kinds)
        detail["policy_supervised_live_write_kinds"] = list(current_supervised_live_write_kinds())
        detail["unexpected_supervised_live_write_kinds"] = unexpected_write_kinds

    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="supervised_proof_report_gap_rollup_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_read_surface_fixture_evidence_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "read_surface_fixture_evidence_must_remain_read_only_and_non_autonomous",
    }
    violations: list[str] = []

    try:
        snapshot = build_read_surface_fixture_evidence_snapshot()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        snapshot = {}
        violations.append(f"snapshot_build_error:{type(exc).__name__}")

    safety = snapshot.get("safety") if isinstance(snapshot.get("safety"), Mapping) else {}
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), Mapping) else {}
    status_counts = summary.get("status_counts") if isinstance(summary.get("status_counts"), Mapping) else {}
    surfaces = snapshot.get("surfaces") if isinstance(snapshot.get("surfaces"), list) else []
    if not isinstance(snapshot.get("surfaces"), list):
        violations.append("read_surface_fixture_surfaces_missing")

    def summary_int(key: str) -> int:
        try:
            return int(summary.get(key) or 0)
        except (TypeError, ValueError):
            violations.append(f"read_surface_fixture_{key}_not_integer")
            return -1

    expected_status_counts = {status: 0 for status in sorted(FIXTURE_EVIDENCE_STATUSES)}
    expected_physical_missing: list[str] = []
    expected_surface_with_fixture_count = 0
    declared_tests_by_path: dict[str, set[str]] = {}
    declared_fixture_path_count = 0
    declared_test_name_count = 0
    for index, row in enumerate(surfaces):
        if not isinstance(row, Mapping):
            violations.append(f"read_surface_fixture_row_{index}_not_object")
            continue
        surface_id = str(row.get("surface_id") or f"index_{index}")
        if not isinstance(row.get("surface_id"), str) or not row.get("surface_id"):
            violations.append(f"read_surface_fixture_row_{surface_id}_surface_id_missing")
        if not isinstance(row.get("target_console_surface"), str) or not row.get("target_console_surface"):
            violations.append(f"read_surface_fixture_row_{surface_id}_target_console_surface_missing")
        status = str(row.get("status") or "")
        if not status:
            violations.append(f"read_surface_fixture_row_{surface_id}_status_missing")
        else:
            if status not in FIXTURE_EVIDENCE_STATUSES:
                violations.append(f"read_surface_fixture_row_{surface_id}_status_unknown")
            expected_status_counts[status] = expected_status_counts.get(status, 0) + 1
            if status != "missing_fixture_evidence":
                expected_surface_with_fixture_count += 1
        if row.get("full_autonomous_ready") is not False:
            violations.append(f"read_surface_fixture_row_{surface_id}_full_autonomous_ready_not_false")
        remaining_gaps = row.get("remaining_gaps")
        if not isinstance(remaining_gaps, list):
            remaining_gaps = []
            violations.append(f"read_surface_fixture_row_{surface_id}_remaining_gaps_missing")
        if "physical_readback_fixture_missing" in remaining_gaps:
            expected_physical_missing.append(surface_id)
        covered_contracts_value = row.get("covered_contracts")
        if not isinstance(covered_contracts_value, list):
            covered_contracts: list[object] = []
            violations.append(f"read_surface_fixture_row_{surface_id}_covered_contracts_not_list")
        else:
            covered_contracts = list(covered_contracts_value)
        invalid_contracts = [
            value
            for value in covered_contracts
            if not isinstance(value, str) or not value.strip()
        ]
        if invalid_contracts:
            violations.append(f"read_surface_fixture_row_{surface_id}_covered_contracts_invalid")
        invalid_gaps = [
            value
            for value in remaining_gaps
            if not isinstance(value, str) or not value.strip()
        ]
        if invalid_gaps:
            violations.append(f"read_surface_fixture_row_{surface_id}_remaining_gaps_invalid")
        fixture_paths_value = row.get("fixture_paths")
        if not isinstance(fixture_paths_value, list):
            fixture_paths: list[object] = []
            violations.append(f"read_surface_fixture_row_{surface_id}_fixture_paths_not_list")
        else:
            fixture_paths = list(fixture_paths_value)
        test_names_value = row.get("test_names")
        if not isinstance(test_names_value, list):
            test_names: list[object] = []
            violations.append(f"read_surface_fixture_row_{surface_id}_test_names_not_list")
        else:
            test_names = list(test_names_value)
        declared_fixture_path_count += len(fixture_paths)
        declared_test_name_count += len(test_names)
        if status == "missing_fixture_evidence":
            if covered_contracts:
                violations.append(f"read_surface_fixture_row_{surface_id}_missing_status_has_covered_contracts")
            if fixture_paths:
                violations.append(f"read_surface_fixture_row_{surface_id}_missing_status_has_fixture_paths")
            if test_names:
                violations.append(f"read_surface_fixture_row_{surface_id}_missing_status_has_test_names")
            continue
        if not covered_contracts:
            violations.append(f"read_surface_fixture_row_{surface_id}_covered_contracts_missing")
        has_physical_contract = "physical_readback_fixture" in covered_contracts
        has_physical_gap = "physical_readback_fixture_missing" in remaining_gaps
        has_hardware_gap = "hardware_parity_unproven" in remaining_gaps
        if status == "partial_fixture_evidence":
            if has_physical_contract:
                violations.append(f"read_surface_fixture_row_{surface_id}_partial_status_has_physical_contract")
            if not has_physical_gap:
                violations.append(f"read_surface_fixture_row_{surface_id}_partial_status_missing_physical_gap")
            if not has_hardware_gap:
                violations.append(f"read_surface_fixture_row_{surface_id}_partial_status_missing_hardware_gap")
        if status == "fixture_evidence_present":
            if not has_physical_contract:
                violations.append(f"read_surface_fixture_row_{surface_id}_present_status_missing_physical_contract")
            if has_physical_gap:
                violations.append(f"read_surface_fixture_row_{surface_id}_present_status_has_physical_gap")
            if has_hardware_gap:
                violations.append(f"read_surface_fixture_row_{surface_id}_present_status_has_hardware_gap")
        if not fixture_paths:
            violations.append(f"read_surface_fixture_row_{surface_id}_fixture_paths_missing")
        if not test_names:
            violations.append(f"read_surface_fixture_row_{surface_id}_test_names_missing")
        available_tests: set[str] = set()
        for value in fixture_paths:
            if not isinstance(value, str) or not value.strip():
                violations.append(f"read_surface_fixture_row_{surface_id}_fixture_path_invalid")
                continue
            relative_path = value.strip()
            path_parts = Path(relative_path).parts
            if Path(relative_path).is_absolute() or ".." in path_parts or not path_parts or path_parts[0] != "tests":
                violations.append(f"read_surface_fixture_row_{surface_id}_fixture_path_not_tests")
                continue
            fixture_path = REPO_ROOT / relative_path
            try:
                resolved_fixture_path = fixture_path.resolve()
                resolved_repo_root = REPO_ROOT.resolve()
                if resolved_fixture_path != resolved_repo_root and resolved_repo_root not in resolved_fixture_path.parents:
                    violations.append(f"read_surface_fixture_row_{surface_id}_fixture_path_escapes_repo")
                    continue
            except OSError:
                violations.append(f"read_surface_fixture_row_{surface_id}_fixture_path_unresolvable")
                continue
            if not fixture_path.exists():
                violations.append(f"read_surface_fixture_row_{surface_id}_fixture_path_missing")
                continue
            if not fixture_path.is_file():
                violations.append(f"read_surface_fixture_row_{surface_id}_fixture_path_not_file")
                continue
            if relative_path not in declared_tests_by_path:
                try:
                    declared_tests_by_path[relative_path] = _declared_pytest_functions(fixture_path)
                except (OSError, SyntaxError, UnicodeDecodeError):
                    declared_tests_by_path[relative_path] = set()
                    violations.append(f"read_surface_fixture_row_{surface_id}_fixture_path_unreadable")
            available_tests.update(declared_tests_by_path[relative_path])
        for value in test_names:
            if not isinstance(value, str) or not value.startswith("test_"):
                violations.append(f"read_surface_fixture_row_{surface_id}_test_name_invalid")
                continue
            if value not in available_tests:
                violations.append(f"read_surface_fixture_row_{surface_id}_test_name_missing")

    normalized_status_counts = {status: 0 for status in sorted(FIXTURE_EVIDENCE_STATUSES)}
    for key, value in status_counts.items():
        status_key = str(key)
        try:
            normalized_status_counts[status_key] = int(value)
        except (TypeError, ValueError):
            normalized_status_counts[status_key] = -1
            violations.append(f"read_surface_fixture_status_count_{status_key}_not_integer")
    physical_missing = summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_missing, list):
        physical_missing = []
        violations.append("physical_fixture_missing_surface_ids_missing")
    expected_next_physical_readback_surface_id = (
        expected_physical_missing[0] if expected_physical_missing else None
    )
    next_physical_readback_surface_id = summary.get("next_physical_readback_surface_id")
    if next_physical_readback_surface_id != expected_next_physical_readback_surface_id:
        violations.append("read_surface_fixture_next_physical_readback_surface_id_mismatch")

    expected_true = ("read_only",)
    for key in expected_true:
        if safety.get(key) is not True:
            violations.append(f"read_surface_fixture_{key}_not_true")
    expected_false = (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "full_autonomous_ready",
    )
    for key in expected_false:
        if safety.get(key) is not False:
            violations.append(f"read_surface_fixture_{key}_not_false")
    if snapshot.get("type") != "read_surface_fixture_evidence":
        violations.append("read_surface_fixture_type_mismatch")
    if snapshot.get("status") == "fixture_evidence_complete" and physical_missing:
        violations.append("fixture_status_complete_but_physical_gaps_remain")
    if not physical_missing:
        violations.append("physical_readback_gaps_not_reported")
    summary_surface_count = summary_int("surface_count")
    summary_surface_with_fixture_count = summary_int("surface_with_fixture_count")
    if summary_surface_count != len(surfaces):
        violations.append("read_surface_fixture_surface_count_mismatch")
    if summary_surface_with_fixture_count != expected_surface_with_fixture_count:
        violations.append("read_surface_fixture_surface_with_fixture_count_mismatch")
    if normalized_status_counts != expected_status_counts:
        violations.append("read_surface_fixture_status_counts_mismatch")
    if list(physical_missing) != expected_physical_missing:
        violations.append("read_surface_fixture_physical_missing_surface_ids_mismatch")

    detail["snapshot_status"] = snapshot.get("status")
    detail["surface_count"] = summary_surface_count
    detail["surface_with_fixture_count"] = summary_surface_with_fixture_count
    detail["status_counts"] = dict(status_counts)
    detail["physical_fixture_missing_surface_ids"] = list(physical_missing)
    detail["next_physical_readback_surface_id"] = next_physical_readback_surface_id
    detail["declared_fixture_path_count"] = declared_fixture_path_count
    detail["declared_test_name_count"] = declared_test_name_count
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="read_surface_fixture_evidence_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_read_surface_remediation_plan_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "read_surface_remediation_plan_must_track_fixture_next_target_without_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        plan = build_read_surface_remediation_plan(fixture_evidence_snapshot=fixture_snapshot)
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        plan = {}
        violations.append(f"read_surface_remediation_plan_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    plan_summary = plan.get("summary") if isinstance(plan.get("summary"), Mapping) else {}
    safety = plan.get("safety") if isinstance(plan.get("safety"), Mapping) else {}
    actions = plan.get("actions") if isinstance(plan.get("actions"), list) else []
    if not isinstance(plan.get("actions"), list):
        violations.append("read_surface_remediation_actions_missing")

    expected_physical_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(expected_physical_ids, list):
        expected_physical_ids = []
        violations.append("read_surface_remediation_fixture_physical_ids_missing")
    expected_physical_ids = [item for item in expected_physical_ids if isinstance(item, str) and item]
    expected_next_target = fixture_summary.get("next_physical_readback_surface_id")
    if not isinstance(expected_next_target, str):
        expected_next_target = expected_physical_ids[0] if expected_physical_ids else None
    expected_next_action_id = f"remediate_{expected_next_target}" if expected_next_target else None

    physical_ids = plan_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_ids, list):
        physical_ids = []
        violations.append("read_surface_remediation_physical_ids_missing")
    physical_ids = [item for item in physical_ids if isinstance(item, str) and item]
    if physical_ids != expected_physical_ids:
        violations.append("read_surface_remediation_physical_ids_mismatch")

    next_target = plan_summary.get("next_physical_readback_surface_id")
    if next_target != expected_next_target:
        violations.append("read_surface_remediation_next_target_mismatch")
    next_action_id = plan_summary.get("next_physical_readback_action_id")
    if next_action_id != expected_next_action_id:
        violations.append("read_surface_remediation_next_action_id_mismatch")

    action_by_id = {
        row.get("id"): row
        for row in actions
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }
    next_action = action_by_id.get(next_action_id) if isinstance(next_action_id, str) else None
    target_action_ids = [
        row.get("id")
        for row in actions
        if isinstance(row, Mapping)
        and isinstance(row.get("id"), str)
        and row.get("next_physical_readback_target") is True
    ]
    expected_target_action_ids = [expected_next_action_id] if expected_next_action_id else []
    if target_action_ids != expected_target_action_ids:
        violations.append("read_surface_remediation_next_target_action_flags_mismatch")
    if expected_next_action_id and not isinstance(next_action, Mapping):
        violations.append("read_surface_remediation_next_action_missing")
    elif isinstance(next_action, Mapping):
        if next_action.get("surface_id") != expected_next_target:
            violations.append("read_surface_remediation_next_action_surface_mismatch")
        if next_action.get("physical_readback_gap") is not True:
            violations.append("read_surface_remediation_next_action_not_physical_gap")
        if next_action.get("next_physical_readback_target") is not True:
            violations.append("read_surface_remediation_next_action_target_flag_missing")
        missing_contracts = next_action.get("missing_contracts")
        if not isinstance(missing_contracts, list) or "physical_readback_fixture" not in missing_contracts:
            violations.append("read_surface_remediation_next_action_missing_physical_contract")

    expected_true = ("read_only",)
    for key in expected_true:
        if safety.get(key) is not True:
            violations.append(f"read_surface_remediation_{key}_not_true")
    expected_false = (
        "live_write",
        "mutates_console_state",
        "live_widening_allowed",
        "full_autonomous_ready",
    )
    for key in expected_false:
        if safety.get(key) is not False:
            violations.append(f"read_surface_remediation_{key}_not_false")
    if plan.get("type") != "read_surface_remediation_plan":
        violations.append("read_surface_remediation_type_mismatch")
    if plan.get("status") == "read_coverage_complete" and physical_ids:
        violations.append("read_surface_remediation_complete_but_physical_gaps_remain")
    if expected_physical_ids and not actions:
        violations.append("read_surface_remediation_actions_empty_with_physical_gaps")

    detail["plan_status"] = plan.get("status")
    detail["action_count"] = int(plan_summary.get("action_count") or len(actions))
    detail["incomplete_surface_count"] = int(plan_summary.get("incomplete_surface_count") or 0)
    detail["physical_fixture_missing_surface_ids"] = list(physical_ids)
    detail["next_physical_readback_surface_id"] = next_target
    detail["next_physical_readback_action_id"] = next_action_id
    detail["next_physical_readback_target_action_ids"] = list(target_action_ids)
    detail["expected_next_physical_readback_surface_id"] = expected_next_target
    detail["expected_next_physical_readback_action_id"] = expected_next_action_id
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="read_surface_remediation_plan_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_supervised_proof_report_gain_dry_run(report_path: Path) -> CheckResult:
    started = datetime.now(timezone.utc)
    payload, load_error = _json_object_from_path(report_path)
    detail: dict[str, object] = {
        "report_path": str(report_path),
        "policy": "supervised_proof_report_must_include_read_only_gain_dry_run_evidence",
    }
    violations: list[str] = []
    gain_rows: list[Mapping[str, Any]] = []

    if load_error is not None:
        violations.append(load_error)
    else:
        assert payload is not None
        safety = payload.get("safety") if isinstance(payload.get("safety"), Mapping) else {}
        summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
        rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []

        if safety.get("read_only") is not True:
            violations.append("proof_report_read_only_not_true")
        for key in ("live_write", "live_widening_allowed", "full_autonomous_ready"):
            if safety.get(key) is not False:
                violations.append(f"proof_report_{key}_not_false")

        gain_rows = [
            row
            for row in rows
            if isinstance(row, Mapping)
            and row.get("value_type") == "gain"
            and row.get("proof_mode") == "dry_run"
            and row.get("status") == "dry_run_ready"
        ]
        if not gain_rows:
            violations.append("gain_dry_run_ready_row_missing")

        value_type_counts = summary.get("value_type_counts")
        if not isinstance(value_type_counts, Mapping) or int(value_type_counts.get("gain") or 0) < 1:
            violations.append("summary_gain_value_type_count_missing")
        dry_run_ids = summary.get("dry_run_ready_proof_ids")
        if not isinstance(dry_run_ids, list) or not dry_run_ids:
            violations.append("summary_dry_run_ready_proof_ids_missing")

        for row in gain_rows:
            if row.get("promotion_allowed") is not False:
                violations.append("gain_dry_run_promotion_allowed_not_false")
            if row.get("autonomous_ready") is not False:
                violations.append("gain_dry_run_autonomous_ready_not_false")
            blockers = row.get("blockers") if isinstance(row.get("blockers"), list) else []
            if "physical_proof_required_for_promotion" not in blockers:
                violations.append("gain_dry_run_physical_proof_blocker_missing")
            evidence = row.get("evidence") if isinstance(row.get("evidence"), Mapping) else {}
            expected_sources = {
                "pre_read_source": "fake_mixer",
                "readback_source": "fake_mixer",
                "rollback_readback_source": "fake_mixer",
            }
            for key, expected in expected_sources.items():
                if evidence.get(key) != expected:
                    violations.append(f"gain_dry_run_{key}_not_fake_mixer")
            if evidence.get("operator_approval_recorded") is not True:
                violations.append("gain_dry_run_operator_approval_missing")
            if evidence.get("idempotency_key_present") is not True:
                violations.append("gain_dry_run_idempotency_key_missing")

        detail["gain_dry_run_row_count"] = len(gain_rows)
        detail["dry_run_ready_proof_ids"] = list(dry_run_ids) if isinstance(dry_run_ids, list) else []
        detail["value_type_counts"] = dict(value_type_counts) if isinstance(value_type_counts, Mapping) else {}

    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="supervised_proof_report_gain_dry_run_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_supervised_proof_report_next_wing_plan(report_path: Path) -> CheckResult:
    started = datetime.now(timezone.utc)
    payload, load_error = _json_object_from_path(report_path)
    detail: dict[str, object] = {
        "report_path": str(report_path),
        "policy": "next_wing_proof_plan_must_be_operator_required_and_non_executing",
    }
    violations: list[str] = []
    planned_value_types: list[str] = []
    planned_channels: list[int] = []
    global_blockers: list[str] = []
    step_count = 0
    plan_status: object = None

    if load_error is not None:
        violations.append(load_error)
    else:
        assert payload is not None
        plan = payload.get("next_wing_proof_plan") if isinstance(payload.get("next_wing_proof_plan"), Mapping) else {}
        safety = plan.get("safety") if isinstance(plan.get("safety"), Mapping) else {}
        summary = plan.get("summary") if isinstance(plan.get("summary"), Mapping) else {}
        steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
        plan_status = plan.get("status")

        if plan.get("type") != "supervised_wing_proof_plan":
            violations.append("next_wing_proof_plan_type_mismatch")
        if plan_status not in {"operator_proof_required", "manual_review_ready", "blocked_by_invalid_artifacts"}:
            violations.append("next_wing_proof_plan_status_unknown")

        expected_true = ("read_only", "operator_required")
        for key in expected_true:
            if safety.get(key) is not True:
                violations.append(f"next_wing_proof_plan_{key}_not_true")
        expected_false = (
            "live_write",
            "execution_allowed_by_plan",
            "live_widening_allowed",
            "automatic_safety_update_allowed",
            "full_autonomous_ready",
        )
        for key in expected_false:
            if safety.get(key) is not False:
                violations.append(f"next_wing_proof_plan_{key}_not_false")

        step_count = int(summary.get("step_count") or 0)
        if step_count != len(steps):
            violations.append("next_wing_proof_plan_step_count_mismatch")
        planned_value_types = [str(item) for item in summary.get("planned_value_types", []) if item] if isinstance(summary.get("planned_value_types"), list) else []
        planned_channels = [
            int(item)
            for item in summary.get("planned_channels", [])
            if isinstance(item, (int, float, str)) and str(item).strip().lstrip("-").isdigit()
        ] if isinstance(summary.get("planned_channels"), list) else []
        global_blockers = [str(item) for item in summary.get("global_blockers", []) if item] if isinstance(summary.get("global_blockers"), list) else []
        if plan_status == "operator_proof_required" and step_count < 1:
            violations.append("next_wing_proof_plan_required_without_steps")
        if plan_status == "operator_proof_required" and "operator_must_approve_and_execute_manually" not in global_blockers:
            violations.append("next_wing_proof_plan_operator_blocker_missing")

        for index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                violations.append(f"next_wing_proof_plan_step_{index}_not_object")
                continue
            if step.get("execution_allowed_by_plan") is not False:
                violations.append(f"next_wing_proof_plan_step_{index}_execution_allowed")
            if step.get("operator_approval_required") is not True:
                violations.append(f"next_wing_proof_plan_step_{index}_operator_approval_not_required")
            if step.get("live_execution_required") is not True:
                violations.append(f"next_wing_proof_plan_step_{index}_live_execution_not_declared")
            try:
                delta_db = float(step.get("delta_db"))
            except (TypeError, ValueError):
                violations.append(f"next_wing_proof_plan_step_{index}_delta_invalid")
                continue
            if delta_db >= 0 or abs(delta_db) > 0.5:
                violations.append(f"next_wing_proof_plan_step_{index}_delta_not_safe_negative_proof")

    detail["plan_status"] = plan_status
    detail["step_count"] = step_count
    detail["planned_value_types"] = planned_value_types
    detail["planned_channels"] = planned_channels
    detail["global_blockers"] = global_blockers
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="next_wing_proof_plan_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_supervised_proof_report_autonomy_overlay(report_path: Path) -> CheckResult:
    started = datetime.now(timezone.utc)
    payload, load_error = _json_object_from_path(report_path)
    detail: dict[str, object] = {
        "report_path": str(report_path),
        "policy": "autonomy_evidence_overlay_must_remain_read_only_and_non_promoting",
    }
    violations: list[str] = []
    overlay_status: object = None
    row_count = 0
    status_counts: Mapping[str, Any] = {}
    global_blockers: list[str] = []
    manual_review_ids: list[str] = []
    remediation_ids: list[str] = []
    dry_run_ids: list[str] = []
    no_evidence_ids: list[str] = []

    if load_error is not None:
        violations.append(load_error)
    else:
        assert payload is not None
        overlay = (
            payload.get("autonomy_evidence_overlay")
            if isinstance(payload.get("autonomy_evidence_overlay"), Mapping)
            else {}
        )
        safety = overlay.get("safety") if isinstance(overlay.get("safety"), Mapping) else {}
        summary = overlay.get("summary") if isinstance(overlay.get("summary"), Mapping) else {}
        rows = overlay.get("rows") if isinstance(overlay.get("rows"), list) else []
        overlay_status = overlay.get("status")

        if overlay.get("type") != "autonomy_evidence_overlay":
            violations.append("autonomy_overlay_type_mismatch")
        if overlay_status not in {
            "manual_remediation_required",
            "manual_review_ready",
            "dry_run_only",
            "no_current_evidence",
        }:
            violations.append("autonomy_overlay_status_unknown")

        expected_true = ("read_only", "operator_required")
        for key in expected_true:
            if safety.get(key) is not True:
                violations.append(f"autonomy_overlay_{key}_not_true")
        expected_false = (
            "live_write",
            "live_widening_allowed",
            "automatic_safety_update_allowed",
            "full_autonomous_ready",
        )
        for key in expected_false:
            if safety.get(key) is not False:
                violations.append(f"autonomy_overlay_{key}_not_false")

        status_counts = summary.get("status_counts") if isinstance(summary.get("status_counts"), Mapping) else {}
        global_blockers = [str(item) for item in summary.get("global_blockers", []) if item] if isinstance(summary.get("global_blockers"), list) else []
        manual_review_ids = [str(item) for item in summary.get("manual_safety_review_candidate_target_ids", []) if item] if isinstance(summary.get("manual_safety_review_candidate_target_ids"), list) else []
        remediation_ids = [str(item) for item in summary.get("remediation_required_target_ids", []) if item] if isinstance(summary.get("remediation_required_target_ids"), list) else []
        dry_run_ids = [str(item) for item in summary.get("dry_run_only_target_ids", []) if item] if isinstance(summary.get("dry_run_only_target_ids"), list) else []
        no_evidence_ids = [str(item) for item in summary.get("no_current_evidence_target_ids", []) if item] if isinstance(summary.get("no_current_evidence_target_ids"), list) else []

        if summary.get("autonomous_ready_target_ids") != []:
            violations.append("autonomy_overlay_autonomous_ready_targets_not_empty")
        for blocker in (
            "overlay_does_not_modify_safety_rows",
            "manual_safety_review_required_before_promotion",
            "full_autonomous_mode_remains_blocked",
        ):
            if blocker not in global_blockers:
                violations.append(f"autonomy_overlay_global_blocker_missing:{blocker}")

        row_count = len(rows)
        row_value_types: set[str] = set()
        expected_row_statuses = {
            "manual_safety_review_candidate",
            "remediation_required",
            "dry_run_only",
            "no_current_evidence",
        }
        actual_status_counts = {status: 0 for status in expected_row_statuses}
        actual_target_ids_by_status = {status: [] for status in expected_row_statuses}
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                violations.append(f"autonomy_overlay_row_{index}_not_object")
                continue
            value_type = str(row.get("value_type") or "")
            row_status = str(row.get("status") or "")
            target_id = str(row.get("target_id") or "")
            row_value_types.add(value_type)
            if value_type not in {"fader", "gain"}:
                violations.append(f"autonomy_overlay_row_{index}_unexpected_value_type")
            if row.get("autonomous_ready") is not False:
                violations.append(f"autonomy_overlay_row_{index}_autonomous_ready_not_false")
            if row_status not in expected_row_statuses:
                violations.append(f"autonomy_overlay_row_{index}_status_unknown")
            else:
                actual_status_counts[row_status] += 1
                if target_id:
                    actual_target_ids_by_status[row_status].append(target_id)
            if row_status == "manual_safety_review_candidate":
                blockers = row.get("blockers") if isinstance(row.get("blockers"), list) else []
                if "automatic_promotion_forbidden" not in blockers:
                    violations.append(f"autonomy_overlay_row_{index}_automatic_promotion_blocker_missing")
        if summary.get("target_count") != row_count:
            violations.append("autonomy_overlay_target_count_mismatch")
        normalized_status_counts: dict[str, int] = {}
        for status in expected_row_statuses:
            try:
                normalized_status_counts[status] = int(status_counts.get(status, -1))
            except (TypeError, ValueError):
                normalized_status_counts[status] = -1
        if normalized_status_counts != actual_status_counts:
            violations.append("autonomy_overlay_status_counts_mismatch")
        expected_summary_buckets = {
            "manual_safety_review_candidate": manual_review_ids,
            "remediation_required": remediation_ids,
            "dry_run_only": dry_run_ids,
            "no_current_evidence": no_evidence_ids,
        }
        for status, summary_ids in expected_summary_buckets.items():
            if summary_ids != actual_target_ids_by_status[status]:
                violations.append(f"autonomy_overlay_{status}_target_ids_mismatch")
        if row_value_types != {"fader", "gain"}:
            violations.append("autonomy_overlay_fader_gain_rows_missing")
        if not dry_run_ids:
            violations.append("autonomy_overlay_dry_run_target_ids_missing")

    detail["overlay_status"] = overlay_status
    detail["row_count"] = row_count
    detail["status_counts"] = dict(status_counts)
    detail["manual_safety_review_candidate_target_ids"] = manual_review_ids
    detail["remediation_required_target_ids"] = remediation_ids
    detail["dry_run_only_target_ids"] = dry_run_ids
    detail["no_current_evidence_target_ids"] = no_evidence_ids
    detail["global_blockers"] = global_blockers
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="autonomy_evidence_overlay_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def autonomy_gap_rollup_summary(results: Sequence[CheckResult]) -> dict[str, object] | None:
    for result in results:
        if result.name != "supervised_proof_report_gap_rollup_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        counts = detail.get("full_auto_gap_bucket_counts")
        supervised_kinds = detail.get("supervised_current_live_write_kinds")
        policy_kinds = detail.get("policy_supervised_live_write_kinds")
        unexpected_kinds = detail.get("unexpected_supervised_live_write_kinds")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "full_autonomous_ready": False,
            "autonomy_gap_rollup_present": bool(detail.get("autonomy_gap_rollup_present")),
            "full_auto_gap_bucket_counts": dict(counts) if isinstance(counts, Mapping) else {},
            "supervised_current_live_write_kinds": list(supervised_kinds) if isinstance(supervised_kinds, list) else [],
            "policy_supervised_live_write_kinds": list(policy_kinds) if isinstance(policy_kinds, list) else [],
            "unexpected_supervised_live_write_kinds": list(unexpected_kinds)
            if isinstance(unexpected_kinds, list)
            else [],
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def autonomy_evidence_overlay_summary(results: Sequence[CheckResult]) -> dict[str, object] | None:
    for result in results:
        if result.name != "autonomy_evidence_overlay_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        status_counts = detail.get("status_counts")
        manual_review_ids = detail.get("manual_safety_review_candidate_target_ids")
        remediation_ids = detail.get("remediation_required_target_ids")
        dry_run_ids = detail.get("dry_run_only_target_ids")
        no_evidence_ids = detail.get("no_current_evidence_target_ids")
        global_blockers = detail.get("global_blockers")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "live_widening_allowed": False,
            "automatic_safety_update_allowed": False,
            "full_autonomous_ready": False,
            "operator_required": True,
            "overlay_status": detail.get("overlay_status"),
            "row_count": int(detail.get("row_count") or 0),
            "status_counts": dict(status_counts) if isinstance(status_counts, Mapping) else {},
            "manual_safety_review_candidate_target_ids": list(manual_review_ids)
            if isinstance(manual_review_ids, list)
            else [],
            "remediation_required_target_ids": list(remediation_ids)
            if isinstance(remediation_ids, list)
            else [],
            "dry_run_only_target_ids": list(dry_run_ids) if isinstance(dry_run_ids, list) else [],
            "no_current_evidence_target_ids": list(no_evidence_ids)
            if isinstance(no_evidence_ids, list)
            else [],
            "global_blockers": list(global_blockers) if isinstance(global_blockers, list) else [],
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def supervised_gain_dry_run_summary(results: Sequence[CheckResult]) -> dict[str, object] | None:
    for result in results:
        if result.name != "supervised_proof_report_gain_dry_run_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        dry_run_ids = detail.get("dry_run_ready_proof_ids")
        value_type_counts = detail.get("value_type_counts")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "promotion_allowed": False,
            "autonomous_ready": False,
            "gain_dry_run_row_count": int(detail.get("gain_dry_run_row_count") or 0),
            "dry_run_ready_proof_ids": list(dry_run_ids) if isinstance(dry_run_ids, list) else [],
            "value_type_counts": dict(value_type_counts) if isinstance(value_type_counts, Mapping) else {},
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def read_surface_fixture_evidence_summary(results: Sequence[CheckResult]) -> dict[str, object] | None:
    for result in results:
        if result.name != "read_surface_fixture_evidence_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        status_counts = detail.get("status_counts")
        physical_missing = detail.get("physical_fixture_missing_surface_ids")
        violations = detail.get("violations")
        physical_missing_ids = list(physical_missing) if isinstance(physical_missing, list) else []
        next_physical_readback_surface_id = detail.get("next_physical_readback_surface_id")
        if not isinstance(next_physical_readback_surface_id, str):
            next_physical_readback_surface_id = (
                physical_missing_ids[0] if physical_missing_ids else None
            )
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "full_autonomous_ready": False,
            "snapshot_status": detail.get("snapshot_status"),
            "surface_count": int(detail.get("surface_count") or 0),
            "surface_with_fixture_count": int(detail.get("surface_with_fixture_count") or 0),
            "declared_fixture_path_count": int(detail.get("declared_fixture_path_count") or 0),
            "declared_test_name_count": int(detail.get("declared_test_name_count") or 0),
            "physical_gap_surface_count": len(physical_missing_ids),
            "summary_recomputed_from_rows": True,
            "fixture_declaration_validation": True,
            "contract_gap_consistency_enforced": True,
            "status_counts": dict(status_counts) if isinstance(status_counts, Mapping) else {},
            "physical_fixture_missing_surface_ids": physical_missing_ids,
            "next_physical_readback_surface_id": next_physical_readback_surface_id,
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def read_surface_remediation_plan_summary(results: Sequence[CheckResult]) -> dict[str, object] | None:
    for result in results:
        if result.name != "read_surface_remediation_plan_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        physical_missing = detail.get("physical_fixture_missing_surface_ids")
        violations = detail.get("violations")
        physical_missing_ids = list(physical_missing) if isinstance(physical_missing, list) else []
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "plan_status": detail.get("plan_status"),
            "action_count": int(detail.get("action_count") or 0),
            "incomplete_surface_count": int(detail.get("incomplete_surface_count") or 0),
            "physical_gap_surface_count": len(physical_missing_ids),
            "physical_fixture_missing_surface_ids": physical_missing_ids,
            "next_physical_readback_surface_id": detail.get("next_physical_readback_surface_id"),
            "next_physical_readback_action_id": detail.get("next_physical_readback_action_id"),
            "next_physical_readback_target_action_ids": (
                list(detail.get("next_physical_readback_target_action_ids"))
                if isinstance(detail.get("next_physical_readback_target_action_ids"), list)
                else []
            ),
            "target_action_consistency": (
                detail.get("next_physical_readback_surface_id")
                == detail.get("expected_next_physical_readback_surface_id")
                and detail.get("next_physical_readback_action_id")
                == detail.get("expected_next_physical_readback_action_id")
                and detail.get("next_physical_readback_target_action_ids")
                == [detail.get("expected_next_physical_readback_action_id")]
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def next_wing_proof_plan_summary(results: Sequence[CheckResult]) -> dict[str, object] | None:
    for result in results:
        if result.name != "next_wing_proof_plan_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        planned_value_types = detail.get("planned_value_types")
        planned_channels = detail.get("planned_channels")
        global_blockers = detail.get("global_blockers")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "execution_allowed_by_plan": False,
            "operator_required": True,
            "full_autonomous_ready": False,
            "plan_status": detail.get("plan_status"),
            "step_count": int(detail.get("step_count") or 0),
            "planned_value_types": list(planned_value_types) if isinstance(planned_value_types, list) else [],
            "planned_channels": list(planned_channels) if isinstance(planned_channels, list) else [],
            "global_blockers": list(global_blockers) if isinstance(global_blockers, list) else [],
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def write_summary(run_dir: Path, results: Sequence[CheckResult], *, generated_at: str) -> dict[str, object]:
    failed = [result.name for result in results if result.status != "passed"]
    gap_rollup = autonomy_gap_rollup_summary(results)
    autonomy_overlay = autonomy_evidence_overlay_summary(results)
    gain_dry_run = supervised_gain_dry_run_summary(results)
    read_fixture = read_surface_fixture_evidence_summary(results)
    read_remediation = read_surface_remediation_plan_summary(results)
    next_plan = next_wing_proof_plan_summary(results)
    summary = {
        "type": "no_wing_release_check",
        "generated_at": generated_at,
        "status": "passed" if not failed else "failed",
        "connection_policy": "no_hardware_no_network_no_live_writes",
        "live_write": False,
        "failed_checks": failed,
        "results": [asdict(result) for result in results],
    }
    if gap_rollup is not None:
        summary["autonomy_gap_rollup"] = gap_rollup
    if autonomy_overlay is not None:
        summary["autonomy_evidence_overlay"] = autonomy_overlay
    if gain_dry_run is not None:
        summary["supervised_gain_dry_run"] = gain_dry_run
    if read_fixture is not None:
        summary["read_surface_fixture_evidence"] = read_fixture
    if read_remediation is not None:
        summary["read_surface_remediation_plan"] = read_remediation
    if next_plan is not None:
        summary["next_wing_proof_plan"] = next_plan
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
    if gap_rollup is not None:
        lines.extend(
            [
                "",
                "## Autonomy Gap Rollup",
                "",
                f"- schema_check: `{gap_rollup['status']}`",
                f"- read_only: `{str(gap_rollup['read_only']).lower()}`",
                f"- live_write: `{str(gap_rollup['live_write']).lower()}`",
                f"- full_autonomous_ready: `{str(gap_rollup['full_autonomous_ready']).lower()}`",
                "- full_auto_gap_bucket_counts: "
                f"`{json.dumps(gap_rollup['full_auto_gap_bucket_counts'], sort_keys=True)}`",
                "- supervised_current_live_write_kinds: "
                f"`{json.dumps(gap_rollup['supervised_current_live_write_kinds'])}`",
                "- policy_supervised_live_write_kinds: "
                f"`{json.dumps(gap_rollup['policy_supervised_live_write_kinds'])}`",
                f"- violations: `{json.dumps(gap_rollup['violations'])}`",
            ]
        )
    if autonomy_overlay is not None:
        lines.extend(
            [
                "",
                "## Autonomy Evidence Overlay",
                "",
                f"- schema_check: `{autonomy_overlay['status']}`",
                f"- read_only: `{str(autonomy_overlay['read_only']).lower()}`",
                f"- live_write: `{str(autonomy_overlay['live_write']).lower()}`",
                f"- live_widening_allowed: `{str(autonomy_overlay['live_widening_allowed']).lower()}`",
                f"- automatic_safety_update_allowed: `{str(autonomy_overlay['automatic_safety_update_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(autonomy_overlay['full_autonomous_ready']).lower()}`",
                f"- operator_required: `{str(autonomy_overlay['operator_required']).lower()}`",
                f"- overlay_status: `{autonomy_overlay['overlay_status']}`",
                f"- row_count: `{autonomy_overlay['row_count']}`",
                "- status_counts: "
                f"`{json.dumps(autonomy_overlay['status_counts'], sort_keys=True)}`",
                "- dry_run_only_target_ids: "
                f"`{json.dumps(autonomy_overlay['dry_run_only_target_ids'])}`",
                "- manual_safety_review_candidate_target_ids: "
                f"`{json.dumps(autonomy_overlay['manual_safety_review_candidate_target_ids'])}`",
                f"- violations: `{json.dumps(autonomy_overlay['violations'])}`",
            ]
        )
    if gain_dry_run is not None:
        lines.extend(
            [
                "",
                "## Supervised Gain Dry-Run Proof",
                "",
                f"- schema_check: `{gain_dry_run['status']}`",
                f"- read_only: `{str(gain_dry_run['read_only']).lower()}`",
                f"- live_write: `{str(gain_dry_run['live_write']).lower()}`",
                f"- promotion_allowed: `{str(gain_dry_run['promotion_allowed']).lower()}`",
                f"- autonomous_ready: `{str(gain_dry_run['autonomous_ready']).lower()}`",
                f"- gain_dry_run_row_count: `{gain_dry_run['gain_dry_run_row_count']}`",
                "- dry_run_ready_proof_ids: "
                f"`{json.dumps(gain_dry_run['dry_run_ready_proof_ids'])}`",
                f"- violations: `{json.dumps(gain_dry_run['violations'])}`",
            ]
        )
    if read_fixture is not None:
        lines.extend(
            [
                "",
                "## Read Surface Fixture Evidence",
                "",
                f"- schema_check: `{read_fixture['status']}`",
                f"- read_only: `{str(read_fixture['read_only']).lower()}`",
                f"- live_write: `{str(read_fixture['live_write']).lower()}`",
                f"- full_autonomous_ready: `{str(read_fixture['full_autonomous_ready']).lower()}`",
                f"- snapshot_status: `{read_fixture['snapshot_status']}`",
                f"- surface_count: `{read_fixture['surface_count']}`",
                f"- surface_with_fixture_count: `{read_fixture['surface_with_fixture_count']}`",
                f"- declared_fixture_path_count: `{read_fixture['declared_fixture_path_count']}`",
                f"- declared_test_name_count: `{read_fixture['declared_test_name_count']}`",
                f"- physical_gap_surface_count: `{read_fixture['physical_gap_surface_count']}`",
                f"- summary_recomputed_from_rows: `{str(read_fixture['summary_recomputed_from_rows']).lower()}`",
                f"- fixture_declaration_validation: `{str(read_fixture['fixture_declaration_validation']).lower()}`",
                f"- contract_gap_consistency_enforced: `{str(read_fixture['contract_gap_consistency_enforced']).lower()}`",
                "- status_counts: "
                f"`{json.dumps(read_fixture['status_counts'], sort_keys=True)}`",
                "- physical_fixture_missing_surface_ids: "
                f"`{json.dumps(read_fixture['physical_fixture_missing_surface_ids'])}`",
                "- next_physical_readback_surface_id: "
                f"`{read_fixture['next_physical_readback_surface_id']}`",
                f"- violations: `{json.dumps(read_fixture['violations'])}`",
            ]
        )
    if read_remediation is not None:
        lines.extend(
            [
                "",
                "## Read Surface Remediation Plan",
                "",
                f"- schema_check: `{read_remediation['status']}`",
                f"- read_only: `{str(read_remediation['read_only']).lower()}`",
                f"- live_write: `{str(read_remediation['live_write']).lower()}`",
                f"- mutates_console_state: `{str(read_remediation['mutates_console_state']).lower()}`",
                f"- live_widening_allowed: `{str(read_remediation['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(read_remediation['full_autonomous_ready']).lower()}`",
                f"- plan_status: `{read_remediation['plan_status']}`",
                f"- action_count: `{read_remediation['action_count']}`",
                f"- incomplete_surface_count: `{read_remediation['incomplete_surface_count']}`",
                f"- physical_gap_surface_count: `{read_remediation['physical_gap_surface_count']}`",
                "- physical_fixture_missing_surface_ids: "
                f"`{json.dumps(read_remediation['physical_fixture_missing_surface_ids'])}`",
                "- next_physical_readback_surface_id: "
                f"`{read_remediation['next_physical_readback_surface_id']}`",
                "- next_physical_readback_action_id: "
                f"`{read_remediation['next_physical_readback_action_id']}`",
                "- next_physical_readback_target_action_ids: "
                f"`{json.dumps(read_remediation['next_physical_readback_target_action_ids'])}`",
                f"- target_action_consistency: `{str(read_remediation['target_action_consistency']).lower()}`",
                f"- violations: `{json.dumps(read_remediation['violations'])}`",
            ]
        )
    if next_plan is not None:
        lines.extend(
            [
                "",
                "## Next WING Proof Plan",
                "",
                f"- schema_check: `{next_plan['status']}`",
                f"- read_only: `{str(next_plan['read_only']).lower()}`",
                f"- live_write: `{str(next_plan['live_write']).lower()}`",
                f"- execution_allowed_by_plan: `{str(next_plan['execution_allowed_by_plan']).lower()}`",
                f"- operator_required: `{str(next_plan['operator_required']).lower()}`",
                f"- full_autonomous_ready: `{str(next_plan['full_autonomous_ready']).lower()}`",
                f"- plan_status: `{next_plan['plan_status']}`",
                f"- step_count: `{next_plan['step_count']}`",
                f"- planned_value_types: `{json.dumps(next_plan['planned_value_types'])}`",
                f"- planned_channels: `{json.dumps(next_plan['planned_channels'])}`",
                f"- global_blockers: `{json.dumps(next_plan['global_blockers'])}`",
                f"- violations: `{json.dumps(next_plan['violations'])}`",
            ]
        )
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
    proof_report_path = proof_report_path_for_run(run_dir)

    checks = build_checks(
        run_dir=run_dir,
        include_frontend=include_frontend,
        include_package=include_package,
        quick=quick,
        command_timeout_sec=command_timeout_sec,
    )
    validate_repo_local_commands(checks)
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
        if check.name == "read_surface_fixture_evidence" and result.status == "passed":
            read_fixture_result = validate_read_surface_fixture_evidence_snapshot()
            results.append(read_fixture_result)
            if read_fixture_result.status != "passed" and not continue_on_failure:
                break
            read_remediation_result = validate_read_surface_remediation_plan_snapshot()
            results.append(read_remediation_result)
            if read_remediation_result.status != "passed" and not continue_on_failure:
                break
        if check.name == "supervised_proof_artifact_report" and result.status == "passed":
            gain_dry_run_result = validate_supervised_proof_report_gain_dry_run(proof_report_path)
            results.append(gain_dry_run_result)
            if gain_dry_run_result.status != "passed" and not continue_on_failure:
                break
            autonomy_overlay_result = validate_supervised_proof_report_autonomy_overlay(proof_report_path)
            results.append(autonomy_overlay_result)
            if autonomy_overlay_result.status != "passed" and not continue_on_failure:
                break
            next_plan_result = validate_supervised_proof_report_next_wing_plan(proof_report_path)
            results.append(next_plan_result)
            if next_plan_result.status != "passed" and not continue_on_failure:
                break
            artifact_result = validate_supervised_proof_report_gap_rollup(proof_report_path)
            results.append(artifact_result)
            if artifact_result.status != "passed" and not continue_on_failure:
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
