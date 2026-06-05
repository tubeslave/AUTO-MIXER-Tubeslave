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
from backend.bus_group_main_dca_physical_readback_contract import (  # noqa: E402
    BUS_GROUP_MAIN_DCA_FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_TYPE,
    BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SCHEMA_ID,
    BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SOURCES,
    BUS_GROUP_MAIN_DCA_REMEDIATION_ACTION_ID,
    BUS_GROUP_MAIN_DCA_REQUIRED_BUS_ASSIGNMENT_FIELDS,
    BUS_GROUP_MAIN_DCA_REQUIRED_BUS_CONTROL_FIELDS,
    BUS_GROUP_MAIN_DCA_REQUIRED_BUS_COVERAGE_FLAGS,
    BUS_GROUP_MAIN_DCA_REQUIRED_BUS_FIELDS,
    BUS_GROUP_MAIN_DCA_REQUIRED_DCA_ASSIGNMENT_FIELDS,
    BUS_GROUP_MAIN_DCA_REQUIRED_DCA_CONTROL_FIELDS,
    BUS_GROUP_MAIN_DCA_REQUIRED_DCA_COVERAGE_FLAGS,
    BUS_GROUP_MAIN_DCA_REQUIRED_DCA_FIELDS,
    BUS_GROUP_MAIN_DCA_REQUIRED_FALSE_SAFETY_FLAGS,
    BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_CONTROL_FIELDS,
    BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_COVERAGE_FLAGS,
    BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_FIELDS,
    BUS_GROUP_MAIN_DCA_REQUIRED_METERING_FIELDS,
    BUS_GROUP_MAIN_DCA_REQUIRED_TOP_LEVEL_FIELDS,
    BUS_GROUP_MAIN_DCA_SURFACE_ID,
    BUS_GROUP_MAIN_DCA_VERIFIED_PHYSICAL_STATUSES,
    build_bus_group_main_dca_physical_readback_contract,
)
from backend.bus_group_main_dca_physical_readback_artifact_intake import (  # noqa: E402
    BUS_GROUP_MAIN_DCA_INTAKE_STATUSES,
    BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_bus_group_main_dca_physical_readback_artifact_intake,
)
from backend.channel_basic_controls_physical_readback_contract import (  # noqa: E402
    CHANNEL_BASIC_CONTROLS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_TYPE,
    CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SCHEMA_ID,
    CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SOURCES,
    CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID,
    CHANNEL_BASIC_CONTROLS_REQUIRED_CHANNEL_FIELDS,
    CHANNEL_BASIC_CONTROLS_REQUIRED_CONTROL_FIELDS,
    CHANNEL_BASIC_CONTROLS_REQUIRED_COVERAGE_FLAGS,
    CHANNEL_BASIC_CONTROLS_REQUIRED_FALSE_SAFETY_FLAGS,
    CHANNEL_BASIC_CONTROLS_REQUIRED_IDENTITY_FIELDS,
    CHANNEL_BASIC_CONTROLS_REQUIRED_TOP_LEVEL_FIELDS,
    CHANNEL_BASIC_CONTROLS_SURFACE_ID,
    CHANNEL_BASIC_CONTROLS_VERIFIED_PHYSICAL_STATUSES,
    build_channel_basic_controls_physical_readback_contract,
)
from backend.channel_basic_controls_physical_readback_artifact_intake import (  # noqa: E402
    CHANNEL_BASIC_CONTROLS_INTAKE_STATUSES,
    CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_channel_basic_controls_physical_readback_artifact_intake,
)
from backend.channel_processing_physical_readback_contract import (  # noqa: E402
    CHANNEL_PROCESSING_FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_TYPE,
    CHANNEL_PROCESSING_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    CHANNEL_PROCESSING_PHYSICAL_READBACK_SCHEMA_ID,
    CHANNEL_PROCESSING_PHYSICAL_READBACK_SOURCES,
    CHANNEL_PROCESSING_REMEDIATION_ACTION_ID,
    CHANNEL_PROCESSING_REQUIRED_CHANNEL_FIELDS,
    CHANNEL_PROCESSING_REQUIRED_COMPRESSOR_FIELDS,
    CHANNEL_PROCESSING_REQUIRED_COVERAGE_FLAGS,
    CHANNEL_PROCESSING_REQUIRED_EQ_BAND_FIELDS,
    CHANNEL_PROCESSING_REQUIRED_EQ_FIELDS,
    CHANNEL_PROCESSING_REQUIRED_FALSE_SAFETY_FLAGS,
    CHANNEL_PROCESSING_REQUIRED_GATE_FIELDS,
    CHANNEL_PROCESSING_REQUIRED_HPF_FIELDS,
    CHANNEL_PROCESSING_REQUIRED_TOP_LEVEL_FIELDS,
    CHANNEL_PROCESSING_SURFACE_ID,
    CHANNEL_PROCESSING_VERIFIED_PHYSICAL_STATUSES,
    MINIMUM_EQ_BAND_COUNT,
    build_channel_processing_physical_readback_contract,
)
from backend.channel_processing_physical_readback_artifact_intake import (  # noqa: E402
    CHANNEL_PROCESSING_INTAKE_STATUSES,
    CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_channel_processing_physical_readback_artifact_intake,
)
from backend.dlive_state_physical_readback_contract import (  # noqa: E402
    DLIVE_STATE_FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE,
    DLIVE_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    DLIVE_STATE_PHYSICAL_READBACK_SCHEMA_ID,
    DLIVE_STATE_PHYSICAL_READBACK_SOURCES,
    DLIVE_STATE_REMEDIATION_ACTION_ID,
    DLIVE_STATE_REQUIRED_CHANNEL_FIELDS,
    DLIVE_STATE_REQUIRED_COVERAGE_FLAGS,
    DLIVE_STATE_REQUIRED_FALSE_SAFETY_FLAGS,
    DLIVE_STATE_REQUIRED_MIDI_STATE_FIELDS,
    DLIVE_STATE_REQUIRED_PARITY_FIELDS,
    DLIVE_STATE_REQUIRED_TOP_LEVEL_FIELDS,
    DLIVE_STATE_SURFACE_ID,
    DLIVE_STATE_VERIFIED_PHYSICAL_STATUSES,
    MINIMUM_DLIVE_STATE_CHANNEL_COUNT,
    build_dlive_state_physical_readback_contract,
)
from backend.fx_slots_inserts_physical_readback_contract import (  # noqa: E402
    FX_INSERT_POSITIONS,
    FX_INSERT_REQUIRED_COVERAGE_FLAGS,
    FX_INSERT_REQUIRED_FIELDS,
    FX_INSERT_TARGET_TYPES,
    FX_PARAMETER_REQUIRED_FIELDS,
    FX_PARAMETER_SAFE_RANGE_FIELDS,
    FX_PARAMETER_SCHEMA_REQUIRED_FIELDS,
    FX_SLOT_REQUIRED_COVERAGE_FLAGS,
    FX_SLOT_REQUIRED_FIELDS,
    FX_SLOTS_INSERTS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_TYPE,
    FX_SLOTS_INSERTS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID,
    FX_SLOTS_INSERTS_PHYSICAL_READBACK_SOURCES,
    FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID,
    FX_SLOTS_INSERTS_REQUIRED_FALSE_SAFETY_FLAGS,
    FX_SLOTS_INSERTS_REQUIRED_TOP_LEVEL_FIELDS,
    FX_SLOTS_INSERTS_SURFACE_ID,
    FX_SLOTS_INSERTS_VERIFIED_PHYSICAL_STATUSES,
    MINIMUM_FX_INSERT_COUNT,
    MINIMUM_FX_PARAMETER_COUNT,
    MINIMUM_FX_SLOT_COUNT,
    build_fx_slots_inserts_physical_readback_contract,
)
from backend.matrix_state_physical_readback_contract import (  # noqa: E402
    MATRIX_STATE_FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE,
    MATRIX_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    MATRIX_STATE_PHYSICAL_READBACK_SCHEMA_ID,
    MATRIX_STATE_PHYSICAL_READBACK_SOURCES,
    MATRIX_STATE_REMEDIATION_ACTION_ID,
    MATRIX_STATE_REQUIRED_CONTROL_FIELDS,
    MATRIX_STATE_REQUIRED_COVERAGE_FLAGS,
    MATRIX_STATE_REQUIRED_DYNAMICS_FIELDS,
    MATRIX_STATE_REQUIRED_EQ_BAND_FIELDS,
    MATRIX_STATE_REQUIRED_EQ_FIELDS,
    MATRIX_STATE_REQUIRED_FALSE_SAFETY_FLAGS,
    MATRIX_STATE_REQUIRED_INSERT_FIELDS,
    MATRIX_STATE_REQUIRED_MATRIX_FIELDS,
    MATRIX_STATE_REQUIRED_METERING_FIELDS,
    MATRIX_STATE_REQUIRED_TOP_LEVEL_FIELDS,
    MATRIX_STATE_SURFACE_ID,
    MATRIX_STATE_VERIFIED_PHYSICAL_STATUSES,
    MINIMUM_MATRIX_EQ_BAND_COUNT,
    build_matrix_state_physical_readback_contract,
)
from backend.matrix_state_physical_readback_artifact_intake import (  # noqa: E402
    MATRIX_STATE_INTAKE_STATUSES,
    MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_matrix_state_physical_readback_artifact_intake,
)
from backend.read_surface_fixture_evidence import (  # noqa: E402
    FIXTURE_EVIDENCE_STATUSES,
    build_read_surface_fixture_evidence_snapshot,
)
from backend.raw_mixer_state_physical_readback_contract import (  # noqa: E402
    FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    PHYSICAL_READBACK_SOURCES,
    RAW_MIXER_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE,
    RAW_MIXER_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    RAW_MIXER_STATE_PHYSICAL_READBACK_SCHEMA_ID,
    RAW_MIXER_STATE_REMEDIATION_ACTION_ID,
    RAW_MIXER_STATE_SURFACE_ID,
    REQUIRED_FALSE_SAFETY_FLAGS,
    REQUIRED_ROW_FIELDS,
    REQUIRED_TOP_LEVEL_FIELDS,
    VERIFIED_PHYSICAL_ROW_STATUSES,
    build_raw_mixer_state_physical_readback_contract,
)
from backend.raw_mixer_state_physical_readback_artifact_intake import (  # noqa: E402
    INTAKE_STATUSES,
    RAW_MIXER_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_raw_mixer_state_physical_readback_artifact_intake,
)
from backend.routing_sends_physical_readback_contract import (  # noqa: E402
    MINIMUM_CHANNEL_BUS_SEND_COUNT,
    MINIMUM_CHANNEL_MAIN_SEND_COUNT,
    MINIMUM_CHANNEL_MATRIX_SEND_COUNT,
    MINIMUM_OUTPUT_ROUTE_COUNT,
    MINIMUM_STATIC_DANTE_ROUTE_COUNT,
    ROUTING_SENDS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_TYPE,
    ROUTING_SENDS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    ROUTING_SENDS_PHYSICAL_READBACK_SCHEMA_ID,
    ROUTING_SENDS_PHYSICAL_READBACK_SOURCES,
    ROUTING_SENDS_REMEDIATION_ACTION_ID,
    ROUTING_SENDS_REQUIRED_BUS_SEND_FIELDS,
    ROUTING_SENDS_REQUIRED_FALSE_SAFETY_FLAGS,
    ROUTING_SENDS_REQUIRED_MAIN_SEND_FIELDS,
    ROUTING_SENDS_REQUIRED_MATRIX_SEND_FIELDS,
    ROUTING_SENDS_REQUIRED_OUTPUT_ROUTE_COVERAGE_FLAGS,
    ROUTING_SENDS_REQUIRED_OUTPUT_ROUTE_FIELDS,
    ROUTING_SENDS_REQUIRED_SEND_COVERAGE_FLAGS,
    ROUTING_SENDS_REQUIRED_STATIC_ROUTE_COVERAGE_FLAGS,
    ROUTING_SENDS_REQUIRED_STATIC_ROUTE_FIELDS,
    ROUTING_SENDS_REQUIRED_TOP_LEVEL_FIELDS,
    ROUTING_SENDS_SURFACE_ID,
    ROUTING_SENDS_VERIFIED_PHYSICAL_STATUSES,
    build_routing_sends_physical_readback_contract,
)
from backend.snapshots_scenes_physical_readback_contract import (  # noqa: E402
    MINIMUM_SNAPSHOT_SCENE_COUNT,
    SNAPSHOTS_SCENES_FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    SNAPSHOTS_SCENES_PHYSICAL_READBACK_ARTIFACT_TYPE,
    SNAPSHOTS_SCENES_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    SNAPSHOTS_SCENES_PHYSICAL_READBACK_SCHEMA_ID,
    SNAPSHOTS_SCENES_PHYSICAL_READBACK_SOURCES,
    SNAPSHOTS_SCENES_REMEDIATION_ACTION_ID,
    SNAPSHOTS_SCENES_REQUIRED_ACTIVE_SCENE_FIELDS,
    SNAPSHOTS_SCENES_REQUIRED_COVERAGE_FLAGS,
    SNAPSHOTS_SCENES_REQUIRED_FALSE_SAFETY_FLAGS,
    SNAPSHOTS_SCENES_REQUIRED_SNAPSHOT_FIELDS,
    SNAPSHOTS_SCENES_REQUIRED_TOP_LEVEL_FIELDS,
    SNAPSHOTS_SCENES_SURFACE_ID,
    SNAPSHOTS_SCENES_VERIFIED_PHYSICAL_STATUSES,
    build_snapshots_scenes_physical_readback_contract,
)
from backend.user_presets_physical_readback_contract import (  # noqa: E402
    MINIMUM_USER_PRESET_COUNT,
    USER_PRESETS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    USER_PRESETS_PHYSICAL_READBACK_ARTIFACT_TYPE,
    USER_PRESETS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    USER_PRESETS_PHYSICAL_READBACK_SCHEMA_ID,
    USER_PRESETS_PHYSICAL_READBACK_SOURCES,
    USER_PRESETS_REMEDIATION_ACTION_ID,
    USER_PRESETS_REQUIRED_COVERAGE_FLAGS,
    USER_PRESETS_REQUIRED_FALSE_SAFETY_FLAGS,
    USER_PRESETS_REQUIRED_PRESET_FIELDS,
    USER_PRESETS_REQUIRED_TOP_LEVEL_FIELDS,
    USER_PRESETS_SURFACE_ID,
    USER_PRESETS_VERIFIED_PHYSICAL_STATUSES,
    build_user_presets_physical_readback_contract,
)
from backend.read_surface_physical_readback_capture_plan import (  # noqa: E402
    READ_SURFACE_PHYSICAL_READBACK_CAPTURE_PLAN_METHOD_ID,
    build_read_surface_physical_readback_capture_plan,
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
                "tests/test_raw_mixer_state_physical_readback_contract.py",
                "tests/test_raw_mixer_state_physical_readback_artifact_intake.py",
                "tests/test_channel_basic_controls_physical_readback_contract.py",
                "tests/test_channel_basic_controls_physical_readback_artifact_intake.py",
                "tests/test_channel_processing_physical_readback_contract.py",
                "tests/test_channel_processing_physical_readback_artifact_intake.py",
                "tests/test_bus_group_main_dca_physical_readback_contract.py",
                "tests/test_bus_group_main_dca_physical_readback_artifact_intake.py",
                "tests/test_matrix_state_physical_readback_contract.py",
                "tests/test_matrix_state_physical_readback_artifact_intake.py",
                "tests/test_fx_slots_inserts_physical_readback_contract.py",
                "tests/test_routing_sends_physical_readback_contract.py",
                "tests/test_snapshots_scenes_physical_readback_contract.py",
                "tests/test_user_presets_physical_readback_contract.py",
                "tests/test_dlive_state_physical_readback_contract.py",
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


def _string_list_map(value: Any) -> tuple[dict[str, list[str]], bool]:
    if not isinstance(value, Mapping):
        return {}, False
    normalized: dict[str, list[str]] = {}
    for key, raw_items in value.items():
        if not isinstance(key, str) or not key:
            return {}, False
        if not isinstance(raw_items, list):
            return {}, False
        items = [str(item) for item in raw_items if isinstance(item, str) and item]
        if len(items) != len(raw_items):
            return {}, False
        if items:
            normalized[key] = items
    return normalized, True


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
        evidence_by_family, evidence_by_family_valid = _string_list_map(
            rollup.get("evidence_requirement_ids_by_action_family")
        )
        next_by_family, next_by_family_valid = _string_list_map(
            rollup.get("next_required_evidence_by_action_family")
        )
        if not evidence_by_family_valid:
            violations.append("autonomy_gap_rollup_evidence_by_family_missing")
        if not next_by_family_valid:
            violations.append("autonomy_gap_rollup_next_by_family_missing")
        if int(rollup_counts.get("evidence_requirement") or 0) > 0 and not evidence_by_family:
            violations.append("autonomy_gap_rollup_evidence_by_family_empty")
        if int(rollup_counts.get("evidence_requirement") or 0) > 0 and not next_by_family:
            violations.append("autonomy_gap_rollup_next_by_family_empty")
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
        detail["evidence_requirement_ids_by_action_family"] = evidence_by_family
        detail["next_required_evidence_by_action_family"] = next_by_family
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


def validate_read_surface_physical_readback_capture_plan_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "read_surface_physical_readback_capture_plan_must_be_operator_required_and_non_executing",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        capture_plan = build_read_surface_physical_readback_capture_plan(
            fixture_evidence_snapshot=fixture_snapshot,
            remediation_plan=remediation_plan,
        )
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        capture_plan = {}
        violations.append(f"read_surface_capture_plan_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_summary = (
        remediation_plan.get("summary")
        if isinstance(remediation_plan.get("summary"), Mapping)
        else {}
    )
    capture_summary = (
        capture_plan.get("summary")
        if isinstance(capture_plan.get("summary"), Mapping)
        else {}
    )
    safety = (
        capture_plan.get("safety")
        if isinstance(capture_plan.get("safety"), Mapping)
        else {}
    )
    steps = capture_plan.get("steps") if isinstance(capture_plan.get("steps"), list) else []
    if not isinstance(capture_plan.get("steps"), list):
        violations.append("read_surface_capture_plan_steps_missing")

    expected_physical_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(expected_physical_ids, list):
        expected_physical_ids = []
        violations.append("read_surface_capture_plan_fixture_physical_ids_missing")
    expected_physical_ids = [
        item for item in expected_physical_ids if isinstance(item, str) and item
    ]
    expected_next_target = fixture_summary.get("next_physical_readback_surface_id")
    if not isinstance(expected_next_target, str):
        expected_next_target = expected_physical_ids[0] if expected_physical_ids else None
    expected_next_action_id = f"remediate_{expected_next_target}" if expected_next_target else None

    next_target = capture_summary.get("next_physical_readback_surface_id")
    if next_target != expected_next_target:
        violations.append("read_surface_capture_plan_next_target_mismatch")
    next_action_id = capture_summary.get("next_physical_readback_action_id")
    if next_action_id != expected_next_action_id:
        violations.append("read_surface_capture_plan_next_action_id_mismatch")
    if capture_summary.get("target_action_consistency") is not bool(expected_next_target):
        violations.append("read_surface_capture_plan_target_action_consistency_mismatch")
    if remediation_summary.get("next_physical_readback_surface_id") != expected_next_target:
        violations.append("read_surface_capture_plan_remediation_target_mismatch")
    if remediation_summary.get("next_physical_readback_action_id") != expected_next_action_id:
        violations.append("read_surface_capture_plan_remediation_action_mismatch")

    expected_true = ("read_only", "operator_required")
    for key in expected_true:
        if safety.get(key) is not True:
            violations.append(f"read_surface_capture_plan_{key}_not_true")
    expected_false = (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_plan",
        "live_widening_allowed",
        "live_authority_granted",
        "full_autonomous_ready",
    )
    for key in expected_false:
        if safety.get(key) is not False:
            violations.append(f"read_surface_capture_plan_{key}_not_false")

    if capture_plan.get("type") != "read_surface_physical_readback_capture_plan":
        violations.append("read_surface_capture_plan_type_mismatch")
    if (
        capture_plan.get("method_id")
        != READ_SURFACE_PHYSICAL_READBACK_CAPTURE_PLAN_METHOD_ID
    ):
        violations.append("read_surface_capture_plan_method_id_mismatch")
    expected_status = (
        "operator_capture_required"
        if expected_next_target
        else "read_surface_physical_readback_complete"
    )
    if capture_plan.get("status") != expected_status:
        violations.append("read_surface_capture_plan_status_mismatch")
    if expected_next_target and len(steps) != 1:
        violations.append("read_surface_capture_plan_step_count_mismatch")
    if not expected_next_target and steps:
        violations.append("read_surface_capture_plan_complete_but_steps_present")

    required_artifact_contracts = capture_summary.get("required_artifact_contracts")
    if not isinstance(required_artifact_contracts, list):
        required_artifact_contracts = []
        violations.append("read_surface_capture_plan_required_contracts_missing")
    elif expected_next_target and "physical_readback_fixture" not in required_artifact_contracts:
        violations.append("read_surface_capture_plan_missing_physical_fixture_contract")

    step = steps[0] if steps and isinstance(steps[0], Mapping) else {}
    if expected_next_target:
        if not step:
            violations.append("read_surface_capture_plan_next_step_missing")
        elif step.get("surface_id") != expected_next_target:
            violations.append("read_surface_capture_plan_step_surface_mismatch")
        elif step.get("remediation_action_id") != expected_next_action_id:
            violations.append("read_surface_capture_plan_step_action_mismatch")
        for key in ("operator_required",):
            if step and step.get(key) is not True:
                violations.append(f"read_surface_capture_plan_step_{key}_not_true")
        for key in (
            "execution_allowed_by_plan",
            "live_write",
            "mutates_console_state",
            "queries_console",
            "executes_tests",
        ):
            if step and step.get(key) is not False:
                violations.append(f"read_surface_capture_plan_step_{key}_not_false")

    detail["plan_status"] = capture_plan.get("status")
    detail["capture_step_count"] = int(capture_summary.get("capture_step_count") or len(steps))
    detail["physical_fixture_missing_surface_ids"] = list(expected_physical_ids)
    detail["next_physical_readback_surface_id"] = next_target
    detail["next_physical_readback_action_id"] = next_action_id
    detail["expected_next_physical_readback_surface_id"] = expected_next_target
    detail["expected_next_physical_readback_action_id"] = expected_next_action_id
    detail["target_action_consistency"] = (
        capture_summary.get("target_action_consistency") is True
    )
    detail["required_artifact_contracts"] = (
        list(required_artifact_contracts)
        if isinstance(required_artifact_contracts, list)
        else []
    )
    detail["blocked_by"] = (
        list(capture_summary.get("blocked_by"))
        if isinstance(capture_summary.get("blocked_by"), list)
        else []
    )
    detail["step_surface_id"] = step.get("surface_id") if step else None
    detail["step_remediation_action_id"] = step.get("remediation_action_id") if step else None
    detail["step_forbidden_actions"] = (
        list(step.get("forbidden_actions"))
        if step and isinstance(step.get("forbidden_actions"), list)
        else []
    )
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="read_surface_physical_readback_capture_plan_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_raw_mixer_state_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "raw_mixer_state_readback_contract_is_schema_only_no_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_raw_mixer_state_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"raw_mixer_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_summary = (
        remediation_plan.get("summary")
        if isinstance(remediation_plan.get("summary"), Mapping)
        else {}
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "raw_mixer_state_physical_readback_contract":
        violations.append("raw_mixer_readback_contract_type_mismatch")
    if contract.get("method_id") != RAW_MIXER_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("raw_mixer_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != RAW_MIXER_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("raw_mixer_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("raw_mixer_readback_contract_status_mismatch")
    if contract.get("surface_id") != RAW_MIXER_STATE_SURFACE_ID:
        violations.append("raw_mixer_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != RAW_MIXER_STATE_REMEDIATION_ACTION_ID:
        violations.append("raw_mixer_readback_contract_action_id_mismatch")

    expected_true = ("read_only",)
    for key in expected_true:
        if safety.get(key) is not True:
            violations.append(f"raw_mixer_readback_contract_{key}_not_true")
    expected_false = (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
    )
    for key in expected_false:
        if safety.get(key) is not False:
            violations.append(f"raw_mixer_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != RAW_MIXER_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("raw_mixer_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != RAW_MIXER_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("raw_mixer_readback_artifact_schema_id_mismatch")
    if artifact_contract.get("minimum_row_count") != 1:
        violations.append("raw_mixer_readback_artifact_minimum_row_count_mismatch")

    required_top_level = artifact_contract.get("required_top_level_fields")
    if not isinstance(required_top_level, list) or required_top_level != list(REQUIRED_TOP_LEVEL_FIELDS):
        violations.append("raw_mixer_readback_required_top_level_fields_mismatch")
        required_top_level = []
    required_row_fields = artifact_contract.get("required_row_fields")
    if not isinstance(required_row_fields, list) or required_row_fields != list(REQUIRED_ROW_FIELDS):
        violations.append("raw_mixer_readback_required_row_fields_mismatch")
        required_row_fields = []
    required_false_flags = artifact_contract.get("required_false_safety_flags")
    if not isinstance(required_false_flags, list) or required_false_flags != list(REQUIRED_FALSE_SAFETY_FLAGS):
        violations.append("raw_mixer_readback_required_false_safety_flags_mismatch")
        required_false_flags = []

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("raw_mixer_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(PHYSICAL_READBACK_SOURCES):
        violations.append("raw_mixer_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("raw_mixer_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_row_statuses")
    if not isinstance(accepted_statuses, list) or accepted_statuses != list(VERIFIED_PHYSICAL_ROW_STATUSES):
        violations.append("raw_mixer_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields != list(FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("raw_mixer_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    fixture_next_target = fixture_summary.get("next_physical_readback_surface_id")
    remediation_next_target = remediation_summary.get("next_physical_readback_surface_id")
    remediation_next_action = remediation_summary.get("next_physical_readback_action_id")
    contract_next_target = contract_summary.get("next_physical_readback_surface_id")
    contract_next_action = contract_summary.get("next_physical_readback_action_id")
    if fixture_next_target != RAW_MIXER_STATE_SURFACE_ID:
        violations.append("raw_mixer_readback_fixture_next_target_mismatch")
    if remediation_next_target != RAW_MIXER_STATE_SURFACE_ID:
        violations.append("raw_mixer_readback_remediation_next_target_mismatch")
    if remediation_next_action != RAW_MIXER_STATE_REMEDIATION_ACTION_ID:
        violations.append("raw_mixer_readback_remediation_next_action_mismatch")
    if contract_next_target != RAW_MIXER_STATE_SURFACE_ID:
        violations.append("raw_mixer_readback_contract_next_target_mismatch")
    if contract_next_action != RAW_MIXER_STATE_REMEDIATION_ACTION_ID:
        violations.append("raw_mixer_readback_contract_next_action_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("raw_mixer_readback_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("raw_mixer_readback_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("raw_mixer_readback_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("raw_mixer_readback_contract_live_authority_not_false")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["next_physical_readback_surface_id"] = contract_next_target
    detail["next_physical_readback_action_id"] = contract_next_action
    detail["fixture_next_physical_readback_surface_id"] = fixture_next_target
    detail["remediation_next_physical_readback_surface_id"] = remediation_next_target
    detail["remediation_next_physical_readback_action_id"] = remediation_next_action
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["required_top_level_field_count"] = len(required_top_level)
    detail["required_row_field_count"] = len(required_row_fields)
    detail["required_false_safety_flags"] = list(required_false_flags)
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_row_statuses"] = list(accepted_statuses)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="raw_mixer_state_physical_readback_contract_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_raw_mixer_state_physical_readback_artifact_intake_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "raw_mixer_state_artifact_intake_is_read_only_operator_review_gate",
    }
    violations: list[str] = []

    try:
        intake = build_raw_mixer_state_physical_readback_artifact_intake()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        intake = {}
        violations.append(f"raw_mixer_artifact_intake_build_error:{type(exc).__name__}")

    safety = intake.get("safety") if isinstance(intake.get("safety"), Mapping) else {}
    summary = intake.get("summary") if isinstance(intake.get("summary"), Mapping) else {}
    artifacts = intake.get("artifacts") if isinstance(intake.get("artifacts"), list) else []

    if intake.get("type") != "raw_mixer_state_physical_readback_artifact_intake":
        violations.append("raw_mixer_artifact_intake_type_mismatch")
    if intake.get("method_id") != RAW_MIXER_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID:
        violations.append("raw_mixer_artifact_intake_method_id_mismatch")
    if intake.get("required_contract_method_id") != RAW_MIXER_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("raw_mixer_artifact_intake_required_contract_mismatch")
    if intake.get("schema_id") != RAW_MIXER_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("raw_mixer_artifact_intake_schema_id_mismatch")
    if intake.get("status") not in INTAKE_STATUSES:
        violations.append("raw_mixer_artifact_intake_status_unknown")
    if intake.get("surface_id") != RAW_MIXER_STATE_SURFACE_ID:
        violations.append("raw_mixer_artifact_intake_surface_id_mismatch")
    if intake.get("remediation_action_id") != RAW_MIXER_STATE_REMEDIATION_ACTION_ID:
        violations.append("raw_mixer_artifact_intake_action_id_mismatch")

    expected_true = ("read_only", "operator_required")
    for key in expected_true:
        if safety.get(key) is not True:
            violations.append(f"raw_mixer_artifact_intake_{key}_not_true")
    expected_false = (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_intake",
        "fixture_closure_allowed_by_intake",
        "live_widening_allowed",
        "full_autonomous_ready",
    )
    for key in expected_false:
        if safety.get(key) is not False:
            violations.append(f"raw_mixer_artifact_intake_{key}_not_false")

    artifact_count = summary.get("artifact_count")
    valid_count = summary.get("valid_candidate_count")
    invalid_count = summary.get("invalid_candidate_count")
    if not isinstance(artifact_count, int) or artifact_count != len(artifacts):
        violations.append("raw_mixer_artifact_intake_artifact_count_mismatch")
        artifact_count = len(artifacts)
    if not isinstance(valid_count, int):
        violations.append("raw_mixer_artifact_intake_valid_count_missing")
        valid_count = 0
    if not isinstance(invalid_count, int):
        violations.append("raw_mixer_artifact_intake_invalid_count_missing")
        invalid_count = 0
    if valid_count + invalid_count != artifact_count:
        violations.append("raw_mixer_artifact_intake_candidate_count_mismatch")
    if invalid_count:
        violations.append("raw_mixer_artifact_intake_invalid_candidate_present")

    if summary.get("operator_review_required") is not True:
        violations.append("raw_mixer_artifact_intake_operator_review_not_true")
    if summary.get("physical_fixture_gap_remains") is not True:
        violations.append("raw_mixer_artifact_intake_physical_gap_not_true")
    if summary.get("fixture_closure_allowed_by_intake") is not False:
        violations.append("raw_mixer_artifact_intake_fixture_closure_not_false")
    if summary.get("live_authority_granted") is not False:
        violations.append("raw_mixer_artifact_intake_live_authority_not_false")
    if summary.get("full_autonomous_ready") is not False:
        violations.append("raw_mixer_artifact_intake_full_autonomous_not_false")
    if summary.get("accepted_as_physical_fixture_candidate") is not bool(valid_count):
        violations.append("raw_mixer_artifact_intake_candidate_flag_mismatch")

    intake_violations = (
        list(summary.get("violations")) if isinstance(summary.get("violations"), list) else []
    )
    artifact_source_ids = [
        row.get("source_id")
        for row in artifacts
        if isinstance(row, Mapping) and isinstance(row.get("source_id"), str)
    ]
    detail["intake_status"] = intake.get("status")
    detail["schema_id"] = intake.get("schema_id")
    detail["surface_id"] = intake.get("surface_id")
    detail["remediation_action_id"] = intake.get("remediation_action_id")
    detail["required_contract_method_id"] = intake.get("required_contract_method_id")
    detail["artifact_search_roots"] = (
        list(summary.get("artifact_search_roots"))
        if isinstance(summary.get("artifact_search_roots"), list)
        else []
    )
    detail["artifact_count"] = artifact_count
    detail["valid_candidate_count"] = valid_count
    detail["invalid_candidate_count"] = invalid_count
    detail["accepted_as_physical_fixture_candidate"] = summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["fixture_closure_allowed_by_intake"] = summary.get(
        "fixture_closure_allowed_by_intake"
    )
    detail["live_authority_granted"] = summary.get("live_authority_granted")
    detail["full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["latest_valid_candidate_source_id"] = summary.get("latest_valid_candidate_source_id")
    detail["latest_valid_candidate_captured_at"] = summary.get(
        "latest_valid_candidate_captured_at"
    )
    detail["blocked_by"] = (
        list(summary.get("blocked_by")) if isinstance(summary.get("blocked_by"), list) else []
    )
    detail["artifact_source_ids"] = artifact_source_ids
    detail["intake_violations"] = intake_violations
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="raw_mixer_state_physical_readback_artifact_intake_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_channel_basic_controls_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "channel_basic_controls_readback_contract_is_schema_only_no_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_channel_basic_controls_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"channel_basic_controls_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_actions = (
        remediation_plan.get("actions")
        if isinstance(remediation_plan.get("actions"), list)
        else []
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "channel_basic_controls_physical_readback_contract":
        violations.append("channel_basic_controls_readback_contract_type_mismatch")
    if contract.get("method_id") != CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("channel_basic_controls_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("channel_basic_controls_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("channel_basic_controls_readback_contract_status_mismatch")
    if contract.get("surface_id") != CHANNEL_BASIC_CONTROLS_SURFACE_ID:
        violations.append("channel_basic_controls_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID:
        violations.append("channel_basic_controls_readback_contract_action_id_mismatch")

    if safety.get("read_only") is not True:
        violations.append("channel_basic_controls_readback_contract_read_only_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"channel_basic_controls_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("channel_basic_controls_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("channel_basic_controls_readback_artifact_schema_id_mismatch")
    if artifact_contract.get("minimum_channel_count") != 1:
        violations.append("channel_basic_controls_readback_artifact_minimum_channel_count_mismatch")

    required_top_level = artifact_contract.get("required_top_level_fields")
    if (
        not isinstance(required_top_level, list)
        or required_top_level != list(CHANNEL_BASIC_CONTROLS_REQUIRED_TOP_LEVEL_FIELDS)
    ):
        violations.append("channel_basic_controls_readback_required_top_level_fields_mismatch")
        required_top_level = []
    required_channel_fields = artifact_contract.get("required_channel_fields")
    if (
        not isinstance(required_channel_fields, list)
        or required_channel_fields != list(CHANNEL_BASIC_CONTROLS_REQUIRED_CHANNEL_FIELDS)
    ):
        violations.append("channel_basic_controls_readback_required_channel_fields_mismatch")
        required_channel_fields = []
    required_identity_fields = artifact_contract.get("required_identity_fields")
    if (
        not isinstance(required_identity_fields, list)
        or required_identity_fields != list(CHANNEL_BASIC_CONTROLS_REQUIRED_IDENTITY_FIELDS)
    ):
        violations.append("channel_basic_controls_readback_required_identity_fields_mismatch")
        required_identity_fields = []
    required_control_fields = artifact_contract.get("required_control_fields")
    if (
        not isinstance(required_control_fields, list)
        or required_control_fields != list(CHANNEL_BASIC_CONTROLS_REQUIRED_CONTROL_FIELDS)
    ):
        violations.append("channel_basic_controls_readback_required_control_fields_mismatch")
        required_control_fields = []
    required_coverage_flags = artifact_contract.get("required_coverage_flags")
    if (
        not isinstance(required_coverage_flags, list)
        or required_coverage_flags != list(CHANNEL_BASIC_CONTROLS_REQUIRED_COVERAGE_FLAGS)
    ):
        violations.append("channel_basic_controls_readback_required_coverage_flags_mismatch")
        required_coverage_flags = []
    required_false_flags = artifact_contract.get("required_false_safety_flags")
    if (
        not isinstance(required_false_flags, list)
        or required_false_flags != list(CHANNEL_BASIC_CONTROLS_REQUIRED_FALSE_SAFETY_FLAGS)
    ):
        violations.append("channel_basic_controls_readback_required_false_safety_flags_mismatch")
        required_false_flags = []

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("channel_basic_controls_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SOURCES):
        violations.append("channel_basic_controls_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("channel_basic_controls_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_channel_statuses")
    if (
        not isinstance(accepted_statuses, list)
        or accepted_statuses != list(CHANNEL_BASIC_CONTROLS_VERIFIED_PHYSICAL_STATUSES)
    ):
        violations.append("channel_basic_controls_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields
        != list(CHANNEL_BASIC_CONTROLS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("channel_basic_controls_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    physical_gap_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_gap_ids, list):
        physical_gap_ids = []
        violations.append("channel_basic_controls_fixture_physical_gap_ids_missing")
    if CHANNEL_BASIC_CONTROLS_SURFACE_ID not in physical_gap_ids:
        violations.append("channel_basic_controls_fixture_gap_not_declared")

    remediation_action = {}
    for action in remediation_actions:
        if (
            isinstance(action, Mapping)
            and action.get("surface_id") == CHANNEL_BASIC_CONTROLS_SURFACE_ID
        ):
            remediation_action = dict(action)
            break
    if not remediation_action:
        violations.append("channel_basic_controls_remediation_action_missing")
    elif remediation_action.get("id") != CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID:
        violations.append("channel_basic_controls_remediation_action_id_mismatch")
    if remediation_action and remediation_action.get("physical_readback_gap") is not True:
        violations.append("channel_basic_controls_remediation_physical_gap_not_true")

    if contract_summary.get("physical_readback_surface_id") != CHANNEL_BASIC_CONTROLS_SURFACE_ID:
        violations.append("channel_basic_controls_contract_surface_summary_mismatch")
    if contract_summary.get("physical_readback_action_id") != CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID:
        violations.append("channel_basic_controls_contract_action_summary_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("channel_basic_controls_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("channel_basic_controls_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("channel_basic_controls_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("channel_basic_controls_contract_live_authority_not_false")
    promotion_families = contract_summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("channel_basic_controls_promotion_families_missing")
    for family in ("fader", "gain", "mute", "pan_phase_delay"):
        if family not in promotion_families:
            violations.append(f"channel_basic_controls_promotion_family_missing:{family}")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["physical_readback_surface_id"] = contract_summary.get("physical_readback_surface_id")
    detail["physical_readback_action_id"] = contract_summary.get("physical_readback_action_id")
    detail["fixture_physical_gap_declared"] = CHANNEL_BASIC_CONTROLS_SURFACE_ID in physical_gap_ids
    detail["remediation_physical_gap_declared"] = bool(
        remediation_action and remediation_action.get("physical_readback_gap") is True
    )
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["required_top_level_field_count"] = len(required_top_level)
    detail["required_channel_field_count"] = len(required_channel_fields)
    detail["required_identity_fields"] = list(required_identity_fields)
    detail["required_control_fields"] = list(required_control_fields)
    detail["required_coverage_flags"] = list(required_coverage_flags)
    detail["required_false_safety_flags"] = list(required_false_flags)
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_channel_statuses"] = list(accepted_statuses)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="channel_basic_controls_physical_readback_contract_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_channel_basic_controls_physical_readback_artifact_intake_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "channel_basic_controls_artifact_intake_is_read_only_operator_review_gate",
    }
    violations: list[str] = []

    try:
        intake = build_channel_basic_controls_physical_readback_artifact_intake()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        intake = {}
        violations.append(f"channel_basic_artifact_intake_build_error:{type(exc).__name__}")

    safety = intake.get("safety") if isinstance(intake.get("safety"), Mapping) else {}
    summary = intake.get("summary") if isinstance(intake.get("summary"), Mapping) else {}
    artifacts = intake.get("artifacts") if isinstance(intake.get("artifacts"), list) else []

    if intake.get("type") != "channel_basic_controls_physical_readback_artifact_intake":
        violations.append("channel_basic_artifact_intake_type_mismatch")
    if intake.get("method_id") != CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID:
        violations.append("channel_basic_artifact_intake_method_id_mismatch")
    if intake.get("required_contract_method_id") != CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("channel_basic_artifact_intake_required_contract_mismatch")
    if intake.get("schema_id") != CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("channel_basic_artifact_intake_schema_id_mismatch")
    if intake.get("status") not in CHANNEL_BASIC_CONTROLS_INTAKE_STATUSES:
        violations.append("channel_basic_artifact_intake_status_unknown")
    if intake.get("surface_id") != CHANNEL_BASIC_CONTROLS_SURFACE_ID:
        violations.append("channel_basic_artifact_intake_surface_id_mismatch")
    if intake.get("remediation_action_id") != CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID:
        violations.append("channel_basic_artifact_intake_action_id_mismatch")

    for key in ("read_only", "operator_required"):
        if safety.get(key) is not True:
            violations.append(f"channel_basic_artifact_intake_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_intake",
        "fixture_closure_allowed_by_intake",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"channel_basic_artifact_intake_{key}_not_false")

    artifact_count = summary.get("artifact_count")
    valid_count = summary.get("valid_candidate_count")
    invalid_count = summary.get("invalid_candidate_count")
    valid_channel_count = summary.get("valid_channel_count")
    if not isinstance(artifact_count, int) or artifact_count != len(artifacts):
        violations.append("channel_basic_artifact_intake_artifact_count_mismatch")
        artifact_count = len(artifacts)
    if not isinstance(valid_count, int):
        violations.append("channel_basic_artifact_intake_valid_count_missing")
        valid_count = 0
    if not isinstance(invalid_count, int):
        violations.append("channel_basic_artifact_intake_invalid_count_missing")
        invalid_count = 0
    if not isinstance(valid_channel_count, int):
        violations.append("channel_basic_artifact_intake_valid_channel_count_missing")
        valid_channel_count = 0
    if valid_count + invalid_count != artifact_count:
        violations.append("channel_basic_artifact_intake_candidate_count_mismatch")
    if invalid_count:
        violations.append("channel_basic_artifact_intake_invalid_candidate_present")

    if summary.get("operator_review_required") is not True:
        violations.append("channel_basic_artifact_intake_operator_review_not_true")
    if summary.get("physical_fixture_gap_remains") is not True:
        violations.append("channel_basic_artifact_intake_physical_gap_not_true")
    if summary.get("fixture_closure_allowed_by_intake") is not False:
        violations.append("channel_basic_artifact_intake_fixture_closure_not_false")
    if summary.get("live_authority_granted") is not False:
        violations.append("channel_basic_artifact_intake_live_authority_not_false")
    if summary.get("full_autonomous_ready") is not False:
        violations.append("channel_basic_artifact_intake_full_autonomous_not_false")
    if summary.get("accepted_as_physical_fixture_candidate") is not bool(valid_count):
        violations.append("channel_basic_artifact_intake_candidate_flag_mismatch")

    promotion_families = summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("channel_basic_artifact_intake_promotion_families_missing")
    for family in ("fader", "gain", "mute", "pan_phase_delay"):
        if family not in promotion_families:
            violations.append(f"channel_basic_artifact_intake_promotion_family_missing:{family}")

    intake_violations = (
        list(summary.get("violations")) if isinstance(summary.get("violations"), list) else []
    )
    artifact_source_ids = [
        row.get("source_id")
        for row in artifacts
        if isinstance(row, Mapping) and isinstance(row.get("source_id"), str)
    ]
    detail["intake_status"] = intake.get("status")
    detail["schema_id"] = intake.get("schema_id")
    detail["surface_id"] = intake.get("surface_id")
    detail["remediation_action_id"] = intake.get("remediation_action_id")
    detail["required_contract_method_id"] = intake.get("required_contract_method_id")
    detail["artifact_search_roots"] = (
        list(summary.get("artifact_search_roots"))
        if isinstance(summary.get("artifact_search_roots"), list)
        else []
    )
    detail["artifact_count"] = artifact_count
    detail["valid_candidate_count"] = valid_count
    detail["invalid_candidate_count"] = invalid_count
    detail["valid_channel_count"] = valid_channel_count
    detail["accepted_as_physical_fixture_candidate"] = summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["fixture_closure_allowed_by_intake"] = summary.get(
        "fixture_closure_allowed_by_intake"
    )
    detail["live_authority_granted"] = summary.get("live_authority_granted")
    detail["full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["latest_valid_candidate_source_id"] = summary.get("latest_valid_candidate_source_id")
    detail["latest_valid_candidate_captured_at"] = summary.get(
        "latest_valid_candidate_captured_at"
    )
    detail["blocked_by"] = (
        list(summary.get("blocked_by")) if isinstance(summary.get("blocked_by"), list) else []
    )
    detail["artifact_source_ids"] = artifact_source_ids
    detail["intake_violations"] = intake_violations
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="channel_basic_controls_physical_readback_artifact_intake_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_channel_processing_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "channel_processing_readback_contract_is_schema_only_no_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_channel_processing_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"channel_processing_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_actions = (
        remediation_plan.get("actions")
        if isinstance(remediation_plan.get("actions"), list)
        else []
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "channel_processing_physical_readback_contract":
        violations.append("channel_processing_readback_contract_type_mismatch")
    if contract.get("method_id") != CHANNEL_PROCESSING_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("channel_processing_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != CHANNEL_PROCESSING_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("channel_processing_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("channel_processing_readback_contract_status_mismatch")
    if contract.get("surface_id") != CHANNEL_PROCESSING_SURFACE_ID:
        violations.append("channel_processing_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != CHANNEL_PROCESSING_REMEDIATION_ACTION_ID:
        violations.append("channel_processing_readback_contract_action_id_mismatch")

    if safety.get("read_only") is not True:
        violations.append("channel_processing_readback_contract_read_only_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"channel_processing_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("channel_processing_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != CHANNEL_PROCESSING_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("channel_processing_readback_artifact_schema_id_mismatch")
    if artifact_contract.get("minimum_channel_count") != 1:
        violations.append("channel_processing_readback_artifact_minimum_channel_count_mismatch")
    if artifact_contract.get("minimum_eq_band_count") != MINIMUM_EQ_BAND_COUNT:
        violations.append("channel_processing_readback_artifact_minimum_eq_band_count_mismatch")

    required_top_level = artifact_contract.get("required_top_level_fields")
    if (
        not isinstance(required_top_level, list)
        or required_top_level != list(CHANNEL_PROCESSING_REQUIRED_TOP_LEVEL_FIELDS)
    ):
        violations.append("channel_processing_readback_required_top_level_fields_mismatch")
        required_top_level = []
    required_channel_fields = artifact_contract.get("required_channel_fields")
    if (
        not isinstance(required_channel_fields, list)
        or required_channel_fields != list(CHANNEL_PROCESSING_REQUIRED_CHANNEL_FIELDS)
    ):
        violations.append("channel_processing_readback_required_channel_fields_mismatch")
        required_channel_fields = []
    required_hpf_fields = artifact_contract.get("required_hpf_fields")
    if (
        not isinstance(required_hpf_fields, list)
        or required_hpf_fields != list(CHANNEL_PROCESSING_REQUIRED_HPF_FIELDS)
    ):
        violations.append("channel_processing_readback_required_hpf_fields_mismatch")
        required_hpf_fields = []
    required_eq_fields = artifact_contract.get("required_eq_fields")
    if (
        not isinstance(required_eq_fields, list)
        or required_eq_fields != list(CHANNEL_PROCESSING_REQUIRED_EQ_FIELDS)
    ):
        violations.append("channel_processing_readback_required_eq_fields_mismatch")
        required_eq_fields = []
    required_eq_band_fields = artifact_contract.get("required_eq_band_fields")
    if (
        not isinstance(required_eq_band_fields, list)
        or required_eq_band_fields != list(CHANNEL_PROCESSING_REQUIRED_EQ_BAND_FIELDS)
    ):
        violations.append("channel_processing_readback_required_eq_band_fields_mismatch")
        required_eq_band_fields = []
    required_compressor_fields = artifact_contract.get("required_compressor_fields")
    if (
        not isinstance(required_compressor_fields, list)
        or required_compressor_fields != list(CHANNEL_PROCESSING_REQUIRED_COMPRESSOR_FIELDS)
    ):
        violations.append("channel_processing_readback_required_compressor_fields_mismatch")
        required_compressor_fields = []
    required_gate_fields = artifact_contract.get("required_gate_fields")
    if (
        not isinstance(required_gate_fields, list)
        or required_gate_fields != list(CHANNEL_PROCESSING_REQUIRED_GATE_FIELDS)
    ):
        violations.append("channel_processing_readback_required_gate_fields_mismatch")
        required_gate_fields = []
    required_coverage_flags = artifact_contract.get("required_coverage_flags")
    if (
        not isinstance(required_coverage_flags, list)
        or required_coverage_flags != list(CHANNEL_PROCESSING_REQUIRED_COVERAGE_FLAGS)
    ):
        violations.append("channel_processing_readback_required_coverage_flags_mismatch")
        required_coverage_flags = []
    required_false_flags = artifact_contract.get("required_false_safety_flags")
    if (
        not isinstance(required_false_flags, list)
        or required_false_flags != list(CHANNEL_PROCESSING_REQUIRED_FALSE_SAFETY_FLAGS)
    ):
        violations.append("channel_processing_readback_required_false_safety_flags_mismatch")
        required_false_flags = []

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("channel_processing_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(CHANNEL_PROCESSING_PHYSICAL_READBACK_SOURCES):
        violations.append("channel_processing_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("channel_processing_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_channel_statuses")
    if (
        not isinstance(accepted_statuses, list)
        or accepted_statuses != list(CHANNEL_PROCESSING_VERIFIED_PHYSICAL_STATUSES)
    ):
        violations.append("channel_processing_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields != list(CHANNEL_PROCESSING_FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("channel_processing_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    physical_gap_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_gap_ids, list):
        physical_gap_ids = []
        violations.append("channel_processing_fixture_physical_gap_ids_missing")
    if CHANNEL_PROCESSING_SURFACE_ID not in physical_gap_ids:
        violations.append("channel_processing_fixture_gap_not_declared")

    remediation_action = {}
    for action in remediation_actions:
        if isinstance(action, Mapping) and action.get("surface_id") == CHANNEL_PROCESSING_SURFACE_ID:
            remediation_action = dict(action)
            break
    if not remediation_action:
        violations.append("channel_processing_remediation_action_missing")
    elif remediation_action.get("id") != CHANNEL_PROCESSING_REMEDIATION_ACTION_ID:
        violations.append("channel_processing_remediation_action_id_mismatch")
    if remediation_action and remediation_action.get("physical_readback_gap") is not True:
        violations.append("channel_processing_remediation_physical_gap_not_true")

    if contract_summary.get("physical_readback_surface_id") != CHANNEL_PROCESSING_SURFACE_ID:
        violations.append("channel_processing_contract_surface_summary_mismatch")
    if contract_summary.get("physical_readback_action_id") != CHANNEL_PROCESSING_REMEDIATION_ACTION_ID:
        violations.append("channel_processing_contract_action_summary_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("channel_processing_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("channel_processing_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("channel_processing_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("channel_processing_contract_live_authority_not_false")
    promotion_families = contract_summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("channel_processing_promotion_families_missing")
    for family in ("eq", "compressor", "gate", "deessing"):
        if family not in promotion_families:
            violations.append(f"channel_processing_promotion_family_missing:{family}")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["physical_readback_surface_id"] = contract_summary.get("physical_readback_surface_id")
    detail["physical_readback_action_id"] = contract_summary.get("physical_readback_action_id")
    detail["fixture_physical_gap_declared"] = CHANNEL_PROCESSING_SURFACE_ID in physical_gap_ids
    detail["remediation_physical_gap_declared"] = bool(
        remediation_action and remediation_action.get("physical_readback_gap") is True
    )
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["required_top_level_field_count"] = len(required_top_level)
    detail["required_channel_field_count"] = len(required_channel_fields)
    detail["required_hpf_fields"] = list(required_hpf_fields)
    detail["required_eq_fields"] = list(required_eq_fields)
    detail["required_eq_band_fields"] = list(required_eq_band_fields)
    detail["required_compressor_fields"] = list(required_compressor_fields)
    detail["required_gate_fields"] = list(required_gate_fields)
    detail["required_coverage_flags"] = list(required_coverage_flags)
    detail["required_false_safety_flags"] = list(required_false_flags)
    detail["minimum_eq_band_count"] = int(artifact_contract.get("minimum_eq_band_count") or 0)
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_channel_statuses"] = list(accepted_statuses)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="channel_processing_physical_readback_contract_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_channel_processing_physical_readback_artifact_intake_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "channel_processing_artifact_intake_is_read_only_operator_review_gate",
    }
    violations: list[str] = []

    try:
        intake = build_channel_processing_physical_readback_artifact_intake()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        intake = {}
        violations.append(f"channel_processing_artifact_intake_build_error:{type(exc).__name__}")

    safety = intake.get("safety") if isinstance(intake.get("safety"), Mapping) else {}
    summary = intake.get("summary") if isinstance(intake.get("summary"), Mapping) else {}
    artifacts = intake.get("artifacts") if isinstance(intake.get("artifacts"), list) else []

    if intake.get("type") != "channel_processing_physical_readback_artifact_intake":
        violations.append("channel_processing_artifact_intake_type_mismatch")
    if intake.get("method_id") != CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID:
        violations.append("channel_processing_artifact_intake_method_id_mismatch")
    if intake.get("required_contract_method_id") != CHANNEL_PROCESSING_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("channel_processing_artifact_intake_required_contract_mismatch")
    if intake.get("schema_id") != CHANNEL_PROCESSING_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("channel_processing_artifact_intake_schema_id_mismatch")
    if intake.get("status") not in CHANNEL_PROCESSING_INTAKE_STATUSES:
        violations.append("channel_processing_artifact_intake_status_unknown")
    if intake.get("surface_id") != CHANNEL_PROCESSING_SURFACE_ID:
        violations.append("channel_processing_artifact_intake_surface_id_mismatch")
    if intake.get("remediation_action_id") != CHANNEL_PROCESSING_REMEDIATION_ACTION_ID:
        violations.append("channel_processing_artifact_intake_action_id_mismatch")

    for key in ("read_only", "operator_required"):
        if safety.get(key) is not True:
            violations.append(f"channel_processing_artifact_intake_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_intake",
        "fixture_closure_allowed_by_intake",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"channel_processing_artifact_intake_{key}_not_false")

    artifact_count = summary.get("artifact_count")
    valid_count = summary.get("valid_candidate_count")
    invalid_count = summary.get("invalid_candidate_count")
    valid_channel_count = summary.get("valid_channel_count")
    if not isinstance(artifact_count, int) or artifact_count != len(artifacts):
        violations.append("channel_processing_artifact_intake_artifact_count_mismatch")
        artifact_count = len(artifacts)
    if not isinstance(valid_count, int):
        violations.append("channel_processing_artifact_intake_valid_count_missing")
        valid_count = 0
    if not isinstance(invalid_count, int):
        violations.append("channel_processing_artifact_intake_invalid_count_missing")
        invalid_count = 0
    if not isinstance(valid_channel_count, int):
        violations.append("channel_processing_artifact_intake_valid_channel_count_missing")
        valid_channel_count = 0
    if valid_count + invalid_count != artifact_count:
        violations.append("channel_processing_artifact_intake_candidate_count_mismatch")
    if invalid_count:
        violations.append("channel_processing_artifact_intake_invalid_candidate_present")

    if summary.get("operator_review_required") is not True:
        violations.append("channel_processing_artifact_intake_operator_review_not_true")
    if summary.get("physical_fixture_gap_remains") is not True:
        violations.append("channel_processing_artifact_intake_physical_gap_not_true")
    if summary.get("fixture_closure_allowed_by_intake") is not False:
        violations.append("channel_processing_artifact_intake_fixture_closure_not_false")
    if summary.get("live_authority_granted") is not False:
        violations.append("channel_processing_artifact_intake_live_authority_not_false")
    if summary.get("full_autonomous_ready") is not False:
        violations.append("channel_processing_artifact_intake_full_autonomous_not_false")
    if summary.get("accepted_as_physical_fixture_candidate") is not bool(valid_count):
        violations.append("channel_processing_artifact_intake_candidate_flag_mismatch")

    promotion_families = summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("channel_processing_artifact_intake_promotion_families_missing")
    for family in (
        "eq",
        "compressor",
        "gate",
        "deessing",
        "frequency_separation",
        "cross_channel_unmasking",
    ):
        if family not in promotion_families:
            violations.append(f"channel_processing_artifact_intake_promotion_family_missing:{family}")

    intake_violations = (
        list(summary.get("violations")) if isinstance(summary.get("violations"), list) else []
    )
    artifact_source_ids = [
        row.get("source_id")
        for row in artifacts
        if isinstance(row, Mapping) and isinstance(row.get("source_id"), str)
    ]
    detail["intake_status"] = intake.get("status")
    detail["schema_id"] = intake.get("schema_id")
    detail["surface_id"] = intake.get("surface_id")
    detail["remediation_action_id"] = intake.get("remediation_action_id")
    detail["required_contract_method_id"] = intake.get("required_contract_method_id")
    detail["artifact_search_roots"] = (
        list(summary.get("artifact_search_roots"))
        if isinstance(summary.get("artifact_search_roots"), list)
        else []
    )
    detail["artifact_count"] = artifact_count
    detail["valid_candidate_count"] = valid_count
    detail["invalid_candidate_count"] = invalid_count
    detail["valid_channel_count"] = valid_channel_count
    detail["accepted_as_physical_fixture_candidate"] = summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["fixture_closure_allowed_by_intake"] = summary.get(
        "fixture_closure_allowed_by_intake"
    )
    detail["live_authority_granted"] = summary.get("live_authority_granted")
    detail["full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["latest_valid_candidate_source_id"] = summary.get("latest_valid_candidate_source_id")
    detail["latest_valid_candidate_captured_at"] = summary.get(
        "latest_valid_candidate_captured_at"
    )
    detail["blocked_by"] = (
        list(summary.get("blocked_by")) if isinstance(summary.get("blocked_by"), list) else []
    )
    detail["artifact_source_ids"] = artifact_source_ids
    detail["intake_violations"] = intake_violations
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="channel_processing_physical_readback_artifact_intake_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_bus_group_main_dca_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "bus_group_main_dca_readback_contract_is_schema_only_no_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_bus_group_main_dca_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"bus_group_main_dca_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_actions = (
        remediation_plan.get("actions")
        if isinstance(remediation_plan.get("actions"), list)
        else []
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "bus_group_main_dca_physical_readback_contract":
        violations.append("bus_group_main_dca_readback_contract_type_mismatch")
    if contract.get("method_id") != BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("bus_group_main_dca_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("bus_group_main_dca_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("bus_group_main_dca_readback_contract_status_mismatch")
    if contract.get("surface_id") != BUS_GROUP_MAIN_DCA_SURFACE_ID:
        violations.append("bus_group_main_dca_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != BUS_GROUP_MAIN_DCA_REMEDIATION_ACTION_ID:
        violations.append("bus_group_main_dca_readback_contract_action_id_mismatch")

    if safety.get("read_only") is not True:
        violations.append("bus_group_main_dca_readback_contract_read_only_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"bus_group_main_dca_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("bus_group_main_dca_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("bus_group_main_dca_readback_artifact_schema_id_mismatch")
    if artifact_contract.get("minimum_bus_count") != 1:
        violations.append("bus_group_main_dca_readback_artifact_minimum_bus_count_mismatch")
    if artifact_contract.get("minimum_main_count") != 1:
        violations.append("bus_group_main_dca_readback_artifact_minimum_main_count_mismatch")
    if artifact_contract.get("minimum_dca_count") != 1:
        violations.append("bus_group_main_dca_readback_artifact_minimum_dca_count_mismatch")

    required_top_level = artifact_contract.get("required_top_level_fields")
    if (
        not isinstance(required_top_level, list)
        or required_top_level != list(BUS_GROUP_MAIN_DCA_REQUIRED_TOP_LEVEL_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_top_level_fields_mismatch")
        required_top_level = []
    required_bus_fields = artifact_contract.get("required_bus_fields")
    if (
        not isinstance(required_bus_fields, list)
        or required_bus_fields != list(BUS_GROUP_MAIN_DCA_REQUIRED_BUS_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_bus_fields_mismatch")
        required_bus_fields = []
    required_main_fields = artifact_contract.get("required_main_fields")
    if (
        not isinstance(required_main_fields, list)
        or required_main_fields != list(BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_main_fields_mismatch")
        required_main_fields = []
    required_dca_fields = artifact_contract.get("required_dca_fields")
    if (
        not isinstance(required_dca_fields, list)
        or required_dca_fields != list(BUS_GROUP_MAIN_DCA_REQUIRED_DCA_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_dca_fields_mismatch")
        required_dca_fields = []
    required_bus_control_fields = artifact_contract.get("required_bus_control_fields")
    if (
        not isinstance(required_bus_control_fields, list)
        or required_bus_control_fields != list(BUS_GROUP_MAIN_DCA_REQUIRED_BUS_CONTROL_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_bus_control_fields_mismatch")
        required_bus_control_fields = []
    required_main_control_fields = artifact_contract.get("required_main_control_fields")
    if (
        not isinstance(required_main_control_fields, list)
        or required_main_control_fields != list(BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_CONTROL_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_main_control_fields_mismatch")
        required_main_control_fields = []
    required_dca_control_fields = artifact_contract.get("required_dca_control_fields")
    if (
        not isinstance(required_dca_control_fields, list)
        or required_dca_control_fields != list(BUS_GROUP_MAIN_DCA_REQUIRED_DCA_CONTROL_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_dca_control_fields_mismatch")
        required_dca_control_fields = []
    required_bus_assignment_fields = artifact_contract.get("required_bus_assignment_fields")
    if (
        not isinstance(required_bus_assignment_fields, list)
        or required_bus_assignment_fields != list(BUS_GROUP_MAIN_DCA_REQUIRED_BUS_ASSIGNMENT_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_bus_assignment_fields_mismatch")
        required_bus_assignment_fields = []
    required_dca_assignment_fields = artifact_contract.get("required_dca_assignment_fields")
    if (
        not isinstance(required_dca_assignment_fields, list)
        or required_dca_assignment_fields != list(BUS_GROUP_MAIN_DCA_REQUIRED_DCA_ASSIGNMENT_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_dca_assignment_fields_mismatch")
        required_dca_assignment_fields = []
    required_metering_fields = artifact_contract.get("required_metering_fields")
    if (
        not isinstance(required_metering_fields, list)
        or required_metering_fields != list(BUS_GROUP_MAIN_DCA_REQUIRED_METERING_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_metering_fields_mismatch")
        required_metering_fields = []
    required_bus_coverage_flags = artifact_contract.get("required_bus_coverage_flags")
    if (
        not isinstance(required_bus_coverage_flags, list)
        or required_bus_coverage_flags != list(BUS_GROUP_MAIN_DCA_REQUIRED_BUS_COVERAGE_FLAGS)
    ):
        violations.append("bus_group_main_dca_readback_required_bus_coverage_flags_mismatch")
        required_bus_coverage_flags = []
    required_main_coverage_flags = artifact_contract.get("required_main_coverage_flags")
    if (
        not isinstance(required_main_coverage_flags, list)
        or required_main_coverage_flags != list(BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_COVERAGE_FLAGS)
    ):
        violations.append("bus_group_main_dca_readback_required_main_coverage_flags_mismatch")
        required_main_coverage_flags = []
    required_dca_coverage_flags = artifact_contract.get("required_dca_coverage_flags")
    if (
        not isinstance(required_dca_coverage_flags, list)
        or required_dca_coverage_flags != list(BUS_GROUP_MAIN_DCA_REQUIRED_DCA_COVERAGE_FLAGS)
    ):
        violations.append("bus_group_main_dca_readback_required_dca_coverage_flags_mismatch")
        required_dca_coverage_flags = []
    required_false_flags = artifact_contract.get("required_false_safety_flags")
    if (
        not isinstance(required_false_flags, list)
        or required_false_flags != list(BUS_GROUP_MAIN_DCA_REQUIRED_FALSE_SAFETY_FLAGS)
    ):
        violations.append("bus_group_main_dca_readback_required_false_safety_flags_mismatch")
        required_false_flags = []

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("bus_group_main_dca_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SOURCES):
        violations.append("bus_group_main_dca_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("bus_group_main_dca_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_surface_statuses")
    if (
        not isinstance(accepted_statuses, list)
        or accepted_statuses != list(BUS_GROUP_MAIN_DCA_VERIFIED_PHYSICAL_STATUSES)
    ):
        violations.append("bus_group_main_dca_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields
        != list(BUS_GROUP_MAIN_DCA_FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    physical_gap_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_gap_ids, list):
        physical_gap_ids = []
        violations.append("bus_group_main_dca_fixture_physical_gap_ids_missing")
    if BUS_GROUP_MAIN_DCA_SURFACE_ID not in physical_gap_ids:
        violations.append("bus_group_main_dca_fixture_gap_not_declared")

    remediation_action = {}
    for action in remediation_actions:
        if isinstance(action, Mapping) and action.get("surface_id") == BUS_GROUP_MAIN_DCA_SURFACE_ID:
            remediation_action = dict(action)
            break
    if not remediation_action:
        violations.append("bus_group_main_dca_remediation_action_missing")
    elif remediation_action.get("id") != BUS_GROUP_MAIN_DCA_REMEDIATION_ACTION_ID:
        violations.append("bus_group_main_dca_remediation_action_id_mismatch")
    if remediation_action and remediation_action.get("physical_readback_gap") is not True:
        violations.append("bus_group_main_dca_remediation_physical_gap_not_true")

    if contract_summary.get("physical_readback_surface_id") != BUS_GROUP_MAIN_DCA_SURFACE_ID:
        violations.append("bus_group_main_dca_contract_surface_summary_mismatch")
    if contract_summary.get("physical_readback_action_id") != BUS_GROUP_MAIN_DCA_REMEDIATION_ACTION_ID:
        violations.append("bus_group_main_dca_contract_action_summary_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("bus_group_main_dca_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("bus_group_main_dca_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("bus_group_main_dca_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("bus_group_main_dca_contract_live_authority_not_false")
    promotion_families = contract_summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("bus_group_main_dca_promotion_families_missing")
    for family in ("bus_group_matrix", "dca", "master_bus"):
        if family not in promotion_families:
            violations.append(f"bus_group_main_dca_promotion_family_missing:{family}")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["physical_readback_surface_id"] = contract_summary.get("physical_readback_surface_id")
    detail["physical_readback_action_id"] = contract_summary.get("physical_readback_action_id")
    detail["fixture_physical_gap_declared"] = BUS_GROUP_MAIN_DCA_SURFACE_ID in physical_gap_ids
    detail["remediation_physical_gap_declared"] = bool(
        remediation_action and remediation_action.get("physical_readback_gap") is True
    )
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["required_top_level_field_count"] = len(required_top_level)
    detail["required_bus_field_count"] = len(required_bus_fields)
    detail["required_main_field_count"] = len(required_main_fields)
    detail["required_dca_field_count"] = len(required_dca_fields)
    detail["required_bus_control_fields"] = list(required_bus_control_fields)
    detail["required_main_control_fields"] = list(required_main_control_fields)
    detail["required_dca_control_fields"] = list(required_dca_control_fields)
    detail["required_bus_assignment_fields"] = list(required_bus_assignment_fields)
    detail["required_dca_assignment_fields"] = list(required_dca_assignment_fields)
    detail["required_metering_fields"] = list(required_metering_fields)
    detail["required_bus_coverage_flags"] = list(required_bus_coverage_flags)
    detail["required_main_coverage_flags"] = list(required_main_coverage_flags)
    detail["required_dca_coverage_flags"] = list(required_dca_coverage_flags)
    detail["required_false_safety_flags"] = list(required_false_flags)
    detail["minimum_bus_count"] = int(artifact_contract.get("minimum_bus_count") or 0)
    detail["minimum_main_count"] = int(artifact_contract.get("minimum_main_count") or 0)
    detail["minimum_dca_count"] = int(artifact_contract.get("minimum_dca_count") or 0)
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_surface_statuses"] = list(accepted_statuses)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="bus_group_main_dca_physical_readback_contract_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_bus_group_main_dca_physical_readback_artifact_intake_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "bus_group_main_dca_artifact_intake_is_read_only_operator_review_gate",
    }
    violations: list[str] = []

    try:
        intake = build_bus_group_main_dca_physical_readback_artifact_intake()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        intake = {}
        violations.append(f"bus_group_main_dca_artifact_intake_build_error:{type(exc).__name__}")

    safety = intake.get("safety") if isinstance(intake.get("safety"), Mapping) else {}
    summary = intake.get("summary") if isinstance(intake.get("summary"), Mapping) else {}
    artifacts = intake.get("artifacts") if isinstance(intake.get("artifacts"), list) else []

    if intake.get("type") != "bus_group_main_dca_physical_readback_artifact_intake":
        violations.append("bus_group_main_dca_artifact_intake_type_mismatch")
    if intake.get("method_id") != BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID:
        violations.append("bus_group_main_dca_artifact_intake_method_id_mismatch")
    if intake.get("required_contract_method_id") != BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("bus_group_main_dca_artifact_intake_required_contract_mismatch")
    if intake.get("schema_id") != BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("bus_group_main_dca_artifact_intake_schema_id_mismatch")
    if intake.get("status") not in BUS_GROUP_MAIN_DCA_INTAKE_STATUSES:
        violations.append("bus_group_main_dca_artifact_intake_status_unknown")
    if intake.get("surface_id") != BUS_GROUP_MAIN_DCA_SURFACE_ID:
        violations.append("bus_group_main_dca_artifact_intake_surface_id_mismatch")
    if intake.get("remediation_action_id") != BUS_GROUP_MAIN_DCA_REMEDIATION_ACTION_ID:
        violations.append("bus_group_main_dca_artifact_intake_action_id_mismatch")

    for key in ("read_only", "operator_required"):
        if safety.get(key) is not True:
            violations.append(f"bus_group_main_dca_artifact_intake_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_intake",
        "fixture_closure_allowed_by_intake",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"bus_group_main_dca_artifact_intake_{key}_not_false")

    artifact_count = summary.get("artifact_count")
    valid_count = summary.get("valid_candidate_count")
    invalid_count = summary.get("invalid_candidate_count")
    valid_bus_count = summary.get("valid_bus_count")
    valid_main_count = summary.get("valid_main_count")
    valid_dca_count = summary.get("valid_dca_count")
    if not isinstance(artifact_count, int) or artifact_count != len(artifacts):
        violations.append("bus_group_main_dca_artifact_intake_artifact_count_mismatch")
        artifact_count = len(artifacts)
    if not isinstance(valid_count, int):
        violations.append("bus_group_main_dca_artifact_intake_valid_count_missing")
        valid_count = 0
    if not isinstance(invalid_count, int):
        violations.append("bus_group_main_dca_artifact_intake_invalid_count_missing")
        invalid_count = 0
    if not isinstance(valid_bus_count, int):
        violations.append("bus_group_main_dca_artifact_intake_valid_bus_count_missing")
        valid_bus_count = 0
    if not isinstance(valid_main_count, int):
        violations.append("bus_group_main_dca_artifact_intake_valid_main_count_missing")
        valid_main_count = 0
    if not isinstance(valid_dca_count, int):
        violations.append("bus_group_main_dca_artifact_intake_valid_dca_count_missing")
        valid_dca_count = 0
    if valid_count + invalid_count != artifact_count:
        violations.append("bus_group_main_dca_artifact_intake_candidate_count_mismatch")
    if invalid_count:
        violations.append("bus_group_main_dca_artifact_intake_invalid_candidate_present")

    if summary.get("operator_review_required") is not True:
        violations.append("bus_group_main_dca_artifact_intake_operator_review_not_true")
    if summary.get("physical_fixture_gap_remains") is not True:
        violations.append("bus_group_main_dca_artifact_intake_physical_gap_not_true")
    if summary.get("fixture_closure_allowed_by_intake") is not False:
        violations.append("bus_group_main_dca_artifact_intake_fixture_closure_not_false")
    if summary.get("live_authority_granted") is not False:
        violations.append("bus_group_main_dca_artifact_intake_live_authority_not_false")
    if summary.get("full_autonomous_ready") is not False:
        violations.append("bus_group_main_dca_artifact_intake_full_autonomous_not_false")
    if summary.get("accepted_as_physical_fixture_candidate") is not bool(valid_count):
        violations.append("bus_group_main_dca_artifact_intake_candidate_flag_mismatch")

    promotion_families = summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("bus_group_main_dca_artifact_intake_promotion_families_missing")
    for family in (
        "bus_group_matrix",
        "dca",
        "master_bus",
        "mix_balance",
        "performance_adaptation",
    ):
        if family not in promotion_families:
            violations.append(f"bus_group_main_dca_artifact_intake_promotion_family_missing:{family}")

    intake_violations = (
        list(summary.get("violations")) if isinstance(summary.get("violations"), list) else []
    )
    artifact_source_ids = [
        row.get("source_id")
        for row in artifacts
        if isinstance(row, Mapping) and isinstance(row.get("source_id"), str)
    ]
    detail["intake_status"] = intake.get("status")
    detail["schema_id"] = intake.get("schema_id")
    detail["surface_id"] = intake.get("surface_id")
    detail["remediation_action_id"] = intake.get("remediation_action_id")
    detail["required_contract_method_id"] = intake.get("required_contract_method_id")
    detail["artifact_search_roots"] = (
        list(summary.get("artifact_search_roots"))
        if isinstance(summary.get("artifact_search_roots"), list)
        else []
    )
    detail["artifact_count"] = artifact_count
    detail["valid_candidate_count"] = valid_count
    detail["invalid_candidate_count"] = invalid_count
    detail["valid_bus_count"] = valid_bus_count
    detail["valid_main_count"] = valid_main_count
    detail["valid_dca_count"] = valid_dca_count
    detail["accepted_as_physical_fixture_candidate"] = summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["fixture_closure_allowed_by_intake"] = summary.get(
        "fixture_closure_allowed_by_intake"
    )
    detail["live_authority_granted"] = summary.get("live_authority_granted")
    detail["full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["latest_valid_candidate_source_id"] = summary.get("latest_valid_candidate_source_id")
    detail["latest_valid_candidate_captured_at"] = summary.get(
        "latest_valid_candidate_captured_at"
    )
    detail["blocked_by"] = (
        list(summary.get("blocked_by")) if isinstance(summary.get("blocked_by"), list) else []
    )
    detail["artifact_source_ids"] = artifact_source_ids
    detail["intake_violations"] = intake_violations
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="bus_group_main_dca_physical_readback_artifact_intake_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_matrix_state_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "matrix_state_readback_contract_is_schema_only_no_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_matrix_state_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"matrix_state_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_actions = (
        remediation_plan.get("actions")
        if isinstance(remediation_plan.get("actions"), list)
        else []
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "matrix_state_physical_readback_contract":
        violations.append("matrix_state_readback_contract_type_mismatch")
    if contract.get("method_id") != MATRIX_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("matrix_state_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != MATRIX_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("matrix_state_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("matrix_state_readback_contract_status_mismatch")
    if contract.get("surface_id") != MATRIX_STATE_SURFACE_ID:
        violations.append("matrix_state_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != MATRIX_STATE_REMEDIATION_ACTION_ID:
        violations.append("matrix_state_readback_contract_action_id_mismatch")

    if safety.get("read_only") is not True:
        violations.append("matrix_state_readback_contract_read_only_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"matrix_state_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("matrix_state_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != MATRIX_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("matrix_state_readback_artifact_schema_id_mismatch")
    if artifact_contract.get("minimum_matrix_count") != 1:
        violations.append("matrix_state_readback_artifact_minimum_matrix_count_mismatch")
    if artifact_contract.get("minimum_eq_band_count") != MINIMUM_MATRIX_EQ_BAND_COUNT:
        violations.append("matrix_state_readback_artifact_minimum_eq_band_count_mismatch")

    expected_lists = {
        "required_top_level_fields": MATRIX_STATE_REQUIRED_TOP_LEVEL_FIELDS,
        "required_matrix_fields": MATRIX_STATE_REQUIRED_MATRIX_FIELDS,
        "required_control_fields": MATRIX_STATE_REQUIRED_CONTROL_FIELDS,
        "required_eq_fields": MATRIX_STATE_REQUIRED_EQ_FIELDS,
        "required_eq_band_fields": MATRIX_STATE_REQUIRED_EQ_BAND_FIELDS,
        "required_dynamics_fields": MATRIX_STATE_REQUIRED_DYNAMICS_FIELDS,
        "required_insert_fields": MATRIX_STATE_REQUIRED_INSERT_FIELDS,
        "required_metering_fields": MATRIX_STATE_REQUIRED_METERING_FIELDS,
        "required_coverage_flags": MATRIX_STATE_REQUIRED_COVERAGE_FLAGS,
        "required_false_safety_flags": MATRIX_STATE_REQUIRED_FALSE_SAFETY_FLAGS,
    }
    list_details: dict[str, list[object]] = {}
    for key, expected in expected_lists.items():
        value = artifact_contract.get(key)
        if not isinstance(value, list) or value != list(expected):
            violations.append(f"matrix_state_readback_{key}_mismatch")
            list_details[key] = []
        else:
            list_details[key] = list(value)

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("matrix_state_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(MATRIX_STATE_PHYSICAL_READBACK_SOURCES):
        violations.append("matrix_state_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("matrix_state_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_matrix_statuses")
    if (
        not isinstance(accepted_statuses, list)
        or accepted_statuses != list(MATRIX_STATE_VERIFIED_PHYSICAL_STATUSES)
    ):
        violations.append("matrix_state_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields != list(MATRIX_STATE_FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("matrix_state_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    physical_gap_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_gap_ids, list):
        physical_gap_ids = []
        violations.append("matrix_state_fixture_physical_gap_ids_missing")
    if MATRIX_STATE_SURFACE_ID not in physical_gap_ids:
        violations.append("matrix_state_fixture_gap_not_declared")

    remediation_action = {}
    for action in remediation_actions:
        if isinstance(action, Mapping) and action.get("surface_id") == MATRIX_STATE_SURFACE_ID:
            remediation_action = dict(action)
            break
    if not remediation_action:
        violations.append("matrix_state_remediation_action_missing")
    elif remediation_action.get("id") != MATRIX_STATE_REMEDIATION_ACTION_ID:
        violations.append("matrix_state_remediation_action_id_mismatch")
    if remediation_action and remediation_action.get("physical_readback_gap") is not True:
        violations.append("matrix_state_remediation_physical_gap_not_true")

    if contract_summary.get("physical_readback_surface_id") != MATRIX_STATE_SURFACE_ID:
        violations.append("matrix_state_contract_surface_summary_mismatch")
    if contract_summary.get("physical_readback_action_id") != MATRIX_STATE_REMEDIATION_ACTION_ID:
        violations.append("matrix_state_contract_action_summary_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("matrix_state_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("matrix_state_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("matrix_state_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("matrix_state_contract_live_authority_not_false")
    promotion_families = contract_summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("matrix_state_promotion_families_missing")
    for family in ("bus_group_matrix", "fx_send_insert", "routing"):
        if family not in promotion_families:
            violations.append(f"matrix_state_promotion_family_missing:{family}")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["physical_readback_surface_id"] = contract_summary.get("physical_readback_surface_id")
    detail["physical_readback_action_id"] = contract_summary.get("physical_readback_action_id")
    detail["fixture_physical_gap_declared"] = MATRIX_STATE_SURFACE_ID in physical_gap_ids
    detail["remediation_physical_gap_declared"] = bool(
        remediation_action and remediation_action.get("physical_readback_gap") is True
    )
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["required_top_level_field_count"] = len(
        list_details.get("required_top_level_fields", [])
    )
    detail["required_matrix_field_count"] = len(
        list_details.get("required_matrix_fields", [])
    )
    detail["required_control_fields"] = list(list_details.get("required_control_fields", []))
    detail["required_eq_fields"] = list(list_details.get("required_eq_fields", []))
    detail["required_eq_band_fields"] = list(
        list_details.get("required_eq_band_fields", [])
    )
    detail["required_dynamics_fields"] = list(
        list_details.get("required_dynamics_fields", [])
    )
    detail["required_insert_fields"] = list(list_details.get("required_insert_fields", []))
    detail["required_metering_fields"] = list(
        list_details.get("required_metering_fields", [])
    )
    detail["required_coverage_flags"] = list(
        list_details.get("required_coverage_flags", [])
    )
    detail["required_false_safety_flags"] = list(
        list_details.get("required_false_safety_flags", [])
    )
    detail["minimum_matrix_count"] = int(artifact_contract.get("minimum_matrix_count") or 0)
    detail["minimum_eq_band_count"] = int(artifact_contract.get("minimum_eq_band_count") or 0)
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_matrix_statuses"] = list(accepted_statuses)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="matrix_state_physical_readback_contract_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_matrix_state_physical_readback_artifact_intake_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "matrix_state_artifact_intake_is_read_only_operator_review_gate",
    }
    violations: list[str] = []

    try:
        intake = build_matrix_state_physical_readback_artifact_intake()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        intake = {}
        violations.append(f"matrix_state_artifact_intake_build_error:{type(exc).__name__}")

    safety = intake.get("safety") if isinstance(intake.get("safety"), Mapping) else {}
    summary = intake.get("summary") if isinstance(intake.get("summary"), Mapping) else {}
    artifacts = intake.get("artifacts") if isinstance(intake.get("artifacts"), list) else []

    if intake.get("type") != "matrix_state_physical_readback_artifact_intake":
        violations.append("matrix_state_artifact_intake_type_mismatch")
    if intake.get("method_id") != MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID:
        violations.append("matrix_state_artifact_intake_method_id_mismatch")
    if intake.get("required_contract_method_id") != MATRIX_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("matrix_state_artifact_intake_required_contract_mismatch")
    if intake.get("schema_id") != MATRIX_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("matrix_state_artifact_intake_schema_id_mismatch")
    if intake.get("status") not in MATRIX_STATE_INTAKE_STATUSES:
        violations.append("matrix_state_artifact_intake_status_unknown")
    if intake.get("surface_id") != MATRIX_STATE_SURFACE_ID:
        violations.append("matrix_state_artifact_intake_surface_id_mismatch")
    if intake.get("remediation_action_id") != MATRIX_STATE_REMEDIATION_ACTION_ID:
        violations.append("matrix_state_artifact_intake_action_id_mismatch")

    for key in ("read_only", "operator_required"):
        if safety.get(key) is not True:
            violations.append(f"matrix_state_artifact_intake_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_intake",
        "fixture_closure_allowed_by_intake",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"matrix_state_artifact_intake_{key}_not_false")

    artifact_count = summary.get("artifact_count")
    valid_count = summary.get("valid_candidate_count")
    invalid_count = summary.get("invalid_candidate_count")
    valid_matrix_count = summary.get("valid_matrix_count")
    if not isinstance(artifact_count, int) or artifact_count != len(artifacts):
        violations.append("matrix_state_artifact_intake_artifact_count_mismatch")
        artifact_count = len(artifacts)
    if not isinstance(valid_count, int):
        violations.append("matrix_state_artifact_intake_valid_count_missing")
        valid_count = 0
    if not isinstance(invalid_count, int):
        violations.append("matrix_state_artifact_intake_invalid_count_missing")
        invalid_count = 0
    if not isinstance(valid_matrix_count, int):
        violations.append("matrix_state_artifact_intake_valid_matrix_count_missing")
        valid_matrix_count = 0
    if valid_count + invalid_count != artifact_count:
        violations.append("matrix_state_artifact_intake_candidate_count_mismatch")
    if invalid_count:
        violations.append("matrix_state_artifact_intake_invalid_candidate_present")

    if summary.get("operator_review_required") is not True:
        violations.append("matrix_state_artifact_intake_operator_review_not_true")
    if summary.get("physical_fixture_gap_remains") is not True:
        violations.append("matrix_state_artifact_intake_physical_gap_not_true")
    if summary.get("fixture_closure_allowed_by_intake") is not False:
        violations.append("matrix_state_artifact_intake_fixture_closure_not_false")
    if summary.get("live_authority_granted") is not False:
        violations.append("matrix_state_artifact_intake_live_authority_not_false")
    if summary.get("full_autonomous_ready") is not False:
        violations.append("matrix_state_artifact_intake_full_autonomous_not_false")
    if summary.get("accepted_as_physical_fixture_candidate") is not bool(valid_count):
        violations.append("matrix_state_artifact_intake_candidate_flag_mismatch")

    promotion_families = summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("matrix_state_artifact_intake_promotion_families_missing")
    for family in (
        "bus_group_matrix",
        "fx_send_insert",
        "master_bus",
        "mix_balance",
        "routing",
        "performance_adaptation",
    ):
        if family not in promotion_families:
            violations.append(f"matrix_state_artifact_intake_promotion_family_missing:{family}")

    intake_violations = (
        list(summary.get("violations")) if isinstance(summary.get("violations"), list) else []
    )
    artifact_source_ids = [
        row.get("source_id")
        for row in artifacts
        if isinstance(row, Mapping) and isinstance(row.get("source_id"), str)
    ]
    detail["intake_status"] = intake.get("status")
    detail["schema_id"] = intake.get("schema_id")
    detail["surface_id"] = intake.get("surface_id")
    detail["remediation_action_id"] = intake.get("remediation_action_id")
    detail["required_contract_method_id"] = intake.get("required_contract_method_id")
    detail["artifact_search_roots"] = (
        list(summary.get("artifact_search_roots"))
        if isinstance(summary.get("artifact_search_roots"), list)
        else []
    )
    detail["artifact_count"] = artifact_count
    detail["valid_candidate_count"] = valid_count
    detail["invalid_candidate_count"] = invalid_count
    detail["valid_matrix_count"] = valid_matrix_count
    detail["accepted_as_physical_fixture_candidate"] = summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["fixture_closure_allowed_by_intake"] = summary.get(
        "fixture_closure_allowed_by_intake"
    )
    detail["live_authority_granted"] = summary.get("live_authority_granted")
    detail["full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["latest_valid_candidate_source_id"] = summary.get("latest_valid_candidate_source_id")
    detail["latest_valid_candidate_captured_at"] = summary.get(
        "latest_valid_candidate_captured_at"
    )
    detail["blocked_by"] = (
        list(summary.get("blocked_by")) if isinstance(summary.get("blocked_by"), list) else []
    )
    detail["artifact_source_ids"] = artifact_source_ids
    detail["intake_violations"] = intake_violations
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="matrix_state_physical_readback_artifact_intake_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_fx_slots_inserts_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "fx_slots_inserts_readback_contract_is_schema_only_no_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_fx_slots_inserts_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"fx_slots_inserts_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_actions = (
        remediation_plan.get("actions")
        if isinstance(remediation_plan.get("actions"), list)
        else []
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "fx_slots_inserts_physical_readback_contract":
        violations.append("fx_slots_inserts_readback_contract_type_mismatch")
    if contract.get("method_id") != FX_SLOTS_INSERTS_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("fx_slots_inserts_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("fx_slots_inserts_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("fx_slots_inserts_readback_contract_status_mismatch")
    if contract.get("surface_id") != FX_SLOTS_INSERTS_SURFACE_ID:
        violations.append("fx_slots_inserts_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID:
        violations.append("fx_slots_inserts_readback_contract_action_id_mismatch")

    if safety.get("read_only") is not True:
        violations.append("fx_slots_inserts_readback_contract_read_only_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"fx_slots_inserts_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("fx_slots_inserts_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("fx_slots_inserts_readback_artifact_schema_id_mismatch")
    if artifact_contract.get("minimum_fx_slot_count") != MINIMUM_FX_SLOT_COUNT:
        violations.append("fx_slots_inserts_readback_artifact_minimum_fx_slot_count_mismatch")
    if artifact_contract.get("minimum_fx_parameter_count") != MINIMUM_FX_PARAMETER_COUNT:
        violations.append("fx_slots_inserts_readback_artifact_minimum_fx_parameter_count_mismatch")
    if artifact_contract.get("minimum_insert_count") != MINIMUM_FX_INSERT_COUNT:
        violations.append("fx_slots_inserts_readback_artifact_minimum_insert_count_mismatch")

    expected_lists = {
        "required_top_level_fields": FX_SLOTS_INSERTS_REQUIRED_TOP_LEVEL_FIELDS,
        "required_fx_slot_fields": FX_SLOT_REQUIRED_FIELDS,
        "required_parameter_fields": FX_PARAMETER_REQUIRED_FIELDS,
        "required_parameter_safe_range_fields": FX_PARAMETER_SAFE_RANGE_FIELDS,
        "required_parameter_schema_fields": FX_PARAMETER_SCHEMA_REQUIRED_FIELDS,
        "required_fx_slot_coverage_flags": FX_SLOT_REQUIRED_COVERAGE_FLAGS,
        "required_insert_fields": FX_INSERT_REQUIRED_FIELDS,
        "required_insert_coverage_flags": FX_INSERT_REQUIRED_COVERAGE_FLAGS,
        "required_false_safety_flags": FX_SLOTS_INSERTS_REQUIRED_FALSE_SAFETY_FLAGS,
    }
    list_details: dict[str, list[object]] = {}
    for key, expected in expected_lists.items():
        value = artifact_contract.get(key)
        if not isinstance(value, list) or value != list(expected):
            violations.append(f"fx_slots_inserts_readback_{key}_mismatch")
            list_details[key] = []
        else:
            list_details[key] = list(value)

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("fx_slots_inserts_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(FX_SLOTS_INSERTS_PHYSICAL_READBACK_SOURCES):
        violations.append("fx_slots_inserts_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("fx_slots_inserts_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_row_statuses")
    if (
        not isinstance(accepted_statuses, list)
        or accepted_statuses != list(FX_SLOTS_INSERTS_VERIFIED_PHYSICAL_STATUSES)
    ):
        violations.append("fx_slots_inserts_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    target_types = artifact_contract.get("allowed_insert_target_types")
    if not isinstance(target_types, list) or target_types != list(FX_INSERT_TARGET_TYPES):
        violations.append("fx_slots_inserts_readback_insert_target_types_mismatch")
        target_types = []
    positions = artifact_contract.get("allowed_insert_positions")
    if not isinstance(positions, list) or positions != list(FX_INSERT_POSITIONS):
        violations.append("fx_slots_inserts_readback_insert_positions_mismatch")
        positions = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields
        != list(FX_SLOTS_INSERTS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("fx_slots_inserts_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    physical_gap_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_gap_ids, list):
        physical_gap_ids = []
        violations.append("fx_slots_inserts_fixture_physical_gap_ids_missing")
    if FX_SLOTS_INSERTS_SURFACE_ID not in physical_gap_ids:
        violations.append("fx_slots_inserts_fixture_gap_not_declared")

    remediation_action = {}
    for action in remediation_actions:
        if isinstance(action, Mapping) and action.get("surface_id") == FX_SLOTS_INSERTS_SURFACE_ID:
            remediation_action = dict(action)
            break
    if not remediation_action:
        violations.append("fx_slots_inserts_remediation_action_missing")
    elif remediation_action.get("id") != FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID:
        violations.append("fx_slots_inserts_remediation_action_id_mismatch")
    if remediation_action and remediation_action.get("physical_readback_gap") is not True:
        violations.append("fx_slots_inserts_remediation_physical_gap_not_true")

    if contract_summary.get("physical_readback_surface_id") != FX_SLOTS_INSERTS_SURFACE_ID:
        violations.append("fx_slots_inserts_contract_surface_summary_mismatch")
    if contract_summary.get("physical_readback_action_id") != FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID:
        violations.append("fx_slots_inserts_contract_action_summary_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("fx_slots_inserts_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("fx_slots_inserts_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("fx_slots_inserts_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("fx_slots_inserts_contract_live_authority_not_false")
    promotion_families = contract_summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("fx_slots_inserts_promotion_families_missing")
    for family in ("fx_send_insert", "routing", "mix_balance"):
        if family not in promotion_families:
            violations.append(f"fx_slots_inserts_promotion_family_missing:{family}")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["physical_readback_surface_id"] = contract_summary.get("physical_readback_surface_id")
    detail["physical_readback_action_id"] = contract_summary.get("physical_readback_action_id")
    detail["fixture_physical_gap_declared"] = FX_SLOTS_INSERTS_SURFACE_ID in physical_gap_ids
    detail["remediation_physical_gap_declared"] = bool(
        remediation_action and remediation_action.get("physical_readback_gap") is True
    )
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["required_top_level_field_count"] = len(
        list_details.get("required_top_level_fields", [])
    )
    detail["required_fx_slot_field_count"] = len(
        list_details.get("required_fx_slot_fields", [])
    )
    detail["required_fx_slot_fields"] = list(list_details.get("required_fx_slot_fields", []))
    detail["required_parameter_fields"] = list(
        list_details.get("required_parameter_fields", [])
    )
    detail["required_parameter_safe_range_fields"] = list(
        list_details.get("required_parameter_safe_range_fields", [])
    )
    detail["required_parameter_schema_fields"] = list(
        list_details.get("required_parameter_schema_fields", [])
    )
    detail["required_fx_slot_coverage_flags"] = list(
        list_details.get("required_fx_slot_coverage_flags", [])
    )
    detail["required_insert_fields"] = list(list_details.get("required_insert_fields", []))
    detail["required_insert_coverage_flags"] = list(
        list_details.get("required_insert_coverage_flags", [])
    )
    detail["required_false_safety_flags"] = list(
        list_details.get("required_false_safety_flags", [])
    )
    detail["minimum_fx_slot_count"] = int(
        artifact_contract.get("minimum_fx_slot_count") or 0
    )
    detail["minimum_fx_parameter_count"] = int(
        artifact_contract.get("minimum_fx_parameter_count") or 0
    )
    detail["minimum_insert_count"] = int(artifact_contract.get("minimum_insert_count") or 0)
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_row_statuses"] = list(accepted_statuses)
    detail["allowed_insert_target_types"] = list(target_types)
    detail["allowed_insert_positions"] = list(positions)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="fx_slots_inserts_physical_readback_contract_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_routing_sends_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "routing_sends_readback_contract_is_schema_only_no_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_routing_sends_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"routing_sends_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_actions = (
        remediation_plan.get("actions")
        if isinstance(remediation_plan.get("actions"), list)
        else []
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "routing_sends_physical_readback_contract":
        violations.append("routing_sends_readback_contract_type_mismatch")
    if contract.get("method_id") != ROUTING_SENDS_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("routing_sends_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != ROUTING_SENDS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("routing_sends_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("routing_sends_readback_contract_status_mismatch")
    if contract.get("surface_id") != ROUTING_SENDS_SURFACE_ID:
        violations.append("routing_sends_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != ROUTING_SENDS_REMEDIATION_ACTION_ID:
        violations.append("routing_sends_readback_contract_action_id_mismatch")

    if safety.get("read_only") is not True:
        violations.append("routing_sends_readback_contract_read_only_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"routing_sends_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("routing_sends_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != ROUTING_SENDS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("routing_sends_readback_artifact_schema_id_mismatch")

    minimums = {
        "minimum_static_dante_route_count": MINIMUM_STATIC_DANTE_ROUTE_COUNT,
        "minimum_channel_bus_send_count": MINIMUM_CHANNEL_BUS_SEND_COUNT,
        "minimum_channel_main_send_count": MINIMUM_CHANNEL_MAIN_SEND_COUNT,
        "minimum_channel_matrix_send_count": MINIMUM_CHANNEL_MATRIX_SEND_COUNT,
        "minimum_output_route_count": MINIMUM_OUTPUT_ROUTE_COUNT,
    }
    for key, expected in minimums.items():
        if artifact_contract.get(key) != expected:
            violations.append(f"routing_sends_readback_artifact_{key}_mismatch")

    expected_lists = {
        "required_top_level_fields": ROUTING_SENDS_REQUIRED_TOP_LEVEL_FIELDS,
        "required_static_route_fields": ROUTING_SENDS_REQUIRED_STATIC_ROUTE_FIELDS,
        "required_bus_send_fields": ROUTING_SENDS_REQUIRED_BUS_SEND_FIELDS,
        "required_main_send_fields": ROUTING_SENDS_REQUIRED_MAIN_SEND_FIELDS,
        "required_matrix_send_fields": ROUTING_SENDS_REQUIRED_MATRIX_SEND_FIELDS,
        "required_output_route_fields": ROUTING_SENDS_REQUIRED_OUTPUT_ROUTE_FIELDS,
        "required_static_route_coverage_flags": (
            ROUTING_SENDS_REQUIRED_STATIC_ROUTE_COVERAGE_FLAGS
        ),
        "required_send_coverage_flags": ROUTING_SENDS_REQUIRED_SEND_COVERAGE_FLAGS,
        "required_output_route_coverage_flags": (
            ROUTING_SENDS_REQUIRED_OUTPUT_ROUTE_COVERAGE_FLAGS
        ),
        "required_false_safety_flags": ROUTING_SENDS_REQUIRED_FALSE_SAFETY_FLAGS,
    }
    list_details: dict[str, list[object]] = {}
    for key, expected in expected_lists.items():
        value = artifact_contract.get(key)
        if not isinstance(value, list) or value != list(expected):
            violations.append(f"routing_sends_readback_{key}_mismatch")
            list_details[key] = []
        else:
            list_details[key] = list(value)

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("routing_sends_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(ROUTING_SENDS_PHYSICAL_READBACK_SOURCES):
        violations.append("routing_sends_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("routing_sends_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_row_statuses")
    if (
        not isinstance(accepted_statuses, list)
        or accepted_statuses != list(ROUTING_SENDS_VERIFIED_PHYSICAL_STATUSES)
    ):
        violations.append("routing_sends_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields
        != list(ROUTING_SENDS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("routing_sends_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    physical_gap_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_gap_ids, list):
        physical_gap_ids = []
        violations.append("routing_sends_fixture_physical_gap_ids_missing")
    if ROUTING_SENDS_SURFACE_ID not in physical_gap_ids:
        violations.append("routing_sends_fixture_gap_not_declared")

    remediation_action = {}
    for action in remediation_actions:
        if isinstance(action, Mapping) and action.get("surface_id") == ROUTING_SENDS_SURFACE_ID:
            remediation_action = dict(action)
            break
    if not remediation_action:
        violations.append("routing_sends_remediation_action_missing")
    elif remediation_action.get("id") != ROUTING_SENDS_REMEDIATION_ACTION_ID:
        violations.append("routing_sends_remediation_action_id_mismatch")
    if remediation_action and remediation_action.get("physical_readback_gap") is not True:
        violations.append("routing_sends_remediation_physical_gap_not_true")

    if contract_summary.get("physical_readback_surface_id") != ROUTING_SENDS_SURFACE_ID:
        violations.append("routing_sends_contract_surface_summary_mismatch")
    if contract_summary.get("physical_readback_action_id") != ROUTING_SENDS_REMEDIATION_ACTION_ID:
        violations.append("routing_sends_contract_action_summary_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("routing_sends_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("routing_sends_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("routing_sends_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("routing_sends_contract_live_authority_not_false")
    promotion_families = contract_summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("routing_sends_promotion_families_missing")
    for family in ("routing", "fx_send_insert", "scene_preset"):
        if family not in promotion_families:
            violations.append(f"routing_sends_promotion_family_missing:{family}")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["physical_readback_surface_id"] = contract_summary.get("physical_readback_surface_id")
    detail["physical_readback_action_id"] = contract_summary.get("physical_readback_action_id")
    detail["fixture_physical_gap_declared"] = ROUTING_SENDS_SURFACE_ID in physical_gap_ids
    detail["remediation_physical_gap_declared"] = bool(
        remediation_action and remediation_action.get("physical_readback_gap") is True
    )
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["required_top_level_field_count"] = len(
        list_details.get("required_top_level_fields", [])
    )
    detail["required_static_route_field_count"] = len(
        list_details.get("required_static_route_fields", [])
    )
    detail["required_static_route_fields"] = list(
        list_details.get("required_static_route_fields", [])
    )
    detail["required_bus_send_fields"] = list(
        list_details.get("required_bus_send_fields", [])
    )
    detail["required_main_send_fields"] = list(
        list_details.get("required_main_send_fields", [])
    )
    detail["required_matrix_send_fields"] = list(
        list_details.get("required_matrix_send_fields", [])
    )
    detail["required_output_route_fields"] = list(
        list_details.get("required_output_route_fields", [])
    )
    detail["required_static_route_coverage_flags"] = list(
        list_details.get("required_static_route_coverage_flags", [])
    )
    detail["required_send_coverage_flags"] = list(
        list_details.get("required_send_coverage_flags", [])
    )
    detail["required_output_route_coverage_flags"] = list(
        list_details.get("required_output_route_coverage_flags", [])
    )
    detail["required_false_safety_flags"] = list(
        list_details.get("required_false_safety_flags", [])
    )
    for key in minimums:
        detail[key] = int(artifact_contract.get(key) or 0)
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_row_statuses"] = list(accepted_statuses)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="routing_sends_physical_readback_contract_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_snapshots_scenes_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "snapshots_scenes_readback_contract_is_schema_only_no_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_snapshots_scenes_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"snapshots_scenes_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_actions = (
        remediation_plan.get("actions")
        if isinstance(remediation_plan.get("actions"), list)
        else []
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "snapshots_scenes_physical_readback_contract":
        violations.append("snapshots_scenes_readback_contract_type_mismatch")
    if contract.get("method_id") != SNAPSHOTS_SCENES_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("snapshots_scenes_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != SNAPSHOTS_SCENES_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("snapshots_scenes_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("snapshots_scenes_readback_contract_status_mismatch")
    if contract.get("surface_id") != SNAPSHOTS_SCENES_SURFACE_ID:
        violations.append("snapshots_scenes_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != SNAPSHOTS_SCENES_REMEDIATION_ACTION_ID:
        violations.append("snapshots_scenes_readback_contract_action_id_mismatch")

    if safety.get("read_only") is not True:
        violations.append("snapshots_scenes_readback_contract_read_only_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
        "scene_load_allowed",
        "scene_save_allowed",
    ):
        if safety.get(key) is not False:
            violations.append(f"snapshots_scenes_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != SNAPSHOTS_SCENES_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("snapshots_scenes_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != SNAPSHOTS_SCENES_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("snapshots_scenes_readback_artifact_schema_id_mismatch")
    if artifact_contract.get("minimum_snapshot_count") != MINIMUM_SNAPSHOT_SCENE_COUNT:
        violations.append("snapshots_scenes_readback_artifact_minimum_snapshot_count_mismatch")

    expected_lists = {
        "required_top_level_fields": SNAPSHOTS_SCENES_REQUIRED_TOP_LEVEL_FIELDS,
        "required_active_scene_fields": SNAPSHOTS_SCENES_REQUIRED_ACTIVE_SCENE_FIELDS,
        "required_snapshot_fields": SNAPSHOTS_SCENES_REQUIRED_SNAPSHOT_FIELDS,
        "required_coverage_flags": SNAPSHOTS_SCENES_REQUIRED_COVERAGE_FLAGS,
        "required_false_safety_flags": SNAPSHOTS_SCENES_REQUIRED_FALSE_SAFETY_FLAGS,
    }
    list_details: dict[str, list[object]] = {}
    for key, expected in expected_lists.items():
        value = artifact_contract.get(key)
        if not isinstance(value, list) or value != list(expected):
            violations.append(f"snapshots_scenes_readback_{key}_mismatch")
            list_details[key] = []
        else:
            list_details[key] = list(value)

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("snapshots_scenes_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(SNAPSHOTS_SCENES_PHYSICAL_READBACK_SOURCES):
        violations.append("snapshots_scenes_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("snapshots_scenes_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_row_statuses")
    if (
        not isinstance(accepted_statuses, list)
        or accepted_statuses != list(SNAPSHOTS_SCENES_VERIFIED_PHYSICAL_STATUSES)
    ):
        violations.append("snapshots_scenes_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields
        != list(SNAPSHOTS_SCENES_FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("snapshots_scenes_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    physical_gap_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_gap_ids, list):
        physical_gap_ids = []
        violations.append("snapshots_scenes_fixture_physical_gap_ids_missing")
    if SNAPSHOTS_SCENES_SURFACE_ID not in physical_gap_ids:
        violations.append("snapshots_scenes_fixture_gap_not_declared")

    remediation_action = {}
    for action in remediation_actions:
        if isinstance(action, Mapping) and action.get("surface_id") == SNAPSHOTS_SCENES_SURFACE_ID:
            remediation_action = dict(action)
            break
    if not remediation_action:
        violations.append("snapshots_scenes_remediation_action_missing")
    elif remediation_action.get("id") != SNAPSHOTS_SCENES_REMEDIATION_ACTION_ID:
        violations.append("snapshots_scenes_remediation_action_id_mismatch")
    if remediation_action and remediation_action.get("physical_readback_gap") is not True:
        violations.append("snapshots_scenes_remediation_physical_gap_not_true")

    if contract_summary.get("physical_readback_surface_id") != SNAPSHOTS_SCENES_SURFACE_ID:
        violations.append("snapshots_scenes_contract_surface_summary_mismatch")
    if contract_summary.get("physical_readback_action_id") != SNAPSHOTS_SCENES_REMEDIATION_ACTION_ID:
        violations.append("snapshots_scenes_contract_action_summary_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("snapshots_scenes_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("snapshots_scenes_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("snapshots_scenes_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("snapshots_scenes_contract_live_authority_not_false")
    promotion_families = contract_summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("snapshots_scenes_promotion_families_missing")
    for family in ("scene_preset", "routing"):
        if family not in promotion_families:
            violations.append(f"snapshots_scenes_promotion_family_missing:{family}")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["physical_readback_surface_id"] = contract_summary.get("physical_readback_surface_id")
    detail["physical_readback_action_id"] = contract_summary.get("physical_readback_action_id")
    detail["fixture_physical_gap_declared"] = SNAPSHOTS_SCENES_SURFACE_ID in physical_gap_ids
    detail["remediation_physical_gap_declared"] = bool(
        remediation_action and remediation_action.get("physical_readback_gap") is True
    )
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["required_top_level_field_count"] = len(
        list_details.get("required_top_level_fields", [])
    )
    detail["required_active_scene_fields"] = list(
        list_details.get("required_active_scene_fields", [])
    )
    detail["required_snapshot_fields"] = list(
        list_details.get("required_snapshot_fields", [])
    )
    detail["required_coverage_flags"] = list(
        list_details.get("required_coverage_flags", [])
    )
    detail["required_false_safety_flags"] = list(
        list_details.get("required_false_safety_flags", [])
    )
    detail["minimum_snapshot_count"] = int(
        artifact_contract.get("minimum_snapshot_count") or 0
    )
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_row_statuses"] = list(accepted_statuses)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="snapshots_scenes_physical_readback_contract_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_user_presets_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "user_presets_readback_contract_is_schema_only_no_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_user_presets_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"user_presets_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_actions = (
        remediation_plan.get("actions")
        if isinstance(remediation_plan.get("actions"), list)
        else []
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "user_presets_physical_readback_contract":
        violations.append("user_presets_readback_contract_type_mismatch")
    if contract.get("method_id") != USER_PRESETS_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("user_presets_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != USER_PRESETS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("user_presets_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("user_presets_readback_contract_status_mismatch")
    if contract.get("surface_id") != USER_PRESETS_SURFACE_ID:
        violations.append("user_presets_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != USER_PRESETS_REMEDIATION_ACTION_ID:
        violations.append("user_presets_readback_contract_action_id_mismatch")

    if safety.get("read_only") is not True:
        violations.append("user_presets_readback_contract_read_only_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
        "preset_recall_allowed",
        "preset_store_allowed",
    ):
        if safety.get(key) is not False:
            violations.append(f"user_presets_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != USER_PRESETS_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("user_presets_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != USER_PRESETS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("user_presets_readback_artifact_schema_id_mismatch")
    if artifact_contract.get("minimum_preset_count") != MINIMUM_USER_PRESET_COUNT:
        violations.append("user_presets_readback_artifact_minimum_preset_count_mismatch")

    expected_lists = {
        "required_top_level_fields": USER_PRESETS_REQUIRED_TOP_LEVEL_FIELDS,
        "required_preset_fields": USER_PRESETS_REQUIRED_PRESET_FIELDS,
        "required_coverage_flags": USER_PRESETS_REQUIRED_COVERAGE_FLAGS,
        "required_false_safety_flags": USER_PRESETS_REQUIRED_FALSE_SAFETY_FLAGS,
    }
    list_details: dict[str, list[object]] = {}
    for key, expected in expected_lists.items():
        value = artifact_contract.get(key)
        if not isinstance(value, list) or value != list(expected):
            violations.append(f"user_presets_readback_{key}_mismatch")
            list_details[key] = []
        else:
            list_details[key] = list(value)

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("user_presets_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(USER_PRESETS_PHYSICAL_READBACK_SOURCES):
        violations.append("user_presets_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("user_presets_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_row_statuses")
    if (
        not isinstance(accepted_statuses, list)
        or accepted_statuses != list(USER_PRESETS_VERIFIED_PHYSICAL_STATUSES)
    ):
        violations.append("user_presets_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields
        != list(USER_PRESETS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("user_presets_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    physical_gap_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_gap_ids, list):
        physical_gap_ids = []
        violations.append("user_presets_fixture_physical_gap_ids_missing")
    if USER_PRESETS_SURFACE_ID not in physical_gap_ids:
        violations.append("user_presets_fixture_gap_not_declared")

    remediation_action = {}
    for action in remediation_actions:
        if isinstance(action, Mapping) and action.get("surface_id") == USER_PRESETS_SURFACE_ID:
            remediation_action = dict(action)
            break
    if not remediation_action:
        violations.append("user_presets_remediation_action_missing")
    elif remediation_action.get("id") != USER_PRESETS_REMEDIATION_ACTION_ID:
        violations.append("user_presets_remediation_action_id_mismatch")
    if remediation_action and remediation_action.get("physical_readback_gap") is not True:
        violations.append("user_presets_remediation_physical_gap_not_true")

    if contract_summary.get("physical_readback_surface_id") != USER_PRESETS_SURFACE_ID:
        violations.append("user_presets_contract_surface_summary_mismatch")
    if contract_summary.get("physical_readback_action_id") != USER_PRESETS_REMEDIATION_ACTION_ID:
        violations.append("user_presets_contract_action_summary_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("user_presets_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("user_presets_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("user_presets_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("user_presets_contract_live_authority_not_false")
    promotion_families = contract_summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("user_presets_promotion_families_missing")
    for family in ("scene_preset", "routing"):
        if family not in promotion_families:
            violations.append(f"user_presets_promotion_family_missing:{family}")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["physical_readback_surface_id"] = contract_summary.get("physical_readback_surface_id")
    detail["physical_readback_action_id"] = contract_summary.get("physical_readback_action_id")
    detail["fixture_physical_gap_declared"] = USER_PRESETS_SURFACE_ID in physical_gap_ids
    detail["remediation_physical_gap_declared"] = bool(
        remediation_action and remediation_action.get("physical_readback_gap") is True
    )
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["required_top_level_field_count"] = len(
        list_details.get("required_top_level_fields", [])
    )
    detail["required_preset_fields"] = list(
        list_details.get("required_preset_fields", [])
    )
    detail["required_coverage_flags"] = list(
        list_details.get("required_coverage_flags", [])
    )
    detail["required_false_safety_flags"] = list(
        list_details.get("required_false_safety_flags", [])
    )
    detail["minimum_preset_count"] = int(
        artifact_contract.get("minimum_preset_count") or 0
    )
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_row_statuses"] = list(accepted_statuses)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="user_presets_physical_readback_contract_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_dlive_state_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "dlive_state_readback_contract_is_schema_only_no_midi_tcp_or_dlive_write_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_dlive_state_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"dlive_state_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_actions = (
        remediation_plan.get("actions")
        if isinstance(remediation_plan.get("actions"), list)
        else []
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "dlive_state_physical_readback_contract":
        violations.append("dlive_state_readback_contract_type_mismatch")
    if contract.get("method_id") != DLIVE_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("dlive_state_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != DLIVE_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("dlive_state_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("dlive_state_readback_contract_status_mismatch")
    if contract.get("surface_id") != DLIVE_STATE_SURFACE_ID:
        violations.append("dlive_state_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != DLIVE_STATE_REMEDIATION_ACTION_ID:
        violations.append("dlive_state_readback_contract_action_id_mismatch")

    if safety.get("read_only") is not True:
        violations.append("dlive_state_readback_contract_read_only_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
        "midi_send_allowed",
        "tcp_send_allowed",
        "dlive_write_allowed",
        "dlive_apply_allowed",
    ):
        if safety.get(key) is not False:
            violations.append(f"dlive_state_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("dlive_state_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != DLIVE_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("dlive_state_readback_artifact_schema_id_mismatch")
    if artifact_contract.get("minimum_channel_count") != MINIMUM_DLIVE_STATE_CHANNEL_COUNT:
        violations.append("dlive_state_readback_artifact_minimum_channel_count_mismatch")

    expected_lists = {
        "required_top_level_fields": DLIVE_STATE_REQUIRED_TOP_LEVEL_FIELDS,
        "required_channel_fields": DLIVE_STATE_REQUIRED_CHANNEL_FIELDS,
        "required_midi_state_fields": DLIVE_STATE_REQUIRED_MIDI_STATE_FIELDS,
        "required_parity_fields": DLIVE_STATE_REQUIRED_PARITY_FIELDS,
        "required_coverage_flags": DLIVE_STATE_REQUIRED_COVERAGE_FLAGS,
        "required_false_safety_flags": DLIVE_STATE_REQUIRED_FALSE_SAFETY_FLAGS,
    }
    list_details: dict[str, list[object]] = {}
    for key, expected in expected_lists.items():
        value = artifact_contract.get(key)
        if not isinstance(value, list) or value != list(expected):
            violations.append(f"dlive_state_readback_{key}_mismatch")
            list_details[key] = []
        else:
            list_details[key] = list(value)

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("dlive_state_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(DLIVE_STATE_PHYSICAL_READBACK_SOURCES):
        violations.append("dlive_state_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("dlive_state_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_row_statuses")
    if (
        not isinstance(accepted_statuses, list)
        or accepted_statuses != list(DLIVE_STATE_VERIFIED_PHYSICAL_STATUSES)
    ):
        violations.append("dlive_state_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields
        != list(DLIVE_STATE_FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("dlive_state_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    physical_gap_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_gap_ids, list):
        physical_gap_ids = []
        violations.append("dlive_state_fixture_physical_gap_ids_missing")
    if DLIVE_STATE_SURFACE_ID not in physical_gap_ids:
        violations.append("dlive_state_fixture_gap_not_declared")

    remediation_action = {}
    for action in remediation_actions:
        if isinstance(action, Mapping) and action.get("surface_id") == DLIVE_STATE_SURFACE_ID:
            remediation_action = dict(action)
            break
    if not remediation_action:
        violations.append("dlive_state_remediation_action_missing")
    elif remediation_action.get("id") != DLIVE_STATE_REMEDIATION_ACTION_ID:
        violations.append("dlive_state_remediation_action_id_mismatch")
    if remediation_action and remediation_action.get("physical_readback_gap") is not True:
        violations.append("dlive_state_remediation_physical_gap_not_true")

    if contract_summary.get("physical_readback_surface_id") != DLIVE_STATE_SURFACE_ID:
        violations.append("dlive_state_contract_surface_summary_mismatch")
    if contract_summary.get("physical_readback_action_id") != DLIVE_STATE_REMEDIATION_ACTION_ID:
        violations.append("dlive_state_contract_action_summary_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("dlive_state_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("dlive_state_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("dlive_state_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("dlive_state_contract_live_authority_not_false")
    promotion_families = contract_summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("dlive_state_promotion_families_missing")
    for family in ("fader", "gain", "mute", "routing", "performance_adaptation"):
        if family not in promotion_families:
            violations.append(f"dlive_state_promotion_family_missing:{family}")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["physical_readback_surface_id"] = contract_summary.get("physical_readback_surface_id")
    detail["physical_readback_action_id"] = contract_summary.get("physical_readback_action_id")
    detail["fixture_physical_gap_declared"] = DLIVE_STATE_SURFACE_ID in physical_gap_ids
    detail["remediation_physical_gap_declared"] = bool(
        remediation_action and remediation_action.get("physical_readback_gap") is True
    )
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["required_top_level_field_count"] = len(
        list_details.get("required_top_level_fields", [])
    )
    detail["required_channel_fields"] = list(
        list_details.get("required_channel_fields", [])
    )
    detail["required_midi_state_fields"] = list(
        list_details.get("required_midi_state_fields", [])
    )
    detail["required_parity_fields"] = list(
        list_details.get("required_parity_fields", [])
    )
    detail["required_coverage_flags"] = list(
        list_details.get("required_coverage_flags", [])
    )
    detail["required_false_safety_flags"] = list(
        list_details.get("required_false_safety_flags", [])
    )
    detail["minimum_channel_count"] = int(
        artifact_contract.get("minimum_channel_count") or 0
    )
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_row_statuses"] = list(accepted_statuses)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="dlive_state_physical_readback_contract_schema",
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
        evidence_by_family = detail.get("evidence_requirement_ids_by_action_family")
        next_by_family = detail.get("next_required_evidence_by_action_family")
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
            "evidence_requirement_ids_by_action_family": dict(evidence_by_family)
            if isinstance(evidence_by_family, Mapping)
            else {},
            "next_required_evidence_by_action_family": dict(next_by_family)
            if isinstance(next_by_family, Mapping)
            else {},
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


def read_surface_physical_readback_capture_plan_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "read_surface_physical_readback_capture_plan_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        physical_missing = detail.get("physical_fixture_missing_surface_ids")
        required_contracts = detail.get("required_artifact_contracts")
        blocked_by = detail.get("blocked_by")
        forbidden_actions = detail.get("step_forbidden_actions")
        violations = detail.get("violations")
        physical_missing_ids = list(physical_missing) if isinstance(physical_missing, list) else []
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_plan": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "plan_status": detail.get("plan_status"),
            "capture_step_count": int(detail.get("capture_step_count") or 0),
            "physical_gap_surface_count": len(physical_missing_ids),
            "physical_fixture_missing_surface_ids": physical_missing_ids,
            "next_physical_readback_surface_id": detail.get("next_physical_readback_surface_id"),
            "next_physical_readback_action_id": detail.get("next_physical_readback_action_id"),
            "expected_next_physical_readback_surface_id": detail.get(
                "expected_next_physical_readback_surface_id"
            ),
            "expected_next_physical_readback_action_id": detail.get(
                "expected_next_physical_readback_action_id"
            ),
            "target_action_consistency": detail.get("target_action_consistency") is True,
            "required_artifact_contracts": (
                list(required_contracts) if isinstance(required_contracts, list) else []
            ),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "step_surface_id": detail.get("step_surface_id"),
            "step_remediation_action_id": detail.get("step_remediation_action_id"),
            "step_forbidden_actions": (
                list(forbidden_actions) if isinstance(forbidden_actions, list) else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def raw_mixer_readback_contract_summary(results: Sequence[CheckResult]) -> dict[str, object] | None:
    for result in results:
        if result.name != "raw_mixer_state_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_row_statuses")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "next_physical_readback_surface_id": detail.get("next_physical_readback_surface_id"),
            "next_physical_readback_action_id": detail.get("next_physical_readback_action_id"),
            "fixture_next_physical_readback_surface_id": detail.get(
                "fixture_next_physical_readback_surface_id"
            ),
            "remediation_next_physical_readback_surface_id": detail.get(
                "remediation_next_physical_readback_surface_id"
            ),
            "remediation_next_physical_readback_action_id": detail.get(
                "remediation_next_physical_readback_action_id"
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_row_field_count": int(detail.get("required_row_field_count") or 0),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_row_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("next_physical_readback_surface_id")
                == detail.get("fixture_next_physical_readback_surface_id")
                == detail.get("remediation_next_physical_readback_surface_id")
                and detail.get("next_physical_readback_action_id")
                == detail.get("remediation_next_physical_readback_action_id")
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def raw_mixer_readback_artifact_intake_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "raw_mixer_state_physical_readback_artifact_intake_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        artifact_search_roots = detail.get("artifact_search_roots")
        blocked_by = detail.get("blocked_by")
        artifact_source_ids = detail.get("artifact_source_ids")
        intake_violations = detail.get("intake_violations")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_intake": False,
            "fixture_closure_allowed_by_intake": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "intake_status": detail.get("intake_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "required_contract_method_id": detail.get("required_contract_method_id"),
            "artifact_search_roots": (
                list(artifact_search_roots) if isinstance(artifact_search_roots, list) else []
            ),
            "artifact_count": int(detail.get("artifact_count") or 0),
            "valid_candidate_count": int(detail.get("valid_candidate_count") or 0),
            "invalid_candidate_count": int(detail.get("invalid_candidate_count") or 0),
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "operator_review_required": detail.get("operator_review_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "latest_valid_candidate_source_id": detail.get("latest_valid_candidate_source_id"),
            "latest_valid_candidate_captured_at": detail.get("latest_valid_candidate_captured_at"),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "artifact_source_ids": (
                list(artifact_source_ids) if isinstance(artifact_source_ids, list) else []
            ),
            "intake_violations": (
                list(intake_violations) if isinstance(intake_violations, list) else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def channel_basic_controls_readback_artifact_intake_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "channel_basic_controls_physical_readback_artifact_intake_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        artifact_search_roots = detail.get("artifact_search_roots")
        blocked_by = detail.get("blocked_by")
        artifact_source_ids = detail.get("artifact_source_ids")
        intake_violations = detail.get("intake_violations")
        promotion_families = detail.get("promotion_families_requiring_surface")
        violations = detail.get("violations")
        source_candidate_claimed = detail.get("accepted_as_physical_fixture_candidate") is True
        operator_review_required = detail.get("operator_review_required") is True
        physical_fixture_gap_remains = detail.get("physical_fixture_gap_remains") is True
        unsafe_claimed = (
            detail.get("fixture_closure_allowed_by_intake") is True
            or detail.get("live_authority_granted") is True
            or detail.get("full_autonomous_ready") is True
        )
        candidate_accepted = (
            result.status == "passed"
            and source_candidate_claimed
            and operator_review_required
            and physical_fixture_gap_remains
            and not unsafe_claimed
        )
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_intake": False,
            "fixture_closure_allowed_by_intake": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "intake_status": detail.get("intake_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "required_contract_method_id": detail.get("required_contract_method_id"),
            "artifact_search_roots": (
                list(artifact_search_roots) if isinstance(artifact_search_roots, list) else []
            ),
            "artifact_count": int(detail.get("artifact_count") or 0),
            "valid_candidate_count": int(detail.get("valid_candidate_count") or 0),
            "invalid_candidate_count": int(detail.get("invalid_candidate_count") or 0),
            "valid_channel_count": int(detail.get("valid_channel_count") or 0),
            "accepted_as_physical_fixture_candidate": candidate_accepted,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "latest_valid_candidate_source_id": detail.get("latest_valid_candidate_source_id"),
            "latest_valid_candidate_captured_at": detail.get("latest_valid_candidate_captured_at"),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "artifact_source_ids": (
                list(artifact_source_ids) if isinstance(artifact_source_ids, list) else []
            ),
            "intake_violations": (
                list(intake_violations) if isinstance(intake_violations, list) else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def channel_basic_controls_readback_contract_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "channel_basic_controls_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_channel_statuses")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        promotion_families = detail.get("promotion_families_requiring_surface")
        required_identity_fields = detail.get("required_identity_fields")
        required_control_fields = detail.get("required_control_fields")
        required_coverage_flags = detail.get("required_coverage_flags")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "physical_readback_surface_id": detail.get("physical_readback_surface_id"),
            "physical_readback_action_id": detail.get("physical_readback_action_id"),
            "fixture_physical_gap_declared": detail.get("fixture_physical_gap_declared") is True,
            "remediation_physical_gap_declared": (
                detail.get("remediation_physical_gap_declared") is True
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_channel_field_count": int(detail.get("required_channel_field_count") or 0),
            "required_identity_fields": (
                list(required_identity_fields) if isinstance(required_identity_fields, list) else []
            ),
            "required_control_fields": (
                list(required_control_fields) if isinstance(required_control_fields, list) else []
            ),
            "required_coverage_flags": (
                list(required_coverage_flags) if isinstance(required_coverage_flags, list) else []
            ),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_channel_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("physical_readback_surface_id") == detail.get("surface_id")
                and detail.get("physical_readback_action_id")
                == detail.get("remediation_action_id")
                and detail.get("fixture_physical_gap_declared") is True
                and detail.get("remediation_physical_gap_declared") is True
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def channel_processing_readback_contract_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "channel_processing_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_channel_statuses")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        promotion_families = detail.get("promotion_families_requiring_surface")
        required_hpf_fields = detail.get("required_hpf_fields")
        required_eq_fields = detail.get("required_eq_fields")
        required_eq_band_fields = detail.get("required_eq_band_fields")
        required_compressor_fields = detail.get("required_compressor_fields")
        required_gate_fields = detail.get("required_gate_fields")
        required_coverage_flags = detail.get("required_coverage_flags")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "physical_readback_surface_id": detail.get("physical_readback_surface_id"),
            "physical_readback_action_id": detail.get("physical_readback_action_id"),
            "fixture_physical_gap_declared": detail.get("fixture_physical_gap_declared") is True,
            "remediation_physical_gap_declared": (
                detail.get("remediation_physical_gap_declared") is True
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_channel_field_count": int(detail.get("required_channel_field_count") or 0),
            "required_hpf_fields": (
                list(required_hpf_fields) if isinstance(required_hpf_fields, list) else []
            ),
            "required_eq_fields": (
                list(required_eq_fields) if isinstance(required_eq_fields, list) else []
            ),
            "required_eq_band_fields": (
                list(required_eq_band_fields) if isinstance(required_eq_band_fields, list) else []
            ),
            "required_compressor_fields": (
                list(required_compressor_fields)
                if isinstance(required_compressor_fields, list)
                else []
            ),
            "required_gate_fields": (
                list(required_gate_fields) if isinstance(required_gate_fields, list) else []
            ),
            "required_coverage_flags": (
                list(required_coverage_flags) if isinstance(required_coverage_flags, list) else []
            ),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "minimum_eq_band_count": int(detail.get("minimum_eq_band_count") or 0),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_channel_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("physical_readback_surface_id") == detail.get("surface_id")
                and detail.get("physical_readback_action_id")
                == detail.get("remediation_action_id")
                and detail.get("fixture_physical_gap_declared") is True
                and detail.get("remediation_physical_gap_declared") is True
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def channel_processing_readback_artifact_intake_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "channel_processing_physical_readback_artifact_intake_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        artifact_search_roots = detail.get("artifact_search_roots")
        blocked_by = detail.get("blocked_by")
        artifact_source_ids = detail.get("artifact_source_ids")
        intake_violations = detail.get("intake_violations")
        promotion_families = detail.get("promotion_families_requiring_surface")
        violations = detail.get("violations")
        source_candidate_claimed = detail.get("accepted_as_physical_fixture_candidate") is True
        operator_review_required = detail.get("operator_review_required") is True
        physical_fixture_gap_remains = detail.get("physical_fixture_gap_remains") is True
        unsafe_claimed = (
            detail.get("fixture_closure_allowed_by_intake") is True
            or detail.get("live_authority_granted") is True
            or detail.get("full_autonomous_ready") is True
        )
        candidate_accepted = (
            result.status == "passed"
            and source_candidate_claimed
            and operator_review_required
            and physical_fixture_gap_remains
            and not unsafe_claimed
        )
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_intake": False,
            "fixture_closure_allowed_by_intake": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "intake_status": detail.get("intake_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "required_contract_method_id": detail.get("required_contract_method_id"),
            "artifact_search_roots": (
                list(artifact_search_roots) if isinstance(artifact_search_roots, list) else []
            ),
            "artifact_count": int(detail.get("artifact_count") or 0),
            "valid_candidate_count": int(detail.get("valid_candidate_count") or 0),
            "invalid_candidate_count": int(detail.get("invalid_candidate_count") or 0),
            "valid_channel_count": int(detail.get("valid_channel_count") or 0),
            "accepted_as_physical_fixture_candidate": candidate_accepted,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "latest_valid_candidate_source_id": detail.get("latest_valid_candidate_source_id"),
            "latest_valid_candidate_captured_at": detail.get("latest_valid_candidate_captured_at"),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "artifact_source_ids": (
                list(artifact_source_ids) if isinstance(artifact_source_ids, list) else []
            ),
            "intake_violations": (
                list(intake_violations) if isinstance(intake_violations, list) else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def bus_group_main_dca_readback_contract_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "bus_group_main_dca_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_surface_statuses")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        promotion_families = detail.get("promotion_families_requiring_surface")
        required_bus_control_fields = detail.get("required_bus_control_fields")
        required_main_control_fields = detail.get("required_main_control_fields")
        required_dca_control_fields = detail.get("required_dca_control_fields")
        required_bus_assignment_fields = detail.get("required_bus_assignment_fields")
        required_dca_assignment_fields = detail.get("required_dca_assignment_fields")
        required_metering_fields = detail.get("required_metering_fields")
        required_bus_coverage_flags = detail.get("required_bus_coverage_flags")
        required_main_coverage_flags = detail.get("required_main_coverage_flags")
        required_dca_coverage_flags = detail.get("required_dca_coverage_flags")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "physical_readback_surface_id": detail.get("physical_readback_surface_id"),
            "physical_readback_action_id": detail.get("physical_readback_action_id"),
            "fixture_physical_gap_declared": detail.get("fixture_physical_gap_declared") is True,
            "remediation_physical_gap_declared": (
                detail.get("remediation_physical_gap_declared") is True
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_bus_field_count": int(detail.get("required_bus_field_count") or 0),
            "required_main_field_count": int(detail.get("required_main_field_count") or 0),
            "required_dca_field_count": int(detail.get("required_dca_field_count") or 0),
            "required_bus_control_fields": (
                list(required_bus_control_fields)
                if isinstance(required_bus_control_fields, list)
                else []
            ),
            "required_main_control_fields": (
                list(required_main_control_fields)
                if isinstance(required_main_control_fields, list)
                else []
            ),
            "required_dca_control_fields": (
                list(required_dca_control_fields)
                if isinstance(required_dca_control_fields, list)
                else []
            ),
            "required_bus_assignment_fields": (
                list(required_bus_assignment_fields)
                if isinstance(required_bus_assignment_fields, list)
                else []
            ),
            "required_dca_assignment_fields": (
                list(required_dca_assignment_fields)
                if isinstance(required_dca_assignment_fields, list)
                else []
            ),
            "required_metering_fields": (
                list(required_metering_fields)
                if isinstance(required_metering_fields, list)
                else []
            ),
            "required_bus_coverage_flags": (
                list(required_bus_coverage_flags)
                if isinstance(required_bus_coverage_flags, list)
                else []
            ),
            "required_main_coverage_flags": (
                list(required_main_coverage_flags)
                if isinstance(required_main_coverage_flags, list)
                else []
            ),
            "required_dca_coverage_flags": (
                list(required_dca_coverage_flags)
                if isinstance(required_dca_coverage_flags, list)
                else []
            ),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "minimum_bus_count": int(detail.get("minimum_bus_count") or 0),
            "minimum_main_count": int(detail.get("minimum_main_count") or 0),
            "minimum_dca_count": int(detail.get("minimum_dca_count") or 0),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_surface_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("physical_readback_surface_id") == detail.get("surface_id")
                and detail.get("physical_readback_action_id")
                == detail.get("remediation_action_id")
                and detail.get("fixture_physical_gap_declared") is True
                and detail.get("remediation_physical_gap_declared") is True
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def bus_group_main_dca_readback_artifact_intake_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "bus_group_main_dca_physical_readback_artifact_intake_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        artifact_search_roots = detail.get("artifact_search_roots")
        blocked_by = detail.get("blocked_by")
        artifact_source_ids = detail.get("artifact_source_ids")
        intake_violations = detail.get("intake_violations")
        promotion_families = detail.get("promotion_families_requiring_surface")
        violations = detail.get("violations")
        source_candidate_claimed = detail.get("accepted_as_physical_fixture_candidate") is True
        operator_review_required = detail.get("operator_review_required") is True
        physical_fixture_gap_remains = detail.get("physical_fixture_gap_remains") is True
        unsafe_claimed = (
            detail.get("fixture_closure_allowed_by_intake") is True
            or detail.get("live_authority_granted") is True
            or detail.get("full_autonomous_ready") is True
        )
        candidate_accepted = (
            result.status == "passed"
            and source_candidate_claimed
            and operator_review_required
            and physical_fixture_gap_remains
            and not unsafe_claimed
        )
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_intake": False,
            "fixture_closure_allowed_by_intake": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "intake_status": detail.get("intake_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "required_contract_method_id": detail.get("required_contract_method_id"),
            "artifact_search_roots": (
                list(artifact_search_roots) if isinstance(artifact_search_roots, list) else []
            ),
            "artifact_count": int(detail.get("artifact_count") or 0),
            "valid_candidate_count": int(detail.get("valid_candidate_count") or 0),
            "invalid_candidate_count": int(detail.get("invalid_candidate_count") or 0),
            "valid_bus_count": int(detail.get("valid_bus_count") or 0),
            "valid_main_count": int(detail.get("valid_main_count") or 0),
            "valid_dca_count": int(detail.get("valid_dca_count") or 0),
            "accepted_as_physical_fixture_candidate": candidate_accepted,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "latest_valid_candidate_source_id": detail.get("latest_valid_candidate_source_id"),
            "latest_valid_candidate_captured_at": detail.get("latest_valid_candidate_captured_at"),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "artifact_source_ids": (
                list(artifact_source_ids) if isinstance(artifact_source_ids, list) else []
            ),
            "intake_violations": (
                list(intake_violations) if isinstance(intake_violations, list) else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def matrix_state_readback_contract_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "matrix_state_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_matrix_statuses")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        promotion_families = detail.get("promotion_families_requiring_surface")
        required_control_fields = detail.get("required_control_fields")
        required_eq_fields = detail.get("required_eq_fields")
        required_eq_band_fields = detail.get("required_eq_band_fields")
        required_dynamics_fields = detail.get("required_dynamics_fields")
        required_insert_fields = detail.get("required_insert_fields")
        required_metering_fields = detail.get("required_metering_fields")
        required_coverage_flags = detail.get("required_coverage_flags")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "physical_readback_surface_id": detail.get("physical_readback_surface_id"),
            "physical_readback_action_id": detail.get("physical_readback_action_id"),
            "fixture_physical_gap_declared": detail.get("fixture_physical_gap_declared") is True,
            "remediation_physical_gap_declared": (
                detail.get("remediation_physical_gap_declared") is True
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_matrix_field_count": int(detail.get("required_matrix_field_count") or 0),
            "required_control_fields": (
                list(required_control_fields)
                if isinstance(required_control_fields, list)
                else []
            ),
            "required_eq_fields": (
                list(required_eq_fields) if isinstance(required_eq_fields, list) else []
            ),
            "required_eq_band_fields": (
                list(required_eq_band_fields)
                if isinstance(required_eq_band_fields, list)
                else []
            ),
            "required_dynamics_fields": (
                list(required_dynamics_fields)
                if isinstance(required_dynamics_fields, list)
                else []
            ),
            "required_insert_fields": (
                list(required_insert_fields)
                if isinstance(required_insert_fields, list)
                else []
            ),
            "required_metering_fields": (
                list(required_metering_fields)
                if isinstance(required_metering_fields, list)
                else []
            ),
            "required_coverage_flags": (
                list(required_coverage_flags)
                if isinstance(required_coverage_flags, list)
                else []
            ),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "minimum_matrix_count": int(detail.get("minimum_matrix_count") or 0),
            "minimum_eq_band_count": int(detail.get("minimum_eq_band_count") or 0),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_matrix_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("physical_readback_surface_id") == detail.get("surface_id")
                and detail.get("physical_readback_action_id")
                == detail.get("remediation_action_id")
                and detail.get("fixture_physical_gap_declared") is True
                and detail.get("remediation_physical_gap_declared") is True
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def matrix_state_readback_artifact_intake_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "matrix_state_physical_readback_artifact_intake_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        artifact_search_roots = detail.get("artifact_search_roots")
        blocked_by = detail.get("blocked_by")
        artifact_source_ids = detail.get("artifact_source_ids")
        intake_violations = detail.get("intake_violations")
        promotion_families = detail.get("promotion_families_requiring_surface")
        violations = detail.get("violations")
        source_candidate_claimed = detail.get("accepted_as_physical_fixture_candidate") is True
        operator_review_required = detail.get("operator_review_required") is True
        physical_fixture_gap_remains = detail.get("physical_fixture_gap_remains") is True
        unsafe_claimed = (
            detail.get("fixture_closure_allowed_by_intake") is True
            or detail.get("live_authority_granted") is True
            or detail.get("full_autonomous_ready") is True
        )
        candidate_accepted = (
            result.status == "passed"
            and source_candidate_claimed
            and operator_review_required
            and physical_fixture_gap_remains
            and not unsafe_claimed
        )
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_intake": False,
            "fixture_closure_allowed_by_intake": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "intake_status": detail.get("intake_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "required_contract_method_id": detail.get("required_contract_method_id"),
            "artifact_search_roots": (
                list(artifact_search_roots) if isinstance(artifact_search_roots, list) else []
            ),
            "artifact_count": int(detail.get("artifact_count") or 0),
            "valid_candidate_count": int(detail.get("valid_candidate_count") or 0),
            "invalid_candidate_count": int(detail.get("invalid_candidate_count") or 0),
            "valid_matrix_count": int(detail.get("valid_matrix_count") or 0),
            "accepted_as_physical_fixture_candidate": candidate_accepted,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "latest_valid_candidate_source_id": detail.get("latest_valid_candidate_source_id"),
            "latest_valid_candidate_captured_at": detail.get("latest_valid_candidate_captured_at"),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "artifact_source_ids": (
                list(artifact_source_ids) if isinstance(artifact_source_ids, list) else []
            ),
            "intake_violations": (
                list(intake_violations) if isinstance(intake_violations, list) else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def fx_slots_inserts_readback_contract_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "fx_slots_inserts_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_row_statuses")
        target_types = detail.get("allowed_insert_target_types")
        positions = detail.get("allowed_insert_positions")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        promotion_families = detail.get("promotion_families_requiring_surface")
        required_fx_slot_fields = detail.get("required_fx_slot_fields")
        required_parameter_fields = detail.get("required_parameter_fields")
        required_parameter_safe_range_fields = detail.get(
            "required_parameter_safe_range_fields"
        )
        required_parameter_schema_fields = detail.get("required_parameter_schema_fields")
        required_fx_slot_coverage_flags = detail.get("required_fx_slot_coverage_flags")
        required_insert_fields = detail.get("required_insert_fields")
        required_insert_coverage_flags = detail.get("required_insert_coverage_flags")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "physical_readback_surface_id": detail.get("physical_readback_surface_id"),
            "physical_readback_action_id": detail.get("physical_readback_action_id"),
            "fixture_physical_gap_declared": detail.get("fixture_physical_gap_declared") is True,
            "remediation_physical_gap_declared": (
                detail.get("remediation_physical_gap_declared") is True
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_fx_slot_field_count": int(detail.get("required_fx_slot_field_count") or 0),
            "required_fx_slot_fields": (
                list(required_fx_slot_fields)
                if isinstance(required_fx_slot_fields, list)
                else []
            ),
            "required_parameter_fields": (
                list(required_parameter_fields)
                if isinstance(required_parameter_fields, list)
                else []
            ),
            "required_parameter_safe_range_fields": (
                list(required_parameter_safe_range_fields)
                if isinstance(required_parameter_safe_range_fields, list)
                else []
            ),
            "required_parameter_schema_fields": (
                list(required_parameter_schema_fields)
                if isinstance(required_parameter_schema_fields, list)
                else []
            ),
            "required_fx_slot_coverage_flags": (
                list(required_fx_slot_coverage_flags)
                if isinstance(required_fx_slot_coverage_flags, list)
                else []
            ),
            "required_insert_fields": (
                list(required_insert_fields)
                if isinstance(required_insert_fields, list)
                else []
            ),
            "required_insert_coverage_flags": (
                list(required_insert_coverage_flags)
                if isinstance(required_insert_coverage_flags, list)
                else []
            ),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "minimum_fx_slot_count": int(detail.get("minimum_fx_slot_count") or 0),
            "minimum_fx_parameter_count": int(
                detail.get("minimum_fx_parameter_count") or 0
            ),
            "minimum_insert_count": int(detail.get("minimum_insert_count") or 0),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_row_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "allowed_insert_target_types": (
                list(target_types) if isinstance(target_types, list) else []
            ),
            "allowed_insert_positions": (
                list(positions) if isinstance(positions, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("physical_readback_surface_id") == detail.get("surface_id")
                and detail.get("physical_readback_action_id")
                == detail.get("remediation_action_id")
                and detail.get("fixture_physical_gap_declared") is True
                and detail.get("remediation_physical_gap_declared") is True
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def routing_sends_readback_contract_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "routing_sends_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_row_statuses")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        promotion_families = detail.get("promotion_families_requiring_surface")
        required_static_route_fields = detail.get("required_static_route_fields")
        required_bus_send_fields = detail.get("required_bus_send_fields")
        required_main_send_fields = detail.get("required_main_send_fields")
        required_matrix_send_fields = detail.get("required_matrix_send_fields")
        required_output_route_fields = detail.get("required_output_route_fields")
        required_static_route_coverage_flags = detail.get(
            "required_static_route_coverage_flags"
        )
        required_send_coverage_flags = detail.get("required_send_coverage_flags")
        required_output_route_coverage_flags = detail.get(
            "required_output_route_coverage_flags"
        )
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "physical_readback_surface_id": detail.get("physical_readback_surface_id"),
            "physical_readback_action_id": detail.get("physical_readback_action_id"),
            "fixture_physical_gap_declared": detail.get("fixture_physical_gap_declared") is True,
            "remediation_physical_gap_declared": (
                detail.get("remediation_physical_gap_declared") is True
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_static_route_field_count": int(
                detail.get("required_static_route_field_count") or 0
            ),
            "required_static_route_fields": (
                list(required_static_route_fields)
                if isinstance(required_static_route_fields, list)
                else []
            ),
            "required_bus_send_fields": (
                list(required_bus_send_fields)
                if isinstance(required_bus_send_fields, list)
                else []
            ),
            "required_main_send_fields": (
                list(required_main_send_fields)
                if isinstance(required_main_send_fields, list)
                else []
            ),
            "required_matrix_send_fields": (
                list(required_matrix_send_fields)
                if isinstance(required_matrix_send_fields, list)
                else []
            ),
            "required_output_route_fields": (
                list(required_output_route_fields)
                if isinstance(required_output_route_fields, list)
                else []
            ),
            "required_static_route_coverage_flags": (
                list(required_static_route_coverage_flags)
                if isinstance(required_static_route_coverage_flags, list)
                else []
            ),
            "required_send_coverage_flags": (
                list(required_send_coverage_flags)
                if isinstance(required_send_coverage_flags, list)
                else []
            ),
            "required_output_route_coverage_flags": (
                list(required_output_route_coverage_flags)
                if isinstance(required_output_route_coverage_flags, list)
                else []
            ),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "minimum_static_dante_route_count": int(
                detail.get("minimum_static_dante_route_count") or 0
            ),
            "minimum_channel_bus_send_count": int(
                detail.get("minimum_channel_bus_send_count") or 0
            ),
            "minimum_channel_main_send_count": int(
                detail.get("minimum_channel_main_send_count") or 0
            ),
            "minimum_channel_matrix_send_count": int(
                detail.get("minimum_channel_matrix_send_count") or 0
            ),
            "minimum_output_route_count": int(
                detail.get("minimum_output_route_count") or 0
            ),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_row_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("physical_readback_surface_id") == detail.get("surface_id")
                and detail.get("physical_readback_action_id")
                == detail.get("remediation_action_id")
                and detail.get("fixture_physical_gap_declared") is True
                and detail.get("remediation_physical_gap_declared") is True
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def snapshots_scenes_readback_contract_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "snapshots_scenes_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_row_statuses")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        promotion_families = detail.get("promotion_families_requiring_surface")
        required_active_scene_fields = detail.get("required_active_scene_fields")
        required_snapshot_fields = detail.get("required_snapshot_fields")
        required_coverage_flags = detail.get("required_coverage_flags")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "scene_load_allowed": False,
            "scene_save_allowed": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "physical_readback_surface_id": detail.get("physical_readback_surface_id"),
            "physical_readback_action_id": detail.get("physical_readback_action_id"),
            "fixture_physical_gap_declared": detail.get("fixture_physical_gap_declared") is True,
            "remediation_physical_gap_declared": (
                detail.get("remediation_physical_gap_declared") is True
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_active_scene_fields": (
                list(required_active_scene_fields)
                if isinstance(required_active_scene_fields, list)
                else []
            ),
            "required_snapshot_fields": (
                list(required_snapshot_fields)
                if isinstance(required_snapshot_fields, list)
                else []
            ),
            "required_coverage_flags": (
                list(required_coverage_flags) if isinstance(required_coverage_flags, list) else []
            ),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "minimum_snapshot_count": int(detail.get("minimum_snapshot_count") or 0),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_row_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("physical_readback_surface_id") == detail.get("surface_id")
                and detail.get("physical_readback_action_id")
                == detail.get("remediation_action_id")
                and detail.get("fixture_physical_gap_declared") is True
                and detail.get("remediation_physical_gap_declared") is True
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def user_presets_readback_contract_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "user_presets_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_row_statuses")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        promotion_families = detail.get("promotion_families_requiring_surface")
        required_preset_fields = detail.get("required_preset_fields")
        required_coverage_flags = detail.get("required_coverage_flags")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "preset_recall_allowed": False,
            "preset_store_allowed": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "physical_readback_surface_id": detail.get("physical_readback_surface_id"),
            "physical_readback_action_id": detail.get("physical_readback_action_id"),
            "fixture_physical_gap_declared": detail.get("fixture_physical_gap_declared") is True,
            "remediation_physical_gap_declared": (
                detail.get("remediation_physical_gap_declared") is True
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_preset_fields": (
                list(required_preset_fields)
                if isinstance(required_preset_fields, list)
                else []
            ),
            "required_coverage_flags": (
                list(required_coverage_flags)
                if isinstance(required_coverage_flags, list)
                else []
            ),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "minimum_preset_count": int(detail.get("minimum_preset_count") or 0),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_row_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("physical_readback_surface_id") == detail.get("surface_id")
                and detail.get("physical_readback_action_id")
                == detail.get("remediation_action_id")
                and detail.get("fixture_physical_gap_declared") is True
                and detail.get("remediation_physical_gap_declared") is True
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def dlive_state_readback_contract_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "dlive_state_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_row_statuses")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        promotion_families = detail.get("promotion_families_requiring_surface")
        required_channel_fields = detail.get("required_channel_fields")
        required_midi_state_fields = detail.get("required_midi_state_fields")
        required_parity_fields = detail.get("required_parity_fields")
        required_coverage_flags = detail.get("required_coverage_flags")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "midi_send_allowed": False,
            "tcp_send_allowed": False,
            "dlive_write_allowed": False,
            "dlive_apply_allowed": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "physical_readback_surface_id": detail.get("physical_readback_surface_id"),
            "physical_readback_action_id": detail.get("physical_readback_action_id"),
            "fixture_physical_gap_declared": detail.get("fixture_physical_gap_declared") is True,
            "remediation_physical_gap_declared": (
                detail.get("remediation_physical_gap_declared") is True
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_channel_fields": (
                list(required_channel_fields)
                if isinstance(required_channel_fields, list)
                else []
            ),
            "required_midi_state_fields": (
                list(required_midi_state_fields)
                if isinstance(required_midi_state_fields, list)
                else []
            ),
            "required_parity_fields": (
                list(required_parity_fields)
                if isinstance(required_parity_fields, list)
                else []
            ),
            "required_coverage_flags": (
                list(required_coverage_flags)
                if isinstance(required_coverage_flags, list)
                else []
            ),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "minimum_channel_count": int(detail.get("minimum_channel_count") or 0),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_row_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("physical_readback_surface_id") == detail.get("surface_id")
                and detail.get("physical_readback_action_id")
                == detail.get("remediation_action_id")
                and detail.get("fixture_physical_gap_declared") is True
                and detail.get("remediation_physical_gap_declared") is True
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
    read_capture_plan = read_surface_physical_readback_capture_plan_summary(results)
    raw_mixer_contract = raw_mixer_readback_contract_summary(results)
    raw_mixer_intake = raw_mixer_readback_artifact_intake_summary(results)
    channel_basic_contract = channel_basic_controls_readback_contract_summary(results)
    channel_basic_intake = channel_basic_controls_readback_artifact_intake_summary(results)
    channel_processing_contract = channel_processing_readback_contract_summary(results)
    channel_processing_intake = channel_processing_readback_artifact_intake_summary(results)
    bus_group_main_dca_contract = bus_group_main_dca_readback_contract_summary(results)
    bus_group_main_dca_intake = bus_group_main_dca_readback_artifact_intake_summary(results)
    matrix_state_contract = matrix_state_readback_contract_summary(results)
    matrix_state_intake = matrix_state_readback_artifact_intake_summary(results)
    fx_slots_inserts_contract = fx_slots_inserts_readback_contract_summary(results)
    routing_sends_contract = routing_sends_readback_contract_summary(results)
    snapshots_scenes_contract = snapshots_scenes_readback_contract_summary(results)
    user_presets_contract = user_presets_readback_contract_summary(results)
    dlive_state_contract = dlive_state_readback_contract_summary(results)
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
    if read_capture_plan is not None:
        summary["read_surface_physical_readback_capture_plan"] = read_capture_plan
    if raw_mixer_contract is not None:
        summary["raw_mixer_state_physical_readback_contract"] = raw_mixer_contract
    if raw_mixer_intake is not None:
        summary["raw_mixer_state_physical_readback_artifact_intake"] = raw_mixer_intake
    if channel_basic_contract is not None:
        summary["channel_basic_controls_physical_readback_contract"] = channel_basic_contract
    if channel_basic_intake is not None:
        summary["channel_basic_controls_physical_readback_artifact_intake"] = channel_basic_intake
    if channel_processing_contract is not None:
        summary["channel_processing_physical_readback_contract"] = channel_processing_contract
    if channel_processing_intake is not None:
        summary["channel_processing_physical_readback_artifact_intake"] = (
            channel_processing_intake
        )
    if bus_group_main_dca_contract is not None:
        summary["bus_group_main_dca_physical_readback_contract"] = bus_group_main_dca_contract
    if bus_group_main_dca_intake is not None:
        summary["bus_group_main_dca_physical_readback_artifact_intake"] = (
            bus_group_main_dca_intake
        )
    if matrix_state_contract is not None:
        summary["matrix_state_physical_readback_contract"] = matrix_state_contract
    if matrix_state_intake is not None:
        summary["matrix_state_physical_readback_artifact_intake"] = matrix_state_intake
    if fx_slots_inserts_contract is not None:
        summary["fx_slots_inserts_physical_readback_contract"] = fx_slots_inserts_contract
    if routing_sends_contract is not None:
        summary["routing_sends_physical_readback_contract"] = routing_sends_contract
    if snapshots_scenes_contract is not None:
        summary["snapshots_scenes_physical_readback_contract"] = snapshots_scenes_contract
    if user_presets_contract is not None:
        summary["user_presets_physical_readback_contract"] = user_presets_contract
    if dlive_state_contract is not None:
        summary["dlive_state_physical_readback_contract"] = dlive_state_contract
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
                "- evidence_requirement_ids_by_action_family: "
                f"`{json.dumps(gap_rollup['evidence_requirement_ids_by_action_family'], sort_keys=True)}`",
                "- next_required_evidence_by_action_family: "
                f"`{json.dumps(gap_rollup['next_required_evidence_by_action_family'], sort_keys=True)}`",
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
    if read_capture_plan is not None:
        lines.extend(
            [
                "",
                "## Read Surface Physical Readback Capture Plan",
                "",
                f"- schema_check: `{read_capture_plan['status']}`",
                f"- read_only: `{str(read_capture_plan['read_only']).lower()}`",
                f"- live_write: `{str(read_capture_plan['live_write']).lower()}`",
                f"- mutates_console_state: `{str(read_capture_plan['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(read_capture_plan['queries_console']).lower()}`",
                f"- executes_tests: `{str(read_capture_plan['executes_tests']).lower()}`",
                f"- operator_required: `{str(read_capture_plan['operator_required']).lower()}`",
                "- execution_allowed_by_plan: "
                f"`{str(read_capture_plan['execution_allowed_by_plan']).lower()}`",
                "- live_authority_granted: "
                f"`{str(read_capture_plan['live_authority_granted']).lower()}`",
                f"- full_autonomous_ready: `{str(read_capture_plan['full_autonomous_ready']).lower()}`",
                f"- plan_status: `{read_capture_plan['plan_status']}`",
                f"- capture_step_count: `{read_capture_plan['capture_step_count']}`",
                f"- physical_gap_surface_count: `{read_capture_plan['physical_gap_surface_count']}`",
                "- physical_fixture_missing_surface_ids: "
                f"`{json.dumps(read_capture_plan['physical_fixture_missing_surface_ids'])}`",
                "- next_physical_readback_surface_id: "
                f"`{read_capture_plan['next_physical_readback_surface_id']}`",
                "- next_physical_readback_action_id: "
                f"`{read_capture_plan['next_physical_readback_action_id']}`",
                f"- target_action_consistency: `{str(read_capture_plan['target_action_consistency']).lower()}`",
                "- required_artifact_contracts: "
                f"`{json.dumps(read_capture_plan['required_artifact_contracts'])}`",
                f"- blocked_by: `{json.dumps(read_capture_plan['blocked_by'])}`",
                f"- step_surface_id: `{read_capture_plan['step_surface_id']}`",
                f"- step_remediation_action_id: `{read_capture_plan['step_remediation_action_id']}`",
                f"- violations: `{json.dumps(read_capture_plan['violations'])}`",
            ]
        )
    if raw_mixer_contract is not None:
        lines.extend(
            [
                "",
                "## Raw Mixer State Physical Readback Contract",
                "",
                f"- schema_check: `{raw_mixer_contract['status']}`",
                f"- read_only: `{str(raw_mixer_contract['read_only']).lower()}`",
                f"- live_write: `{str(raw_mixer_contract['live_write']).lower()}`",
                f"- mutates_console_state: `{str(raw_mixer_contract['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(raw_mixer_contract['queries_console']).lower()}`",
                f"- executes_tests: `{str(raw_mixer_contract['executes_tests']).lower()}`",
                f"- live_widening_allowed: `{str(raw_mixer_contract['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(raw_mixer_contract['full_autonomous_ready']).lower()}`",
                f"- schema_id: `{raw_mixer_contract['schema_id']}`",
                f"- contract_status: `{raw_mixer_contract['contract_status']}`",
                f"- surface_id: `{raw_mixer_contract['surface_id']}`",
                f"- remediation_action_id: `{raw_mixer_contract['remediation_action_id']}`",
                "- next_physical_readback_surface_id: "
                f"`{raw_mixer_contract['next_physical_readback_surface_id']}`",
                "- next_physical_readback_action_id: "
                f"`{raw_mixer_contract['next_physical_readback_action_id']}`",
                f"- artifact_required: `{str(raw_mixer_contract['artifact_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(raw_mixer_contract['physical_fixture_gap_remains']).lower()}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(raw_mixer_contract['accepted_as_physical_fixture_candidate']).lower()}`",
                "- live_authority_granted: "
                f"`{str(raw_mixer_contract['live_authority_granted']).lower()}`",
                "- required_top_level_field_count: "
                f"`{raw_mixer_contract['required_top_level_field_count']}`",
                f"- required_row_field_count: `{raw_mixer_contract['required_row_field_count']}`",
                "- allowed_sources: "
                f"`{json.dumps(raw_mixer_contract['allowed_sources'])}`",
                "- accepted_row_statuses: "
                f"`{json.dumps(raw_mixer_contract['accepted_row_statuses'])}`",
                "- forbidden_mutation_evidence_fields: "
                f"`{json.dumps(raw_mixer_contract['forbidden_mutation_evidence_fields'])}`",
                f"- target_consistency: `{str(raw_mixer_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(raw_mixer_contract['violations'])}`",
            ]
        )
    if raw_mixer_intake is not None:
        lines.extend(
            [
                "",
                "## Raw Mixer State Physical Readback Artifact Intake",
                "",
                f"- schema_check: `{raw_mixer_intake['status']}`",
                f"- read_only: `{str(raw_mixer_intake['read_only']).lower()}`",
                f"- live_write: `{str(raw_mixer_intake['live_write']).lower()}`",
                f"- mutates_console_state: `{str(raw_mixer_intake['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(raw_mixer_intake['queries_console']).lower()}`",
                f"- executes_tests: `{str(raw_mixer_intake['executes_tests']).lower()}`",
                f"- operator_required: `{str(raw_mixer_intake['operator_required']).lower()}`",
                "- execution_allowed_by_intake: "
                f"`{str(raw_mixer_intake['execution_allowed_by_intake']).lower()}`",
                "- fixture_closure_allowed_by_intake: "
                f"`{str(raw_mixer_intake['fixture_closure_allowed_by_intake']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(raw_mixer_intake['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(raw_mixer_intake['full_autonomous_ready']).lower()}`",
                f"- intake_status: `{raw_mixer_intake['intake_status']}`",
                f"- schema_id: `{raw_mixer_intake['schema_id']}`",
                f"- surface_id: `{raw_mixer_intake['surface_id']}`",
                f"- remediation_action_id: `{raw_mixer_intake['remediation_action_id']}`",
                "- required_contract_method_id: "
                f"`{raw_mixer_intake['required_contract_method_id']}`",
                f"- artifact_count: `{raw_mixer_intake['artifact_count']}`",
                f"- valid_candidate_count: `{raw_mixer_intake['valid_candidate_count']}`",
                f"- invalid_candidate_count: `{raw_mixer_intake['invalid_candidate_count']}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(raw_mixer_intake['accepted_as_physical_fixture_candidate']).lower()}`",
                "- operator_review_required: "
                f"`{str(raw_mixer_intake['operator_review_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(raw_mixer_intake['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(raw_mixer_intake['live_authority_granted']).lower()}`",
                "- artifact_search_roots: "
                f"`{json.dumps(raw_mixer_intake['artifact_search_roots'])}`",
                "- artifact_source_ids: "
                f"`{json.dumps(raw_mixer_intake['artifact_source_ids'])}`",
                f"- blocked_by: `{json.dumps(raw_mixer_intake['blocked_by'])}`",
                "- intake_violations: "
                f"`{json.dumps(raw_mixer_intake['intake_violations'])}`",
                f"- violations: `{json.dumps(raw_mixer_intake['violations'])}`",
            ]
        )
    if channel_basic_contract is not None:
        lines.extend(
            [
                "",
                "## Channel Basic Controls Physical Readback Contract",
                "",
                f"- schema_check: `{channel_basic_contract['status']}`",
                f"- read_only: `{str(channel_basic_contract['read_only']).lower()}`",
                f"- live_write: `{str(channel_basic_contract['live_write']).lower()}`",
                f"- mutates_console_state: `{str(channel_basic_contract['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(channel_basic_contract['queries_console']).lower()}`",
                f"- executes_tests: `{str(channel_basic_contract['executes_tests']).lower()}`",
                f"- live_widening_allowed: `{str(channel_basic_contract['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(channel_basic_contract['full_autonomous_ready']).lower()}`",
                f"- schema_id: `{channel_basic_contract['schema_id']}`",
                f"- contract_status: `{channel_basic_contract['contract_status']}`",
                f"- surface_id: `{channel_basic_contract['surface_id']}`",
                f"- remediation_action_id: `{channel_basic_contract['remediation_action_id']}`",
                "- physical_readback_surface_id: "
                f"`{channel_basic_contract['physical_readback_surface_id']}`",
                "- physical_readback_action_id: "
                f"`{channel_basic_contract['physical_readback_action_id']}`",
                "- fixture_physical_gap_declared: "
                f"`{str(channel_basic_contract['fixture_physical_gap_declared']).lower()}`",
                "- remediation_physical_gap_declared: "
                f"`{str(channel_basic_contract['remediation_physical_gap_declared']).lower()}`",
                f"- artifact_required: `{str(channel_basic_contract['artifact_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(channel_basic_contract['physical_fixture_gap_remains']).lower()}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(channel_basic_contract['accepted_as_physical_fixture_candidate']).lower()}`",
                "- live_authority_granted: "
                f"`{str(channel_basic_contract['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(channel_basic_contract['promotion_families_requiring_surface'])}`",
                "- required_top_level_field_count: "
                f"`{channel_basic_contract['required_top_level_field_count']}`",
                "- required_channel_field_count: "
                f"`{channel_basic_contract['required_channel_field_count']}`",
                "- required_identity_fields: "
                f"`{json.dumps(channel_basic_contract['required_identity_fields'])}`",
                "- required_control_fields: "
                f"`{json.dumps(channel_basic_contract['required_control_fields'])}`",
                "- required_coverage_flags: "
                f"`{json.dumps(channel_basic_contract['required_coverage_flags'])}`",
                "- allowed_sources: "
                f"`{json.dumps(channel_basic_contract['allowed_sources'])}`",
                "- accepted_channel_statuses: "
                f"`{json.dumps(channel_basic_contract['accepted_channel_statuses'])}`",
                "- forbidden_mutation_evidence_fields: "
                f"`{json.dumps(channel_basic_contract['forbidden_mutation_evidence_fields'])}`",
                f"- target_consistency: `{str(channel_basic_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(channel_basic_contract['violations'])}`",
            ]
        )
    if channel_basic_intake is not None:
        lines.extend(
            [
                "",
                "## Channel Basic Controls Physical Readback Artifact Intake",
                "",
                f"- schema_check: `{channel_basic_intake['status']}`",
                f"- read_only: `{str(channel_basic_intake['read_only']).lower()}`",
                f"- live_write: `{str(channel_basic_intake['live_write']).lower()}`",
                f"- mutates_console_state: `{str(channel_basic_intake['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(channel_basic_intake['queries_console']).lower()}`",
                f"- executes_tests: `{str(channel_basic_intake['executes_tests']).lower()}`",
                f"- operator_required: `{str(channel_basic_intake['operator_required']).lower()}`",
                "- execution_allowed_by_intake: "
                f"`{str(channel_basic_intake['execution_allowed_by_intake']).lower()}`",
                "- fixture_closure_allowed_by_intake: "
                f"`{str(channel_basic_intake['fixture_closure_allowed_by_intake']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(channel_basic_intake['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(channel_basic_intake['full_autonomous_ready']).lower()}`",
                f"- intake_status: `{channel_basic_intake['intake_status']}`",
                f"- schema_id: `{channel_basic_intake['schema_id']}`",
                f"- surface_id: `{channel_basic_intake['surface_id']}`",
                f"- remediation_action_id: `{channel_basic_intake['remediation_action_id']}`",
                "- required_contract_method_id: "
                f"`{channel_basic_intake['required_contract_method_id']}`",
                f"- artifact_count: `{channel_basic_intake['artifact_count']}`",
                f"- valid_candidate_count: `{channel_basic_intake['valid_candidate_count']}`",
                f"- invalid_candidate_count: `{channel_basic_intake['invalid_candidate_count']}`",
                f"- valid_channel_count: `{channel_basic_intake['valid_channel_count']}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(channel_basic_intake['accepted_as_physical_fixture_candidate']).lower()}`",
                "- operator_review_required: "
                f"`{str(channel_basic_intake['operator_review_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(channel_basic_intake['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(channel_basic_intake['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(channel_basic_intake['promotion_families_requiring_surface'])}`",
                "- artifact_search_roots: "
                f"`{json.dumps(channel_basic_intake['artifact_search_roots'])}`",
                "- artifact_source_ids: "
                f"`{json.dumps(channel_basic_intake['artifact_source_ids'])}`",
                f"- blocked_by: `{json.dumps(channel_basic_intake['blocked_by'])}`",
                "- intake_violations: "
                f"`{json.dumps(channel_basic_intake['intake_violations'])}`",
                f"- violations: `{json.dumps(channel_basic_intake['violations'])}`",
            ]
        )
    if channel_processing_contract is not None:
        lines.extend(
            [
                "",
                "## Channel Processing Physical Readback Contract",
                "",
                f"- schema_check: `{channel_processing_contract['status']}`",
                f"- read_only: `{str(channel_processing_contract['read_only']).lower()}`",
                f"- live_write: `{str(channel_processing_contract['live_write']).lower()}`",
                f"- mutates_console_state: `{str(channel_processing_contract['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(channel_processing_contract['queries_console']).lower()}`",
                f"- executes_tests: `{str(channel_processing_contract['executes_tests']).lower()}`",
                f"- live_widening_allowed: `{str(channel_processing_contract['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(channel_processing_contract['full_autonomous_ready']).lower()}`",
                f"- schema_id: `{channel_processing_contract['schema_id']}`",
                f"- contract_status: `{channel_processing_contract['contract_status']}`",
                f"- surface_id: `{channel_processing_contract['surface_id']}`",
                f"- remediation_action_id: `{channel_processing_contract['remediation_action_id']}`",
                "- physical_readback_surface_id: "
                f"`{channel_processing_contract['physical_readback_surface_id']}`",
                "- physical_readback_action_id: "
                f"`{channel_processing_contract['physical_readback_action_id']}`",
                "- fixture_physical_gap_declared: "
                f"`{str(channel_processing_contract['fixture_physical_gap_declared']).lower()}`",
                "- remediation_physical_gap_declared: "
                f"`{str(channel_processing_contract['remediation_physical_gap_declared']).lower()}`",
                f"- artifact_required: `{str(channel_processing_contract['artifact_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(channel_processing_contract['physical_fixture_gap_remains']).lower()}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(channel_processing_contract['accepted_as_physical_fixture_candidate']).lower()}`",
                "- live_authority_granted: "
                f"`{str(channel_processing_contract['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(channel_processing_contract['promotion_families_requiring_surface'])}`",
                "- required_top_level_field_count: "
                f"`{channel_processing_contract['required_top_level_field_count']}`",
                "- required_channel_field_count: "
                f"`{channel_processing_contract['required_channel_field_count']}`",
                "- required_hpf_fields: "
                f"`{json.dumps(channel_processing_contract['required_hpf_fields'])}`",
                "- required_eq_fields: "
                f"`{json.dumps(channel_processing_contract['required_eq_fields'])}`",
                "- required_eq_band_fields: "
                f"`{json.dumps(channel_processing_contract['required_eq_band_fields'])}`",
                "- required_compressor_fields: "
                f"`{json.dumps(channel_processing_contract['required_compressor_fields'])}`",
                "- required_gate_fields: "
                f"`{json.dumps(channel_processing_contract['required_gate_fields'])}`",
                "- required_coverage_flags: "
                f"`{json.dumps(channel_processing_contract['required_coverage_flags'])}`",
                f"- minimum_eq_band_count: `{channel_processing_contract['minimum_eq_band_count']}`",
                "- allowed_sources: "
                f"`{json.dumps(channel_processing_contract['allowed_sources'])}`",
                "- accepted_channel_statuses: "
                f"`{json.dumps(channel_processing_contract['accepted_channel_statuses'])}`",
                "- forbidden_mutation_evidence_fields: "
                f"`{json.dumps(channel_processing_contract['forbidden_mutation_evidence_fields'])}`",
                f"- target_consistency: `{str(channel_processing_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(channel_processing_contract['violations'])}`",
            ]
        )
    if channel_processing_intake is not None:
        lines.extend(
            [
                "",
                "## Channel Processing Physical Readback Artifact Intake",
                "",
                f"- schema_check: `{channel_processing_intake['status']}`",
                f"- read_only: `{str(channel_processing_intake['read_only']).lower()}`",
                f"- live_write: `{str(channel_processing_intake['live_write']).lower()}`",
                f"- mutates_console_state: `{str(channel_processing_intake['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(channel_processing_intake['queries_console']).lower()}`",
                f"- executes_tests: `{str(channel_processing_intake['executes_tests']).lower()}`",
                f"- operator_required: `{str(channel_processing_intake['operator_required']).lower()}`",
                "- execution_allowed_by_intake: "
                f"`{str(channel_processing_intake['execution_allowed_by_intake']).lower()}`",
                "- fixture_closure_allowed_by_intake: "
                f"`{str(channel_processing_intake['fixture_closure_allowed_by_intake']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(channel_processing_intake['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(channel_processing_intake['full_autonomous_ready']).lower()}`",
                f"- intake_status: `{channel_processing_intake['intake_status']}`",
                f"- schema_id: `{channel_processing_intake['schema_id']}`",
                f"- surface_id: `{channel_processing_intake['surface_id']}`",
                f"- remediation_action_id: `{channel_processing_intake['remediation_action_id']}`",
                "- required_contract_method_id: "
                f"`{channel_processing_intake['required_contract_method_id']}`",
                f"- artifact_count: `{channel_processing_intake['artifact_count']}`",
                f"- valid_candidate_count: `{channel_processing_intake['valid_candidate_count']}`",
                f"- invalid_candidate_count: `{channel_processing_intake['invalid_candidate_count']}`",
                f"- valid_channel_count: `{channel_processing_intake['valid_channel_count']}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(channel_processing_intake['accepted_as_physical_fixture_candidate']).lower()}`",
                "- operator_review_required: "
                f"`{str(channel_processing_intake['operator_review_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(channel_processing_intake['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(channel_processing_intake['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(channel_processing_intake['promotion_families_requiring_surface'])}`",
                "- artifact_search_roots: "
                f"`{json.dumps(channel_processing_intake['artifact_search_roots'])}`",
                "- artifact_source_ids: "
                f"`{json.dumps(channel_processing_intake['artifact_source_ids'])}`",
                f"- blocked_by: `{json.dumps(channel_processing_intake['blocked_by'])}`",
                "- intake_violations: "
                f"`{json.dumps(channel_processing_intake['intake_violations'])}`",
                f"- violations: `{json.dumps(channel_processing_intake['violations'])}`",
            ]
        )
    if bus_group_main_dca_contract is not None:
        lines.extend(
            [
                "",
                "## Bus Group Main DCA Physical Readback Contract",
                "",
                f"- schema_check: `{bus_group_main_dca_contract['status']}`",
                f"- read_only: `{str(bus_group_main_dca_contract['read_only']).lower()}`",
                f"- live_write: `{str(bus_group_main_dca_contract['live_write']).lower()}`",
                f"- mutates_console_state: `{str(bus_group_main_dca_contract['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(bus_group_main_dca_contract['queries_console']).lower()}`",
                f"- executes_tests: `{str(bus_group_main_dca_contract['executes_tests']).lower()}`",
                f"- live_widening_allowed: `{str(bus_group_main_dca_contract['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(bus_group_main_dca_contract['full_autonomous_ready']).lower()}`",
                f"- schema_id: `{bus_group_main_dca_contract['schema_id']}`",
                f"- contract_status: `{bus_group_main_dca_contract['contract_status']}`",
                f"- surface_id: `{bus_group_main_dca_contract['surface_id']}`",
                f"- remediation_action_id: `{bus_group_main_dca_contract['remediation_action_id']}`",
                "- physical_readback_surface_id: "
                f"`{bus_group_main_dca_contract['physical_readback_surface_id']}`",
                "- physical_readback_action_id: "
                f"`{bus_group_main_dca_contract['physical_readback_action_id']}`",
                "- fixture_physical_gap_declared: "
                f"`{str(bus_group_main_dca_contract['fixture_physical_gap_declared']).lower()}`",
                "- remediation_physical_gap_declared: "
                f"`{str(bus_group_main_dca_contract['remediation_physical_gap_declared']).lower()}`",
                f"- artifact_required: `{str(bus_group_main_dca_contract['artifact_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(bus_group_main_dca_contract['physical_fixture_gap_remains']).lower()}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(bus_group_main_dca_contract['accepted_as_physical_fixture_candidate']).lower()}`",
                "- live_authority_granted: "
                f"`{str(bus_group_main_dca_contract['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(bus_group_main_dca_contract['promotion_families_requiring_surface'])}`",
                "- required_top_level_field_count: "
                f"`{bus_group_main_dca_contract['required_top_level_field_count']}`",
                "- required_bus_field_count: "
                f"`{bus_group_main_dca_contract['required_bus_field_count']}`",
                "- required_main_field_count: "
                f"`{bus_group_main_dca_contract['required_main_field_count']}`",
                "- required_dca_field_count: "
                f"`{bus_group_main_dca_contract['required_dca_field_count']}`",
                "- required_bus_control_fields: "
                f"`{json.dumps(bus_group_main_dca_contract['required_bus_control_fields'])}`",
                "- required_main_control_fields: "
                f"`{json.dumps(bus_group_main_dca_contract['required_main_control_fields'])}`",
                "- required_dca_control_fields: "
                f"`{json.dumps(bus_group_main_dca_contract['required_dca_control_fields'])}`",
                "- required_bus_assignment_fields: "
                f"`{json.dumps(bus_group_main_dca_contract['required_bus_assignment_fields'])}`",
                "- required_dca_assignment_fields: "
                f"`{json.dumps(bus_group_main_dca_contract['required_dca_assignment_fields'])}`",
                "- required_metering_fields: "
                f"`{json.dumps(bus_group_main_dca_contract['required_metering_fields'])}`",
                "- required_bus_coverage_flags: "
                f"`{json.dumps(bus_group_main_dca_contract['required_bus_coverage_flags'])}`",
                "- required_main_coverage_flags: "
                f"`{json.dumps(bus_group_main_dca_contract['required_main_coverage_flags'])}`",
                "- required_dca_coverage_flags: "
                f"`{json.dumps(bus_group_main_dca_contract['required_dca_coverage_flags'])}`",
                f"- minimum_bus_count: `{bus_group_main_dca_contract['minimum_bus_count']}`",
                f"- minimum_main_count: `{bus_group_main_dca_contract['minimum_main_count']}`",
                f"- minimum_dca_count: `{bus_group_main_dca_contract['minimum_dca_count']}`",
                "- allowed_sources: "
                f"`{json.dumps(bus_group_main_dca_contract['allowed_sources'])}`",
                "- accepted_surface_statuses: "
                f"`{json.dumps(bus_group_main_dca_contract['accepted_surface_statuses'])}`",
                "- forbidden_mutation_evidence_fields: "
                f"`{json.dumps(bus_group_main_dca_contract['forbidden_mutation_evidence_fields'])}`",
                f"- target_consistency: `{str(bus_group_main_dca_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(bus_group_main_dca_contract['violations'])}`",
            ]
        )
    if bus_group_main_dca_intake is not None:
        lines.extend(
            [
                "",
                "## Bus Group Main DCA Physical Readback Artifact Intake",
                "",
                f"- schema_check: `{bus_group_main_dca_intake['status']}`",
                f"- read_only: `{str(bus_group_main_dca_intake['read_only']).lower()}`",
                f"- live_write: `{str(bus_group_main_dca_intake['live_write']).lower()}`",
                f"- mutates_console_state: `{str(bus_group_main_dca_intake['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(bus_group_main_dca_intake['queries_console']).lower()}`",
                f"- executes_tests: `{str(bus_group_main_dca_intake['executes_tests']).lower()}`",
                f"- operator_required: `{str(bus_group_main_dca_intake['operator_required']).lower()}`",
                "- execution_allowed_by_intake: "
                f"`{str(bus_group_main_dca_intake['execution_allowed_by_intake']).lower()}`",
                "- fixture_closure_allowed_by_intake: "
                f"`{str(bus_group_main_dca_intake['fixture_closure_allowed_by_intake']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(bus_group_main_dca_intake['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(bus_group_main_dca_intake['full_autonomous_ready']).lower()}`",
                f"- intake_status: `{bus_group_main_dca_intake['intake_status']}`",
                f"- schema_id: `{bus_group_main_dca_intake['schema_id']}`",
                f"- surface_id: `{bus_group_main_dca_intake['surface_id']}`",
                f"- remediation_action_id: `{bus_group_main_dca_intake['remediation_action_id']}`",
                "- required_contract_method_id: "
                f"`{bus_group_main_dca_intake['required_contract_method_id']}`",
                f"- artifact_count: `{bus_group_main_dca_intake['artifact_count']}`",
                f"- valid_candidate_count: `{bus_group_main_dca_intake['valid_candidate_count']}`",
                f"- invalid_candidate_count: `{bus_group_main_dca_intake['invalid_candidate_count']}`",
                f"- valid_bus_count: `{bus_group_main_dca_intake['valid_bus_count']}`",
                f"- valid_main_count: `{bus_group_main_dca_intake['valid_main_count']}`",
                f"- valid_dca_count: `{bus_group_main_dca_intake['valid_dca_count']}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(bus_group_main_dca_intake['accepted_as_physical_fixture_candidate']).lower()}`",
                "- operator_review_required: "
                f"`{str(bus_group_main_dca_intake['operator_review_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(bus_group_main_dca_intake['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(bus_group_main_dca_intake['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(bus_group_main_dca_intake['promotion_families_requiring_surface'])}`",
                "- artifact_search_roots: "
                f"`{json.dumps(bus_group_main_dca_intake['artifact_search_roots'])}`",
                "- artifact_source_ids: "
                f"`{json.dumps(bus_group_main_dca_intake['artifact_source_ids'])}`",
                f"- blocked_by: `{json.dumps(bus_group_main_dca_intake['blocked_by'])}`",
                "- intake_violations: "
                f"`{json.dumps(bus_group_main_dca_intake['intake_violations'])}`",
                f"- violations: `{json.dumps(bus_group_main_dca_intake['violations'])}`",
            ]
        )
    if matrix_state_contract is not None:
        lines.extend(
            [
                "",
                "## Matrix State Physical Readback Contract",
                "",
                f"- schema_check: `{matrix_state_contract['status']}`",
                f"- read_only: `{str(matrix_state_contract['read_only']).lower()}`",
                f"- live_write: `{str(matrix_state_contract['live_write']).lower()}`",
                f"- mutates_console_state: `{str(matrix_state_contract['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(matrix_state_contract['queries_console']).lower()}`",
                f"- executes_tests: `{str(matrix_state_contract['executes_tests']).lower()}`",
                f"- live_widening_allowed: `{str(matrix_state_contract['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(matrix_state_contract['full_autonomous_ready']).lower()}`",
                f"- schema_id: `{matrix_state_contract['schema_id']}`",
                f"- contract_status: `{matrix_state_contract['contract_status']}`",
                f"- surface_id: `{matrix_state_contract['surface_id']}`",
                f"- remediation_action_id: `{matrix_state_contract['remediation_action_id']}`",
                "- physical_readback_surface_id: "
                f"`{matrix_state_contract['physical_readback_surface_id']}`",
                "- physical_readback_action_id: "
                f"`{matrix_state_contract['physical_readback_action_id']}`",
                "- fixture_physical_gap_declared: "
                f"`{str(matrix_state_contract['fixture_physical_gap_declared']).lower()}`",
                "- remediation_physical_gap_declared: "
                f"`{str(matrix_state_contract['remediation_physical_gap_declared']).lower()}`",
                f"- artifact_required: `{str(matrix_state_contract['artifact_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(matrix_state_contract['physical_fixture_gap_remains']).lower()}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(matrix_state_contract['accepted_as_physical_fixture_candidate']).lower()}`",
                "- live_authority_granted: "
                f"`{str(matrix_state_contract['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(matrix_state_contract['promotion_families_requiring_surface'])}`",
                "- required_top_level_field_count: "
                f"`{matrix_state_contract['required_top_level_field_count']}`",
                "- required_matrix_field_count: "
                f"`{matrix_state_contract['required_matrix_field_count']}`",
                "- required_control_fields: "
                f"`{json.dumps(matrix_state_contract['required_control_fields'])}`",
                "- required_eq_fields: "
                f"`{json.dumps(matrix_state_contract['required_eq_fields'])}`",
                "- required_eq_band_fields: "
                f"`{json.dumps(matrix_state_contract['required_eq_band_fields'])}`",
                "- required_dynamics_fields: "
                f"`{json.dumps(matrix_state_contract['required_dynamics_fields'])}`",
                "- required_insert_fields: "
                f"`{json.dumps(matrix_state_contract['required_insert_fields'])}`",
                "- required_metering_fields: "
                f"`{json.dumps(matrix_state_contract['required_metering_fields'])}`",
                "- required_coverage_flags: "
                f"`{json.dumps(matrix_state_contract['required_coverage_flags'])}`",
                f"- minimum_matrix_count: `{matrix_state_contract['minimum_matrix_count']}`",
                f"- minimum_eq_band_count: `{matrix_state_contract['minimum_eq_band_count']}`",
                "- allowed_sources: "
                f"`{json.dumps(matrix_state_contract['allowed_sources'])}`",
                "- accepted_matrix_statuses: "
                f"`{json.dumps(matrix_state_contract['accepted_matrix_statuses'])}`",
                "- forbidden_mutation_evidence_fields: "
                f"`{json.dumps(matrix_state_contract['forbidden_mutation_evidence_fields'])}`",
                f"- target_consistency: `{str(matrix_state_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(matrix_state_contract['violations'])}`",
            ]
        )
    if matrix_state_intake is not None:
        lines.extend(
            [
                "",
                "## Matrix State Physical Readback Artifact Intake",
                "",
                f"- schema_check: `{matrix_state_intake['status']}`",
                f"- read_only: `{str(matrix_state_intake['read_only']).lower()}`",
                f"- live_write: `{str(matrix_state_intake['live_write']).lower()}`",
                f"- mutates_console_state: `{str(matrix_state_intake['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(matrix_state_intake['queries_console']).lower()}`",
                f"- executes_tests: `{str(matrix_state_intake['executes_tests']).lower()}`",
                f"- operator_required: `{str(matrix_state_intake['operator_required']).lower()}`",
                "- execution_allowed_by_intake: "
                f"`{str(matrix_state_intake['execution_allowed_by_intake']).lower()}`",
                "- fixture_closure_allowed_by_intake: "
                f"`{str(matrix_state_intake['fixture_closure_allowed_by_intake']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(matrix_state_intake['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(matrix_state_intake['full_autonomous_ready']).lower()}`",
                f"- intake_status: `{matrix_state_intake['intake_status']}`",
                f"- schema_id: `{matrix_state_intake['schema_id']}`",
                f"- surface_id: `{matrix_state_intake['surface_id']}`",
                f"- remediation_action_id: `{matrix_state_intake['remediation_action_id']}`",
                "- required_contract_method_id: "
                f"`{matrix_state_intake['required_contract_method_id']}`",
                f"- artifact_count: `{matrix_state_intake['artifact_count']}`",
                f"- valid_candidate_count: `{matrix_state_intake['valid_candidate_count']}`",
                f"- invalid_candidate_count: `{matrix_state_intake['invalid_candidate_count']}`",
                f"- valid_matrix_count: `{matrix_state_intake['valid_matrix_count']}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(matrix_state_intake['accepted_as_physical_fixture_candidate']).lower()}`",
                "- operator_review_required: "
                f"`{str(matrix_state_intake['operator_review_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(matrix_state_intake['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(matrix_state_intake['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(matrix_state_intake['promotion_families_requiring_surface'])}`",
                "- artifact_search_roots: "
                f"`{json.dumps(matrix_state_intake['artifact_search_roots'])}`",
                "- artifact_source_ids: "
                f"`{json.dumps(matrix_state_intake['artifact_source_ids'])}`",
                f"- blocked_by: `{json.dumps(matrix_state_intake['blocked_by'])}`",
                "- intake_violations: "
                f"`{json.dumps(matrix_state_intake['intake_violations'])}`",
                f"- violations: `{json.dumps(matrix_state_intake['violations'])}`",
            ]
        )
    if fx_slots_inserts_contract is not None:
        lines.extend(
            [
                "",
                "## FX Slots Inserts Physical Readback Contract",
                "",
                f"- schema_check: `{fx_slots_inserts_contract['status']}`",
                f"- read_only: `{str(fx_slots_inserts_contract['read_only']).lower()}`",
                f"- live_write: `{str(fx_slots_inserts_contract['live_write']).lower()}`",
                "- mutates_console_state: "
                f"`{str(fx_slots_inserts_contract['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(fx_slots_inserts_contract['queries_console']).lower()}`",
                f"- executes_tests: `{str(fx_slots_inserts_contract['executes_tests']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(fx_slots_inserts_contract['live_widening_allowed']).lower()}`",
                "- full_autonomous_ready: "
                f"`{str(fx_slots_inserts_contract['full_autonomous_ready']).lower()}`",
                f"- schema_id: `{fx_slots_inserts_contract['schema_id']}`",
                f"- contract_status: `{fx_slots_inserts_contract['contract_status']}`",
                f"- surface_id: `{fx_slots_inserts_contract['surface_id']}`",
                "- remediation_action_id: "
                f"`{fx_slots_inserts_contract['remediation_action_id']}`",
                "- physical_readback_surface_id: "
                f"`{fx_slots_inserts_contract['physical_readback_surface_id']}`",
                "- physical_readback_action_id: "
                f"`{fx_slots_inserts_contract['physical_readback_action_id']}`",
                "- fixture_physical_gap_declared: "
                f"`{str(fx_slots_inserts_contract['fixture_physical_gap_declared']).lower()}`",
                "- remediation_physical_gap_declared: "
                f"`{str(fx_slots_inserts_contract['remediation_physical_gap_declared']).lower()}`",
                "- artifact_required: "
                f"`{str(fx_slots_inserts_contract['artifact_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(fx_slots_inserts_contract['physical_fixture_gap_remains']).lower()}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(fx_slots_inserts_contract['accepted_as_physical_fixture_candidate']).lower()}`",
                "- live_authority_granted: "
                f"`{str(fx_slots_inserts_contract['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(fx_slots_inserts_contract['promotion_families_requiring_surface'])}`",
                "- required_top_level_field_count: "
                f"`{fx_slots_inserts_contract['required_top_level_field_count']}`",
                "- required_fx_slot_field_count: "
                f"`{fx_slots_inserts_contract['required_fx_slot_field_count']}`",
                "- required_fx_slot_fields: "
                f"`{json.dumps(fx_slots_inserts_contract['required_fx_slot_fields'])}`",
                "- required_parameter_fields: "
                f"`{json.dumps(fx_slots_inserts_contract['required_parameter_fields'])}`",
                "- required_parameter_safe_range_fields: "
                f"`{json.dumps(fx_slots_inserts_contract['required_parameter_safe_range_fields'])}`",
                "- required_parameter_schema_fields: "
                f"`{json.dumps(fx_slots_inserts_contract['required_parameter_schema_fields'])}`",
                "- required_fx_slot_coverage_flags: "
                f"`{json.dumps(fx_slots_inserts_contract['required_fx_slot_coverage_flags'])}`",
                "- required_insert_fields: "
                f"`{json.dumps(fx_slots_inserts_contract['required_insert_fields'])}`",
                "- required_insert_coverage_flags: "
                f"`{json.dumps(fx_slots_inserts_contract['required_insert_coverage_flags'])}`",
                "- required_false_safety_flags: "
                f"`{json.dumps(fx_slots_inserts_contract['required_false_safety_flags'])}`",
                f"- minimum_fx_slot_count: `{fx_slots_inserts_contract['minimum_fx_slot_count']}`",
                "- minimum_fx_parameter_count: "
                f"`{fx_slots_inserts_contract['minimum_fx_parameter_count']}`",
                f"- minimum_insert_count: `{fx_slots_inserts_contract['minimum_insert_count']}`",
                "- allowed_sources: "
                f"`{json.dumps(fx_slots_inserts_contract['allowed_sources'])}`",
                "- accepted_row_statuses: "
                f"`{json.dumps(fx_slots_inserts_contract['accepted_row_statuses'])}`",
                "- allowed_insert_target_types: "
                f"`{json.dumps(fx_slots_inserts_contract['allowed_insert_target_types'])}`",
                "- allowed_insert_positions: "
                f"`{json.dumps(fx_slots_inserts_contract['allowed_insert_positions'])}`",
                "- forbidden_mutation_evidence_fields: "
                f"`{json.dumps(fx_slots_inserts_contract['forbidden_mutation_evidence_fields'])}`",
                "- target_consistency: "
                f"`{str(fx_slots_inserts_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(fx_slots_inserts_contract['violations'])}`",
            ]
        )
    if routing_sends_contract is not None:
        lines.extend(
            [
                "",
                "## Routing Sends Physical Readback Contract",
                "",
                f"- schema_check: `{routing_sends_contract['status']}`",
                f"- read_only: `{str(routing_sends_contract['read_only']).lower()}`",
                f"- live_write: `{str(routing_sends_contract['live_write']).lower()}`",
                "- mutates_console_state: "
                f"`{str(routing_sends_contract['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(routing_sends_contract['queries_console']).lower()}`",
                f"- executes_tests: `{str(routing_sends_contract['executes_tests']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(routing_sends_contract['live_widening_allowed']).lower()}`",
                "- full_autonomous_ready: "
                f"`{str(routing_sends_contract['full_autonomous_ready']).lower()}`",
                f"- schema_id: `{routing_sends_contract['schema_id']}`",
                f"- contract_status: `{routing_sends_contract['contract_status']}`",
                f"- surface_id: `{routing_sends_contract['surface_id']}`",
                f"- remediation_action_id: `{routing_sends_contract['remediation_action_id']}`",
                "- physical_readback_surface_id: "
                f"`{routing_sends_contract['physical_readback_surface_id']}`",
                "- physical_readback_action_id: "
                f"`{routing_sends_contract['physical_readback_action_id']}`",
                "- fixture_physical_gap_declared: "
                f"`{str(routing_sends_contract['fixture_physical_gap_declared']).lower()}`",
                "- remediation_physical_gap_declared: "
                f"`{str(routing_sends_contract['remediation_physical_gap_declared']).lower()}`",
                "- artifact_required: "
                f"`{str(routing_sends_contract['artifact_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(routing_sends_contract['physical_fixture_gap_remains']).lower()}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(routing_sends_contract['accepted_as_physical_fixture_candidate']).lower()}`",
                "- live_authority_granted: "
                f"`{str(routing_sends_contract['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(routing_sends_contract['promotion_families_requiring_surface'])}`",
                "- required_top_level_field_count: "
                f"`{routing_sends_contract['required_top_level_field_count']}`",
                "- required_static_route_field_count: "
                f"`{routing_sends_contract['required_static_route_field_count']}`",
                "- required_static_route_fields: "
                f"`{json.dumps(routing_sends_contract['required_static_route_fields'])}`",
                "- required_bus_send_fields: "
                f"`{json.dumps(routing_sends_contract['required_bus_send_fields'])}`",
                "- required_main_send_fields: "
                f"`{json.dumps(routing_sends_contract['required_main_send_fields'])}`",
                "- required_matrix_send_fields: "
                f"`{json.dumps(routing_sends_contract['required_matrix_send_fields'])}`",
                "- required_output_route_fields: "
                f"`{json.dumps(routing_sends_contract['required_output_route_fields'])}`",
                "- required_static_route_coverage_flags: "
                f"`{json.dumps(routing_sends_contract['required_static_route_coverage_flags'])}`",
                "- required_send_coverage_flags: "
                f"`{json.dumps(routing_sends_contract['required_send_coverage_flags'])}`",
                "- required_output_route_coverage_flags: "
                f"`{json.dumps(routing_sends_contract['required_output_route_coverage_flags'])}`",
                "- required_false_safety_flags: "
                f"`{json.dumps(routing_sends_contract['required_false_safety_flags'])}`",
                "- minimum_static_dante_route_count: "
                f"`{routing_sends_contract['minimum_static_dante_route_count']}`",
                "- minimum_channel_bus_send_count: "
                f"`{routing_sends_contract['minimum_channel_bus_send_count']}`",
                "- minimum_channel_main_send_count: "
                f"`{routing_sends_contract['minimum_channel_main_send_count']}`",
                "- minimum_channel_matrix_send_count: "
                f"`{routing_sends_contract['minimum_channel_matrix_send_count']}`",
                f"- minimum_output_route_count: `{routing_sends_contract['minimum_output_route_count']}`",
                "- allowed_sources: "
                f"`{json.dumps(routing_sends_contract['allowed_sources'])}`",
                "- accepted_row_statuses: "
                f"`{json.dumps(routing_sends_contract['accepted_row_statuses'])}`",
                "- forbidden_mutation_evidence_fields: "
                f"`{json.dumps(routing_sends_contract['forbidden_mutation_evidence_fields'])}`",
                "- target_consistency: "
                f"`{str(routing_sends_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(routing_sends_contract['violations'])}`",
            ]
        )
    if snapshots_scenes_contract is not None:
        lines.extend(
            [
                "",
                "## Snapshots Scenes Physical Readback Contract",
                "",
                f"- schema_check: `{snapshots_scenes_contract['status']}`",
                f"- read_only: `{str(snapshots_scenes_contract['read_only']).lower()}`",
                f"- live_write: `{str(snapshots_scenes_contract['live_write']).lower()}`",
                "- scene_load_allowed: "
                f"`{str(snapshots_scenes_contract['scene_load_allowed']).lower()}`",
                "- scene_save_allowed: "
                f"`{str(snapshots_scenes_contract['scene_save_allowed']).lower()}`",
                f"- schema_id: `{snapshots_scenes_contract['schema_id']}`",
                f"- contract_status: `{snapshots_scenes_contract['contract_status']}`",
                f"- surface_id: `{snapshots_scenes_contract['surface_id']}`",
                f"- remediation_action_id: `{snapshots_scenes_contract['remediation_action_id']}`",
                "- physical_readback_surface_id: "
                f"`{snapshots_scenes_contract['physical_readback_surface_id']}`",
                "- physical_readback_action_id: "
                f"`{snapshots_scenes_contract['physical_readback_action_id']}`",
                "- fixture_physical_gap_declared: "
                f"`{str(snapshots_scenes_contract['fixture_physical_gap_declared']).lower()}`",
                "- remediation_physical_gap_declared: "
                f"`{str(snapshots_scenes_contract['remediation_physical_gap_declared']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(snapshots_scenes_contract['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(snapshots_scenes_contract['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(snapshots_scenes_contract['promotion_families_requiring_surface'])}`",
                "- required_snapshot_fields: "
                f"`{json.dumps(snapshots_scenes_contract['required_snapshot_fields'])}`",
                "- required_coverage_flags: "
                f"`{json.dumps(snapshots_scenes_contract['required_coverage_flags'])}`",
                "- required_false_safety_flags: "
                f"`{json.dumps(snapshots_scenes_contract['required_false_safety_flags'])}`",
                f"- minimum_snapshot_count: `{snapshots_scenes_contract['minimum_snapshot_count']}`",
                "- target_consistency: "
                f"`{str(snapshots_scenes_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(snapshots_scenes_contract['violations'])}`",
            ]
        )
    if user_presets_contract is not None:
        lines.extend(
            [
                "",
                "## User Presets Physical Readback Contract",
                "",
                f"- schema_check: `{user_presets_contract['status']}`",
                f"- read_only: `{str(user_presets_contract['read_only']).lower()}`",
                f"- live_write: `{str(user_presets_contract['live_write']).lower()}`",
                "- preset_recall_allowed: "
                f"`{str(user_presets_contract['preset_recall_allowed']).lower()}`",
                "- preset_store_allowed: "
                f"`{str(user_presets_contract['preset_store_allowed']).lower()}`",
                f"- schema_id: `{user_presets_contract['schema_id']}`",
                f"- contract_status: `{user_presets_contract['contract_status']}`",
                f"- surface_id: `{user_presets_contract['surface_id']}`",
                f"- remediation_action_id: `{user_presets_contract['remediation_action_id']}`",
                "- physical_readback_surface_id: "
                f"`{user_presets_contract['physical_readback_surface_id']}`",
                "- physical_readback_action_id: "
                f"`{user_presets_contract['physical_readback_action_id']}`",
                "- fixture_physical_gap_declared: "
                f"`{str(user_presets_contract['fixture_physical_gap_declared']).lower()}`",
                "- remediation_physical_gap_declared: "
                f"`{str(user_presets_contract['remediation_physical_gap_declared']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(user_presets_contract['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(user_presets_contract['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(user_presets_contract['promotion_families_requiring_surface'])}`",
                "- required_preset_fields: "
                f"`{json.dumps(user_presets_contract['required_preset_fields'])}`",
                "- required_coverage_flags: "
                f"`{json.dumps(user_presets_contract['required_coverage_flags'])}`",
                "- required_false_safety_flags: "
                f"`{json.dumps(user_presets_contract['required_false_safety_flags'])}`",
                f"- minimum_preset_count: `{user_presets_contract['minimum_preset_count']}`",
                "- target_consistency: "
                f"`{str(user_presets_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(user_presets_contract['violations'])}`",
            ]
        )
    if dlive_state_contract is not None:
        lines.extend(
            [
                "",
                "## dLive State Physical Readback Contract",
                "",
                f"- schema_check: `{dlive_state_contract['status']}`",
                f"- read_only: `{str(dlive_state_contract['read_only']).lower()}`",
                f"- live_write: `{str(dlive_state_contract['live_write']).lower()}`",
                "- midi_send_allowed: "
                f"`{str(dlive_state_contract['midi_send_allowed']).lower()}`",
                "- tcp_send_allowed: "
                f"`{str(dlive_state_contract['tcp_send_allowed']).lower()}`",
                "- dlive_write_allowed: "
                f"`{str(dlive_state_contract['dlive_write_allowed']).lower()}`",
                f"- schema_id: `{dlive_state_contract['schema_id']}`",
                f"- contract_status: `{dlive_state_contract['contract_status']}`",
                f"- surface_id: `{dlive_state_contract['surface_id']}`",
                f"- remediation_action_id: `{dlive_state_contract['remediation_action_id']}`",
                "- physical_readback_surface_id: "
                f"`{dlive_state_contract['physical_readback_surface_id']}`",
                "- physical_readback_action_id: "
                f"`{dlive_state_contract['physical_readback_action_id']}`",
                "- fixture_physical_gap_declared: "
                f"`{str(dlive_state_contract['fixture_physical_gap_declared']).lower()}`",
                "- remediation_physical_gap_declared: "
                f"`{str(dlive_state_contract['remediation_physical_gap_declared']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(dlive_state_contract['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(dlive_state_contract['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(dlive_state_contract['promotion_families_requiring_surface'])}`",
                "- required_channel_fields: "
                f"`{json.dumps(dlive_state_contract['required_channel_fields'])}`",
                "- required_midi_state_fields: "
                f"`{json.dumps(dlive_state_contract['required_midi_state_fields'])}`",
                "- required_parity_fields: "
                f"`{json.dumps(dlive_state_contract['required_parity_fields'])}`",
                "- required_coverage_flags: "
                f"`{json.dumps(dlive_state_contract['required_coverage_flags'])}`",
                "- required_false_safety_flags: "
                f"`{json.dumps(dlive_state_contract['required_false_safety_flags'])}`",
                f"- minimum_channel_count: `{dlive_state_contract['minimum_channel_count']}`",
                "- target_consistency: "
                f"`{str(dlive_state_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(dlive_state_contract['violations'])}`",
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
            read_capture_plan_result = (
                validate_read_surface_physical_readback_capture_plan_snapshot()
            )
            results.append(read_capture_plan_result)
            if read_capture_plan_result.status != "passed" and not continue_on_failure:
                break
            raw_mixer_contract_result = validate_raw_mixer_state_physical_readback_contract_snapshot()
            results.append(raw_mixer_contract_result)
            if raw_mixer_contract_result.status != "passed" and not continue_on_failure:
                break
            raw_mixer_intake_result = (
                validate_raw_mixer_state_physical_readback_artifact_intake_snapshot()
            )
            results.append(raw_mixer_intake_result)
            if raw_mixer_intake_result.status != "passed" and not continue_on_failure:
                break
            channel_basic_contract_result = (
                validate_channel_basic_controls_physical_readback_contract_snapshot()
            )
            results.append(channel_basic_contract_result)
            if channel_basic_contract_result.status != "passed" and not continue_on_failure:
                break
            channel_basic_intake_result = (
                validate_channel_basic_controls_physical_readback_artifact_intake_snapshot()
            )
            results.append(channel_basic_intake_result)
            if channel_basic_intake_result.status != "passed" and not continue_on_failure:
                break
            channel_processing_contract_result = (
                validate_channel_processing_physical_readback_contract_snapshot()
            )
            results.append(channel_processing_contract_result)
            if channel_processing_contract_result.status != "passed" and not continue_on_failure:
                break
            channel_processing_intake_result = (
                validate_channel_processing_physical_readback_artifact_intake_snapshot()
            )
            results.append(channel_processing_intake_result)
            if channel_processing_intake_result.status != "passed" and not continue_on_failure:
                break
            bus_group_main_dca_contract_result = (
                validate_bus_group_main_dca_physical_readback_contract_snapshot()
            )
            results.append(bus_group_main_dca_contract_result)
            if (
                bus_group_main_dca_contract_result.status != "passed"
                and not continue_on_failure
            ):
                break
            bus_group_main_dca_intake_result = (
                validate_bus_group_main_dca_physical_readback_artifact_intake_snapshot()
            )
            results.append(bus_group_main_dca_intake_result)
            if (
                bus_group_main_dca_intake_result.status != "passed"
                and not continue_on_failure
            ):
                break
            matrix_state_contract_result = (
                validate_matrix_state_physical_readback_contract_snapshot()
            )
            results.append(matrix_state_contract_result)
            if (
                matrix_state_contract_result.status != "passed"
                and not continue_on_failure
            ):
                break
            matrix_state_intake_result = (
                validate_matrix_state_physical_readback_artifact_intake_snapshot()
            )
            results.append(matrix_state_intake_result)
            if (
                matrix_state_intake_result.status != "passed"
                and not continue_on_failure
            ):
                break
            fx_slots_inserts_contract_result = (
                validate_fx_slots_inserts_physical_readback_contract_snapshot()
            )
            results.append(fx_slots_inserts_contract_result)
            if (
                fx_slots_inserts_contract_result.status != "passed"
                and not continue_on_failure
            ):
                break
            routing_sends_contract_result = (
                validate_routing_sends_physical_readback_contract_snapshot()
            )
            results.append(routing_sends_contract_result)
            if (
                routing_sends_contract_result.status != "passed"
                and not continue_on_failure
            ):
                break
            snapshots_scenes_contract_result = (
                validate_snapshots_scenes_physical_readback_contract_snapshot()
            )
            results.append(snapshots_scenes_contract_result)
            if (
                snapshots_scenes_contract_result.status != "passed"
                and not continue_on_failure
            ):
                break
            user_presets_contract_result = (
                validate_user_presets_physical_readback_contract_snapshot()
            )
            results.append(user_presets_contract_result)
            if (
                user_presets_contract_result.status != "passed"
                and not continue_on_failure
            ):
                break
            dlive_state_contract_result = (
                validate_dlive_state_physical_readback_contract_snapshot()
            )
            results.append(dlive_state_contract_result)
            if (
                dlive_state_contract_result.status != "passed"
                and not continue_on_failure
            ):
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
