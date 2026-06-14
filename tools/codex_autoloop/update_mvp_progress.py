#!/usr/bin/env python3
"""Generate the latest supervised WING MVP progress report.

This tool is read-only except for the requested report output. It parses the
repo-local blocker table and writes a compact operator-facing status report.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BLOCKERS = PROJECT_ROOT / ".ai" / "project-layer" / "wing_mvp_blockers.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "Docs" / "reports" / "MVP_PROGRESS_LATEST.md"
CLOSED_STATUS = "closed"
CURRENT_FIRST_COMMAND_HEADING = "## Current First Command"
CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS = (
    "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick",
    "Docs/runbooks/WING_READONLY_PREFLIGHT.md",
    "operator_sequence_count: 10",
    "operator_sequence_sha256_matches_actual: true",
    "operator_sequence_matches_contract: true",
    "operator_sequence_sha256_matches_contract: true",
    "backend_port_prestart_gate",
    "expected_backend_port_prestart_gate",
    "backend_port_prestart_gate.grants_live_authority: false",
    "backend_port_prestart_gate.must_pass_before_backend_wing_session: true",
    "backend_port_prestart_check_command",
    "backend_port_prestart_check_command_text",
    "backend_port_prestart_check_required: true",
    "backend_port_prestart_check_must_pass_before_starting_backend: true",
    "backend_port_prestart_gate_execution_allowed: false",
    "backend_port_prestart_gate_readiness_closure_allowed: false",
    "backend_port_prestart_gate_live_authority_granted: false",
    "backend_port_prestart_check_failure_detail_fields",
    "listener_confirmation_template_status: pending_operator_edit",
    "listener_confirmation_template_safe_to_stop_listener_by_gate: false",
    "listener_confirmation_template_live_authority_granted: false",
    "listener_confirmation_template_accepted_by_template: false",
    "listener_confirmation_template_operator_placeholder_fields",
    "listener_confirmation_template_execution_allowed_by_template: false",
    "listener_confirmation_template_readiness_closure_allowed_by_template: false",
    "manual_review_required_before_readiness_update: true",
    "readiness_matrix_update_allowed_by_packet: false",
    "readiness_closure_allowed_by_packet: false",
    "read_only: true",
    "live_write: false",
    "mutates_console_state: false",
    "queries_console: false",
    "executes_tests: false",
    "operator_required: true",
    "expected_plan_required_for_real_run: true",
    "physical_preflight_requires_existing_backend_connection: true",
    "backend_port_prestart_summary_required: true",
    "backend_port_prestart_summary_type: no_wing_release_check",
    "backend_port_prestart_summary_connection_policy: no_hardware_no_network_no_live_writes",
    "backend_port_prestart_summary_status: passed",
    "backend_port_prestart_summary_live_write: false",
    "backend_port_prestart_summary_contacts_backend: false",
    "backend_port_prestart_summary_queries_console: false",
    "backend_port_prestart_summary_executes_tests: false",
    "backend_port_prestart_check_status: passed",
    "execution_allowed_by_packet: false",
    "live_authority_granted: false",
    "full_autonomous_ready: false",
    "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json",
    "expected_artifacts",
    "expected_artifacts_match_plan: true",
    "summary_glob_matches_plan: true",
    "event_log_glob_matches_plan: true",
    "mvp_blocker_anchor_matches_current_first_open: true",
    "mvp_blocker_anchor_sha256_matches_current_first_open: true",
    "mvp_blocker_anchor.packet_does_not_close_blocker: true",
    "operator_sequence_has_supervised_prestart_gate: true",
    "operator_sequence_supervised_prestart_gate_is_final: true",
    "required_supervised_prestart_gate_markers",
    "supervised_prestart_gate",
    "expected_supervised_prestart_gate",
    "supervised_prestart_gate.grants_live_authority: false",
    "supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true",
    "required_supervised_proof_plan_summary_fields",
    "expected_required_supervised_proof_plan_summary_fields",
    "required_supervised_proof_plan_summary_fields_match_contract: true",
    "readonly_preflight_review_decision_expected_candidate_prestart_source_ids",
    "readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source",
    "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_blocker_table(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    in_table = False
    headers: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            if in_table and rows:
                break
            continue
        cells = split_markdown_row(line)
        if not headers:
            headers = [cell.lower() for cell in cells]
            in_table = True
            continue
        if set(cells) <= {"---", ":---", "---:", ":---:"}:
            continue
        if len(cells) != len(headers):
            continue
        row = dict(zip(headers, cells))
        if {"blocker", "status", "evidence", "next action"} <= set(row):
            rows.append(row)
    return rows


def first_open_blocker(rows: Sequence[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if row.get("status") != CLOSED_STATUS:
            return dict(row)
    return None


def markdown_section_text(path: Path, heading: str) -> str | None:
    section_lines: list[str] = []
    in_section = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == heading:
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section and line:
            section_lines.append(line)
    if not section_lines:
        return None
    return " ".join(section_lines)


def current_first_command(path: Path) -> str | None:
    return markdown_section_text(path, CURRENT_FIRST_COMMAND_HEADING)


def current_first_command_covers_baseline(current_first_command: str | None) -> bool:
    if not current_first_command:
        return False
    return all(
        marker in current_first_command
        for marker in CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS
    )


def selected_action_text(
    *,
    blocker_next_action: str | None,
    current_first_command: str | None,
) -> tuple[str, str]:
    if current_first_command and blocker_next_action:
        if current_first_command == blocker_next_action:
            return current_first_command, "Current First Command"
        if current_first_command_covers_baseline(current_first_command):
            return current_first_command, "Current First Command"
        return (
            f"{current_first_command} Baseline blocker next action remains: "
            f"{blocker_next_action}",
            "Current First Command + blocker table baseline",
        )
    if current_first_command:
        return current_first_command, "Current First Command"
    if blocker_next_action:
        return blocker_next_action, "blocker table"
    return "All MVP blockers are closed.", "all blockers closed"


def build_progress_report(
    rows: Sequence[dict[str, str]],
    *,
    generated_at: str | None = None,
    current_first_command: str | None = None,
) -> str:
    generated = generated_at or utc_now()
    first_open = first_open_blocker(rows)
    total = len(rows)
    closed = sum(1 for row in rows if row.get("status") == CLOSED_STATUS)
    first_name = first_open["blocker"] if first_open else "none"
    next_action, next_action_source = selected_action_text(
        blocker_next_action=first_open["next action"] if first_open else None,
        current_first_command=current_first_command if first_open else None,
    )

    lines = [
        "# MVP Progress Latest",
        "",
        f"Generated: `{generated}`",
        "",
        "## Current MVP",
        "",
        "Supervised WING Rack proof.",
        "",
        "## Summary",
        "",
        f"- Total blockers: {total}",
        f"- Closed blockers: {closed}",
        f"- Current first open blocker: {first_name}",
        f"- Next required action source: {next_action_source}",
        f"- Next required action: {next_action}",
        "",
        "## Blockers",
        "",
        "| Blocker | Status | Next action |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['blocker']} | {row['status']} | {row['next action']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This report does not grant live authority. Offline tooling, shadow",
            "evidence, and readiness candidates are not live readiness. A blocker",
            "closes only after real WING artifacts are manually reviewed.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update supervised WING MVP progress report.")
    parser.add_argument("--blockers", default=str(DEFAULT_BLOCKERS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--print", action="store_true", help="Print report instead of writing only.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    blockers_path = Path(args.blockers)
    rows = parse_blocker_table(blockers_path)
    if not rows:
        print(f"No blocker rows found in {blockers_path}", file=sys.stderr)
        return 2
    command_text = current_first_command(blockers_path)
    report = build_progress_report(rows, current_first_command=command_text)
    write_report(Path(args.output), report)
    if args.print:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
