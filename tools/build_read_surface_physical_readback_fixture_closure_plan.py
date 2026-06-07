#!/usr/bin/env python3
"""Build an offline operator plan for manual readback fixture closure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
for path in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.read_surface_physical_readback_fixture_closure_plan import (  # noqa: E402
    build_read_surface_physical_readback_fixture_closure_plan,
)


DEFAULT_OUTPUT = "Docs/reports/READ_SURFACE_PHYSICAL_READBACK_FIXTURE_CLOSURE_PLAN_LATEST.md"
FOCUSED_VALIDATION_COMMAND = [
    "PYTHONPATH=backend",
    "python3",
    "-m",
    "pytest",
    "-q",
    "tests/test_read_surface_physical_readback_fixture_closure_plan.py",
    "tests/test_no_wing_release_check.py",
]
OPERATOR_SEQUENCE = [
    "Confirm every physical readback candidate was reviewed by an operator.",
    "Reject this plan if any acceptance record is invalid, ambiguous, or unreviewed.",
    "Keep fixture_update_allowed_by_plan and fixture_closure_allowed_by_plan false.",
    "Perform any fixture baseline update as a separate manual repository change.",
    "Run the focused validation command before considering a separate closure decision.",
]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _bool(value: Any) -> str:
    return str(bool(value)).lower()


def _command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def build_closure_plan_artifact() -> dict[str, Any]:
    plan = dict(build_read_surface_physical_readback_fixture_closure_plan())
    plan["operator_sequence"] = list(OPERATOR_SEQUENCE)
    plan["commands"] = {
        "focused_validation": list(FOCUSED_VALIDATION_COMMAND),
    }
    return plan


def render_markdown(plan: Mapping[str, Any]) -> str:
    safety = _mapping(plan.get("safety"))
    summary = _mapping(plan.get("summary"))
    commands = _mapping(plan.get("commands"))
    candidates = _list(plan.get("closure_candidates"))
    lines = [
        "# Read Surface Physical Readback Fixture Closure Plan",
        "",
        f"Generated: `{plan.get('generated_at')}`",
        f"Status: `{plan.get('status')}`",
        "",
        "## Safety Boundary",
        "",
        f"- read_only: `{_bool(safety.get('read_only'))}`",
        f"- live_write: `{_bool(safety.get('live_write'))}`",
        f"- mutates_console_state: `{_bool(safety.get('mutates_console_state'))}`",
        f"- queries_console: `{_bool(safety.get('queries_console'))}`",
        f"- executes_tests: `{_bool(safety.get('executes_tests'))}`",
        f"- operator_required: `{_bool(safety.get('operator_required'))}`",
        f"- execution_allowed_by_plan: `{_bool(safety.get('execution_allowed_by_plan'))}`",
        f"- fixture_update_allowed_by_plan: `{_bool(safety.get('fixture_update_allowed_by_plan'))}`",
        f"- fixture_closure_allowed_by_plan: `{_bool(safety.get('fixture_closure_allowed_by_plan'))}`",
        f"- live_authority_granted: `{_bool(safety.get('live_authority_granted'))}`",
        f"- full_autonomous_ready: `{_bool(safety.get('full_autonomous_ready'))}`",
        "",
        "## Summary",
        "",
        f"- ledger_status: `{summary.get('ledger_status')}`",
        f"- review_candidate_count: `{summary.get('review_candidate_count')}`",
        f"- acceptance_record_count: `{summary.get('acceptance_record_count')}`",
        f"- accepted_record_count: `{summary.get('accepted_record_count')}`",
        f"- invalid_record_count: `{summary.get('invalid_record_count')}`",
        f"- unreviewed_candidate_count: `{summary.get('unreviewed_candidate_count')}`",
        f"- closure_candidate_count: `{summary.get('closure_candidate_count')}`",
        "- manual_fixture_closure_candidate_available: "
        f"`{_bool(summary.get('manual_fixture_closure_candidate_available'))}`",
        f"- physical_fixture_gap_remains: `{_bool(summary.get('physical_fixture_gap_remains'))}`",
        f"- blocked_by: `{json.dumps(summary.get('blocked_by') or [])}`",
        f"- next_global_step: `{summary.get('next_global_step')}`",
        "",
        "## Operator Sequence",
        "",
    ]
    for index, item in enumerate(plan.get("operator_sequence") or [], start=1):
        lines.append(f"{index}. {item}")
    lines.extend(["", "## Closure Candidates", ""])
    if candidates:
        for candidate in candidates:
            row = _mapping(candidate)
            lines.append(
                "- "
                f"{row.get('surface_id')} from {row.get('artifact_source_id')} "
                f"(manual_required={_bool(row.get('manual_fixture_closure_required'))})"
            )
    else:
        lines.append("- none")
    focused_validation = commands.get("focused_validation")
    if isinstance(focused_validation, list):
        lines.extend(
            [
                "",
                "## Focused Validation",
                "",
                "```bash",
                _command_text([str(item) for item in focused_validation]),
                "```",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def write_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an offline read-surface physical readback fixture closure plan."
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_closure_plan_artifact()
    text = (
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True)
        if args.format == "json"
        else render_markdown(plan)
    )
    write_output(Path(args.output), text)
    if not args.quiet:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
