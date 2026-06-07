#!/usr/bin/env python3
"""Build an offline operator review queue for physical readback artifacts."""

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

from backend.read_surface_physical_readback_artifact_review_queue import (  # noqa: E402
    build_read_surface_physical_readback_artifact_review_queue,
)


DEFAULT_OUTPUT = "Docs/reports/READ_SURFACE_PHYSICAL_READBACK_ARTIFACT_REVIEW_QUEUE_LATEST.md"
FOCUSED_VALIDATION_COMMAND = [
    "PYTHONPATH=backend",
    "python3",
    "-m",
    "pytest",
    "-q",
    "tests/test_read_surface_physical_readback_artifact_review_queue.py",
    "tests/test_no_wing_release_check.py",
]
OPERATOR_SEQUENCE = [
    "Confirm every listed candidate came from a physical readback artifact intake.",
    "Reject the queue if any candidate reports live, mutation, query, or closure authority.",
    "Review valid candidates manually before preparing acceptance records.",
    "Capture or replace any missing or invalid physical readback artifacts.",
    "Run the focused validation command before using the acceptance template.",
]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _bool(value: Any) -> str:
    return str(bool(value)).lower()


def _command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def build_review_queue_artifact() -> dict[str, Any]:
    queue = dict(build_read_surface_physical_readback_artifact_review_queue())
    queue["operator_sequence"] = list(OPERATOR_SEQUENCE)
    queue["commands"] = {"focused_validation": list(FOCUSED_VALIDATION_COMMAND)}
    return queue


def render_markdown(queue: Mapping[str, Any]) -> str:
    safety = _mapping(queue.get("safety"))
    summary = _mapping(queue.get("summary"))
    commands = _mapping(queue.get("commands"))
    items = _list(queue.get("queue_items"))
    lines = [
        "# Read Surface Physical Readback Artifact Review Queue",
        "",
        f"Generated: `{queue.get('generated_at')}`",
        f"Status: `{queue.get('status')}`",
        "",
        "## Safety Boundary",
        "",
        f"- read_only: `{_bool(safety.get('read_only'))}`",
        f"- live_write: `{_bool(safety.get('live_write'))}`",
        f"- mutates_console_state: `{_bool(safety.get('mutates_console_state'))}`",
        f"- queries_console: `{_bool(safety.get('queries_console'))}`",
        f"- executes_tests: `{_bool(safety.get('executes_tests'))}`",
        f"- operator_required: `{_bool(safety.get('operator_required'))}`",
        f"- execution_allowed_by_queue: `{_bool(safety.get('execution_allowed_by_queue'))}`",
        f"- fixture_closure_allowed_by_queue: `{_bool(safety.get('fixture_closure_allowed_by_queue'))}`",
        f"- live_authority_granted: `{_bool(safety.get('live_authority_granted'))}`",
        f"- full_autonomous_ready: `{_bool(safety.get('full_autonomous_ready'))}`",
        "",
        "## Summary",
        "",
        f"- intake_surface_count: `{summary.get('intake_surface_count')}`",
        f"- available_intake_count: `{summary.get('available_intake_count')}`",
        f"- artifact_count: `{summary.get('artifact_count')}`",
        f"- valid_candidate_count: `{summary.get('valid_candidate_count')}`",
        f"- invalid_candidate_count: `{summary.get('invalid_candidate_count')}`",
        f"- review_candidate_count: `{summary.get('review_candidate_count')}`",
        f"- missing_artifact_surface_count: `{summary.get('missing_artifact_surface_count')}`",
        f"- unsafe_authority_claim_count: `{summary.get('unsafe_authority_claim_count')}`",
        "- candidate_source_id_missing_count: "
        f"`{summary.get('candidate_source_id_missing_count')}`",
        f"- target_inconsistent_surface_count: `{summary.get('target_inconsistent_surface_count')}`",
        f"- physical_fixture_gap_remains: `{_bool(summary.get('physical_fixture_gap_remains'))}`",
        f"- blocked_by: `{json.dumps(summary.get('blocked_by') or [])}`",
        "",
        "## Operator Sequence",
        "",
    ]
    for index, item in enumerate(queue.get("operator_sequence") or [], start=1):
        lines.append(f"{index}. {item}")
    lines.extend(["", "## Queue Items", ""])
    for item in items:
        row = _mapping(item)
        lines.append(
            "- "
            f"{row.get('surface_id')} status={row.get('status')} "
            f"valid={row.get('valid_candidate_count')} invalid={row.get('invalid_candidate_count')} "
            f"sources={json.dumps(row.get('artifact_source_ids') or [])}"
        )
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
        description="Build an offline read-surface physical readback artifact review queue."
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    queue = build_review_queue_artifact()
    text = (
        json.dumps(queue, ensure_ascii=False, indent=2, sort_keys=True)
        if args.format == "json"
        else render_markdown(queue)
    )
    write_output(Path(args.output), text)
    if not args.quiet:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
