#!/usr/bin/env python3
"""Lint Automixer Markdown memory files without modifying the project."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemoryFile:
    path: str
    headings: tuple[str, ...] = ()
    required_terms: tuple[str, ...] = ()


REQUIRED_FILES: tuple[MemoryFile, ...] = (
    MemoryFile(
        ".ai/agent-core/rules.md",
        ("# Agent Core Rules", "## Purpose", "## Required Startup Reads"),
    ),
    MemoryFile(".ai/agent-core/workflow.md", ("# Agent Core Workflow", "## Default Cycle")),
    MemoryFile(
        ".ai/agent-core/safety.md",
        ("# Agent Core Safety", "## Non Negotiable Rules", "## Mode Definitions"),
        (
            "live osc writes are forbidden without explicit approval",
            "live wing writes are forbidden without explicit approval",
            "audit-only",
            "dry-run",
            "supervised-write",
            "live-write",
        ),
    ),
    MemoryFile(".ai/agent-core/source_policy.md", ("# Agent Core Source Policy",)),
    MemoryFile(".ai/agent-core/lint_policy.md", ("# Agent Core Lint Policy",)),
    MemoryFile(
        ".ai/project-layer/automixer_goal.md",
        ("# Automixer Goal", "## Main Goal"),
        ("automatic live mixer", "behringer wing rack", "osc"),
    ),
    MemoryFile(".ai/project-layer/architecture.md", ("# Automixer Memory Architecture",)),
    MemoryFile(
        ".ai/project-layer/wing_osc_policy.md",
        ("# WING OSC Policy", "## Mode Separation", "## Hard Restrictions"),
        (
            "audit-only",
            "dry-run",
            "supervised-write",
            "live-write",
            "live osc writes are forbidden without explicit approval",
        ),
    ),
    MemoryFile(".ai/project-layer/gainstaging_method.md", ("# Gain-Staging Method",)),
    MemoryFile(
        ".ai/project-layer/live_safety_rules.md",
        ("# Live Safety Rules", "## Hard Rules", "## Mode Definitions"),
        (
            "live osc writes are forbidden without explicit approval",
            "pre-read",
            "rollback",
            "recommendations and real control commands",
            "audit-only",
            "dry-run",
            "live-write",
        ),
    ),
    MemoryFile(
        ".ai/project-layer/current_tasks.md",
        ("# Current Tasks", "## Active Tasks", "## Watch List"),
        (
            "dry-run or audit-only",
            "do not treat shadow-only or offline-only evidence as live readiness",
        ),
    ),
    MemoryFile(".ai/project-layer/decisions.md", ("# Decisions", "## Records")),
    MemoryFile(".ai/project-layer/experiments.md", ("# Experiments", "## Records")),
    MemoryFile(".ai/project-layer/session_reports.md", ("# Session Reports", "## Records")),
    MemoryFile(".ai/project-layer/glossary.md", ("# Glossary", "## Terms")),
    MemoryFile(".ai/pipeline/memory_pipeline.md", ("# Memory Pipeline", "## Flow")),
    MemoryFile(
        ".ai/pipeline/capture_classify_propose_approve_apply_log_lint.md",
        ("# CAPTURE CLASSIFY PROPOSE APPROVE APPLY LOG LINT", "## CAPTURE", "## LINT"),
    ),
    MemoryFile(".ai/pipeline/codex_cycle.md", ("# Codex Cycle", "## Cycle")),
    MemoryFile(".ai/pipeline/paperclip_cycle.md", ("# Paperclip Cycle", "## Cycle")),
    MemoryFile(".ai/pipeline/templates/decision_record.md"),
    MemoryFile(".ai/pipeline/templates/experiment_record.md"),
    MemoryFile(".ai/pipeline/templates/session_report.md"),
    MemoryFile(".ai/pipeline/templates/task_record.md"),
    MemoryFile(".ai/pipeline/templates/safety_check.md"),
    MemoryFile("Docs/memory_pipeline.md"),
)


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_memory_file(root: Path, item: MemoryFile) -> list[str]:
    errors: list[str] = []
    path = root / item.path

    if not path.exists():
        return [f"missing required file: {item.path}"]
    if not path.is_file():
        return [f"required path is not a file: {item.path}"]

    text = read_text(path)
    lower_text = text.lower()

    for heading in item.headings:
        if heading not in text:
            errors.append(f"{item.path}: missing heading {heading!r}")

    for term in item.required_terms:
        if term.lower() not in lower_text:
            errors.append(f"{item.path}: missing required term {term!r}")

    return errors


def run_lint(root: Path) -> tuple[bool, list[str]]:
    root = root.resolve()
    errors: list[str] = []

    for item in REQUIRED_FILES:
        errors.extend(check_memory_file(root, item))

    mode_files = [
        root / ".ai/agent-core/safety.md",
        root / ".ai/project-layer/wing_osc_policy.md",
        root / ".ai/project-layer/live_safety_rules.md",
    ]
    for path in mode_files:
        if not path.exists():
            continue
        text = read_text(path).lower()
        for mode in ("audit-only", "dry-run", "supervised-write", "live-write"):
            if mode not in text:
                errors.append(f"{path.relative_to(root)}: missing mode {mode!r}")

    return not errors, errors


def build_report(root: Path, ok: bool, errors: list[str]) -> str:
    lines = [
        "# Memory Lint Report",
        "",
        f"Root: {root.resolve()}",
        f"Required files: {len(REQUIRED_FILES)}",
        f"Status: {'PASS' if ok else 'FAIL'}",
        "",
    ]

    if errors:
        lines.append("## Errors")
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("## Checks")
        lines.append("- Required files exist.")
        lines.append("- Mandatory headings are present.")
        lines.append("- Safety files forbid live OSC/WING writes without approval.")
        lines.append("- audit-only, dry-run, supervised-write, and live-write modes are separated.")

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root(),
        help="Repository root to lint. Defaults to this script's repo.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ok, errors = run_lint(args.root)
    print(build_report(args.root, ok, errors))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
