#!/usr/bin/env python3
"""Append a session report to .ai/project-layer/session_reports.md."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


TARGET = Path(".ai/project-layer/session_reports.md")
TEMPLATE = Path(".ai/pipeline/templates/session_report.md")


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def render_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value.strip() or "Not specified.")
    return rendered


def append_record(root: Path, record: str) -> Path:
    target = root / TARGET
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        current = target.read_text(encoding="utf-8")
    else:
        current = "# Session Reports\n\n## Records\n"

    separator = "\n\n---\n\n" if current.strip() else ""
    target.write_text(current.rstrip() + separator + record.rstrip() + "\n", encoding="utf-8")
    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--title", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--agent", default="codex")
    parser.add_argument("--mode", default="audit-only")
    parser.add_argument("--files", default="Not specified.")
    parser.add_argument("--tests", default="Not specified.")
    parser.add_argument("--memory-lint", default="Not specified.")
    parser.add_argument(
        "--safety",
        default="No live OSC/WING writes. No runtime mixer state changes.",
    )
    parser.add_argument("--risks", default="Not specified.")
    parser.add_argument("--next", default="Not specified.")
    parser.add_argument("--date", default=utc_date())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    template_path = root / TEMPLATE

    if not template_path.exists():
        print(f"Missing template: {template_path}", file=sys.stderr)
        return 1

    template = template_path.read_text(encoding="utf-8")
    record = render_template(
        template,
        {
            "date": args.date,
            "title": args.title,
            "agent": args.agent,
            "mode": args.mode,
            "summary": args.summary,
            "files": args.files,
            "tests": args.tests,
            "memory_lint": args.memory_lint,
            "safety": args.safety,
            "risks": args.risks,
            "next": args.next,
        },
    )
    target = append_record(root, record)
    print(f"Appended session report to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
