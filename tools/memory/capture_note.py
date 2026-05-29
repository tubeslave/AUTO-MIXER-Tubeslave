#!/usr/bin/env python3
"""Append a raw memory note to experiments.md or a capture file."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENTS_TARGET = Path(".ai/project-layer/experiments.md")
CAPTURE_TARGET = Path(".ai/project-layer/capture_notes.md")


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def append_note(root: Path, target: Path, note: str) -> Path:
    path = root / target
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        current = path.read_text(encoding="utf-8")
    elif target == EXPERIMENTS_TARGET:
        current = "# Experiments\n\n## Records\n"
    else:
        current = "# Capture Notes\n\n## Records\n"

    separator = "\n\n---\n\n" if current.strip() else ""
    path.write_text(current.rstrip() + separator + note.rstrip() + "\n", encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--title", required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--mode", default="audit-only")
    parser.add_argument("--source", default="Not specified.")
    parser.add_argument("--author", default="codex")
    parser.add_argument("--date", default=utc_date())
    parser.add_argument(
        "--capture-file",
        action="store_true",
        help="Append to .ai/project-layer/capture_notes.md instead of experiments.md.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target = CAPTURE_TARGET if args.capture_file else EXPERIMENTS_TARGET
    note = "\n".join(
        [
            f"### {args.date} - {args.title}",
            "",
            f"Mode: {args.mode}",
            "",
            f"Source: {args.source}",
            "",
            "Note:",
            args.note,
            "",
            f"Author: {args.author}",
        ]
    )
    path = append_note(args.root.resolve(), target, note)
    print(f"Appended note to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
