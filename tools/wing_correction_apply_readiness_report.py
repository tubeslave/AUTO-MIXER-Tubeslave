#!/usr/bin/env python3
"""Emit the WING correction apply readiness snapshot as JSON or Markdown."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_path in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from wing_correction_apply_readiness import (  # noqa: E402
    build_wing_correction_apply_readiness_snapshot,
    render_wing_correction_apply_readiness_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the read-only WING correction apply readiness matrix."
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="markdown",
        help="Output format",
    )
    parser.add_argument("--output", help="Optional output file")
    parser.add_argument(
        "--no-method-status",
        action="store_true",
        help="Omit per-method registry status details from JSON.",
    )
    return parser


def render_payload(args: argparse.Namespace) -> str:
    snapshot: dict[str, Any] = build_wing_correction_apply_readiness_snapshot(
        include_method_status=not args.no_method_status
    )
    if args.format == "json":
        return json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return render_wing_correction_apply_readiness_markdown(snapshot)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = render_payload(args)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
        if not payload.endswith("\n"):
            sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

