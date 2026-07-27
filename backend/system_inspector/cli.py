"""CLI for read-only System Inspector snapshot replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .engine import inspect_snapshot_bundle
from .io import InspectorInputError
from .reports import canonical_json, render_text_report, write_json_report
from .safety import InspectorReadOnlyConfigurationError, assert_read_only_mode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="automixer")
    top = parser.add_subparsers(dest="product_command", required=True)
    inspector = top.add_parser("inspector", help="Read-only System Inspector")
    actions = inspector.add_subparsers(dest="inspector_command", required=True)
    for action in ("scan", "replay-snapshot", "findings", "routes", "devices"):
        command = actions.add_parser(action)
        command.add_argument("--profile", required=True, type=Path)
        command.add_argument("--snapshot", required=True, type=Path)
        command.add_argument("--format", choices=("text", "json"), default="text")
        command.add_argument("--output", type=Path)
        command.add_argument(
            "--read-only",
            action="store_true",
            default=True,
            help="Always enabled in v1; there is no disable flag.",
        )
        command.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        assert_read_only_mode()
        report = inspect_snapshot_bundle(args.profile, args.snapshot)
        payload = (
            canonical_json(report)
            if args.format == "json"
            else render_text_report(report) + "\n"
        )
        if args.output:
            if args.format != "json":
                parser.error("--output currently requires --format json")
            write_json_report(report, args.output)
        sys.stdout.write(payload)
        return 2 if any(
            finding.severity.value in {"CRITICAL", "HIGH"}
            for finding in report.findings
        ) else 0
    except (
        InspectorInputError,
        InspectorReadOnlyConfigurationError,
        OSError,
        ValueError,
    ) as exc:
        sys.stderr.write(f"System Inspector failed closed: {exc}\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

