#!/usr/bin/env python3
"""Run the no-hardware Automixer operator validation harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
for path in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.offline_validation_harness import (  # noqa: E402
    run_offline_validation,
    write_offline_validation_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Automixer proposal flow offline without connecting to real mixer hardware."
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts/offline_validation",
        help="Directory where summary.json and events.jsonl will be written.",
    )
    parser.add_argument(
        "--total-channels",
        type=int,
        default=16,
        help="Channel count exposed to the Product Layer inventory builder.",
    )
    parser.add_argument(
        "--no-write-artifacts",
        action="store_true",
        help="Print the summary only and skip artifact files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_offline_validation(total_channels=args.total_channels)
    if not args.no_write_artifacts:
        result["artifacts"] = write_offline_validation_artifacts(
            result,
            artifact_root=args.artifact_root,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") in {"passed", "clear"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
