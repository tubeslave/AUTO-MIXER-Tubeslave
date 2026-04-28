#!/usr/bin/env python3
"""Create, clear, or inspect the Mixing Station emergency stop flag."""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLAG = REPO_ROOT / "runtime" / "EMERGENCY_STOP"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["stop", "clear", "status"])
    parser.add_argument("--flag", default=str(DEFAULT_FLAG))
    args = parser.parse_args()

    flag = Path(args.flag).expanduser()
    if args.command == "stop":
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text("Mixing Station emergency stop active\n", encoding="utf-8")
        print(f"Emergency stop active: {flag}")
        return 0
    if args.command == "clear":
        if flag.exists():
            flag.unlink()
        print(f"Emergency stop cleared: {flag}")
        return 0

    print("active" if flag.exists() else "inactive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
