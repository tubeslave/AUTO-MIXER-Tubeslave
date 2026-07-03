#!/usr/bin/env python3
"""Discover available Mixing Station API dataPath-like values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from integrations.mixing_station.adapter import MixingStationAdapter
from integrations.mixing_station.config import MixingStationConfig, normalize_console_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--console", default="wing_rack")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    profile = normalize_console_profile(args.console)
    out = args.out or f"logs/mixing_station_discovered_paths_{profile}.json"
    config = MixingStationConfig.from_mapping({
        "console_profile": profile,
        "host": args.host,
        "rest_port": args.port,
        "dry_run": True,
        "enabled": True,
    })
    adapter = MixingStationAdapter(config)
    result = adapter.discover_available_paths(out=out)
    print(json.dumps({"out": out, **result}, indent=2, ensure_ascii=True))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
