#!/usr/bin/env python3
"""Send one test AutomixCorrection to Mixing Station."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from integrations.mixing_station.adapter import MixingStationAdapter
from integrations.mixing_station.config import MixingStationConfig, normalize_console_profile
from integrations.mixing_station.models import AutomixCorrection


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_value(value: str, unit: str):
    if unit == "bool":
        return parse_bool(value)
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--console", default="wing_rack")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--strip-type", default="input")
    parser.add_argument("--parameter", default="fader")
    parser.add_argument("--value", required=True)
    parser.add_argument("--unit", default="db")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--osc-port", type=int, default=9000)
    parser.add_argument("--transport", default="websocket")
    parser.add_argument("--dry-run", default="true")
    parser.add_argument("--mode", default="offline_visualization")
    args = parser.parse_args()

    profile = normalize_console_profile(args.console)
    config = MixingStationConfig.from_mapping({
        "enabled": True,
        "console_profile": profile,
        "host": args.host,
        "rest_port": args.port,
        "osc_port": args.osc_port,
        "transport": args.transport,
        "dry_run": parse_bool(args.dry_run),
        "mode": args.mode,
    })
    adapter = MixingStationAdapter(config)
    correction = AutomixCorrection(
        console_profile=profile,
        mode=args.mode,
        channel_index=args.channel,
        strip_type=args.strip_type,
        parameter=args.parameter,
        value=parse_value(args.value, args.unit),
        value_unit=args.unit,
        reason="manual Mixing Station integration test",
        dry_run=config.dry_run,
    )
    result = adapter.send_correction(correction)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
