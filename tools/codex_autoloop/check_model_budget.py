#!/usr/bin/env python3
"""Select a conservative Codex model profile for one Automixer autoloop cycle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_PROFILE = SCRIPT_DIR / "model_budget_profile.json"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def period_keys(now: datetime) -> dict[str, str]:
    iso_year, iso_week, _ = now.isocalendar()
    return {
        "day": now.strftime("%Y-%m-%d"),
        "week": f"{iso_year}-W{iso_week:02d}",
        "month": now.strftime("%Y-%m"),
    }


def count_events(events: list[dict[str, Any]], *, profile: str, keys: dict[str, str]) -> dict[str, int]:
    counts = {"day": 0, "week": 0, "month": 0}
    for event in events:
        if event.get("profile") != profile:
            continue
        for period, key in keys.items():
            if event.get(period) == key:
                counts[period] += 1
    return counts


def profile_has_budget(profile: dict[str, Any], counts: dict[str, int]) -> bool:
    limits = {
        "day": int(profile.get("daily_limit", 0)),
        "week": int(profile.get("weekly_limit", 0)),
        "month": int(profile.get("monthly_limit", 0)),
    }
    return all(counts[period] < limit for period, limit in limits.items())


def choose_profile(config: dict[str, Any], requested: str | None, task_text: str) -> str | None:
    profiles = config["profiles"]
    if requested:
        if requested not in profiles:
            raise ValueError(f"Unknown profile: {requested}")
        return requested

    lowered = task_text.lower()
    for route in config.get("task_routes", []):
        terms = [str(term).lower() for term in route.get("match", [])]
        if any(term in lowered for term in terms):
            profile = str(route["profile"])
            if profile in profiles:
                return profile
    return str(config.get("default_profile", "budget_balanced"))


def fallback_order(config: dict[str, Any], first: str) -> list[str]:
    profiles = config["profiles"]
    order = [first]
    fallback = str(config.get("fallback_profile", "budget_light"))
    if fallback not in order and fallback in profiles:
        order.append(fallback)
    for name in ("budget_light", "budget_balanced", "budget_strict"):
        if name not in order and name in profiles:
            order.append(name)
    return order


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def render_env(selection: dict[str, Any]) -> str:
    lines = []
    for key, value in selection.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        lines.append(f"{key}={shell_quote(str(value))}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-config", default=str(DEFAULT_PROFILE))
    parser.add_argument("--state", default=None)
    parser.add_argument("--task", default="")
    parser.add_argument("--reserve", action="store_true")
    parser.add_argument("--format", choices=("env", "json"), default="env")
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.profile_config).resolve()
    config = load_json(config_path, {})
    if config.get("schema") != "automixer.codex_autoloop.model_budget.v1":
        raise ValueError(f"Unexpected model budget schema in {config_path}")

    state_path = Path(args.state or config["state_path"])
    if not state_path.is_absolute():
        state_path = PROJECT_ROOT / state_path

    state = load_json(state_path, {"schema": config["schema"], "events": []})
    events = state.setdefault("events", [])
    if not isinstance(events, list):
        raise ValueError(f"Expected events list in {state_path}")

    now = utc_now()
    keys = period_keys(now)
    requested = os.environ.get(config["overrides"]["profile_env"])
    requested = requested.strip() if requested else None
    requested_model = os.environ.get(config["overrides"]["model_env"])
    requested_reasoning = os.environ.get(config["overrides"]["reasoning_env"])

    preferred = choose_profile(config, requested, args.task)
    selected_name = None
    selected_profile = None
    selected_counts = None
    for candidate in fallback_order(config, preferred):
        profile = config["profiles"][candidate]
        counts = count_events(events, profile=candidate, keys=keys)
        if profile_has_budget(profile, counts):
            selected_name = candidate
            selected_profile = profile
            selected_counts = counts
            break

    if not selected_name or not selected_profile or selected_counts is None:
        message = config.get("skip_policy", {}).get(
            "message",
            "Automixer autoloop skipped because all model profiles are exhausted.",
        )
        selection = {
            "CODEX_AUTOLOOP_SHOULD_RUN": "0",
            "CODEX_AUTOLOOP_SKIP_REASON": message,
            "CODEX_AUTOLOOP_BUDGET_STATE": str(state_path),
        }
    else:
        model = requested_model or str(selected_profile["model"])
        reasoning = requested_reasoning or str(selected_profile["reasoning_effort"])
        event = {
            "reserved_at": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "profile": selected_name,
            "model": model,
            "reasoning_effort": reasoning,
            **keys,
        }
        if args.reserve:
            events.append(event)
            state["last_selection"] = event
            write_json(state_path, state)
        selection = {
            "CODEX_AUTOLOOP_SHOULD_RUN": "1",
            "CODEX_AUTOLOOP_SELECTED_PROFILE": selected_name,
            "CODEX_AUTOLOOP_SELECTED_MODEL": model,
            "CODEX_AUTOLOOP_SELECTED_REASONING_EFFORT": reasoning,
            "CODEX_AUTOLOOP_BUDGET_STATE": str(state_path),
            "CODEX_AUTOLOOP_BUDGET_COUNTS": selected_counts,
        }

    text = (
        json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else render_env(selection)
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
