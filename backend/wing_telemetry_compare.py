"""Offline summary and comparison helpers for WING telemetry JSONL sessions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


METER_HINTS = ("/$meters/",)
FALLBACK_HINTS = ("/fdr", "/mute", "/name", "/$name")


def _resolve_session_paths(path: str | Path) -> Tuple[Path, Path]:
    candidate = Path(path)
    if candidate.is_dir():
        events_path = candidate / "events.jsonl"
        metadata_path = candidate / "metadata.json"
    else:
        if candidate.name != "events.jsonl":
            raise FileNotFoundError(f"Expected a session directory or events.jsonl file, got: {candidate}")
        events_path = candidate
        metadata_path = candidate.with_name("metadata.json")

    if not events_path.exists():
        raise FileNotFoundError(f"Telemetry events not found: {events_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Telemetry metadata not found: {metadata_path}")
    return events_path, metadata_path


def load_session(path: str | Path) -> Dict[str, Any]:
    events_path, metadata_path = _resolve_session_paths(path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with events_path.open("r", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    return {
        "session_dir": str(events_path.parent),
        "events_path": str(events_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
        "events": events,
    }


def _count_addresses(events: Iterable[Dict[str, Any]], *, event_type: str | None = None) -> Counter[str]:
    counts: Counter[str] = Counter()
    for event in events:
        if event_type is not None and event.get("event_type") != event_type:
            continue
        address = event.get("address")
        if address:
            counts[address] += 1
    return counts


def _classify_observed_addresses(addresses: Iterable[str]) -> Dict[str, List[str]]:
    meter = []
    fallback = []
    other = []
    for address in sorted(set(addresses)):
        if any(hint in address for hint in METER_HINTS):
            meter.append(address)
        elif any(address.endswith(hint) for hint in FALLBACK_HINTS):
            fallback.append(address)
        else:
            other.append(address)
    return {
        "meter": meter,
        "fallback": fallback,
        "other": other,
    }


def summarize_session(path: str | Path) -> Dict[str, Any]:
    session = load_session(path)
    events = session["events"]
    event_type_counts = Counter(event.get("event_type", "unknown") for event in events)
    address_counts = _count_addresses(events)
    observed_address_counts = _count_addresses(events, event_type="console_observed")
    observed_classes = _classify_observed_addresses(observed_address_counts.keys())
    timestamps = [event.get("timestamp") for event in events if isinstance(event.get("timestamp"), (int, float))]

    return {
        "session_dir": session["session_dir"],
        "session_id": session["metadata"].get("session_id"),
        "session_label": session["metadata"].get("session_label"),
        "event_count": len(events),
        "duration_sec": (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0,
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "address_counts": dict(sorted(address_counts.items())),
        "observed_address_counts": dict(sorted(observed_address_counts.items())),
        "observed_address_classes": observed_classes,
        "blocked_write_count": event_type_counts.get("blocked_write", 0),
        "write_sent_count": event_type_counts.get("write_sent", 0),
        "query_sent_count": event_type_counts.get("query_sent", 0),
        "console_observed_count": event_type_counts.get("console_observed", 0),
        "fallback_only_observation": bool(
            observed_classes["fallback"]
            and not observed_classes["meter"]
            and not observed_classes["other"]
        ),
        "metadata": session["metadata"].get("metadata", {}),
    }


def compare_sessions(left_path: str | Path, right_path: str | Path) -> Dict[str, Any]:
    left = summarize_session(left_path)
    right = summarize_session(right_path)

    left_event_types = Counter(left["event_type_counts"])
    right_event_types = Counter(right["event_type_counts"])
    left_observed = Counter(left["observed_address_counts"])
    right_observed = Counter(right["observed_address_counts"])

    shared_addresses = sorted(set(left_observed) & set(right_observed))
    count_mismatches = []
    for address in shared_addresses:
        if left_observed[address] != right_observed[address]:
            count_mismatches.append(
                {
                    "address": address,
                    "left_count": left_observed[address],
                    "right_count": right_observed[address],
                }
            )

    return {
        "left": left,
        "right": right,
        "event_type_delta": {
            key: left_event_types.get(key, 0) - right_event_types.get(key, 0)
            for key in sorted(set(left_event_types) | set(right_event_types))
        },
        "observed_address_only_in_left": sorted(set(left_observed) - set(right_observed)),
        "observed_address_only_in_right": sorted(set(right_observed) - set(left_observed)),
        "observed_address_count_mismatches": count_mismatches,
        "safety_flags": {
            "left_has_write_sent": left["write_sent_count"] > 0,
            "right_has_write_sent": right["write_sent_count"] > 0,
            "left_fallback_only": left["fallback_only_observation"],
            "right_fallback_only": right["fallback_only_observation"],
        },
    }


def _main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize or diff WING telemetry sessions.")
    parser.add_argument("left", help="Session directory or events.jsonl for the left/live capture")
    parser.add_argument("right", nargs="?", help="Optional session directory or events.jsonl for the right/replay capture")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args(argv)

    payload = summarize_session(args.left) if args.right is None else compare_sessions(args.left, args.right)
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
