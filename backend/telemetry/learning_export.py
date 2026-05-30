"""Load telemetry sessions and export training timelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from .schemas import EVENT_FILES, SCHEMA_VERSION, utc_now
    from .session_manager import DEFAULT_STORAGE_DIR
except ImportError:  # pragma: no cover - direct PYTHONPATH=backend compatibility
    from telemetry.schemas import EVENT_FILES, SCHEMA_VERSION, utc_now
    from telemetry.session_manager import DEFAULT_STORAGE_DIR


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def load_session(session_id: str, *, storage_dir: str | Path = DEFAULT_STORAGE_DIR) -> Dict[str, Any]:
    session_dir = Path(storage_dir) / session_id
    events = {
        event_type: _read_jsonl(session_dir / filename)
        for event_type, filename in EVENT_FILES.items()
    }
    return {
        "session_id": session_id,
        "session_dir": str(session_dir),
        "session": _read_json(session_dir / "session.json"),
        "scene_metadata": _read_json(session_dir / "scene_metadata.json"),
        "manifest": _read_json(session_dir / "manifest.json"),
        "events": events,
    }


def export_training_timeline(
    session_id: str,
    *,
    storage_dir: str | Path = DEFAULT_STORAGE_DIR,
    output_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    loaded = load_session(session_id, storage_dir=storage_dir)
    timeline: List[Dict[str, Any]] = []
    for event_type, rows in loaded["events"].items():
        for row in rows:
            timeline.append({
                "event_type": event_type,
                "timestamp_ms": int(row.get("timestamp_ms") or 0),
                "session_id": row.get("session_id") or session_id,
                "payload": row,
            })
    timeline.sort(key=lambda item: (item["timestamp_ms"], item["event_type"]))
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in timeline:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return timeline


def _event_counts(events: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    return {event_type: len(rows) for event_type, rows in events.items()}


def generate_summary_report(
    session_id: str,
    *,
    storage_dir: str | Path = DEFAULT_STORAGE_DIR,
    write: bool = True,
) -> Dict[str, Any]:
    loaded = load_session(session_id, storage_dir=storage_dir)
    timeline = export_training_timeline(session_id, storage_dir=storage_dir)
    counts = _event_counts(loaded["events"])
    summary = {
        "type": "telemetry_learning_summary",
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "session_id": session_id,
        "session": loaded["session"],
        "scene_metadata": loaded["scene_metadata"],
        "event_counts": counts,
        "timeline_event_count": len(timeline),
        "safety": {
            "read_only": True,
            "live_write": False,
        },
    }
    if write:
        session_dir = Path(storage_dir) / session_id
        report_path = session_dir / "summary_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_summary_markdown(summary), encoding="utf-8")
        summary["report_path"] = str(report_path)
    return summary


def _summary_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        f"# Telemetry Learning Summary {summary['session_id']}",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Schema version: {summary['schema_version']}",
        f"- Timeline events: {summary['timeline_event_count']}",
        "- Safety: read-only, no live mixer writes",
        "",
        "## Event Counts",
        "",
    ]
    for event_type, count in sorted(summary["event_counts"].items()):
        lines.append(f"- {event_type}: {count}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export Automixer learning telemetry")
    parser.add_argument("--storage-dir", default=DEFAULT_STORAGE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    summarize = sub.add_parser("summarize")
    summarize.add_argument("--session-id", required=True)

    timeline = sub.add_parser("timeline")
    timeline.add_argument("--session-id", required=True)
    timeline.add_argument("--output", default=None)

    args = parser.parse_args(argv)
    if args.command == "summarize":
        result = generate_summary_report(args.session_id, storage_dir=args.storage_dir)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "timeline":
        rows = export_training_timeline(args.session_id, storage_dir=args.storage_dir, output_path=args.output)
        print(json.dumps({"session_id": args.session_id, "timeline_event_count": len(rows)}, ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
