#!/usr/bin/env python3
"""Audit long-run soundcheck proposal logs for drift-prone actions.

The audit is offline only. It reads JSONL evidence from prior observation or
proposal loops and writes a summary artifact. A passing audit proves only that
the supplied log stayed within the proposal/drift contract; it does not grant
WING readiness or live authority.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


DEFAULT_ARTIFACT_ROOT = "artifacts/soundcheck_long_run_drift_audit"
DEFAULT_MIN_DURATION_SEC = 600.0
DEFAULT_MAX_CUMULATIVE_DB = 3.0
EPSILON_DB = 1e-9

ACTION_COLLECTIONS = ("applied", "proposals", "proposed", "recommendations")
ABSOLUTE_TARGET_KEYS = (
    "target_fader_db",
    "target_trim_db",
    "target_gain_db",
    "target_value_db",
    "target_db",
    "new_value_db",
    "absolute_target_db",
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _read_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                yield line_number, {
                    "audit_parse_error": True,
                    "error": f"invalid_json:{exc.msg}",
                    "raw": stripped[:200],
                }
                continue
            if isinstance(payload, dict):
                yield line_number, payload
            else:
                yield line_number, {"audit_parse_error": True, "error": "jsonl_row_not_object"}


def _record_value(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested_search(mapping: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    key_set = set(keys)
    stack: List[Any] = [mapping]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, item in value.items():
                if key in key_set:
                    number = _safe_float(item)
                    if number is not None:
                        return number
                if isinstance(item, (dict, list)):
                    stack.append(item)
        elif isinstance(value, list):
            stack.extend(value)
    return None


def _relative_move_db(action: Dict[str, Any]) -> Optional[float]:
    params = _record_value(action.get("parameters"))
    proposed = _record_value(action.get("proposed_action"))
    for source in (params, proposed, action):
        for key in ("gain_db", "delta_db", "move_db", "value_db"):
            number = _safe_float(source.get(key))
            if number is not None:
                return number
    return None


def _absolute_target_db(action: Dict[str, Any]) -> Optional[float]:
    return _nested_search(action, ABSOLUTE_TARGET_KEYS)


def _surface_id(action: Dict[str, Any]) -> str:
    action_id = str(action.get("id") or action.get("proposal_id") or "").strip()
    target = str(action.get("target") or "").strip()
    if action_id:
        return action_id
    if target:
        return target
    return str(action.get("action_type") or "unknown").strip() or "unknown"


def _surface_kind(action: Dict[str, Any]) -> str:
    surface = _surface_id(action).lower()
    target = str(action.get("target") or "").lower()
    action_type = str(action.get("action_type") or "").lower()
    if surface.startswith("eq.") or ".eq." in target or "/eq/" in target:
        return "eq"
    if surface.startswith("fader.") or target.endswith(".fdr") or ".fdr" in target:
        return "fader"
    if surface.startswith("trim.") or re.fullmatch(r"ch\d+", target):
        return "trim"
    if "gain" in action_type or "trim" in action_type:
        return "gain"
    return action_type or "other"


def _safety_gate(action: Dict[str, Any]) -> Dict[str, Any]:
    return _record_value(_record_value(action.get("metadata")).get("safety_gate"))


def _action_rows(event: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    for collection in ACTION_COLLECTIONS:
        rows = event.get(collection)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    yield collection, row
    blocked = event.get("blocked")
    if isinstance(blocked, list):
        for row in blocked:
            if not isinstance(row, dict):
                continue
            decision = row.get("decision")
            if isinstance(decision, dict):
                yield "blocked.decision", decision


def build_audit_summary(
    evidence_paths: Iterable[Path],
    *,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    max_cumulative_db: float = DEFAULT_MAX_CUMULATIVE_DB,
) -> Dict[str, Any]:
    paths = [Path(path) for path in evidence_paths]
    events = 0
    parse_errors: List[Dict[str, Any]] = []
    cycles: List[int] = []
    timestamps: List[float] = []
    actions = 0
    applied_actions = 0
    live_executor_actions = 0
    positive_fader_boosts: List[Dict[str, Any]] = []
    positive_gain_boosts: List[Dict[str, Any]] = []
    relative_target_actions: List[Dict[str, Any]] = []
    cumulative_relative_db: Dict[str, float] = defaultdict(float)
    action_counts_by_surface: Dict[str, int] = defaultdict(int)
    action_counts_by_collection: Dict[str, int] = defaultdict(int)

    for path in paths:
        for line_number, event in _read_jsonl(path):
            events += 1
            if event.get("audit_parse_error"):
                parse_errors.append({"path": str(path), "line": line_number, "error": event.get("error")})
                continue
            cycle = event.get("cycle")
            if isinstance(cycle, int):
                cycles.append(cycle)
            event_time = _safe_float(event.get("t"))
            if event_time is not None:
                timestamps.append(event_time)
            for collection, action in _action_rows(event):
                actions += 1
                action_counts_by_collection[collection] += 1
                surface = _surface_id(action)
                kind = _surface_kind(action)
                move_db = _relative_move_db(action)
                absolute_target = _absolute_target_db(action)
                action_counts_by_surface[surface] += 1
                if collection == "applied":
                    applied_actions += 1
                    gate = _safety_gate(action)
                    dry_run = gate.get("dry_run")
                    executor = gate.get("send_to_executor")
                    if dry_run is not True and executor is not False:
                        live_executor_actions += 1
                if move_db is not None and move_db > EPSILON_DB:
                    row = {
                        "path": str(path),
                        "line": line_number,
                        "surface": surface,
                        "kind": kind,
                        "move_db": move_db,
                        "collection": collection,
                    }
                    if kind == "fader":
                        positive_fader_boosts.append(row)
                    elif kind in {"trim", "gain"}:
                        positive_gain_boosts.append(row)
                if kind in {"fader", "trim", "gain"} and absolute_target is None:
                    relative_target_actions.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "surface": surface,
                            "kind": kind,
                            "move_db": move_db,
                            "collection": collection,
                        }
                    )
                    if move_db is not None:
                        cumulative_relative_db[surface] += move_db

    duration_sec = 0.0
    if timestamps:
        duration_sec = round(max(timestamps) - min(timestamps), 3)
    cumulative_drift = [
        {"surface": surface, "cumulative_relative_db": round(value, 3)}
        for surface, value in sorted(cumulative_relative_db.items())
        if abs(value) > max_cumulative_db + EPSILON_DB
    ]
    failures: List[str] = []
    if parse_errors:
        failures.append("invalid_jsonl_rows")
    if duration_sec < min_duration_sec:
        failures.append("duration_below_minimum")
    if actions == 0:
        failures.append("proposal_actions_missing")
    if live_executor_actions:
        failures.append("live_executor_actions_present")
    if positive_fader_boosts:
        failures.append("positive_fader_boosts_present")
    if positive_gain_boosts:
        failures.append("positive_gain_or_trim_boosts_present")
    if relative_target_actions:
        failures.append("relative_only_targets_present")
    if cumulative_drift:
        failures.append("cumulative_relative_drift_present")

    status = "passed" if not failures else "failed"
    return {
        "schema": "soundcheck_long_run_drift_audit.v1",
        "generated_at": iso_now(),
        "status": status,
        "failures": failures,
        "proof_scope": "offline_log_audit_only_no_wing_readiness",
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "readiness_matrix_update_allowed": False,
        "evidence_paths": [str(path) for path in paths],
        "min_duration_sec": min_duration_sec,
        "duration_sec": duration_sec,
        "event_count": events,
        "cycle_count": len(set(cycles)),
        "cycle_min": min(cycles) if cycles else None,
        "cycle_max": max(cycles) if cycles else None,
        "action_count": actions,
        "applied_action_count": applied_actions,
        "live_executor_action_count": live_executor_actions,
        "positive_fader_boost_count": len(positive_fader_boosts),
        "positive_gain_or_trim_boost_count": len(positive_gain_boosts),
        "relative_only_target_count": len(relative_target_actions),
        "cumulative_drift_surface_count": len(cumulative_drift),
        "max_cumulative_db": max_cumulative_db,
        "action_counts_by_collection": dict(sorted(action_counts_by_collection.items())),
        "action_counts_by_surface": dict(sorted(action_counts_by_surface.items())),
        "positive_fader_boosts": positive_fader_boosts[:50],
        "positive_gain_or_trim_boosts": positive_gain_boosts[:50],
        "relative_only_targets": relative_target_actions[:50],
        "cumulative_relative_drift": cumulative_drift[:50],
        "parse_errors": parse_errors[:50],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit soundcheck long-run drift evidence.")
    parser.add_argument("evidence", nargs="+", help="JSONL evidence files to audit")
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--min-duration-sec", type=float, default=DEFAULT_MIN_DURATION_SEC)
    parser.add_argument("--max-cumulative-db", type=float, default=DEFAULT_MAX_CUMULATIVE_DB)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    evidence_paths = [Path(item) for item in args.evidence]
    missing = [str(path) for path in evidence_paths if not path.exists()]
    run_id = f"{utc_stamp()}_soundcheck_long_run_drift_audit"
    run_dir = Path(args.artifact_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if missing:
        summary = {
            "schema": "soundcheck_long_run_drift_audit.v1",
            "generated_at": iso_now(),
            "status": "blocked",
            "failures": ["evidence_file_missing"],
            "proof_scope": "offline_log_audit_only_no_wing_readiness",
            "missing_evidence_paths": missing,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "readiness_matrix_update_allowed": False,
        }
    else:
        summary = build_audit_summary(
            evidence_paths,
            min_duration_sec=args.min_duration_sec,
            max_cumulative_db=args.max_cumulative_db,
        )
    summary["summary_path"] = str(summary_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
