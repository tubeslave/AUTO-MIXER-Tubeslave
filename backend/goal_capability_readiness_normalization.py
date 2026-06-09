"""Shared helpers for readiness payload normalization used across snapshots."""

from __future__ import annotations

from typing import Any, Dict, List


def _safe_int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_str_list(raw_value: Any) -> List[str]:
    if isinstance(raw_value, (list, tuple)):
        return [str(item) for item in raw_value]
    return []


def normalize_goal_capability_readiness_goal_area_priorities(
    raw_priorities: Any,
) -> List[Any]:
    """Normalize list-like goal area priority rows for reliable UI snapshots."""
    if isinstance(raw_priorities, tuple):
        raw_priorities = list(raw_priorities)
    return list(raw_priorities) if isinstance(raw_priorities, list) else []


def normalize_goal_capability_readiness_rows(raw_rows: Any) -> Dict[str, Any]:
    """Normalize objective readiness metrics by goal area.

    The function intentionally preserves only stable, typed values and drops malformed
    entries so downstream UI/persistence layers can compare snapshots reliably.
    """
    if not isinstance(raw_rows, dict):
        return {}

    normalized: Dict[str, Dict[str, Any]] = {}
    for goal_area, raw_row in raw_rows.items():
        if not isinstance(raw_row, dict):
            continue

        normalized_row: Dict[str, Any] = {}
        raw_goal_area = raw_row.get("goal_area")
        if isinstance(raw_goal_area, str):
            normalized_goal_area = raw_goal_area.strip()
            if normalized_goal_area:
                normalized_row["goal_area"] = normalized_goal_area

        has_field = False

        for field in (
            "required_row_count",
            "ready_row_count",
            "gap_row_count",
            "readiness_score",
            "blocker_row_count",
            "target_live_writable_row_count",
            "target_proof_gap_count",
        ):
            raw_value = raw_row.get(field)
            if raw_value is None:
                continue
            normalized_value = _safe_int_value(raw_value, default=None)  # type: ignore[arg-type]
            if normalized_value is not None:
                normalized_row[field] = normalized_value
                has_field = True

        for field in ("status_counts", "ready_row_ids", "gap_row_ids"):
            raw_value = raw_row.get(field)
            if field == "status_counts":
                if isinstance(raw_value, dict):
                    normalized_row[field] = {
                        str(status): _safe_int_value(value)
                        for status, value in raw_value.items()
                    }
                    has_field = has_field or bool(normalized_row[field])
            else:
                normalized_values = _normalize_str_list(raw_value)
                if normalized_values:
                    normalized_row[field] = normalized_values
                has_field = has_field or bool(normalized_values)

        raw_blocker_rows = raw_row.get("blocker_rows")
        if isinstance(raw_blocker_rows, list):
            normalized_blocker_rows = [str(item) for item in raw_blocker_rows]
            if normalized_blocker_rows:
                normalized_row["blocker_rows"] = normalized_blocker_rows
                normalized_row["blocker_row_ids"] = normalized_blocker_rows
                has_field = True
        elif "blocker_rows" in raw_row:
            has_field = has_field or False

        raw_blocker_row_ids = raw_row.get("blocker_row_ids")
        if isinstance(raw_blocker_row_ids, list):
            normalized_blocker_row_ids = [str(item) for item in raw_blocker_row_ids]
            normalized_row["blocker_row_ids"] = normalized_blocker_row_ids
            has_field = has_field or bool(normalized_blocker_row_ids)

        for status_field, payload_field in (
            ("full_autonomous_status", "full_autonomous_status"),
            ("full_autonomous_status_label", "full_autonomous_status_label"),
        ):
            raw_status = raw_row.get(status_field)
            if isinstance(raw_status, str) and raw_status:
                normalized_row[payload_field] = raw_status
                has_field = True

        if not has_field:
            continue

        normalized[goal_area] = normalized_row

    return normalized


def normalize_goal_capability_readiness_hotspots(raw_rows: Any) -> List[Dict[str, Any]]:
    """Normalize objective readiness hotspot rows from progress snapshots."""
    if not isinstance(raw_rows, list):
        return []

    normalized: list[Dict[str, Any]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue

        area = str(raw_row.get("goal_area", "")).strip()
        if not area:
            continue

        normalized_row: Dict[str, Any] = {"goal_area": area}
        has_field = False

        for field in ("gap_row_count", "readiness_score", "blocker_row_count"):
            raw_value = raw_row.get(field)
            if raw_value is None:
                continue
            parsed = _safe_int_value(raw_value, default=None)
            if parsed is not None:
                normalized_row[field] = parsed
                has_field = True

        raw_blocker_row_ids = raw_row.get("blocker_row_ids")
        normalized_blocker_row_ids = _normalize_str_list(raw_blocker_row_ids)
        if normalized_blocker_row_ids:
            normalized_row["blocker_row_ids"] = normalized_blocker_row_ids
        has_field = has_field or bool(normalized_blocker_row_ids)

        if has_field:
            normalized.append(normalized_row)

    return normalized
