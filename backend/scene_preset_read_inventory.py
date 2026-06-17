"""Read-only scene and preset catalog inventory.

This module is transport-free. It normalizes already captured Product Layer
scene and preset catalog rows into a read-only inventory artifact that
downstream scene/preset proposal layers can inspect before any operator review.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

try:
    from read_surface_inventory import get_read_surface
except ImportError:  # pragma: no cover - package import compatibility
    from backend.read_surface_inventory import get_read_surface


SCENE_PRESET_READ_INVENTORY_METHOD_ID = "scene_preset_read_inventory"
SCENE_PRESET_READ_INVENTORY_VERSION = "scene_preset_read_inventory_v1"
REQUIRED_SURFACE_IDS = ("snapshots_scenes", "user_presets")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric >= 0 else None


def _token(value: Any, default: str = "") -> str:
    return str(value if value is not None else default).strip()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "active"}
    return bool(value)


def _freshness_payload(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"status": "not_provided", "provided": False}
    allowed = (
        "status",
        "source",
        "field_count",
        "fresh_field_count",
        "stale_field_count",
        "unknown_field_count",
        "missing_field_count",
        "stale_fields",
        "missing_fields",
        "stale_after_sec",
        "read_only",
        "live_write",
        "queries_console",
    )
    normalized = {key: payload.get(key) for key in allowed if key in payload}
    normalized.setdefault("status", _token(payload.get("status"), "unknown").lower())
    normalized["provided"] = True
    return normalized


def _surface_freshness(
    surface_id: str,
    surface_freshness: Optional[Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Any]:
    if not isinstance(surface_freshness, Mapping):
        return {"status": "not_provided", "provided": False}
    return _freshness_payload(surface_freshness.get(surface_id))


def _row_id(row: Mapping[str, Any], *keys: str, fallback: str) -> str:
    for key in keys:
        value = _token(row.get(key))
        if value:
            return value
    return fallback


def _normalize_scene_rows(rows: Iterable[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        scene_id = _row_id(row, "scene_id", "snapshot_id", "id", fallback=f"scene:{index}")
        load_policy = _token(row.get("scene_load_policy") or row.get("load_policy"))
        save_policy = _token(row.get("scene_save_policy") or row.get("save_policy"))
        marker = _token(row.get("destructive_change_marker"))
        normalized.append(
            {
                "id": scene_id,
                "surface_id": "snapshots_scenes",
                "snapshot_index": _safe_int(
                    row.get("snapshot_index") or row.get("scene_index") or row.get("index")
                ),
                "snapshot_name": _token(
                    row.get("snapshot_name") or row.get("scene_name") or row.get("name")
                ),
                "active": _bool(row.get("active") or row.get("is_active")),
                "active_snapshot_name": _token(row.get("active_snapshot_name")),
                "current_show": _token(row.get("current_show")),
                "scene_load_policy": load_policy,
                "scene_load_policy_present": bool(load_policy),
                "scene_save_policy": save_policy,
                "scene_save_policy_present": bool(save_policy),
                "destructive_change_marker": marker,
                "destructive_change_marker_present": bool(marker),
                "cache_freshness": _freshness_payload(row.get("cache_freshness")),
                "read_coverage": row.get("read_coverage"),
            }
        )
    return tuple(normalized)


def _normalize_preset_rows(rows: Iterable[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        preset_id = _row_id(row, "preset_id", "user_preset_id", "id", fallback=f"preset:{index}")
        scope = _token(row.get("preset_scope") or row.get("scope") or row.get("target_type"))
        recall_policy = _token(row.get("preset_recall_policy") or row.get("recall_policy"))
        store_policy = _token(row.get("preset_store_policy") or row.get("store_policy"))
        marker = _token(row.get("destructive_recall_marker"))
        normalized.append(
            {
                "id": preset_id,
                "surface_id": "user_presets",
                "preset_index": _safe_int(row.get("preset_index") or row.get("index")),
                "preset_name": _token(row.get("preset_name") or row.get("name")),
                "preset_scope": scope,
                "preset_scope_present": bool(scope),
                "target_type": _token(row.get("target_type")),
                "target_label": _token(row.get("target_label")),
                "preset_recall_policy": recall_policy,
                "preset_recall_policy_present": bool(recall_policy),
                "preset_store_policy": store_policy,
                "preset_store_policy_present": bool(store_policy),
                "destructive_recall_marker": marker,
                "destructive_recall_marker_present": bool(marker),
                "cache_freshness": _freshness_payload(row.get("cache_freshness")),
                "read_coverage": row.get("read_coverage"),
            }
        )
    return tuple(normalized)


def build_scene_preset_read_inventory(
    *,
    scene_rows: Iterable[Mapping[str, Any]] = (),
    preset_rows: Iterable[Mapping[str, Any]] = (),
    surface_freshness: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a read-only inventory from cached scene and preset catalog rows."""

    scenes = _normalize_scene_rows(scene_rows)
    presets = _normalize_preset_rows(preset_rows)
    rows_by_surface = {
        "snapshots_scenes": scenes,
        "user_presets": presets,
    }
    blockers = []
    surface_payloads: Dict[str, Dict[str, Any]] = {}
    for surface_id in REQUIRED_SURFACE_IDS:
        inventory = get_read_surface(surface_id)
        freshness = _surface_freshness(surface_id, surface_freshness)
        freshness_status = str(freshness.get("status") or "unknown")
        if not freshness.get("provided"):
            blockers.append(f"surface_snapshot_missing:{surface_id}")
        elif freshness_status != "fresh":
            blockers.append(f"surface_not_fresh:{surface_id}:{freshness_status}")
        surface_payloads[surface_id] = {
            "surface_id": surface_id,
            "inventory_status": inventory.status if inventory else "missing",
            "inventory_blocker": inventory.blocker if inventory else "inventory_row_missing",
            "freshness": freshness,
            "row_count": len(rows_by_surface[surface_id]),
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
        }

    for index, row in enumerate(scenes):
        row_freshness_status = str(row.get("cache_freshness", {}).get("status") or "unknown")
        if row_freshness_status != "fresh":
            blockers.append(f"scene_row_{index}_cache_not_fresh:{row_freshness_status}")
        if not row.get("destructive_change_marker_present"):
            blockers.append(f"scene_row_{index}_destructive_change_marker_missing")
        if not row.get("scene_load_policy_present"):
            blockers.append(f"scene_row_{index}_load_policy_missing")
        if not row.get("scene_save_policy_present"):
            blockers.append(f"scene_row_{index}_save_policy_missing")
    for index, row in enumerate(presets):
        row_freshness_status = str(row.get("cache_freshness", {}).get("status") or "unknown")
        if row_freshness_status != "fresh":
            blockers.append(f"preset_row_{index}_cache_not_fresh:{row_freshness_status}")
        if not row.get("preset_scope_present"):
            blockers.append(f"preset_row_{index}_scope_missing")
        if not row.get("destructive_recall_marker_present"):
            blockers.append(f"preset_row_{index}_destructive_recall_marker_missing")
        if not row.get("preset_recall_policy_present"):
            blockers.append(f"preset_row_{index}_recall_policy_missing")
        if not row.get("preset_store_policy_present"):
            blockers.append(f"preset_row_{index}_store_policy_missing")
    if not scenes and not presets:
        blockers.append("scene_preset_read_inventory_empty")

    status = "read_inventory_ready" if not blockers else "read_inventory_incomplete"
    return {
        "type": "scene_preset_read_inventory",
        "version": SCENE_PRESET_READ_INVENTORY_VERSION,
        "generated_at": utc_now(),
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "queue_mutation": False,
            "scene_load_allowed": False,
            "scene_save_allowed": False,
            "preset_recall_allowed": False,
            "preset_store_allowed": False,
        },
        "required_surface_ids": list(REQUIRED_SURFACE_IDS),
        "summary": {
            "status": status,
            "blockers": sorted(set(blockers)),
            "scene_count": len(scenes),
            "preset_count": len(presets),
            "fresh_surface_ids": [
                surface_id
                for surface_id, payload in surface_payloads.items()
                if payload["freshness"].get("status") == "fresh"
            ],
        },
        "surfaces": surface_payloads,
        "inventory": {
            "scenes": [dict(row) for row in scenes],
            "presets": [dict(row) for row in presets],
        },
    }
