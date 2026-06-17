"""Read-only routing, group, DCA, and matrix topology map.

This module is transport-free. It normalizes already captured Product Layer
read rows into a topology artifact that downstream routing and bus/group
proposal layers can inspect before proposing changes.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

try:
    from read_surface_inventory import get_read_surface
except ImportError:  # pragma: no cover - package import compatibility
    from backend.read_surface_inventory import get_read_surface


ROUTING_GROUP_READ_MAP_METHOD_ID = "routing_group_read_map"
ROUTING_GROUP_READ_MAP_VERSION = "routing_group_read_map_v1"
REQUIRED_SURFACE_IDS = (
    "routing_and_sends",
    "bus_group_main_dca_state",
    "matrix_state",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _token(value: Any, default: str = "") -> str:
    return str(value or default).strip()


def _upper_token(value: Any, default: str = "") -> str:
    return _token(value, default).upper()


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


def _endpoint(group: Any, channel: Any = None, *, label: Any = "") -> Dict[str, Any]:
    return {
        "group": _upper_token(group),
        "channel": _safe_int(channel),
        "label": _token(label),
    }


def _endpoint_id(endpoint: Mapping[str, Any], fallback: str) -> str:
    group = _upper_token(endpoint.get("group"))
    channel = _safe_int(endpoint.get("channel"))
    if group and channel is not None:
        return f"{group}:{channel}"
    if group:
        return group
    return fallback


def _row_id(row: Mapping[str, Any], *keys: str, fallback: str) -> str:
    for key in keys:
        value = _token(row.get(key))
        if value:
            return value
    return fallback


def _members_from_row(row: Mapping[str, Any]) -> Tuple[str, ...]:
    raw_members = (
        row.get("members")
        or row.get("member_ids")
        or row.get("assigned_channels")
        or row.get("channels")
        or ()
    )
    if isinstance(raw_members, Mapping):
        raw_members = raw_members.values()
    members = []
    for item in raw_members if isinstance(raw_members, Iterable) and not isinstance(raw_members, (str, bytes)) else ():
        if isinstance(item, Mapping):
            endpoint = _endpoint(
                item.get("group") or item.get("source_group") or item.get("kind") or "CH",
                item.get("channel") or item.get("source_channel") or item.get("id"),
                label=item.get("label"),
            )
            member_id = _endpoint_id(endpoint, _token(item.get("id")))
        else:
            member_id = _token(item)
        if member_id:
            members.append(member_id)
    return tuple(sorted(set(members)))


def _normalize_route_rows(rows: Iterable[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        route_id = _row_id(row, "route_id", "id", fallback=f"route:{index}")
        source = _endpoint(
            row.get("source_group") or row.get("current_source_group"),
            row.get("source_channel") or row.get("current_source_channel"),
            label=row.get("source_label") or row.get("current_source_label"),
        )
        target = _endpoint(
            row.get("target_group") or row.get("output_group") or row.get("target_kind"),
            row.get("target_channel") or row.get("output_number"),
            label=row.get("target_label") or row.get("output_label"),
        )
        marker = _token(row.get("destructive_change_marker"))
        normalized.append(
            {
                "id": route_id,
                "surface_id": "routing_and_sends",
                "kind": _token(row.get("route_kind") or row.get("kind"), "route"),
                "source": source,
                "target": target,
                "readback_status": _token(row.get("readback_status"), "cached_or_static"),
                "cache_freshness": _freshness_payload(row.get("cache_freshness")),
                "destructive_change_marker": marker,
                "destructive_change_marker_present": bool(marker),
            }
        )
    return tuple(normalized)


def _normalize_bus_group_rows(rows: Iterable[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        bus_id = _row_id(row, "bus_id", "group_id", "dca_id", "id", fallback=f"bus_group:{index}")
        bus_kind = _token(row.get("bus_kind") or row.get("group_kind") or row.get("kind"), "bus").lower()
        members = _members_from_row(row)
        normalized.append(
            {
                "id": bus_id,
                "surface_id": "bus_group_main_dca_state",
                "kind": bus_kind,
                "label": _token(row.get("label") or row.get("name")),
                "members": list(members),
                "member_count": len(members),
                "fader_db": _safe_float(row.get("fader_db")),
                "mute": row.get("mute"),
                "cache_freshness": _freshness_payload(row.get("cache_freshness")),
            }
        )
    return tuple(normalized)


def _matrix_sources_from_row(row: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    raw_sources = row.get("sources") or row.get("sends") or row.get("matrix_sends") or ()
    if isinstance(raw_sources, Mapping):
        raw_sources = raw_sources.values()
    sources = []
    for index, item in enumerate(raw_sources if isinstance(raw_sources, Iterable) and not isinstance(raw_sources, (str, bytes)) else ()):
        if not isinstance(item, Mapping):
            source_id = _token(item)
            if source_id:
                sources.append({"source_id": source_id, "level_db": None, "on": None})
            continue
        endpoint = _endpoint(
            item.get("source_group") or item.get("group") or item.get("kind"),
            item.get("source_channel") or item.get("channel") or item.get("id"),
            label=item.get("label"),
        )
        fallback = _row_id(item, "source_id", "id", fallback=f"matrix_source:{index}")
        sources.append(
            {
                "source_id": _endpoint_id(endpoint, fallback),
                "level_db": _safe_float(item.get("level_db")),
                "on": item.get("on"),
            }
        )
    return tuple(sources)


def _normalize_matrix_rows(rows: Iterable[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        matrix_id = _row_id(row, "matrix_id", "id", fallback=f"matrix:{index + 1}")
        sources = _matrix_sources_from_row(row)
        normalized.append(
            {
                "id": matrix_id,
                "surface_id": "matrix_state",
                "kind": "matrix",
                "label": _token(row.get("label") or row.get("name")),
                "sources": list(sources),
                "source_count": len(sources),
                "fader_db": _safe_float(row.get("fader_db")),
                "mute": row.get("mute"),
                "cache_freshness": _freshness_payload(row.get("cache_freshness")),
            }
        )
    return tuple(normalized)


def _build_edges(
    routing_rows: Iterable[Mapping[str, Any]],
    bus_group_rows: Iterable[Mapping[str, Any]],
    matrix_rows: Iterable[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    edges = []
    for row in routing_rows:
        source = _endpoint_id(row.get("source", {}), f"{row.get('id')}:source")
        target = _endpoint_id(row.get("target", {}), str(row.get("id")))
        edges.append({"kind": "route", "from": source, "to": target, "surface_id": row.get("surface_id")})
    for row in bus_group_rows:
        for member in row.get("members", ()):
            edges.append({"kind": row.get("kind") or "bus_group", "from": member, "to": row.get("id"), "surface_id": row.get("surface_id")})
    for row in matrix_rows:
        for source in row.get("sources", ()):
            edges.append({"kind": "matrix_send", "from": source.get("source_id"), "to": row.get("id"), "level_db": source.get("level_db"), "surface_id": row.get("surface_id")})
    return tuple(edges)


def build_routing_group_read_map(
    *,
    routing_rows: Iterable[Mapping[str, Any]] = (),
    bus_group_rows: Iterable[Mapping[str, Any]] = (),
    matrix_rows: Iterable[Mapping[str, Any]] = (),
    surface_freshness: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a read-only topology map from cached/static routing surfaces."""

    routes = _normalize_route_rows(routing_rows)
    buses = _normalize_bus_group_rows(bus_group_rows)
    matrices = _normalize_matrix_rows(matrix_rows)
    rows_by_surface = {
        "routing_and_sends": routes,
        "bus_group_main_dca_state": buses,
        "matrix_state": matrices,
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

    for index, row in enumerate(routes):
        row_freshness_status = str(row.get("cache_freshness", {}).get("status") or "unknown")
        if row_freshness_status != "fresh":
            blockers.append(f"routing_row_{index}_cache_not_fresh:{row_freshness_status}")
        if not row.get("destructive_change_marker_present"):
            blockers.append(f"routing_row_{index}_destructive_change_marker_missing")
    for index, row in enumerate(buses):
        row_freshness_status = str(row.get("cache_freshness", {}).get("status") or "unknown")
        if row_freshness_status != "fresh":
            blockers.append(f"bus_group_row_{index}_cache_not_fresh:{row_freshness_status}")
        if row.get("kind") in {"group", "dca", "stem"} and not row.get("members"):
            blockers.append(f"bus_group_row_{index}_membership_missing")
    for index, row in enumerate(matrices):
        row_freshness_status = str(row.get("cache_freshness", {}).get("status") or "unknown")
        if row_freshness_status != "fresh":
            blockers.append(f"matrix_row_{index}_cache_not_fresh:{row_freshness_status}")
    if not routes and not buses and not matrices:
        blockers.append("routing_group_read_map_empty")

    edges = _build_edges(routes, buses, matrices)
    status = "read_map_ready" if not blockers else "read_map_incomplete"
    return {
        "type": "routing_group_read_map",
        "version": ROUTING_GROUP_READ_MAP_VERSION,
        "generated_at": utc_now(),
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "queue_mutation": False,
        },
        "required_surface_ids": list(REQUIRED_SURFACE_IDS),
        "summary": {
            "status": status,
            "blockers": sorted(set(blockers)),
            "route_count": len(routes),
            "bus_group_count": len(buses),
            "matrix_count": len(matrices),
            "edge_count": len(edges),
            "fresh_surface_ids": [
                surface_id
                for surface_id, payload in surface_payloads.items()
                if payload["freshness"].get("status") == "fresh"
            ],
        },
        "surfaces": surface_payloads,
        "topology": {
            "routing_rows": [dict(row) for row in routes],
            "bus_group_rows": [dict(row) for row in buses],
            "matrix_rows": [dict(row) for row in matrices],
            "edges": [dict(edge) for edge in edges],
        },
    }
