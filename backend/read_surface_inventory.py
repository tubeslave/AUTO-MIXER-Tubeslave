"""Declarative inventory for console read surfaces.

The inventory is read-only. It documents what the Automixer can inspect today,
what is only partially covered, and what still needs explicit readback support
before broad autonomous console control can be claimed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Tuple


READ_SURFACE_STATUSES = {
    "cached",
    "queryable",
    "partial",
    "planned",
    "missing",
}

READ_MODES = {
    "cached_state",
    "osc_read_query",
    "midi_tcp_state",
    "product_snapshot",
    "static_config",
    "not_available",
}


@dataclass(frozen=True)
class ReadSurface:
    id: str
    label: str
    target_console_surface: str
    status: str
    read_mode: str
    paths: Tuple[str, ...]
    entrypoints: Tuple[str, ...]
    read_fields: Tuple[str, ...]
    current_coverage: str
    blocker: str = ""
    next_action: str = ""
    requires_connection: bool = True
    mutates_console_state: bool = False
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in READ_SURFACE_STATUSES:
            raise ValueError(f"Unknown read surface status: {self.status}")
        if self.read_mode not in READ_MODES:
            raise ValueError(f"Unknown read mode: {self.read_mode}")
        if not self.paths:
            raise ValueError(f"{self.id} must declare at least one path")
        if not self.entrypoints:
            raise ValueError(f"{self.id} must declare at least one entrypoint")
        if self.mutates_console_state:
            raise ValueError(f"{self.id} is not read-only")
        if self.status in {"cached", "queryable", "partial"} and not self.read_fields:
            raise ValueError(f"{self.id} must declare read fields")
        if self.status == "missing" and self.read_mode != "not_available":
            raise ValueError(f"{self.id} missing rows must use not_available read mode")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["paths"] = list(self.paths)
        payload["entrypoints"] = list(self.entrypoints)
        payload["read_fields"] = list(self.read_fields)
        payload["notes"] = list(self.notes)
        return payload


READ_SURFACES: Tuple[ReadSurface, ...] = (
    ReadSurface(
        id="raw_mixer_state_cache",
        label="Raw mixer state cache",
        target_console_surface="all_cached_addresses",
        status="cached",
        read_mode="cached_state",
        paths=("backend/wing_client.py", "backend/dlive_client.py", "backend/handlers/mixer_handlers.py"),
        entrypoints=("get_state", "handle_get_state"),
        read_fields=("raw_address_values", "client_state"),
        current_coverage="Returns currently cached client state; completeness depends on prior subscriptions/queries.",
        next_action="Add per-surface completeness counters so cached state cannot be mistaken for full console proof.",
        notes=("read_only", "no_apply_authority"),
    ),
    ReadSurface(
        id="channel_identity_and_basic_controls",
        label="Channel identity and basic controls",
        target_console_surface="channels",
        status="partial",
        read_mode="product_snapshot",
        paths=("backend/operator_product_state.py", "backend/wing_client.py"),
        entrypoints=("build_channel_inventory", "get_channel_info", "get_channel_fader"),
        read_fields=("name", "source", "routing_role", "fader_db", "gain_db", "mute", "pan", "meter", "read_coverage"),
        current_coverage="Product snapshot publishes per-channel read coverage for identity, routing, metering, basic controls, and cached gate/dynamics.",
        blocker="Color/icon, source patch, HPF, EQ, and stale-cache timestamps are not fully normalized into one channel read contract.",
        next_action="Extend channel processing and source-patch DTOs with stale-cache timestamps and fixture snapshots.",
    ),
    ReadSurface(
        id="channel_eq_dynamics_gate",
        label="Channel EQ, dynamics, and gate state",
        target_console_surface="channels.processing",
        status="partial",
        read_mode="osc_read_query",
        paths=("backend/wing_client.py", "backend/handlers/eq_handlers.py", "backend/handlers/compressor_handlers.py"),
        entrypoints=(
            "get_eq_on",
            "get_eq_band_gain",
            "get_eq_band_frequency",
            "get_compressor_on",
            "get_compressor_threshold",
            "get_compressor_ratio",
            "get_compressor_gain",
        ),
        read_fields=(
            "hpf_enabled",
            "hpf_freq",
            "eq_on",
            "eq_band_gain",
            "eq_band_frequency",
            "eq_band_q",
            "compressor_on",
            "compressor_model",
            "threshold",
            "ratio",
            "makeup",
            "gain_reduction",
            "gate_on",
            "gate_threshold",
            "gate_timing",
            "read_coverage",
        ),
        current_coverage="Product snapshot exposes cached HPF, EQ, compressor, and gate DTOs with per-section coverage flags.",
        blocker="Physical query fixtures, stale-cache timestamps, de-esser/dynamic-EQ readback, and WING/dLive parity remain incomplete.",
        next_action="Add physical-query fixture snapshots and stale-cache timestamps before promoting EQ/dynamics/gate proposals.",
    ),
    ReadSurface(
        id="bus_group_main_dca_state",
        label="Bus, group, main, and DCA state",
        target_console_surface="buses_groups_main_dca",
        status="partial",
        read_mode="osc_read_query",
        paths=("backend/wing_client.py", "backend/operator_product_state.py"),
        entrypoints=(
            "get_bus_fader",
            "get_bus_settings",
            "get_main_fader",
            "get_dca_fader",
            "get_dca_settings",
            "build_bus_main_dca_snapshot",
            "build_groups_buses_snapshot",
        ),
        read_fields=(
            "bus_fader",
            "bus_mute",
            "bus_pan_width",
            "bus_eq",
            "bus_dynamics",
            "bus_main_send",
            "main_fader",
            "main_eq",
            "dca_fader",
            "dca_mute",
            "dca_identity",
            "dca_assignments",
            "read_coverage",
            "group_summary",
        ),
        current_coverage="Product snapshot exposes cached bus, main, and DCA DTOs with per-section coverage flags; group summary remains synthetic from channel inventory.",
        blocker="Physical query fixtures, stale-cache timestamps, full group membership, matrix sends, and WING/dLive parity remain incomplete.",
        next_action="Add physical-query fixture snapshots and stale-cache timestamps before bus/DCA/group balance proposals become live candidates.",
    ),
    ReadSurface(
        id="matrix_state",
        label="Matrix state",
        target_console_surface="matrices",
        status="partial",
        read_mode="cached_state",
        paths=("backend/operator_product_state.py", "backend/wing_client.py", "Docs/TOOLS.md"),
        entrypoints=("build_matrix_snapshot", "get_matrix_inserts", "get_state"),
        read_fields=("matrix_fader", "matrix_mute", "matrix_pan", "matrix_eq", "matrix_insert_state", "read_coverage"),
        current_coverage="Product snapshot exposes cached matrix fader/mute/pan, EQ, and pre/post insert DTOs with per-section coverage flags.",
        blocker="Physical query fixtures, stale-cache timestamps, matrix sends, dynamics, and WING/dLive parity remain incomplete.",
        next_action="Add physical-query matrix fixtures, send-state DTOs, and stale-cache timestamps before master/matrix optimization claims.",
    ),
    ReadSurface(
        id="fx_slots_and_inserts",
        label="FX slots and inserts",
        target_console_surface="fx_inserts",
        status="queryable",
        read_mode="osc_read_query",
        paths=("backend/wing_client.py", "backend/handlers/fx_handlers.py"),
        entrypoints=("query_fx_slot", "get_fx_slot_info", "query_insert", "get_insert", "get_fx_overview", "get_insert_state"),
        read_fields=("fx_model", "fx_on", "fx_mix", "fx_source", "insert_on", "insert_slot", "insert_status", "fx_parameters"),
        current_coverage="FX overview and insert state can be read from cached/query helpers.",
        blocker="FX parameter schemas and safe semantic names are not normalized per FX model.",
        next_action="Add FX-model parameter schema mapping before FX proposal confidence is used for live work.",
    ),
    ReadSurface(
        id="routing_and_sends",
        label="Routing and send state",
        target_console_surface="routing_sends",
        status="partial",
        read_mode="static_config",
        paths=("backend/dante_routing_config.py", "backend/operator_product_state.py", "backend/handlers/routing_handlers.py", "backend/wing_client.py"),
        entrypoints=("build_routing_sends_snapshot", "get_routing_as_dict", "get_state"),
        read_fields=(
            "static_dante_role",
            "tap_point",
            "channel_bus_send_on",
            "channel_bus_send_level",
            "channel_bus_send_mode",
            "channel_main_send",
            "output_route_source_group",
            "output_route_source_channel",
            "destructive_change_marker",
            "read_coverage",
        ),
        current_coverage="Product snapshot exposes static Dante roles plus cached channel bus sends, channel main sends, and output route source fields with coverage flags.",
        blocker="Physical query fixtures, stale-cache timestamps, complete route/send diffing, and destructive-change review gates are incomplete.",
        next_action="Add physical-query fixtures and route/send diff schema before any routing or send proposal can become queueable.",
    ),
    ReadSurface(
        id="snapshots_scenes",
        label="Snapshots and scenes",
        target_console_surface="snapshots_scenes",
        status="partial",
        read_mode="product_snapshot",
        paths=("backend/operator_product_state.py", "backend/wing_client.py", "backend/handlers/snapshot_handlers.py"),
        entrypoints=("build_snapshots_scenes_snapshot", "get_snapshots_scenes_snapshot", "find_snap_by_name"),
        read_fields=(
            "snapshot_index",
            "snapshot_name",
            "active_snapshot_name",
            "current_show",
            "scene_load_policy",
            "scene_save_policy",
            "destructive_change_marker",
            "read_coverage",
        ),
        current_coverage="Product snapshot exposes cached snapshot/scene catalog names and active-scene markers without loading snapshots.",
        blocker="Scene contents, restore diff, stale-cache timestamps, and rollback proof are not read-only audited before load/save operations.",
        next_action="Keep scene apply blocked; add read-only scene diff artifacts and rollback evidence before any automation.",
        notes=("find_snap_by_name_must_not_load_snapshot",),
    ),
    ReadSurface(
        id="user_presets",
        label="User presets",
        target_console_surface="presets",
        status="partial",
        read_mode="product_snapshot",
        paths=("backend/operator_product_state.py", "backend/wing_client.py"),
        entrypoints=("build_user_presets_snapshot", "get_user_presets_snapshot"),
        read_fields=(
            "preset_index",
            "preset_name",
            "preset_scope",
            "target_type",
            "target_label",
            "preset_recall_policy",
            "preset_store_policy",
            "destructive_recall_marker",
            "read_coverage",
        ),
        current_coverage="Product snapshot exposes cached user preset catalog names, scope, and target metadata without recalling or storing presets.",
        blocker="Physical preset enumeration, preset parameter contents, scoped diff, stale-cache timestamps, and rollback proof are incomplete.",
        next_action="Add physical-query fixtures and preset diff artifacts before any preset recall/store proposal can become queueable.",
        requires_connection=False,
    ),
    ReadSurface(
        id="dlive_state",
        label="Allen & Heath dLive state",
        target_console_surface="dlive",
        status="partial",
        read_mode="midi_tcp_state",
        paths=("backend/dlive_client.py",),
        entrypoints=("get_state", "get_fader", "get_channel_info"),
        read_fields=("raw_state", "fader_db", "channel_info"),
        current_coverage="dLive client exposes basic state and fader reads; advanced surfaces are not audited.",
        blocker="dLive EQ/dynamics/routing read coverage is not equivalent to the WING inventory.",
        next_action="Build a dLive-specific read matrix before claiming parity.",
    ),
)


def iter_read_surfaces() -> Iterable[ReadSurface]:
    return READ_SURFACES


def get_read_surface(surface_id: str) -> ReadSurface | None:
    return next((surface for surface in READ_SURFACES if surface.id == surface_id), None)


def build_read_surface_inventory_snapshot() -> Dict[str, Any]:
    status_counts: Dict[str, int] = {status: 0 for status in sorted(READ_SURFACE_STATUSES)}
    partial_or_missing_ids = []
    for surface in READ_SURFACES:
        status_counts[surface.status] += 1
        if surface.status in {"partial", "planned", "missing"}:
            partial_or_missing_ids.append(surface.id)

    return {
        "type": "read_surface_inventory",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ok",
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
        },
        "summary": {
            "surface_count": len(READ_SURFACES),
            "status_counts": status_counts,
            "partial_or_missing_surface_ids": partial_or_missing_ids,
            "claim": "read coverage is inventoried but not complete",
            "next_global_step": "Normalize channel processing, matrix, routing/send, scene diff, preset, and dLive read DTOs.",
        },
        "surfaces": [surface.to_dict() for surface in READ_SURFACES],
    }
