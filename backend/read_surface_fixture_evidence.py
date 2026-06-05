"""Read-only fixture evidence map for console read surfaces.

This module does not inspect hardware and does not execute tests. It exposes the
current fixture coverage contract so Product Layer snapshots can distinguish
cached DTO evidence from the still-missing physical readback proof required for
full autonomous console awareness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from read_surface_inventory import ReadSurface, iter_read_surfaces
except ImportError:  # pragma: no cover - package import compatibility
    from backend.read_surface_inventory import ReadSurface, iter_read_surfaces


READ_SURFACE_FIXTURE_EVIDENCE_METHOD_ID = "read_surface_fixture_evidence"
FIXTURE_EVIDENCE_STATUSES = {
    "fixture_evidence_present",
    "partial_fixture_evidence",
    "missing_fixture_evidence",
}


@dataclass(frozen=True)
class ReadSurfaceFixtureEvidence:
    surface_id: str
    target_console_surface: str
    status: str
    fixture_paths: Tuple[str, ...]
    test_names: Tuple[str, ...]
    covered_contracts: Tuple[str, ...]
    remaining_gaps: Tuple[str, ...]
    full_autonomous_ready: bool = False

    def __post_init__(self) -> None:
        if self.status not in FIXTURE_EVIDENCE_STATUSES:
            raise ValueError(f"Unknown fixture evidence status: {self.status}")
        if self.full_autonomous_ready:
            raise ValueError("Fixture evidence alone must not claim full autonomous readiness")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in ("fixture_paths", "test_names", "covered_contracts", "remaining_gaps"):
            payload[key] = list(payload[key])
        return payload


FIXTURE_EVIDENCE_BY_SURFACE: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "raw_mixer_state_cache": {
        "fixture_paths": ("tests/test_read_surface_inventory.py",),
        "test_names": (
            "test_read_surface_inventory_is_read_only_and_unique",
            "test_partial_and_missing_surfaces_block_full_readiness_claim",
        ),
        "covered_contracts": ("inventory_row_validation", "read_only_safety_flags"),
    },
    "channel_identity_and_basic_controls": {
        "fixture_paths": ("tests/test_operator_product_state.py",),
        "test_names": (
            "test_channel_inventory_publishes_cached_source_patch_readback_only",
            "test_channel_inventory_publishes_cached_channel_detail_readback_only",
            "test_channel_inventory_marks_missing_cached_read_sections_without_writes",
        ),
        "covered_contracts": (
            "cached_channel_identity",
            "cached_source_patch",
            "basic_control_cache",
            "cache_freshness",
            "no_console_query",
        ),
    },
    "channel_eq_dynamics_gate": {
        "fixture_paths": ("tests/test_operator_product_state.py",),
        "test_names": (
            "test_channel_processing_snapshot_publishes_cached_processing_readback_only",
            "test_channel_processing_snapshot_marks_stale_cache_timestamps_without_queries",
            "test_channel_processing_snapshot_marks_missing_cache_without_queries",
        ),
        "covered_contracts": (
            "cached_hpf_eq_dynamics_gate",
            "field_cache_freshness",
            "stale_cache_detection",
            "no_console_query",
        ),
    },
    "bus_group_main_dca_state": {
        "fixture_paths": ("tests/test_operator_product_state.py",),
        "test_names": (
            "test_bus_main_dca_snapshot_publishes_cached_console_surfaces_without_queries",
            "test_bus_main_dca_snapshot_marks_missing_cache_without_queries",
        ),
        "covered_contracts": (
            "cached_bus_main_dca_state",
            "bus_main_dca_cache_freshness",
            "stale_cache_detection",
            "coverage_flags",
            "no_console_query",
        ),
    },
    "matrix_state": {
        "fixture_paths": ("tests/test_operator_product_state.py",),
        "test_names": (
            "test_matrix_snapshot_publishes_cached_console_state_without_queries",
            "test_matrix_snapshot_marks_missing_cache_without_queries",
        ),
        "covered_contracts": (
            "cached_matrix_state",
            "cached_matrix_dynamics",
            "cached_matrix_sends",
            "matrix_cache_freshness",
            "stale_cache_detection",
            "coverage_flags",
            "no_console_query",
        ),
    },
    "fx_slots_and_inserts": {
        "fixture_paths": ("tests/test_operator_product_state.py",),
        "test_names": (
            "test_fx_inserts_snapshot_publishes_cached_fx_and_insert_state_without_queries",
            "test_fx_inserts_snapshot_uses_generic_schema_for_unknown_fx_models",
            "test_fx_inserts_snapshot_marks_missing_cache_without_queries",
        ),
        "covered_contracts": (
            "cached_fx_insert_state",
            "fx_parameter_schema_hints",
            "fx_insert_cache_freshness",
            "stale_cache_detection",
            "coverage_flags",
            "no_console_query",
        ),
    },
    "routing_and_sends": {
        "fixture_paths": ("tests/test_operator_product_state.py",),
        "test_names": (
            "test_routing_sends_snapshot_publishes_static_and_cached_state_without_queries",
            "test_routing_sends_snapshot_marks_missing_cache_without_queries",
        ),
        "covered_contracts": (
            "static_routing_roles",
            "cached_send_state",
            "cached_matrix_send_state",
            "send_cache_freshness",
            "stale_cache_detection",
            "destructive_route_marker",
            "no_console_query",
        ),
    },
    "snapshots_scenes": {
        "fixture_paths": ("tests/test_operator_product_state.py",),
        "test_names": (
            "test_snapshots_scenes_snapshot_publishes_cached_catalog_without_queries",
            "test_snapshots_scenes_snapshot_marks_stale_cache_timestamps_without_queries",
            "test_snapshots_scenes_snapshot_marks_missing_cache_without_queries",
        ),
        "covered_contracts": (
            "cached_scene_catalog",
            "snapshot_scene_cache_freshness",
            "stale_cache_detection",
            "destructive_scene_marker",
            "no_scene_load",
            "no_console_query",
        ),
    },
    "user_presets": {
        "fixture_paths": ("tests/test_operator_product_state.py",),
        "test_names": (
            "test_user_presets_snapshot_publishes_cached_catalog_without_queries",
            "test_user_presets_snapshot_marks_stale_cache_timestamps_without_queries",
            "test_user_presets_snapshot_marks_missing_cache_without_queries",
        ),
        "covered_contracts": (
            "cached_preset_catalog",
            "user_preset_cache_freshness",
            "stale_cache_detection",
            "destructive_preset_marker",
            "no_preset_recall",
            "no_console_query",
        ),
    },
    "dlive_state": {
        "fixture_paths": ("tests/test_operator_product_state.py",),
        "test_names": (
            "test_dlive_state_snapshot_publishes_cached_midi_tcp_state_without_queries",
            "test_dlive_state_snapshot_marks_missing_cache_without_queries",
        ),
        "covered_contracts": (
            "cached_dlive_midi_state",
            "dlive_apply_blocked",
            "midi_send_blocked",
            "no_console_query",
        ),
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _remaining_gaps(
    surface: ReadSurface,
    evidence: Mapping[str, Tuple[str, ...]],
) -> Tuple[str, ...]:
    gaps = ["physical_readback_fixture_missing", "hardware_parity_unproven"]
    if surface.read_mode == "osc_read_query":
        gaps.append("osc_query_fixture_missing")
    if surface.target_console_surface in {"snapshots_scenes", "presets"}:
        gaps.extend(("scoped_diff_fixture_missing", "rollback_diff_fixture_missing"))
    if surface.target_console_surface == "routing_sends":
        gaps.append("route_send_diff_fixture_missing")
    if surface.target_console_surface == "dlive":
        gaps.append("dlive_wing_parity_mapping_missing")
    if "stale_cache_detection" not in evidence.get("covered_contracts", ()):
        gaps.append("per_field_stale_cache_fixture_incomplete")
    return tuple(dict.fromkeys(gaps))


def _evidence_for_surface(surface: ReadSurface) -> ReadSurfaceFixtureEvidence:
    evidence = FIXTURE_EVIDENCE_BY_SURFACE.get(surface.id, {})
    fixture_paths = tuple(evidence.get("fixture_paths", ()))
    test_names = tuple(evidence.get("test_names", ()))
    covered_contracts = tuple(evidence.get("covered_contracts", ()))
    if not fixture_paths or not test_names:
        status = "missing_fixture_evidence"
    elif "physical_readback_fixture" in covered_contracts:
        status = "fixture_evidence_present"
    else:
        status = "partial_fixture_evidence"
    return ReadSurfaceFixtureEvidence(
        surface_id=surface.id,
        target_console_surface=surface.target_console_surface,
        status=status,
        fixture_paths=fixture_paths,
        test_names=test_names,
        covered_contracts=covered_contracts,
        remaining_gaps=_remaining_gaps(surface, evidence),
        full_autonomous_ready=False,
    )


def _status_counts(rows: Iterable[ReadSurfaceFixtureEvidence]) -> Dict[str, int]:
    counts = {status: 0 for status in sorted(FIXTURE_EVIDENCE_STATUSES)}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    return counts


def build_read_surface_fixture_evidence_snapshot(
    surfaces: Iterable[ReadSurface] | None = None,
) -> Dict[str, Any]:
    """Return fixture evidence for read-surface DTOs without executing fixtures."""

    all_surfaces = tuple(surfaces if surfaces is not None else iter_read_surfaces())
    rows = tuple(_evidence_for_surface(surface) for surface in all_surfaces)
    status_counts = _status_counts(rows)
    rows_with_physical_gaps = tuple(
        row.surface_id
        for row in rows
        if "physical_readback_fixture_missing" in row.remaining_gaps
    )
    next_physical_readback_surface_id = (
        rows_with_physical_gaps[0] if rows_with_physical_gaps else None
    )
    return {
        "type": "read_surface_fixture_evidence",
        "generated_at": _utc_now(),
        "method_id": READ_SURFACE_FIXTURE_EVIDENCE_METHOD_ID,
        "status": "fixture_evidence_incomplete"
        if rows_with_physical_gaps
        else "fixture_evidence_complete",
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "surface_count": len(all_surfaces),
            "surface_with_fixture_count": sum(
                1 for row in rows if row.status != "missing_fixture_evidence"
            ),
            "status_counts": status_counts,
            "physical_fixture_missing_surface_ids": list(rows_with_physical_gaps),
            "next_physical_readback_surface_id": next_physical_readback_surface_id,
            "claim": "read DTO fixtures are tracked but physical readback proof is incomplete",
            "next_global_step": (
                "Add physical readback fixtures and parity mappings before claiming "
                "complete console read awareness."
            ),
        },
        "surfaces": [row.to_dict() for row in rows],
    }
