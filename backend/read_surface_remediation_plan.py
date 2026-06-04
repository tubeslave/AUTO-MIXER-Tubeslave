"""Read-only remediation plan for incomplete console read surfaces.

The plan converts the read-surface inventory into concrete follow-up actions
for full-console observability. It does not import mixer transports, does not
query a console, and does not create live write authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from read_surface_inventory import ReadSurface, iter_read_surfaces
except ImportError:  # pragma: no cover - package import compatibility
    from backend.read_surface_inventory import ReadSurface, iter_read_surfaces


READ_SURFACE_REMEDIATION_METHOD_ID = "read_surface_remediation_plan"
REMEDIATION_STATUSES = {
    "read_coverage_complete",
    "read_remediation_required",
}
ACTION_TYPE_BY_STATUS = {
    "partial": "complete_read_dto",
    "planned": "implement_read_surface",
    "missing": "define_read_surface_contract",
}


@dataclass(frozen=True)
class ReadSurfaceRemediationAction:
    id: str
    surface_id: str
    target_console_surface: str
    status: str
    action_type: str
    read_mode: str
    requires_connection: bool
    mutates_console_state: bool
    current_fields: Tuple[str, ...]
    missing_contracts: Tuple[str, ...]
    source_paths: Tuple[str, ...]
    entrypoints: Tuple[str, ...]
    rationale: str
    expected_artifacts: Tuple[str, ...]
    blockers: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.mutates_console_state:
            raise ValueError("Read remediation actions must not mutate console state")
        if self.status not in ACTION_TYPE_BY_STATUS:
            raise ValueError(f"{self.surface_id} is not an incomplete read surface")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in (
            "current_fields",
            "missing_contracts",
            "source_paths",
            "entrypoints",
            "expected_artifacts",
            "blockers",
        ):
            payload[key] = list(payload[key])
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _missing_contracts(surface: ReadSurface) -> Tuple[str, ...]:
    contracts = [
        "product_layer_read_dto",
        "coverage_completeness_flags",
        "stale_cache_or_query_source_metadata",
        "fixture_or_snapshot_test",
    ]
    if surface.read_mode == "osc_read_query":
        contracts.append("physical_query_readback_fixture")
    if surface.target_console_surface in {"routing_sends", "snapshots_scenes", "presets"}:
        contracts.append("destructive_change_diff_marker")
    if surface.status == "missing" or surface.target_console_surface == "presets":
        contracts.append("enumeration_contract")
    if surface.target_console_surface == "dlive":
        contracts.append("dlive_wing_parity_mapping")
    return tuple(contracts)


def _expected_artifacts(surface: ReadSurface) -> Tuple[str, ...]:
    return (
        f"Read DTO for {surface.target_console_surface} with explicit coverage flags.",
        f"Fixture test proving {surface.id} does not mutate console state.",
        "Read source metadata distinguishing cache, physical query, static config, and unavailable fields.",
        "Updated read surface inventory row with remaining blockers.",
    )


def _action_for_surface(surface: ReadSurface) -> ReadSurfaceRemediationAction:
    return ReadSurfaceRemediationAction(
        id=f"remediate_{surface.id}",
        surface_id=surface.id,
        target_console_surface=surface.target_console_surface,
        status=surface.status,
        action_type=ACTION_TYPE_BY_STATUS[surface.status],
        read_mode=surface.read_mode,
        requires_connection=bool(surface.requires_connection),
        mutates_console_state=False,
        current_fields=surface.read_fields,
        missing_contracts=_missing_contracts(surface),
        source_paths=surface.paths,
        entrypoints=surface.entrypoints,
        rationale=surface.blocker or surface.next_action or "Read coverage is incomplete.",
        expected_artifacts=_expected_artifacts(surface),
        blockers=tuple(item for item in (surface.blocker, surface.next_action) if item),
    )


def _status_counts(surfaces: Iterable[ReadSurface]) -> Dict[str, int]:
    counts: Dict[str, int] = {"partial": 0, "planned": 0, "missing": 0}
    for surface in surfaces:
        counts[surface.status] = counts.get(surface.status, 0) + 1
    return dict(sorted(counts.items()))


def build_read_surface_remediation_plan(
    surfaces: Iterable[ReadSurface] | None = None,
) -> Dict[str, Any]:
    """Build a non-executing plan for incomplete console read coverage."""

    all_surfaces = tuple(surfaces if surfaces is not None else iter_read_surfaces())
    incomplete = tuple(surface for surface in all_surfaces if surface.status in ACTION_TYPE_BY_STATUS)
    actions = tuple(_action_for_surface(surface) for surface in incomplete)
    status = "read_coverage_complete" if not actions else "read_remediation_required"

    return {
        "type": "read_surface_remediation_plan",
        "generated_at": _utc_now(),
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "surface_count": len(all_surfaces),
            "incomplete_surface_count": len(incomplete),
            "action_count": len(actions),
            "status_counts": _status_counts(all_surfaces),
            "incomplete_surface_ids": [surface.id for surface in incomplete],
            "global_blockers": [
                "full_console_read_coverage_incomplete",
                "read_surface_remediation_does_not_contact_mixer",
                "full_autonomous_requires_complete_read_dtos",
            ]
            if actions
            else [],
            "next_global_step": (
                "Complete read DTOs and fixture tests for partial/missing surfaces before broad autonomous console control."
                if actions
                else "Read surface inventory has no partial, planned, or missing rows."
            ),
        },
        "actions": [action.to_dict() for action in actions],
    }
