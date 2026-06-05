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
    from bus_group_main_dca_physical_readback_contract import (
        BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        BUS_GROUP_MAIN_DCA_SURFACE_ID,
    )
    from channel_basic_controls_physical_readback_contract import (
        CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        CHANNEL_BASIC_CONTROLS_SURFACE_ID,
    )
    from channel_processing_physical_readback_contract import (
        CHANNEL_PROCESSING_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        CHANNEL_PROCESSING_SURFACE_ID,
    )
    from fx_slots_inserts_physical_readback_contract import (
        FX_SLOTS_INSERTS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        FX_SLOTS_INSERTS_SURFACE_ID,
    )
    from matrix_state_physical_readback_contract import (
        MATRIX_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        MATRIX_STATE_SURFACE_ID,
    )
    from read_surface_inventory import ReadSurface, iter_read_surfaces
    from read_surface_fixture_evidence import build_read_surface_fixture_evidence_snapshot
    from raw_mixer_state_physical_readback_contract import (
        RAW_MIXER_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        RAW_MIXER_STATE_SURFACE_ID,
    )
    from routing_sends_physical_readback_contract import (
        ROUTING_SENDS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        ROUTING_SENDS_SURFACE_ID,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.bus_group_main_dca_physical_readback_contract import (
        BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        BUS_GROUP_MAIN_DCA_SURFACE_ID,
    )
    from backend.channel_basic_controls_physical_readback_contract import (
        CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        CHANNEL_BASIC_CONTROLS_SURFACE_ID,
    )
    from backend.channel_processing_physical_readback_contract import (
        CHANNEL_PROCESSING_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        CHANNEL_PROCESSING_SURFACE_ID,
    )
    from backend.fx_slots_inserts_physical_readback_contract import (
        FX_SLOTS_INSERTS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        FX_SLOTS_INSERTS_SURFACE_ID,
    )
    from backend.matrix_state_physical_readback_contract import (
        MATRIX_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        MATRIX_STATE_SURFACE_ID,
    )
    from backend.read_surface_inventory import ReadSurface, iter_read_surfaces
    from backend.read_surface_fixture_evidence import build_read_surface_fixture_evidence_snapshot
    from backend.raw_mixer_state_physical_readback_contract import (
        RAW_MIXER_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        RAW_MIXER_STATE_SURFACE_ID,
    )
    from backend.routing_sends_physical_readback_contract import (
        ROUTING_SENDS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        ROUTING_SENDS_SURFACE_ID,
    )


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
PHYSICAL_READBACK_ACTION_TYPE = "add_physical_readback_fixture"
REMEDIATION_ACTION_TYPES = set(ACTION_TYPE_BY_STATUS.values()) | {PHYSICAL_READBACK_ACTION_TYPE}


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
    fixture_evidence_status: str = ""
    physical_readback_gap: bool = False
    next_physical_readback_target: bool = False
    fixture_remaining_gaps: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mutates_console_state:
            raise ValueError("Read remediation actions must not mutate console state")
        if self.action_type not in REMEDIATION_ACTION_TYPES:
            raise ValueError(f"Unknown read remediation action type: {self.action_type}")
        if self.status not in ACTION_TYPE_BY_STATUS and not self.physical_readback_gap:
            raise ValueError(f"{self.surface_id} is not an incomplete read surface or physical gap")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in (
            "current_fields",
            "missing_contracts",
            "source_paths",
            "entrypoints",
            "expected_artifacts",
            "blockers",
            "fixture_remaining_gaps",
        ):
            payload[key] = list(payload[key])
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _string_tuple(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _fixture_rows_by_surface(snapshot: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    rows = snapshot.get("surfaces")
    if not isinstance(rows, list):
        return {}
    return {
        row["surface_id"]: row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("surface_id"), str)
    }


def _physical_gap_surface_ids(
    fixture_snapshot: Mapping[str, Any],
    fixture_rows: Mapping[str, Mapping[str, Any]],
) -> Tuple[str, ...]:
    summary = fixture_snapshot.get("summary") if isinstance(fixture_snapshot.get("summary"), Mapping) else {}
    declared = _string_tuple(summary.get("physical_fixture_missing_surface_ids"))
    if declared:
        return declared
    return tuple(
        surface_id
        for surface_id, row in fixture_rows.items()
        if "physical_readback_fixture_missing" in _string_tuple(row.get("remaining_gaps"))
    )


def _missing_contracts(
    surface: ReadSurface,
    *,
    physical_readback_gap: bool,
    fixture_remaining_gaps: Tuple[str, ...],
) -> Tuple[str, ...]:
    contracts = []
    if surface.status in ACTION_TYPE_BY_STATUS:
        contracts.extend(
            [
                "product_layer_read_dto",
                "coverage_completeness_flags",
                "stale_cache_or_query_source_metadata",
                "fixture_or_snapshot_test",
            ]
        )
    if surface.read_mode == "osc_read_query":
        contracts.append("physical_query_readback_fixture")
    if surface.target_console_surface in {"routing_sends", "snapshots_scenes", "presets"}:
        contracts.append("destructive_change_diff_marker")
    if surface.status == "missing" or surface.target_console_surface == "presets":
        contracts.append("enumeration_contract")
    if surface.target_console_surface == "dlive":
        contracts.append("dlive_wing_parity_mapping")
    if physical_readback_gap:
        contracts.append("physical_readback_fixture")
        contracts.extend(fixture_remaining_gaps)
    return tuple(dict.fromkeys(contracts))


def _expected_artifacts(surface: ReadSurface, *, physical_readback_gap: bool) -> Tuple[str, ...]:
    artifacts = [
        f"Read DTO for {surface.target_console_surface} with explicit coverage flags.",
        f"Fixture test proving {surface.id} does not mutate console state.",
        "Read source metadata distinguishing cache, physical query, static config, and unavailable fields.",
        "Updated read surface inventory row with remaining blockers.",
    ]
    if physical_readback_gap:
        artifacts.append(
            f"Physical readback fixture for {surface.id} proving cached or queryable evidence matches console state."
        )
        if surface.id == RAW_MIXER_STATE_SURFACE_ID:
            artifacts.append(
                "Raw mixer-state physical readback artifact validated by "
                f"{RAW_MIXER_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID}."
            )
        if surface.id == CHANNEL_BASIC_CONTROLS_SURFACE_ID:
            artifacts.append(
                "Channel identity/basic-control physical readback artifact validated by "
                f"{CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_CONTRACT_METHOD_ID}."
            )
        if surface.id == CHANNEL_PROCESSING_SURFACE_ID:
            artifacts.append(
                "Channel HPF/EQ/dynamics/gate physical readback artifact validated by "
                f"{CHANNEL_PROCESSING_PHYSICAL_READBACK_CONTRACT_METHOD_ID}."
            )
        if surface.id == BUS_GROUP_MAIN_DCA_SURFACE_ID:
            artifacts.append(
                "Bus/group/main/DCA physical readback artifact validated by "
                f"{BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_CONTRACT_METHOD_ID}."
            )
        if surface.id == MATRIX_STATE_SURFACE_ID:
            artifacts.append(
                "Matrix-state physical readback artifact validated by "
                f"{MATRIX_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID}."
            )
        if surface.id == FX_SLOTS_INSERTS_SURFACE_ID:
            artifacts.append(
                "FX slots/inserts physical readback artifact validated by "
                f"{FX_SLOTS_INSERTS_PHYSICAL_READBACK_CONTRACT_METHOD_ID}."
            )
        if surface.id == ROUTING_SENDS_SURFACE_ID:
            artifacts.append(
                "Routing/sends physical readback artifact validated by "
                f"{ROUTING_SENDS_PHYSICAL_READBACK_CONTRACT_METHOD_ID}."
            )
    return tuple(artifacts)


def _action_type_for_surface(surface: ReadSurface, *, physical_readback_gap: bool) -> str:
    if surface.status in ACTION_TYPE_BY_STATUS:
        return ACTION_TYPE_BY_STATUS[surface.status]
    if physical_readback_gap:
        return PHYSICAL_READBACK_ACTION_TYPE
    raise ValueError(f"{surface.id} has no read remediation action")


def _action_for_surface(
    surface: ReadSurface,
    *,
    fixture_row: Mapping[str, Any],
    physical_readback_gap: bool,
    next_physical_readback_target: bool,
) -> ReadSurfaceRemediationAction:
    fixture_remaining_gaps = _string_tuple(fixture_row.get("remaining_gaps"))
    return ReadSurfaceRemediationAction(
        id=f"remediate_{surface.id}",
        surface_id=surface.id,
        target_console_surface=surface.target_console_surface,
        status=surface.status,
        action_type=_action_type_for_surface(surface, physical_readback_gap=physical_readback_gap),
        read_mode=surface.read_mode,
        requires_connection=bool(surface.requires_connection),
        mutates_console_state=False,
        current_fields=surface.read_fields,
        missing_contracts=_missing_contracts(
            surface,
            physical_readback_gap=physical_readback_gap,
            fixture_remaining_gaps=fixture_remaining_gaps,
        ),
        source_paths=surface.paths,
        entrypoints=surface.entrypoints,
        rationale=surface.blocker or surface.next_action or "Read coverage is incomplete.",
        expected_artifacts=_expected_artifacts(surface, physical_readback_gap=physical_readback_gap),
        blockers=tuple(item for item in (surface.blocker, surface.next_action) if item),
        fixture_evidence_status=str(fixture_row.get("status") or ""),
        physical_readback_gap=physical_readback_gap,
        next_physical_readback_target=next_physical_readback_target,
        fixture_remaining_gaps=fixture_remaining_gaps,
    )


def _status_counts(surfaces: Iterable[ReadSurface]) -> Dict[str, int]:
    counts: Dict[str, int] = {"partial": 0, "planned": 0, "missing": 0}
    for surface in surfaces:
        counts[surface.status] = counts.get(surface.status, 0) + 1
    return dict(sorted(counts.items()))


def build_read_surface_remediation_plan(
    surfaces: Iterable[ReadSurface] | None = None,
    fixture_evidence_snapshot: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a non-executing plan for incomplete console read coverage."""

    all_surfaces = tuple(surfaces if surfaces is not None else iter_read_surfaces())
    fixture_snapshot = (
        fixture_evidence_snapshot
        if fixture_evidence_snapshot is not None
        else build_read_surface_fixture_evidence_snapshot(all_surfaces)
    )
    fixture_rows = _fixture_rows_by_surface(fixture_snapshot)
    physical_gap_ids = _physical_gap_surface_ids(fixture_snapshot, fixture_rows)
    physical_gap_id_set = set(physical_gap_ids)
    summary = fixture_snapshot.get("summary") if isinstance(fixture_snapshot.get("summary"), Mapping) else {}
    source_next_target = summary.get("next_physical_readback_surface_id")
    next_physical_readback_surface_id = (
        source_next_target
        if isinstance(source_next_target, str) and source_next_target in physical_gap_id_set
        else physical_gap_ids[0]
        if physical_gap_ids
        else None
    )
    incomplete = tuple(
        surface
        for surface in all_surfaces
        if surface.status in ACTION_TYPE_BY_STATUS or surface.id in physical_gap_id_set
    )
    actions = tuple(
        _action_for_surface(
            surface,
            fixture_row=fixture_rows.get(surface.id, {}),
            physical_readback_gap=surface.id in physical_gap_id_set,
            next_physical_readback_target=surface.id == next_physical_readback_surface_id,
        )
        for surface in incomplete
    )
    next_action_id = (
        f"remediate_{next_physical_readback_surface_id}"
        if next_physical_readback_surface_id
        and any(action.surface_id == next_physical_readback_surface_id for action in actions)
        else None
    )
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
            "physical_fixture_missing_surface_ids": list(physical_gap_ids),
            "next_physical_readback_surface_id": next_physical_readback_surface_id,
            "next_physical_readback_action_id": next_action_id,
            "global_blockers": [
                "full_console_read_coverage_incomplete",
                "physical_readback_fixtures_incomplete",
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
