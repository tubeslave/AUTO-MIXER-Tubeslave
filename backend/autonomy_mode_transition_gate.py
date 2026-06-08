"""Read-only gate for the four target Automixer operating modes.

The gate combines runtime mode policy, static autonomy readiness, and current
proof evidence. It reports what is currently available, limited, or blocked,
but it never changes the operator mode and never creates console actions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from autonomy_evidence_overlay import build_autonomy_evidence_overlay
    from autonomy_promotion_readiness import build_autonomy_promotion_readiness_snapshot
    from operator_mode_policy import (
        MODE_ANALYZE,
        MODE_ASSIST,
        MODE_AUTO,
        MODE_SUPERVISED,
        VALID_OPERATOR_MODES,
        build_operator_mode_status,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.autonomy_evidence_overlay import build_autonomy_evidence_overlay
    from backend.autonomy_promotion_readiness import build_autonomy_promotion_readiness_snapshot
    from backend.operator_mode_policy import (
        MODE_ANALYZE,
        MODE_ASSIST,
        MODE_AUTO,
        MODE_SUPERVISED,
        VALID_OPERATOR_MODES,
        build_operator_mode_status,
    )


AUTONOMY_MODE_TRANSITION_GATE_METHOD_ID = "autonomy_mode_transition_gate"
MODE_GATE_STATUSES = {
    "available",
    "limited_available",
    "blocked",
}


@dataclass(frozen=True)
class AutonomyModeGateRow:
    mode: str
    project_mode_name: str
    status: str
    runtime_entry_allowed: bool
    proposal_creation_allowed: bool
    live_apply_allowed: bool
    autonomous_apply_allowed: bool
    mode_expansion_allowed: bool
    current_live_write_kinds: Tuple[str, ...]
    current_live_write_surfaces: Tuple[str, ...]
    promotion_target_ids: Tuple[str, ...]
    evidence_statuses: Tuple[str, ...]
    readiness_gap_bucket_counts: Dict[str, int]
    readiness_gap_target_ids_by_bucket: Dict[str, Tuple[str, ...]]
    blockers: Tuple[str, ...]
    next_required_actions: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in MODE_GATE_STATUSES:
            raise ValueError(f"Unknown autonomy mode gate status: {self.status}")
        if self.autonomous_apply_allowed:
            raise ValueError("Mode transition gate cannot allow Full Autonomous apply")
        if self.mode_expansion_allowed:
            raise ValueError("Mode transition gate cannot widen live authority")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in (
            "current_live_write_kinds",
            "current_live_write_surfaces",
            "promotion_target_ids",
            "evidence_statuses",
            "readiness_gap_target_ids_by_bucket",
            "blockers",
            "next_required_actions",
        ):
            if key == "readiness_gap_target_ids_by_bucket":
                payload[key] = {bucket: list(ids) for bucket, ids in payload[key].items()}
            else:
                payload[key] = list(payload[key])
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _list_of_maps(value: Any) -> Tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _mode_policy(mode: str, *, connection_mode: str, wing_boundary_active: bool) -> Mapping[str, Any]:
    return build_operator_mode_status(
        mode=mode,
        connection_mode=connection_mode,
        wing_boundary_active=wing_boundary_active,
        auto_mode_unlocked=False,
    )


def _promotion_readiness(proof_context: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = proof_context.get("autonomy_promotion_readiness")
    return nested if isinstance(nested, Mapping) else build_autonomy_promotion_readiness_snapshot()


def _evidence_overlay(proof_context: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = proof_context.get("autonomy_evidence_overlay")
    return nested if isinstance(nested, Mapping) else build_autonomy_evidence_overlay(proof_context)


def _overlay_rows_by_target(overlay: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(row.get("target_id")): row
        for row in _list_of_maps(overlay.get("rows"))
        if row.get("target_id")
    }


def _readiness_rows_by_id(readiness: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(row.get("id")): row
        for row in _list_of_maps(readiness.get("rows"))
        if row.get("id")
    }


def _unique(values: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _supervised_target_ids(readiness: Mapping[str, Any]) -> Tuple[str, ...]:
    summary = readiness.get("summary") if isinstance(readiness.get("summary"), Mapping) else {}
    readiness_ids = tuple(str(item) for item in summary.get("supervised_live_allowed_target_ids", ()) if item)
    return _unique(readiness_ids)


def _proof_overlay_blockers(overlay: Mapping[str, Any]) -> Tuple[str, ...]:
    summary = overlay.get("summary") if isinstance(overlay.get("summary"), Mapping) else {}
    blockers: list[str] = []
    if summary.get("remediation_required_target_ids"):
        blockers.append("proof_remediation_required_before_promotion")
    if summary.get("dry_run_only_target_ids"):
        blockers.append("physical_proof_required_before_promotion")
    if summary.get("no_current_evidence_target_ids"):
        blockers.append("current_proof_evidence_missing")
    return tuple(blockers)


def _promotion_gap_summary(readiness: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = readiness.get("summary") if isinstance(readiness.get("summary"), Mapping) else {}
    gap_summary = summary.get("promotion_gap_summary")
    return gap_summary if isinstance(gap_summary, Mapping) else {}


def _gap_bucket_counts(gap_summary: Mapping[str, Any]) -> Dict[str, int]:
    counts = gap_summary.get("gap_bucket_counts")
    if not isinstance(counts, Mapping):
        return {}
    return {
        str(bucket): max(0, int(count or 0))
        for bucket, count in counts.items()
        if int(count or 0) > 0
    }


def _gap_target_ids_by_bucket(gap_summary: Mapping[str, Any]) -> Dict[str, Tuple[str, ...]]:
    targets = gap_summary.get("blocked_target_ids_by_bucket")
    if not isinstance(targets, Mapping):
        return {}
    normalized: Dict[str, Tuple[str, ...]] = {}
    for bucket, values in targets.items():
        if not isinstance(values, list):
            continue
        ids = tuple(str(value) for value in values if value)
        if ids:
            normalized[str(bucket)] = ids
    return normalized


def _gap_bucket_blockers(gap_bucket_counts: Mapping[str, int]) -> Tuple[str, ...]:
    return tuple(
        f"readiness_gap_{bucket}"
        for bucket, count in gap_bucket_counts.items()
        if count > 0
    )


def _row_for_summary(rows: Tuple["AutonomyModeGateRow", ...], mode: str) -> "AutonomyModeGateRow":
    return next(row for row in rows if row.mode == mode)


def _target_ids_by_bucket_for_summary(values: Mapping[str, Tuple[str, ...]]) -> Dict[str, list[str]]:
    return {bucket: list(target_ids) for bucket, target_ids in values.items()}


def _row_for_mode(
    mode: str,
    *,
    readiness: Mapping[str, Any],
    overlay: Mapping[str, Any],
    connection_mode: str,
    wing_boundary_active: bool,
) -> AutonomyModeGateRow:
    policy = _mode_policy(mode, connection_mode=connection_mode, wing_boundary_active=wing_boundary_active)
    project_mode = policy.get("project_mode") if isinstance(policy.get("project_mode"), Mapping) else {}
    capabilities = policy.get("capabilities") if isinstance(policy.get("capabilities"), Mapping) else {}
    readiness_rows = _readiness_rows_by_id(readiness)
    gap_summary = _promotion_gap_summary(readiness)
    gap_bucket_counts = _gap_bucket_counts(gap_summary)
    gap_target_ids_by_bucket = _gap_target_ids_by_bucket(gap_summary)
    overlay_rows = _overlay_rows_by_target(overlay)
    blockers = []
    next_actions = []
    target_ids: Tuple[str, ...] = ()
    evidence_statuses: Tuple[str, ...] = ()

    if mode in {MODE_ANALYZE, MODE_ASSIST}:
        status = "available"
        runtime_entry_allowed = True
        mode_expansion_allowed = False
        blockers.append("mode_has_no_live_apply_authority")
    elif mode == MODE_SUPERVISED:
        target_ids = _supervised_target_ids(readiness)
        evidence_statuses = tuple(
            str(overlay_rows[target_id].get("status"))
            for target_id in target_ids
            if target_id in overlay_rows
        )
        runtime_entry_allowed = bool(capabilities.get("can_apply_to_console") and target_ids)
        status = "limited_available" if runtime_entry_allowed else "blocked"
        mode_expansion_allowed = False
        blockers.extend(_proof_overlay_blockers(overlay))
        blockers.append("supervised_mode_is_narrow_fader_gain_only")
        for target_id in target_ids:
            target_row = readiness_rows.get(target_id, {})
            next_actions.extend(str(item) for item in target_row.get("next_required_evidence", ()) if item)
            overlay_row = overlay_rows.get(target_id, {})
            next_actions.extend(str(item) for item in overlay_row.get("next_required_actions", ()) if item)
    else:
        status = "blocked"
        runtime_entry_allowed = False
        mode_expansion_allowed = False
        blockers.extend(str(item) for item in policy.get("blocked_reasons", ()) if item)
        blockers.extend(
            str(item)
            for item in (readiness.get("summary") if isinstance(readiness.get("summary"), Mapping) else {}).get(
                "global_blockers",
                (),
            )
            if item
        )
        blockers.extend(_proof_overlay_blockers(overlay))
        blockers.extend(_gap_bucket_blockers(gap_bucket_counts))
        next_actions.append("Complete all supervised proof gates before Full Autonomous mode can be reviewed.")

    if mode != MODE_AUTO:
        blockers.append("full_autonomous_mode_remains_blocked")
    if mode in {MODE_SUPERVISED, MODE_AUTO} and wing_boundary_active:
        blockers.append("wing_deployment_requires_manual_safety_review")

    current_live_write_kinds = tuple(str(item) for item in capabilities.get("allowed_live_write_kinds", ()) if item)
    current_live_write_surfaces = tuple(
        str(item) for item in (project_mode.get("effective_live_write_surfaces") or ()) if item
    )
    live_apply_allowed = bool(runtime_entry_allowed and mode == MODE_SUPERVISED and current_live_write_kinds)

    return AutonomyModeGateRow(
        mode=mode,
        project_mode_name=str(project_mode.get("name") or mode),
        status=status,
        runtime_entry_allowed=runtime_entry_allowed,
        proposal_creation_allowed=bool(capabilities.get("can_create_proposals")),
        live_apply_allowed=live_apply_allowed,
        autonomous_apply_allowed=False,
        mode_expansion_allowed=mode_expansion_allowed,
        current_live_write_kinds=current_live_write_kinds if live_apply_allowed else (),
        current_live_write_surfaces=current_live_write_surfaces if live_apply_allowed else (),
        promotion_target_ids=target_ids,
        evidence_statuses=_unique(evidence_statuses),
        readiness_gap_bucket_counts=gap_bucket_counts,
        readiness_gap_target_ids_by_bucket=gap_target_ids_by_bucket,
        blockers=_unique(tuple(blockers)),
        next_required_actions=_unique(tuple(next_actions)),
    )


def build_autonomy_mode_transition_gate(
    proof_context: Mapping[str, Any] | None = None,
    *,
    connection_mode: str = "wing",
    wing_boundary_active: bool = True,
) -> Dict[str, Any]:
    """Build a read-only gate for Analysis, Assisted, Supervised, and Auto."""

    context = proof_context or {}
    readiness = _promotion_readiness(context)
    overlay = _evidence_overlay(context)
    rows = tuple(
        _row_for_mode(
            mode,
            readiness=readiness,
            overlay=overlay,
            connection_mode=connection_mode,
            wing_boundary_active=wing_boundary_active,
        )
        for mode in VALID_OPERATOR_MODES
    )
    status_counts: Dict[str, int] = {status: 0 for status in sorted(MODE_GATE_STATUSES)}
    for row in rows:
        status_counts[row.status] += 1

    entry_allowed = [row.mode for row in rows if row.runtime_entry_allowed]
    live_apply_modes = [row.mode for row in rows if row.live_apply_allowed]
    blocked_modes = [row.mode for row in rows if row.status == "blocked"]
    auto_row = _row_for_summary(rows, MODE_AUTO)
    supervised_row = _row_for_summary(rows, MODE_SUPERVISED)

    return {
        "type": "autonomy_mode_transition_gate",
        "generated_at": _utc_now(),
        "status": "ok",
        "safety": {
            "read_only": True,
            "live_write": False,
            "live_widening_allowed": False,
            "automatic_mode_change_allowed": False,
            "automatic_safety_update_allowed": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "mode_count": len(rows),
            "status_counts": status_counts,
            "runtime_entry_allowed_modes": entry_allowed,
            "live_apply_allowed_modes": live_apply_modes,
            "blocked_modes": blocked_modes,
            "autonomous_apply_allowed_modes": [],
            "full_auto_readiness_gap_bucket_counts": dict(auto_row.readiness_gap_bucket_counts),
            "full_auto_readiness_gap_target_ids_by_bucket": _target_ids_by_bucket_for_summary(
                auto_row.readiness_gap_target_ids_by_bucket
            ),
            "supervised_current_live_write_kinds": list(supervised_row.current_live_write_kinds),
            "supervised_current_live_write_surfaces": list(
                supervised_row.current_live_write_surfaces
            ),
            "supervised_promotion_target_ids": list(supervised_row.promotion_target_ids),
            "global_blockers": [
                "mode_gate_does_not_change_runtime_mode",
                "supervised_mode_limited_to_existing_policy",
                "full_autonomous_mode_remains_blocked",
            ],
        },
        "rows": [row.to_dict() for row in rows],
    }
