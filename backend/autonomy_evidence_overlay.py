"""Dynamic proof-evidence overlay for autonomy promotion targets.

The overlay connects current supervised proof reports to the static autonomy
readiness targets. It is read-only: it never changes safety rows, never creates
queue entries, and never executes remediation commands.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple
try:
    from operator_mode_policy import WING_DEPLOYMENT_LIVE_WRITE_KINDS
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_mode_policy import WING_DEPLOYMENT_LIVE_WRITE_KINDS

try:
    from autonomy_promotion_readiness import iter_promotion_readiness_rows
    from supervised_promotion_evidence import build_supervised_promotion_evidence_snapshot
    from supervised_proof_remediation_plan import build_supervised_proof_remediation_plan
except ImportError:  # pragma: no cover - package import compatibility
    from backend.autonomy_promotion_readiness import iter_promotion_readiness_rows
    from backend.supervised_promotion_evidence import build_supervised_promotion_evidence_snapshot
    from backend.supervised_proof_remediation_plan import build_supervised_proof_remediation_plan


AUTONOMY_EVIDENCE_OVERLAY_METHOD_ID = "autonomy_evidence_overlay"
OVERLAY_STATUSES = {
    "manual_safety_review_candidate",
    "remediation_required",
    "dry_run_only",
    "no_current_evidence",
}
TARGET_BY_VALUE_TYPE = {
    "fader": "fader_supervised_promotion",
    "gain": "gain_trim_supervised_promotion",
    "pan": "pan_phase_supervised_promotion",
}
PROMOTION_VALUE_TYPES = tuple(WING_DEPLOYMENT_LIVE_WRITE_KINDS)


@dataclass(frozen=True)
class AutonomyEvidenceOverlayRow:
    id: str
    target_id: str
    value_type: str
    status: str
    safety_review_candidate: bool
    supervised_live_allowed_static: bool
    autonomous_ready: bool
    proof_artifact_count: int
    physical_candidate_count: int
    dry_run_ready_count: int
    blocked_artifact_count: int
    remediation_action_count: int
    physical_candidate_proof_ids: Tuple[str, ...]
    dry_run_ready_proof_ids: Tuple[str, ...]
    blocked_proof_ids: Tuple[str, ...]
    remediation_action_ids: Tuple[str, ...]
    blockers: Tuple[str, ...]
    next_required_actions: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in OVERLAY_STATUSES:
            raise ValueError(f"Unknown autonomy evidence overlay status: {self.status}")
        if self.autonomous_ready:
            raise ValueError("Dynamic proof evidence cannot mark Full Autonomous ready")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in (
            "physical_candidate_proof_ids",
            "dry_run_ready_proof_ids",
            "blocked_proof_ids",
            "remediation_action_ids",
            "blockers",
            "next_required_actions",
        ):
            payload[key] = list(payload[key])
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _list_of_maps(value: Any) -> Tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _ids(rows: Tuple[Mapping[str, Any], ...]) -> Tuple[str, ...]:
    return tuple(str(row.get("proof_id")) for row in rows if row.get("proof_id"))


def _unique(items: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


def _promotion_evidence(proof_report: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = proof_report.get("promotion_evidence")
    return nested if isinstance(nested, Mapping) else build_supervised_promotion_evidence_snapshot(proof_report)


def _remediation_plan(proof_report: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = proof_report.get("proof_remediation_plan")
    return nested if isinstance(nested, Mapping) else build_supervised_proof_remediation_plan(proof_report)


def _promotion_row(evidence: Mapping[str, Any], value_type: str) -> Mapping[str, Any]:
    for row in _list_of_maps(evidence.get("rows")):
        if row.get("value_type") == value_type:
            return row
    return {}


def _static_target_rows() -> Dict[str, Any]:
    return {row.id: row for row in iter_promotion_readiness_rows()}


def _row_for_value_type(
    proof_report: Mapping[str, Any],
    *,
    value_type: str,
    evidence: Mapping[str, Any],
    remediation: Mapping[str, Any],
    static_rows: Mapping[str, Any],
) -> AutonomyEvidenceOverlayRow:
    report_rows = tuple(row for row in _list_of_maps(proof_report.get("rows")) if row.get("value_type") == value_type)
    physical = tuple(
        row
        for row in report_rows
        if row.get("promotion_allowed") is True
        and row.get("proof_mode") == "physical"
        and row.get("status") == "supervised_candidate"
    )
    dry_run = tuple(row for row in report_rows if row.get("status") == "dry_run_ready")
    blocked = tuple(row for row in report_rows if row.get("status") == "blocked")
    actions = tuple(
        action
        for action in _list_of_maps(remediation.get("actions"))
        if action.get("value_type") == value_type
    )
    target_id = TARGET_BY_VALUE_TYPE[value_type]
    target = static_rows.get(target_id)
    promotion = _promotion_row(evidence, value_type)
    safety_review_candidate = bool(promotion.get("safety_row_review_candidate"))

    if safety_review_candidate:
        status = "manual_safety_review_candidate"
    elif actions or blocked:
        status = "remediation_required"
    elif dry_run:
        status = "dry_run_only"
    else:
        status = "no_current_evidence"

    blockers = []
    blockers.extend(str(item) for item in promotion.get("blockers", ()) if item)
    blockers.extend(str(item) for item in getattr(target, "blockers", ()) if item)
    if status == "no_current_evidence":
        blockers.append("proof_artifact_missing")
    if status == "manual_safety_review_candidate":
        blockers.append("manual_safety_review_required")
        blockers.append("automatic_promotion_forbidden")
    if actions:
        blockers.append("proof_remediation_actions_present")

    next_actions = []
    next_actions.extend(str(item) for item in promotion.get("next_required_evidence", ()) if item)
    next_actions.extend(str(item) for item in getattr(target, "next_required_evidence", ()) if item)
    for action in actions:
        rationale = action.get("rationale")
        if rationale:
            next_actions.append(str(rationale))

    return AutonomyEvidenceOverlayRow(
        id=f"{value_type}_autonomy_evidence_overlay",
        target_id=target_id,
        value_type=value_type,
        status=status,
        safety_review_candidate=safety_review_candidate,
        supervised_live_allowed_static=bool(getattr(target, "supervised_live_allowed", False)),
        autonomous_ready=False,
        proof_artifact_count=len(report_rows),
        physical_candidate_count=len(physical),
        dry_run_ready_count=len(dry_run),
        blocked_artifact_count=len(blocked),
        remediation_action_count=len(actions),
        physical_candidate_proof_ids=_ids(physical),
        dry_run_ready_proof_ids=_ids(dry_run),
        blocked_proof_ids=_ids(blocked),
        remediation_action_ids=tuple(str(action.get("id")) for action in actions if action.get("id")),
        blockers=_unique(tuple(blockers)),
        next_required_actions=_unique(tuple(next_actions)),
    )


def build_autonomy_evidence_overlay(proof_report: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a proof-report overlay for autonomy promotion targets."""

    evidence = _promotion_evidence(proof_report)
    remediation = _remediation_plan(proof_report)
    static_rows = _static_target_rows()
    rows = tuple(
        _row_for_value_type(
            proof_report,
            value_type=value_type,
            evidence=evidence,
            remediation=remediation,
            static_rows=static_rows,
        )
        for value_type in PROMOTION_VALUE_TYPES
    )
    status_counts: Dict[str, int] = {status: 0 for status in sorted(OVERLAY_STATUSES)}
    for row in rows:
        status_counts[row.status] += 1

    review_ids = [row.target_id for row in rows if row.safety_review_candidate]
    remediation_ids = [row.target_id for row in rows if row.status == "remediation_required"]
    dry_run_ids = [row.target_id for row in rows if row.status == "dry_run_only"]
    no_evidence_ids = [row.target_id for row in rows if row.status == "no_current_evidence"]
    if remediation_ids:
        status = "manual_remediation_required"
    elif review_ids:
        status = "manual_review_ready"
    elif dry_run_ids:
        status = "dry_run_only"
    else:
        status = "no_current_evidence"

    return {
        "type": "autonomy_evidence_overlay",
        "generated_at": _utc_now(),
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "live_widening_allowed": False,
            "automatic_safety_update_allowed": False,
            "full_autonomous_ready": False,
            "operator_required": True,
        },
        "summary": {
            "target_count": len(rows),
            "status_counts": status_counts,
            "manual_safety_review_candidate_target_ids": review_ids,
            "remediation_required_target_ids": remediation_ids,
            "dry_run_only_target_ids": dry_run_ids,
            "no_current_evidence_target_ids": no_evidence_ids,
            "autonomous_ready_target_ids": [],
            "global_blockers": [
                "overlay_does_not_modify_safety_rows",
                "manual_safety_review_required_before_promotion",
                "full_autonomous_mode_remains_blocked",
            ],
        },
        "rows": [row.to_dict() for row in rows],
    }
