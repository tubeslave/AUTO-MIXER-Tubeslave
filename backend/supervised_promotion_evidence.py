"""Read-only bridge from proof artifact reports to safety-row review decisions.

This module does not update the safety matrix. It only decides whether current
proof artifacts are strong enough to be reviewed by an operator before a manual
SafetyCoverageRow change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple
try:
    from operator_mode_policy import CURRENT_PROVEN_LIVE_WRITE_KINDS
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_mode_policy import CURRENT_PROVEN_LIVE_WRITE_KINDS

try:
    from safety_coverage_matrix import get_safety_coverage_row
except ImportError:  # pragma: no cover - package import compatibility
    from backend.safety_coverage_matrix import get_safety_coverage_row


SUPERVISED_PROMOTION_EVIDENCE_METHOD_ID = "supervised_promotion_evidence_bridge"
EVIDENCE_STATUSES = {"safety_review_candidate", "dry_run_only", "evidence_blocked"}
PROMOTION_VALUE_TYPES = tuple(CURRENT_PROVEN_LIVE_WRITE_KINDS)
SAFETY_ROW_BY_VALUE_TYPE = {
    "fader": "fader_supervised",
    "gain": "gain_supervised",
    "pan": "pan_phase_delay",
}
DEFAULT_MIN_PHYSICAL_CANDIDATES = {value_type: 1 for value_type in PROMOTION_VALUE_TYPES}


@dataclass(frozen=True)
class SupervisedPromotionEvidenceRow:
    id: str
    value_type: str
    safety_row_id: str
    current_safety_status: str
    current_live_allowed: bool
    status: str
    safety_row_review_candidate: bool
    automatic_safety_update_allowed: bool
    autonomous_ready: bool
    artifact_count: int
    physical_candidate_count: int
    dry_run_ready_count: int
    blocked_artifact_count: int
    required_physical_candidate_count: int
    physical_candidate_proof_ids: Tuple[str, ...]
    dry_run_ready_proof_ids: Tuple[str, ...]
    blocked_proof_ids: Tuple[str, ...]
    blockers: Tuple[str, ...]
    next_required_evidence: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in EVIDENCE_STATUSES:
            raise ValueError(f"Unknown supervised promotion evidence status: {self.status}")
        if self.automatic_safety_update_allowed:
            raise ValueError("Proof evidence bridge must not auto-update safety rows")
        if self.autonomous_ready:
            raise ValueError("Proof evidence bridge must not mark autonomous mode ready")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in (
            "physical_candidate_proof_ids",
            "dry_run_ready_proof_ids",
            "blocked_proof_ids",
            "blockers",
            "next_required_evidence",
        ):
            payload[key] = list(payload[key])
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _summary(report: Mapping[str, Any]) -> Mapping[str, Any]:
    value = report.get("summary")
    return value if isinstance(value, Mapping) else {}


def _rows(report: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    value = report.get("rows")
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _ids(rows: Tuple[Mapping[str, Any], ...]) -> Tuple[str, ...]:
    return tuple(str(row.get("proof_id", "")) for row in rows if row.get("proof_id"))


def _next_evidence_for(value_type: str, *, has_physical_candidate: bool) -> Tuple[str, ...]:
    if has_physical_candidate:
        return (
            "Review physical proof artifacts manually before changing the safety coverage matrix.",
            "Keep Full Autonomous blocked until repeated representative proof and read coverage are complete.",
        )
    if value_type == "gain":
        return (
            "Run one small supervised gain-down proof with physical pre-read, apply readback, rollback readback, and emergency-stop block evidence.",
            "Keep gain-up blocked unless the operator authored the reason and headroom/bleed safety is proven.",
        )
    if value_type == "pan":
        return (
            "Run one controlled supervised pan-space proof with physical pre-read, apply readback, and rollback evidence across representative channels.",
            "Keep phase/delay related pan transforms proposal-only until proposal-level proof and stereo safety are established.",
        )
    return (
        "Run repeated controlled fader proofs across representative channels with connection-loss and emergency-stop reset evidence.",
        "Keep fader automation supervised-only until repeated proof closes partial hardware evidence.",
    )


def _build_row(
    report: Mapping[str, Any],
    *,
    value_type: str,
    min_physical_candidates: int,
) -> SupervisedPromotionEvidenceRow:
    safety_row_id = SAFETY_ROW_BY_VALUE_TYPE[value_type]
    safety = get_safety_coverage_row(safety_row_id)
    rows = tuple(row for row in _rows(report) if row.get("value_type") == value_type)
    physical = tuple(
        row
        for row in rows
        if row.get("promotion_allowed") is True
        and row.get("proof_mode") == "physical"
        and row.get("status") == "supervised_candidate"
    )
    dry_run = tuple(row for row in rows if row.get("status") == "dry_run_ready")
    blocked = tuple(row for row in rows if row.get("status") == "blocked")
    summary = _summary(report)
    invalid_count = int(summary.get("invalid_summary_count", 0) or 0)

    blockers = []
    if safety is None:
        blockers.append("safety_row_missing")
    if invalid_count:
        blockers.append("invalid_proof_summaries_present")
    if blocked:
        blockers.append("blocked_artifacts_for_value_type")
    if len(physical) < min_physical_candidates:
        blockers.append("physical_candidate_missing")

    evidence_blockers = tuple(sorted(set(blockers)))
    has_physical_candidate = len(physical) >= min_physical_candidates
    safety_row_review_candidate = bool(safety is not None and not evidence_blockers and has_physical_candidate)
    if safety_row_review_candidate:
        status = "safety_review_candidate"
    elif dry_run and not physical and not invalid_count and not blocked:
        status = "dry_run_only"
    else:
        status = "evidence_blocked"

    return SupervisedPromotionEvidenceRow(
        id=f"{value_type}_supervised_evidence",
        value_type=value_type,
        safety_row_id=safety_row_id,
        current_safety_status=safety.status if safety else "missing",
        current_live_allowed=bool(safety and safety.live_allowed),
        status=status,
        safety_row_review_candidate=safety_row_review_candidate,
        automatic_safety_update_allowed=False,
        autonomous_ready=False,
        artifact_count=len(rows),
        physical_candidate_count=len(physical),
        dry_run_ready_count=len(dry_run),
        blocked_artifact_count=len(blocked),
        required_physical_candidate_count=min_physical_candidates,
        physical_candidate_proof_ids=_ids(physical),
        dry_run_ready_proof_ids=_ids(dry_run),
        blocked_proof_ids=_ids(blocked),
        blockers=evidence_blockers,
        next_required_evidence=_next_evidence_for(value_type, has_physical_candidate=safety_row_review_candidate),
    )


def build_supervised_promotion_evidence_snapshot(
    proof_report: Mapping[str, Any],
    *,
    min_physical_candidates: Mapping[str, int] | None = None,
) -> Dict[str, Any]:
    """Summarize whether proof artifacts justify a manual safety-row review."""

    minimums = dict(DEFAULT_MIN_PHYSICAL_CANDIDATES)
    if min_physical_candidates:
        minimums.update({key: max(0, int(value)) for key, value in min_physical_candidates.items()})
    rows = tuple(
        _build_row(
            proof_report,
            value_type=value_type,
            min_physical_candidates=minimums.get(value_type, DEFAULT_MIN_PHYSICAL_CANDIDATES[value_type]),
        )
        for value_type in PROMOTION_VALUE_TYPES
    )
    status_counts: Dict[str, int] = {status: 0 for status in sorted(EVIDENCE_STATUSES)}
    for row in rows:
        status_counts[row.status] += 1

    review_ids = [row.id for row in rows if row.safety_row_review_candidate]
    blocked_ids = [row.id for row in rows if not row.safety_row_review_candidate]
    return {
        "type": "supervised_promotion_evidence",
        "generated_at": _utc_now(),
        "status": "ok",
        "safety": {
            "read_only": True,
            "live_write": False,
            "live_widening_allowed": False,
            "automatic_safety_update_allowed": False,
            "full_autonomous_ready": False,
            "supervised_only": True,
        },
        "summary": {
            "row_count": len(rows),
            "status_counts": status_counts,
            "safety_row_review_candidate_ids": review_ids,
            "blocked_evidence_row_ids": blocked_ids,
            "autonomous_ready_row_ids": [],
            "global_blockers": [
                "manual_safety_review_required",
                "full_autonomous_mode_remains_blocked",
                "proof_report_does_not_modify_safety_matrix",
            ],
        },
        "rows": [row.to_dict() for row in rows],
    }
