"""Proposal-only bus, group, DCA, and matrix balance recommendations.

This module is transport-free. It turns summed stem/group evidence into
operator-facing balance proposals, but it never creates a live console write
request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, Optional, Tuple


BUS_GROUP_METHOD_ID = "bus_group_balance_proposal"
BUS_GROUP_VERSION = "bus_group_balance_proposal_v1"
SUPPORTED_BUS_KINDS = {"stem", "group", "dca", "matrix", "bus", "fx_return"}
BUS_TRUE_PEAK_LIMIT_DBFS = -3.0
TRUE_PEAK_RESERVE_DB = 1.0
MAX_UP_DELTA_DB = 1.0
MAX_DOWN_DELTA_DB = 2.5
MIN_DELTA_DB = 0.4
FADER_CEILING_DB = 0.0
HIGH_FEEDBACK_RISK = 0.55
MIN_ASSIGNMENT_CONFIDENCE = 0.55

AYAIC_TARGETS_LUFS = {
    "lead_main": -22.0,
    "strong_group": -25.0,
    "backing_vocal": -27.0,
    "support": -30.0,
    "fx_return": -35.0,
}

ROLE_TOKENS = (
    ("lead_vocal", ("lead_vocal", "lead vocal", "main_vocal", "main vocal", "vox lead")),
    ("backing_vocal", ("backing_vocal", "backing vocal", "bgv", "bv", "harmony")),
    ("drums", ("drum", "drums", "kick", "snare", "tom", "overhead", "oh")),
    ("bass", ("bass", "sub")),
    ("guitars", ("gtr", "guitar", "guitars")),
    ("keys", ("key", "keys", "piano", "synth", "organ")),
    ("playback", ("playback", "track", "stem", "loop")),
    ("fx_return", ("fx", "reverb", "delay", "plate", "room", "return")),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _clip01(value: Any) -> float:
    return _clip(_safe_float(value), 0.0, 1.0)


def normalize_bus_kind(bus_kind: Any) -> str:
    text = str(bus_kind or "bus").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"group", "grp", "subgroup"}:
        return "group"
    if text in {"dca", "vca"}:
        return "dca"
    if text in {"matrix", "mtx"}:
        return "matrix"
    if text in {"fx", "fx_return", "return"}:
        return "fx_return"
    if text in {"stem", "bus"}:
        return text
    return text


def classify_bus_family(source_role: str, bus_label: str = "", bus_kind: str = "") -> str:
    text = f"{source_role} {bus_label} {bus_kind}".strip().lower().replace("-", "_")
    spaced = text.replace("_", " ")
    for family, tokens in ROLE_TOKENS:
        if any(token in text or token in spaced for token in tokens):
            return family
    return "default"


def resolve_ayaic_bus_target_lufs(
    bus_kind: str,
    source_role: str,
    arrangement_role: str = "",
    explicit_target_lufs: Optional[float] = None,
) -> Tuple[float, str]:
    if explicit_target_lufs is not None:
        return round(_clip(_safe_float(explicit_target_lufs), -40.0, -16.0), 1), "explicit"

    kind = normalize_bus_kind(bus_kind)
    family = classify_bus_family(source_role, bus_kind=kind)
    arrangement = str(arrangement_role or "").strip().lower()
    if kind == "fx_return" or family == "fx_return" or "deep" in arrangement:
        return AYAIC_TARGETS_LUFS["fx_return"], "fx_return"
    if "lead" in arrangement or "main" in arrangement or "anchor" in arrangement or family == "lead_vocal":
        return AYAIC_TARGETS_LUFS["lead_main"], "lead_main"
    if family == "backing_vocal" or "backing" in arrangement or "tucked" in arrangement:
        return AYAIC_TARGETS_LUFS["backing_vocal"], "backing_vocal"
    if family in {"keys", "playback"} or "support" in arrangement or "ambient" in arrangement:
        return AYAIC_TARGETS_LUFS["support"], "support"
    return AYAIC_TARGETS_LUFS["strong_group"], "strong_group"


@dataclass(frozen=True)
class BusGroupBalanceAnalysisInput:
    bus_id: str
    bus_kind: str
    source_role: str
    bus_label: str = ""
    arrangement_role: str = ""
    current_fader_db: Optional[float] = 0.0
    integrated_lufs: float = -25.0
    short_term_lufs: float = -25.0
    true_peak_dbfs: float = -8.0
    bus_true_peak_headroom_db: Optional[float] = None
    summed_child_lufs: Optional[float] = None
    child_count: int = 1
    dominant_child_share: float = 0.0
    main_bus_headroom_db: float = 6.0
    feedback_risk: float = 0.0
    assignment_confidence: float = 0.75
    target_plane_lufs: Optional[float] = None
    confidence: float = 0.75
    persistence_cycles: int = 1


@dataclass(frozen=True)
class BusGroupBalanceRecommendation:
    id: str
    status: str
    action_type: str
    bus_id: str
    bus_kind: str
    bus_label: str
    source_role: str
    source_family: str
    target_plane_lufs: float
    target_plane_reason: str
    current_integrated_lufs: float
    short_term_lufs: float
    summed_child_lufs: Optional[float]
    current_fader_db: Optional[float]
    suggested_delta_db: float
    target_fader_db: Optional[float]
    true_peak_dbfs: float
    true_peak_headroom_db: float
    confidence: float
    severity: str
    reason: str
    blockers: Tuple[str, ...] = ()
    evidence: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["blockers"] = list(self.blockers)
        return payload

    def to_operator_proposal(self) -> Dict[str, Any]:
        """Return a Product Layer style proposal without live-apply fields."""

        return {
            "id": self.id,
            "method_id": BUS_GROUP_METHOD_ID,
            "source": BUS_GROUP_METHOD_ID,
            "title": f"{self.bus_label or self.bus_id}: bus/group balance proposal",
            "reason": self.reason,
            "target": self.bus_id,
            "kind": "bus_group_balance",
            "severity": self.severity,
            "confidence": self.confidence,
            "queue_action": "not_queueable_until_promoted",
            "can_import_to_queue": False,
            "can_apply_live": False,
            "blockers": [
                "unsupported_value_type:bus_group_balance",
                "not_in_supervised_fader_gain_executor",
                "hardware_readback_and_rollback_not_proven",
                *self.blockers,
            ],
            "proposed_action": {
                "action_type": self.action_type,
                "bus_id": self.bus_id,
                "bus_kind": self.bus_kind,
                "target_plane_lufs": self.target_plane_lufs,
                "current_integrated_lufs": self.current_integrated_lufs,
                "suggested_delta_db": self.suggested_delta_db,
                "target_fader_db": self.target_fader_db,
                "true_peak_headroom_db": self.true_peak_headroom_db,
            },
            "evidence": {
                **self.evidence,
                "live_write": False,
                "queue_mutation": False,
                "readback_required_before_promotion": True,
                "rollback_required_before_promotion": True,
            },
        }


def _derive_headroom(analysis: BusGroupBalanceAnalysisInput) -> float:
    if analysis.bus_true_peak_headroom_db is not None:
        return _safe_float(analysis.bus_true_peak_headroom_db)
    return BUS_TRUE_PEAK_LIMIT_DBFS - _safe_float(analysis.true_peak_dbfs, -120.0)


def _bounded_delta(desired_delta: float, current_fader: Optional[float], headroom: float) -> Tuple[float, Optional[float]]:
    if current_fader is None:
        return 0.0, None
    current = _safe_float(current_fader)
    if desired_delta > 0.0:
        available_by_peak = max(0.0, headroom - TRUE_PEAK_RESERVE_DB)
        available_by_fader = max(0.0, FADER_CEILING_DB - current)
        delta = min(desired_delta, MAX_UP_DELTA_DB, available_by_peak, available_by_fader)
    else:
        delta = max(desired_delta, -MAX_DOWN_DELTA_DB)

    delta = round(delta, 1)
    return delta, round(_clip(current + delta, -144.0, FADER_CEILING_DB), 1)


def _severity(desired_delta: float, headroom: float, blockers: Iterable[str]) -> str:
    blocker_set = set(blockers)
    if "true_peak_headroom_insufficient_for_bus_up" in blocker_set or abs(desired_delta) >= 5.0:
        return "high"
    if abs(desired_delta) >= 1.5 or headroom <= 1.5:
        return "medium"
    return "low"


def analyze_bus_group_balance_need(
    analysis: BusGroupBalanceAnalysisInput,
    *,
    min_confidence: float = 0.45,
) -> BusGroupBalanceRecommendation:
    bus_kind = normalize_bus_kind(analysis.bus_kind)
    label = analysis.bus_label or analysis.bus_id
    family = classify_bus_family(analysis.source_role, label, bus_kind)
    target_lufs, target_reason = resolve_ayaic_bus_target_lufs(
        bus_kind,
        analysis.source_role,
        analysis.arrangement_role,
        analysis.target_plane_lufs,
    )
    integrated_lufs = _safe_float(analysis.integrated_lufs, -100.0)
    short_term_lufs = _safe_float(analysis.short_term_lufs, integrated_lufs)
    true_peak = _safe_float(analysis.true_peak_dbfs, -120.0)
    current_fader = None if analysis.current_fader_db is None else round(_safe_float(analysis.current_fader_db), 1)
    headroom = round(_derive_headroom(analysis), 2)
    desired_delta = round(target_lufs - integrated_lufs, 2)
    suggested_delta, target_fader = _bounded_delta(desired_delta, current_fader, headroom)
    child_count = max(0, int(analysis.child_count or 0))
    feedback_risk = _clip01(analysis.feedback_risk)
    assignment_confidence = _clip01(analysis.assignment_confidence)
    main_headroom = _safe_float(analysis.main_bus_headroom_db)
    dominant_child_share = _clip01(analysis.dominant_child_share)
    confidence_input = _clip01(analysis.confidence)
    persistence = max(1, int(analysis.persistence_cycles or 1))

    blockers = []
    if bus_kind not in SUPPORTED_BUS_KINDS:
        blockers.append("unsupported_bus_kind")
    if current_fader is None:
        blockers.append("current_bus_fader_readback_missing")
    if integrated_lufs <= -70.0 or true_peak <= -90.0:
        blockers.append("bus_signal_below_analysis_floor")
    if bus_kind in {"stem", "group", "dca"} and child_count <= 0:
        blockers.append("bus_assignment_children_missing")
    if assignment_confidence < MIN_ASSIGNMENT_CONFIDENCE:
        blockers.append("bus_assignment_confidence_low")
    if desired_delta > 0.0 and headroom <= TRUE_PEAK_RESERVE_DB:
        blockers.append("true_peak_headroom_insufficient_for_bus_up")
    if desired_delta > 0.0 and main_headroom <= TRUE_PEAK_RESERVE_DB:
        blockers.append("main_bus_headroom_insufficient_for_bus_up")
    if desired_delta > 0.0 and feedback_risk >= HIGH_FEEDBACK_RISK:
        blockers.append("feedback_risk_high")
    if current_fader is not None and desired_delta > 0.0 and current_fader >= FADER_CEILING_DB:
        blockers.append("fader_ceiling_would_be_exceeded")
    if abs(suggested_delta) < MIN_DELTA_DB:
        blockers.append("bus_balance_change_not_needed")

    summed_child_lufs = None
    child_to_bus_delta = None
    if analysis.summed_child_lufs is not None:
        summed_child_lufs = round(_safe_float(analysis.summed_child_lufs), 2)
        child_to_bus_delta = round(summed_child_lufs - integrated_lufs, 2)

    confidence = _clip01(
        0.35 * confidence_input
        + 0.25 * assignment_confidence
        + 0.15 * min(persistence / 3.0, 1.0)
        + 0.10 * min(max(headroom, 0.0) / 6.0, 1.0)
        + 0.10 * min(abs(suggested_delta) / 1.5, 1.0)
        + 0.05 * (1.0 - feedback_risk)
    )
    blockers_tuple = tuple(sorted(set(blockers)))
    status = "proposal_only" if not blockers_tuple and confidence >= min_confidence else "analysis_only"
    if status == "proposal_only":
        reason = (
            f"{label} can move {suggested_delta:+.1f} dB toward the "
            f"{target_lufs:.1f} LUFS Ayaic {target_reason} plane."
        )
    else:
        reason = f"{label} does not meet safe bus/group balance proposal criteria."

    evidence = {
        "source": BUS_GROUP_VERSION,
        "bus_kind": bus_kind,
        "source_role": analysis.source_role,
        "source_family": family,
        "arrangement_role": analysis.arrangement_role,
        "target_plane_lufs": target_lufs,
        "target_plane_reason": target_reason,
        "integrated_lufs": round(integrated_lufs, 2),
        "short_term_lufs": round(short_term_lufs, 2),
        "desired_delta_db": desired_delta,
        "true_peak_dbfs": round(true_peak, 2),
        "true_peak_headroom_db": headroom,
        "main_bus_headroom_db": round(main_headroom, 2),
        "summed_child_lufs": summed_child_lufs,
        "child_to_bus_delta_db": child_to_bus_delta,
        "child_count": child_count,
        "dominant_child_share": round(dominant_child_share, 3),
        "feedback_risk": round(feedback_risk, 3),
        "assignment_confidence": round(assignment_confidence, 3),
        "current_fader_db": current_fader,
        "target_fader_db": target_fader,
        "persistence_cycles": persistence,
        "transport_free": True,
        "ayaic_level_plane": True,
    }

    return BusGroupBalanceRecommendation(
        id=f"{BUS_GROUP_METHOD_ID}:{analysis.bus_id}",
        status=status,
        action_type="bus_group_balance_proposal",
        bus_id=str(analysis.bus_id),
        bus_kind=bus_kind,
        bus_label=label,
        source_role=str(analysis.source_role or "unknown"),
        source_family=family,
        target_plane_lufs=target_lufs,
        target_plane_reason=target_reason,
        current_integrated_lufs=round(integrated_lufs, 2),
        short_term_lufs=round(short_term_lufs, 2),
        summed_child_lufs=summed_child_lufs,
        current_fader_db=current_fader,
        suggested_delta_db=suggested_delta,
        target_fader_db=target_fader,
        true_peak_dbfs=round(true_peak, 2),
        true_peak_headroom_db=headroom,
        confidence=round(confidence, 3),
        severity=_severity(desired_delta, headroom, blockers_tuple),
        reason=reason,
        blockers=blockers_tuple,
        evidence=evidence,
    )


def build_bus_group_balance_proposals(
    analyses: Iterable[BusGroupBalanceAnalysisInput],
) -> Tuple[Dict[str, Any], ...]:
    proposals = []
    for analysis in analyses:
        recommendation = analyze_bus_group_balance_need(analysis)
        if recommendation.status == "proposal_only":
            proposals.append(recommendation.to_operator_proposal())
    return tuple(proposals)
