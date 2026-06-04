"""Proposal-only routing management recommendations.

This module is transport-free. It turns intended route changes, readback state,
rollback evidence, activity checks, and operator intent into routing proposals,
but it never creates a live console write request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, Optional, Tuple


ROUTING_METHOD_ID = "routing_management_proposal"
ROUTING_VERSION = "routing_management_proposal_v1"
SUPPORTED_ROUTE_KINDS = {
    "channel_input",
    "output_patch",
    "bus_source",
    "matrix_source",
    "insert_source",
    "send_source",
}
SAFE_ROUTE_WINDOWS = {"offline", "soundcheck", "rehearsal", "line_check"}
MIN_CONFIDENCE = 0.45
MIN_SOURCE_CONFIDENCE = 0.6
MIN_ROUTE_CONFIDENCE = 0.6
ACTIVE_SIGNAL_DBFS = -55.0
MAX_DESTRUCTIVE_SCOPE_SCORE = 0.45
MAX_AFFECTED_TARGETS = 2


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


def normalize_route_kind(route_kind: Any) -> str:
    text = str(route_kind or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "input": "channel_input",
        "channel": "channel_input",
        "channel_route": "channel_input",
        "output": "output_patch",
        "patch": "output_patch",
        "matrix": "matrix_source",
        "bus": "bus_source",
        "insert": "insert_source",
        "send": "send_source",
    }
    return aliases.get(text, text)


def _normalize_group(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_channel(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        channel = int(value)
    except (TypeError, ValueError):
        return None
    return channel if channel > 0 else None


@dataclass(frozen=True)
class RoutingEndpoint:
    group: str
    channel: Optional[int] = None
    label: str = ""

    def normalized(self) -> Tuple[str, Optional[int]]:
        return (_normalize_group(self.group), _normalize_channel(self.channel))

    def to_dict(self) -> Dict[str, Any]:
        group, channel = self.normalized()
        return {"group": group, "channel": channel, "label": self.label}


@dataclass(frozen=True)
class RoutingManagementAnalysisInput:
    route_id: str
    route_kind: str
    target_label: str = ""
    current_source_group: str = ""
    current_source_channel: Optional[int] = None
    current_source_label: str = ""
    proposed_source_group: str = ""
    proposed_source_channel: Optional[int] = None
    proposed_source_label: str = ""
    expected_source_role: str = ""
    route_readback_available: bool = False
    rollback_route_available: bool = False
    snapshot_baseline_id: str = ""
    operator_intent: str = ""
    route_change_window: str = "soundcheck"
    active_performance: bool = False
    current_route_signal_dbfs: float = -100.0
    proposed_source_signal_dbfs: float = -100.0
    proposed_source_verified: bool = True
    destructive_scope_score: float = 0.0
    affected_target_count: int = 1
    source_confidence: float = 0.75
    route_confidence: float = 0.75
    persistence_cycles: int = 1


@dataclass(frozen=True)
class RoutingManagementRecommendation:
    id: str
    status: str
    action_type: str
    route_id: str
    route_kind: str
    target_label: str
    current_source: Dict[str, Any]
    proposed_source: Dict[str, Any]
    expected_source_role: str
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
            "method_id": ROUTING_METHOD_ID,
            "source": ROUTING_METHOD_ID,
            "title": f"{self.target_label or self.route_id}: routing proposal",
            "reason": self.reason,
            "target": self.route_id,
            "kind": "routing",
            "severity": self.severity,
            "confidence": self.confidence,
            "queue_action": "not_queueable_until_promoted",
            "can_import_to_queue": False,
            "can_apply_live": False,
            "blockers": [
                "unsupported_value_type:routing",
                "not_in_supervised_fader_gain_executor",
                "hardware_readback_and_rollback_not_proven",
                *self.blockers,
            ],
            "proposed_action": {
                "action_type": self.action_type,
                "route_id": self.route_id,
                "route_kind": self.route_kind,
                "current_source": self.current_source,
                "proposed_source": self.proposed_source,
            },
            "evidence": {
                **self.evidence,
                "live_write": False,
                "queue_mutation": False,
                "readback_required_before_promotion": True,
                "rollback_required_before_promotion": True,
            },
        }


def _severity(blockers: Iterable[str], destructive_scope_score: float, affected_count: int) -> str:
    blocker_set = set(blockers)
    if (
        "active_performance_routing_blocked" in blocker_set
        or "routing_scope_too_broad" in blocker_set
        or destructive_scope_score >= 0.7
        or affected_count > 4
    ):
        return "high"
    if destructive_scope_score >= 0.25 or affected_count > 1:
        return "medium"
    return "low"


def analyze_routing_management_need(
    analysis: RoutingManagementAnalysisInput,
    *,
    min_confidence: float = MIN_CONFIDENCE,
) -> RoutingManagementRecommendation:
    route_kind = normalize_route_kind(analysis.route_kind)
    target_label = analysis.target_label or analysis.route_id
    current = RoutingEndpoint(
        group=analysis.current_source_group,
        channel=analysis.current_source_channel,
        label=analysis.current_source_label,
    )
    proposed = RoutingEndpoint(
        group=analysis.proposed_source_group,
        channel=analysis.proposed_source_channel,
        label=analysis.proposed_source_label,
    )
    current_norm = current.normalized()
    proposed_norm = proposed.normalized()
    current_signal = _safe_float(analysis.current_route_signal_dbfs, -100.0)
    proposed_signal = _safe_float(analysis.proposed_source_signal_dbfs, -100.0)
    destructive_scope = _clip01(analysis.destructive_scope_score)
    affected_count = max(1, int(analysis.affected_target_count or 1))
    source_confidence = _clip01(analysis.source_confidence)
    route_confidence = _clip01(analysis.route_confidence)
    persistence = max(1, int(analysis.persistence_cycles or 1))
    window = str(analysis.route_change_window or "").strip().lower().replace("-", "_")
    operator_intent = str(analysis.operator_intent or "").strip().lower()

    blockers = []
    if route_kind not in SUPPORTED_ROUTE_KINDS:
        blockers.append("unsupported_route_kind")
    if not current_norm[0] or current_norm[1] is None:
        blockers.append("current_route_endpoint_missing")
    if not proposed_norm[0] or proposed_norm[1] is None:
        blockers.append("proposed_route_endpoint_missing")
    if current_norm == proposed_norm:
        blockers.append("route_change_not_needed")
    if not analysis.route_readback_available:
        blockers.append("route_readback_missing")
    if not analysis.rollback_route_available or not analysis.snapshot_baseline_id:
        blockers.append("rollback_route_missing")
    if not operator_intent:
        blockers.append("operator_intent_required")
    if analysis.active_performance or window not in SAFE_ROUTE_WINDOWS:
        blockers.append("active_performance_routing_blocked")
    if destructive_scope > MAX_DESTRUCTIVE_SCOPE_SCORE or affected_count > MAX_AFFECTED_TARGETS:
        blockers.append("routing_scope_too_broad")
    if current_signal >= ACTIVE_SIGNAL_DBFS:
        blockers.append("current_route_active_requires_manual_patch")
    if not analysis.proposed_source_verified:
        blockers.append("proposed_source_readback_missing")
    if source_confidence < MIN_SOURCE_CONFIDENCE:
        blockers.append("source_confidence_low")
    if route_confidence < MIN_ROUTE_CONFIDENCE:
        blockers.append("route_confidence_low")

    confidence = _clip01(
        0.25 * source_confidence
        + 0.25 * route_confidence
        + 0.15 * min(persistence / 3.0, 1.0)
        + 0.10 * (1.0 if analysis.route_readback_available else 0.0)
        + 0.10 * (1.0 if analysis.rollback_route_available and analysis.snapshot_baseline_id else 0.0)
        + 0.10 * (1.0 - destructive_scope)
        + 0.05 * (1.0 if operator_intent else 0.0)
    )
    blockers_tuple = tuple(sorted(set(blockers)))
    status = "proposal_only" if not blockers_tuple and confidence >= min_confidence else "analysis_only"
    if status == "proposal_only":
        reason = (
            f"{target_label} can be proposed from "
            f"{current_norm[0]}/{current_norm[1]} to {proposed_norm[0]}/{proposed_norm[1]}."
        )
    else:
        reason = f"{target_label} does not meet safe routing proposal criteria."

    evidence = {
        "source": ROUTING_VERSION,
        "route_kind": route_kind,
        "expected_source_role": analysis.expected_source_role,
        "route_readback_available": bool(analysis.route_readback_available),
        "rollback_route_available": bool(analysis.rollback_route_available),
        "snapshot_baseline_id": analysis.snapshot_baseline_id,
        "operator_intent": operator_intent,
        "route_change_window": window,
        "active_performance": bool(analysis.active_performance),
        "current_route_signal_dbfs": round(current_signal, 2),
        "proposed_source_signal_dbfs": round(proposed_signal, 2),
        "proposed_source_verified": bool(analysis.proposed_source_verified),
        "destructive_scope_score": round(destructive_scope, 3),
        "affected_target_count": affected_count,
        "source_confidence": round(source_confidence, 3),
        "route_confidence": round(route_confidence, 3),
        "persistence_cycles": persistence,
        "transport_free": True,
        "destructive_console_state": True,
    }

    return RoutingManagementRecommendation(
        id=f"{ROUTING_METHOD_ID}:{analysis.route_id}",
        status=status,
        action_type="routing_proposal",
        route_id=str(analysis.route_id),
        route_kind=route_kind,
        target_label=target_label,
        current_source=current.to_dict(),
        proposed_source=proposed.to_dict(),
        expected_source_role=str(analysis.expected_source_role or "unknown"),
        confidence=round(confidence, 3),
        severity=_severity(blockers_tuple, destructive_scope, affected_count),
        reason=reason,
        blockers=blockers_tuple,
        evidence=evidence,
    )


def build_routing_management_proposals(
    analyses: Iterable[RoutingManagementAnalysisInput],
) -> Tuple[Dict[str, Any], ...]:
    proposals = []
    for analysis in analyses:
        recommendation = analyze_routing_management_need(analysis)
        if recommendation.status == "proposal_only":
            proposals.append(recommendation.to_operator_proposal())
    return tuple(proposals)
