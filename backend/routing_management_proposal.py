"""Proposal-only routing management recommendations.

This module is transport-free. It turns intended route changes, readback state,
rollback evidence, activity checks, and operator intent into routing proposals,
but it never creates a live console write request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

try:
    from source_identity_downstream_gate import evaluate_source_identity_downstream_trust
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_downstream_gate import evaluate_source_identity_downstream_trust

try:
    from processing_proposal_adapters import build_operator_correction_payload
except ImportError:  # pragma: no cover - package import compatibility
    from backend.processing_proposal_adapters import build_operator_correction_payload


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


def _ordered_unique(values: Iterable[str]) -> Tuple[str, ...]:
    ordered = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _routing_required_safety_checks() -> Dict[str, str]:
    return {
        "read_only_route_map": "required_before_live_promotion",
        "route_diff": "required_before_live_promotion",
        "current_route_readback": "required_before_live_promotion",
        "proposed_source_readback": "required_before_live_promotion",
        "rollback_route": "required_before_live_promotion",
        "operator_intent": "required_before_live_promotion",
        "active_performance_guard": "required_before_live_promotion",
        "destructive_scope_check": "required_before_live_promotion",
        "current_route_activity": "required_before_live_promotion",
        "source_identity": "required_before_live_promotion",
    }


def _routing_rollback_expectation() -> Dict[str, Any]:
    return {
        "required_before_live_promotion": True,
        "operator_reviewed_route_snapshot_required": True,
        "restore_previous_route_assignment_after_physical_readback": True,
        "restore_previous_bus_matrix_send_after_physical_readback": True,
        "manual_operator_rollback_required_if_promoted": True,
    }


def _routing_future_wing_readback_contract() -> Dict[str, Any]:
    return {
        "required": True,
        "required_before_live_promotion": True,
        "target_kind": "route_diff",
        "old_value_source": "cached_or_physical_route_map_read_required_before_promotion",
        "readback_targets": [
            "routing_map",
            "bus_assignment",
            "matrix_send",
            "current_route_source",
            "proposed_route_source",
            "rollback_route",
        ],
        "blocked_live_actions": [
            "route_map_mutation_write",
            "bus_assignment_write",
            "matrix_send_write",
        ],
        "requires_real_wing_proof": True,
        "requires_operator_approval": True,
        "requires_separate_non_fader_adr": True,
    }


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
    current_state_cache_freshness: Optional[Dict[str, Any]] = None
    source_identity_trust_state: str = "not_provided"
    source_identity_trust_confidence: float = 0.0
    source_identity_replay_outcome_count: int = 0
    source_identity_calibration_freshness: Optional[Dict[str, Any]] = None


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

    def operator_safety_blockers(self) -> Tuple[str, ...]:
        return _ordered_unique(
            (
                "routing_live_apply_quarantined",
                "unsupported_value_type:routing",
                "not_in_supervised_fader_gain_executor",
                "real_wing_proof_required_before_live_apply",
                "non_fader_live_expansion_requires_separate_adr",
                "operator_approval_required_before_supervised_routing_apply",
                "hardware_readback_and_rollback_not_proven",
                "route_map_read_only_until_promoted",
                *self.blockers,
            )
        )

    def routing_safety_summary(self) -> Dict[str, Any]:
        source_identity = dict(
            self.evidence.get("source_identity_downstream_trust") or {}
        )
        return {
            "status": "passed" if not self.blockers else "blocked",
            "route_id": self.route_id,
            "route_kind": self.route_kind,
            "current_source": dict(self.current_source),
            "proposed_source": dict(self.proposed_source),
            "expected_source_role": self.expected_source_role,
            "route_diff": self.evidence.get("route_diff"),
            "current_route_readback": self.evidence.get("current_route_readback"),
            "proposed_source_readback": self.evidence.get("proposed_source_readback"),
            "rollback_route": self.evidence.get("rollback_route"),
            "route_readback_available": self.evidence.get(
                "route_readback_available"
            ),
            "rollback_route_available": self.evidence.get(
                "rollback_route_available"
            ),
            "snapshot_baseline_id": self.evidence.get("snapshot_baseline_id"),
            "operator_intent": self.evidence.get("operator_intent"),
            "route_change_window": self.evidence.get("route_change_window"),
            "active_performance": self.evidence.get("active_performance"),
            "current_route_signal_dbfs": self.evidence.get(
                "current_route_signal_dbfs"
            ),
            "proposed_source_verified": self.evidence.get(
                "proposed_source_verified"
            ),
            "destructive_scope_score": self.evidence.get(
                "destructive_scope_score"
            ),
            "destructive_scope_check": self.evidence.get(
                "destructive_scope_check"
            ),
            "active_performance_guard": self.evidence.get(
                "active_performance_guard"
            ),
            "current_route_activity": self.evidence.get(
                "current_route_activity"
            ),
            "affected_target_count": self.evidence.get("affected_target_count"),
            "current_route_cache_freshness_status": self.evidence.get(
                "current_state_cache_freshness_status"
            ),
            "current_route_cache_no_live_authority": self.evidence.get(
                "current_route_cache_no_live_authority"
            ),
            "source_identity_trust_status": source_identity.get("status"),
            "source_identity_usable_for_downstream": source_identity.get(
                "usable_for_downstream"
            ),
            "required_safety_checks": _routing_required_safety_checks(),
            "safety_blockers": list(self.operator_safety_blockers()),
            "operator_explanation": (
                "Routing evidence is read-only. Review the route diff, current "
                "route readback, proposed source readback, active-performance "
                "guard, destructive scope and rollback route snapshot before "
                "any future WING promotion."
            ),
            "rollback_expectation": _routing_rollback_expectation(),
            "future_wing_readback_contract": _routing_future_wing_readback_contract(),
            "readback_required_before_promotion": True,
            "rollback_required_before_promotion": True,
            "live_write": False,
            "queue_mutation": False,
            "can_apply_live": False,
            "can_import_to_queue": False,
            "queue_import_allowed": False,
            "live_apply_allowed": False,
            "read_only_or_dry_run_only": True,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "queue_action": "not_queueable_until_promoted",
            "mutates_mixer": False,
            "blockers": list(self.blockers),
        }

    def to_operator_proposal(self) -> Dict[str, Any]:
        """Return a Product Layer style proposal without live-apply fields."""

        routing_safety_summary = self.routing_safety_summary()
        blockers = list(self.operator_safety_blockers())
        proposed_action = {
            "action_type": self.action_type,
            "route_id": self.route_id,
            "route_kind": self.route_kind,
            "current_source": self.current_source,
            "proposed_source": self.proposed_source,
            "route_diff": routing_safety_summary["route_diff"],
            "current_route_readback": routing_safety_summary[
                "current_route_readback"
            ],
            "proposed_source_readback": routing_safety_summary[
                "proposed_source_readback"
            ],
            "rollback_route": routing_safety_summary["rollback_route"],
            "destructive_scope_check": routing_safety_summary[
                "destructive_scope_check"
            ],
            "current_route_activity": routing_safety_summary[
                "current_route_activity"
            ],
            "required_safety_checks": routing_safety_summary[
                "required_safety_checks"
            ],
            "rollback_expectation": routing_safety_summary[
                "rollback_expectation"
            ],
            "future_wing_readback_contract": routing_safety_summary[
                "future_wing_readback_contract"
            ],
            "routing_safety_summary": routing_safety_summary,
        }
        evidence = {
            **self.evidence,
            "routing_safety_summary": routing_safety_summary,
            "required_safety_checks": routing_safety_summary[
                "required_safety_checks"
            ],
            "operator_explanation": routing_safety_summary[
                "operator_explanation"
            ],
            "rollback_expectation": routing_safety_summary[
                "rollback_expectation"
            ],
            "future_wing_readback_contract": routing_safety_summary[
                "future_wing_readback_contract"
            ],
            "live_write": False,
            "queue_mutation": False,
            "can_apply_live": False,
            "can_import_to_queue": False,
            "queue_import_allowed": False,
            "live_apply_allowed": False,
            "read_only_or_dry_run_only": True,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "queue_action": "not_queueable_until_promoted",
            "mutates_mixer": False,
            "readback_required_before_promotion": True,
            "rollback_required_before_promotion": True,
        }
        proposal = {
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
            "live_write": False,
            "queue_mutation": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
            "queue_import_allowed": False,
            "live_apply_allowed": False,
            "read_only_or_dry_run_only": True,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "mutates_mixer": False,
            "blockers": blockers,
            "safety_blockers": routing_safety_summary["safety_blockers"],
            "operator_explanation": routing_safety_summary[
                "operator_explanation"
            ],
            "required_safety_checks": routing_safety_summary[
                "required_safety_checks"
            ],
            "rollback_expectation": routing_safety_summary[
                "rollback_expectation"
            ],
            "future_wing_readback_contract": routing_safety_summary[
                "future_wing_readback_contract"
            ],
            "proposed_action": proposed_action,
            "routing_safety_summary": routing_safety_summary,
            "evidence": evidence,
        }
        proposal["operator_correction"] = build_operator_correction_payload(
            proposal=proposal,
            method_id=ROUTING_METHOD_ID,
            target=self.route_id,
            proposed_action=proposed_action,
            blockers=blockers,
        )
        return proposal


def _severity(blockers: Iterable[str], destructive_scope_score: float, affected_count: int) -> str:
    blocker_set = set(blockers)
    if (
        "active_performance_routing_blocked" in blocker_set
        or "routing_scope_too_broad" in blocker_set
        or "source_identity_blocked_from_downstream_trust" in blocker_set
        or "source_identity_replay_context_not_fresh" in blocker_set
        or destructive_scope_score >= 0.7
        or affected_count > 4
    ):
        return "high"
    if (
        "source_identity_not_trusted_for_downstream" in blocker_set
        or "source_identity_trust_state_unknown" in blocker_set
    ):
        return "medium"
    if destructive_scope_score >= 0.25 or affected_count > 1:
        return "medium"
    return "low"


def _freshness_status(payload: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(payload, Mapping):
        return "not_provided"
    return str(payload.get("status") or "unknown").strip().lower() or "unknown"


def _freshness_payload(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"status": "not_provided", "provided": False}
    allowed_keys = (
        "status",
        "source",
        "field_count",
        "fresh_field_count",
        "stale_field_count",
        "unknown_field_count",
        "missing_field_count",
        "stale_fields",
        "missing_fields",
        "untimestamped_present_fields",
        "stale_after_sec",
        "read_only",
        "live_write",
        "queries_console",
    )
    normalized = {key: payload.get(key) for key in allowed_keys if key in payload}
    normalized["provided"] = True
    normalized.setdefault("status", _freshness_status(payload))
    return normalized


def _freshness_is_no_authority(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("read_only") is True
        and payload.get("live_write") is False
        and payload.get("queries_console") is False
    )


def _route_source_identity(
    analysis: RoutingManagementAnalysisInput,
    proposed_norm: Tuple[str, Optional[int]],
    route_kind: str,
) -> Tuple[str, str]:
    if proposed_norm[0] and proposed_norm[1] is not None:
        source_id = f"{proposed_norm[0]}:{proposed_norm[1]}"
    else:
        source_id = str(analysis.route_id)
    source_type = (
        str(analysis.expected_source_role or "").strip()
        or str(analysis.proposed_source_label or "").strip()
        or route_kind
    )
    return source_id, source_type


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
    current_state_freshness = _freshness_payload(analysis.current_state_cache_freshness)
    current_state_freshness_status = str(current_state_freshness.get("status") or "unknown")
    current_state_no_authority = _freshness_is_no_authority(current_state_freshness)
    source_identity_id, source_identity_type = _route_source_identity(
        analysis,
        proposed_norm,
        route_kind,
    )
    source_identity_trust = evaluate_source_identity_downstream_trust(
        source_id=source_identity_id,
        source_type=source_identity_type,
        trust_state=analysis.source_identity_trust_state,
        trust_confidence=analysis.source_identity_trust_confidence,
        replay_outcome_count=analysis.source_identity_replay_outcome_count,
        current_state_cache_freshness=analysis.source_identity_calibration_freshness,
    )

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
    if current_state_freshness_status != "fresh":
        blockers.append("routing_current_state_cache_not_fresh")
    if current_state_freshness_status == "fresh" and not current_state_no_authority:
        blockers.append("routing_current_state_cache_live_authority_present")
    blockers.extend(source_identity_trust.blockers)

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

    route_diff = {
        "status": "passed" if current_norm != proposed_norm else "blocked",
        "route_id": str(analysis.route_id),
        "route_kind": route_kind,
        "current_source": current.to_dict(),
        "proposed_source": proposed.to_dict(),
        "affected_target_count": affected_count,
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
    }
    current_route_readback = {
        "status": "available" if analysis.route_readback_available else "missing",
        "available": bool(analysis.route_readback_available),
        "current_source": current.to_dict(),
        "current_route_cache_freshness_status": current_state_freshness_status,
        "read_only": current_state_no_authority,
        "live_write": False,
    }
    proposed_source_readback = {
        "status": "verified" if analysis.proposed_source_verified else "missing",
        "verified": bool(analysis.proposed_source_verified),
        "proposed_source": proposed.to_dict(),
        "proposed_source_signal_dbfs": round(proposed_signal, 2),
        "source_confidence": round(source_confidence, 3),
    }
    rollback_route = {
        "status": (
            "available"
            if analysis.rollback_route_available and analysis.snapshot_baseline_id
            else "missing"
        ),
        "available": bool(
            analysis.rollback_route_available and analysis.snapshot_baseline_id
        ),
        "snapshot_baseline_id": analysis.snapshot_baseline_id,
        "source": "operator_reviewed_route_snapshot_required",
        "read_only": True,
        "live_write": False,
    }
    active_performance_guard = {
        "status": "passed"
        if not analysis.active_performance and window in SAFE_ROUTE_WINDOWS
        else "blocked",
        "active_performance": bool(analysis.active_performance),
        "route_change_window": window,
        "safe_route_windows": sorted(SAFE_ROUTE_WINDOWS),
    }
    destructive_scope_check = {
        "status": (
            "passed"
            if destructive_scope <= MAX_DESTRUCTIVE_SCOPE_SCORE
            and affected_count <= MAX_AFFECTED_TARGETS
            else "blocked"
        ),
        "destructive_scope_score": round(destructive_scope, 3),
        "max_destructive_scope_score": MAX_DESTRUCTIVE_SCOPE_SCORE,
        "affected_target_count": affected_count,
        "max_affected_target_count": MAX_AFFECTED_TARGETS,
    }
    current_route_activity = {
        "status": "inactive" if current_signal < ACTIVE_SIGNAL_DBFS else "active",
        "current_route_signal_dbfs": round(current_signal, 2),
        "active_signal_threshold_dbfs": ACTIVE_SIGNAL_DBFS,
        "manual_patch_required_when_active": current_signal >= ACTIVE_SIGNAL_DBFS,
    }

    evidence = {
        "source": ROUTING_VERSION,
        "route_kind": route_kind,
        "expected_source_role": analysis.expected_source_role,
        "route_diff": route_diff,
        "current_route_readback": current_route_readback,
        "proposed_source_readback": proposed_source_readback,
        "rollback_route": rollback_route,
        "active_performance_guard": active_performance_guard,
        "destructive_scope_check": destructive_scope_check,
        "current_route_activity": current_route_activity,
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
        "current_state_cache_freshness_status": current_state_freshness_status,
        "current_state_cache_freshness": current_state_freshness,
        "current_route_cache_no_live_authority": current_state_no_authority,
        "source_identity_downstream_trust": source_identity_trust.to_dict(),
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
