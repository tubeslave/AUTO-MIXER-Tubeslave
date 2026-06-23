"""Proposal-only decision evidence packets.

This module is transport-free. It turns audio, psychoacoustic, statistical,
history, reference, inter-channel, arrangement, genre, and previous-decision
signals into an operator-facing evidence packet, but it never creates a live
console write request.
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


DECISION_EVIDENCE_METHOD_ID = "decision_evidence_packet_proposal"
DECISION_EVIDENCE_VERSION = "decision_evidence_packet_proposal_v1"
SUPPORTED_DECISION_KINDS = {
    "analysis",
    "operator_review",
    "proposal",
    "shadow_hold",
    "shadow_rejection",
    "learning",
}
SUPPORTED_ACTION_FAMILIES = {
    "analysis_only",
    "gain_fader",
    "eq",
    "dynamics",
    "gate",
    "pan",
    "phase",
    "fx",
    "master_bus",
    "bus_group",
    "mute",
    "routing",
    "scene_preset",
    "mix_variant",
    "arrangement_role",
    "performance_adaptation",
}
EVIDENCE_FAMILIES = (
    "audio_analysis",
    "psychoacoustic_model",
    "statistical_model",
    "project_history",
    "reference_comparison",
    "previous_decision_outcomes",
    "inter_channel_context",
    "arrangement_context",
    "genre_context",
)
CRITICAL_EVIDENCE_FAMILIES = (
    "audio_analysis",
    "inter_channel_context",
    "arrangement_context",
)
MIN_CONFIDENCE = 0.5
MIN_FAMILY_CONFIDENCE = 0.55
MIN_CRITICAL_CONFIDENCE = 0.6
MIN_REQUIRED_FAMILY_COUNT = 7
MIN_PERSISTENCE_CYCLES = 2
MAX_CONFLICTING_EVIDENCE_COUNT = 1
MIN_TRUE_PEAK_HEADROOM_DB = 1.0
HIGH_FEEDBACK_RISK = 0.62
CONSOLE_AFFECTING_FAMILIES = SUPPORTED_ACTION_FAMILIES - {"analysis_only"}


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


def _normalize_token(value: Any, default: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text or default


def normalize_decision_kind(kind: Any) -> str:
    text = _normalize_token(kind, "proposal")
    aliases = {
        "recommendation": "proposal",
        "operator": "operator_review",
        "review": "operator_review",
        "hold": "shadow_hold",
        "reject": "shadow_rejection",
        "rejection": "shadow_rejection",
        "memory": "learning",
    }
    return aliases.get(text, text)


def normalize_action_family(action_family: Any) -> str:
    text = _normalize_token(action_family, "analysis_only")
    aliases = {
        "fader": "gain_fader",
        "gain": "gain_fader",
        "level": "gain_fader",
        "compressor": "dynamics",
        "compression": "dynamics",
        "deesser": "dynamics",
        "effect": "fx",
        "effects": "fx",
        "group": "bus_group",
        "dca": "bus_group",
        "matrix": "bus_group",
        "scene": "scene_preset",
        "preset": "scene_preset",
        "variant": "mix_variant",
        "arrangement": "arrangement_role",
        "role": "arrangement_role",
        "adaptation": "performance_adaptation",
        "analysis": "analysis_only",
    }
    return aliases.get(text, text)


@dataclass(frozen=True)
class DecisionEvidenceAnalysisInput:
    decision_id: str
    target_id: str
    decision_kind: str = "proposal"
    proposed_action_family: str = "analysis_only"
    operator_intent: str = ""
    operator_review_enabled: bool = True
    audio_analysis_available: bool = False
    audio_analysis_confidence: float = 0.0
    psychoacoustic_model_available: bool = False
    psychoacoustic_model_confidence: float = 0.0
    statistical_model_available: bool = False
    statistical_model_confidence: float = 0.0
    project_history_available: bool = False
    project_history_confidence: float = 0.0
    reference_comparison_available: bool = False
    reference_comparison_confidence: float = 0.0
    previous_decision_outcomes_available: bool = False
    previous_decision_outcomes_confidence: float = 0.0
    inter_channel_context_available: bool = False
    inter_channel_context_confidence: float = 0.0
    arrangement_context_available: bool = False
    arrangement_context_confidence: float = 0.0
    genre_context_available: bool = False
    genre_context_confidence: float = 0.0
    current_state_readback_available: bool = False
    rollback_plan_available: bool = False
    true_peak_headroom_db: float = 6.0
    feedback_risk: float = 0.0
    latency_budget_ms: float = 250.0
    latency_budget_met: bool = True
    decision_persistence_cycles: int = 1
    conflicting_evidence_count: int = 0
    manual_override_active: bool = False
    auto_apply_requested: bool = False
    active_performance: bool = False
    current_state_cache_freshness: Optional[Dict[str, Any]] = None
    source_identity_trust_state: str = "not_provided"
    source_identity_trust_confidence: float = 0.0
    source_identity_replay_outcome_count: int = 0
    source_identity_calibration_freshness: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class DecisionEvidenceRecommendation:
    id: str
    status: str
    action_type: str
    decision_id: str
    target_id: str
    decision_kind: str
    proposed_action_family: str
    evidence_family_count: int
    missing_evidence_families: Tuple[str, ...]
    low_confidence_families: Tuple[str, ...]
    confidence: float
    severity: str
    reason: str
    blockers: Tuple[str, ...] = ()
    evidence: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["blockers"] = list(self.blockers)
        payload["missing_evidence_families"] = list(self.missing_evidence_families)
        payload["low_confidence_families"] = list(self.low_confidence_families)
        return payload

    def to_operator_proposal(self) -> Dict[str, Any]:
        """Return a Product Layer style evidence packet without live-apply fields."""

        proposed_action = {
            "action_type": self.action_type,
            "decision_id": self.decision_id,
            "target_id": self.target_id,
            "decision_kind": self.decision_kind,
            "proposed_action_family": self.proposed_action_family,
            "evidence_family_count": self.evidence_family_count,
            "missing_evidence_families": list(self.missing_evidence_families),
            "low_confidence_families": list(self.low_confidence_families),
        }
        proposal = {
            "id": self.id,
            "method_id": DECISION_EVIDENCE_METHOD_ID,
            "source": DECISION_EVIDENCE_METHOD_ID,
            "title": f"{self.target_id}: decision evidence packet",
            "reason": self.reason,
            "target": self.target_id,
            "kind": "decision_evidence_packet",
            "severity": self.severity,
            "confidence": self.confidence,
            "queue_action": "not_queueable_until_promoted",
            "can_import_to_queue": False,
            "can_apply_live": False,
            "blockers": [
                "unsupported_value_type:decision_evidence_packet",
                "not_in_supervised_fader_gain_executor",
                "evidence_packet_requires_operator_review",
                *self.blockers,
            ],
            "proposed_action": proposed_action,
            "evidence": {
                **self.evidence,
                "live_write": False,
                "queue_mutation": False,
                "readback_required_before_promotion": True,
                "rollback_required_before_promotion": self.proposed_action_family
                in CONSOLE_AFFECTING_FAMILIES,
            },
        }
        proposal["operator_correction"] = build_operator_correction_payload(
            proposal=proposal,
            method_id=DECISION_EVIDENCE_METHOD_ID,
            target=self.target_id,
            proposed_action=proposed_action,
            blockers=proposal["blockers"],
        )
        return proposal


def _family_status(analysis: DecisionEvidenceAnalysisInput) -> Tuple[Dict[str, Dict[str, Any]], Tuple[str, ...], Tuple[str, ...]]:
    family_status: Dict[str, Dict[str, Any]] = {}
    missing = []
    low_confidence = []
    for family in EVIDENCE_FAMILIES:
        available = bool(getattr(analysis, f"{family}_available"))
        confidence = _clip01(getattr(analysis, f"{family}_confidence"))
        family_status[family] = {
            "available": available,
            "confidence": round(confidence, 3),
        }
        if not available:
            missing.append(family)
            continue
        threshold = MIN_CRITICAL_CONFIDENCE if family in CRITICAL_EVIDENCE_FAMILIES else MIN_FAMILY_CONFIDENCE
        if confidence < threshold:
            low_confidence.append(family)
    return family_status, tuple(missing), tuple(low_confidence)


def _severity(blockers: Iterable[str], action_family: str, active_performance: bool) -> str:
    blocker_set = set(blockers)
    if (
        "feedback_risk_high" in blocker_set
        or "true_peak_headroom_low" in blocker_set
        or "auto_apply_not_supported" in blocker_set
        or "source_identity_blocked_from_downstream_trust" in blocker_set
        or "source_identity_replay_context_not_fresh" in blocker_set
    ):
        return "high"
    if (
        "source_identity_not_trusted_for_downstream" in blocker_set
        or "source_identity_trust_state_unknown" in blocker_set
    ):
        return "medium"
    if action_family in {"routing", "scene_preset", "master_bus"} or active_performance:
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


def analyze_decision_evidence_packet_need(
    analysis: DecisionEvidenceAnalysisInput,
    *,
    min_confidence: float = MIN_CONFIDENCE,
) -> DecisionEvidenceRecommendation:
    decision_kind = normalize_decision_kind(analysis.decision_kind)
    action_family = normalize_action_family(analysis.proposed_action_family)
    operator_intent = str(analysis.operator_intent or "").strip().lower()
    family_status, missing, low_confidence = _family_status(analysis)
    available_count = sum(1 for item in family_status.values() if item["available"])
    available_confidences = [
        item["confidence"]
        for item in family_status.values()
        if item["available"]
    ]
    average_family_confidence = (
        sum(available_confidences) / len(available_confidences)
        if available_confidences
        else 0.0
    )
    persistence = max(1, int(analysis.decision_persistence_cycles or 1))
    conflicts = max(0, int(analysis.conflicting_evidence_count or 0))
    headroom = _safe_float(analysis.true_peak_headroom_db, 0.0)
    feedback_risk = _clip01(analysis.feedback_risk)
    latency_budget = max(0.0, _safe_float(analysis.latency_budget_ms, 0.0))
    console_affecting = action_family in CONSOLE_AFFECTING_FAMILIES
    current_state_freshness = _freshness_payload(analysis.current_state_cache_freshness)
    current_state_freshness_status = str(current_state_freshness.get("status") or "unknown")
    current_state_no_authority = _freshness_is_no_authority(current_state_freshness)
    source_identity_trust = evaluate_source_identity_downstream_trust(
        source_id=str(analysis.target_id),
        source_type=str(action_family),
        trust_state=analysis.source_identity_trust_state,
        trust_confidence=analysis.source_identity_trust_confidence,
        replay_outcome_count=analysis.source_identity_replay_outcome_count,
        current_state_cache_freshness=analysis.source_identity_calibration_freshness,
    )

    blockers = []
    if decision_kind not in SUPPORTED_DECISION_KINDS:
        blockers.append("unsupported_decision_kind")
    if action_family not in SUPPORTED_ACTION_FAMILIES:
        blockers.append("unsupported_action_family")
    if not analysis.operator_review_enabled:
        blockers.append("operator_review_disabled")
    if console_affecting and not operator_intent:
        blockers.append("operator_intent_required")
    if available_count < MIN_REQUIRED_FAMILY_COUNT:
        blockers.append("insufficient_evidence_family_count")
    for family in CRITICAL_EVIDENCE_FAMILIES:
        if family in missing:
            blockers.append(f"{family}_missing")
    for family in low_confidence:
        blockers.append(f"{family}_confidence_low")
    if console_affecting and not analysis.current_state_readback_available:
        blockers.append("current_state_readback_missing")
    if console_affecting and not analysis.rollback_plan_available:
        blockers.append("rollback_plan_missing")
    if console_affecting and headroom < MIN_TRUE_PEAK_HEADROOM_DB:
        blockers.append("true_peak_headroom_low")
    if console_affecting and feedback_risk >= HIGH_FEEDBACK_RISK:
        blockers.append("feedback_risk_high")
    if not analysis.latency_budget_met or latency_budget <= 0.0:
        blockers.append("latency_budget_missing")
    if persistence < MIN_PERSISTENCE_CYCLES:
        blockers.append("decision_persistence_low")
    if conflicts > MAX_CONFLICTING_EVIDENCE_COUNT:
        blockers.append("conflicting_evidence_high")
    if analysis.manual_override_active:
        blockers.append("manual_override_active")
    if analysis.auto_apply_requested:
        blockers.append("auto_apply_not_supported")
    if current_state_freshness_status != "fresh":
        blockers.append("decision_evidence_context_cache_not_fresh")
    if current_state_freshness_status == "fresh" and not current_state_no_authority:
        blockers.append("decision_evidence_context_cache_live_authority_present")
    blockers.extend(source_identity_trust.blockers)

    coverage_score = available_count / len(EVIDENCE_FAMILIES)
    persistence_score = min(persistence / 4.0, 1.0)
    safety_score = 0.5 * min(max(headroom, 0.0) / 6.0, 1.0) + 0.5 * (1.0 - feedback_risk)
    conflict_penalty = min(conflicts * 0.12, 0.36)
    confidence = _clip01(
        0.38 * average_family_confidence
        + 0.24 * coverage_score
        + 0.12 * persistence_score
        + 0.08 * (1.0 if analysis.current_state_readback_available or not console_affecting else 0.0)
        + 0.07 * (1.0 if analysis.rollback_plan_available or not console_affecting else 0.0)
        + 0.06 * (1.0 if analysis.latency_budget_met and latency_budget > 0.0 else 0.0)
        + 0.05 * safety_score
        - conflict_penalty
    )
    blockers_tuple = tuple(sorted(set(blockers)))
    status = "proposal_only" if not blockers_tuple and confidence >= min_confidence else "analysis_only"
    if status == "proposal_only":
        reason = f"{analysis.target_id} has enough evidence for operator review of {action_family}."
    else:
        reason = f"{analysis.target_id} does not meet decision-evidence packet criteria."

    evidence = {
        "source": DECISION_EVIDENCE_VERSION,
        "decision_id": str(analysis.decision_id),
        "decision_kind": decision_kind,
        "proposed_action_family": action_family,
        "operator_intent": operator_intent,
        "operator_review_enabled": bool(analysis.operator_review_enabled),
        "family_status": family_status,
        "audio_analysis": family_status["audio_analysis"],
        "psychoacoustic_model": family_status["psychoacoustic_model"],
        "statistical_model": family_status["statistical_model"],
        "project_history": family_status["project_history"],
        "reference_comparison": family_status["reference_comparison"],
        "previous_decision_outcomes": family_status["previous_decision_outcomes"],
        "inter_channel_context": family_status["inter_channel_context"],
        "arrangement_context": family_status["arrangement_context"],
        "genre_context": family_status["genre_context"],
        "available_evidence_family_count": available_count,
        "required_evidence_family_count": MIN_REQUIRED_FAMILY_COUNT,
        "critical_evidence_families": list(CRITICAL_EVIDENCE_FAMILIES),
        "average_family_confidence": round(average_family_confidence, 3),
        "coverage_score": round(coverage_score, 3),
        "current_state_readback_available": bool(analysis.current_state_readback_available),
        "readback_state": {
            "available": bool(analysis.current_state_readback_available),
            "required_before_live_promotion": True,
        },
        "rollback_plan_available": bool(analysis.rollback_plan_available),
        "rollback_plan": {
            "available": bool(analysis.rollback_plan_available),
            "required_for_console_affecting_family": bool(console_affecting),
        },
        "true_peak_headroom_db": round(headroom, 2),
        "true_peak_headroom": {
            "headroom_db": round(headroom, 2),
            "minimum_db": MIN_TRUE_PEAK_HEADROOM_DB,
        },
        "feedback_risk": round(feedback_risk, 3),
        "latency_budget_ms": round(latency_budget, 2),
        "latency_budget_met": bool(analysis.latency_budget_met),
        "decision_persistence_cycles": persistence,
        "conflicting_evidence_count": conflicts,
        "manual_override_active": bool(analysis.manual_override_active),
        "auto_apply_requested": bool(analysis.auto_apply_requested),
        "active_performance": bool(analysis.active_performance),
        "console_affecting_family": bool(console_affecting),
        "current_state_cache_freshness_status": current_state_freshness_status,
        "current_state_cache_freshness": current_state_freshness,
        "current_decision_evidence_context_cache_freshness": current_state_freshness,
        "current_decision_evidence_context_cache_no_live_authority": current_state_no_authority,
        "source_identity_downstream_trust": source_identity_trust.to_dict(),
        "transport_free": True,
        "decision_evidence_packet": True,
    }

    return DecisionEvidenceRecommendation(
        id=f"{DECISION_EVIDENCE_METHOD_ID}:{analysis.decision_id}",
        status=status,
        action_type="decision_evidence_packet_proposal",
        decision_id=str(analysis.decision_id),
        target_id=str(analysis.target_id),
        decision_kind=decision_kind,
        proposed_action_family=action_family,
        evidence_family_count=available_count,
        missing_evidence_families=missing,
        low_confidence_families=low_confidence,
        confidence=round(confidence, 3),
        severity=_severity(blockers_tuple, action_family, bool(analysis.active_performance)),
        reason=reason,
        blockers=blockers_tuple,
        evidence=evidence,
    )


def build_decision_evidence_packet_proposals(
    analyses: Iterable[DecisionEvidenceAnalysisInput],
) -> Tuple[Dict[str, Any], ...]:
    proposals = []
    for analysis in analyses:
        recommendation = analyze_decision_evidence_packet_need(analysis)
        if recommendation.status == "proposal_only":
            proposals.append(recommendation.to_operator_proposal())
    return tuple(proposals)
