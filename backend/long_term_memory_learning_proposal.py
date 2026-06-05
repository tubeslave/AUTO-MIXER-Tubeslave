"""Proposal-only long-term memory learning recommendations.

This module is transport-free. It evaluates verified decisions, offline replay
runs, telemetry sessions, evidence packets, and operator outcomes to decide
whether a learning record is ready for operator review. It does not update the
mixer, mutate the proposal queue, or grant live authority.
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


LONG_TERM_MEMORY_METHOD_ID = "long_term_memory_learning_proposal"
LONG_TERM_MEMORY_VERSION = "long_term_memory_learning_proposal_v1"
SUPPORTED_SCOPE_KINDS = {
    "project",
    "show",
    "session",
    "venue",
    "artist",
    "song",
    "source",
    "channel",
    "bus",
    "mix_profile",
    "global",
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
    "decision_evidence",
}
MIN_CONFIDENCE = 0.5
MIN_VERIFIED_OR_OFFLINE_EVIDENCE = 2
MIN_OUTCOME_LABEL_COUNT = 2
MIN_EVIDENCE_PACKET_COUNT = 1
MIN_TELEMETRY_SESSION_COUNT = 1
MIN_REPLAY_CORRELATION = 0.6
MIN_OUTCOME_CONSISTENCY = 0.62
MAX_CONFLICTING_OUTCOME_COUNT = 1
MAX_SAFETY_REGRESSION_COUNT = 0
MAX_MANUAL_OVERRIDE_BURST = 3


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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_token(value: Any, default: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text or default


def normalize_scope_kind(scope_kind: Any) -> str:
    text = _normalize_token(scope_kind, "project")
    aliases = {
        "concert": "show",
        "performance": "show",
        "track": "song",
        "input": "channel",
        "stem": "bus",
        "profile": "mix_profile",
    }
    return aliases.get(text, text)


def normalize_learning_action_family(action_family: Any) -> str:
    text = _normalize_token(action_family, "analysis_only")
    aliases = {
        "analysis": "analysis_only",
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
        "evidence": "decision_evidence",
    }
    return aliases.get(text, text)


@dataclass(frozen=True)
class LongTermMemoryLearningInput:
    learning_id: str
    scope_id: str
    scope_kind: str = "project"
    proposed_action_family: str = "analysis_only"
    operator_intent: str = ""
    operator_review_enabled: bool = True
    source_session_count: int = 0
    telemetry_session_count: int = 0
    verified_decision_count: int = 0
    offline_replay_count: int = 0
    accepted_decision_count: int = 0
    rejected_decision_count: int = 0
    evidence_packet_count: int = 0
    engineer_action_count: int = 0
    reference_comparison_count: int = 0
    genre_context_count: int = 0
    arrangement_context_count: int = 0
    replay_correlation_score: float = 0.0
    outcome_consistency_score: float = 0.0
    safety_regression_count: int = 0
    conflicting_outcome_count: int = 0
    recent_manual_override_count: int = 0
    latency_budget_met: bool = True
    manual_review_required: bool = True
    auto_apply_requested: bool = False
    active_performance: bool = False
    current_state_cache_freshness: Optional[Dict[str, Any]] = None
    source_identity_trust_state: str = "not_provided"
    source_identity_trust_confidence: float = 0.0
    source_identity_replay_outcome_count: int = 0
    source_identity_calibration_freshness: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class LongTermMemoryLearningRecommendation:
    id: str
    status: str
    action_type: str
    learning_id: str
    scope_id: str
    scope_kind: str
    proposed_action_family: str
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
        """Return a Product Layer style memory proposal without live-apply fields."""

        return {
            "id": self.id,
            "method_id": LONG_TERM_MEMORY_METHOD_ID,
            "source": LONG_TERM_MEMORY_METHOD_ID,
            "title": f"{self.scope_id}: long-term memory learning proposal",
            "reason": self.reason,
            "target": self.scope_id,
            "kind": "long_term_memory_learning",
            "severity": self.severity,
            "confidence": self.confidence,
            "queue_action": "not_queueable_until_promoted",
            "can_import_to_queue": False,
            "can_apply_live": False,
            "blockers": [
                "unsupported_value_type:long_term_memory_learning",
                "not_in_supervised_fader_gain_executor",
                "memory_learning_requires_replay_and_operator_acceptance",
                *self.blockers,
            ],
            "proposed_action": {
                "action_type": self.action_type,
                "learning_id": self.learning_id,
                "scope_id": self.scope_id,
                "scope_kind": self.scope_kind,
                "proposed_action_family": self.proposed_action_family,
                "learning_record_state": "operator_review_required",
            },
            "evidence": {
                **self.evidence,
                "live_write": False,
                "queue_mutation": False,
                "memory_mutation": False,
                "operator_acceptance_required_before_memory_write": True,
                "readback_required_before_promotion": False,
                "rollback_required_before_promotion": False,
            },
        }


def _severity(blockers: Iterable[str], active_performance: bool) -> str:
    blocker_set = set(blockers)
    if (
        "source_identity_blocked_from_downstream_trust" in blocker_set
        or "source_identity_replay_context_not_fresh" in blocker_set
        or "safety_regression_observed" in blocker_set
        or "auto_apply_not_supported" in blocker_set
        or "active_performance_learning_write_blocked" in blocker_set
    ):
        return "high"
    if (
        "source_identity_not_trusted_for_downstream" in blocker_set
        or "source_identity_trust_state_unknown" in blocker_set
    ):
        return "medium"
    if active_performance or "conflicting_outcomes_high" in blocker_set:
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


def analyze_long_term_memory_learning_need(
    analysis: LongTermMemoryLearningInput,
    *,
    min_confidence: float = MIN_CONFIDENCE,
) -> LongTermMemoryLearningRecommendation:
    scope_kind = normalize_scope_kind(analysis.scope_kind)
    action_family = normalize_learning_action_family(analysis.proposed_action_family)
    verified = max(0, _safe_int(analysis.verified_decision_count))
    offline = max(0, _safe_int(analysis.offline_replay_count))
    accepted = max(0, _safe_int(analysis.accepted_decision_count))
    rejected = max(0, _safe_int(analysis.rejected_decision_count))
    source_sessions = max(0, _safe_int(analysis.source_session_count))
    telemetry_sessions = max(0, _safe_int(analysis.telemetry_session_count))
    evidence_packets = max(0, _safe_int(analysis.evidence_packet_count))
    engineer_actions = max(0, _safe_int(analysis.engineer_action_count))
    references = max(0, _safe_int(analysis.reference_comparison_count))
    genre_contexts = max(0, _safe_int(analysis.genre_context_count))
    arrangement_contexts = max(0, _safe_int(analysis.arrangement_context_count))
    safety_regressions = max(0, _safe_int(analysis.safety_regression_count))
    conflicts = max(0, _safe_int(analysis.conflicting_outcome_count))
    manual_overrides = max(0, _safe_int(analysis.recent_manual_override_count))
    replay_correlation = _clip01(analysis.replay_correlation_score)
    outcome_consistency = _clip01(analysis.outcome_consistency_score)
    outcome_labels = accepted + rejected
    verified_or_offline = verified + offline
    operator_intent = str(analysis.operator_intent or "").strip().lower()
    current_state_freshness = _freshness_payload(analysis.current_state_cache_freshness)
    current_state_freshness_status = str(current_state_freshness.get("status") or "unknown")
    source_identity_trust = evaluate_source_identity_downstream_trust(
        source_id=str(analysis.scope_id),
        source_type=str(action_family),
        trust_state=analysis.source_identity_trust_state,
        trust_confidence=analysis.source_identity_trust_confidence,
        replay_outcome_count=analysis.source_identity_replay_outcome_count,
        current_state_cache_freshness=analysis.source_identity_calibration_freshness,
    )

    blockers = []
    if scope_kind not in SUPPORTED_SCOPE_KINDS:
        blockers.append("unsupported_learning_scope")
    if action_family not in SUPPORTED_ACTION_FAMILIES:
        blockers.append("unsupported_learning_action_family")
    if not analysis.operator_review_enabled:
        blockers.append("operator_review_disabled")
    if not analysis.manual_review_required:
        blockers.append("manual_review_required_flag_missing")
    if action_family != "analysis_only" and not operator_intent:
        blockers.append("operator_intent_required")
    if verified_or_offline < MIN_VERIFIED_OR_OFFLINE_EVIDENCE:
        blockers.append("insufficient_verified_or_offline_evidence")
    if outcome_labels < MIN_OUTCOME_LABEL_COUNT:
        blockers.append("decision_outcome_labels_missing")
    if evidence_packets < MIN_EVIDENCE_PACKET_COUNT:
        blockers.append("evidence_packets_missing")
    if telemetry_sessions < MIN_TELEMETRY_SESSION_COUNT:
        blockers.append("telemetry_history_missing")
    if replay_correlation < MIN_REPLAY_CORRELATION:
        blockers.append("replay_correlation_low")
    if outcome_consistency < MIN_OUTCOME_CONSISTENCY:
        blockers.append("outcome_consistency_low")
    if safety_regressions > MAX_SAFETY_REGRESSION_COUNT:
        blockers.append("safety_regression_observed")
    if conflicts > MAX_CONFLICTING_OUTCOME_COUNT:
        blockers.append("conflicting_outcomes_high")
    if manual_overrides > MAX_MANUAL_OVERRIDE_BURST:
        blockers.append("manual_override_burst_high")
    if not analysis.latency_budget_met:
        blockers.append("latency_budget_missing")
    if analysis.auto_apply_requested:
        blockers.append("auto_apply_not_supported")
    if analysis.active_performance:
        blockers.append("active_performance_learning_write_blocked")
    if current_state_freshness_status not in {"fresh", "not_provided"}:
        blockers.append("long_term_memory_context_cache_not_fresh")
    blockers.extend(source_identity_trust.blockers)

    evidence_volume_score = min(verified_or_offline / 6.0, 1.0)
    outcome_score = min(outcome_labels / 8.0, 1.0)
    context_score = min((references + genre_contexts + arrangement_contexts) / 6.0, 1.0)
    telemetry_score = min((telemetry_sessions + source_sessions) / 4.0, 1.0)
    engineer_score = min(engineer_actions / 6.0, 1.0)
    conflict_penalty = min(conflicts * 0.12 + safety_regressions * 0.3 + manual_overrides * 0.04, 0.5)
    confidence = _clip01(
        0.22 * evidence_volume_score
        + 0.18 * replay_correlation
        + 0.17 * outcome_consistency
        + 0.12 * outcome_score
        + 0.10 * telemetry_score
        + 0.08 * min(evidence_packets / 4.0, 1.0)
        + 0.07 * context_score
        + 0.04 * engineer_score
        + 0.02 * (1.0 if analysis.operator_review_enabled else 0.0)
        - conflict_penalty
    )
    blockers_tuple = tuple(sorted(set(blockers)))
    status = "proposal_only" if not blockers_tuple and confidence >= min_confidence else "analysis_only"
    if status == "proposal_only":
        reason = f"{analysis.scope_id} has enough replay and operator evidence for memory-learning review."
    else:
        reason = f"{analysis.scope_id} does not meet long-term memory learning criteria."

    evidence = {
        "source": LONG_TERM_MEMORY_VERSION,
        "learning_id": str(analysis.learning_id),
        "scope_kind": scope_kind,
        "proposed_action_family": action_family,
        "operator_intent": operator_intent,
        "operator_review_enabled": bool(analysis.operator_review_enabled),
        "manual_review_required": bool(analysis.manual_review_required),
        "source_session_count": source_sessions,
        "telemetry_session_count": telemetry_sessions,
        "verified_decision_count": verified,
        "offline_replay_count": offline,
        "verified_or_offline_evidence_count": verified_or_offline,
        "accepted_decision_count": accepted,
        "rejected_decision_count": rejected,
        "outcome_label_count": outcome_labels,
        "evidence_packet_count": evidence_packets,
        "engineer_action_count": engineer_actions,
        "reference_comparison_count": references,
        "genre_context_count": genre_contexts,
        "arrangement_context_count": arrangement_contexts,
        "replay_correlation_score": round(replay_correlation, 3),
        "outcome_consistency_score": round(outcome_consistency, 3),
        "safety_regression_count": safety_regressions,
        "conflicting_outcome_count": conflicts,
        "recent_manual_override_count": manual_overrides,
        "latency_budget_met": bool(analysis.latency_budget_met),
        "auto_apply_requested": bool(analysis.auto_apply_requested),
        "active_performance": bool(analysis.active_performance),
        "current_state_cache_freshness_status": current_state_freshness_status,
        "current_state_cache_freshness": current_state_freshness,
        "source_identity_downstream_trust": source_identity_trust.to_dict(),
        "transport_free": True,
        "long_term_memory_learning": True,
        "learning_requires_replay": True,
    }

    return LongTermMemoryLearningRecommendation(
        id=f"{LONG_TERM_MEMORY_METHOD_ID}:{analysis.learning_id}",
        status=status,
        action_type="long_term_memory_learning_proposal",
        learning_id=str(analysis.learning_id),
        scope_id=str(analysis.scope_id),
        scope_kind=scope_kind,
        proposed_action_family=action_family,
        confidence=round(confidence, 3),
        severity=_severity(blockers_tuple, bool(analysis.active_performance)),
        reason=reason,
        blockers=blockers_tuple,
        evidence=evidence,
    )


def build_long_term_memory_learning_proposals(
    analyses: Iterable[LongTermMemoryLearningInput],
) -> Tuple[Dict[str, Any], ...]:
    proposals = []
    for analysis in analyses:
        recommendation = analyze_long_term_memory_learning_need(analysis)
        if recommendation.status == "proposal_only":
            proposals.append(recommendation.to_operator_proposal())
    return tuple(proposals)
