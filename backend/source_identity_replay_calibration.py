"""Proposal-only replay calibration for source identity confidence.

This module is transport-free. It summarizes accepted/rejected source identity
outcomes so downstream proposal layers can see whether a classifier label should
be trusted, held for review, or blocked from source-specific decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


SOURCE_IDENTITY_REPLAY_METHOD_ID = "source_identity_replay_calibration"
SOURCE_IDENTITY_REPLAY_VERSION = "source_identity_replay_calibration_v1"
SUPPORTED_OPERATOR_DECISIONS = {
    "accepted",
    "corrected",
    "rejected",
    "dismissed",
}
MIN_OUTCOME_COUNT = 3
MIN_ACCEPTANCE_RATE = 0.72
MIN_REPLAY_MATCH_RATE = 0.65
MIN_AVERAGE_CONFIDENCE = 0.55


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _clip01(value: Any) -> float:
    return _clip(_safe_float(value), 0.0, 1.0)


def _normalize_token(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text or default


def normalize_source_type(source_type: Any) -> str:
    text = _normalize_token(source_type)
    aliases = {
        "lead": "lead_vocal",
        "vox": "lead_vocal",
        "vocal": "lead_vocal",
        "bv": "backing_vocal",
        "bgv": "backing_vocal",
        "bass_guitar": "bass",
        "bass_di": "bass",
        "bass_mic": "bass",
        "bd": "kick",
        "sn": "snare",
        "sd": "snare",
        "hihat": "hi_hat",
        "hh": "hi_hat",
        "gtr": "electric_guitar",
        "egtr": "electric_guitar",
        "agtr": "acoustic_guitar",
        "keys_piano": "keys",
        "track": "playback",
        "tracks": "playback",
        "return": "fx_return",
    }
    return aliases.get(text, text)


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
        "unknown_fields",
        "missing_fields",
        "max_age_seconds",
        "oldest_age_seconds",
        "read_only",
        "live_write",
        "queries_console",
    )
    normalized = {key: payload[key] for key in allowed_keys if key in payload}
    normalized.setdefault("status", _freshness_status(payload))
    normalized.setdefault("provided", True)
    return normalized


@dataclass(frozen=True)
class SourceIdentityReplayEvent:
    event_id: str
    source_id: str
    predicted_source_type: str
    predicted_confidence: float
    operator_decision: str
    accepted_source_type: str = ""
    replay_matched: bool = False
    downstream_family: str = "analysis_only"
    evidence_packet_id: str = ""
    safety_regression: bool = False
    manual_override: bool = False
    session_id: str = ""


@dataclass(frozen=True)
class SourceIdentityCalibrationRecommendation:
    id: str
    status: str
    action_type: str
    source_type: str
    recommended_trust_state: str
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
        """Return a Product Layer style calibration proposal without live apply."""

        return {
            "id": self.id,
            "method_id": SOURCE_IDENTITY_REPLAY_METHOD_ID,
            "source": SOURCE_IDENTITY_REPLAY_METHOD_ID,
            "title": f"{self.source_type}: source identity replay calibration",
            "reason": self.reason,
            "target": self.source_type,
            "kind": "source_identity_calibration",
            "severity": self.severity,
            "confidence": self.confidence,
            "queue_action": "not_queueable_until_promoted",
            "can_import_to_queue": False,
            "can_apply_live": False,
            "blockers": [
                "unsupported_value_type:source_identity_calibration",
                "not_in_supervised_fader_gain_executor",
                "calibration_requires_operator_review",
                *self.blockers,
            ],
            "proposed_action": {
                "action_type": self.action_type,
                "source_type": self.source_type,
                "recommended_trust_state": self.recommended_trust_state,
                "calibration_state": "operator_review_required",
            },
            "evidence": {
                **self.evidence,
                "live_write": False,
                "queue_mutation": False,
                "model_mutation": False,
                "memory_mutation": False,
                "operator_acceptance_required_before_calibration_write": True,
            },
        }


@dataclass(frozen=True)
class SourceIdentityReplayCalibrationResult:
    method_id: str
    version: str
    status: str
    recommendations: Tuple[SourceIdentityCalibrationRecommendation, ...]
    warnings: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_id": self.method_id,
            "version": self.version,
            "status": self.status,
            "recommendations": [item.to_dict() for item in self.recommendations],
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def _decision(value: Any) -> str:
    return _normalize_token(value)


def _accepted(event: SourceIdentityReplayEvent) -> bool:
    decision = _decision(event.operator_decision)
    predicted = normalize_source_type(event.predicted_source_type)
    accepted = normalize_source_type(event.accepted_source_type or event.predicted_source_type)
    return decision == "accepted" or (decision == "corrected" and predicted == accepted)


def _corrected_away(event: SourceIdentityReplayEvent) -> bool:
    decision = _decision(event.operator_decision)
    predicted = normalize_source_type(event.predicted_source_type)
    accepted = normalize_source_type(event.accepted_source_type or predicted)
    return decision == "corrected" and predicted != accepted


def _summary_for(source_type: str, events: Tuple[SourceIdentityReplayEvent, ...]) -> Dict[str, Any]:
    accepted_count = sum(1 for event in events if _accepted(event))
    corrected_away_count = sum(1 for event in events if _corrected_away(event))
    rejected_count = sum(1 for event in events if _decision(event.operator_decision) == "rejected")
    dismissed_count = sum(1 for event in events if _decision(event.operator_decision) == "dismissed")
    replay_match_count = sum(1 for event in events if event.replay_matched)
    safety_regression_count = sum(1 for event in events if event.safety_regression)
    manual_override_count = sum(1 for event in events if event.manual_override)
    total = len(events)
    confidence_sum = sum(_clip01(event.predicted_confidence) for event in events)
    return {
        "source_type": source_type,
        "outcome_count": total,
        "accepted_count": accepted_count,
        "corrected_away_count": corrected_away_count,
        "rejected_count": rejected_count,
        "dismissed_count": dismissed_count,
        "replay_match_count": replay_match_count,
        "safety_regression_count": safety_regression_count,
        "manual_override_count": manual_override_count,
        "acceptance_rate": round(accepted_count / total if total else 0.0, 3),
        "replay_match_rate": round(replay_match_count / total if total else 0.0, 3),
        "average_predicted_confidence": round(confidence_sum / total if total else 0.0, 3),
        "downstream_families": sorted({
            _normalize_token(event.downstream_family, "analysis_only")
            for event in events
        }),
        "evidence_packet_ids": sorted({
            str(event.evidence_packet_id)
            for event in events
            if str(event.evidence_packet_id or "").strip()
        }),
        "session_ids": sorted({
            str(event.session_id)
            for event in events
            if str(event.session_id or "").strip()
        }),
    }


def _recommendation_from_summary(
    source_type: str,
    summary: Dict[str, Any],
    current_state_freshness: Dict[str, Any],
) -> SourceIdentityCalibrationRecommendation:
    acceptance_rate = _clip01(summary["acceptance_rate"])
    replay_match_rate = _clip01(summary["replay_match_rate"])
    average_confidence = _clip01(summary["average_predicted_confidence"])
    safety_regressions = _safe_int(summary["safety_regression_count"])
    corrected_away = _safe_int(summary["corrected_away_count"])
    rejected = _safe_int(summary["rejected_count"])
    total = _safe_int(summary["outcome_count"])

    blockers = []
    if total < MIN_OUTCOME_COUNT:
        blockers.append("source_identity_outcome_count_low")
    if average_confidence < MIN_AVERAGE_CONFIDENCE:
        blockers.append("classifier_average_confidence_low")
    if replay_match_rate < MIN_REPLAY_MATCH_RATE:
        blockers.append("source_identity_replay_match_low")
    if safety_regressions > 0:
        blockers.append("source_identity_safety_regression_observed")

    if safety_regressions > 0:
        trust_state = "blocked_from_downstream_trust"
        severity = "high"
    elif acceptance_rate >= MIN_ACCEPTANCE_RATE and not blockers:
        trust_state = "trusted_for_downstream_priority"
        severity = "low"
    elif corrected_away + rejected > 0:
        trust_state = "hold_for_operator_review"
        severity = "medium"
    else:
        trust_state = "needs_more_replay_outcomes"
        severity = "low"

    confidence = _clip01(
        0.34 * acceptance_rate
        + 0.28 * replay_match_rate
        + 0.22 * average_confidence
        + 0.16 * min(total / 8.0, 1.0)
        - min(safety_regressions * 0.35 + corrected_away * 0.04 + rejected * 0.03, 0.5)
    )
    if trust_state == "trusted_for_downstream_priority":
        reason = f"{source_type} has enough accepted replay outcomes for downstream priority review."
    elif trust_state == "blocked_from_downstream_trust":
        reason = f"{source_type} replay outcomes include safety regression evidence."
    elif trust_state == "hold_for_operator_review":
        reason = f"{source_type} replay outcomes disagree with classifier predictions often enough to hold downstream trust."
    else:
        reason = f"{source_type} needs more accepted/rejected replay outcomes before downstream trust."

    evidence = {
        "source": SOURCE_IDENTITY_REPLAY_VERSION,
        **summary,
        "min_outcome_count": MIN_OUTCOME_COUNT,
        "min_acceptance_rate": MIN_ACCEPTANCE_RATE,
        "min_replay_match_rate": MIN_REPLAY_MATCH_RATE,
        "min_average_confidence": MIN_AVERAGE_CONFIDENCE,
        "recommended_trust_state": trust_state,
        "current_source_identity_replay_context_cache_freshness": current_state_freshness.get("status"),
        "current_state_cache_freshness": current_state_freshness,
        "transport_free": True,
        "model_mutation": False,
        "memory_mutation": False,
        "downstream_trust_requires_operator_acceptance": True,
    }
    return SourceIdentityCalibrationRecommendation(
        id=f"{SOURCE_IDENTITY_REPLAY_METHOD_ID}:{source_type}",
        status="proposal_only",
        action_type="source_identity_replay_calibration_candidate",
        source_type=source_type,
        recommended_trust_state=trust_state,
        confidence=round(confidence, 3),
        severity=severity,
        reason=reason,
        blockers=tuple(sorted(set(blockers))),
        evidence=evidence,
    )


def analyze_source_identity_replay_calibration(
    events: Iterable[SourceIdentityReplayEvent],
    *,
    current_state_cache_freshness: Optional[Mapping[str, Any]] = None,
    operator_review_enabled: bool = True,
) -> SourceIdentityReplayCalibrationResult:
    current_state_freshness = _freshness_payload(current_state_cache_freshness)
    current_state_freshness_status = str(current_state_freshness.get("status") or "unknown")
    event_tuple = tuple(events)
    metadata = {
        "proposal_only": True,
        "live_write_allowed": False,
        "queue_import_allowed": False,
        "model_mutation_allowed": False,
        "memory_mutation_allowed": False,
        "operator_review_enabled": bool(operator_review_enabled),
        "current_state_cache_freshness_status": current_state_freshness_status,
        "current_state_cache_freshness": current_state_freshness,
        "event_count": len(event_tuple),
        "blockers": (),
    }

    if current_state_freshness_status not in {"fresh", "not_provided"}:
        blocker = "source_identity_replay_context_cache_not_fresh"
        metadata["blockers"] = (blocker,)
        metadata["calibration_candidate_count"] = 0
        return SourceIdentityReplayCalibrationResult(
            method_id=SOURCE_IDENTITY_REPLAY_METHOD_ID,
            version=SOURCE_IDENTITY_REPLAY_VERSION,
            status="analysis_only",
            recommendations=(),
            warnings=("source identity replay context cache is not fresh; calibration blocked",),
            metadata=metadata,
        )

    if not operator_review_enabled:
        blocker = "operator_review_disabled"
        metadata["blockers"] = (blocker,)
        metadata["calibration_candidate_count"] = 0
        return SourceIdentityReplayCalibrationResult(
            method_id=SOURCE_IDENTITY_REPLAY_METHOD_ID,
            version=SOURCE_IDENTITY_REPLAY_VERSION,
            status="analysis_only",
            recommendations=(),
            warnings=("operator review is disabled; source identity calibration blocked",),
            metadata=metadata,
        )

    grouped: Dict[str, list[SourceIdentityReplayEvent]] = {}
    unsupported_decisions = 0
    for event in event_tuple:
        decision = _decision(event.operator_decision)
        if decision not in SUPPORTED_OPERATOR_DECISIONS:
            unsupported_decisions += 1
            continue
        source_type = normalize_source_type(event.predicted_source_type)
        grouped.setdefault(source_type, []).append(event)

    recommendations = tuple(
        _recommendation_from_summary(
            source_type,
            _summary_for(source_type, tuple(items)),
            current_state_freshness,
        )
        for source_type, items in sorted(grouped.items())
        if len(items) >= MIN_OUTCOME_COUNT
    )
    metadata["unsupported_decision_count"] = unsupported_decisions
    metadata["source_type_count"] = len(grouped)
    metadata["calibration_candidate_count"] = len(recommendations)
    status = "proposal_only" if recommendations else "analysis_only"
    warnings = ()
    if unsupported_decisions:
        warnings = (f"{unsupported_decisions} source identity replay events used unsupported decisions",)
    return SourceIdentityReplayCalibrationResult(
        method_id=SOURCE_IDENTITY_REPLAY_METHOD_ID,
        version=SOURCE_IDENTITY_REPLAY_VERSION,
        status=status,
        recommendations=recommendations,
        warnings=warnings,
        metadata=metadata,
    )


def build_source_identity_replay_calibration_proposals(
    events: Iterable[SourceIdentityReplayEvent],
    *,
    current_state_cache_freshness: Optional[Mapping[str, Any]] = None,
    operator_review_enabled: bool = True,
) -> Tuple[Dict[str, Any], ...]:
    result = analyze_source_identity_replay_calibration(
        events,
        current_state_cache_freshness=current_state_cache_freshness,
        operator_review_enabled=operator_review_enabled,
    )
    return tuple(item.to_operator_proposal() for item in result.recommendations)
