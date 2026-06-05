"""Proposal-only source identity confidence review.

This module is transport-free. It turns classifier/name-recognizer confidence
evidence into operator-facing source-identity review records, but it never
creates a live console write request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


SOURCE_IDENTITY_METHOD_ID = "source_identity_confidence_proposal"
SOURCE_IDENTITY_VERSION = "source_identity_confidence_proposal_v1"
MIN_TRUSTED_SOURCE_CONFIDENCE = 0.70
MIN_ACTIVITY_CONFIDENCE = 0.25
MIN_ALTERNATIVE_MARGIN = 0.15
DOWNSTREAM_TARGET_FAMILIES = (
    "gain_staging",
    "equalization",
    "compression",
    "gating",
    "deessing",
    "panning",
    "effects_management",
    "arrangement_role",
    "frequency_separation",
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


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def normalize_source_type(source_type: Any) -> str:
    text = _normalize_token(source_type)
    aliases = {
        "lead": "lead_vocal",
        "leadvox": "lead_vocal",
        "lead_voc": "lead_vocal",
        "vox": "lead_vocal",
        "vocal": "lead_vocal",
        "bv": "backing_vocal",
        "bgv": "backing_vocal",
        "back_vocal": "backing_vocal",
        "backvocal": "backing_vocal",
        "bass_guitar": "bass",
        "bass_di": "bass",
        "bass_mic": "bass",
        "bd": "kick",
        "bass_drum": "kick",
        "sn": "snare",
        "sd": "snare",
        "hh": "hi_hat",
        "hihat": "hi_hat",
        "egtr": "electric_guitar",
        "e_gtr": "electric_guitar",
        "gtr": "electric_guitar",
        "agtr": "acoustic_guitar",
        "ac_gtr": "acoustic_guitar",
        "keys_piano": "keys",
        "keyboard": "keys",
        "loop": "playback",
        "track": "playback",
        "tracks": "playback",
        "return": "fx_return",
        "reverb": "fx_return",
        "delay": "fx_return",
        "tb": "talkback",
    }
    return aliases.get(text, text or "unknown")


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


def _tuple_str(values: Any) -> Tuple[str, ...]:
    if isinstance(values, str):
        return (values,)
    if not isinstance(values, Iterable):
        return ()
    return tuple(str(value) for value in values if str(value or "").strip())


@dataclass(frozen=True)
class SourceIdentityAlternative:
    source_type: str
    confidence: float
    source: str = ""

    def normalized(self) -> "SourceIdentityAlternative":
        return SourceIdentityAlternative(
            source_type=normalize_source_type(self.source_type),
            confidence=round(_clip01(self.confidence), 3),
            source=str(self.source or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        item = self.normalized()
        return asdict(item)


@dataclass(frozen=True)
class SourceIdentityObservation:
    source_id: str
    channel_id: Optional[int] = None
    channel_label: str = ""
    predicted_source_type: str = "unknown"
    source_type_confidence: float = 0.0
    match_type: str = "unknown"
    classifier_source: str = "unknown"
    alternatives: Tuple[SourceIdentityAlternative, ...] = ()
    stem_roles: Tuple[str, ...] = ()
    allowed_controls: Tuple[str, ...] = ()
    priority: float = 0.0
    signal_present: bool = True
    activity_confidence: float = 1.0
    operator_label_confirmed: bool = False
    downstream_target_families: Tuple[str, ...] = DOWNSTREAM_TARGET_FAMILIES

    @classmethod
    def from_channel_classification(
        cls,
        source_id: str,
        classification: Any,
        *,
        channel_id: Optional[int] = None,
        classifier_source: str = "channel_recognizer",
    ) -> "SourceIdentityObservation":
        if hasattr(classification, "to_dict"):
            payload = classification.to_dict()
        elif isinstance(classification, Mapping):
            payload = dict(classification)
        else:
            payload = {}

        source_type = (
            payload.get("source_role")
            or payload.get("predicted_source_type")
            or payload.get("source_type")
            or "unknown"
        )
        return cls(
            source_id=source_id,
            channel_id=channel_id,
            channel_label=str(payload.get("name") or payload.get("channel_name") or source_id),
            predicted_source_type=str(source_type),
            source_type_confidence=_clip01(payload.get("confidence", 0.0)),
            match_type=str(payload.get("match_type") or "unknown"),
            classifier_source=classifier_source,
            stem_roles=_tuple_str(payload.get("stem_roles", ())),
            allowed_controls=_tuple_str(payload.get("allowed_controls", ())),
            priority=_safe_float(payload.get("priority"), 0.0),
            operator_label_confirmed=bool(payload.get("operator_label_confirmed", False)),
        )


@dataclass(frozen=True)
class SourceIdentityRecommendation:
    id: str
    status: str
    action_type: str
    source_id: str
    channel_id: Optional[int]
    channel_label: str
    proposed_source_type: str
    confidence: float
    severity: str
    proposal_priority: int
    downstream_priority_multiplier: float
    reason: str
    review_reasons: Tuple[str, ...] = ()
    downstream_rejection_reasons: Tuple[str, ...] = ()
    evidence: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["review_reasons"] = list(self.review_reasons)
        payload["downstream_rejection_reasons"] = list(self.downstream_rejection_reasons)
        return payload

    def to_operator_proposal(self) -> Dict[str, Any]:
        """Return a Product Layer style proposal without live-apply fields."""

        blockers = [
            "unsupported_value_type:source_identity",
            "not_in_supervised_fader_gain_executor",
            "source_identity_requires_operator_acceptance",
            *self.review_reasons,
            *self.downstream_rejection_reasons,
        ]
        return {
            "id": self.id,
            "method_id": SOURCE_IDENTITY_METHOD_ID,
            "source": SOURCE_IDENTITY_METHOD_ID,
            "title": f"{self.channel_label or self.source_id}: source identity review",
            "reason": self.reason,
            "target": self.source_id,
            "kind": "source_identity",
            "severity": self.severity,
            "confidence": self.confidence,
            "proposal_priority": self.proposal_priority,
            "queue_action": "not_queueable_until_promoted",
            "can_import_to_queue": False,
            "can_apply_live": False,
            "blockers": sorted(set(blockers)),
            "proposed_action": {
                "action_type": self.action_type,
                "source_id": self.source_id,
                "channel_id": self.channel_id,
                "channel_label": self.channel_label,
                "proposed_source_type": self.proposed_source_type,
                "classifier_confidence": self.confidence,
                "review_reasons": list(self.review_reasons),
                "downstream_priority_multiplier": self.downstream_priority_multiplier,
                "downstream_rejection_reasons": list(self.downstream_rejection_reasons),
            },
            "evidence": {
                **self.evidence,
                "live_write": False,
                "queue_mutation": False,
                "operator_acceptance_required_before_promotion": True,
                "readback_required_before_promotion": False,
                "rollback_required_before_promotion": False,
            },
        }


@dataclass(frozen=True)
class SourceIdentityResult:
    method_id: str
    version: str
    status: str
    recommendations: Tuple[SourceIdentityRecommendation, ...]
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


def _alternative_margin(
    predicted_source_type: str,
    confidence: float,
    alternatives: Sequence[SourceIdentityAlternative],
) -> Tuple[float, Optional[SourceIdentityAlternative]]:
    predicted = normalize_source_type(predicted_source_type)
    competitors = [
        alt.normalized()
        for alt in alternatives
        if alt.normalized().source_type != predicted
    ]
    if not competitors:
        return 1.0, None
    competitor = max(competitors, key=lambda item: item.confidence)
    return round(confidence - competitor.confidence, 3), competitor


def _review_reasons(
    observation: SourceIdentityObservation,
    *,
    min_trusted_confidence: float,
) -> Tuple[str, ...]:
    source_type = normalize_source_type(observation.predicted_source_type)
    confidence = _clip01(observation.source_type_confidence)
    activity_confidence = _clip01(observation.activity_confidence)
    margin, competitor = _alternative_margin(source_type, confidence, observation.alternatives)
    match_type = _normalize_token(observation.match_type)

    reasons = []
    if source_type == "unknown":
        reasons.append("source_identity_unknown")
    if confidence < min_trusted_confidence:
        reasons.append("source_type_confidence_low")
    if not observation.signal_present or activity_confidence < MIN_ACTIVITY_CONFIDENCE:
        reasons.append("source_signal_not_reliable")
    if match_type in {"unknown", "empty"}:
        reasons.append("classifier_match_unknown")
    if competitor is not None and margin < MIN_ALTERNATIVE_MARGIN:
        reasons.append("source_identity_ambiguous")
    return tuple(sorted(set(reasons)))


def _downstream_rejection_reasons(review_reasons: Iterable[str]) -> Tuple[str, ...]:
    reasons = set(review_reasons)
    downstream = set()
    if "source_identity_unknown" in reasons:
        downstream.add("source_type_required_for_downstream_processing")
    if "source_type_confidence_low" in reasons:
        downstream.add("source_type_confidence_below_downstream_threshold")
    if "source_identity_ambiguous" in reasons:
        downstream.add("source_identity_ambiguity_requires_operator_review")
    if "source_signal_not_reliable" in reasons:
        downstream.add("signal_activity_required_for_classifier_trust")
    if "classifier_match_unknown" in reasons:
        downstream.add("classifier_match_required_for_source_specific_processing")
    return tuple(sorted(downstream))


def _proposal_priority(review_reasons: Iterable[str], confidence: float) -> int:
    reasons = set(review_reasons)
    if {"source_identity_unknown", "source_signal_not_reliable"} & reasons:
        return 1
    if confidence < 0.45 or "source_identity_ambiguous" in reasons:
        return 2
    return 3


def _severity(review_reasons: Iterable[str], confidence: float) -> str:
    priority = _proposal_priority(review_reasons, confidence)
    if priority == 1:
        return "high"
    if priority == 2:
        return "medium"
    return "low"


def _priority_multiplier(review_reasons: Iterable[str], confidence: float) -> float:
    reasons = set(review_reasons)
    if {"source_identity_unknown", "source_signal_not_reliable"} & reasons:
        return 0.0
    if confidence < 0.45:
        return 0.25
    if "source_identity_ambiguous" in reasons:
        return 0.5
    return 0.6


def _recommendation_for_observation(
    observation: SourceIdentityObservation,
    *,
    current_state_freshness: Dict[str, Any],
    min_trusted_confidence: float,
) -> Optional[SourceIdentityRecommendation]:
    source_type = normalize_source_type(observation.predicted_source_type)
    confidence = round(_clip01(observation.source_type_confidence), 3)
    review_reasons = _review_reasons(
        observation,
        min_trusted_confidence=min_trusted_confidence,
    )
    if not review_reasons:
        return None

    margin, competitor = _alternative_margin(source_type, confidence, observation.alternatives)
    downstream_reasons = _downstream_rejection_reasons(review_reasons)
    label = observation.channel_label or observation.source_id
    reason = (
        f"{label} source identity needs operator review before source-specific "
        "processing proposals can trust it."
    )
    evidence = {
        "source": SOURCE_IDENTITY_VERSION,
        "classifier_source": observation.classifier_source,
        "match_type": observation.match_type,
        "predicted_source_type": source_type,
        "source_type_confidence": confidence,
        "min_trusted_source_confidence": min_trusted_confidence,
        "alternative_margin": margin,
        "competing_source_type": competitor.source_type if competitor else None,
        "competing_source_confidence": competitor.confidence if competitor else None,
        "alternatives": [alt.to_dict() for alt in observation.alternatives],
        "stem_roles": list(observation.stem_roles),
        "allowed_controls": list(observation.allowed_controls),
        "source_priority": observation.priority,
        "signal_present": bool(observation.signal_present),
        "activity_confidence": round(_clip01(observation.activity_confidence), 3),
        "operator_label_confirmed": bool(observation.operator_label_confirmed),
        "downstream_target_families": list(observation.downstream_target_families),
        "downstream_rejection_reasons": list(downstream_reasons),
        "current_source_identity_context_cache_freshness": current_state_freshness.get("status"),
        "current_state_cache_freshness": current_state_freshness,
        "transport_free": True,
        "decision_evidence_packet": True,
    }

    return SourceIdentityRecommendation(
        id=f"{SOURCE_IDENTITY_METHOD_ID}:{observation.source_id}",
        status="proposal_only",
        action_type="source_identity_review_candidate",
        source_id=str(observation.source_id),
        channel_id=observation.channel_id,
        channel_label=label,
        proposed_source_type=source_type,
        confidence=confidence,
        severity=_severity(review_reasons, confidence),
        proposal_priority=_proposal_priority(review_reasons, confidence),
        downstream_priority_multiplier=round(_priority_multiplier(review_reasons, confidence), 3),
        reason=reason,
        review_reasons=review_reasons,
        downstream_rejection_reasons=downstream_reasons,
        evidence=evidence,
    )


def analyze_source_identity_confidence(
    observations: Iterable[SourceIdentityObservation],
    *,
    current_state_cache_freshness: Optional[Mapping[str, Any]] = None,
    min_trusted_confidence: float = MIN_TRUSTED_SOURCE_CONFIDENCE,
    operator_review_enabled: bool = True,
) -> SourceIdentityResult:
    current_state_freshness = _freshness_payload(current_state_cache_freshness)
    current_state_freshness_status = str(current_state_freshness.get("status") or "unknown")
    min_trusted = _clip01(min_trusted_confidence)
    observations_tuple = tuple(observations)

    metadata = {
        "proposal_only": True,
        "live_write_allowed": False,
        "queue_import_allowed": False,
        "operator_review_enabled": bool(operator_review_enabled),
        "current_state_cache_freshness_status": current_state_freshness_status,
        "current_state_cache_freshness": current_state_freshness,
        "min_trusted_source_confidence": min_trusted,
        "observation_count": len(observations_tuple),
        "blockers": (),
    }

    if current_state_freshness_status not in {"fresh", "not_provided"}:
        blocker = "source_identity_context_cache_not_fresh"
        metadata["blockers"] = (blocker,)
        metadata["review_candidate_count"] = 0
        return SourceIdentityResult(
            method_id=SOURCE_IDENTITY_METHOD_ID,
            version=SOURCE_IDENTITY_VERSION,
            status="analysis_only",
            recommendations=(),
            warnings=("source identity context cache is not fresh; proposals blocked",),
            metadata=metadata,
        )

    if not operator_review_enabled:
        blocker = "operator_review_disabled"
        metadata["blockers"] = (blocker,)
        metadata["review_candidate_count"] = 0
        return SourceIdentityResult(
            method_id=SOURCE_IDENTITY_METHOD_ID,
            version=SOURCE_IDENTITY_VERSION,
            status="analysis_only",
            recommendations=(),
            warnings=("operator review is disabled; source identity proposals blocked",),
            metadata=metadata,
        )

    recommendations = tuple(
        item
        for item in (
            _recommendation_for_observation(
                observation,
                current_state_freshness=current_state_freshness,
                min_trusted_confidence=min_trusted,
            )
            for observation in observations_tuple
        )
        if item is not None
    )
    metadata["review_candidate_count"] = len(recommendations)
    status = "proposal_only" if recommendations else "analysis_only"
    return SourceIdentityResult(
        method_id=SOURCE_IDENTITY_METHOD_ID,
        version=SOURCE_IDENTITY_VERSION,
        status=status,
        recommendations=recommendations,
        metadata=metadata,
    )


def build_source_identity_confidence_proposals(
    observations: Iterable[SourceIdentityObservation],
    *,
    current_state_cache_freshness: Optional[Mapping[str, Any]] = None,
    min_trusted_confidence: float = MIN_TRUSTED_SOURCE_CONFIDENCE,
    operator_review_enabled: bool = True,
) -> Tuple[Dict[str, Any], ...]:
    result = analyze_source_identity_confidence(
        observations,
        current_state_cache_freshness=current_state_cache_freshness,
        min_trusted_confidence=min_trusted_confidence,
        operator_review_enabled=operator_review_enabled,
    )
    return tuple(item.to_operator_proposal() for item in result.recommendations)
