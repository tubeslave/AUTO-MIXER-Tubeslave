"""Proposal-only arrangement role recommendations.

This module is transport-free. It turns source-type, section, inter-channel,
reference, and persistence evidence into operator-facing arrangement-role
proposals, but it never creates a live console write request.
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


ARRANGEMENT_ROLE_METHOD_ID = "arrangement_role_proposal"
ARRANGEMENT_ROLE_VERSION = "arrangement_role_proposal_v1"
SUPPORTED_SOURCE_TYPES = {
    "lead_vocal",
    "vocal",
    "backing_vocal",
    "kick",
    "snare",
    "drums",
    "bass",
    "guitar",
    "keys",
    "synth",
    "playback",
    "fx_return",
    "talkback",
    "reference",
    "unknown",
}
SUPPORTED_ARRANGEMENT_ROLES = {
    "lead",
    "featured",
    "rhythmic_anchor",
    "low_anchor",
    "harmonic_support",
    "backing",
    "texture",
    "effect",
    "utility",
    "reference",
    "unknown",
}
MIN_CONFIDENCE = 0.5
MIN_SOURCE_TYPE_CONFIDENCE = 0.55
MIN_ROLE_CONFIDENCE = 0.62
MIN_CONTEXT_CONFIDENCE = 0.55
MIN_PERSISTENCE_CYCLES = 2
MAX_COMPETING_ROLE_COUNT = 1
MAX_RECENT_ROLE_SWITCHES = 1


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
        "vox": "vocal",
        "bv": "backing_vocal",
        "bgv": "backing_vocal",
        "bd": "kick",
        "bass_drum": "kick",
        "sn": "snare",
        "kit": "drums",
        "gtr": "guitar",
        "key": "keys",
        "loop": "playback",
        "track": "playback",
        "return": "fx_return",
        "fx": "fx_return",
        "tb": "talkback",
        "ref": "reference",
    }
    return aliases.get(text, text or "unknown")


def normalize_arrangement_role(role: Any) -> str:
    text = _normalize_token(role)
    aliases = {
        "main": "lead",
        "lead_vocal": "lead",
        "solo": "featured",
        "feature": "featured",
        "rhythm": "rhythmic_anchor",
        "rhythm_anchor": "rhythmic_anchor",
        "groove": "rhythmic_anchor",
        "bass_anchor": "low_anchor",
        "low": "low_anchor",
        "support": "harmonic_support",
        "pad": "texture",
        "ambient": "texture",
        "backing_vocal": "backing",
        "bv": "backing",
        "fx": "effect",
        "utility_source": "utility",
        "talkback": "utility",
        "ref": "reference",
    }
    return aliases.get(text, text or "unknown")


def infer_candidate_role_from_scores(
    source_type: str,
    *,
    lead_salience_score: float = 0.0,
    rhythmic_stability_score: float = 0.0,
    low_anchor_score: float = 0.0,
    harmonic_support_score: float = 0.0,
    texture_score: float = 0.0,
) -> str:
    source = normalize_source_type(source_type)
    if source in {"talkback"}:
        return "utility"
    if source in {"reference"}:
        return "reference"
    if source in {"fx_return"}:
        return "effect"
    if source in {"lead_vocal"} or lead_salience_score >= 0.72:
        return "lead"
    if source in {"kick", "snare", "drums"} or rhythmic_stability_score >= 0.72:
        return "rhythmic_anchor"
    if source in {"bass"} or low_anchor_score >= 0.72:
        return "low_anchor"
    if source in {"backing_vocal"}:
        return "backing"
    if harmonic_support_score >= 0.62 or source in {"guitar", "keys", "synth", "playback"}:
        return "harmonic_support"
    if texture_score >= 0.62:
        return "texture"
    return "unknown"


@dataclass(frozen=True)
class ArrangementRoleAnalysisInput:
    source_id: str
    source_type: str
    channel_label: str = ""
    current_arrangement_role: str = "unknown"
    proposed_arrangement_role: str = ""
    section_label: str = ""
    source_type_confidence: float = 0.75
    role_confidence: float = 0.75
    context_confidence: float = 0.7
    arrangement_context_available: bool = False
    inter_channel_context_available: bool = False
    reference_role_available: bool = False
    accepted_history_available: bool = False
    activity_persistence_cycles: int = 1
    competing_role_count: int = 0
    recent_role_switches: int = 0
    lead_salience_score: float = 0.0
    rhythmic_stability_score: float = 0.0
    low_anchor_score: float = 0.0
    harmonic_support_score: float = 0.0
    texture_score: float = 0.0
    masking_impact_score: float = 0.0
    operator_review_enabled: bool = True
    current_state_cache_freshness: Optional[Dict[str, Any]] = None
    source_identity_trust_state: str = "not_provided"
    source_identity_trust_confidence: float = 0.0
    source_identity_replay_outcome_count: int = 0
    source_identity_calibration_freshness: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ArrangementRoleRecommendation:
    id: str
    status: str
    action_type: str
    source_id: str
    source_type: str
    channel_label: str
    current_arrangement_role: str
    proposed_arrangement_role: str
    section_label: str
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

        proposed_action = {
            "action_type": self.action_type,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "current_arrangement_role": self.current_arrangement_role,
            "proposed_arrangement_role": self.proposed_arrangement_role,
            "section_label": self.section_label,
        }
        proposal = {
            "id": self.id,
            "method_id": ARRANGEMENT_ROLE_METHOD_ID,
            "source": ARRANGEMENT_ROLE_METHOD_ID,
            "title": f"{self.channel_label or self.source_id}: arrangement role proposal",
            "reason": self.reason,
            "target": self.source_id,
            "kind": "arrangement_role",
            "severity": self.severity,
            "confidence": self.confidence,
            "queue_action": "not_queueable_until_promoted",
            "can_import_to_queue": False,
            "can_apply_live": False,
            "blockers": [
                "unsupported_value_type:arrangement_role",
                "not_in_supervised_fader_gain_executor",
                "role_learning_requires_operator_acceptance",
                *self.blockers,
            ],
            "proposed_action": proposed_action,
            "evidence": {
                **self.evidence,
                "live_write": False,
                "queue_mutation": False,
                "readback_required_before_promotion": True,
                "rollback_required_before_promotion": False,
            },
        }
        proposal["operator_correction"] = build_operator_correction_payload(
            proposal=proposal,
            method_id=ARRANGEMENT_ROLE_METHOD_ID,
            target=self.source_id,
            proposed_action=proposed_action,
            blockers=proposal["blockers"],
        )
        return proposal


def _severity(blockers: Iterable[str], proposed_role: str, masking_impact_score: float) -> str:
    blocker_set = set(blockers)
    if (
        "source_identity_blocked_from_downstream_trust" in blocker_set
        or "source_identity_replay_context_not_fresh" in blocker_set
    ):
        return "high"
    if "role_ambiguity_high" in blocker_set or "recent_role_switches_high" in blocker_set:
        return "high"
    if proposed_role in {"lead", "low_anchor", "rhythmic_anchor"} or masking_impact_score >= 0.65:
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


def analyze_arrangement_role_need(
    analysis: ArrangementRoleAnalysisInput,
    *,
    min_confidence: float = MIN_CONFIDENCE,
) -> ArrangementRoleRecommendation:
    source_type = normalize_source_type(analysis.source_type)
    current_role = normalize_arrangement_role(analysis.current_arrangement_role)
    inferred_role = infer_candidate_role_from_scores(
        source_type,
        lead_salience_score=_clip01(analysis.lead_salience_score),
        rhythmic_stability_score=_clip01(analysis.rhythmic_stability_score),
        low_anchor_score=_clip01(analysis.low_anchor_score),
        harmonic_support_score=_clip01(analysis.harmonic_support_score),
        texture_score=_clip01(analysis.texture_score),
    )
    proposed_role = normalize_arrangement_role(analysis.proposed_arrangement_role or inferred_role)
    source_type_confidence = _clip01(analysis.source_type_confidence)
    role_confidence = _clip01(analysis.role_confidence)
    context_confidence = _clip01(analysis.context_confidence)
    persistence = max(1, int(analysis.activity_persistence_cycles or 1))
    competing_roles = max(0, int(analysis.competing_role_count or 0))
    role_switches = max(0, int(analysis.recent_role_switches or 0))
    masking_impact = _clip01(analysis.masking_impact_score)
    current_state_freshness = _freshness_payload(analysis.current_state_cache_freshness)
    current_state_freshness_status = str(current_state_freshness.get("status") or "unknown")
    current_state_no_authority = _freshness_is_no_authority(current_state_freshness)
    source_identity_trust = evaluate_source_identity_downstream_trust(
        source_id=str(analysis.source_id),
        source_type=source_type,
        trust_state=analysis.source_identity_trust_state,
        trust_confidence=analysis.source_identity_trust_confidence,
        replay_outcome_count=analysis.source_identity_replay_outcome_count,
        current_state_cache_freshness=analysis.source_identity_calibration_freshness,
    )

    blockers = []
    if source_type not in SUPPORTED_SOURCE_TYPES:
        blockers.append("unsupported_source_type_for_role")
    if proposed_role not in SUPPORTED_ARRANGEMENT_ROLES:
        blockers.append("unsupported_arrangement_role")
    if not analysis.arrangement_context_available:
        blockers.append("arrangement_context_missing")
    if not analysis.inter_channel_context_available:
        blockers.append("inter_channel_context_missing")
    if not analysis.operator_review_enabled:
        blockers.append("operator_review_disabled")
    if source_type_confidence < MIN_SOURCE_TYPE_CONFIDENCE:
        blockers.append("source_type_confidence_low")
    if role_confidence < MIN_ROLE_CONFIDENCE:
        blockers.append("role_confidence_low")
    if context_confidence < MIN_CONTEXT_CONFIDENCE:
        blockers.append("context_confidence_low")
    if persistence < MIN_PERSISTENCE_CYCLES:
        blockers.append("role_persistence_low")
    if competing_roles > MAX_COMPETING_ROLE_COUNT:
        blockers.append("role_ambiguity_high")
    if role_switches > MAX_RECENT_ROLE_SWITCHES:
        blockers.append("recent_role_switches_high")
    if current_role == proposed_role:
        blockers.append("arrangement_role_change_not_needed")
    if proposed_role == "unknown":
        blockers.append("arrangement_role_unknown")
    if current_state_freshness_status != "fresh":
        blockers.append("arrangement_role_context_cache_not_fresh")
    if current_state_freshness_status == "fresh" and not current_state_no_authority:
        blockers.append("arrangement_role_context_cache_live_authority_present")
    blockers.extend(source_identity_trust.blockers)

    confidence = _clip01(
        0.26 * source_type_confidence
        + 0.28 * role_confidence
        + 0.18 * context_confidence
        + 0.10 * min(persistence / 4.0, 1.0)
        + 0.06 * (1.0 if analysis.reference_role_available else 0.0)
        + 0.06 * (1.0 if analysis.accepted_history_available else 0.0)
        + 0.04 * masking_impact
        + 0.02 * (1.0 if analysis.operator_review_enabled else 0.0)
    )
    blockers_tuple = tuple(sorted(set(blockers)))
    status = "proposal_only" if not blockers_tuple and confidence >= min_confidence else "analysis_only"
    label = analysis.channel_label or analysis.source_id
    if status == "proposal_only":
        reason = f"{label} can be proposed as {proposed_role} in the arrangement."
    else:
        reason = f"{label} does not meet safe arrangement-role proposal criteria."

    evidence = {
        "source": ARRANGEMENT_ROLE_VERSION,
        "section_label": analysis.section_label,
        "arrangement_context_available": bool(analysis.arrangement_context_available),
        "arrangement_context": {
            "available": bool(analysis.arrangement_context_available),
            "section_label": analysis.section_label,
            "context_confidence": round(context_confidence, 3),
        },
        "inter_channel_context_available": bool(analysis.inter_channel_context_available),
        "inter_channel_context": {
            "available": bool(analysis.inter_channel_context_available),
            "competing_role_count": competing_roles,
            "recent_role_switches": role_switches,
        },
        "reference_role_available": bool(analysis.reference_role_available),
        "reference_role": {
            "available": bool(analysis.reference_role_available),
            "current_role": current_role,
            "proposed_role": proposed_role,
        },
        "accepted_history_available": bool(analysis.accepted_history_available),
        "accepted_history": {
            "available": bool(analysis.accepted_history_available),
            "role_persistence_cycles": persistence,
            "operator_review_enabled": bool(analysis.operator_review_enabled),
        },
        "activity_persistence_cycles": persistence,
        "role_persistence": {
            "cycles": persistence,
            "minimum_cycles": MIN_PERSISTENCE_CYCLES,
            "persistent": persistence >= MIN_PERSISTENCE_CYCLES,
        },
        "competing_role_count": competing_roles,
        "recent_role_switches": role_switches,
        "role_ambiguity": {
            "competing_role_count": competing_roles,
            "max_competing_role_count": MAX_COMPETING_ROLE_COUNT,
            "recent_role_switches": role_switches,
            "max_recent_role_switches": MAX_RECENT_ROLE_SWITCHES,
        },
        "source_type_confidence": round(source_type_confidence, 3),
        "role_confidence": round(role_confidence, 3),
        "context_confidence": round(context_confidence, 3),
        "lead_salience_score": round(_clip01(analysis.lead_salience_score), 3),
        "rhythmic_stability_score": round(_clip01(analysis.rhythmic_stability_score), 3),
        "low_anchor_score": round(_clip01(analysis.low_anchor_score), 3),
        "harmonic_support_score": round(_clip01(analysis.harmonic_support_score), 3),
        "texture_score": round(_clip01(analysis.texture_score), 3),
        "masking_impact_score": round(masking_impact, 3),
        "masking_impact": {
            "score": round(masking_impact, 3),
            "requires_downstream_unmasking_review": masking_impact >= 0.5,
        },
        "operator_review_enabled": bool(analysis.operator_review_enabled),
        "current_state_cache_freshness_status": current_state_freshness_status,
        "current_state_cache_freshness": current_state_freshness,
        "current_arrangement_role_context_cache_freshness": current_state_freshness,
        "current_arrangement_role_context_cache_no_live_authority": current_state_no_authority,
        "source_identity_downstream_trust": source_identity_trust.to_dict(),
        "transport_free": True,
        "decision_evidence_packet": True,
    }

    return ArrangementRoleRecommendation(
        id=f"{ARRANGEMENT_ROLE_METHOD_ID}:{analysis.source_id}",
        status=status,
        action_type="arrangement_role_proposal",
        source_id=str(analysis.source_id),
        source_type=source_type,
        channel_label=label,
        current_arrangement_role=current_role,
        proposed_arrangement_role=proposed_role,
        section_label=str(analysis.section_label or ""),
        confidence=round(confidence, 3),
        severity=_severity(blockers_tuple, proposed_role, masking_impact),
        reason=reason,
        blockers=blockers_tuple,
        evidence=evidence,
    )


def build_arrangement_role_proposals(
    analyses: Iterable[ArrangementRoleAnalysisInput],
) -> Tuple[Dict[str, Any], ...]:
    proposals = []
    for analysis in analyses:
        recommendation = analyze_arrangement_role_need(analysis)
        if recommendation.status == "proposal_only":
            proposals.append(recommendation.to_operator_proposal())
    return tuple(proposals)
