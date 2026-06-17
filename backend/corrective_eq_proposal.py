"""Proposal-only corrective EQ recommendations.

This module is transport-free. It turns channel EQ evidence into operator-facing
EQ band proposals, but it never creates a live console write request.
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


CORRECTIVE_EQ_METHOD_ID = "corrective_eq_proposal"
CORRECTIVE_EQ_VERSION = "corrective_eq_proposal_v1"
EQ_READ_REQUIRED_SOURCE = "cached_or_physical_eq_band_read_required_before_promotion"

MIN_EQ_FREQUENCY_HZ = 20.0
MAX_EQ_FREQUENCY_HZ = 18000.0
MIN_EQ_GAIN_DB = -12.0
MAX_EQ_BOOST_DB = 3.0
MAX_DEFAULT_BOOST_DB = 2.0
MIN_EQ_Q = 0.2
MAX_EQ_Q = 12.0
MAX_BAND_MATCH_OCTAVES = 0.35
MIN_TRUE_PEAK_MARGIN_AFTER_BOOST_DB = 1.0
HIGH_FEEDBACK_RISK = 0.45
ALLOWED_FILTER_TYPES = (
    "peaking",
    "notch",
    "low_shelf",
    "high_shelf",
    "low_pass",
    "high_pass",
)
FILTER_TYPE_ALIASES = {
    "bell": "peaking",
    "peak": "peaking",
    "parametric": "peaking",
    "low_shelf": "low_shelf",
    "lowshelf": "low_shelf",
    "low-shelf": "low_shelf",
    "high_shelf": "high_shelf",
    "highshelf": "high_shelf",
    "high-shelf": "high_shelf",
    "low_pass": "low_pass",
    "lowpass": "low_pass",
    "low-pass": "low_pass",
    "high_pass": "high_pass",
    "highpass": "high_pass",
    "high-pass": "high_pass",
}

ROLE_TOKENS = (
    ("vocal", ("vocal", "vox", "voice", "lead", "backing")),
    ("bass", ("bass", "sub")),
    ("drum", ("kick", "snare", "tom", "drum", "overhead", "oh", "hat")),
    ("guitar", ("gtr", "guitar")),
    ("keys", ("key", "keys", "piano", "synth", "organ")),
    ("playback", ("playback", "track", "stem", "loop")),
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


def classify_source_family(source_role: str) -> str:
    text = str(source_role or "").strip().lower().replace("-", "_").replace(" ", "_")
    for family, tokens in ROLE_TOKENS:
        if any(token in text for token in tokens):
            return family
    return "default"


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


def _band_match_octaves(current_hz: Any, target_hz: Any) -> Optional[float]:
    current = _safe_float(current_hz, 0.0)
    target = _safe_float(target_hz, 0.0)
    if current <= 0.0 or target <= 0.0:
        return None
    return abs(math.log2(target / current))


def _normalize_filter_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    return FILTER_TYPE_ALIASES.get(text, text)


@dataclass(frozen=True)
class CorrectiveEQAnalysisInput:
    channel_id: int
    source_role: str
    channel_label: str = ""
    problem: str = "corrective_eq"
    current_band_id: Optional[int] = None
    current_band_frequency_hz: Optional[float] = None
    current_band_gain_db: Optional[float] = None
    current_band_q: Optional[float] = None
    current_band_is_feedback_notch: bool = False
    current_band_is_manual: bool = False
    target_frequency_hz: float = 1000.0
    proposed_gain_db: float = -2.0
    proposed_q: float = 1.4
    filter_type: str = "peaking"
    true_peak_headroom_db: float = 6.0
    feedback_risk: float = 0.0
    phase_risk: float = 0.0
    confidence_hint: float = 0.7
    current_state_cache_freshness: Optional[Dict[str, Any]] = None
    phase_context_freshness: Optional[Dict[str, Any]] = None
    source_identity_trust_state: str = "not_provided"
    source_identity_trust_confidence: float = 0.0
    source_identity_replay_outcome_count: int = 0
    source_identity_calibration_freshness: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class CorrectiveEQRecommendation:
    id: str
    status: str
    action_type: str
    channel_id: int
    channel_label: str
    source_role: str
    source_family: str
    current_band_id: Optional[int]
    target_frequency_hz: float
    proposed_gain_db: float
    proposed_q: float
    filter_type: str
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

        evidence = {
            **self.evidence,
            "old_value_source": EQ_READ_REQUIRED_SOURCE,
            "rollback_value_source": EQ_READ_REQUIRED_SOURCE,
            "live_write": False,
            "queue_mutation": False,
            "readback_required_before_promotion": True,
            "rollback_required_before_promotion": True,
        }
        return {
            "id": self.id,
            "method_id": CORRECTIVE_EQ_METHOD_ID,
            "source": CORRECTIVE_EQ_METHOD_ID,
            "title": f"{self.channel_label or f'Ch {self.channel_id}'}: EQ proposal",
            "reason": self.reason,
            "channel": self.channel_id,
            "target": self.channel_id,
            "kind": "eq",
            "severity": self.severity,
            "confidence": self.confidence,
            "queue_action": "not_queueable_until_promoted",
            "can_import_to_queue": False,
            "can_apply_live": False,
            "blockers": [
                "unsupported_value_type:eq",
                "not_in_supervised_fader_gain_executor",
                "hardware_readback_and_rollback_not_proven",
                *self.blockers,
            ],
            "proposed_action": {
                "action_type": self.action_type,
                "band_id": self.current_band_id,
                "filter_type": self.filter_type,
                "frequency_hz": self.target_frequency_hz,
                "gain_db": self.proposed_gain_db,
                "q": self.proposed_q,
            },
            "old_value": {
                "source": EQ_READ_REQUIRED_SOURCE,
                "required_before_apply": True,
            },
            "rollback_value": {
                "source": EQ_READ_REQUIRED_SOURCE,
                "required_before_apply": True,
            },
            "evidence": evidence,
        }


def _severity(gain_db: float, blockers: Iterable[str]) -> str:
    blocker_set = set(blockers)
    if (
        "source_identity_blocked_from_downstream_trust" in blocker_set
        or "source_identity_replay_context_not_fresh" in blocker_set
        or "feedback_notch_band_preserved" in blocker_set
    ):
        return "high"
    if blocker_set:
        return "medium"
    if abs(gain_db) >= 5.0:
        return "high"
    if abs(gain_db) >= 2.5:
        return "medium"
    return "low"


def analyze_corrective_eq_need(
    analysis: CorrectiveEQAnalysisInput,
    *,
    min_confidence: float = 0.45,
) -> CorrectiveEQRecommendation:
    family = classify_source_family(analysis.source_role)
    target_frequency = round(
        _clip(_safe_float(analysis.target_frequency_hz, 1000.0), MIN_EQ_FREQUENCY_HZ, MAX_EQ_FREQUENCY_HZ),
        1,
    )
    proposed_gain = round(_clip(_safe_float(analysis.proposed_gain_db), MIN_EQ_GAIN_DB, MAX_EQ_BOOST_DB), 2)
    proposed_q = round(_clip(_safe_float(analysis.proposed_q, 1.4), MIN_EQ_Q, MAX_EQ_Q), 2)
    requested_filter_type = str(analysis.filter_type or "").strip()
    normalized_filter_type = _normalize_filter_type(analysis.filter_type)
    filter_type = (
        normalized_filter_type
        if normalized_filter_type in ALLOWED_FILTER_TYPES
        else "unsupported"
    )
    feedback_risk = _clip01(analysis.feedback_risk)
    phase_risk = _clip01(analysis.phase_risk)
    confidence_hint = _clip01(analysis.confidence_hint)
    true_peak_headroom = _safe_float(analysis.true_peak_headroom_db)
    current_state_freshness = _freshness_payload(analysis.current_state_cache_freshness)
    current_state_freshness_status = str(current_state_freshness.get("status") or "unknown")
    current_state_no_authority = _freshness_is_no_authority(current_state_freshness)
    phase_context_freshness = _freshness_payload(analysis.phase_context_freshness)
    phase_context_freshness_status = str(phase_context_freshness.get("status") or "unknown")
    phase_context_no_authority = _freshness_is_no_authority(phase_context_freshness)
    band_match = _band_match_octaves(analysis.current_band_frequency_hz, target_frequency)
    source_identity_trust = evaluate_source_identity_downstream_trust(
        source_id=f"ch:{int(analysis.channel_id)}",
        source_type=str(analysis.source_role or "unknown"),
        trust_state=analysis.source_identity_trust_state,
        trust_confidence=analysis.source_identity_trust_confidence,
        replay_outcome_count=analysis.source_identity_replay_outcome_count,
        current_state_cache_freshness=analysis.source_identity_calibration_freshness,
    )

    blockers = []
    if analysis.current_band_id is None:
        blockers.append("eq_band_identity_missing")
    if band_match is None:
        blockers.append("eq_band_frequency_readback_missing")
    elif band_match > MAX_BAND_MATCH_OCTAVES:
        blockers.append("eq_band_frequency_mismatch")
    if analysis.current_band_is_feedback_notch:
        blockers.append("feedback_notch_band_preserved")
    if analysis.current_band_is_manual:
        blockers.append("manual_eq_band_preserved")
    if current_state_freshness_status != "fresh":
        blockers.append("eq_current_state_cache_not_fresh")
    if current_state_freshness_status == "fresh" and not current_state_no_authority:
        blockers.append("eq_current_state_cache_live_authority_present")
    if phase_context_freshness_status != "fresh":
        blockers.append("eq_phase_context_not_fresh")
    if phase_context_freshness_status == "fresh" and not phase_context_no_authority:
        blockers.append("eq_phase_context_cache_live_authority_present")
    if normalized_filter_type not in ALLOWED_FILTER_TYPES:
        blockers.append("unsupported_eq_filter_type")
    if proposed_gain > MAX_DEFAULT_BOOST_DB:
        blockers.append("eq_boost_exceeds_default_bound")
    if proposed_gain > 0.0 and true_peak_headroom - proposed_gain < MIN_TRUE_PEAK_MARGIN_AFTER_BOOST_DB:
        blockers.append("insufficient_true_peak_headroom_for_eq_boost")
    if proposed_gain > 0.0 and feedback_risk >= HIGH_FEEDBACK_RISK:
        blockers.append("feedback_risk_blocks_eq_boost")
    blockers.extend(source_identity_trust.blockers)
    blockers_tuple = tuple(sorted(set(blockers)))

    bounded_gain = {
        "status": "blocked"
        if {
            "eq_boost_exceeds_default_bound",
            "insufficient_true_peak_headroom_for_eq_boost",
            "feedback_risk_blocks_eq_boost",
        }.intersection(blockers_tuple)
        else "passed",
        "proposed_gain_db": proposed_gain,
        "max_cut_db": MIN_EQ_GAIN_DB,
        "max_default_boost_db": MAX_DEFAULT_BOOST_DB,
        "true_peak_headroom_db": round(true_peak_headroom, 2),
        "min_true_peak_margin_after_boost_db": MIN_TRUE_PEAK_MARGIN_AFTER_BOOST_DB,
    }
    band_identity = {
        "status": "passed"
        if not {
            "eq_band_identity_missing",
            "eq_band_frequency_readback_missing",
            "eq_band_frequency_mismatch",
        }.intersection(blockers_tuple)
        else "blocked",
        "band_id": analysis.current_band_id,
        "current_band_frequency_hz": analysis.current_band_frequency_hz,
        "target_frequency_hz": target_frequency,
        "band_match_octaves": None if band_match is None else round(band_match, 3),
        "max_band_match_octaves": MAX_BAND_MATCH_OCTAVES,
    }
    protected_band_policy = {
        "status": "passed"
        if not {"feedback_notch_band_preserved", "manual_eq_band_preserved"}.intersection(blockers_tuple)
        else "blocked",
        "current_band_is_feedback_notch": bool(analysis.current_band_is_feedback_notch),
        "current_band_is_manual": bool(analysis.current_band_is_manual),
    }
    filter_type_policy = {
        "status": "passed"
        if "unsupported_eq_filter_type" not in blockers_tuple
        else "blocked",
        "requested_filter_type": requested_filter_type,
        "normalized_filter_type": normalized_filter_type,
        "selected_filter_type": filter_type,
        "allowed_filter_types": list(ALLOWED_FILTER_TYPES),
    }
    confidence = _clip01(
        0.35 * confidence_hint
        + 0.25 * min(abs(proposed_gain) / 4.0, 1.0)
        + 0.15 * (1.0 - feedback_risk)
        + 0.15 * (1.0 - phase_risk)
        + 0.10 * min(max(true_peak_headroom, 0.0) / 8.0, 1.0)
    )

    evidence = {
        "source": CORRECTIVE_EQ_VERSION,
        "source_role": analysis.source_role,
        "source_family": family,
        "problem": analysis.problem,
        "filter_type": filter_type,
        "filter_type_policy": filter_type_policy,
        "frequency_match": band_identity,
        "band_identity": band_identity,
        "protected_band_policy": protected_band_policy,
        "bounded_gain_delta": bounded_gain,
        "feedback_notch_preserved": not analysis.current_band_is_feedback_notch,
        "manual_band_preserved": not analysis.current_band_is_manual,
        "true_peak_headroom": bounded_gain,
        "feedback_risk": round(feedback_risk, 3),
        "phase_risk": round(phase_risk, 3),
        "current_state_cache_freshness_status": current_state_freshness_status,
        "current_eq_context_cache_freshness": current_state_freshness,
        "current_eq_context_cache_no_live_authority": current_state_no_authority,
        "current_phase_audio_context_cache_freshness_status": phase_context_freshness_status,
        "current_phase_audio_context_cache_freshness": phase_context_freshness,
        "current_phase_audio_context_cache_no_live_authority": phase_context_no_authority,
        "source_identity_downstream_trust": source_identity_trust.to_dict(),
        "old_value_source": EQ_READ_REQUIRED_SOURCE,
        "rollback_value_source": EQ_READ_REQUIRED_SOURCE,
        "readback_required_before_promotion": True,
        "rollback_required_before_promotion": True,
        "transport_free": True,
    }

    status = "proposal_only" if not blockers_tuple and confidence >= min_confidence else "analysis_only"
    label = analysis.channel_label or f"Ch {analysis.channel_id}"
    if status == "proposal_only":
        reason = f"{label} can review {proposed_gain:+.1f} dB EQ at {target_frequency:.0f} Hz."
    else:
        reason = f"{label} does not meet safe corrective EQ proposal criteria."

    return CorrectiveEQRecommendation(
        id=f"{CORRECTIVE_EQ_METHOD_ID}:ch{int(analysis.channel_id)}:band{analysis.current_band_id}",
        status=status,
        action_type="eq_band_proposal",
        channel_id=int(analysis.channel_id),
        channel_label=label,
        source_role=str(analysis.source_role or "unknown"),
        source_family=family,
        current_band_id=analysis.current_band_id,
        target_frequency_hz=target_frequency,
        proposed_gain_db=proposed_gain,
        proposed_q=proposed_q,
        filter_type=filter_type,
        confidence=round(confidence, 3),
        severity=_severity(proposed_gain, blockers_tuple),
        reason=reason,
        blockers=blockers_tuple,
        evidence=evidence,
    )


def build_corrective_eq_proposals(
    analyses: Iterable[CorrectiveEQAnalysisInput],
) -> Tuple[Dict[str, Any], ...]:
    proposals = []
    for analysis in analyses:
        recommendation = analyze_corrective_eq_need(analysis)
        if recommendation.status == "proposal_only":
            proposals.append(recommendation.to_operator_proposal())
    return tuple(proposals)
