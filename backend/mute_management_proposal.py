"""Proposal-only mute management recommendations.

This module is transport-free. It turns channel state, signal activity,
source role, operator intent, and safety evidence into mute/unmute proposals,
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


MUTE_METHOD_ID = "mute_management_proposal"
MUTE_VERSION = "mute_management_proposal_v1"
MIN_CONFIDENCE = 0.45
IDLE_MUTE_RMS_DBFS = -60.0
IDLE_MUTE_PEAK_DBFS = -48.0
ACTIVE_UNMUTE_RMS_DBFS = -42.0
ACTIVE_UNMUTE_PEAK_DBFS = -30.0
MIN_IDLE_SECONDS = 18.0
HIGH_FEEDBACK_RISK = 0.82
HIGH_DROPOUT_RISK = 0.55
MIN_SOURCE_CONFIDENCE = 0.55

SAFE_IDLE_FAMILIES = {"tom", "percussion", "backing_vocal", "guitar", "keys", "fx_return"}
CRITICAL_FAMILIES = {"lead_vocal", "kick", "snare", "bass", "playback", "master", "reference"}
ROLE_TOKENS = (
    ("lead_vocal", ("lead_vocal", "lead vocal", "main_vocal", "main vocal", "vox lead")),
    ("backing_vocal", ("backing_vocal", "backing vocal", "bgv", "bv", "harmony")),
    ("kick", ("kick", "bd", "bass_drum")),
    ("snare", ("snare", "sn")),
    ("bass", ("bass", "di bass", "bass_di")),
    ("tom", ("tom", "floor_tom", "rack_tom", "floor tom", "rack tom")),
    ("percussion", ("perc", "clap", "shaker", "tamb", "cowbell")),
    ("guitar", ("gtr", "guitar")),
    ("keys", ("key", "keys", "piano", "synth", "organ")),
    ("playback", ("playback", "track", "stem", "loop")),
    ("fx_return", ("fx", "reverb", "delay", "plate", "room", "return")),
    ("master", ("master", "main", "matrix")),
    ("reference", ("reference", "talkback", "measurement")),
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


def classify_source_family(source_role: str, channel_label: str = "") -> str:
    text = f"{source_role} {channel_label}".strip().lower().replace("-", "_")
    spaced = text.replace("_", " ")
    for family, tokens in ROLE_TOKENS:
        if any(token in text or token in spaced for token in tokens):
            return family
    return "default"


@dataclass(frozen=True)
class MuteManagementAnalysisInput:
    channel_id: int
    source_role: str
    channel_label: str = ""
    current_muted: bool = False
    auto_muted_by_system: bool = False
    allow_auto_mute_candidate: bool = False
    rms_dbfs: float = -100.0
    peak_dbfs: float = -100.0
    noise_floor_dbfs: float = -90.0
    idle_duration_sec: float = 0.0
    recent_signal_cycles: int = 0
    expected_next_cue_sec: Optional[float] = None
    source_confidence: float = 0.75
    bleed_risk: float = 0.0
    feedback_risk: float = 0.0
    dropout_risk: float = 0.0
    operator_intent: str = ""
    persistence_cycles: int = 1
    current_state_cache_freshness: Optional[Dict[str, Any]] = None
    source_identity_trust_state: str = "not_provided"
    source_identity_trust_confidence: float = 0.0
    source_identity_replay_outcome_count: int = 0
    source_identity_calibration_freshness: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class MuteManagementRecommendation:
    id: str
    status: str
    action_type: str
    channel_id: int
    channel_label: str
    source_role: str
    source_family: str
    current_muted: bool
    proposed_muted: bool
    mute_reason: str
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
            "method_id": MUTE_METHOD_ID,
            "source": MUTE_METHOD_ID,
            "title": f"{self.channel_label or f'Ch {self.channel_id}'}: mute-state proposal",
            "reason": self.reason,
            "channel": self.channel_id,
            "target": self.channel_id,
            "kind": "mute_state",
            "severity": self.severity,
            "confidence": self.confidence,
            "queue_action": "not_queueable_until_promoted",
            "can_import_to_queue": False,
            "can_apply_live": False,
            "blockers": [
                "unsupported_value_type:mute_state",
                "not_in_supervised_fader_gain_executor",
                "hardware_readback_and_rollback_not_proven",
                *self.blockers,
            ],
            "proposed_action": {
                "action_type": self.action_type,
                "channel": self.channel_id,
                "current_muted": self.current_muted,
                "proposed_muted": self.proposed_muted,
                "mute_reason": self.mute_reason,
            },
            "evidence": {
                **self.evidence,
                "live_write": False,
                "queue_mutation": False,
                "readback_required_before_promotion": True,
                "rollback_required_before_promotion": True,
            },
        }


def _is_idle(rms_dbfs: float, peak_dbfs: float, idle_duration_sec: float) -> bool:
    return (
        rms_dbfs <= IDLE_MUTE_RMS_DBFS
        and peak_dbfs <= IDLE_MUTE_PEAK_DBFS
        and idle_duration_sec >= MIN_IDLE_SECONDS
    )


def _is_active(rms_dbfs: float, peak_dbfs: float, recent_signal_cycles: int) -> bool:
    return (
        rms_dbfs >= ACTIVE_UNMUTE_RMS_DBFS
        or peak_dbfs >= ACTIVE_UNMUTE_PEAK_DBFS
        or recent_signal_cycles >= 2
    )


def _severity(action_type: str, feedback_risk: float, blockers: Iterable[str]) -> str:
    blocker_set = set(blockers)
    if (
        "source_identity_blocked_from_downstream_trust" in blocker_set
        or "source_identity_replay_context_not_fresh" in blocker_set
    ):
        return "high"
    if "source_identity_not_trusted_for_downstream" in blocker_set:
        return "medium"
    if action_type == "emergency_mute_proposal" or feedback_risk >= HIGH_FEEDBACK_RISK:
        return "high"
    if "manual_mute_authority_required" in blocker_set or "critical_source_manual_only" in blocker_set:
        return "high"
    if action_type == "unmute_proposal":
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


def analyze_mute_management_need(
    analysis: MuteManagementAnalysisInput,
    *,
    min_confidence: float = MIN_CONFIDENCE,
) -> MuteManagementRecommendation:
    channel = int(analysis.channel_id)
    label = analysis.channel_label or f"Ch {channel}"
    family = classify_source_family(analysis.source_role, label)
    current_muted = bool(analysis.current_muted)
    auto_muted = bool(analysis.auto_muted_by_system)
    allow_candidate = bool(analysis.allow_auto_mute_candidate)
    rms = _safe_float(analysis.rms_dbfs, -100.0)
    peak = _safe_float(analysis.peak_dbfs, -100.0)
    noise_floor = _safe_float(analysis.noise_floor_dbfs, -100.0)
    idle_seconds = max(0.0, _safe_float(analysis.idle_duration_sec))
    recent_signal_cycles = max(0, int(analysis.recent_signal_cycles or 0))
    source_confidence = _clip01(analysis.source_confidence)
    bleed_risk = _clip01(analysis.bleed_risk)
    feedback_risk = _clip01(analysis.feedback_risk)
    dropout_risk = _clip01(analysis.dropout_risk)
    operator_intent = str(analysis.operator_intent or "").strip().lower()
    persistence = max(1, int(analysis.persistence_cycles or 1))
    cue_seconds = None if analysis.expected_next_cue_sec is None else max(0.0, _safe_float(analysis.expected_next_cue_sec))
    current_state_freshness = _freshness_payload(analysis.current_state_cache_freshness)
    current_state_freshness_status = str(current_state_freshness.get("status") or "unknown")
    current_state_no_authority = _freshness_is_no_authority(current_state_freshness)
    source_identity_trust = evaluate_source_identity_downstream_trust(
        source_id=f"ch:{channel}",
        source_type=str(analysis.source_role or "unknown"),
        trust_state=analysis.source_identity_trust_state,
        trust_confidence=analysis.source_identity_trust_confidence,
        replay_outcome_count=analysis.source_identity_replay_outcome_count,
        current_state_cache_freshness=analysis.source_identity_calibration_freshness,
    )

    idle = _is_idle(rms, peak, idle_seconds)
    active = _is_active(rms, peak, recent_signal_cycles)
    emergency_intent = operator_intent in {"emergency_feedback", "feedback", "unsafe_signal"}
    emergency_mute = emergency_intent and feedback_risk >= 0.65 and not current_muted
    unmute_for_cue = cue_seconds is not None and cue_seconds <= 5.0 and current_muted and auto_muted

    proposed_muted = current_muted
    action_type = "mute_state_analysis"
    mute_reason = "no_change"
    blockers = []

    if emergency_mute:
        proposed_muted = True
        action_type = "emergency_mute_proposal"
        mute_reason = "operator_flagged_feedback_or_unsafe_signal"
    elif current_muted:
        if active or unmute_for_cue:
            proposed_muted = False
            action_type = "unmute_proposal"
            mute_reason = "auto_muted_channel_has_returned_signal_or_upcoming_cue"
            if not auto_muted:
                blockers.append("manual_mute_authority_required")
        else:
            blockers.append("mute_state_change_not_needed")
    else:
        if idle and family in CRITICAL_FAMILIES:
            blockers.append("critical_source_manual_only")
        elif idle and (family in SAFE_IDLE_FAMILIES or allow_candidate):
            proposed_muted = True
            action_type = "idle_mute_proposal"
            mute_reason = "idle_safe_source_below_signal_threshold"
        else:
            blockers.append("mute_state_change_not_needed")

    if source_confidence < MIN_SOURCE_CONFIDENCE:
        blockers.append("source_confidence_low")
    if action_type == "idle_mute_proposal" and not allow_candidate and family not in SAFE_IDLE_FAMILIES:
        blockers.append("auto_mute_candidate_not_allowed")
    if action_type == "idle_mute_proposal" and dropout_risk >= HIGH_DROPOUT_RISK:
        blockers.append("dropout_risk_high")
    if action_type == "idle_mute_proposal" and cue_seconds is not None and cue_seconds <= 12.0:
        blockers.append("upcoming_cue_too_close_for_mute")
    if action_type == "unmute_proposal" and feedback_risk >= HIGH_FEEDBACK_RISK:
        blockers.append("feedback_risk_high_for_unmute")
    if current_state_freshness_status != "fresh":
        blockers.append("mute_current_state_cache_not_fresh")
    if current_state_freshness_status == "fresh" and not current_state_no_authority:
        blockers.append("mute_current_state_cache_live_authority_present")
    blockers.extend(source_identity_trust.blockers)

    changed = proposed_muted != current_muted
    if not changed and "mute_state_change_not_needed" not in blockers:
        blockers.append("mute_state_change_not_needed")

    confidence = _clip01(
        0.30 * source_confidence
        + 0.20 * min(persistence / 3.0, 1.0)
        + 0.15 * min(idle_seconds / 45.0, 1.0)
        + 0.10 * (1.0 - dropout_risk)
        + 0.10 * (1.0 - feedback_risk if action_type == "unmute_proposal" else feedback_risk)
        + 0.10 * (1.0 if changed else 0.0)
        + 0.05 * min(max(noise_floor + 85.0, 0.0) / 25.0, 1.0)
    )
    blockers_tuple = tuple(sorted(set(blockers)))
    status = "proposal_only" if changed and not blockers_tuple and confidence >= min_confidence else "analysis_only"
    if status == "proposal_only":
        target_text = "mute" if proposed_muted else "unmute"
        reason = f"{label} can be proposed to {target_text}: {mute_reason}."
    else:
        reason = f"{label} does not meet safe mute-state proposal criteria."

    evidence = {
        "source": MUTE_VERSION,
        "source_role": analysis.source_role,
        "source_family": family,
        "current_muted": current_muted,
        "auto_muted_by_system": auto_muted,
        "allow_auto_mute_candidate": allow_candidate,
        "rms_dbfs": round(rms, 2),
        "peak_dbfs": round(peak, 2),
        "noise_floor_dbfs": round(noise_floor, 2),
        "idle_duration_sec": round(idle_seconds, 2),
        "recent_signal_cycles": recent_signal_cycles,
        "expected_next_cue_sec": cue_seconds,
        "source_confidence": round(source_confidence, 3),
        "bleed_risk": round(bleed_risk, 3),
        "feedback_risk": round(feedback_risk, 3),
        "dropout_risk": round(dropout_risk, 3),
        "operator_intent": operator_intent,
        "idle_signal": idle,
        "active_signal": active,
        "persistence_cycles": persistence,
        "current_state_cache_freshness_status": current_state_freshness_status,
        "current_state_cache_freshness": current_state_freshness,
        "current_mute_cache_no_live_authority": current_state_no_authority,
        "source_identity_downstream_trust": source_identity_trust.to_dict(),
        "transport_free": True,
        "manual_mute_authority_preserved": True,
    }

    return MuteManagementRecommendation(
        id=f"{MUTE_METHOD_ID}:ch{channel}",
        status=status,
        action_type=action_type,
        channel_id=channel,
        channel_label=label,
        source_role=str(analysis.source_role or "unknown"),
        source_family=family,
        current_muted=current_muted,
        proposed_muted=proposed_muted,
        mute_reason=mute_reason,
        confidence=round(confidence, 3),
        severity=_severity(action_type, feedback_risk, blockers_tuple),
        reason=reason,
        blockers=blockers_tuple,
        evidence=evidence,
    )


def build_mute_management_proposals(
    analyses: Iterable[MuteManagementAnalysisInput],
) -> Tuple[Dict[str, Any], ...]:
    proposals = []
    for analysis in analyses:
        recommendation = analyze_mute_management_need(analysis)
        if recommendation.status == "proposal_only":
            proposals.append(recommendation.to_operator_proposal())
    return tuple(proposals)
