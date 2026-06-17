"""Proposal-only spatial panning recommendations.

This module is transport-free. It turns source role, stereo-pair, stage-position,
and mono-compatibility evidence into operator-facing pan proposals, but it never
creates a live console write request.
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


PAN_METHOD_ID = "pan_spatial_proposal"
PAN_VERSION = "pan_spatial_proposal_v1"
PAN_READ_REQUIRED_SOURCE = "cached_or_physical_pan_read_required_before_promotion"
PAN_PHASE_REQUIRED_SOURCE = "fresh_mono_phase_check_required_before_non_center_pan_promotion"
MAX_PAN_PERCENT = 70.0
MIN_PAN_DELTA_PERCENT = 8.0
HIGH_MONO_RISK = 0.65
MIN_SAFE_PHASE_CORRELATION = 0.15
MIN_MONO_SUM_DELTA_DB = -3.0

CENTER_FAMILIES = {"lead_vocal", "kick", "snare", "bass", "master", "sub"}
LOW_FREQUENCY_FAMILIES = {"kick", "bass", "sub", "floor_tom"}
ROLE_TOKENS = (
    ("lead_vocal", ("lead_vocal", "lead vocal", "main_vocal", "main vocal", "vox lead")),
    ("backing_vocal", ("backing_vocal", "backing vocal", "back_vocal", "bgv", "bv")),
    ("kick", ("kick", "bd", "bass_drum")),
    ("snare", ("snare", "sn")),
    ("bass", ("bass", "di bass", "bass_di")),
    ("sub", ("sub", "subkick")),
    ("floor_tom", ("floor_tom", "floor tom")),
    ("tom", ("tom",)),
    ("overhead", ("overhead", "oh", "cymbal")),
    ("guitar", ("gtr", "guitar")),
    ("keys", ("keys", "key", "piano", "synth", "organ")),
    ("percussion", ("perc", "clap", "shaker", "tamb")),
    ("playback", ("playback", "track", "stem", "loop")),
    ("master", ("master", "main", "reference")),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _clip01(value: Any) -> float:
    return _clip(_safe_float(value), 0.0, 1.0)


def _normalize_side(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if text in {"l", "left", "stage_left"}:
        return "left"
    if text in {"r", "right", "stage_right"}:
        return "right"
    return None


def classify_source_family(source_role: str) -> str:
    text = str(source_role or "").strip().lower().replace("-", "_")
    spaced = text.replace("_", " ")
    for family, tokens in ROLE_TOKENS:
        if any(token in text or token in spaced for token in tokens):
            return family
    return "default"


def _role_width(family: str) -> float:
    if family in CENTER_FAMILIES:
        return 0.0
    if family in {"backing_vocal", "tom", "overhead", "percussion"}:
        return 35.0
    if family in {"guitar", "keys"}:
        return 48.0
    if family == "playback":
        return 55.0
    return 32.0


@dataclass(frozen=True)
class PanAnalysisInput:
    channel_id: int
    source_role: str
    channel_label: str = ""
    current_pan_percent: float = 0.0
    stage_position: Optional[float] = None
    stereo_pair_id: Optional[str] = None
    stereo_pair_side: Optional[str] = None
    spectral_centroid_hz: float = 1000.0
    low_frequency_share: float = 0.0
    mono_compatibility_risk: float = 0.0
    phase_correlation: Optional[float] = None
    mono_sum_delta_db: Optional[float] = None
    mono_phase_check_freshness: Optional[Dict[str, Any]] = None
    contribution_share: float = 0.0
    role_confidence: float = 0.7
    arrangement_role: str = ""
    persistence_cycles: int = 1
    current_state_cache_freshness: Optional[Dict[str, Any]] = None
    source_identity_trust_state: str = "not_provided"
    source_identity_trust_confidence: float = 0.0
    source_identity_replay_outcome_count: int = 0
    source_identity_calibration_freshness: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class PanRecommendation:
    id: str
    status: str
    action_type: str
    channel_id: int
    channel_label: str
    source_role: str
    source_family: str
    current_pan_percent: float
    target_pan_percent: float
    delta_pan_percent: float
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
            "old_value_source": PAN_READ_REQUIRED_SOURCE,
            "rollback_value_source": PAN_READ_REQUIRED_SOURCE,
            "live_write": False,
            "queue_mutation": False,
            "readback_required_before_promotion": True,
            "rollback_required_before_promotion": True,
        }
        return {
            "id": self.id,
            "method_id": PAN_METHOD_ID,
            "source": PAN_METHOD_ID,
            "title": f"{self.channel_label or f'Ch {self.channel_id}'}: pan proposal",
            "reason": self.reason,
            "channel": self.channel_id,
            "target": self.channel_id,
            "kind": "pan",
            "severity": self.severity,
            "confidence": self.confidence,
            "queue_action": "not_queueable_until_promoted",
            "can_import_to_queue": False,
            "can_apply_live": False,
            "blockers": [
                "unsupported_value_type:pan",
                "not_in_supervised_fader_gain_executor",
                "hardware_readback_and_rollback_not_proven",
                *self.blockers,
            ],
            "proposed_action": {
                "action_type": self.action_type,
                "current_pan_percent": self.current_pan_percent,
                "target_pan_percent": self.target_pan_percent,
                "delta_pan_percent": self.delta_pan_percent,
            },
            "old_value": {
                "source": PAN_READ_REQUIRED_SOURCE,
                "required_before_apply": True,
            },
            "rollback_value": {
                "source": PAN_READ_REQUIRED_SOURCE,
                "required_before_apply": True,
            },
            "evidence": evidence,
        }


def _severity_from_delta(delta: float, blockers: Iterable[str] = ()) -> str:
    blocker_set = set(blockers)
    if (
        "source_identity_blocked_from_downstream_trust" in blocker_set
        or "source_identity_replay_context_not_fresh" in blocker_set
    ):
        return "high"
    if "source_identity_not_trusted_for_downstream" in blocker_set:
        return "medium"
    if blocker_set & {
        "pan_current_state_cache_live_authority_present",
        "pan_mono_phase_check_live_authority_present",
    }:
        return "medium"
    magnitude = abs(delta)
    if magnitude >= 35.0:
        return "high"
    if magnitude >= 16.0:
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


def _target_from_context(analysis: PanAnalysisInput, family: str) -> Tuple[float, str, Tuple[str, ...]]:
    current = _clip(_safe_float(analysis.current_pan_percent), -100.0, 100.0)
    low_share = _clip01(analysis.low_frequency_share)
    blockers = []

    if family in CENTER_FAMILIES or low_share >= 0.55:
        return 0.0, "center_anchor_or_low_frequency", tuple(blockers)

    side = _normalize_side(analysis.stereo_pair_side)
    if side and analysis.stereo_pair_id:
        width = _role_width(family)
        return (-width if side == "left" else width), "stereo_pair_side", tuple(blockers)

    if analysis.stage_position is not None:
        stage = _clip(_safe_float(analysis.stage_position), -1.0, 1.0)
        width = min(_role_width(family), MAX_PAN_PERCENT)
        if family in {"backing_vocal", "tom", "overhead", "percussion"}:
            width = min(width, 38.0)
        return stage * width, "stage_position", tuple(blockers)

    if abs(current) > MIN_PAN_DELTA_PERCENT and family in CENTER_FAMILIES:
        return 0.0, "center_anchor_correction", tuple(blockers)

    blockers.append("stage_or_stereo_pair_context_missing")
    return current, "insufficient_context", tuple(blockers)


def analyze_pan_need(
    analysis: PanAnalysisInput,
    *,
    min_confidence: float = 0.45,
) -> PanRecommendation:
    family = classify_source_family(analysis.source_role)
    current = round(_clip(_safe_float(analysis.current_pan_percent), -100.0, 100.0), 1)
    mono_risk = _clip01(analysis.mono_compatibility_risk)
    contribution = _clip01(analysis.contribution_share)
    role_confidence = _clip01(analysis.role_confidence)
    persistence = max(1, int(analysis.persistence_cycles or 1))
    current_state_freshness = _freshness_payload(analysis.current_state_cache_freshness)
    current_state_freshness_status = str(current_state_freshness.get("status") or "unknown")
    current_state_no_authority = _freshness_is_no_authority(current_state_freshness)
    mono_phase_freshness = _freshness_payload(analysis.mono_phase_check_freshness)
    mono_phase_freshness_status = str(mono_phase_freshness.get("status") or "unknown")
    mono_phase_no_authority = _freshness_is_no_authority(mono_phase_freshness)
    phase_correlation = _optional_float(analysis.phase_correlation)
    if phase_correlation is not None:
        phase_correlation = _clip(phase_correlation, -1.0, 1.0)
    mono_sum_delta_db = _optional_float(analysis.mono_sum_delta_db)
    source_identity_trust = evaluate_source_identity_downstream_trust(
        source_id=f"ch:{int(analysis.channel_id)}",
        source_type=str(analysis.source_role or "unknown"),
        trust_state=analysis.source_identity_trust_state,
        trust_confidence=analysis.source_identity_trust_confidence,
        replay_outcome_count=analysis.source_identity_replay_outcome_count,
        current_state_cache_freshness=analysis.source_identity_calibration_freshness,
    )

    target, decision_source, blockers = _target_from_context(analysis, family)
    target = round(_clip(target, -MAX_PAN_PERCENT, MAX_PAN_PERCENT), 1)
    delta = round(target - current, 1)
    blocker_list = list(blockers)

    if abs(delta) < MIN_PAN_DELTA_PERCENT:
        blocker_list.append("pan_change_not_needed")
    if target != 0.0 and mono_risk >= HIGH_MONO_RISK:
        blocker_list.append("mono_compatibility_risk_high")
    if current_state_freshness_status != "fresh":
        blocker_list.append("pan_current_state_cache_not_fresh")
    if current_state_freshness_status == "fresh" and not current_state_no_authority:
        blocker_list.append("pan_current_state_cache_live_authority_present")
    non_center_pan_requires_phase = target != 0.0 and abs(delta) >= MIN_PAN_DELTA_PERCENT
    if non_center_pan_requires_phase:
        if phase_correlation is None:
            blocker_list.append("mono_phase_check_missing")
        elif phase_correlation < MIN_SAFE_PHASE_CORRELATION:
            blocker_list.append("pan_phase_correlation_unsafe")
        if mono_sum_delta_db is not None and mono_sum_delta_db < MIN_MONO_SUM_DELTA_DB:
            blocker_list.append("pan_mono_sum_loss_high")
        if mono_phase_freshness_status != "fresh":
            blocker_list.append("pan_mono_phase_check_not_fresh")
        if mono_phase_freshness_status == "fresh" and not mono_phase_no_authority:
            blocker_list.append("pan_mono_phase_check_live_authority_present")
    blocker_list.extend(source_identity_trust.blockers)

    phase_safety_blockers = {
        "mono_phase_check_missing",
        "pan_phase_correlation_unsafe",
        "pan_mono_sum_loss_high",
        "pan_mono_phase_check_not_fresh",
    }
    if not non_center_pan_requires_phase:
        mono_phase_status = "not_required_for_center_or_noop"
    elif phase_safety_blockers.intersection(blocker_list):
        mono_phase_status = "blocked"
    else:
        mono_phase_status = "passed"

    confidence = _clip01(
        0.40 * role_confidence
        + 0.20 * min(abs(delta) / 40.0, 1.0)
        + 0.15 * (1.0 - mono_risk)
        + 0.15 * min(persistence / 3.0, 1.0)
        + 0.10 * contribution
    )

    evidence = {
        "source": PAN_VERSION,
        "source_role": analysis.source_role,
        "source_family": family,
        "decision_source": decision_source,
        "stage_position": analysis.stage_position,
        "stereo_pair_id": analysis.stereo_pair_id,
        "stereo_pair_side": _normalize_side(analysis.stereo_pair_side),
        "spectral_centroid_hz": round(_safe_float(analysis.spectral_centroid_hz), 1),
        "low_frequency_share": round(_clip01(analysis.low_frequency_share), 3),
        "mono_compatibility_risk": round(mono_risk, 3),
        "phase_correlation": None if phase_correlation is None else round(phase_correlation, 3),
        "mono_sum_delta_db": None if mono_sum_delta_db is None else round(mono_sum_delta_db, 2),
        "mono_phase_check_freshness_status": mono_phase_freshness_status,
        "mono_phase_check_freshness": mono_phase_freshness,
        "mono_phase_safety": {
            "status": mono_phase_status,
            "required_for_non_center_pan": non_center_pan_requires_phase,
            "phase_evidence_source": PAN_PHASE_REQUIRED_SOURCE,
            "min_safe_phase_correlation": MIN_SAFE_PHASE_CORRELATION,
            "min_mono_sum_delta_db": MIN_MONO_SUM_DELTA_DB,
        },
        "contribution_share": round(contribution, 3),
        "role_confidence": round(role_confidence, 3),
        "arrangement_role": analysis.arrangement_role,
        "persistence_cycles": persistence,
        "current_pan_percent": current,
        "target_pan_percent": target,
        "delta_pan_percent": delta,
        "current_pan_cache_freshness": current_state_freshness,
        "current_pan_cache_no_live_authority": current_state_no_authority,
        "current_state_cache_freshness_status": current_state_freshness_status,
        "current_state_cache_freshness": current_state_freshness,
        "mono_phase_check_no_live_authority": mono_phase_no_authority,
        "source_identity_downstream_trust": source_identity_trust.to_dict(),
        "old_value_source": PAN_READ_REQUIRED_SOURCE,
        "rollback_value_source": PAN_READ_REQUIRED_SOURCE,
        "readback_required_before_promotion": True,
        "rollback_required_before_promotion": True,
        "transport_free": True,
    }

    status = "proposal_only" if not blocker_list and confidence >= min_confidence else "analysis_only"
    label = analysis.channel_label or f"Ch {analysis.channel_id}"
    if status == "proposal_only":
        reason = f"{label} pan can move {current:.1f} -> {target:.1f} using {decision_source} evidence."
    else:
        reason = f"{label} does not meet safe pan proposal criteria."

    return PanRecommendation(
        id=f"{PAN_METHOD_ID}:ch{int(analysis.channel_id)}",
        status=status,
        action_type="pan_proposal",
        channel_id=int(analysis.channel_id),
        channel_label=label,
        source_role=str(analysis.source_role or "unknown"),
        source_family=family,
        current_pan_percent=current,
        target_pan_percent=target,
        delta_pan_percent=delta,
        confidence=round(confidence, 3),
        severity=_severity_from_delta(delta, blocker_list),
        reason=reason,
        blockers=tuple(sorted(set(blocker_list))),
        evidence=evidence,
    )


def build_pan_proposals(
    analyses: Iterable[PanAnalysisInput],
) -> Tuple[Dict[str, Any], ...]:
    proposals = []
    for analysis in analyses:
        recommendation = analyze_pan_need(analysis)
        if recommendation.status == "proposal_only":
            proposals.append(recommendation.to_operator_proposal())
    return tuple(proposals)
