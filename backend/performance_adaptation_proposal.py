"""Proposal-only performance adaptation recommendations.

This module is transport-free. It turns level drift, baseline evidence,
readback, headroom, feedback risk, cooldown, and operator intent into
operator-facing adaptation proposals, but it never creates a live console write
request.
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


PERFORMANCE_ADAPTATION_METHOD_ID = "performance_adaptation_proposal"
PERFORMANCE_ADAPTATION_VERSION = "performance_adaptation_proposal_v1"
SUPPORTED_TARGET_KINDS = {"channel", "stem", "group", "bus", "dca", "matrix"}
SAFE_ADAPTATION_WINDOWS = {"soundcheck", "rehearsal", "performance", "live_show"}
MIN_CONFIDENCE = 0.5
MIN_SOURCE_CONFIDENCE = 0.55
MIN_TREND_CONFIDENCE = 0.6
MIN_SAFETY_CONFIDENCE = 0.65
MIN_DRIFT_DB = 0.5
MAX_UP_DELTA_DB = 0.8
MAX_DOWN_DELTA_DB = 1.5
FADER_CEILING_DB = 0.0
TRUE_PEAK_RESERVE_DB = 1.0
HIGH_FEEDBACK_RISK = 0.62
HIGH_BLEED_RISK = 0.7
MIN_PERSISTENCE_CYCLES = 2
MIN_COOLDOWN_SEC = 5.0


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


def normalize_target_kind(target_kind: Any) -> str:
    text = str(target_kind or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ch": "channel",
        "input": "channel",
        "subgroup": "group",
        "grp": "group",
        "vca": "dca",
        "mtx": "matrix",
    }
    return aliases.get(text, text or "channel")


def _normalize_window(window: Any) -> str:
    text = str(window or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "live": "performance",
        "show": "performance",
        "concert": "performance",
        "line_check": "soundcheck",
    }
    return aliases.get(text, text or "performance")


def _source_family(source_role: str, label: str = "") -> str:
    text = f"{source_role} {label}".strip().lower().replace("-", "_")
    spaced = text.replace("_", " ")
    if any(token in text or token in spaced for token in ("lead_vocal", "lead vocal", "main_vocal", "main vocal")):
        return "lead_vocal"
    if any(token in text or token in spaced for token in ("kick", "snare", "bass", "playback", "reference", "main")):
        return "critical"
    if any(token in text or token in spaced for token in ("tom", "overhead", "oh", "room", "ride", "hihat")):
        return "high_bleed"
    return "default"


def _bounded_delta(desired_delta_db: float, current_fader_db: Optional[float], main_bus_headroom_db: float) -> float:
    if desired_delta_db > 0.0:
        available_by_headroom = max(0.0, main_bus_headroom_db - TRUE_PEAK_RESERVE_DB)
        available_by_fader = MAX_UP_DELTA_DB
        if current_fader_db is not None:
            available_by_fader = max(0.0, FADER_CEILING_DB - _safe_float(current_fader_db))
        return round(_clip(desired_delta_db, 0.0, min(MAX_UP_DELTA_DB, available_by_headroom, available_by_fader)), 1)
    return round(_clip(desired_delta_db, -MAX_DOWN_DELTA_DB, 0.0), 1)


@dataclass(frozen=True)
class PerformanceAdaptationAnalysisInput:
    target_id: str
    target_kind: str
    source_role: str
    target_label: str = ""
    arrangement_role: str = ""
    baseline_level_lufs: Optional[float] = None
    current_level_lufs: float = -25.0
    short_term_lufs: float = -25.0
    target_plane_lufs: Optional[float] = None
    current_fader_db: Optional[float] = None
    current_gain_db: Optional[float] = None
    true_peak_dbfs: float = -8.0
    main_bus_headroom_db: float = 6.0
    feedback_risk: float = 0.0
    bleed_risk: float = 0.0
    baseline_available: bool = False
    current_readback_available: bool = False
    rollback_available: bool = False
    operator_intent: str = ""
    adaptation_window: str = "performance"
    active_performance: bool = True
    manual_override_active: bool = False
    auto_apply_requested: bool = False
    cooldown_elapsed_sec: float = MIN_COOLDOWN_SEC
    persistence_cycles: int = 1
    latency_budget_ms: float = 250.0
    latency_budget_met: bool = True
    source_confidence: float = 0.75
    trend_confidence: float = 0.75
    safety_confidence: float = 0.75
    current_state_cache_freshness: Optional[Dict[str, Any]] = None
    source_identity_trust_state: str = "not_provided"
    source_identity_trust_confidence: float = 0.0
    source_identity_replay_outcome_count: int = 0
    source_identity_calibration_freshness: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class PerformanceAdaptationRecommendation:
    id: str
    status: str
    action_type: str
    target_id: str
    target_kind: str
    target_label: str
    source_role: str
    source_family: str
    baseline_level_lufs: Optional[float]
    current_level_lufs: float
    reference_level_lufs: float
    level_drift_db: float
    suggested_delta_db: float
    target_fader_db: Optional[float]
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
            "target_id": self.target_id,
            "target_kind": self.target_kind,
            "source_role": self.source_role,
            "reference_level_lufs": self.reference_level_lufs,
            "current_level_lufs": self.current_level_lufs,
            "level_drift_db": self.level_drift_db,
            "suggested_delta_db": self.suggested_delta_db,
            "target_fader_db": self.target_fader_db,
            "followup_proposal_methods": [
                "bus_group_balance_proposal",
                "lufs_gain_staging",
                "master_bus_optimization_proposal",
            ],
        }
        proposal = {
            "id": self.id,
            "method_id": PERFORMANCE_ADAPTATION_METHOD_ID,
            "source": PERFORMANCE_ADAPTATION_METHOD_ID,
            "title": f"{self.target_label or self.target_id}: performance adaptation proposal",
            "reason": self.reason,
            "target": self.target_id,
            "kind": "performance_adaptation",
            "severity": self.severity,
            "confidence": self.confidence,
            "queue_action": "not_queueable_until_promoted",
            "can_import_to_queue": False,
            "can_apply_live": False,
            "blockers": [
                "unsupported_value_type:performance_adaptation",
                "not_in_supervised_fader_gain_executor",
                "adaptation_requires_replay_and_hardware_proof",
                *self.blockers,
            ],
            "proposed_action": proposed_action,
            "evidence": {
                **self.evidence,
                "live_write": False,
                "queue_mutation": False,
                "readback_required_before_promotion": True,
                "rollback_required_before_promotion": True,
            },
        }
        proposal["operator_correction"] = build_operator_correction_payload(
            proposal=proposal,
            method_id=PERFORMANCE_ADAPTATION_METHOD_ID,
            target=self.target_id,
            proposed_action=proposed_action,
            blockers=proposal["blockers"],
        )
        return proposal


def _severity(blockers: Iterable[str], level_drift_db: float, active_performance: bool) -> str:
    blocker_set = set(blockers)
    if (
        "source_identity_blocked_from_downstream_trust" in blocker_set
        or "source_identity_replay_context_not_fresh" in blocker_set
    ):
        return "high"
    if "source_identity_not_trusted_for_downstream" in blocker_set:
        return "medium"
    if "feedback_risk_high_for_upward_adaptation" in blocker_set or abs(level_drift_db) >= 5.0:
        return "high"
    if active_performance and abs(level_drift_db) >= 2.0:
        return "medium"
    if abs(level_drift_db) >= 1.5:
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


def analyze_performance_adaptation_need(
    analysis: PerformanceAdaptationAnalysisInput,
    *,
    min_confidence: float = MIN_CONFIDENCE,
) -> PerformanceAdaptationRecommendation:
    target_kind = normalize_target_kind(analysis.target_kind)
    label = analysis.target_label or analysis.target_id
    family = _source_family(analysis.source_role, label)
    current_level = _safe_float(analysis.current_level_lufs, -25.0)
    short_term = _safe_float(analysis.short_term_lufs, current_level)
    baseline = None if analysis.baseline_level_lufs is None else _safe_float(analysis.baseline_level_lufs)
    reference_level = (
        _safe_float(analysis.target_plane_lufs)
        if analysis.target_plane_lufs is not None
        else (baseline if baseline is not None else current_level)
    )
    drift = round(current_level - reference_level, 2)
    desired_delta = -drift
    upward_requested = desired_delta > 0.0
    headroom = max(0.0, _safe_float(analysis.main_bus_headroom_db, 0.0))
    suggested_delta = _bounded_delta(desired_delta, analysis.current_fader_db, headroom)
    target_fader = None
    if analysis.current_fader_db is not None:
        target_fader = round(_clip(_safe_float(analysis.current_fader_db) + suggested_delta, -144.0, FADER_CEILING_DB), 1)

    feedback_risk = _clip01(analysis.feedback_risk)
    bleed_risk = _clip01(analysis.bleed_risk)
    source_confidence = _clip01(analysis.source_confidence)
    trend_confidence = _clip01(analysis.trend_confidence)
    safety_confidence = _clip01(analysis.safety_confidence)
    persistence = max(1, int(analysis.persistence_cycles or 1))
    cooldown = max(0.0, _safe_float(analysis.cooldown_elapsed_sec, 0.0))
    latency_budget = max(0.0, _safe_float(analysis.latency_budget_ms, 0.0))
    window = _normalize_window(analysis.adaptation_window)
    operator_intent = str(analysis.operator_intent or "").strip().lower()
    current_state_freshness = _freshness_payload(analysis.current_state_cache_freshness)
    current_state_freshness_status = str(current_state_freshness.get("status") or "unknown")
    current_state_no_authority = _freshness_is_no_authority(current_state_freshness)
    source_identity_trust = evaluate_source_identity_downstream_trust(
        source_id=str(analysis.target_id),
        source_type=str(analysis.source_role or "unknown"),
        trust_state=analysis.source_identity_trust_state,
        trust_confidence=analysis.source_identity_trust_confidence,
        replay_outcome_count=analysis.source_identity_replay_outcome_count,
        current_state_cache_freshness=analysis.source_identity_calibration_freshness,
    )

    blockers = []
    if target_kind not in SUPPORTED_TARGET_KINDS:
        blockers.append("unsupported_adaptation_target_kind")
    if not analysis.baseline_available or baseline is None:
        blockers.append("baseline_level_missing")
    if not analysis.current_readback_available:
        blockers.append("current_level_readback_missing")
    if not analysis.rollback_available:
        blockers.append("rollback_state_missing")
    if not operator_intent:
        blockers.append("operator_intent_required")
    if window not in SAFE_ADAPTATION_WINDOWS:
        blockers.append("unsupported_adaptation_window")
    if analysis.manual_override_active:
        blockers.append("manual_override_active")
    if analysis.auto_apply_requested:
        blockers.append("auto_apply_not_supported")
    if persistence < MIN_PERSISTENCE_CYCLES:
        blockers.append("trend_persistence_low")
    if cooldown < MIN_COOLDOWN_SEC:
        blockers.append("cooldown_not_elapsed")
    if not analysis.latency_budget_met or latency_budget <= 0.0:
        blockers.append("latency_budget_missing")
    if abs(drift) < MIN_DRIFT_DB or suggested_delta == 0.0:
        blockers.append("adaptation_change_not_needed")
    if upward_requested:
        required_headroom = TRUE_PEAK_RESERVE_DB + min(desired_delta, MAX_UP_DELTA_DB)
        if headroom < required_headroom:
            blockers.append("true_peak_headroom_insufficient_for_upward_adaptation")
        if feedback_risk >= HIGH_FEEDBACK_RISK:
            blockers.append("feedback_risk_high_for_upward_adaptation")
        if bleed_risk >= HIGH_BLEED_RISK or family == "high_bleed":
            blockers.append("bleed_risk_high_for_upward_adaptation")
    if analysis.current_fader_db is not None and target_fader is not None and target_fader > FADER_CEILING_DB:
        blockers.append("fader_ceiling_would_be_exceeded")
    if source_confidence < MIN_SOURCE_CONFIDENCE:
        blockers.append("source_confidence_low")
    if trend_confidence < MIN_TREND_CONFIDENCE:
        blockers.append("trend_confidence_low")
    if safety_confidence < MIN_SAFETY_CONFIDENCE:
        blockers.append("safety_confidence_low")
    if current_state_freshness_status != "fresh":
        blockers.append("performance_adaptation_current_state_cache_not_fresh")
    if current_state_freshness_status == "fresh" and not current_state_no_authority:
        blockers.append("performance_adaptation_current_state_cache_live_authority_present")
    blockers.extend(source_identity_trust.blockers)

    confidence = _clip01(
        0.22 * source_confidence
        + 0.22 * trend_confidence
        + 0.20 * safety_confidence
        + 0.12 * min(persistence / 4.0, 1.0)
        + 0.08 * min(cooldown / 12.0, 1.0)
        + 0.06 * min(abs(drift) / 4.0, 1.0)
        + 0.05 * min(headroom / 6.0, 1.0)
        + 0.05 * (1.0 - max(feedback_risk, bleed_risk))
    )
    blockers_tuple = tuple(sorted(set(blockers)))
    status = "proposal_only" if not blockers_tuple and confidence >= min_confidence else "analysis_only"
    direction = "raise" if suggested_delta > 0.0 else "lower"
    if status == "proposal_only":
        reason = f"{label} can be proposed to {direction} {abs(suggested_delta):.1f} dB after persistent level drift."
    else:
        reason = f"{label} does not meet safe performance-adaptation proposal criteria."

    evidence = {
        "source": PERFORMANCE_ADAPTATION_VERSION,
        "target_kind": target_kind,
        "source_role": analysis.source_role,
        "arrangement_role": analysis.arrangement_role,
        "baseline_available": bool(analysis.baseline_available),
        "current_readback_available": bool(analysis.current_readback_available),
        "rollback_available": bool(analysis.rollback_available),
        "operator_intent": operator_intent,
        "adaptation_window": window,
        "active_performance": bool(analysis.active_performance),
        "manual_override_active": bool(analysis.manual_override_active),
        "auto_apply_requested": bool(analysis.auto_apply_requested),
        "baseline_level_lufs": None if baseline is None else round(baseline, 2),
        "baseline_level": (
            None
            if baseline is None
            else {
                "value_lufs": round(baseline, 2),
                "available": bool(analysis.baseline_available),
            }
        ),
        "target_plane_lufs": None if analysis.target_plane_lufs is None else round(_safe_float(analysis.target_plane_lufs), 2),
        "short_term_lufs": round(short_term, 2),
        "current_level_readback": {
            "value_lufs": round(current_level, 2),
            "available": bool(analysis.current_readback_available),
        },
        "current_true_peak_dbfs": round(_safe_float(analysis.true_peak_dbfs, -120.0), 2),
        "main_bus_headroom_db": round(headroom, 2),
        "true_peak_headroom": {
            "main_bus_headroom_db": round(headroom, 2),
            "reserve_db": TRUE_PEAK_RESERVE_DB,
            "current_true_peak_dbfs": round(_safe_float(analysis.true_peak_dbfs, -120.0), 2),
        },
        "feedback_risk": round(feedback_risk, 3),
        "bleed_risk": round(bleed_risk, 3),
        "cooldown_elapsed_sec": round(cooldown, 2),
        "cooldown": {
            "elapsed_sec": round(cooldown, 2),
            "minimum_sec": MIN_COOLDOWN_SEC,
            "elapsed": cooldown >= MIN_COOLDOWN_SEC,
        },
        "latency_budget_ms": round(latency_budget, 2),
        "latency_budget_met": bool(analysis.latency_budget_met),
        "latency_budget": {
            "budget_ms": round(latency_budget, 2),
            "met": bool(analysis.latency_budget_met),
        },
        "persistence_cycles": persistence,
        "trend_persistence": {
            "persistence_cycles": persistence,
            "minimum_cycles": MIN_PERSISTENCE_CYCLES,
            "trend_confidence": round(trend_confidence, 3),
        },
        "level_drift": {
            "drift_db": drift,
            "minimum_db": MIN_DRIFT_DB,
            "reference_level_lufs": round(reference_level, 2),
            "current_level_lufs": round(current_level, 2),
        },
        "manual_override_guard": {
            "manual_override_active": bool(analysis.manual_override_active),
            "operator_intent": operator_intent,
        },
        "rollback_state": {
            "available": bool(analysis.rollback_available),
            "required_before_live_promotion": True,
        },
        "source_confidence": round(source_confidence, 3),
        "trend_confidence": round(trend_confidence, 3),
        "safety_confidence": round(safety_confidence, 3),
        "current_state_cache_freshness_status": current_state_freshness_status,
        "current_state_cache_freshness": current_state_freshness,
        "current_performance_context_cache_freshness": current_state_freshness,
        "current_performance_context_cache_no_live_authority": current_state_no_authority,
        "source_identity_downstream_trust": source_identity_trust.to_dict(),
        "rate_limited": True,
        "transport_free": True,
        "reversible_candidate": True,
    }

    return PerformanceAdaptationRecommendation(
        id=f"{PERFORMANCE_ADAPTATION_METHOD_ID}:{analysis.target_id}",
        status=status,
        action_type="performance_adaptation_proposal",
        target_id=str(analysis.target_id),
        target_kind=target_kind,
        target_label=label,
        source_role=str(analysis.source_role or "unknown"),
        source_family=family,
        baseline_level_lufs=None if baseline is None else round(baseline, 2),
        current_level_lufs=round(current_level, 2),
        reference_level_lufs=round(reference_level, 2),
        level_drift_db=drift,
        suggested_delta_db=suggested_delta,
        target_fader_db=target_fader,
        confidence=round(confidence, 3),
        severity=_severity(blockers_tuple, drift, bool(analysis.active_performance)),
        reason=reason,
        blockers=blockers_tuple,
        evidence=evidence,
    )


def build_performance_adaptation_proposals(
    analyses: Iterable[PerformanceAdaptationAnalysisInput],
) -> Tuple[Dict[str, Any], ...]:
    proposals = []
    for analysis in analyses:
        recommendation = analyze_performance_adaptation_need(analysis)
        if recommendation.status == "proposal_only":
            proposals.append(recommendation.to_operator_proposal())
    return tuple(proposals)
